"""Official GME REST API client (GME = Gestore dei Mercati Energetici).

AUTHORITATIVE source for Italian power & gas market results: the ex-post PUN
Index, the 7 zonal MGP prices, intraday and the gas day-ahead (MGP-GAS / PSV).
Talks to the public ``PublicMarketResults`` service at
``https://api.mercatoelettrico.org``.

Protocol:
* ``POST {base}/api/v1/Auth`` with ``{Username, Password}`` -> JWT bearer token.
* ``POST {base}/api/v1/RequestData`` with ``Authorization: Bearer <jwt>`` and a
  JSON body ``{"Platform","Segment","DataName","IntervalStart","IntervalEnd",
  "Attributes"}`` (dates ``YYYYMMDD``). For delivery dates on/after
  ``QUARTER_HOUR_GOLIVE`` (2025-10-01) we add ``Attributes={"GranularityType":
  "PT15"}`` to request the 15-minute MTU.
* Response is JSON with a ``ContentResponse`` field holding a base64-encoded
  ``.zip``; ``FormatType`` ("JSON"/"XML") tells us how to parse the inner file.
  decode -> ``io.BytesIO`` -> ``zipfile`` -> read inner member -> parse.

Each record carries ``FlowDate`` (YYYYMMDD), ``Hour`` (1..25, 25 = autumn long
DST day), ``Period`` (15-min sub-index 1..4, may be null for hourly), ``Zone``
and ``Price``. We convert ``(FlowDate, Hour, Period)`` from Italian local
wall-clock (Europe/Rome) to a tz-aware UTC ``delivery_start``, handling the
1..25 DST convention and the spring short day.

DEFENSIVE NOTES: the public schema varies by dataset/over time, so field names
are matched case-insensitively against several aliases (see the ``*_KEYS`` and
``*_ALIASES`` tables below). The gas DataName / price field in particular are
best-effort (GAS_PGasResults / MGP-GAS; price field PGas / Price, EUR/MWh) and
should be confirmed against live GME responses.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import logging
import zipfile
from typing import Any
from zoneinfo import ZoneInfo

import requests
from lxml import etree
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from energy_prices.config import Market, Zone, get_settings
from energy_prices.config.enums import QUARTER_HOUR_GOLIVE
from energy_prices.storage.repositories import IngestionRepository, PriceRepository

logger = logging.getLogger(__name__)

SOURCE = "gme"
_ROME = ZoneInfo("Europe/Rome")
_UTC = dt.UTC

# Default request timeout (connect, read) in seconds.
_TIMEOUT = (10, 60)

# Physical zonal codes as they appear in GME payloads -> our Zone enum value.
# Keyed by the normalised (upper, alnum-only) token; built once at import time.
_ZONE_ALIASES: dict[str, str] = {
    "NORD": Zone.NORD.value, "NORTH": Zone.NORD.value,
    "CNOR": Zone.CNOR.value, "CENTRONORD": Zone.CNOR.value,
    "CSUD": Zone.CSUD.value, "CENTROSUD": Zone.CSUD.value,
    "SUD": Zone.SUD.value, "SOUTH": Zone.SUD.value,
    "CALA": Zone.CALA.value, "CALABRIA": Zone.CALA.value,
    "SICI": Zone.SICI.value, "SICILIA": Zone.SICI.value, "SICILY": Zone.SICI.value,
    "SARD": Zone.SARD.value, "SARDEGNA": Zone.SARD.value, "SARDINIA": Zone.SARD.value,
}
# Codes that denote the national PUN index rather than a physical zone.
_PUN_ALIASES = {"PUN", "ITALIA", "IT", "ITALY", "NAT", "NATIONAL", "PREZZOUNICO"}

# Candidate field-name aliases (matched case-insensitively, no separators).
_DATE_KEYS = ("flowdate", "date", "data", "gmedate", "deliverydate", "giorno")
_HOUR_KEYS = ("hour", "ora", "ore", "period", "periodo")
_QUARTER_KEYS = ("quarter", "interval", "subhour", "subperiod", "qh", "quartodora")
_ZONE_KEYS = ("zone", "zona", "biddingzone", "marketzone", "area")
_PRICE_KEYS = ("price", "prezzo", "value", "valore", "pun", "pgas", "prezzounico")


class GmeError(RuntimeError):
    """Raised on unrecoverable GME API or payload errors."""


def _norm(key: str) -> str:
    """Normalise a field name for tolerant matching (lower, strip separators)."""
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _build_lookup(record: dict[str, Any]) -> dict[str, Any]:
    """Return a {normalised_key: value} view of a record for tolerant access."""
    return {_norm(k): v for k, v in record.items()}


def _first(lookup: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in lookup and lookup[k] not in (None, ""):
            return lookup[k]
    return None


def _parse_flow_date(raw: Any) -> dt.date:
    """Parse a GME date which is usually ``YYYYMMDD`` but may be ISO ``YYYY-MM-DD``."""
    if isinstance(raw, dt.datetime):
        return raw.date()
    if isinstance(raw, dt.date):
        return raw
    text = str(raw).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return dt.date(int(digits[0:4]), int(digits[4:6]), int(digits[6:8]))
    # Fallback to ISO parsing (handles separators / trailing time component).
    return dt.date.fromisoformat(text[:10])


def _as_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def _as_float(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    text = str(raw).strip()
    # GME sometimes uses a decimal comma in localized payloads.
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _delivery_start_utc(
    flow_date: dt.date, hour: int, quarter: int | None
) -> tuple[dt.datetime, int]:
    """Convert (FlowDate, Hour 1..25, Period 1..4|None) to a UTC delivery start.

    Returns ``(utc_datetime, resolution_minutes)``.

    GME numbers hours 1..24 on normal days, 1..23 on the spring short day, and
    1..25 on the autumn long day (the repeated 02:00-03:00 local hour). We
    therefore build the local timeline by walking concrete UTC instants from
    local midnight, which naturally yields 23/24/25 hourly slots per day and
    resolves the DST fold without ambiguity.
    """
    minutes_per_slot = 15 if quarter is not None else 60
    slots_per_hour = 4 if quarter is not None else 1
    sub = (quarter - 1) if quarter is not None else 0
    slot_index = (hour - 1) * slots_per_hour + sub  # zero-based slot in the local day
    midnight = dt.datetime(flow_date.year, flow_date.month, flow_date.day, tzinfo=_ROME)
    start_utc = midnight.astimezone(_UTC)
    return start_utc + dt.timedelta(minutes=minutes_per_slot * slot_index), minutes_per_slot


def _quarter_attributes(start: dt.date, end: dt.date) -> dict[str, Any]:
    """Attributes requesting 15-min granularity if the window touches the PT15 era."""
    if end >= QUARTER_HOUR_GOLIVE:
        return {"GranularityType": "PT15"}
    return {}


def _ymd(day: dt.date) -> str:
    return day.strftime("%Y%m%d")


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient network errors and 5xx / 429 (quota) responses."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class GmeClient:
    """Thin authenticated client over the GME PublicMarketResults REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: tuple[int, int] = _TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.gme_api_base_url).rstrip("/")
        self._username = username or settings.gme_api_username
        self._password = password or settings.gme_api_password
        self._timeout = timeout
        self._http = session or requests.Session()
        self._token: str | None = None

    # -- low level ---------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _post(self, path: str, payload: dict[str, Any], auth: bool) -> requests.Response:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if auth:
            if not self._token:
                self.authenticate()
            headers["Authorization"] = f"Bearer {self._token}"
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._http.post(url, json=payload, headers=headers, timeout=self._timeout)
        if auth and resp.status_code == 401:  # stale token: refresh once, then retry
            logger.info("GME token rejected (401); re-authenticating.")
            self._token = None
            self.authenticate()
            headers["Authorization"] = f"Bearer {self._token}"
            resp = self._http.post(url, json=payload, headers=headers, timeout=self._timeout)
        if not resp.ok and not _is_retryable(resp_to_http_error(resp)):
            raise GmeError(f"GME POST {path} failed: {resp.status_code} {resp.text[:300]}")
        resp.raise_for_status()
        return resp

    def authenticate(self) -> str:
        """Authenticate and cache a JWT bearer token. Returns the token."""
        if not (self._username and self._password):
            raise GmeError("GME credentials missing (set ENERGY_GME_API_USERNAME/PASSWORD).")
        # NB: the GME Auth endpoint expects the field "Login" (not "Username").
        payload = {"Login": self._username, "Password": self._password}
        url = f"{self.base_url}/api/v1/Auth"
        headers = {"Content-Type": "application/json"}
        try:
            resp = self._http.post(url, json=payload, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:  # network/HTTP
            raise GmeError(f"GME authentication failed: {exc}") from exc
        token = _extract_token(resp)
        if not token:
            raise GmeError("GME authentication returned no token.")
        self._token = token
        logger.info("GME authentication succeeded.")
        return token

    def request_data(
        self,
        segment: str,
        data_name: str,
        start: dt.date,
        end: dt.date,
        attributes: dict[str, Any] | None = None,
        platform: str = "PublicMarketResults",
    ) -> list[dict[str, Any]]:
        """Request a dataset and return its parsed records (list of dicts)."""
        body = {
            "Platform": platform,
            "Segment": segment,
            "DataName": data_name,
            "IntervalStart": _ymd(start),
            "IntervalEnd": _ymd(end),
            "Attributes": attributes or {},
        }
        resp = self._post("api/v1/RequestData", body, auth=True)
        return _parse_response(resp)

    # -- high level --------------------------------------------------------

    def fetch_zonal_prices(self, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
        """Fetch MGP zonal + PUN prices as PriceRepository observation dicts.

        Zone ``PUN`` -> ``zone=Zone.PUN.value`` (market still ELEC_DAYAHEAD); the
        seven physical codes -> their zonal value. Unknown codes are skipped.
        """
        attrs = _quarter_attributes(start, end)
        records = self.request_data("MGP", "ME_ZonalPrices", start, end, attrs)
        observations: list[dict[str, Any]] = []
        for rec in records:
            lookup = _build_lookup(rec)
            zone_value = _map_zone(_first(lookup, _ZONE_KEYS))
            if zone_value is None:
                continue
            obs = _build_price_obs(lookup, Market.ELEC_DAYAHEAD.value, zone_value)
            if obs is not None:
                observations.append(obs)
        logger.info("GME zonal prices %s..%s -> %d observations.", start, end, len(observations))
        return observations

    def fetch_gas_dayahead(self, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
        """Fetch MGP-GAS day-ahead prices as PriceRepository observation dicts.

        ``market=Market.GAS_DAYAHEAD.value``, ``zone=None``. The gas DataName /
        field names are best-effort (see module docstring): we try a couple of
        known DataNames and parse defensively. Gas day-ahead is a single national
        daily price, so resolution is DAILY (1440 minutes) when no hour is given.
        """
        records: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for data_name in ("GAS_PGasResults", "MGP-GAS", "MGPGAS_Prices"):
            try:
                records = self.request_data("MGP-GAS", data_name, start, end, {})
                if records:
                    break
            except (GmeError, requests.RequestException) as exc:  # try next name
                last_exc = exc
                logger.debug("GME gas DataName %s failed: %s", data_name, exc)
        if not records and last_exc is not None:
            logger.warning("GME gas day-ahead unavailable: %s", last_exc)
            return []

        observations: list[dict[str, Any]] = []
        for rec in records:
            obs = _build_price_obs(_build_lookup(rec), Market.GAS_DAYAHEAD.value, None)
            if obs is not None:
                observations.append(obs)
        logger.info("GME gas day-ahead %s..%s -> %d observations.", start, end, len(observations))
        return observations


# --- parsing helpers (module level, reusable & testable) -------------------


def resp_to_http_error(resp: requests.Response) -> requests.HTTPError:
    """Build an HTTPError carrying the response (for retry classification)."""
    err = requests.HTTPError(f"{resp.status_code} for {resp.url}")
    err.response = resp
    return err


def _extract_token(resp: requests.Response) -> str | None:
    """Pull a JWT out of an Auth response (JSON object, bare string, or header)."""
    try:
        data = resp.json()
    except ValueError:
        data = resp.text
    if isinstance(data, str):
        token = data.strip().strip('"')
        return token or resp.headers.get("Authorization", "").removeprefix("Bearer ").strip() or None
    if isinstance(data, dict):
        lookup = _build_lookup(data)
        for key in ("token", "accesstoken", "jwt", "bearertoken", "idtoken"):
            value = lookup.get(key)
            if isinstance(value, str) and value:
                return value
    header = resp.headers.get("Authorization", "")
    return header.removeprefix("Bearer ").strip() or None


def _parse_response(resp: requests.Response) -> list[dict[str, Any]]:
    """Decode a RequestData response: base64 zip -> inner json/xml -> records."""
    try:
        payload = resp.json()
    except ValueError as exc:
        raise GmeError(f"GME RequestData returned non-JSON body: {resp.text[:200]}") from exc

    lookup = _build_lookup(payload) if isinstance(payload, dict) else {}
    content_b64 = _first(lookup, ("contentresponse", "content", "data", "payload"))
    fmt = str(_first(lookup, ("formattype", "format", "type")) or "").upper()

    if content_b64 is None:
        # Some datasets may inline records directly (no zip wrapper).
        inline = _first(lookup, ("records", "result", "results", "rows", "items"))
        if isinstance(inline, list):
            return [r for r in inline if isinstance(r, dict)]
        raise GmeError("GME RequestData response missing ContentResponse.")

    try:
        raw = base64.b64decode(content_b64)
    except (ValueError, TypeError) as exc:
        raise GmeError(f"GME ContentResponse is not valid base64: {exc}") from exc

    inner_bytes, inner_name = _unzip_single(raw)
    if not fmt:
        fmt = "XML" if inner_name.lower().endswith(".xml") else "JSON"

    if fmt == "XML" or inner_bytes.lstrip()[:1] == b"<":
        return _parse_xml(inner_bytes)
    return _parse_json(inner_bytes)


def _unzip_single(raw: bytes) -> tuple[bytes, str]:
    """Return (bytes, name) of the first real member of a zip archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
            if not members:
                raise GmeError("GME zip archive is empty.")
            name = members[0]
            return zf.read(name), name
    except zipfile.BadZipFile as exc:
        # Not a zip — assume the content was the raw document itself.
        logger.debug("GME content not a zip (%s); treating as raw document.", exc)
        return raw, ""


def _parse_json(data: bytes) -> list[dict[str, Any]]:
    """Parse a JSON document into a flat list of record dicts."""
    obj = json.loads(data.decode("utf-8-sig"))
    return _flatten_records(obj)


def _flatten_records(obj: Any) -> list[dict[str, Any]]:
    """Find the list of record dicts inside an arbitrary JSON structure."""
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        lookup = _build_lookup(obj)
        for key in ("records", "result", "results", "rows", "items", "data", "prices"):
            value = lookup.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
        # Otherwise recurse into the first list-valued field we find.
        for value in obj.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
            if isinstance(value, (dict, list)):
                nested = _flatten_records(value)
                if nested:
                    return nested
    return []


def _parse_xml(data: bytes) -> list[dict[str, Any]]:
    """Parse a GME XML document into a list of record dicts.

    GME XML wraps repeated row elements (each holding the fields as child tags
    and/or attributes). We treat the deepest repeated element type as the record
    and merge its attributes with its child element texts.
    """
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(data, parser=parser)
    if root is None:
        return []

    # Group elements by tag; the most frequent leaf-ish tag is the record row.
    candidates: dict[str, list[Any]] = {}
    for el in root.iter():
        if el is root:
            continue
        tag = etree.QName(el).localname
        candidates.setdefault(tag, []).append(el)

    record_tag = None
    best = 0
    for tag, els in candidates.items():
        # Rows have children or attributes and appear repeatedly.
        if len(els) > best and any(len(e) > 0 or e.attrib for e in els):
            record_tag = tag
            best = len(els)

    rows: list[dict[str, Any]] = []
    for el in candidates.get(record_tag, []) if record_tag else []:
        record: dict[str, Any] = {k: v for k, v in el.attrib.items()}
        for child in el:
            name = etree.QName(child).localname
            text = (child.text or "").strip()
            if text:
                record[name] = text
            elif child.attrib:
                record.update({k: v for k, v in child.attrib.items()})
        if record:
            rows.append(record)
    return rows


def _map_zone(raw: Any) -> str | None:
    """Map a raw GME zone token to a Zone enum value, PUN, or None (unknown).

    Alias keys are already normalised (upper, alnum-only), so we normalise the
    incoming token the same way and do direct membership tests.
    """
    if raw is None:
        return None
    token = _norm(str(raw)).upper()
    if not token:
        return None
    if token in _PUN_ALIASES:
        return Zone.PUN.value
    return _ZONE_ALIASES.get(token)


def _build_price_obs(
    lookup: dict[str, Any], market: str, zone: str | None
) -> dict[str, Any] | None:
    """Build one PriceRepository observation dict from a parsed record lookup.

    Returns None when essential fields (date or price) are missing. A record with
    no hour position is treated as a daily national price (typical for gas DA).
    """
    date_raw = _first(lookup, _DATE_KEYS)
    price = _as_float(_first(lookup, _PRICE_KEYS))
    if date_raw is None or price is None:
        return None
    try:
        flow_date = _parse_flow_date(date_raw)
    except (ValueError, TypeError):
        return None

    hour = _as_int(_first(lookup, _HOUR_KEYS))
    quarter = _as_int(_first(lookup, _QUARTER_KEYS))

    if hour is None:
        midnight = dt.datetime(flow_date.year, flow_date.month, flow_date.day, tzinfo=_ROME)
        delivery, resolution = midnight.astimezone(_UTC), 1440
    else:
        delivery, resolution = _delivery_start_utc(flow_date, hour, quarter)

    return {
        "market": market,
        "zone": zone,
        "delivery_start": delivery,
        "resolution_minutes": resolution,
        "price": price,
        "source": SOURCE,
        "unit": "EUR/MWh",
    }


def inspect(
    segment: str = "MGP",
    data_name: str = "ME_ZonalPrices",
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> dict[str, Any]:
    """Fetch a small GME dataset and report its raw structure for validation.

    Use this once when real credentials arrive to confirm the live field names /
    formats before trusting the parser. Returns a diagnostics dict (never raises
    for the common no-credentials case): keys ``ok``, ``n_records``,
    ``field_names`` (union across the sample), ``samples`` (first raw records),
    ``mapped_preview`` (what _build_price_obs would produce), and any ``error``.
    """
    settings = get_settings()
    if not settings.has_gme:
        return {"ok": False, "error": "GME credentials not configured (.env)."}

    end = end or dt.datetime.now(tz=_UTC).date()
    start = start or (end - dt.timedelta(days=2))
    try:
        client = GmeClient()
        client.authenticate()
        attrs = _quarter_attributes(start, end) if data_name == "ME_ZonalPrices" else {}
        records = client.request_data(segment, data_name, start, end, attrs)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash the CLI
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    field_names: set[str] = set()
    for rec in records[:50]:
        field_names.update(rec.keys())

    mapped_preview: list[dict[str, Any]] = []
    is_elec = segment in ("MGP", "MI-A1", "MI-A2", "MI-A3")
    for rec in records[:3]:
        lookup = _build_lookup(rec)
        zone = _map_zone(_first(lookup, _ZONE_KEYS)) if is_elec else None
        market = Market.ELEC_DAYAHEAD.value if is_elec else Market.GAS_DAYAHEAD.value
        mapped_preview.append(_build_price_obs(lookup, market, zone) or {"<unmapped>": rec})

    return {
        "ok": True,
        "segment": segment,
        "data_name": data_name,
        "window": f"{start}..{end}",
        "n_records": len(records),
        "field_names": sorted(field_names),
        "samples": records[:3],
        "mapped_preview": mapped_preview,
    }


def ingest(session: Session, start: dt.date, end: dt.date) -> int:
    """Ingest GME zonal/PUN + gas day-ahead prices for [start, end] (inclusive).

    Writes through PriceRepository and records an IngestionRun. Returns the
    number of price observations upserted. If GME credentials are not configured
    (``not settings.has_gme``) this logs a warning and returns 0 (no-op), so the
    pipeline can run in ENTSO-E-only / demo mode.
    """
    settings = get_settings()
    if not settings.has_gme:
        logger.warning("GME credentials not configured; skipping ingestion (%s..%s).", start, end)
        return 0

    prices = PriceRepository(session)
    runs = IngestionRepository(session)
    run = runs.start(SOURCE, dt.datetime.now(tz=_UTC))

    total = 0
    try:
        client = GmeClient()
        client.authenticate()
        observations = client.fetch_zonal_prices(start, end)
        try:
            observations += client.fetch_gas_dayahead(start, end)
        except (GmeError, requests.RequestException) as exc:
            # Gas is secondary: don't fail the whole run if it's unavailable.
            logger.warning("GME gas day-ahead ingestion failed: %s", exc)

        total = prices.upsert(observations)
        runs.finish(run, "success", dt.datetime.now(tz=_UTC), rows=total,
                    message=f"GME {start}..{end}: {total} rows")
        logger.info("GME ingestion complete: %d rows (%s..%s).", total, start, end)
    except Exception as exc:  # noqa: BLE001 — record failure, then re-raise
        runs.finish(run, "failed", dt.datetime.now(tz=_UTC), rows=total, message=str(exc))
        logger.exception("GME ingestion failed (%s..%s).", start, end)
        raise

    return total
