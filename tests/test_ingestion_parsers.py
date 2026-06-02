"""Unit tests for the ingestion/parsing layer (pure, hermetic — no network/DB).

These lock in three live-validated bug fixes that previously had ZERO coverage:

1. GME 15-minute decode: ``Period`` is the ABSOLUTE quarter-hour slot of the
   local day, ``Hour`` only used as an hourly fallback — so the four quarters of
   one hour map to four distinct UTC instants, never collapse onto one. The UTC
   anchor is the instant of *local* midnight, which makes the spring (23h) and
   autumn (25h) DST days fall out correctly.
2. PUN / NAT dedup: ``PUN``/``PrezzoUnico`` -> ``Zone.PUN``, but ``NAT`` (a
   separate national series) and any unknown/garbage token -> ``None`` (skipped),
   so PUN and NAT never collide on the same (market, zone, delivery_start) key.
3. Gas day-ahead anchoring: the auction row's ``Product`` field carries the
   DELIVERY gas day, anchored at 00:00 UTC / daily resolution (FlowDate+1 only as
   a fallback) so gas aligns with the TTF grid.

Everything is built from synthetic in-memory payloads (dicts / pandas frames /
XML strings). The autouse ``_hermetic_env`` fixture in conftest.py blanks creds.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from energy_prices.config import Market, Zone
from energy_prices.config.enums import PUN_ZONE_WEIGHTS

_UTC = dt.UTC
_ROME = ZoneInfo("Europe/Rome")


# ===========================================================================
# 1. GME 15-minute Period -> UTC timestamp decode
# ===========================================================================
from energy_prices.ingestion import gme_client as gme  # noqa: E402


class TestGmeDeliveryStart:
    """``_delivery_start_utc(flow_date, hour, quarter)`` -> (utc_dt, res_min)."""

    def test_hourly_mapping_to_utc(self):
        """No quarter -> hourly slot anchored at local-midnight UTC + (Hour-1)h."""
        day = dt.date(2025, 1, 15)  # winter: Rome is CET = UTC+1
        # Hour 1 == local 00:00 == 23:00 UTC of the previous day.
        ts, res = gme._delivery_start_utc(day, hour=1, quarter=None)
        assert res == 60
        assert ts == dt.datetime(2025, 1, 14, 23, 0, tzinfo=_UTC)

        # Hour 13 == local 12:00 == 11:00 UTC.
        ts13, res13 = gme._delivery_start_utc(day, hour=13, quarter=None)
        assert res13 == 60
        assert ts13 == dt.datetime(2025, 1, 15, 11, 0, tzinfo=_UTC)

    def test_four_quarters_are_four_distinct_timestamps(self):
        """Period 1..4 -> four distinct 15-min UTC instants, all resolution 15."""
        day = dt.date(2025, 10, 5)  # PT15 era; summer time CEST = UTC+2
        out = [gme._delivery_start_utc(day, hour=1, quarter=q) for q in (1, 2, 3, 4)]
        timestamps = [t for t, _ in out]
        resolutions = {r for _, r in out}

        assert resolutions == {15}
        # Four DISTINCT timestamps (the bug being guarded: they collapsed to one).
        assert len(set(timestamps)) == 4
        # Local midnight 2025-10-05 in CEST == 22:00 UTC the previous day; the
        # four quarters are +0/+15/+30/+45 minutes from that anchor.
        base = dt.datetime(2025, 10, 4, 22, 0, tzinfo=_UTC)
        assert timestamps == [
            base,
            base + dt.timedelta(minutes=15),
            base + dt.timedelta(minutes=30),
            base + dt.timedelta(minutes=45),
        ]

    def test_quarter_is_absolute_slot_not_within_hour(self):
        """Period is the ABSOLUTE quarter index of the day, independent of Hour."""
        day = dt.date(2025, 10, 5)
        base = dt.datetime(2025, 10, 4, 22, 0, tzinfo=_UTC)
        # Period 5 (regardless of the Hour field) -> the 5th quarter == +60 min.
        ts, res = gme._delivery_start_utc(day, hour=99, quarter=5)
        assert res == 15
        assert ts == base + dt.timedelta(minutes=60)

    def test_autumn_dst_long_day_100_quarters(self):
        """Autumn 25h day (2025-10-26): 100 quarters, monotonic, spans 25h real.

        Local midnight is CEST (UTC+2); adding real minutes walks straight
        through the fall-back without ambiguity. The last quarter (100) is at
        +99*15 minutes; the full day spans 25h == 100 quarters.
        """
        day = dt.date(2025, 10, 26)
        base = day_midnight_utc = dt.datetime(2025, 10, 26, tzinfo=_ROME).astimezone(_UTC)
        # That local midnight is 22:00 UTC (CEST, UTC+2) on 2025-10-25.
        assert base == dt.datetime(2025, 10, 25, 22, 0, tzinfo=_UTC)

        first, _ = gme._delivery_start_utc(day, hour=1, quarter=1)
        last, res = gme._delivery_start_utc(day, hour=25, quarter=100)
        assert res == 15
        assert first == day_midnight_utc
        # 100 quarters -> last slot starts 99*15 min after midnight == +24h45m.
        assert last == day_midnight_utc + dt.timedelta(minutes=99 * 15)
        # The autumn day is 25h long: quarter 100 starts before the next local
        # midnight (which is 25h == 100 quarters after this one).
        next_midnight = dt.datetime(2025, 10, 27, tzinfo=_ROME).astimezone(_UTC)
        assert (next_midnight - day_midnight_utc) == dt.timedelta(hours=25)
        assert last < next_midnight

    def test_spring_dst_short_day_92_quarters(self):
        """Spring 23h day (2025-03-30): 92 quarters span exactly 23h real time."""
        day = dt.date(2025, 3, 30)
        day_midnight_utc = dt.datetime(2025, 3, 30, tzinfo=_ROME).astimezone(_UTC)
        # Local midnight is CET (UTC+1) -> 23:00 UTC on 2025-03-29.
        assert day_midnight_utc == dt.datetime(2025, 3, 29, 23, 0, tzinfo=_UTC)

        last, res = gme._delivery_start_utc(day, hour=24, quarter=92)
        assert res == 15
        assert last == day_midnight_utc + dt.timedelta(minutes=91 * 15)
        # The spring day is only 23h == 92 quarters long.
        next_midnight = dt.datetime(2025, 3, 31, tzinfo=_ROME).astimezone(_UTC)
        assert (next_midnight - day_midnight_utc) == dt.timedelta(hours=23)


class TestGmeBuildPriceObs:
    """``_build_price_obs`` ties decode + zone + price into an observation dict."""

    def test_quarter_hour_record_resolution_15(self):
        rec = {"FlowDate": "20251005", "Hour": "1", "Period": "3",
               "Zone": "NORD", "Price": "123.45"}
        obs = gme._build_price_obs(
            gme._build_lookup(rec), Market.ELEC_DAYAHEAD.value, Zone.NORD.value
        )
        assert obs is not None
        assert obs["resolution_minutes"] == 15
        assert obs["price"] == pytest.approx(123.45)
        assert obs["zone"] == Zone.NORD.value
        assert obs["market"] == Market.ELEC_DAYAHEAD.value
        assert obs["source"] == "gme"
        # Period 3 == +30 min from local-midnight UTC (22:00 UTC prev day in CEST).
        assert obs["delivery_start"] == dt.datetime(2025, 10, 4, 22, 30, tzinfo=_UTC)

    def test_pun_local_midnight_anchoring_for_first_quarter(self):
        """PUN record at Hour=1/Period=1 anchors to local-midnight UTC."""
        rec = {"FlowDate": "20250115", "Hour": "1", "Period": None,
               "Zone": "PUN", "Price": "100.0"}
        zone = gme._map_zone(gme._first(gme._build_lookup(rec), gme._ZONE_KEYS))
        obs = gme._build_price_obs(
            gme._build_lookup(rec), Market.ELEC_DAYAHEAD.value, zone
        )
        assert obs is not None
        assert obs["zone"] == Zone.PUN.value
        # Winter CET (UTC+1): local midnight 2025-01-15 == 23:00 UTC 2025-01-14.
        assert obs["delivery_start"] == dt.datetime(2025, 1, 14, 23, 0, tzinfo=_UTC)
        assert obs["resolution_minutes"] == 60

    def test_prereform_hourly_period_zero_sentinel(self):
        """Pre-reform hourly rows carry Period="0" (not None): must decode hourly.

        Regression: Period="0" is the live API's "no sub-period" sentinel for the
        pre-2025-10-01 hourly era. It must NOT be read as a 15-min slot, or all 24
        hours of a day collapse onto (local-midnight - 15 min) and the upsert keeps
        only one price per day. Hours 1 and 24 must map to distinct hourly UTC
        instants 23h apart, both at 60-min resolution.
        """
        rows = []
        for hour in (1, 24):
            rec = {"FlowDate": "20200601", "Hour": str(hour), "Period": "0",
                   "Zone": "NORD", "Price": "20.0"}
            obs = gme._build_price_obs(
                gme._build_lookup(rec), Market.ELEC_DAYAHEAD.value, Zone.NORD.value
            )
            assert obs is not None
            assert obs["resolution_minutes"] == 60
            rows.append(obs["delivery_start"])
        # Summer CEST (UTC+2): local midnight 2020-06-01 == 22:00 UTC 2020-05-31.
        assert rows[0] == dt.datetime(2020, 5, 31, 22, 0, tzinfo=_UTC)
        assert rows[1] == dt.datetime(2020, 6, 1, 21, 0, tzinfo=_UTC)
        assert (rows[1] - rows[0]) == dt.timedelta(hours=23)

    def test_no_hour_is_treated_as_daily(self):
        """A record with no Hour/Period -> daily (1440-min) national price."""
        rec = {"Data": "2025-01-15", "Price": "55.5"}
        obs = gme._build_price_obs(
            gme._build_lookup(rec), Market.GAS_DAYAHEAD.value, None
        )
        assert obs is not None
        assert obs["resolution_minutes"] == 1440
        assert obs["delivery_start"] == dt.datetime(2025, 1, 14, 23, 0, tzinfo=_UTC)

    @pytest.mark.parametrize("rec", [
        {"Hour": "1", "Price": "10.0"},          # missing date
        {"FlowDate": "20250115", "Hour": "1"},   # missing price
        {"FlowDate": "notadate", "Price": "x"},  # unparseable price -> None
    ])
    def test_missing_essentials_returns_none(self, rec):
        obs = gme._build_price_obs(
            gme._build_lookup(rec), Market.ELEC_DAYAHEAD.value, Zone.NORD.value
        )
        assert obs is None

    def test_decimal_comma_price_is_parsed(self):
        rec = {"FlowDate": "20250115", "Hour": "2", "Prezzo": "123,45"}
        obs = gme._build_price_obs(
            gme._build_lookup(rec), Market.ELEC_DAYAHEAD.value, Zone.PUN.value
        )
        assert obs is not None
        assert obs["price"] == pytest.approx(123.45)


# ===========================================================================
# 2. Zone mapping (PUN / NAT dedup collision guard)
# ===========================================================================
class TestGmeMapZone:
    @pytest.mark.parametrize("raw,expected", [
        ("PUN", Zone.PUN.value),
        ("PrezzoUnico", Zone.PUN.value),
        ("Prezzo Unico", Zone.PUN.value),   # separators stripped by _norm
        ("prezzounico", Zone.PUN.value),
    ])
    def test_pun_aliases_map_to_pun(self, raw, expected):
        assert gme._map_zone(raw) == expected

    @pytest.mark.parametrize("raw", [
        "NAT",          # the distinct national series — MUST NOT alias to PUN
        "Nazionale",
        "ZZZ",
        "garbage",
        "",
        "   ",
        None,
    ])
    def test_nat_and_unknown_map_to_none(self, raw):
        """The PUN/NAT dedup fix: NAT and unknown tokens map to None (skipped)."""
        assert gme._map_zone(raw) is None

    @pytest.mark.parametrize("raw,expected", [
        ("NORD", Zone.NORD.value),
        ("North", Zone.NORD.value),
        ("CNOR", Zone.CNOR.value),
        ("Centro Nord", Zone.CNOR.value),
        ("CSUD", Zone.CSUD.value),
        ("Centro Sud", Zone.CSUD.value),
        ("SUD", Zone.SUD.value),
        ("South", Zone.SUD.value),
        ("CALA", Zone.CALA.value),
        ("Calabria", Zone.CALA.value),
        ("SICI", Zone.SICI.value),
        ("Sicilia", Zone.SICI.value),
        ("Sicily", Zone.SICI.value),
        ("SARD", Zone.SARD.value),
        ("Sardegna", Zone.SARD.value),
        ("Sardinia", Zone.SARD.value),
    ])
    def test_physical_zones_map_correctly(self, raw, expected):
        assert gme._map_zone(raw) == expected

    def test_pun_and_nat_do_not_collide(self):
        """PUN -> Zone.PUN, NAT -> None: the two never share a dedup key."""
        assert gme._map_zone("PUN") == Zone.PUN.value
        assert gme._map_zone("NAT") is None
        assert gme._map_zone("PUN") != gme._map_zone("NAT")


# ===========================================================================
# 3. Gas Product/date anchoring
# ===========================================================================
class TestGmeGasAnchoring:
    def test_gas_delivery_date_from_product(self):
        """``Product='MGP-2026-05-26'`` -> delivery date 2026-05-26 (not FlowDate+1)."""
        lookup = gme._build_lookup({"Product": "MGP-2026-05-26"})
        flow = dt.date(2026, 5, 20)  # unrelated session day
        assert gme._gas_delivery_date(lookup, flow) == dt.date(2026, 5, 26)

    @pytest.mark.parametrize("product", [
        "MGP-20260526",          # no separators
        "Delivery 2026/05/26",   # slash separators
    ])
    def test_gas_delivery_date_separator_variants(self, product):
        lookup = gme._build_lookup({"Prodotto": product})
        assert gme._gas_delivery_date(lookup, dt.date(2026, 1, 1)) == dt.date(2026, 5, 26)

    def test_gas_delivery_date_fallback_to_flowdate_plus_one(self):
        """Missing/unparseable Product -> FlowDate + 1 day."""
        assert gme._gas_delivery_date(gme._build_lookup({}), dt.date(2026, 5, 20)) == \
            dt.date(2026, 5, 21)
        bad = gme._build_lookup({"Product": "no-date-here"})
        assert gme._gas_delivery_date(bad, dt.date(2026, 5, 20)) == dt.date(2026, 5, 21)

    def test_build_gas_obs_anchors_utc_midnight_daily(self):
        rec = {"FlowDate": "20260525", "Product": "MGP-2026-05-26", "Price": "33,5"}
        obs = gme._build_gas_obs(gme._build_lookup(rec))
        assert obs is not None
        assert obs["market"] == Market.GAS_DAYAHEAD.value
        assert obs["zone"] is None
        assert obs["resolution_minutes"] == 1440
        assert obs["price"] == pytest.approx(33.5)
        # Anchored to the DELIVERY day at 00:00 UTC (NOT the session FlowDate).
        assert obs["delivery_start"] == dt.datetime(2026, 5, 26, tzinfo=_UTC)
        assert obs["source"] == "gme"

    def test_build_gas_obs_missing_price_returns_none(self):
        rec = {"FlowDate": "20260525", "Product": "MGP-2026-05-26"}
        assert gme._build_gas_obs(gme._build_lookup(rec)) is None


# ===========================================================================
# 3b. GME base64-zip-JSON / XML ContentResponse decode (end-to-end of parser)
# ===========================================================================
class _FakeResponse:
    """Minimal stand-in for requests.Response for the parser helpers."""

    def __init__(self, json_obj):
        self._json = json_obj
        self.text = str(json_obj)

    def json(self):
        return self._json


def _make_zip_b64(inner_name: str, inner_bytes: bytes) -> str:
    import base64
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(inner_name, inner_bytes)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestGmeContentResponseDecode:
    def test_base64_zipped_json_records(self):
        inner = b'{"records": [{"FlowDate": "20250115", "Zone": "NORD", "Price": "10"}]}'
        b64 = _make_zip_b64("data.json", inner)
        resp = _FakeResponse({"ContentResponse": b64, "FormatType": "JSON"})
        records = gme._parse_response(resp)
        assert records == [{"FlowDate": "20250115", "Zone": "NORD", "Price": "10"}]

    def test_base64_zipped_xml_records(self):
        inner = (
            b"<root><Row Zone='NORD'><FlowDate>20250115</FlowDate>"
            b"<Price>10.5</Price></Row>"
            b"<Row Zone='SUD'><FlowDate>20250115</FlowDate>"
            b"<Price>12.5</Price></Row></root>"
        )
        b64 = _make_zip_b64("data.xml", inner)
        resp = _FakeResponse({"ContentResponse": b64, "FormatType": "XML"})
        records = gme._parse_response(resp)
        assert len(records) == 2
        # Attributes + child texts are merged on each row.
        assert records[0]["Zone"] == "NORD"
        assert records[0]["FlowDate"] == "20250115"
        assert records[0]["Price"] == "10.5"
        assert records[1]["Zone"] == "SUD"

    def test_inline_records_without_zip_wrapper(self):
        resp = _FakeResponse({"records": [{"FlowDate": "20250115", "Price": "1"}]})
        assert gme._parse_response(resp) == [{"FlowDate": "20250115", "Price": "1"}]

    def test_missing_content_raises(self):
        resp = _FakeResponse({"unrelated": "field"})
        with pytest.raises(gme.GmeError):
            gme._parse_response(resp)


# ===========================================================================
# 4. ENTSO-E reconstruct_pun (volume-weighted PUN proxy)
# ===========================================================================
from energy_prices.ingestion import entsoe_client as ent  # noqa: E402


class TestEntsoeReconstructPun:
    def test_weighted_average_matches_pun_zone_weights(self):
        delivery = dt.datetime(2025, 1, 15, 10, tzinfo=_UTC)
        # One price per physical zone for a single delivery_start.
        prices = {z: 100.0 + i * 10 for i, z in enumerate(PUN_ZONE_WEIGHTS)}
        rows = [
            {
                "market": Market.ELEC_DAYAHEAD.value,
                "zone": z.value,
                "delivery_start": delivery,
                "resolution_minutes": 60,
                "price": prices[z],
                "source": "entsoe",
            }
            for z in PUN_ZONE_WEIGHTS
        ]
        out = ent.EntsoeClient.reconstruct_pun(rows)
        assert len(out) == 1
        pun = out[0]

        expected = sum(PUN_ZONE_WEIGHTS[z] * prices[z] for z in PUN_ZONE_WEIGHTS) / sum(
            PUN_ZONE_WEIGHTS.values()
        )
        assert pun["price"] == pytest.approx(expected)
        assert pun["zone"] == Zone.PUN.value
        assert pun["source"] == "entsoe"
        assert pun["market"] == Market.ELEC_DAYAHEAD.value
        assert pun["delivery_start"] == delivery
        assert pun["resolution_minutes"] == 60

    def test_partial_zones_renormalise_over_present_weights(self):
        """With only a subset of zones, the weighted mean re-normalises over them."""
        delivery = dt.datetime(2025, 1, 15, 10, tzinfo=_UTC)
        present = {Zone.NORD: 100.0, Zone.SUD: 200.0}
        rows = [
            {
                "market": Market.ELEC_DAYAHEAD.value,
                "zone": z.value,
                "delivery_start": delivery,
                "resolution_minutes": 60,
                "price": p,
                "source": "entsoe",
            }
            for z, p in present.items()
        ]
        out = ent.EntsoeClient.reconstruct_pun(rows)
        assert len(out) == 1
        w_sum = PUN_ZONE_WEIGHTS[Zone.NORD] + PUN_ZONE_WEIGHTS[Zone.SUD]
        expected = (
            PUN_ZONE_WEIGHTS[Zone.NORD] * 100.0 + PUN_ZONE_WEIGHTS[Zone.SUD] * 200.0
        ) / w_sum
        assert out[0]["price"] == pytest.approx(expected)

    def test_pun_zone_rows_are_excluded_from_reconstruction(self):
        """A pre-existing PUN row is ignored (only physical zones contribute)."""
        delivery = dt.datetime(2025, 1, 15, 10, tzinfo=_UTC)
        rows = [
            {"market": Market.ELEC_DAYAHEAD.value, "zone": Zone.NORD.value,
             "delivery_start": delivery, "resolution_minutes": 60,
             "price": 100.0, "source": "entsoe"},
            {"market": Market.ELEC_DAYAHEAD.value, "zone": Zone.PUN.value,
             "delivery_start": delivery, "resolution_minutes": 60,
             "price": 999.0, "source": "entsoe"},
        ]
        out = ent.EntsoeClient.reconstruct_pun(rows)
        # Only NORD contributes -> price equals NORD's price, PUN row ignored.
        assert len(out) == 1
        assert out[0]["price"] == pytest.approx(100.0)

    def test_empty_input_returns_empty(self):
        assert ent.EntsoeClient.reconstruct_pun([]) == []

    def test_groups_by_delivery_start(self):
        d1 = dt.datetime(2025, 1, 15, 10, tzinfo=_UTC)
        d2 = dt.datetime(2025, 1, 15, 11, tzinfo=_UTC)
        rows = [
            {"market": Market.ELEC_DAYAHEAD.value, "zone": Zone.NORD.value,
             "delivery_start": d, "resolution_minutes": 60, "price": 100.0,
             "source": "entsoe"}
            for d in (d1, d2)
        ]
        out = ent.EntsoeClient.reconstruct_pun(rows)
        assert {r["delivery_start"] for r in out} == {d1, d2}


class TestEntsoeInferResolution:
    def test_hourly_index(self):
        idx = pd.date_range("2025-01-15", periods=24, freq="h", tz="UTC")
        assert ent._infer_resolution_minutes(idx) == 60

    def test_quarter_hour_index(self):
        idx = pd.date_range("2025-10-05", periods=96, freq="15min", tz="UTC")
        assert ent._infer_resolution_minutes(idx) == 15

    def test_single_point_falls_back_to_default(self):
        idx = pd.date_range("2025-01-15", periods=1, freq="h", tz="UTC")
        assert ent._infer_resolution_minutes(idx) == 60
        assert ent._infer_resolution_minutes(idx, default=30) == 30

    def test_irregular_uses_minimum_spacing(self):
        idx = pd.DatetimeIndex(
            ["2025-01-15 00:00", "2025-01-15 00:15", "2025-01-15 01:15"], tz="UTC"
        )
        # min spacing is 15 min.
        assert ent._infer_resolution_minutes(idx) == 15


# ===========================================================================
# 5. TTF parsing helpers
# ===========================================================================
from energy_prices.ingestion import ttf_client as ttf  # noqa: E402


class TestTtfExtractClose:
    def _idx(self, n=3):
        return pd.date_range("2025-01-15", periods=n, freq="D")

    def test_flat_columns(self):
        df = pd.DataFrame(
            {"Open": [1, 2, 3], "Close": [10.0, 11.0, 12.0]}, index=self._idx()
        )
        close = ttf._extract_close(df)
        assert close is not None
        assert list(close) == [10.0, 11.0, 12.0]

    def test_multiindex_close_then_ticker(self):
        cols = pd.MultiIndex.from_tuples(
            [("Close", "TTF=F"), ("Open", "TTF=F")]
        )
        df = pd.DataFrame([[10.0, 1.0], [11.0, 2.0]], columns=cols,
                          index=self._idx(2))
        close = ttf._extract_close(df)
        assert close is not None
        assert list(close) == [10.0, 11.0]

    def test_multiindex_ticker_then_close(self):
        cols = pd.MultiIndex.from_tuples(
            [("TTF=F", "Close"), ("TTF=F", "Open")]
        )
        df = pd.DataFrame([[10.0, 1.0], [11.0, 2.0]], columns=cols,
                          index=self._idx(2))
        close = ttf._extract_close(df)
        assert close is not None
        assert list(close) == [10.0, 11.0]

    def test_adj_close_fallback(self):
        df = pd.DataFrame({"Adj Close": [5.0, 6.0]}, index=self._idx(2))
        close = ttf._extract_close(df)
        assert close is not None
        assert list(close) == [5.0, 6.0]

    def test_close_preferred_over_adj_close(self):
        df = pd.DataFrame(
            {"Close": [10.0, 11.0], "Adj Close": [5.0, 6.0]}, index=self._idx(2)
        )
        close = ttf._extract_close(df)
        assert list(close) == [10.0, 11.0]

    def test_empty_and_none_return_none(self):
        assert ttf._extract_close(None) is None
        assert ttf._extract_close(pd.DataFrame()) is None

    def test_no_close_column_returns_none(self):
        df = pd.DataFrame({"Open": [1, 2], "High": [3, 4]}, index=self._idx(2))
        assert ttf._extract_close(df) is None


class TestTtfCoercePrice:
    @pytest.mark.parametrize("value,expected", [
        (10.5, 10.5),
        ("12.25", 12.25),
        (0, 0.0),
        (-3.5, -3.5),
    ])
    def test_valid_numbers(self, value, expected):
        assert ttf._coerce_price(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "not-a-number",
        "",
    ])
    def test_rejects_nan_inf_nonnumeric(self, value):
        assert ttf._coerce_price(value) is None


class TestTtfToUtcMidnight:
    def test_anchors_to_midnight_utc(self):
        ts = ttf._to_utc_midnight(pd.Timestamp("2025-01-15 14:37:00"))
        assert ts == dt.datetime(2025, 1, 15, tzinfo=_UTC)

    def test_drops_intraday_component(self):
        ts = ttf._to_utc_midnight(pd.Timestamp("2025-01-15 23:59:59"))
        assert ts == dt.datetime(2025, 1, 15, tzinfo=_UTC)

    def test_nat_returns_none(self):
        assert ttf._to_utc_midnight(pd.NaT) is None


# ===========================================================================
# 6. GIE AGSI+ paged-envelope parser + sentinel handling
# ===========================================================================
from energy_prices.ingestion import gie_client as gie  # noqa: E402


class TestGieToFloat:
    @pytest.mark.parametrize("value,expected", [
        ("12.5", 12.5),
        ("1,234.5", 1234.5),     # thousands separator stripped
        ("1,000,000", 1000000.0),
        (42, 42.0),
        (3.14, 3.14),
    ])
    def test_valid_values(self, value, expected):
        assert gie._to_float(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [
        "-", "N/A", "n/a", "null", "None", "", "  ", None, "garbage",
    ])
    def test_sentinels_and_garbage_return_none(self, value):
        assert gie._to_float(value) is None


class TestGieGasDayStart:
    def test_parses_iso_date_to_utc_midnight(self):
        assert gie._gas_day_start_utc("2025-01-15") == dt.datetime(
            2025, 1, 15, tzinfo=_UTC
        )

    def test_parses_iso_with_time_suffix(self):
        assert gie._gas_day_start_utc("2025-01-15T06:00:00Z") == dt.datetime(
            2025, 1, 15, tzinfo=_UTC
        )

    @pytest.mark.parametrize("value", ["", None, "not-a-date"])
    def test_bad_values_return_none(self, value):
        assert gie._gas_day_start_utc(value) is None


class TestGieRecordsToObservations:
    def test_full_and_net_withdrawal(self):
        records = [
            {"gasDayStart": "2025-01-15", "full": "85.5",
             "withdrawal": "120.0", "injection": "20.0"},
        ]
        obs = gie._records_to_observations(records)
        by_series = {o["series"]: o for o in obs}
        assert set(by_series) == {
            gie.SERIES_STORAGE_PCT, gie.SERIES_NET_WITHDRAWAL
        }
        assert by_series[gie.SERIES_STORAGE_PCT]["value"] == pytest.approx(85.5)
        assert by_series[gie.SERIES_STORAGE_PCT]["unit"] == "%"
        # net withdrawal == withdrawal - injection.
        assert by_series[gie.SERIES_NET_WITHDRAWAL]["value"] == pytest.approx(100.0)
        assert by_series[gie.SERIES_NET_WITHDRAWAL]["unit"] == "GWh/d"
        for o in obs:
            assert o["valid_start"] == dt.datetime(2025, 1, 15, tzinfo=_UTC)
            assert o["resolution_minutes"] == 1440
            assert o["source"] == "agsi"
            assert o["zone"] is None

    def test_sentinel_fullness_skips_pct_series(self):
        """A '-' full value yields no storage_pct row, but net withdrawal stays."""
        records = [
            {"gasDayStart": "2025-01-15", "full": "-",
             "withdrawal": "120.0", "injection": "20.0"},
        ]
        obs = gie._records_to_observations(records)
        series = {o["series"] for o in obs}
        assert gie.SERIES_STORAGE_PCT not in series
        assert gie.SERIES_NET_WITHDRAWAL in series

    def test_missing_one_flow_skips_net_withdrawal(self):
        """Net withdrawal needs BOTH withdrawal and injection present."""
        records = [
            {"gasDayStart": "2025-01-15", "full": "80.0",
             "withdrawal": "120.0", "injection": "N/A"},
        ]
        obs = gie._records_to_observations(records)
        series = {o["series"] for o in obs}
        assert gie.SERIES_STORAGE_PCT in series
        assert gie.SERIES_NET_WITHDRAWAL not in series

    def test_thousands_separator_in_flow_values(self):
        records = [
            {"gasDayStart": "2025-01-15", "full": "80.0",
             "withdrawal": "1,250.5", "injection": "250.5"},
        ]
        obs = gie._records_to_observations(records)
        net = next(o for o in obs if o["series"] == gie.SERIES_NET_WITHDRAWAL)
        assert net["value"] == pytest.approx(1000.0)

    def test_unparseable_gas_day_skips_record(self):
        records = [{"gasDayStart": "not-a-date", "full": "80.0"}]
        assert gie._records_to_observations(records) == []


# ===========================================================================
# 5b. ENTSO-E exogenous decoding path (_exog_rows / _sum_wind_solar)
# ===========================================================================
class TestEntsoeExogRows:
    """Guard the _exog_rows and _fetch_load DataFrame->Series reduction (Fix #4).

    All tests run under -W error::DeprecationWarning (configured in pytest.ini /
    pyproject.toml or passed on the command line) to catch NumPy scalar-coercion
    regressions immediately.
    """

    def _make_series(self, values, freq="h"):
        idx = pd.date_range("2025-01-15", periods=len(values), freq=freq, tz="UTC")
        return pd.Series(values, index=idx, dtype=float)

    def test_exog_rows_from_series_values_are_python_floats(self):
        """_exog_rows on a Series produces rows whose 'value' is a Python float."""
        series = self._make_series([100.0, 200.0])
        rows = ent.EntsoeClient._exog_rows(Zone.NORD, series, "load_forecast", unit="MW")
        assert len(rows) == 2
        assert rows[0]["value"] == pytest.approx(100.0)
        assert rows[1]["value"] == pytest.approx(200.0)
        # Must be a plain Python float, not a numpy ndarray or numpy scalar.
        for row in rows:
            assert type(row["value"]) is float, (
                f"expected float, got {type(row['value'])}"
            )

    def test_exog_rows_resolution_and_tz(self):
        """_exog_rows infers resolution_minutes and stores tz-aware valid_start."""
        series = self._make_series([50.0], freq="h")
        rows = ent.EntsoeClient._exog_rows(Zone.NORD, series, "load_forecast", unit="MW")
        assert len(rows) == 1
        assert rows[0]["resolution_minutes"] == 60
        assert rows[0]["valid_start"].tzinfo is not None
        assert rows[0]["valid_start"].utcoffset() == dt.timedelta(0)

    def test_exog_rows_from_single_column_dataframe_values_are_python_floats(self):
        """Simulates Fix #4: a DataFrame is reduced to its numeric first column
        before being passed to _exog_rows, so 'value' must be a plain float."""
        idx = pd.date_range("2025-01-15", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({"Forecasted Load": [100.0, 200.0]}, index=idx)
        # Mimic the _fetch_load reduction introduced by Fix A.
        numeric = df.select_dtypes(include="number")
        assert not numeric.empty
        series = numeric.iloc[:, 0]
        rows = ent.EntsoeClient._exog_rows(Zone.NORD, series, "load_forecast", unit="MW")
        assert len(rows) == 2
        assert rows[0]["value"] == pytest.approx(100.0)
        assert rows[1]["value"] == pytest.approx(200.0)
        for row in rows:
            assert type(row["value"]) is float, (
                f"expected float, got {type(row['value'])}"
            )
        # valid_start must be tz-aware UTC.
        assert rows[0]["valid_start"] == dt.datetime(2025, 1, 15, 0, 0, tzinfo=_UTC)
        assert rows[1]["valid_start"] == dt.datetime(2025, 1, 15, 1, 0, tzinfo=_UTC)

    def test_sum_wind_solar_sums_rowwise(self):
        """_sum_wind_solar sums all numeric columns across axis=1."""
        idx = pd.date_range("2025-01-15", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"Solar": [10.0, 20.0, 30.0], "Wind Onshore": [5.0, 15.0, 25.0]},
            index=idx,
        )
        result = ent.EntsoeClient._sum_wind_solar(df)
        assert result is not None
        assert list(result) == pytest.approx([15.0, 35.0, 55.0])

    def test_sum_wind_solar_passthrough_series(self):
        """_sum_wind_solar returns a Series unchanged when passed a Series."""
        series = self._make_series([42.0, 43.0])
        result = ent.EntsoeClient._sum_wind_solar(series)
        assert result is series

    def test_sum_wind_solar_empty_df_returns_none(self):
        result = ent.EntsoeClient._sum_wind_solar(pd.DataFrame())
        assert result is None

    def test_exog_rows_skips_nan_values(self):
        """NaN entries must be silently dropped, not produce rows."""
        import math
        series = self._make_series([100.0, float("nan"), 300.0])
        rows = ent.EntsoeClient._exog_rows(Zone.NORD, series, "load_forecast", unit="MW")
        assert len(rows) == 2
        for row in rows:
            assert not math.isnan(row["value"])


# ===========================================================================
# 7. Weather: population-weighted temp + HDD/CDD + renormalisation
# ===========================================================================
from energy_prices.ingestion import weather_client as wx  # noqa: E402


class TestWeatherPopulationWeighted:
    def test_all_cities_present_weighted_mean(self):
        weights = wx._normalised_weights(wx.CITIES)
        assert sum(weights.values()) == pytest.approx(1.0)

        day = dt.date(2025, 1, 15)
        # Give every city the same temperature -> weighted mean equals it.
        per_city = {c.name: {day: 10.0} for c in wx.CITIES}
        weighted = wx._population_weighted(per_city, weights)
        assert weighted[day] == pytest.approx(10.0)

    def test_distinct_temps_match_explicit_weighted_average(self):
        weights = wx._normalised_weights(wx.CITIES)
        day = dt.date(2025, 1, 15)
        temps = {c.name: 5.0 + i for i, c in enumerate(wx.CITIES)}
        per_city = {name: {day: t} for name, t in temps.items()}
        weighted = wx._population_weighted(per_city, weights)
        expected = sum(weights[name] * t for name, t in temps.items())
        assert weighted[day] == pytest.approx(expected)

    def test_renormalises_when_one_city_missing(self):
        """A missing city-day must re-normalise weights over the PRESENT cities."""
        weights = wx._normalised_weights(wx.CITIES)
        day = dt.date(2025, 1, 15)
        # Milano (largest weight) reports nothing for this day; others report 10.
        per_city = {c.name: {day: 10.0} for c in wx.CITIES if c.name != "Milano"}
        per_city["Milano"] = {}  # missing for this day
        weighted = wx._population_weighted(per_city, weights)
        # All present cities are 10.0 -> renormalised mean must still be 10.0,
        # NOT diluted toward 0 by Milano's missing 30% weight.
        assert weighted[day] == pytest.approx(10.0)

    def test_renormalisation_value_with_distinct_present_temps(self):
        weights = wx._normalised_weights(wx.CITIES)
        day = dt.date(2025, 1, 15)
        present = {c.name: 20.0 for c in wx.CITIES if c.name != "Milano"}
        per_city = {name: {day: t} for name, t in present.items()}
        per_city["Milano"] = {}
        weighted = wx._population_weighted(per_city, weights)
        num = sum(weights[name] * t for name, t in present.items())
        denom = sum(weights[name] for name in present)
        assert weighted[day] == pytest.approx(num / denom)

    def test_day_with_no_data_is_omitted(self):
        weights = wx._normalised_weights(wx.CITIES)
        day = dt.date(2025, 1, 15)
        per_city = {c.name: {} for c in wx.CITIES}
        weighted = wx._population_weighted(per_city, weights)
        assert day not in weighted
        assert weighted == {}


class TestWeatherBuildRows:
    def test_hdd_cdd_derivation(self):
        # Cold day -> HDD positive, CDD zero.
        cold = dt.date(2025, 1, 15)
        warm = dt.date(2025, 7, 15)
        rows = wx._build_rows({cold: 5.0, warm: 30.0})
        by_key = {(r["series"], r["valid_start"]): r["value"] for r in rows}

        cold_mid = dt.datetime(2025, 1, 15, tzinfo=_UTC)
        warm_mid = dt.datetime(2025, 7, 15, tzinfo=_UTC)
        # HDD = max(0, 18 - tavg); CDD = max(0, tavg - 21).
        assert by_key[("temp_pop_it", cold_mid)] == pytest.approx(5.0)
        assert by_key[("hdd", cold_mid)] == pytest.approx(wx.HDD_BASE - 5.0)
        assert by_key[("cdd", cold_mid)] == pytest.approx(0.0)

        assert by_key[("hdd", warm_mid)] == pytest.approx(0.0)
        assert by_key[("cdd", warm_mid)] == pytest.approx(30.0 - wx.CDD_BASE)

    def test_three_series_per_day(self):
        rows = wx._build_rows({dt.date(2025, 1, 15): 10.0})
        assert {r["series"] for r in rows} == {"temp_pop_it", "hdd", "cdd"}
        assert len(rows) == 3
        for r in rows:
            assert r["resolution_minutes"] == 1440
            assert r["source"] == "open-meteo"
            assert r["zone"] is None
