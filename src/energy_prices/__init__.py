"""energy-prices: GME (Italian) electricity & gas price dashboard + forecasting service."""

__version__ = "0.1.0"

# Force UTF-8 on stdout/stderr so emoji / € / accented log+console output never
# crash on a legacy Windows code page (cp1252). Must run before rich's Console
# is constructed (it caches the stream encoding at creation).
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    try:  # pragma: no cover - environment-dependent
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - older/odd streams lack reconfigure
        pass

# Use the OS (Windows/macOS) trust store for TLS so HTTPS works behind corporate
# proxies / antivirus that inject their own root CA (which `certifi` does not
# know about). Safe and optional: no-op if `truststore` is not installed.
try:  # pragma: no cover - environment-dependent
    import truststore as _truststore

    _truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - never block import on TLS trust setup
    pass

# `truststore` covers Python's `ssl` (so `requests` works), but NOT `curl_cffi`
# (used by yfinance), which embeds its own libcurl. Build a combined CA bundle
# from the OS trust store and point libcurl at it via CURL_CA_BUNDLE so TTF
# downloads work behind the same intercepting proxy/AV CA.
try:  # pragma: no cover - environment-dependent
    from energy_prices._tls import ensure_curl_ca_bundle as _ensure_curl_ca_bundle

    _ensure_curl_ca_bundle()
except Exception:  # noqa: BLE001 - never block import on TLS trust setup
    pass
