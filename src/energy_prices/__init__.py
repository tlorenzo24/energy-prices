"""energy-prices: GME (Italian) electricity & gas price dashboard + forecasting service."""

__version__ = "0.1.0"

# Use the OS (Windows/macOS) trust store for TLS so HTTPS works behind corporate
# proxies / antivirus that inject their own root CA (which `certifi` does not
# know about). Safe and optional: no-op if `truststore` is not installed.
try:  # pragma: no cover - environment-dependent
    import truststore as _truststore

    _truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - never block import on TLS trust setup
    pass
