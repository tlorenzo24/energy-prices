"""TLS trust setup for HTTP clients that bypass Python's ``ssl`` module.

``truststore`` (injected in :mod:`energy_prices.__init__`) makes the stdlib
``ssl`` module — and therefore ``requests`` — use the OS trust store, which is
what we need behind a corporate proxy / antivirus that injects its own root CA.

But ``yfinance`` talks to Yahoo through ``curl_cffi``, which embeds its own
libcurl and a bundled ``certifi`` CA set; it never touches Python's ``ssl``, so
``truststore`` does not help it and TLS fails with *curl (60) unable to get
local issuer certificate*.

The fix here is to materialise a combined PEM bundle (``certifi`` ∪ the Windows
ROOT/CA stores, which include the intercepting corporate CA) on disk and point
libcurl at it via ``CURL_CA_BUNDLE``. This is the same trick ``pip-system-certs``
uses. It is a no-op on platforms without ``ssl.enum_certificates`` (non-Windows)
and never raises — TLS trust setup must not block import.
"""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache the generated bundle next to the package's data directory (repo root).
_BUNDLE_PATH = Path(__file__).resolve().parents[2] / "data" / "ca-bundle.pem"

# Only libcurl's CA-bundle var. We deliberately do NOT set SSL_CERT_FILE: Python
# TLS already goes through `truststore` (Windows store), and pointing OpenSSL at
# this concatenated file instead can make it fail with "[X509] PEM lib" on certs
# its parser rejects — breaking the working `requests` path (GME). curl_cffi only
# needs CURL_CA_BUNDLE.
_CA_ENV_VARS = ("CURL_CA_BUNDLE",)


def _windows_ca_pems() -> list[bytes]:
    """Return PEM-encoded certs from the Windows ROOT + CA stores (DER → PEM).

    ``ssl.enum_certificates`` only exists on Windows; elsewhere this returns [].
    """
    enum = getattr(ssl, "enum_certificates", None)
    if enum is None:
        return []
    pems: list[bytes] = []
    seen: set[bytes] = set()
    for store in ("ROOT", "CA"):
        try:
            for cert_bytes, encoding, _trust in enum(store):
                if encoding == "x509_asn" and cert_bytes not in seen:
                    seen.add(cert_bytes)
                    pems.append(ssl.DER_cert_to_PEM_cert(cert_bytes).encode("ascii"))
        except Exception as exc:  # noqa: BLE001 - per-store best effort
            logger.debug("Could not enumerate Windows cert store %s: %s", store, exc)
    return pems


def _build_bundle() -> bytes | None:
    """Concatenate certifi's bundle with the OS-store certs. None if unavailable."""
    try:
        import certifi

        base = Path(certifi.where()).read_bytes()
    except Exception as exc:  # noqa: BLE001 - certifi should exist, but be safe
        logger.debug("certifi unavailable for CA bundle: %s", exc)
        base = b""

    extra = _windows_ca_pems()
    if not extra and not base:
        return None
    return base + b"\n" + b"\n".join(extra)


def ensure_curl_ca_bundle() -> str | None:
    """Write the combined CA bundle and export it to the curl/OpenSSL env vars.

    Idempotent: rewrites the on-disk file only when its contents change, and
    never overrides a CA-bundle env var the user has already set. Returns the
    bundle path, or None if no bundle could be built (e.g. non-Windows without
    certifi). Safe to call at import time — swallows all errors.
    """
    try:
        # Respect an explicit user override for the primary curl var.
        if os.environ.get("CURL_CA_BUNDLE"):
            return os.environ["CURL_CA_BUNDLE"]

        bundle = _build_bundle()
        if not bundle:
            return None

        _BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not (_BUNDLE_PATH.exists() and _BUNDLE_PATH.read_bytes() == bundle):
            _BUNDLE_PATH.write_bytes(bundle)

        path = str(_BUNDLE_PATH)
        for var in _CA_ENV_VARS:
            os.environ.setdefault(var, path)
        logger.debug("curl/OpenSSL CA bundle ready at %s", path)
        return path
    except Exception as exc:  # noqa: BLE001 - never block import on TLS setup
        logger.debug("ensure_curl_ca_bundle failed: %s", exc)
        return None
