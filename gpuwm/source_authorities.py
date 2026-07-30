"""Immutable source-family authorities shipped with the RW-WPS wheel."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


_AUTHORITY_ROOT = Path(__file__).with_name("authorities")
_TWENTYCRV3_NAMES = MappingProxyType({
    "mapping": "rw-wps-20crv3-member-grib2.mapping.json",
    "composition": "rw-wps-20crv3-member-grib2.composition.json",
    "provenance": "rw-wps-20crv3-member-grib2.provenance.json",
})
_TWENTYCRV3_SHA256 = MappingProxyType({
    "mapping": "2e9877d51d9c993e83311c87236467b99ce9022638a343985da10ccd195efe09",
    "composition": "aa4f3fac03c09e8461c5e6c5e04a6bed48b5ad477babc4c75e8dd10fd92fe7b2",
    "provenance": "d1248e1b091f59841757a98a024cbe2868cebc25308f4eb4f9608e2c1755f3b1",
})


def twentycrv3_authorities() -> Mapping[str, Path]:
    """Resolve and byte-verify the exact packaged 20CRv3 authorities."""

    resolved: dict[str, Path] = {}
    for role, name in _TWENTYCRV3_NAMES.items():
        path = (_AUTHORITY_ROOT / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"packaged 20CRv3 {role} authority is missing: {path}"
            )
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = _TWENTYCRV3_SHA256[role]
        if observed != expected:
            raise RuntimeError(
                f"packaged 20CRv3 {role} authority hash differs: "
                f"expected {expected}, got {observed}"
            )
        resolved[role] = path
    return MappingProxyType(resolved)


def twentycrv3_authority_sha256() -> Mapping[str, str]:
    """Return the immutable SHA-256 contract without touching the filesystem."""

    return _TWENTYCRV3_SHA256


__all__ = ["twentycrv3_authorities", "twentycrv3_authority_sha256"]
