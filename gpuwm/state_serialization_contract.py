"""Shared prepared-state/restart serialization identity primitives.

This deliberately small module is part of the standalone RW-WPS package.
Prepared-cache export and the full forecast restart reader must serialize the
same state inventory and compute the same setup fingerprint without making the
preprocessor depend on the full forecast I/O implementation.
"""

from __future__ import annotations

import hashlib

import numpy as np


STATE_SERIALIZED_ATTRS = (
    "u", "v", "w", "thp", "php", "mup",
    "p", "al", "alt",
    "qv", "qc", "qr", "h_diabatic",
    "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng",
    "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh",
    "effc", "effr", "effi", "effs",
)

STATE_SETUP_ARRAYS = (
    "thb", "pb", "alb", "phb", "mub2d", "ht",
    "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f",
    "msft", "msfu", "msfv", "f", "e", "sina", "cosa",
    "dnw", "rdnw", "dn", "rdn", "fnp", "fnm", "znu", "znw",
)

STATE_SETUP_SCALARS = (
    "mub", "p_top", "cf1", "cf2", "cf3", "cfn", "cfn1",
    "has_msf", "rotational",
)


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _digest_array(digest, name: str, value) -> None:
    host = _host(value)
    digest.update(name.encode())
    digest.update(str(host.shape).encode())
    digest.update(str(host.dtype).encode())
    digest.update(host.tobytes())


def setup_fingerprint(state, *, error_type: type[Exception] = ValueError) -> str:
    """Hash deterministic setup state and attached LBC forcing tables."""
    digest = hashlib.sha256()
    for name in STATE_SETUP_ARRAYS:
        _digest_array(digest, name, getattr(state, name))
    for name in STATE_SETUP_SCALARS:
        value = getattr(state, name)
        if value is not None and not isinstance(value, bool):
            value = float(value)
        digest.update(f"{name}={value!r};".encode())
    nest_class = getattr(state, "_nest_restart_classification", None)
    if nest_class is not None:
        if nest_class != "REBUILT":
            raise error_type(
                f"unknown nest restart classification {nest_class!r}")
        digest.update(b"nest_tables=REBUILT;")
    else:
        boundaries = getattr(state, "lateral_boundaries", None)
        if boundaries is None:
            digest.update(b"lateral_boundaries=None;")
        else:
            digest.update(
                f"lbc:width={boundaries.spec_bdy_width};"
                f"spec={boundaries.spec_zone};relax={boundaries.relax_zone};"
                f"intervals={len(boundaries.intervals)};".encode())
            for interval in boundaries.intervals:
                digest.update(
                    f"[{interval.start_seconds!r},"
                    f"{interval.end_seconds!r}]".encode())
                for name in sorted(interval.fields):
                    boundary = interval.fields[name]
                    for side_name in ("west", "east", "south", "north"):
                        side = getattr(boundary, side_name)
                        _digest_array(
                            digest, f"{name}/{side_name}/value", side.value)
                        _digest_array(
                            digest, f"{name}/{side_name}/tendency",
                            side.tendency)
    return digest.hexdigest()


__all__ = [
    "STATE_SERIALIZED_ATTRS",
    "STATE_SETUP_ARRAYS",
    "STATE_SETUP_SCALARS",
    "setup_fingerprint",
]
