"""Bind prescribed smoke source bytes without processing profile values."""
from __future__ import annotations

SMOKE_IDENTITY_KEY = "rrtmg_smoke_manifest_identity"


def bind_smoke_source_identity(values, *, start_time=None, nz=None):
    """Add a checked source identity to a live config document in place.

    Empty is absent before any provider import or file read. A saved
    identity remains saved: digesting a stored checkpoint must not replace
    its original source bytes with the current file at the same path.
    """
    path = values.get("rrtmg_smoke_manifest", "")
    if not path:
        values.pop("rrtmg_smoke_manifest", None)
        values.pop(SMOKE_IDENTITY_KEY, None)
        return values
    if SMOKE_IDENTITY_KEY not in values:
        from gpuwm.core.rrtmg_smoke_manifest import describe_smoke_source
        values[SMOKE_IDENTITY_KEY] = describe_smoke_source(
            path, start_time=start_time,
            nz=int(values["nz"]) if nz is None else int(nz))
    return values
