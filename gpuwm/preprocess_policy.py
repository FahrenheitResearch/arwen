"""Preparation routing shared by sizing and command composition; no GPU imports."""
from __future__ import annotations

from collections.abc import Mapping


def resolve_preprocess_backend(*, source: str, experiment=None, tables=None,
                               requested: str | None = None) -> str:
    """Honor an explicit backend; prepare GFS host-tiled experiments on CPU.

    The [tiles] declaration already records that full-domain device residency
    is avoidable. Its prepared GFS road must make that true during preparation
    as well as during integration. Other sources/contracts retain CUDA until
    their CPU preparation route has been explicitly covered. ``auto`` remains
    an explicit request for the existing runtime resolver, never a CPU claim.

    ``tables`` serves the source CLI's already captured TOML authority without
    importing forecast code. ``experiment`` serves callers that validated it.
    """
    if requested is not None:
        if not isinstance(requested, str) or requested not in {"cpu", "cuda", "auto"}:
            raise ValueError("preprocess backend must be cpu, cuda or auto")
        return requested
    if experiment is not None and tables is not None:
        raise ValueError("provide an experiment or TOML tables, not both")
    if str(source).strip().lower() != "gfs":
        return "cuda"
    if experiment is not None:
        default = getattr(experiment, "tiles", None)
        for domain in getattr(experiment, "domains", ()):
            choice = getattr(domain, "tiles", None)
            choice = default if choice is None else choice
            if (getattr(choice, "mode", "off") in ("auto", "on")
                    and getattr(choice, "store", "host") == "host"):
                return "cpu"
    elif isinstance(tables, Mapping):
        default = tables.get("tiles")
        for domain in tables.get("domain", ()):
            if not isinstance(domain, Mapping):
                continue
            choice = domain.get("tiles")
            choice = default if choice is None else choice
            if (isinstance(choice, Mapping)
                    and choice.get("mode", "off") in ("auto", "on")
                    and choice.get("store", "host") == "host"):
                return "cpu"
    return "cuda"
