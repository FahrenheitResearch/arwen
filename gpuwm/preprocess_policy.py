"""Preparation routing shared by sizing and command composition; no GPU imports."""
from __future__ import annotations

from collections.abc import Mapping

#: The sources whose CPU preparation road is covered end to end, so a
#: ``[tiles]`` host-store declaration can be honoured during preparation as
#: well as during integration.  A TABLE, not a per-source branch: adding a
#: source whose CPU road has been covered is one entry here and nothing
#: else.  Every source absent from it resolves an unrequested backend to
#: ``auto``, as the docstring below states: the door prices its
#: preparation and prepares on the card only when that price fits.
#:
#: ``met_em`` joins ``gfs`` because the met_em route already prepares one
#: domain and one forcing interval at a time and releases each
#: (``gpuwm/metem_forecast.py`` per-interval ``del``/``gc.collect``), and
#: ``gpuwm.ingest.real.initialize_real`` already accepts the resolved
#: backend, so its CPU road needs no new ingest code.
CPU_PREPARED_SOURCES: tuple[str, ...] = ("gfs", "met_em")

#: Why this policy prepares a host-tiled experiment on the CPU.  It travels
#: with the choice into the preparation receipt's ``selection`` block and
#: into the one line the preparation prints, so a CPU preparation the
#: reader did not ask for names the declaration that asked for it.
HOST_TILED_CPU_REASON = (
    'the configuration keeps its tiled domain state in host memory '
    '([tiles] store = "host"), so preparation stays off the card')


def resolve_preprocess_backend(*, source: str, experiment=None, tables=None,
                               requested: str | None = None) -> str:
    """The backend :func:`preprocess_backend_choice` selects, without its reason."""
    return preprocess_backend_choice(
        source=source, experiment=experiment, tables=tables,
        requested=requested)[0]


def preprocess_backend_choice(*, source: str, experiment=None, tables=None,
                              requested: str | None = None
                              ) -> tuple[str, str | None]:
    """Honor an explicit backend; prepare host-tiled experiments on CPU.

    Returns ``(backend, reason)``.  ``reason`` is :data:`HOST_TILED_CPU_REASON`
    when the host-tiling declaration, not the caller, chose the CPU, and
    ``None`` for every other answer.

    Unrequested and not host-tiled, the answer is ``auto`` (fixed means
    default): the door then prices its preparation and prepares on the
    card only when that price fits the card's free memory
    (:func:`gpuwm.ingest.preprocess_backend.admit_preparation`).  It used
    to be ``cuda``, which is a request the fit may only refuse, never
    redirect.

    The [tiles] declaration already records that full-domain device residency
    is avoidable. The prepared road of every source in
    :data:`CPU_PREPARED_SOURCES` must make that true during preparation
    as well as during integration. Other sources resolve an unrequested
    backend to ``auto`` whatever ``[tiles]`` declares, until their CPU
    preparation route has been explicitly covered. ``auto`` is a request for
    the runtime resolver, never a CPU claim.

    ``tables`` serves the source CLI's already captured TOML authority without
    importing forecast code. ``experiment`` serves callers that validated it.
    """
    if requested is not None:
        if not isinstance(requested, str) or requested not in {"cpu", "cuda", "auto"}:
            raise ValueError("preprocess backend must be cpu, cuda or auto")
        return requested, None
    if experiment is not None and tables is not None:
        raise ValueError("provide an experiment or TOML tables, not both")
    if str(source).strip().lower() not in CPU_PREPARED_SOURCES:
        return "auto", None
    if experiment is not None:
        default = getattr(experiment, "tiles", None)
        for domain in getattr(experiment, "domains", ()):
            choice = getattr(domain, "tiles", None)
            choice = default if choice is None else choice
            if (getattr(choice, "mode", "off") in ("auto", "on")
                    and getattr(choice, "store", "host") == "host"):
                return "cpu", HOST_TILED_CPU_REASON
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
                return "cpu", HOST_TILED_CPU_REASON
    return "auto", None
