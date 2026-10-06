"""Pinned input assets acquired by a configured forcing fetch."""
from __future__ import annotations

import os
from pathlib import Path


def wif_fetch_domains(experiment, fetch_hints) -> tuple[int, ...]:
    """Domains whose fetch can supply the default monthly aerosol dataset.

    Named files and named climatology directories remain authoritative.
    A working-directory dataset also keeps its existing ingest path. Only
    the canonical table cache is acquired or verified automatically.
    """
    if not fetch_hints or not fetch_hints.get("source") or fetch_hints.get("wif") is False:
        return ()
    from gpuwm.ingest.wif_climatology import (
        WIF_CLIMATOLOGY_PATH_ENV, WIF_CLIMATOLOGY_ROOT_ENV,
        resolve_wif_climatology)
    from gpuwm.ingest.wif_dataset import WIF_DATASET_FILE, resolve_wif_data_root

    if os.environ.get(WIF_CLIMATOLOGY_PATH_ENV) or os.environ.get(WIF_CLIMATOLOGY_ROOT_ENV):
        return ()
    cache_file = resolve_wif_data_root() / WIF_DATASET_FILE
    requested = []
    for domain in experiment.domains:
        cfg = domain.run
        if cfg.mp_physics != 28 or cfg.wif_climatology_path:
            continue
        source = cfg.mp28_aerosol_source
        surface_monthly = cfg.use_rap_aero_icbc
        if source == "synthetic" or (source == "analysis" and not surface_monthly):
            continue
        if not cfg.specified and not surface_monthly:
            continue
        resolution = resolve_wif_climatology()
        if resolution.resolved and Path(resolution.path) != cache_file:
            continue
        requested.append(int(domain.grid_id))
    return tuple(requested)


def wif_fetch_resolution(domains: tuple[int, ...]) -> dict:
    """The pending acquisition plan, with no download or cache creation."""
    from gpuwm.ingest.wif_dataset import WIF_DATASET_ASSET, resolve_wif_data_root

    asset = WIF_DATASET_ASSET
    return {
        "scope": "fetch", "key": "wif_climatology",
        "value": str(resolve_wif_data_root() / asset.filename),
        "basis": "configured forcing fetch and selected aerosol physics",
        "domains": list(domains), "bytes": asset.bytes, "sha256": asset.sha256,
        "note": "The fetch stages the monthly aerosol dataset before forcing "
                "transfer and preparation, reusing its verified cache. Offline: "
                "gpuwm fetch-tables --wif --from DIR.",
    }
