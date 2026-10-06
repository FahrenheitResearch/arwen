"""Acquire the exact WRF 3.9 fork coefficient set without changing its pins.

The measured oracle set has no verified public binary publisher. It can ship
under gpuwm-data/data/thompson/wrf39-noaa, come from a pinned local oracle
build, or come from an explicitly selected mirror. Without those sources,
the runtime builds unchanged, hash-pinned public Fortran inputs with a pinned
portable libc on a local CPU. Every route enforces the same canonical manifest
before installation. Different official HRRR floating-point bytes are refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import shutil

from gpuwm import fetch_guard
from gpuwm.core.thompson_contract import (
    FORK_REFERENCE_SOURCE, FORK_TABLE_ASSETS, FORK_TABLE_SET_ID,
    validate_table_assets,
)
from gpuwm.table_assets import (
    TableAssetError, classify_assets, fetch_asset_from_dir, fetch_asset_from_url,
)

FORK_TABLE_SOURCE_ROOT_ENV = "GPUWM_THOMPSON_FORK_TABLE_SOURCE_ROOT"
FORK_TABLE_ASSET_URL_BASE_ENV = "GPUWM_THOMPSON_FORK_TABLE_ASSET_URL_BASE"
_VERIFIED_ROOTS: dict[str, tuple] = {}


def _signature(root: Path) -> tuple | None:
    try:
        return tuple((asset.filename, (root / asset.filename).stat().st_size,
                      (root / asset.filename).stat().st_mtime_ns,
                      (root / asset.filename).stat().st_ctime_ns,
                      (root / asset.filename).stat().st_ino)
                     for asset in FORK_TABLE_ASSETS)
    except OSError:
        return None


def _packaged_source() -> Path | None:
    try:
        from gpuwm_data import data_root
    except ImportError:
        return None
    candidate = data_root() / "thompson" / "wrf39-noaa"
    return candidate if candidate.is_dir() else None


def _build_source(work: Path, log: Path) -> Path:
    from gpuwm.thompson_fork_build import build_canonical_fork_tables
    return build_canonical_fork_tables(work, log_path=log)


def _cleanup_build(work: Path, root: Path) -> None:
    # This unique work directory was created by this call, never supplied by
    # the operator. Record every spent file and size before removing it.
    if work.parent.resolve() != root.resolve() or not work.name.startswith(".fork-build-"):
        raise RuntimeError("Fork build cleanup escaped its owned cache directory")
    if not work.exists():
        return
    files = []
    for directory, subdirs, names in os.walk(work, followlinks=False):
        for name in list(subdirs):
            candidate = Path(directory) / name
            if candidate.is_symlink():
                files.append({"path": str(candidate.relative_to(work)), "bytes": candidate.lstat().st_size})
                subdirs.remove(name)
        for name in names:
            candidate = Path(directory) / name
            files.append({"path": str(candidate.relative_to(work)), "bytes": candidate.lstat().st_size})
    fetch_guard.atomic_write_text(root / "fork-table-build-cleanup.json", json.dumps(
        {"owned_work": work.name, "files": files, "logical_bytes": sum(item["bytes"] for item in files)}, indent=2) + "\n")
    shutil.rmtree(work)


def ensure_thompson_fork_tables(root=None, *, source_dir=None) -> Path:
    """Verify or acquire the selected fork's complete set under one root lock.

    Existing wrong bytes are refused without overwrite. A named source wins
    over packaged data and a mirror and must contain the complete canonical
    set. Transfers use unique temporary files and per-file atomic publication;
    a failed transfer can leave verified files to resume, never wrong bytes.
    All consumers enter this transaction before reading the selected set.
    """
    if root is None:
        from gpuwm.physics_compat import thompson_fork_table_root
        root = thompson_fork_table_root()
    root = Path(root)
    with fetch_guard.hold("fetch-tables", root):
        key = str(root.resolve())
        signature = _signature(root)
        if signature is not None and _VERIFIED_ROOTS.get(key) == signature:
            return root
        _valid, invalid, absent = classify_assets(root, FORK_TABLE_ASSETS)
        if invalid:
            raise TableAssetError(
                "fork Thompson cache contains different bytes; refusing "
                "without overwrite: " + "; ".join(invalid))
        if not absent:
            _VERIFIED_ROOTS[key] = signature
            return root
        source = source_dir
        if source is None:
            source = os.environ.get(FORK_TABLE_SOURCE_ROOT_ENV) or _packaged_source()
        mirror = os.environ.get(FORK_TABLE_ASSET_URL_BASE_ENV, "").strip().rstrip("/")
        owned_work = None
        if source is None and not mirror:
            root.mkdir(parents=True, exist_ok=True)
            owned_work = root / (".fork-build-" + secrets.token_hex(8))
            try:
                source = _build_source(owned_work, root / "fork-table-source-build.log")
            except (OSError, ValueError, TableAssetError) as error:
                _cleanup_build(owned_work, root)
                raise FileNotFoundError(
                    "thompson_version='wrf_39_noaa' canonical table acquisition "
                    f"failed at {root}: {error}. Offline: gpuwm fetch-tables "
                    "--thompson-fork --thompson-fork-only --from DIR; alternatively "
                    f"set {FORK_TABLE_SOURCE_ROOT_ENV} or {FORK_TABLE_ASSET_URL_BASE_ENV}. "
                    "The exact oracle route is tools/thompson_fork_oracle/build.sh; "
                    "different generated bytes are refused, never re-pinned.") from error
        if source is not None:
            source = Path(source)
            try:
                validate_table_assets(source, FORK_TABLE_ASSETS)
            except (OSError, ValueError) as error:
                if owned_work is not None:
                    _cleanup_build(owned_work, root)
                raise TableAssetError(
                    f"fork Thompson source {source} is not the canonical "
                    f"complete set: {error}") from error
        try:
            root.mkdir(parents=True, exist_ok=True)
            for asset in absent:
                if source is not None:
                    fetch_asset_from_dir(root, asset, source)
                else:
                    fetch_asset_from_url(root, asset, mirror + "/" + asset.filename)
            validate_table_assets(root, FORK_TABLE_ASSETS)
            receipt = {
                "schema": 1, "table_set": FORK_TABLE_SET_ID,
                "reference_source": FORK_REFERENCE_SOURCE,
                "acquired_from": "pinned-public-source-build" if owned_work is not None else (str(source) if source is not None else mirror),
                "assets": [{"filename": asset.filename, "bytes": asset.bytes,
                            "sha256": asset.sha256} for asset in FORK_TABLE_ASSETS],
            }
            if owned_work is not None:
                receipt["source_build"] = json.loads((owned_work / "source-build-receipt.json").read_text())
            fetch_guard.atomic_write_text(
                root / "fork-table-acquisition.json", json.dumps(receipt, indent=2) + "\n")
            _VERIFIED_ROOTS[key] = _signature(root)
            return root
        finally:
            if owned_work is not None:
                _cleanup_build(owned_work, root)


def stage_thompson_fork_tables(source_dir=None, root=None) -> int:
    """CLI acquisition with the same transaction the runtime uses."""
    try:
        staged = ensure_thompson_fork_tables(root, source_dir=source_dir)
    except (OSError, ValueError, TableAssetError, fetch_guard.FetchLockBusy) as error:
        print(f"gpuwm fetch-tables --thompson-fork: REFUSED: {error}")
        return 2
    print(f"gpuwm fetch-tables --thompson-fork: verified {staged} "
          f"({FORK_TABLE_SET_ID})")
    return 0
