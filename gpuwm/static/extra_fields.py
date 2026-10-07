"""Optional continuous source rows. All decode and sampling stays in Rust."""
from __future__ import annotations

import os
from pathlib import Path

from gpuwm.geog_assets import archive_for
from . import rust_bridge


def _dataset(root: Path, name: str) -> str:
    relative = Path(name)
    # Containment is LEXICAL.  A dataset linked into the root from a shared
    # WPS_GEOG tree -- the layout gpuwm fetch-geog stages -- resolves
    # outside the root and is a valid dataset; resolving before the check
    # refused every such root.  Only a name that climbs out is refused.
    path = Path(os.path.normpath(root / relative))
    if (relative.is_absolute() or ".." in relative.parts
            or not path.is_relative_to(root) or path == root):
        raise ValueError(f"dataset_path must name a directory under geog_root: {name!r}")
    if not (path / "index").is_file():
        try:
            selector = archive_for(relative.parts[0]).required_by[0]
        except ValueError:
            selector = name
        raise FileNotFoundError(
            f"missing WPS dataset {name!r} at {path}; stage geography "
            f"with gpuwm fetch-geog --datasets {selector} --root \"{root}\"")
    return str(path)


def build_extra_fields(grid, geog_root, specs: list[dict], *, halo=None):
    """Resolve plain rows and sample them on a ProjectedGrid.

    dataset_path (including water_mask.dataset_path) is relative to geog_root.
    The mask source should be the land-use directory selected for the native
    static build. Rows and dataset content identities belong in the prepared
    identity only when these rows are active.
    """
    if not specs:
        return {}
    root = Path(geog_root).resolve()
    resolved = []
    for source in specs:
        row = dict(source)
        row["dataset_path"] = _dataset(root, row["dataset_path"])
        if row.get("water_mask") is not None:
            rule = dict(row["water_mask"])
            rule["dataset_path"] = _dataset(root, rule["dataset_path"])
            row["water_mask"] = rule
        resolved.append(row)
    rust_bridge.require_extra_fields()
    return rust_bridge.build_extra_fields(
        grid._rust_sampling_handle(rust_bridge), resolved, halo)
