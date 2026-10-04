"""WPS lake-depth geography, sampled by the native static-field engine.

The row is WPS v4.6.0 GEOGRID.TBL.ARW:430-438. Its fill is 10 m; the
lake model's separate lakedepth_default option does not replace this row.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path


LAKE_DEPTH_ROW = {
    "name": "LAKE_DEPTH", "gcell": True, "gcell_ratio": 1.0,
    "search_depth": 5, "masked_water": False, "masked_land": True,
    "fill_missing": 10.0,
}


def with_lake_statics(selection, cfg):
    """Request the depth dataset only for a lake model that consumes it."""
    if (int(getattr(cfg, "sf_lake_physics", 0)) == 1
            and int(getattr(cfg, "use_lakedepth", 1)) == 1):
        return replace(selection, lake_depth=True)
    return selection


def build_lake_fields(grid, geog_root, *, landuse_path, halo=None,
                      coverage_report=None):
    """Sample the declared WPS dataset, preserving its mask and fill rules.

    Absent geography cannot stand in for a supplied depth field: WRF's
    default use_lakedepth=1 refuses a missing input dataset.
    """
    from . import rust_bridge
    from .build import HALO

    dataset = Path(geog_root) / "lake_depth"
    if not (dataset / "index").is_file():
        raise FileNotFoundError(
            "sf_lake_physics=1 with use_lakedepth=1 requires the WPS_GEOG "
            f"lake_depth dataset at {dataset}; no bathymetry can be inferred "
            "from land use. Run gpuwm fetch-geog --datasets lake_depth "
            "--source ncar --root with this GEOG root, or explicitly "
            "choose use_lakedepth=0.")
    bridge = rust_bridge.route("build_orographic")
    if bridge is None or not hasattr(grid, "_rust_sampling_handle"):
        raise RuntimeError(
            "LAKE_DEPTH requires the current native static-fields bridge "
            "for WPS average_gcell(1.0)+search(5); rebuild static-fields")
    request = {"landuse": str(landuse_path), "fields": [
        dict(LAKE_DEPTH_ROW, path=str(dataset))]}
    fieldset = bridge.build_orographic(
        grid._rust_sampling_handle(bridge), request,
        HALO if halo is None else int(halo))
    try:
        fields = bridge.fieldset_to_dict(fieldset)
        if coverage_report is not None:
            import json
            coverage = json.loads(
                bridge.field_coverage_json(fieldset, "LAKE_DEPTH"))
            coverage["field"] = "lake_depth"
            coverage["output_field"] = "LAKE_DEPTH"
            coverage_report["lake_depth"] = coverage
        return {"LAKE_DEPTH": fields["LAKE_DEPTH"]}
    finally:
        bridge.fieldset_free(fieldset)
