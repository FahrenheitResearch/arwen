"""Carry supplied lake mask and depth to the selected column model."""
from __future__ import annotations


_LAKE_INPUTS = {"LAKE_DEPTH": "lake_depth", "LAKE_DEPTH_FLAG": "lake_depth_flag",
                "LAKEMASK": "lakemask", "LAKEFLAG": "lake_mask_flag"}


def lake_physics_inputs(cfg, fields):
    """The off route reads nothing; the model validates required depth input."""
    if int(getattr(cfg, "sf_lake_physics", 0)) != 1:
        return {}
    return {keyword: fields.get(name) for name, keyword in _LAKE_INPUTS.items()}


def wrfinput_lake_physics_inputs(restored, cfg):
    """Read optional lake inputs through Rust, preserving WRF's axes."""
    if int(getattr(cfg, "sf_lake_physics", 0)) != 1:
        return {}
    fields = {name: restored.raw[name] for name in _LAKE_INPUTS
              if name in restored.raw}
    missing = set(_LAKE_INPUTS) - fields.keys()
    if missing:
        from gpuwm import netcdf_bridge
        from gpuwm.ingest.wrfinput import _read_numeric
        with netcdf_bridge.open_dataset(restored.path) as dataset:
            for name in _LAKE_INPUTS:
                if name not in missing:
                    continue
                variable = dataset.variables.get(name)
                if variable is None:
                    continue
                dimensions = tuple(variable.dimensions)
                if dimensions and dimensions[0] == "Time":
                    dimensions = dimensions[1:]
                expected = () if name in ("LAKE_DEPTH_FLAG", "LAKEFLAG") else (
                    "south_north", "west_east")
                if dimensions != expected:
                    raise ValueError(
                        f"{name} axes {dimensions} != {expected}; lake "
                        "inputs would be assigned to different columns")
                fields[name] = _read_numeric(variable)
    return lake_physics_inputs(cfg, fields)
