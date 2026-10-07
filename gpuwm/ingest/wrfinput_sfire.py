"""Cold-start input fields from WRF v4.7.1 Registry/registry.fire.

Only the registry's input fields enter this cold-start door. Evolving RK
buffers and firebrand particles belong to the checkpoint door, which binds
their complete inventory and clocks before restoring any word.
"""
from __future__ import annotations

FINE_DIMS = ("south_north_subgrid", "west_east_subgrid")
MASS_DIMS = ("south_north", "west_east")
SFIRE_INPUT_DIMENSIONS = {
    **{name: FINE_DIMS for name in (
        "LFN_HIST", "NFUEL_CAT", "ZSF", "DZDXF", "DZDYF", "FMC_G",
        "FXLONG", "FXLAT", "FZ0")},
    "LFN_TIME": ("i_lfn_history",),
    "FMC_GC": ("fuel_moisture_classes", *MASS_DIMS),
    "FMEP": ("fuel_moisture_extended_parameters", *MASS_DIMS),
}
# WRF's generated dimension spelling is truncated by its native IO layer.
# Accept only this observed alias and only for the registry FMEP field.
FMEP_DIMENSION_ALIAS = "fuel_moisture_extended_paramete"
FMC_GC_DIMENSION_ALIAS = "fuel_moisture_classes_stag"


def fire_input_extents(cfg, atmospheric_extents):
    """Pin WRF's terminal refinement extension to the producing namelist."""
    if int(getattr(cfg, "ifire", 0)) != 2:
        return {}
    return {
        "west_east_subgrid": (int(atmospheric_extents["west_east"]) + 1) * int(cfg.sr_x),
        "south_north_subgrid": (int(atmospheric_extents["south_north"]) + 1) * int(cfg.sr_y),
        "i_lfn_history": 1,
        "fuel_moisture_classes": int(cfg.nfmc),
        "fuel_moisture_extended_parameters": 2,
    }


def fire_static_fields(restored, cfg):
    """Pass native fine static and moisture operands to their real consumer."""
    if int(getattr(cfg, "ifire", 0)) != 2:
        return None
    fields = {name: restored.raw[name] for name in SFIRE_INPUT_DIMENSIONS
              if name in restored.raw}
    if cfg.fire_static and fields:
        raise ValueError("WRF fine fire input and fire_static both supply SFIRE setup; "
                         "select one bound static authority")
    if not fields:
        return None
    attrs = restored.global_attributes
    for name in ("MAP_PROJ", "CEN_LAT", "FIRE_COORDINATE_MODE"):
        if name in attrs:
            fields[name] = attrs[name]
    return fields
