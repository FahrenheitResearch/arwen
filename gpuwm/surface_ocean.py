"""Surface-ocean metadata shared by config, input, runtime and restart."""
from functools import lru_cache
import json
from pathlib import Path

@lru_cache(maxsize=1)
def ocean_option(option=1):
    table = json.loads((Path(__file__).parent / "data/surface_ocean.json").read_text(encoding="utf-8"))
    return table["options"][str(option)]

def ocean_fields(cfg):
    option = int(getattr(cfg, "sf_ocean_physics", 0) or 0)
    return tuple(ocean_option(option)["state_fields"]) if option else ()

def ocean_namelist_defaults():
    return dict(ocean_option()["namelist_defaults"])

def drop_inert_ocean(values):
    if not values.get("sf_ocean_physics", 0):
        for name in ("sf_ocean_physics", *ocean_namelist_defaults()):
            values.pop(name, None)

def ocean_input_fields(cfg, fields):
    if not getattr(cfg, "sf_ocean_physics", 0):
        return {}
    return {"ocean_initial": {name.lower(): fields[name] for name in ocean_option()["input_fields"] if name in fields}}

def validate_ocean_inputs(cfg, initial):
    if not cfg.sf_ocean_physics:
        return
    initial = initial or {}
    if cfg.oml_hml0 <= 0 and initial.get("h0ml") is None:
        raise ValueError(f"sf_ocean_physics=1 with oml_hml0={cfg.oml_hml0} requires a 2D H0ML input: otherwise mixed-layer depth is uninitialised")
    if cfg.oml_hml0 < 0:
        missing = [name.upper() for name in ("huml", "hvml", "tmoml") if initial.get(name) is None]
        if missing:
            raise ValueError("oml_hml0<0 requires supplied " + ", ".join(missing) + ": WRF's 2D branch preserves currents and the ocean mean temperature; these states would be uninitialised")
