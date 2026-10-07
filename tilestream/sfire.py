"""SFIRE compute-window setup and continuation clocks for atmospheric tiles."""
from __future__ import annotations

GRID_CLOCKS = ("time_seconds", "step_count", "moisture_initialized",
               "moisture_lasttime", "moisture_nexttime", "perimeter_applied")
GRID_DIAGNOSTICS = ("last_cfl_bound", "last_cfl_exceeded", "last_ignited_counts")


def tile_static(owner, cfg):
    """Allocate neutral operands that the native carrier gather replaces."""
    import cupy as cp
    fine = (int(cfg.ny) * owner.sr_y, int(cfg.nx) * owner.sr_x)
    result = {name: cp.zeros(fine, cp.float32) for name in
              ("ZSF", "DZDXF", "DZDYF", "FMC_G", "FZ0")}
    if owner.spotting is not None:
        result["_SPOTTING_OWNER"] = owner.spotting
    result["NFUEL_CAT"] = cp.full(fine, int(cfg.fire_fuel_cat), cp.float32)
    result["FZ0"].fill(0.001)
    result["LFN_HIST"] = cp.zeros(fine, cp.float32)
    result["LFN_TIME"] = owner.data["lfn_time"]
    if owner.geometry["coordinate_mode"] == "geographic":
        result.update(FXLAT=cp.zeros(fine, cp.float32), FXLONG=cp.zeros(fine, cp.float32),
                      FIRE_COORDINATE_MODE="geographic", CEN_LAT=0.0)
    if cfg.fire_is_real_perim:
        if "historical_tign" in owner.grid.data:
            result["HISTORICAL_TIGN"] = cp.zeros(fine, cp.float32)
    if owner.grid.moisture is not None:
        for key, value in owner.grid.moisture.arrays().items():
            shape = tuple(value.shape[:-2]) + (int(cfg.ny), int(cfg.nx))
            result["FMC_TEND" if key == "fmc_lag" else key.upper()] = cp.zeros(shape, cp.float32)
    return result


def configure_tile(state, spec):
    """Locate physical fire boundaries inside an atmospheric compute window."""
    fire = getattr(getattr(state, "physics", None), "fire", None)
    if fire is None:
        return
    rx, ry = fire.sr_x, fire.sr_y
    xl, xh = max(0, -spec.ci0), min(spec.cnx, spec.nx - spec.ci0)
    yl, yh = max(0, -spec.cj0), min(spec.cny, spec.ny - spec.cj0)
    fire._tile_spec = spec
    fire._tile_atmos_domain = (1 + xl, xh, 1 + yl, yh)
    fire.grid.domain = (1 + xl * rx, xh * rx, 1 + yl * ry, yh * ry)
    fire.grid.tiles = (fire.grid.domain,)
    fire.grid.owned_domain = (1 + (spec.i0 - spec.ci0) * rx, (spec.i1 - spec.ci0) * rx,
                              1 + (spec.j0 - spec.cj0) * ry, (spec.j1 - spec.cj0) * ry)
    fire.grid.global_guard_domain = (1 - spec.ci0 * rx, (spec.nx - spec.ci0) * rx,
                                     1 - spec.cj0 * ry, (spec.ny - spec.cj0) * ry)
    fire._tile_exchange_domain = (max(0, 1 - spec.ci0), min(spec.cnx - 1, spec.nx - 2 - spec.ci0),
                                  max(0, 1 - spec.cj0), min(spec.cny - 1, spec.ny - 2 - spec.cj0))


def clock_values(fire):
    return {name: getattr(fire.grid, name) for name in GRID_CLOCKS}


def restore_clocks(fire, values):
    for name in GRID_CLOCKS:
        setattr(fire.grid, name, values[name])


def header_values(driver):
    """Global immutable setup with this step's actual continuation scalars."""
    import copy
    from gpuwm.io.restart import _fire_header
    fire = driver.fire
    result = copy.deepcopy(getattr(fire, "_streamed_header", None) or _fire_header(driver))
    result["fire"]["grid"].update(clock_values(fire))
    result["fire"]["grid"].update({key: getattr(fire.grid, key) for key in GRID_DIAGNOSTICS})
    result["fire"]["grid"]["last_ignited_counts"] = list(fire.grid.last_ignited_counts)
    return result


def fold_diagnostics(records, values):
    if not records:
        return
    counts = [tuple(row["last_ignited_counts"]) for row in records]
    if len({len(row) for row in counts}) != 1:
        raise ValueError("SFIRE tile ignition inventories differ")
    target = values["fire_header"]["fire"]["grid"]
    target["last_cfl_bound"] = min(row["last_cfl_bound"] for row in records)
    target["last_cfl_exceeded"] = any(row["last_cfl_exceeded"] for row in records)
    target["last_ignited_counts"] = [sum(row[k] for row in counts) for k in range(len(counts[0]))]


def dependency_halo(cfg):
    if int(getattr(cfg, "ifire", 0)) != 2:
        return 0
    # Three ENO stages per spread/reinitialization, radius at most two.
    fine_reach = 6 * (1 + (cfg.fire_lsm_reinit_iter if cfg.fire_lsm_reinit else 0)) + 3
    refinement = min(int(cfg.sr_x), int(cfg.sr_y))
    return (fine_reach + refinement - 1) // refinement
