"""Endogenous bulk WRF-SFIRE smoke, before the tracer transport step.

Native source is consumed-step fraction times FGIP, without a second dt
or the heat routine's dry-fuel correction. It is bulk smoke, not PM2.5.
The native g/kg-air increment is converted once to AQ ug/kg-dry storage.
"""
from __future__ import annotations

import math
from functools import lru_cache
import numpy as np

from gpuwm.core.chem_context import ChemAllocation

KEY = "emission.sfire"
TIMING = "pre_transport"
LEDGER = "emitted"
REQUIRES = ("rho", "dryrho", "dz8w", "z_at_w", "msftx", "msfty")
# sfire_atm is already part of the ifire=2 source compositions and budget.
KERNEL_MODULES = ("chem_sfire",)
MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
ALLOCATES = (
    ChemAllocation("sfire_native_increment", "3d", restart="rebuild"),
    ChemAllocation("sfire_native_output", "3d", restart="rebuild", output_name="fire_smoke",
        units="g_smoke/kg_air", description="WRF-SFIRE bulk smoke tracer"),
    ChemAllocation("sfire_source_expected", "rows_2d", dtype="float64",
        output_name="SFIRE_SMOKE_FUEL_SOURCE", units="kg",
        description="Accumulated native fuel-derived bulk smoke source per atmospheric cell"),
    ChemAllocation("sfire_source_injected", "rows_2d", dtype="float64",
        output_name="SFIRE_SMOKE_INJECTED", units="kg",
        description="Accumulated bulk smoke added using dry density and layer thickness"),
    ChemAllocation("sfire_source_areal", "rows_2d", output_name="SFIRE_SMOKE_EMITTED", units="ug m-2",
        description="Accumulated bulk SFIRE smoke added per physical cell area"),
    ChemAllocation("sfire_source_lasttime", "2d", dtype="float64"),
)


def module_source(kernel_dir=None):
    from gpuwm.core.kernels import module_source as source
    return source("chem_sfire") if kernel_dir is None else source("chem_sfire", kernel_dir=kernel_dir)


def rows(table):
    return table.rows_for(KEY)


def pre_transport_active(state, cfg):
    return getattr(getattr(state, "physics", None), "fire", None) is not None


def refusal(cfg):
    if int(getattr(cfg, "ifire", 0)) not in (0, 2):
        return "emission.sfire accepts an ifire=2 source or a passive ifire=0 nest carrier"
    return None


def init(ctx):
    active = rows(ctx.table)
    if any(row.units != "ug kg-1" for row in active):
        raise ValueError("emission.sfire rows must carry ug kg-1 of dry air")
    fire = getattr(getattr(ctx.state, "physics", None), "fire", None)
    last = float(ctx.diag["sfire_source_lasttime"][0, 0].item())
    clock = 0.0 if fire is None else fire.grid.time_seconds
    if not math.isfinite(last) or last < 0 or last > clock:
        raise ValueError("SFIRE smoke source continuation clock exceeds the fire clock")
    ctx.state.chem.sfire_source_lasttime = last


def step(ctx, dt, ktau):
    import cupy as cp
    from gpuwm.core.sfire_atm import add_fire_tracer_emissions

    fire = getattr(getattr(ctx.state, "physics", None), "fire", None)
    if fire is None:
        return
    now = float(fire.grid.time_seconds)
    # A tile buffer is reused for different domain windows. Its gathered
    # continuation field is authoritative, rather than a cached host scalar.
    last = float(ctx.diag["sfire_source_lasttime"][0,0].item())
    if not math.isfinite(last) or last < 0:
        raise ValueError("SFIRE smoke continuation clock must be finite and nonnegative")
    if now == last:
        return
    if now < last or not math.isclose(now, ctx.clock.curr_secs + float(dt), rel_tol=0, abs_tol=1e-9):
        raise ValueError("SFIRE smoke requires one completed fire step on the atmosphere clock")
    interior = ((slice(1,-1),slice(1,-1)) if hasattr(fire, "_tile_spec")
                else fire.grid.interior)
    burnt = cp.ascontiguousarray(fire.grid.data["burnt_area_dt"][interior])
    fuel = cp.ascontiguousarray(fire.grid.data["fgip"][interior])
    if not bool(cp.isfinite(burnt).all()) or not bool(cp.isfinite(fuel).all()):
        raise ValueError("SFIRE smoke requires finite consumed fraction and native fuel loading")
    if bool(cp.any(burnt < 0)) or bool(cp.any(fuel < 0)):
        raise ValueError("SFIRE smoke requires nonnegative consumed fraction and native fuel loading")
    rho, dryrho, dz = (ctx.met(name) for name in ("rho", "dryrho", "dz8w"))
    if bool(cp.any(dryrho <= 0)) or not bool(cp.isfinite(dryrho).all()):
        raise ValueError("SFIRE smoke requires finite positive dry density")
    nz, ny, nx = rho.shape
    bounds = getattr(fire, "_tile_exchange_domain", (1, nx-2, 1, ny-2))
    native = ctx.diag["sfire_native_increment"]
    native.fill(0)
    cfg = ctx.cfg
    add_fire_tracer_emissions(native, burnt, fuel, rho=rho, dz8w=dz,
        sr_x=fire.sr_x, sr_y=fire.sr_y, smoke_yield=cfg.fire_tracer_smoke,
        scheme=cfg.fire_smk_scheme, z_at_w=ctx.met("z_at_w"), terrain=ctx.state.ht,
        peak=cfg.fire_smk_peak, upper=cfg.fire_tg_ub, extinction=cfg.fire_smk_ext, domain=bounds)
    mx, my = (ctx.met(name) for name in ("msftx", "msfty"))
    if bool(cp.any(mx <= 0)) or bool(cp.any(my <= 0)):
        raise ValueError("SFIRE smoke physical cell area requires positive map factors")
    for index, row in enumerate(rows(ctx.table)):
        _module(cp.cuda.Device().id).get_function("chem_sfire_apply")(((nx*ny+127)//128,), (128,),
            (ctx.field(row), native, rho, dryrho, dz, burnt, fuel, mx, my,
             ctx.diag["sfire_source_expected"][index], ctx.diag["sfire_source_injected"][index],
             ctx.diag["sfire_source_areal"][index], np.int32(nx), np.int32(ny), np.int32(nz),
             np.int32(fire.sr_x), np.int32(fire.sr_y), np.float32(cfg.fire_tracer_smoke),
             np.float64(cfg.dx*cfg.dy), *map(np.int32, bounds)))
    ctx.diag["sfire_source_lasttime"].fill(now)
    ctx.state.chem.sfire_source_lasttime = now


@lru_cache(maxsize=None)
def _module(device):
    # Preserve native subnormal increments before their g -> ug scaling.
    # CuPy RawModule appends FTZ=true, so use the same direct NVRTC path
    # as the native SFIRE atmospheric helper.
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.kernel_compile_notice import observe_module_compile
    source = module_source()
    with cp.cuda.Device(device), observe_module_compile("gpuwm.core.chem_sfire:chem_sfire"):
        ptx, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None, "chem_sfire.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module("gpuwm.core.chem_sfire:chem_sfire", source=source, options=MODULE_OPTIONS, module=None)
    return module


def source_receipt(state):
    import cupy as cp
    from gpuwm.core.chem_state import process_attr
    alloc = {item.name: item for item in ALLOCATES}
    from gpuwm.core.streaming import domain_store
    store = domain_store(state)
    def field(key):
        attr = process_attr(alloc[key])
        if store is not None:
            from tilestream.sfire_spotting import mapped_host_array
            return mapped_host_array(store['state/' + attr])
        return getattr(state, attr)
    sums = {key: cp.asnumpy(field(key).sum(axis=(1, 2)))
            for key in ("sfire_source_expected", "sfire_source_injected")}
    return {row.name: {"expected_fuel_source_kg": float(sums["sfire_source_expected"][index]),
        "injected_geometry_kg": float(sums["sfire_source_injected"][index]),
        "yield_basis": "native completed-step burnt_area_dt times FGIP, wet fuel loading",
        "state_units": "ug kg-1 dry air", "native_units": "g kg-1 air",
        "source_present": getattr(getattr(state, "physics", None), "fire", None) is not None,
        "pm25_fraction_defined": False}
        for index, row in enumerate(rows(state.chem.table))}


def _native_row(state):
    selected = () if state.chem is None else rows(state.chem.table)
    if len(selected) != 1:
        raise ValueError("native fire_smoke input/output requires one active endogenous bulk smoke row")
    return selected[0]


def _convert_units(state, value, out, *, to_native):
    import cupy as cp
    if state.qv is None or value.shape != state.alt.shape:
        raise ValueError("native fire_smoke conversion requires matching moist mass-grid fields")
    _module(cp.cuda.Device().id).get_function("chem_sfire_units")(
        ((out.size+127)//128,), (128,),
        (out, value, state.alt, state.qv, np.int32(out.size), np.int32(to_native)))
    return out


def import_native(state, value):
    """Restore WRF's g_smoke/kg_air cold-start field into the dry-air row."""
    import cupy as cp
    row = _native_row(state)
    value = cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
    if not bool(cp.isfinite(value).all()) or bool(cp.any(value < 0)):
        raise ValueError("native fire_smoke must be finite and nonnegative")
    _convert_units(state, value, getattr(state, row.state_attr), to_native=False)
    getattr(state, row.time_attr)[...] = getattr(state, row.state_attr)
    refresh_output(state)


def refresh_output(state):
    """Refresh the declared native history buffer after transport or feedback."""
    from gpuwm.core.chem_state import process_attr
    row = _native_row(state)
    alloc = next(item for item in ALLOCATES if item.name == "sfire_native_output")
    native = getattr(state, process_attr(alloc))
    _convert_units(state, getattr(state, row.state_attr), native, to_native=True)


prepare_history = refresh_output
