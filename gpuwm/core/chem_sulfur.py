"""Pure launchers for WRF 4.7.1 GOCART_SIMPLE sulfur chemistry.

Rows declare sulfur.role = dms | so2 | so4 | msa. These are reaction roles,
not state names. pack_parameters returns four int32 row indices in that order,
with -1 for an absent role. An absent role is zero on entry, never written.
All species are ppmv. Prescribed oxidants are mol/mol daily means.
"""
from __future__ import annotations

import numpy as np

from gpuwm.core.chem_column import columns, field, parameter, timestep

_ROLES = ("dms", "so2", "so4", "msa")


def pack_parameters(rows):
    role_row = np.full(4, -1, np.int32)
    for r, row in enumerate(rows):
        sulfur = row.get("sulfur")
        if sulfur is None:
            continue
        role = sulfur["role"]
        if role not in _ROLES:
            raise ValueError(f"unknown sulfur reaction role {role!r}")
        slot = _ROLES.index(role)
        if role_row[slot] >= 0:
            raise ValueError(f"duplicate sulfur reaction role {role!r}")
        role_row[slot] = r
    return role_row


def upload_parameters(packed):
    import cupy as cp
    return cp.asarray(packed, dtype=cp.int32)


def _solar_fields(lat, lon):
    shape = lat.shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("latitude must have shape (ny, nx)")
    field(lat, shape, "latitude")
    field(lon, shape, "longitude")
    return shape


def _clock(gmt, julday):
    gmt = np.float32(gmt)
    if not np.isfinite(gmt) or not 0 <= gmt < 24:
        raise ValueError("gmt must be in [0, 24)")
    if isinstance(julday, bool) or int(julday) != julday or not 1 <= julday <= 366:
        raise ValueError("julday must be an integer in 1..366")
    return gmt, np.int32(julday)


def launch_solar_init(lat, lon, dt, gmt, julday, *, tcosz=None, ttday=None):
    """One-time float32 daily sum. Two (ny, nx) float32 persistent outputs.

    The caller owns their restart/carry classification. No hidden state.
    Polar night retains WRF's zero tcosz; the step retains its 0/0 OH.
    """
    from gpuwm.core.kernels import get_kernel
    import cupy as cp
    shape = _solar_fields(lat, lon)
    dt = timestep(dt)
    gmt, julday = _clock(gmt, julday)
    if tcosz is None:
        tcosz = cp.empty(shape, dtype=cp.float32)
    if ttday is None:
        ttday = cp.empty(shape, dtype=cp.float32)
    field(tcosz, shape, "tcosz"); field(ttday, shape, "ttday")
    nc = lat.size
    get_kernel("chem_sulfur", "chem_sulfur_solar_init")(
        ((nc+127)//128,), (128,), (lat, lon, tcosz, ttday, np.int32(nc), dt, gmt, julday))
    return tcosz, ttday


def launch_solar_step(lat, lon, curr_secs, gmt, julday, *, cossza=None):
    """Driver float64 clock, then float32 szangle. One temporary 2-D output."""
    from gpuwm.core.kernels import get_kernel
    import cupy as cp
    shape = _solar_fields(lat, lon)
    curr_secs = np.float64(curr_secs)
    if not np.isfinite(curr_secs) or curr_secs < 0:
        raise ValueError("curr_secs must be finite and nonnegative")
    gmt, julday = _clock(gmt, julday)
    if cossza is None:
        cossza = cp.empty(shape, dtype=cp.float32)
    field(cossza, shape, "cossza")
    nc = lat.size
    get_kernel("chem_sulfur", "chem_sulfur_solar_step")(
        ((nc+127)//128,), (128,),
        (lat, lon, cossza, np.int32(nc), curr_secs, gmt, julday))
    return cossza


def launch_sulfur(species, role_row, temp, rho, backg_oh, backg_h2o2,
                  backg_no3, cossza, tcosz, ttday, dt, *, qc=None, qi=None,
                  gd_cldf=None):
    """Update levels 0..nz-2 in place. Top mass level and oxidants stay intact.

    cossza comes from launch_solar_step, tcosz/ttday from launch_solar_init.
    Splitting solar and chemistry permits attribution against oracle solar
    fields. qc/qi None means the corresponding WRF index is absent. Like WRF,
    an ice-only package does not override gd_cldf. No 3-D workspace is needed.
    Device role_row is trusted output from pack/upload_parameters.
    """
    from gpuwm.core.kernels import get_kernel
    shape, pointers = columns(species)
    nz, ny, nx = shape
    dt = timestep(dt)
    parameter(role_row, 4, np.int32, "role_row")
    for label, a in (("temp", temp), ("rho", rho), ("backg_oh", backg_oh),
                     ("backg_h2o2", backg_h2o2), ("backg_no3", backg_no3)):
        field(a, shape, label)
    for label, a in (("qc", qc), ("qi", qi), ("gd_cldf", gd_cldf)):
        if a is not None:
            field(a, shape, label)
    for label, a in (("cossza", cossza), ("tcosz", tcosz), ("ttday", ttday)):
        field(a, (ny, nx), label)
    nc = ny * nx
    get_kernel("chem_sulfur", "chem_sulfur_gocart")(
        ((nc+127)//128,), (128,),
        (pointers, role_row, temp, rho, qc if qc is not None else np.uint64(0),
         qi if qi is not None else np.uint64(0),
         gd_cldf if gd_cldf is not None else np.uint64(0),
         backg_oh, backg_h2o2, backg_no3, cossza, tcosz, ttday,
         np.int32(nz), np.int32(nc), dt, np.int32(qc is not None), np.int32(qi is not None)))


# ---------------------------------------------------------------------------
# The process (``chem.sulfur``): the chem driver's view.
# ---------------------------------------------------------------------------

from gpuwm.core.chem_context import ChemAllocation  # noqa: E402
from gpuwm.core.chem_statics import (  # noqa: E402,F401  -- re-exported
    SOLAR_ALLOCATIONS, attach_solar_geometry)

KEY = "chem.sulfur"
LEDGER = "chemistry"
#: The kernel translation units step() launches (priced by the preflight).
KERNEL_MODULES = ("chem_sulfur",)
REQUIRES = ("t_phy", "rho")
#: The prescribed daily-mean oxidants a row supplies by its sulfur role.
OXIDANT_ROLES = ("oh", "h2o2", "no3")
ALLOCATES = (
    *SOLAR_ALLOCATIONS,
    ChemAllocation("sulfur_tcosz", "2d", restart="rebuild",
                   description="sum of cos(SZA) over one day of steps from "
                               "the run's start (chemics_init.F:1765-1791)"),
    ChemAllocation("sulfur_ttday", "2d", restart="rebuild", units="s",
                   description="daylight seconds in that day"),
    ChemAllocation("sulfur_cossza", "2d", restart="rebuild",
                   description="cos(SZA) this chem step"),
)


def rows(table):
    return table.rows_for(KEY)


def _roles(table):
    species, oxidants = [], {}
    for row in rows(table):
        role = getattr(row, "sulfur_role", None)
        if role in OXIDANT_ROLES:
            if role in oxidants:
                raise ValueError(f"two rows supply the {role} oxidant")
            oxidants[role] = row
        elif role is not None:
            species.append(row)
    return species, oxidants


def _oxidant_refusal(table, enabled_sources):
    """Name absent prescribed inputs before a forecast prepares its forcing."""
    species, oxidants = _roles(table)
    if not species:
        return None
    missing = [role for role in OXIDANT_ROLES if role not in oxidants]
    if missing:
        return (f"the sulfur rows need the {missing} prescribed background "
                "oxidant rows in the active chem table, and they are absent: "
                "enabling an oxidant source alone supplies no state field, so "
                "SO2 would never oxidize and sulfate would be emission-only")
    enabled = set(enabled_sources)
    for role in OXIDANT_ROLES:
        row = oxidants[role]
        if row.transported or row.units != "mol mol-1":
            return (f"background oxidant {role!r} row {row.name!r} must be "
                    "prescribed in mol mol-1: the sulfur kernel reads a held "
                    "daily mean, so transport or ppmv would change its "
                    "reaction rates")
        entries = [entry for entry in row.boundary
                   if entry["source"] in enabled]
        source = (table.sources.get(entries[0]["source"])
                  if entries else None)
        if (source is None or source.kind != "oxidant"
                or source.acquisition is None
                or (source.time or {}).get("interpolation") != "daily_mean"):
            return (f"background oxidant {role!r} row {row.name!r} has no "
                    "enabled data-store oxidant source with daily-mean "
                    "initialization: its allocated field would stay zero "
                    "or read an instantaneous frame instead of GOCART's "
                    "prescribed daily mean")
    return None


def refusal(cfg):
    """Refuse sulfur whose required oxidant rows cannot be initialized.

    An acquisition row is only a request grammar. It does not create the
    three prescribed species fields the source initializer and sulfur
    kernel consume. This check runs through the configuration's ordinary
    process-refusal hook, before meteorological preparation or GPU work.
    """
    from gpuwm.chem_table import chem_names, load

    table = load(cfg)
    if table is None:
        return None
    return _oxidant_refusal(
        table, chem_names(getattr(cfg, "chem_sources", ""), "chem_sources"))


def _stepchem(cfg) -> int:
    """WRF's grid%stepchem, ``max(nint(chemdt*60/dt), 1)``
    (chem/chemics_init.F:644-649): the model steps per chem step.  Fortran's
    nint rounds half away from zero; Python's round would round half to
    even (chemdt = 1.25 min at dt = 30 s is 2.5 steps: 3 in WRF, 2 here)."""
    chemdt = float(getattr(cfg, "chemdt", 0.0))
    if chemdt <= 0:
        return 1
    return max(int(np.floor(chemdt * 60.0 / float(cfg.dt) + 0.5)), 1)


def chem_step_dt(cfg, ktau, dt):
    """``(run, dtstepc)``: WRF runs the mechanism at ktau 1 and every
    stepchem-th step after, over the time since the last chem step
    (chem_driver.F:393-416)."""
    step = _stepchem(cfg)
    if ktau == 1:
        return True, float(dt)
    if step == 1 or ktau % step == 0:
        return True, float(dt) * (step if ktau > step else step - 1)
    return False, 0.0


def init(ctx):
    species, oxidants = _roles(ctx.table)
    if not species:
        return
    from gpuwm.chem_table import chem_names

    reason = _oxidant_refusal(
        ctx.table, chem_names(getattr(ctx.cfg, "chem_sources", ""),
                              "chem_sources"))
    if reason:
        raise ValueError(reason)
    import cupy as cp
    lat = ctx.diag["sulfur_xlat"]
    if not bool(cp.any(lat != 0)):
        raise ValueError("the sulfur process's latitude was never written "
                         "for this domain; szangle would put every column on "
                         "the equator")
    clock = ctx.clock
    launch_solar_init(lat, ctx.diag["sulfur_xlong"], float(ctx.cfg.dt),
                      float(clock.gmt), int(clock.julday),
                      tcosz=ctx.diag["sulfur_tcosz"],
                      ttday=ctx.diag["sulfur_ttday"])


def step(ctx, dt, ktau):
    species, oxidants = _roles(ctx.table)
    if not species:
        return
    run, dtstepc = chem_step_dt(ctx.cfg, ktau, dt)
    if not run:
        return
    chem = ctx.state.chem
    role_row = getattr(chem, "gocart_sulfur_roles", None)
    if role_row is None:
        role_row = chem.gocart_sulfur_roles = upload_parameters(
            pack_parameters([{"sulfur": {"role": r.sulfur_role}}
                             for r in species]))
    clock = ctx.clock
    launch_solar_step(ctx.diag["sulfur_xlat"], ctx.diag["sulfur_xlong"],
                      float(clock.curr_secs), float(clock.gmt),
                      int(clock.julday), cossza=ctx.diag["sulfur_cossza"])
    # WRF sets the cloud fraction to 1 where qc (and, when the scheme carries
    # it, qi) is positive (module_gocart_chem.F:93-101); the state's own
    # moisture fields are those arrays.
    qc = getattr(ctx.state, "qc", None)
    qi = getattr(ctx.state, "qi", None)
    launch_sulfur([ctx.field(r) for r in species], role_row,
                  ctx.met("t_phy"), ctx.met("rho"),
                  ctx.field(oxidants["oh"]), ctx.field(oxidants["h2o2"]),
                  ctx.field(oxidants["no3"]), ctx.diag["sulfur_cossza"],
                  ctx.diag["sulfur_tcosz"], ctx.diag["sulfur_ttday"],
                  dtstepc, qc=qc, qi=qi)
