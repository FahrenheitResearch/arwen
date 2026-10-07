"""WRF GOCART settling of one row group with a shared growth arm.

All mass levels 0..nz-1 are processed. There is no fixed vertical limit.
Workspace is one float64 (nz,ny,nx) array reused across rows and one int32
(nrow,ny,nx) substep diagnostic. Row fields are updated in place.
"""
from __future__ import annotations

import numpy as np



def _array(a, shape, dtype, label):
    import cupy as cp
    if not isinstance(a, cp.ndarray) or a.shape != shape or a.dtype != np.dtype(dtype) or not a.flags.c_contiguous:
        raise ValueError(f"{label} must be a contiguous CuPy {dtype} array of shape {shape}")
    return a


def _pointers(arrays):
    import cupy as cp
    return cp.asarray([0 if a is None else a.data.ptr for a in arrays], dtype=cp.uint64)


def pack_settling_rows(rows):
    """Pack dict keys radius_m, density_kg_m3, growth_arm (none or gerber).

    Radius and density are float64, matching WRF REAL*8 reff/den arrays.
    A group must share one growth arm. No species identifiers are read.
    The optional UoC density override is expressed by the row density value.
    """
    import cupy as cp
    rows = list(rows)
    if not rows:
        raise ValueError("settling requires at least one row")
    arms = {r["growth_arm"] for r in rows}
    if len(arms) != 1 or not arms <= {"none", "gerber"}:
        raise ValueError("a settling group must share growth_arm none or gerber")
    radius = np.asarray([r["radius_m"] for r in rows], dtype=np.float64)
    density = np.asarray([r["density_kg_m3"] for r in rows], dtype=np.float64)
    if not np.all(np.isfinite(radius) & (radius > 0)) or not np.all(np.isfinite(density) & (density > 0)):
        raise ValueError("settling radius and density must be finite and positive")
    return {"radius": cp.asarray(radius), "density": cp.asarray(density),
            "growth_arm": next(iter(arms))}


def launch_settling(rows, parameters, *, temp, pressure, dz, rho, qv, dt,
                    accumulators=None, velocities=None, gravity=9.81,
                    dyn_visc=1.5e-5, workspace=None, nsteps=None):
    """Settle N separate float32 fields; diagnostics are optional per row.

    Inputs are C-order (nz,ny,nx). Dust accumulators are kg/m2 and velocity
    outputs are m/s, both float32 (ny,nx). Gerber groups do not accumulate
    dust diagnostics. The returned workspace/nsteps may be reused next call.
    Positive dz, pressure, temperature and rho are the caller's met contract.
    """
    from gpuwm.core.kernels import get_kernel
    import cupy as cp
    rows = list(rows)
    shape = temp.shape
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("settling needs a nonempty (nz,ny,nx) domain")
    nz, ny, nx = shape
    nr = len(rows)
    if nr < 1:
        raise ValueError("settling requires at least one row")
    for label, a in (("temp",temp),("pressure",pressure),("dz",dz),("rho",rho),("qv",qv)):
        _array(a,shape,"float32",label)
    for a in rows:
        _array(a,shape,"float32","row")
    arm = parameters["growth_arm"]
    if arm not in ("none", "gerber"):
        raise ValueError("growth_arm must be none or gerber")
    for name in ("radius", "density"):
        _array(parameters[name],(nr,),"float64",name)
    accumulators = list(accumulators) if accumulators is not None else [None]*nr
    velocities = list(velocities) if velocities is not None else [None]*nr
    if len(accumulators) != nr or len(velocities) != nr:
        raise ValueError("diagnostics must have one entry per row")
    if arm == "gerber" and any(a is not None for a in accumulators+velocities):
        raise ValueError("Gerber rows have no dust diagnostics in WRF")
    for a in accumulators+velocities:
        if a is not None:
            _array(a,(ny,nx),"float32","diagnostic")
    if not np.isfinite(dt) or dt <= 0 or dt > np.iinfo(np.int32).max:
        raise ValueError("dt must be finite, positive and fit INT(dt)")
    if not np.isfinite(gravity) or gravity <= 0 or not np.isfinite(dyn_visc) or dyn_visc <= 0:
        raise ValueError("gravity and dyn_visc must be finite and positive")
    if workspace is None:
        workspace = cp.empty(shape,dtype=cp.float64)
    if nsteps is None:
        nsteps = cp.empty((nr,ny,nx),dtype=cp.int32)
    _array(workspace,shape,"float64","workspace")
    _array(nsteps,(nr,ny,nx),"int32","nsteps")
    args = (_pointers(rows),parameters["radius"],parameters["density"],
            _pointers(accumulators),_pointers(velocities),temp,pressure,dz,rho,qv,
            workspace,nsteps,*map(np.int32,(nz,ny*nx,nr,arm=="gerber")),
            np.float32(dt),np.float32(gravity),np.float32(dyn_visc))
    get_kernel("chem_settling","chem_settling_gocart")(((ny*nx+127)//128,),(128,),args)
    return {"workspace":workspace,"nsteps":nsteps}


# ---------------------------------------------------------------------------
# The process (``settling.gocart``): the chem driver's view.
# ---------------------------------------------------------------------------

from gpuwm.core.chem_context import ChemAllocation  # noqa: E402

KEY = "settling.gocart"
LEDGER = "settled"
#: The kernel translation units step() launches (priced by the preflight).
KERNEL_MODULES = ("chem_settling",)
REQUIRES = ("t_phy", "p_phy", "dz8w", "rho", "qv")
ALLOCATES = (
    ChemAllocation("graset", "rows_2d",
                   units="kg m-2",
                   description="accumulated gravitational settling out of "
                               "the lowest layer, WRF DUSTGRASET "
                               "(chem/module_gocart_settling.F:164-168); "
                               "accumulated for the no-growth rows only, "
                               "as WRF does"),
    ChemAllocation("setvel", "rows_2d", restart="rebuild", units="m s-1",
                   description="lowest-layer settling velocity, WRF SETVEL"),
    ChemAllocation("settling_work", "3d", dtype="float64", restart="rebuild",
                   description="the settling sweep's REAL*8 column copy"),
    ChemAllocation("settling_nsteps", "rows_2d", dtype="int32",
                   restart="rebuild",
                   description="sub-steps each row took, per column"),
)
#: WRF's gravity and GOCART's dynamic viscosity
#: (share/module_model_constants.F:17; phys/module_data_gocart_dust.F:16).
G = 9.81
DYN_VISC = 1.5e-5
#: The settling growth arms of the schema and the kernel's names for them.
GROWTH = {"none": "none", "gerber_seasalt": "gerber"}


def rows(table):
    return table.rows_for(KEY)


def _groups(table):
    """``[(arm, [(index, row), ...])]`` in first-appearance order.

    WRF settles the dust bins (no growth) and the sea-salt bins (Gerber
    growth) in two calls of ``settling``; a group is every row sharing an
    arm, which is what one call shares (``growth_fac``, the column dzmin).
    """
    groups: dict = {}
    for i, row in enumerate(rows(table)):
        if row.settling is None:
            raise ValueError(f"row {row.name!r} names {KEY} with no settling "
                             "column")
        arm = GROWTH[row.settling.get("growth", "none")]
        groups.setdefault(arm, []).append((i, row))
    for arm, members in groups.items():
        index = [i for i, _ in members]
        if index != list(range(index[0], index[0] + len(index))):
            raise ValueError(
                f"the {arm!r} settling rows are not contiguous in table "
                "order; their sub-step counts share one slice of the "
                "process's per-row array")
    return list(groups.items())


def active(cfg) -> bool:
    """WRF settles inside dry_dep_driver (chem_driver.F:1046-1049, gated by
    vertmix_onoff and ktau > 2) when a dust or sea-salt scheme is on
    (dry_dep_driver.F:1385-1387)."""
    dust = int(getattr(cfg, "dust_opt", 0))
    return (int(getattr(cfg, "vertmix_onoff", 1)) > 0
            and (dust == 1 or dust >= 3 or int(getattr(cfg, "seas_opt", 0)) >= 1))


def init(ctx):
    if not active(ctx.cfg):
        return
    missing = [name for name in REQUIRES if not ctx.has(name)]
    if missing:
        raise ValueError(f"settling reads {missing}, which this run does not "
                         "produce")
    _groups(ctx.table)


def step(ctx, dt, ktau):
    if not active(ctx.cfg) or ktau <= 2:
        return
    chem = ctx.state.chem
    cache = getattr(chem, "gocart_settling_cache", None)
    if cache is None:
        cache = chem.gocart_settling_cache = [
            (arm, [i for i, _ in members], [r for _, r in members],
             pack_settling_rows([{"radius_m": r.settling["radius_m"],
                                  "density_kg_m3": r.settling["density_kg_m3"],
                                  "growth_arm": arm} for _, r in members]))
            for arm, members in _groups(ctx.table)]
    graset, setvel = ctx.diag["graset"], ctx.diag["setvel"]
    nsteps = ctx.diag["settling_nsteps"]
    met = dict(temp=ctx.met("t_phy"), pressure=ctx.met("p_phy"),
               dz=ctx.met("dz8w"), rho=ctx.met("rho"), qv=ctx.met("qv"))
    for arm, index, members, params in cache:
        accumulate = arm == "none"
        launch_settling(
            [ctx.field(r) for r in members], params, dt=float(dt),
            accumulators=[graset[i] for i in index] if accumulate else None,
            velocities=[setvel[i] for i in index] if accumulate else None,
            gravity=G, dyn_visc=DYN_VISC, workspace=ctx.diag["settling_work"],
            nsteps=nsteps[index[0]:index[0] + len(index)], **met)
