"""GOCART aerosol velocity and dust deposition diagnostics.

No vertical state is changed. The velocity is the lower boundary for vertmx.
Only level 0 is read, matching chem(i,1,j,p_dust_n). There is no nz bound.
Workspace/output: three float32 (ny,nx) arrays, supplied or allocated here.
"""
from __future__ import annotations

import numpy as np

from gpuwm.core.chem_settling import _array, _pointers


def pack_drydep_rows(rows):
    """Pack dict key is_dust as int32 flags. No species identifiers are read."""
    import cupy as cp
    rows = list(rows)
    if any(type(r["is_dust"]) is not bool for r in rows):
        raise ValueError("is_dust must be a bool")
    return {"is_dust":cp.asarray([r["is_dust"] for r in rows],dtype=cp.int32),
            "is_dust_host":np.asarray([r["is_dust"] for r in rows],dtype=np.int32)}


def launch_drydep(rows, parameters, *, rho, rmol, ust, znt, pbl, dt,
                  chem_opt, domain_extent, tile_origin=(0,0), accumulators=None,
                  ddvel=None, aer_res=None, rmol_used=None):
    """Compute one velocity shared by aerosol rows, with domain edges zero.

    domain_extent=(ids,ide,jds,jde) and tile_origin=(its,jts) use the same
    global index coordinates. WRF processes ids<i<ide and jds<j<jde only.
    chem_opt is 300 (Wesely aer_res_def) or 401 (local resistance).
    The rmol input remains unchanged. rmol_used exposes the local value.
    Fields are float32; rho and rows are (nz,ny,nx), surface fields (ny,nx).
    HFX, TSK, XLAND, surface pressure and layer depth do not affect dvel:
    the HFX-derived obk is overwritten, and the other values feed unused
    outputs in WRF. They are deliberately absent from this pure interface.
    """
    from gpuwm.core.kernels import get_kernel
    import cupy as cp
    if chem_opt not in (300,401):
        raise ValueError("GOCART dry deposition supports chem_opt 300 or 401")
    shape = rho.shape
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("rho must have nonempty shape (nz,ny,nx)")
    _, ny, nx = shape
    surface = (ny,nx)
    _array(rho,shape,"float32","rho")
    for label,a in (("rmol",rmol),("ust",ust),("znt",znt),("pbl",pbl)):
        _array(a,surface,"float32",label)
    rows = list(rows)
    for a in rows:
        _array(a,shape,"float32","row")
    nr = len(rows)
    _array(parameters["is_dust"],(nr,),"int32","is_dust")
    accumulators = list(accumulators) if accumulators is not None else [None]*nr
    if len(accumulators) != nr:
        raise ValueError("accumulators must have one entry per row")
    # Require a buffer for every flagged row; copying these small flags is setup only.
    flags = parameters.get("is_dust_host")
    if flags is None:
        flags = parameters["is_dust"].get()
    if not np.all((flags==0)|(flags==1)):
        raise ValueError("is_dust flags must be 0 or 1")
    for flag,a in zip(flags,accumulators):
        if flag and a is None:
            raise ValueError("a dust row requires an accumulator")
        if a is not None:
            _array(a,surface,"float32","accumulator")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    ids,ide,jds,jde = domain_extent
    x0,y0 = tile_origin
    if any(int(x)!=x for x in (*domain_extent,*tile_origin)) or ide<=ids or jde<=jds:
        raise ValueError("domain extents and tile origin must be integer indices")
    outputs=[]
    for label,a in (("ddvel",ddvel),("aer_res",aer_res),("rmol_used",rmol_used)):
        if a is None:
            a=cp.empty(surface,dtype=cp.float32)
        outputs.append(_array(a,surface,"float32",label))
    args=(_pointers(rows),parameters["is_dust"],_pointers(accumulators),rho,
          rmol,ust,znt,pbl,*outputs,
          *map(np.int32,(nx,ny,nr,x0,y0,ids,ide,jds,jde,chem_opt==401)),np.float32(dt))
    get_kernel("chem_drydep_gocart","chem_drydep_gocart")(((ny*nx+127)//128,),(128,),args)
    return dict(zip(("ddvel","aer_res","rmol_used"),outputs))


# ---------------------------------------------------------------------------
# The process (``drydep.gocart``): the chem driver's view.
# ---------------------------------------------------------------------------

from gpuwm.core.chem_context import ChemAllocation  # noqa: E402

KEY = "drydep.gocart"
#: The process writes deposition velocities, not species: the mass leaves
#: through vertmx's lower boundary, which books it.  So this bucket closes at
#: zero by construction.
LEDGER = "deposited"
#: The kernel translation units step() launches (priced by the preflight).
KERNEL_MODULES = ("chem_drydep_gocart",)
REQUIRES = ("rho", "rmol", "ust", "znt", "pblh")
ALLOCATES = (
    ChemAllocation("drydep", "rows_2d", units="kg m-2",
                   description="accumulated dust dry deposition, WRF "
                               "DUSTDRYDEP (chem/module_gocart_drydep.F:121-125), "
                               "for the dust rows"),
    ChemAllocation("drydepvel", "2d", restart="rebuild",
                   output_name="DRYDEPVEL", units="m s-1",
                   description="GOCART aerosol dry deposition velocity, WRF "
                               "DEPVELOCITY (chem/module_gocart_drydep.F:119)"),
    ChemAllocation("drydep_aer_res", "2d", restart="rebuild", units="s m-1",
                   description="Wesely aer_res_def the velocity used "
                               "(chem/module_dep_simple.F:1614)"),
    ChemAllocation("drydep_rmol_used", "2d", restart="rebuild", units="m-1",
                   description="rmol after Wesely's |rmol| < 1e-6 zeroing, "
                               "applied to a copy; the model's rmol is not "
                               "written"),
)


def rows(table):
    return table.rows_for(KEY)


def resistance_arm(table) -> int:
    """WRF's two aerosol arms: 401 (DUST) computes the aerodynamic resistance
    locally, every other GOCART package takes Wesely's aer_res_def
    (chem/module_gocart_drydep.F:271-274).  The active sets' own
    ``wrf_chem_opt`` decides: DUST only when every set that names a WRF
    package names 401."""
    from gpuwm.chem_table import catalog

    sets = catalog().sets
    options = {sets[name].wrf_chem_opt for name in table.sets
               if sets[name].wrf_chem_opt is not None}
    return 401 if options == {401} else 300


def active(cfg) -> bool:
    """dry_dep_driver runs when vertmix_onoff > 0 and ktau > 2
    (chem/chem_driver.F:1046-1049)."""
    return int(getattr(cfg, "vertmix_onoff", 1)) > 0


def init(ctx):
    if not active(ctx.cfg):
        return
    missing = [name for name in REQUIRES if not ctx.has(name)]
    if missing:
        raise ValueError(f"GOCART aerosol deposition reads {missing}, which "
                         "this run's surface layer and PBL do not produce")


def step(ctx, dt, ktau):
    if not active(ctx.cfg) or ktau <= 2:
        return
    chem = ctx.state.chem
    cache = getattr(chem, "gocart_drydep_cache", None)
    if cache is None:
        rs = rows(ctx.table)
        flags = [r.family == "dust" for r in rs]
        cache = chem.gocart_drydep_cache = {
            "rows": rs, "is_dust": flags,
            "params": pack_drydep_rows([{"is_dust": f} for f in flags]),
            "arm": resistance_arm(ctx.table)}
    rs = cache["rows"]
    rho = ctx.met("rho")
    nz, ny, nx = rho.shape
    drydep = ctx.diag["drydep"]
    out = launch_drydep(
        [ctx.field(r) for r in rs], cache["params"], rho=rho,
        rmol=ctx.met("rmol"), ust=ctx.met("ust"), znt=ctx.met("znt"),
        pbl=ctx.met("pblh"), dt=float(dt), chem_opt=cache["arm"],
        # WRF's global mass-point indices: the loop excludes the domain's
        # first row and column (i = max(ids+1,its) .. min(ide-1,ite) with
        # ide the staggered end, chem/module_gocart_drydep.F:88-89).
        domain_extent=(1, nx + 1, 1, ny + 1), tile_origin=(1, 1),
        accumulators=[drydep[i] if f else None
                      for i, f in enumerate(cache["is_dust"])],
        ddvel=ctx.diag["drydepvel"], aer_res=ctx.diag["drydep_aer_res"],
        rmol_used=ctx.diag["drydep_rmol_used"])
    velocity = out["ddvel"]
    for row in rs:
        ctx.ddvel[ctx.row_index(row)] = velocity
