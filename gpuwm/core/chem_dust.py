"""Pure surface dust launchers. Species arrays are float32 (nz, ny, nx).

Only level 0 is changed. AFWA diagnostics cover levels 0 through nz-1.
All input arrays must be C contiguous. Erodibility is (3, ny, nx).
Packing keeps WRF REAL arrays in float32 and REAL*8 arrays in float64.
"""
from __future__ import annotations

import numpy as np



def pack_dust_rows(rows):
    """Pack dict keys den_dust, reff_dust, frac_s, ipoint, ch_dust,
    distr_dust, and vis_coefficient. No species identity is inspected.

    vis_coefficient carries the REAL coefficients in AFWA.F:357-361.
    Unused process keys can be omitted: packing is process specific below.
    """
    kinds = {"den_dust": np.float64, "reff_dust": np.float64,
             "frac_s": np.float32, "ipoint": np.int32,
             "ch_dust": np.float32, "distr_dust": np.float64,
             "vis_coefficient": np.float32}
    rows = tuple(rows)
    if not rows:
        raise ValueError("dust requires at least one row")
    return {key: np.asarray([r[key] for r in rows], dtype=kind)
            for key, kind in kinds.items() if all(key in r for r in rows)}


def _arrays(species, emissions, fields, integer=()):
    import cupy as cp
    if not species or len(species) != len(emissions):
        raise ValueError("one emission array is required per species row")
    shape = species[0].shape
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("species must have shape (nz, ny, nx)")
    for a in species:
        if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 or a.shape != shape or not a.flags.c_contiguous:
            raise ValueError("species must be contiguous float32 arrays of equal shape")
    for a in emissions:
        if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 or a.shape != shape[1:] or not a.flags.c_contiguous:
            raise ValueError("emissions must be contiguous float32 (ny, nx)")
    for key, a in fields.items():
        dtype = cp.int32 if key in integer else cp.float32
        if not isinstance(a, cp.ndarray) or a.dtype != dtype or not a.flags.c_contiguous:
            raise ValueError(f"{key} must be a contiguous {dtype} CuPy array")
    pointers = lambda arrays: cp.asarray([a.data.ptr for a in arrays], dtype=cp.uint64)
    return cp, shape, pointers(species), pointers(emissions)


def _params(cp, params, keys, count):
    kinds = {"den_dust": np.float64, "reff_dust": np.float64,
             "frac_s": np.float32, "ipoint": np.int32, "ch_dust": np.float32,
             "distr_dust": np.float64, "vis_coefficient": np.float32}
    out = []
    for key in keys:
        a = np.asarray(params[key])
        if a.dtype != kinds[key] or a.shape != (count,):
            raise ValueError(f"{key} must have WRF dtype {kinds[key]} and shape ({count},)")
        out.append(cp.asarray(a))
    return out


def launch_dust_gocart(species, emissions, params, *, xland, u10, v10,
                       smois, isltyp, erod, rho_phy, dz8w, u_phy, v_phy,
                       dt, g=9.81):
    """Update level 0 mixing ratios and accumulate EDUST in kg/m2.

    Workspace: two row pointer arrays and five small parameter arrays.
    smois is the surface soil layer, (ny, nx). g and dt are WRF REAL.
    """
    from gpuwm.core.kernels import get_kernel
    fields = dict(xland=xland, u10=u10, v10=v10, smois=smois, isltyp=isltyp,
                  erod=erod, rho_phy=rho_phy, dz8w=dz8w, u_phy=u_phy, v_phy=v_phy)
    cp, shape, ptr, eptr = _arrays(species, emissions, fields, ("isltyp",))
    nz, ny, nx = shape
    for key in ("xland", "u10", "v10", "smois", "isltyp"):
        if fields[key].shape != (ny, nx):
            raise ValueError(f"{key} must have shape (ny, nx)")
    if erod.shape != (3, ny, nx):
        raise ValueError("erod must have shape (3, ny, nx)")
    for a in (rho_phy, dz8w, u_phy, v_phy):
        if a.shape != shape:
            raise ValueError("mass level fields must match species shape")
    packed = _params(cp, params, ("den_dust", "reff_dust", "frac_s", "ipoint", "ch_dust"), len(species))
    if np.any((params["ipoint"] < 1) | (params["ipoint"] > 3)):
        raise ValueError("ipoint must be 1, 2 or 3")
    nc = nx * ny
    get_kernel("chem_dust", "chem_dust_gocart")(((nc+127)//128,), (128,),
        (ptr, eptr, np.int32(len(species)), np.int32(nc), *packed,
         xland, u10, v10, smois, isltyp, erod, rho_phy, dz8w, u_phy, v_phy,
         np.float32(dt), np.float32(g)))
    return species, emissions


def launch_dust_afwa(species, emissions, params, *, xland, u10, v10,
                     smois, isltyp, erod, erod_dri, rho_phy, dz8w, snowh,
                     vegfra, lai_vegmask, ust, znt, clay_wrf, sand_wrf,
                     clay_nga, sand_nga, afwa_dustloft, tot_dust, tot_edust,
                     vis_dust, dust_dsr, dust_veg, dust_soils, dust_smois,
                     sf_surface_physics, alpha, gamma, smtune, ustune, dt, g=9.81):
    """All AFWA switches are passed directly, with WRF's default else arms.

    Workspace: two pointer arrays and four row parameter arrays. No shared
    persistent workspace. Diagnostics are updated in place; tot_edust is
    an accumulator even though WRF declares it INTENT(OUT).
    """
    from gpuwm.core.kernels import get_kernel
    fields = {k: v for k, v in locals().items() if hasattr(v, "data") and hasattr(v, "dtype")}
    cp, shape, ptr, eptr = _arrays(species, emissions, fields, ("isltyp",))
    nz, ny, nx = shape
    for key, a in fields.items():
        wanted = (3, ny, nx) if key in ("erod", "erod_dri") else shape if key in ("rho_phy", "dz8w", "tot_dust", "vis_dust") else (ny, nx)
        if a.shape != wanted:
            raise ValueError(f"{key} must have shape {wanted}")
    packed = _params(cp, params, ("den_dust", "reff_dust", "distr_dust", "vis_coefficient"), len(species))
    nc = ny*nx
    args = (ptr, eptr, np.int32(len(species)), np.int32(nc), np.int32(nz), *packed,
            xland, u10, v10, smois, isltyp, erod, erod_dri, rho_phy, dz8w,
            snowh, vegfra, lai_vegmask, ust, znt, clay_wrf, sand_wrf, clay_nga,
            sand_nga, afwa_dustloft, tot_dust, tot_edust, vis_dust,
            *map(np.int32, (dust_dsr, dust_veg, dust_soils, dust_smois, sf_surface_physics)),
            *map(np.float32, (alpha, gamma, smtune, ustune, dt, g)))
    get_kernel("chem_dust", "chem_dust_afwa")(((nc+127)//128,), (128,), args)
    return species, emissions


# ---------------------------------------------------------------------------
# The process (``emission.dust``, DESIGN 3 and 6): the chem driver's view.
# ---------------------------------------------------------------------------

from gpuwm.core.chem_context import ChemAllocation  # noqa: E402
from gpuwm.core.chem_statics import (  # noqa: E402,F401  -- re-exported
    DUST_STATIC_ALLOCATIONS, STATIC_VARIABLES, attach_dust_statics, static_specs)

KEY = "emission.dust"
LEDGER = "emitted"
#: The kernel translation units step() launches (priced by the preflight).
KERNEL_MODULES = ("chem_dust",)
#: What both dust schemes read (chem_prep fields and physics fields by their
#: WRF names).  AFWA adds ust, znt, snowh and vegfra, checked at init when
#: dust_opt = 3 selects it.
REQUIRES = ("xland", "u10", "v10", "smois", "isltyp", "rho", "dz8w",
            "u_phy", "v_phy")
AFWA_REQUIRES = ("ust", "znt", "snowh", "vegfra")

ALLOCATES = (
    ChemAllocation("edust", "rows_2d", output_name="EDUST{n}",
                   units="kg m-2",
                   description="accumulated dust emission, WRF EMIS_DUST "
                               "(chem/module_gocart_dust.F:103-107)"),
    *DUST_STATIC_ALLOCATIONS,
    ChemAllocation("tot_edust", "2d", output_name="TOT_EDUST",
                   units="kg m-2",
                   description="AFWA total accumulated dust emission"),
    ChemAllocation("afwa_dustloft", "2d", restart="rebuild",
                   output_name="AFWA_DUSTLOFT", units="m s-1",
                   description="AFWA dust lofting potential (-99 masked)"),
)

#: WRF's gravity (share/module_model_constants.F:17, REAL 9.81).
G = 9.81


def rows(table):
    return table.rows_for(KEY)


def row_parameters(row) -> dict:
    """One dust row's WRF bin constants, from its settling and dust_emission
    columns (the same REAL*8 den_dust/reff_dust the settling reads)."""
    if row.settling is None or row.dust_emission is None:
        raise ValueError(f"dust row {row.name!r} needs settling (radius, "
                         "density) and dust_emission columns")
    e = row.dust_emission
    return {"den_dust": row.settling["density_kg_m3"],
            "reff_dust": row.settling["radius_m"],
            "frac_s": e["frac_s"], "ipoint": e["erod_class"],
            "ch_dust": e["ch_dust"], "distr_dust": e["afwa_distr"],
            "vis_coefficient": e["afwa_vis_coefficient"]}


def _cache(ctx):
    chem = ctx.state.chem
    cache = getattr(chem, "gocart_dust_cache", None)
    if cache is None:
        rs = rows(ctx.table)
        cache = chem.gocart_dust_cache = {
            "rows": rs,
            "params": pack_dust_rows([row_parameters(r) for r in rs])}
    return cache


def init(ctx):
    option = int(getattr(ctx.cfg, "dust_opt", 0))
    if option == 0:
        return
    needed = REQUIRES + (AFWA_REQUIRES if option == 3 else ())
    missing = [name for name in needed if not ctx.has(name)]
    if missing:
        raise ValueError(
            f"dust_opt={option} reads {missing}, which this run's schemes "
            "do not produce: the emission threshold would read garbage "
            "(dust needs a land surface scheme's smois and the surface "
            "layer's u10/v10, ust and znt)")
    import cupy as cp
    ready = ctx.diag["dust_statics_ready"]
    if not bool(cp.any(ready > 0)):
        raise ValueError(
            "the dust statics (EROD, CLAYFRAC, SANDFRAC) were never sampled "
            "onto this domain: without them no cell is erodible and the "
            "dust rows would emit nothing.  Stage them with gpuwm fetch-geog "
            "--datasets chem-dust and prepare the case again")
    _cache(ctx)


def step(ctx, dt, ktau):
    option = int(getattr(ctx.cfg, "dust_opt", 0))
    if option == 0:
        return
    import cupy as cp

    cache = _cache(ctx)
    rs = cache["rows"]
    species = [ctx.field(r) for r in rs]
    edust = ctx.diag["edust"]
    emissions = [edust[i] for i in range(len(rs))]
    if "erod" not in cache:
        cache["erod"] = cp.ascontiguousarray(cp.stack(
            [ctx.diag[f"dust_erod_{k}"] for k in (1, 2, 3)]))
        cache["isltyp"] = cp.ascontiguousarray(
            ctx.met("isltyp").astype(cp.int32))
    smois = ctx.met("smois")
    smois0 = cp.ascontiguousarray(smois[0] if smois.ndim == 3 else smois)
    common = dict(xland=ctx.met("xland"), u10=ctx.met("u10"),
                  v10=ctx.met("v10"), smois=smois0,
                  isltyp=cache["isltyp"], erod=cache["erod"],
                  rho_phy=ctx.met("rho"), dz8w=ctx.met("dz8w"),
                  dt=float(dt), g=G)
    if option == 1:
        launch_dust_gocart(species, emissions, cache["params"],
                           u_phy=ctx.met("u_phy"), v_phy=ctx.met("v_phy"),
                           **common)
        return
    if option != 3:
        raise ValueError(f"dust_opt={option} is not transcribed")
    cfg = ctx.cfg
    if "zeros2" not in cache:
        ny, nx = common["xland"].shape
        cache["zeros2"] = cp.zeros((ny, nx), cp.float32)
        # AFWA's DRI and NGA fields are optional WRF inputs ArWen does not
        # carry; their negative sentinel takes WRF's own fallback to EROD and
        # to the WRF clay/sand fractions (module_gocart_dust_afwa.F).
        cache["neg3"] = cp.full((3, ny, nx), -1.0, cp.float32)
        cache["neg2"] = cp.full((ny, nx), -1.0, cp.float32)
        cache["tot_dust"] = cp.zeros_like(common["rho_phy"])
        cache["vis_dust"] = cp.zeros_like(common["rho_phy"])
    launch_dust_afwa(
        species, emissions, cache["params"],
        erod_dri=cache["neg3"], snowh=ctx.met("snowh"),
        vegfra=ctx.met("vegfra"), lai_vegmask=cache["zeros2"],
        ust=ctx.met("ust"), znt=ctx.met("znt"),
        clay_wrf=ctx.diag["dust_clayfrac"], sand_wrf=ctx.diag["dust_sandfrac"],
        clay_nga=cache["neg2"], sand_nga=cache["neg2"],
        afwa_dustloft=ctx.diag["afwa_dustloft"], tot_dust=cache["tot_dust"],
        tot_edust=ctx.diag["tot_edust"], vis_dust=cache["vis_dust"],
        dust_dsr=0, dust_veg=0, dust_soils=0, dust_smois=0,
        sf_surface_physics=int(getattr(cfg, "sf_surface_physics", 2)),
        alpha=float(cfg.dust_alpha), gamma=float(cfg.dust_gamma),
        smtune=float(cfg.dust_smtune), ustune=float(cfg.dust_ustune),
        **common)
