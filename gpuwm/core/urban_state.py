"""Per-domain urban state (WRF's urban Registry arrays) and its cold start.

One :class:`UrbanState` per domain, attached as ``PhysicsDriver.urban``;
``None`` when ``sf_urban_physics == 0`` so the default driver allocates
nothing.  Arrays carry WRF's Registry names in lower case, float32 (int32 for
``utype_urb2d``), laid out ``(k, ny, nx)`` like every gpuwm field.

WHERE THE ARRAYS LIVE.  In the driver's ``fields`` dict, under those names.
``fields`` is the surface inventory the checkpoint writer already serializes
and restores IN PLACE, so every urban prognostic rides a checkpoint with no
second serialization path and a resumed run continues the unbroken one byte
for byte.  (WRF's own restart re-runs ``urban_var_init``, which resets
``QC_URB2D`` to 0.01 and ``SH/LH/G/RN_URB2D`` to 0 at every start, restart
included -- module_sf_urban.F:2698-2702, 2801 -- so a WRF restart is not
continuous there.  gpuwm keeps its restart law, continuity, and records the
difference here.)

:func:`urban_var_init_host` transcribes ``urban_var_init``
(module_sf_urban.F:2557-3052, WRF v4.7.1) in NumPy float32 so the Fortran
column oracle can check it on any host; :func:`init_urban_state` runs it and
uploads the result.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gpuwm.config import URBAN_MODEL_MODULES, urban_model_in_build
from gpuwm.core.urban_tables import (URBAN_LAYERS, UrbanCategories,
                                     UrbanParams)

#: WRF's BEP and BEP_BEM module dimensions (``bep_ndm()`` .. in
#: module_sf_bep.F / module_sf_bep_bem.F headers, as check_a_mundo.F:468-485
#: reads them).  The model lanes export their own ``DIMENSIONS``; these are
#: the reference they must agree with, and what the infra oracle builds with.
WRF_URBAN_DIMENSIONS = MappingProxyType({
    2: MappingProxyType({"ndm": 2, "nz_um": 18, "ng_u": 10, "nwr_u": 10,
                         "nf_u": 1, "ngb_u": 1, "nbui_max": 1, "ngr_u": 1}),
    3: MappingProxyType({"ndm": 2, "nz_um": 18, "ng_u": 10, "nwr_u": 10,
                         "nf_u": 10, "ngb_u": 10, "nbui_max": 15,
                         "ngr_u": 10}),
})


def urban_maps(dims: Mapping[str, int]) -> dict[str, int]:
    """check_a_mundo.F:3274-3302: the ``urban_map_*`` sizes."""
    ndm, nz, ng = dims["ndm"], dims["nz_um"], dims["ng_u"]
    nwr, nbui = dims["nwr_u"], dims.get("nbui_max", 1)
    ngb, nf, ngr = dims.get("ngb_u", 1), dims.get("nf_u", 1), dims.get("ngr_u", 1)
    return {
        "zrd": ndm * nwr * nz, "zwd": ndm * nwr * nz * nbui,
        "gd": ndm * ng, "zd": ndm * nz * nbui, "zdf": ndm * nz,
        "bd": nz * nbui, "wd": ndm * nz * nbui, "gbd": ndm * ngb * nbui,
        "fbd": ndm * (nz - 1) * nf * nbui, "zgrd": ndm * ngr * nz,
        "ndm": ndm,
    }


#: Arrays every option carries.  name -> (dtype, layers); layers 0 = 2-D,
#: a string names a size (``hi`` = num_urban_hi, ``layers`` = 4).
COMMON_SPEC = MappingProxyType({
    "frc_urb2d": ("f4", 0), "utype_urb2d": ("i4", 0), "ts_urb2d": ("f4", 0),
    "sh_urb2d": ("f4", 0), "lh_urb2d": ("f4", 0), "g_urb2d": ("f4", 0),
    "rn_urb2d": ("f4", 0), "lp_urb2d": ("f4", 0), "lb_urb2d": ("f4", 0),
    "hgt_urb2d": ("f4", 0), "mh_urb2d": ("f4", 0), "stdh_urb2d": ("f4", 0),
    "lf_urb2d": ("f4", 4), "hi_urb2d": ("f4", "hi"), "z0_urb2d": ("f4", 0),
    "lf_urb2d_s": ("f4", 0), "zd_urb2d": ("f4", 0), "tsk_rural": ("f4", 0),
})
#: Option 1 (the single-layer UCM), Registry.EM_COMMON:916-945.
UCM_SPEC = MappingProxyType({
    **{n: ("f4", 0) for n in (
        "tr_urb2d", "tb_urb2d", "tg_urb2d", "tc_urb2d", "qc_urb2d",
        "uc_urb2d", "xxxr_urb2d", "xxxb_urb2d", "xxxg_urb2d", "xxxc_urb2d",
        "cmcr_urb2d", "tgr_urb2d", "drelr_urb2d", "drelb_urb2d",
        "drelg_urb2d", "flxhumr_urb2d", "flxhumb_urb2d", "flxhumg_urb2d",
        "psim_urb2d", "psih_urb2d", "gz1oz0_urb2d", "u10_urb2d",
        "v10_urb2d", "th2_urb2d", "q2_urb2d", "ust_urb2d", "akms_urb2d")},
    **{n: ("f4", "layers") for n in (
        "trl_urb3d", "tbl_urb3d", "tgl_urb3d", "tgrl_urb3d", "smr_urb3d")},
})
#: Options 2 and 3 (BEP), Registry.EM_COMMON:948-968.
BEP_SPEC = MappingProxyType({
    "tsk_rural_bep": ("f4", 0),
    "trb_urb4d": ("f4", "zrd"), "tw1_urb4d": ("f4", "zwd"),
    "tw2_urb4d": ("f4", "zwd"), "tgb_urb4d": ("f4", "gd"),
    "sfw1_urb3d": ("f4", "zd"), "sfw2_urb3d": ("f4", "zd"),
    "sfr_urb3d": ("f4", "zdf"), "sfg_urb3d": ("f4", "ndm"),
})
#: Option 3 adds BEM, Registry.EM_COMMON:952-982.
BEM_SPEC = MappingProxyType({
    "tlev_urb3d": ("f4", "bd"), "qlev_urb3d": ("f4", "bd"),
    "tw1lev_urb3d": ("f4", "wd"), "tw2lev_urb3d": ("f4", "wd"),
    "tglev_urb3d": ("f4", "gbd"), "tflev_urb3d": ("f4", "fbd"),
    **{n: ("f4", 0) for n in ("sf_ac_urb3d", "lf_ac_urb3d", "cm_ac_urb3d",
                             "sfvent_urb3d", "lfvent_urb3d", "ep_pv_urb3d",
                             "qgr_urb3d", "tgr_urb3d", "draingr_urb3d")},
    "sfwin1_urb3d": ("f4", "wd"), "sfwin2_urb3d": ("f4", "wd"),
    "t_pv_urb3d": ("f4", "zdf"), "trv_urb4d": ("f4", "zgrd"),
    "qr_urb4d": ("f4", "zgrd"), "drain_urb4d": ("f4", "zdf"),
    "sfrv_urb3d": ("f4", "zdf"), "lfrv_urb3d": ("f4", "zdf"),
    "dgr_urb3d": ("f4", "zdf"), "lfr_urb3d": ("f4", "zdf"),
    "dg_urb3d": ("f4", "ndm"), "lfg_urb3d": ("f4", "ndm"),
})
#: The BEP -> PBL handoff (Registry.EM_COMMON ``a_u_bep`` ..), mass levels
#: except ``sf_bep`` which is face-staggered (nz + 1).
PBL_TERM_NAMES = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
                  "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
                  "dlg_bep", "dl_u_bep", "sf_bep", "vl_bep")

#: The rural values the land-surface scheme hands the urban model, WRF's
#: local names.  Noah: written by noah.cu on urban columns (``q1``, ``q2k``,
#: ``zlvl``) and snapshot from ``fields`` after the LSM for the rest.
NOAH_RURAL_KERNEL_FIELDS = ("q1", "q2k", "zlvl")
NOAH_RURAL_FIELD_SOURCES = MappingProxyType({
    "t1": "tsk", "sheat": "hfx", "eta_kinematic": "qfx", "eta": "lh",
    "ssoil": "grdflx", "albedok": "albedo", "emissi": "emiss",
    "sfctmp": "sfctmp", "sfcprs": "sfcprs", "soldn": "swdown",
    "rainbl_used": "rainbl", "ust_in": "ust", "qsfc": "qsfc",
})
#: Noah-MP: noahmpdrv.F:3345-3370 reads these after noahmplsm.
NOAHMP_RURAL_FIELDS = ("hfx", "qfx", "lh", "grdflx", "tsk", "emiss",
                       "albedo", "qsfc", "t2mvxy", "t2mbxy", "q2mvxy",
                       "q2mbxy", "fvegxy", "ust", "swdown", "glw", "rainbl")


def _normalized_row(row) -> tuple[str, object]:
    """One STATE_SPEC row in infra's spelling.

    Model lanes may spell the dtype ``float32``/``int32`` and the layer
    count as WRF's Registry dimension names (``urban_map_zrd``,
    ``num_urban_ndm``, ``num_urban_hi``, ``num_soil_layers``); infra keys
    the same sizes as ``zrd``, ``ndm``, ``hi`` and ``layers``.
    """
    dtype = {"f4": "f4", "float32": "f4", "i4": "i4", "int32": "i4"}.get(
        str(row[0]), str(row[0]))
    layers = row[1]
    if isinstance(layers, str):
        if layers.startswith("urban_map_"):
            layers = layers[len("urban_map_"):]
        layers = {"num_urban_ndm": "ndm", "num_urban_hi": "hi",
                  "num_soil_layers": "layers"}.get(layers, layers)
        if layers == "layers":
            layers = URBAN_LAYERS
    return dtype, layers


def option_spec(option: int, module=None) -> dict[str, tuple[str, object]]:
    """The full array spec for ``option``: infra's rows plus the module's."""
    spec = dict(COMMON_SPEC)
    if option == 1:
        spec.update(UCM_SPEC)
    if option in (2, 3):
        spec.update(BEP_SPEC)
    if option == 3:
        spec.update(BEM_SPEC)
    extra = getattr(module, "STATE_SPEC", None) or {}

    def norm(row):
        return _normalized_row(row)

    def same(a, b):
        # A single-level row and infra's 2-D row describe one plane.
        (ta, la), (tb, lb) = norm(a), norm(b)
        return ta == tb and ({la, lb} <= {0, 1} or la == lb)

    for name, row in extra.items():
        if name in spec and not same(spec[name], row):
            raise ValueError(
                f"{getattr(module, '__name__', module)} declares {name} as "
                f"{tuple(row)}, infra's Registry row is {spec[name]}")
        if name not in spec:
            spec[name] = norm(row)
    return spec


def resolve_dimensions(option: int, module=None) -> dict[str, int]:
    """The model's DIMENSIONS, checked against WRF's, plus the urban maps."""
    if option not in (2, 3):
        return {}
    ref = dict(WRF_URBAN_DIMENSIONS[option])
    exported = getattr(module, "DIMENSIONS", None)
    if exported:
        for key, value in exported.items():
            if key in ref and int(ref[key]) != int(value):
                raise ValueError(
                    f"{module.__name__} exports {key}={value}; WRF v4.7.1's "
                    f"{'BEP_BEM' if option == 3 else 'BEP'} header has "
                    f"{ref[key]} (check_a_mundo.F:468-485)")
            ref[key] = int(value)
    return {**ref, **{f"urban_map_{k}" if k != "ndm" else "num_urban_ndm": v
                      for k, v in urban_maps(ref).items()}}


#: What :func:`urban_array_shapes` prices that is NOT held for the whole
#: run at its priced size: BEP's column scratch and its class-table
#: scratch are allocated on each call (``gpuwm.core.urban_bep``), so they
#: ride the memory pool like any step transient.  Every other urban array
#: is allocated ONCE and held at exactly its priced shape --
#: :func:`init_urban_state`'s Registry arrays and PBL handoff, the surface
#: planes ``gpuwm.core.physics._attach_urban`` adds, the coupler's rural
#: hand-over and BEP_BEM's ``ColumnPlan`` -- which is what lets the memory
#: gate price them at their allocated size (A163).
URBAN_PER_CALL_ARRAYS = frozenset({"bep_column_workspace",
                                   "bep_class_scratch"})


def urban_held_array_shapes(cfg, *, urban_columns: int | None = None
                            ) -> dict[str, tuple[int, ...]]:
    """The :func:`urban_array_shapes` entries held for the whole run."""
    return {name: shape for name, shape in urban_array_shapes(
                cfg, urban_columns=urban_columns).items()
            if name not in URBAN_PER_CALL_ARRAYS}


def urban_array_shapes(cfg, *, urban_columns: int | None = None
                       ) -> dict[str, tuple[int, ...]]:
    """Every array ``init_urban_state`` allocates for ``cfg``, by name.

    For the memory checks (``gpuwm.core.preflight.physics_array_shapes``),
    which run on CPU-only installs: infra's Registry rows and WRF's
    reference dimensions only, so no model module (which may import CuPy)
    is loaded.  BEM's arrays are large -- ``tw1/tw2_urb4d`` alone are 5,400
    words per column -- which is why an unpriced option 3 would pass the
    admission check and then fail to allocate.

    ``urban_columns`` is the domain's urban column count where its land
    cover is known (:func:`prepared_urban_columns`, the prepared doors);
    ``None`` prices BEP+BEM's column workspace at every column urban
    (:func:`_column_workspace_shapes`).
    """
    option = int(getattr(cfg, "sf_urban_physics", 0))
    if option == 0:
        return {}
    ny, nx, nz = int(cfg.ny), int(cfg.nx), int(cfg.nz)
    dims = resolve_dimensions(option)
    hi = int(getattr(cfg, "num_urban_hi", 15))
    shapes: dict[str, tuple[int, ...]] = {}
    for name, row in option_spec(option).items():
        layers = _layers(row, dims, hi)
        shapes[name] = ((layers,) if layers else ()) + (ny, nx)
    if option in (2, 3):
        for name in PBL_TERM_NAMES:
            shapes[name] = ((nz + 1) if name == "sf_bep" else nz, ny, nx)
    # The surface-layer diagnostics the urban overrides write, where the
    # configuration's surface layer does not already allocate them
    # (gpuwm.core.physics._attach_urban); priced whether or not they are
    # new, which over-counts by at most seven planes.
    for name in ("psim", "psih", "gz1oz0", "akhs", "akms"):
        shapes.setdefault(f"override_{name}", (ny, nx))
    if option == 3:
        # The direct/diffuse surface shortwave BEP_BEM reads.
        shapes["swddir"] = (ny, nx)
        shapes["swddif"] = (ny, nx)
    # The rural hand-over the coupler holds between the LSM and the model.
    rural = (NOAH_RURAL_KERNEL_FIELDS + tuple(NOAH_RURAL_FIELD_SOURCES)
             if int(cfg.sf_surface_physics) == 2 else NOAHMP_RURAL_FIELDS)
    for name in rural:
        shapes[f"rural_{name}"] = (ny, nx)
    shapes.update(_column_workspace_shapes(option, ny * nx, nz,
                                           urban_columns=urban_columns))
    return shapes


def _column_workspace_shapes(option: int, ncol: int, nz: int, *,
                             urban_columns: int | None = None
                             ) -> dict[str, tuple[int, ...]]:
    """The device scratch the BEP and BEP+BEM column kernels allocate at
    run time, for this grid.

    Neither lives in ``fields``, and neither is small: BEP holds up to
    ``urban_bep.BEP_WORKSPACE_BYTES`` (256 MiB) of column scratch per call
    plus its class tables' 11 x 8,192 x 32-word scratch, and a BEP+BEM plan
    holds up to ``urban_bem.FORECAST_WORKSPACE_BYTES`` (1 GiB).

    BEP+BEM's plan (``urban_bem.ColumnPlan``) holds ``min(urban columns,
    FORECAST_WORKSPACE_BYTES / workspace_bytes_per_column(nz))`` columns'
    workspace (4,792 at 59 levels) and an error word per column of that
    chunk, and no workspace at all on a domain with no urban column.  With
    ``urban_columns`` (the count :func:`urban_column_count` reads off the
    domain's land cover, the columns the plan is built from) that is its
    price, exactly.  Without it, the configuration door's case, the land
    cover does not exist yet, so the price is the upper bound of every
    column urban: the allocation once a domain holds 4,792 urban columns,
    which a 750 m city nest does, and over by the rural columns below
    that (A176: up to 1 GiB per domain; a prepared door prices tighter).
    BEP's per-call scratch keeps its every-column price either way.  The
    model modules import NumPy only, so this stays CPU-safe.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    if option == 2 and urban_model_in_build(2):
        from gpuwm.core import urban_bep as bep

        slots = bep.bep_workspace_slots(nz)
        lanes = bep._TPB
        cap = max(lanes, (bep.BEP_WORKSPACE_BYTES // (4 * slots * lanes))
                  * lanes)
        tile = min(-(-ncol // lanes) * lanes, cap)
        shapes["bep_column_workspace"] = (tile // lanes, slots, lanes)
        shapes["bep_class_scratch"] = (bep.NURBMAX, 8192, lanes)
        shapes["bep_class_views"] = (bep.NURBMAX, 795)
    if option == 3 and urban_model_in_build(3):
        from gpuwm.core import urban_bem as bem

        per_column = bem.workspace_bytes_per_column(nz)
        urban = (ncol if urban_columns is None
                 else max(0, min(int(urban_columns), ncol)))
        chunk = max(1, min(urban, bem.FORECAST_WORKSPACE_BYTES // per_column))
        shapes["bem_column_workspace"] = (
            (chunk if urban else 0) * (per_column // 4),)
        shapes["bem_column_errors"] = (chunk,)
    return shapes


def urban_column_count(ivgtyp, *, categories: UrbanCategories,
                       use_wudapt_lcz: int, frc_urb_tbl,
                       frc_urb2d=None) -> int:
    """The urban columns the urban model runs on, read off a land cover.

    The columns :func:`urban_fraction` leaves above 0, the very set
    ``urban_bem.ColumnPlan`` lists (``frc_urb2d > 0``).  Read off the
    PREPARED land use, before ``initialize_landuse`` has folded lakes and
    sea ice into water and ice, so it is the runtime's count or more,
    never fewer.
    """
    utype = urban_types(ivgtyp, categories, use_wudapt_lcz)
    _validate_urban_types(utype, use_wudapt_lcz, table_bounds_only=True)
    frc = (np.zeros(utype.shape, np.float32) if frc_urb2d is None
           else frc_urb2d)
    return int(np.count_nonzero(urban_fraction(utype, frc, frc_urb_tbl) > 0))


def urban_categories_for(cfg, landuse_dataset: str) -> UrbanCategories:
    """The urban categories ``gpuwm.core.physics._attach_urban`` gives a
    domain of ``cfg`` on ``landuse_dataset``, without a card.

    Noah-MP reads them from its own land-use identity (MPTABLE's ISURBAN,
    NATURAL and LCZ_1..11 for the dataset, ``NoahmpRuntimeParameters``);
    Noah reads NATURAL and the LCZ rows from VEGPARM.TBL beside the same
    ISURBAN (:func:`gpuwm.core.urban_tables.urban_category_set`).
    """
    from gpuwm.core.noahmp import load_noahmp_parameters
    from gpuwm.core.urban_tables import urban_category_set

    _, veg = load_noahmp_parameters().vegetation_groups(str(landuse_dataset))
    if int(cfg.sf_surface_physics) == 4:
        return UrbanCategories(
            isurban=int(veg.scalar("ISURBAN")),
            natural=int(veg.scalar("NATURAL")),
            lcz=tuple(int(veg.scalar(f"LCZ_{n}")) for n in range(1, 12)))
    return urban_category_set(str(landuse_dataset),
                              isurban=int(veg.scalar("ISURBAN")))


def bem_workspace_counted(cfg) -> bool:
    """Whether a domain's urban column count moves its memory price.

    Only BEP+BEM's column workspace (``sf_urban_physics`` 3) is priced by
    the count (:func:`_column_workspace_shapes`); every other urban array
    is a full plane.  A door asks this before it reads any land-use
    attribute, so a domain that runs no BEP+BEM reads nothing.
    """
    return int(getattr(cfg, "sf_urban_physics", 0) or 0) == 3


def prepared_urban_columns(cfg, lu_index, *, landuse_dataset: str,
                           frc_urb2d=None) -> int | None:
    """A prepared domain's urban column count, for the memory price.

    ``None`` where the count does not move a price: no BEP+BEM
    (``sf_urban_physics`` 3, whose column workspace is
    :func:`_column_workspace_shapes`'s only count-priced array), or no
    land use to read.  ``lu_index`` is the prepared static's LU_INDEX and
    ``frc_urb2d`` the input's own urban fraction where it carries one (a
    wrfinput), exactly what the domain's urban cold start reads.
    """
    if not bem_workspace_counted(cfg) or lu_index is None:
        return None
    from gpuwm.core.urban_tables import load_urban_params

    lcz = int(getattr(cfg, "use_wudapt_lcz", 0) or 0)
    params = load_urban_params(3, lcz)
    return urban_column_count(
        lu_index, categories=urban_categories_for(cfg, landuse_dataset),
        use_wudapt_lcz=lcz, frc_urb_tbl=params.FRC_URB_TBL,
        frc_urb2d=frc_urb2d)


def urban_columns_line(counts) -> str | None:
    """The one line a prepared door prints when it prices the urban column
    workspace from the land cover rather than the configuration's bound."""
    if not counts:
        return None
    named = ", ".join(f"d{gid:02d} {count:,}"
                      for gid, count in sorted(counts.items()))
    return ("BEP+BEM column workspace priced from the land cover's urban "
            f"columns ({named}), not at every column urban as the "
            "configuration check prices it")


def _layers(row, dims: Mapping[str, int], num_urban_hi: int) -> int:
    layers = row[1]
    if isinstance(layers, int):
        return layers
    if layers == "hi":
        return int(num_urban_hi)
    if layers == "layers":
        return URBAN_LAYERS
    if layers == "ndm":
        return int(dims["num_urban_ndm"])
    return int(dims[f"urban_map_{layers}"])


def load_model_module(option: int):
    """The model lane's module for ``option``, or None when absent."""
    name = URBAN_MODEL_MODULES.get(int(option))
    if name is None:
        return None
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None
        raise


# ---------------------------------------------------------------------------
# urban_var_init, on the host
# ---------------------------------------------------------------------------

def urban_types(ivgtyp, categories: UrbanCategories,
                use_wudapt_lcz: int) -> np.ndarray:
    """``urban_var_init``'s UTYPE_URB2D for each column (int32, 0 = not
    urban; module_sf_urban.F:2748-2786)."""
    ivgtyp = np.asarray(ivgtyp).astype(np.int64)
    lookup = categories.utype_lookup(use_wudapt_lcz)
    inside = (ivgtyp >= 0) & (ivgtyp < lookup.size)
    utype = np.where(inside, lookup[np.clip(ivgtyp, 0, lookup.size - 1)], 0)
    return utype.astype(np.int32)


def urban_fraction(utype, frc_urb2d, frc_urb_tbl) -> np.ndarray:
    """FRC_URB2D as ``urban_var_init`` leaves it (float32, a new array).

    An input fraction in (0, 1] is kept on an urban column, any other takes
    ``FRC_URB_TBL(UTYPE)`` (module_sf_urban.F:2767-2777), and a column that
    is not urban holds 0 (:2785-2799).  The columns it leaves above 0 are
    the ones every urban model runs on: BEP+BEM's ``ColumnPlan`` is built
    from exactly them, so :func:`urban_column_count` reads this same rule.
    """
    utype = np.asarray(utype)
    frc = np.array(frc_urb2d, dtype=np.float32)
    switch = utype > 0
    keep = (frc > 0.0) & (frc <= 1.0)
    from_table = switch & ~keep
    frc[from_table] = np.asarray(frc_urb_tbl)[utype[from_table] - 1]
    frc[~switch] = 0.0
    return frc


def _validate_urban_types(utype, use_wudapt_lcz: int, *,
                          table_bounds_only: bool = False) -> None:
    """Refuse a land-cover legend whose urban types have no table rows."""
    # module_physics_init.F:3349-3356: validate before indexing FRC_URB_TBL.
    max_utype = int(utype.max()) if utype.size else 0
    if int(use_wudapt_lcz) == 0 and max_utype > 3:
        raise ValueError("USING 10 WUDAPT LCZ WITHOUT URBPARM_LCZ.TBL. SET "
                         "USE_WUDAPT_LCZ=1 (WRF's fatal: LCZ types 4-11 have "
                         "no row in URBPARM.TBL)")
    if (not table_bounds_only and int(use_wudapt_lcz) == 1
            and max_utype <= 3):
        raise ValueError("USING URBPARM_LCZ.TBL WITH OLD 3 URBAN CLASSES. SET "
                         "USE_WUDAPT_LCZ=0 (WRF's fatal: the LCZ table's rows "
                         "1-3 are not the three NLCD urban classes)")


def urban_var_init_host(*, option: int, use_wudapt_lcz: int,
                        params: UrbanParams, categories: UrbanCategories,
                        ivgtyp, tsk, tslb, tmn, smois, frc_urb2d=None,
                        num_urban_hi: int = 15, nz: int = 1,
                        dims: Mapping[str, int] | None = None,
                        spec: Mapping | None = None,
                        restart: bool = False) -> dict[str, np.ndarray]:
    """``urban_var_init`` (module_sf_urban.F:2557-3052) for one domain.

    Returns every array of ``spec`` (default: :func:`option_spec`) plus the
    PBL handoff for options 2/3 (``nz`` mass levels, ``sf_bep`` ``nz + 1``).
    Inputs are host arrays: ``ivgtyp`` (ny, nx) int, ``tsk``/``tmn`` (ny, nx),
    ``tslb``/``smois`` (4, ny, nx), optional ``frc_urb2d`` (ny, nx) from the
    input file (zeros when the input has none, which is WRF's stock case).

    ``distributed_aerodynamics_option`` (``slucm_distributed_drag``) is off:
    the namelist door refuses it, because its arm is not transcribed.
    """
    f4 = np.float32
    ivgtyp = np.asarray(ivgtyp).astype(np.int64)
    ny, nx = ivgtyp.shape
    tsk = np.asarray(tsk, dtype=f4)
    tslb = np.asarray(tslb, dtype=f4)
    smois = np.asarray(smois, dtype=f4)
    dims = dict(dims or {})
    spec = dict(spec if spec is not None else option_spec(option))
    out: dict[str, np.ndarray] = {}
    for name, row in spec.items():
        layers = _layers(row, dims, num_urban_hi)
        dtype = np.int32 if row[0] == "i4" else f4
        out[name] = np.zeros(((layers,) if layers else ()) + (ny, nx),
                             dtype=dtype)
    if frc_urb2d is not None:
        out["frc_urb2d"][...] = np.asarray(frc_urb2d, dtype=f4)
    utype = urban_types(ivgtyp, categories, use_wudapt_lcz)
    switch = utype > 0
    # module_physics_init.F:3349-3356, WRF's own fatals.  WRF raises them
    # AFTER urban_var_init, which has by then read FRC_URB_TBL(UTYPE) past
    # the end of a 3-row table for an LCZ type above 3 -- undefined in WRF,
    # so the refusal comes first here.
    _validate_urban_types(utype, use_wudapt_lcz)
    # :2716-2719 and :2725
    for name in ("sh_urb2d", "lh_urb2d", "g_urb2d", "rn_urb2d"):
        out[name][...] = 0.0
    out["utype_urb2d"][...] = utype
    # :2749-2767 -- default morphology when HGT_URB2D <= 0 (always, until
    # gridded morphology is ingested), and :2785-2799 off the city.
    zero_morph = ~switch | ~(out["hgt_urb2d"] > 0.0)
    for name in ("lp_urb2d", "lb_urb2d", "hgt_urb2d"):
        out[name][zero_morph] = 0.0
    if option == 1:
        out["mh_urb2d"][zero_morph] = 0.0
        out["stdh_urb2d"][zero_morph] = 0.0
        out["lf_urb2d"][:, zero_morph] = 0.0
    else:
        out["hi_urb2d"][:, zero_morph] = 0.0
    out["frc_urb2d"][...] = urban_fraction(utype, out["frc_urb2d"],
                                           params.FRC_URB_TBL)
    if "qc_urb2d" in out:
        out["qc_urb2d"][...] = f4(0.01)          # :2801, even on restart
    if not restart:
        t0 = tsk
        tl1, tl2, tl3 = tslb[0], tslb[1], tslb[2]
        tl_mid = (f4(0.5) * (tl1 + tl2)).astype(f4)
        tl_4 = (tl2 + ((tl3 - tl2).astype(f4) * f4(0.29)).astype(f4)).astype(f4)
        ladder = np.stack([tl1 + f4(0.0), tl_mid, tl2 + f4(0.0), tl_4])
        for name in ("xxxr_urb2d", "xxxb_urb2d", "xxxg_urb2d", "xxxc_urb2d"):
            if name in out:
                out[name][...] = 0.0
        if option == 1:
            for name in ("drelr_urb2d", "drelb_urb2d", "drelg_urb2d",
                         "flxhumr_urb2d", "flxhumb_urb2d", "flxhumg_urb2d",
                         "cmcr_urb2d"):
                out[name][...] = 0.0
            out["tgr_urb2d"][...] = t0 + f4(0.0)
        for name in ("tc_urb2d", "tr_urb2d", "tb_urb2d", "tg_urb2d",
                     "ts_urb2d"):
            if name in out:
                out[name][...] = t0 + f4(0.0)
        for name in ("trl_urb3d", "tbl_urb3d"):
            if name in out:
                out[name][...] = ladder
        if option == 1:
            out["tgrl_urb3d"][...] = ladder
            out["smr_urb3d"][...] = np.array(
                [0.2, 0.2, 0.2, 0.0], dtype=f4)[:, None, None]
        if "tgl_urb3d" in out:
            out["tgl_urb3d"][...] = tslb[:URBAN_LAYERS] + f4(0.0)
        if option in (2, 3):
            tblend = params.TBLEND_TBL
            walls = np.where(switch, tblend[np.maximum(utype, 1) - 1], tl1)
            for name in ("trb_urb4d", "tw1_urb4d", "tw2_urb4d"):
                out[name][...] = walls.astype(f4)
            out["tgb_urb4d"][...] = tl1
            for name in ("sfw1_urb3d", "sfw2_urb3d", "sfr_urb3d",
                         "sfg_urb3d"):
                out[name][...] = 0.0
        if option == 3:
            for name in ("lf_ac_urb3d", "sf_ac_urb3d", "cm_ac_urb3d",
                         "sfvent_urb3d", "lfvent_urb3d", "ep_pv_urb3d",
                         "drain_urb4d", "sfrv_urb3d", "lfrv_urb3d",
                         "dgr_urb3d", "dg_urb3d", "lfr_urb3d", "lfg_urb3d",
                         "draingr_urb3d", "sfwin1_urb3d", "sfwin2_urb3d"):
                out[name][...] = 0.0
            out["t_pv_urb3d"][...] = tl1
            out["qr_urb4d"][...] = smois[0]
            out["qgr_urb3d"][...] = smois[0]
            # :2891 then :2899: TGR_URB3D = tlayer0 then = 0.
            out["tgr_urb3d"][...] = 0.0
            out["trv_urb4d"][...] = np.where(
                switch, tblend[np.maximum(utype, 1) - 1], tl1).astype(f4)
            for name in ("tlev_urb3d", "tw1lev_urb3d", "tw2lev_urb3d",
                         "tglev_urb3d", "tflev_urb3d"):
                out[name][...] = tl1
            out["qlev_urb3d"][...] = f4(0.01)
    if option in (2, 3):
        for name in PBL_TERM_NAMES:
            levels = nz + 1 if name == "sf_bep" else nz
            out[name] = np.zeros((levels, ny, nx), dtype=f4)
        out["sf_bep"][...] = 1.0
        out["vl_bep"][...] = 1.0
    return out


# ---------------------------------------------------------------------------
# the device state
# ---------------------------------------------------------------------------

@dataclass
class UrbanSolar:
    """The solar geometry the urban models read, on the radiation cadence.

    ``declin`` radians, ``hrang`` (= WRF's ``OMG_URB2D``) radians per column,
    ``coszen`` per column, ``xlat``/``xlong`` degrees, ``gmt`` hours,
    ``julday`` day of year, ``julian`` fractional day, ``julyr`` year.
    """

    declin: float = 0.0
    coszen: object = None
    hrang: object = None
    xlat: object = None
    xlong: object = None
    gmt: float = 0.0
    julday: int = 1
    julian: float = 0.0
    julyr: int = 2000
    solcon: float = 1370.0
    model_time: float | None = None


@dataclass
class UrbanState:
    """The urban arrays of one domain, as views of the driver's ``fields``."""

    option: int
    use_wudapt_lcz: int
    num_urban_hi: int
    params: UrbanParams
    categories: UrbanCategories
    dimensions: Mapping[str, int]
    names: tuple[str, ...]
    fields: dict = field(repr=False)
    rural: dict = field(default_factory=dict, repr=False)
    pbl_terms: Mapping | None = field(default=None, repr=False)
    solar: UrbanSolar = field(default_factory=UrbanSolar, repr=False)
    urban_mask: object = field(default=None, repr=False)
    category_lookup: object = field(default=None, repr=False)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        names = object.__getattribute__(self, "names")
        if name in names:
            return object.__getattribute__(self, "fields")[name]
        raise AttributeError(name)

    def arrays(self) -> dict:
        return {name: self.fields[name] for name in self.names}


def init_urban_state(cfg, params: UrbanParams, categories: UrbanCategories,
                     fields: dict, *, nz: int, frc_urb2d=None,
                     restart: bool = False, module=None) -> UrbanState:
    """Build one domain's :class:`UrbanState` into ``fields`` (in place).

    Runs :func:`urban_var_init_host` on the LSM's initialized ``tsk``,
    ``tslb``, ``tmn`` and ``smois`` (WRF calls it after LSMINIT /
    NOAHMP_INIT, module_physics_init.F:3294, 3437), uploads each array into
    ``fields`` under its Registry name, then lets the model lane's
    ``init_state`` hook finish anything option-specific.
    """
    import cupy as cp

    option = int(cfg.sf_urban_physics)
    if module is None:
        module = load_model_module(option)
    spec = option_spec(option, module)
    dims = resolve_dimensions(option, module)
    host = urban_var_init_host(
        option=option, use_wudapt_lcz=int(cfg.use_wudapt_lcz),
        params=params, categories=categories,
        ivgtyp=cp.asnumpy(fields["ivgtyp"]), tsk=cp.asnumpy(fields["tsk"]),
        tslb=cp.asnumpy(fields["tslb"]), tmn=cp.asnumpy(fields["tmn"]),
        smois=cp.asnumpy(fields["smois"]),
        frc_urb2d=(None if frc_urb2d is None
                   else cp.asnumpy(cp.asarray(frc_urb2d))),
        num_urban_hi=int(cfg.num_urban_hi), nz=int(nz), dims=dims,
        spec=spec, restart=restart)
    names = []
    for name, value in host.items():
        if name in PBL_TERM_NAMES:
            continue
        if name in fields:
            raise ValueError(f"urban array {name!r} collides with an existing "
                             "surface field")
        fields[name] = cp.ascontiguousarray(cp.asarray(value))
        names.append(name)
    pbl_terms = None
    if option in (2, 3):
        pbl_terms = MappingProxyType({
            name: cp.ascontiguousarray(cp.asarray(host[name]))
            for name in PBL_TERM_NAMES})
    lookup = categories.utype_lookup(int(cfg.use_wudapt_lcz))
    state = UrbanState(
        option=option, use_wudapt_lcz=int(cfg.use_wudapt_lcz),
        num_urban_hi=int(cfg.num_urban_hi), params=params,
        categories=categories, dimensions=MappingProxyType(dims),
        names=tuple(names), fields=fields, pbl_terms=pbl_terms,
        urban_mask=fields["utype_urb2d"] > 0,
        category_lookup=cp.asarray((lookup > 0).astype(np.int32)))
    hook = getattr(module, "init_state", None)
    if hook is not None:
        hook(state, params, tlayer0=fields["tslb"], tsurface0=fields["tsk"],
             tdeep0=fields["tmn"], smois=fields["smois"], restart=restart)
    return state


__all__ = [
    "BEM_SPEC", "BEP_SPEC", "COMMON_SPEC", "NOAHMP_RURAL_FIELDS",
    "NOAH_RURAL_FIELD_SOURCES", "NOAH_RURAL_KERNEL_FIELDS", "PBL_TERM_NAMES",
    "UCM_SPEC", "UrbanSolar", "UrbanState", "WRF_URBAN_DIMENSIONS",
    "bem_workspace_counted",
    "init_urban_state", "load_model_module", "option_spec",
    "URBAN_PER_CALL_ARRAYS", "urban_array_shapes", "urban_held_array_shapes",
    "prepared_urban_columns", "resolve_dimensions", "urban_categories_for",
    "urban_column_count", "urban_columns_line", "urban_fraction",
    "urban_maps", "urban_types",
    "urban_var_init_host",
]
