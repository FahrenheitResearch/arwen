"""``sf_urban_physics = 3``: WRF v4.7.1 BEP+BEM on the GPU.

The multi-layer building effect parameterisation with the building energy
model -- indoor temperature and humidity, air conditioning and its waste
heat, windows, floors, green roofs and rooftop photovoltaics -- is
``phys/module_sf_bep_bem.F`` (``BEP_BEM`` and its own copies of the BEP
routines; WRF keeps two diverging BEP copies and identity means transcribing
the one that runs) and ``phys/module_sf_bem.F``.  Both live, in full, in
``kernels/urban_bep_bem.cu`` and ``kernels/urban_bem.cuh``, generated from the
pinned Fortran by ``tools/transcribe_urban_bem.py``: one thread per urban
column (``FRC_URB2D > 0``, module_sf_bep_bem.F:710), FP32 in Fortran's
evaluation order, no FMA contraction, no flush to zero, glibc 2.43's float
libm words (``glibc_flt32.cuh`` expf/logf/powf, ``glibc_trig_flt32.cuh``
sinf/cosf/asinf/acosf/atanf/tanf, CORE-MATH log10f in ``urban_bem.cuh``).  Every WRF local array lives in a column-private, interleaved
global workspace (``ws[offset * nthreads + tid]``), so the per-thread local
frame stays small.  Every FP32/FP64 operator is emitted as its IEEE
intrinsic (``__fmul_rn`` ...), so no compiler can reassociate or contract
it: NVRTC 13.3 for sm_120 rewrote ``(c*ch)/cm`` as ``c*(ch/cm)`` in
``flux_flat`` until it was, and the RTX 5090 was then up to 4552 ULP off
WRF where the RTX 4090 was bit-identical.

Proof: ``tests/test_urban_bem_wrf471_parity.py`` runs the kernel against WRF's
own ``BEP_BEM`` (``tools/urban_wrf471_oracle/run_bep_bem.F90``, five fixtures,
up to 30 consecutive calls) and asserts max ULP 0 on every output and every
prognostic array; measured on an RTX 4090 (sm_89) and an RTX 5090 (sm_120).  WRF's compiled Fortran is the CPU reference.

Coupling (DESIGN.md 3.4): :func:`after_lsm` zeroes what the land-surface
driver zeroes, runs the column and hands over to the shared BEP couple
(``gpuwm.core.urban_bep_couple``), exactly as ``module_sf_noahdrv.F:1636-1776``
and ``module_sf_noahmpdrv.F:3644-3776`` do.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gpuwm.core.urban_bem_layout import METADATA

# module_sf_bep_bem.F:22-53, the BEP_BEM module parameters.
NDM = 2
NZ_UM = 18
NG_U = 10
NGR_U = 10
NWR_U = 10
NF_U = 10
NGB_U = 10
NBUI_MAX = 15
NURBMAX = 11
DZ_U = 5.0

#: What check_a_mundo.F:476-485 reads from bep_bem_ndm() .. bep_bem_ngr_u().
DIMENSIONS = MappingProxyType({
    "ndm": NDM, "nz_um": NZ_UM, "ng_u": NG_U, "nwr_u": NWR_U, "nf_u": NF_U,
    "ngb_u": NGB_U, "nbui_max": NBUI_MAX, "ngr_u": NGR_U,
})

#: The option-3 arrays (Registry.EM_COMMON), in infra's row convention:
#: layers 0 = 2-D, a string names an ``urban_map_*`` size or ``ndm``.
STATE_SPEC = MappingProxyType({
    "tsk_rural_bep": ("f4", 0),
    "trb_urb4d": ("f4", "zrd"), "tw1_urb4d": ("f4", "zwd"),
    "tw2_urb4d": ("f4", "zwd"), "tgb_urb4d": ("f4", "gd"),
    "sfw1_urb3d": ("f4", "zd"), "sfw2_urb3d": ("f4", "zd"),
    "sfr_urb3d": ("f4", "zdf"), "sfg_urb3d": ("f4", "ndm"),
    "tlev_urb3d": ("f4", "bd"), "qlev_urb3d": ("f4", "bd"),
    "tw1lev_urb3d": ("f4", "wd"), "tw2lev_urb3d": ("f4", "wd"),
    "tglev_urb3d": ("f4", "gbd"), "tflev_urb3d": ("f4", "fbd"),
    **{name: ("f4", 0) for name in (
        "sf_ac_urb3d", "lf_ac_urb3d", "cm_ac_urb3d", "sfvent_urb3d",
        "lfvent_urb3d", "ep_pv_urb3d", "qgr_urb3d", "tgr_urb3d",
        "draingr_urb3d")},
    "sfwin1_urb3d": ("f4", "wd"), "sfwin2_urb3d": ("f4", "wd"),
    "t_pv_urb3d": ("f4", "zdf"), "trv_urb4d": ("f4", "zgrd"),
    "qr_urb4d": ("f4", "zgrd"), "drain_urb4d": ("f4", "zdf"),
    "sfrv_urb3d": ("f4", "zdf"), "lfrv_urb3d": ("f4", "zdf"),
    "dgr_urb3d": ("f4", "zdf"), "lfr_urb3d": ("f4", "zdf"),
    "dg_urb3d": ("f4", "ndm"), "lfg_urb3d": ("f4", "ndm"),
})

# ---------------------------------------------------------------------------
# the generated layouts
# ---------------------------------------------------------------------------

def _layout(section: str, kind: str) -> Mapping[str, tuple[int, tuple]]:
    return MappingProxyType({
        name: (offset, tuple(shape))
        for name, (k, offset, shape) in METADATA[section].items()
        if k == kind})


#: module_sf_urban arrays BEP_BEM reads, packed in Fortran order.
BEM_TABLE_FLOAT_LAYOUT = _layout("table", "float")
BEM_TABLE_INT_LAYOUT = _layout("table", "int")
#: The ``save``d class arrays BEP_BEM's first call builds (init_para, icBEP).
#: ``clsi[0]`` is the class kernel's error word.
BEM_CLASS_FLOAT_LAYOUT = _layout("class", "float")
BEM_CLASS_INT_LAYOUT = MappingProxyType(
    {"_error": (0, ()), **_layout("class", "int")})
WORKSPACE_FIXED_FLOATS = int(METADATA["fixed"])
WORKSPACE_FLOATS_PER_LEVEL = int(METADATA["per_level"])
COLUMN_ARRAY_ARGUMENTS = tuple(METADATA["array_arguments"])
COLUMN_SCALAR_ARGUMENTS = tuple(METADATA["scalar_arguments"])
#: Error code -> the Fortran check it stands for (file:line, WRF's text).
ERRORS = MappingProxyType({
    number: f"{text} ({where})"
    for where, (number, text) in METADATA["errors"].items()})

#: No FMA contraction (gfortran -O0 contracts nothing) and no flush to zero
#: (gfortran keeps subnormals); the parity test fails with either dropped.
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "gpuwm.core.urban_bem:urban_bep_bem"
_SOURCES = ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "urban_bem.cuh",
            "urban_bep_bem.cu")

#: Per-category rows are laid out for NURBMAX classes; the hourly profiles
#: are 24 long in every table.
_HOURLY = ("hsequip_tbl", "irho_tbl")


def _length(layout) -> int:
    return max((offset + int(np.prod(shape, dtype=np.int64))
                for offset, shape in layout.values()), default=0)


def pack_bem_tables(tbl: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Pack every module_sf_urban variable BEP_BEM reads (Fortran order).

    ``tbl`` is keyed by the lower-case module_sf_urban names (``capb_tbl``,
    ``street_direction_tbl``, ``icate``, ...).  Per-category arrays may carry
    ICATE (3 or 11) categories; the layout reserves eleven and pads.
    """
    for name in METADATA["table"]:
        if name not in tbl:
            raise KeyError(name)
    icate = int(tbl["icate"])
    if icate not in (3, 11):
        raise ValueError(
            f"icate={icate}: WRF allocates 3 (URBPARM.TBL) or 11 "
            "(URBPARM_LCZ.TBL) urban classes (module_check_a_mundo.F:459-461)")
    packed = []
    for layout, dtype in ((BEM_TABLE_FLOAT_LAYOUT, np.float32),
                          (BEM_TABLE_INT_LAYOUT, np.int32)):
        out = np.zeros(_length(layout), dtype=dtype)
        for name, (offset, shape) in layout.items():
            value = np.asarray(tbl[name], dtype=dtype)
            if shape:
                expected = (shape if name in _HOURLY
                            else shape[:-1] + (icate,))
                if value.shape not in (shape, expected):
                    raise ValueError(
                        f"{name}: expected Fortran shape {expected} or "
                        f"{shape}, got {value.shape}")
                padded = np.zeros(shape, dtype=dtype)
                padded[tuple(slice(0, n) for n in value.shape)] = value
                words = padded.ravel(order="F")
            else:
                if value.shape != ():
                    raise ValueError(f"{name}: expected a scalar, got "
                                     f"{value.shape}")
                words = value.reshape(1)
            out[offset:offset + words.size] = words
        packed.append(out)
    return packed[0], packed[1]


def tables_from_params(params) -> dict[str, object]:
    """The :func:`pack_bem_tables` input from infra's ``UrbanParams``."""
    out: dict[str, object] = {}
    for name in METADATA["table"]:
        if name == "icate":
            out[name] = int(params.icate)
            continue
        value = np.asarray(params.values[name.upper()])
        out[name] = value.item() if value.ndim == 0 else value
    return out


# ---------------------------------------------------------------------------
# the device module
# ---------------------------------------------------------------------------

def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from gpuwm.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in _SOURCES)


@lru_cache(maxsize=None)
def _bem_module():
    """Dedicated loader: this module alone needs both option overrides."""
    import cupy as cp

    from gpuwm.certify.kernel_manifest import record_module

    source = module_source()
    module = cp.RawModule(code=source, options=MODULE_OPTIONS,
                          name_expressions=None)
    module.compile()
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=module)
    return module


class BepBemError(RuntimeError):
    """A WRF fatal (``stop`` / ``wrf_error_fatal``) inside BEP_BEM or BEM."""


_EXTRA_ERRORS = {
    9001: "the column workspace is smaller than WRF's locals need",
    9002: "ICATE is outside the 1..11 class arrays"}


def _message(code: int) -> str:
    return ERRORS.get(code, _EXTRA_ERRORS.get(code, f"error {code}"))


def _refuse(code: int) -> None:
    raise BepBemError(f"WRF BEP_BEM stopped where WRF stops: {_message(code)}")


def _check_errors(err, cols=None) -> None:
    """Immediate read (class tables, tests): name the first failing column."""
    import cupy as cp

    host = cp.asnumpy(err)
    bad = np.flatnonzero(host)
    if not bad.size:
        return
    first = int(bad[0])
    where = ("class initialization" if cols is None
             else f"column {int(cols[first].item())}")
    raise BepBemError(f"WRF BEP_BEM {where}: {_message(int(host[first]))}")


def build_class_tables(tblf, tbli) -> dict:
    """Run BEP_BEM's first-call block (init_para, icBEP) on the device."""
    import cupy as cp

    tblf = cp.asarray(tblf, dtype=cp.float32)
    tbli = cp.asarray(tbli, dtype=cp.int32)
    if tblf.ndim != 1 or tblf.size != _length(BEM_TABLE_FLOAT_LAYOUT):
        raise ValueError("tblf does not match BEM_TABLE_FLOAT_LAYOUT")
    if tbli.ndim != 1 or tbli.size != _length(BEM_TABLE_INT_LAYOUT):
        raise ValueError("tbli does not match BEM_TABLE_INT_LAYOUT")
    cls = {"clsf": cp.zeros(_length(BEM_CLASS_FLOAT_LAYOUT), cp.float32),
           "clsi": cp.zeros(_length(BEM_CLASS_INT_LAYOUT), cp.int32)}
    _bem_module().get_function("bep_bem_class_init")(
        (1,), (1,), (tblf, tbli, cls["clsf"], cls["clsi"]))
    _check_errors(cls["clsi"][:1])
    return cls


_CLASS_CACHE: dict[tuple, dict] = {}


def class_tables(params) -> dict:
    """:func:`build_class_tables` for one ``UrbanParams``, once per device.

    WRF builds these once, on BEP_BEM's first call (``first``,
    module_sf_bep_bem.F:690-705), from the one table every domain reads.
    """
    import cupy as cp

    key = (params.sha256, int(params.use_wudapt_lcz), int(params.icate),
           cp.cuda.Device().id)
    held = _CLASS_CACHE.get(key)
    if held is None:
        held = build_class_tables(*pack_bem_tables(tables_from_params(params)))
        _CLASS_CACHE[key] = held
    return held


def workspace_bytes_per_column(nz: int) -> int:
    """Upper bound of WRF's local arrays for one column (float32 words)."""
    return 4 * (WORKSPACE_FIXED_FLOATS
                + WORKSPACE_FLOATS_PER_LEVEL * (int(nz) + 1))


class ColumnPlan:
    """The urban column list, chunked to a workspace budget, and its buffers.

    FRC_URB2D is fixed for the run once the cold start has set it, so a
    domain builds its plan once (:func:`after_lsm` keeps it on the state)
    and the per-step launch reads nothing back from the device.
    """

    def __init__(self, frc_urb2d, nz: int, workspace_bytes: int) -> None:
        import cupy as cp

        cols = cp.flatnonzero(frc_urb2d.ravel() > cp.float32(0))
        cols = cols.astype(cp.int32)
        self.count = int(cols.size)
        self.nz = int(nz)
        self.shape = tuple(frc_urb2d.shape)
        per_column = workspace_bytes_per_column(nz)
        chunk = min(max(self.count, 1), int(workspace_bytes) // per_column)
        if chunk < 1:
            raise ValueError(f"workspace_bytes must be at least {per_column}")
        self.words = per_column // 4
        self.parts = [cols[start:start + chunk]
                      for start in range(0, self.count, chunk)]
        self.ws = (cp.empty(self.words * chunk, cp.float32)
                   if self.count else None)
        self.err = cp.zeros(chunk, cp.int32)


def launch_bep_bem_columns(dev: Mapping, cls: Mapping, *, gmt, julday,
                           declin_urb, dt, itimestep, num_urban_hi,
                           workspace_bytes: int = 256 << 20,
                           plan: ColumnPlan | None = None) -> ColumnPlan:
    """BEP_BEM's column loop (module_sf_bep_bem.F:708-1150), in place.

    ``dev`` holds every name in :data:`COLUMN_ARRAY_ARGUMENTS` as C-contiguous
    device arrays in gpuwm's ``(k, ny, nx)`` layout (``(ny, nx)`` for 2-D;
    ``utype_urb2d`` int32, everything else float32).  Atmospheric inputs are
    ``(nz, ny, nx)`` mass levels; the level outputs ``a_u .. vl`` are
    ``(nz, ny, nx)`` and ``sf`` is ``(nz + 1, ny, nx)``.  Columns with
    ``frc_urb2d <= 0`` are never touched.  The urban columns are launched in
    chunks whose workspace fits ``workspace_bytes``; pass back the returned
    :class:`ColumnPlan` to skip rebuilding it.  A WRF ``stop`` inside the
    column is reported through :mod:`gpuwm.core.health_ledger` (at once, or
    at the forecast's drain when one is active) as :class:`BepBemError`.
    """
    import cupy as cp

    from gpuwm.core import health_ledger

    nz, ny, nx = dev["dz8w"].shape
    if nz < 2:
        raise ValueError("BEP_BEM needs at least two mass levels: z(kte+1) "
                         "is built from dz8w and urban_meso reads level 2")
    if not 0 < int(num_urban_hi) < NZ_UM:
        raise ValueError(f"num_urban_hi={num_urban_hi}: BEP_BEM stops unless "
                         f"it is below nz_um={NZ_UM} (module_sf_bep_bem.F:656)")
    for name in COLUMN_ARRAY_ARGUMENTS:
        array = dev[name]
        dtype = cp.int32 if name == "utype_urb2d" else cp.float32
        if (not isinstance(array, cp.ndarray) or array.dtype != dtype
                or not array.flags.c_contiguous):
            raise TypeError(f"{name}: requires a C-contiguous device "
                            f"{np.dtype(dtype).name} array")
        if array.shape[-2:] != (ny, nx):
            raise ValueError(f"{name}: horizontal shape {array.shape[-2:]} "
                             f"is not {(ny, nx)}")
    if plan is None or plan.nz != nz or plan.shape != (ny, nx):
        plan = ColumnPlan(dev["frc_urb2d"], nz, workspace_bytes)
    if not plan.count:
        return plan
    scalars = {"gmt": np.float32(gmt), "julday": np.int32(julday),
               "declin_urb": np.float32(declin_urb), "dt": np.float32(dt),
               "itimestep": np.int32(itimestep), "nz": np.int32(nz),
               "ny": np.int32(ny), "nx": np.int32(nx),
               "num_urban_hi": np.int32(num_urban_hi)}
    kernel = _bem_module().get_function("bep_bem_columns")
    for part in plan.parts:
        count = int(part.size)
        err = plan.err[:count]
        args = (part, np.int32(count), cls["clsf"], cls["clsi"])
        args += tuple(dev[name] for name in COLUMN_ARRAY_ARGUMENTS)
        args += tuple(scalars[name] for name in COLUMN_SCALAR_ARGUMENTS)
        args += (plan.ws, np.int32(plan.words), err)
        kernel(((count + 31) // 32,), (32,), args)
        status = err.max().reshape(1)
        code = health_ledger.read_status(status, site="urban_bem",
                                         describe=_refuse)
        if code:
            _check_errors(err, part)
    return plan


# ---------------------------------------------------------------------------
# DESIGN 3.4 entry points (infra's UrbanCoupler calls these by name)
# ---------------------------------------------------------------------------

#: BEP_BEM's atmospheric inputs and the ``_prepare_atmosphere`` key each one
#: is (module_sf_noahdrv.F:1650-1677, module_sf_noahmpdrv.F:3658-3685).
ATMOSPHERE_SOURCES = MappingProxyType({
    "dz8w": "dz", "u_phy": "u", "v_phy": "v", "th_phy": "theta",
    "rho": "rho", "p_phy": "pressure", "qv_phy": "qv",
})
#: Kernel surface output -> its name in ``UrbanState.bep_out``, where the
#: shared couple reads it (``urban_bep_couple.BEP_OUTPUTS``).
SURFACE_OUTPUTS = MappingProxyType({
    "rl_up": "rl_up_urb", "rs_abs": "rs_abs_urb", "emiss": "emiss_urb",
    "grdflx_urb": "grdflx_urb",
})
#: Kernel level output -> its ``UrbanState.pbl_terms`` key.
PBL_OUTPUTS = MappingProxyType({
    "a_u": "a_u_bep", "a_v": "a_v_bep", "a_t": "a_t_bep", "a_e": "a_e_bep",
    "b_u": "b_u_bep", "b_v": "b_v_bep", "b_t": "b_t_bep", "b_e": "b_e_bep",
    "b_q": "b_q_bep", "dlg": "dlg_bep", "dl_u": "dl_u_bep", "sf": "sf_bep",
    "vl": "vl_bep",
})
_FIELD_INPUTS = ("swdown", "glw", "swddir", "swddif", "rainbl")
_SOLAR_INPUTS = MappingProxyType({
    "cosz_urb2d": "coszen", "omg_urb2d": "hrang", "xlat": "xlat",
    "xlong": "xlong"})
#: Kernel arguments that are UrbanState arrays under their own names.
STATE_ARGUMENTS = tuple(
    name for name in COLUMN_ARRAY_ARGUMENTS
    if name not in ATMOSPHERE_SOURCES and name not in SURFACE_OUTPUTS
    and name not in PBL_OUTPUTS and name not in _FIELD_INPUTS
    and name not in _SOLAR_INPUTS)


#: The column workspace budget a forecast domain's plan may hold.  A chunk
#: of urban columns costs about one column's serial time (63 ms on the RTX
#: 4090 at 30 levels) whatever its width up to ~10,000 columns, so the
#: budget sets the cost: measured for 10,000 urban columns, 569 ms per call
#: at 256 MiB (9 chunks) and 79 ms at 2.1 GiB (1 chunk).  The plan holds
#: min(urban columns, budget / per-column) columns' worth
#: (:func:`workspace_bytes_per_column`), so a small city holds only what it
#: uses.
FORECAST_WORKSPACE_BYTES = 1 << 30


class ShortwaveSplitMissing(RuntimeError):
    """Option 3 reached the surface step without SWDDIR/SWDDIF."""


def init_state(state, params, *, tlayer0, tsurface0, tdeep0, smois,
               restart: bool) -> None:
    """Nothing to add: infra's cold start already writes option 3's block.

    :func:`gpuwm.core.urban_state.urban_var_init_host` transcribes the whole
    option-3 part of ``urban_var_init`` (module_sf_urban.F:2912-2992);
    ``tests/test_urban_bem_wrf471_parity.py::
    test_option3_cold_start_is_urban_var_init`` grades it against the state
    WRF's own ``urban_var_init`` leaves, word for word.
    """
    del state, params, tlayer0, tsurface0, tdeep0, smois, restart


def after_lsm(state, params, *, lsm: int, fields: dict, atmosphere,
              dt: float, itimestep: int, solar, cfg) -> None:
    """BEP_BEM on every urban column, then the shared BEP couple."""
    del cfg
    from gpuwm.core import urban_bep_couple

    for name in ("swddir", "swddif"):
        if name not in fields:
            raise ShortwaveSplitMissing(
                "sf_urban_physics=3 needs the radiation step's direct and "
                f"diffuse surface shortwave ({name.upper()}): BEP_BEM's "
                "shadow_mas and short_rad_dd put the sun on roads, walls and "
                "windows from them, and without them every building would "
                "be heated by zero sun")
    bep_out = _bep_out(state, urban_bep_couple.BEP_OUTPUTS)
    # module_sf_noahdrv.F:1639-1647 / module_sf_noahmpdrv.F:3646-3656:
    # zeroed on every column before every call.
    for name in urban_bep_couple.BEP_OUTPUTS:
        bep_out[name][...] = 0.0
    pbl = state.pbl_terms
    pbl["b_q_bep"][...] = 0.0
    dev = {name: state.fields[name] for name in STATE_ARGUMENTS}
    for name, source in ATMOSPHERE_SOURCES.items():
        dev[name] = atmosphere[source]
    for name in _FIELD_INPUTS:
        dev[name] = fields[name]
    for name, source in _SOLAR_INPUTS.items():
        dev[name] = getattr(solar, source)
    for name, key in PBL_OUTPUTS.items():
        dev[name] = pbl[key]
    for name, key in SURFACE_OUTPUTS.items():
        dev[name] = bep_out[key]
    state.bem_plan = launch_bep_bem_columns(
        dev, class_tables(params),
        gmt=float(solar.gmt), julday=int(solar.julday),
        declin_urb=float(solar.declin), dt=float(dt),
        itimestep=int(itimestep), num_urban_hi=int(state.num_urban_hi),
        workspace_bytes=FORECAST_WORKSPACE_BYTES,
        plan=getattr(state, "bem_plan", None))
    urban_bep_couple.couple(state, lsm=lsm, fields=fields,
                            atmosphere=atmosphere, dt=dt, bep_out=bep_out)


def after_surface_diagnostics(state, *, lsm: int, fields: dict, atmosphere,
                              cfg) -> None:
    """Options 2 and 3 share the 2 m overrides (the BEP lane's couple)."""
    from gpuwm.core import urban_bep_couple

    urban_bep_couple.after_surface_diagnostics(
        state, lsm=lsm, fields=fields, atmosphere=atmosphere, cfg=cfg)


def _bep_out(state, names) -> dict:
    """The BEP column outputs the couple reads, held on the state.

    Options 2 and 3 share them (``urban_bep_couple.couple`` reads
    ``state.bep_out``); whichever model runs first on a domain allocates.
    """
    import cupy as cp

    held = getattr(state, "bep_out", None)
    shape = state.fields["frc_urb2d"].shape
    if held is None or any(n not in held or held[n].shape != shape
                           for n in names):
        held = {name: cp.zeros(shape, dtype=cp.float32) for name in names}
        state.bep_out = held
    return held
