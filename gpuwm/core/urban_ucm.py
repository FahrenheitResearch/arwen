"""Single-layer urban canopy model (``sf_urban_physics = 1``): host side.

The physics is ``gpuwm/core/kernels/urban_ucm.cu``, a line-faithful
transcription of WRF v4.7.1 ``phys/module_sf_urban.F`` (subroutine ``urban``
and every helper it reaches) plus the three places WRF couples it:

* Noah (``sf_surface_physics = 2``): ``module_sf_noahdrv.F`` 1317-1600, which
  in WRF runs INSIDE Noah's column loop.  Columns are independent, so here it
  runs as a post-LSM kernel fed the rural values Noah itself computed
  (``UrbanState.rural``: ``t1 sheat eta_kinematic eta ssoil albedok q1 sfctmp
  q2k sfcprs zlvl soldn rainbl_used``) -- never re-derived.
* Noah-MP (``sf_surface_physics = 4``): ``noahmp_urban``'s option-1 arm,
  ``module_sf_noahmpdrv.F`` 3374-3598, which WRF calls after ``noahmplsm``
  on the grid fields it left.
* The surface-driver overrides after the 2 m diagnostics:
  ``module_surface_driver.F`` 3001-3021 (Noah) and 3383-3404 (Noah-MP, whose
  T2/Q2/TH2 are also blended here).

This module owns the model-lane entry points the urban driver loads by name
(DESIGN.md section 3.4): :data:`STATE_SPEC`, :data:`DIMENSIONS`,
:func:`init_state`, :func:`after_lsm` and :func:`after_surface_diagnostics`.

Proven against a WRF column oracle (``tools/urban_wrf471_oracle/run_ucm.F90``,
fixture ``gpuwm/data/urban/oracle/ucm/``); see
``tests/test_urban_ucm_wrf471_parity.py``.

Not transcribed, and refused rather than approximated:

* ``slucm_distributed_drag`` (``distributed_aerodynamics_option``): its
  ``Z0_URB2D``/``LF_URB2D_S`` inputs are gridded morphology the engine does
  not carry (DESIGN.md section 2, named follow-up).
* the NUDAPT arm, ``mh_urb > 0`` (``module_sf_urban.F`` 649-788): gridded
  morphology again.  :func:`init_state` refuses a state that carries it.

Two WRF reads of undefined memory are given defined behaviour, identical in
the kernel and in the CPU reference ``gpuwm/verify/urban_ucm_ref.py``:

* ``ETR`` (``module_sf_urban.F:571``) is read by ``SMFLX``/``SRT`` on the
  green-roof dew arm (:1161-1168) before ``TRANSP`` ever wrote it in that
  call.  Defined as zero: no transpiration under dew.  The oracle's ``snan``
  build proves the read is real; its ``zero`` build is the fixture.
* ``tloc`` is set only when ``AHOPTION == 1`` (:623-627) but read by the
  ``IRI_SCHEME == 1`` arm (:874-883).  Defined by the same formula whenever
  either arm needs it.
"""
from __future__ import annotations

import calendar
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

__all__ = [
    "STATE_SPEC",
    "DIMENSIONS",
    "UCM_TABLE_COLUMNS",
    "UCM_STATE_PLANES",
    "UcmParams",
    "pack_params",
    "pack_params_from_rows",
    "init_state",
    "after_lsm",
    "after_surface_diagnostics",
    "run_columns",
    "UrbanCanopyError",
]

#: WRF's soil-layer count for Noah and Noah-MP; the UCM's roof, wall and
#: road layers are ``num_soil_layers`` (``urban_param_init``, :2090-2092).
NUM_URBAN_LAYERS = 4

_TPB = 128

#: The per-UTYPE values ``read_param`` (:1942-2037) hands ``urban``, in the
#: kernel's PT_* column order.  Keys are WRF's local names; the table arrays
#: are the ``<NAME>_TBL`` of ``module_sf_urban``.
UCM_TABLE_COLUMNS = (
    "zr", "z0c", "z0hc", "zdc", "svf", "r", "rw", "hgt", "ah", "alh",
    "betr", "betb", "betg", "capr", "capb", "capg", "aksr", "aksb", "aksg",
    "albr", "albb", "albg", "epsr", "epsb", "epsg", "z0r", "z0b", "z0g",
    "z0hb", "z0hg", "trlend", "tblend", "tglend", "akanda_urban",
)
_TBL_NAME = {name: f"{name.upper()}_TBL" for name in UCM_TABLE_COLUMNS}

#: The module-level arrays and scalars ``urban`` reads, in the kernel's PG_*
#: layout: (name, length).
UCM_GLOBAL_LAYOUT = (
    ("dzr", 4), ("dzb", 4), ("dzg", 4), ("dzgr", 4), ("porimp", 3),
    ("dengimp", 3), ("ahdiuprf", 24), ("alhseason", 4), ("alhdiuprf", 48),
    ("fgr", 1),
)
#: The integer switches, in the kernel's SW_* order.
UCM_SWITCHES = ("boundr", "boundb", "boundg", "ch_scheme", "ts_scheme",
                "ahoption", "alhoption", "imp_scheme", "iri_scheme",
                "groption")

#: The admitted values of each switch (``module_sf_urban.F``'s IF arms).  A
#: table row outside them would be silently read as the default arm.
_SWITCH_DOMAIN = MappingProxyType({
    "boundr": (1, 2), "boundb": (1, 2), "boundg": (1, 2),
    "ch_scheme": (1, 2), "ts_scheme": (1, 2), "ahoption": (0, 1),
    "alhoption": (0, 1), "imp_scheme": (1, 2), "iri_scheme": (0, 1),
    "groption": (0, 1),
})

#: Kernel state planes (SP_* order): Registry name -> layers (0 = 2-D).
UCM_STATE_PLANES = (
    ("tr_urb2d", 0), ("tb_urb2d", 0), ("tg_urb2d", 0), ("tc_urb2d", 0),
    ("qc_urb2d", 0), ("uc_urb2d", 0),
    ("trl_urb3d", NUM_URBAN_LAYERS), ("tbl_urb3d", NUM_URBAN_LAYERS),
    ("tgl_urb3d", NUM_URBAN_LAYERS),
    ("xxxr_urb2d", 0), ("xxxb_urb2d", 0), ("xxxg_urb2d", 0),
    ("xxxc_urb2d", 0),
    ("cmr_sfcdif", 0), ("chr_sfcdif", 0), ("cmc_sfcdif", 0),
    ("chc_sfcdif", 0), ("cmgr_sfcdif", 0), ("chgr_sfcdif", 0),
    ("cmcr_urb2d", 0), ("tgr_urb2d", 0),
    ("tgrl_urb3d", NUM_URBAN_LAYERS), ("smr_urb3d", NUM_URBAN_LAYERS),
    ("drelr_urb2d", 0), ("drelb_urb2d", 0), ("drelg_urb2d", 0),
    ("flxhumr_urb2d", 0), ("flxhumb_urb2d", 0), ("flxhumg_urb2d", 0),
    ("ts_urb2d", 0), ("sh_urb2d", 0), ("lh_urb2d", 0), ("g_urb2d", 0),
    ("rn_urb2d", 0), ("psim_urb2d", 0), ("psih_urb2d", 0),
    ("gz1oz0_urb2d", 0), ("u10_urb2d", 0), ("v10_urb2d", 0),
    ("th2_urb2d", 0), ("q2_urb2d", 0), ("ust_urb2d", 0), ("akms_urb2d", 0),
)

#: Arrays every urban option carries (DESIGN.md 3.1); the infra state owns
#: them, this option reads and writes them.
_SHARED_ARRAYS = frozenset(("ts_urb2d", "sh_urb2d", "lh_urb2d", "g_urb2d",
                            "rn_urb2d"))

#: Option-1 state, Registry names lowercase: name -> (dtype, layers).
#: ``layers`` 0 is a 2-D ``(ny, nx)`` field, n > 0 a ``(n, ny, nx)`` one.
#: ``cm*/ch*_sfcdif`` are the SFCDIF_URB exchange coefficients WRF carries
#: between calls (Registry.EM_COMMON:984-989, restart "r").
STATE_SPEC: Mapping[str, tuple[str, int | str]] = MappingProxyType({
    name: ("float32", layers)
    for name, layers in UCM_STATE_PLANES if name not in _SHARED_ARRAYS})

#: Option 1 has no BEP-style dimensions.
DIMENSIONS: Mapping[str, int] = MappingProxyType({})

#: Rural hand-off names the Noah kernel reads (RH_* order).
_NOAH_RURAL = ("t1", "sheat", "eta_kinematic", "eta", "ssoil", "albedok",
               "q1", "sfctmp", "q2k", "sfcprs", "zlvl", "soldn",
               "rainbl_used")
#: Grid fields the Noah/Noah-MP kernels read or write (GF_* order).
_GRID_FIELDS = ("albedo", "hfx", "qfx", "lh", "grdflx", "tsk", "qsfc", "ust",
                "chs", "chs2", "cqs2", "glw", "znt", "swdown", "rainbl")
#: Grid fields the override kernel writes (OF_* order).
_OVERRIDE_FIELDS = ("u10", "v10", "psim", "psih", "gz1oz0", "akhs", "akms",
                    "chs", "t2", "th2", "q2", "psfc")
#: Noah-MP's own 2 m parts (surface_driver.F:3389-3392), from the hand-off.
_NOAHMP_T2_PARTS = ("fvegxy", "t2mvxy", "t2mbxy", "q2mvxy", "q2mbxy")

_ERRORS = {
    1: ("ZDC+Z0C+2m is larger than the 1st WRF level "
        "(module_sf_urban.F:825, WRF's own fatal): the canopy displacement "
        "height plus roughness reaches the first model level, so the "
        "log-profile the UCM integrates is undefined there"),
    2: ("an urban column's UTYPE is outside the loaded URBPARM table: the "
        "kernel would read another category's row"),
}


class UrbanCanopyError(RuntimeError):
    """A condition WRF stops the model on, raised with its breakage named."""


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UcmParams:
    """Packed host arrays for the kernel (float32 / int32)."""

    table: np.ndarray        # (n_utype, len(UCM_TABLE_COLUMNS))
    globals_: np.ndarray     # (99,)
    switches: np.ndarray     # (len(UCM_SWITCHES) + 1,) last = n_utype
    frc_urb: np.ndarray      # (n_utype,) FRC_URB_TBL, for init defaulting

    @property
    def n_utype(self) -> int:
        return int(self.table.shape[0])


def _pack(table_rows: np.ndarray, glob: Mapping[str, object],
          switches: Mapping[str, int], frc_urb) -> UcmParams:
    table = np.ascontiguousarray(np.asarray(table_rows, dtype=np.float32))
    if table.ndim != 2 or table.shape[1] != len(UCM_TABLE_COLUMNS):
        raise ValueError("UCM table must be (n_utype, "
                         f"{len(UCM_TABLE_COLUMNS)}) float32")
    flat = []
    for name, length in UCM_GLOBAL_LAYOUT:
        values = np.atleast_1d(np.asarray(glob[name], dtype=np.float32))
        if values.size < length:
            raise ValueError(f"UCM global {name!r} needs {length} values, "
                             f"got {values.size}")
        flat.append(values[:length])
    globals_ = np.ascontiguousarray(np.concatenate(flat).astype(np.float32))
    ints = []
    for name in UCM_SWITCHES:
        value = int(switches[name])
        if value not in _SWITCH_DOMAIN[name]:
            raise ValueError(
                f"URBPARM {name.upper()} = {value} is outside WRF's arms "
                f"{_SWITCH_DOMAIN[name]}: module_sf_urban.F would silently "
                "take its fall-through arm")
        ints.append(value)
    ints.append(table.shape[0])
    return UcmParams(table=table, globals_=globals_,
                     switches=np.asarray(ints, dtype=np.int32),
                     frc_urb=np.asarray(frc_urb, dtype=np.float32))


def _field(obj, *names):
    """First of ``names`` found on ``obj`` as a mapping key or attribute."""
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise KeyError(f"none of {names} on {type(obj).__name__}")


def pack_params(params) -> UcmParams:
    """Pack the infra ``UrbanParams`` (``urban_param_init``'s product).

    The table arrays are read by WRF's ``<NAME>_TBL`` names and the switches
    and module arrays by WRF's own names, so this is the only place that
    knows the kernel's column order.
    """
    rows = np.stack([np.asarray(_field(params, _TBL_NAME[c], _TBL_NAME[c].lower(), c),
                                dtype=np.float32)
                     for c in UCM_TABLE_COLUMNS], axis=1)
    glob = {name: _field(params, name.upper(), name) for name, _ in UCM_GLOBAL_LAYOUT}
    sw = {}
    for name in UCM_SWITCHES:
        if name.startswith("bound") or name in ("ch_scheme", "ts_scheme"):
            sw[name] = _field(params, f"{name.upper()}_DATA", f"{name}_data",
                              name.upper(), name)
        else:
            sw[name] = _field(params, name.upper(), name)
    frc = np.asarray(_field(params, "FRC_URB_TBL", "frc_urb_tbl"), np.float32)
    return _pack(rows, glob, sw, frc)


_PACKED: dict = {}


def _packed(params) -> UcmParams:
    """``pack_params`` once per parameter object (a reference is held so
    its identity cannot be recycled)."""
    if isinstance(params, UcmParams):
        return params
    held = _PACKED.get(id(params))
    if held is None or held[0] is not params:
        held = (params, pack_params(params))
        _PACKED[id(params)] = held
    return held[1]


def pack_params_from_rows(table: Mapping[str, np.ndarray],
                          switches: Mapping[str, object]) -> UcmParams:
    """Pack from the oracle's own dump of ``read_param`` and the switches.

    ``table`` maps every :data:`UCM_TABLE_COLUMNS` name (and ``frc_urb``) to
    a per-UTYPE array; ``switches`` carries the :data:`UCM_SWITCHES` ints and
    the :data:`UCM_GLOBAL_LAYOUT` arrays.
    """
    rows = np.stack([np.asarray(table[c], np.float32) for c in UCM_TABLE_COLUMNS],
                    axis=1)
    glob = {name: switches[name] for name, _ in UCM_GLOBAL_LAYOUT}
    sw = {name: switches[name] for name in UCM_SWITCHES}
    return _pack(rows, glob, sw, table["frc_urb"])


_DEVICE_PARAMS: dict = {}


def _device_params(p: UcmParams):
    import cupy as cp

    key = id(p)
    held = _DEVICE_PARAMS.get(key)
    if held is None:
        held = (p, (cp.asarray(p.table.ravel()), cp.asarray(p.globals_),
                    cp.asarray(p.switches)))
        _DEVICE_PARAMS[key] = held
    return held[1]


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def _get(obj, name):
    if isinstance(obj, Mapping):
        return obj[name]
    getter = getattr(obj, "__getitem__", None)
    if getter is not None:
        try:
            return getter(name)
        except (KeyError, TypeError):
            pass
    return getattr(obj, name)


def init_state(state, params, *, tlayer0, tsurface0, tdeep0, smois,
               restart: bool) -> None:
    """Option-1 hook after infra's ``urban_var_init``.

    ``gpuwm.core.urban_state.urban_var_init_host`` already transcribes every
    option-1 line of ``urban_var_init`` (module_sf_urban.F:2840-2889) and is
    graded against the Fortran routine; the ``cm*/ch*_sfcdif`` coefficients
    this module adds to the spec start at WRF's Registry zero there too.  So
    the hook only refuses what this port cannot run: a column carrying
    gridded morphology (``MH_URB2D > 0``) would take the NUDAPT arm
    (:649-788), which is not transcribed, and run on table morphology while
    claiming gridded.
    """
    import cupy as cp

    del params, tlayer0, tsurface0, tdeep0, smois, restart
    mh = _get(state, "mh_urb2d")
    if bool(cp.any(cp.asarray(mh) > 0)):
        raise UrbanCanopyError(
            "MH_URB2D > 0 selects module_sf_urban.F's NUDAPT gridded-"
            "morphology arm (:649-788), which this port does not transcribe; "
            "the column would run on table morphology while claiming gridded")


# ---------------------------------------------------------------------------
# launches
# ---------------------------------------------------------------------------

_PTR_CACHE: dict = {}


def _pointer_table(arrays):
    """One device uint64 array of data pointers (built once per identity)."""
    import cupy as cp

    key = tuple(int(a.data.ptr) for a in arrays)
    held = _PTR_CACHE.get(key)
    if held is None:
        held = (tuple(arrays), cp.asarray(np.asarray(key, dtype=np.uint64)))
        _PTR_CACHE[key] = held
    return held[1]


def _checked(arr, shape, name, dtype=None):
    import cupy as cp

    dtype = dtype or cp.float32
    if not isinstance(arr, cp.ndarray) or arr.dtype != dtype \
            or tuple(arr.shape) != tuple(shape) or not arr.flags.c_contiguous:
        raise ValueError(f"{name}: expected C-contiguous {tuple(shape)} "
                         f"{np.dtype(dtype).name} on the device")
    return arr


def _state_arrays(state, ny, nx):
    out = []
    for name, layers in UCM_STATE_PLANES:
        shape = (ny, nx) if layers == 0 else (layers, ny, nx)
        out.append(_checked(_get(state, name), shape, name))
    return out


_STATUS: dict = {}


def _status_word(state):
    """A persistent one-word status buffer per urban state, zeroed on the
    device before each launch (no per-step allocation, capturable)."""
    import cupy as cp

    held = _STATUS.get(id(state))
    if held is None or held[0] is not state:
        held = (state, cp.zeros(1, dtype=cp.uint32))
        _STATUS[id(state)] = held
    held[1].fill(0)
    return held[1]


def _describe_status(flags: int) -> None:
    """Raise for a UCM status word: bit ``code - 1`` per kernel code."""
    reasons = [text for code, text in sorted(_ERRORS.items())
               if flags & (1 << (code - 1))]
    raise UrbanCanopyError("; ".join(reasons) or f"UCM status {flags:#x}")


def _raise_on(err) -> None:
    """Report the kernel's status word through the engine's health path:
    read now when no ledger is active, else recorded for the ledger's drain
    (``gpuwm.core.health_ledger``), so a deferring step keeps a launch
    sequence that does not depend on device values."""
    from gpuwm.core import health_ledger

    flags = health_ledger.read_status(err, site="urban_ucm",
                                      describe=_describe_status)
    if flags:
        _describe_status(flags)


def _jmonth(solar) -> int:
    """``cal_mon_day(julian, julyr, jmonth, jday)`` (module_ra_gfdleta).

    WRF's leap rule there is ``MOD(julyr,4) == 0``; its MONTH table carries
    SAVE via DATA, so after a leap-year call February keeps 29 days for the
    life of the process.  The month is computed here from the rule alone.
    """
    for name in ("jmonth",):
        if hasattr(solar, name):
            return int(getattr(solar, name))
    julday = int(_field(solar, "julday", "julian"))
    year = int(_field(solar, "julyr", "year"))
    days = [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31,
            30, 31]
    for month, length in enumerate(days, start=1):
        if julday <= length:
            return month
        julday -= length
    return 12


_MASKS: dict = {}


def _urban_mask(state):
    """The urban-column mask as int32 (``UrbanState.urban_mask`` is
    ``utype_urb2d > 0``, fixed for the run), converted once per mask."""
    import cupy as cp

    mask = _field(state, "urban_mask", "urban_columns")
    if mask.dtype == cp.int32 and mask.flags.c_contiguous:
        return mask
    held = _MASKS.get(id(mask))
    if held is None or held[0] is not mask:
        held = (mask, cp.ascontiguousarray(mask.astype(cp.int32)))
        _MASKS[id(mask)] = held
    return _checked(held[1], mask.shape, "urban mask", cp.int32)


def after_lsm(state, params, *, lsm: int, fields: dict, atmosphere: Mapping,
              dt: float, itimestep: int, solar, cfg) -> None:
    """Run the UCM on every urban column and blend it, in place.

    Called by the urban driver after the LSM and BEFORE ``rainbl`` is
    zeroed.  ``lsm`` 2 is Noah (the rural values come from
    ``state.rural``), 4 is Noah-MP (the rural values are the grid fields).
    """
    import cupy as cp

    from gpuwm.core.kernels import get_kernel

    del itimestep, cfg
    p = _packed(params)
    tab, glob, isw = _device_params(p)
    tsk = fields["tsk"]
    ny, nx = tsk.shape
    urban = _urban_mask(state)
    utype = _checked(_get(state, "utype_urb2d"), (ny, nx), "utype_urb2d", cp.int32)
    frc = _checked(_get(state, "frc_urb2d"), (ny, nx), "frc_urb2d")
    u1 = cp.ascontiguousarray(atmosphere["u"][0])
    v1 = cp.ascontiguousarray(atmosphere["v"][0])
    hrang = _checked(cp.ascontiguousarray(_field(solar, "hrang", "omg")),
                     (ny, nx), "hrang")
    sp = _pointer_table(_state_arrays(state, ny, nx))
    fp = _pointer_table([_checked(fields[n], (ny, nx), n) for n in _GRID_FIELDS])
    err = _status_word(state)
    blocks = (ny * nx + _TPB - 1) // _TPB
    jmonth = np.int32(_jmonth(solar))
    if int(lsm) == 2:
        rural = _field(state, "rural")
        rp = _pointer_table([_checked(_get(rural, n), (ny, nx), f"rural {n}")
                             for n in _NOAH_RURAL])
        kern = get_kernel("urban_ucm", "ucm_noah_after_lsm")
        kern((blocks,), (_TPB,),
             (urban, utype, frc, u1, v1, hrang, rp, fp, sp, tab, glob, isw,
              np.float32(dt), jmonth, err, np.int32(ny), np.int32(nx)))
    elif int(lsm) == 4:
        t3d1 = cp.ascontiguousarray(atmosphere["temperature"][0])
        qv1 = cp.ascontiguousarray(atmosphere["qv"][0])
        p8w = atmosphere["p_interface"]
        p8w1 = cp.ascontiguousarray(p8w[0])
        p8w2 = cp.ascontiguousarray(p8w[1])
        dz1 = cp.ascontiguousarray(atmosphere["dz"][0])
        kern = get_kernel("urban_ucm", "ucm_noahmp_after_lsm")
        kern((blocks,), (_TPB,),
             (urban, utype, frc, u1, v1, t3d1, qv1, p8w1, p8w2, dz1, hrang,
              fp, sp, tab, glob, isw, np.float32(dt), jmonth, err,
              np.int32(ny), np.int32(nx)))
    else:
        raise ValueError(
            f"sf_surface_physics={lsm} never calls module_sf_urban (WRF couples "
            "the UCM only through Noah and Noah-MP)")
    _raise_on(err)


def after_surface_diagnostics(state, *, lsm: int, fields: dict,
                              atmosphere: Mapping, cfg) -> None:
    """The option-1 overrides after the 2 m diagnostics, in place."""
    import cupy as cp

    from gpuwm.core.kernels import get_kernel

    del atmosphere, cfg
    tsk = fields["tsk"]
    ny, nx = tsk.shape
    urban = _urban_mask(state)
    frc = _checked(_get(state, "frc_urb2d"), (ny, nx), "frc_urb2d")
    sp = _pointer_table(_state_arrays(state, ny, nx))
    fp = _pointer_table([_checked(fields[n], (ny, nx), n) for n in _OVERRIDE_FIELDS])
    if int(lsm) == 4:
        rural = _field(state, "rural")
        parts = [_checked(_get(rural, n), (ny, nx), f"rural {n}")
                 for n in _NOAHMP_T2_PARTS]
        noahmp = 1
    elif int(lsm) == 2:
        parts = [frc] * len(_NOAHMP_T2_PARTS)
        noahmp = 0
    else:
        raise ValueError(f"sf_surface_physics={lsm} never calls module_sf_urban")
    kern = get_kernel("urban_ucm", "ucm_overrides")
    blocks = (ny * nx + _TPB - 1) // _TPB
    kern((blocks,), (_TPB,),
         (urban, frc, fp, sp, *parts, np.int32(noahmp), np.int32(ny), np.int32(nx)))


# ---------------------------------------------------------------------------
# column entry (tests and the oracle)
# ---------------------------------------------------------------------------

#: ``urban``'s inputs as the column kernel takes them (CI_* order).
UCM_COLUMN_INPUTS = ("utype", "jmonth", "ta", "qa", "ua", "u1", "v1", "ssg",
                     "llg", "rain", "rhoo", "za", "omg", "delt", "znt_in",
                     "chs", "chs2")
#: ``urban``'s outputs (CO_* order).
UCM_COLUMN_OUTPUTS = ("ts", "qs", "sh", "lh", "lh_kinematic", "sw", "alb",
                      "lw", "g", "rn", "psim", "psih", "gz1oz0", "u10", "v10",
                      "th2", "q2", "ust", "znt")


def run_columns(params: UcmParams, inputs: Mapping[str, np.ndarray],
                state: Mapping[str, np.ndarray]):
    """Run ``urban`` over n independent columns on the device.

    ``state`` maps every :data:`UCM_STATE_PLANES` name to a host array of
    shape ``(n,)`` or ``(4, n)``; returns ``(outputs, state_after, codes)``
    as host arrays.
    """
    import cupy as cp

    from gpuwm.core.kernels import get_kernel

    n = int(np.asarray(inputs["ta"]).shape[0])
    dev_in = []
    for name in UCM_COLUMN_INPUTS:
        dtype = np.int32 if name in ("utype", "jmonth") else np.float32
        dev_in.append(cp.asarray(np.ascontiguousarray(inputs[name], dtype=dtype)))
    dev_out = [cp.zeros(n, dtype=cp.float32) for _ in UCM_COLUMN_OUTPUTS]
    dev_state = []
    for name, layers in UCM_STATE_PLANES:
        shape = (n,) if layers == 0 else (layers, n)
        host = np.ascontiguousarray(state.get(name, np.zeros(shape, np.float32)),
                                    dtype=np.float32)
        if host.shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {host.shape}")
        dev_state.append(cp.asarray(host))
    codes = cp.zeros(n, dtype=cp.int32)
    tab, glob, isw = _device_params(params)
    kern = get_kernel("urban_ucm", "ucm_column_test")
    ip = cp.asarray(np.asarray([a.data.ptr for a in dev_in], dtype=np.uint64))
    op = cp.asarray(np.asarray([a.data.ptr for a in dev_out], dtype=np.uint64))
    sp = cp.asarray(np.asarray([a.data.ptr for a in dev_state], dtype=np.uint64))
    kern(((n + _TPB - 1) // _TPB,), (_TPB,),
         (ip, op, sp, tab, glob, isw, codes, np.int32(n)))
    outputs = {name: a.get() for name, a in zip(UCM_COLUMN_OUTPUTS, dev_out)}
    after = {name: a.get() for (name, _), a in zip(UCM_STATE_PLANES, dev_state)}
    return outputs, after, codes.get()
