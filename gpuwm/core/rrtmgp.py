"""GPU RTE+RRTMGP longwave/shortwave radiation.

The coefficient loader in this first section mirrors the transformations in
RTE+RRTMGP ``mo_optics_utils_rrtmgp.F90`` and
``mo_gas_optics_rrtmgp.F90:init_abs_coeffs``.  NetCDF values remain float64 on
the host; :meth:`GasTables.to_device` and :meth:`CloudTables.to_device` create
and cache packed FP32 device copies.

Reference: earth-system-radiation/rte-rrtmgp commit
fa107a16120051c4124305c6b3d4c87059119f58; coefficient data are rrtmgp-data
v1.9.  See ``gpuwm/data/rrtmgp/PROVENANCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset, chartostring

from gpuwm.config import DEFAULT_COLUMN_CHUNK
from gpuwm.physics_compat import (
    WRF_RRTMG_TO_RTE_RRTMGP,
    WRF_RRTMG_TO_RTE_RRTMGP_V1,
)
from gpuwm.core.mynn_radiation import (
    merge_mynn_bl_clouds,
    mynn_bl_cloud_active,
    wrf_itimestep,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rrtmgp"
DTYPE = np.float32

# ---------------------------------------------------------------------------
# Radiation-facing effective radii are contracted in MICRONS across every
# microphysics writer (state.py background fills, thompson_contract.py,
# the WSM6/Thompson/Morrison kernels).  These physical-plausibility bands
# gate that contract at the radiation boundary: clouds with re_liq of
# 0.0000025 um or 2,500,000 um do not exist, so any writer that emits a
# radius in the wrong metric unit must fail here instead of silently
# radiating at a clip floor.  Every writer background-fills clear cells
# with 2.49-25 um values, and every background times or divided by any
# metric-prefix factor (>= 1000) leaves its band, so a metre, millimetre,
# or nanometre emission trips the gate on the very first radiation call
# regardless of scheme or cloud state.  Bounds admit each scheme's clamp
# extremes: WSM6/Thompson clamps 2.49-50/4.99-125/9.99-999 um; Morrison
# lambda bounds give cloud (pgam+3)/(2(pgam+1)) = 0.59-0.83 um through
# 50 um, ice 1.5 through 525 um (1.5e6*(2*MDCS+100e-6)), snow 15 through
# 3000 um (1.5e6*2000e-6); backgrounds 2.49-25 um.
EFFECTIVE_RADIUS_PLAUSIBLE_UM = {
    "effc": (0.5, 100.0),
    "effi": (1.0, 600.0),
    "effs": (1.0, 5000.0),
}

# Snow treatment in the radiative ice path for schemes that provide an
# explicit snow effective radius (the WSM6/Thompson coupling surface).
#   full-snow-mass-into-ice: the adapter's original coupling -- snow joins
#     the ice optical path at full mass with its native radius.
#   wrf-rrtmg-130um-snow-discount: WRF v4.6.1's option-4 explicit-radius
#     coupling (inflg/iceflg=5) -- the ice path is cloud ice only, snow
#     mass is discounted by MIN(0.99, (130/re_s)^2) and re_s is capped at
#     130 um (module_ra_rrtmg_lw.F:12500-12532, module_ra_rrtmg_sw.F:
#     11040-11067; fixture: tests/data/wrf_rrtmg_snow_discount_fixture.csv).
# Selection is bound to the wrf_rrtmg_compatibility receipt token so no
# already-issued run is relabeled; unknown tokens fail closed.
SNOW_TREATMENT_FULL_MASS = "full-snow-mass-into-ice"
SNOW_TREATMENT_WRF_DISCOUNT = "wrf-rrtmg-130um-snow-discount"
SNOW_TREATMENTS = (SNOW_TREATMENT_FULL_MASS, SNOW_TREATMENT_WRF_DISCOUNT)

_SNOW_TREATMENT_BY_COMPATIBILITY = {
    # Native RTE+RRTMGP selection: not a WRF-mapped run, keeps the
    # adapter's original coupling unchanged.
    "none": SNOW_TREATMENT_FULL_MASS,
    # -v1 receipts predate the WRF-matching snow coupling; they keep the
    # behavior they were issued under.
    WRF_RRTMG_TO_RTE_RRTMGP_V1: SNOW_TREATMENT_FULL_MASS,
    # -v2 (current importer default): WRF-matching snow discount.
    WRF_RRTMG_TO_RTE_RRTMGP: SNOW_TREATMENT_WRF_DISCOUNT,
}


def snow_treatment_for_compatibility(token: str) -> str:
    """Map a ``wrf_rrtmg_compatibility`` receipt token to a snow treatment.

    Fails closed: a token this build does not recognize must never be
    silently coerced onto either behavior.
    """
    try:
        return _SNOW_TREATMENT_BY_COMPATIBILITY[token]
    except KeyError:
        raise ValueError(
            "unknown wrf_rrtmg_compatibility token for the radiation "
            f"snow treatment: {token!r}; known tokens: "
            f"{sorted(_SNOW_TREATMENT_BY_COMPATIBILITY)}") from None


# ---------------------------------------------------------------------------
# Which cloud-optics coupling each microphysics selector gets.
#
# This table decides THREE things at once inside
# :meth:`RRTMGPRadiation.__call__`, which is why it is stated here as data
# with its WRF citations rather than inlined as a ``.get(mp, "kessler")``
# default: the branch :func:`hydrometeor_paths` takes (scheme-native radii
# vs. Kessler's constant 10 um / 50 um pair), whether the scheme's
# effc/effi/effs columns are read out of state at all, and -- through
# :data:`_ICE_ACTIVE_SCHEMES` -- the ``f_qi``/``f_qs`` flags
# :func:`cal_cldfra1` is called with.  A scheme that falls through to
# "kessler" therefore does not merely lose its radii: its ice and snow stop
# producing cloud fraction, and an overcast ice cloud radiates as clear sky.
#
# THE ENTRY THAT WAS MISSING.  ``28`` (THOMPSONAERO) is the aerosol-aware
# Thompson package and takes the SAME radiative coupling as classic
# Thompson.  WRF's authority for that, all in the stock v4.6.1 tree:
#
#   * ``Registry/Registry.EM_COMMON:3036`` declares
#     ``package thompsonaero mp_physics==28 - moist:qv,qc,qr,qi,qs,qg;
#     scalar:...;state:re_cloud,re_ice,re_snow`` -- character for character
#     the same ``moist:`` inventory and the same three ``re_*`` state
#     fields as line 3024's ``thompson`` (mp==8).  ``F_QI``/``F_QS`` are
#     therefore both true for mp=28, which is exactly what
#     ``cal_cldfra1`` keys on.
#   * ``phys/module_physics_init.F:1005-1006`` lists ``THOMPSON`` and
#     ``THOMPSONAERO`` as two members of ONE disjunction that sets
#     ``has_reqc = has_reqi = has_reqs = 1`` (:1021-1023); the P3 /
#     Jensen-Ishmael ``has_reqs = 0`` override at :1027-1033 does not
#     name THOMPSONAERO.  So mp=28 hands radiation all three radii.
#     Those same three flags are the ONLY gate on the block that computes
#     them -- ``module_mp_thompson.F:1466`` opens
#     ``IF (has_reqc.ne.0 .and. has_reqi.ne.0 .and. has_reqs.ne.0)`` around
#     calc_effectRad and the :1475-1477 clamps -- so in WRF a scheme
#     computes re_cloud/re_ice/re_snow if and only if radiation consumes
#     them.  Computing them for mp=28 and then discarding them at the
#     radiation boundary is not a WRF configuration at all.
#   * Neither RRTMG wrapper branches on the selector for any of this:
#     ``phys/module_ra_rrtmg_lw.F`` tests ``mp_physics`` only at
#     :12131-12136 and ``_sw.F`` only at :10732-10737, both for
#     FER_MP_HIRES / FER_MP_HIRES_ADVECT / ETAMP_HWRF.  ``cal_cldfra1``'s
#     only ``mp_physics`` branch is the same Ferrier one
#     (``phys/module_radiation_driver.F:3926-3937``); mp=8 and mp=28 both
#     take the ``F_QI .and. F_QC .and. F_QS`` arm at :3870-3877.
#
# gpuwm's own side of the contract is already in place: ``mp_physics == 28``
# allocates effc/effi/effs and background-fills them with the same
# RE_QC_BG/RE_QI_BG/RE_QS_BG values mp=8 uses
# (``gpuwm/core/state.py``), and the mp=28 adapter writes them every step
# through ``launch_aerosol_effective_radius``
# (``gpuwm/core/microphysics_aerosol.py``) under mp_gt_driver's OWN clamps
# (module_mp_thompson.F:1466-1479), which is the identical clamp pair mp=8
# takes because mp_gt_driver is one driver serving both packages.
# ``gpuwm/core/rrtmg_legacy.py`` already carried this same judgement
# (``_MP_DECLARES_RADII[28] = True``, ``_LEGACY_ICE_ACTIVE_MICROPHYSICS``);
# this table is the RTE+RRTMGP half of it, and the two are pinned equal by
# ``tests/test_rrtmgp.py``.
#
# FAILS CLOSED.  Every selector ``gpuwm/config.py`` accepts has a row.  A
# selector without one raises instead of silently resolving to Kessler --
# a silent default is precisely how mp=28 spent four waves radiating its
# ice clouds as clear sky.
_MP_CLOUD_OPTICS_SCHEME = {
    # Registry.EM_COMMON:3014, package passiveqv mp_physics==0 - moist:qv.
    # No condensate species exist at all, so the constant-radius branch is
    # inert: it multiplies zero paths.
    0: "kessler",
    # Registry.EM_COMMON:3015, package kesslerscheme mp_physics==1 -
    # moist:qv,qc,qr.  No qi, no qs, no re_* state.
    1: "kessler",
    6: "wsm6",       # Registry.EM_COMMON:3021, wsm6scheme
    8: "thompson",   # Registry.EM_COMMON:3024, thompson
    10: "morrison",  # Registry.EM_COMMON:3026, morr_two_moment
    18: "nssl",      # Registry.EM_COMMON:3033, nssl_2mom
    28: "thompson",  # Registry.EM_COMMON:3036, thompsonaero
}

#: Schemes whose Registry package carries ``qi`` and ``qs`` in ``moist``,
#: i.e. the ones for which the radiation driver's ``F_QI``/``F_QS`` are
#: true and :func:`cal_cldfra1` takes its QCLD = QI + QC + QS arm
#: (module_radiation_driver.F:3870-3877).  Derived from the table above so
#: the two can never disagree; Kessler's package has neither species.
_ICE_ACTIVE_SCHEMES = ("wsm6", "thompson", "morrison", "nssl")


def cloud_optics_scheme(mp_physics) -> str:
    """Resolve an ``mp_physics`` selector to its cloud-optics coupling.

    See :data:`_MP_CLOUD_OPTICS_SCHEME`.  Raises rather than defaulting:
    an unmapped selector must not silently inherit Kessler's constant
    radii and ice-free cloud fraction.
    """
    selector = int(mp_physics)
    try:
        return _MP_CLOUD_OPTICS_SCHEME[selector]
    except KeyError:
        raise NotImplementedError(
            f"mp_physics={selector} has no RTE+RRTMGP cloud-optics coupling; "
            "add a row to gpuwm.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME with its "
            "Registry package (which fixes F_QI/F_QS for cal_cldfra1) and "
            "its module_physics_init.F use_mp_re membership (which fixes "
            "whether its effective radii reach cloud optics) rather than "
            "letting it fall through to Kessler's constant 10 um / 50 um "
            "radii") from None


def scheme_is_ice_active(scheme: str) -> bool:
    """``F_QI``/``F_QS`` for a resolved cloud-optics scheme name."""

    return scheme in _ICE_ACTIVE_SCHEMES


# RRTMGP v1.9's lowest reference pressure.  WRF's wrappers use 0/1e-5 mb
# at TOA, but the pinned RRTMGP example raises that interface to the gas-table
# floor before computing dry-column amounts (the same adaptation formerly
# applied directly to gpuwm's model top).
RRTMGP_TOA_PRESSURE_PA = 1.005183574463
WRF_LW_UPPER_DELTA_P_PA = 400.0
MAX_RADIATION_LAYERS = 128

# module_ra_rrtmg_lw.F:11904-11932.  Pressures are hPa.  The table is the
# weighted standard-atmosphere mean used by WRF to temperature its 4-hPa
# model-top buffer layers.
_WRF_LW_PPROF_HPA = np.array([
    1000.00, 855.47, 731.82, 626.05, 535.57, 458.16,
    391.94, 335.29, 286.83, 245.38, 209.91, 179.57,
    153.62, 131.41, 112.42, 96.17, 82.27, 70.38,
    60.21, 51.51, 44.06, 37.69, 32.25, 27.59,
    23.60, 20.19, 17.27, 14.77, 12.64, 10.81,
    9.25, 7.91, 6.77, 5.79, 4.95, 4.24,
    3.63, 3.10, 2.65, 2.27, 1.94, 1.66,
    1.42, 1.22, 1.04, 0.89, 0.76, 0.65,
    0.56, 0.48, 0.41, 0.35, 0.30, 0.26,
    0.22, 0.19, 0.16, 0.14, 0.12, 0.10,
], np.float64)
_WRF_LW_TPROF_K = np.array([
    286.96, 281.07, 275.16, 268.11, 260.56, 253.02,
    245.62, 238.41, 231.57, 225.91, 221.72, 217.79,
    215.06, 212.74, 210.25, 210.16, 210.69, 212.14,
    213.74, 215.37, 216.82, 217.94, 219.03, 220.18,
    221.37, 222.64, 224.16, 225.88, 227.63, 229.51,
    231.50, 233.73, 236.18, 238.78, 241.60, 244.44,
    247.35, 250.33, 253.32, 256.30, 259.22, 262.12,
    264.80, 266.50, 267.59, 268.44, 268.69, 267.76,
    266.13, 263.96, 261.54, 258.93, 256.15, 253.23,
    249.89, 246.67, 243.48, 240.25, 236.66, 233.86,
], np.float64)


@dataclass(frozen=True)
class _RadiationColumnProfile:
    play: object
    plev: object
    tlay: object
    tlev: object
    qv: object
    model_nlay: int
    upper_nlay: int


def rrtmgp_above_model_layer_counts(
        p_top: float, *, pressure_floor: float = RRTMGP_TOA_PRESSURE_PA,
) -> tuple[int, int]:
    """Return WRF v4.6.1's ``(LW, SW)`` above-model layer counts.

    LW uses ``nint(p_top*.01/4)`` 4-hPa layers
    (module_ra_rrtmg_lw.F:11565,12998-13001).  SW uses one model-top-to-TOA
    layer (:10756-10760).  A column whose possible layer midpoint is already
    below the pinned coefficient floor needs no representable extra layer.
    """
    p_top = float(p_top)
    pressure_floor = float(pressure_floor)
    if (not np.isfinite(p_top) or not np.isfinite(pressure_floor)
            or p_top < 0.0 or pressure_floor <= 0.0):
        raise ValueError("radiation top pressures must be finite and nonnegative")
    if p_top <= pressure_floor:
        return 0, 0
    # Fortran NINT is nearest integer, with a positive half rounded upward.
    lw = int(np.floor(p_top / WRF_LW_UPPER_DELTA_P_PA + 0.5))
    sw = int(0.5 * p_top >= pressure_floor)
    return max(0, lw), sw


def _extend_above_model_profile(
        play, plev, tlay, tlev, qv, *, p_top: float, kind: str,
        pressure_floor: float = RRTMGP_TOA_PRESSURE_PA, xp=None,
        validate_top: bool = True) -> _RadiationColumnProfile:
    """Append WRF's clear upper atmosphere in bottom-to-top layout.

    The LW pressure/temperature construction transcribes
    ``module_ra_rrtmg_lw.F:12322-12393``: 400-Pa interfaces, a final TOA
    interface adapted to the RRTMGP coefficient floor, the 60-level WRF
    standard-atmosphere temperature interpolant shifted to meet the live
    model-top temperature, and layer temperatures averaged from interfaces.
    SW transcribes ``module_ra_rrtmg_sw.F:10910-10925``: one layer with
    ``play=.5*ptop`` and an isothermal top.  Both hold top-layer water vapor
    constant; well-mixed gases and pressure-interpolated ozone are filled by
    :meth:`RRTMGPRadiation._gas_vmr` after extension.
    """
    if kind not in ("lw", "sw"):
        raise ValueError("above-model profile kind must be 'lw' or 'sw'")
    if xp is None:
        import cupy as xp

    play = xp.ascontiguousarray(xp.asarray(play, dtype=DTYPE))
    plev = xp.ascontiguousarray(xp.asarray(plev, dtype=DTYPE))
    tlay = xp.ascontiguousarray(xp.asarray(tlay, dtype=DTYPE))
    tlev = xp.ascontiguousarray(xp.asarray(tlev, dtype=DTYPE))
    qv = xp.ascontiguousarray(xp.asarray(qv, dtype=DTYPE))
    if play.ndim != 2:
        raise ValueError("play must have shape (ncol,nlay)")
    ncol, model_nlay = play.shape
    expected = {
        "plev": (plev, (ncol, model_nlay + 1)),
        "tlay": (tlay, play.shape), "tlev": (tlev, plev.shape),
        "qv": (qv, play.shape),
    }
    for name, (value, shape) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")

    lw_upper, sw_upper = rrtmgp_above_model_layer_counts(
        p_top, pressure_floor=pressure_floor)
    upper_nlay = lw_upper if kind == "lw" else sw_upper
    if model_nlay + upper_nlay > MAX_RADIATION_LAYERS:
        raise ValueError(
            "RRTMGP CUDA RTE supports at most "
            f"{MAX_RADIATION_LAYERS} layers including the above-model "
            f"column; got {model_nlay + upper_nlay}")
    if validate_top and not bool(xp.allclose(
            plev[:, -1], DTYPE(p_top), rtol=DTYPE(0.0),
            atol=DTYPE(max(1.0e-3, abs(float(p_top)) * 2.0e-7)))):
        raise ValueError(
            "radiation workspace top pressure does not match the model-top "
            f"interface ({p_top:g} Pa)")
    if upper_nlay == 0:
        return _RadiationColumnProfile(
            play, plev, tlay, tlev, qv, model_nlay, 0)

    top_pressure = plev[:, -1:]
    top_temperature = tlev[:, -1:]
    top_qv = qv[:, -1:]
    if kind == "sw":
        upper_plev = xp.full(
            (ncol, 1), DTYPE(pressure_floor), dtype=DTYPE)
        upper_play = DTYPE(0.5) * top_pressure
        upper_tlev = top_temperature.copy()
        upper_tlay = top_temperature.copy()
    else:
        offsets = (WRF_LW_UPPER_DELTA_P_PA
                   * xp.arange(1, upper_nlay + 1, dtype=DTYPE))[None, :]
        upper_plev = top_pressure - offsets
        upper_plev[:, -1] = DTYPE(pressure_floor)
        all_upper_interfaces = xp.concatenate(
            (top_pressure, upper_plev), axis=1)
        upper_play = DTYPE(0.5) * (
            all_upper_interfaces[:, :-1] + all_upper_interfaces[:, 1:])

        order = np.argsort(_WRF_LW_PPROF_HPA)
        pprof = xp.asarray(_WRF_LW_PPROF_HPA[order], dtype=DTYPE)
        tprof = xp.asarray(_WRF_LW_TPROF_K[order], dtype=DTYPE)
        climo_top = xp.interp(
            top_pressure.ravel() * DTYPE(0.01), pprof, tprof).reshape(ncol, 1)
        climo_upper = xp.interp(
            upper_plev.ravel() * DTYPE(0.01), pprof, tprof).reshape(
                ncol, upper_nlay)
        upper_tlev = climo_upper + (top_temperature - climo_top)
        all_upper_tlev = xp.concatenate(
            (top_temperature, upper_tlev), axis=1)
        upper_tlay = DTYPE(0.5) * (
            all_upper_tlev[:, :-1] + all_upper_tlev[:, 1:])
    upper_qv = xp.broadcast_to(top_qv, (ncol, upper_nlay))
    return _RadiationColumnProfile(
        xp.ascontiguousarray(xp.concatenate((play, upper_play), axis=1)),
        xp.ascontiguousarray(xp.concatenate((plev, upper_plev), axis=1)),
        xp.ascontiguousarray(xp.concatenate((tlay, upper_tlay), axis=1)),
        xp.ascontiguousarray(xp.concatenate((tlev, upper_tlev), axis=1)),
        xp.ascontiguousarray(xp.concatenate((qv, upper_qv), axis=1)),
        model_nlay, upper_nlay)


def _model_flux_interfaces(flux, model_nlay: int, *, xp=None):
    """Discard upper-atmosphere heating levels while retaining model-top flux."""
    if xp is None:
        import cupy as xp
    flux = xp.asarray(flux, dtype=DTYPE)
    if flux.ndim != 2 or flux.shape[1] < int(model_nlay) + 1:
        raise ValueError("radiation flux column does not reach model top")
    return flux[:, :int(model_nlay) + 1]


def _append_clear_upper_layers(value, upper_nlay: int, *, xp=None):
    """Append zero-valued cloud/path layers to a model-layer field."""
    if xp is None:
        import cupy as xp
    value = xp.ascontiguousarray(xp.asarray(value, dtype=DTYPE))
    if value.ndim != 2:
        raise ValueError("clear-layer input must have shape (ncol,nlay)")
    upper_nlay = int(upper_nlay)
    if upper_nlay < 0:
        raise ValueError("upper_nlay must be nonnegative")
    if upper_nlay == 0:
        return value
    clear = xp.zeros((value.shape[0], upper_nlay), dtype=DTYPE)
    return xp.ascontiguousarray(xp.concatenate((value, clear), axis=1))


def _array(value, dtype):
    """Materialize a masked/NetCDF value as a C-contiguous plain array."""
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _packed_variable(variable, dimensions, dtype):
    """Pack a NetCDF variable in a kernel layout selected by dimension name."""
    source = tuple(variable.dimensions)
    target = tuple(dimensions)
    if len(source) != len(target) or set(source) != set(target):
        raise ValueError(
            f"{getattr(variable, 'name', 'variable')} dimensions {source} "
            f"do not match required dimensions {target}")
    permutation = tuple(source.index(name) for name in target)
    return _array(np.transpose(variable[:], permutation), dtype)


def _strings(variable) -> tuple[str, ...]:
    values = chartostring(variable[:])
    return tuple(str(value).strip().lower() for value in values.tolist())


def _make_flavors(key_species: np.ndarray,
                  band_lims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transcribe ``create_flavor``/``create_gpoint_flavor`` (0-based).

    Gas slot zero is dry air.  Upstream rewrites the special ``(0,0)`` pair
    to ``(2,2)`` because those coefficients are identically zero.
    """
    pairs: list[tuple[int, int]] = []
    for iband in range(key_species.shape[0]):
        for iatm in range(2):
            pair = tuple(int(x) for x in key_species[iband, iatm])
            if pair == (0, 0):
                pair = (2, 2)
            if pair not in pairs:
                pairs.append(pair)
    flavors = _array(pairs, np.int32)
    gpoint_flavor = np.empty((2, int(band_lims[-1, 1]) + 1), np.int32)
    lookup = {pair: i for i, pair in enumerate(pairs)}
    for iband, (start, end) in enumerate(band_lims):
        for iatm in range(2):
            pair = tuple(int(x) for x in key_species[iband, iatm])
            if pair == (0, 0):
                pair = (2, 2)
            gpoint_flavor[iatm, start:end + 1] = lookup[pair]
    return flavors, np.ascontiguousarray(gpoint_flavor)


def _minor_gas_indices(gas_names: tuple[str, ...], gas_minor,
                       identifier_minor, minor_identifiers) -> np.ndarray:
    identifier_map = {name: i for i, name in enumerate(identifier_minor)}
    gas_map = {name: i + 1 for i, name in enumerate(gas_names)}
    return _array([gas_map[gas_minor[identifier_map[name]]]
                   for name in minor_identifiers], np.int32)


def _scaling_gas_indices(gas_names: tuple[str, ...], names) -> np.ndarray:
    gas_map = {name: i + 1 for i, name in enumerate(gas_names)}
    return _array([gas_map.get(name, -1) for name in names], np.int32)


@dataclass
class DeviceTables:
    """Namespace holding cached device arrays plus scalar metadata."""

    _values: dict[str, object]

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class GasOpticsResult:
    """Gas extinction optical depth and optional SW scattering properties."""

    tau: object
    ssa: object | None = None
    g: object | None = None
    col_dry: object | None = None


@dataclass
class FluxResult:
    flux_up: object
    flux_dn: object
    flux_dir: object | None = None


@dataclass
class PlanckSourceResult:
    lay_source: object
    lev_source: object
    sfc_source: object


@dataclass
class _InterpolationMetadata:
    """Driver-owned pressure/temperature indices and FP32 fractions.

    Instances are created for one ``RRTMGPRadiation.__call__`` and are never
    accepted by the exported gas/source entry points.  Keeping the object
    private makes its table/input provenance and one-call lifetime explicit.
    """

    iatm: object
    jt: object
    jp: object
    ftemp: object
    fpress: object

    def __getitem__(self, key):
        return _InterpolationMetadata(
            self.iatm[key], self.jt[key], self.jp[key],
            self.ftemp[key], self.fpress[key])


@dataclass
class CloudOpticsResult:
    """Band-resolved cloud extinction and scattering properties."""

    tau: object
    ssa: object
    g: object


@dataclass
class HydrometeorPaths:
    """Condensate paths (g m-2) and RRTMGP particle sizes (microns)."""

    clwp: object
    ciwp: object
    reliq: object
    dgice: object


@dataclass(frozen=True)
class _RadiationColumnChunk:
    """One solver chunk's synthetic upper atmosphere and interpolation."""

    profile: _RadiationColumnProfile
    paths: HydrometeorPaths
    cldfra: object
    metadata: _InterpolationMetadata


def _prepare_above_model_chunk(
        *, tables, play, plev, tlay, tlev, qv, paths, cldfra, columns,
        p_top: float, kind: str,
        pressure_floor: float = RRTMGP_TOA_PRESSURE_PA, xp=None,
        validate_top: bool = True, validate: bool = True,
) -> _RadiationColumnChunk:
    """Build every above-model temporary for exactly ``columns``.

    The model-layer columns remain full-domain inputs, but the synthetic
    thermodynamic cap, clear cloud/path layers, and gas-table interpolation
    coordinates are deliberately materialized only for the active solver
    chunk.  In particular, an irregular final slice retains its true column
    count rather than allocating or exposing a padded ``column_chunk`` tail.
    """
    if xp is None:
        import cupy as xp

    profile = _extend_above_model_profile(
        play[columns], plev[columns], tlay[columns], tlev[columns],
        qv[columns], p_top=p_top, kind=kind,
        pressure_floor=pressure_floor, xp=xp, validate_top=validate_top)
    chunk_paths = HydrometeorPaths(*(
        _append_clear_upper_layers(
            value[columns], profile.upper_nlay, xp=xp)
        for value in (paths.clwp, paths.ciwp, paths.reliq, paths.dgice)))
    chunk_cldfra = _append_clear_upper_layers(
        cldfra[columns], profile.upper_nlay, xp=xp)
    metadata = _interpolation_metadata(
        tables, profile.play, profile.tlay, validate=validate)
    return _RadiationColumnChunk(
        profile=profile, paths=chunk_paths, cldfra=chunk_cldfra,
        metadata=metadata)


# Shared-workspace counterpart to SCRATCH_SLOT_LIFETIME_AUDIT.  Each value
# names the operation that completely writes the slot after every reuse and
# before its first consumer.  RTE "carried" entries are identical-offset views
# of values fully produced in the same chunk's immediately preceding optics
# phase; no value is carried across chunks or domains.
RRTMGP_WORKSPACE_LIFETIME_AUDIT = {
    "lw_optics": {
        "gas_tau": "rrtmgp_gas_optics kernel",
        "optics_tau": "rrtmgp_finalize_cloud_lw kernel",
        "vmr": "full zero fill plus complete active-gas assignment",
        "cld_tau": "rrtmgp_cloud_optics kernel",
        "cld_ssa": "rrtmgp_cloud_optics kernel",
        "cld_asy": "rrtmgp_cloud_optics kernel",
        "col_dry": "complete expression assignment",
        "mcica_mask": "rrtmgp_mcica_maxran kernel",
    },
    "lw_rte": {
        "gas_tau": "same-chunk lw_optics producer at identical offset",
        "optics_tau": "same-chunk lw_optics producer at identical offset",
        "vmr": "same-chunk lw_optics producer at identical offset",
        "cld_tau": "same-chunk lw_optics producer at identical offset",
        "cld_ssa": "same-chunk lw_optics producer at identical offset",
        "cld_asy": "same-chunk lw_optics producer at identical offset",
        "col_dry": "same-chunk lw_optics producer at identical offset",
        "lay_source": "rrtmgp_planck_sources kernel",
        "lev_source": "rrtmgp_planck_sources kernel",
        "sfc_source": "rrtmgp_planck_sources kernel",
        "emiss_gpt": "complete band-expansion assignment",
        "incident": "full zero fill",
        "flux_up": "rrtmgp_lw_noscat kernel",
        "flux_dn": "rrtmgp_lw_noscat kernel",
    },
    "sw_optics": {
        "gas_tau": "rrtmgp_gas_optics kernel",
        "gas_ssa": "rrtmgp_gas_optics kernel",
        "optics_tau": "rrtmgp_finalize_cloud_sw kernel",
        "optics_ssa": "rrtmgp_finalize_cloud_sw kernel",
        "optics_g": "rrtmgp_finalize_cloud_sw kernel",
        "vmr": "full zero fill plus complete active-gas assignment",
        "cld_tau": "rrtmgp_cloud_optics kernel",
        "cld_ssa": "rrtmgp_cloud_optics kernel",
        "cld_asy": "rrtmgp_cloud_optics kernel",
        "col_dry": "complete expression assignment",
        "mcica_mask": "rrtmgp_mcica_maxran kernel",
    },
    "sw_rte": {
        "gas_tau": "same-chunk sw_optics producer at identical offset",
        "gas_ssa": "same-chunk sw_optics producer at identical offset",
        "optics_tau": "same-chunk sw_optics producer at identical offset",
        "optics_ssa": "same-chunk sw_optics producer at identical offset",
        "optics_g": "same-chunk sw_optics producer at identical offset",
        "vmr": "same-chunk sw_optics producer at identical offset",
        "cld_tau": "same-chunk sw_optics producer at identical offset",
        "cld_ssa": "same-chunk sw_optics producer at identical offset",
        "cld_asy": "same-chunk sw_optics producer at identical offset",
        "col_dry": "same-chunk sw_optics producer at identical offset",
        "albedo_gpt": "complete surface broadcast assignment",
        "inc_gpt": "complete solar broadcast assignment",
        "mu0": "complete cosine broadcast assignment",
        "flux_up": "rrtmgp_sw_2stream kernel",
        "flux_dn": "rrtmgp_sw_2stream kernel",
        "flux_dir": "rrtmgp_sw_2stream kernel",
    },
}


_VALIDATION_MESSAGES = (
    (1 << 0, "play is non-finite or outside the gas-table pressure range"),
    (1 << 1, "plev is non-finite or negative"),
    (1 << 2, "tlay is non-finite or outside the gas-table temperature range"),
    (1 << 3, "tlev is non-finite or outside the LW temperature range"),
    (1 << 4, "tsfc is non-finite or outside the LW temperature range"),
    (1 << 5, "qv is non-finite or negative"),
    (1 << 6, "qc is non-finite or negative"),
    (1 << 7, "qr is non-finite or negative"),
    (1 << 8, "qi is non-finite or negative"),
    (1 << 9, "qs is non-finite or negative"),
    (1 << 10, "cldfra is non-finite or outside [0, 1]"),
    (1 << 11, "nc is non-finite or negative"),
    (1 << 12, "nr is non-finite or negative"),
    (1 << 13, "ni is non-finite or negative"),
    (1 << 14, "ns is non-finite or negative"),
    (1 << 15, "effc is non-finite or negative"),
    (1 << 16, "effr is non-finite or negative"),
    (1 << 17, "effi is non-finite or negative"),
    (1 << 18, "effs is non-finite or negative"),
    (1 << 19, "surface emissivity is non-finite or outside [0, 1]"),
    (1 << 20, "the bottom four layer pressures are not bottom-to-top"),
    (1 << 21, "pressure thickness or Exner is non-positive"),
    (1 << 22, "effc is outside the physical-plausibility band "
              "(microns contract; radii writer unit defect?)"),
    (1 << 23, "effi is outside the physical-plausibility band "
              "(microns contract; radii writer unit defect?)"),
    (1 << 24, "effs is outside the physical-plausibility band "
              "(microns contract; radii writer unit defect?)"),
)


@dataclass
class RFMIPResult:
    lw_up: object
    lw_dn: object
    sw_up: object
    sw_dn: object


_RFMIP_GAS_NAMES = {
    "co2": "carbon_dioxide", "n2o": "nitrous_oxide",
    "co": "carbon_monoxide", "ch4": "methane", "o2": "oxygen",
    "n2": "nitrogen", "ccl4": "carbon_tetrachloride",
    "cfc11": "cfc11", "cfc12": "cfc12", "cfc22": "hcfc22",
    "hfc143a": "hfc143a", "hfc125": "hfc125", "hfc23": "hfc23",
    "hfc32": "hfc32", "hfc134a": "hfc134a", "cf4": "cf4",
}


# NOAA Global Monitoring Laboratory, "Globally averaged marine surface
# annual mean CO2", dry-air mole fraction in ppm.  Pinned 2026-07-16 from
# https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt
# (file creation 2026-07-05; DOI https://doi.org/10.15138/9N0H-ZH07).
# The source begins in 1979.  Dates outside the pinned range hold the nearest
# published annual mean unless the case declares an override.  Runtime
# performs no network access.
_NOAA_GML_CO2_ANNUAL_PPM = {
    1979: 336.85, 1980: 338.91, 1981: 340.11, 1982: 340.85,
    1983: 342.53, 1984: 344.07, 1985: 345.54, 1986: 346.97,
    1987: 348.68, 1988: 351.16, 1989: 352.79, 1990: 354.06,
    1991: 355.40, 1992: 356.09, 1993: 356.84, 1994: 358.33,
    1995: 360.18, 1996: 361.93, 1997: 363.04, 1998: 365.70,
    1999: 367.80, 2000: 368.96, 2001: 370.57, 2002: 372.58,
    2003: 375.14, 2004: 376.95, 2005: 378.98, 2006: 381.15,
    2007: 382.90, 2008: 385.02, 2009: 386.50, 2010: 388.75,
    2011: 390.62, 2012: 392.65, 2013: 395.40, 2014: 397.34,
    2015: 399.65, 2016: 403.07, 2017: 405.22, 2018: 407.61,
    2019: 410.07, 2020: 412.44, 2021: 414.70, 2022: 417.08,
    2023: 419.35, 2024: 422.79, 2025: 425.64,
}


def trace_gases(valid_date: date | datetime,
                override: Mapping[str, float] | None = None
                ) -> dict[str, float]:
    """Select date-indexed well-mixed gas VMRs, then apply case overrides.

    NOAA's annual global CO2 mean is selected by calendar year, holding the
    earliest/latest published value outside the table range.  Overrides are
    mole fractions and win over the dated selection.
    """
    if not isinstance(valid_date, (date, datetime)):
        raise TypeError("trace-gas selection date must be a date or datetime")
    years = tuple(_NOAA_GML_CO2_ANNUAL_PPM)
    selected_year = max((year for year in years if year <= valid_date.year),
                        default=min(years))
    selected = {
        "co2": _NOAA_GML_CO2_ANNUAL_PPM[selected_year] * 1.0e-6,
    }
    if override is None:
        return selected
    if not isinstance(override, Mapping):
        raise TypeError("trace-gas override must be a mapping or None")
    unknown = sorted(set(override) - set(_RFMIP_GAS_NAMES))
    if unknown:
        raise ValueError(
            f"unknown trace gas(es) {unknown} in override; known well-mixed "
            f"gases: {sorted(_RFMIP_GAS_NAMES)}")
    for gas, raw_value in override.items():
        if isinstance(raw_value, bool):
            raise ValueError(
                f"trace-gas override[{gas!r}] = {raw_value!r} must be a "
                "finite mole fraction in (0, 1e-2)")
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 < value < 1.0e-2:
            raise ValueError(
                f"trace-gas override[{gas!r}] = {value!r} must be a finite "
                "mole fraction in (0, 1e-2)")
        selected[gas] = value
    return selected


@dataclass
class GasTables:
    kind: str
    gas_names: tuple[str, ...]
    gas_index: Mapping[str, int]
    nband: int
    ngpt: int
    ntemp: int
    npres: int
    neta: int
    press_ref: np.ndarray
    temp_ref: np.ndarray
    press_ref_trop: float
    vmr_ref: np.ndarray
    band_lims_gpt: np.ndarray
    gpoint_bands: np.ndarray
    flavor: np.ndarray
    gpoint_flavor: np.ndarray
    kmajor: np.ndarray
    kminor_lower: np.ndarray
    kminor_upper: np.ndarray
    minor_limits_gpt_lower: np.ndarray
    minor_limits_gpt_upper: np.ndarray
    minor_scales_with_density_lower: np.ndarray
    minor_scales_with_density_upper: np.ndarray
    scale_by_complement_lower: np.ndarray
    scale_by_complement_upper: np.ndarray
    idx_minor_lower: np.ndarray
    idx_minor_upper: np.ndarray
    idx_minor_scaling_lower: np.ndarray
    idx_minor_scaling_upper: np.ndarray
    kminor_start_lower: np.ndarray
    kminor_start_upper: np.ndarray
    rayleigh: np.ndarray | None = None
    planck_fraction: np.ndarray | None = None
    temperature_planck: np.ndarray | None = None
    totplnk: np.ndarray | None = None
    solar_source: np.ndarray | None = None
    tsi_default: float | None = None
    optimal_angle_fit: np.ndarray | None = None
    _device: DeviceTables | None = field(default=None, init=False, repr=False)

    @property
    def ngas(self) -> int:
        return len(self.gas_names)

    @property
    def nflav(self) -> int:
        return self.flavor.shape[0]

    def packed_arrays(self) -> dict[str, np.ndarray]:
        return {name: value for name, value in vars(self).items()
                if isinstance(value, np.ndarray)}

    def to_device(self) -> DeviceTables:
        if self._device is None:
            import cupy as cp
            values: dict[str, object] = {}
            for name, value in vars(self).items():
                if isinstance(value, np.ndarray):
                    dtype = (cp.float32 if np.issubdtype(value.dtype,
                                                         np.floating)
                             else cp.int32 if np.issubdtype(value.dtype,
                                                            np.integer)
                             else cp.bool_)
                    values[name] = cp.ascontiguousarray(cp.asarray(value,
                                                                   dtype=dtype))
                elif not name.startswith("_"):
                    values[name] = value
            self._device = DeviceTables(values)
        return self._device


@dataclass
class CloudTables:
    kind: str
    nband: int
    nsize_liq: int
    nsize_ice: int
    nrghice: int
    radliq_lwr: float
    radliq_upr: float
    diamice_lwr: float
    diamice_upr: float
    extliq: np.ndarray
    ssaliq: np.ndarray
    asyliq: np.ndarray
    extice: np.ndarray
    ssaice: np.ndarray
    asyice: np.ndarray
    _device: DeviceTables | None = field(default=None, init=False, repr=False)

    @property
    def liq_step_size(self) -> float:
        return (self.radliq_upr - self.radliq_lwr) / (self.nsize_liq - 1)

    @property
    def ice_step_size(self) -> float:
        return (self.diamice_upr - self.diamice_lwr) / (self.nsize_ice - 1)

    def to_device(self) -> DeviceTables:
        if self._device is None:
            import cupy as cp
            values = {name: (cp.ascontiguousarray(cp.asarray(value,
                                                               dtype=cp.float32))
                             if isinstance(value, np.ndarray) else value)
                      for name, value in vars(self).items()
                      if not name.startswith("_")}
            self._device = DeviceTables(values)
        return self._device


@lru_cache(maxsize=2)
def load_gas_tables(kind: str) -> GasTables:
    """Load and pack the v1.9 LW or SW gas k-distribution in float64."""
    kind = kind.lower()
    if kind not in ("lw", "sw"):
        raise ValueError("kind must be 'lw' or 'sw'")
    filename = ("rrtmgp-gas-lw-g256.nc" if kind == "lw"
                else "rrtmgp-gas-sw-g224.nc")
    with Dataset(DATA_DIR / filename, "r") as nc:
        gas_names = _strings(nc["gas_names"])
        gas_minor = _strings(nc["gas_minor"])
        identifier_minor = _strings(nc["identifier_minor"])
        band_lims = _array(nc["bnd_limits_gpt"][:] - 1, np.int32)
        key_species = _array(nc["key_species"][:], np.int32)
        flavor, gpoint_flavor = _make_flavors(key_species, band_lims)
        gpoint_bands = np.empty(len(nc.dimensions["gpt"]), np.int32)
        for iband, (start, end) in enumerate(band_lims):
            gpoint_bands[start:end + 1] = iband

        lower_names = _strings(nc["minor_gases_lower"])
        upper_names = _strings(nc["minor_gases_upper"])
        scaling_lower = _strings(nc["scaling_gas_lower"])
        scaling_upper = _strings(nc["scaling_gas_upper"])
        kwargs = dict(
            kind=kind,
            gas_names=gas_names,
            gas_index={name: i + 1 for i, name in enumerate(gas_names)},
            nband=len(nc.dimensions["bnd"]),
            ngpt=len(nc.dimensions["gpt"]),
            ntemp=len(nc.dimensions["temperature"]),
            npres=len(nc.dimensions["pressure"]),
            neta=len(nc.dimensions["mixing_fraction"]),
            press_ref=_array(nc["press_ref"][:], np.float64),
            temp_ref=_array(nc["temp_ref"][:], np.float64),
            press_ref_trop=float(nc["press_ref_trop"].getValue()),
            # NetCDF dimension order is (temperature, absorber, atmosphere).
            # The numerical kernels use (atmosphere, absorber, temperature).
            vmr_ref=_array(np.transpose(nc["vmr_ref"][:], (2, 1, 0)),
                           np.float64),
            band_lims_gpt=band_lims,
            gpoint_bands=np.ascontiguousarray(gpoint_bands),
            flavor=flavor,
            gpoint_flavor=gpoint_flavor,
            # Kernel layouts are selected from declared NetCDF dimension
            # names, never from an assumed positional file order.
            kmajor=_packed_variable(
                nc["kmajor"],
                ("temperature", "mixing_fraction", "pressure_interp", "gpt"),
                np.float64),
            kminor_lower=_packed_variable(
                nc["kminor_lower"],
                ("temperature", "mixing_fraction", "contributors_lower"),
                np.float64),
            kminor_upper=_packed_variable(
                nc["kminor_upper"],
                ("temperature", "mixing_fraction", "contributors_upper"),
                np.float64),
            minor_limits_gpt_lower=_array(
                nc["minor_limits_gpt_lower"][:] - 1, np.int32),
            minor_limits_gpt_upper=_array(
                nc["minor_limits_gpt_upper"][:] - 1, np.int32),
            minor_scales_with_density_lower=_array(
                nc["minor_scales_with_density_lower"][:], bool),
            minor_scales_with_density_upper=_array(
                nc["minor_scales_with_density_upper"][:], bool),
            scale_by_complement_lower=_array(
                nc["scale_by_complement_lower"][:], bool),
            scale_by_complement_upper=_array(
                nc["scale_by_complement_upper"][:], bool),
            idx_minor_lower=_minor_gas_indices(
                gas_names, gas_minor, identifier_minor, lower_names),
            idx_minor_upper=_minor_gas_indices(
                gas_names, gas_minor, identifier_minor, upper_names),
            idx_minor_scaling_lower=_scaling_gas_indices(
                gas_names, scaling_lower),
            idx_minor_scaling_upper=_scaling_gas_indices(
                gas_names, scaling_upper),
            kminor_start_lower=_array(
                nc["kminor_start_lower"][:] - 1, np.int32),
            kminor_start_upper=_array(
                nc["kminor_start_upper"][:] - 1, np.int32),
        )
        if kind == "lw":
            kwargs.update(
                planck_fraction=_packed_variable(
                    nc["plank_fraction"],
                    ("temperature", "mixing_fraction", "pressure_interp",
                     "gpt"), np.float64),
                temperature_planck=_array(nc["temperature_Planck"][:],
                                          np.float64),
                totplnk=_array(np.transpose(nc["totplnk"][:]), np.float64),
                optimal_angle_fit=_array(nc["optimal_angle_fit"][:],
                                         np.float64),
            )
        else:
            rayleigh = np.stack((nc["rayl_lower"][:],
                                 nc["rayl_upper"][:]), axis=0)
            quiet = _array(nc["solar_source_quiet"][:], np.float64)
            facular = _array(nc["solar_source_facular"][:], np.float64)
            sunspot = _array(nc["solar_source_sunspot"][:], np.float64)
            mg = float(nc["mg_default"].getValue())
            sb = float(nc["sb_default"].getValue())
            solar = quiet + (mg - 0.1495954) * facular \
                + (sb - 0.00066696) * sunspot
            kwargs.update(
                rayleigh=_array(rayleigh, np.float64),
                solar_source=_array(solar, np.float64),
                tsi_default=float(nc["tsi_default"].getValue()),
            )
    return GasTables(**kwargs)


@lru_cache(maxsize=2)
def load_cloud_tables(kind: str) -> CloudTables:
    """Load band-resolved v1.9 liquid/ice cloud-optics tables."""
    kind = kind.lower()
    if kind not in ("lw", "sw"):
        raise ValueError("kind must be 'lw' or 'sw'")
    with Dataset(DATA_DIR / f"rrtmgp-clouds-{kind}-bnd.nc", "r") as nc:
        return CloudTables(
            kind=kind,
            nband=len(nc.dimensions["nband"]),
            nsize_liq=len(nc.dimensions["nsize_liq"]),
            nsize_ice=len(nc.dimensions["nsize_ice"]),
            nrghice=len(nc.dimensions["nrghice"]),
            radliq_lwr=float(nc["radliq_lwr"].getValue()),
            radliq_upr=float(nc["radliq_upr"].getValue()),
            diamice_lwr=float(nc["diamice_lwr"].getValue()),
            diamice_upr=float(nc["diamice_upr"].getValue()),
            extliq=_array(np.transpose(nc["extliq"][:]), np.float64),
            ssaliq=_array(np.transpose(nc["ssaliq"][:]), np.float64),
            asyliq=_array(np.transpose(nc["asyliq"][:]), np.float64),
            extice=_array(np.transpose(nc["extice"][:], (2, 1, 0)),
                          np.float64),
            ssaice=_array(np.transpose(nc["ssaice"][:], (2, 1, 0)),
                          np.float64),
            asyice=_array(np.transpose(nc["asyice"][:], (2, 1, 0)),
                          np.float64),
        )


def hydrometeor_paths(plev, qc, qr=None, qi=None, qs=None, *,
                      microphysics="kessler", play=None, tlay=None,
                      nc=None, nr=None, ni=None, ns=None,
                      effc=None, effr=None, effi=None,
                      effs=None, cldfra=None,
                      snow_treatment=SNOW_TREATMENT_FULL_MASS,
                      validate=True) -> HydrometeorPaths:
    """Convert hydrometeor mixing ratios to cloud-optics inputs on device.

    Mixing ratios and Morrison number concentrations are per kg dry air.
    Per WRF the radiation liquid path is cloud water only and the ice path
    is cloud ice plus snow; rain never feeds the paths
    (module_ra_rrtmg_sw.F:11029-11034, module_ra_rrtmg_lw.F:12488-12493:
    ``gliqwp = qc1d(k) * pdel*100/gravmks*1000``).  When ``cldfra`` is
    given the grid-box paths become in-cloud paths through WRF's
    ``max(0.01, cldfrac)`` division (same lines).
    Kessler uses a 10 micron liquid radius and 50 micron ice diameter.
    WSM6, Thompson, and NSSL consume their scheme-native cloud/ice/snow
    effective radii (MICRONS, the state contract) and merge ice plus snow
    into the single RRTMGP ice species.  ``"thompson"`` serves BOTH Thompson
    packages: mp_physics=8 (Registry.EM_COMMON:3024) and the aerosol-aware
    mp_physics=28 (:3036) declare the same ``moist`` inventory and the same
    ``re_cloud/re_ice/re_snow`` state, and module_physics_init.F:1005-1006
    lists them together in one ``has_req*`` disjunction, so their radiative
    coupling is identical -- see :data:`_MP_CLOUD_OPTICS_SCHEME`.
    ``snow_treatment`` selects how snow joins that path:
    ``full-snow-mass-into-ice`` keeps the adapter's original
    full-mass merge; ``wrf-rrtmg-130um-snow-discount`` reproduces WRF
    v4.6.1's option-4 explicit-radius coupling -- ice path from cloud ice
    only (module_ra_rrtmg_lw.F:12500-12505, _sw.F:11040-11045), snow mass
    multiplied by ``MIN(0.99, (130/re_s)^2)`` with ``re_s`` floored at 10
    and capped at 130 microns (_lw.F:12242,12515-12532, _sw.F:10824,
    11055-11067; the FP32 discount expression is bitwise-pinned by
    tests/data/wrf_rrtmg_snow_discount_fixture.csv).  WRF's dead
    ``gicewp`` accumulation of the 1% remainder (_lw.F:12518, _sw.F:11058
    -- computed but never stored to ``cicewp``) is intentionally not
    reproduced.  The mass-weighted single-species diameter remains a
    documented adapter divergence from WRF's separate Fu snow species.
    ``snow_treatment`` is validated for every scheme but only alters the
    explicit-snow-radius (WSM6/Thompson/NSSL) coupling: WRF discounts snow
    only when the scheme supplies re_snow (iceflg=5); Morrison and Kessler
    keep WRF's merged path (_lw.F:12488-12493).
    Morrison liquid size is the cloud-droplet gamma radius alone -- rain
    carries no radiative mass, so it contributes no radius either
    (``effr`` is accepted for interface parity with Morrison's
    diagnostics and ignored); ice/snow combine by number before clipping
    to the shipped RRTMGP table domains.
    """
    import cupy as cp

    if snow_treatment not in SNOW_TREATMENTS:
        raise ValueError(
            f"snow_treatment must be one of {SNOW_TREATMENTS}, got "
            f"{snow_treatment!r}")

    plev = cp.ascontiguousarray(cp.asarray(plev, dtype=DTYPE))
    qc = cp.ascontiguousarray(cp.asarray(qc, dtype=DTYPE))
    if qc.ndim != 2 or plev.shape != (qc.shape[0], qc.shape[1] + 1):
        raise ValueError("plev/qc must have shapes (ncol,nlay+1)/(ncol,nlay)")

    def field(value, name):
        if value is None:
            return cp.zeros_like(qc)
        return _device_profile(value, qc.shape, name)

    qr, qi, qs = field(qr, "qr"), field(qi, "qi"), field(qs, "qs")
    if validate:
        if bool(cp.any(~cp.isfinite(plev))):
            raise ValueError("hydrometeor pressure inputs must be finite")
        _require_finite_nonnegative(qc=qc, qr=qr, qi=qi, qs=qs)
    mass_path = (cp.abs(cp.diff(plev, axis=1))
                 * DTYPE(1000.0 / 9.80665))
    clwp = qc * mass_path
    ciwp = (qi + qs) * mass_path
    if cldfra is not None:
        cldfra = _device_profile(cldfra, qc.shape, "cldfra")
        if validate and (bool(cp.any(~cp.isfinite(cldfra)))
                         or bool(cp.any(cldfra < 0.0))
                         or bool(cp.any(cldfra > 1.0))):
            raise ValueError("cldfra must be finite and within [0, 1]")
        incloud = cp.maximum(DTYPE(0.01), cldfra)
        clwp = clwp / incloud
        ciwp = ciwp / incloud
    clwp = cp.ascontiguousarray(clwp)
    ciwp = cp.ascontiguousarray(ciwp)
    scheme = str(microphysics).lower()
    if scheme == "kessler":
        return HydrometeorPaths(
            clwp, ciwp, cp.full_like(qc, DTYPE(10.0)),
            cp.full_like(qc, DTYPE(50.0)))
    if scheme in ("wsm6", "thompson", "nssl"):
        if any(x is None for x in (effc, effi, effs)):
            raise ValueError(
                f"{scheme} radii require effc, effi, and effs")
        re_c, re_i, re_s = (field(value, name) for value, name in
                            ((effc, "effc"), (effi, "effi"),
                             (effs, "effs")))
        if validate:
            _require_finite_nonnegative(effc=re_c, effi=re_i, effs=re_s)
            _require_plausible_radii_um(effc=re_c, effi=re_i, effs=re_s)
        tiny = DTYPE(1.0e-20)
        if snow_treatment == SNOW_TREATMENT_WRF_DISCOUNT:
            # WRF v4.6.1 option-4 explicit-snow-radius coupling, FP32 in
            # WRF's operation order (fixture-pinned, max_ulp 0):
            #   resnow = MAX(10., re_s)            (_lw.F:12242, _sw.F:10824;
            #                                       state is already microns)
            #   factor = 0.99; if resnow > 130:
            #       factor = MIN(0.99, (130/resnow)*(130/resnow))
            #       resnow = 130                   (_lw.F:12515-12528)
            #   snow path mass = qs * factor       (_lw.F:12529)
            # and the ice path is cloud ice only (_lw.F:12500-12505).
            re_s0 = cp.maximum(DTYPE(10.0), re_s)
            quotient = DTYPE(130.0) / re_s0
            factor = cp.where(
                re_s0 > DTYPE(130.0),
                cp.minimum(DTYPE(0.99), quotient * quotient),
                DTYPE(0.99))
            re_s_eff = cp.minimum(re_s0, DTYPE(130.0))
            qs_eff = qs * factor
            ciwp = (qi + qs_eff) * mass_path
            if cldfra is not None:
                ciwp = ciwp / incloud
            ciwp = cp.ascontiguousarray(ciwp)
        else:
            qs_eff = qs
            re_s_eff = re_s
        frozen = qi + qs_eff
        reice = cp.where(
            frozen > tiny,
            (qi * re_i + qs_eff * re_s_eff) / cp.maximum(frozen, tiny),
            DTYPE(25.0))
        reliq = cp.where(qc > tiny, re_c, DTYPE(10.0))
        return HydrometeorPaths(
            clwp, ciwp,
            cp.ascontiguousarray(cp.clip(reliq, DTYPE(2.5), DTYPE(21.5))),
            cp.ascontiguousarray(cp.clip(DTYPE(2.0) * reice,
                                         DTYPE(10.0), DTYPE(180.0))))
    if scheme != "morrison":
        raise ValueError(
            "microphysics must be 'kessler', 'wsm6', 'thompson', 'nssl', "
            "or 'morrison'")
    if play is None or tlay is None or any(x is None for x in (nc, nr, ni, ns)):
        raise ValueError("Morrison radii require play, tlay, nc, nr, ni, ns")
    play = _device_profile(play, qc.shape, "play")
    tlay = _device_profile(tlay, qc.shape, "tlay")
    ncp, nrp = field(nc, "nc"), field(nr, "nr")
    nip, nsp = field(ni, "ni"), field(ns, "ns")
    if validate and (bool(cp.any(~cp.isfinite(play)))
                     or bool(cp.any(~cp.isfinite(tlay)))):
        raise ValueError("Morrison thermodynamic inputs must be finite")
    if validate:
        _require_finite_nonnegative(nc=ncp, nr=nrp, ni=nip, ns=nsp)
    tiny = DTYPE(1.0e-20)

    rho_air = play / (DTYPE(287.15) * tlay)
    shape = DTYPE(0.0005714) * (ncp / DTYPE(1.0e6) * rho_air) \
        + DTYPE(0.2714)
    pgam = cp.clip(DTYPE(1.0) / (shape * shape) - DTYPE(1.0),
                   DTYPE(2.0), DTYPE(10.0))
    gamma_ratio = (pgam + DTYPE(1.0)) * (pgam + DTYPE(2.0)) \
        * (pgam + DTYPE(3.0))
    lam_c = cp.power(DTYPE(np.pi * 997.0 / 6.0) * ncp * gamma_ratio
                     / cp.maximum(qc, tiny), DTYPE(1.0 / 3.0))
    re_c = (pgam + DTYPE(3.0)) / cp.maximum(DTYPE(2.0) * lam_c, tiny) \
        * DTYPE(1.0e6)

    def exponential_radius(qmass, number, density):
        lam = cp.power(DTYPE(np.pi * density) * number
                       / cp.maximum(qmass, tiny), DTYPE(1.0 / 3.0))
        return DTYPE(1.5e6) / cp.maximum(lam, tiny)

    supplied_effective = (effc, effr, effi, effs)
    if any(value is not None for value in supplied_effective):
        if any(value is None for value in supplied_effective):
            raise ValueError("Morrison effective radii require effc/effr/"
                             "effi/effs together")
        re_c, re_r, re_i, re_s = (
            field(value, name) for value, name in zip(
                supplied_effective, ("effc", "effr", "effi", "effs")))
        if validate:
            _require_finite_nonnegative(
                effc=re_c, effr=re_r, effi=re_i, effs=re_s)
            # effr is interface parity only (ignored input, not gated).
            _require_plausible_radii_um(effc=re_c, effi=re_i, effs=re_s)
    else:
        re_i = exponential_radius(qi, nip, 500.0)
        re_s = exponential_radius(qs, nsp, 100.0)
    wc = cp.where((qc > 0) & (ncp > 0), ncp, DTYPE(0.0))
    wi = cp.where((qi > 0) & (nip > 0), nip, DTYPE(0.0))
    ws = cp.where((qs > 0) & (nsp > 0), nsp, DTYPE(0.0))
    reliq = cp.where(wc > 0, re_c, DTYPE(10.0))
    reice = cp.where(wi + ws > 0, (wi * re_i + ws * re_s)
                     / cp.maximum(wi + ws, tiny), DTYPE(25.0))
    return HydrometeorPaths(
        clwp, ciwp,
        cp.ascontiguousarray(cp.clip(reliq, DTYPE(2.5), DTYPE(21.5))),
        cp.ascontiguousarray(cp.clip(DTYPE(2.0) * reice,
                                     DTYPE(10.0), DTYPE(180.0))))


def cal_cldfra1(qv, qc, qi, qs, tlay, play, *, f_qc=True, f_qi=True,
                f_qs=True):
    """WRF icloud=1 Xu-Randall cloud fraction on device (FP32).

    Exact transcription of WRF v4.6.1 ``module_radiation_driver.F``
    ``cal_cldfra1`` (lines 3761-3986), the routine the radiation driver
    calls for the Registry default ``icloud=1`` (driver lines 1320-1332,
    Registry.EM_COMMON:2498).  Saturation follows Murray (1966) with the
    driver's constants (lines 3806-3816, 3861-3865); the liquid/ice
    saturation blend uses the condensate ice weight (line 3945) and the
    fraction is Xu and Randall (1996) with ALPHA0=100, GAMMA=0.49,
    QCLDMIN=1e-12, PEXP=0.25, RHGRID=1.0 plus the -6.9 ARG clamp and the
    0.01 truncation (lines 3950-3979).

    Moisture-set dispatch mirrors the driver flags: ``f_qc and f_qi and
    f_qs`` is the Morrison-class branch (lines 3870-3877, QCLD=QI+QC+QS,
    weight=(QI+QS)/QCLD); ``f_qc`` alone is the Kessler-class branch
    (lines 3891-3899, QCLD=QC, 273.15 K phase threshold).  Rain never
    enters QCLD (lines 3904-3916).
    """
    import cupy as cp

    qv = cp.ascontiguousarray(cp.asarray(qv, dtype=DTYPE))
    if qv.ndim != 2:
        raise ValueError("qv must have shape (ncol,nlay)")
    qc = _device_profile(qc, qv.shape, "qc")
    qi = _device_profile(qi, qv.shape, "qi")
    qs = _device_profile(qs, qv.shape, "qs")
    tlay = _device_profile(tlay, qv.shape, "tlay")
    play = _device_profile(play, qv.shape, "play")
    if not f_qc or f_qi != f_qs:
        raise NotImplementedError(
            "cal_cldfra1 port supports the qc-only (Kessler) and qc+qi+qs "
            "(Morrison) WRF moisture sets")
    qcldmin = DTYPE(1.0e-12)
    svpt0 = DTYPE(273.15)
    tc = tlay - svpt0
    esw = DTYPE(1000.0) * DTYPE(0.61078) * cp.exp(
        DTYPE(17.2693882) * tc / (tlay - DTYPE(35.86)))
    esi = DTYPE(1000.0) * DTYPE(0.61078) * cp.exp(
        DTYPE(21.8745584) * tc / (tlay - DTYPE(7.66)))
    ep2 = DTYPE(287.0) / DTYPE(461.6)
    qvsw = ep2 * esw / (play - esw)
    qvsi = ep2 * esi / (play - esi)
    if f_qi:
        qcld = qi + qc + qs
        weight = cp.where(qcld < qcldmin, DTYPE(0.0),
                          (qi + qs) / cp.maximum(qcld, qcldmin))
    else:
        qcld = qc
        weight = cp.where(qcld < qcldmin, DTYPE(0.0),
                          cp.where(tlay > svpt0, DTYPE(0.0), DTYPE(1.0)))
    qvs_weight = (DTYPE(1.0) - weight) * qvsw + weight * qvsi
    rhum = qv / qvs_weight
    subsat = cp.maximum(DTYPE(1.0e-10), qvs_weight - qv)
    arg = cp.maximum(DTYPE(-6.9),
                     DTYPE(-100.0) * qcld / cp.power(subsat, DTYPE(0.49)))
    fraction = cp.power(cp.maximum(DTYPE(1.0e-10), rhum), DTYPE(0.25)) \
        * (DTYPE(1.0) - cp.exp(arg))
    fraction = cp.where(fraction < DTYPE(0.01), DTYPE(0.0), fraction)
    return cp.ascontiguousarray(
        cp.where(qcld < qcldmin, DTYPE(0.0),
                 cp.where(rhum >= DTYPE(1.0), DTYPE(1.0), fraction)))


# WRF drives the LW/SW stochastic generators with distinct seed advances
# (module_ra_rrtmg_sw.F:11220-11222, module_ra_rrtmg_lw.F:12687-12689).
MCICA_PERMUTESEED_SW = 1
MCICA_PERMUTESEED_LW = 150


def mcica_cloud_masks(play, cldfra, ngpt, permuteseed, *, validate=True):
    return _mcica_cloud_masks(
        play, cldfra, ngpt, permuteseed, validate=validate, out=None)


def _mcica_cloud_masks(play, cldfra, ngpt, permuteseed, *, validate, out):
    """WRF RRTMG McICA maximum-random subcolumn cloud masks on device.

    Transcribes the kissvec generator, pmid-fraction seeding, cldmin
    floor, icld=2 maximum-random overlap walk (WRF Registry default
    ``cldovrlp=2``, Registry.EM_COMMON:2499), and the ``CDF >= 1-cldf``
    subcolumn decision of WRF v4.6.1 ``module_ra_rrtmg_sw.F``
    (lines 1692-1744, 1778-1813, 1941-1977, 2008-2040) with one
    subcolumn per g-point (line 1476).  Returns a boolean
    ``(ncol,nlay,ngpt)`` mask; ``play`` must be bottom-to-top in Pa,
    matching the generator's seed requirement.
    """
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2 or play.shape[1] < 4:
        raise ValueError("play must have shape (ncol,nlay) with nlay >= 4")
    cldfra = _device_profile(cldfra, play.shape, "cldfra")
    if validate and bool(cp.any(play[:, 0] < play[:, 1])):
        # module_ra_rrtmg_sw.F:1734-1736 stops unless pmid is supplied
        # bottom-to-top.
        raise ValueError(
            "kissvec seeding requires pmid from the bottom four layers")
    ncol, nlay = play.shape
    mask = _workspace_output(
        out, (ncol, nlay, int(ngpt)), "mcica_mask", dtype=cp.bool_)
    threads = 64
    get_kernel("rrtmgp_mcica", "rrtmgp_mcica_maxran")(
        ((ncol + threads - 1) // threads,), (threads,),
        (play, cldfra, mask, np.int32(ncol), np.int32(nlay),
         np.int32(ngpt), np.int32(permuteseed)))
    return mask


def cloud_optics(tables: CloudTables, clwp, ciwp, reliq,
                 dgice) -> CloudOpticsResult:
    return _cloud_optics(tables, clwp, ciwp, reliq, dgice, out=None)


def _cloud_optics(tables: CloudTables, clwp, ciwp, reliq,
                  dgice, *, out) -> CloudOpticsResult:
    """Interpolate v1.9 cloud tables into band-resolved FP32 optics.

    Non-positive water paths are reference-defined clear-sky masks in
    ``mo_cloud_optics_rrtmgp.F90:332-340``; the reference kernel writes zero
    properties for them in ``mo_cloud_optics_rrtmgp_kernels.F90:45-60``.
    """
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    clwp = cp.ascontiguousarray(cp.asarray(clwp, dtype=DTYPE))
    if clwp.ndim != 2:
        raise ValueError("clwp must have shape (ncol,nlay)")
    ciwp = _device_profile(ciwp, clwp.shape, "ciwp")
    reliq = _device_profile(reliq, clwp.shape, "reliq")
    dgice = _device_profile(dgice, clwp.shape, "dgice")
    d = tables.to_device()
    shape = (*clwp.shape, tables.nband)
    if out is None:
        tau, ssa, asym = (cp.empty(shape, dtype=DTYPE) for _ in range(3))
    else:
        tau, ssa, asym = (
            _workspace_output(value, shape, name)
            for value, name in zip(out, ("cld_tau", "cld_ssa", "cld_asy")))
    n = clwp.size * tables.nband
    threads = 256
    get_kernel("rrtmgp_cloud", "rrtmgp_cloud_optics")(
        ((n + threads - 1) // threads,), (threads,),
        (clwp, ciwp, reliq, dgice, d.extliq, d.ssaliq, d.asyliq,
         d.extice, d.ssaice, d.asyice, tau, ssa, asym,
         np.int32(clwp.size), np.int32(tables.nband),
         np.int32(tables.nsize_liq), np.int32(tables.nsize_ice),
         np.int32(tables.nrghice), DTYPE(tables.radliq_lwr),
         DTYPE(tables.liq_step_size), DTYPE(tables.diamice_lwr),
         DTYPE(tables.ice_step_size)))
    return CloudOpticsResult(tau, ssa, asym)


def add_cloud_optics(tables: GasTables, gas: GasOpticsResult,
                     cloud: CloudOpticsResult,
                     cloud_mask=None) -> GasOpticsResult:
    """Add band cloud properties to g-point gas properties on device.

    ``cloud_mask`` is an optional boolean ``(ncol,nlay,ngpt)`` McICA
    subcolumn mask: cloud properties are applied only to cloudy
    subcolumns, mirroring how WRF's RRTMG solvers consume the stochastic
    ``cldfmc``/``taucmc`` arrays per g-point (module_ra_rrtmg_sw.F:
    1951-1977, module_ra_rrtmg_lw.F:3288).  Without a mask every layer
    with condensate is treated as overcast.
    """
    import cupy as cp

    tau_gas = cp.ascontiguousarray(cp.asarray(gas.tau, dtype=DTYPE))
    if tau_gas.ndim != 3 or tau_gas.shape[2] != tables.ngpt:
        raise ValueError("gas optics do not match the gas table")
    band_shape = (*tau_gas.shape[:2], tables.nband)
    tau_cloud = _device_profile(cloud.tau, band_shape, "cloud.tau")
    ssa_cloud = _device_profile(cloud.ssa, band_shape, "cloud.ssa")
    g_cloud = _device_profile(cloud.g, band_shape, "cloud.g")
    bands = tables.to_device().gpoint_bands
    tc, wc, gc = (x[:, :, bands]
                  for x in (tau_cloud, ssa_cloud, g_cloud))
    if cloud_mask is not None:
        mask = cp.asarray(cloud_mask)
        if mask.shape != tau_gas.shape:
            raise ValueError(
                f"cloud_mask must have shape {tau_gas.shape}, "
                f"got {mask.shape}")
        tc = tc * mask
    if tables.kind == "lw":
        return GasOpticsResult(
            cp.ascontiguousarray(tau_gas + tc * (DTYPE(1.0) - wc)),
            col_dry=gas.col_dry)
    ssa_gas = _device_profile(gas.ssa, tau_gas.shape, "gas.ssa")
    g_gas = _device_profile(gas.g, tau_gas.shape, "gas.g")
    total_tau = tau_gas + tc
    scatter = tau_gas * ssa_gas + tc * wc
    floor = DTYPE(3.0 * np.finfo(np.float32).tiny)
    total_ssa = scatter / cp.maximum(floor, total_tau)
    total_g = (tau_gas * ssa_gas * g_gas + tc * wc * gc) \
        / cp.maximum(floor, scatter)
    return GasOpticsResult(cp.ascontiguousarray(total_tau),
                           cp.ascontiguousarray(total_ssa),
                           cp.ascontiguousarray(total_g), gas.col_dry)


def _finalize_cloud_optics(tables: GasTables, gas: GasOpticsResult,
                           cloud: CloudOpticsResult,
                           cloud_mask=None, *, out=None) -> GasOpticsResult:
    """Expand/add cloud optics and, for SW, delta-scale in one kernel."""
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    tau_gas = cp.ascontiguousarray(cp.asarray(gas.tau, dtype=DTYPE))
    if tau_gas.ndim != 3 or tau_gas.shape[2] != tables.ngpt:
        raise ValueError("gas optics do not match the gas table")
    band_shape = (*tau_gas.shape[:2], tables.nband)
    tau_cloud = _device_profile(cloud.tau, band_shape, "cloud.tau")
    ssa_cloud = _device_profile(cloud.ssa, band_shape, "cloud.ssa")
    g_cloud = _device_profile(cloud.g, band_shape, "cloud.g")
    if cloud_mask is None:
        mask = tau_gas  # Valid dummy pointer; the kernels do not read it.
        have_mask = False
    else:
        mask = cp.ascontiguousarray(cp.asarray(cloud_mask, dtype=cp.bool_))
        if mask.shape != tau_gas.shape:
            raise ValueError(
                f"cloud_mask must have shape {tau_gas.shape}, "
                f"got {mask.shape}")
        have_mask = True

    n = tau_gas.size
    threads = 256
    launch = ((n + threads - 1) // threads,), (threads,)
    bands = tables.to_device().gpoint_bands
    if tables.kind == "lw":
        tau = (cp.empty_like(tau_gas) if out is None else
               _workspace_output(out[0], tau_gas.shape, "optics_tau"))
        get_kernel("rrtmgp_cloud", "rrtmgp_finalize_cloud_lw")(
            *launch, (tau_gas, tau_cloud, ssa_cloud, bands, mask, tau,
                      np.int32(n), np.int32(tables.ngpt),
                      np.int32(tables.nband), np.int32(have_mask)))
        return GasOpticsResult(tau=tau, col_dry=gas.col_dry)

    if gas.g is not None:
        raise ValueError("fused SW optics require the zero gas-g sentinel")
    ssa_gas = _device_profile(gas.ssa, tau_gas.shape, "gas.ssa")
    if out is None:
        tau, ssa, asym = (cp.empty_like(tau_gas) for _ in range(3))
    else:
        tau, ssa, asym = (
            _workspace_output(value, tau_gas.shape, name)
            for value, name in zip(
                out, ("optics_tau", "optics_ssa", "optics_g")))
    get_kernel("rrtmgp_cloud", "rrtmgp_finalize_cloud_sw")(
        *launch, (tau_gas, ssa_gas, tau_cloud, ssa_cloud, g_cloud,
                  bands, mask, tau, ssa, asym, np.int32(n),
                  np.int32(tables.ngpt), np.int32(tables.nband),
                  np.int32(have_mask)))
    return GasOpticsResult(tau, ssa, asym, gas.col_dry)


def _surface_emissivity_bands(value, tables: GasTables, ny: int, nx: int, *,
                              validate=True):
    """Normalize scalar or band-first surface emissivity to (ncol,nband)."""
    import cupy as cp

    emissivity = cp.asarray(value, dtype=DTYPE)
    ncol = ny * nx
    if emissivity.ndim == 0:
        bands = cp.broadcast_to(emissivity, (ncol, tables.nband))
    elif emissivity.shape == (ny, nx):
        bands = cp.broadcast_to(
            emissivity.reshape(ncol, 1), (ncol, tables.nband))
    elif emissivity.shape == (tables.nband,):
        bands = cp.broadcast_to(emissivity[None, :], (ncol, tables.nband))
    elif emissivity.shape == (tables.nband, ny, nx):
        bands = emissivity.transpose(1, 2, 0).reshape(ncol, tables.nband)
    else:
        raise ValueError(
            "emiss must be scalar, (ny,nx), (nband,), or (nband,ny,nx); "
            f"got {emissivity.shape}")
    if validate and (bool(cp.any(~cp.isfinite(bands)))
                     or bool(cp.any(bands < 0.0))
                     or bool(cp.any(bands > 1.0))):
        raise ValueError("surface emissivity must be finite and within [0, 1]")
    return cp.ascontiguousarray(bands)


def _expand_band_to_gpoint(values, tables: GasTables, name="band_values", *,
                           out=None):
    """Expand band values unchanged over each band's g-points.

    Transcribes ``mo_rte_lw.F90:188-191,266-268,476-496`` at fa107a1.
    Input is ``(ncol,nband)`` after adapting Fortran's ``(nband,ncol)``.
    """
    import cupy as cp

    values = cp.ascontiguousarray(cp.asarray(values, dtype=DTYPE))
    if values.ndim != 2 or values.shape[1] != tables.nband:
        raise ValueError(
            f"{name} must have shape (ncol,{tables.nband}), got {values.shape}")
    if out is None:
        return cp.ascontiguousarray(
            values[:, tables.to_device().gpoint_bands])
    target = _workspace_output(
        out, (values.shape[0], tables.ngpt), name)
    # cupy.take writes the full shared view directly; advanced indexing would
    # allocate a second call-local g-point array and defeat real sharing.
    cp.take(values, tables.to_device().gpoint_bands, axis=1, out=target)
    return target


def _validation_error_messages(flags: int) -> tuple[str, ...]:
    """Decode the production validation bitset without device dependencies."""
    return tuple(message for bit, message in _VALIDATION_MESSAGES if flags & bit)


def _validate_device_call_shapes(*, play, plev, tlay, tlev, tsfc, exner,
                                 qv, qc, qr, qi, qs, cldfra, emiss,
                                 numbers, effective):
    """Reject every extent mismatch before the fused kernel sees a pointer."""
    if len(play.shape) != 2:
        raise ValueError("play must have shape (ncol,nlay)")
    ncol, nlay = play.shape
    if ncol < 1 or nlay < 2:
        raise ValueError(
            "RRTMGP profiles must contain at least one column and two layers")
    cell_shape = (ncol, nlay)
    level_shape = (ncol, nlay + 1)
    expected = {
        "plev": (plev, level_shape), "tlay": (tlay, cell_shape),
        "tlev": (tlev, level_shape), "tsfc": (tsfc, (ncol,)),
        "exner": (exner, cell_shape), "qv": (qv, cell_shape),
        "qc": (qc, cell_shape), "qr": (qr, cell_shape),
        "qi": (qi, cell_shape), "qs": (qs, cell_shape),
        "cldfra": (cldfra, cell_shape),
    }
    expected.update((name, (value, cell_shape))
                    for name, value in numbers.items())
    expected.update((name, (value, cell_shape))
                    for name, value in effective.items())
    for name, (value, shape) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if len(emiss.shape) != 2 or emiss.shape[0] != ncol:
        raise ValueError(
            f"surface emissivity must have {ncol} columns, got {emiss.shape}")


def _validation_flags_device(*, play, plev, tlay, tlev, tsfc, exner, qv,
                             qc, qr, qi, qs, cldfra, emiss,
                             numbers, effective, tables_lw, tables_sw):
    """Run the production predicate scan and return its host bitset."""
    _validate_device_call_shapes(
        play=play, plev=plev, tlay=tlay, tlev=tlev, tsfc=tsfc,
        exner=exner, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs,
        cldfra=cldfra, emiss=emiss, numbers=numbers,
        effective=effective)
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    ncol, nlay = play.shape
    dummy = qc
    morrison = bool(numbers)
    have_effective = bool(effective)

    def optional(fields, name):
        return fields[name] if name in fields else dummy

    flags = cp.zeros((1,), dtype=cp.uint32)
    n = max(play.size, plev.size, emiss.size)
    threads = 256
    play_lower = max(float(np.min(tables_lw.press_ref)),
                     float(np.min(tables_sw.press_ref)))
    play_upper = min(float(np.max(tables_lw.press_ref)),
                     float(np.max(tables_sw.press_ref)))
    temp_lower = max(float(np.min(tables_lw.temp_ref)),
                     float(np.min(tables_sw.temp_ref)))
    temp_upper = min(float(np.max(tables_lw.temp_ref)),
                     float(np.max(tables_sw.temp_ref)))
    radii_bands = EFFECTIVE_RADIUS_PLAUSIBLE_UM
    get_kernel("rrtmgp_validation", "rrtmgp_validate_call")(
        ((n + threads - 1) // threads,), (threads,), (
            play, plev, tlay, tlev, tsfc, exner, qv, qc, qr, qi, qs,
            cldfra,
            optional(numbers, "nc"), optional(numbers, "nr"),
            optional(numbers, "ni"), optional(numbers, "ns"),
            optional(effective, "effc"), optional(effective, "effr"),
            optional(effective, "effi"), optional(effective, "effs"),
            emiss, flags, np.int32(ncol), np.int32(nlay),
            np.int32(emiss.size), np.int32(morrison),
            np.int32(have_effective), np.float64(play_lower),
            np.float64(play_upper),
            DTYPE(temp_lower), DTYPE(temp_upper),
            DTYPE(radii_bands["effc"][0]), DTYPE(radii_bands["effc"][1]),
            DTYPE(radii_bands["effi"][0]), DTYPE(radii_bands["effi"][1]),
            DTYPE(radii_bands["effs"][0]), DTYPE(radii_bands["effs"][1])))
    # This is the production path's sole validation synchronization/D2H read.
    return int(cp.asnumpy(flags)[0])


def _validate_device_call(*, diagnose=None, **profiles):
    """Run fused guards, replaying legacy diagnostics only on failure."""
    observed = _validation_flags_device(**profiles)
    if observed:
        if diagnose is not None:
            diagnose()
        raise ValueError(
            "RRTMGP input validation failed: "
            + "; ".join(_validation_error_messages(observed)))


def _raise_full_call_validation_error(*, play, plev, tlay, tlev, tsfc,
                                      exner, qv, qc, qr, qi, qs, cldfra,
                                      emiss, numbers, effective, tables_lw,
                                      tables_sw, column_chunk):
    """Replay the former validators in their observable failure order."""
    import cupy as cp

    if bool(cp.any(~cp.isfinite(plev))):
        raise ValueError("hydrometeor pressure inputs must be finite")
    _require_finite_nonnegative(qc=qc, qr=qr, qi=qi, qs=qs)
    if (bool(cp.any(~cp.isfinite(cldfra)))
            or bool(cp.any(cldfra < 0.0))
            or bool(cp.any(cldfra > 1.0))):
        raise ValueError("cldfra must be finite and within [0, 1]")
    if numbers:
        if (bool(cp.any(~cp.isfinite(play)))
                or bool(cp.any(~cp.isfinite(tlay)))):
            raise ValueError("Morrison thermodynamic inputs must be finite")
        _require_finite_nonnegative(**numbers)
        if effective:
            _require_finite_nonnegative(**effective)
    if effective:
        _require_plausible_radii_um(**{
            name: value for name, value in effective.items()
            if name in EFFECTIVE_RADIUS_PLAUSIBLE_UM})
    if (bool(cp.any(~cp.isfinite(emiss)))
            or bool(cp.any(emiss < 0.0))
            or bool(cp.any(emiss > 1.0))):
        raise ValueError("surface emissivity must be finite and within [0, 1]")
    def replay_gas_chunk(tables, sl, *, planck):
        _require_finite_nonnegative(qv=qv[sl])
        _validate_host_range(
            "play", play[sl], float(np.min(tables.press_ref)),
            float(np.max(tables.press_ref)), "Pa")
        _validate_host_range("plev", plev[sl], 0.0, None, "Pa")
        _validate_host_range(
            "tlay", tlay[sl], float(np.min(tables.temp_ref)),
            float(np.max(tables.temp_ref)), "K")
        if bool(cp.any(play[sl, 0] < play[sl, 1])):
            raise ValueError(
                "kissvec seeding requires pmid from the bottom four layers")
        if planck:
            for name, value in (("tlev", tlev[sl]), ("tsfc", tsfc[sl])):
                _validate_host_range(
                    name, value, float(np.min(tables.temp_ref)),
                    float(np.max(tables.temp_ref)), "K")

    ncol = play.shape[0]
    for start in range(0, ncol, column_chunk):
        sl = slice(start, min(start + column_chunk, ncol))
        replay_gas_chunk(tables_lw, sl, planck=True)
    for start in range(0, ncol, column_chunk):
        sl = slice(start, min(start + column_chunk, ncol))
        replay_gas_chunk(tables_sw, sl, planck=False)
    dp = cp.abs(plev[:, 1:] - plev[:, :-1])
    if (bool(cp.any(dp <= DTYPE(0.0)))
            or bool(cp.any(exner <= DTYPE(0.0)))):
        raise ValueError("radiation pressure thickness and Exner must be positive")


@dataclass
class RRTMGPRadiation:
    """RTE+RRTMGP column driver for the frozen Phase-4 radiation slot.

    State arrays remain in gpuwm's bottom-to-top ``(nz,ny,nx)`` layout;
    columns are packed only at the scheme boundary.  Trace gases use RFMIP
    experiment-zero climatology plus :func:`trace_gases` date selection and
    explicit case overrides.  Water vapor comes from the model and ozone is
    interpolated from the median RFMIP climatological profile.
    """

    #: RTE resolves the full level stack, so the top level's upward
    #: longwave flux IS WRF's OLR; the driver reads this declaration to
    #: decide whether the run's wrfout carries the field.  Unannotated on
    #: purpose: this is a class constant, not a dataclass field.
    publishes_olr = True

    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    # Controller benchmark on the 250x200x49 d01 grid (2026-07-15):
    # 256=21.9 s, 1024=5.54 s, 4096=1.71 s, 12500=1.12 s, 50000=1.36 s
    # per call, with memory flat at 0.65 GiB.  The multi-domain default is
    # capacity-led; keep it configurable for throughput tuning.
    column_chunk: int = DEFAULT_COLUMN_CHUNK
    validation_mode: str = "fused"
    # Trace-gas policy hook (Phase 5, Task 2): a mapping of well-mixed
    # RFMIP gas name -> mole fraction applied over the climatological
    # values.  None applies the pinned date-indexed policy; the frozen 1974
    # profile passes its declared 330 ppm choice explicitly.
    trace_gas_overrides: Mapping[str, float] | None = None
    update_count: int = field(default=0, init=False)
    trace_vmr: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        import cupy as cp

        if not isinstance(self.start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        self.latitude_deg = cp.ascontiguousarray(
            cp.asarray(self.latitude_deg, dtype=DTYPE))
        self.longitude_deg = cp.ascontiguousarray(
            cp.asarray(self.longitude_deg, dtype=DTYPE))
        if self.latitude_deg.shape != self.longitude_deg.shape:
            raise ValueError("radiation latitude/longitude shapes must match")
        if self.column_chunk < 1:
            raise ValueError("column_chunk must be positive")
        if self.validation_mode not in ("fused", "full"):
            raise ValueError("validation_mode must be 'fused' or 'full'")
        self.lw_tables = load_gas_tables("lw")
        self.sw_tables = load_gas_tables("sw")
        if (not np.array_equal(self.lw_tables.press_ref,
                               self.sw_tables.press_ref)
                or not np.array_equal(self.lw_tables.temp_ref,
                                      self.sw_tables.temp_ref)
                or self.lw_tables.press_ref_trop
                != self.sw_tables.press_ref_trop):
            raise ValueError(
                "LW/SW gas tables must share an interpolation grid")
        if float(np.min(self.lw_tables.press_ref)) != RRTMGP_TOA_PRESSURE_PA:
            raise ValueError(
                "RRTMGP coefficient pressure floor drifted from the "
                "above-model column adapter")
        self.lw_cloud_tables = load_cloud_tables("lw")
        self.sw_cloud_tables = load_cloud_tables("sw")
        with Dataset(DATA_DIR / "rfmip-clear-sky-inputs.nc", "r") as ncfile:
            ncfile.set_auto_mask(False)
            for gas, rfmip_name in _RFMIP_GAS_NAMES.items():
                variable = ncfile[rfmip_name + "_GM"]
                scale = float(getattr(variable, "units", "1").replace(" ", ""))
                self.trace_vmr[gas] = float(variable[0]) * scale
            for gas, value in trace_gases(
                    self.start_time, self.trace_gas_overrides).items():
                if gas not in self.trace_vmr:
                    # Defensive parity with the pure policy validation: table
                    # and packaged RFMIP names must never drift silently.
                    raise ValueError(
                        f"unknown trace gas {gas!r}; known well-mixed gases: "
                        f"{sorted(self.trace_vmr)}")
                self.trace_vmr[gas] = value
            pressure = np.median(
                np.asarray(ncfile["pres_layer"][:], np.float64), axis=0)
            ozone = np.median(
                np.asarray(ncfile["ozone"][0], np.float64), axis=0)
        order = np.argsort(pressure)
        self._ozone_logp = cp.asarray(np.log(pressure[order]), dtype=DTYPE)
        self._ozone_vmr = cp.asarray(ozone[order], dtype=DTYPE)

    @staticmethod
    def _columns(array):
        """Pack ``(nz,ny,nx)`` into bottom-to-top ``(ncol,nlay)``."""
        import cupy as cp
        return cp.ascontiguousarray(array.transpose(1, 2, 0).reshape(
            array.shape[1] * array.shape[2], array.shape[0]))

    @staticmethod
    def _field_from_state(state, names, fallback):
        for name in names:
            value = getattr(state, name, None)
            if value is not None:
                return value
        return fallback

    def _gas_vmr(self, tables, play, qv, *, validate=True, out=None):
        import cupy as cp

        shape = (*play.shape, tables.ngas + 1)
        if out is None:
            vmr = cp.zeros(shape, dtype=DTYPE)
        else:
            vmr = _workspace_output(out, shape, "vmr")
            # Slot zero and absent trace gases are real inputs to the gas
            # kernel.  The full fill is the write-before-read producer for
            # every reused byte, exactly matching cp.zeros numerically.
            vmr.fill(DTYPE(0.0))
        # qv is kg water / kg dry air; RRTMGP consumes mole / mole dry air.
        if validate:
            _require_finite_nonnegative(qv=qv)
        vmr[:, :, tables.gas_index["h2o"]] = qv \
            * DTYPE(0.028964 / 0.018016)
        vmr[:, :, tables.gas_index["o3"]] = cp.interp(
            cp.log(play).ravel(), self._ozone_logp, self._ozone_vmr).reshape(
                play.shape)
        for gas, value in self.trace_vmr.items():
            index = tables.gas_index.get(gas)
            if index is not None:
                vmr[:, :, index] = DTYPE(value)
        return cp.ascontiguousarray(vmr)

    @staticmethod
    def _solar_constant(valid_time) -> float:
        """WRF SOLCON = 1370 * ECCFAC (Paltridge & Platt eccentricity).

        Transcribes ``radconst`` (module_radiation_driver.F:3504-3509) with
        WRF's 0-based fractional julian day (frame/module_domain.F:2165):
        1369.704 W/m2 at julian 92.75 (eccfac 0.999784).
        """
        hour = (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0
                + valid_time.microsecond / 3.6e9)
        julian = (valid_time.timetuple().tm_yday - 1) + hour / 24.0
        da = 2.0 * np.pi * julian / 365.0
        eccfac = (1.000110 + 0.034221 * np.cos(da) + 0.001280 * np.sin(da)
                  + 0.000719 * np.cos(2.0 * da)
                  + 0.000077 * np.sin(2.0 * da))
        return 1370.0 * eccfac

    def _cosine_zenith(self, valid_time, *, hour_offset_seconds=0.0):
        """WRF v4.6.1 ``radconst``/``calc_coszen`` solar geometry.

        This transcribes the standard real74 path in
        ``module_radiation_driver.F:3469-3541``.  WRF deliberately uses a
        fixed 365-day orbital phase even in leap years.  The absolute UTC
        ``valid_time`` supplies WRF's zero-based fractional ``julian`` and
        its ``gmt + mod(xtime, 1440)/60`` clock; ``hour_offset_seconds``
        shifts only the hour angle, matching the ``radt*0.5`` midpoint call.

        This is geometric COSZEN only.  It does not emulate optional WRF
        eclipse, slope-shadow, or shortwave-interpolation corrections, none
        of which are selected by the frozen real74 configuration.
        """
        import cupy as cp

        hour = (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0
                + valid_time.microsecond / 3.6e9)
        julian = valid_time.timetuple().tm_yday - 1.0 + hour / 24.0
        degrad = np.pi / 180.0
        dpd = 360.0 / 365.0
        if julian >= 80.0:
            solar_longitude = dpd * (julian - 80.0)
        else:
            solar_longitude = dpd * (julian + 285.0)
        declination = np.arcsin(
            np.sin(23.5 * degrad)
            * np.sin(solar_longitude * degrad))
        da = 2.0 * np.pi * (julian - 1.0) / 365.0
        equation = 229.18 * (
            0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
            - 0.014615 * np.cos(2.0 * da) - 0.04089 * np.sin(2.0 * da))
        solar_minutes = (60.0 * (hour + hour_offset_seconds / 3600.0)
                         + equation + 4.0 * self.longitude_deg)
        hour_angle = cp.deg2rad(solar_minutes / 4.0 - 180.0)
        latitude = cp.deg2rad(self.latitude_deg)
        mu = (cp.sin(latitude) * DTYPE(np.sin(declination))
              + cp.cos(latitude) * DTYPE(np.cos(declination))
              * cp.cos(hour_angle))
        return cp.clip(mu, DTYPE(-1.0), DTYPE(1.0))

    def __call__(self, *, atmosphere, fields, state, cfg):
        pressure = atmosphere["pressure"]
        nz, ny, nx = pressure.shape
        full_validation = self.validation_mode == "full"
        if self.latitude_deg.shape != (ny, nx):
            raise ValueError("radiation latitude/longitude must match state grid")
        declared_p_top = getattr(state, "p_top", None)
        if declared_p_top is not None:
            lw_upper, _ = rrtmgp_above_model_layer_counts(declared_p_top)
            if nz + lw_upper > 128:
                raise ValueError(
                    "RRTMGP radiation supports at most 128 layers including "
                    f"the above-model column, got {nz + lw_upper}")
        if atmosphere["exner"].shape != pressure.shape:
            raise ValueError(
                f"exner must have shape {pressure.shape}, "
                f"got {atmosphere['exner'].shape}")
        if fields["tsk"].size != ny * nx:
            raise ValueError(
                f"tsfc must have {ny * nx} surface values, "
                f"got {fields['tsk'].size}")

        import cupy as cp

        play = self._columns(pressure)
        plev = self._columns(atmosphere["p_interface"])
        tlay = self._columns(atmosphere["temperature"])
        exner = _device_profile(
            self._columns(atmosphere["exner"]), play.shape, "exner")
        qv = self._columns(atmosphere["qv"])
        if declared_p_top is None:
            # Legacy/synthetic direct callers do not carry BaseState.p_top.
            # Production always takes the scalar path above, so this fallback
            # does not add a synchronization to the forecast radiation loop.
            p_top = float(cp.asnumpy(plev[0, -1]))
        else:
            p_top = float(declared_p_top)
        # Both v1.9 gas tables share their lower pressure bound.  Clamp the
        # interface exactly as the upstream RFMIP example does.  For real74
        # this is now the appended TOA interface, not the 100-hPa model top.
        plev[:, -1] = cp.maximum(plev[:, -1],
                                 DTYPE(RRTMGP_TOA_PRESSURE_PA))
        profile_p_top = max(p_top, RRTMGP_TOA_PRESSURE_PA)
        tlev = _interface_temperatures(play, plev, tlay)
        tsfc = _device_profile(
            fields["tsk"].reshape(-1), (play.shape[0],), "tsfc")

        qc3 = self._field_from_state(state, ("qc",), atmosphere["qc"])
        qr3 = self._field_from_state(state, ("qr",), cp.zeros_like(qc3))
        qi3 = self._field_from_state(state, ("qi",), atmosphere["qi"])
        qs3 = self._field_from_state(state, ("qs",), cp.zeros_like(qi3))
        mp_physics = int(getattr(cfg, "mp_physics", 1))
        # Fail-closed table with its WRF citations; see
        # _MP_CLOUD_OPTICS_SCHEME.  mp=28 (THOMPSONAERO) resolves to
        # "thompson" -- the same coupling classic Thompson gets, which is
        # what Registry.EM_COMMON:3036 and module_physics_init.F:1005-1006
        # say.  Until 2026-08-01 this was a ``.get(mp_physics, "kessler")``
        # default and mp=28 silently landed on Kessler: no scheme radii,
        # and f_qi = f_qs = False into cal_cldfra1, so an overcast ice
        # cloud radiated as clear sky.
        scheme = cloud_optics_scheme(mp_physics)
        # The snow radiative treatment is bound to the compatibility
        # receipt token (fail closed on unknown values): -v2 selects the
        # WRF option-4 snow discount, -v1 and native 'none' keep the
        # original full-mass merge, so no already-issued run is relabeled.
        kwargs = {"snow_treatment": snow_treatment_for_compatibility(
            str(getattr(cfg, "wrf_rrtmg_compatibility", "none")))}
        numbers = {}
        effective_fields = {}
        if scheme == "morrison":
            for category, names in {
                    "nc": ("nc", "qnc"), "nr": ("nr", "qnr"),
                    "ni": ("ni", "qni"), "ns": ("ns", "qns")}.items():
                value = self._field_from_state(state, names, None)
                if value is None:
                    raise ValueError(
                        f"Morrison radiation coupling requires state.{names[0]} "
                        f"or state.{names[1]}")
                numbers[category] = self._columns(value)
            kwargs.update({
                "play": play, "tlay": tlay,
                **numbers,
            })
            # These diagnostics describe the just-completed Morrison update and
            # therefore become cloud optics on the *next* radiation call.  On
            # initialisation they are zero-filled state storage, not valid PSD
            # diagnostics, so retain the number-moment reconstruction until the
            # named post-RK contract has accepted at least one update.
            physics = getattr(state, "physics", None)
            have_effective = (getattr(physics, "microphysics_updates", 0) > 0)
            effective = {
                name: (self._field_from_state(state, (name,), None)
                       if have_effective else None)
                for name in ("effc", "effr", "effi", "effs")}
            if any(value is not None for value in effective.values()):
                if any(value is None for value in effective.values()):
                    raise ValueError(
                        "Morrison radiation coupling requires all of state."
                        "effc/effr/effi/effs")
                kwargs.update({name: self._columns(value)
                               for name, value in effective.items()})
                effective_fields = {
                    name: kwargs[name] for name in effective}
        elif scheme in ("wsm6", "thompson", "nssl"):
            effective = {
                name: self._field_from_state(state, (name,), None)
                for name in ("effc", "effi", "effs")}
            if any(value is None for value in effective.values()):
                raise ValueError(
                    f"{scheme} radiation coupling requires "
                    "state.effc/effi/effs")
            kwargs.update({name: self._columns(value)
                           for name, value in effective.items()})
            effective_fields = {name: kwargs[name] for name in effective}
        qc_cols = self._columns(qc3)
        qr_cols = self._columns(qr3)
        qi_cols = self._columns(qi3)
        qs_cols = self._columns(qs3)
        # WRF radiation driver, icloud=1 default: CLDFRA from cal_cldfra1
        # (module_radiation_driver.F:1320-1332), grid-box paths divided by
        # max(0.01, CLDFRA) into in-cloud paths, and McICA subcolumn masks
        # per g-point in the chunk loops below.
        ice_active = scheme_is_ice_active(scheme)
        cldfra = cal_cldfra1(qv, qc_cols, qi_cols, qs_cols, tlay, play,
                             f_qc=True, f_qi=ice_active, f_qs=ice_active)
        active_bl = mynn_bl_cloud_active(
            getattr(cfg, "bl_pbl_physics", 0), getattr(cfg, "icloud_bl", 0))
        qc_bl = self._columns(fields["qc_bl"]) if active_bl else None
        qi_bl = self._columns(fields["qi_bl"]) if active_bl else None
        cldfra_bl = (
            self._columns(fields["cldfra_bl"]) if active_bl else None)
        qc_cols, qi_cols, cldfra = merge_mynn_bl_clouds(
            qc_cols, qi_cols, cldfra, qc_bl=qc_bl, qi_bl=qi_bl,
            cldfra_bl=cldfra_bl,
            bl_pbl_physics=getattr(cfg, "bl_pbl_physics", 0),
            icloud_bl=getattr(cfg, "icloud_bl", 0),
            itimestep=(wrf_itimestep(state.elapsed_seconds, cfg.dt)
                       if active_bl else 1),
        )
        ncol = play.shape[0]
        if full_validation:
            # Validate the caller-supplied model column before constructing
            # WRF's synthetic above-model atmosphere.  Otherwise full mode
            # can report an appended cap pressure as the observed minimum,
            # while the fused production validator diagnoses the original
            # input.  Full mode exists to replay that legacy diagnostic
            # contract exactly, so run the common replay up front and let the
            # downstream cap solvers operate on already-valid profiles.
            emiss_bands = _surface_emissivity_bands(
                fields["emiss"], self.lw_tables, ny, nx, validate=False)
            _raise_full_call_validation_error(
                play=play, plev=plev, tlay=tlay, tlev=tlev,
                tsfc=tsfc, exner=exner, qv=qv, qc=qc_cols,
                qr=qr_cols, qi=qi_cols, qs=qs_cols, cldfra=cldfra,
                emiss=emiss_bands, numbers=numbers,
                effective=effective_fields, tables_lw=self.lw_tables,
                tables_sw=self.sw_tables,
                column_chunk=self.column_chunk)
            paths = hydrometeor_paths(
                plev, qc_cols, qr_cols, qi_cols, qs_cols,
                microphysics=scheme, cldfra=cldfra, validate=False,
                **kwargs)
        else:
            emiss_bands = _surface_emissivity_bands(
                fields["emiss"], self.lw_tables, ny, nx, validate=False)
            _validate_device_call(
                play=play, plev=plev, tlay=tlay, tlev=tlev, tsfc=tsfc,
                exner=exner, qv=qv, qc=qc_cols, qr=qr_cols, qi=qi_cols,
                qs=qs_cols, cldfra=cldfra, emiss=emiss_bands,
                numbers=numbers, effective=effective_fields,
                tables_lw=self.lw_tables, tables_sw=self.sw_tables,
                diagnose=lambda: _raise_full_call_validation_error(
                    play=play, plev=plev, tlay=tlay, tlev=tlev,
                    tsfc=tsfc, exner=exner, qv=qv, qc=qc_cols,
                    qr=qr_cols, qi=qi_cols, qs=qs_cols, cldfra=cldfra,
                    emiss=emiss_bands, numbers=numbers,
                    effective=effective_fields, tables_lw=self.lw_tables,
                    tables_sw=self.sw_tables,
                    column_chunk=self.column_chunk))
            # Invalid hydrometeors are rejected above, before these path/radius
            # allocations and arithmetic.
            paths = hydrometeor_paths(
                plev, qc_cols, qr_cols, qi_cols, qs_cols,
                microphysics=scheme, cldfra=cldfra, validate=False, **kwargs)

        workspace = getattr(self, "chunk_workspace", None)
        if workspace is not None:
            if (int(workspace.nz) != nz
                    or int(workspace.column_chunk) != self.column_chunk
                    or float(workspace.p_top) != p_top):
                raise ValueError(
                    "RRTMGP adapter/workspace shape drift: "
                    f"adapter nz/chunk/p_top={(nz, self.column_chunk, p_top)}, "
                    "workspace "
                    f"{(workspace.nz, workspace.column_chunk, workspace.p_top)}")

        lw_up = cp.empty((ncol, nz + 1), dtype=DTYPE)
        lw_dn = cp.empty_like(lw_up)
        for start in range(0, ncol, self.column_chunk):
            sl = slice(start, min(start + self.column_chunk, ncol))
            chunk_ncol = sl.stop - sl.start
            lw_chunk = _prepare_above_model_chunk(
                tables=self.lw_tables, play=play, plev=plev, tlay=tlay,
                tlev=tlev, qv=qv, paths=paths, cldfra=cldfra, columns=sl,
                p_top=profile_p_top, kind="lw",
                pressure_floor=RRTMGP_TOA_PRESSURE_PA, xp=cp,
                validate_top=declared_p_top is None,
                validate=full_validation)
            lw_profile = lw_chunk.profile
            lw_paths = lw_chunk.paths
            lw_cldfra = lw_chunk.cldfra
            chunk_metadata = lw_chunk.metadata
            if workspace is None:
                vmr_lw = self._gas_vmr(
                    self.lw_tables, lw_profile.play, lw_profile.qv,
                    validate=full_validation)
                gas_lw = _gas_optics(
                    self.lw_tables, lw_profile.play, lw_profile.plev,
                    lw_profile.tlay, vmr_lw,
                    metadata=chunk_metadata, validate=full_validation,
                    zero_g_sentinel=True)
                cld_lw = cloud_optics(
                    self.lw_cloud_tables, lw_paths.clwp, lw_paths.ciwp,
                    lw_paths.reliq, lw_paths.dgice)
                mask_lw = mcica_cloud_masks(
                    lw_profile.play, lw_cldfra, self.lw_tables.ngpt,
                    MCICA_PERMUTESEED_LW, validate=full_validation)
                optics_lw = _finalize_cloud_optics(
                    self.lw_tables, gas_lw, cld_lw, cloud_mask=mask_lw)
                del mask_lw
                sources = _planck_sources(
                    self.lw_tables, lw_profile.play, lw_profile.plev,
                    lw_profile.tlay, lw_profile.tlev, tsfc[sl],
                    vmr_lw, metadata=chunk_metadata,
                    validate=full_validation)
                emiss = _expand_band_to_gpoint(
                    emiss_bands[sl], self.lw_tables, "surface emissivity")
                flux = lw_rte(
                    optics_lw.tau, sources.lay_source, sources.lev_source,
                    sources.sfc_source, emiss, top_at_1=False)
            else:
                # SCRATCH_SLOT_LIFETIME_AUDIT analogue: every optics slot is
                # a kernel output or full fill before its first read in this
                # chunk.  The RTE layout preserves common-slot offsets and
                # overwrites only the now-dead mask tail with its own outputs.
                work = workspace.phase("lw_optics", chunk_ncol)
                vmr_lw = self._gas_vmr(
                    self.lw_tables, lw_profile.play, lw_profile.qv,
                    validate=full_validation, out=work["vmr"])
                gas_lw = _gas_optics(
                    self.lw_tables, lw_profile.play, lw_profile.plev,
                    lw_profile.tlay, vmr_lw,
                    metadata=chunk_metadata, validate=full_validation,
                    zero_g_sentinel=True, out=(work["gas_tau"],),
                    col_dry_out=work["col_dry"])
                cld_lw = _cloud_optics(
                    self.lw_cloud_tables, lw_paths.clwp, lw_paths.ciwp,
                    lw_paths.reliq, lw_paths.dgice,
                    out=(work["cld_tau"], work["cld_ssa"],
                         work["cld_asy"]))
                mask_lw = _mcica_cloud_masks(
                    lw_profile.play, lw_cldfra, self.lw_tables.ngpt,
                    MCICA_PERMUTESEED_LW, validate=full_validation,
                    out=work["mcica_mask"])
                optics_lw = _finalize_cloud_optics(
                    self.lw_tables, gas_lw, cld_lw, cloud_mask=mask_lw,
                    out=(work["optics_tau"],))
                del mask_lw
                work = workspace.phase("lw_rte", chunk_ncol)
                sources = _planck_sources(
                    self.lw_tables, lw_profile.play, lw_profile.plev,
                    lw_profile.tlay, lw_profile.tlev, tsfc[sl],
                    vmr_lw, metadata=chunk_metadata,
                    validate=full_validation,
                    out=(work["lay_source"], work["lev_source"],
                         work["sfc_source"]))
                emiss = _expand_band_to_gpoint(
                    emiss_bands[sl], self.lw_tables, "surface emissivity",
                    out=work["emiss_gpt"])
                flux = _lw_rte(
                    optics_lw.tau, sources.lay_source, sources.lev_source,
                    sources.sfc_source, emiss, top_at_1=False,
                    out=(work["flux_up"], work["flux_dn"]),
                    incident_out=work["incident"])
            lw_up[sl] = _model_flux_interfaces(
                flux.flux_up, nz, xp=cp)
            lw_dn[sl] = _model_flux_interfaces(
                flux.flux_dn, nz, xp=cp)
            del vmr_lw, gas_lw, cld_lw, optics_lw, sources, emiss, flux
            del chunk_metadata, lw_profile, lw_paths, lw_cldfra, lw_chunk

        valid_time = (self.start_time
                      + timedelta(seconds=float(state.elapsed_seconds)))
        # WRF evaluates the HOUR ANGLE at the CENTER of the radiation
        # interval: calc_coszen receives xtime + radt*0.5 inside the
        # Solar_step block (module_radiation_driver.F:1206-1208, 'jararias
        # 2013/08/10') while declination/EOT stay at the call-time julian
        # (:3514-3541); that coszen is what RRTMG SW consumes (driver:2636).
        from gpuwm.core.physics import (
            _model_clock_dt, _physics_interval_seconds)
        radt_minutes = cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes
        radt_seconds = _physics_interval_seconds(
            radt_minutes, _model_clock_dt(cfg))
        mu_raw = self._cosine_zenith(
            valid_time, hour_offset_seconds=0.5 * radt_seconds)
        daylight = mu_raw > DTYPE(0.0)
        mu = cp.where(daylight, mu_raw, DTYPE(1.0)).reshape(-1)
        albedo_surface = cp.asarray(
            fields["albedo"], dtype=DTYPE).reshape(-1)
        solar = cp.asarray(self.sw_tables.solar_source, dtype=DTYPE)
        # WRF scales every SW band by scon/rrsw_scon so TOA irradiance
        # equals radconst's SOLCON (module_ra_rrtmg_sw.F:10872 'scon =
        # solcon*(1-obscur)', 9867-9871 'solvar(ib) = scon/rrsw_scon');
        # the RRTMGP-native equivalent normalizes the g-point source total
        # (the shipped table sums to tsi_default = 1360.8577 W/m2).
        solar = solar * DTYPE(
            self._solar_constant(valid_time)
            / float(np.sum(np.asarray(self.sw_tables.solar_source,
                                      dtype=np.float64))))
        sw_up = cp.empty_like(lw_up)
        sw_dn = cp.empty_like(lw_up)
        for start in range(0, ncol, self.column_chunk):
            sl = slice(start, min(start + self.column_chunk, ncol))
            chunk_ncol = sl.stop - sl.start
            sw_chunk = _prepare_above_model_chunk(
                tables=self.sw_tables, play=play, plev=plev, tlay=tlay,
                tlev=tlev, qv=qv, paths=paths, cldfra=cldfra, columns=sl,
                p_top=profile_p_top, kind="sw",
                pressure_floor=RRTMGP_TOA_PRESSURE_PA, xp=cp,
                validate_top=declared_p_top is None,
                validate=full_validation)
            sw_profile = sw_chunk.profile
            sw_paths = sw_chunk.paths
            sw_cldfra = sw_chunk.cldfra
            chunk_metadata = sw_chunk.metadata
            if workspace is None:
                vmr_sw = self._gas_vmr(
                    self.sw_tables, sw_profile.play, sw_profile.qv,
                    validate=full_validation)
                gas_sw = _gas_optics(
                    self.sw_tables, sw_profile.play, sw_profile.plev,
                    sw_profile.tlay, vmr_sw,
                    metadata=chunk_metadata, validate=full_validation,
                    zero_g_sentinel=True)
                cld_sw = cloud_optics(
                    self.sw_cloud_tables, sw_paths.clwp, sw_paths.ciwp,
                    sw_paths.reliq, sw_paths.dgice)
                mask_sw = mcica_cloud_masks(
                    sw_profile.play, sw_cldfra, self.sw_tables.ngpt,
                    MCICA_PERMUTESEED_SW, validate=full_validation)
                optics_sw = _finalize_cloud_optics(
                    self.sw_tables, gas_sw, cld_sw, cloud_mask=mask_sw)
                del mask_sw
                albedo = cp.ascontiguousarray(cp.broadcast_to(
                    albedo_surface[sl, None],
                    (chunk_ncol, self.sw_tables.ngpt)))
                inc = cp.ascontiguousarray(cp.broadcast_to(
                    solar[None, :],
                    (chunk_ncol, self.sw_tables.ngpt)))
                flux = sw_rte(
                    optics_sw.tau, optics_sw.ssa, optics_sw.g, mu[sl],
                    albedo, albedo, inc, top_at_1=False)
            else:
                work = workspace.phase("sw_optics", chunk_ncol)
                vmr_sw = self._gas_vmr(
                    self.sw_tables, sw_profile.play, sw_profile.qv,
                    validate=full_validation, out=work["vmr"])
                gas_sw = _gas_optics(
                    self.sw_tables, sw_profile.play, sw_profile.plev,
                    sw_profile.tlay, vmr_sw,
                    metadata=chunk_metadata, validate=full_validation,
                    zero_g_sentinel=True,
                    out=(work["gas_tau"], work["gas_ssa"]),
                    col_dry_out=work["col_dry"])
                cld_sw = _cloud_optics(
                    self.sw_cloud_tables, sw_paths.clwp, sw_paths.ciwp,
                    sw_paths.reliq, sw_paths.dgice,
                    out=(work["cld_tau"], work["cld_ssa"],
                         work["cld_asy"]))
                mask_sw = _mcica_cloud_masks(
                    sw_profile.play, sw_cldfra, self.sw_tables.ngpt,
                    MCICA_PERMUTESEED_SW, validate=full_validation,
                    out=work["mcica_mask"])
                optics_sw = _finalize_cloud_optics(
                    self.sw_tables, gas_sw, cld_sw, cloud_mask=mask_sw,
                    out=(work["optics_tau"], work["optics_ssa"],
                         work["optics_g"]))
                del mask_sw
                work = workspace.phase("sw_rte", chunk_ncol)
                albedo = work["albedo_gpt"]
                albedo[...] = albedo_surface[sl, None]
                inc = work["inc_gpt"]
                inc[...] = solar[None, :]
                mu_chunk = work["mu0"]
                mu_chunk[...] = mu[sl, None]
                flux = _sw_rte(
                    optics_sw.tau, optics_sw.ssa, optics_sw.g, mu_chunk,
                    albedo, albedo, inc, top_at_1=False,
                    out=(work["flux_up"], work["flux_dn"],
                         work["flux_dir"]))
            mask = daylight.reshape(-1)[sl, None]
            sw_up[sl] = cp.where(
                mask, _model_flux_interfaces(flux.flux_up, nz, xp=cp),
                DTYPE(0.0))
            sw_dn[sl] = cp.where(
                mask, _model_flux_interfaces(flux.flux_dn, nz, xp=cp),
                DTYPE(0.0))
            del vmr_sw, gas_sw, cld_sw, optics_sw
            del albedo, inc, flux, mask, chunk_metadata
            del sw_profile, sw_paths, sw_cldfra, sw_chunk

        result = _fluxes_to_radiation(
            lw_up, lw_dn, sw_up, sw_dn, plev, exner, ny=ny, nx=nx,
            coszen=mu_raw, validate=full_validation)
        self.update_count += 1
        return result


def _device_profile(value, shape, name):
    import cupy as cp
    out = cp.ascontiguousarray(cp.asarray(value, dtype=DTYPE))
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    return out


def _workspace_output(out, shape, name, *, dtype=DTYPE):
    """Return an exact contiguous output view, allocating only if absent."""
    import cupy as cp

    shape = tuple(int(extent) for extent in shape)
    if out is None:
        return cp.empty(shape, dtype=dtype)
    if tuple(out.shape) != shape:
        raise ValueError(
            f"RRTMGP workspace {name} shape {tuple(out.shape)} != {shape}")
    if np.dtype(out.dtype) != np.dtype(dtype):
        raise ValueError(
            f"RRTMGP workspace {name} dtype {out.dtype} != {np.dtype(dtype)}")
    if not bool(out.flags.c_contiguous):
        raise ValueError(f"RRTMGP workspace {name} must be C-contiguous")
    return out


def _interface_temperatures(play, plev, tlay):
    """Construct pressure-weighted level temperatures.

    Exact transcription of ``mo_gas_optics_rrtmgp.F90:890-909`` at the
    pinned fa107a1 commit, including pressure-linear boundary extrapolation
    and the reference's pressure-weighted interior interpolation.
    """
    import cupy as cp

    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2 or play.shape[1] < 2:
        raise ValueError("play must have shape (ncol,nlay), nlay >= 2")
    ncol, nlay = play.shape
    plev = _device_profile(plev, (ncol, nlay + 1), "plev")
    tlay = _device_profile(tlay, (ncol, nlay), "tlay")
    tlev = cp.empty((ncol, nlay + 1), dtype=DTYPE)
    tlev[:, 0] = tlay[:, 0] + (plev[:, 0] - play[:, 0]) * (
        tlay[:, 1] - tlay[:, 0]) / (play[:, 1] - play[:, 0])
    tlev[:, -1] = tlay[:, -1] + (plev[:, -1] - play[:, -1]) * (
        tlay[:, -1] - tlay[:, -2]) / (play[:, -1] - play[:, -2])
    tlev[:, 1:-1] = (
        play[:, :-1] * tlay[:, :-1] * (plev[:, 1:-1] - play[:, 1:])
        + play[:, 1:] * tlay[:, 1:] * (play[:, :-1] - plev[:, 1:-1])
    ) / (plev[:, 1:-1] * (play[:, :-1] - play[:, 1:]))
    return cp.ascontiguousarray(tlev)


def _validate_host_range(name, value, lower, upper, unit):
    """Validate a device profile on the host before launching RRTMGP."""
    import cupy as cp

    host = np.asarray(cp.asnumpy(value), dtype=np.float64)
    if host.size == 0 or not np.all(np.isfinite(host)):
        raise ValueError(f"{name} range contains non-finite values")
    observed = (float(np.min(host)), float(np.max(host)))
    if observed[0] < lower or (upper is not None and observed[1] > upper):
        bound = (f"[{lower:.9g}, {upper:.9g}]" if upper is not None
                 else f"[{lower:.9g}, infinity)")
        raise ValueError(
            f"{name} range [{observed[0]:.9g}, {observed[1]:.9g}] {unit} "
            f"is outside allowed range {bound} {unit}")


def _require_finite_nonnegative(**fields):
    """Reject upstream physics defects instead of silently clearing them."""
    import cupy as cp

    for name, value in fields.items():
        finite = cp.isfinite(value)
        negative = finite & (value < 0.0)
        invalid = ~finite | negative
        if bool(cp.any(invalid)):
            # This replay runs only after the fused production predicate has
            # already failed.  Spend the extra failure-path synchronizations
            # to preserve the actual upstream defect in the capsule instead
            # of reducing a multi-million-cell field to a generic label.
            flat_index = int(cp.asnumpy(cp.argmax(invalid.reshape(-1))))
            first_value = float(cp.asnumpy(value.reshape(-1)[flat_index]))
            negative_count = int(cp.asnumpy(cp.count_nonzero(negative)))
            nonfinite_count = int(cp.asnumpy(cp.count_nonzero(~finite)))
            index = tuple(int(part) for part in np.unravel_index(
                flat_index, tuple(int(extent) for extent in value.shape)))
            raise ValueError(
                f"{name} must be finite and non-negative: "
                f"first_index={index}, first_value={first_value:.9g}, "
                f"negative_count={negative_count}, "
                f"nonfinite_count={nonfinite_count}")


def _require_plausible_radii_um(**fields):
    """Gate the micron contract of every radiation-facing radii writer.

    Keyword names select the band from :data:`EFFECTIVE_RADIUS_PLAUSIBLE_UM`.
    A writer that emits metres (or any other metric prefix) lands outside
    its band on background-filled cells alone and fails here instead of
    silently radiating at a clip floor.
    """
    import cupy as cp

    for name, value in fields.items():
        lower, upper = EFFECTIVE_RADIUS_PLAUSIBLE_UM[name]
        if bool(cp.any(value < DTYPE(lower))) \
                or bool(cp.any(value > DTYPE(upper))):
            raise ValueError(
                f"{name} is outside the physical-plausibility band "
                f"[{lower}, {upper}] microns; the state contract is "
                "microns -- a radii writer probably emitted another unit")


def _fluxes_to_radiation(lw_up, lw_dn, sw_up, sw_dn, plev, exner, *,
                         ny, nx, coszen=None, validate=True):
    """Map bottom-to-top broadband column fluxes into the radiation slot."""
    import cupy as cp
    from gpuwm.core import constants
    from gpuwm.core.physics import RadiationResult

    exner = cp.ascontiguousarray(cp.asarray(exner, dtype=DTYPE))
    if exner.ndim != 2 or exner.shape[0] != ny * nx:
        raise ValueError("exner columns must have shape (ny*nx,nlay)")
    ncol, nlay = exner.shape
    level_shape = (ncol, nlay + 1)
    lw_up = _device_profile(lw_up, level_shape, "lw_up")
    lw_dn = _device_profile(lw_dn, level_shape, "lw_dn")
    sw_up = _device_profile(sw_up, level_shape, "sw_up")
    sw_dn = _device_profile(sw_dn, level_shape, "sw_dn")
    plev = _device_profile(plev, level_shape, "plev")
    dp = cp.abs(plev[:, 1:] - plev[:, :-1])
    if validate and (bool(cp.any(dp <= DTYPE(0.0)))
                     or bool(cp.any(exner <= DTYPE(0.0)))):
        raise ValueError("radiation pressure thickness and Exner must be positive")

    def theta_heating(flux_up, flux_dn):
        # RTE+RRTMGP ``rte/extensions/mo_heating_rates.F90:30-63`` diagnoses
        # temperature tendency from pressure-coordinate flux convergence.
        # The frozen gpuwm slot consumes potential-temperature tendency.
        net_down = flux_dn - flux_up
        convergence = net_down[:, 1:] - net_down[:, :-1]
        packed = DTYPE(constants.G / constants.CP) * convergence / dp / exner
        return cp.ascontiguousarray(packed.reshape(ny, nx, nlay).transpose(
            2, 0, 1))

    return RadiationResult(
        rthratenlw=theta_heating(lw_up, lw_dn),
        rthratensw=theta_heating(sw_up, sw_dn),
        swdown=cp.ascontiguousarray(sw_dn[:, 0].reshape(ny, nx)),
        glw=cp.ascontiguousarray(lw_dn[:, 0].reshape(ny, nx)),
        # OLR: the upward longwave flux at the TOP level of the same
        # bottom-to-top level stack whose level 0 supplies GLW above.
        olr=cp.ascontiguousarray(lw_up[:, -1].reshape(ny, nx)),
        gsw=cp.ascontiguousarray(
            (sw_dn[:, 0] - sw_up[:, 0]).reshape(ny, nx)),
        coszen=(None if coszen is None else cp.ascontiguousarray(
            cp.asarray(coszen, dtype=DTYPE).reshape(ny, nx))))


def _interpolation_metadata(tables: GasTables, play, tlay, *,
                            validate=True) -> _InterpolationMetadata:
    """Compute driver-owned reference interpolation coordinates once.

    The CUDA prepass transcribes the expressions formerly evaluated inside
    each gas-optics call and inside Planck's g-point loop.  It is intentionally
    recomputed for every radiation call; no cross-call cache or public reuse
    contract is kept.
    """
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2:
        raise ValueError("play must be a 2-D (ncol,nlay) array")
    tlay = _device_profile(tlay, play.shape, "tlay")
    if validate:
        _validate_host_range("play", play, float(np.min(tables.press_ref)),
                             float(np.max(tables.press_ref)), "Pa")
        _validate_host_range("tlay", tlay, float(np.min(tables.temp_ref)),
                             float(np.max(tables.temp_ref)), "K")
    d = tables.to_device()
    integer = tuple(cp.empty(play.shape, dtype=cp.int32) for _ in range(3))
    fraction = tuple(cp.empty(play.shape, dtype=DTYPE) for _ in range(2))
    n = play.size
    threads = 256
    get_kernel("rrtmgp_gas", "rrtmgp_interpolation_prepass")(
        ((n + threads - 1) // threads,), (threads,), (
            play, tlay, d.press_ref, d.temp_ref,
            DTYPE(tables.press_ref_trop), *integer, *fraction,
            np.int32(n), np.int32(tables.ntemp), np.int32(tables.npres)))
    return _InterpolationMetadata(*integer, *fraction)


def _normalize_interpolation_metadata(metadata, shape):
    import cupy as cp

    def profile(value, dtype, name):
        out = cp.ascontiguousarray(cp.asarray(value, dtype=dtype))
        if out.shape != shape:
            raise ValueError(
                f"interpolation metadata {name} must have shape {shape}, "
                f"got {out.shape}")
        return out

    if not isinstance(metadata, _InterpolationMetadata):
        raise TypeError("metadata must be driver-owned interpolation metadata")
    return _InterpolationMetadata(
        profile(metadata.iatm, cp.int32, "iatm"),
        profile(metadata.jt, cp.int32, "jt"),
        profile(metadata.jp, cp.int32, "jp"),
        profile(metadata.ftemp, DTYPE, "ftemp"),
        profile(metadata.fpress, DTYPE, "fpress"))


def gas_optics(tables: GasTables, play, plev, tlay, vmr) -> GasOpticsResult:
    """Compute FP32 gas optical properties on device.

    ``vmr`` has shape ``(ncol,nlay,ngas+1)``; slot zero is reserved for dry
    air and ignored on input.  The CUDA transcription uses one thread per
    column and retains the reference pressure/temperature/eta interpolation,
    all available minor gases, and SW Rayleigh scattering.
    """
    return _gas_optics(
        tables, play, plev, tlay, vmr, metadata=None, validate=True,
        zero_g_sentinel=False, out=None, col_dry_out=None)


def _gas_optics(tables: GasTables, play, plev, tlay, vmr, *, metadata,
                validate, zero_g_sentinel, out=None,
                col_dry_out=None) -> GasOpticsResult:
    """Internal gas optics path supporting one-call shared metadata."""
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2:
        raise ValueError("play must be a 2-D (ncol,nlay) array")
    ncol, nlay = play.shape
    plev = _device_profile(plev, (ncol, nlay + 1), "plev")
    tlay = _device_profile(tlay, (ncol, nlay), "tlay")
    vmr = _device_profile(vmr, (ncol, nlay, tables.ngas + 1), "vmr")
    if validate:
        _validate_host_range("play", play, float(np.min(tables.press_ref)),
                             float(np.max(tables.press_ref)), "Pa")
        _validate_host_range("plev", plev, 0.0, None, "Pa")
        _validate_host_range("tlay", tlay, float(np.min(tables.temp_ref)),
                             float(np.max(tables.temp_ref)), "K")
    if metadata is None:
        metadata = _interpolation_metadata(
            tables, play, tlay, validate=False)
    else:
        metadata = _normalize_interpolation_metadata(metadata, play.shape)
    d = tables.to_device()
    shape = (ncol, nlay, tables.ngpt)
    if out is None:
        tau = cp.empty(shape, dtype=DTYPE)
        ssa = (cp.empty_like(tau) if tables.kind == "sw"
               else cp.empty((1,), dtype=DTYPE))
    else:
        tau = _workspace_output(out[0], shape, "gas_tau")
        ssa = (_workspace_output(out[1], shape, "gas_ssa")
               if tables.kind == "sw" else tau)
    # LW has no Rayleigh array; pass a valid dummy pointer which is never read.
    rayleigh = (d.rayleigh if tables.rayleigh is not None
                else (tau if out is not None
                      else cp.empty((1,), dtype=DTYPE)))
    threads = 64
    blocks = (ncol + threads - 1) // threads
    kernel = get_kernel("rrtmgp_gas", "rrtmgp_gas_optics")
    kernel((blocks,), (threads,), (
        play, plev, tlay, vmr, metadata.iatm, metadata.jt, metadata.jp,
        metadata.ftemp, metadata.fpress, d.vmr_ref, d.flavor,
        d.gpoint_flavor, d.kmajor, d.kminor_lower, d.kminor_upper,
        d.minor_limits_gpt_lower, d.minor_limits_gpt_upper,
        d.minor_scales_with_density_lower,
        d.minor_scales_with_density_upper, d.scale_by_complement_lower,
        d.scale_by_complement_upper, d.idx_minor_lower, d.idx_minor_upper,
        d.idx_minor_scaling_lower, d.idx_minor_scaling_upper,
        d.kminor_start_lower, d.kminor_start_upper, rayleigh, tau, ssa,
        np.int32(ncol), np.int32(nlay), np.int32(tables.ngas),
        np.int32(tables.nflav), np.int32(tables.ngpt),
        np.int32(tables.ntemp), np.int32(tables.npres),
        np.int32(tables.neta),
        np.int32(tables.minor_limits_gpt_lower.shape[0]),
        np.int32(tables.minor_limits_gpt_upper.shape[0]),
        np.int32(tables.kminor_lower.shape[2]),
        np.int32(tables.kminor_upper.shape[2]),
        np.int32(tables.gas_index["h2o"]),
        np.int32(tables.kind == "sw")))
    h2o = vmr[:, :, tables.gas_index["h2o"]]
    fact = DTYPE(1.0) / (DTYPE(1.0) + h2o)
    m_air = (DTYPE(0.028964) + DTYPE(0.018016) * h2o) * fact
    col_dry_value = (cp.abs(plev[:, 1:] - plev[:, :-1])
                     * DTYPE(6.02214076e23) * fact
                     / (DTYPE(10000.0) * m_air * DTYPE(9.80665)))
    if col_dry_out is None:
        col_dry = col_dry_value
    else:
        col_dry = _workspace_output(
            col_dry_out, (ncol, nlay), "col_dry")
        col_dry[...] = col_dry_value
    if tables.kind == "lw":
        return GasOpticsResult(tau=tau, col_dry=col_dry)
    # The zero-field sentinel is private to the production fused finalizer.
    # Exported gas_optics retains its historical allocated zero array.
    asym = None if zero_g_sentinel else cp.zeros_like(tau)
    return GasOpticsResult(tau=tau, ssa=ssa, g=asym, col_dry=col_dry)


def delta_scale(tau, ssa, g):
    """Return reference-default delta-scaled FP32 two-stream properties."""
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    tau = cp.ascontiguousarray(cp.asarray(tau, dtype=DTYPE))
    ssa = _device_profile(ssa, tau.shape, "ssa")
    g = _device_profile(g, tau.shape, "g")
    out = tuple(cp.empty_like(tau) for _ in range(3))
    n = tau.size
    threads = 256
    get_kernel("rrtmgp_rte", "rrtmgp_delta_scale")(
        ((n + threads - 1) // threads,), (threads,),
        (tau, ssa, g, *out, np.int32(n)))
    return out


def lw_rte(tau, lay_source, lev_source, sfc_source, sfc_emis,
           incident_flux=None, *, top_at_1: bool) -> FluxResult:
    """Run the FP32 one-angle LW no-scattering solver on device."""
    return _lw_rte(
        tau, lay_source, lev_source, sfc_source, sfc_emis,
        incident_flux=incident_flux, top_at_1=top_at_1,
        out=None, incident_out=None)


def _lw_rte(tau, lay_source, lev_source, sfc_source, sfc_emis,
            incident_flux=None, *, top_at_1: bool, out,
            incident_out) -> FluxResult:
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    tau = cp.ascontiguousarray(cp.asarray(tau, dtype=DTYPE))
    if tau.ndim != 3:
        raise ValueError("tau must have shape (ncol,nlay,ngpt)")
    ncol, nlay, ngpt = tau.shape
    if nlay > 128:
        raise ValueError("RRTMGP CUDA RTE supports at most 128 layers")
    lay_source = _device_profile(lay_source, tau.shape, "lay_source")
    lev_source = _device_profile(
        lev_source, (ncol, nlay + 1, ngpt), "lev_source")
    sfc_source = _device_profile(sfc_source, (ncol, ngpt), "sfc_source")
    sfc_emis = _device_profile(sfc_emis, (ncol, ngpt), "sfc_emis")
    if incident_flux is None:
        if incident_out is None:
            incident = cp.zeros((ncol, ngpt), dtype=DTYPE)
        else:
            incident = _workspace_output(
                incident_out, (ncol, ngpt), "incident")
            incident.fill(DTYPE(0.0))
    else:
        incident = _device_profile(
            incident_flux, (ncol, ngpt), "incident_flux")
    flux_shape = (ncol, nlay + 1)
    if out is None:
        up = cp.empty(flux_shape, dtype=DTYPE)
        down = cp.empty_like(up)
    else:
        up = _workspace_output(out[0], flux_shape, "flux_up")
        down = _workspace_output(out[1], flux_shape, "flux_dn")
    threads = 64
    get_kernel("rrtmgp_rte", "rrtmgp_lw_noscat")(
        ((ncol + threads - 1) // threads,), (threads,),
        (tau, lay_source, lev_source, sfc_source, sfc_emis, incident,
         up, down, np.int32(ncol), np.int32(nlay), np.int32(ngpt),
         np.int32(top_at_1)))
    return FluxResult(up, down)


def sw_rte(tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
           *, top_at_1: bool) -> FluxResult:
    """Run the FP32 delta-scaled PIFM two-stream SW solver on device."""
    return _sw_rte(
        tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
        top_at_1=top_at_1, out=None)


def _sw_rte(tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
            *, top_at_1: bool, out) -> FluxResult:
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    tau = cp.ascontiguousarray(cp.asarray(tau, dtype=DTYPE))
    if tau.ndim != 3:
        raise ValueError("tau must have shape (ncol,nlay,ngpt)")
    ncol, nlay, ngpt = tau.shape
    if nlay > 128:
        raise ValueError("RRTMGP CUDA RTE supports at most 128 layers")
    ssa = _device_profile(ssa, tau.shape, "ssa")
    g = _device_profile(g, tau.shape, "g")
    mu0 = cp.asarray(mu0, dtype=DTYPE)
    if mu0.shape == (ncol,):
        mu0 = cp.broadcast_to(mu0[:, None], (ncol, nlay))
    mu0 = _device_profile(mu0, (ncol, nlay), "mu0")
    alb_dir = _device_profile(sfc_alb_dir, (ncol, ngpt), "sfc_alb_dir")
    alb_dif = _device_profile(sfc_alb_dif, (ncol, ngpt), "sfc_alb_dif")
    inc = _device_profile(inc_flux_dir, (ncol, ngpt), "inc_flux_dir")
    flux_shape = (ncol, nlay + 1)
    if out is None:
        up = cp.empty(flux_shape, dtype=DTYPE)
        down = cp.empty_like(up)
        direct = cp.empty_like(up)
    else:
        up, down, direct = (
            _workspace_output(value, flux_shape, name)
            for value, name in zip(
                out, ("flux_up", "flux_dn", "flux_dir")))
    threads = 64
    get_kernel("rrtmgp_rte", "rrtmgp_sw_2stream")(
        ((ncol + threads - 1) // threads,), (threads,),
        (tau, ssa, g, mu0, alb_dir, alb_dif, inc, up, down, direct,
         np.int32(ncol), np.int32(nlay), np.int32(ngpt),
         np.int32(top_at_1), np.int32(0)))
    return FluxResult(up, down, direct)


def planck_sources(tables: GasTables, play, plev, tlay, tlev, tsfc,
                   vmr) -> PlanckSourceResult:
    """Compute FP32 RRTMGP LW Planck sources on device."""
    return _planck_sources(
        tables, play, plev, tlay, tlev, tsfc, vmr,
        metadata=None, validate=True, out=None)


def _planck_sources(tables: GasTables, play, plev, tlay, tlev, tsfc,
                    vmr, *, metadata, validate, out=None) -> PlanckSourceResult:
    """Internal Planck path supporting driver-owned shared metadata."""
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    if tables.kind != "lw":
        raise ValueError("Planck sources require LW gas tables")
    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2:
        raise ValueError("play must have shape (ncol,nlay)")
    ncol, nlay = play.shape
    # plev is validated because it is part of the public gas/source profile
    # contract even though Planck interpolation itself only consumes play.
    plev = _device_profile(plev, (ncol, nlay + 1), "plev")
    tlay = _device_profile(tlay, (ncol, nlay), "tlay")
    tlev = _device_profile(tlev, (ncol, nlay + 1), "tlev")
    tsfc = _device_profile(tsfc, (ncol,), "tsfc")
    vmr = _device_profile(vmr, (ncol, nlay, tables.ngas + 1), "vmr")
    if validate:
        _validate_host_range("play", play, float(np.min(tables.press_ref)),
                             float(np.max(tables.press_ref)), "Pa")
        _validate_host_range("plev", plev, 0.0, None, "Pa")
        for name, value in (("tlay", tlay), ("tlev", tlev), ("tsfc", tsfc)):
            _validate_host_range(name, value, float(np.min(tables.temp_ref)),
                                 float(np.max(tables.temp_ref)), "K")
    if metadata is None:
        metadata = _interpolation_metadata(
            tables, play, tlay, validate=False)
    else:
        metadata = _normalize_interpolation_metadata(metadata, play.shape)
    d = tables.to_device()
    shapes = ((ncol, nlay, tables.ngpt),
              (ncol, nlay + 1, tables.ngpt),
              (ncol, tables.ngpt))
    if out is None:
        lay = cp.empty(shapes[0], dtype=DTYPE)
        lev = cp.empty(shapes[1], dtype=DTYPE)
        sfc = cp.empty(shapes[2], dtype=DTYPE)
    else:
        lay, lev, sfc = (
            _workspace_output(value, shape, name)
            for value, shape, name in zip(
                out, shapes, ("lay_source", "lev_source", "sfc_source")))
    threads = 64
    get_kernel("rrtmgp_gas", "rrtmgp_planck_sources")(
        ((ncol + threads - 1) // threads,), (threads,),
        (play, tlay, tlev, tsfc, vmr, metadata.iatm, metadata.jt,
         metadata.jp, metadata.ftemp, metadata.fpress, d.temp_ref,
         d.vmr_ref, d.flavor,
         d.gpoint_flavor, d.gpoint_bands, d.planck_fraction, d.totplnk,
         lay, lev, sfc, np.int32(ncol), np.int32(nlay),
         np.int32(tables.ngas), np.int32(tables.ngpt),
         np.int32(tables.ntemp), np.int32(tables.npres),
         np.int32(tables.neta), np.int32(tables.nband),
         np.int32(tables.totplnk.shape[0])))
    return PlanckSourceResult(lay, lev, sfc)


def _rfmip_profiles(tables, sites, experiments):
    sites = np.asarray(sites, dtype=np.intp)
    experiments = np.asarray(experiments, dtype=np.intp)
    with Dataset(DATA_DIR / "rfmip-clear-sky-inputs.nc", "r") as nc:
        nc.set_auto_mask(False)
        nsite, nexp = sites.size, experiments.size
        play_site = np.asarray(nc["pres_layer"][sites], np.float64)
        plev_site = np.asarray(nc["pres_level"][sites], np.float64)
        play = np.broadcast_to(play_site[None], (nexp, *play_site.shape))
        plev = np.broadcast_to(
            plev_site[None], (nexp, *plev_site.shape)).copy()
        # The RFMIP top boundary is 0.01 Pa, below the coefficient grid.
        # Match the upstream example's explicit input sanitization.
        top = 0 if play_site[0, 0] < play_site[0, -1] else -1
        plev[..., top] = tables.press_ref[-1] + np.finfo(np.float64).eps
        tlay = np.asarray(nc["temp_layer"][experiments][:, sites], np.float64)
        tlev = np.asarray(nc["temp_level"][experiments][:, sites], np.float64)
        tsfc = np.asarray(
            nc["surface_temperature"][experiments][:, sites], np.float64)
        vmr = np.zeros((*tlay.shape, tables.ngas + 1), np.float64)
        vmr[..., tables.gas_index["h2o"]] = np.asarray(
            nc["water_vapor"][experiments][:, sites], np.float64)
        vmr[..., tables.gas_index["o3"]] = np.asarray(
            nc["ozone"][experiments][:, sites], np.float64)
        for gas, rfmip_name in _RFMIP_GAS_NAMES.items():
            variable = nc[rfmip_name + "_GM"]
            scale = float(getattr(variable, "units", "1").replace(" ", ""))
            values = np.asarray(variable[experiments], np.float64) * scale
            vmr[..., tables.gas_index[gas]] = values[:, None, None]
        emiss = np.asarray(nc["surface_emissivity"][sites], np.float64)
        albedo = np.asarray(nc["surface_albedo"][sites], np.float64)
        sza = np.asarray(nc["solar_zenith_angle"][sites], np.float64)
        tsi = np.asarray(nc["total_solar_irradiance"][sites], np.float64)
    def flat(a):
        return np.ascontiguousarray(a.reshape(nexp * nsite, *a.shape[2:]))
    return (flat(play), flat(plev), flat(tlay), flat(tlev),
            np.ascontiguousarray(tsfc.reshape(-1)),
            flat(vmr), np.tile(emiss, nexp), np.tile(albedo, nexp),
            np.tile(sza, nexp), np.tile(tsi, nexp))


def rfmip_clear_sky(*, sites=None, experiments=None) -> RFMIPResult:
    """Run the shipped RFMIP clear-sky oracle profiles on the GPU.

    This reproduces the upstream physics-index-1/forcing-index-1 examples:
    one-angle LW, default solar spectrum normalized to each RFMIP TSI, and
    nighttime columns explicitly zeroed after the SW solve.
    """
    import cupy as cp

    sites = np.arange(100) if sites is None else np.asarray(sites)
    experiments = (np.arange(18) if experiments is None
                   else np.asarray(experiments))
    lw = load_gas_tables("lw")
    (play, plev, tlay, tlev, tsfc, vmr, emiss, _albedo,
     _sza, _tsi) = _rfmip_profiles(lw, sites, experiments)
    dplay = cp.asarray(play, dtype=DTYPE)
    dplev = cp.asarray(plev, dtype=DTYPE)
    dtlay = cp.asarray(tlay, dtype=DTYPE)
    dvmr = cp.asarray(vmr, dtype=DTYPE)
    optics_lw = gas_optics(lw, dplay, dplev, dtlay, dvmr)
    sources = planck_sources(
        lw, dplay, dplev, dtlay, cp.asarray(tlev, dtype=DTYPE),
        cp.asarray(tsfc, dtype=DTYPE), dvmr)
    emis = cp.asarray(emiss, dtype=DTYPE)
    emis_band = cp.broadcast_to(emis[:, None], (play.shape[0], lw.nband))
    emis_gpt = _expand_band_to_gpoint(
        emis_band, lw, "RFMIP surface emissivity")
    lw_flux = lw_rte(optics_lw.tau, sources.lay_source,
                     sources.lev_source, sources.sfc_source, emis_gpt,
                     top_at_1=True)

    sw = load_gas_tables("sw")
    (play, plev, tlay, _tlev, _tsfc, vmr, _emiss, albedo,
     sza, tsi) = _rfmip_profiles(sw, sites, experiments)
    optics_sw = gas_optics(
        sw, cp.asarray(play, dtype=DTYPE), cp.asarray(plev, dtype=DTYPE),
        cp.asarray(tlay, dtype=DTYPE), cp.asarray(vmr, dtype=DTYPE))
    tau, ssa, asym = delta_scale(optics_sw.tau, optics_sw.ssa, optics_sw.g)
    mu_raw = cp.cos(cp.asarray(sza, dtype=DTYPE) * DTYPE(np.pi / 180.0))
    daylight = mu_raw > DTYPE(0.0)
    mu = cp.where(daylight, mu_raw, DTYPE(1.0))
    alb = cp.asarray(albedo, dtype=DTYPE)
    alb_gpt = cp.ascontiguousarray(cp.broadcast_to(
        alb[:, None], (play.shape[0], sw.ngpt)))
    inc = _normalized_solar_incident(sw.solar_source, tsi)
    sw_flux = sw_rte(tau, ssa, asym, mu, alb_gpt, alb_gpt, inc,
                     top_at_1=True)
    mask = daylight[:, None]
    sw_up = cp.where(mask, sw_flux.flux_up, DTYPE(0.0))
    sw_dn = cp.where(mask, sw_flux.flux_dn, DTYPE(0.0))
    return RFMIPResult(lw_flux.flux_up, lw_flux.flux_dn, sw_up, sw_dn)


def _normalized_solar_incident(solar_source, tsi):
    """Normalize a solar spectrum with a binding float64 host reduction."""
    import cupy as cp

    solar_host = np.asarray(solar_source, dtype=np.float64)
    norm = np.sum(solar_host, dtype=np.float64)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("solar-source normalization must be finite and positive")
    scale = np.asarray(tsi, dtype=np.float64) / norm
    solar = cp.asarray(solar_host, dtype=DTYPE)
    scale_device = cp.asarray(scale, dtype=DTYPE)
    return cp.ascontiguousarray(solar[None, :] * scale_device[:, None])


__all__ = ["CloudOpticsResult", "CloudTables", "DATA_DIR", "FluxResult",
           "GasOpticsResult", "GasTables", "HydrometeorPaths",
           "MCICA_PERMUTESEED_LW", "MCICA_PERMUTESEED_SW",
           "PlanckSourceResult", "RFMIPResult", "RRTMGPRadiation",
           "RRTMGP_TOA_PRESSURE_PA",
           "add_cloud_optics", "cal_cldfra1",
           "cloud_optics", "delta_scale", "gas_optics",
           "hydrometeor_paths", "load_cloud_tables", "load_gas_tables",
           "lw_rte", "mcica_cloud_masks", "planck_sources",
           "rfmip_clear_sky", "rrtmgp_above_model_layer_counts",
           "sw_rte", "trace_gases"]
