"""Shared launcher utilities for aerosol-aware Thompson (``mp_physics=28``).

Every ``gpuwm.core.thompson_aerosol_*`` launcher module imports its argument
validation, its grid arithmetic and its CUDA module names from here, so the
six aerosol translation units cannot drift apart on conventions that are
invisible in a diff.

Two things this module deliberately does *not* do:

* It does not import anything private from :mod:`gpuwm.core.thompson`.  That
  module is read-only for this port and its ``_validate_fields`` is a private
  name; :func:`validate_fields` below is a transcription of
  ``thompson.py:50-64``, not an alias, so a future change to the classic
  launcher cannot silently retarget mp=28.
* It does not make ``mp_physics=28`` selectable.  Importing it compiles
  nothing and registers nothing.

It also carries the pointwise probe launchers for
``gpuwm/core/kernels/thompson_aerosol_probe.cu``.  Those exist so the five
aerosol kernel packages can unit-test one ``__device__`` helper at a time
against ``gpuwm/data/thompson/oracle-aero/probe-*.csv`` without standing up a
network kernel first.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE


# ---------------------------------------------------------------------------
# CUDA module names.  These are the six .cu files that receive
# thompson_aerosol_common.cuh from gpuwm/core/kernels/__init__.py's
# _EXTRA_HEADERS allow-list; the tuple below and that dict must agree, which
# tests/test_kernel_loader_inert.py asserts.
# ---------------------------------------------------------------------------

AEROSOL_COMMON_HEADER = "thompson_aerosol_common.cuh"

STATE_MODULE = "thompson_aerosol_state"
SAT_MODULE = "thompson_aerosol_sat"
COLD_MODULE = "thompson_aerosol_cold"
WARM_MODULE = "thompson_aerosol_warm"
SED_MODULE = "thompson_aerosol_sed"
PROBE_MODULE = "thompson_aerosol_probe"

AEROSOL_KERNEL_MODULES = (
    STATE_MODULE, SAT_MODULE, COLD_MODULE, WARM_MODULE, SED_MODULE,
    PROBE_MODULE,
)

#: The mp=8 translation unit.  It is byte-frozen and is never given a header.
CLASSIC_MODULE = "thompson"

#: gpuwm's default 1-D launch width, matching gpuwm/core/thompson.py.
DEFAULT_THREADS = 256

#: Fortran shape of ``tnccn_act`` (module_mp_thompson.F:247-251, :393).
CCN_ACTIVATION_SHAPE = (7, 9, 7, 5, 4)

#: Row order of ``thompson_aa_probe_constant_tables``'s output.  Part of the
#: published probe API; do not reorder.
PROBE_TABLE_ROWS = (
    "cce1", "cce2", "cce3", "cce4", "cce5",
    "ccg1", "ccg2", "ccg3", "ccg4", "ccg5",
    "ocg1", "ocg2", "g_ratio",
)
PROBE_TABLE_COLS = 16

#: Eff_aero collector species selectors; must match
#: THOMPSON_AA_SPECIES_* in thompson_aerosol_common.cuh.
SPECIES_RAIN = 0
SPECIES_SNOW = 1
SPECIES_GRAUPEL = 2
SPECIES_CODES = {"r": SPECIES_RAIN, "s": SPECIES_SNOW, "g": SPECIES_GRAUPEL}


# ---------------------------------------------------------------------------
# Argument validation.
# ---------------------------------------------------------------------------

def validate_fields(fields: dict[str, object]) -> tuple[tuple[int, ...], int]:
    """Require every named field to be one float32 C-contiguous shape.

    Transcribed from ``gpuwm/core/thompson.py:50-64``.  Returns the common
    shape and its element count.
    """
    first = next(iter(fields.values()))
    shape = first.shape
    if not shape:
        raise ValueError("Thompson aerosol fields must be arrays")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape, int(np.prod(shape, dtype=np.int64))


def validate_fp64_fortran_table(name, value, shape) -> None:
    """Require a float64 Fortran-ordered device table of the given shape.

    Transcribed from ``gpuwm/core/thompson.py:64-71``.  mp=28's aerosol table
    contract (:mod:`gpuwm.core.thompson_aerosol_runtime`) uploads float64
    Fortran arrays exactly like the classic one, so the same check applies.
    """
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be float64, got {value.dtype}")
    if not value.flags.f_contiguous:
        raise ValueError(f"{name} must be Fortran-contiguous")


def validate_int_fields(fields: dict[str, object],
                        shape: tuple[int, ...]) -> None:
    """Require every named field to be one int32 C-contiguous ``shape``."""
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != np.dtype(np.int32):
            raise TypeError(f"{name} must be int32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def launch_grid(size: int, threads: int = DEFAULT_THREADS
                ) -> tuple[tuple[int], tuple[int]]:
    """Return ``(grid, block)`` for a flat ``size``-element 1-D kernel."""
    if size <= 0:
        raise ValueError("launch size must be positive")
    blocks = (int(size) + threads - 1) // threads
    return (blocks,), (threads,)


# ---------------------------------------------------------------------------
# Pointwise probes of thompson_aerosol_common.cuh.
# ---------------------------------------------------------------------------

def _empty_like_int(reference):
    import cupy as cp
    return cp.empty(reference.shape, dtype=cp.int32)


def _empty_like_float(reference):
    import cupy as cp
    return cp.empty(reference.shape, dtype=DTYPE)


def _probe(func: str, size: int, args: tuple) -> None:
    grid, block = launch_grid(size)
    get_kernel(PROBE_MODULE, func)(grid, block, args)


def probe_nint(x):
    """Fortran ``NINT`` (round half away from zero) on device."""
    _, size = validate_fields({"x": x})
    out = _empty_like_int(x)
    _probe("thompson_aa_probe_nint", size, (x, out, np.int32(size)))
    return out


def probe_nu_c(nc_m3):
    """``nu_c = MIN(15, NINT(1000.E6/nc) + 2)``, module_mp_thompson.F:2171."""
    _, size = validate_fields({"nc_m3": nc_m3})
    out = _empty_like_int(nc_m3)
    _probe("thompson_aa_probe_nu_c", size, (nc_m3, out, np.int32(size)))
    return out


def probe_droplet_bin(nc_m3):
    """Zero-based ``idx_n`` into ``tnc_wev``, module_mp_thompson.F:3447."""
    _, size = validate_fields({"nc_m3": nc_m3})
    out = _empty_like_int(nc_m3)
    _probe("thompson_aa_probe_droplet_bin", size,
           (nc_m3, out, np.int32(size)))
    return out


def probe_decade_index(value, first_exponent: int, table_size: int):
    """Zero-based base-ten decade/mantissa bin."""
    _, size = validate_fields({"value": value})
    out = _empty_like_int(value)
    _probe("thompson_aa_probe_decade_index", size,
           (value, np.int32(first_exponent), np.int32(table_size), out,
            np.int32(size)))
    return out


def probe_in_bin(xni):
    """Zero-based ``idx_IN``, module_mp_thompson.F:2579-2591."""
    _, size = validate_fields({"xni": xni})
    out = _empty_like_int(xni)
    _probe("thompson_aa_probe_in_bin", size, (xni, out, np.int32(size)))
    return out


def probe_inu_c_effrad(nc_m3):
    """``calc_effectRad``'s three-branch shape selector, :5637-5643."""
    _, size = validate_fields({"nc_m3": nc_m3})
    out = _empty_like_int(nc_m3)
    _probe("thompson_aa_probe_inu_c_effrad", size,
           (nc_m3, out, np.int32(size)))
    return out


def probe_clamps(nc_m3, nwfa_m3, nifa_m3):
    """WRF's terminal aerosol clamps, :1805-1806 and :3979-3981."""
    _, size = validate_fields(
        {"nc_m3": nc_m3, "nwfa_m3": nwfa_m3, "nifa_m3": nifa_m3})
    nc_out = _empty_like_float(nc_m3)
    nwfa_out = _empty_like_float(nwfa_m3)
    nifa_out = _empty_like_float(nifa_m3)
    _probe("thompson_aa_probe_clamps", size,
           (nc_m3, nwfa_m3, nifa_m3, nc_out, nwfa_out, nifa_out,
            np.int32(size)))
    return nc_out, nwfa_out, nifa_out


def probe_saturation(pressure, temperature):
    """``RSLF`` and ``RSIF``, module_mp_thompson.F:5378-5446."""
    _, size = validate_fields(
        {"pressure": pressure, "temperature": temperature})
    rslf = _empty_like_float(pressure)
    rsif = _empty_like_float(pressure)
    _probe("thompson_aa_probe_saturation", size,
           (pressure, temperature, rslf, rsif, np.int32(size)))
    return rslf, rsif


def probe_field_ab(tc, moment):
    """Field et al snow-moment power-law coefficients."""
    _, size = validate_fields({"tc": tc, "moment": moment})
    a_out = _empty_like_float(tc)
    b_out = _empty_like_float(tc)
    _probe("thompson_aa_probe_field_ab", size,
           (tc, moment, a_out, b_out, np.int32(size)))
    return a_out, b_out


def probe_activ_ncloud(temperature, w, nccn, tnccn_act):
    """``activ_ncloud``, module_mp_thompson.F:5178-5253.

    ``tnccn_act`` is the float64 Fortran-ordered ``(7,9,7,5,4)`` device array
    published by :mod:`gpuwm.core.thompson_aerosol_runtime`.
    """
    _, size = validate_fields(
        {"temperature": temperature, "w": w, "nccn": nccn})
    validate_fp64_fortran_table(
        "tnccn_act", tnccn_act, CCN_ACTIVATION_SHAPE)
    out = _empty_like_float(temperature)
    _probe("thompson_aa_probe_activ_ncloud", size,
           (temperature, w, nccn, tnccn_act, out, np.int32(size)))
    return out


def probe_ice_demott(tempc, rho, nifa_m3):
    """``iceDeMott``, module_mp_thompson.F:5448-5518 (pure in tempc/rho/nifa)."""
    _, size = validate_fields(
        {"tempc": tempc, "rho": rho, "nifa_m3": nifa_m3})
    out = _empty_like_float(tempc)
    _probe("thompson_aa_probe_ice_demott", size,
           (tempc, rho, nifa_m3, out, np.int32(size)))
    return out


def probe_ice_koop(temperature, qv, qvs, naero, dt):
    """``iceKoop``, module_mp_thompson.F:5521-5546."""
    _, size = validate_fields({
        "temperature": temperature, "qv": qv, "qvs": qvs, "naero": naero,
        "dt": dt,
    })
    out = _empty_like_float(temperature)
    _probe("thompson_aa_probe_ice_koop", size,
           (temperature, qv, qvs, naero, dt, out, np.int32(size)))
    return out


def probe_eff_aero(d_collector, d_aerosol, visc, rhoa, temperature, species):
    """``Eff_aero``, module_mp_thompson.F:4965-5001.

    ``species`` is an int32 array of :data:`SPECIES_RAIN`,
    :data:`SPECIES_SNOW` or :data:`SPECIES_GRAUPEL`.
    """
    shape, size = validate_fields({
        "d_collector": d_collector, "d_aerosol": d_aerosol, "visc": visc,
        "rhoa": rhoa, "temperature": temperature,
    })
    validate_int_fields({"species": species}, shape)
    out = _empty_like_float(d_collector)
    _probe("thompson_aa_probe_eff_aero", size,
           (d_collector, d_aerosol, visc, rhoa, temperature, species, out,
            np.int32(size)))
    return out


def probe_snow_number(smob, smoc):
    """WRF's explicit two-gamma ``ns(k)``, module_mp_thompson.F:2081-2088."""
    _, size = validate_fields({"smob": smob, "smoc": smoc})
    out = _empty_like_float(smob)
    _probe("thompson_aa_probe_snow_number", size,
           (smob, smoc, out, np.int32(size)))
    return out


def probe_cloud_dist(rc, nc_per_kg, rho):
    """Entry droplet-distribution diagnosis, module_mp_thompson.F:1826-1842.

    Returns ``(nc_m3, nu_c, lamc)``; ``lamc`` is float64 because WRF declares
    it DOUBLE PRECISION.
    """
    import cupy as cp
    _, size = validate_fields(
        {"rc": rc, "nc_per_kg": nc_per_kg, "rho": rho})
    nc_out = _empty_like_float(rc)
    nu_c_out = _empty_like_int(rc)
    lamc_out = cp.empty(rc.shape, dtype=cp.float64)
    _probe("thompson_aa_probe_cloud_dist", size,
           (rc, nc_per_kg, rho, nc_out, nu_c_out, lamc_out, np.int32(size)))
    return nc_out, nu_c_out, lamc_out


def probe_effect_rad(temperature, pressure, qv, qc, nc_per_kg, qi, ni_per_kg,
                     qs):
    """``calc_effectRad``, module_mp_thompson.F:5594-5699.  Returns METRES."""
    _, size = validate_fields({
        "temperature": temperature, "pressure": pressure, "qv": qv, "qc": qc,
        "nc_per_kg": nc_per_kg, "qi": qi, "ni_per_kg": ni_per_kg, "qs": qs,
    })
    effc = _empty_like_float(temperature)
    effi = _empty_like_float(temperature)
    effs = _empty_like_float(temperature)
    _probe("thompson_aa_probe_effect_rad", size,
           (temperature, pressure, qv, qc, nc_per_kg, qi, ni_per_kg, qs,
            effc, effi, effs, np.int32(size)))
    return effc, effi, effs


def probe_constant_tables():
    """Read the ``__constant__`` gamma tables back as a ``(13, 16)`` array."""
    import cupy as cp
    out = cp.empty((len(PROBE_TABLE_ROWS), PROBE_TABLE_COLS), dtype=DTYPE)
    get_kernel(PROBE_MODULE, "thompson_aa_probe_constant_tables")(
        (1,), (PROBE_TABLE_COLS,), (out,))
    return out


__all__ = [
    "AEROSOL_COMMON_HEADER",
    "AEROSOL_KERNEL_MODULES",
    "CCN_ACTIVATION_SHAPE",
    "CLASSIC_MODULE",
    "COLD_MODULE",
    "DEFAULT_THREADS",
    "PROBE_MODULE",
    "PROBE_TABLE_COLS",
    "PROBE_TABLE_ROWS",
    "SAT_MODULE",
    "SED_MODULE",
    "SPECIES_CODES",
    "SPECIES_GRAUPEL",
    "SPECIES_RAIN",
    "SPECIES_SNOW",
    "STATE_MODULE",
    "WARM_MODULE",
    "launch_grid",
    "probe_activ_ncloud",
    "probe_clamps",
    "probe_cloud_dist",
    "probe_constant_tables",
    "probe_decade_index",
    "probe_droplet_bin",
    "probe_eff_aero",
    "probe_effect_rad",
    "probe_field_ab",
    "probe_ice_demott",
    "probe_ice_koop",
    "probe_in_bin",
    "probe_inu_c_effrad",
    "probe_nint",
    "probe_nu_c",
    "probe_saturation",
    "probe_snow_number",
    "validate_fields",
    "validate_fp64_fortran_table",
    "validate_int_fields",
]
