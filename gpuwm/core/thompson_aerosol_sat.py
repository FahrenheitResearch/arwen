"""Launchers for the aerosol-aware Thompson saturation adjustment (mp=28).

This module owns exactly two device entry points, both compiled from
``gpuwm/core/kernels/thompson_aerosol_sat.cu``:

``launch_aerosol_saturation_adjust``
    module_mp_thompson.F:3399-3494 -- condensation, CCN activation via
    ``activ_ncloud``/``tnccn_act``, the aerosol-only droplet-evaporation
    branch via ``tnc_wev``, and the one-for-one CCN return.

``launch_aerosol_rain_evaporation``
    module_mp_thompson.F:3236-3255 + :3384-3388 + :3500-3574 -- a direct
    port, including the mean-volume-diameter clamp ArWen's mp=8 kernel does
    not carry, plus WRF's ``nwfaten += pnr_rev`` and the ``prw_vcd`` gate.

Both are BITWISE against WRF v4.6.1, measured on WRF's own working column
lifted out of an instrumented copy of the pristine module -- see the two
``*_instrumented_wrf_oracle`` tests in
``tests/test_thompson_aerosol_sat_gpu.py``.

Four contract points that are easy to get wrong and are therefore enforced
here rather than documented:

* ``nc`` and ``nwfa`` state are **read-only entry state**.  These launchers
  write only the ``ncten``/``nwfaten`` scratch accumulators and the in-place
  mass/temperature fields.  The single terminal apply with WRF's clamps
  (:3972-4021) belongs to :mod:`gpuwm.core.thompson_aerosol_state`.
* ``nwfa_work_m3`` is WRF's **second** aerosol snapshot (:3211,
  ``MAX(11.1E6, (nwfa1d + nwfaten*DT)*rho)``, no upper bound, no nifa
  counterpart), not the entry snapshot at :1805.  Passing the entry snapshot
  instead changes activated droplet number wherever scavenging mattered.
* ``entry_density`` for rain evaporation is WRF's :3193 density, NOT the
  post-condensation one the same loop uses for everything else.  :3242-3243
  forms ``rr``/``nr`` from it while :3490 has already replaced ``rho(k)`` by
  the time :3505-3520 reads it, so two densities are live at once.  The
  saturation-adjustment launcher writes exactly that array into its
  ``reference_density`` output; pass that buffer through.  Omitting it costs
  up to 2.0e-03 on ``qr`` and ``nr``.
* ``w`` is the **entry** vertical velocity on the lower full level, i.e.
  ``state.w[:-1]``.  ``mp_gt_driver`` copies ``w1d(k) = w(i,k,j)`` once at
  :1224 with no averaging and never refreshes it.  See
  ``tests/test_thompson_aerosol_sat_gpu.py::test_ccn_sweep_settles_the_w_source``
  for the oracle measurement behind that choice.

Importing this module compiles nothing and does not make ``mp_physics=28``
selectable.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.thompson_aerosol_launch import (
    CCN_ACTIVATION_SHAPE,
    SAT_MODULE,
    launch_grid,
    validate_fields,
    validate_fp64_fortran_table,
)


#: Fortran shape of ``tnc_wev`` (thompson_contract.AUXILIARY_TABLE_RECORDS,
#: module_mp_thompson.F:647).  Axis 0 is droplet diameter with
#: ``Dc(i) = i micron`` LINEARLY (:831-836) -- unlike every other bin family
#: in this scheme; axis 1 is cloud water 1e-6..1e-2 kg m-3; axis 2 is the
#: ``t_Nc`` log droplet-number grid.
DROP_EVAP_SHAPE = (100, 37, 100)

SATURATION_ADJUST_KERNEL = "thompson_aa_saturation_adjust"
RAIN_EVAPORATION_KERNEL = "thompson_aa_rain_evaporation"
DROPLET_EVAP_PROBE_KERNEL = "thompson_aa_droplet_evap_probe"
RAIN_EVAP_PROBE_KERNEL = "thompson_aa_rain_evaporation_probe"


def _require_positive_dt(dt: float) -> float:
    value = float(dt)
    if not value > 0.0:
        raise ValueError("dt must be positive")
    return value


def launch_aerosol_saturation_adjust(
        temperature, pressure, qv, qc, nc_entry, ncten, nwfaten,
        nwfa_work_m3, w, tnccn_act, tnc_wev, dt, *,
        reference_density=None, reference_temperature=None,
        condensation_rate=None) -> None:
    """Run WRF's mp=28 cloud condensation/evaporation block on device.

    Parameters
    ----------
    temperature, qv, qc
        Post-network state, updated in place (:3479-3489).
    pressure
        Dry-air pressure, read only.
    nc_entry
        ``nc1d``, the per-kilogram droplet number frozen at call entry.  Read
        only -- the working per-cubic-metre value is rebuilt inside the
        kernel as ``MAX(2, MIN((nc_entry + ncten*dt)*rho, 1999e6))``
        exactly as WRF does at :3216-3223.
    ncten, nwfaten
        Per-kilogram-per-second scratch accumulators, updated in place.
    nwfa_work_m3
        The :3211 snapshot, per cubic metre.  See the module docstring.
    w
        Entry vertical velocity, m/s, on the lower full level.
    tnccn_act
        float64 Fortran-ordered ``(7,9,7,5,4)`` device array from
        :mod:`gpuwm.core.thompson_aerosol_runtime`.
    tnc_wev
        float64 Fortran-ordered ``(100,37,100)`` device array -- record 7 of
        ``thompson_aux_tables.dat``, already resident for every mp=8 launch
        and read here for the first time.
    condensation_rate
        Optional output receiving ``prw_vcd`` (kg/kg/s).  Feed it to
        :func:`launch_aerosol_rain_evaporation` to reproduce WRF's :3502
        gate, which suppresses rain evaporation in a cell that just
        condensed.
    """
    fields = {
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "qc": qc,
        "nc_entry": nc_entry,
        "ncten": ncten,
        "nwfaten": nwfaten,
        "nwfa_work_m3": nwfa_work_m3,
        "w": w,
    }
    if reference_density is not None:
        fields["reference_density"] = reference_density
    if reference_temperature is not None:
        if reference_density is None:
            raise ValueError(
                "reference_temperature requires reference_density")
        fields["reference_temperature"] = reference_temperature
    if condensation_rate is not None:
        fields["condensation_rate"] = condensation_rate
    _, size = validate_fields(fields)
    validate_fp64_fortran_table("tnccn_act", tnccn_act, CCN_ACTIVATION_SHAPE)
    validate_fp64_fortran_table("tnc_wev", tnc_wev, DROP_EVAP_SHAPE)
    step = _require_positive_dt(dt)

    grid, block = launch_grid(size)
    get_kernel(SAT_MODULE, SATURATION_ADJUST_KERNEL)(
        grid, block,
        (temperature, pressure, qv, qc, nc_entry, ncten, nwfaten,
         nwfa_work_m3, w, tnccn_act, tnc_wev,
         reference_density, reference_temperature, condensation_rate,
         np.float32(step), np.int32(size)))


def launch_aerosol_rain_evaporation(
        qr, nr, temperature, pressure, qv, nwfaten, dt, *,
        reference_density=None, reference_temperature=None,
        graupel_melt_marker=None, condensation_rate=None,
        entry_density=None) -> None:
    """Run WRF's mp=28 rain evaporation, returning one CCN per raindrop.

    A direct port of :3236-3255 + :3384-3388 + :3500-3574, including the
    mean-volume-diameter clamp ArWen's mp=8 kernel does not carry.  ``nwfaten``
    may be ``None`` to suppress :3565 (``nwfaten += pnr_rev``), which is what
    makes the aerosol term auditable in isolation.

    Parameters
    ----------
    entry_density
        WRF's :3193 density -- the one diagnosed after the source networks
        and BEFORE the condensation block.  :3242-3243 forms ``rr`` and
        ``nr`` from it while :3505-3520 uses the post-condensation density
        for everything else, and reproducing that mixture is worth up to
        1.9e-03 on ``qr``/``nr`` wherever condensation moved the column.
        The saturation-adjustment kernel writes exactly this array into its
        ``reference_density`` output, unconditionally and at every level, so
        the caller already has it; pass that buffer here.  ``None`` falls
        back to the locally recomputed post-condensation density, i.e. mp=8's
        behaviour.
    """
    fields = {
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    }
    if entry_density is not None:
        fields["entry_density"] = entry_density
    if nwfaten is not None:
        fields["nwfaten"] = nwfaten
    if reference_density is not None:
        fields["reference_density"] = reference_density
    if reference_temperature is not None:
        if reference_density is None:
            raise ValueError(
                "reference_temperature requires reference_density")
        fields["reference_temperature"] = reference_temperature
    if graupel_melt_marker is not None:
        fields["graupel_melt_marker"] = graupel_melt_marker
    if condensation_rate is not None:
        fields["condensation_rate"] = condensation_rate
    _, size = validate_fields(fields)
    step = _require_positive_dt(dt)

    grid, block = launch_grid(size)
    get_kernel(SAT_MODULE, RAIN_EVAPORATION_KERNEL)(
        grid, block,
        (qr, nr, temperature, pressure, qv, nwfaten,
         reference_density, reference_temperature, graupel_melt_marker,
         condensation_rate, entry_density,
         np.float32(step), np.int32(size)))


def probe_droplet_evaporation_indices(
        temperature, pressure, qv, qc, nc_work_m3, tnc_wev, dt):
    """Return the ``(idx_d, idx_c, idx_n)`` triple the kernel would read.

    A diagnostic, not a physics path.  ``tnc_wev`` has been parsed,
    SHA-validated and uploaded since the mp=8 port and has never been read by
    any kernel, so an order defect in the ``(100,37,100)`` upload would first
    appear as an mp=28 physics error.  This probe lets a test attribute it to
    the table instead.

    Returns ``(idx_d, idx_c, idx_n, tnc, pnc_wcd)``; the three indices are
    ONE-BASED int32 arrays matching WRF's own subscripts, ``tnc`` is the
    float64 table value read, and ``pnc_wcd`` is the resulting rate.
    """
    import cupy as cp

    shape, size = validate_fields({
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "qc": qc,
        "nc_work_m3": nc_work_m3,
    })
    validate_fp64_fortran_table("tnc_wev", tnc_wev, DROP_EVAP_SHAPE)
    step = _require_positive_dt(dt)

    idx_d = cp.empty(shape, dtype=cp.int32)
    idx_c = cp.empty(shape, dtype=cp.int32)
    idx_n = cp.empty(shape, dtype=cp.int32)
    tnc = cp.empty(shape, dtype=cp.float64)
    pnc = cp.empty(shape, dtype=cp.float64)

    grid, block = launch_grid(size)
    get_kernel(SAT_MODULE, DROPLET_EVAP_PROBE_KERNEL)(
        grid, block,
        (temperature, pressure, qv, qc, nc_work_m3, tnc_wev,
         idx_d, idx_c, idx_n, tnc, pnc,
         np.float32(step), np.int32(size)))
    return idx_d, idx_c, idx_n, tnc, pnc


def probe_rain_evaporation_rates(
        qr, nr, temperature, pressure, qv, dt, *, graupel_melt_marker=None,
        entry_density=None):
    """Return ``(prv_rev, pnr_rev, nr_bound)`` without touching caller state.

    A diagnostic, not a physics path.  ``prv_rev`` (:3540) and ``pnr_rev``
    (:3559) are the two rates WRF's rain-evaporation loop actually produces,
    and ``nr_bound`` is the working per-cubic-metre rain number AFTER the
    mean-volume-diameter clamp at :3247-3255.  Pinning all three against an
    instrumented copy of the pristine Fortran is what makes the clamp -- which
    ArWen's mp=8 kernel does not carry -- auditable in isolation instead of
    being visible only as a shifted ``nr`` three launchers later.

    The four mutable state fields are copied first, so this call is
    observationally inert.
    """
    import cupy as cp

    shape, size = validate_fields({
        "qr": qr,
        "nr": nr,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
    })
    if graupel_melt_marker is not None:
        validate_fields({"pressure": pressure,
                         "graupel_melt_marker": graupel_melt_marker})
    if entry_density is not None:
        validate_fields({"pressure": pressure,
                         "entry_density": entry_density})
    step = _require_positive_dt(dt)

    scratch = [cp.array(field, copy=True)
               for field in (qr, nr, temperature, qv)]
    prv = cp.empty(shape, dtype=cp.float64)
    pnr = cp.empty(shape, dtype=cp.float64)
    bound = cp.empty(shape, dtype=cp.float32)

    grid, block = launch_grid(size)
    get_kernel(SAT_MODULE, RAIN_EVAP_PROBE_KERNEL)(
        grid, block,
        (scratch[0], scratch[1], scratch[2], pressure, scratch[3],
         graupel_melt_marker, entry_density, prv, pnr, bound,
         np.float32(step), np.int32(size)))
    return prv, pnr, bound


__all__ = [
    "DROPLET_EVAP_PROBE_KERNEL",
    "DROP_EVAP_SHAPE",
    "RAIN_EVAPORATION_KERNEL",
    "RAIN_EVAP_PROBE_KERNEL",
    "SATURATION_ADJUST_KERNEL",
    "launch_aerosol_rain_evaporation",
    "launch_aerosol_saturation_adjust",
    "probe_droplet_evaporation_indices",
    "probe_rain_evaporation_rates",
]
