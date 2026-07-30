"""WRF-real-like moist initialization on an explicit hybrid eta grid.

Setup and hydrostatic recurrences are float64.  The pressure-column vertical
interpolations follow WRF real's ratified defaults (interp_theta=F,
lagrange_order=2, use_surface=T with force_sfc_in_vinterp=1 and
zap_close_levels=500) through the common CUDA/parallel-CPU preprocessing
contract, and the completed prognostic state is FP32 on device.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.grid import (BaseState, VerticalCoord,
                             finalize_vertical_coord)
from gpuwm.core.state import DomainState
from gpuwm.ingest.horiz import (
    HorizontalSnapshot,
    source_orography_from_catalog as _source_orography_from_catalog,
)
from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend


def _column_worker_count(value) -> int:
    """Validate an explicit setup-only CPU column-worker count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("column_workers must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError("column_workers must be positive")
    return value


def _axis0_chunks(length: int, workers: int):
    """Return non-empty, deterministic contiguous chunks of axis zero."""
    workers = min(int(workers), int(length))
    q, r = divmod(int(length), workers)
    start = 0
    chunks = []
    for index in range(workers):
        stop = start + q + (index < r)
        chunks.append((start, stop))
        start = stop
    return tuple(chunks)


def _ordered_levels(array, order):
    """Apply a vertical order without copying monotonic source inventories."""
    array = np.asarray(array)
    order = np.asarray(order)
    identity = np.arange(array.shape[0], dtype=order.dtype)
    if np.array_equal(order, identity):
        return array
    if np.array_equal(order, identity[::-1]):
        return array[::-1]
    return array[order]


def _fill_axis0_chunk(output, start, stop, operation, args, kwargs=None):
    """Evaluate and immediately store one disjoint threaded output slab."""
    output[start:stop] = operation(*args, **({} if kwargs is None else kwargs))


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def _saturation_mixing_ratio_serial(temperature, pressure,
                                    relative_humidity):
    """Evaluate one contiguous WRF/Bolton humidity chunk."""
    temperature, pressure, rh = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.clip(np.asarray(relative_humidity, dtype=np.float64), 0.0, 100.0),
    )
    # SVP1 is kPa in module_model_constants; convert to hPa to pair with p/100.
    # rh_to_mxrat1 uses its own local EPS = 0.622, NOT module ep_2 = 0.62175
    # (module_initialize_real.F:7379, q = MAX(eps*es/(p/100.-es), 1.E-6)).
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        es_hpa = (rh * 0.01) * (10.0 * c.SVP1) * np.exp(
            c.SVP2 * (temperature - c.SVPT0) / (temperature - c.SVP3))
        candidate = 0.622 * es_hpa / (pressure / 100.0 - es_hpa)
    valid = ((temperature != 0.0) & np.isfinite(es_hpa)
             & (es_hpa < pressure / 100.0))
    return np.where(valid, np.maximum(candidate, 1.0e-6), 1.0e-6)


def _saturation_mixing_ratio(temperature, pressure, relative_humidity=100.0,
                             *, column_workers=1):
    """WRF/Bolton liquid-water saturation mixing ratio (kg kg-1).

    Large setup arrays are divided into deterministic contiguous chunks.
    NumPy performs the unchanged float64 expression in independent threads,
    preserving every element's arithmetic while using otherwise-idle host
    cores during native initial-condition preparation.
    """
    workers = _column_worker_count(column_workers)
    temperature, pressure, rh = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.clip(np.asarray(relative_humidity, dtype=np.float64), 0.0, 100.0),
    )
    if workers == 1 or temperature.ndim == 0 or temperature.shape[0] < 2:
        return _saturation_mixing_ratio_serial(temperature, pressure, rh)

    chunks = _axis0_chunks(temperature.shape[0], workers)
    output = np.empty(temperature.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _saturation_mixing_ratio_serial,
                (temperature[start:stop], pressure[start:stop],
                 rh[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _cap_stratospheric_qv_serial(qv, pressure):
    """Evaluate one contiguous stratospheric-cap chunk."""
    qv = np.asarray(qv, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    return np.where((pressure < 10000.0) & (qv > 1.0e-5), 3.0e-6, qv)


def _cap_stratospheric_qv(qv, pressure, *, column_workers=1):
    """WRF ``rh_to_mxrat1`` stratospheric qv sanity cap.

    module_initialize_real.F:7490-7498 with the Registry defaults
    (Registry.EM_COMMON:2306-2308): where ``p < qv_max_p_safe`` (10000 Pa,
    strict) and ``qv > qv_max_flag`` (1e-5, strict), force ``qv_max_value``
    (3e-6).  The companion qv_min cap (:7499-7503, p < 110000 Pa, qv < 1e-6
    -> 1e-6) is already realized by the unconditional 1e-6 floor in
    :func:`_saturation_mixing_ratio` for this domain's pressure range.
    """
    workers = _column_worker_count(column_workers)
    qv, pressure = np.broadcast_arrays(
        np.asarray(qv, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if workers == 1 or qv.ndim == 0 or qv.shape[0] < 2:
        return _cap_stratospheric_qv_serial(qv, pressure)

    chunks = _axis0_chunks(qv.shape[0], workers)
    output = np.empty(qv.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _cap_stratospheric_qv_serial,
                (qv[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


_WPS_SPFH_UNDERSHOOT_LOWER_BOUND = -0.028126


def _specific_humidity_to_mixing_ratio_serial(
        specific_humidity, *, allow_wps_undershoot=False):
    """Convert and validate one contiguous specific-humidity chunk."""
    specific = np.asarray(specific_humidity, dtype=np.float64)
    lower = (_WPS_SPFH_UNDERSHOOT_LOWER_BOUND
             if allow_wps_undershoot else 0.0)
    if (not np.isfinite(specific).all() or np.any(specific < lower)
            or np.any(specific >= 1.0)):
        interval = (f"[{_WPS_SPFH_UNDERSHOOT_LOWER_BOUND}, 1)"
                    if allow_wps_undershoot else "[0, 1)")
        raise ValueError(f"specific humidity must be finite in {interval}")
    return specific / (1.0 - specific)


def _specific_humidity_to_mixing_ratio(
        specific_humidity, *, allow_wps_undershoot=False,
        column_workers=1):
    """Convert HRRR/WPS specific humidity to dry-air mixing ratio.

    This is WRF ``module_initialize_real.F``'s ``flag_sh`` branch:
    ``qv_gc = sh_gc / (1 - sh_gc)``.  Unlike the RH path, WRF does not
    apply the ``rh_to_mxrat1`` floor or stratospheric cap when
    ``use_sh_qv`` is active.  WPS's overlapping-parabolic horizontal
    interpolation can create negative undershoots from an everywhere
    non-negative source field.  ``real.exe`` retains those values in
    ``qv_gc`` for ``integ_moist`` even when ``use_sh_qv = .false.``.  HRRR
    SPFH is gated to 0..0.1 upstream; the 2-D sixteen-point operator's most
    negative coefficient sum is -9/32, so -0.028125 is its exact lower
    envelope.  One extra micro-unit covers FP32 evaluation rounding without
    weakening the explicit direct-qv lane's physical-range check.
    """
    if not isinstance(allow_wps_undershoot, (bool, np.bool_)):
        raise TypeError("allow_wps_undershoot must be boolean")
    workers = _column_worker_count(column_workers)
    specific = np.asarray(specific_humidity, dtype=np.float64)
    if (workers == 1 or specific.ndim == 0 or specific.shape[0] < 2):
        return _specific_humidity_to_mixing_ratio_serial(
            specific, allow_wps_undershoot=allow_wps_undershoot)

    chunks = _axis0_chunks(specific.shape[0], workers)
    output = np.empty(specific.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _specific_humidity_to_mixing_ratio_serial,
                (specific[start:stop],),
                {"allow_wps_undershoot": allow_wps_undershoot})
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _mixing_ratio_to_relative_humidity_serial(
        temperature, pressure, mixing_ratio, *, allow_wps_undershoot=False):
    """Diagnose and validate one contiguous relative-humidity chunk."""
    temperature, pressure, qv = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.asarray(mixing_ratio, dtype=np.float64),
    )
    minimum_qv = (_WPS_SPFH_UNDERSHOOT_LOWER_BOUND
                  / (1.0 - _WPS_SPFH_UNDERSHOOT_LOWER_BOUND)
                  if allow_wps_undershoot else 0.0)
    if (not np.isfinite(temperature).all()
            or not np.isfinite(pressure).all()
            or not np.isfinite(qv).all()
            or np.any(pressure <= 0.0) or np.any(qv < minimum_qv)):
        qv_requirement = ("inside the bounded WPS undershoot envelope"
                          if allow_wps_undershoot else "non-negative")
        raise ValueError(
            "temperature, pressure, and mixing ratio must be finite; "
            f"pressure must be positive and mixing ratio {qv_requirement}")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        es_hpa = (10.0 * c.SVP1) * np.exp(
            c.SVP2 * (temperature - c.SVPT0) / (temperature - c.SVP3))
        vapor_hpa = qv * (pressure / 100.0) / (qv + 0.622)
        rh = 100.0 * vapor_hpa / es_hpa
    if not np.isfinite(rh).all():
        raise ValueError("diagnosed relative humidity is non-finite")
    return rh


def _mixing_ratio_to_relative_humidity(
        temperature, pressure, mixing_ratio, *, allow_wps_undershoot=False,
        column_workers=1):
    """Diagnose WPS/WRF relative humidity (%) from dry-air mixing ratio.

    HRRR supplies specific humidity.  WPS horizontally maps SPECHUMD into its
    intermediate output; real.exe's FLAG_SH branch converts it to qv_gc
    and overwrites rh_gc before vertical interpolation.  This is the algebraic
    inverse of its later ``rh_to_mxrat1`` Bolton saturation relation, before
    that routine's 0--100 percent RH clipping.
    """
    if not isinstance(allow_wps_undershoot, (bool, np.bool_)):
        raise TypeError("allow_wps_undershoot must be boolean")
    workers = _column_worker_count(column_workers)
    temperature, pressure, qv = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
        np.asarray(mixing_ratio, dtype=np.float64),
    )
    if (workers == 1 or temperature.ndim == 0
            or temperature.shape[0] < 2):
        return _mixing_ratio_to_relative_humidity_serial(
            temperature, pressure, qv,
            allow_wps_undershoot=allow_wps_undershoot)

    chunks = _axis0_chunks(temperature.shape[0], workers)
    output = np.empty(temperature.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _mixing_ratio_to_relative_humidity_serial,
                (temperature[start:stop], pressure[start:stop],
                 qv[start:stop]),
                {"allow_wps_undershoot": allow_wps_undershoot})
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _potential_temperature_from_temperature_serial(temperature, pressure):
    """Evaluate WRF's T-to-theta relation over one contiguous chunk."""
    return temperature * (c.P0 / pressure) ** c.RCP


def _potential_temperature_from_temperature(
        temperature, pressure, *, column_workers=1):
    """Column-parallel, byte-stable WRF T-to-theta conversion."""
    workers = _column_worker_count(column_workers)
    temperature, pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if (workers == 1 or temperature.ndim == 0
            or temperature.shape[0] < 2):
        return _potential_temperature_from_temperature_serial(
            temperature, pressure)
    chunks = _axis0_chunks(temperature.shape[0], workers)
    output = np.empty(temperature.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _potential_temperature_from_temperature_serial,
                (temperature[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _temperature_from_potential_temperature_serial(theta, pressure):
    """Evaluate WRF's theta-to-T relation over one contiguous chunk."""
    return theta * (pressure / c.P0) ** c.RCP


def _temperature_from_potential_temperature(
        theta, pressure, *, column_workers=1):
    """Column-parallel, byte-stable WRF theta-to-T conversion."""
    workers = _column_worker_count(column_workers)
    theta, pressure = np.broadcast_arrays(
        np.asarray(theta, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if workers == 1 or theta.ndim == 0 or theta.shape[0] < 2:
        return _temperature_from_potential_temperature_serial(theta, pressure)
    chunks = _axis0_chunks(theta.shape[0], workers)
    output = np.empty(theta.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _temperature_from_potential_temperature_serial,
                (theta[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _moist_specific_volume_serial(theta, qv, pressure):
    """Evaluate one contiguous chunk of WRF moist specific volume."""
    theta_m = theta * (1.0 + c.RVOVRD * qv)
    return c.RD * theta_m * (pressure / c.P0) ** c.RCP / pressure


def _moist_specific_volume(theta, qv, pressure, *, column_workers=1):
    """Column-parallel, byte-stable moist specific-volume diagnostic."""
    workers = _column_worker_count(column_workers)
    theta, qv, pressure = np.broadcast_arrays(
        np.asarray(theta, dtype=np.float64),
        np.asarray(qv, dtype=np.float64),
        np.asarray(pressure, dtype=np.float64),
    )
    if workers == 1 or theta.ndim == 0 or theta.shape[0] < 2:
        return _moist_specific_volume_serial(theta, qv, pressure)
    chunks = _axis0_chunks(theta.shape[0], workers)
    output = np.empty(theta.shape, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fill_axis0_chunk, output, start, stop,
                _moist_specific_volume_serial,
                (theta[start:stop], qv[start:stop], pressure[start:stop]))
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _wrf_flag_sh_surface_specific_humidity(
        q2, spfh, pressure, *, force_fallback=None):
    """Apply real.exe's whole-domain FLAG_SH surface fallback.

    WPS's surface SPECHUMD occupies metgrid level one.  WRF checks its first
    valid horizontal point; when that value is below ``1e-6``, it replaces the
    complete surface field with the nearest atmospheric SPECHUMD level before
    converting specific humidity to mixing ratio.  Native HRRR stores that
    surface value separately as Q2, so reproduce the same decision explicitly.
    """
    q2 = np.asarray(q2, dtype=np.float64)
    spfh = np.asarray(spfh, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    if spfh.ndim != 3 or pressure.shape != spfh.shape:
        raise ValueError("SPFH and pressure must share shape (level, y, x)")
    if q2.shape != spfh.shape[1:]:
        raise ValueError("Q2 must match the SPFH horizontal grid")
    if force_fallback is not None and not isinstance(
            force_fallback, (bool, np.bool_)):
        raise TypeError("force_fallback must be boolean or None")
    fallback = (q2[0, 0] < 1.0e-6
                if force_fallback is None else bool(force_fallback))
    if not fallback:
        return q2
    nearest = 0 if pressure[-1, 0, 0] < pressure[0, 0, 0] else -1
    return spfh[nearest].copy()


def _surface_relative_humidity(dewpoint, temperature):
    """ungrib's 2 m relative humidity from D2/T2 (surface RH level 1).

    WPS v4.6 ``ungrib/src/rrpr.F:compute_rh_dewpt`` (:1168-1185):
    ``RH2m = 100 * exp((Xlv/Rv) * (1/T2 - 1/D2))`` with ``Xlv = 2.5e6`` and
    ``Rv = 461.5`` -- a constant-latent-heat Clausius-Clapeyron ratio, not
    the Bolton/Magnus curve used elsewhere in this module.
    The value is deliberately NOT clipped to 100: WRF interpolates ``rh_gc``
    as delivered and clips only inside ``rh_to_mxrat1``
    (module_initialize_real.F:7392-7399), which
    :func:`_saturation_mixing_ratio` mirrors.
    """
    dewpoint = _host(dewpoint)
    temperature = _host(temperature)
    xlv_over_rv = 2.5e6 / 461.5
    return 100.0 * np.exp(
        xlv_over_rv * (1.0 / temperature - 1.0 / dewpoint))


def surface_pressure_from_surface(psfc_in, source_orography, terrain,
                                  surface_temperature, surface_qv):
    """WRF ``sfcprs2`` surface-to-surface pressure adjustment.

    ``source_orography`` is the invariant geopotential-height surface that
    belongs to ``psfc_in`` (the retained profile's declared ``SOILHGT``
    artifact); ``terrain`` is the target WRF terrain.
    """
    psfc = _host(psfc_in)
    source_z = _host(source_orography)
    target_z = _host(terrain)
    temperature = _host(surface_temperature)
    qv = _host(surface_qv)
    if len({a.shape for a in (psfc, source_z, target_z, temperature, qv)}) != 1:
        raise ValueError("surface-pressure input shapes differ")
    # Equal source and target terrain is WRF's DEFINED behavior: sfcprs2
    # returns psfc_in unchanged (exp(0) = 1, module_initialize_real.F:8496).
    # Generic code performs that no-op; protecting a specific case against
    # a silently regressed source-orography artifact (e.g. a SOILHGT
    # regenerated byte-equal to HGT_M) is the job of that case's pinned
    # wrfinput gates, which fail by orders of magnitude on such data.
    virtual_temperature = temperature * (1.0 + 0.608 * qv)
    if (not np.isfinite(virtual_temperature).all()
            or np.any(virtual_temperature <= 0.0)):
        raise ValueError("surface virtual temperature must be finite and positive")
    result = psfc * np.exp(
        c.G * (source_z - target_z) / (c.RD * virtual_temperature))
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("adjusted surface pressure is invalid")
    return result


def _integrate_moisture_scalar_reference(
        qv, pressure, temperature, height, psfc, tsfc, qsfc,
        surface_height):
    """Pre-vectorization row/column oracle retained for byte-parity tests."""
    """Float64 transcription of ``module_initialize_real.F:integ_moist``.

    Inputs are pressure levels only, normalized here to bottom-to-top order;
    the separate surface values play the WPS surface pseudo-level role.
    Returns bottom-to-top dry pressure and the column-integrated vapor
    pressure removed from surface pressure.
    """
    order = np.argsort(-pressure[:, 0, 0])
    p = pressure[order].copy()
    q = qv[order].copy()
    t = temperature[order].copy()
    z = height[order].copy()
    nlev, ny, nx = p.shape
    pd = np.empty_like(p)
    intq = np.zeros((ny, nx), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            above = np.flatnonzero(p[:, j, i] < psfc[j, i])
            if above.size == 0:
                raise ValueError(
                    f"no pressure level above the surface at column ({j}, {i})")
            ka = int(above[0])
            cumulative = np.zeros(nlev, dtype=np.float64)
            pd[-1, j, i] = p[-1, j, i]
            for k in range(nlev - 2, ka - 1, -1):
                rhobar = 0.5 * (p[k, j, i] / (c.RD * t[k, j, i])
                                + p[k + 1, j, i] / (c.RD * t[k + 1, j, i]))
                qbar = 0.5 * (q[k, j, i] + q[k + 1, j, i])
                dz = z[k + 1, j, i] - z[k, j, i]
                if dz > 0.0:
                    cumulative[k] = (cumulative[k + 1]
                                     + c.G * qbar * rhobar / (1.0 + qbar) * dz)
                else:
                    cumulative[k] = cumulative[k + 1]
                pd[k, j, i] = p[k, j, i] - cumulative[k]
            rhobar = 0.5 * (psfc[j, i] / (c.RD * tsfc[j, i])
                            + p[ka, j, i] / (c.RD * t[ka, j, i]))
            qbar = 0.5 * (qsfc[j, i] + q[ka, j, i])
            dz = z[ka, j, i] - surface_height[j, i]
            surface_intq = cumulative[ka]
            if dz > 0.1:
                surface_intq += c.G * qbar * rhobar / (1.0 + qbar) * dz
            intq[j, i] = surface_intq
            pd[:ka, j, i] = p[:ka, j, i] - surface_intq
            # ka was assigned in the loop unless it is the topmost level.
            pd[ka:, j, i] = p[ka:, j, i] - cumulative[ka:]
    return pd, intq, order


def _integrate_moisture_vectorized_slab(
        qv, pressure, temperature, height, psfc, tsfc, qsfc,
        surface_height, *, order, out_pd, out_intq, row_offset=0):
    """Evaluate one contiguous row slab of WRF ``integ_moist``.

    Vertical recurrence order remains top-down within every column, while
    NumPy evaluates all independent horizontal columns in one native loop.
    ``order`` is resolved once from the complete domain so worker partitioning
    cannot change the source-level ordering contract.
    """
    nlev, ny, nx = pressure.shape
    ka = np.full((ny, nx), nlev, dtype=np.intp)
    for k in range(nlev):
        source_k = int(order[k])
        take = (ka == nlev) & (pressure[source_k] < psfc)
        ka[take] = k
    missing = ka == nlev
    if bool(np.any(missing)):
        j, i = np.argwhere(missing)[0]
        raise ValueError(
            "no pressure level above the surface at column "
            f"({int(j) + int(row_offset)}, {i})")

    running = np.zeros_like(psfc, dtype=pressure.dtype)
    surface_intq = np.zeros_like(psfc, dtype=pressure.dtype)
    out_pd[-1] = pressure[int(order[-1])]
    for k in range(nlev - 2, -1, -1):
        source_k = int(order[k])
        source_kp1 = int(order[k + 1])
        pk = pressure[source_k]
        pkp1 = pressure[source_kp1]
        qk = qv[source_k]
        qkp1 = qv[source_kp1]
        tk = temperature[source_k]
        tkp1 = temperature[source_kp1]
        zk = height[source_k]
        zkp1 = height[source_kp1]
        active = ka <= k
        rhobar = 0.5 * (
            pk / (c.RD * tk) + pkp1 / (c.RD * tkp1))
        qbar = 0.5 * (qk + qkp1)
        dz = zkp1 - zk
        increment = c.G * qbar * rhobar / (1.0 + qbar) * dz
        running = np.where(
            active,
            running + np.where(dz > 0.0, increment, 0.0),
            0.0)
        out_pd[k] = pk - running
        surface_intq = np.where(ka == k, running, surface_intq)

    p_ka = np.empty_like(psfc, dtype=pressure.dtype)
    q_ka = np.empty_like(psfc, dtype=qv.dtype)
    t_ka = np.empty_like(psfc, dtype=temperature.dtype)
    z_ka = np.empty_like(psfc, dtype=height.dtype)
    for k in range(nlev):
        selected = ka == k
        source_k = int(order[k])
        p_ka[selected] = pressure[source_k][selected]
        q_ka[selected] = qv[source_k][selected]
        t_ka[selected] = temperature[source_k][selected]
        z_ka[selected] = height[source_k][selected]
    rhobar = 0.5 * (
        psfc / (c.RD * tsfc) + p_ka / (c.RD * t_ka))
    qbar = 0.5 * (qsfc + q_ka)
    dz = z_ka - surface_height
    surface_increment = c.G * qbar * rhobar / (1.0 + qbar) * dz
    surface_intq = np.where(
        dz > 0.1, surface_intq + surface_increment, surface_intq)
    for k in range(nlev):
        below = k < ka
        if bool(np.any(below)):
            source_k = int(order[k])
            out_pd[k, below] = (
                pressure[source_k, below] - surface_intq[below])
    out_intq[...] = surface_intq


def _integrate_moisture(qv, pressure, temperature, height, psfc, tsfc, qsfc,
                        surface_height, *, column_workers=1):
    """Column-parallel transcription of WRF ``integ_moist``.

    The original Python row/column loop is vectorized within each row slab.
    Large domains may additionally divide those independent slabs among an
    explicit number of setup-only host threads.  Every column keeps the same
    vertical operation order, and source-level ordering is resolved once from
    the complete domain, so the threaded result is byte-identical to the
    single-thread vectorized implementation.
    """
    workers = _column_worker_count(column_workers)
    pressure = np.asarray(pressure)
    if pressure.ndim != 3:
        raise ValueError("pressure must have shape (level, y, x)")
    order = np.argsort(-pressure[:, 0, 0])
    pd = np.empty(pressure.shape, dtype=pressure.dtype)
    intq = np.empty(pressure.shape[1:], dtype=pressure.dtype)
    if workers == 1 or pressure.shape[1] < 2:
        _integrate_moisture_vectorized_slab(
            qv, pressure, temperature, height, psfc, tsfc, qsfc,
            surface_height, order=order, out_pd=pd, out_intq=intq)
        return pd, intq, order

    chunks = _axis0_chunks(pressure.shape[1], workers)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _integrate_moisture_vectorized_slab,
                qv[:, start:stop], pressure[:, start:stop],
                temperature[:, start:stop], height[:, start:stop],
                psfc[start:stop], tsfc[start:stop], qsfc[start:stop],
                surface_height[start:stop], order=order,
                out_pd=pd[:, start:stop], out_intq=intq[start:stop],
                row_offset=start)
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return pd, intq, order


def _make_real_base_serial(coord: VerticalCoord, terrain: np.ndarray,
                           p_top: float, base_temp: float,
                           hypsometric_opt: int = 1) -> BaseState:
    """WRF ``module_initialize_real.F`` analytic hydrostatic base state.

    The base geopotential integration is keyed on ``hypsometric_opt``
    exactly as WRF (module_initialize_real.F:3811-3825): opt 1 is the
    discrete ``phb(k+1) = phb(k) - dnw(k)*(c1h*mub+c2h)*alb(k)``
    recurrence; opt 2 the log-pressure form ``phb(k+1) = phb(k) +
    alb(k)*phm*LOG(pfd/pfu)`` on the base-state reference dry pressures
    ``pf/ph = c3*MUB + c4 + p_top`` (F:3816-3822).
    """
    finalize_vertical_coord(coord, p_top)
    terrain = np.array(terrain, dtype=np.float64, copy=True)
    lapse = 50.0
    iso_temperature = 200.0
    root = (base_temp / lapse) ** 2 - (
        2.0 * c.G * terrain / (lapse * c.RD))
    if np.any(root <= 0.0):
        raise ValueError("terrain is outside the analytic base-state range")
    ps_base = c.P0 * np.exp(-base_temp / lapse + np.sqrt(root))
    mub = ps_base - p_top
    pb = (coord.c3h[:, None, None] * mub[None]
          + coord.c4h[:, None, None] + p_top)
    if np.any(pb <= 0.0) or not np.all(np.diff(pb, axis=0) < 0.0):
        raise ValueError("hybrid base pressure is not monotonic")
    temperature = np.maximum(
        iso_temperature, base_temp + lapse * np.log(pb / c.P0))
    thb = temperature * (c.P0 / pb) ** c.RCP
    alb = c.RD * thb * (pb / c.P0) ** c.RCP / pb
    phb = np.empty((coord.znw.size,) + terrain.shape, dtype=np.float64)
    phb[0] = c.G * terrain
    if hypsometric_opt == 1:
        for k in range(coord.dnw.size):
            phb[k + 1] = (phb[k] - coord.dnw[k]
                          * (coord.c1h[k] * mub + coord.c2h[k]) * alb[k])
    elif hypsometric_opt == 2:
        # module_initialize_real.F:3816-3822 (indices shifted to 0-based:
        # WRF's k/k-1 full levels and k-1 half level become k+1/k and k).
        for k in range(coord.dnw.size):
            pfu = coord.c3f[k + 1] * mub + coord.c4f[k + 1] + p_top
            pfd = coord.c3f[k] * mub + coord.c4f[k] + p_top
            phm = coord.c3h[k] * mub + coord.c4h[k] + p_top
            phb[k + 1] = phb[k] + alb[k] * phm * np.log(pfd / pfu)
    else:
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    return BaseState(mub=mub, p_top=float(p_top), pb=pb, alb=alb, thb=thb,
                     phb=phb, terrain_z=terrain)


def _fill_real_base_rows(output, coord, terrain, start, stop, p_top,
                         base_temp, hypsometric_opt):
    """Build then immediately copy one independent analytic-base row tile."""
    part = _make_real_base_serial(
        coord, terrain[start:stop], p_top, base_temp, hypsometric_opt)
    output.mub[start:stop] = part.mub
    output.pb[:, start:stop] = part.pb
    output.alb[:, start:stop] = part.alb
    output.thb[:, start:stop] = part.thb
    output.phb[:, start:stop] = part.phb
    output.terrain_z[start:stop] = part.terrain_z


def _make_real_base(coord: VerticalCoord, terrain: np.ndarray, p_top: float,
                    base_temp: float, hypsometric_opt: int = 1, *,
                    column_workers=1) -> BaseState:
    """Build the exact analytic base over independent horizontal chunks."""
    workers = _column_worker_count(column_workers)
    finalize_vertical_coord(coord, p_top)
    terrain = np.asarray(terrain, dtype=np.float64)
    if workers == 1 or terrain.shape[0] < 2:
        return _make_real_base_serial(
            coord, terrain, p_top, base_temp, hypsometric_opt)

    ny, nx = terrain.shape
    nz = coord.dnw.size
    output = BaseState(
        mub=np.empty((ny, nx), dtype=np.float64),
        p_top=float(p_top),
        pb=np.empty((nz, ny, nx), dtype=np.float64),
        alb=np.empty((nz, ny, nx), dtype=np.float64),
        thb=np.empty((nz, ny, nx), dtype=np.float64),
        phb=np.empty((nz + 1, ny, nx), dtype=np.float64),
        terrain_z=np.empty((ny, nx), dtype=np.float64),
    )
    # More tiles than workers bounds live temporary BaseStates and avoids a
    # serial multi-gigabyte concatenate on large domains.
    chunks = _axis0_chunks(ny, min(ny, workers * 4))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _fill_real_base_rows, output, coord, terrain, start, stop,
                p_top, base_temp, hypsometric_opt)
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return output


def _pressure_at_u(pressure):
    out = np.empty(pressure.shape[:2] + (pressure.shape[2] + 1,),
                   dtype=np.float64)
    out[..., 0] = pressure[..., 0]
    out[..., -1] = pressure[..., -1]
    out[..., 1:-1] = 0.5 * (pressure[..., :-1] + pressure[..., 1:])
    return out


def _pressure_at_v(pressure):
    out = np.empty((pressure.shape[0], pressure.shape[1] + 1,
                    pressure.shape[2]), dtype=np.float64)
    out[:, 0, :] = pressure[:, 0, :]
    out[:, -1, :] = pressure[:, -1, :]
    out[:, 1:-1, :] = 0.5 * (pressure[:, :-1, :] + pressure[:, 1:, :])
    return out


def _slice_base_rows(base: BaseState, start: int, stop: int) -> BaseState:
    """View one contiguous mass-grid row slab of an analytic base state."""
    return BaseState(
        mub=base.mub[start:stop], p_top=base.p_top,
        pb=base.pb[:, start:stop], alb=base.alb[:, start:stop],
        thb=base.thb[:, start:stop], phb=base.phb[:, start:stop],
        terrain_z=base.terrain_z[start:stop])


def _rebalance_moist_pressure_serial(pressure_guess, qv, dry_mass, base,
                                     coord, *, out=None):
    """Integrate the discrete WRF moist w-balance pressure recurrence.

    This is the initialization counterpart of ``pg_buoy_w``: it chooses
    perturbation-pressure differences so the large-step vertical pressure
    gradient, dry-mass perturbation, and vapor loading cancel row by row.
    """
    if out is None:
        out = np.empty_like(pressure_guess)
    elif out.shape != pressure_guess.shape or out.dtype != pressure_guess.dtype:
        raise ValueError("rebalance output shape/dtype differs from pressure")
    perturbation = out
    mup = dry_mass - base.mub
    nz = pressure_guess.shape[0]
    # WRF initializes the top half-level pressure from the rigid-lid row,
    # then integrates downward (module_initialize_real/ideal qvf1/qvf2
    # recurrence).  Anchoring at the top avoids importing horizontally
    # varying surface interpolation error into every pressure level.
    cq = 1.0 / (1.0 + qv[-1])
    load = qv[-1] * cq
    perturbation[-1] = (
        -0.5 * (coord.c1f[nz] * mup
                + load * (coord.c1f[nz] * base.mub + coord.c2f[nz]))
        / coord.rdnw[nz - 1] / cq)
    for k in range(nz - 2, -1, -1):
        kw = k + 1
        qbar = 0.5 * (qv[k] + qv[k + 1])
        cq = 1.0 / (1.0 + qbar)
        load = qbar * cq
        perturbation[k] = (
            perturbation[k + 1]
            - (coord.c1f[kw] * mup
               + load * (coord.c1f[kw] * base.mub + coord.c2f[kw]))
            / cq / coord.rdn[kw])
    np.add(base.pb, perturbation, out=out)
    if not np.isfinite(out).all() or np.any(out <= 0.0):
        raise ValueError("moist hydrostatic pressure recurrence failed")
    return out


def _rebalance_moist_pressure(pressure_guess, qv, dry_mass, base, coord, *,
                              column_workers=1):
    """Run the unchanged vertical recurrence over parallel row slabs."""
    workers = _column_worker_count(column_workers)
    pressure = np.empty_like(pressure_guess)
    if workers == 1 or pressure_guess.shape[1] < 2:
        return _rebalance_moist_pressure_serial(
            pressure_guess, qv, dry_mass, base, coord, out=pressure)

    chunks = _axis0_chunks(pressure_guess.shape[1], workers)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _rebalance_moist_pressure_serial,
                pressure_guess[:, start:stop], qv[:, start:stop],
                dry_mass[start:stop], _slice_base_rows(base, start, stop),
                coord, out=pressure[:, start:stop])
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return pressure


def _fp32_geopotential_split_serial(base, coord, dry_mass, alpha,
                                    hypsometric_opt: int = 1):
    """Choose phi' ulps that best preserve the float64 hydrostatic layers.

    Over high terrain the lowest explicit eta layers are only a few metres
    thick.  Forming ``phi=phb+phi'`` in FP32 can otherwise lose enough of a
    layer geopotential difference to create an artificial vertical impulse.
    This setup-only quantizer tests the nearest phi' ulp and its two
    neighbors against the exact diagnostic-alpha equation at every level.

    The target layer thickness and the diagnostic operator are keyed on
    ``hypsometric_opt`` to match the runtime EOS (calc_p_alpha) and WRF's
    real-init geopotential integration: opt 1 inverts ``alpha =
    -d(phi)*rdnw/(c1h*mu+c2h)``; opt 2 integrates ``d(phi) =
    alt*phm*LOG(pfd/pfu)`` on the TOTAL dry-mass reference pressures
    (module_initialize_real.F:3970-3981) and diagnoses ``alt =
    d(phi)/phm/LOG(pfd/pfu)`` (F:4002-4010), all in the FP32 arithmetic
    the device kernel uses.
    """
    if hypsometric_opt not in (1, 2):
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    phb = np.asarray(base.phb, dtype=np.float32)
    dnw = np.asarray(coord.dnw, dtype=np.float32)
    rdnw = np.asarray(coord.rdnw, dtype=np.float32)
    increment = np.asarray(
        coord.c1h[:, None, None] * dry_mass[None]
        + coord.c2h[:, None, None], dtype=np.float32)
    target = np.asarray(alpha, dtype=np.float32)
    if hypsometric_opt == 2:
        # Per-layer reference dry pressures on the TOTAL dry mass (WRF
        # MU0 = mub + mu'), in the kernel's FP32 arithmetic.
        c3f = np.asarray(coord.c3f, dtype=np.float32)
        c4f = np.asarray(coord.c4f, dtype=np.float32)
        c3h = np.asarray(coord.c3h, dtype=np.float32)
        c4h = np.asarray(coord.c4h, dtype=np.float32)
        mu32 = np.asarray(dry_mass, dtype=np.float32)
        pt32 = np.float32(base.p_top)
    php = np.zeros_like(phb, dtype=np.float32)
    total_low = np.asarray(phb[0] + php[0], dtype=np.float32)
    for k in range(coord.dnw.size):
        if hypsometric_opt == 2:
            pfu = c3f[k + 1] * mu32 + c4f[k + 1] + pt32
            pfd = c3f[k] * mu32 + c4f[k] + pt32
            phm = c3h[k] * mu32 + c4h[k] + pt32
            log_ratio = np.log(pfd / pfu)              # float32
            desired_dphi = np.asarray(target[k] * phm * log_ratio,
                                      dtype=np.float32)
        else:
            desired_dphi = np.asarray(-dnw[k] * increment[k] * target[k],
                                      dtype=np.float32)
        desired_total = np.asarray(total_low + desired_dphi, dtype=np.float32)
        centre = np.asarray(desired_total - phb[k + 1], dtype=np.float32)
        candidates = (
            np.nextafter(centre, np.float32(-np.inf)), centre,
            np.nextafter(centre, np.float32(np.inf)),
        )
        best = None
        best_error = None
        best_total = None
        for candidate in candidates:
            total = np.asarray(phb[k + 1] + candidate, dtype=np.float32)
            dphi = np.asarray(total - total_low, dtype=np.float32)
            if hypsometric_opt == 2:
                diagnosed = np.asarray(
                    np.asarray(dphi / phm, dtype=np.float32) / log_ratio,
                    dtype=np.float32)
            else:
                diagnosed = np.asarray(
                    np.asarray(-dphi * rdnw[k], dtype=np.float32)
                    / increment[k], dtype=np.float32)
            error = np.abs(diagnosed.astype(np.float64)
                           - target[k].astype(np.float64))
            if best is None:
                best, best_error, best_total = candidate, error, total
            else:
                choose = error < best_error
                best = np.where(choose, candidate, best)
                best_error = np.where(choose, error, best_error)
                best_total = np.where(choose, total, best_total)
        php[k + 1] = best
        total_low = best_total
    return php


def _fp32_geopotential_split(base, coord, dry_mass, alpha,
                             hypsometric_opt: int = 1, *,
                             column_workers=1):
    """Quantize geopotential independently over parallel mass-grid slabs."""
    workers = _column_worker_count(column_workers)
    if workers == 1 or dry_mass.shape[0] < 2:
        return _fp32_geopotential_split_serial(
            base, coord, dry_mass, alpha, hypsometric_opt)

    chunks = _axis0_chunks(dry_mass.shape[0], workers)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _fp32_geopotential_split_serial,
                _slice_base_rows(base, start, stop), coord,
                dry_mass[start:stop], alpha[:, start:stop],
                hypsometric_opt)
            for start, stop in chunks
        ]
        parts = [future.result() for future in futures]
    return np.concatenate(parts, axis=1)


@dataclass(frozen=True)
class RealInitResult:
    state: DomainState
    coord: VerticalCoord
    base: BaseState
    surface_pressure: np.ndarray
    surface_qv: np.ndarray
    dry_mass: np.ndarray
    dry_pressure: np.ndarray
    total_pressure: np.ndarray
    total_geopotential: np.ndarray
    total_specific_volume: np.ndarray
    integrated_moisture_pressure: np.ndarray
    #: The hypsometric_opt the geopotential/base construction was keyed on;
    #: hydrostatic_residual grades the state against the same operator.
    hypsometric_opt: int = 1


def initialize_real(snapshot: HorizontalSnapshot, cfg: RunConfig,
                    coord: VerticalCoord, terrain, *, source_orography=None,
                    p_top=10000.0, sfcp_to_sfcp=True,
                    use_sh_qv=False,
                    column_workers=1,
                    preprocess_backend="cuda",
                    preprocess_workers=None,
                    cpu_bridge=None,
                    state_backend="cuda",
                    flag_sh_surface_fallback=None,
                    timing_report=None,
                    scratch_arena=None,
                    dycore_state_workspace=None) -> RealInitResult:
    """Construct a moist, discretely hydrostatic :class:`DomainState`.

    The pressure-level/RH lane requires TT, RH, GHT, UU, VV, PSFC, T2,
    exactly one of D2 or RH2, U10, and V10.  Native HRRR requires
    per-column PRES, SPFH, RH, and Q2, and for
    WSM6/Thompson/Morrison requires analyzed QC/QR/QI/QS/QG.  Classic
    Thompson's source-absent QNICE/QNRAIN moments are initialized to exact
    zero, matching real.exe, while the five analyzed HRRR mass categories are
    retained.  WRF's default
    ``use_sh_qv = .false.`` vertically interpolates HRRR RH and then diagnoses
    qv; direct SPFH/qv interpolation is available only when this function's
    explicit ``use_sh_qv=True`` option is selected.  With WRF's default
    ``use_sh_qv=False``, RH is diagnosed from the already horizontally mapped
    SPFH, temperature, and pressure exactly where real.exe handles FLAG_SH.
    Surface fields build WRF's ``use_surface`` pseudo-level that anchors
    every vertical-interpolation column.
    ``source_orography`` is mandatory for ``sfcp_to_sfcp`` unless the
    horizontal snapshot carries catalog-resolved ``SOURCE_OROGRAPHY``.
    ``preprocess_backend`` selects the interpolation engine independently of
    ``state_backend``.  The latter defaults to ``'cuda'`` so a GPU forecast
    may use CPU transforms and still receive the historical device state.
    Native export callers select ``state_backend='preprocess'``: CPU
    preprocessing then retains the completed setup/export state in NumPy
    host memory, while CUDA preprocessing keeps it on device.
    """
    if timing_report is not None:
        if not isinstance(timing_report, MutableMapping):
            raise TypeError("timing_report must be a mutable mapping")
        if len(timing_report):
            raise ValueError("timing_report must be empty")
    timing_start = timing_last = perf_counter()

    def mark_timing(name):
        nonlocal timing_last
        now = perf_counter()
        if timing_report is not None:
            timing_report[name] = now - timing_last
        timing_last = now

    if not sfcp_to_sfcp:
        raise ValueError(
            "sfcp_to_sfcp=false branch is not implemented: WRF requires "
            "the PMSL and pressure/GHT profile sfcprs3 reconstruction; "
            "copying the input PSFC is not supported")
    if not cfg.moist:
        raise ValueError("real initialization requires cfg.moist=True")
    if not isinstance(use_sh_qv, (bool, np.bool_)):
        raise TypeError("use_sh_qv must be boolean")
    use_sh_qv = bool(use_sh_qv)
    column_workers = _column_worker_count(column_workers)
    if cfg.nz != coord.dnw.size:
        raise ValueError("RunConfig.nz and vertical coordinate differ")
    finalize_vertical_coord(coord, float(p_top))
    terrain = _host(terrain)
    if terrain.shape != (cfg.ny, cfg.nx):
        raise ValueError("terrain must have shape (ny, nx)")
    specific_markers = ("PRES", "SPFH", "Q2")
    marker_count = sum(name in snapshot.fields for name in specific_markers)
    if marker_count not in (0, len(specific_markers)):
        present = [name for name in specific_markers if name in snapshot.fields]
        missing_specific = [name for name in specific_markers
                            if name not in snapshot.fields]
        raise KeyError(
            "partial specific-humidity forcing inventory: "
            f"present={present}, missing={missing_specific}")
    has_specific_humidity = marker_count == len(specific_markers)
    if use_sh_qv and not has_specific_humidity:
        raise ValueError(
            "use_sh_qv=True requires PRES, SPFH, and Q2 forcing")
    if has_specific_humidity:
        required = ("TT", "PRES", "SPFH", "GHT", "UU", "VV", "PSFC",
                    "T2", "Q2", "U10", "V10")
        if cfg.mp_physics in (6, 8, 10):
            required += ("QC", "QR", "QI", "QS", "QG")
    else:
        surface_rh_markers = tuple(
            name for name in ("D2", "RH2") if name in snapshot.fields)
        if len(surface_rh_markers) != 1:
            raise KeyError(
                "pressure-level RH forcing requires exactly one of D2 or RH2")
        surface_rh_name = surface_rh_markers[0]
        required = ("TT", "RH", "GHT", "UU", "VV", "PSFC", "T2",
                    surface_rh_name, "U10", "V10")
    missing = [name for name in required if name not in snapshot.fields]
    if missing:
        raise KeyError(f"missing real-data field(s): {missing}")
    # Only fields consumed by float64 WRF-real setup are materialized on the
    # host.  Winds and analyzed hydrometeors remain in their mapped FP32
    # device representation until the FP32 vertical interpolation.  The old
    # FP32 -> host-FP64 -> device-FP32 round trip changed no bits but cost
    # gigabytes of transfer and host residency on large domains.
    host_required = {"TT", "GHT", "PSFC", "T2"}
    if has_specific_humidity:
        host_required.update({"PRES", "SPFH", "Q2"})
    else:
        host_required.update({"RH", surface_rh_name})
    fields = {
        name: (_host(snapshot.fields[name])
               if name in host_required else snapshot.fields[name])
        for name in required
    }
    # Source precedence is explicit: a case may declare an artifact OR use
    # the forcing catalog's validated era5_z_invariant provider.  Silently
    # replacing a declaration would make provenance depend on GRIB inventory.
    if source_orography is not None and "SOURCE_OROGRAPHY" in snapshot.fields:
        raise ValueError(
            "source-orography conflict: both declared source_orography "
            "argument and forcing catalog SOURCE_OROGRAPHY "
            "(era5_z_invariant/SOILGEO) are present; declare exactly one")
    if "SOURCE_OROGRAPHY" in snapshot.fields:
        source_orography = _host(snapshot.fields["SOURCE_OROGRAPHY"])
    mark_timing("validate_and_materialize_host_fields")
    nsource = snapshot.levels_hpa.size
    mass_shape = (nsource, cfg.ny, cfg.nx)
    mass_names = ["TT", "GHT"]
    mass_names += (["PRES", "SPFH"] if has_specific_humidity else ["RH"])
    if cfg.mp_physics in (6, 8, 10) and has_specific_humidity:
        mass_names += ["QC", "QR", "QI", "QS", "QG"]
    if any(fields[name].shape != mass_shape for name in mass_names):
        raise ValueError(
            f"mass-field shapes do not match levels and mass grid: {mass_names}")
    if fields["UU"].shape != (nsource, cfg.ny, cfg.nx + 1):
        raise ValueError("UU does not have WRF u staggering")
    if fields["VV"].shape != (nsource, cfg.ny + 1, cfg.nx):
        raise ValueError("VV does not have WRF v staggering")
    if fields["U10"].shape != (cfg.ny, cfg.nx + 1):
        raise ValueError("U10 does not have WRF u staggering")
    if fields["V10"].shape != (cfg.ny + 1, cfg.nx):
        raise ValueError("V10 does not have WRF v staggering")

    if has_specific_humidity:
        pressure = fields["PRES"]
        if (not np.isfinite(pressure).all() or np.any(pressure <= 0.0)):
            raise ValueError("PRES must be finite and positive")
        surface_specific = _wrf_flag_sh_surface_specific_humidity(
            fields["Q2"], fields["SPFH"], pressure,
            force_fallback=flag_sh_surface_fallback)
        surface_qv = _specific_humidity_to_mixing_ratio(
            surface_specific, allow_wps_undershoot=True,
            column_workers=column_workers)
        source_qv = _specific_humidity_to_mixing_ratio(
            fields["SPFH"], allow_wps_undershoot=not use_sh_qv,
            column_workers=column_workers)
    else:
        pressure = np.broadcast_to(
            snapshot.levels_hpa[:, None, None] * 100.0, mass_shape).copy()
        if surface_rh_name == "D2":
            surface_qv = _saturation_mixing_ratio(
                fields["D2"], fields["PSFC"], 100.0,
                column_workers=column_workers)
        else:
            surface_qv = _saturation_mixing_ratio(
                fields["T2"], fields["PSFC"], fields["RH2"],
                column_workers=column_workers)
        source_qv = _cap_stratospheric_qv(
            _saturation_mixing_ratio(
                fields["TT"], pressure, fields["RH"],
                column_workers=column_workers),
            pressure, column_workers=column_workers)
    if source_orography is None:
        raise ValueError(
            "source_orography is required when sfcp_to_sfcp=True")
    surface_pressure = surface_pressure_from_surface(
        fields["PSFC"], source_orography, terrain, fields["T2"], surface_qv)
    # WRF integrates moisture on the ORIGINAL met surface (integ_moist is
    # called with p_gc whose level 1 is the met PSFC on SOILHGT,
    # module_initialize_real.F:1457/7022); only p_dts (:1482) pairs the
    # sfcprs2-adjusted psfc with the resulting intq for the dry mass.
    source_pd, intq, order = _integrate_moisture(
        source_qv, pressure, fields["TT"], fields["GHT"], fields["PSFC"],
        fields["T2"], surface_qv,
        _host(source_orography) if source_orography is not None else terrain,
        column_workers=column_workers)
    pressure = _ordered_levels(pressure, order)
    source_temperature = _ordered_levels(fields["TT"], order)
    source_qv = _ordered_levels(source_qv, order)
    if use_sh_qv:
        source_rh = None
    elif has_specific_humidity:
        # module_initialize_real.F:1138-1167: FLAG_SH first converts the
        # horizontally mapped SPECHUMD to qv_gc, then overwrites rh_gc at the
        # same target points.  Deriving RH on the HRRR source grid and mapping
        # it separately is not equivalent because both transforms are
        # nonlinear.  Negative WPS SPFH undershoots produce negative RH here;
        # rh_to_mxrat1 clips that RH to zero only after vertical interpolation.
        source_rh = _mixing_ratio_to_relative_humidity(
            source_temperature, pressure, source_qv,
            allow_wps_undershoot=True,
            column_workers=column_workers)
    else:
        source_rh = _ordered_levels(fields["RH"], order)
    # WRF's vert-interp source column carries the surface pseudo-level at
    # the met-source dry surface pressure pd_gc(:,1,:) = p_gc(:,1,:) - intq
    # (integ_moist:7130), i.e. the ORIGINAL met PSFC minus the vapor column.
    surface_pd = fields["PSFC"] - intq
    if use_sh_qv:
        surface_rh = None
    elif has_specific_humidity:
        surface_rh = _mixing_ratio_to_relative_humidity(
            fields["T2"], fields["PSFC"], surface_qv,
            allow_wps_undershoot=True,
            column_workers=column_workers)
    else:
        surface_rh = (
            _surface_relative_humidity(fields["D2"], fields["T2"])
            if surface_rh_name == "D2" else fields["RH2"].copy())
    dry_mass = surface_pressure - intq - float(p_top)
    if np.any(dry_mass <= 0.0):
        raise ValueError("non-positive dry column mass")
    dry_pressure = (coord.c3h[:, None, None] * dry_mass[None]
                    + coord.c4h[:, None, None] + float(p_top))
    if not np.all(np.diff(dry_pressure, axis=0) < 0.0):
        raise ValueError("target dry pressure is not monotonic")
    mark_timing("source_moisture_and_dry_mass")

    # Backend-selected FP32 vertical interpolation, after the float64
    # pressure/mass setup.
    # WRF (interp_theta=F defaults) interpolates TEMPERATURE in LOG(p) with
    # the t_extrap_type=2 below-ground branch (module_initialize_real.F:
    # 1784-1802), full pressure linearly in p through the same machinery
    # (:1805-1820, var type 'T' so it shares the temperature extrapolation
    # branch), RH in the default LOG(p) with constant extrapolation
    # (:1736-1748), and converts T -> theta only afterwards with the
    # interpolated pressure (t_to_theta, :1862-1867).
    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_bridge)
    backend_xp = preprocess.array_module
    if not isinstance(state_backend, str):
        raise TypeError("state_backend must be 'cuda', 'cpu', or 'preprocess'")
    normalized_state_backend = state_backend.strip().lower()
    if normalized_state_backend == "preprocess":
        state_xp = backend_xp
    elif normalized_state_backend == "cpu":
        state_xp = np
    elif normalized_state_backend == "cuda":
        import cupy as cp
        state_xp = cp
    else:
        raise ValueError(
            "state_backend must be 'cuda', 'cpu', or 'preprocess'")
    order_backend = backend_xp.asarray(order, dtype=backend_xp.int32)

    def backend_ordered_levels(value):
        return backend_xp.take(
            preprocess.float32(value), order_backend, axis=0)

    mass_vertical_plan = preprocess.prepare_wrf_vertical(
        preprocess.float32(source_pd),
        preprocess.float32(surface_pd),
        preprocess.float32(dry_pressure))
    temperature = mass_vertical_plan.apply(
        preprocess.float32(source_temperature),
        preprocess.float32(fields["T2"]),
        interp_in_logp=True, extrap="temperature")
    if use_sh_qv:
        qv = mass_vertical_plan.apply(
            preprocess.float32(source_qv),
            preprocess.float32(surface_qv),
            interp_in_logp=True, extrap="constant")
        rh = None
    else:
        rh = mass_vertical_plan.apply(
            preprocess.float32(source_rh),
            preprocess.float32(surface_rh),
            interp_in_logp=True, extrap="constant")
        qv = None
    total_pressure = mass_vertical_plan.apply(
        preprocess.float32(pressure),
        preprocess.float32(fields["PSFC"]),
        interp_in_logp=False, extrap="temperature")
    temperature_h = _host(temperature).astype(np.float64)
    rh_h = (None if rh is None else
            _host(rh).astype(np.float64))
    total_pressure_h = np.maximum(
        _host(total_pressure).astype(np.float64), dry_pressure)
    theta_h = _potential_temperature_from_temperature(
        temperature_h, total_pressure_h,
        column_workers=column_workers)
    if use_sh_qv:
        qv_h = _host(qv).astype(np.float64)
        if not np.isfinite(qv_h).all() or np.any(qv_h < 0.0):
            raise ValueError("interpolated specific-humidity qv is invalid")
    else:
        qv_h = _cap_stratospheric_qv(
            _saturation_mixing_ratio(
                temperature_h, total_pressure_h, rh_h,
                column_workers=column_workers),
            total_pressure_h, column_workers=column_workers)
    mark_timing("thermodynamic_vertical_interpolation")

    base = _make_real_base(coord, terrain, float(p_top), cfg.base_temp,
                           hypsometric_opt=cfg.hypsometric_opt,
                           column_workers=column_workers)
    if use_sh_qv:
        # WRF use_sh_qv retains the directly interpolated dry-air mixing
        # ratio while diagnosing the final moist-hydrostatic pressure.
        total_pressure_h = _rebalance_moist_pressure(
            total_pressure_h, qv_h, dry_mass, base, coord,
            column_workers=column_workers)
    else:
        # WRF diagnoses qv from interpolated RH, then recomputes a
        # hydrostatic pressure and diagnoses qv once more from that pressure.
        # Two passes make the q/pressure coupling converge below FP32 setup
        # precision.
        for _ in range(2):
            total_pressure_h = _rebalance_moist_pressure(
                total_pressure_h, qv_h, dry_mass, base, coord,
                column_workers=column_workers)
            temperature_h = _temperature_from_potential_temperature(
                theta_h, total_pressure_h,
                column_workers=column_workers)
            qv_h = _cap_stratospheric_qv(
                _saturation_mixing_ratio(
                    temperature_h, total_pressure_h, rh_h,
                    column_workers=column_workers),
                total_pressure_h, column_workers=column_workers)
        total_pressure_h = _rebalance_moist_pressure(
            total_pressure_h, qv_h, dry_mass, base, coord,
            column_workers=column_workers)
        # WRF invokes rh_to_mxrat1 again against its final hydrostatic
        # pressure; apply the strict pressure-side cap once more in case a
        # target level crossed 10 kPa during the last rebalance.
        qv_h = _cap_stratospheric_qv(
            qv_h, total_pressure_h, column_workers=column_workers)
    mark_timing("base_state_and_moist_rebalance")

    # U/V columns include the 10 m surface pseudo-level with pd averaged to
    # the staggered points exactly like the interior levels
    # (module_initialize_real.F:2785-2811 with vert_interp's 'U'/'V'
    # pressure averaging at :5664-5713; extrap_type=2 constant).
    source_pd_u = _pressure_at_u(source_pd)
    source_pd_v = _pressure_at_v(source_pd)
    surface_pd_u = _pressure_at_u(surface_pd[None])[0]
    surface_pd_v = _pressure_at_v(surface_pd[None])[0]
    target_pd_u = _pressure_at_u(dry_pressure)
    target_pd_v = _pressure_at_v(dry_pressure)
    u_plan = preprocess.prepare_wrf_vertical(
        preprocess.float32(source_pd_u),
        preprocess.float32(surface_pd_u),
        preprocess.float32(target_pd_u))
    v_plan = preprocess.prepare_wrf_vertical(
        preprocess.float32(source_pd_v),
        preprocess.float32(surface_pd_v),
        preprocess.float32(target_pd_v))
    u = u_plan.apply(
        backend_ordered_levels(fields["UU"]),
        preprocess.float32(fields["U10"]),
        interp_in_logp=True, extrap="constant")
    v = v_plan.apply(
        backend_ordered_levels(fields["VV"]),
        preprocess.float32(fields["V10"]),
        interp_in_logp=True, extrap="constant")

    hydrometeors = {}
    if has_specific_humidity and cfg.mp_physics in (6, 8, 10):
        # WRF's hydrometeor vert_interp calls use var_type='Q' with
        # linear_interp and no dedicated surface analysis.  The metgrid
        # surface pseudo-level is therefore zero; setting vboundb above the
        # target column keeps the shared kernel linear at every eta level.
        zero_surface = backend_xp.zeros(
            (cfg.ny, cfg.nx), dtype=backend_xp.float32)
        invalid_source = []
        for name in ("QC", "QR", "QI", "QS", "QG"):
            source_value = preprocess.float32(fields[name])
            if (not bool(backend_xp.isfinite(source_value).all())
                    or bool((source_value < 0.0).any())):
                invalid_source.append(name)
        if invalid_source:
            raise ValueError(
                "mapped HRRR hydrometeor forcing is non-finite or negative: "
                f"{invalid_source}")
        for name in ("QC", "QR", "QI", "QS", "QG"):
            value = mass_vertical_plan.apply(
                backend_ordered_levels(fields[name]),
                zero_surface,
                interp_in_logp=True, extrap="constant",
                vboundb=cfg.nz + 1, values_are_finite=True)
            if (not bool(backend_xp.isfinite(value).all())
                    or bool((value < 0.0).any())):
                raise ValueError(
                    f"interpolated HRRR hydrometeor {name} is invalid")
            hydrometeors[name] = value

    alpha = _moist_specific_volume(
        theta_h, qv_h, total_pressure_h,
        column_workers=column_workers)
    mark_timing("wind_hydrometeor_interpolation_and_alpha")

    state_kwargs = {}
    if scratch_arena is not None:
        state_kwargs["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        state_kwargs["dycore_state_workspace"] = dycore_state_workspace
    state = DomainState(cfg, array_module=state_xp, **state_kwargs)
    mark_timing("domain_state_allocation")
    state.load_base(coord, base)
    mark_timing("base_state_upload")
    state.mup[...] = state_xp.asarray(
        dry_mass - base.mub, dtype=state_xp.float32)
    state.thp[...] = state_xp.asarray(
        theta_h - base.thb, dtype=state_xp.float32)
    mark_timing("mass_theta_upload")
    state.php[...] = state_xp.asarray(
        _fp32_geopotential_split(base, coord, dry_mass, alpha,
                                 hypsometric_opt=cfg.hypsometric_opt,
                                 column_workers=column_workers),
        dtype=state_xp.float32)
    mark_timing("fp32_geopotential_split_and_upload")
    state.qv[...] = state_xp.asarray(qv_h, dtype=state_xp.float32)
    if hydrometeors:
        state.qc[...] = state_xp.asarray(
            hydrometeors["QC"], dtype=state_xp.float32)
        state.qr[...] = state_xp.asarray(
            hydrometeors["QR"], dtype=state_xp.float32)
        state.qi[...] = state_xp.asarray(
            hydrometeors["QI"], dtype=state_xp.float32)
        state.qs[...] = state_xp.asarray(
            hydrometeors["QS"], dtype=state_xp.float32)
        state.qg[...] = state_xp.asarray(
            hydrometeors["QG"], dtype=state_xp.float32)
        if cfg.mp_physics == 8:
            # HRRR provides the shared five WRF mass species but not classic
            # Thompson's Registry scalar QNICE/QNRAIN fields.  Pin real.exe's
            # source-absent policy explicitly rather than diagnosing a
            # distribution here: both transported number moments begin at
            # exact FP32 zero and Thompson owns their first physical update.
            state.ni[...] = state_xp.float32(0.0)
            state.nr[...] = state_xp.float32(0.0)
    else:
        state.qc[...] = 0.0
        state.qr[...] = 0.0
    state.u[...] = state_xp.asarray(u, dtype=state_xp.float32)
    state.v[...] = state_xp.asarray(v, dtype=state_xp.float32)
    state.w[...] = 0.0
    total_phi = _host(state.phb + state.php)
    mark_timing("remaining_state_upload_and_geopotential_readback")
    if timing_report is not None:
        timing_report["total_seconds"] = perf_counter() - timing_start
    return RealInitResult(
        state=state, coord=coord, base=base,
        surface_pressure=surface_pressure, surface_qv=surface_qv,
        dry_mass=dry_mass, dry_pressure=dry_pressure,
        total_pressure=total_pressure_h, total_geopotential=total_phi,
        total_specific_volume=alpha,
        integrated_moisture_pressure=intq,
        hypsometric_opt=cfg.hypsometric_opt)


def source_orography_from_catalog(catalog, grid, *,
                                  provider="era5_z_invariant",
                                  valid_time=None) -> np.ndarray:
    """Public real-ingest resolver for catalog-declared source terrain."""
    return _source_orography_from_catalog(
        catalog, grid, provider=provider, valid_time=valid_time)


def hydrostatic_residual(result: RealInitResult) -> np.ndarray:
    """Maximum discrete moist-hydrostatic residual of the live FP32 state.

    Pressure remains the setup-time thermodynamic target, while geopotential,
    dry mass, potential temperature, vapor, coefficients, and arithmetic inputs
    are read back from the initialized :class:`DomainState`.  This makes the
    gate sensitive to the actual ``state.php`` quantization loaded on device.
    """
    state = result.state
    total_phi = _host(state.phb + state.php)
    dry_mass = _host(state.mub2d + state.mup)
    theta = _host(state.thb + state.thp)
    qv = _host(state.qv)
    pressure = np.asarray(result.total_pressure, dtype=np.float64)
    alpha = (c.RD * theta * (1.0 + c.RVOVRD * qv)
             * (pressure / c.P0) ** c.RCP / pressure)
    if result.hypsometric_opt == 2:
        # Log-pressure hydrostatic operator on the total dry mass, the
        # opt-2 counterpart of the discrete d(phi)/d(eta) relation
        # (module_initialize_real.F:3970-3981 / :4002-4010).
        c3h = _host(state.c3h)[:, None, None]
        c4h = _host(state.c4h)[:, None, None]
        c3f = _host(state.c3f)[:, None, None]
        c4f = _host(state.c4f)[:, None, None]
        p_top = float(state.p_top)
        pfu = c3f[1:] * dry_mass[None] + c4f[1:] + p_top
        pfd = c3f[:-1] * dry_mass[None] + c4f[:-1] + p_top
        phm = c3h * dry_mass[None] + c4h + p_top
        residual = (np.diff(total_phi, axis=0)
                    - alpha * phm * np.log(pfd / pfu))
    else:
        c1h = _host(state.c1h)[:, None, None]
        c2h = _host(state.c2h)[:, None, None]
        dnw = _host(state.dnw)[:, None, None]
        increment = c1h * dry_mass[None] + c2h
        residual = np.diff(total_phi, axis=0) + dnw * increment * alpha
    return np.max(np.abs(residual), axis=0)


__all__ = ["RealInitResult", "hydrostatic_residual", "initialize_real",
           "source_orography_from_catalog", "surface_pressure_from_surface"]
