"""Device twins of REAL setup helpers, with the CPU arithmetic as oracle.

No library fallback is provided: the portable header must be installed before
the transcendental unit can compile. Arithmetic helpers compile separately.
"""
from __future__ import annotations

from types import SimpleNamespace
import sys

from gpuwm.core import constants as c
from gpuwm.core.kernels import load_module


def _workers(value):
    from gpuwm.ingest.real import _column_worker_count
    return _column_worker_count(value)


def _cp():
    import cupy
    return cupy


def _contiguous(value, dtype=None):
    cp = _cp()
    value = cp.asarray(value)
    if dtype == cp.float64 and value.dtype == cp.float32:
        value = widen(value)
    return cp.ascontiguousarray(cp.asarray(value, dtype=dtype))


def _arrays(*values):
    cp = _cp()
    return tuple(_contiguous(v, dtype=cp.float64)
                 for v in cp.broadcast_arrays(*(_contiguous(v, dtype=cp.float64)
                                                for v in values)))


def _launch(unit, name, n, args):
    if n:
        load_module(unit).get_function(name)(((n + 127) // 128,), (128,), args)


def _flag():
    cp = _cp()
    return cp.zeros(1, dtype=cp.uint32)


def widen(value):
    """Float32 to float64 by IEEE words, preserving sm_120 subnormals."""
    cp = _cp()
    value = cp.asarray(value)
    if value.dtype != cp.float32:
        return cp.asarray(value, dtype=cp.float64)
    value = cp.ascontiguousarray(value)
    out = cp.empty(value.shape, dtype=cp.float64)
    _launch("real_init", "real_widen", value.size,
            (value.view(cp.uint32), out, value.size))
    return out


def float32(value):
    """Narrow REAL's double operands with explicit subnormal rounding."""
    cp = _cp()
    value = cp.asarray(value)
    if value.dtype != cp.float64:
        return cp.asarray(value, dtype=cp.float32)
    value = cp.ascontiguousarray(value)
    out = cp.empty(value.shape, dtype=cp.float32)
    _launch("real_init", "real_narrow", value.size, (value, out, value.size))
    return out


def _fp32_probe(a, b):
    """Grade the quantizer's FP32 primitives without layer cancellation."""
    cp = _cp()
    a, b = (cp.ascontiguousarray(cp.asarray(v, dtype=cp.float32)) for v in (a, b))
    out = cp.empty((4, a.size), dtype=cp.float32)
    _launch("real_init", "real_fp32_probe", a.size, (a, b, out, a.size))
    return out


def base_residual(phi):
    cp = _cp()
    phi = _contiguous(phi, dtype=cp.float64)
    out = cp.empty((phi.shape[0] - 1, *phi.shape[1:]), dtype=cp.float32)
    ncol = phi.size // phi.shape[0]
    _launch("real_init", "real_base_residual", out.size, (phi, out, out.size, ncol))
    return out


def _ordered_levels(array, order):
    cp = _cp()
    array = cp.asarray(array)
    # Source order is integer metadata. Monotonic inventories keep a view,
    # just as the reference does, instead of duplicating a whole field.
    if not isinstance(order, cp.ndarray):
        levels = tuple(int(v) for v in order)
        if levels == tuple(range(array.shape[0])):
            return array
        if levels == tuple(range(array.shape[0] - 1, -1, -1)):
            return array[::-1]
    return cp.take(cp.asarray(array), cp.asarray(order, dtype=cp.int32), axis=0)


def _specific_humidity_to_mixing_ratio(specific_humidity, *,
        allow_wps_undershoot=False, undershoot_floor=None, column_workers=1):
    from gpuwm.ingest import real
    cp = _cp()
    if not isinstance(allow_wps_undershoot, (bool, _cp().bool_)):
        raise TypeError("allow_wps_undershoot must be boolean")
    real._column_worker_count(column_workers)
    lower, _ = real._specific_humidity_undershoot_bound(
        allow_wps_undershoot, undershoot_floor)
    q, = _arrays(specific_humidity)
    out, bad = cp.empty_like(q), _flag()
    _launch("real_init", "real_specific", q.size,
            (q, out, bad, q.size, float(lower)))
    if int(bad.item()):
        # Only a refusal copies the field. The host constructs its exact text,
        # including the first failing threaded slab when workers were named.
        real._specific_humidity_to_mixing_ratio(q.get(),
            allow_wps_undershoot=allow_wps_undershoot,
            undershoot_floor=undershoot_floor, column_workers=column_workers)
        raise AssertionError("device refusal disagrees with host")
    return out


def _cap_stratospheric_qv(qv, pressure, *, column_workers=1):
    _workers(column_workers)
    cp = _cp()
    q, p = _arrays(qv, pressure)
    out = cp.empty_like(q)
    _launch("real_init", "real_cap", q.size, (q, p, out, q.size))
    return out


def _wrf_flag_sh_surface_specific_humidity(q2, spfh, pressure, *, force_fallback=None):
    cp = _cp()
    q2, spfh, pressure = (cp.asarray(v, dtype=cp.float64) for v in (q2, spfh, pressure))
    if spfh.ndim != 3 or pressure.shape != spfh.shape:
        raise ValueError("SPFH and pressure must share shape (level, y, x)")
    if q2.shape != spfh.shape[1:]:
        raise ValueError("Q2 must match the SPFH horizontal grid")
    if force_fallback is not None and not isinstance(force_fallback, (bool, cp.bool_)):
        raise TypeError("force_fallback must be boolean or None")
    fallback = bool((q2[0, 0] < 1.0e-6).item()) if force_fallback is None else bool(force_fallback)
    if not fallback:
        return q2
    nearest = 0 if bool((pressure[-1, 0, 0] < pressure[0, 0, 0]).item()) else -1
    return spfh[nearest].copy()


def _integrate_moisture(qv, pressure, temperature, height, psfc, tsfc, qsfc,
                        surface_height, *, column_workers=1, _order=None):
    _workers(column_workers)
    cp = _cp()
    p = _contiguous(pressure, dtype=cp.float64)
    if p.ndim != 3:
        raise ValueError("pressure must have shape (level, y, x)")
    # This tiny metadata vector is ordered by NumPy's exact argsort. CuPy's
    # stable sort differs on ties from NumPy's default quicksort.
    import numpy as np
    order = np.argsort((-p[:, 0, 0]).get()) if _order is None else _order
    od = cp.asarray(order, dtype=cp.int32)
    q, t, z = (_contiguous(v, dtype=cp.float64) for v in (qv, temperature, height))
    ps, ts, qs, zs = (_contiguous(v, dtype=cp.float64)
                      for v in (psfc, tsfc, qsfc, surface_height))
    pd, intq = cp.empty_like(p), cp.empty_like(ps)
    missing = cp.full(1, ps.size, dtype=cp.uint32)
    _launch("real_init", "real_integrate", ps.size,
            (q, p, t, z, ps, ts, qs, zs, od, pd, intq, missing,
             ps.size, p.shape[0], float(c.RD), float(c.G)))
    first = int(missing.item())
    if first < ps.size:
        j, i = divmod(first, p.shape[2])
        raise ValueError(f"no pressure level above the surface at column ({j}, {i})")
    return pd, intq, order


def _pressure_at(pressure, axis):
    cp = _cp()
    p = _contiguous(pressure, dtype=cp.float64)
    shape = list(p.shape)
    shape[axis] += 1
    out = cp.empty(tuple(shape), dtype=cp.float64)
    _launch("real_init", "real_stagger", out.size, (p, out, *p.shape, axis))
    return out


def _pressure_at_u(pressure):
    return _pressure_at(pressure, 2)


def _pressure_at_v(pressure):
    return _pressure_at(pressure, 1)


def dry_pressure_ladder(dry_mass, c3, c4, p_top):
    cp = _cp()
    mu = _contiguous(dry_mass, dtype=cp.float64)
    c3, c4 = (_contiguous(v, dtype=cp.float64) for v in (c3, c4))
    out = cp.empty((c3.size, *mu.shape), dtype=cp.float64)
    _launch("real_init", "real_dry_ladder", out.size,
            (mu, c3, c4, out, mu.size, c3.size, float(p_top)))
    return out


def upload_base(base):
    cp = _cp()
    return SimpleNamespace(**{name: (cp.asarray(value, dtype=cp.float64)
        if name in ("mub", "pb", "alb", "thb", "phb", "terrain_z") and value is not None
        else value) for name, value in vars(base).items()})


def _rebalance_moist_pressure(pressure_guess, qtot, dry_mass, base, coord, *, column_workers=1):
    _workers(column_workers)
    cp = _cp()
    q = _contiguous(qtot, dtype=cp.float64)
    mu, mub, pb = (_contiguous(v, dtype=cp.float64)
                   for v in (dry_mass, base.mub, base.pb))
    coeff = tuple(_contiguous(getattr(coord, v), dtype=cp.float64)
                  for v in ("c1f", "c2f", "rdnw", "rdn"))
    out, bad = cp.empty_like(q), _flag()
    _launch("real_init", "real_rebalance", mu.size,
            (q, mu, mub, pb, *coeff, out, bad, mu.size, q.shape[0]))
    if int(bad.item()):
        raise ValueError("moist hydrostatic pressure recurrence failed")
    return out


def _fp32_geopotential_split(base, coord, dry_mass, alpha, hypsometric_opt=1, *, column_workers=1):
    _workers(column_workers)
    cp = _cp()
    if hypsometric_opt not in (1, 2):
        raise ValueError(f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    phi, mu, a = (_contiguous(v, dtype=cp.float64) for v in (base.phb, dry_mass, alpha))
    out = cp.empty(phi.shape, dtype=cp.float32)
    if hypsometric_opt == 1:
        coeff = tuple(_contiguous(getattr(coord, v), dtype=dtype)
                      for v, dtype in (("c1h", cp.float64), ("c2h", cp.float64),
                                       ("dnw", cp.float32), ("rdnw", cp.float32)))
        _launch("real_init", "real_split_opt1", mu.size,
                (phi, mu, a, *coeff, out, mu.size, a.shape[0]))
    else:
        coeff = tuple(_contiguous(getattr(coord, v), dtype=cp.float32)
                      for v in ("c3f", "c4f", "c3h", "c4h"))
        values = tuple(cp.asarray(getattr(coord, v), dtype=cp.float64)
                       for v in ("c3f", "c4f"))
        drops = tuple(v[:-1] - v[1:] for v in values)
        drops = tuple(float32(v) for v in drops)
        _launch("real_init_math", "real_split_opt2", mu.size,
                (phi, mu, a, *coeff, *drops, out, mu.size, a.shape[0], cp.float32(base.p_top)))
    return out


def _thermo(t, p, q, operation, minimum_qv=0.0):
    cp = _cp()
    if operation in (0, 1, 5):
        t, p = _arrays(t, p)
        q = t
    else:
        t, p, q = _arrays(t, p, q)
    out, bad = cp.empty_like(t), _flag()
    _launch("real_init_math", "real_thermo", t.size,
        (t, p, q, out, bad, t.size, operation, float(c.P0), float(c.RCP),
         float(c.RD), float(c.RVOVRD), float(10.0 * c.SVP1), float(c.SVP2),
         float(c.SVPT0), float(c.SVP3), 100.0, float(minimum_qv), float(2.5e6 / 461.5)))
    return out, int(bad.item())


def _potential_temperature_from_temperature(temperature, pressure, *, column_workers=1):
    _workers(column_workers)
    return _thermo(temperature, pressure, 0.0, 0)[0]


def _temperature_from_potential_temperature(theta, pressure, *, column_workers=1):
    _workers(column_workers)
    return _thermo(theta, pressure, 0.0, 1)[0]


def _moist_specific_volume(theta, qv, pressure, *, column_workers=1):
    _workers(column_workers)
    return _thermo(theta, pressure, qv, 2)[0]


def _saturation_mixing_ratio(temperature, pressure, relative_humidity=100.0, *, column_workers=1):
    _workers(column_workers)
    return _thermo(temperature, pressure, relative_humidity, 3)[0]


def _mixing_ratio_to_relative_humidity(temperature, pressure, mixing_ratio, *,
        allow_wps_undershoot=False, column_workers=1):
    from gpuwm.ingest import real
    bound = real._WPS_SPFH_UNDERSHOOT_LOWER_BOUND
    if not isinstance(allow_wps_undershoot, (bool, _cp().bool_)):
        raise TypeError("allow_wps_undershoot must be boolean")
    _workers(column_workers)
    minimum = bound / (1.0 - bound) if allow_wps_undershoot else 0.0
    out, bad = _thermo(temperature, pressure, mixing_ratio, 4, minimum)
    if bad:
        # Threaded refusals follow the first failing slab, not the global
        # priority of the two predicates. Copy only on a device refusal.
        values = tuple(_cp().asarray(v).get() for v in (temperature, pressure, mixing_ratio))
        real._mixing_ratio_to_relative_humidity(*values,
            allow_wps_undershoot=allow_wps_undershoot, column_workers=column_workers)
        raise AssertionError("device refusal disagrees with host")
    return out


def _surface_relative_humidity(dewpoint, temperature):
    return _thermo(dewpoint, temperature, 0.0, 5)[0]


def surface_pressure_from_surface(psfc_in, source_orography, terrain, surface_temperature, surface_qv):
    cp = _cp()
    arrays = tuple(_contiguous(v, dtype=cp.float64)
                   for v in (psfc_in, source_orography, terrain, surface_temperature, surface_qv))
    if len({v.shape for v in arrays}) != 1:
        raise ValueError("surface-pressure input shapes differ")
    out, bad = cp.empty_like(arrays[0]), _flag()
    _launch("real_init_math", "real_surface_pressure", out.size,
            (*arrays, out, bad, out.size, float(c.RD), float(c.G)))
    bits = int(bad.item())
    if bits & 1:
        raise ValueError("surface virtual temperature must be finite and positive")
    if bits & 2:
        raise ValueError("adjusted surface pressure is invalid")
    return out


def _refuse_non_finite_prognostic_qv(qv):
    cp = _cp()
    if not bool(cp.isfinite(qv).all().item()):
        from gpuwm.ingest.real import _refuse_non_finite_prognostic_qv as host
        host(qv.get())


def _floor_flag_sh_surface_mixing_ratio(surface_qv, surface_pressure):
    from gpuwm.ingest import real
    cp = _cp()
    q, p = (cp.asarray(v, dtype=cp.float64) for v in (surface_qv, surface_pressure))
    if p.shape != q.shape:
        raise ValueError("surface pressure and surface mixing ratio shapes differ")
    mask = (p < real._WRF_QV_MIN_P_SAFE) & (q < real._WRF_QV_MIN_VALUE)
    count = int(cp.count_nonzero(mask).item())
    if not count:
        return q, {}
    negative = int(cp.count_nonzero(mask & (q < 0.0)).item())
    minimum = float(cp.min(q[mask]).item())
    receipt = dict(policy="flag-sh-surface-qv-floored-to-wrf-qv-min-value",
        wrf_reference=dict(real._SURFACE_QV_FLOOR_WRF_REFERENCE),
        qv_min_value=real._WRF_QV_MIN_VALUE, qv_min_p_safe=real._WRF_QV_MIN_P_SAFE,
        floored_cells=count, negative_cells=negative, min_pre_floor=minimum)
    print(f"surface moisture floor: {count} FLAG_SH 2 m mixing-ratio "
        f"value(s) below WRF's qv_min_value {real._WRF_QV_MIN_VALUE:g} floored "
        f"to it ({negative} of them negative; min pre-floor {minimum:.6g}); "
        "WPS's overshooting sixteen_pt operator undershoots SPECHUMD that is "
        "exactly zero over high dry terrain, and real.exe carries that sign "
        "into grid%q2 unfloored (module_initialize_real.F:1157,1257)", file=sys.stderr)
    return cp.where(mask, real._WRF_QV_MIN_VALUE, q), receipt


def _floor_sh_vertical_undershoot(qv, pressure):
    from gpuwm.ingest import real
    cp = _cp()
    q, p = (cp.asarray(v, dtype=cp.float64) for v in (qv, pressure))
    if p.shape != q.shape:
        raise ValueError("pressure and interpolated qv shapes differ")
    negative = q < 0.0
    count = int(cp.count_nonzero(negative).item())
    if not count:
        return q, {}
    unreached = int(cp.count_nonzero(negative & ~(p < real._WRF_QV_MIN_P_SAFE)).item())
    levels = cp.flatnonzero(cp.any(negative, axis=(1, 2))).get().tolist()
    columns = int(cp.count_nonzero(cp.any(negative, axis=0)).item())
    minimum = float(cp.min(q[negative]).item())
    index = int(cp.argmin(cp.where(negative, q, cp.inf)).item())
    k, rem = divmod(index, q.shape[1] * q.shape[2])
    j, i = divmod(rem, q.shape[2])
    pa = float(p[k, j, i].item())
    receipt = dict(policy="use-sh-qv-vertical-undershoot-floored-to-wrf-qv-min-value",
        wrf_reference=dict(real._PROGNOSTIC_QV_FLOOR_WRF_REFERENCE),
        qv_min_value=real._WRF_QV_MIN_VALUE, qv_min_p_safe=real._WRF_QV_MIN_P_SAFE,
        floored_cells=count, floored_cells_at_or_above_qv_min_p_safe=unreached,
        columns=columns, levels=levels, min_pre_floor=minimum,
        min_pre_floor_at=dict(level=k, row=j, column=i, pressure_pa=pa))
    print(f"prognostic moisture floor: {count} interpolated vapour "
        f"value(s) in {columns} column(s) on level(s) {levels} were taken "
        "below zero by WRF's second-order vertical operator at a sharp dry "
        f"slot (min {minimum:.6g} at level {k} row {j} column {i}, {pa:.0f} Pa) "
        f"and are floored to WRF's qv_min_value {real._WRF_QV_MIN_VALUE:g}, "
        "the floor rh_to_mxrat1 applies on the RH lane and real.exe never "
        "applies on the use_sh_qv lane (module_initialize_real.F:1744-1758, :1831)"
        + (f"; {unreached} of them at or above qv_min_p_safe "
           f"{real._WRF_QV_MIN_P_SAFE:g} Pa, where only the RH lane's "
           "unconditional floor reaches" if unreached else ""), file=sys.stderr)
    return cp.where(negative, real._WRF_QV_MIN_VALUE, q), receipt


def export_result(result, *, base=None):
    """Root exports own one host copy of each published result array."""
    from dataclasses import replace
    return replace(result, base=result.base if base is None else base,
                   **{name: getattr(result, name).get() for name in (
        "surface_pressure", "surface_qv", "dry_mass", "dry_pressure",
        "total_pressure", "total_geopotential", "total_specific_volume",
        "integrated_moisture_pressure")})


def load_base(state, coord, host_base, base):
    """Install the uploaded base without recomputing column arrays on host."""
    cp = _cp()
    for name in ("dnw", "rdnw", "dn", "rdn", "fnp", "fnm", "znu", "znw",
                 "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f"):
        getattr(state, name)[...] = cp.asarray(getattr(coord, name), dtype=cp.float32)
    for name in ("thb", "pb", "alb"):
        target, source = getattr(state, name), getattr(base, name)
        if source.ndim != target.ndim:
            raise ValueError(f"base state {name} is {source.ndim}-D but the state was "
                f"allocated for {target.ndim}-D profiles: cfg.terrain_opt "
                "must match the terrain_z the base state was built with")
        target[...] = float32(source)
    for name in ("c3f", "c4f"):
        values = cp.asarray(getattr(coord, name), dtype=cp.float64)
        getattr(state, "d" + name)[...] = float32(values[:-1] - values[1:])
    state.p_top = cp.float32(base.p_top)
    if base.mub.ndim == 0:
        state.mub = cp.float32(base.mub.item())
        state.mub2d[...] = state.mub
    else:
        state.mub = None
        state.mub2d[...] = float32(base.mub)
    state.ht[...] = 0.0 if base.terrain_z is None else float32(base.terrain_z)
    source = base.phb
    if source.ndim != state.phb.ndim:
        raise ValueError(f"base state phb is {source.ndim}-D but the state was "
            f"allocated for {state.phb.ndim}-D profiles: cfg.terrain_opt "
            "must match the terrain_z the base state was built with")
    if source.shape != state.phb.shape:
        raise ValueError(f"base state phb has shape {source.shape}, the state was "
            f"allocated for {tuple(state.phb.shape)}: the grid this column profile "
            "was built on is not the grid the state holds, so nz or the "
            "horizontal extent disagree.  Writing it would either overrun the "
            "allocation or silently broadcast one column's geopotential across "
            "the domain, and dphb_resid would then describe a profile no cell has")
    stored = float32(source)
    state.phb[...] = stored
    state.dphb_resid[...] = base_residual(source)
    # BaseState stays the host source. Retaining a copy is a cache ownership
    # operation; the spacing calculation runs on the device copy.
    state._phb_host = host_base.phb.copy()
    half = 0.5 * (source[:-1] + source[1:]) / float(c.G)
    state._dz_min = float(cp.diff(half, axis=0).min().item()) if half.shape[0] > 1 else None
    if coord.dnw.size >= 3:
        dn, dnw, fnp, fnm = coord.dn, coord.dnw, coord.fnp, coord.fnm
        cof1 = (2.0 * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
        cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
        state.cf1 = cp.float32(fnp[1] + cof1)
        state.cf2 = cp.float32(fnm[1] - cof1 - cof2)
        state.cf3 = cp.float32(cof2)
    if coord.dnw.size >= 1:
        state.cfn = cp.float32(1.0 + coord.fnp[-1])
        state.cfn1 = cp.float32(-coord.fnp[-1])
