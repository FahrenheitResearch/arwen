"""GPU log-pressure vertical interpolation for WRF real-data ingest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpuwm.core.kernels import get_kernel, get_kernel_int_defines

_THREADS = 256

#: Column capacities ``wrf_real_vertical_interpolate`` is compiled at.  A
#: column is the source levels plus the surface pseudo-level, and each tier
#: is the ``WRF_VI_MAX_LEVELS`` size of the kernel's three per-thread column
#: arrays.  64 is the kernel source's own default, so every source up to 63
#: levels (GFS and ERA5 pressure levels, HRRR native levels) keeps the
#: binary it always ran; 160 holds ERA5's 137 model levels plus the surface
#: and ICON's model levels; 256 is headroom above those.  A column deeper
#: than the top tier runs on the packaged CPU bridge instead
#: (:func:`wrf_vertical_route`), which takes any level count.
WRF_VERT_INTERP_LEVEL_TIERS = (64, 160, 256)

_WRF_VI_DEFINE = "WRF_VI_MAX_LEVELS"


def wrf_vert_interp_level_tier(column_levels: int) -> int | None:
    """Smallest compiled tier holding ``column_levels``, or ``None`` above
    the top tier (that column runs on the CPU bridge)."""

    column_levels = int(column_levels)
    for tier in WRF_VERT_INTERP_LEVEL_TIERS:
        if column_levels <= tier:
            return tier
    return None


def wrf_vertical_route(source_levels: int) -> dict[str, object]:
    """Which engine interpolates a ``source_levels``-deep column, and why.

    This is the entry the CUDA preprocessing receipt records for every
    vertical geometry it prepares, so a reader of a preparation can tell
    which backend ran the vertical interpolation and at what depth.
    """

    source_levels = int(source_levels)
    column_levels = source_levels + 1
    tier = wrf_vert_interp_level_tier(column_levels)
    if tier is None:
        return {
            "source_levels": source_levels,
            "column_levels": column_levels,
            "backend": "cpu",
            "kernel_level_tier": None,
            "reason": (
                f"the {column_levels}-level column ({source_levels} source "
                "levels plus the surface) is deeper than the CUDA kernel's "
                f"top tier of {WRF_VERT_INTERP_LEVEL_TIERS[-1]} levels, so "
                "the parallel CPU bridge ran this vertical interpolation"),
        }
    return {
        "source_levels": source_levels,
        "column_levels": column_levels,
        "backend": "cuda",
        "kernel_level_tier": tier,
        "reason": (
            f"the {column_levels}-level column ({source_levels} source "
            f"levels plus the surface) fits the CUDA kernel's {tier}-level "
            "tier"),
    }


def _wrf_vert_kernel(tier: int):
    """The WRF-real vertical kernel compiled for one column tier."""

    if tier == WRF_VERT_INTERP_LEVEL_TIERS[0]:
        # The source's own #ifndef default: the module every shallow
        # source has always launched, shared with vertical_interpolate_logp.
        return get_kernel("vert_interp", "wrf_real_vertical_interpolate")
    return get_kernel_int_defines(
        "vert_interp", "wrf_real_vertical_interpolate",
        ((_WRF_VI_DEFINE, int(tier)),))


def _deep_column_bridge(column_levels: int):
    """The CPU bridge for a column above the top CUDA tier, resolved up front."""

    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend

    try:
        return CpuPreprocessBackend()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"a {column_levels}-level vertical column is deeper than the "
            "CUDA kernel's top tier of "
            f"{WRF_VERT_INTERP_LEVEL_TIERS[-1]} levels, so its vertical "
            "interpolation runs on the parallel CPU bridge, and that "
            f"bridge is not installed here.\n{exc}") from exc


def _bridge_vertical(bridge, values, sfc_value, source, sfc_pressure, target,
                     *, interp_in_logp, extrap, force_sfc_in_vinterp,
                     zap_close_levels, vboundb):
    """Run a descending, validated column set on the CPU bridge; device out."""

    cp = _cupy()
    output = bridge.wrf_vertical_interpolate(
        values, sfc_value, source, sfc_pressure, target,
        interp_in_logp=bool(interp_in_logp), extrap=extrap,
        force_sfc_in_vinterp=int(force_sfc_in_vinterp),
        zap_close_levels=float(zap_close_levels), vboundb=int(vboundb))
    return cp.asarray(output, dtype=cp.float32)


@dataclass(frozen=True)
class _WrfVertGeometryPlan:
    """Validated, contiguous pressure geometry shared by value fields.

    ``kernel_level_tier`` is the compiled column tier that runs this
    geometry, or ``None`` when the column is deeper than the top tier and
    ``cpu_bridge`` runs it instead.
    """

    source: object
    surface_pressure: object
    target: object
    source_shape: tuple[int, int, int]
    reverse_values: bool
    kernel_level_tier: int | None = WRF_VERT_INTERP_LEVEL_TIERS[0]
    cpu_bridge: object | None = None


def _prepare_wrf_vert_interp_geometry(source_pressure, surface_pressure,
                                      target_pressure):
    """Validate one WRF-real pressure geometry for repeated field applies."""
    cp = _cupy()
    source = cp.asarray(source_pressure, dtype=cp.float32)
    sfc_pressure = cp.asarray(surface_pressure, dtype=cp.float32)
    target = cp.asarray(target_pressure, dtype=cp.float32)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source_pressure and target_pressure must be (level, y, x)")
    if sfc_pressure.shape != source.shape[1:]:
        raise ValueError("surface_pressure must be (y, x)")
    if target.shape[1:] != source.shape[1:]:
        raise ValueError("source and target horizontal shapes differ")
    column_levels = int(source.shape[0]) + 1
    tier = wrf_vert_interp_level_tier(column_levels)
    bridge = None if tier is not None else _deep_column_bridge(column_levels)
    if (not bool(cp.isfinite(source).all())
            or not bool(cp.isfinite(sfc_pressure).all())
            or not bool(cp.isfinite(target).all())):
        raise ValueError("vertical interpolation pressure geometry must be finite")
    if (bool((source <= 0.0).any()) or bool((target <= 0.0).any())
            or bool((sfc_pressure <= 0.0).any())):
        raise ValueError("vertical interpolation pressures must be positive")
    descending = bool((source[:-1] > source[1:]).all())
    ascending = bool((source[:-1] < source[1:]).all())
    if not descending and not ascending:
        raise ValueError(
            "source pressure must be strictly monotonic in every column")
    if ascending:
        source = source[::-1]
    if not bool(((source < sfc_pressure[None]).any(axis=0)).all()):
        raise ValueError("every column needs a source level above the surface")
    if bool((target < source[-1][None, :, :]).any()):
        raise ValueError("target pressure lies above source top")
    return _WrfVertGeometryPlan(
        source=cp.ascontiguousarray(source),
        surface_pressure=cp.ascontiguousarray(sfc_pressure),
        target=cp.ascontiguousarray(target),
        source_shape=tuple(map(int, source.shape)),
        reverse_values=bool(ascending),
        kernel_level_tier=tier,
        cpu_bridge=bridge,
    )


def _wrf_vert_interp_gpu_prepared(
        field, surface_value, plan: _WrfVertGeometryPlan, *,
        interp_in_logp=True, extrap="constant", force_sfc_in_vinterp=1,
        zap_close_levels=500.0, vboundb=4,
        values_are_finite=False):
    """Apply values to a pressure geometry validated exactly once."""
    cp = _cupy()
    if not isinstance(plan, _WrfVertGeometryPlan):
        raise TypeError("plan must be a prepared WRF vertical geometry")
    if extrap not in ("constant", "temperature"):
        raise ValueError("extrap must be 'constant' or 'temperature'")
    if not isinstance(values_are_finite, (bool, np.bool_)):
        raise TypeError("values_are_finite must be boolean")
    values = cp.asarray(field, dtype=cp.float32)
    sfc_value = cp.asarray(surface_value, dtype=cp.float32)
    if tuple(map(int, values.shape)) != plan.source_shape:
        raise ValueError("field shape does not match prepared source pressure")
    if sfc_value.shape != values.shape[1:]:
        raise ValueError("surface_value must be (y, x)")
    if not 0 <= int(force_sfc_in_vinterp) <= plan.target.shape[0]:
        raise ValueError("force_sfc_in_vinterp must be within target levels")
    if ((not values_are_finite and not bool(cp.isfinite(values).all()))
            or not bool(cp.isfinite(sfc_value).all())):
        raise ValueError("vertical interpolation values must be finite")
    if plan.reverse_values:
        values = values[::-1]
    values = cp.ascontiguousarray(values)
    sfc_value = cp.ascontiguousarray(sfc_value)
    if plan.kernel_level_tier is None:
        return _bridge_vertical(
            plan.cpu_bridge, values, sfc_value, plan.source,
            plan.surface_pressure, plan.target,
            interp_in_logp=interp_in_logp, extrap=extrap,
            force_sfc_in_vinterp=force_sfc_in_vinterp,
            zap_close_levels=zap_close_levels, vboundb=vboundb)
    output = cp.empty(plan.target.shape, dtype=cp.float32)
    nsource, ny, nx = plan.source_shape
    ntarget = int(plan.target.shape[0])
    ncolumn = ny * nx
    kernel = _wrf_vert_kernel(plan.kernel_level_tier)
    kernel(((ncolumn + _THREADS - 1) // _THREADS,), (_THREADS,),
           (values, sfc_value, plan.source, plan.surface_pressure,
            plan.target, output, np.int32(nsource), np.int32(ntarget),
            np.int32(ncolumn), np.int32(bool(interp_in_logp)),
            np.int32(extrap == "temperature"),
            np.int32(force_sfc_in_vinterp),
            np.float32(zap_close_levels), np.int32(vboundb)))
    return output


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - CPU-only installations
        raise RuntimeError("CuPy is required for GPU vertical interpolation") from exc
    return cp


def interpolate_logp_gpu(field, source_pressure, target_pressure, *,
                         below="constant", above="error"):
    """Interpolate ``field`` from source to target pressure columns on GPU.

    Arrays are ``(level, y, x)`` and device FP32.  Source pressure may be
    ascending or descending but must be strictly monotonic in every column;
    ascending inputs are reversed before the kernel.  Interior interpolation
    is linear in log pressure.  The extrapolation names and equations match
    :func:`gpuwm.verify.npref.np_vertical_interpolate_logp`.
    """
    cp = _cupy()
    if below not in ("constant", "temperature"):
        raise ValueError("below must be 'constant' or 'temperature'")
    if above != "error":
        raise ValueError("above must be 'error'")
    values = cp.asarray(field, dtype=cp.float32)
    source = cp.asarray(source_pressure, dtype=cp.float32)
    target = cp.asarray(target_pressure, dtype=cp.float32)
    if values.ndim != 3 or target.ndim != 3:
        raise ValueError("field and target_pressure must be (level, y, x)")
    try:
        source = cp.broadcast_to(source, values.shape)
    except ValueError as exc:
        raise ValueError("source_pressure is not broadcastable to field") from exc
    if target.shape[1:] != values.shape[1:]:
        raise ValueError("source and target horizontal shapes differ")
    if (not bool(cp.isfinite(values).all()) or not bool(cp.isfinite(source).all())
            or not bool(cp.isfinite(target).all())):
        raise ValueError("vertical interpolation inputs must be finite")
    if bool((source <= 0.0).any()) or bool((target <= 0.0).any()):
        raise ValueError("vertical interpolation pressures must be positive")
    descending = bool((source[:-1] > source[1:]).all())
    ascending = bool((source[:-1] < source[1:]).all())
    if not descending and not ascending:
        raise ValueError("source pressure must be strictly monotonic in every column")
    if ascending:
        values = values[::-1]
        source = source[::-1]
    if bool((target < source[-1][None, :, :]).any()):
        raise ValueError("target pressure lies above source top")
    values = cp.ascontiguousarray(values)
    source = cp.ascontiguousarray(source)
    target = cp.ascontiguousarray(target)
    output = cp.empty(target.shape, dtype=cp.float32)
    nsource, ny, nx = values.shape
    ntarget = target.shape[0]
    count = ntarget * ny * nx
    kernel = get_kernel("vert_interp", "vertical_interpolate_logp")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,),
           (values, source, target, output,
            np.int32(nsource), np.int32(ntarget), np.int32(ny * nx),
            np.int32(below == "temperature")))
    return output


def wrf_vert_interp_gpu(field, surface_value, source_pressure,
                        surface_pressure, target_pressure, *,
                        interp_in_logp=True, extrap="constant",
                        force_sfc_in_vinterp=1, zap_close_levels=500.0,
                        vboundb=4):
    """WRF real's default vertical interpolation on GPU columns.

    Device FP32 transcription of ``module_initialize_real.F:vert_interp`` at
    the reference run's Registry defaults (``use_surface=T``,
    ``use_levels_below_ground=T``, ``lagrange_order=2``,
    ``force_sfc_in_vinterp=1``, ``zap_close_levels=500``); the float64
    authority is :func:`gpuwm.verify.npref.np_wrf_real_vert_interp`.

    ``field``/``source_pressure`` are ``(nsource, y, x)`` isobaric columns
    WITHOUT the surface, ascending or descending but strictly monotonic
    (ascending inputs are reversed to the kernel's bottom-up layout);
    ``surface_value``/``surface_pressure`` carry the surface pseudo-level.
    ``interp_in_logp=False`` is WRF's forced ``interp_type=1`` for the
    full-pressure field; ``extrap='temperature'`` selects the
    ``t_extrap_type=2`` CRC below-ground branch.  A target above the source
    top is WRF-fatal and rejected here before launch.  The kernel runs at
    the smallest :data:`WRF_VERT_INTERP_LEVEL_TIERS` tier holding the
    column; a column deeper than the top tier runs on the CPU bridge and
    comes back on the device all the same.
    """
    cp = _cupy()
    if extrap not in ("constant", "temperature"):
        raise ValueError("extrap must be 'constant' or 'temperature'")
    values = cp.asarray(field, dtype=cp.float32)
    sfc_value = cp.asarray(surface_value, dtype=cp.float32)
    source = cp.asarray(source_pressure, dtype=cp.float32)
    sfc_pressure = cp.asarray(surface_pressure, dtype=cp.float32)
    target = cp.asarray(target_pressure, dtype=cp.float32)
    if values.ndim != 3 or target.ndim != 3:
        raise ValueError("field and target_pressure must be (level, y, x)")
    if source.shape != values.shape:
        raise ValueError("source_pressure shape does not match field")
    if (sfc_value.shape != values.shape[1:]
            or sfc_pressure.shape != values.shape[1:]):
        raise ValueError("surface fields must be (y, x)")
    if target.shape[1:] != values.shape[1:]:
        raise ValueError("source and target horizontal shapes differ")
    column_levels = int(values.shape[0]) + 1
    tier = wrf_vert_interp_level_tier(column_levels)
    bridge = None if tier is not None else _deep_column_bridge(column_levels)
    if not 0 <= int(force_sfc_in_vinterp) <= target.shape[0]:
        raise ValueError("force_sfc_in_vinterp must be within target levels")
    finite = (bool(cp.isfinite(values).all())
              and bool(cp.isfinite(sfc_value).all())
              and bool(cp.isfinite(source).all())
              and bool(cp.isfinite(sfc_pressure).all())
              and bool(cp.isfinite(target).all()))
    if not finite:
        raise ValueError("vertical interpolation inputs must be finite")
    if (bool((source <= 0.0).any()) or bool((target <= 0.0).any())
            or bool((sfc_pressure <= 0.0).any())):
        raise ValueError("vertical interpolation pressures must be positive")
    descending = bool((source[:-1] > source[1:]).all())
    ascending = bool((source[:-1] < source[1:]).all())
    if not descending and not ascending:
        raise ValueError(
            "source pressure must be strictly monotonic in every column")
    if ascending:
        values = values[::-1]
        source = source[::-1]
    if not bool(((source < sfc_pressure[None]).any(axis=0)).all()):
        raise ValueError("every column needs a source level above the surface")
    if bool((target < source[-1][None, :, :]).any()):
        raise ValueError("target pressure lies above source top")
    values = cp.ascontiguousarray(values)
    source = cp.ascontiguousarray(source)
    sfc_value = cp.ascontiguousarray(sfc_value)
    sfc_pressure = cp.ascontiguousarray(sfc_pressure)
    target = cp.ascontiguousarray(target)
    if tier is None:
        return _bridge_vertical(
            bridge, values, sfc_value, source, sfc_pressure, target,
            interp_in_logp=interp_in_logp, extrap=extrap,
            force_sfc_in_vinterp=force_sfc_in_vinterp,
            zap_close_levels=zap_close_levels, vboundb=vboundb)
    output = cp.empty(target.shape, dtype=cp.float32)
    nsource, ny, nx = values.shape
    ntarget = target.shape[0]
    ncolumn = ny * nx
    kernel = _wrf_vert_kernel(tier)
    kernel(((ncolumn + _THREADS - 1) // _THREADS,), (_THREADS,),
           (values, sfc_value, source, sfc_pressure, target, output,
            np.int32(nsource), np.int32(ntarget), np.int32(ncolumn),
            np.int32(bool(interp_in_logp)),
            np.int32(extrap == "temperature"),
            np.int32(force_sfc_in_vinterp),
            np.float32(zap_close_levels), np.int32(vboundb)))
    return output


__all__ = ["WRF_VERT_INTERP_LEVEL_TIERS", "interpolate_logp_gpu",
           "wrf_vert_interp_gpu", "wrf_vert_interp_level_tier",
           "wrf_vertical_route"]
