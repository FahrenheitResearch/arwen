"""WRF specified/relaxation lateral-boundary arrays and GPU application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from gpuwm.core.kernels import get_kernel

_THREADS = 256
_THETA_OFFSET_K = np.float32(300.0)


def _host(value):
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class SideBoundary:
    """One immutable host-side boundary table and its time tendency.

    A boundary table becomes resident device state when it is attached.  Own
    immutable copies prevent an attached device mirror from silently becoming
    stale through an in-place edit; replace forcing by building a new
    :class:`LateralBoundaries` and calling :func:`attach_lateral_boundaries`
    again.
    """

    value: np.ndarray
    tendency: np.ndarray

    def __post_init__(self):
        def immutable_array(value):
            source = np.asarray(value)
            if source.dtype.hasobject:
                raise TypeError("boundary tables must have a numeric dtype")
            packed = np.ascontiguousarray(source)
            # A bytes backing makes immutability irreversible through
            # ``setflags(write=True)`` as well as ordinary item assignment.
            result = np.frombuffer(packed.tobytes(order="C"),
                                   dtype=packed.dtype).reshape(packed.shape)
            return result

        value = immutable_array(self.value)
        tendency = immutable_array(self.tendency)
        if tendency.shape != value.shape:
            raise ValueError("boundary value and tendency shapes differ")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "tendency", tendency)


@dataclass(frozen=True)
class FieldBoundary:
    west: SideBoundary
    east: SideBoundary
    south: SideBoundary
    north: SideBoundary


@dataclass(frozen=True)
class BoundaryInterval:
    start_seconds: float
    end_seconds: float
    fields: Mapping[str, FieldBoundary]

    def __post_init__(self):
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


@dataclass(frozen=True)
class LateralBoundaries:
    intervals: tuple[BoundaryInterval, ...]
    spec_bdy_width: int = 5
    spec_zone: int = 1
    relax_zone: int = 4

    def interval_at(self, elapsed_seconds: float) -> BoundaryInterval:
        t = float(elapsed_seconds)
        for interval in self.intervals:
            if interval.start_seconds <= t < interval.end_seconds:
                return interval
        if t == self.intervals[-1].end_seconds:
            return self.intervals[-1]
        raise ValueError(f"boundary time {t} s is outside the available intervals")


@dataclass(frozen=True)
class RollingNestBoundaries:
    """Metadata-only marker for one mutable, device-resident nest frame.

    Nested forcing has no host interval search: FORCE refreshes one rolling
    interval in place and :class:`DomainClock` supplies recurrent FP32 dtbc.
    """

    spec_bdy_width: int
    spec_zone: int
    relax_zone: int


@dataclass(frozen=True)
class _DeviceSideBoundary:
    value: object
    tendency: object


@dataclass(frozen=True)
class _DeviceFieldBoundary:
    west: _DeviceSideBoundary
    east: _DeviceSideBoundary
    south: _DeviceSideBoundary
    north: _DeviceSideBoundary


@dataclass(frozen=True)
class _DeviceBoundaryInterval:
    fields: Mapping[str, _DeviceFieldBoundary]


@dataclass
class _DeviceLateralBoundaries:
    """Metadata for state-owned FP32 mirrors of host lateral forcing.

    Device storage is registered in ``DomainState.scratch``; this object owns
    only views and lookup metadata.  Boundary values/tendencies are rounded
    and uploaded at attach time.  Davies weights also remain resident, but are
    created on first use because their ``dt`` and ``spec_exp`` live on
    RunConfig rather than on the forcing object passed to
    :func:`attach_lateral_boundaries`.
    """

    intervals: tuple[_DeviceBoundaryInterval, ...]
    host_interval_indices: Mapping[int, int]
    weights: dict[tuple[int, int, int, float, float], tuple[object, object]]
    scratch_slots: set[str]
    device_nbytes: int
    next_weight_slot: int = 0
    rolling: bool = False
    clock: object | None = None
    valid: bool = True
    streaming_external: bool = False
    active_host_interval_id: int | None = None
    packed_forcing: object | None = None
    external_reload_count: int = 0


def _seconds(times: Sequence[datetime | float]) -> np.ndarray:
    if not times:
        raise ValueError("at least two boundary times are required")
    if isinstance(times[0], datetime):
        if not all(isinstance(t, datetime) for t in times):
            raise TypeError("boundary times must all be datetime or all numeric")
        origin = times[0]
        values = np.array([(t - origin).total_seconds() for t in times],
                          dtype=np.float64)
    else:
        values = np.asarray(times, dtype=np.float64)
        values = values - values[0]
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise ValueError("boundary times must be a strictly increasing 1-D sequence")
    return values


def _field_boundary(first, second, duration, width):
    if first.ndim == 2:
        first, second = first[None], second[None]
    if first.ndim != 3 or second.shape != first.shape:
        raise ValueError("boundary fields must be matching 2-D or 3-D arrays")
    if min(first.shape[-2:]) < 2 * width:
        raise ValueError("domain is too small for the requested boundary width")

    def side(a, b):
        return SideBoundary(np.ascontiguousarray(a),
                            np.ascontiguousarray((b - a) / duration))
    return FieldBoundary(
        west=side(first[..., :width], second[..., :width]),
        east=side(first[..., -width:][..., ::-1],
                  second[..., -width:][..., ::-1]),
        south=side(first[..., :width, :], second[..., :width, :]),
        north=side(first[..., -width:, :][..., ::-1, :],
                   second[..., -width:, :][..., ::-1, :]),
    )


def build_lateral_boundaries(snapshots: Sequence[Mapping[str, object]],
                             times: Sequence[datetime | float], *,
                             spec_bdy_width=5, spec_zone=1, relax_zone=4
                             ) -> LateralBoundaries:
    """Build 6-hourly (or arbitrary-time) linear WRF boundary intervals."""
    if len(snapshots) != len(times) or len(snapshots) < 2:
        raise ValueError("snapshots and times must have the same length >= 2")
    if spec_zone < 1 or relax_zone < 2:
        raise ValueError("spec_zone must be >=1 and relax_zone >=2")
    if spec_bdy_width < spec_zone + relax_zone:
        raise ValueError("spec_bdy_width must cover spec_zone + relax_zone")
    names = set(snapshots[0])
    if not names or any(set(snapshot) != names for snapshot in snapshots[1:]):
        raise ValueError("boundary snapshot field inventories differ")
    seconds = _seconds(times)
    intervals = []
    for n in range(len(snapshots) - 1):
        duration = float(seconds[n + 1] - seconds[n])
        fields = {
            name: _field_boundary(_host(snapshots[n][name]),
                                  _host(snapshots[n + 1][name]),
                                  duration, spec_bdy_width)
            for name in sorted(names)
        }
        intervals.append(BoundaryInterval(float(seconds[n]),
                                          float(seconds[n + 1]), fields))
    return LateralBoundaries(tuple(intervals), int(spec_bdy_width),
                             int(spec_zone), int(relax_zone))


def _weights(width, spec_zone, relax_zone, dt, spec_exp, *, wrf_real=False):
    fcx = np.zeros(width, dtype=np.float32)
    gcx = np.zeros(width, dtype=np.float32)
    for loop in range(spec_zone + 1, spec_zone + relax_zone + 1):
        index = loop - 1
        if index >= width:
            break
        if wrf_real:
            # module_bc_em.F:1329-1331, nested branch: default-REAL
            # left-to-right operations and no sponge multiplication.
            dt32 = np.float32(dt)
            numerator = np.float32(spec_zone + relax_zone - loop)
            denominator = np.float32(relax_zone - 1)
            f = np.float32(np.float32(0.1) / dt32)
            f = np.float32(f * numerator)
            fcx[index] = np.float32(f / denominator)
            g = np.float32(np.float32(1.0) / dt32)
            g = np.float32(g / np.float32(50.0))
            g = np.float32(g * numerator)
            gcx[index] = np.float32(g / denominator)
            continue
        ramp = (spec_zone + relax_zone - loop) / (relax_zone - 1)
        sponge = np.exp(-(loop - (spec_zone + 1)) * spec_exp)
        fcx[index] = 0.1 / dt * ramp * sponge
        gcx[index] = 1.0 / dt / 50.0 * ramp * sponge
    return fcx, gcx


def lateral_boundary_clock_dt(cfg) -> float:
    """Return WRF's model-clock ``dt`` for lateral-BC coefficients."""
    clock_dt = float(getattr(cfg, "clock_dt", 0.0))
    return clock_dt if clock_dt > 0.0 else float(cfg.dt)


def _perimeter_count(ny: int, nx: int, width: int) -> int:
    """Number of cells in ``width`` nested frames (Y sides own corners)."""
    return sum(2 * (nx - 2 * d) + 2 * max(ny - 2 * d - 2, 0)
               for d in range(width))


def _validate_frame_domain(ny: int, nx: int, width: int, purpose: str) -> None:
    if width < 1 or min(ny, nx) <= 2 * width:
        raise ValueError(f"{purpose} width leaves no unique interior frame")


def _boundary_field_shape(boundary: FieldBoundary
                          ) -> tuple[int, int, int, int]:
    """Validate side layout and return ``(nz, ny, nx, width)``."""
    west = boundary.west.value.shape
    east = boundary.east.value.shape
    south = boundary.south.value.shape
    north = boundary.north.value.shape
    if any(len(shape) != 3 for shape in (west, east, south, north)):
        raise ValueError("boundary side tables must be 3-D")
    nz, ny, width = west
    nx = south[2]
    if east != west or south != (nz, width, nx) or north != south:
        raise ValueError("boundary side table shapes are inconsistent")
    return nz, ny, nx, width


def _release_resident_scratch(state) -> None:
    """Release LBC-owned buffers from the sanctioned state scratch pool."""
    resident = getattr(state, "_lateral_boundary_device", None)
    pool = getattr(state, "_scratch", None)
    if resident is not None and isinstance(pool, dict):
        for slot in resident.scratch_slots:
            pool.pop(slot, None)


def _lbc_scratch(state, shape, slot):
    scratch = getattr(state, "scratch", None)
    if not callable(scratch):
        raise TypeError("lateral-boundary state must provide scratch(shape, slot)")
    return scratch(shape, slot)


def lateral_boundary_resident_bytes(state) -> int:
    """Return persistent device bytes owned by attached LBC tables/weights."""
    resident = getattr(state, "_lateral_boundary_device", None)
    return 0 if resident is None else resident.device_nbytes


def lateral_boundary_reload_count(state) -> int:
    """Return successful host-to-device external interval uploads."""
    resident = getattr(state, "_lateral_boundary_device", None)
    return 0 if resident is None else resident.external_reload_count


def _resident_interval(state, interval: BoundaryInterval
                       ) -> _DeviceBoundaryInterval:
    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None:
        raise RuntimeError("lateral boundaries were not attached to the state")
    if resident.streaming_external:
        if resident.active_host_interval_id != id(interval):
            _reload_streaming_external_interval(state, interval)
        return resident.intervals[0]
    try:
        index = resident.host_interval_indices[id(interval)]
    except KeyError as exc:
        raise RuntimeError("selected lateral interval is not attached to state") \
            from exc
    return resident.intervals[index]


def _reload_streaming_external_interval(
        state, interval: BoundaryInterval) -> None:
    """Overwrite the one external-LBC device slot from immutable host data."""
    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None or not resident.streaming_external:
        raise RuntimeError("state has no streaming external LBC attachment")
    device_interval = resident.intervals[0]
    if set(interval.fields) != set(device_interval.fields):
        raise RuntimeError("streaming lateral interval field inventory changed")
    if getattr(state, "_host_setup_state", False):
        xp = np
    else:
        import cupy as cp
        xp = cp
    for name, host_field in interval.fields.items():
        device_field = device_interval.fields[name]
        for side_name in ("west", "east", "south", "north"):
            host_side = getattr(host_field, side_name)
            device_side = getattr(device_field, side_name)
            if (host_side.value.shape != device_side.value.shape or
                    host_side.tendency.shape != device_side.tendency.shape):
                raise RuntimeError(
                    f"streaming lateral interval layout changed for "
                    f"{name}/{side_name}")
            device_side.value[...] = xp.asarray(
                host_side.value, dtype=xp.float32)
            device_side.tendency[...] = xp.asarray(
                host_side.tendency, dtype=xp.float32)
    resident.active_host_interval_id = id(interval)
    resident.external_reload_count += 1


def _active_device_interval(state, cfg):
    """Return ``(device interval, dtbc, child dt, spec_exp)``."""
    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None:
        raise RuntimeError("lateral boundaries were not attached to the state")
    if getattr(cfg, "nested", False):
        if not resident.rolling or not resident.valid:
            raise RuntimeError("nested boundary tables are invalid before FORCE")
        if resident.clock is None:
            raise RuntimeError("nested boundary attachment has no child clock")
        return (resident.intervals[0], resident.clock.dtbc_launch_fp32,
                resident.clock.spec.dt_fp32, 0.0)
    boundaries = state.lateral_boundaries
    elapsed = (state.elapsed_seconds if resident.clock is None
               else resident.clock.elapsed_seconds)
    interval = boundaries.interval_at(elapsed)
    return (_resident_interval(state, interval),
            (state.elapsed_seconds - interval.start_seconds
             if resident.clock is None
             else resident.clock.dtbc_launch_fp32),
            lateral_boundary_clock_dt(cfg), cfg.spec_exp)


def _resident_weights(state, width, spec_zone, relax_zone, dt, spec_exp, *,
                      wrf_real=False):
    import cupy as cp

    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None:
        raise RuntimeError("lateral boundaries were not attached to the state")
    dt_key = np.float32(dt) if wrf_real else float(dt)
    key = (int(width), int(spec_zone), int(relax_zone), dt_key,
           float(spec_exp), bool(wrf_real))
    hit = resident.weights.get(key)
    if hit is None:
        fcx, gcx = _weights(
            key[0], key[1], key[2], key[3], key[4], wrf_real=key[5])
        slot = f"lbc_weights_{resident.next_weight_slot}"
        packed = _lbc_scratch(state, (2, int(width)), slot)
        packed[0] = cp.asarray(fcx, dtype=cp.float32)
        packed[1] = cp.asarray(gcx, dtype=cp.float32)
        hit = (packed[0], packed[1])
        resident.weights[key] = hit
        resident.scratch_slots.add(slot)
        resident.device_nbytes += int(packed.nbytes)
        resident.next_weight_slot += 1
    return hit


def apply_specified_relaxation(field, tendency, boundary: FieldBoundary, *,
                               dtbc, dt, spec_zone=1, relax_zone=4,
                               spec_exp=0.0, apply_relax=True,
                               state=None, field_name=None, weights=None,
                               clear_specified=False, add_held=None,
                               divide_msf=False, source_field=None,
                               source_mup=None):
    """Apply ``spec_bdytend`` + ``relax_bdytend_core`` to device arrays."""
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CuPy is required for specified boundaries") from exc
    if state is not None:
        supported = {"u", "v", "w", "theta", "phi", "mu", "qv", "qc",
                     "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng",
                     "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh",
                     "qnn", "qvolg", "qvolh"}
        if field_name not in supported:
            raise ValueError(f"unsupported state LBC field {field_name!r}")
        if field_name == "mu":
            shape = (1, *state.mup.shape)
        else:
            target_name = {"theta": "thp", "phi": "php"}.get(
                field_name, field_name)
            target = getattr(state, target_name)
            if target is None:
                raise ValueError(f"state has no {field_name} field")
            shape = target.shape
        if tendency.shape != shape:
            raise ValueError(
                f"{field_name} tendency has shape {tendency.shape}, expected {shape}")
        _launch_state_relaxation(
            state, field_name, tendency, boundary, dtbc=dtbc, dt=dt,
            spec_zone=spec_zone, relax_zone=relax_zone, spec_exp=spec_exp,
            apply_relax=apply_relax, weights=weights,
            clear_specified=clear_specified, add_held=add_held,
            divide_msf=divide_msf, source_field=source_field,
            source_mup=source_mup)
        return

    field = cp.asarray(field, dtype=cp.float32)
    if field.ndim != 3 or tendency.shape != field.shape:
        raise ValueError("field and tendency must have the same 3-D shape")
    nz, ny, nx = field.shape
    width = boundary.west.value.shape[-1]
    if width < spec_zone + relax_zone:
        raise ValueError("boundary width is smaller than spec_zone + relax_zone")
    sides = []
    for side in (boundary.west, boundary.east, boundary.south, boundary.north):
        sides.extend((cp.asarray(side.value, dtype=cp.float32),
                      cp.asarray(side.tendency, dtype=cp.float32)))
    if weights is None:
        fcx, gcx = _weights(width, spec_zone, relax_zone, dt, spec_exp)
        fcx = cp.asarray(fcx)
        gcx = cp.asarray(gcx)
    else:
        fcx, gcx = weights
    count = nz * ny * nx
    kernel = get_kernel("spec_bdy", "specified_relaxation")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,),
           (field, tendency, *sides, fcx, gcx, cp.float32(dtbc),
            np.int32(width), np.int32(spec_zone), np.int32(relax_zone),
            np.int32(apply_relax),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _apply_legacy_held_interior(tendency, held, msft, active_width, *,
                                divide_msf, add_held):
    """Restore legacy whole-array ufunc effects outside the CUDA frame.

    The perimeter kernel retains the existing optimized arithmetic.  These
    view ufuncs cover only the complementary interior, preserving legacy
    signed-zero and non-finite behavior without double-applying frame work.
    """
    if not (divide_msf or add_held):
        return
    ny, nx = tendency.shape[-2:]
    interior = (slice(None), slice(active_width, ny - active_width),
                slice(active_width, nx - active_width))
    if divide_msf:
        tendency[interior] /= msft[
            None, active_width:ny - active_width,
            active_width:nx - active_width]
    if add_held:
        tendency[interior] += held[interior]


def _launch_state_relaxation(state, field_name, tendency, boundary, *,
                             dtbc, dt, spec_zone, relax_zone, spec_exp,
                             apply_relax, weights, clear_specified, add_held,
                             divide_msf, source_field, source_mup):
    """Perimeter-only coupled-field relaxation without 3-D temporaries."""
    import cupy as cp

    nz, ny, nx = tendency.shape
    width = boundary.west.value.shape[-1]
    if width < spec_zone + relax_zone:
        raise ValueError("boundary width is smaller than spec_zone + relax_zone")
    if weights is None:
        fcx, gcx = _weights(width, spec_zone, relax_zone, dt, spec_exp)
        weights = (cp.asarray(fcx), cp.asarray(gcx))
    fcx, gcx = weights
    active_width = max(spec_zone, relax_zone)
    _validate_frame_domain(
        ny, nx, active_width, f"{field_name} specified/relaxation")
    frame_count = _perimeter_count(ny, nx, active_width)
    count = nz * frame_count
    sides = []
    for side in (boundary.west, boundary.east,
                 boundary.south, boundary.north):
        sides.extend((side.value, side.tendency))
    # CUDA arguments cannot be null. ``scalar`` is read only for a scalar
    # kind; the theta pointer is a harmless sentinel for dry calls.
    target_name = {"theta": "thp", "phi": "php"}.get(
        field_name, field_name)
    source = (getattr(state, target_name, None) if source_field is None
              else source_field)
    scalar = source
    if scalar is None:
        scalar = state.thp
    u = source if field_name == "u" else state.u
    v = source if field_name == "v" else state.v
    w = source if field_name == "w" else state.w
    thp = source if field_name == "theta" else state.thp
    php = source if field_name == "phi" else state.php
    held = tendency if add_held is None else add_held
    mass_perturbation = state.mup if source_mup is None else source_mup
    kind = {"u": 0, "v": 1, "theta": 2, "phi": 3, "mu": 4,
            "qv": 5, "w": 6}.get(field_name, 7)
    kernel = get_kernel("lbc_state", "state_specified_relaxation")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        tendency, held, state.mub2d, mass_perturbation, u, v, w,
        thp, state.thb, php, scalar,
        state.c1h, state.c2h, state.c1f, state.c2f,
        state.msft, state.msfu, state.msfv, *sides, fcx, gcx,
        cp.float32(dtbc),
        np.int32(width), np.int32(spec_zone), np.int32(relax_zone),
        np.int32(apply_relax), np.int32(clear_specified),
        np.int32(add_held is not None), np.int32(divide_msf),
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(kind), np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(frame_count)))
    _apply_legacy_held_interior(
        tendency, held, state.msft, active_width,
        divide_msf=divide_msf, add_held=(add_held is not None))


def _coupled_device_fields(state):
    """Fields in the same coupled units as WRF boundary tendencies."""
    from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces

    mu = state.total_mu()
    mux = mu_at_u_faces(mu)
    muy = mu_at_v_faces(mu)
    # Specified boundary faces use their adjacent boundary cell's mass, not
    # the periodic seam average.
    mux[:, 0] = mu[:, 0]
    mux[:, -1] = mu[:, -1]
    muy[0, :] = mu[0, :]
    muy[-1, :] = mu[-1, :]
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    c1f = state.c1f[:, None, None]
    c2f = state.c2f[:, None, None]
    chm = c1h * mu[None] + c2h
    chf = c1f * mu[None] + c2f
    result = {
        "u": (c1h * mux[None] + c2h) * state.u,
        "v": (c1h * muy[None] + c2h) * state.v,
        # WRF module_bc.F:2081,2139-2142: boundary scalars use mu-only
        # coupling; 1/msf belongs only to u/v/w.  T is perturbation theta.
        "theta": chm * (state.total_theta() - _THETA_OFFSET_K),
        "phi": chf * state.php,
        "mu": state.mup[None],
    }
    if state.has_msf:
        result["u"] /= state.msfu[None]
        result["v"] /= state.msfv[None]
    # WRF specified-domain moisture policy without hydrometeor boundary
    # arrays: only water vapor is carried by the external LBC snapshots.
    # Condensate species use flow_dep_bdy after their RK scalar update.
    if state.qv is not None:
        result["qv"] = chm * state.qv
    return result


def couple_nest_field(state, field_name: str, *, out):
    """Write one full field in WRF coupled units into preallocated ``out``.

    This is the force-time counterpart of ``_coupled_device_fields`` with
    the ratified extensions: w uses full-level c1f/c2f mass weight and
    1/msft, while every active hydrometeor/number scalar uses half-level
    mu coupling.  No source state is mutated and no device temporary is
    allocated.  Authority: WRF v4.6.1
    ``dyn_em/couple_or_uncouple_em.F:119-166`` and
    ``dyn_em/module_bc_em.F:320-345``.
    """
    target_name = {"t": "thp", "ph": "php"}.get(field_name, field_name)
    target = state.mup[None] if field_name == "mu" else getattr(
        state, target_name, None)
    if target is None:
        raise ValueError(f"state has no active nest field {field_name!r}")
    expected = tuple(int(n) for n in target.shape)
    if tuple(out.shape) != expected:
        raise ValueError(
            f"nest coupled output for {field_name} has shape {out.shape}, "
            f"expected {expected}")
    scalar_names = {"qv", "qc", "qr", "qi", "qs", "qg",
                    "nr", "ni", "ns", "ng", "qh", "qndrop", "qnr",
                    "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh"}
    if field_name not in {"u", "v", "w", "t", "ph", "mu"} | scalar_names:
        raise ValueError(f"unsupported nest coupling field {field_name!r}")
    kind = {"u": 0, "v": 1, "t": 2, "ph": 3, "mu": 4,
            "qv": 5, "w": 6}.get(field_name, 7)
    nz, ny, nx = expected
    mny, mnx = state.mup.shape
    count = int(out.size)
    kernel = get_kernel("lbc_state", "couple_nest_field")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        target, state.mub2d, state.mup, state.thb,
        state.c1h, state.c2h, state.c1f, state.c2f,
        state.msft, state.msfu, state.msfv, out,
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(kind), np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(mny), np.int32(mnx)))
    return out


def uncouple_feedback_field(state, field_name: str, coupled, reg, *,
                            spec_zone=1):
    """Write one restricted coupled field into the parent prognostic state.

    ``coupled`` already contains ``copy_fcn`` output in the parent's field
    geometry.  Only the exact WRF feedback rectangle is uncoupled; all cells
    outside it remain byte-untouched.  MU is restricted directly and is not
    accepted here, so momentum/scalar inverses always see the already
    restricted current parent mass.
    """
    from gpuwm.core.nest_interp import feedback_parent_bounds

    if field_name == "mu":
        raise ValueError("feedback MU is restricted directly")
    target_name = {"t": "thp", "ph": "php"}.get(field_name, field_name)
    target = getattr(state, target_name, None)
    if target is None:
        raise ValueError(f"state has no active feedback field {field_name!r}")
    if tuple(coupled.shape) != tuple(target.shape):
        raise ValueError(
            f"coupled feedback field {field_name} has shape "
            f"{tuple(coupled.shape)}, expected {tuple(target.shape)}")
    scalar_names = {"qv", "qc", "qr", "qi", "qs", "qg",
                    "nr", "ni", "ns", "ng", "qh", "qndrop", "qnr",
                    "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh"}
    if field_name not in {"u", "v", "w", "t", "ph"} | scalar_names:
        raise ValueError(f"unsupported feedback field {field_name!r}")
    kind = {"u": 0, "v": 1, "t": 2, "ph": 3,
            "qv": 5, "w": 6}.get(field_name, 7)
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(
        reg, spec_zone=spec_zone)
    ni = max(0, i_hi - i_lo + 1)
    nj = max(0, j_hi - j_lo + 1)
    if not ni or not nj:
        return target
    nz, ny, nx = target.shape
    mny, mnx = state.mup.shape
    count = nz * nj * ni
    kernel = get_kernel("lbc_state", "uncouple_feedback_field")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        target, coupled, state.mub2d, state.mup, state.thb,
        state.c1h, state.c2h, state.c1f, state.c2f,
        state.msft, state.msfu, state.msfv,
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(kind), np.int32(i_lo), np.int32(i_hi),
        np.int32(j_lo), np.int32(j_hi),
        np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(mny), np.int32(mnx)))
    return target


def domain_boundary_snapshot(state) -> Mapping[str, np.ndarray]:
    """Return a float64 coupled snapshot suitable for the LBC builder."""
    return MappingProxyType({
        name: _host(value) for name, value in _coupled_device_fields(state).items()
    })


def extract_lateral_side(snapshot: Mapping[str, object], side: str,
                         width: int) -> Mapping[str, np.ndarray]:
    """Extract one side exactly as :func:`build_lateral_boundaries` does.

    A caller may initialize only a narrow W/E/S/N rectangle, couple its
    fields, and retain the requested side instead of materializing a complete
    analysis-domain state.  East/north are outermost-first, matching WRF's
    boundary-file layout and the established full-domain builder.
    """
    if side not in {"west", "east", "south", "north"}:
        raise ValueError(f"unknown lateral side {side!r}")
    width = int(width)
    if width < 1:
        raise ValueError("lateral side width must be positive")
    result = {}
    for name in sorted(snapshot):
        value = _host(snapshot[name])
        if value.ndim == 2:
            value = value[None]
        if value.ndim != 3:
            raise ValueError(
                f"boundary snapshot field {name!r} must be 2-D or 3-D")
        if side in {"west", "east"} and value.shape[-1] < width:
            raise ValueError(
                f"boundary snapshot field {name!r} is narrower than {width}")
        if side in {"south", "north"} and value.shape[-2] < width:
            raise ValueError(
                f"boundary snapshot field {name!r} is shorter than {width}")
        if side == "west":
            selected = value[..., :width]
        elif side == "east":
            selected = value[..., -width:][..., ::-1]
        elif side == "south":
            selected = value[..., :width, :]
        else:
            selected = value[..., -width:, :][..., ::-1, :]
        result[name] = np.ascontiguousarray(selected)
    if not result:
        raise ValueError("boundary snapshot field inventory is empty")
    return MappingProxyType(result)


def build_lateral_interval_from_sides(
        first: Mapping[str, Mapping[str, object]],
        second: Mapping[str, Mapping[str, object]], *,
        start_seconds: float, end_seconds: float) -> BoundaryInterval:
    """Assemble one exact interval from W/E/S/N side-only snapshots."""
    side_names = ("west", "east", "south", "north")
    if set(first) != set(side_names) or set(second) != set(side_names):
        raise ValueError("side snapshots must contain west/east/south/north")
    start = float(start_seconds)
    end = float(end_seconds)
    duration = end - start
    if not np.isfinite(start) or not np.isfinite(end) or duration <= 0.0:
        raise ValueError("boundary interval times must be finite and increasing")
    inventory = set(first["west"])
    if not inventory:
        raise ValueError("side snapshot field inventory is empty")
    for collection in (first, second):
        for side in side_names:
            if set(collection[side]) != inventory:
                raise ValueError("side snapshot field inventories differ")
    fields = {}
    for name in sorted(inventory):
        packed = {}
        for side in side_names:
            before = _host(first[side][name])
            after = _host(second[side][name])
            if before.shape != after.shape:
                raise ValueError(
                    f"boundary side {side}/{name} shapes differ")
            packed[side] = SideBoundary(
                np.ascontiguousarray(before),
                np.ascontiguousarray((after - before) / duration))
        fields[name] = FieldBoundary(**packed)
    return BoundaryInterval(start, end, fields)


def build_state_lateral_boundaries(states, times, *, spec_bdy_width=5,
                                   spec_zone=1, relax_zone=4):
    """Build WRF boundary arrays from initialized DomainState snapshots."""
    return build_lateral_boundaries(
        [domain_boundary_snapshot(state) for state in states], times,
        spec_bdy_width=spec_bdy_width, spec_zone=spec_zone,
        relax_zone=relax_zone)


def _validate_lateral_attachment(state, boundaries: LateralBoundaries) -> None:
    """Validate external forcing geometry against one target state."""
    if not boundaries.intervals:
        raise ValueError("at least one lateral-boundary interval is required")
    if boundaries.spec_zone < 1 or boundaries.relax_zone < 2:
        raise ValueError("spec_zone must be >=1 and relax_zone >=2")
    if boundaries.spec_bdy_width < (
            boundaries.spec_zone + boundaries.relax_zone):
        raise ValueError("spec_bdy_width must cover spec_zone + relax_zone")
    required = {"u", "v", "theta", "phi", "mu"}
    for interval in boundaries.intervals:
        missing = required - set(interval.fields)
        if missing:
            raise ValueError(f"state boundary interval is missing {sorted(missing)}")

        for name, boundary in interval.fields.items():
            nz, ny, nx, width = _boundary_field_shape(boundary)
            if width != boundaries.spec_bdy_width:
                raise ValueError(
                    f"{name} boundary width {width} does not match "
                    f"spec_bdy_width {boundaries.spec_bdy_width}")
            _validate_frame_domain(
                ny, nx, max(boundaries.spec_zone, boundaries.relax_zone),
                f"{name} specified/relaxation")
            if name == "mu":
                target = getattr(state, "mup", None)
                expected = None if target is None else (1, *target.shape)
            else:
                target_name = {"theta": "thp", "phi": "php"}.get(name, name)
                target = getattr(state, target_name, None)
                expected = None if target is None else target.shape
            if expected is not None and (nz, ny, nx) != expected:
                raise ValueError(
                    f"{name} boundary shape {(nz, ny, nx)} does not match "
                    f"state shape {expected}")

def attach_lateral_boundaries(state, boundaries: LateralBoundaries) -> None:
    """Attach validated, immutable specified forcing to a model state.

    All forcing intervals are uploaded eagerly into a packed persistent
    ``DomainState.scratch`` buffer.  This intentionally makes setup account
    for the complete forcing set and moves conversion/allocation failures to
    attach time.  Davies-weight buffers join the same accounting when first
    used.  Reattachment releases those registered buffers and rebuilds the
    mirrors.  A copied or deserialized state must likewise call this function
    to establish its device-resident attachment.
    """
    _validate_lateral_attachment(state, boundaries)
    total_values = sum(
        array.size
        for interval in boundaries.intervals
        for boundary in interval.fields.values()
        for side in (boundary.west, boundary.east,
                     boundary.south, boundary.north)
        for array in (side.value, side.tendency)
    )
    _release_resident_scratch(state)
    forcing_slot = "lbc_forcing_tables"
    packed = _lbc_scratch(state, (total_values,), forcing_slot)
    if getattr(state, "_host_setup_state", False):
        xp = np
    else:
        import cupy as cp
        xp = cp
    offset = 0

    def upload(host):
        nonlocal offset
        size = host.size
        device = packed[offset:offset + size].reshape(host.shape)
        # Preserve the former direct NumPy -> CuPy FP32 conversion exactly;
        # the temporary is copied into state-owned registered storage once.
        device[...] = xp.asarray(host, dtype=xp.float32)
        offset += size
        return device

    device_intervals = []
    for interval in boundaries.intervals:
        fields = {}
        for name, boundary in interval.fields.items():
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                side = getattr(boundary, side_name)
                # This is deliberately the same dtype path formerly executed
                # in every hot call: NumPy float64 -> CuPy float32.
                sides[side_name] = _DeviceSideBoundary(
                    upload(side.value), upload(side.tendency))
            fields[name] = _DeviceFieldBoundary(**sides)
        device_intervals.append(_DeviceBoundaryInterval(
            MappingProxyType(fields)))
    state._lateral_boundary_device = _DeviceLateralBoundaries(
        tuple(device_intervals),
        MappingProxyType({id(interval): index for index, interval in
                          enumerate(boundaries.intervals)}),
        {}, {forcing_slot}, int(packed.nbytes))
    state.lateral_boundaries = boundaries
    state.elapsed_seconds = 0.0


def attach_streaming_lateral_boundaries(
        state, boundaries: LateralBoundaries) -> None:
    """Attach one reusable device mirror for an external forcing series.

    Host intervals remain immutable and authoritative, but only the interval
    selected by model time is resident on the device. Crossing an interval
    boundary overwrites the same packed scratch allocation before the next
    specified-boundary kernel launch. This opt-in path is intended for long
    offline-child archives where eager residency scales as
    ``interval_count * perimeter``; :func:`attach_lateral_boundaries` retains
    its established eager behavior.
    """
    _validate_lateral_attachment(state, boundaries)
    first = boundaries.intervals[0]
    inventory = tuple(first.fields)

    def layout(interval):
        return tuple(
            (name,) + _boundary_field_shape(interval.fields[name])
            for name in inventory)

    reference_layout = layout(first)
    for interval in boundaries.intervals[1:]:
        if (tuple(interval.fields) != inventory or
                layout(interval) != reference_layout):
            raise ValueError(
                "streaming lateral intervals must have identical inventories "
                "and side layouts")
    total_values = sum(
        array.size
        for boundary in first.fields.values()
        for side in (boundary.west, boundary.east,
                     boundary.south, boundary.north)
        for array in (side.value, side.tendency)
    )
    _release_resident_scratch(state)
    forcing_slot = "lbc_forcing_tables"
    packed = _lbc_scratch(state, (total_values,), forcing_slot)
    offset = 0

    def view(host):
        nonlocal offset
        size = host.size
        result = packed[offset:offset + size].reshape(host.shape)
        offset += size
        return result

    fields = {}
    for name, boundary in first.fields.items():
        sides = {}
        for side_name in ("west", "east", "south", "north"):
            side = getattr(boundary, side_name)
            sides[side_name] = _DeviceSideBoundary(
                view(side.value), view(side.tendency))
        fields[name] = _DeviceFieldBoundary(**sides)
    device_interval = _DeviceBoundaryInterval(MappingProxyType(fields))
    state._lateral_boundary_device = _DeviceLateralBoundaries(
        (device_interval,), MappingProxyType({}), {}, {forcing_slot},
        int(packed.nbytes), streaming_external=True,
        packed_forcing=packed)
    state.lateral_boundaries = boundaries
    state.elapsed_seconds = 0.0
    _reload_streaming_external_interval(state, first)


def bind_lateral_boundary_clock(state, clock) -> None:
    """Bind T9's integer/FP32 clock to an existing external LBC mirror.

    PRODUCTION CALLERS (Davies clock bind, 2026-07-28; retires the F20
    adjudication): ``gpuwm.core.model.build_experiment`` binds the root
    immediately after the root ``DomainNode`` is constructed, and the
    N5S restored-model builder (``real74_n5s.build_restored_model``)
    binds its manually constructed root after attachment.  Binding
    switches every external Davies consumer -- relaxation, the held
    moist/scalar targets, and the final specified-ring overwrite -- to
    WRF's post-increment ``dtbc_launch_fp32`` recurrence
    (dyn_em/solve_em.F:371-372) with interval selection from the clock's
    solve-entry time.  No extra reset is added here: the executor
    already zeroes the root dtbc at every external interval seam
    (share/mediation_integrate.F:1522 position) and runs
    ``prepare_step()`` before each solve.  Unbound mirrors (legacy
    era5/gfs direct paths without a DomainClock) keep the established
    ``state.elapsed_seconds`` compatibility calculation bit-for-bit;
    restart headers record which semantic integrated a checkpoint
    (``gpuwm/io/restart.py`` ``root_external_lbc_clock``) so the two can
    never be silently mixed.
    """
    resident = getattr(state, "_lateral_boundary_device", None)
    if resident is None or resident.rolling:
        raise RuntimeError("external lateral boundaries must be attached first")
    resident.clock = clock


def attach_nest_boundaries(state, fields: Mapping[str, Mapping[str, tuple]],
                           *, clock, spec_bdy_width=5, spec_zone=1,
                           relax_zone=4) -> None:
    """Attach one rolling child-owned device frame without host copies.

    ``fields`` is the direct ``bdy_interp1`` output mapping.  FORCE refreshes
    those same manifest-backed arrays in place; this function installs only
    lookup metadata and preserves the Davies-weight cache across forces.
    """
    if spec_zone < 1 or relax_zone < 2:
        raise ValueError("spec_zone must be >=1 and relax_zone >=2")
    if spec_bdy_width < spec_zone + relax_zone:
        raise ValueError("spec_bdy_width must cover spec_zone + relax_zone")
    frame_width = min(max(int(spec_zone), int(relax_zone) + 1),
                      int(spec_bdy_width))
    required = {"u", "v", "w", "theta", "phi", "mu"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"nest boundary frame is missing {sorted(missing)}")
    converted = {}
    for name, sides in fields.items():
        if set(sides) != {"west", "east", "south", "north"}:
            raise ValueError(f"{name} nest boundary side inventory differs")
        device_sides = {}
        for side_name in ("west", "east", "south", "north"):
            value, tendency = sides[side_name]
            if value.shape != tendency.shape:
                raise ValueError(f"{name}/{side_name} value/tendency differ")
            device_sides[side_name] = _DeviceSideBoundary(value, tendency)
        boundary = _DeviceFieldBoundary(**device_sides)
        nz, ny, nx, width = _boundary_field_shape(boundary)
        if width != frame_width:
            raise ValueError(
                f"{name} nest width {width} != bdy_interp1 width "
                f"{frame_width}")
        target_name = {"theta": "thp", "phi": "php"}.get(name, name)
        target = state.mup[None] if name == "mu" else getattr(
            state, target_name, None)
        if target is None or tuple(target.shape) != (nz, ny, nx):
            raise ValueError(f"{name} nest frame does not match child state")
        converted[name] = boundary

    existing = getattr(state, "_lateral_boundary_device", None)
    if existing is not None and existing.rolling:
        weights = existing.weights
        scratch_slots = existing.scratch_slots
        next_weight_slot = existing.next_weight_slot
        device_nbytes = existing.device_nbytes
    else:
        weights, scratch_slots, next_weight_slot, device_nbytes = {}, set(), 0, 0
    state._lateral_boundary_device = _DeviceLateralBoundaries(
        (_DeviceBoundaryInterval(MappingProxyType(converted)),),
        MappingProxyType({}), weights, scratch_slots, device_nbytes,
        next_weight_slot=next_weight_slot, rolling=True, clock=clock,
        valid=True)
    state.lateral_boundaries = RollingNestBoundaries(
        frame_width, int(spec_zone), int(relax_zone))


def apply_state_lateral_boundaries(state, cfg, *, rk_stage: int) -> None:
    """Apply WRF dry specified/relaxation tendencies for one RK stage.

    WRF computes u/v/theta/phi relaxation once on stage 1 into held
    ``*_save`` tendencies and adds those fixed increments on all three RK
    stages.  Mu relaxation is stage-1-only.  ``spec_bdytend`` still replaces
    the outer specified-zone tendency on every stage for every dry field.
    """
    boundaries = state.lateral_boundaries
    if not (cfg.specified or cfg.nested):
        return
    if rk_stage not in (0, 1, 2):
        raise ValueError("rk_stage must be 0, 1, or 2")
    if boundaries is None:
        raise RuntimeError(
            "cfg.specified=True requires attach_lateral_boundaries(state, ...)")
    device_interval, dtbc, dt, spec_exp = _active_device_interval(state, cfg)
    held_rows = [
        ("u", state.ru_t), ("v", state.rv_t),
        ("theta", state.rth_t), ("phi", state.rph_t),
    ]
    common = dict(
        dtbc=dtbc, dt=dt,
        spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone, spec_exp=spec_exp)
    weights = _resident_weights(
        state, device_interval.fields["u"].west.value.shape[-1],
        cfg.spec_zone, cfg.relax_zone, common["dt"], spec_exp,
        wrf_real=bool(cfg.nested))
    if cfg.specified:
        for name, tendency in held_rows:
            held = state.scratch(tendency.shape, "lbc_relax_" + name)
            if rk_stage == 0:
                held[...] = 0
                # Preserve the frozen d01 held-tendency path byte-for-byte.
                apply_specified_relaxation(
                    getattr(state, {"theta": "thp", "phi": "php"}.get(
                        name, name)), held, device_interval.fields[name],
                    **common, apply_relax=True, state=state, field_name=name,
                    weights=weights, clear_specified=True,
                    divide_msf=(state.has_msf and name in ("theta", "phi")))
            apply_specified_relaxation(
                getattr(state, {"theta": "thp", "phi": "php"}.get(
                    name, name)), tendency, device_interval.fields[name],
                **common, apply_relax=False, state=state, field_name=name,
                weights=weights, add_held=held)
    else:
        # Nested tables are rolling and all RK time-t copies already exist.
        # Re-evaluate the same held increment from those copies on each stage
        # into one force/launch-local arena temporary.  Its backing aliases
        # acoustic_a when that W-shaped slot has enough capacity; otherwise
        # preflight retains an explicit bounded backing for valid skinny
        # grids where U or V is larger than W.
        sources = {"u": state.u0, "v": state.v0, "w": state.w0,
                   "theta": state.thp0, "phi": state.php0}
        nested_rows = held_rows + [("w", state.rw_t)]
        relax_capacity = max(int(tendency.size)
                             for _name, tendency in nested_rows)
        backing = state.scratch((relax_capacity,), "lbc_nested_relax")
        for name, tendency in nested_rows:
            count = tendency.size
            if count > backing.size:
                raise RuntimeError("nested LBC temporary is too small")
            held = backing.reshape(-1)[:count].reshape(tendency.shape)
            held[...] = 0
            apply_specified_relaxation(
                getattr(state, {"theta": "thp", "phi": "php"}.get(
                    name, name)), held, device_interval.fields[name],
                **common, apply_relax=True, state=state, field_name=name,
                weights=weights, source_field=sources[name],
                source_mup=state.mup0,
                clear_specified=True,
                divide_msf=(state.has_msf and name in ("theta", "phi")))
            apply_specified_relaxation(
                getattr(state, {"theta": "thp", "phi": "php"}.get(
                    name, name)), tendency, device_interval.fields[name],
                **common, apply_relax=False, state=state, field_name=name,
                weights=weights, add_held=held)

    apply_specified_relaxation(
        state.mup[None], state.rmu_t[None], device_interval.fields["mu"],
        **common, apply_relax=(rk_stage == 0), state=state, field_name="mu",
        weights=weights)


def apply_state_qv_lateral_boundary(state, cfg, tendency, *,
                                    apply_relax=True) -> None:
    """Compatibility wrapper for the qv scalar forcing path."""
    return apply_state_scalar_lateral_boundary(
        state, cfg, "qv", tendency, apply_relax=apply_relax)


def apply_state_scalar_lateral_boundary(state, cfg, field_name, tendency, *,
                                        apply_relax=True,
                                        source_field=None) -> None:
    """Apply one resident moist/scalar boundary tendency."""
    if state.lateral_boundaries is None:
        raise RuntimeError("boundary-forced state has no lateral forcing")
    device_interval, dtbc, dt, spec_exp = _active_device_interval(state, cfg)
    if field_name not in device_interval.fields:
        return
    weights = _resident_weights(
        state, device_interval.fields[field_name].west.value.shape[-1],
        cfg.spec_zone, cfg.relax_zone, dt, spec_exp,
        wrf_real=bool(cfg.nested))
    apply_specified_relaxation(
        getattr(state, field_name), tendency, device_interval.fields[field_name],
        dtbc=dtbc, dt=dt,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone,
        spec_exp=spec_exp, apply_relax=apply_relax,
        state=state, field_name=field_name, weights=weights,
        source_field=source_field,
        source_mup=(state.mup0 if cfg.nested else None))


def _launch_mu_boundary_values(state, boundary, dtbc, spec_zone):
    """Install MU and retain its old boundary frame for exact uncoupling."""
    import cupy as cp

    ny, nx = state.mup.shape
    frame_count = _perimeter_count(ny, nx, spec_zone)
    old_frame = state.scratch(
        (frame_count,), f"lbc_old_mup_frame_{spec_zone}")
    sides = []
    for side in (boundary.west, boundary.east,
                 boundary.south, boundary.north):
        sides.extend((side.value, side.tendency))
    kernel = get_kernel("lbc_state", "install_mu_boundary")
    kernel(((frame_count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        state.mup, old_frame, *sides, cp.float32(dtbc),
        np.int32(boundary.west.value.shape[-1]), np.int32(spec_zone),
        np.int32(ny), np.int32(nx), np.int32(frame_count)))
    return old_frame


def _launch_finalize_field(state, name, boundary, old_mup_frame, dtbc,
                           spec_zone):
    """Fused exact old-couple/install/new-uncouple transformation."""
    import cupy as cp

    target_name = {"theta": "thp", "phi": "php"}.get(name, name)
    target = getattr(state, target_name)
    nz, ny, nx = target.shape
    count = nz * ny * nx
    sides = []
    for side in (boundary.west, boundary.east,
                 boundary.south, boundary.north):
        sides.extend((side.value, side.tendency))
    scalar = target
    kind = {"u": 0, "v": 1, "theta": 2, "phi": 3,
            "qv": 5, "w": 6}.get(name, 7)
    kernel = get_kernel("lbc_state", "finalize_state_field")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        target, old_mup_frame, state.mub2d, state.mup,
        state.thb, scalar, state.c1h, state.c2h, state.c1f, state.c2f,
        state.msft, state.msfu, state.msfv, *sides, cp.float32(dtbc),
        np.int32(boundary.west.value.shape[-1]), np.int32(spec_zone),
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(kind), np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(state.mup.shape[0]), np.int32(state.mup.shape[1])))


def apply_state_boundary_values(state, cfg, elapsed_seconds=None) -> None:
    """WRF ``spec_bdy_final``: force prognostics back to boundary values.

    A BOUND root (production: the tree build assigns the root
    ``DomainClock`` to the external mirror) takes the same
    ``_active_device_interval`` selection as every other bound consumer:
    the interval is chosen from the clock's solve-entry time and ``dtbc``
    is the step-constant post-increment value (dyn_em/solve_em.F:371-372,
    :4531-4639).  On the last step of a boundary interval that is the OLD
    record at ``dtbc = T_bdy`` -- WRF's seam record ownership; the new
    record is read only at the top of the following step
    (frame/module_integrate.F:393-396, share/mediation_integrate.F:
    1431-1471).  The half-open end-of-step ``interval_at`` lookup (new
    record at ``dtbc = 0``: equal-valued for continuous tables but not
    FP32-bit-identical to the old endpoint reconstruction) remains only
    as the UNBOUND compatibility fallback for direct paths without a
    DomainClock.
    """
    if not (cfg.specified or cfg.nested):
        return
    boundaries = state.lateral_boundaries
    if boundaries is None:
        raise RuntimeError("specified state has no lateral boundary forcing")
    resident = getattr(state, "_lateral_boundary_device", None)
    if cfg.nested or (resident is not None and resident.clock is not None):
        device_interval, dtbc, _, _ = _active_device_interval(state, cfg)
    else:
        t = (state.elapsed_seconds if elapsed_seconds is None
             else float(elapsed_seconds))
        interval = boundaries.interval_at(t)
        device_interval = _resident_interval(state, interval)
        dtbc = t - interval.start_seconds
    # The legacy path coupled every full field using old MU, installed MU,
    # then uncoupled every full field using new MU.  The fused finalizers
    # retain that roundoff-visible operation order without materializing the
    # five coupled 3-D arrays.
    old_mup_frame = _launch_mu_boundary_values(
        state, device_interval.fields["mu"], dtbc, cfg.spec_zone)
    names = (("u", "v", "theta", "phi", "qv") if cfg.specified else
             tuple(name for name in device_interval.fields if name != "mu"))
    for name in names:
        if name not in device_interval.fields or (
                name == "qv" and state.qv is None):
            continue
        _launch_finalize_field(
            state, name, device_interval.fields[name], old_mup_frame,
            dtbc, cfg.spec_zone)


def _raw_float32_c_arrays(arrays, cp) -> bool:
    """Whether raw ``real*`` CUDA kernels may safely index every operand."""
    return all(isinstance(array, cp.ndarray)
               and array.dtype == np.dtype(np.float32)
               and bool(array.flags.c_contiguous)
               for array in arrays)


def _apply_flow_dependent_boundary_generic(field, u_flux, v_flux,
                                           spec_zone, cp) -> None:
    """Stride/dtype-aware compatibility path from the original public API."""
    nz, ny, nx = field.shape
    zero = cp.zeros((), dtype=field.dtype)
    for d in range(spec_zone):
        cols = cp.arange(d, nx - d)
        inner_i = cp.clip(cols, spec_zone, nx - 1 - spec_zone)
        field[:, d, cols] = cp.where(
            v_flux[:, d, cols] < 0.0,
            field[:, spec_zone, inner_i], zero)
        jn = ny - 1 - d
        field[:, jn, cols] = cp.where(
            v_flux[:, jn + 1, cols] > 0.0,
            field[:, ny - 1 - spec_zone, inner_i], zero)

        rows = cp.arange(d + 1, ny - d - 1)
        if rows.size:
            inner_j = cp.clip(rows, spec_zone, ny - 1 - spec_zone)
            field[:, rows, d] = cp.where(
                u_flux[:, rows, d] < 0.0,
                field[:, inner_j, spec_zone], zero)
            ie = nx - 1 - d
            field[:, rows, ie] = cp.where(
                u_flux[:, rows, ie + 1] > 0.0,
                field[:, inner_j, nx - 1 - spec_zone], zero)


def apply_flow_dependent_boundaries(fields, u_flux, v_flux, spec_zone) -> None:
    """Batched WRF ``flow_dep_bdy`` for unstaggered hydrometeor fields.

    Outflow copies the first interior row/column (zero gradient); inflow is
    zero because no external hydrometeor boundary data exist.  Y sides own
    the four corners exactly as in ``share/module_bc.F``.
    """
    import cupy as cp

    fields = tuple(fields)
    if not fields or len(fields) > 9:
        raise ValueError("flow-dependent boundary batch needs 1..9 fields")
    if fields[0].ndim != 3:
        raise ValueError("flow-dependent boundary fields must be 3-D")
    nz, ny, nx = fields[0].shape
    if any(field.shape != (nz, ny, nx) for field in fields):
        raise ValueError("flow-dependent boundary fields must share a shape")
    if u_flux.shape != (nz, ny, nx + 1):
        raise ValueError("u_flux must have shape (nz, ny, nx + 1)")
    if v_flux.shape != (nz, ny + 1, nx):
        raise ValueError("v_flux must have shape (nz, ny + 1, nx)")
    if spec_zone < 1 or min(ny, nx) <= 2 * spec_zone:
        raise ValueError("spec_zone leaves no flow-dependent boundary interior")
    if not _raw_float32_c_arrays((*fields, u_flux, v_flux), cp):
        for field in fields:
            _apply_flow_dependent_boundary_generic(
                field, u_flux, v_flux, spec_zone, cp)
        return
    padded = fields + (fields[-1],) * (9 - len(fields))
    frame_count = _perimeter_count(ny, nx, spec_zone)
    count = nz * frame_count
    kernel = get_kernel("lbc_flow", "flow_dependent_batch")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        *padded, u_flux, v_flux, np.int32(len(fields)),
        np.int32(spec_zone), np.int32(nz), np.int32(ny), np.int32(nx),
        np.int32(frame_count)))


def apply_flow_dependent_boundary(field, u_flux, v_flux, spec_zone) -> None:
    """Single-field compatibility wrapper for :func:`apply_flow_dependent_boundaries`."""
    apply_flow_dependent_boundaries((field,), u_flux, v_flux, spec_zone)


def _apply_specified_w_zero_gradient_generic(w, spec_zone, cp) -> None:
    """Stride/dtype-aware compatibility path from the original public API."""
    _, ny, nx = w.shape
    source_i = cp.clip(cp.arange(nx), spec_zone, nx - 1 - spec_zone)
    for d in range(spec_zone):
        lo, hi = d, -1 - d
        inner_lo, inner_hi = spec_zone, -1 - spec_zone
        cols = source_i[d:nx - d]
        w[:, lo, d:nx - d] = w[:, inner_lo, cols]
        w[:, hi, d:nx - d] = w[:, inner_hi, cols]
        if ny - d - 1 > d + 1:
            rows = cp.arange(d + 1, ny - d - 1)
            inner_j = cp.clip(rows, spec_zone,
                              ny - 1 - spec_zone)
            w[:, rows, lo] = w[:, inner_j, inner_lo]
            w[:, rows, hi] = w[:, inner_j, inner_hi]


def apply_specified_w_zero_gradient(state, cfg, field=None) -> None:
    """Apply d01's zero-gradient w rule; nested children use real w tables.

    The nested slow tendency and stage-final state are handled by
    ``apply_state_lateral_boundaries``/``apply_state_boundary_values``.
    Applying d01's zero-gradient rule here would erase that forcing.
    """
    if getattr(cfg, "nested", False):
        return
    if not cfg.specified:
        return
    import cupy as cp

    w = state.w if field is None else field
    nz, ny, nx = w.shape
    if (ny, nx) != (cfg.ny, cfg.nx):
        raise ValueError("w boundary field horizontal shape does not match cfg")
    if cfg.spec_zone < 1 or min(ny, nx) <= 2 * cfg.spec_zone:
        raise ValueError("spec_zone leaves no zero-gradient boundary interior")
    if not _raw_float32_c_arrays((w,), cp):
        _apply_specified_w_zero_gradient_generic(w, cfg.spec_zone, cp)
        return
    frame_count = _perimeter_count(ny, nx, cfg.spec_zone)
    count = nz * frame_count
    kernel = get_kernel("lbc_flow", "zero_gradient_w")
    kernel(((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
        w, np.int32(cfg.spec_zone), np.int32(nz), np.int32(ny),
        np.int32(nx), np.int32(frame_count)))


__all__ = ["BoundaryInterval", "FieldBoundary", "LateralBoundaries",
           "RollingNestBoundaries", "SideBoundary",
           "apply_flow_dependent_boundary",
           "apply_flow_dependent_boundaries",
           "apply_specified_relaxation",
           "apply_specified_w_zero_gradient", "apply_state_boundary_values",
           "apply_state_lateral_boundaries",
           "apply_state_qv_lateral_boundary",
           "apply_state_scalar_lateral_boundary",
           "attach_lateral_boundaries", "attach_nest_boundaries",
           "attach_streaming_lateral_boundaries",
           "build_lateral_boundaries", "build_state_lateral_boundaries",
           "build_lateral_interval_from_sides", "extract_lateral_side",
           "couple_nest_field", "domain_boundary_snapshot",
           "lateral_boundary_clock_dt", "lateral_boundary_reload_count",
           "lateral_boundary_resident_bytes"]
