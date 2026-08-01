"""Phase 3 Task 8: WRF specified/relaxation lateral boundaries."""

import sys
import inspect
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ingest.lateral_bc import (
    BoundaryInterval,
    StateBoundaryFrames,
    FieldBoundary,
    LateralBoundaries,
    SideBoundary,
    _apply_legacy_held_interior,
    _apply_specified_w_zero_gradient_generic,
    _perimeter_count,
    _resident_interval,
    _resident_weights,
    _weights,
    apply_flow_dependent_boundary,
    apply_flow_dependent_boundaries,
    apply_specified_relaxation,
    apply_specified_w_zero_gradient,
    apply_state_boundary_values,
    apply_state_lateral_boundaries,
    attach_lateral_boundaries,
    attach_streaming_lateral_boundaries,
    build_lateral_boundaries,
    build_lateral_interval_from_sides,
    build_state_lateral_boundaries,
    domain_boundary_snapshot,
    extract_lateral_side,
    lateral_boundary_resident_bytes,
)
from gpuwm.verify.npref import (np_flow_dependent_boundary,
                                np_rk_specified_relaxation,
                                np_specified_relaxation)


def _snapshots():
    nz, ny, nx = 3, 10, 12
    base = np.arange(nz * ny * nx, dtype=np.float64).reshape(nz, ny, nx)
    return [{"theta": base + offset} for offset in (0.0, 12.0, 30.0)]


def test_side_only_interval_is_byte_identical_to_full_domain_builder():
    rng = np.random.default_rng(7401)
    shapes = {
        "u": (4, 18, 21), "v": (4, 19, 20),
        "theta": (4, 18, 20), "phi": (5, 18, 20),
        "mu": (1, 18, 20), "qv": (4, 18, 20),
    }
    first = {name: rng.standard_normal(shape).astype(np.float32)
             for name, shape in shapes.items()}
    second = {name: rng.standard_normal(shape).astype(np.float32)
              for name, shape in shapes.items()}
    reference = build_lateral_boundaries(
        [first, second], [0.0, 3600.0], spec_bdy_width=5).intervals[0]
    side_names = ("west", "east", "south", "north")
    first_sides = {
        side: extract_lateral_side(first, side, 5) for side in side_names}
    second_sides = {
        side: extract_lateral_side(second, side, 5) for side in side_names}
    candidate = build_lateral_interval_from_sides(
        first_sides, second_sides, start_seconds=0.0, end_seconds=3600.0)

    assert set(candidate.fields) == set(reference.fields)
    for name in reference.fields:
        for side in side_names:
            expected = getattr(reference.fields[name], side)
            actual = getattr(candidate.fields[name], side)
            np.testing.assert_array_equal(actual.value, expected.value)
            np.testing.assert_array_equal(actual.tendency, expected.tendency)


def test_davies_coefficients_match_wrf_outer_clock_f1_f2_rows():
    """WRF module_bc_em.F:lbc_fcx_gcx, loops 2:5 at dt=60 s."""
    f1, f2 = _weights(5, spec_zone=1, relax_zone=4, dt=60.0,
                      spec_exp=0.0)
    expected_f1 = np.array([
        0.0, 1.0 / 600.0, 1.0 / 900.0, 1.0 / 1800.0, 0.0,
    ], dtype=np.float32)
    expected_f2 = np.array([
        0.0, 1.0 / 3000.0, 1.0 / 4500.0, 1.0 / 9000.0, 0.0,
    ], dtype=np.float32)
    np.testing.assert_array_equal(f1, expected_f1)
    np.testing.assert_array_equal(f2, expected_f2)


def test_attach_rounds_each_boundary_table_once_and_reuses_weights(monkeypatch):
    """The hot-call FP64->FP32 path is retained exactly, but only at attach."""
    calls = []

    def asarray(value, dtype=None):
        calls.append((value, dtype))
        return np.asarray(value, dtype=dtype)

    fake_cupy = SimpleNamespace(float32=np.float32, asarray=asarray)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    base = np.linspace(-1.0, 1.0, 3 * 10 * 12, dtype=np.float64).reshape(
        3, 10, 12)
    snapshots = [
        {name: base + offset for name in ("u", "v", "theta", "phi", "mu")}
        for offset in (0.125, 0.375)
    ]
    boundaries = build_lateral_boundaries(snapshots, [0.0, 60.0])
    scratch_buffers = {}

    def scratch(shape, slot):
        shape = tuple(shape)
        buffer = scratch_buffers.get(slot)
        if buffer is None:
            buffer = np.zeros(shape, dtype=np.float32)
            scratch_buffers[slot] = buffer
        elif buffer.shape != shape:
            raise ValueError("scratch shape changed")
        return buffer

    state = SimpleNamespace(_scratch=scratch_buffers, scratch=scratch)

    attach_lateral_boundaries(state, boundaries)

    # 1 interval * 5 fields * 4 sides * (value+tendency).
    assert len(calls) == 40
    assert all(dtype is np.float32 for _, dtype in calls)
    device = state._lateral_boundary_device.intervals[0]
    for name, host_field in boundaries.intervals[0].fields.items():
        device_field = device.fields[name]
        for side_name in ("west", "east", "south", "north"):
            host_side = getattr(host_field, side_name)
            device_side = getattr(device_field, side_name)
            np.testing.assert_array_equal(
                device_side.value, np.asarray(host_side.value, np.float32))
            np.testing.assert_array_equal(
                device_side.tendency,
                np.asarray(host_side.tendency, np.float32))

    first = _resident_weights(state, 5, 1, 4, 60.0, 0.0)
    second = _resident_weights(state, 5, 1, 4, 60.0, 0.0)
    assert first is second
    assert len(calls) == 42
    expected_bytes = sum(
        array.size * np.dtype(np.float32).itemsize
        for boundary in boundaries.intervals[0].fields.values()
        for side in (boundary.west, boundary.east,
                     boundary.south, boundary.north)
        for array in (side.value, side.tendency)
    ) + 2 * 5 * np.dtype(np.float32).itemsize
    assert lateral_boundary_resident_bytes(state) == expected_bytes

    # Host forcing is irreversibly immutable, so a resident mirror cannot be
    # made stale by either ordinary assignment or re-enabling write flags.
    host_value = boundaries.intervals[0].fields["u"].west.value
    with pytest.raises(ValueError):
        host_value[...] = 99.0
    with pytest.raises(ValueError):
        host_value.setflags(write=True)

    replacement = build_lateral_boundaries(
        [{name: base + offset for name in ("u", "v", "theta", "phi", "mu")}
         for offset in (1.125, 1.375, 1.875)],
        [0.0, 60.0, 120.0])
    attach_lateral_boundaries(state, replacement)
    np.testing.assert_array_equal(
        state._lateral_boundary_device.intervals[0].fields["u"].west.value,
        np.asarray(replacement.intervals[0].fields["u"].west.value,
                   np.float32))
    assert (_resident_interval(state, replacement.interval_at(60.0))
            is state._lateral_boundary_device.intervals[1])
    assert set(scratch_buffers) == {"lbc_forcing_tables"}


def test_streaming_attachment_reuses_one_interval_device_buffer(monkeypatch):
    calls = []

    def asarray(value, dtype=None):
        calls.append((value, dtype))
        return np.asarray(value, dtype=dtype)

    monkeypatch.setitem(
        sys.modules, "cupy",
        SimpleNamespace(float32=np.float32, asarray=asarray))
    nz, ny, nx = 3, 10, 12
    base = np.arange(nz * ny * nx, dtype=np.float64).reshape(nz, ny, nx)
    snapshots = [
        {name: base + offset
         for name in ("u", "v", "theta", "phi", "mu")}
        for offset in (0.125, 0.375, 0.875)
    ]
    boundaries = build_lateral_boundaries(
        snapshots, [0.0, 60.0, 120.0])
    scratch_buffers = {}

    def scratch(shape, slot):
        shape = tuple(shape)
        buffer = scratch_buffers.get(slot)
        if buffer is None:
            buffer = np.zeros(shape, dtype=np.float32)
            scratch_buffers[slot] = buffer
        elif buffer.shape != shape:
            raise ValueError("streaming scratch shape changed")
        return buffer

    state = SimpleNamespace(_scratch=scratch_buffers, scratch=scratch)
    attach_streaming_lateral_boundaries(state, boundaries)
    resident = state._lateral_boundary_device
    assert resident.streaming_external
    assert len(resident.intervals) == 1
    assert resident.active_host_interval_id == id(boundaries.intervals[0])
    assert resident.external_reload_count == 1
    one_interval_bytes = sum(
        array.size * np.dtype(np.float32).itemsize
        for boundary in boundaries.intervals[0].fields.values()
        for side in (boundary.west, boundary.east,
                     boundary.south, boundary.north)
        for array in (side.value, side.tendency)
    )
    assert lateral_boundary_resident_bytes(state) == one_interval_bytes
    assert scratch_buffers["lbc_forcing_tables"].nbytes == one_interval_bytes

    device = resident.intervals[0]
    np.testing.assert_array_equal(
        device.fields["u"].west.value,
        np.asarray(boundaries.intervals[0].fields["u"].west.value,
                   np.float32))
    selected = _resident_interval(state, boundaries.interval_at(60.0))
    assert selected is device
    assert resident.active_host_interval_id == id(boundaries.intervals[1])
    assert resident.external_reload_count == 2
    np.testing.assert_array_equal(
        device.fields["u"].west.value,
        np.asarray(boundaries.intervals[1].fields["u"].west.value,
                   np.float32))
    assert lateral_boundary_resident_bytes(state) == one_interval_bytes
    assert set(scratch_buffers) == {"lbc_forcing_tables"}
    # Two interval uploads, each 5 fields * 4 sides * value+tendency.
    assert len(calls) == 80


def test_attach_rejects_overlapping_nested_relaxation_frames():
    nz, ny, nx, width = 1, 4, 5, 3

    def side(shape):
        return SideBoundary(np.zeros(shape), np.zeros(shape))

    field = FieldBoundary(
        side((nz, ny, width)), side((nz, ny, width)),
        side((nz, width, nx)), side((nz, width, nx)))
    interval = BoundaryInterval(
        0.0, 60.0,
        {name: field for name in ("u", "v", "theta", "phi", "mu")})
    boundaries = LateralBoundaries(
        (interval,), spec_bdy_width=width, spec_zone=1, relax_zone=2)
    with pytest.raises(ValueError, match="no unique interior frame"):
        attach_lateral_boundaries(SimpleNamespace(), boundaries)


def test_perimeter_count_matches_unique_nested_frames():
    ny, nx, width = 10, 12, 3
    mask = np.zeros((ny, nx), dtype=bool)
    for d in range(width):
        mask[d, d:nx - d] = True
        mask[ny - 1 - d, d:nx - d] = True
        mask[d + 1:ny - d - 1, d] = True
        mask[d + 1:ny - d - 1, nx - 1 - d] = True
    assert _perimeter_count(ny, nx, width) == int(mask.sum())


def _numpy_cupy_namespace():
    return SimpleNamespace(
        ndarray=np.ndarray, float32=np.float32, asarray=np.asarray,
        zeros=np.zeros, arange=np.arange, clip=np.clip, where=np.where)


@pytest.mark.parametrize("layout", ["float64", "strided"])
def test_public_flow_helper_preserves_dtype_and_strides(monkeypatch, layout):
    """Raw float kernels must fall back for every formerly generic operand."""
    import gpuwm.ingest.lateral_bc as lbc

    monkeypatch.setitem(sys.modules, "cupy", _numpy_cupy_namespace())
    monkeypatch.setattr(
        lbc, "get_kernel",
        lambda *_: pytest.fail("unsafe raw kernel used for generic operand"))
    nz, ny, nx = 2, 7, 9
    source = (100.0 * np.arange(nz)[:, None, None]
              + 10.0 * np.arange(ny)[None, :, None]
              + np.arange(nx)[None, None, :])
    u_source = np.zeros((nz, ny, nx + 1))
    v_source = np.zeros((nz, ny + 1, nx))
    u_source[..., 0] = -1.0
    u_source[..., -1] = 1.0
    v_source[:, 0, :] = -1.0
    v_source[:, -1, :] = 1.0
    if layout == "float64":
        field, u_flux, v_flux = source.copy(), u_source, v_source
    else:
        field_store = np.empty((nz, ny, 2 * nx), dtype=np.float32)
        u_store = np.empty((nz, ny, 2 * (nx + 1)), dtype=np.float32)
        v_store = np.empty((nz, 2 * (ny + 1), nx), dtype=np.float32)
        field = field_store[..., ::2]
        u_flux = u_store[..., ::2]
        v_flux = v_store[:, ::2, :]
        field[...] = source
        u_flux[...] = u_source
        v_flux[...] = v_source
        assert not field.flags.c_contiguous
        assert not u_flux.flags.c_contiguous
        assert not v_flux.flags.c_contiguous
    expected = np_flow_dependent_boundary(
        np.asarray(field), np.asarray(u_flux), np.asarray(v_flux),
        spec_zone=2)
    apply_flow_dependent_boundary(field, u_flux, v_flux, spec_zone=2)
    np.testing.assert_array_equal(field, expected)
    assert field.dtype == (np.float64 if layout == "float64" else np.float32)


def test_two_field_flow_batch_uses_one_fp32_kernel(monkeypatch):
    """Kessler's qc/qr pair remains a two-field raw-kernel batch."""
    import gpuwm.ingest.lateral_bc as lbc

    monkeypatch.setitem(sys.modules, "cupy", _numpy_cupy_namespace())
    launches = []

    def kernel(grid, block, args):
        launches.append((grid, block, args))

    monkeypatch.setattr(lbc, "get_kernel", lambda *_: kernel)
    fields = tuple(np.zeros((2, 7, 9), dtype=np.float32) for _ in range(2))
    u_flux = np.zeros((2, 7, 10), dtype=np.float32)
    v_flux = np.zeros((2, 8, 9), dtype=np.float32)
    apply_flow_dependent_boundaries(fields, u_flux, v_flux, spec_zone=2)
    assert len(launches) == 1
    assert launches[0][2][11] == np.int32(2)


@pytest.mark.parametrize("layout", ["float64", "strided"])
def test_public_w_helper_preserves_dtype_and_strides(monkeypatch, layout):
    import gpuwm.ingest.lateral_bc as lbc

    monkeypatch.setitem(sys.modules, "cupy", _numpy_cupy_namespace())
    monkeypatch.setattr(
        lbc, "get_kernel",
        lambda *_: pytest.fail("unsafe raw kernel used for generic operand"))
    nz, ny, nx, width = 3, 9, 11, 3
    source = np.arange(nz * ny * nx).reshape(nz, ny, nx)
    if layout == "float64":
        field = source.astype(np.float64)
    else:
        store = np.empty((nz, ny, 2 * nx), dtype=np.float32)
        field = store[..., ::2]
        field[...] = source
        assert not field.flags.c_contiguous
    expected = field.copy()
    _apply_specified_w_zero_gradient_generic(
        expected, width, _numpy_cupy_namespace())
    cfg = SimpleNamespace(specified=True, ny=ny, nx=nx, spec_zone=width)
    apply_specified_w_zero_gradient(SimpleNamespace(w=field), cfg)
    np.testing.assert_array_equal(field, expected)
    assert field.dtype == (np.float64 if layout == "float64" else np.float32)


def test_legacy_held_interior_restores_signed_zero_and_nonfinite_msf():
    width = 2
    tendency = np.full((1, 7, 9), np.float32(-0.0))
    held = np.zeros_like(tendency)
    msft = np.ones((7, 9), dtype=np.float32)
    _apply_legacy_held_interior(
        tendency, held, msft, width, divide_msf=False, add_held=True)
    assert not np.signbit(tendency[:, width:-width, width:-width]).any()
    assert np.signbit(tendency[:, :width, :]).all()

    divided = np.zeros_like(tendency)
    msft[width, width] = 0.0
    msft[width, width + 1] = np.nan
    with np.errstate(invalid="ignore"):
        _apply_legacy_held_interior(
            divided, held, msft, width, divide_msf=True, add_held=False)
    assert np.isnan(divided[0, width, width])
    assert np.isnan(divided[0, width, width + 1])


def test_nested_held_relaxation_uses_time_t_mass_for_dry_and_scalar(
        monkeypatch):
    """WRF forms nested held tendencies from the complete time-t state.

    Re-evaluating the one-buffer representation on later RK stages is valid
    only if both the source field and dry column mass remain the RK1 copies.
    """
    import gpuwm.ingest.lateral_bc as lbc

    # Exact skinny topology that exposed the production capacity assumption:
    # U/V are larger than the W-shaped acoustic scratch when nx < nz.
    nz, ny, nx = 80, 60, 60

    class FakeState:
        def __init__(self):
            self.u = np.zeros((nz, ny, nx + 1), np.float32)
            self.v = np.zeros((nz, ny + 1, nx), np.float32)
            self.w = np.zeros((nz + 1, ny, nx), np.float32)
            self.thp = np.zeros((nz, ny, nx), np.float32)
            self.php = np.zeros((nz + 1, ny, nx), np.float32)
            self.qv = np.zeros((nz, ny, nx), np.float32)
            self.u0, self.v0, self.w0 = (a.copy() for a in
                                         (self.u, self.v, self.w))
            self.thp0, self.php0, self.qv0 = (a.copy() for a in
                                               (self.thp, self.php, self.qv))
            self.ru_t = np.zeros_like(self.u)
            self.rv_t = np.zeros_like(self.v)
            self.rw_t = np.zeros_like(self.w)
            self.rth_t = np.zeros_like(self.thp)
            self.rph_t = np.zeros_like(self.php)
            self.rmu_t = np.zeros((ny, nx), np.float32)
            self.mup = np.full((ny, nx), 7.0, np.float32)
            self.mup0 = np.full((ny, nx), 3.0, np.float32)
            self.has_msf = False
            self.lateral_boundaries = object()
            self._scratch = {}

        def scratch(self, shape, slot, dtype=None):
            value = self._scratch.get(slot)
            if value is None:
                value = np.zeros(shape, np.float32)
                self._scratch[slot] = value
            return value

    def boundary(shape):
        side = SimpleNamespace(value=np.zeros(shape, np.float32))
        return SimpleNamespace(west=side)

    state = FakeState()
    assert state.u.size > state.w.size
    fields = {
        "u": boundary((nz, ny, 1)),
        "v": boundary((nz, ny + 1, 1)),
        "w": boundary((nz + 1, ny, 1)),
        "theta": boundary((nz, ny, 1)),
        "phi": boundary((nz + 1, ny, 1)),
        "mu": boundary((1, ny, 1)),
        "qv": boundary((nz, ny, 1)),
    }
    interval = SimpleNamespace(fields=fields)
    monkeypatch.setattr(
        lbc, "_active_device_interval",
        lambda *_args, **_kwargs: (interval, 0.0, 12.0, 0.0))
    monkeypatch.setattr(
        lbc, "_resident_weights",
        lambda *_args, **_kwargs: (np.ones(1, np.float32),
                                   np.ones(1, np.float32)))

    calls = []

    def capture(*_args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(lbc, "apply_specified_relaxation", capture)
    cfg = SimpleNamespace(
        specified=False, nested=True, spec_zone=1, relax_zone=0)
    lbc.apply_state_lateral_boundaries(state, cfg, rk_stage=1)
    lbc.apply_state_scalar_lateral_boundary(
        state, cfg, "qv", np.zeros_like(state.qv), source_field=state.qv0)

    assert state._scratch["lbc_nested_relax"].shape == (state.u.size,)

    held = [call for call in calls if call.get("apply_relax") is True]
    assert {call["field_name"] for call in held} == {
        "u", "v", "w", "theta", "phi", "qv"}
    assert all(call.get("source_mup") is state.mup0 for call in held)


def test_acoustic_rk_finalizes_nested_state_once_after_rk_loop():
    """WRF calls ``spec_bdy_final`` once after ``Runge_Kutta_loop``."""
    import gpuwm.core.dycore as dycore

    source = inspect.getsource(dycore.step)
    rk_start = source.index("for istage, (nsub, dtau) in enumerate(stages):")
    final_call = source.index(
        "apply_state_boundary_values(state, cfg,\n"
        "                                state.elapsed_seconds + cfg.dt)",
        rk_start)
    rk_body = source[rk_start:final_call]
    assert "apply_state_boundary_values(" not in rk_body


@requires_gpu
@pytest.mark.gpu
def test_davies_run_wiring_uses_outer_clock_for_dry_and_moist(monkeypatch):
    """The 7.5 s real74 substep installs coefficients from its 60 s clock."""
    import cupy as cp

    import gpuwm.ingest.lateral_bc as lbc
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import advance_scalars_stage
    from gpuwm.core.state import init_at_rest

    cfg = RunConfig(nx=12, ny=10, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=7.5, clock_dt=60.0,
                    run_seconds=60.0, moist=True, specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    states = [init_at_rest(cfg, coord, base) for _ in range(3)]
    for state in states:
        state.qv[...] = cp.float32(0.01)
        state.qv0[...] = state.qv
    boundaries = build_state_lateral_boundaries(
        states, [0.0, 21600.0, 43200.0])
    attach_lateral_boundaries(states[0], boundaries)
    qv_boundary = boundaries.intervals[0].fields["qv"]

    installed = []
    real_apply = lbc.apply_specified_relaxation

    def capture_coefficients(field, tendency, boundary, **kwargs):
        f1, f2 = _weights(
            boundary.west.value.shape[-1], kwargs["spec_zone"],
            kwargs["relax_zone"], kwargs["dt"], kwargs["spec_exp"])
        installed.append((kwargs.get("field_name") == "qv"
                          or boundary is qv_boundary,
                          f1.copy(), f2.copy()))
        return real_apply(field, tendency, boundary, **kwargs)

    monkeypatch.setattr(lbc, "apply_specified_relaxation",
                        capture_coefficients)
    apply_state_lateral_boundaries(states[0], cfg, rk_stage=0)
    advance_scalars_stage(
        states[0], cfg, cp.zeros_like(states[0].u),
        cp.zeros_like(states[0].v), cp.zeros_like(states[0].w),
        dt_eff=cfg.dt / 3.0, final=False)

    expected_f1, expected_f2 = _weights(
        cfg.spec_bdy_width, cfg.spec_zone, cfg.relax_zone,
        dt=60.0, spec_exp=cfg.spec_exp)
    assert any(is_qv for is_qv, _, _ in installed)
    assert any(not is_qv for is_qv, _, _ in installed)
    for _, actual_f1, actual_f2 in installed:
        np.testing.assert_array_equal(actual_f1, expected_f1)
        np.testing.assert_array_equal(actual_f2, expected_f2)


@requires_gpu
@pytest.mark.gpu
def test_hydrometeor_boundary_is_zero_gradient_outflow_zero_inflow():
    """WRF flow_dep_bdy on a synthetic west/south-divergent flow."""
    import cupy as cp

    nz, ny, nx = 2, 6, 8
    field = (100.0 * np.arange(nz)[:, None, None]
             + 10.0 * np.arange(ny)[None, :, None]
             + np.arange(nx)[None, None, :]).astype(np.float32)
    u_flux = np.zeros((nz, ny, nx + 1), dtype=np.float32)
    v_flux = np.zeros((nz, ny + 1, nx), dtype=np.float32)
    u_flux[:, :, 0] = -2.0       # west outflow
    u_flux[:, :, -1] = -2.0      # east inflow
    v_flux[:, 0, :] = -3.0       # south outflow
    v_flux[:, -1, :] = -3.0      # north inflow

    expected = np_flow_dependent_boundary(
        field, u_flux, v_flux, spec_zone=1)
    actual = cp.asarray(field)
    apply_flow_dependent_boundary(
        actual, cp.asarray(u_flux), cp.asarray(v_flux), spec_zone=1)
    np.testing.assert_array_equal(cp.asnumpy(actual), expected.astype(np.float32))

    # Y sides own corners: south copies the first interior row, north is
    # zero.  On the remaining rows west copies the first interior column
    # while east inflow is zero.
    np.testing.assert_array_equal(expected[:, 0, 1:-1], field[:, 1, 1:-1])
    np.testing.assert_array_equal(expected[:, -1, :], 0.0)
    np.testing.assert_array_equal(expected[:, 1:-1, 0], field[:, 1:-1, 1])
    np.testing.assert_array_equal(expected[:, 1:-1, -1], 0.0)


@requires_gpu
@pytest.mark.gpu
def test_specified_w_x_sides_clamp_j_inner_for_wide_spec_zone():
    """WRF X-copy clamp and CUDA frame mapping cover odd, wide domains."""
    from types import SimpleNamespace

    import cupy as cp

    nz, ny, nx, spec_zone = 3, 9, 11, 3
    source = np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx)
    expected = source.copy()
    for d in range(spec_zone):
        cols = np.clip(np.arange(d, nx - d), spec_zone,
                       nx - 1 - spec_zone)
        expected[:, d, d:nx - d] = expected[:, spec_zone, cols]
        expected[:, ny - 1 - d, d:nx - d] = (
            expected[:, ny - 1 - spec_zone, cols])
        rows = np.arange(d + 1, ny - d - 1)
        inner_rows = np.clip(rows, spec_zone, ny - 1 - spec_zone)
        expected[:, rows, d] = expected[:, inner_rows, spec_zone]
        expected[:, rows, nx - 1 - d] = (
            expected[:, inner_rows, nx - 1 - spec_zone])

    actual = cp.asarray(source)
    cfg = SimpleNamespace(specified=True, ny=ny, nx=nx,
                          spec_zone=spec_zone)
    apply_specified_w_zero_gradient(SimpleNamespace(w=actual), cfg)
    np.testing.assert_array_equal(cp.asnumpy(actual), expected)


@requires_gpu
@pytest.mark.gpu
def test_state_boundary_snapshot_carries_qv_but_no_hydrometeors():
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    cfg = RunConfig(nx=12, ny=10, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=60.0, run_seconds=60.0,
                    moist=True, specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state.qv[...] = cp.float32(0.01)
    state.qc[...] = cp.float32(0.002)
    state.qr[...] = cp.float32(0.003)

    snapshot = domain_boundary_snapshot(state)
    assert "qv" in snapshot
    assert "qc" not in snapshot and "qr" not in snapshot


@requires_gpu
@pytest.mark.gpu
def test_theta_boundary_uses_300k_perturbation_and_round_trips():
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    cfg = RunConfig(nx=12, ny=10, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=60.0, run_seconds=60.0,
                    specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    source = init_at_rest(cfg, coord, base)
    source.thp += cp.float32(7.25)

    snapshot = domain_boundary_snapshot(source)
    chm = (source.c1h[:, None, None] * source.total_mu()[None, :, :]
           + source.c2h[:, None, None])
    expected = chm * (source.total_theta() - cp.float32(300.0))
    np.testing.assert_allclose(snapshot["theta"], cp.asnumpy(expected),
                               rtol=2.0e-6, atol=2.0e-5)

    later = init_at_rest(cfg, coord, base)
    later.thp += cp.float32(9.0)
    boundaries = build_state_lateral_boundaries(
        [source, later], [0.0, 60.0],
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone,
    )
    target = init_at_rest(cfg, coord, base)
    target.thp -= cp.float32(13.0)
    before = cp.asnumpy(target.total_theta())
    attach_lateral_boundaries(target, boundaries)
    apply_state_boundary_values(target, cfg, elapsed_seconds=0.0)

    got = cp.asnumpy(target.total_theta())
    want = cp.asnumpy(source.total_theta())
    mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    mask[:cfg.spec_zone, :] = True
    mask[-cfg.spec_zone:, :] = True
    mask[:, :cfg.spec_zone] = True
    mask[:, -cfg.spec_zone:] = True
    np.testing.assert_allclose(got[:, mask], want[:, mask],
                               rtol=2.0e-6, atol=2.0e-5)
    np.testing.assert_allclose(got[:, ~mask], before[:, ~mask],
                               rtol=2.0e-6, atol=2.0e-5)


@requires_gpu
@pytest.mark.gpu
def test_fused_boundary_finalizer_is_bitwise_legacy_equivalent():
    """Pin the roundoff-visible full-domain couple/uncouple operation order."""
    import cupy as cp

    import gpuwm.ingest.lateral_bc as lbc
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import (init_at_rest, mu_at_u_faces,
                                  mu_at_v_faces)

    cfg = RunConfig(nx=12, ny=10, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=60.0, run_seconds=60.0,
                    moist=True, specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    reference = init_at_rest(cfg, coord, base)
    actual = init_at_rest(cfg, coord, base)
    forcing = [init_at_rest(cfg, coord, base) for _ in range(2)]
    msft = np.linspace(0.83, 1.17, cfg.ny * cfg.nx,
                       dtype=np.float64).reshape(cfg.ny, cfg.nx)
    msfu = np.linspace(0.82, 1.18, cfg.ny * (cfg.nx + 1),
                       dtype=np.float64).reshape(cfg.ny, cfg.nx + 1)
    msfv = np.linspace(0.81, 1.19, (cfg.ny + 1) * cfg.nx,
                       dtype=np.float64).reshape(cfg.ny + 1, cfg.nx)
    for state in (reference, actual, *forcing):
        state.set_map_coriolis(msft=msft, msfu=msfu, msfv=msfv)
    rng = np.random.default_rng(90210)
    for name in ("u", "v", "thp", "php", "mup", "qv"):
        values = rng.normal(0.0, 1.0, getattr(reference, name).shape).astype(
            np.float32)
        if name == "qv":
            values = np.abs(values) * np.float32(1.0e-3)
        getattr(reference, name)[...] = cp.asarray(values)
        getattr(actual, name)[...] = cp.asarray(values)
    for n, state in enumerate(forcing):
        state.u += cp.float32(2.0 + n)
        state.v -= cp.float32(1.0 + n)
        state.thp += cp.float32(4.0 + n)
        state.php += cp.float32(20.0 + n)
        state.mup += cp.float32(50.0 + n)
        state.qv += cp.float32(0.01 + 0.001 * n)
    boundaries = build_state_lateral_boundaries(forcing, [0.0, 60.0])
    attach_lateral_boundaries(reference, boundaries)
    attach_lateral_boundaries(actual, boundaries)

    def install(coupled, boundary, dtbc):
        nz, ny, nx = coupled.shape
        for d in range(cfg.spec_zone):
            south = (cp.asarray(boundary.south.value[:, d, :], cp.float32)
                     + cp.float32(dtbc) * cp.asarray(
                         boundary.south.tendency[:, d, :], cp.float32))
            north = (cp.asarray(boundary.north.value[:, d, :], cp.float32)
                     + cp.float32(dtbc) * cp.asarray(
                         boundary.north.tendency[:, d, :], cp.float32))
            coupled[:, d, d:nx - d] = south[:, d:nx - d]
            coupled[:, ny - 1 - d, d:nx - d] = north[:, d:nx - d]
            west = (cp.asarray(boundary.west.value[:, :, d], cp.float32)
                    + cp.float32(dtbc) * cp.asarray(
                        boundary.west.tendency[:, :, d], cp.float32))
            east = (cp.asarray(boundary.east.value[:, :, d], cp.float32)
                    + cp.float32(dtbc) * cp.asarray(
                        boundary.east.tendency[:, :, d], cp.float32))
            coupled[:, d + 1:ny - d - 1, d] = west[:, d + 1:ny - d - 1]
            coupled[:, d + 1:ny - d - 1, nx - 1 - d] = (
                east[:, d + 1:ny - d - 1])

    dtbc = 17.0
    interval = boundaries.intervals[0]
    fields = lbc._coupled_device_fields(reference)
    install(fields["mu"], interval.fields["mu"], dtbc)
    reference.mup[...] = fields["mu"][0]
    mu = reference.total_mu()
    mux, muy = mu_at_u_faces(mu), mu_at_v_faces(mu)
    mux[:, 0], mux[:, -1] = mu[:, 0], mu[:, -1]
    muy[0, :], muy[-1, :] = mu[0, :], mu[-1, :]
    c1h, c2h = reference.c1h[:, None, None], reference.c2h[:, None, None]
    c1f, c2f = reference.c1f[:, None, None], reference.c2f[:, None, None]
    chm = c1h * mu[None] + c2h
    chf = c1f * mu[None] + c2f
    for name in ("u", "v", "theta", "phi", "qv"):
        install(fields[name], interval.fields[name], dtbc)
    reference.u[...] = fields["u"] * reference.msfu[None] / (
        c1h * mux[None] + c2h)
    reference.v[...] = fields["v"] * reference.msfv[None] / (
        c1h * muy[None] + c2h)
    thb = (reference.thb if reference.thb.ndim == 3
           else reference.thb[:, None, None])
    reference.thp[...] = fields["theta"] / chm + cp.float32(300.0) - thb
    reference.php[...] = fields["phi"] / chf
    reference.qv[...] = (fields["qv"] / chm).clip(min=0.0)

    apply_state_boundary_values(actual, cfg, elapsed_seconds=dtbc)
    for name in ("u", "v", "thp", "php", "mup", "qv"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)),
            cp.asnumpy(getattr(reference, name)), err_msg=name)


@requires_gpu
@pytest.mark.gpu
def test_scalar_frame_increments_use_wrf_mu_only_coupling_with_map_factors():
    """WRF module_bc.F:2081/2139-2142: only u/v/w couple with 1/msf."""
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import advance_scalars_stage
    from gpuwm.core.state import init_at_rest

    cfg = RunConfig(nx=12, ny=10, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=60.0, run_seconds=60.0,
                    moist=True, specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    states = [init_at_rest(cfg, coord, base) for _ in range(3)]
    msft = 0.85 + 0.3 * np.arange(cfg.ny * cfg.nx).reshape(cfg.ny, cfg.nx) \
        / (cfg.ny * cfg.nx - 1)
    for state in states:
        state.set_map_coriolis(msft=msft)
        state.qv[...] = cp.float32(0.010)
        state.qv0[...] = state.qv
    states[1].thp += cp.float32(6.0)
    states[1].php += cp.float32(120.0)
    states[1].qv += cp.float32(0.001)
    states[2].thp += cp.float32(12.0)
    states[2].php += cp.float32(240.0)
    states[2].qv += cp.float32(0.002)

    boundaries = build_state_lateral_boundaries(states, [0.0, 60.0, 120.0])
    target = states[0]
    attach_lateral_boundaries(target, boundaries)
    apply_state_lateral_boundaries(target, cfg, rk_stage=0)

    mu = cp.asnumpy(target.total_mu())
    chm = coord.c1h[:, None, None] * mu[None] + coord.c2h[:, None, None]
    chf = coord.c1f[:, None, None] * mu[None] + coord.c2f[:, None, None]
    wrf_rth = chm * (6.0 / 60.0)
    wrf_rph = chf * (120.0 / 60.0)
    frame = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    frame[[0, -1], :] = True
    frame[:, [0, -1]] = True
    np.testing.assert_allclose(cp.asnumpy(target.rth_t)[:, frame],
                               wrf_rth[:, frame], rtol=2.0e-6, atol=2.0e-4)
    np.testing.assert_allclose(cp.asnumpy(target.rph_t)[:, frame],
                               wrf_rph[:, frame], rtol=2.0e-6, atol=2.0e-4)
    wrf_rqv = chm * (0.001 / 60.0)
    np.testing.assert_allclose(
        boundaries.intervals[0].fields["qv"].west.tendency,
        wrf_rqv[..., :cfg.spec_bdy_width], rtol=2.0e-6, atol=2.0e-5)

    dtau = 0.5
    acoustic_substep(target, cfg, dtau=dtau, first=True)
    np.testing.assert_allclose(cp.asnumpy(target.th_pp)[:, frame],
                               dtau * wrf_rth[:, frame],
                               rtol=3.0e-6, atol=3.0e-4)
    np.testing.assert_allclose(cp.asnumpy(target.ph_pp)[1:, frame],
                               dtau * wrf_rph[1:, frame] / chf[1:, frame],
                               rtol=3.0e-6, atol=3.0e-5)
    advance_scalars_stage(
        target, cfg, cp.zeros_like(target.u), cp.zeros_like(target.v),
        cp.zeros_like(target.w), dt_eff=dtau, final=False)
    np.testing.assert_allclose(
        cp.asnumpy(target.qv)[:, frame], 0.010 + dtau * 0.001 / 60.0,
        rtol=2.0e-6, atol=2.0e-8)


def test_boundary_builder_preserves_six_hour_deltas_at_all_zone_edges():
    start = datetime(1974, 4, 3, 12)
    times = [start, start + timedelta(hours=6), start + timedelta(hours=12)]
    snapshots = _snapshots()
    bc = build_lateral_boundaries(snapshots, times, spec_bdy_width=5,
                                  spec_zone=1, relax_zone=4)
    assert len(bc.intervals) == 2
    interval = bc.intervals[0]
    field = interval.fields["theta"]
    dt = 6.0 * 3600.0
    expected = snapshots[1]["theta"]
    np.testing.assert_array_equal(field.west.value + dt * field.west.tendency,
                                  expected[..., :5])
    np.testing.assert_array_equal(field.east.value + dt * field.east.tendency,
                                  expected[..., -5:][..., ::-1])
    np.testing.assert_array_equal(field.south.value + dt * field.south.tendency,
                                  expected[..., :5, :])
    np.testing.assert_array_equal(field.north.value + dt * field.north.tendency,
                                  expected[..., -5:, :][..., ::-1, :])


def test_interval_at_18z_selects_the_18z_to_00z_boundary_record():
    """The half-open WRF boundary clock advances records exactly at 18Z."""
    start = datetime(1974, 4, 3, 12)
    times = [start, start + timedelta(hours=6), start + timedelta(hours=12)]
    bc = build_lateral_boundaries(_snapshots(), times, spec_bdy_width=5,
                                  spec_zone=1, relax_zone=4)
    assert bc.interval_at(21600.0) is bc.intervals[1]
    assert (bc.interval_at(21600.0).start_seconds,
            bc.interval_at(21600.0).end_seconds) == (21600.0, 43200.0)


def test_rk_mirror_holds_dry_relaxation_but_not_mu_or_moist_scalars():
    snapshots = _snapshots()
    bc = build_lateral_boundaries(snapshots, [0.0, 21600.0, 43200.0],
                                  spec_bdy_width=5, spec_zone=1,
                                  relax_zone=4)
    boundary = bc.intervals[0].fields["theta"]
    field = np.full_like(snapshots[0]["theta"], -10.0)
    stages = tuple(np.full_like(field, value) for value in (1.0, 2.0, 3.0))
    spec_only = tuple(
        np_specified_relaxation(
            field, tendency, boundary, dtbc=0.0, dt=60.0,
            spec_zone=1, relax_zone=4, apply_relax=False)
        for tendency in stages
    )

    dry = np_rk_specified_relaxation(
        field, stages, boundary, dtbc=0.0, dt=60.0,
        hold_relaxation=True, spec_zone=1, relax_zone=4)
    first_stage_only = np_rk_specified_relaxation(
        field, stages, boundary, dtbc=0.0, dt=60.0,
        hold_relaxation=False, spec_zone=1, relax_zone=4)

    relax_point = (0, 1, 5)
    held_increment = dry[0][relax_point] - spec_only[0][relax_point]
    assert held_increment != 0.0
    for rk_stage in range(3):
        assert (dry[rk_stage][relax_point]
                == spec_only[rk_stage][relax_point] + held_increment)
    assert (first_stage_only[0][relax_point]
            == spec_only[0][relax_point] + held_increment)
    np.testing.assert_array_equal(first_stage_only[1], spec_only[1])
    np.testing.assert_array_equal(first_stage_only[2], spec_only[2])


@requires_gpu
@pytest.mark.gpu
def test_spec_bdy_cuda_matches_mirror_in_specified_and_relaxation_zones():
    import cupy as cp

    snapshots = _snapshots()
    times = [0.0, 21600.0, 43200.0]
    bc = build_lateral_boundaries(snapshots, times, spec_bdy_width=5,
                                  spec_zone=1, relax_zone=4)
    boundary = bc.intervals[0].fields["theta"]
    rng = np.random.default_rng(8)
    field = rng.normal(270.0, 4.0, snapshots[0]["theta"].shape)
    tendency0 = rng.normal(0.0, 0.01, field.shape)
    dtbc = 2700.0
    ref = np_specified_relaxation(
        field, tendency0, boundary, dtbc=dtbc, dt=60.0,
        spec_zone=1, relax_zone=4, spec_exp=0.0,
    )
    tend = cp.asarray(tendency0, cp.float32)
    apply_specified_relaxation(
        cp.asarray(field, cp.float32), tend, boundary, dtbc=dtbc, dt=60.0,
        spec_zone=1, relax_zone=4, spec_exp=0.0,
    )
    np.testing.assert_allclose(cp.asnumpy(tend), ref, rtol=8.0e-5, atol=2.0e-5)
    # spec_zone=1 replaces the outer tendency with the boundary tendency.
    np.testing.assert_allclose(cp.asnumpy(tend)[:, :, 0],
                               boundary.west.tendency[:, :, 0],
                               rtol=2.0e-6, atol=2.0e-7)
    # The centre lies outside the five-point specified+relaxation strip.
    np.testing.assert_allclose(cp.asnumpy(tend)[:, 5, 6], tendency0[:, 5, 6],
                               rtol=2.0e-6, atol=2.0e-7)


@requires_gpu
@pytest.mark.gpu
def test_state_lbc_holds_dry_relaxation_across_rk_stages_but_not_mu():
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    cfg = RunConfig(nx=12, ny=10, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=60.0, run_seconds=60.0,
                    specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    forcing = [init_at_rest(cfg, coord, base) for _ in range(3)]
    for n, forced in enumerate(forcing, start=1):
        forced.u[...] = cp.float32(2.0 * n)
        forced.v[...] = cp.float32(-1.5 * n)
        forced.thp[...] = cp.float32(3.0 * n)
        forced.php[...] = cp.float32(20.0 * n)
        forced.mup[...] = cp.float32(50.0 * n)
    boundaries = build_state_lateral_boundaries(
        forcing, [0.0, 21600.0, 43200.0])
    attach_lateral_boundaries(state, boundaries)
    current = domain_boundary_snapshot(state)
    interval = boundaries.intervals[0]

    rng = np.random.default_rng(42)
    targets = {
        "u": state.ru_t,
        "v": state.rv_t,
        "theta": state.rth_t,
        "phi": state.rph_t,
        "mu": state.rmu_t[None],
    }
    inputs = {
        name: tuple(rng.normal(0.0, 0.01, target.shape).astype(np.float32)
                    for _ in range(3))
        for name, target in targets.items()
    }
    actual = {name: [] for name in targets}
    for rk_stage in range(3):
        for name, target in targets.items():
            target[...] = cp.asarray(inputs[name][rk_stage])
        apply_state_lateral_boundaries(state, cfg, rk_stage=rk_stage)
        for name, target in targets.items():
            actual[name].append(cp.asnumpy(target).copy())

    for name in ("u", "v", "theta", "phi", "mu"):
        expected = np_rk_specified_relaxation(
            current[name], inputs[name], interval.fields[name], dtbc=0.0,
            dt=cfg.dt, hold_relaxation=(name != "mu"),
            spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone,
            spec_exp=cfg.spec_exp)
        for rk_stage in range(3):
            np.testing.assert_allclose(actual[name][rk_stage],
                                       expected[rk_stage],
                                       rtol=8.0e-5, atol=2.0e-5,
                                       err_msg=f"{name} RK stage {rk_stage + 1}")


@requires_gpu
@pytest.mark.gpu
def test_coriolis_open_bounds_zero_boundary_normal_face_increments():
    import cupy as cp

    from gpuwm.core.dycore import launch_coriolis_curvature
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.verify.npref import np_coriolis_curvature

    rng = np.random.default_rng(19)
    nz, ny, nx = 6, 5, 8
    coord = make_vertical_coord(nz, stretch=1.2)
    arrays = {
        "u": rng.normal(0.0, 8.0, (nz, ny, nx + 1)),
        "v": rng.normal(0.0, 8.0, (nz, ny + 1, nx)),
        "w": rng.normal(0.0, 0.5, (nz + 1, ny, nx)),
        "ru": rng.normal(0.0, 7.0e5, (nz, ny, nx + 1)),
        "rv": rng.normal(0.0, 7.0e5, (nz, ny + 1, nx)),
        "mut": rng.normal(8.5e4, 500.0, (ny, nx)),
        "msft": rng.uniform(0.9, 1.1, (ny, nx)),
        "msfu": rng.uniform(0.9, 1.1, (ny, nx + 1)),
        "msfv": rng.uniform(0.9, 1.1, (ny + 1, nx)),
        "f": rng.uniform(7.0e-5, 1.1e-4, (ny, nx)),
        "e": rng.uniform(7.0e-5, 1.1e-4, (ny, nx)),
    }
    dev = {name: cp.asarray(value, cp.float32) for name, value in arrays.items()}
    ru_t = cp.zeros_like(dev["ru"])
    rv_t = cp.zeros_like(dev["rv"])
    rw_t = cp.zeros_like(dev["w"])
    launch_coriolis_curvature(
        dev["ru"], dev["rv"], dev["u"], dev["v"], dev["w"], dev["mut"],
        dev["msft"], dev["msfu"], dev["msfv"], dev["f"], dev["e"],
        cp.asarray(coord.c1f, cp.float32), cp.asarray(coord.c2f, cp.float32),
        cp.asarray(coord.fnm, cp.float32), cp.asarray(coord.fnp, cp.float32),
        12000.0, 12000.0, ru_t, rv_t, rw_t, boundary_x=True, boundary_y=True,
    )
    ref = np_coriolis_curvature(
        arrays["ru"], arrays["rv"], arrays["u"], arrays["v"], arrays["w"],
        arrays["mut"], arrays["msft"], arrays["msfu"], arrays["msfv"],
        arrays["f"], arrays["e"], coord, 12000.0, 12000.0,
        boundary_x=True, boundary_y=True,
    )
    for got, expected in zip((ru_t, rv_t, rw_t), ref):
        np.testing.assert_allclose(cp.asnumpy(got), expected,
                                   rtol=3.0e-4, atol=4.0e-2)
    assert bool((ru_t[:, :, (0, -1)] == 0.0).all())
    assert bool((rv_t[:, (0, -1), :] == 0.0).all())


@requires_gpu
@pytest.mark.gpu
def test_specified_boundaries_are_applied_in_rk_loop_and_advance_clock():
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    cfg = RunConfig(nx=12, ny=10, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=12.0, run_seconds=24.0,
                    specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    forcing_a = init_at_rest(cfg, coord, base)
    forcing_b = init_at_rest(cfg, coord, base)
    forcing_c = init_at_rest(cfg, coord, base)
    boundaries = build_state_lateral_boundaries(
        [forcing_a, forcing_b, forcing_c], [0.0, 21600.0, 43200.0])
    attach_lateral_boundaries(state, boundaries)

    step(state, cfg)
    assert state.elapsed_seconds == 12.0
    for name in ("u", "v", "w", "thp", "php", "mup"):
        assert bool(cp.isfinite(getattr(state, name)).all())


@requires_gpu
@pytest.mark.gpu
def test_completed_step_relaxes_qv_toward_driving_data():
    """A COMPLETED model step integrates the qv Davies relaxation.

    WRF captures qv's spec+relax tendency once per step into moist_tend
    (solve_em.F:2255-2292) and every RK stage integrates it, including
    the final PD fold (module_em.F:1889-1893).  Final-review MAJOR: the
    relax previously reached only the discarded stage-1 provisional, so
    a completed step nudged relax-row qv by exactly zero.  At rest with
    a uniform +0.002 boundary deviation the one-step movement in relax
    row r is dt*f1(r)*0.002 with f1 = (1/600, 1/900, 1/1800, 0) for the
    outer-clock Davies weights (module_bdy_em.F lbc_fcx_gcx at dt=60).
    """
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    cfg = RunConfig(nx=14, ny=12, nz=6, dx=12000.0, dy=12000.0,
                    ztop=12000.0, dt=60.0, run_seconds=60.0, moist=True,
                    moist_adv_opt=1, specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    boundary_states = [init_at_rest(cfg, coord, base) for _ in range(3)]
    for s in boundary_states:
        s.qv[...] = cp.float32(0.012)
        s.qv0[...] = s.qv
    boundaries = build_state_lateral_boundaries(
        boundary_states, [0.0, 21600.0, 43200.0])
    state = init_at_rest(cfg, coord, base)
    state.qv[...] = cp.float32(0.010)
    state.qv0[...] = state.qv
    attach_lateral_boundaries(state, boundaries)

    step(state, cfg)

    qv = cp.asnumpy(state.qv)
    deviation = 0.012 - 0.010
    # Sample the west-side relax rows midway along y (away from corners).
    j = cfg.ny // 2
    for row, rate in ((1, 1.0 / 600.0), (2, 1.0 / 900.0),
                      (3, 1.0 / 1800.0)):
        moved = qv[:, j, row] - np.float32(0.010)
        expected = cfg.dt * rate * deviation
        # rtol covers the per-level chm0/chm mass-coupling variation of
        # the coupled update (measured +/-5%); the defect this pins
        # produced exactly ZERO movement.
        np.testing.assert_allclose(moved, expected, rtol=0.08,
                                   err_msg=f"relax row {row}")
    # Interior beyond the relaxation zone stays put (no relax, no motion).
    np.testing.assert_allclose(qv[:, j, 5:-5] - np.float32(0.010), 0.0,
                               atol=1.0e-6)
    # The spec row follows the driving data outright.
    np.testing.assert_allclose(qv[:, j, 0], 0.012, rtol=1.0e-3)


def _multi_time_snapshots(count=9, seed=20260731):
    """One coupled-snapshot series in the shape ingest actually builds."""
    rng = np.random.default_rng(seed)
    shapes = {
        "u": (10, 120, 151), "v": (10, 121, 150),
        "theta": (10, 120, 150), "phi": (11, 120, 150),
        "mu": (1, 120, 150), "qv": (10, 120, 150),
    }
    return [
        {name: rng.standard_normal(shape).astype(np.float32)
         for name, shape in shapes.items()}
        for _ in range(count)
    ]


def test_state_boundary_frames_match_the_all_at_once_builder_bit_for_bit():
    """The streaming accumulator is a memory change, not a numeric one.

    Ingest used to hold every forcing time's complete state so this
    builder could see them all at once.  The accumulator keeps only the
    perimeter frames, so what it produces has to be the same bytes --
    every side, every field, every interval, value and tendency.
    """
    snapshots = _multi_time_snapshots()
    start = datetime(2026, 7, 30, 0, 0, 0)
    times = [start + timedelta(hours=3 * n) for n in range(len(snapshots))]
    reference = build_lateral_boundaries(
        snapshots, times, spec_bdy_width=5, spec_zone=1, relax_zone=4)

    frames = StateBoundaryFrames(
        spec_bdy_width=5, spec_zone=1, relax_zone=4)
    for snapshot in snapshots:
        frames.add_snapshot(snapshot)
    candidate = frames.build(times)

    assert len(candidate.intervals) == len(reference.intervals) == 8
    assert candidate.spec_bdy_width == reference.spec_bdy_width
    assert candidate.spec_zone == reference.spec_zone
    assert candidate.relax_zone == reference.relax_zone
    for expected_interval, actual_interval in zip(reference.intervals,
                                                  candidate.intervals):
        assert (actual_interval.start_seconds
                == expected_interval.start_seconds)
        assert actual_interval.end_seconds == expected_interval.end_seconds
        assert set(actual_interval.fields) == set(expected_interval.fields)
        for name in expected_interval.fields:
            for side in ("west", "east", "south", "north"):
                expected = getattr(expected_interval.fields[name], side)
                actual = getattr(actual_interval.fields[name], side)
                assert actual.value.dtype == expected.value.dtype
                assert actual.value.shape == expected.value.shape
                assert (actual.value.tobytes()
                        == expected.value.tobytes()), f"{name}/{side} value"
                assert (actual.tendency.tobytes()
                        == expected.tendency.tobytes()), (
                            f"{name}/{side} tendency")


def test_state_boundary_frames_retain_only_the_perimeter():
    """The whole point: a retained time costs a perimeter, not a volume."""
    width = 5
    snapshots = _multi_time_snapshots()
    frames = StateBoundaryFrames(spec_bdy_width=width)
    for snapshot in snapshots:
        frames.add_snapshot(snapshot)

    def perimeter_elements(shape):
        nz, ny, nx = shape
        return 2 * nz * ny * width + 2 * nz * nx * width

    expected = len(snapshots) * 8 * sum(
        perimeter_elements(np.asarray(value).shape)
        for value in snapshots[0].values())
    one_time_float64 = sum(
        np.asarray(value).size * 8 for value in snapshots[0].values())

    assert len(frames) == len(snapshots)
    # Exactly the four frames, nothing else retained.
    assert frames.retained_bytes == expected
    # A retained time is a small fraction of the time it came from; the
    # all-at-once builder retained the whole thing, on the device.
    assert frames.retained_bytes / len(frames) < 0.2 * one_time_float64


def test_state_boundary_frames_reject_the_same_geometry_faults():
    snapshots = _multi_time_snapshots(count=3)
    times = [0.0, 10800.0, 21600.0]

    frames = StateBoundaryFrames(spec_bdy_width=5)
    frames.add_snapshot(snapshots[0])
    with pytest.raises(ValueError, match="inventories differ"):
        frames.add_snapshot({"u": snapshots[1]["u"]})

    narrow = {"theta": np.zeros((3, 8, 6), dtype=np.float32)}
    with pytest.raises(ValueError, match="domain is too small"):
        StateBoundaryFrames(spec_bdy_width=5).add_snapshot(narrow)

    short = StateBoundaryFrames(spec_bdy_width=5)
    short.add_snapshot(snapshots[0])
    short.add_snapshot(snapshots[1])
    with pytest.raises(ValueError, match="same length"):
        short.build(times)

    with pytest.raises(ValueError, match="spec_zone"):
        StateBoundaryFrames(spec_bdy_width=5, spec_zone=0)
    with pytest.raises(ValueError, match="spec_bdy_width must cover"):
        StateBoundaryFrames(spec_bdy_width=3, spec_zone=1, relax_zone=4)


def test_release_backend_memory_returns_cuda_blocks_and_spares_cpu():
    """The Python-level release only reaches the device through the pool."""
    from gpuwm.ingest.preprocess_backend import release_backend_memory

    freed = []

    class _Pool:
        def __init__(self, label):
            self.label = label

        def free_all_blocks(self):
            freed.append(self.label)

    class _Module:
        @staticmethod
        def get_default_memory_pool():
            return _Pool("device")

        @staticmethod
        def get_default_pinned_memory_pool():
            return _Pool("pinned")

    release_backend_memory(SimpleNamespace(name="cpu"))
    assert freed == []

    release_backend_memory(
        SimpleNamespace(name="cuda", array_module=_Module()))
    assert freed == ["device", "pinned"]

    # An array module with no pool has nothing to hand back, and this
    # helper must never be the reason a preparation fails: it decides
    # nothing and computes nothing.  A CPU stand-in swapped in for cupy
    # (which tests/test_runtime.py does) used to raise straight through
    # the middle of setup.
    import numpy as np_module

    release_backend_memory(
        SimpleNamespace(name="cuda", array_module=np_module))
    assert freed == ["device", "pinned"]
