"""Real dry RK boundary updates compared with independent scalar helpers."""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]
MEMBERS = (1, 4, 10, 20, 40)


def _setup(members, mapped, mode, nonlinear=False):
    import cupy as cp
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.core.preflight import scratch_slot_registry
    from gpuwm.ensemble.batch_state import BatchedDomainState, PreparedHostMember, SHARED_STATE_CANDIDATES
    from gpuwm.ingest import lateral_bc as scalar
    from test_ensemble_batch_acoustic_gpu import _pack_physical

    old, references, before_cfg, _ = _pack_physical(members, mapped=mapped,
        boundary="nested" if mode == "nested" else "specified", ny=11, nx=13)
    del old
    cfg = replace(before_cfg, specified=mode != "nested", nested=mode == "nested",
                  spec_bdy_width=3, spec_zone=1, relax_zone=2, relax_w=mode == "relax_w")
    boundaries, prepared = [], []
    for member, state in enumerate(references):
        for name in ("u", "v", "w", "thp", "php", "mup"):
            getattr(state, name + "0")[...] = getattr(state, name)
        first = dict(scalar.domain_boundary_snapshot(state))
        if mode in ("nested", "relax_w"):
            mass = state.mub2d + state.mup
            w = (state.c1f[:, None, None] * mass[None] + state.c2f[:, None, None]) * state.w
            if mapped:
                w /= state.msft[None]
            first["w"] = cp.asnumpy(w).astype(np.float64)
        second = {name: value + (member + 1) * 0.0125 * 90.0 for name, value in first.items()}
        bdy = scalar.build_lateral_boundaries([first, second], [0.0, 90.0],
            spec_bdy_width=3, spec_zone=1, relax_zone=2)
        if nonlinear:
            rows = {}
            for name, field in bdy.intervals[0].fields.items():
                sides = {}
                for side in ("west", "east", "south", "north"):
                    record = getattr(field, side)
                    sides[side] = scalar.SideBoundary(record.value, record.tendency,
                        scalar.RationalTimeLaw(np.full(record.value.shape, 1.0e-5 * (member + 1)),
                                               np.full(record.value.shape, 1.0e-4)))
                rows[name] = scalar.FieldBoundary(**sides)
            bdy = scalar.LateralBoundaries((scalar.BoundaryInterval(0.0, 90.0, rows),), 3, 1, 2)
        boundaries.append(bdy)
        shapes = state_array_shapes(cfg)
        controls = {"physics", "lateral_boundaries", "_scratch", "_scratch_arena", "_host_setup_state", "_phb_host"}
        scalars = {name: value for name, value in vars(state).items() if name not in shapes and name not in controls}
        clock_snapshot = {"ticks": 0, "step_ticks": 3, "tick_den": 1, "run_ticks": 30,
            "step_count": 0, "dt_fp32": np.float32(3.0), "dtbc_fp32": np.float32(0.0)}
        prepared.append(PreparedHostMember(cfg, {name: cp.asnumpy(getattr(state, name)) for name in shapes},
            scalars, clock_snapshot, scratch={name: cp.asnumpy(value) for name, value in state._scratch.items()},
            phb_host=state._phb_host))
    registry = scratch_slot_registry(cfg)
    slots = {name: np.float32 for name in registry if name.startswith("lbc_")}
    state = BatchedDomainState.from_prepared(prepared, array_module=cp, available_bytes=2**30,
        shared_fields=tuple(SHARED_STATE_CANDIDATES & state_array_shapes(cfg).keys()), scratch_slots=slots)
    clock = SimpleNamespace(elapsed_seconds=4.25, dt_fp32=np.float32(3.0), dtbc_launch_fp32=np.float32(7.25))
    for member, reference in enumerate(references):
        scalar.attach_lateral_boundaries(reference, boundaries[member])
        if mode == "nested":
            reference._lateral_boundary_device.rolling = True
            reference._lateral_boundary_device.valid = True
            reference._lateral_boundary_device.clock = clock
        else:
            reference._lateral_boundary_device.clock = clock
    return state, references, cfg, tuple(boundaries), clock


def _words(cp, observed, expected):
    assert cp.asnumpy(observed).view(np.uint32).tobytes() == cp.asnumpy(expected).view(np.uint32).tobytes()


@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("mapped", (False, True))
@pytest.mark.parametrize("mode", ("specified", "relax_w", "nested"))
def test_dry_rk_tendencies_and_final_prognostics_equal_scalar(members, mapped, mode):
    import cupy as cp
    from gpuwm.ensemble.batch_boundaries import (
        MemberBoundaryTables, prepare_dry_lateral_tendencies, prepare_dry_boundary_values,
        prepare_specified_w_zero_gradient)
    from gpuwm.ingest import lateral_bc as scalar

    state, references, cfg, sources, clock = _setup(members, mapped, mode)
    before = cp.get_default_memory_pool().used_bytes()
    tables = MemberBoundaryTables.from_prepared(sources, cfg, array_module=cp, available_bytes=2**30)
    cp.cuda.get_current_stream().synchronize()
    assert cp.get_default_memory_pool().used_bytes() - before == tables.plan.required_bytes(members)
    tendencies = ("ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t")
    for stage in range(3):
        for name in tendencies:
            getattr(state, name).fill(np.float32(0.0))
            for reference in references:
                getattr(reference, name).fill(np.float32(0.0))
        prepare_dry_lateral_tendencies(state, cfg, tables, clock, rk_stage=stage)()
        for reference in references:
            scalar.apply_state_lateral_boundaries(reference, cfg, rk_stage=stage)
        cp.cuda.get_current_stream().synchronize()
        for member, reference in enumerate(references):
            for name in tendencies:
                _words(cp, state.member_view(name, member), getattr(reference, name))
        # RK stage estimates change while saved time-t fields stay fixed.
        for name in ("u", "v", "thp", "php", "mup"):
            getattr(state, name)[:] += np.float32(0.001 * (stage + 1))
            for reference in references:
                getattr(reference, name)[:] += np.float32(0.001 * (stage + 1))
    prepare_dry_boundary_values(state, cfg, tables, clock)()
    prepare_specified_w_zero_gradient(state, cfg, tables)()
    for reference in references:
        scalar.apply_state_boundary_values(reference, cfg)
        scalar.apply_specified_w_zero_gradient(reference, cfg)
    cp.cuda.get_current_stream().synchronize()
    for member, reference in enumerate(references):
        for name in ("u", "v", "w", "thp", "php", "mup"):
            _words(cp, state.member_view(name, member), getattr(reference, name))


@pytest.mark.parametrize("members", MEMBERS)
def test_rational_member_time_tables_preserve_scalar_reconstruction(members):
    import cupy as cp
    from gpuwm.ensemble.batch_boundaries import MemberBoundaryTables, prepare_dry_lateral_tendencies, prepare_dry_boundary_values
    from gpuwm.ingest import lateral_bc as scalar

    state, references, cfg, sources, clock = _setup(members, True, "specified", nonlinear=True)
    tables = MemberBoundaryTables.from_prepared(sources, cfg, array_module=cp, available_bytes=2**30)
    for stage in range(3):
        prepare_dry_lateral_tendencies(state, cfg, tables, clock, rk_stage=stage)()
        for reference in references:
            scalar.apply_state_lateral_boundaries(reference, cfg, rk_stage=stage)
    prepare_dry_boundary_values(state, cfg, tables, clock)()
    for reference in references:
        scalar.apply_state_boundary_values(reference, cfg)
    cp.cuda.get_current_stream().synchronize()
    for member, reference in enumerate(references):
        for name in ("ru_t", "rv_t", "rth_t", "rph_t", "rmu_t", "u", "v", "thp", "php", "mup"):
            _words(cp, state.member_view(name, member), getattr(reference, name))
    selection, evaluated = tables.evaluate(tables.select(clock))
    assert evaluated and selection.offset.tobytes() == np.float32(0.0).tobytes()


@pytest.mark.parametrize("members", MEMBERS)
def test_interval_endpoint_uses_solve_entry_record_before_next_step(members):
    import cupy as cp
    from gpuwm.ensemble.batch_boundaries import MemberBoundaryTables, prepare_dry_boundary_values
    from gpuwm.ingest import lateral_bc as scalar

    state, references, cfg, sources, _clock = _setup(members, True, "specified")
    extended = []
    for source in sources:
        first = source.intervals[0]
        fields = {}
        for name, field in first.fields.items():
            fields[name] = scalar.FieldBoundary(**{
                side: scalar.SideBoundary(
                    getattr(field, side).value + 90.0 * getattr(field, side).tendency,
                    1.125 * getattr(field, side).tendency)
                for side in ("west", "east", "south", "north")})
        extended.append(scalar.LateralBoundaries(
            (first, scalar.BoundaryInterval(90.0, 180.0, fields)), 3, 1, 2))
    tables = MemberBoundaryTables.from_prepared(extended, cfg, array_module=cp, available_bytes=2**30)
    for member, reference in enumerate(references):
        scalar.attach_lateral_boundaries(reference, extended[member])
    # The final old-record reconstruction and the following solve's new-record
    # reconstruction have different FP32 round points even for continuous data.
    for elapsed, offset, interval in ((87.0, 90.0, 0), (90.0, 3.0, 1)):
        clock = SimpleNamespace(elapsed_seconds=elapsed, dt_fp32=np.float32(3.0),
                                dtbc_launch_fp32=np.float32(offset))
        assert tables.select(clock).interval == interval
        prepare_dry_boundary_values(state, cfg, tables, clock)()
        for reference in references:
            reference._lateral_boundary_device.clock = clock
            scalar.apply_state_boundary_values(reference, cfg)
        cp.cuda.get_current_stream().synchronize()
        for member, reference in enumerate(references):
            for name in ("u", "v", "thp", "php", "mup"):
                _words(cp, state.member_view(name, member), getattr(reference, name))


def test_common_clock_schema_and_admission_refusals():
    import cupy as cp
    from gpuwm.ensemble.batch_boundaries import MemberBoundaryTables

    state, references, cfg, sources, clock = _setup(4, False, "specified")
    tables = MemberBoundaryTables.from_prepared(sources, cfg, array_module=cp, available_bytes=2**30)
    with pytest.raises(MemoryError, match="resident ensemble members"):
        MemberBoundaryTables.from_prepared(sources, cfg, array_module=cp, available_bytes=0)
    with pytest.raises(ValueError, match="fixed admitted model step"):
        tables.select(SimpleNamespace(elapsed_seconds=0.0, dt_fp32=np.float32(6.0), dtbc_launch_fp32=np.float32(3.0)))
    with pytest.raises(TypeError, match="float32"):
        tables.select(SimpleNamespace(elapsed_seconds=0.0, dt_fp32=3.0, dtbc_launch_fp32=3.0))


def test_periodic_without_forcing_needs_no_tables_or_device_arrays():
    from gpuwm.ensemble.batch_boundaries import (
        prepare_dry_lateral_tendencies, prepare_dry_boundary_values, prepare_specified_w_zero_gradient)

    cfg = SimpleNamespace(specified=False, nested=False)
    assert prepare_dry_lateral_tendencies(None, cfg, None, None, rk_stage=0)() is None
    assert prepare_dry_boundary_values(None, cfg, None, None)() is None
    assert prepare_specified_w_zero_gradient(None, cfg, None)() is None


def test_real_singleton_nested_prefix_reshape_retains_actual_parent_allocation():
    import cupy as cp
    from gpuwm.ensemble.batch_boundaries import MemberBoundaryTables

    tables = MemberBoundaryTables()
    tables.members = 1
    tables._views, tables._view_owners, tables._view_cache = {}, {}, {}
    parent = cp.empty((1, 1200), cp.float32)
    held = tables.workspace_view(parent, (5, 13, 14))
    assert held.strides[0] != parent.strides[0]
    assert held.data.ptr == parent.data.ptr
    assert held.data.mem.ptr == parent.data.mem.ptr
    assert tables._views[id(held)] == ("member", parent.strides[0])
    assert tables._view_owners[id(held)][0] is held
    assert tables._view_owners[id(held)][1] is parent
    assert tables.workspace_view(parent, (5, 13, 14)) is held


def test_member_prefix_cpu_bounds_and_stride_guards():
    from gpuwm.ensemble.batch_boundaries import MemberBoundaryTables

    def owner(members):
        tables = MemberBoundaryTables()
        tables.members = members
        tables._views, tables._view_owners = {}, {}
        return tables

    def array(shape, strides, *, address=1024, allocation=1024, capacity=32768,
              dtype="float32", memory=None):
        return SimpleNamespace(shape=shape, strides=strides, dtype=np.dtype(dtype),
            nbytes=int(np.prod(shape)) * np.dtype(dtype).itemsize,
            __cuda_array_interface__={"data": (address, False)},
            data=SimpleNamespace(ptr=address, mem=memory or SimpleNamespace(ptr=allocation, size=capacity)))

    parent = array((1, 1200), (4800, 4))
    valid = array((1, 5, 13, 14), (3640, 728, 56, 4))
    singleton = owner(1)
    singleton.register_member_view(valid, parent)
    assert singleton._views[id(valid)] == ("member", 4800)
    assert singleton._view_owners[id(valid)][0] is valid
    assert singleton._view_owners[id(valid)][1] is parent
    batched = owner(4)
    padded = array((4, 5, 13, 14), (4800, 728, 56, 4))
    batched.register_member_view(padded, array((4, 1200), (4800, 4)))
    assert batched._views[id(padded)] == ("member", 4800)
    with pytest.raises(ValueError, match="admitted member backing"):
        owner(4).register_member_view(array((4, 5, 13, 14), (3640, 728, 56, 4)),
                                       array((4, 1200), (4800, 4)))
    with pytest.raises(ValueError, match="contiguous inner axes"):
        singleton.register_member_view(array((1, 5, 13, 14), (3640, 728, 56, 8)), parent)
    with pytest.raises(ValueError, match="admitted member backing"):
        singleton.register_member_view(array((1, 5, 13, 14), (3640, 728, 56, 4), address=3024), parent)
    with pytest.raises(ValueError, match="parent allocation"):
        singleton.register_member_view(array((1, 5, 13, 14), (3640, 728, 56, 4), allocation=0), parent)
    with pytest.raises(ValueError, match="allocation bounds"):
        singleton.register_member_view(array((1, 5, 13, 14), (3640, 728, 56, 4), capacity=100), parent)
    unowned = type("UnownedMemory", (), {"ptr": 1024, "size": 32768, "owner": None})()
    with pytest.raises(ValueError, match="owned allocation"):
        singleton.register_member_view(array((1, 5, 13, 14), (3640, 728, 56, 4), memory=unowned), parent)
