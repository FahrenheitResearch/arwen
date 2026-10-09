"""Ordinary clock windows, one-frame handoff and borrowed member output."""
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from gpuwm.core.clock import DomainClock, DomainTicks
from gpuwm.ensemble.native_forecast import member_column_view, member_output_view, _require_step_health


def test_member_column_views_preserve_words_and_share_the_original_storage():
    words = np.arange(3 * 4 * 5 * 6, dtype=np.uint32).reshape(3, 20, 6)
    words[0, 0, 0], words[1, 5, 1] = 0x80000000, 0x7FC00019
    values = words.view(np.float32)
    for member in range(4):
        view = member_column_view(values, member=member, members=4, ny=5, nx=6)
        assert view.tobytes() == values[:, member * 5:(member + 1) * 5].tobytes()
        assert np.shares_memory(view, values)
    faces = np.zeros((3, 24, 6), np.float32)
    assert member_column_view(faces, member=3, members=4, ny=5, nx=6).shape == (3, 6, 6)
    shared = np.zeros((5, 6), np.float32)
    assert member_column_view(shared, member=3, members=4, ny=5, nx=6) is shared


def test_member_output_retains_soil_routing_and_borrowed_precipitation_diagnostics(monkeypatch):
    from gpuwm.core.microphysics import MicrophysicsDiagnostics
    from gpuwm.io.history_layout import live_state_history_fields
    import gpuwm.ensemble.batch_dycore as dycore
    monkeypatch.setattr(dycore, "member_domain_view", lambda batch, member: SimpleNamespace())
    ny, nx, members = 3, 5, 4
    plane = np.arange(members * ny * nx, dtype=np.float32).reshape(members * ny, nx)
    soil = np.arange(4 * members * ny * nx, dtype=np.float32).reshape(4, members * ny, nx)
    fields = {name: plane for name in ("snow", "snowh", "snowc", "tmn", "vegfra")}
    fields.update({name: soil for name in ("tslb", "smois", "sh2o")})
    micro = MicrophysicsDiagnostics(plane, plane, plane, snownc=plane, graupelnc=plane)
    params = object()
    dispatch = {"sf_surface_physics": "_run_noah"}
    driver = SimpleNamespace(fields=fields, microphysics=micro, noah_params=params,
        scheme_dispatch=dispatch, surface_enabled=True, output_fields=lambda: {})
    owners = SimpleNamespace(batch=SimpleNamespace(members=members, cfg=SimpleNamespace(ny=ny, nx=nx)),
                             physics=SimpleNamespace(driver=driver))
    view = member_output_view(owners, 2)
    assert view.physics.scheme_dispatch is dispatch
    assert view.physics.noah_params is params
    published = live_state_history_fields(view)
    for name in ("SNOW", "SNOWH", "SNOWC", "RAINNC", "SNOWNC", "GRAUPELNC"):
        assert published[name].tobytes() == plane[2 * ny:3 * ny].tobytes()
        assert np.shares_memory(published[name], plane)
    for name in ("TSLB", "SMOIS", "SH2O"):
        assert published[name].tobytes() == soil[:, 2 * ny:3 * ny].tobytes()
        assert np.shares_memory(published[name], soil)
    assert micro.rainnc.shape == (members * ny, nx)


def test_stability_rejects_nonfinite_members_without_inventing_a_cfl_ceiling():
    _require_step_health(({"nan": False, "cfl": 2.0},), member_ids=(7,), step=1)
    with pytest.raises(RuntimeError, match="member 7"):
        _require_step_health(({"nan": True, "cfl": 0.1},), member_ids=(7,), step=1)
    with pytest.raises(RuntimeError, match="Courant"):
        _require_step_health(({"nan": False, "cfl": float("nan")},), member_ids=(7,), step=1)
    with pytest.raises(RuntimeError, match="incomplete"):
        _require_step_health((), member_ids=(7,), step=1)


@dataclass
class _Health:
    ok: bool = True
    status_mask: int = 0


@pytest.mark.parametrize("begin,end,expected", [(0, None, (0, 24, 48)), (12, 36, (12, 36)), (0, 24, (0, 24))])
@pytest.mark.parametrize("observe_counters", [False, True])
def test_executor_uses_original_history_window_and_advances_each_clock_once(monkeypatch, begin, end, expected, observe_counters):
    import gpuwm.ensemble.native_forecast as module
    spec = DomainTicks(1, 0, 1, 12, np.float32(12), 24, None, None, None,
                       None, None, None, None, lbc_interval_ticks=24,
                       history_begin_ticks=begin, history_end_ticks=end)
    clock = DomainClock(spec, 1, 48)
    cfg = SimpleNamespace(ny=5, nx=6, dt=12.0, spec_bdy_width=1, mp_physics=8, cu_physics=0)
    batch = SimpleNamespace(cfg=cfg, members=4, elapsed_seconds=0.0,
        clock={"ticks": 0, "step_ticks": 12, "step_count": 0},
        plan=SimpleNamespace(required_bytes=lambda members: 256))
    physics = SimpleNamespace(state=SimpleNamespace(), driver=SimpleNamespace(
        refl_10cm=None, output_fields=lambda: {}, rainc=None,
        microphysics=SimpleNamespace(rainnc=np.zeros((20, 6), np.float32))))
    seen_steps, consumes, resets, captures, validations = [], [], [], [], []
    def advance(*, refl_10cm_due):
        assert not clock.at_stop_time
        seen_steps.append((clock.ticks, clock.dtbc_launch_fp32.tobytes(), refl_10cm_due))
        if refl_10cm_due:
            assert physics.driver.refl_10cm is None
            physics.driver.refl_10cm = np.full((3, 20, 6), clock.ticks + 12, np.float32)
        batch.clock["ticks"] += 12
        batch.clock["step_count"] += 1
        batch.elapsed_seconds += 12.0
        physics.driver.microphysics.rainnc += np.float32(1)
    owners = SimpleNamespace(batch=batch, physics=physics, clock=clock, tables=None,
        member_ids=(0, 1, 2, 3), member_seeds=(10, 11, 12, 13), advance=advance,
        receipt=lambda: {"ordinary_initialized_bootstraps": 1})
    monkeypatch.setattr(module, "native_prepared_eligibility", lambda *args, **kwargs: SimpleNamespace(eligible=True))
    monkeypatch.setattr(module, "prepare_native_member_batch", lambda *args, **kwargs: owners)
    monkeypatch.setattr(module, "member_output_view", lambda owners, member, **kwargs: SimpleNamespace(member=member))
    def consume(state):
        assert state is physics.state and physics.driver.refl_10cm is not None
        consumes.append(clock.ticks)
        field = physics.driver.refl_10cm
        physics.driver.refl_10cm = None
        return field
    from gpuwm.core.refl import analysis_refl_10cm
    physics.state.qv = np.zeros((3, 20, 6), np.float32)
    monkeypatch.setitem(sys.modules, "gpuwm.core.refl", SimpleNamespace(
        consume_refl_10cm=consume, analysis_refl_10cm=analysis_refl_10cm,
        refl_10cm_stash_is_due=lambda ticks, **kwargs: ticks != 0))
    monkeypatch.setitem(sys.modules, "gpuwm.core.uh_diag", SimpleNamespace(
        reset_up_heli_max=lambda state: resets.append((clock.ticks, state.member))))
    class Stability:
        def __init__(self, *args, **kwargs):
            self.plan = SimpleNamespace(required_bytes=lambda members: 256)
            self.receipt = {"members": 4}
        def __call__(self):
            return tuple({"nan": False, "cfl": 0.1} for _ in range(4))
    class Validator:
        def __init__(self, *args, **kwargs):
            self.receipt = {"members": 4}
        def require_healthy(self, *, phase):
            return tuple(_Health() for _ in range(4))
    monkeypatch.setitem(sys.modules, "gpuwm.ensemble.batch_health", SimpleNamespace(
        PreparedBatchStability=Stability, PreparedBatchStateHealth=Validator))
    pool = SimpleNamespace(used_bytes=lambda: 1024, total_bytes=lambda: 2048)
    xp = SimpleNamespace(get_default_memory_pool=lambda: pool,
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
    def submit(**kwargs):
        captures.append((clock.ticks, kwargs["member_id"], kwargs["refl_field"]))
    collector = SimpleNamespace(members=4, keep_member_files=False, submit=submit,
        require_complete=lambda: {"complete": True})
    counters = []
    if observe_counters:
        collector.capture_rain_counters = lambda fields, **kwargs: counters.append(
            (clock.ticks, kwargs["member_id"], fields["RAINNC"].copy(), kwargs["absent_zero_fields"]))
    inputs = SimpleNamespace(source="prepared", execution_plan={"kind": "root"},
        experiment=SimpleNamespace(start_time=datetime(2024, 1, 1), run_seconds=48))
    node = SimpleNamespace(clock=clock, cfg=SimpleNamespace(grid_id=1, run=cfg))
    logged, prepared = [], []
    report = module.run_initialized_native_ensemble(inputs, node, members=4, collector=collector,
        available_bytes=1 << 20, array_module=xp, output_metadata={"XLAT": np.zeros((5, 6), np.float32),
            "XLONG": np.ones((5, 6), np.float32)},
        validation_callback=lambda **kwargs: validations.append((kwargs["phase"], clock.ticks)),
        step_observer=lambda **step: logged.append(step),
        launch_prepared=lambda: prepared.append((clock.ticks, len(captures), len(validations))))
    assert report["status"] == "PASS" and report["completed_seconds"] == 48
    # The pack's steps reach the ordinary step log in its own signature.
    assert [(row["grid_id"], row["step_count"], row["model_seconds"], row["dt"]) for row in logged] == [
        (1, 1, 12.0, 12.0), (1, 2, 24.0, 12.0), (1, 3, 36.0, 12.0), (1, 4, 48.0, 12.0)]
    assert all(row["step_wall_seconds"] >= 0 for row in logged)
    # Launch preparation ended before any frame, validation or step.
    assert prepared == [(0, 0, 0)]
    import json
    json.dumps(report, allow_nan=False)
    assert clock is owners.clock and clock.ticks == 48 and clock.step_count == 4
    assert batch.clock["ticks"] == 48 and batch.clock["step_count"] == 4 and batch.elapsed_seconds == 48
    assert [tick for tick, dtbc, due in seen_steps] == [0, 12, 24, 36]
    assert [np.frombuffer(dtbc, np.float32).item() for tick, dtbc, due in seen_steps] == [12, 24, 12, 24]
    assert [tick for tick, member, field in captures] == [tick for tick in expected for _ in range(4)]
    assert consumes == [tick for tick in expected if tick]
    assert resets == [(tick, member) for tick in expected for member in range(4)]
    assert validations[0] == ("initialized", 0) and validations[-1] == ("final", 48)
    assert report["executor"]["steps"] == 4 and report["executor"]["lbc_resets"] == 2
    assert physics.driver.refl_10cm is None
    if observe_counters:
        required = sorted({0, *expected})
        assert [tick for tick, member, field, zeros in counters] == [tick for tick in required for _ in range(4)]
        for tick, member, field, zeros in counters:
            assert field.shape == (5, 6) and np.all(field == tick // 12)
            assert zeros == ("RAINC", "RAINSH")
        assert all(row["captured_ticks"] == required for row in report["precipitation_counter_calendar"])
    else:
        assert report["precipitation_counter_calendar"] is None
    for tick, member, field in captures:
        if tick:
            assert field.shape == (3, 5, 6) and np.all(field == tick)
        else:
            assert field.shape == (3, 5, 6) and not np.any(field)


@pytest.mark.parametrize("stage", ["pack", "health"])
def test_an_audit_refusal_during_launch_preparation_is_a_launch_refusal(monkeypatch, stage):
    """Nothing has advanced, so the caller may decline to the ordinary runner."""
    import gpuwm.ensemble.native_forecast as module
    from gpuwm.ensemble.batch_state import BatchStateUnsupported
    spec = DomainTicks(1, 0, 1, 12, np.float32(12), 24, None, None, None,
                       None, None, None, None, lbc_interval_ticks=24)
    clock = DomainClock(spec, 1, 48)
    cfg = SimpleNamespace(ny=5, nx=6, dt=12.0, spec_bdy_width=1, mp_physics=8, cu_physics=0)
    advanced = []
    owners = SimpleNamespace(batch=SimpleNamespace(cfg=cfg, members=2,
            plan=SimpleNamespace(required_bytes=lambda members: 256)),
        physics=SimpleNamespace(driver=SimpleNamespace()), clock=clock, tables=None,
        member_ids=(0, 1), member_seeds=(1, 2), advance=lambda **kw: advanced.append(kw))
    def prepare(*args, **kwargs):
        if stage == "pack":
            raise ValueError("legacy radiation member call-site audit changed: {'shape': 2}")
        return owners
    class Stability:
        def __init__(self, *args, **kwargs):
            self.plan = SimpleNamespace(required_bytes=lambda members: 256)
    class Validator:
        def __init__(self, *args, **kwargs):
            raise BatchStateUnsupported("native full-state health indexing changed")
    monkeypatch.setattr(module, "native_prepared_eligibility", lambda *args, **kwargs: SimpleNamespace(eligible=True))
    monkeypatch.setattr(module, "prepare_native_member_batch", prepare)
    monkeypatch.setitem(sys.modules, "gpuwm.ensemble.batch_health", SimpleNamespace(
        PreparedBatchStability=Stability, PreparedBatchStateHealth=Validator))
    monkeypatch.setitem(sys.modules, "gpuwm.core.refl", SimpleNamespace(
        consume_refl_10cm=None, refl_10cm_stash_is_due=None))
    monkeypatch.setitem(sys.modules, "gpuwm.core.uh_diag", SimpleNamespace(reset_up_heli_max=None))
    pool = SimpleNamespace(used_bytes=lambda: 1024, total_bytes=lambda: 2048)
    xp = SimpleNamespace(get_default_memory_pool=lambda: pool)
    submitted, began = [], []
    collector = SimpleNamespace(members=2, keep_member_files=False,
                                submit=lambda **kwargs: submitted.append(kwargs))
    inputs = SimpleNamespace(source="prepared", execution_plan={"kind": "root"},
        experiment=SimpleNamespace(start_time=datetime(2024, 1, 1), run_seconds=48))
    node = SimpleNamespace(clock=clock, cfg=SimpleNamespace(grid_id=1, run=cfg))
    with pytest.raises(module.NativeLaunchRefused) as caught:
        module.run_initialized_native_ensemble(inputs, node, members=2, collector=collector,
            available_bytes=1 << 20, array_module=xp, output_metadata={},
            launch_prepared=lambda: began.append(True))
    assert "before any member advanced" in caught.value.reason
    assert ("call-site audit changed" if stage == "pack" else "health indexing changed") in caught.value.reason
    assert isinstance(caught.value.__cause__, ValueError)
    assert not advanced and not submitted and not began
    assert clock.ticks == 0 and clock.step_count == 0


def test_an_ineligible_n1_exits_before_any_device_or_collector_operation(monkeypatch):
    import gpuwm.ensemble.native_forecast as module
    monkeypatch.setattr(module, "native_prepared_eligibility", lambda *args, **kwargs: SimpleNamespace(eligible=False))
    assert module.run_initialized_native_ensemble(None, None, members=1,
        collector=SimpleNamespace(keep_member_files=False), available_bytes=0) is None


@pytest.mark.parametrize("dt", [0.1, 0.3, 7.5, 1.0 / 3.0])
def test_fractional_launch_time_and_physics_cadence_match_original_refresh(dt):
    import ast
    from fractions import Fraction
    from pathlib import Path
    import struct
    from gpuwm.core.state import refresh_model_time
    from gpuwm.ensemble.batch_physics_init import InitializedMemberPhysics
    rational = Fraction(str(dt)).limit_denominator(1000)
    spec = DomainTicks(1, 0, 1, rational.numerator, np.float32(dt),
        rational.numerator * 24, None, None, None, None, None, None, None)
    clock = DomainClock(spec, rational.denominator, rational.numerator * 24)
    ordinary = SimpleNamespace()
    batch = SimpleNamespace(cfg=SimpleNamespace(dt=dt), elapsed_seconds=0.0,
                            storage=SimpleNamespace(arrays={}))
    packed = SimpleNamespace()
    adapter = InitializedMemberPhysics(batch, None, packed, None, None, None, {"model_member_fields": []})
    # Execute the actual pure fixed-clock predicates without importing the
    # CUDA driver module on a host test. The scheduler body remains source
    # authority, including its domain epoch and STEPRA/STEPBL rounding.
    path = Path(__file__).parents[1] / "gpuwm/core/physics.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_physics_interval_steps", "_radiation_step_due", "_surface_pbl_step_due", "_cumulus_step_due"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    driver = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PhysicsDriver")
    compute = next(node for node in driver.body if isinstance(node, ast.FunctionDef) and node.name == "compute")
    statements = [node for node in compute.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id in ("now", "epoch", "itimestep")
                          for target in node.targets)]
    statements.append(ast.Return(ast.Name("itimestep", ast.Load())))
    probe = ast.FunctionDef("time_index", ast.arguments(posonlyargs=[],
        args=[ast.arg("state"), ast.arg("cfg")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        statements, decorator_list=[])
    namespace = {"np": np, "Fraction": Fraction}
    code = ast.fix_missing_locations(ast.Module(functions + [probe], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    cfg = SimpleNamespace(dt=dt)
    for _ in range(24):
        refresh_model_time(ordinary, clock, kernel_launch=True)
        batch.elapsed_seconds = float(clock.elapsed_seconds_fp32)
        adapter.pack_model_state()
        assert struct.pack("!d", ordinary.elapsed_seconds) == struct.pack("!d", packed.elapsed_seconds)
        assert ordinary.domain_start_offset == 0.0
        original_step, packed_step = (namespace["time_index"](state, cfg) for state in (ordinary, packed))
        assert original_step == packed_step
        for minutes in (0.0, 0.01, 0.5):
            period = namespace["_physics_interval_steps"](minutes, dt)
            for function in ("_radiation_step_due", "_surface_pbl_step_due", "_cumulus_step_due"):
                assert namespace[function](original_step, period, minutes) == namespace[function](packed_step, period, minutes)
        refresh_model_time(ordinary, clock, after_step=True)
        clock.advance()
        batch.elapsed_seconds = clock.elapsed_seconds
        assert struct.pack("!d", ordinary.elapsed_seconds) == struct.pack("!d", batch.elapsed_seconds)
