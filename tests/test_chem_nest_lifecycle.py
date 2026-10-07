"""Chemical fields follow native/GPU child birth, relocation and restart.

The treatment is a nonzero parent plume and a distinct child plume. A zero
child or an unstamped relocation cannot satisfy these comparisons. No
weather skill or HRRR-Smoke parity is asserted.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from conftest import requires_gpu


def _tree(*, ratio=3, i_parent_start=4, sets="tracer_test", mp=0,
          child_sets=None, run_seconds=3.0, restart_interval_s=0.0,
          relocation=None, run_overrides=None):
    import cupy as cp
    from test_chem_transport import _bubble_case
    from gpuwm.experiment import (DomainConfig, ExperimentConfig,
                                  ProjectionConfig, VerticalConfig)
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree

    values = dict(chem_sets=sets, mp_physics=mp, dt=3.0,
                  run_seconds=run_seconds, output_interval_s=3.0)
    values.update(run_overrides or {})
    cfg, state = _bubble_case(**values)
    for index, row in enumerate(state.chem.transported):
        z, y, x = cp.indices(state.p.shape, dtype=cp.float32)
        getattr(state, row.state_attr)[...] = (
            cp.float32(index + 2) + x * cp.float32(0.125)
            + y * cp.float32(0.25) + z * cp.float32(0.5))
    root = DomainConfig(1, 0, 1, 1, 1, 1, 3.0, cfg, time_step=3)
    child_cfg = replace(cfg, nx=12 if ratio == 3 else 6,
                        ny=12 if ratio == 3 else 6,
                        dx=cfg.dx / ratio, dy=cfg.dy / ratio,
                        dt=3.0 / ratio, nested=True, specified=False,
                        open_x=False, open_y=False, grid_id=2)
    if child_sets is not None:
        child_cfg = replace(child_cfg, chem_sets=child_sets)
    child = DomainConfig(2, 1, i_parent_start, 4, ratio, ratio,
                          3.0, child_cfg)
    exp = ExperimentConfig(
        name="chem_nest_lifecycle", start_time=datetime(2025, 1, 1),
        run_seconds=run_seconds, domains=(root, child),
        restart_interval_s=restart_interval_s,
        projection=ProjectionConfig("lambert", 35.0, -97.0, 30.0, 60.0, -97.0),
        vertical=VerticalConfig((), 0.0, 1, 0.2))
    if relocation is not None:
        exp = replace(exp, relocation=relocation)
    return exp, assemble_idealized_tree(exp, state)


def _words(array):
    if hasattr(array, "get"):
        array = array.get()
    return np.ascontiguousarray(array).view(np.uint32)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("ratio", [1, 3])
def test_gpu_child_plume_and_time_copies_use_parent_sint(ratio):
    from gpuwm.verify.npref import np_sint

    _, model = _tree(ratio=ratio)
    parent, child = model.root, model.node(2)
    reg = child.coupler.registrations["m"]
    for row in child.state.chem.transported:
        expected = np_sint(getattr(parent.state, row.state_attr).get(),
                           reg, dtype=np.float32)
        assert expected.min() > 0
        np.testing.assert_array_equal(_words(getattr(child.state, row.state_attr)),
                                      _words(expected))
        np.testing.assert_array_equal(_words(getattr(child.state, row.time_attr)),
                                      _words(expected))
    assert child.state.chem.stacked(child.state) is child.state.chem.arena


@pytest.mark.gpu
@requires_gpu
def test_gpu_child_chemical_subset_cold_fills_from_parent_superset():
    from gpuwm.verify.npref import np_sint

    _, model = _tree(sets="gocart_primary", child_sets="dust", mp=1)
    parent, child = model.root, model.node(2)
    assert len(child.state.chem.transported) < len(parent.state.chem.transported)
    reg = child.coupler.registrations["m"]
    for row in child.state.chem.transported:
        expected = np_sint(getattr(parent.state, row.state_attr).get(),
                           reg, dtype=np.float32)
        assert expected.min() > 0
        np.testing.assert_array_equal(_words(getattr(child.state, row.state_attr)),
                                      _words(expected))
        np.testing.assert_array_equal(_words(getattr(child.state, row.time_attr)),
                                      _words(expected))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("weather_clamp", [True, False])
def test_gpu_sharp_chemical_cold_fill_repairs_only_sint_roundoff(weather_clamp):
    import cupy as cp
    from gpuwm.ingest.nest_init import parent_only_init
    from gpuwm.verify.npref import np_sint

    _, model = _tree()
    parent, child = model.root, model.node(2)
    rng = np.random.default_rng(2)
    source = np.where(rng.random((1, 12, 16)) < 0.2,
                      rng.uniform(1, 100, (1, 12, 16)), 0).astype(np.float32)
    source = np.ascontiguousarray(np.broadcast_to(source, parent.state.p.shape))
    parent.state.chem_passive_1[...] = cp.asarray(source)
    raw = np_sint(source, child.coupler.registrations["m"], dtype=np.float32)
    assert np.all(source >= 0)
    assert np.any(raw < 0), "sharp profile must reproduce SINT roundoff"
    initialized = parent_only_init(child.cfg, parent, clamp_undershoot=weather_clamp)
    actual = initialized.state.chem_passive_1.get()
    expected = np.maximum(raw, np.float32(0))
    np.testing.assert_array_equal(_words(actual), _words(expected))
    np.testing.assert_array_equal(_words(initialized.state.chem0_passive_1),
                                  _words(expected))
    assert np.all(actual >= 0)
    account = initialized.positive_definite_clamp["chem_passive_1"]
    assert account["cells"] == int((raw < 0).sum())
    assert account["reference_peak"] == float(source.max())
    assert abs(account["most_negative"]) <= account["tolerance"]


@pytest.mark.gpu
@requires_gpu
def test_host_chemical_cold_fill_refuses_the_python_numerical_backend():
    from gpuwm.ingest.nest_init import parent_only_init

    dc = SimpleNamespace(run=SimpleNamespace(chem_sets="tracer_test"))
    with pytest.raises(ValueError, match="requires a CUDA child.*host SINT backend"):
        parent_only_init(dc, object(), array_module=np)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("staging", ["host", "device"])
def test_gpu_move_carries_child_overlap_parent_strip_and_releases_owner(staging):
    import cupy as cp
    from gpuwm.core.nest_relocation import relocate_child
    from gpuwm.ingest.nest_init import parent_only_init
    from gpuwm.verify.npref import np_sint

    _, model = _tree()
    child = model.node(2)
    old_state = child.state
    from gpuwm.core.chem_prep import ChemPrep

    old_state.chem.prep = ChemPrep(old_state)
    old_state.chem.ddvel = cp.zeros((len(old_state.chem.table.rows),
                                    *old_state.p.shape[-2:]), cp.float32)
    owner_refs = [weakref.ref(old_state.chem.arena),
                  weakref.ref(old_state.chem.prep.get("rho")),
                  weakref.ref(old_state.chem.ddvel)]
    before = {}
    for index, row in enumerate(old_state.chem.transported):
        getattr(old_state, row.state_attr)[...] += cp.float32(20 + index)
        before[row.state_attr] = getattr(old_state, row.state_attr).get()
    cold = {}

    def initialize(new_dc, parent_node, **kwargs):
        if staging == "host":
            assert old_state.chem is None
            assert all(ref() is None for ref in owner_refs)
        else:
            assert all(ref() is not None for ref in owner_refs)
        initialized = parent_only_init(new_dc, parent_node, **kwargs)
        for row in initialized.state.chem.transported:
            cold[row.state_attr] = getattr(initialized.state, row.state_attr).get()
        return initialized

    receipt = relocate_child(
        child, i_parent_start=child.cfg.i_parent_start + 1,
        j_parent_start=child.cfg.j_parent_start, staging=staging,
        initializer=initialize, static_provenance="parent-interpolated ideal statics")
    shift = child.cfg.parent_grid_ratio
    reg = child.coupler.registrations["m"]
    for row in child.state.chem.transported:
        actual = getattr(child.state, row.state_attr).get()
        expected_cold = np_sint(getattr(child.parent.state, row.state_attr).get(),
                                reg, dtype=np.float32)
        np.testing.assert_array_equal(_words(cold[row.state_attr]), _words(expected_cold))
        np.testing.assert_array_equal(_words(actual[..., :-shift]),
                                      _words(before[row.state_attr][..., shift:]))
        np.testing.assert_array_equal(_words(actual[..., -shift:]),
                                      _words(expected_cold[..., -shift:]))
        assert np.any(_words(actual[..., :-shift]) != _words(expected_cold[..., :-shift]))
        np.testing.assert_array_equal(_words(getattr(child.state, row.time_attr)),
                                      _words(actual))
        assert row.state_attr in receipt["transplant"]["stamped"]
    assert child.state.chem.stacked(child.state) is child.state.chem.arena


@pytest.mark.gpu
@requires_gpu
def test_gpu_missing_parent_chemical_donor_refuses_before_host_release():
    from gpuwm.core.nest_relocation import relocate_child

    _, model = _tree()
    child = model.node(2)
    old_state, old_chem = child.state, child.state.chem
    arena_ref = weakref.ref(old_chem.arena)
    model.root.state.chem_passive_1 = None
    with pytest.raises(ValueError, match="requires parent field 'chem_passive_1'"):
        relocate_child(child, i_parent_start=5, j_parent_start=4, staging="host")
    assert child.state is old_state
    assert child.state.chem is old_chem
    assert arena_ref() is old_chem.arena
    assert old_chem.stacked(old_state) is old_chem.arena


@pytest.mark.gpu
@requires_gpu
def test_gpu_move_carries_serialized_smoke_process_fields_and_domain_ledgers():
    import cupy as cp
    from gpuwm.core.nest_relocation import relocate_child, relocatable_attrs
    from gpuwm.state_serialization_contract import serialized_state_attrs

    _, model = _tree(sets="smoke", run_overrides={"chem_sources": "rave-3km",
                                                "bl_pbl_physics": 1,
                                                "sf_sfclay_physics": 1})
    child = model.node(2)
    before = {}
    for index, name in enumerate(serialized_state_attrs(child.state)):
        if name.startswith("chemdiag_"):
            target = getattr(child.state, name)
            target[...] = cp.arange(target.size, dtype=target.dtype).reshape(target.shape) + index + 1
            before[name] = target.get()
    assert "chemdiag_fire_ebu" in before
    assert "chemdiag_plume_k_max" in before
    receipt = relocate_child(child, i_parent_start=5, j_parent_start=4, staging="host")
    for name, expected in before.items():
        actual = getattr(child.state, name).get()
        if name.startswith("chemdiag_ledger_"):
            np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
            assert receipt["transplant"]["stamped"][name]["scope"] == "domain"
        else:
            np.testing.assert_array_equal(actual[..., :-3].view(np.uint8),
                                          np.ascontiguousarray(expected[..., 3:]).view(np.uint8))
            assert not np.any(actual[..., -3:])
        assert name in relocatable_attrs(child.state)
        assert name in receipt["transplant"]["stamped"]


@pytest.mark.gpu
@requires_gpu
def test_gpu_moved_smoke_refreshes_fresh_columns_then_crosses_source_hour(tmp_path):
    import json
    import cupy as cp
    from gpuwm.core import chem_fire, chem_plumerise
    from gpuwm.core.chem_context import ChemClock
    from gpuwm.core.chem_driver import _context, _masses
    from gpuwm.core.nest_relocation import relocate_child

    exp, model = _tree(sets="smoke", run_seconds=7200.0,
                       run_overrides={"chem_sources": "rave-3km",
                                      "fire_emission_mode": "observed_hourly",
                                      "plumerisefire_frq": 60,
                                      "bl_pbl_physics": 1,
                                      "sf_sfclay_physics": 1})
    child = model.node(2)
    cfg = child.cfg.run
    state = child.state
    ny, nx = state.p.shape[-2:]
    reference_time = exp.start_time.replace(tzinfo=timezone.utc)
    stamp = int(reference_time.timestamp() // 3600)
    state.chemdiag_fire_cache_hour.fill(stamp)
    state.chemdiag_fire_cache_ref.fill(0)
    state.chemdiag_fire_hist_carry.fill(2)
    state.chemdiag_fire_coef_carry.fill(1)
    state.chemdiag_fire_coef_bb_dc.fill(1)
    state.chemdiag_plume_k_min.fill(3)
    state.chemdiag_plume_k_max.fill(6)
    state.chemdiag_plume_flam_frac.fill(0.5)
    state.chemdiag_plume_top.fill(1234)
    state.chemdiag_fire_ebu.fill(0)
    state.chemdiag_fire_ebu[:, 2:5].fill(0.5)
    old_ebu = state.chemdiag_fire_ebu.get()
    for node in model.walk_parent_first():
        node.clock.ticks = 600 * node.clock.tick_den
    relocate_child(child, i_parent_start=5, j_parent_start=4, staging="host")
    state = child.state
    state.elapsed_seconds = 600.0
    ctx = _context(state, cfg, 601)
    plane = cp.ones((ny, nx), cp.float32)
    ctx.physics_fields = {"pblh": plane * cp.float32(1000), "oro": state.ht,
                          "u10": plane * cp.float32(2), "v10": plane * cp.float32(2)}
    hours = (reference_time, reference_time + timedelta(hours=1))
    frames_read = []
    emission = ctx.table.rows_for(chem_fire.KEY)[0].emissions[0]
    emission_field = emission["field"]
    frp_field = ctx.table.sources[emission["source"]].variables[emission_field]["fire"]["frp"]

    def frame(_source, field, hour):
        frames_read.append((field, hour.isoformat()))
        assert field in (frp_field, emission_field)
        value = 20.0 if field == frp_field else (2.0 if hour == hours[1] else 1.0)
        return np.full((ny, nx), value, np.float32)

    ctx.frames = SimpleNamespace(reference_time=reference_time,
                                 available_hours=lambda *_args: hours, at=frame)
    chem_plumerise.init(ctx)
    chem_fire.init(ctx)
    initial_mass = float(_masses(state, cfg, state.chem.transported,
                                  state.chem.fields(state), state.mup).get()[0])
    chem_plumerise.step(ctx, 1.0, 601)
    assert np.all(state.chemdiag_plume_top.get()[..., :-3] == 1234)
    assert np.all(state.chemdiag_plume_k_max.get()[..., :-3] == 6)
    assert np.all(state.chemdiag_plume_k_max.get()[..., -3:] > 2)
    assert np.any(state.chemdiag_plume_top.get()[..., -3:] > 0)
    chem_fire.step(ctx, 1.0, 601)
    np.testing.assert_array_equal(state.chemdiag_fire_ebu.get()[..., :-3], old_ebu[..., 3:])
    assert np.all(state.chemdiag_fire_cache_hour.get() == stamp)
    assert np.all(state.chemdiag_fire_hist_carry.get()[..., :-3] == 2)
    assert np.all(state.chemdiag_fire_hist_carry.get()[..., -3:] == 1)
    assert np.all(state.chemdiag_fire_emitted_mass.get()[..., -3:] > 0)
    moved_mass = float(_masses(state, cfg, state.chem.transported,
                                state.chem.fields(state), state.mup).get()[0])
    assert moved_mass > initial_mass
    # GSL's end-step plume clock reaches 3600 s one step before the
    # source's start-step clock selects the next observed hourly frame.
    ctx.clock = ChemClock(0.0, 1, 3599.0, 3600, 1.0)
    chem_plumerise.step(ctx, 1.0, 3600)
    chem_fire.step(ctx, 1.0, 3600)
    hour_zero_ebu = state.chemdiag_fire_ebu.get()
    assert np.all(state.chemdiag_fire_cache_hour.get() == stamp)
    ctx.clock = ChemClock(0.0, 1, 3600.0, 3601, 1.0)
    chem_plumerise.step(ctx, 1.0, 3601)
    chem_fire.step(ctx, 1.0, 3601)
    assert np.all(state.chemdiag_fire_cache_hour.get() == stamp + 1)
    np.testing.assert_array_equal(state.chemdiag_fire_ebu.get(), hour_zero_ebu * np.float32(2))
    assert (emission_field, hours[1].isoformat()) in frames_read
    (tmp_path / "moved-smoke-process-proof.json").write_text(json.dumps({
        "source": "synthetic model-grid frames through SourceFrames interface",
        "fresh_columns": ny * 3, "retained_overlap_columns": ny * (nx - 3),
        "non_cadence_seconds": 600, "source_hour_crossed_seconds": 3600,
        "initial_mass_kg": initial_mass, "after_emission_mass_kg": moved_mass,
        "fresh_strip_emitted": True, "overlap_profile_exact": True,
        "next_source_hour_profile_doubled_exact": True,
    }, indent=2) + "\n", encoding="utf-8")


def _moved_tree_trajectory(*, checkpoint_dir=None, restart_from=None):
    """The uninterrupted and resumed routes share only their declaration."""
    import hashlib
    import cupy as cp
    from gpuwm.core.model import execute_experiment, publish_declared_experiment
    from gpuwm.core.relocation_runner import RelocationRunner
    from gpuwm.experiment import RelocationConfig, ScheduledRelocationMove
    from gpuwm.io import restart
    from gpuwm.prepared_domain_tree_forecast import fingerprint_across_stored_chain
    from gpuwm.runtime import publish_lifecycle_runners, restore_nest_followers
    from gpuwm.state_serialization_contract import serialized_state_attrs
    from gpuwm.verify.cases.nest_ideal_common import prepare_idealized_domain

    relocation = RelocationConfig(
        enabled=True, grid_id=2, cadence_seconds=3.0,
        max_move_parent_cells=1, min_overlap_fraction=0.25,
        moves=(ScheduledRelocationMove(3.0, 1, 0),))
    exp, model = _tree(run_seconds=9.0, restart_interval_s=6.0,
                       relocation=relocation)
    child = model.node(2)
    for row in child.state.chem.transported:
        getattr(child.state, row.state_attr)[...] += cp.float32(20.0)
    publish_declared_experiment(model, exp)
    ledger_checks = []
    carried_ledger = {}

    def prepare(initialized, dc, _parent):
        prepare_idealized_domain(initialized.state, dc, initialized.grid,
                                 exp.start_time)

    def capture(node):
        carried_ledger.clear()
        for name in serialized_state_attrs(node.state):
            if name.startswith("chemdiag_ledger_"):
                carried_ledger[name] = getattr(node.state, name).get()

    def after_move(node):
        for name, expected in carried_ledger.items():
            np.testing.assert_array_equal(
                np.ascontiguousarray(getattr(node.state, name).get()).view(np.uint8),
                np.ascontiguousarray(expected).view(np.uint8), err_msg=name)
        if node.clock.elapsed_seconds > 0.0:
            assert np.all(carried_ledger["chemdiag_ledger_started"] == 1)
            assert np.all(carried_ledger["chemdiag_ledger_initial"] > 0.0)
            ledger_checks.append(node.clock.elapsed_seconds)

    prepare.capture_outgoing = capture
    prepare.after_move = after_move

    runner = RelocationRunner(config=relocation, schedule=model.schedule,
                               on_child_built=prepare, staging="host")
    publish_lifecycle_runners(model, relocation_runner=runner)
    restored_at = None
    if restart_from is not None:
        header = restart.read_restart_header(restart_from)
        placements = restart.checkpoint_placements(restart_from, {1, 2})
        assert placements[2]["i_parent_start"] == 5
        assert placements[2]["j_parent_start"] == 4
        assert header["relocation"]["moved_grid_ids"] == [2]
        as_built = model.experiment_fingerprint
        runner.adopt_placement(model, child, i_parent_start=5,
                               j_parent_start=4, force=True)
        lifecycle = restart.read_tree_lifecycle_header(restart_from, model)
        assert restore_nest_followers(model, lifecycle) == [2]
        model.experiment_fingerprint = fingerprint_across_stored_chain(
            header, as_built, model)
        restart.restore_tree_restart(restart_from, model)
        restored_at = [node.clock.elapsed_seconds
                       for node in model.walk_parent_first()]
        assert restored_at == [6.0, 6.0]
        assert runner.moves_executed == 1
        for node in model.walk_parent_first():
            assert node.state.chem.stacked(node.state) is node.state.chem.arena
    checkpoints = []

    def checkpoint(tree, ticks):
        checkpoints.append(restart.write_tree_restart(
            checkpoint_dir, tree,
            exp.start_time + timedelta(seconds=ticks / tree.schedule.clock.tick_den)))

    execute_experiment(model, relocation_runner=runner,
                       restart_handler=checkpoint if checkpoint_dir else None,
                       pool_trim_per_period=False)
    cp.cuda.runtime.deviceSynchronize()
    digests = {}
    clocks = {}
    for node in model.walk_parent_first():
        gid = str(node.cfg.grid_id)
        clocks[gid] = {"ticks": node.clock.ticks,
                       "tick_den": node.clock.tick_den,
                       "steps": node.clock.step_count,
                       "elapsed_seconds": node.clock.elapsed_seconds,
                       "placement": [node.cfg.i_parent_start,
                                     node.cfg.j_parent_start]}
        names = [name for name in serialized_state_attrs(node.state)
                 if name.startswith(("chem_", "chemdiag_"))]
        assert names
        digests[gid] = {name: hashlib.sha256(
            np.ascontiguousarray(getattr(node.state, name).get()).tobytes()).hexdigest()
                       for name in names}
        assert np.all(node.state.chemdiag_ledger_started.get() == 1)
    assert clocks["2"]["placement"] == [5, 4]
    assert all(clock["elapsed_seconds"] == 9.0 for clock in clocks.values())
    assert runner.moves_executed == 1
    if restart_from is None:
        assert ledger_checks == [3.0]
    return {"fields": digests, "clocks": clocks,
            "restored_at": restored_at,
            "ledger_carried_at": ledger_checks,
            "checkpoints": [str(path) for path in checkpoints]}


@pytest.mark.gpu
@requires_gpu
def test_gpu_moved_child_tree_checkpoint_continues_in_fresh_process(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    control = _moved_tree_trajectory(checkpoint_dir=tmp_path / "control")
    assert len(control["checkpoints"]) == 1
    # A new interpreter cannot retain the old arena, RK copies, ledger flag,
    # nest table cache or movement history from the uninterrupted trajectory.
    result_path = tmp_path / "resumed.json"
    command = (
        "import json,sys; from pathlib import Path; "
        "from test_chem_nest_lifecycle import _moved_tree_trajectory; "
        "result=_moved_tree_trajectory(restart_from=sys.argv[1]); "
        "Path(sys.argv[2]).write_text(json.dumps(result,sort_keys=True),encoding='utf-8')")
    completed = subprocess.run(
        [sys.executable, "-c", command, control["checkpoints"][0], str(result_path)],
        cwd=Path(__file__).parent, capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    resumed = json.loads(result_path.read_text(encoding="utf-8"))
    assert resumed["restored_at"] == [6.0, 6.0]
    assert resumed["fields"] == control["fields"]
    assert resumed["clocks"] == control["clocks"]


@pytest.mark.gpu
@requires_gpu
def test_incomplete_dust_corridor_refuses_before_model_state_is_read():
    from gpuwm.runtime import build_prepared_tree_relocation_runner

    dc = SimpleNamespace(run=SimpleNamespace(chem_sets="dust", chem_sources=""))
    exp = SimpleNamespace(relocation=SimpleNamespace(enabled=True, follow=object(),
                                                    moves=(), grid_id=2),
                          domain=lambda _gid: dc)
    def unexpected_read(_gid):
        pytest.fail("incomplete corridor reached live child state")
    with pytest.raises(ValueError, match="requires sealed corridor statics"):
        build_prepared_tree_relocation_runner(
            exp, statics_corridor={2: SimpleNamespace(fields={})},
            model=SimpleNamespace(node=unexpected_read), outdir="unused")


@pytest.mark.gpu
@requires_gpu
def test_gpu_rebuilt_child_attaches_sealed_dust_statics():
    from gpuwm.runtime import rebuild_child_driver_from_land_state
    from gpuwm.core.chem_statics import DUST_STATIC_ALLOCATIONS
    from gpuwm.core.chem_state import process_attr

    exp, model = _tree(sets="dust", mp=1)
    child = model.node(2)
    ny, nx = child.cfg.run.ny, child.cfg.run.nx
    plane = np.ones((ny, nx), np.float64)
    static = {name: plane.copy() for name in
              ("LU_INDEX", "SCT_DOM", "LANDMASK", "SNOALB")}
    static.update(TMN=plane * 285.0,
                  GREENFRAC=np.ones((12, ny, nx), np.float64) * 0.5,
                  LAI12M=np.ones((12, ny, nx), np.float64),
                  EROD=np.ones((3, ny, nx), np.float64) * 0.25,
                  CLAYFRAC=plane * 0.125, SANDFRAC=plane * 0.75)
    rebuild_child_driver_from_land_state(
        exp=exp, data=None, model=model,
        initialized=SimpleNamespace(state=child.state, grid=child.grid,
                                    static_fields=static),
        child_dc=child.cfg, parent_node=child.parent, land={},
        landuse_attrs={"MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
                       "ISWATER": 17, "ISLAKE": 21, "ISICE": 15},
        radiation_factory=lambda *_args: None)
    by_name = {alloc.name: alloc for alloc in DUST_STATIC_ALLOCATIONS}
    for name, expected in (("dust_erod_1", 0.25), ("dust_erod_2", 0.25),
                           ("dust_erod_3", 0.25), ("dust_clayfrac", 0.125),
                           ("dust_sandfrac", 0.75), ("dust_statics_ready", 1.0)):
        target = getattr(child.state, process_attr(by_name[name]))
        np.testing.assert_array_equal(target.get(), np.full((ny, nx), expected, np.float32))
    assert child.state.chem.start_time == exp.start_time
    lat, lon = child.grid.latlon_mass()
    np.testing.assert_array_equal(child.state.physics.fields["xlat"].get(), lat.astype(np.float32))
    np.testing.assert_array_equal(child.state.physics.fields["xlong"].get(), lon.astype(np.float32))
