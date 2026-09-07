"""Two live follower clocks over the actual shared CUDA tree executor."""
from dataclasses import replace
from datetime import timedelta
import gc

import numpy as np
import pytest
from conftest import requires_gpu


def trajectory(monkeypatch, *, streamed, checkpoint_dir=None, restart_from=None):
    import cupy as cp
    from tilestream import test_moving_nest as moving, test_nest_executor as executor
    from gpuwm.core import streaming, uh_diag
    from gpuwm.core.model import execute_experiment, publish_declared_experiment
    from gpuwm.core.nest_lifecycle import DomainFollowConfig
    from gpuwm.core.relocation_runner import RelocationRunner, RelocationRunnerCollection
    from gpuwm.core.storm_tracking import FollowConfig, StormTracker
    from gpuwm.experiment import RelocationConfig
    from gpuwm.io import restart
    from gpuwm.runtime import publish_lifecycle_runners, restore_nest_followers

    monkeypatch.setattr(executor, 'I_PARENT_START', 41)
    monkeypatch.setattr(executor, 'J_PARENT_START', 41)
    monkeypatch.setattr(executor, 'PARENT_DT', 6.)
    cfg = moving.parent_cfg(nx=96, ny=96, nz=24, dt=6., run_seconds=30.,
                            nwp_diagnostics=1, radt_minutes=.5, cudt_minutes=.5)
    child_cfg = moving.child_cfg(cfg, nx=48, ny=48)
    boundaries = moving.domain_boundaries(cfg, seconds=21600.)
    state, geo = moving.build_parent(cfg, boundaries=boundaries)
    exp = executor.experiment(cfg, child_cfg, steps=5)
    tracker = FollowConfig(field='uh', threshold=1.e9, fallback_threshold=100.,
                           search_margin_cells=4, min_shift_cells=1,
                           max_shift_cells=4, cooldown_seconds=600.)
    follow = DomainFollowConfig(tracker=tracker, cadence_seconds=12.,
                                max_move_parent_cells=4, min_overlap_fraction=.25)
    child = replace(exp.domains[1], follow=follow, history_interval_s=6.)
    other = replace(child, grid_id=3, run=replace(child.run, grid_id=3), i_parent_start=25, j_parent_start=25,
                    follow=replace(follow, cadence_seconds=18.))
    exp = replace(exp, domains=(replace(exp.root, history_interval_s=6.), child, other),
                  restart_interval_s=18.)
    model = executor.assemble(exp, state, geo.grid, feedback=0)
    publish_declared_experiment(model, exp)
    runners = []
    for dc in exp.domains[1:]:
        slot = uh_diag.follow_window_slot(dc.grid_id)
        state.scratch((cfg.ny, cfg.nx), slot).fill(40.25 + dc.grid_id)
        config = RelocationConfig(enabled=True, grid_id=dc.grid_id,
            cadence_seconds=dc.follow.cadence_seconds, follow=tracker,
            max_move_parent_cells=4, min_overlap_fraction=.25)
        runners.append(RelocationRunner(config=config, schedule=model.schedule,
            provider=StormTracker(tracker, uh_slot=slot), on_child_built=lambda *_: None))
    collection = RelocationRunnerCollection(runners)
    publish_lifecycle_runners(model, relocation_runner=collection)
    steppers = {}
    if streamed:
        options = streaming.StreamingOptions(mode='on', tile_nx=48, tile_ny=48,
                                             nbuffers=2, store='host')
        steppers[1] = streaming.make_stepper(state, cfg, options,
            decision=streaming.decide(cfg, options),
            build=streaming.prepared_domain_builder(model.root, check_geography=False))
    if restart_from is not None:
        header = restart.read_tree_lifecycle_header(restart_from, model)
        assert restore_nest_followers(model, header) == [2, 3]
        restart.restore_tree_restart(restart_from, model)
    checkpoints = []
    def checkpoint(tree, ticks):
        checkpoints.append(restart.write_tree_restart(checkpoint_dir, tree,
            exp.start_time + timedelta(seconds=ticks / tree.schedule.clock.tick_den)))
    result = execute_experiment(model, steppers=steppers, validate_state=True,
        pool_trim_per_period=False, relocation_runner=collection,
        restart_handler=checkpoint if checkpoint_dir else None)
    cp.cuda.runtime.deviceSynchronize()
    digests = {dc.grid_id: moving.carrier_digest(
        steppers[dc.grid_id].store if dc.grid_id in steppers else model.node(dc.grid_id).state)
        for dc in exp.domains}
    assert model._relocation_runner is collection
    assert all(r.moves_executed == 0 for r in runners)
    calls = {gid: [row['elapsed_seconds'] for row in runner.receipts if row['event'] == 'held']
             for gid, runner in collection.runners.items()}
    out = dict(domains=digests, cadence=moving.cadence_census(model, steppers),
               calls=calls, checkpoints=checkpoints, steps=result.steps)
    del model, collection, runners, steppers, state, boundaries, geo
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    return out


@requires_gpu
@pytest.mark.gpu
def test_two_follower_windows_match_resident_and_both_restart_directions(monkeypatch, tmp_path):
    resident = trajectory(monkeypatch, streamed=False, checkpoint_dir=tmp_path/'resident')
    streamed = trajectory(monkeypatch, streamed=True, checkpoint_dir=tmp_path/'streamed')
    assert resident['calls'] == streamed['calls'] == {2: [12., 24.], 3: [18.]}
    assert streamed['domains'] == resident['domains']
    assert streamed['cadence'] == resident['cadence']
    for original, target in ((streamed, True), (streamed, False), (resident, True)):
        resumed = trajectory(monkeypatch, streamed=target, restart_from=original['checkpoints'][0])
        assert resumed['domains'] == original['domains']
        assert resumed['cadence'] == original['cadence']
        assert resumed['calls'] == {2: [24.], 3: [18.]}
    # t18 is after d02's first reset and before d03's first consultation.
    with np.load(streamed['checkpoints'][0], allow_pickle=False) as data:
        first, second = (data[f'scratch/uh_follow_window.d0{gid}'] for gid in (2, 3))
        assert second.min() >= 43.25
        assert first.max() < second.min()
    import json
    (tmp_path / 'proof.json').write_text(json.dumps({
        'seconds': 30, 'checkpoint_seconds': 18,
        'parent_tiles': 4, 'reused_buffers': 2,
        'consultations': streamed['calls'],
        'carried_field_counts': {gid: len(row[1]) for gid, row in streamed['domains'].items()},
        'resident_streamed_exact': True, 'resume_directions_exact': 3,
        'actual_relocations': 0,
    }, indent=2) + '\n', encoding='utf-8')


@requires_gpu
@pytest.mark.gpu
def test_missing_generated_tile_slots_recreates_inventory_failure(monkeypatch):
    from gpuwm.io import restart
    original = restart.lifecycle_window_slots
    monkeypatch.setattr(restart, 'lifecycle_window_slots', lambda state:
        tuple(slot for slot in original(state) if '.d' not in slot))
    with pytest.raises((ValueError, RuntimeError), match='uh_follow_window.d0[23]'):
        trajectory(monkeypatch, streamed=True)
