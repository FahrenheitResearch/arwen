"""Independent follower windows share transport, never values or cadence."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import streaming, uh_diag
from gpuwm.core.relocation_runner import RelocationRunner, RelocationRunnerCollection
from gpuwm.io import restart
from tilestream import autoplan, physics_inventory, restart_stream
from test_restart import _lifecycle_experiment, _lifecycle_tree_fixture
from test_streamed_lifecycle_restart import attach


def two_followers():
    exp = _lifecycle_experiment(follow=True)
    child = exp.domains[1]
    other = replace(child, grid_id=3, follow=replace(child.follow, cadence_seconds=720.))
    return replace(exp, domains=(*exp.domains, other))


def test_declared_parent_inventory_and_both_memory_prices():
    from gpuwm.core.preflight import estimate_experiment
    exp = two_followers()
    slots = uh_diag.declared_follower_slots(exp.domains)
    assert slots == {1: ('uh_follow_window.d02', 'uh_follow_window.d03')}
    # Removing follow alone leaves every numerical, forcing and tiling choice.
    baseline = replace(exp, domains=tuple(replace(dc, follow=None) for dc in exp.domains))
    old = estimate_experiment(baseline)
    new = estimate_experiment(exp)
    expected = 8 * exp.root.run.nx * exp.root.run.ny
    assert new.resident_bytes - old.resident_bytes == expected
    for before, after in zip(old.domains, new.domains):
        added = {row.name: row.nbytes for row in after.items if row not in before.items}
        assert added == ({slot: expected // 2 for slot in slots[1]} if after.grid_id == 1 else {})
    opts = streaming.options_for_domain(exp.root, exp.tiles)
    fp = streaming.radiation_footprint(exp.root.run, opts)
    base = autoplan.footprint_for(exp.root.run, radiation_context=opts.radiation_context)
    columns = 17 * 23
    cells = columns * exp.root.run.nz
    assert fp.buffer_bytes(cells) - base.buffer_bytes(cells) == pytest.approx(8 * columns)
    assert fp.store_bytes(cells) - base.store_bytes(cells) == pytest.approx(8 * columns * autoplan.STORE_SAFETY)
    for dc in exp.domains[1:]:
        assert streaming.options_for_domain(dc, exp.tiles).follower_context.slots == ()


def test_parent_resolution_survives_domain_override_and_nested_parent():
    exp = two_followers()
    second = replace(exp.domains[1], tiles=streaming.StreamingOptions(mode='on'))
    third = replace(exp.domains[2], parent_id=2)
    exp = replace(exp, domains=(exp.root, second, third))
    assert streaming.options_for_domain(exp.root, exp.tiles).follower_context.slots == ('uh_follow_window.d02',)
    assert streaming.options_for_domain(exp.domains[1], exp.tiles).follower_context.slots == ('uh_follow_window.d03',)
    assert streaming.options_for_domain(exp.domains[2], exp.tiles).follower_context.slots == ()


def test_no_follow_declaration_keeps_original_footprint():
    exp = _lifecycle_experiment(follow=False)
    assert exp.tiles.follower_context is None
    assert uh_diag.declared_follower_slots(exp.domains) == {}
    assert autoplan.footprint_for(exp.root.run, follower_slots=()) == autoplan.footprint_for(exp.root.run)

    former = two_followers()
    removed = replace(former, domains=tuple(replace(dc, follow=None) for dc in former.domains))
    assert removed.tiles.follower_context is None


def test_declared_dormant_windows_exist_before_attach_without_resetting_live_values(monkeypatch):
    model, _ = _lifecycle_tree_fixture(monkeypatch, follow=True, spawn=False)
    exp = two_followers()
    first = model.root.state.existing_scratch('uh_follow_window.d02')
    first.fill(18.5)
    assert 3 not in model.nodes_by_grid_id
    uh_diag.allocate_declared_follower_windows(exp, model)
    assert model.root.state.existing_scratch('uh_follow_window.d02') is first
    assert np.all(first == 18.5)
    assert not model.root.state.existing_scratch('uh_follow_window.d03').any()
    endpoint = attach(model.root)
    assert 'scratch/uh_follow_window.d03' in endpoint.store


def test_newborn_follower_resets_only_its_reserved_episode_window(monkeypatch):
    from test_nest_lifecycle_runtime import _leg_walk_model
    from gpuwm.core.model import publish_declared_experiment
    from gpuwm.runtime import walk_spawn_legs
    exp, model, spawn_runner, coupler = _leg_walk_model(monkeypatch, [])
    declared = _lifecycle_experiment(follow=True).domains[1].follow
    child = replace(exp.domains[1], follow=declared)
    exp = replace(exp, domains=(exp.root, child))
    spawn_runner.experiment = exp
    publish_declared_experiment(model, exp)
    uh_diag.allocate_declared_follower_windows(exp, model)
    shape = (model.root.cfg.run.ny, model.root.cfg.run.nx)
    sibling = model.root.state.scratch(shape, 'uh_follow_window.d03')
    sibling.fill(39.)
    born = model.root.state.existing_scratch('uh_follow_window.d02')
    born.fill(77.)
    live = {'uh_follow_window.d02': born.copy(), 'uh_follow_window.d03': sibling.copy()}
    model.root.state._streamed_scratch = live
    walk_spawn_legs(model, exp, None, spawn_runner=spawn_runner, writers=None,
                    lbc_interval_s=None, coupler_factory=coupler, validate_state=False)
    assert 2 in model.nodes_by_grid_id
    assert not born.any() and not live['uh_follow_window.d02'].any()
    assert np.all(sibling == 39.) and np.all(live['uh_follow_window.d03'] == 39.)


@pytest.mark.parametrize('streamed_restore', [False, True])
def test_generated_lifecycle_checkpoint_uses_store_and_preserves_identity(monkeypatch, tmp_path, streamed_restore):
    source, start = _lifecycle_tree_fixture(monkeypatch, follow=True, spawn=False)
    endpoint = attach(source.root)
    key = 'scratch/uh_follow_window.d02'
    endpoint.store[key].fill(92.5)
    source.root.state._scratch[key.split('/', 1)[1]].fill(-8)
    path = restart.write_tree_restart(tmp_path/'source', source, start+timedelta(seconds=3600))
    target, _ = _lifecycle_tree_fixture(monkeypatch, follow=True, spawn=False)
    restored_endpoint = attach(target.root) if streamed_restore else None
    restart.restore_tree_restart(path, target)
    actual = restored_endpoint.store[key] if streamed_restore else target.root.state._scratch[key.split('/', 1)[1]]
    np.testing.assert_array_equal(actual, endpoint.store[key])
    block = restart.read_tree_lifecycle_header(path, target)
    assert block.followers['2']['last_proposal_t'] == 60.
    assert block.followers['2']['last_move_t'] is None


def test_two_generated_carriers_have_explicit_checkpoint_opt_in(monkeypatch, tmp_path):
    model, _ = _lifecycle_tree_fixture(monkeypatch, follow=True, spawn=False)
    slots = (uh_diag.follow_window_slot(2), uh_diag.follow_window_slot(3))
    shape = (model.root.cfg.run.ny, model.root.cfg.run.nx)
    model.root.state.scratch(shape, slots[1])
    endpoint = attach(model.root)
    names = tuple('scratch/' + slot for slot in slots)
    assert set(names) <= set(physics_inventory.streaming_only_members(model.root.state))
    for i, key in enumerate(names): endpoint.store[key].fill(41.25 + i)
    selected = restart_stream._checkpoint_carriers(endpoint.store, slots)
    assert all(selected[key] is endpoint.store[key] for key in names)
    ordinary = endpoint.write_restart(tmp_path/'ordinary.npz', model.root.cfg.run)
    with np.load(ordinary.path, allow_pickle=False) as data:
        assert not set(names) & set(data.files)
    endpoint.apply_restart(endpoint.validate_restart(ordinary.path, model.root.cfg.run))
    assert all(not endpoint.store[key].any() for key in names)
    with pytest.raises(restart_stream.RestartRefused, match='missing|absent|allocated'):
        restart_stream._checkpoint_carriers(endpoint.store, (uh_diag.follow_window_slot(4),))
    with pytest.raises(restart_stream.RestartRefused, match='tracker|window|opt'):
        restart_stream._checkpoint_carriers(endpoint.store, ('scratch_typo',))


def test_two_followers_consume_only_their_own_live_window_and_keep_collection(monkeypatch):
    model, _ = _lifecycle_tree_fixture(monkeypatch, follow=True, spawn=False)
    first = model._relocation_runner.runners[2]
    other_node = SimpleNamespace(cfg=SimpleNamespace(**{**vars(model.node(2).cfg), 'grid_id': 3}), parent=model.root)
    model.nodes_by_grid_id[3] = other_node
    shape = (model.root.cfg.run.ny, model.root.cfg.run.nx)
    slots = tuple(uh_diag.follow_window_slot(gid) for gid in (2, 3))
    model.root.state.scratch(shape, slots[1])
    live = {slot: np.full(shape, 20.+i, dtype=np.float32) for i, slot in enumerate(slots)}
    model.root.state._streamed_scratch = live
    seen = []
    class ReadAndHold:
        def __init__(self, slot): self.uh_slot = slot
        def __call__(self, state, cfg, elapsed):
            from gpuwm.core.storm_tracking import signal_plane
            seen.append((cfg.grid_id, elapsed, float(signal_plane(state, 'uh', uh_slot=self.uh_slot).max())))
            return None
    runners = [RelocationRunner(config=replace(first.config, grid_id=gid, cadence_seconds=cadence),
                                schedule=model.schedule, provider=ReadAndHold(slot), on_child_built=lambda *_: None)
               for gid, cadence, slot in ((2, 360., slots[0]), (3, 720., slots[1]))]
    collection = RelocationRunnerCollection(runners)
    model._relocation_runner = collection
    clocks = {1: SimpleNamespace(ticks=360 * model.schedule.clock.tick_den, elapsed_seconds=360.)}
    collection.on_period_begin(model, clocks)
    assert seen == [(2, 360., 20.)]
    assert not live[slots[0]].any() and np.all(live[slots[1]] == 21.)
    live[slots[0]].fill(5.)
    clocks[1].ticks = 720 * model.schedule.clock.tick_den; clocks[1].elapsed_seconds = 720.
    collection.on_period_begin(model, clocks)
    assert seen[-2:] == [(2, 720., 5.), (3, 720., 21.)]
    assert not any(value.any() for value in live.values())
    assert model._relocation_runner is collection
    assert collection.target_grid_ids == (2, 3)


def test_interleaved_follower_move_receipts_keep_the_tree_chain_in_execution_order(monkeypatch):
    model, _ = _lifecycle_tree_fixture(monkeypatch, follow=True, spawn=False)
    first = model._relocation_runner.runners[2]
    second = RelocationRunner(config=replace(first.config, grid_id=3), schedule=model.schedule,
                              provider=first.provider, on_child_built=lambda *_: None)
    collection = RelocationRunnerCollection((first, second))
    model._relocation_runner = collection
    for runner, event, grid_id, mark in ((first, 'relocated', 2, 'a'),
                                       (second, 'contained', 1, 'b'),
                                       (first, 'relocated', 2, 'c')):
        runner._record(model, dict(event=event, grid_id=grid_id, record_sha256=mark * 64))
    assert [row['record_sha256'] for row in model._relocation_receipts] == [c * 64 for c in 'abc']
    assert [row['follower_grid_id'] for row in model._relocation_receipts] == [2, 3, 2]
    first._record(model, {'event': 'summary', 'value': 1}, unique=True)
    second._record(model, {'event': 'summary', 'value': 2}, unique=True)
    first._record(model, {'event': 'summary', 'value': 3}, unique=True)
    assert [row['value'] for row in model._relocation_receipts if row['event'] == 'summary'] == [2, 3]
    assert [row['record_sha256'] for row in model._relocation_receipts if 'record_sha256' in row] == [c * 64 for c in 'abc']
    assert model._relocation_runner is collection
    assert all('follower_grid_id' not in row for row in first.receipts + second.receipts)
