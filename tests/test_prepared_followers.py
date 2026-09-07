"""Prepared followers bind independent declarations before any CUDA allocation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpuwm import runtime
from gpuwm.core.relocation_runner import RelocationRunnerCollection, RelocationRefusal
from gpuwm.core.storm_tracking import FollowConfig
from gpuwm.experiment import RelocationConfig
from gpuwm.static.corridor import (
    ChildStaticsCorridor, config_declares_follow_source, corridor_frame_kwargs,
    moving_grid_ids, relocating_subtree_grid_ids,
)
from test_follow_window_transport import two_followers


def cohort():
    exp = two_followers()
    tracker = FollowConfig(field='pressure', threshold=1., search_margin_cells=4,
                          min_shift_cells=1, max_shift_cells=4, cooldown_seconds=60.)
    return replace(exp, domains=(exp.root, *(
        replace(dc, spawn=None, follow=replace(dc.follow, tracker=tracker))
        for dc in exp.domains[1:])))


def test_corridor_authority_covers_all_followers_and_descendants():
    exp = cohort()
    assert not exp.relocation.enabled
    assert config_declares_follow_source(exp)
    assert moving_grid_ids(exp) == {2, 3}
    assert relocating_subtree_grid_ids(exp) == (2, 3)
    assert corridor_frame_kwargs(exp, exp.domains[2]) == {}
    nested = replace(exp, domains=(exp.root, exp.domains[1],
                                   replace(exp.domains[2], parent_id=2)))
    assert corridor_frame_kwargs(nested, nested.domains[2])
    assert relocating_subtree_grid_ids(nested, moving_roots={3}) == (3,)
    assert relocating_subtree_grid_ids(nested, moving_roots={2}) == (2, 3)
    removed = replace(exp, domains=tuple(replace(dc, follow=None) for dc in exp.domains))
    assert not config_declares_follow_source(removed)
    assert moving_grid_ids(removed) == set()


def test_runplan_decision_has_every_target_without_legacy_identity():
    from gpuwm.runplan import follow_statics_decision
    decision = follow_statics_decision(cohort(), chain='prepared:go')
    assert decision['follower_grid_ids'] == [2, 3]
    assert decision['statics_corridor'] and decision['refusal'] is None


def test_factory_keeps_target_windows_settings_and_preparers_separate(monkeypatch, tmp_path):
    from gpuwm.static.lambert import LambertGrid
    exp = cohort()
    grid = LambertGrid(ref_lat=35., ref_lon=-97., truelat1=30., truelat2=60.,
                       stand_lon=-97., dx=1000., dy=1000., e_we=13, e_sn=13)
    nodes = {int(dc.grid_id): SimpleNamespace(cfg=dc, grid=grid) for dc in exp.domains}
    model = SimpleNamespace(nodes_by_grid_id=nodes, node=nodes.__getitem__,
        schedule=SimpleNamespace(period_ticks=60, clock=SimpleNamespace(tick_den=1)))
    allocated = []
    monkeypatch.setattr('gpuwm.core.uh_diag.allocate_declared_follower_windows',
                        lambda got, model: allocated.append(got))
    corridors = {gid: ChildStaticsCorridor(geometry={
        'grid_id':gid, 'parent_grid_ratio':3, 'child_nx':12,
        'child_ny':12, 'corridor_nx':36, 'corridor_ny':36},
        fields={}, cache_sha256=str(gid)*64) for gid in (2, 3)}
    workspace = object()
    result = runtime.build_prepared_tree_relocation_runners(
        exp, model=model, statics_corridor=corridors, outdir=tmp_path,
        radiation_workspace=workspace)
    assert isinstance(result, RelocationRunnerCollection)
    assert result.target_grid_ids == (2, 3)
    assert allocated == [exp]
    for dc in exp.domains[1:]:
        runner = result.runners[dc.grid_id]
        assert runner.config.follow is dc.follow.tracker
        assert runner.config.cadence_seconds == dc.follow.cadence_seconds
        assert runner.config.containment is None and runner.config.track is None
        assert runner.provider.uh_slot == f'uh_follow_window.d{dc.grid_id:02d}'
        assert runner.on_child_built._radiation_workspace is workspace
        assert runner.reground_descendant is None
    assert result.runners[2].on_child_built is not result.runners[3].on_child_built
    assert result.runners[2].initializer is not result.runners[3].initializer


def test_legacy_factory_is_returned_without_collection_or_window_allocation(monkeypatch, tmp_path):
    exp = cohort()
    exp = replace(exp, domains=tuple(replace(dc, follow=None) for dc in exp.domains))
    sentinel = object()
    monkeypatch.setattr(runtime, 'build_prepared_tree_relocation_runner', lambda *a, **k: sentinel)
    monkeypatch.setattr('gpuwm.core.uh_diag.allocate_declared_follower_windows',
                        lambda *a: pytest.fail('legacy path added generated windows'))
    assert runtime.build_prepared_tree_relocation_runners(
        exp, model=object(), statics_corridor=object(), outdir=tmp_path) is sentinel


def test_collection_restores_target_and_containment_through_correct_owner():
    calls = []
    def member(gid, containment=None):
        return SimpleNamespace(config=RelocationConfig(enabled=True, grid_id=gid),
            adopt_placement=lambda model, node, **kw: calls.append((gid, node.cfg.grid_id, kw)))
    first, second = member(2), member(3)
    result = RelocationRunnerCollection((first, second))
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=3))
    result.adopt_placement(object(), node, i_parent_start=24, j_parent_start=19, force=True)
    assert calls == [(3, 3, dict(i_parent_start=24, j_parent_start=19, force=True))]
    with pytest.raises(RelocationRefusal, match='no unique'):
        result.adopt_placement(object(), SimpleNamespace(cfg=SimpleNamespace(grid_id=4)),
                               i_parent_start=1, j_parent_start=1)


def test_collection_writer_attachment_includes_descendant_and_containment():
    calls = []
    runners = []
    for gid in (2, 3):
        def attach(label):
            return SimpleNamespace(attach_writers=lambda writer: calls.append((label, writer)))
        runners.append(SimpleNamespace(config=SimpleNamespace(grid_id=gid),
            on_child_built=attach((gid, 'target')),
            reground_descendant=attach((gid, 'descendants')),
            containment_preparer=attach((gid, 'containment'))))
    writer = object()
    RelocationRunnerCollection(runners).attach_writers(writer)
    assert calls == [((gid, role), writer) for gid in (2, 3)
                     for role in ('target', 'descendants', 'containment')]

@pytest.mark.parametrize('document', ['hrrr', 'gfs'])
def test_prepared_manifest_requires_and_loads_every_declared_corridor(tmp_path, monkeypatch, document):
    import tomllib
    import test_prepared_domain_tree_forecast as fixture
    from gpuwm.branch import emit_experiment_toml
    original = fixture._write_two_domain_config
    def write(directory):
        path = original(directory)
        raw = tomllib.loads(path.read_text(encoding='utf-8'))
        raw['domain'].append({**raw['domain'][1], 'grid_id':3})
        for gid, table in enumerate(raw['domain'][1:], 2):
            table['follow'] = {key:value for key,value in cohort().domains[gid-1].follow.to_json().items() if key in {'field','threshold','search_margin_cells','min_shift_cells','max_shift_cells','cooldown_seconds','level_hpa','cadence_seconds','max_move_parent_cells','min_overlap_fraction'}}
        path.write_text(emit_experiment_toml(raw), encoding='utf-8')
        return path
    monkeypatch.setattr(fixture, '_write_two_domain_config', write)
    prepared, receipt, config = fixture._synthetic_prepared_tree(tmp_path, monkeypatch)
    # Extend the old two-domain fixture's envelope to its three actual caches.
    import json
    preparation = json.loads(receipt.read_text(encoding='utf-8'))
    preparation['domain_count'] = 3
    preparation['artifact_receipt']['domain_count'] = 3
    preparation['artifact_receipt']['boundary_inventory']['nested_parent_forced'] = [2, 3]
    (prepared/'hierarchy-artifacts/receipt.json').write_text(
        json.dumps(preparation['artifact_receipt']), encoding='utf-8')
    receipt.write_text(json.dumps(preparation), encoding='utf-8')
    monkeypatch.setattr(fixture.runner, 'grids_from_projection_config',
                        lambda exp: tuple(object() for _ in exp.domains))
    receipt = fixture._as_document(prepared, receipt, document)
    with pytest.raises(ValueError, match='--statics-corridor'):
        fixture._preflight(prepared, receipt, config)
    fixture._bind_corridor(receipt, fixture._D02_CORRIDOR_SET)
    with pytest.raises(ValueError, match=r"missing \['d03'\]"):
        fixture._preflight(prepared, receipt, config)
    corridor_set = {'schema':'gpuwm-statics-corridor-set-v1', 'status':'READY',
                   'domains':{f'd{gid:02d}':{'cache':{'path':f'd{gid:02d}.npz',
                                                    'sha256':str(gid)*64}}
                              for gid in (2, 3)}}
    fixture._bind_corridor(receipt, corridor_set)
    calls = []
    def load(directory, **kwargs):
        calls.append(kwargs)
        return kwargs['grid_id']
    monkeypatch.setattr('gpuwm.static.corridor.load_child_statics_corridor', load)
    result = fixture._preflight(prepared, receipt, config)
    assert result.statics_corridor == {2:2, 3:3}
    assert [row['grid_id'] for row in calls] == [2, 3]
    assert all(row['expected_set_receipt'] == corridor_set for row in calls)
    assert all(row['frame_kwargs'] == {} for row in calls)

def test_collection_summary_preserves_tree_restore_event_and_other_follower():
    collection = RelocationRunnerCollection()
    tree = {'event':'follower-state-restored', 'grid_ids':[2, 3]}
    collection.receipts.append(tree)
    collection.record(2, {'event':'summary', 'moves_executed':1}, unique=True)
    collection.record(3, {'event':'summary', 'moves_executed':2}, unique=True)
    collection.record(2, {'event':'summary', 'moves_executed':3}, unique=True)
    assert collection.receipts == [tree,
        {'event':'summary', 'follower_grid_id':3, 'moves_executed':2},
        {'event':'summary', 'follower_grid_id':2, 'moves_executed':3}]

@pytest.mark.parametrize('activation', ['spawn', 'delayed'])
def test_prepared_follower_activation_keeps_route_and_corridor_contracts(
        tmp_path, monkeypatch, activation):
    import tomllib
    import test_prepared_domain_tree_forecast as fixture
    from gpuwm.branch import emit_experiment_toml
    from gpuwm.core.nest_lifecycle import DOMAIN_FOLLOW_EXTRA_KEYS
    from gpuwm.core.storm_tracking import FOLLOW_KEYS
    original = fixture._write_two_domain_config
    def write(directory):
        config = original(directory)
        raw = tomllib.loads(config.read_text(encoding='utf-8'))
        child = raw['domain'][1]
        child['follow'] = {
            key:value for key,value in cohort().domains[1].follow.to_json().items()
            if key in FOLLOW_KEYS or key in DOMAIN_FOLLOW_EXTRA_KEYS}
        if activation == 'spawn':
            child['spawn'] = {'trigger':'time', 'at_s':120.}
        config.write_text(emit_experiment_toml(raw), encoding='utf-8')
        return config

    monkeypatch.setattr(fixture, '_write_two_domain_config', write)
    prepared, receipt, config = fixture._synthetic_prepared_tree(
        tmp_path, monkeypatch, delayed=activation == 'delayed')
    if activation == 'spawn':
        # This route still has no trigger evaluator; refuse before loading
        # corridors or accepting a permanently absent configured child.
        with pytest.raises(ValueError, match='does not implement spawn-triggered nests'):
            fixture._preflight(prepared, receipt, config)
        return

    # A fixed delayed start is implemented. Its dated child analysis must
    # retain the follow declaration and require the same sealed corridor.
    with pytest.raises(ValueError, match='--statics-corridor'):
        fixture._preflight(prepared, receipt, config)
    fixture._bind_corridor(receipt, fixture._D02_CORRIDOR_SET)
    corridor = object()
    loaded = []
    def load(directory, **kwargs):
        loaded.append(kwargs)
        return corridor
    monkeypatch.setattr('gpuwm.static.corridor.load_child_statics_corridor', load)
    inputs = fixture._preflight(prepared, receipt, config)
    assert inputs.experiment.domain_start_offset_exact(2) == 3600
    assert inputs.experiment.domains[1].follow is not None
    assert inputs.domains[1].cache_reader.header['metadata']['user']['initial_valid_time'] \
        == '2026-07-23T01:00:00'
    assert inputs.statics_corridor == {2:corridor}
    assert len(loaded) == 1 and loaded[0]['grid_id'] == 2
    assert loaded[0]['expected_set_receipt'] == fixture._D02_CORRIDOR_SET

def test_bounds_only_legacy_parent_does_not_become_an_extra_mover():
    exp = cohort()
    parent = replace(exp.domains[1], follow=None)
    child = replace(exp.domains[2], parent_id=2)
    without_bounds = replace(exp, domains=(exp.root, parent, child))
    with_bounds = replace(without_bounds,
        relocation=RelocationConfig(enabled=True, grid_id=2))
    for declared in (without_bounds, with_bounds):
        assert config_declares_follow_source(declared)
        assert moving_grid_ids(declared) == {3}
        assert relocating_subtree_grid_ids(declared) == (3,)
        assert corridor_frame_kwargs(declared, declared.domains[2]) == {}
    bounds_only = replace(with_bounds, domains=tuple(
        replace(dc, follow=None) for dc in with_bounds.domains))
    assert not config_declares_follow_source(bounds_only)
    assert moving_grid_ids(bounds_only) == set()
    assert relocating_subtree_grid_ids(bounds_only) == ()
