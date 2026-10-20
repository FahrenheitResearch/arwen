"""Registry sources, family ladders and per-member acquisition contracts, no CUDA."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.wofs_recipe import (
    LadderRefusal, WofsOptions, parse_source_spec, plan_families, plan_ladder,
    summary_lines, expand_config,
)
from gpuwm.ensemble.seeds import member_seed
from gpuwm.source_adapters import (
    canonical_source_token, ensemble_source_ids, get_source_adapter,
)

START = datetime(2022, 6, 13, 18, tzinfo=timezone.utc)


@pytest.mark.parametrize('value,members,expected', [
    ('gfs', None, [('gfs', 36)]), ('rap', 4, [('rap', 4)]),
    ('gefs:18,gfs:18', None, [('gefs', 18), ('gfs', 18)]),
    ('earth2-fourcastnet3:18,gefs:18', 36, [('earth2-fcn3', 18), ('gefs', 18)]),
    ([{'source': 'rap-native', 'members': 4}], 4, [('rap-native', 4)]),
])
def test_source_spec(value, members, expected):
    assert [(r['source'], r['members']) for r in parse_source_spec(value, members)] == expected


@pytest.mark.parametrize('value,members,match', [
    ('gefs:18,gfs:18', 35, '36.*35'), ('gfs:4', 36, '4.*36'),
    ('gefs,gfs', None, 'positive whole'), ('rap:0', None, 'positive whole'),
    ('rap:1.5', None, 'positive whole'), ('rap:2,rap:2', None, 'more than once'),
    ('not-a-source', None, 'eligible ids:.*rap'), ('rrfs-firewx', None, 'eligible ids:'),
    ([{'source': 'rap', 'members': True}], None, 'positive whole'),
])
def test_source_spec_refusal(value, members, match):
    with pytest.raises(LadderRefusal, match=match):
        parse_source_spec(value, members)


def test_cli_choices_follow_registry_and_aliases(capsys):
    import argparse
    from gpuwm.go_cli import register_cli
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest='command'))
    help_text = parser._subparsers._group_actions[0].choices['ensemble'].format_help()
    for source in ensemble_source_ids():
        assert source in "".join(help_text.split())
    assert parser.parse_args(['ensemble', '--source', 'rap-native']).source == 'rap-native'
    assert parser.parse_args(['ensemble', '--source', 'earth2-fourcastnet3:2,gefs:2']).source
    with pytest.raises(SystemExit):
        parser.parse_args(['ensemble', '--source', 'bad-source'])
    assert 'eligible ids:' in capsys.readouterr().err


def test_parameterized_route_is_generic_table_data(tmp_path, monkeypatch):
    from gpuwm import source_adapters as registry
    table = tmp_path / 'models.json'
    table.write_text(json.dumps({'models': {'example': {
        'aliases': ['other'], 'profile': 'earth2-pl13-q-netcdf-v1', 'step_hours': 3,
    }}}))
    row = replace(get_source_adapter('earth2'), source_id='synthetic', aliases=(), model_table=str(table))
    monkeypatch.setattr(registry, '_ADAPTERS', (*registry.source_adapters(), row))
    monkeypatch.setattr(registry, '_ALIASES', {**registry._ALIASES, 'synthetic': row})
    chosen = get_source_adapter('synthetic-other')
    assert chosen.source_id == 'synthetic'
    assert chosen.model_key == 'example'
    assert chosen.forcing_interval_seconds == 10800
    assert canonical_source_token('synthetic-other') == 'synthetic-example'
    assert 'synthetic-example' in ensemble_source_ids()
    # Materialized rows must obey the same contract as dynamic table lookup.
    materialized, = registry._model_table_rows((row,))
    assert materialized.source_id == 'synthetic-example'
    assert materialized.model_key == 'example'
    assert materialized.aliases == ('synthetic-other',)
    assert materialized.forcing_interval_seconds == 10800
    monkeypatch.setattr(registry, '_ALIASES', {**registry._ALIASES,
        'synthetic-example': materialized, 'synthetic-other': materialized})
    assert canonical_source_token('synthetic-other') == 'synthetic-example'
    from gpuwm.ensemble.recipes import SourceTrajectory
    trajectory = SourceTrajectory('synthetic-other', START)
    assert trajectory.source == 'synthetic-example'
    assert trajectory.model == 'example'


def test_each_model_row_cadence_reaches_its_sealed_preparation_contract():
    from gpuwm.source_adapters import source_adapters
    from gpuwm.source_authorities import packaged_mapping_target, boundary_interval_refusal
    for adapter in source_adapters():
        if adapter.model_key is None or not adapter.runnable:
            continue
        target = packaged_mapping_target(adapter.packaged_profile)
        assert boundary_interval_refusal(target, adapter.forcing_interval_seconds) is None, adapter.source_id
        assert boundary_interval_refusal(target, 1800) is not None, adapter.source_id


def test_every_declared_model_key_and_alias_resolves_to_its_own_row():
    from gpuwm.source_adapters import source_adapters, model_rows
    inventory = source_adapters()
    for template in inventory:
        if template.model_table is None or template.model_key is not None:
            continue
        assert sum(row.source_id == template.source_id for row in inventory) == 1
        for key, model in model_rows(template).items():
            expected = template.source_id + '-' + key
            for spelling in (key, *model.get('aliases', ())):
                token = template.source_id + '-' + spelling
                selected = get_source_adapter(token)
                assert selected.model_key == key, token
                assert selected.model_row == model, token
                assert canonical_source_token(token) == expected, token
                assert get_source_adapter(token.upper().replace('-', '_')) == selected


def test_model_without_profile_keeps_its_composition_refusal(tmp_path, monkeypatch):
    from gpuwm import source_adapters as registry
    table = tmp_path / 'models.json'
    table.write_text(json.dumps({'models': {'pending': {'needs': 'missing pressure column'}}}))
    template = replace(get_source_adapter('earth2'), source_id='synthetic', aliases=(), model_table=str(table))
    monkeypatch.setattr(registry, '_ADAPTERS', (template,))
    for row in (registry.get_source_adapter('synthetic-pending'), *registry._model_table_rows((template,))):
        assert not row.runnable
        assert row.packaged_profile is None
        assert row.status is registry.AdapterStatus.COMPOSITION_REQUIRED
        assert row.notes == 'missing pressure column'


def test_mixed_ladders_use_family_rungs_global_seeds_and_arms():
    options = WofsOptions(members=36, max_lag_hours=6, base_seed=333)
    ladder = plan_families(sources='gefs:18,gfs:18', cycle=START, start=START,
                          end=START + timedelta(hours=6), options=options)
    assert ladder.family_counts() == {
        'gefs': {'source-member': 18, 'time-lagged': 0, 'perturbation': 0},
        'gfs': {'source-member': 1, 'time-lagged': 1, 'perturbation': 16}}
    assert [m.index for m in ladder.members] == list(range(36))
    assert [m.seed for m in ladder.members] == [member_seed(333, i) for i in range(36)]
    assert ladder.members[20].surface is not None
    assert ladder.members[20].physics_arm == 'thompson-ysu'
    assert ladder.describe()['families'] == ladder.family_counts()
    assert any('wofs family gfs: 18 members' in line for line in summary_lines(ladder))
    assert {m.provenance()['family'] for m in ladder.members} == {'gfs', 'gefs'}


@pytest.mark.parametrize('source', ['rap', 'rap-native'])
def test_rap_hourly_cycle_ladder(source):
    ladder = plan_ladder(source=source, cycle=START, start=START,
                         end=START + timedelta(hours=6), options=WofsOptions(members=4))
    assert [m.lag_hours for m in ladder.members] == [0, 1, 2, 3]
    assert ladder.counts() == {'source-member': 1, 'time-lagged': 3, 'perturbation': 0}
    for member in ladder.members:
        assert member.trajectory()['source'] == source


def test_owner_receives_parameterized_family_and_global_seed_context(monkeypatch):
    from gpuwm.ensemble import wofs_recipe as w
    from gpuwm.ensemble.recipes import SourceTrajectory
    seen = []
    # Test the owner contract without implementing the batch owner here.
    def members(**kwargs):
        seen.append(kwargs)
        return [None]  # a deterministic test owner; production owner owns noise grammar
    monkeypatch.setattr(w, 'selection_owner', lambda adapter: SimpleNamespace(ensemble_members=members))
    w.source_population('earth2-fcn3', START, members=18, base_seed=333, index_offset=18)
    assert seen == [{'provider': None, 'source': 'earth2-fcn3', 'members': 18,
                     'base_seed': 333, 'index_offset': 18}]


def _config(tmp_path, sources):
    from pathlib import Path
    import tomllib
    from gpuwm.toml_document import emit_experiment_toml
    repo = Path(__file__).resolve().parents[1]
    path = tmp_path / 'experiment.toml'
    raw = tomllib.loads((repo / 'configs/gfs_12km_quickstart.toml').read_text())
    raw['ensemble'] = {'sources': sources}
    path.write_text(emit_experiment_toml(raw))
    path.with_suffix('.namelist.wps').write_bytes((repo / 'configs/gfs_12km_quickstart.namelist.wps').read_bytes())
    return path


def test_config_sources_count_and_mixed_member_preparation(tmp_path):
    import tomllib
    from gpuwm.experiment import load_experiment
    from gpuwm.ensemble.request import EnsembleRequest
    from gpuwm.ensemble.recipe_door import plan_recipe, member_configuration
    path = _config(tmp_path, [{'source': 'gfs', 'members': 2}, {'source': 'rap', 'members': 2}])
    target, ladder = expand_config(path, WofsOptions(members=4))
    raw = tomllib.loads(target.read_text())
    request = EnsembleRequest.from_mapping(raw['ensemble'])
    assert request.sources == tuple(raw['ensemble']['sources'])
    recipe = plan_recipe(request, raw, load_experiment(target))
    for i, member in enumerate(recipe.members):
        config, wps, experiment = member_configuration(raw, target, target.with_suffix('.namelist.wps'),
            recipe, member, tmp_path / str(i))
        table = tomllib.loads(config.read_text())['fetch']
        assert table['source'] == ('gfs' if i < 2 else 'rap')
        assert table['cycle'] == member.trajectory.cycle.strftime('%Y-%m-%dT%H')
    assert len(ladder.members) == 4


def test_generated_family_binds_one_batch_before_any_member_preparation(tmp_path):
    import tomllib
    path = _config(tmp_path, [{'source': 'earth2-fcn3', 'members': 3},
                              {'source': 'gefs', 'members': 2}])
    target, ladder = expand_config(path, WofsOptions(members=5, base_seed=333))
    raw = tomllib.loads(target.read_text())
    fetches = raw['ensemble']['member_fetches']
    assert len(fetches) == 5
    generated = fetches[:3]
    assert len({row['earth2_batch_root'] for row in generated}) == 1
    assert {row['earth2_members'] for row in generated} == {'n0333,n0334,n0335'}
    assert [row['member'] for row in generated] == ['n0333', 'n0334', 'n0335']
    assert all(row['model'] == 'fcn3' for row in generated)
    assert all(row['out'].endswith('/' + row['member']) for row in generated)
    assert all('earth2_members' not in row for row in fetches[3:])


def test_secondary_family_coverage_refused_at_plan_time(tmp_path):
    path = _config(tmp_path, [{'source': 'gfs', 'members': 2}, {'source': 'icon-eu', 'members': 2}])
    with pytest.raises(LadderRefusal, match='family icon-eu:.*cell corners'):
        expand_config(path, WofsOptions(members=4))


def test_parameterized_route_keeps_model_drivability(monkeypatch):
    from gpuwm import source_drivability as routing, fetch_routes
    monkeypatch.setattr(fetch_routes, 'publishes_prep_handoff', lambda source: True)
    assert routing.drivability_for('earth2-fcn3')['chain'] == 'prepared:staged'
    # Base route is runnable but a table row without a profile remains refused.
    from gpuwm.earth2_run import model_table
    unavailable = next(key for key, row in model_table()["models"].items()
                       if not row.get("profile") and row.get("kind") != "diagnostic")
    assert routing.drivability_for("earth2-" + unavailable)["routes"] == []


def test_selection_owner_fetches_are_batched_per_family(tmp_path, monkeypatch):
    import tomllib
    from gpuwm.ensemble import wofs_recipe as w
    from gpuwm import earth2_members as owner, fetch_routes
    from gpuwm.ensemble.request import EnsembleRequest
    from gpuwm.ensemble.recipe_door import member_configuration, plan_recipe
    from gpuwm.experiment import load_experiment
    calls = []
    def members(*, source, members, base_seed, index_offset, provider=None):
        calls.append((source, provider, index_offset))
        return [owner.noise_id(base_seed + index_offset+i) for i in range(members)]
    monkeypatch.setattr(owner, 'ensemble_members', members)
    monkeypatch.setattr(fetch_routes, 'publishes_prep_handoff', lambda source: True)
    path = _config(tmp_path, [{'source': 'earth2-fcn3', 'members': 2},
                              {'source': 'earth2-sfno', 'members': 2}])
    target, ladder = expand_config(path, WofsOptions(members=4, base_seed=333))
    raw = tomllib.loads(target.read_text())
    hints = raw['ensemble']['member_fetches']
    assert [m.seed for m in ladder.members] == [member_seed(333, i) for i in range(4)]
    assert calls == [('earth2-fcn3', 'fcn3', 0), ('earth2-sfno', 'sfno', 2)]
    assert [h['source'] for h in hints] == ['earth2-fcn3']*2+['earth2-sfno']*2
    assert [h['model'] for h in hints] == ['fcn3']*2+['sfno']*2
    assert hints[0]['earth2_batch_root'] != hints[2]['earth2_batch_root']
    for begin in (0,2):
        assert hints[begin]['earth2_members'] == ','.join(h['member'] for h in hints[begin:begin+2])
        assert hints[begin]['earth2_members'] == hints[begin+1]['earth2_members']
    recipe = plan_recipe(EnsembleRequest.from_mapping(raw['ensemble']), raw, load_experiment(target))
    for member in recipe.members:
        cfg, _, _ = member_configuration(raw, target, target.with_suffix('.namelist.wps'),
                                         recipe, member, tmp_path / f'm{member.index}')
        assert tomllib.loads(cfg.read_text())['fetch'] == hints[member.index]

@pytest.mark.parametrize('sources', [
    [{'source': 'gfs', 'members': 2}, {'source': 'era5', 'members': 2}],
    [{'source': 'era5', 'members': 2}, {'source': 'gfs', 'members': 2}],
])
def test_config_driven_family_binds_own_case_inputs_in_either_position(tmp_path, monkeypatch, sources):
    import tomllib
    from pathlib import Path
    from gpuwm import era5_member
    from gpuwm.ensemble.request import EnsembleRequest
    from gpuwm.ensemble.recipe_door import plan_recipe, review_members
    from gpuwm.experiment import load_experiment
    monkeypatch.setattr(era5_member, 'ensemble_members', lambda **kwargs: ())
    path = _config(tmp_path, sources)
    # The first row owns geometry; its fetch and the shared valid-time window
    # must agree even when the first family is config-driven.
    raw = tomllib.loads(path.read_text())
    raw['fetch']['source'] = sources[0]['source']
    from gpuwm.toml_document import emit_experiment_toml
    path.write_text(emit_experiment_toml(raw))
    target, _ = expand_config(path, WofsOptions(members=4))
    raw = tomllib.loads(target.read_text())
    assert 'case_data' not in raw  # roster goes through independent preparation
    recipe = plan_recipe(EnsembleRequest.from_mapping(raw['ensemble']), raw, load_experiment(target))
    plans = review_members(raw, target, target.with_suffix('.namelist.wps'), recipe,
                           scratch=tmp_path / 'review')
    assert len(plans) == 4
    for plan in plans:
        doc = tomllib.loads(plan.config.read_text())
        if plan.member.trajectory.source == 'era5':
            assert plan.chain == 'prepared:case-data'
            assert doc['fetch']['era5_provider'] == 'arco'
            assert Path(doc['case_data']['forcing'][0]).name.endswith('.nc')
            assert Path(doc['case_data']['vtable']).is_file()
        else:
            assert 'case_data' not in doc


def test_direct_snapshot_service_accepts_native_netcdf(tmp_path, monkeypatch):
    from gpuwm.ingest import grib
    path = tmp_path / 'analysis.nc'
    path.write_bytes(b'fixture')
    snapshots = (object(),)
    calls = []
    def decode(paths, vtable, **kwargs):
        calls.append((paths, kwargs))
        return SimpleNamespace(snapshots=snapshots)
    monkeypatch.setattr(grib, 'cached_era5_forcing', decode)
    assert grib.cached_era5_snapshots(path, tmp_path / 'vtable') is snapshots
    assert calls == [((path,), {'container': 'netcdf'})]


def test_config_family_preparer_consumes_its_own_download_directory(tmp_path, monkeypatch):
    import tomllib
    from pathlib import Path
    from gpuwm import era5_preparation as owner, runplan, netcdf_bridge, era5_direct
    from gpuwm.toml_document import emit_experiment_toml
    bridge = tmp_path / 'rw_netcdf'
    bridge.write_bytes(b'native-decoder-fixture')
    monkeypatch.setattr(netcdf_bridge, 'find_netcdf_bin', lambda: bridge)
    calls = []
    def fetch(argv, run_dir, **kwargs):
        out = Path(argv[argv.index('--out')+1])
        out.mkdir(parents=True)
        (out / 'era5-combined.nc').write_bytes(b'native-netcdf-fixture')
        return {'out': str(out)}
    monkeypatch.setattr(runplan, '_run_fetch', fetch)
    def prep(argv):
        config = Path(argv[argv.index('--experiment-config')+1])
        raw = tomllib.loads(config.read_text())
        manifest = Path(argv[argv.index('--source-sha256s')+1])
        roles = {key: Path(argv[argv.index(flag)+1]) for key, flag in {
            'grib':'--grib', 'vtable':'--vtable', 'bridge':'--bridge',
            'wps_namelist':'--wps-namelist', 'experiment_config':'--experiment-config'}.items()}
        era5_direct._verify_input_manifest(manifest,
            argv[argv.index('--source-sha256s-sha256')+1], roles)
        calls.append((roles['grib'], raw['case_data']))
    monkeypatch.setattr(runplan, '_run_prep', prep)
    monkeypatch.setattr(runplan, '_prepare_stage', lambda output, *, run, **kwargs: run())
    monkeypatch.setattr(runplan, '_prepared_chain_result', lambda root, config, wps: {'prepared_root': str(root)})
    observer = SimpleNamespace(enter_stage=lambda *a, **k: None, finish_stage=lambda **k: None)
    for index in range(2):
        directory = tmp_path / f'member-{index}'
        directory.mkdir()
        target = directory / 'experiment.toml'
        raw = {'fetch': {'source':'era5', 'cycle':'2022-06-13T18', 'hours':6,
                         'out':str(directory/'original'), 'area':'30,-100,40,-90'}}
        owner.configure_member(raw, target=target, original=target)
        target.write_text(emit_experiment_toml(raw))
        target.with_suffix('.namelist.wps').write_text('member namelist')
        plan = SimpleNamespace(config_intent={}, run_options={
            'data_dir':str(directory/'managed'), 'geog_root':str(tmp_path/'geog')})
        owner.prepare(plan, config_path=target, exp=None, observer=observer,
                      run_dir=directory/'prepare', prepare_only=True)
    assert calls[0][0] != calls[1][0]
    for forcing, data in calls:
        assert Path(data['forcing'][0]) == forcing
        assert forcing.parent.name == 'managed'
        assert data['geog_root'] == str(tmp_path/'geog')


def test_nonfirst_deterministic_owner_retains_family_provider(tmp_path, monkeypatch):
    import tomllib
    from gpuwm import era5_member
    monkeypatch.setattr(era5_member, 'ensemble_members', lambda **kwargs: ())
    path = _config(tmp_path, [{'source':'gfs','members':2}, {'source':'era5','members':2}])
    target, ladder = expand_config(path, WofsOptions(members=4,
                                    family_providers={'era5':'cds'}))
    raw = tomllib.loads(target.read_text())
    hints = raw['ensemble']['member_fetches']
    assert len(hints) == 4
    assert 'era5_provider' not in hints[0]
    assert [h['era5_provider'] for h in hints[2:]] == ['cds', 'cds']
    assert all(m.source_member is None for m in ladder.members[2:])


def test_each_family_reads_its_own_declared_physics_arm_table(monkeypatch):
    from gpuwm import source_adapters as registry
    rows = {token: replace(get_source_adapter(token), ensemble_physics_arms=(
                ('base', {}), (token + '-arm', {'mp_physics': mp})))
            for token, mp in [('gfs', 8), ('rap', 10)]}
    monkeypatch.setattr(registry, '_ALIASES', {**registry._ALIASES, **rows})
    ladder = plan_families(sources='gfs:2,rap:2', cycle=START, start=START,
        end=START+timedelta(hours=6), options=WofsOptions(members=4, physics='all'))
    assert [m.physics for m in ladder.members] == [{}, {'mp_physics':8}, {}, {'mp_physics':10}]
    assert [m.seed for m in ladder.members] == [member_seed(0, i) for i in range(4)]
