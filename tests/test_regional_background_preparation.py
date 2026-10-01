"""The actual input reader and local snapshot retain regional identities."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest

from gpuwm.local_da import Card, Request, configuration, derive_rung
from gpuwm.regional_preparation import read_prepared, review_background


@pytest.mark.parametrize('source', ['gfs', 'era5', '20crv3'])
def test_portable_reader_preserves_native_source_cache_and_member(tmp_path, monkeypatch, source):
    from test_prepared_single_domain_forecast import _prepared_fixture
    from gpuwm import prepared_single_domain_forecast as reader
    from gpuwm.experiment import load_experiment
    fixture = _prepared_fixture(tmp_path, source, physics_profile=None)
    # The shared fixture carries a minimal source/cache payload. Geography
    # interpolation is outside this identity and dispatch control.
    monkeypatch.setattr(reader, 'validate_native_lambert_contract',
                        lambda exp, path, *, source_name: SimpleNamespace(source=source))
    monkeypatch.setattr(reader, 'verify_native_static_receipt', lambda *args: {'status': 'PASS'})
    monkeypatch.setattr(reader, 'load_native_static_cache',
                        lambda path, grid, ny, nx: {'STATIC': np.ones((ny, nx))})
    exp = load_experiment(fixture.experiment)
    inputs = read_prepared(fixture.prepared, fixture.experiment, fixture.wps,
                          source=source, run_seconds=fixture.run_seconds,
                          history_interval_seconds=exp.root.history_interval_s)
    assert inputs.source == source
    assert inputs.prepared_cache_path.is_dir()
    assert inputs.forcing_hours[0] == 0
    assert inputs.source_member == ('072' if source == '20crv3' else None)
    before = fixture.experiment.read_bytes()
    fixture.experiment.write_bytes(before + b'\n# changed authority\n')
    with pytest.raises(ValueError, match='differ|SHA|sha|identity'):
        read_prepared(fixture.prepared, fixture.experiment, fixture.wps,
                      source=source, run_seconds=fixture.run_seconds,
                      history_interval_seconds=exp.root.history_interval_s)


def test_local_member_is_bound_without_an_acquisition_member_axis(tmp_path):
    from test_audit_area1_acquisition import _local_member_archive
    root = tmp_path / 'inputs'
    _local_member_archive(root, member='009')
    request = Request(epoch='2026-07-29T00:00:00Z', point=(40., -100.), card=Card(24.),
                      source='20crv3', source_root=str(root), source_member='009')
    text, _, _, background = configuration(request, derive_rung(request, 1))
    assert background['selection']['member'] == '009'
    assert background['selection']['cycle'] is None
    assert background['selection']['time_axis'] == 'analysis_times'
    assert background['inputs']['snapshot']['manifest']['member'] == '009'
    assert 'member =' not in text and 'out =' not in text
    with pytest.raises(ValueError, match='member'):
        configuration(replace(request, source_member='008'), derive_rung(request, 1))


def test_local_mapped_review_retains_order_after_external_list_changes(tmp_path, monkeypatch):
    from test_acquisition_review_repairs import _snapshot
    from gpuwm import runplan, stage_cli
    snapshot, listing, paths = _snapshot(tmp_path / 'inputs')
    request = Request(epoch='2026-07-29T00:00:00Z', point=(40., -100.), card=Card(24.),
                      source=snapshot['source'], source_root=snapshot['source_root'])
    text, wps, exp, background = configuration(request, derive_rung(request, 1))
    reviewed = background['inputs']['snapshot']
    listing.write_text(''.join(f'{path}\n' for path in reversed(paths)))
    config = tmp_path / 'experiment.toml'
    config.write_text(text)
    config.with_name('experiment.namelist.wps').write_text(wps)
    observed = []
    def prepare(argv):
        selected = Path(argv[argv.index('--input-list') + 1])
        observed.extend(selected.read_text().splitlines())
        root = Path(argv[argv.index('--output-root') + 1])
        root.mkdir(parents=True, exist_ok=True)
        (root / 'proof.json').write_text('{}')
    monkeypatch.setattr(runplan, '_run_fetch', lambda *a, **k: pytest.fail('local inputs were downloaded'))
    monkeypatch.setattr(runplan, '_run_prep', prepare)
    monkeypatch.setattr(stage_cli, 'resolve_bundle', lambda root: {'document': root / 'proof.json'})
    observer = SimpleNamespace(enter_stage=lambda *a, **k: None, finish_stage=lambda *a, **k: None)
    plan = SimpleNamespace(config_intent={}, run_options={'geog_root': str(tmp_path / 'geog')})
    result = runplan._staged_chain(plan, config_path=config, exp=exp, observer=observer,
        run_dir=tmp_path / 'run', prepare_only=True, reviewed_inputs=reviewed)
    assert observed == [str(path) for path in paths]
    assert result['forecast_started'] is False


@pytest.mark.parametrize('committed_state', [False, True])
def test_stale_owned_preparation_preserves_old_files_and_rebuilds_before_members(tmp_path, monkeypatch, committed_state):
    from test_prepared_single_domain_forecast import _prepared_fixture
    from gpuwm import prepared_single_domain_forecast as reader
    from gpuwm import regional_preparation as preparation
    from gpuwm.background_contract import plan as select
    from gpuwm.experiment import load_experiment
    from datetime import datetime, timezone
    fixture = _prepared_fixture(tmp_path, 'gfs', physics_profile=None)
    monkeypatch.setattr(reader, 'validate_native_lambert_contract',
                        lambda exp, path, *, source_name: SimpleNamespace(source=source_name))
    monkeypatch.setattr(reader, 'verify_native_static_receipt', lambda *args: {'status': 'PASS'})
    monkeypatch.setattr(reader, 'load_native_static_cache',
                        lambda path, grid, ny, nx: {'STATIC': np.ones((ny, nx))})
    exp = load_experiment(fixture.experiment)
    initial = exp.start_time.replace(tzinfo=timezone.utc)
    selected = select('gfs', init=initial, now=datetime(2026, 9, 12, tzinfo=timezone.utc),
                      run_seconds=fixture.run_seconds,
                      cadence_hours=json.loads(fixture.proof.read_text())['boundary_interval_seconds'] // 3600)
    background = dict(schema=preparation.SCHEMA, selection=selected.record(),
                      inputs={'kind': 'automatic', 'files': []}, fetch_hints=selected.fetch_hints(), supplements=[])
    plan = dict(background=background, review_sha256='c' * 64,
                selected={'cadence_seconds': exp.root.history_interval_s}, request={})
    (tmp_path / 'experiment.toml').write_bytes(fixture.experiment.read_bytes())
    (tmp_path / 'experiment.namelist.wps').write_bytes(fixture.wps.read_bytes())
    monkeypatch.setattr(preparation, 'validate_saved_background', lambda *args: selected)
    # A real v3 physics receipt from another registry reaches the ordinary
    # reader's compatibility refusal. No input-reader arithmetic is replaced.
    # The receipt names a registry document whose physics this build cannot
    # establish: since A153 a document digest alone (a citation edit) is
    # not a physics difference, so the receipt drops the physics parts that
    # would resolve it, as a receipt from an unknown registry carries none.
    proof = json.loads(fixture.proof.read_text())
    proof['physics'].pop('registry_physics')
    proof['physics']['registry_sha256'] = '0' * 64
    fixture.proof.write_text(json.dumps(proof))
    stale_bytes = fixture.proof.read_bytes()
    result = dict(prepared_root=str(fixture.prepared), experiment_config=str(fixture.experiment),
                  wps_namelist=str(fixture.wps))
    receipt = dict(review_sha256=plan['review_sha256'], background=background, result=result,
                   files=preparation._files([fixture.proof, fixture.source_manifest, fixture.experiment, fixture.wps]))
    record = tmp_path / 'background-preparation.json'
    record.write_text(json.dumps(receipt))
    original_receipt = record.read_bytes()
    calls = []
    def prepare(*args, run_dir, **kwargs):
        calls.append(run_dir)
        inner = run_dir / 'native'
        inner.mkdir()
        fresh = _prepared_fixture(inner, 'gfs', physics_profile=None)
        return dict(prepared_root=str(fresh.prepared), experiment_config=str(fresh.experiment),
                    wps_namelist=str(fresh.wps))
    monkeypatch.setattr(preparation, 'preparation_chains', lambda: {'prepared:go': prepare})
    if committed_state:
        states = tmp_path / 'cycles' / 'cycle_000'
        states.mkdir(parents=True)
        np.savez(states / 'analysis.npz', state=np.array([1.]))
        with pytest.raises(ValueError, match='committed regional state'):
            preparation.prepare_background(plan, tmp_path, exp, geog=tmp_path / 'geog')
        assert not calls
        assert record.read_bytes() == original_receipt
    else:
        inputs = preparation.prepare_background(plan, tmp_path, exp, geog=tmp_path / 'geog')
        assert len(calls) == 1
        assert inputs.prepared_root != fixture.prepared
        saved = json.loads(record.read_text())
        assert Path(saved['recovery']['previous_receipt']).read_bytes() == original_receipt
        assert saved['recovery']['previous_files_preserved'] is True
        preparation.prepare_background(plan, tmp_path, exp, geog=tmp_path / 'geog')
        assert len(calls) == 1, 'a current cache is restored without preparation'
    assert fixture.proof.read_bytes() == stale_bytes


@pytest.mark.parametrize('source,member', [('gefs', 'c00'), ('gefs', 'p30'), ('aigefs', 'mem000'), ('aigefs', 'mem030')])
def test_prepared_table_member_uses_the_bound_primary_inputs(tmp_path, monkeypatch, source, member):
    from test_forcing_member import fixture
    from gpuwm import forcing_member
    from gpuwm.regional_preparation import _prepared_member
    from gpuwm.source_adapters import get_source_adapter
    hints, handoff, path, row = fixture(tmp_path, monkeypatch, source=source, member=member)
    receipt = forcing_member.verify_handoff(hints, handoff, out=tmp_path)
    root = tmp_path / 'prepared'
    root.mkdir()
    (root / 'forcing-member.json').write_text(json.dumps(receipt))
    inputs = SimpleNamespace(source_member=None, prepared_root=root,
                             source_manifest={'primary_files': [{'sha256': receipt['files'][0]['sha256']}]})
    request = SimpleNamespace(source_member=member)
    assert _prepared_member(request, inputs, get_source_adapter(source)) == member
    # The existing fixture injects only decoded record inventory. The real
    # grammar and its complete identity-field verifier remain active here.
    row['member'] = '999'
    with pytest.raises(ValueError, match='member'):
        _prepared_member(request, inputs, get_source_adapter(source))
    row['member'] = str(forcing_member.member_contract(source, member)[1].member(member).ordinal)
    inputs.source_manifest['primary_files'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='primary input order'):
        _prepared_member(request, inputs, get_source_adapter(source))


def test_older_review_keeps_its_fetch_recipe_and_retains_the_recovered_pointer(tmp_path, monkeypatch):
    from test_local_da_plan import request, availability, price
    from gpuwm.local_da import build_plan, publish
    from gpuwm import go_cli, regional_preparation as preparation
    from gpuwm.experiment import load_experiment
    plan = build_plan(request(), availability=availability, price=price)
    root = tmp_path / 'case'
    publish(plan, root)
    exp = load_experiment(root / 'experiment.toml')
    go = go_cli.plan_from_config(root / 'experiment.toml', outdir=root / 'background', run_stamp=False)
    go['prepared'].mkdir(parents=True)
    original = go['prepared'] / 'proof.json'
    original.write_bytes(b'stale receipt control')
    builds, reads = [], []
    def read(prepared, config, namelist, **kwargs):
        reads.append(Path(prepared))
        if (Path(prepared) / 'proof.json').read_bytes() != b'current receipt control':
            raise ValueError('controlled stale reader contract')
        assert Path(config).read_bytes() == (root / 'experiment.toml').read_bytes()
        assert Path(namelist).read_bytes() == (root / 'experiment.namelist.wps').read_bytes()
        return SimpleNamespace(prepared_root=Path(prepared))
    def prepare(options, *, run_dir, **kwargs):
        builds.append(dict(directory=run_dir, data=options.run_options['data_dir']))
        prepared = run_dir / 'prepared'
        authority = run_dir / 'authority'
        prepared.mkdir(); authority.mkdir()
        (prepared / 'proof.json').write_bytes(b'current receipt control')
        config, namelist = authority / 'experiment.toml', authority / 'namelist.wps'
        config.write_bytes((root / 'experiment.toml').read_bytes())
        namelist.write_bytes((root / 'experiment.namelist.wps').read_bytes())
        return dict(prepared_root=str(prepared), experiment_config=str(config), wps_namelist=str(namelist),
                    bundle={'unused': Path('not serialized')}, forecast_started=False)
    # This test observes recovery and the real configuration/command owner.
    # The portable reader's real physics rejection is exercised above.
    monkeypatch.setattr(preparation, 'read_prepared', read)
    monkeypatch.setattr(preparation, '_prepare_native', prepare)
    kwargs = dict(geog=root / 'geog', cadence=exp.root.history_interval_s, review_sha256=plan['review_sha256'])
    first, resolved = preparation.prepare_legacy_background(go, root, exp, **kwargs)
    second, repeated = preparation.prepare_legacy_background(go, root, exp, **kwargs)
    assert first.prepared_root == second.prepared_root == resolved['prepared'] == repeated['prepared']
    assert len(builds) == 1
    assert builds[0]['data'] == str(go['data'])
    assert original.read_bytes() == b'stale receipt control'
    state = root / 'cycles' / 'cycle_000'
    state.mkdir(parents=True)
    np.savez(state / 'analysis.npz', state=np.array([1.]))
    (resolved['prepared'] / 'proof.json').write_bytes(b'stale again')
    with pytest.raises(ValueError, match='committed regional state'):
        preparation.prepare_legacy_background(go, root, exp, **kwargs)
    assert len(builds) == 1


@pytest.mark.parametrize('member', [0, 9])
def test_supplied_ensemble_product_reaches_its_native_member_owner(tmp_path, monkeypatch, member):
    from datetime import datetime
    from gpuwm import era5_member
    from gpuwm.regional_preparation import _prepared_member, _files
    from gpuwm.source_adapters import get_source_adapter
    path = tmp_path / 'source.grib'
    path.write_bytes(b'role binding control; native inventory is substituted')
    inputs = SimpleNamespace(source_member=None, source_manifest={'files': {'grib': _files([path])[0]}},
                             boundary_interval_seconds=10800, experiment=SimpleNamespace(start_time=datetime(2026, 7, 29)))
    request = SimpleNamespace(source_product='ensemble_members', source_member=member,
                              forcing_cadence_hours=3, source_provider='cds', source_inputs={'grib': str(path)})
    calls = []
    monkeypatch.setattr(era5_member, 'check_member', lambda source, number: calls.append((source, number)))
    assert _prepared_member(request, inputs, get_source_adapter('era5')) == member
    assert calls == [(path, member)]
    request.source_product = None
    with pytest.raises(ValueError, match='reanalysis has no EDA member'):
        _prepared_member(request, inputs, get_source_adapter('era5'))
    request.source_product = 'ensemble_members'
    path.write_bytes(b'changed original bytes')
    with pytest.raises(ValueError, match='differ from the prepared source manifest'):
        _prepared_member(request, inputs, get_source_adapter('era5'))
    assert len(calls) == 1


def test_prepared_inputs_cannot_silently_change_reviewed_outputs():
    from dataclasses import replace
    from gpuwm.io.history_selection import HistorySelection
    from gpuwm.regional_preparation import _experiment_identity
    request = Request(epoch='2026-07-29T00:00:00Z', point=(40., -100.), card=Card(24.))
    _, _, exp, _ = configuration(request, derive_rung(request, 1))
    changed = replace(exp, output=HistorySelection(history_drop=('T2',)))
    assert _experiment_identity(exp) != _experiment_identity(changed)
