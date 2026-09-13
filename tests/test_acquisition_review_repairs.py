"""Reviewed input identity and published forcing lead subsets."""
from datetime import datetime
import json
from pathlib import Path

import pytest

from gpuwm import fetch, fetch_routes, local_preparation


def _snapshot(root):
    root.mkdir()
    paths = [root / name for name in ('later.nc', 'earlier.nc')]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]) * 9)
    donor = root / 'invariant.nc'
    donor.write_bytes(b'invariant')
    listing = root / 'input-list.txt'
    listing.write_text(''.join(f'{path}\n' for path in paths))
    role = 'twentycrv3_netcdf_recovered_invariant'
    tokens = ['--source', '20crv3-cf', '--input-list', str(listing),
              '--supplement', f'{role}={donor}',
              '--author-input-manifest', str(root / 'inputs.json')]
    fetch_routes.write_prep_arguments(root, source='20crv3-cf', prep_source='20crv3-cf',
                                     cycle=datetime(2026, 7, 29), tokens=tokens)
    snapshot = local_preparation.inspect_local_inputs('20crv3-cf', root,
        cycle=datetime(2026, 7, 29), hours=6, cadence=3)
    return snapshot, listing, paths


def _published_list(snapshot, destination):
    path = local_preparation.publish_local_handoff(snapshot, destination)
    tokens = json.loads(path.read_text())['argv']
    return Path(tokens[tokens.index('--input-list') + 1])


@pytest.mark.parametrize('change', ['reverse', 'replace', 'remove'])
def test_external_list_cannot_change_the_reviewed_input_order(tmp_path, change):
    snapshot, listing, paths = _snapshot(tmp_path / 'inputs')
    if change == 'remove':
        listing.unlink()
    elif change == 'reverse':
        listing.write_text(''.join(f'{p}\n' for p in reversed(paths)))
    else:
        alternate = listing.parent / 'alternate.nc'
        alternate.write_bytes(b'alternate')
        listing.write_text(str(alternate) + '\n')
    published = _published_list(snapshot, tmp_path / 'out')
    assert published != listing
    assert published.read_text().splitlines() == list(map(str, paths))
    local_preparation.verify_local_snapshot(snapshot, input_list=published)


def test_owned_list_reuse_and_corruption_recovery_preserve_prior_files(tmp_path):
    snapshot, listing, paths = _snapshot(tmp_path / 'inputs')
    first = _published_list(snapshot, tmp_path / 'out')
    assert _published_list(snapshot, tmp_path / 'out') == first
    first.write_bytes(b'damaged list')
    recovered = _published_list(snapshot, tmp_path / 'out')
    assert recovered != first
    assert first.read_bytes() == b'damaged list'
    assert recovered.read_text().splitlines() == list(map(str, paths))
    assert listing.read_text().splitlines() == list(map(str, paths))


@pytest.mark.parametrize('changed', ['input', 'owned_list'])
def test_preparation_verification_detects_replacement(tmp_path, changed):
    snapshot, _, paths = _snapshot(tmp_path / 'inputs')
    published = _published_list(snapshot, tmp_path / 'out')
    target = paths[0] if changed == 'input' else published
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        local_preparation.verify_local_snapshot(snapshot, input_list=published)


def test_old_snapshot_without_an_order_requires_new_review(tmp_path):
    snapshot, _, _ = _snapshot(tmp_path / 'inputs')
    snapshot.pop('ordered_inputs', None)
    snapshot.pop('sha256')
    snapshot['sha256'] = local_preparation._identity(snapshot)
    with pytest.raises(ValueError, match='ordered inputs'):
        _published_list(snapshot, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('start,hours,cadence', [
    (0,12,2), (0,12,4), (0,24,6), (0,24,12), (0,48,24),
    (1,12,2), (5,8,4), (120,24,6), (123,24,12), (119,4,4),
    (0,384,6), (123,0,1), (117,6,6),
])
def test_published_subsets_keep_exact_request_through_both_python_owners(tmp_path, start, hours, cadence):
    expected = tuple(range(start, start+hours+1, cadence))
    assert fetch.gfs_forecast_hours(hours, cadence, start) == expected
    fetch.validate_fetch_hints({'source':'gfs','cycle':'2026-07-29T00',
        'hours':hours,'cadence':cadence,'forecast_start_hour':start}, source='request')
    if hours:
        from gpuwm.gfs_direct import _read_series
        lines = []
        for lead in expected:
            path = tmp_path / f'f{lead:03d}'
            path.write_bytes(b'parser fixture')
            lines.append(f'{lead}\t{path}\t{81 if lead == 0 else 96}\n')
        series = tmp_path / 'series.tsv'
        series.write_text(''.join(lines))
        assert tuple(lead for lead,_ in _read_series(series)) == expected


@pytest.mark.parametrize('start,hours,cadence', [
    (120,6,2), (121,0,3), (118,6,2), (0,7,2), (384,6,6),
    (True,6,3), (0,True,1), (0,6,True), (0,6,3.0), (0,6,'3'),
    (0,6,0), (-1,6,1), (0,-1,1),
])
def test_unpublished_or_malformed_windows_refuse_before_io(start, hours, cadence):
    with pytest.raises(ValueError):
        fetch.gfs_forecast_hours(hours, cadence, start)


@pytest.mark.parametrize('surface', ['registry', 'public_page'])
def test_legacy_preset_does_not_restore_retired_modern_coupling_refusal(surface):
    from test_authority_agreement import test_mp9_cloud_optics_gate_covers_both_selector_spellings
    from gpuwm.physics_registry import physics_registry
    test_mp9_cloud_optics_gate_covers_both_selector_spellings('rte-rrtmgp','milbrandt2mom-mp9','rte-rrtmgp',False)
    template=physics_registry()['templates']['milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1']
    assert template['parameters']['ra_rrtmg_variant']=='rrtmg_legacy'
    if surface == 'registry':
        assert not any('conditional refusal blocks the RTE+RRTMGP' in w for w in template['warnings'])
    else:
        page=(Path(__file__).resolve().parents[1]/'docs/public/PHYSICS.md').read_text()
        row=next(line for line in page.splitlines() if line.startswith('| Milbrandt-Yau 2-moment |'))
        assert 'refuses the RTE+RRTMGP pairing' not in row


def test_current_soil_document_does_not_claim_an_estimate_based_refusal():
    text=(Path(__file__).resolve().parents[1]/'docs/wrf_ruc_runtime_admission.md').read_text()
    assert 'refused (`tests/test_ruc_admission.py`) because nothing has measured it' not in text
@pytest.mark.parametrize('changed', ['input', 'owned_list'])
def test_local_chain_stops_before_completion_binding_when_inputs_change(tmp_path, monkeypatch, changed):
    from types import SimpleNamespace
    from gpuwm import runplan, stage_cli
    snapshot, _, paths = _snapshot(tmp_path / 'inputs')
    config = tmp_path / 'case.toml'
    config.write_text('[fetch]\nsource="20crv3-cf"\ncycle="2026-07-29T00"\nhours=6\ncadence=3\nsource_root='+json.dumps(str(tmp_path / 'inputs'))+'\n')
    config.with_suffix('.namelist.wps').write_text('&share /\n')
    run_dir = tmp_path / 'run'
    previous = run_dir / 'chain' / 'prep' / 'prior-output'
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b'prior completed state')
    completed = []
    def preparation(arguments):
        target = paths[0] if changed == 'input' else Path(arguments[arguments.index('--input-list')+1])
        target.write_bytes(b'changed')
        return {'status':'PASS'}
    def prepare_stage(root, **kwargs):
        value = kwargs['run']()
        completed.append('binding')
        return value
    monkeypatch.setattr(runplan, '_run_prep', preparation)
    monkeypatch.setattr(runplan, '_prepare_stage', prepare_stage)
    monkeypatch.setattr(stage_cli, 'resolve_bundle', lambda *a:pytest.fail('changed inputs reached forecast binding'))
    plan = SimpleNamespace(run_options={'geog_root':str(tmp_path)}, config_intent=None)
    observer = SimpleNamespace(enter_stage=lambda *a,**k:None, finish_stage=lambda *a,**k:None)
    with pytest.raises(ValueError, match='changed'):
        runplan._staged_chain(plan, config_path=config, exp=None, observer=observer, run_dir=run_dir)
    assert completed == []
    assert previous.read_bytes() == b'prior completed state'
