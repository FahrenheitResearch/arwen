"""CPU metadata and orchestration checks; no synthetic forecast evidence."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gpuwm.ensemble.member_weights import roster_weights, weight_receipt, validate_weights
from gpuwm.ensemble.batch_products import FieldProducts, product_memory_plan, probability_matched_mean
from gpuwm.ensemble import cyclone_products


def test_frame_weights_and_weighted_pmm():
    assert roster_weights((7, 3), [{'member_id': 3, 'weight': .75}, {'member_id': 7, 'weight': .25}]) == (.25, .75)
    row = weight_receipt((7, 3), (.25, .75), [3])
    assert row['member_weights'] == [dict(member_id=3, weight=.75, normalized_weight=1.)]
    for weights in ([1], [0, 1], [True, 1], [float('nan'), 1]):
        with pytest.raises(ValueError):
            validate_weights(weights, 2)
    values = np.array([[[8, 0]], [[4, 2]]], dtype=np.float32)
    mean = np.array([[5, 1.5]], dtype=np.float32)
    np.testing.assert_array_equal(probability_matched_mean(values, mean, np, [1, 3]), [[8, 2]])
    requests = (FieldProducts('wind10', 'm s-1', (10,)),)
    base = product_memory_plan(requests, {'wind10': (2, 3)}, members=2)
    equal = product_memory_plan(requests, {'wind10': (2, 3)}, members=2, member_weights=[.5, .5])
    weighted = product_memory_plan(requests, {'wind10': (2, 3)}, members=2, member_weights=[.25, .75])
    assert equal == base
    assert weighted.required_bytes(2) == base.required_bytes(2)+512


def test_selection_activates_cyclone_products_and_saved_best_track(tmp_path):
    from gpuwm.tc_tracks import best_track
    tracks = tmp_path/'tracks.json'
    best = dict(storm_id='fixture', fixes=[dict(valid='2024-01-01T00:00:00Z', lat=10, lon=20)])
    tracks.write_text(json.dumps(dict(schema='gpuwm.tc-tracks.v1', best_track=best)))
    roster = [dict(source='gefs', cycle='2024-01-01T00:00:00Z', stratum='S1', tracks=str(tracks))]
    request = cyclone_products.roster_product_request(roster)
    assert request['best_track'] == str(tracks)
    assert best_track(tracks)['fixes'] == best['fixes']
    with pytest.raises(ValueError, match='another storm'):
        best_track(tracks, 'different')
    from gpuwm.ensemble.request import EnsembleRequest
    ensemble = EnsembleRequest(members=1, recipe='member-roster', trajectories=tuple(roster), cyclone_products=request)
    assert ensemble.keep_member_files
    assert EnsembleRequest.from_mapping(ensemble.receipt()).receipt() == ensemble.receipt()


def test_production_admission_prices_weighted_replay():
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    from gpuwm.ensemble.production_memory import ordinary_output_memory_components
    collector = SimpleNamespace(members=2, tile_rows=32, events={},
        requests=(FieldProducts('wind10', 'm s-1', (10,)),))
    collector.memory_plan = lambda shape, **kw: HeadlineDiagnosticCollector.memory_plan(collector, shape, **kw)
    experiment = SimpleNamespace(domains=(SimpleNamespace(run=SimpleNamespace(nx=3, ny=2, nz=4)),))
    equal = ordinary_output_memory_components(experiment, collector)[-1].inventory(1)['required_bytes']
    collector.member_weights = (.25, .75)
    weighted = ordinary_output_memory_components(experiment, collector)[-1].inventory(1)['required_bytes']
    assert weighted == equal + 512


def test_roster_completion_renders_every_grid_tracks_weights_and_retires_raw(tmp_path, monkeypatch):
    from gpuwm.ensemble.history_ledger import EnsembleHistoryLedger
    from gpuwm import rustwx, tc_tracks
    ledger = EnsembleHistoryLedger(tmp_path, member_order=(0, 1))
    for member in (0, 1):
        for grid in (1, 2):
            path = tmp_path/f'members/member-{member:04d}/wrfout_d{grid:02d}_fixture'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes([member, grid]))
            relative = str(path.relative_to(tmp_path))
            ledger.records[relative] = dict(path=relative, bytes=2, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                member_id=member, grid_id=grid, episode=0, domain=f'd{grid:02d}', valid_time='2024-01-01_00:00:00', retirement_state='registered')
    ledger._write()
    maps = []
    def render(renderer, wrfouts, **kwargs):
        assert len(wrfouts) == 1
        assert kwargs['products'] == 'refl,wind10,mslp'
        kwargs['store_root'].mkdir(parents=True)
        paths = []
        for field in ('refl', 'wind10', 'mslp'):
            path = kwargs['out_dir']/field/'frame.png'
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'fixture map')
            paths.append(path)
        maps.extend(paths)
        return paths, [], []
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[3] == 'ml-export':
            export = Path(command[command.index('--out')+1])
            (export/'fixture.zarr').mkdir(parents=True)
            (export/'fixture.zarr/.zgroup').write_text('{}')
        else:
            assert command[3] == 'tc-tracks'
            manifest = json.loads(Path(command[command.index('--manifest')+1]).read_text())
            assert [row['weight'] for row in manifest['members']] == [.25, .75]
            assert all(row['family'] == 'arwen' for row in manifest['members'])
            assert [row['ancestry']['stratum'] for row in manifest['members']] == ['S1', 'S2']
            out = Path(command[command.index('--out')+1]); out.mkdir()
            (out/'tracks.json').write_text('{}'); (out/'receipt.json').write_text('{}')
            plume = out/'d01/tc-track-plume/2024-01-01/track-plume.png'
            plume.parent.mkdir(parents=True); plume.write_bytes(b'fixture plume')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(rustwx, 'find_renderer', lambda: Path('rw_wrfbatch'))
    monkeypatch.setattr(rustwx, 'run_renderer_series', render)
    monkeypatch.setattr(cyclone_products.subprocess, 'run', run)
    roster = [dict(source='gefs', cycle='2024-01-01T00:00:00Z', weight=w, stratum=f'S{m+1}', track_member_id=f'track-{m}')
              for m, w in enumerate((.25, .75))]
    request = dict(backend='tempestextremes', workers=2, member_maps=['refl','wind10','mslp'], tracking_domain='d01')
    result = cyclone_products.render_roster(tmp_path, roster, request, member_order=(0,1), history_ledger=ledger)
    assert len(maps) == 12 and len(result['member_products']) == 4
    assert len(result['retired_histories']) == 4
    assert all(row['retirement_state'] == 'retired' for row in ledger.validate_retirements())
    assert not list((tmp_path/'cyclone').glob('.roster-products-*'))
    assert all(path.is_file() for path in maps)
    assert len(calls) == 3


def test_hydro_wrapper_preserves_atmosphere_product_dictionary(tmp_path):
    import threading
    from gpuwm.ensemble.nextgen_capture import NextGenCollector
    wrapper = NextGenCollector.__new__(NextGenCollector)
    wrapper.original = SimpleNamespace(root=tmp_path, finish_run=lambda: {'domain_manifests': ['fixture']})
    wrapper.nextgen_points = SimpleNamespace(require_complete=lambda: None,
        finish=lambda root: ['reach.nc'], frames={})
    wrapper.channel_depth_archive = SimpleNamespace(finish=lambda frames: ['depth.nc'])
    wrapper._capture_lock = threading.RLock()
    result = wrapper.finish_run()
    assert result == {'domain_manifests': ['fixture'], 'nextgen': ['reach.nc', 'depth.nc']}


def test_reach_weights_survive_product_and_frame_receipts():
    from gpuwm.ensemble.nextgen_points import ReachAccumulator, _frame_contract
    points = ReachAccumulator((4, 7), member_weights=(.25, .75))
    assert points.receipt()['member_weights'][1]['normalized_weight'] == .75
    frame = _frame_contract(Path('fixture.nc'), 'fixture', (4, 7), 3, 1, (.25, .75))
    assert frame['member_weights'][0]['normalized_weight'] == .25
    assert frame['denominator'] == 'sum of present roster masses'
    assert points.weighted
    assert not ReachAccumulator((4, 7), member_weights=(.5, .5)).weighted
