"""Failures from the hurricane roster door, not forecast-science evidence."""
from datetime import timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import tc_accumulator, tc_tracks
from gpuwm.ensemble.cyclone_products import _track_member


@pytest.mark.parametrize('entry', ['ensemble', 'go-flags', 'go-config', 'go-sources'])
def test_roster_products_reach_the_ensemble_door(tmp_path, entry):
    config = tmp_path / 'experiment.toml'
    config.write_text('[ensemble]\nmembers = 6\nrecipe = "wofs"\n' if entry == 'go-config' else '')
    request = tmp_path / 'cyclone.json'
    payload = dict(backend='wuduan', workers=6, member_maps=['mslp_10m_winds', 'qpf_1h'],
                   tracking_domain='d01', best_track='bal092022.dat.gz', storm_id='AL092022')
    request.write_text(json.dumps(payload))
    args = SimpleNamespace(command='ensemble' if entry == 'ensemble' else 'go',
        config=config, cyclone_products=request,
        members=6 if entry == 'go-flags' else None, recipe='wofs' if entry == 'go-flags' else None,
        source='gfs:1,gefs:5' if entry == 'go-sources' else None)
    @tc_accumulator.go_scope
    def ensemble_door(received):
        assert not tc_accumulator.active()
        assert json.loads(received.cyclone_products.read_text()) == payload
        return 73
    assert ensemble_door(args) == 73


def test_roster_cycle_is_utc_before_manifest_validation(tmp_path):
    trajectory = dict(cycle='2022-09-28T00', source='gfs')
    row = _track_member(trajectory, 5, tmp_path/'parent.zarr', 6)
    assert row['init'] == '2022-09-28T00:00:00Z'
    assert tc_tracks.utc(row['init']).tzinfo == timezone.utc
    assert row['ancestry']['cycle'] == '2022-09-28T00'
    assert row['ancestry']['ensemble_member_id'] == 5
    # An explicit non-UTC clock must be converted, not relabelled.
    row = _track_member(dict(cycle='2022-09-27T20:00:00-04:00'), 0, 'parent.zarr', 1)
    assert row['init'] == '2022-09-28T00:00:00Z'


@pytest.mark.parametrize('filename', ['bal092022.dat.gz', 'bdeck.dat'])
def test_gzip_bdeck_reaches_native_reader_unchanged(tmp_path, monkeypatch, filename):
    from gpuwm import cyclone_tracks
    # Exact anonymous NOAA archive used by the failing real-case reader.
    path = tmp_path/filename
    compressed = (Path(__file__).parent/'fixtures'/'bal092022.dat.gz').read_bytes()
    path.write_bytes(compressed)
    monkeypatch.setattr(cyclone_tracks, 'native_track_binary', lambda command: 'native-reader')
    def native(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        assert Path(request['input']).read_bytes() == compressed
        Path(request['output']).write_text(json.dumps(dict(schema='gpuwm.tc-tracks.v1',
            tracker=dict(source='atcf'), members=[dict(member_id='best')],
            tracks=[dict(track_id='AL092022', member_id='best', weight=1., fixes=[dict(
                valid='2022-09-26T00:00:00Z', lat=16.8, lon=-80.9,
                mslp_pa=99100, max_wind10_ms=28.29, lead_hours=0, flags=[])])])) )
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(tc_tracks.subprocess, 'run', native)
    result = tc_tracks.best_track(path, 'AL092022')
    assert result['storm_id'] == 'AL092022'
    assert result['fixes'][0]['lat'] == 16.8
    assert result['sha256'] == tc_tracks.sha(path)


def test_real_initial_rain_window_does_not_abort_member_wind_maps():
    from gpuwm.ensemble.cyclone_products import _member_map_plan
    fixture = json.loads((Path(__file__).parent/'fixtures'/'ian-product-warmup.json').read_text())
    assert fixture['current_cyclone_products_would_refuse']
    assert fixture['skipped'][0][0] == 'qpf_6h'
    jobs, unavailable = _member_map_plan([fixture['writer_record']],
        ['10m_wind_speed_and_direction', 'qpf_6h'])
    assert len(jobs) == 1
    assert jobs[0][1] == [fixture['writer_record']]
    assert jobs[0][2] == ['10m_wind_speed_and_direction']
    assert unavailable == [dict(product='qpf_6h', valid_time='2022-09-26_00:00:00',
        reason='accumulation window precedes the first stored frame')]


def test_moving_member_maps_use_the_current_frame_and_exact_rain_endpoints():
    from gpuwm.ensemble.cyclone_products import _member_map_plan
    rows = [dict(valid_time=f'2022-09-26_{hour:02d}:00:00', path=f'placement-{hour}')
            for hour in (0, 1, 6)]
    jobs, unavailable = _member_map_plan(rows, ['mslp_10m_winds', 'qpf_6h'])
    assert [inputs for _, inputs, products in jobs if products == ['mslp_10m_winds']] == [[r] for r in rows]
    assert [inputs for _, inputs, products in jobs if products == ['qpf_6h']] == [[rows[0], rows[2]]]
    assert len(unavailable) == 2
    # Once the window can close, a missing exact start is a real defect.
    with pytest.raises(RuntimeError, match='missing exact accumulation endpoint'):
        _member_map_plan(rows, ['qpf_1h'])
