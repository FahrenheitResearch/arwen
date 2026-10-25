"""The reported +6 detection, +12 miss, and out-of-domain automatic low."""
from datetime import datetime, timedelta
import json

import pytest

from gpuwm.cyclone_tracks import early_start_fix, select_track

START = datetime(2024, 9, 24, 12)


def fix(hour, lat=None, lon=None, pressure=None):
    return dict(valid=(START + timedelta(hours=hour)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        lead_hours=hour, lat=lat, lon=lon, mslp_pa=pressure,
        max_wind10_ms=None, flags=[])


@pytest.mark.parametrize('search_area', [None, [16, -95, 34, -75]])
def test_missed_fix_does_not_send_nest_to_offshore_low(search_area):
    # Exact native tracker output regenerated from the pinned FCN3 input.
    # The unmodified selector chose 0-001 at 38.9444 N, 58.6047 W.
    from pathlib import Path
    path = Path(__file__).parent/'fixtures/fcn3-20240924T12-tracks.json'
    result = select_track(path, storm='auto', member=None, source='earth2-fcn3',
                          start=START, hours=24, search_area=search_area)['lineage']
    assert result['track_id'] == '0-004'
    assert result['interpolation']['missed_fixes'] == 1
    assert result['interpolation']['motion_interval_hours'] == 12
    assert result['interpolation']['estimated']
    assert result['selected']['cycle_mslp_pa'] is None
    assert result['selected']['pressure_valid'] == '2024-09-24T18:00:00Z'
    assert 19 < result['center_fix']['lat'] < 19.762220720735158
    if search_area is not None:
        wrong = next(t for t in result['alternatives'] if t['track_id'] == '0-001')
        assert wrong['reason'] == 'track-outside-search-area'


@pytest.mark.parametrize('fixes', [
    [fix(6, 20, -85, 100000), fix(12), fix(18), fix(24, 21, -86, 99000)],
    [fix(6, 20, -85, 100000), fix(12), fix(30, 21, -86, 99000)],
    [fix(6, 20, -85, 100000), fix(12), fix(18, 40, -60, 99000)],
])
def test_gap_time_and_displacement_bounds_still_refuse(fixes):
    assert early_start_fix(fixes, START, dict(init='2024-09-24T12:00:00Z')) == (None, {})


@pytest.mark.parametrize('region,explicit,expected', [
    (dict(id='forecast-area', area=[16, -95, 34, -75]), None, '--search-area=16,-95,34,-75'),
    (dict(id='north-atlantic', area=[0, -100, 55, -20], polygon_lon_lat=[[0, 0]]), None, None),
    (dict(id='forecast-area', area=[16, -95, 34, -75]), [18, -90, 30, -80], '--search-area=18.0,-90.0,30.0,-80.0'),
])
def test_job_forwards_verified_regional_bounds(tmp_path, monkeypatch, region, explicit, expected):
    from gpuwm import earth2_job
    from gpuwm.earth2_door import build_spec
    spec = build_spec(model='fcn3', cycle='2024-09-24T12', hours=24,
                      storm='auto', nest='2.5km', area=explicit)
    monkeypatch.setattr(earth2_job, '_preflight', lambda spec: None)
    commands = {}
    def stage(name, command, root, receipt):
        commands[name] = [str(v) for v in command]
        if name == 'fetch-model':
            (root/'source').mkdir()
            (root/'source'/'earth2-run.json').write_text(json.dumps(dict(
                forecast='forecast.nc', wall_seconds=0, runner={},
                cache_reuse=dict(wall_seconds=0, coverage=dict(search_region=region)))))
        if name == 'setup':
            raise RuntimeError('captured setup command')
    monkeypatch.setattr(earth2_job, '_command', stage)
    with pytest.raises(RuntimeError, match='captured setup command'):
        earth2_job.run_job(spec, tmp_path/'job')
    assert [s for s in commands['setup'] if s.startswith('--search-area=')] == ([] if expected is None else [expected])
