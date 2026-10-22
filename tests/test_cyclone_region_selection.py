"""The wrong-basin regression, with retained tracker positions and pressures."""
from datetime import datetime
import json

import pytest

from gpuwm.cyclone_tracks import select_track

START = datetime(2024, 9, 24, 12)


def document(tmp_path, *, inside=True):
    def fix(hour, lat, lon, pressure):
        from datetime import timedelta
        return dict(valid=(START+timedelta(hours=hour)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    lead_hours=hour, lat=lat, lon=lon, mslp_pa=pressure,
                    max_wind10_ms=15., flags=[])
    tracks = [dict(track_id='pacific', name='John', member_id='m', weight=1.,
                   fixes=[fix(0,16.396468,-99.821191,100275.484),fix(6,17.17296,-100.77799,100488.875)])]
    if inside:
        tracks.extend([
            dict(track_id='developing', name='Helene', member_id='m', weight=1.,
                 fixes=[fix(0,None,None,None),fix(6,None,None,None),fix(12,None,None,None),
                        fix(18,20.553,-85.332,99234.),fix(24,21.145,-85.909,98427.)]),
            dict(track_id='weaker', member_id='m', weight=1., fixes=[fix(0,38.944,-58.605,100445.664)])])
    path = tmp_path/'tracks.json'
    path.write_text(json.dumps(dict(schema='gpuwm.tc-tracks.v1', tracker={},
        members=[dict(member_id='m',init='2024-09-24T12:00:00Z')],tracks=tracks)))
    return path


def select(path):
    return select_track(path, storm='auto', member=None, source='earth2-fcn3', start=START,hours=1)['lineage']


def test_john_outside_helene_inside_with_late_start(tmp_path):
    row = select(document(tmp_path))
    assert row['track_id'] == 'developing'
    assert row['selected']['name'] == 'Helene'
    assert row['interpolation']['estimated'] is True
    assert row['interpolation']['delay_hours'] == 18
    assert row['interpolation']['displacement_km'] < 300
    assert 18 < row['center_fix']['lat'] < 20
    assert row['selected']['pressure_valid'] == '2024-09-25T06:00:00Z'
    assert row['search_region']['polygon_lon_lat']


def test_no_storm_inside_refuses_with_region_cycle_and_nearest(tmp_path):
    with pytest.raises(ValueError) as error:
        select(document(tmp_path,inside=False))
    message = str(error.value)
    assert all(word in message for word in ('north-atlantic','2024-09-24T12:00:00Z','John','pacific','track-outside-search-area','distance_to_region_km'))


def test_receipt_records_pick_rank_and_every_alternative(tmp_path):
    row = select(document(tmp_path))
    assert row['selected']['rank'] == 1
    assert row['selected']['ranking_mslp_pa'] == 99234
    assert row['selected']['cycle_mslp_pa'] is None
    assert row['selected']['reason'] == 'selected'
    alternatives = {r['track_id']:r for r in row['alternatives']}
    assert set(alternatives) == {'pacific','weaker'}
    assert alternatives['pacific']['reason'] == 'track-outside-search-area'
    assert alternatives['pacific']['center_fix']['lon'] == -99.821191
    assert alternatives['weaker']['reason'] == 'weaker'
    assert alternatives['weaker']['rank'] == 2
    explicit = select_track(document(tmp_path), storm='weaker', member=None,
                            source='earth2-fcn3', start=START, hours=1)['lineage']
    assert all(r['reason'] == 'not-requested' for r in explicit['alternatives'])
