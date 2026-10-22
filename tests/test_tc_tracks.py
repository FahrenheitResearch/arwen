"""Completed-output contract, family weights, native field mapping and NHC times."""
import csv
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest
from gpuwm import tc_tracks as tc


def member(name, family='ai', **extra):
    return dict(member_id=name, family=family, source='fixture', init='2024-09-24T12:00:00Z',
                ancestry=['parent', {'seed': 7}], frames=[{'path': 'frame.nc', 'valid': '2024-09-24T12:00:00Z'}], **extra)


def test_weights_are_within_family_and_ancestry_is_lossless():
    doc = tc.normalize_manifest(dict(schema=tc.MANIFEST, members=[member('a'), member('b'), member('c', 'arwen')]))
    assert [m['weight'] for m in doc['members']] == [.5, .5, 1]
    assert doc['members'][0]['ancestry'] == ['parent', {'seed': 7}]
    with pytest.raises(ValueError, match='all weights or none'):
        tc.normalize_manifest(dict(schema=tc.MANIFEST, members=[member('a', weight=.2), member('b')]))
    with pytest.raises(ValueError, match='duplicate valid'):
        m = member('a'); m['frames'] *= 2
        tc.normalize_manifest(dict(schema=tc.MANIFEST, members=[m]))


def test_independent_lead_and_ensemble_axes_expand_without_weight_loss(monkeypatch):
    inv = {'variables': {
        'time': dict(dims=['time'], shape=[1], values=[0], attrs={'units': 'hours since 2024-09-24 12:00:00'}),
        'lead_time': dict(dims=['lead_time'], shape=[2], values=[0, 6], attrs={'units': 'hours'}),
        'ensemble': dict(dims=['ensemble'], shape=[2], values=[5, 9], attrs={})}}
    monkeypatch.setattr(tc, 'inventory', lambda path: inv)
    doc = tc.normalize_manifest(dict(schema=tc.MANIFEST, members=[dict(member_id='fcn', family='ai', source='earth2-fcn3', file='all.nc', weight=.8, ancestry={'opaque': [3]})]))
    a, b = doc['members']
    assert a['member_id'] == 'fcn-ens000' and b['member_id'] == 'fcn-ens001'
    assert a['weight'] == b['weight'] == .4
    assert b['frames'][1]['select'] == {'time': 0, 'lead_time': 1, 'ensemble': 1}
    assert b['frames'][1]['valid'] == '2024-09-24T18:00:00Z'
    assert b['ancestry'] == {'opaque': [3]}


def test_mapping_declares_height_conversion_and_every_backend_requirement():
    table = json.loads(tc.TABLE.read_text())
    for vocab in table['vocabularies'].values():
        assert vocab['wind_basis'] == 'earth-relative'
        for config in table['backends'].values():
            for name in config['fields']:
                assert name in vocab['fields']
                assert isinstance(vocab['fields'][name]['scale'], (int, float))
    assert table['vocabularies']['arwen-grib2']['fields']['z200']['scale'] == 9.80665
    assert table['vocabularies']['era5-zarr']['fields']['z500']['units'] == 'm**2 s**-2'


def test_csv_nulls_weights_scores_and_special_fix_minutes(tmp_path):
    m = member('a', weight=.25)
    fixes = [dict(tc.missing('2024-09-24T12:00:00Z', m['init']), lat=20, lon=-85, mslp_pa=99000, max_wind10_ms=30, flags=['edge']), tc.missing('2024-09-24T18:00:00Z', m['init'])]
    tracks = [dict(track_id='a-0', member_id='a', weight=.25, fixes=fixes)]
    tc.write_csv(tmp_path/'tracks.csv', tracks)
    with (tmp_path/'tracks.csv').open() as stream:
        reader=csv.DictReader(stream); rows=list(reader)
        assert reader.fieldnames == tc.COLUMNS
    assert rows[1]['lat'] == '' and rows[1]['weight'] == '0.25' and json.loads(rows[1]['flags']) == ['missing']
    result=tc.scores([m], tracks, {'fixes': [dict(f, mslp_pa=98000) for f in fixes]})['a']
    assert (result['detections'],result['misses']) == (1,1)
    assert result['common_valid_times'][0]['pressure_error_hpa'] == 10
    deck=tmp_path/'b.dat'; deck.write_text('AL, 09, 2024092703, 10, BEST, 0, 300N, 837W, 120, 939, HU\n')
    best=tc.best_track(deck)
    assert best['fixes'][0]['valid'] == '2024-09-27T03:10:00Z'
    assert best['fixes'][0]['lat'] == 30 and best['fixes'][0]['lon'] == -83.7
    hurdat=tmp_path/'h.txt'; hurdat.write_text('AL092024, HELENE, 1,\n20240927, 0310, L, HU, 30.0N, 83.7W, 120, 939,\n')
    assert tc.best_track(hurdat)['fixes'][0]['flags'] == ['landfall']


def test_native_zarr_scalar_and_data_declared_unit_conversion(tmp_path):
    binary=os.environ.get('GPUWM_RW_TC_FIELDS')
    if not binary: pytest.skip('set GPUWM_RW_TC_FIELDS to the built native reader')
    store=tmp_path/'fields.zarr'; store.mkdir(); (store/'.zgroup').write_text('{"zarr_format":2}')
    def array(name, shape, dims, values, units, dtype='<f8'):
        d=store/name;d.mkdir()
        (d/'.zarray').write_text(json.dumps(dict(zarr_format=2,shape=shape,chunks=shape,dtype=dtype,compressor=None,filters=None,fill_value=None,order='C')))
        (d/'.zattrs').write_text(json.dumps({'_ARRAY_DIMENSIONS':dims,'units':units}))
        (d/('.'.join('0' for _ in shape) or '0')).write_bytes(struct.pack('<' + ('q' if dtype=='<i8' else 'd')*len(values),*values))
    array('init_time', [], [], [1727179200], 'seconds since 1970-01-01 00:00:00', '<i8')
    array('latitude',[5],['latitude'],[1,.75,.5,.25,0],'degrees_north')
    array('longitude',[5],['longitude'],[-2,-1.75,-1.5,-1.25,-1],'degrees_east')
    array('pressure',[5,5],['latitude','longitude'],[1000]*25,'hPa')
    inv=json.loads(subprocess.check_output([binary,'inspect',str(store)],text=True))
    assert inv['variables']['init_time']['values']==[1727179200]
    req=dict(path=str(store),out=str(tmp_path/'native'),select={},fields=['msl'],vocabulary=dict(format='zarr',lat='latitude',lon='longitude',fields={'msl':dict(name='pressure',scale=100,offset=0,units='hPa')}))
    request=tmp_path/'request.json';request.write_text(json.dumps(req))
    subprocess.run([binary,'extract',str(request)],check=True)
    assert struct.unpack('<25f',(tmp_path/'native/msl.f32').read_bytes()) == (100000.,)*25
    assert (tmp_path/'native/mask.u8').read_bytes() == bytes([1]*25)
    req['vocabulary']['fields']['msl']['units']='Pa';request.write_text(json.dumps(req))
    bad=subprocess.run([binary,'extract',str(request)],capture_output=True,text=True)
    assert bad.returncode != 0 and 'units' in bad.stderr


def test_native_reader_uses_engine_artifact_lookup(monkeypatch, tmp_path):
    from gpuwm import bridges
    calls = []
    binary = tmp_path / "rw_tc_fields"
    binary.write_text("native tracker")
    def find(env, name, **kwargs):
        calls.append((env, name, kwargs))
        return binary
    monkeypatch.setattr(bridges, "find_artifact", find)
    assert tc.binary("rw_tc_fields") == str(binary)
    assert calls == [("GPUWM_RW_TC_FIELDS", bridges.executable_name("rw_tc_fields"),
                      {"crate_relative": "tools/rustwx"})]


def test_verification_does_not_replace_missing_storm_with_other_basin():
    m = member('a', weight=.25)
    valid = m['frames'][0]['valid']
    best = dict(valid=valid, lat=20., lon=-85., mslp_pa=99000.)
    foreign = dict(best, lon=-105.)
    track = dict(member_id='a', track_id='pacific', weight=.25, fixes=[foreign])
    result = tc.scores([m], [track], {'fixes': [best]})['a']
    assert (result['detections'], result['misses']) == (0, 1)
    assert result['weight'] == .25
    assert result['common_valid_times'][0]['track_id'] is None
    assert result['common_valid_times'][0]['centre_distance_km'] is None
    assert track['fixes'] == [foreign]


def test_history_export_vocabulary_is_shared_with_roster_products():
    from gpuwm.tc_tracks import history_export_arguments
    args = history_export_arguments('d07', 4)
    assert args[:2] == ['--domains', 'd07']
    assert args[args.index('--threads')+1] == '4'
    assert 'mean_sea_level_pressure' in args[args.index('--variables')+1].split(',')


@pytest.mark.parametrize('hours,minimum,provisional', [(6, 21600, True), (24, 86400, False), (72, 86400, False)])
def test_tempest_short_forecast_requires_full_available_record(hours, minimum, provisional):
    from datetime import timedelta
    start = tc.utc('2024-09-24T12:00:00Z')
    frames = [dict(valid=start.isoformat()), dict(valid=(start + timedelta(hours=hours)).isoformat())]
    config = json.loads(tc.TABLE.read_text())['backends']['tempestextremes']
    duration = tc.tempest_duration(frames, config)
    assert duration['effective_minimum_seconds'] == minimum
    assert duration['configured_minimum_seconds'] == 86400
    assert duration['provisional'] is provisional
    assert '--mintime' not in config['stitch']
