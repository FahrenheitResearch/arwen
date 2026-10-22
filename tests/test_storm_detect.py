"""Physical criteria and tracking failure modes for the storm detection door."""
import copy
import json
import pytest
from gpuwm.storm_detect import load_rows,eligible,verify_reference


def test_basin_table_drives_fields_and_thresholds():
    rows={r['kind']:r for r in load_rows()['basins']}
    assert set(rows)=={'medicane','polar-low'}
    assert {'z900','z600','z300'}<=set(rows['medicane']['fields'])
    assert rows['polar-low']['sst_t500_min_k']==43
    assert rows['medicane']['hart_b_abs_max_m']==10


def test_qualification_does_not_bridge_failed_or_missing_times():
    row=load_rows()['basins'][0]
    def f(h,c=True):return dict(valid=f'2023-09-08T{h:02d}:00:00Z',classified=c)
    assert eligible(dict(fixes=[f(0),f(3),f(6)]),row)==(True,6)
    assert eligible(dict(fixes=[f(0),f(3,False),f(6)]),row)==(False,0)
    assert eligible(dict(fixes=[f(0),f(9)]),row)==(False,0)


def test_reference_requires_explicit_source_licence(tmp_path):
    p=tmp_path/'ref.json';p.write_text(json.dumps(dict(source_url='https://example.org/track',license='CC-BY-4.0',fixes=[])))
    with pytest.raises(ValueError,match='requires --reference-license'):verify_reference([],p,None)
    with pytest.raises(ValueError,match='matching license'):verify_reference([],p,'MIT')
    assert verify_reference([],p,'CC-BY-4.0')['comparisons']==[]


@pytest.fixture
def cuda():
    cp=pytest.importorskip('cupy')
    try:cp.cuda.runtime.getDeviceCount()
    except Exception as e:pytest.skip(str(e))
    return cp


def fixture_detector(cp,kind='medicane'):
    from gpuwm.storm_detect_gpu import Detector
    row=copy.deepcopy(next(r for r in load_rows()['basins'] if r['kind']==kind))
    grid=dict(lat=[40-i*.25 for i in range(81)],lon=[10+i*.25 for i in range(81)],shape=[81,81])
    det=Detector(grid,row);d=det.distance(30,20);profile=cp.exp(-(d/90)**2)
    data={k:cp.zeros(det.shape,dtype=cp.float64) for k in row['fields']}
    data.update(msl=101000-1500*profile,mask=cp.ones(det.shape,dtype=bool),land=cp.zeros(det.shape),sst=cp.full(det.shape,290.),t500=cp.full(det.shape,240.),wind=25*cp.exp(-((d-80)/60)**2),vort=cp.full(det.shape,2.e-4))
    for p in range(300,901,50):data[f'z{p}']=16000-9000*cp.log(cp.asarray(p)/300)-100*cp.log(cp.asarray(p)/250)*profile
    return det,data,row


def test_hart_warm_cold_and_asymmetry(cuda):
    det,data,_=fixture_detector(cuda)
    warm=det.diagnostic(data,40,40,motion=(1,0))
    assert warm['hart_lower_m']>0 and warm['hart_upper_m']>0
    assert warm['hart_b_abs_m']<1
    assert warm['classified'],warm
    profile=cuda.exp(-(det.distance(30,20)/90)**2)
    for p in range(300,901,50):data[f'z{p}']=16000-9000*cuda.log(cuda.asarray(p)/300)-100*cuda.log(1000/cuda.asarray(p))*profile
    cold=det.diagnostic(data,40,40,motion=(1,0))
    assert cold['hart_lower_m']<0 and cold['hart_upper_m']<0
    assert not cold['classified'] and 'hart_lower_m' in cold['failed_criteria']
    det,data,_=fixture_detector(cuda)
    data['z600']+=cuda.broadcast_to((det.lat[:,None]-30)*40,det.shape)
    assert 'hart_b_abs_m' in det.diagnostic(data,40,40,motion=(1,0))['failed_criteria']


def test_polar_contrast_ocean_and_coverage(cuda):
    det,data,_=fixture_detector(cuda,'polar-low')
    assert det.diagnostic(data,40,40)['classified']
    data['sst'][:]=280
    assert 'sst_t500_k' in det.diagnostic(data,40,40)['failed_criteria']
    data['sst'][:]=290;data['land'][:]=1
    assert 'ocean_fraction' in det.diagnostic(data,40,40)['failed_criteria']
    data['land'][:]=0;data['mask'][40,41]=False
    assert 'coverage' in det.diagnostic(data,40,40)['failed_criteria']


def test_spherical_vorticity_includes_metric_term(cuda):
    from gpuwm.storm_detect_gpu import Detector
    grid=dict(lat=[70-i*.25 for i in range(161)],lon=[i*.25 for i in range(1440)],shape=[161,1440])
    det=Detector(grid,load_rows()['basins'][0])
    z=det.vorticity(cuda.full(det.shape,20.),cuda.zeros(det.shape))
    expected=20*cuda.tan(det.phi)/6371000
    assert float(cuda.max(cuda.abs(z[1:-1]-expected[1:-1])).item())<2.e-10


def test_association_one_to_one_speed_and_dateline(cuda):
    from gpuwm.storm_detect_gpu import associate
    det,_,_=fixture_detector(cuda)
    tracks=[dict(track_id='a',fixes=[dict(valid='2023-09-08T00:00:00Z',lat=30,lon=179.5)]),dict(track_id='b',fixes=[dict(valid='2023-09-08T00:00:00Z',lat=32,lon=179.5)])]
    links=associate(det,tracks,[dict(lat=30,lon=-179.5),dict(lat=32,lon=-179.5)],'2023-09-08T03:00:00Z',40,6)
    assert {k:v['track_id'] for k,v in links.items()}=={0:'a',1:'b'}
    assert associate(det,tracks,[dict(lat=60,lon=0)],'2023-09-08T03:00:00Z',40,6)=={}
    assert associate(det,tracks,[dict(lat=30,lon=179.5)],'2023-09-08T12:00:00Z',40,6)=={}


def test_polar_candidates_are_vortices_not_pressure_minima(cuda):
    det,data,row=fixture_detector(cuda,'polar-low')
    row['bounds']=[20,40,10,30]
    data['vort'][:]=0
    data['vort'][40,40]=2.e-4
    data['msl'][:]=101000
    data['msl'][40,35]=99000
    assert det.candidates(data)==[(40,40)]


def test_land_sst_missing_does_not_invalidate_coastal_environment(cuda,tmp_path):
    import numpy as np
    det,data,row=fixture_detector(cuda,'polar-low')
    data['land'][:4,:]=1
    data['sst'][:4,:]=cuda.nan
    for key in row['fields']:
        cuda.asnumpy(data[key]).astype('<f4').tofile(tmp_path/(key+'.f32'))
    read=det.read(tmp_path)
    assert bool(cuda.all(read['mask']).item())
    read['sst'][40,40]=cuda.nan
    # Atmospheric missingness always fails the physical coverage requirement.
    read['mask'][40,40]=False
    assert 'coverage' in det.diagnostic(read,40,40)['failed_criteria']


def test_nonfinite_diagnostic_writes_strict_json(cuda):
    det,data,row=fixture_detector(cuda,'polar-low')
    data['sst'][:]=cuda.nan
    result=det.diagnostic(data,40,40)
    assert not result['classified'] and result['sst_t500_k'] is None
    json.dumps(result,allow_nan=False)


def test_reference_does_not_reward_unrelated_timestamp_overlap(tmp_path):
    p=tmp_path/'ref.json'
    fixes=[dict(valid=f'2023-09-08T{h:02d}:00:00Z',lat=35,lon=20) for h in (0,3,6)]
    p.write_text(json.dumps(dict(source_url='https://example.org/track',license='CC0-1.0',fixes=fixes)))
    far=dict(track_id='far',fixes=[dict(f,lat=45,classified=True) for f in fixes])
    near=dict(track_id='near',fixes=[dict(f,classified=True) for f in fixes[:2]])
    result=verify_reference([far,near],p,'CC0-1.0')
    assert result['comparisons'][0]['track_id']=='near'
    assert result['comparisons'][0]['matched']==2
    assert result['comparisons'][1]['matched']==0


def test_motion_preserves_crossing_vortex_identities(cuda):
    from gpuwm.storm_detect_gpu import associate
    det,_,_=fixture_detector(cuda)
    def f(hour,lon):return dict(valid=f'2023-09-08T{hour:02d}:00:00Z',lat=35,lon=lon)
    tracks=[dict(track_id='east',fixes=[f(0,18),f(3,20)]),dict(track_id='west',fixes=[f(0,23),f(3,21)])]
    links=associate(det,tracks,[dict(lat=35,lon=19),dict(lat=35,lon=22)],'2023-09-08T06:00:00Z',40,6)
    assert {k:v['track_id'] for k,v in links.items()}=={0:'west',1:'east'}


def test_global_longitude_seam_keeps_candidates(cuda):
    from gpuwm.storm_detect_gpu import Detector
    row=copy.deepcopy(load_rows()['basins'][0]);row['bounds']=[30,40,-7,42]
    grid=dict(lat=[40-i*.25 for i in range(41)],lon=[i*.25 for i in range(1440)],shape=[41,1440])
    det=Detector(grid,row)
    data=dict(mask=cuda.ones(det.shape,dtype=bool),vort=cuda.zeros(det.shape),msl=cuda.full(det.shape,103000.))
    data['msl'][20,0]=99000
    assert det.candidates(data)==[(20,0)]


def test_batched_neighborhoods_keep_independent_coverage_and_ocean_masks(cuda):
    det,data,row=fixture_detector(cuda)
    results=det.diagnostics(data,[(40,40),(1,40)],[(1,0),None])
    assert results[0]['classified'] and results[0]['hart_b_abs_m']<1
    assert not results[1]['coverage_complete'] and 'coverage' in results[1]['failed_criteria']
    det,data,row=fixture_detector(cuda,'polar-low')
    data['land'][:,:30]=1;data['wind'][:]=25
    results=det.diagnostics(data,[(40,20),(40,60)],[None,None])
    assert 'ocean_fraction' in results[0]['failed_criteria']
    assert results[1]['classified']


def test_batched_polar_cap_wraps_seam_without_losing_environment(cuda):
    from gpuwm.storm_detect_gpu import Detector
    row=copy.deepcopy(next(r for r in load_rows()['basins'] if r['kind']=='polar-low'))
    grid=dict(lat=[80-i*.25 for i in range(81)],lon=[i*.25 for i in range(1440)],shape=[81,1440])
    det=Detector(grid,row)
    data={k:cuda.zeros(det.shape) for k in row['fields']}
    data.update(mask=cuda.ones(det.shape,dtype=bool),msl=cuda.full(det.shape,99000.),wind=cuda.full(det.shape,20.),vort=cuda.full(det.shape,2.e-4),sst=cuda.full(det.shape,290.),t500=cuda.full(det.shape,240.))
    results=det.diagnostics(data,[(40,0),(40,1439),(1,0)],[None,None,None])
    for result in results[:2]:
        assert result['classified'] and result['ocean_fraction']==pytest.approx(1)
        assert result['sst_t500_k']==pytest.approx(50)
        assert result['vorticity850_s']==pytest.approx(2.e-4)
    assert not results[2]['coverage_complete']


@pytest.mark.parametrize("motion",[(0,1),(1,0),(1,1),None])
def test_batched_hart_splits_boundary_cell_area_independently_of_batch(cuda,motion):
    import math
    det,data,row=fixture_detector(cuda)
    data['z600']+=cuda.broadcast_to((det.lon[None,:]-20)*25,det.shape)
    disk=det.distance(30,20)<=row['diagnostic_radius_km']
    dx=cuda.cos(det.phi)*cuda.sin(det.lam-math.radians(20))
    dy=math.cos(math.radians(30))*cuda.sin(det.phi)-math.sin(math.radians(30))*cuda.cos(det.phi)*cuda.cos(det.lam-math.radians(20))
    angles=[math.atan2(motion[1],motion[0])] if motion else [i*math.pi/18 for i in range(18)]
    thick=data['z600']-data['z900']
    def mean(mask):
        w=cuda.where(disk,det.weights*mask,0.)
        return cuda.sum(cuda.where(disk,thick,0.)*w)/w.sum()
    expected=[]
    for angle in angles:
        signed=dx*math.sin(angle)-dy*math.cos(angle)
        right=cuda.where(cuda.abs(signed)<=1.e-14,.5,(signed>0).astype(cuda.float64))
        expected.append(float(cuda.abs(mean(right)-mean(1-right)).item()))
    expected=max(expected)
    actual=det.diagnostics(data,[(40,40),(39,41)],[motion,None])[0]['hart_b_abs_m']
    assert actual==pytest.approx(expected,abs=1.e-9)
    assert det.diagnostic(data,40,40,motion)["hart_b_abs_m"]==pytest.approx(actual,abs=1.e-9)
