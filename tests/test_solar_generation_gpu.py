"""Real device solar consumers, including midnight and fractional clocks."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('generation,release', [('wrf_461','v4.6.1'),('wrf_480','v4.8.0')])
@pytest.mark.parametrize('prefix',['','polar_'])
def test_all_fortran_solar_words(generation,release,prefix):
    import cupy as cp
    from gpuwm.core.solar import solar_geometry_device
    fixture = np.load(Path(__file__).parent/'data/oracles/solar_generations'/('oracle-'+release)/'solar_refs.npz')
    for index,(j,xt,g) in enumerate(fixture[prefix+'scalars']):
        ds,cz,ha = solar_geometry_device(j,xt,g,fixture[prefix+'lat'],fixture[prefix+'lon'],generation=generation)
        got = np.concatenate([cp.asnumpy(ds),cp.asnumpy(cz).ravel(),cp.asnumpy(ha).ravel()])
        np.testing.assert_array_equal(got.view('u4'),fixture[prefix+'ref'][index].view('u4'))


@pytest.mark.parametrize('elapsed',[0., 0.001, 60., 900., 86400., 172800.25])
def test_corrected_consumers_share_the_run_clock(elapsed):
    import cupy as cp
    from gpuwm.core.solar import geometry_at
    from gpuwm.core.dudhia import wrf_solar_geometry
    from gpuwm.core.rrtmgp import RRTMGPRadiation
    from gpuwm.core.urban_driver import solar_geometry
    from gpuwm.core.noahmp_runtime import NoahmpSolarGeometry
    from gpuwm.core.swint import ShortwaveInterpolation
    start = datetime(2024,2,28,23,59,31,250000,tzinfo=timezone.utc)
    valid = start + timedelta(seconds=elapsed)
    lat = np.asarray([[-90.,-60.,0.,45.,90.]],np.float32)
    lon = np.asarray([[-180.,-90.,0.,90.,180.]],np.float32)
    half_interval = 217.5
    ds,cz,ha = geometry_at(valid,lat,lon,start_time=start,elapsed_seconds=elapsed,
                          hour_offset_seconds=half_interval)
    expected = cp.asnumpy(cz).view('u4')
    got,solcon = wrf_solar_geometry(valid,lat,lon,start_time=start,elapsed_seconds=elapsed,
                                   hour_offset_seconds=half_interval)
    np.testing.assert_array_equal(got.view('u4'),expected)
    assert np.float32(solcon).view('u4') == cp.asnumpy(ds)[1].view('u4')
    assert np.float32(RRTMGPRadiation._solar_constant(valid)).view('u4') == cp.asnumpy(ds)[1].view('u4')
    adapter = SimpleNamespace(latitude_deg=cp.asarray(lat),longitude_deg=cp.asarray(lon),start_time=start)
    got = RRTMGPRadiation._cosine_zenith(adapter,valid,hour_offset_seconds=half_interval)
    np.testing.assert_array_equal(cp.asnumpy(got).view('u4'),expected)
    urban = solar_geometry(start,elapsed,lat,lon,radt_seconds=half_interval*2.)
    np.testing.assert_array_equal(urban.coszen.view('u4'),expected)
    np.testing.assert_array_equal(urban.hrang.view('u4'),cp.asnumpy(ha).view('u4'))
    no_offset = cp.asnumpy(geometry_at(valid,lat,lon,start_time=start,elapsed_seconds=elapsed)[1])
    geometry = NoahmpSolarGeometry(start,lat,lon)
    np.testing.assert_array_equal(geometry.cosz(elapsed).view('u4'),no_offset.view('u4'))
    carrier = ShortwaveInterpolation(start_time=start,latitude_deg=cp.asarray(lat),
                                    longitude_deg=cp.asarray(lon),shape=lat.shape)
    np.testing.assert_array_equal(cp.asnumpy(carrier.coszen_loc(elapsed)).reshape(lat.shape).view('u4'),
                                  no_offset.view('u4'))


def test_calendar_bounds_and_empty_grid():
    import cupy as cp
    from gpuwm.core.solar import solar_geometry_device
    with pytest.raises(ValueError,match='finite'):
        solar_geometry_device(np.nan,0.,0.,[0.],[0.])
    with pytest.raises(ValueError,match='shapes'):
        solar_geometry_device(0.,0.,0.,[0.],[0.,1.])
    ds,cz,ha = solar_geometry_device(0.,0.,0.,np.empty((0,),np.float32),np.empty((0,),np.float32))
    assert cz.size == ha.size == 0
    assert np.isfinite(cp.asnumpy(ds)).all()


def test_calc_coszen_keeps_supplied_declination():
    from gpuwm.core.rrtmg_legacy import calc_coszen
    lat = np.asarray([0.,45.,90.],np.float32)
    lon = np.zeros_like(lat)
    equator = calc_coszen(172.5,720.,0.,lat,lon,np.float32(0.))
    tilted = calc_coszen(172.5,720.,0.,lat,lon,np.float32(0.4))
    assert np.any(equator.view('u4') != tilted.view('u4'))


@pytest.mark.parametrize('generation',['wrf_480','wrf_461'])
def test_interpolation_rebuild_preserves_every_carried_word(generation):
    import cupy as cp
    from gpuwm.core.swint import ShortwaveInterpolation, STATE_FIELDS
    start = datetime(2024,6,20,23,58,0,tzinfo=timezone.utc)
    lat = cp.asarray([[-90.,-45.,0.,45.,90.]],cp.float32)
    lon = cp.asarray([[-180.,-90.,0.,90.,180.]],cp.float32)
    kwargs = dict(start_time=start,latitude_deg=lat,longitude_deg=lon,shape=lat.shape,generation=generation)
    original = ShortwaveInterpolation(**kwargs)
    fields = {name:cp.zeros_like(lat) for name in (*STATE_FIELDS,'albedo','swdown','gsw')}
    fields['albedo'].fill(np.float32(.3))
    for elapsed in (0.,435.):
        cosine = original.coszen_loc(elapsed+217.5).reshape(lat.shape).copy()
        fields['swdown'][...] = cp.maximum(cosine,cp.float32(0.))*cp.float32(1000.)
        original.radiation_step(fields,coszen=cosine,swddir=fields['swdown']*cp.float32(.8),
                                elapsed_seconds=elapsed)
    restored = ShortwaveInterpolation(**kwargs)
    copied = {name:array.copy() for name,array in fields.items()}
    for elapsed in (450.001,900.,1800.,86400.25):
        original.between_calls(fields,elapsed_seconds=elapsed)
        restored.between_calls(copied,elapsed_seconds=elapsed)
        for name in fields:
            np.testing.assert_array_equal(cp.asnumpy(fields[name]).view('u4'),cp.asnumpy(copied[name]).view('u4'))
