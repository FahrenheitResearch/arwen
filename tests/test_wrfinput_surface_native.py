"""Real compiled WRF surface-driver output words and native startup carriers."""
import json
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.surface_humidity import cap_land_q2

DATA = Path(__file__).parent / 'data'


def _words(values):
    return np.array([int(value, 16) for value in values], np.uint32).view(np.float32)


def _native_cap_columns():
    receipt = json.loads((DATA / 'wrf471_surface_q2_columns.json').read_text())
    rows = receipt['rows']
    return tuple(_words([row[key] for row in rows]) for key in (
        'q2_after_sfcd_bits', 'qv1_bits', 'xland_bits', 'wrf_final_q2_bits'))


def test_final_surface_q2_matches_observed_native_columns():
    # Expected words came from the actual unchanged WRF history, after the
    # independent observer saw SFCDIAGS return. They are not a Python mirror.
    q2, qv1, xland, expected = _native_cap_columns()
    cap_land_q2(q2, qv1, xland, xp=np)
    np.testing.assert_array_equal(q2.view(np.uint32), expected.view(np.uint32))


def test_final_surface_q2_uses_strict_native_xland_boundary():
    # This boundary is absent from the captured real grid (classes 1 and 2).
    xland = np.array([np.nextafter(np.float32(1.5), np.float32(0.)),
                      1.5, np.nextafter(np.float32(1.5), np.float32(2.))], np.float32)
    q2 = np.full(3, .05, np.float32)
    qv1 = np.full(3, .01, np.float32)
    cap_land_q2(q2, qv1, xland, xp=np)
    # Rounded binary32 1.05 times rounded binary32 .01 is 0x3c2c0830.
    expected = _words(['3c2c0830', '3d4ccccd', '3d4ccccd'])
    np.testing.assert_array_equal(q2.view(np.uint32), expected.view(np.uint32))


@pytest.mark.gpu
def test_final_surface_q2_device_matches_observed_native_columns():
    cp = pytest.importorskip('cupy')
    q2, qv1, xland, expected = _native_cap_columns()
    device = cp.asarray(q2)
    cap_land_q2(device, cp.asarray(qv1), cp.asarray(xland), xp=cp)
    np.testing.assert_array_equal(cp.asnumpy(device).view(np.uint32),
                                  expected.view(np.uint32))


def _native_albbck_landuse():
    from gpuwm.core.landuse import initialize_landuse
    receipt = json.loads((DATA / 'wrf471_landuse_albbck_columns.json').read_text())
    raw = {name: _words(spec['fp32_bits']).reshape(spec['shape'])
           for name, spec in receipt['raw_fields'].items()}
    attrs = receipt['attributes']
    landuse = initialize_landuse(
        raw['LU_INDEX'][None, :], soil_type=raw['ISLTYP'][None, :],
        landmask=raw['LANDMASK'][None, :], snow=raw['SNOW'][None, :],
        xice=raw['XICE'][None, :], valid_time=datetime.fromisoformat(receipt['valid_time']),
        cen_lat=attrs['CEN_LAT'], mminlu=attrs['MMINLU'],
        iswater=int(attrs['ISWATER']), islake=int(attrs['ISLAKE']),
        isice=int(attrs['ISICE']), isoilwater=int(attrs['ISOILWATER']),
        soil_temperature=raw['TSLB'][:, None, :], sst=raw['SST'][None, :])
    expected = _words(receipt['expected_albbck_bits'])[None, :]
    np.testing.assert_array_equal(landuse.albbck.view(np.uint32), expected.view(np.uint32))
    return receipt, raw, landuse


@pytest.mark.parametrize('surface,monthly,has_landuse',
                         [(2, False, True), (2, True, True),
                          (2, False, False), (1, False, True)])
def test_native_cold_start_preserves_selected_background_albedo(
        monkeypatch, surface, monthly, has_landuse):
    from gpuwm.config import RunConfig
    from gpuwm.ingest import wrfinput as wi
    receipt, samples, initialized = _native_albbck_landuse()
    shape = (1, 14)
    cfg = RunConfig(nx=14, ny=1, nz=3, dx=3000., dy=3000., dt=12.,
                    ztop=10000., run_seconds=0.,
                    moist=True, mp_physics=8, sf_surface_physics=surface,
                    sf_sfclay_physics=1, bl_pbl_physics=1, usemonalb=monthly)
    raw = {name: value[None, :] for name, value in samples.items()
           if name != 'TSLB'}
    raw.update(ALBBCK=_words(receipt['raw_albbck_bits'])[None, :],
               TSLB=samples['TSLB'][:, None, :],
               SMOIS=np.full((4, *shape), .3, np.float32),
               SH2O=np.full((4, *shape), .3, np.float32),
               TSK=np.full(shape, 290., np.float32),
               TMN=np.full(shape, 288., np.float32),
               VEGFRA=np.full(shape, 60., np.float32),
               SNOWH=np.zeros(shape, np.float32),
               LAI=np.full(shape, 2., np.float32), GLW=np.full(shape, 300., np.float32))
    fields = {'albbck': initialized.albbck.copy(), 'lai': np.zeros(shape, np.float32)}
    driver = SimpleNamespace(fields=fields, noah_params=None, rainc=None)
    # Exercise the real native restoration function on CPU. The already
    # independently checked land-use initializer supplies the expected words.
    monkeypatch.setitem(sys.modules, 'cupy', np)
    monkeypatch.setitem(sys.modules, 'gpuwm.core.physics', SimpleNamespace(
        initialize_physics=lambda *a, **kw: driver,
        NOAH_LAYER_THICKNESS_M=(.1, .3, .6, 1.)))
    monkeypatch.setitem(sys.modules, 'gpuwm.core.noah', SimpleNamespace(
        initialize_noah_liquid_water=lambda *a: None))
    restored = SimpleNamespace(raw=raw, global_attributes={'MMINLU':receipt['attributes']['MMINLU']})
    wi.initialize_wrfinput_physics(object(), restored, cfg,
                                  landuse=initialized if has_landuse else None)
    expected = (initialized.albbck if surface == 2 and not monthly and has_landuse
                else raw['ALBBCK'])
    np.testing.assert_array_equal(fields['albbck'].view(np.uint32), expected.view(np.uint32))
