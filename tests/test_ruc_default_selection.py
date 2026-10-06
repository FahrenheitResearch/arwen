"""Generic RUC retains the snow and irrigation arithmetic before the switches.

The historical irrigation function is copied exactly from ``74135cfe3^``.
The snow inverse is pinned to the complete CUDA file from ``04496587c^``.
Those independent anchors supplement repeated whole-driver word comparisons;
comparing two current calls alone would not establish historical identity.
SOILPROP stays at its separately qualified ``wrf_45`` forecast default.
"""

from dataclasses import fields
import hashlib
from pathlib import Path
import re

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.ruc import ruc_land_surface_step
from gpuwm.core.ruc_mosaic import irrigate

ROOT = Path(__file__).resolve().parents[1]

# Exact ast.get_source_segment for irrigate from 74135cfe3^, plus one newline.
HISTORICAL_IRRIGATION = '''def irrigate(soilm1d, *, landusef, vegfrac, shdmin, shdmax, wilt, qmin,
             nroot, crop, natural, active, arrays=np):
    """WRF LSMRUC's post-SFCTMP irrigation, including its water addition.

    Only soil moisture changes. WRF does not update liquid water or fluxes
    here and does not record the added water in a budget accumulator.
    """
    f = arrays.float32
    if landusef.shape[0] < max(crop, natural):
        raise ValueError("RUC landusef omits the table's crop/natural categories; "
                         "LSMRUC irrigation cannot index the source fractions")
    croparea, naturalarea = landusef[crop - 1], landusef[natural - 1]
    factor = arrays.maximum(f(0), arrays.minimum(f(1),
        ((vegfrac - shdmin).astype(f) / arrays.maximum(f(1), (shdmax - shdmin).astype(f))).astype(f)))
    enabled = active & ((croparea > 0) | (naturalarea > 0)) & (factor > f(.75))
    cropsm = ((f(1.1) * wilt).astype(f) - qmin).astype(f)
    cropfr = arrays.minimum(f(1), (croparea + (f(.4) * naturalarea).astype(f)).astype(f))
    for k in range(soilm1d.shape[0]):
        newsm = ((cropsm * cropfr).astype(f) +
                 ((f(1) - cropfr).astype(f) * soilm1d[k]).astype(f)).astype(f)
        soilm1d[k] = arrays.where(enabled & (k < nroot) & (soilm1d[k] < newsm), newsm, soilm1d[k])
'''
HISTORICAL_IRRIGATION_SHA256 = '6a90bfeba4bb96a4b4fb9c848920499f6cd626d7bfe3d29702c0bd4cd70760a6'
PRE_SNOW_KERNEL_SHA256 = '844546e307b148bab0cd19dfdc786845d16d298ddd4345684c158539ca99e8d8'
PRE_IRRIGATION_DEVICE_ARM_SHA256 = 'e9bc42ce7cbdb51d55e3ab0c02ca57b39b0b9fda0f7ec04ba6df44dc0546b00d'


def _generic_config():
    return RunConfig(nx=8, ny=6, nz=12, dx=3000., dy=3000., ztop=10000.,
                     dt=20., run_seconds=120., moist=True, mp_physics=6,
                     sf_surface_physics=3, num_soil_layers=9,
                     sf_sfclay_physics=1, bl_pbl_physics=1, bldt=0.)


def _words(left, right):
    a, b = np.asarray(left), np.asarray(right)
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),
                                  np.ascontiguousarray(b).view(np.uint8))


def _braced_block(source, marker):
    begin = source.index('{', source.index(marker))
    depth = 0
    for end in range(begin, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if not depth:
            return source[begin:end + 1]
    raise AssertionError(f'unclosed source block after {marker!r}')


def test_generic_snow_dispatch_selects_the_exact_pre_switch_kernel():
    from test_ruc_nzs_tier import _reconstruct_pre_snow
    from gpuwm.core.ruc_tier import ruc_module_defines

    source = (ROOT / 'gpuwm/core/kernels/ruc.cu').read_text(encoding='utf-8')
    historical = _reconstruct_pre_snow(source)
    assert hashlib.sha256(historical.encode()).hexdigest() == PRE_SNOW_KERNEL_SHA256
    # The unchanged CUDA file uses this define for the historical arm.
    assert ruc_module_defines(9, soilprop='wrf_45') == (('GPUWM_SNOW_WRF461', 1),)
    assert ruc_module_defines(9, soilprop='wrf_45', snow='wrf_45') == ()
    # The historical comparison must reject an unrelated one-word edit.
    changed = historical.replace('0.023f', '0.024f', 1)
    assert changed != historical
    assert hashlib.sha256(changed.encode()).hexdigest() != PRE_SNOW_KERNEL_SHA256


def test_generic_irrigation_device_arm_is_the_pre_switch_arithmetic():
    source = (ROOT / 'tools/ruc_fused/driver_body.cuh').read_text(encoding='utf-8')
    legacy = _braced_block(source, 'if(irrigation==1)')
    arithmetic = _braced_block(legacy, 'if(mosaic_lu)')
    normalized = re.sub(r'\s+', '', arithmetic)
    assert hashlib.sha256(normalized.encode()).hexdigest() == PRE_IRRIGATION_DEVICE_ARM_SHA256


@pytest.mark.parametrize('nzs', [6, 9])
def test_generic_irrigation_matches_the_historical_function_after_every_step(nzs):
    assert hashlib.sha256(HISTORICAL_IRRIGATION.encode()).hexdigest() == HISTORICAL_IRRIGATION_SHA256
    namespace = {'np': np}
    exec(compile(HISTORICAL_IRRIGATION, '<pre-switch-irrigation>', 'exec'), namespace)
    historical_irrigate = namespace['irrigate']
    n = 12
    landusef = np.zeros((21, n), np.float32)
    landusef[11] = np.asarray([.05, .8, 0, 0, 1, .3, .05, .8, 0, 0, 1, .3], np.float32)
    landusef[13] = np.asarray([0, 0, .5, 1, 0, .7, 0, 0, .5, 1, 0, .7], np.float32)
    common = dict(landusef=landusef, vegfrac=np.full(n, 80, np.float32),
                  shdmin=np.full(n, 10, np.float32), shdmax=np.full(n, 90, np.float32),
                  wilt=np.full(n, .137, np.float32), qmin=np.full(n, .05, np.float32),
                  nroot=np.asarray([4, 2, 3, 4, 1, 4] * 2, np.int32),
                  crop=12, natural=14, active=np.asarray([True] * 10 + [False] * 2))
    common['vegfrac'][5] = 40
    generic = np.full((nzs, n), .03, np.float32)
    historical = generic.copy()
    fork = generic.copy()
    for step in range(24):
        # Keep a controlled moisture sink so this also exercises repeated gates.
        if step % 3 == 2:
            for soil in (generic, historical, fork):
                soil[0] = np.maximum(np.float32(0), soil[0] - np.float32(.001))
        irrigate(generic, **common)
        historical_irrigate(historical, **common)
        irrigate(fork, **common, form='wrf_45',
                 lai=np.full(n, 5.68, np.float32),
                 ivgtyp=np.asarray([12, 12, 14, 14, 12, 14] * 2, np.int32))
        _words(generic, historical)
    assert not np.array_equal(generic.view(np.uint32), fork.view(np.uint32))


@pytest.mark.parametrize('case', range(12))
def test_generic_full_driver_matches_named_legacy_after_every_host_call(case):
    import test_ruc as oracle

    cfg = _generic_config()
    assert cfg.ruc_soilprop == 'wrf_45'
    assert cfg.ruc_qvg_cold_start == 'wrf'
    assert cfg.ruc_2m_diagnostic == 'flux'
    groups, fixture = oracle._lsmruc_oracle(oracle.LSMRUC_ORACLE)
    values, keywords = oracle._lsmruc_call(fixture, case)
    generic = {name: value.copy() for name, value in values.items()}
    legacy = {name: value.copy() for name, value in values.items()}
    for step in range(1, 7):
        keywords['ktau'] = step
        actual = ruc_land_surface_step(generic, **keywords,
                                      soilprop=cfg.ruc_soilprop,
                                      irrigation=cfg.ruc_irrigation,
                                      snow=cfg.ruc_snow,
                                      qvg_cold_start=cfg.ruc_qvg_cold_start)
        expected = ruc_land_surface_step(legacy, **keywords,
                                        soilprop='wrf_45', irrigation='wrf_461',
                                        snow='wrf_461', qvg_cold_start='wrf')
        for descriptor in fields(actual):
            name = descriptor.name
            _words(getattr(actual, name), getattr(expected, name))
        for name in oracle.RUC_DRIVER_PROFILE_STATE + oracle.RUC_DRIVER_COLUMN_STATE:
            generic[name] = getattr(actual, name).copy()
            legacy[name] = getattr(expected, name).copy()
    assert groups[case][1] == 1


def test_generic_snow_comparison_reaches_a_changed_fork_column():
    import test_ruc as oracle

    cfg = _generic_config()
    _, fixture = oracle._lsmruc_oracle(oracle.LSMRUC_ORACLE)
    # Thin urban snow and a deep forest pack expose cover and heat-flow changes.
    changed = set()
    for case in (2, 3, 8, 9):
        values, keywords = oracle._lsmruc_call(fixture, case)
        actual = ruc_land_surface_step(values, **keywords,
                                      soilprop=cfg.ruc_soilprop,
                                      irrigation=cfg.ruc_irrigation, snow=cfg.ruc_snow)
        fork = ruc_land_surface_step(values, **keywords, soilprop='wrf_45',
                                    irrigation='wrf_461', snow='wrf_45')
        for descriptor in fields(actual):
            name = descriptor.name
            if not np.array_equal(np.asarray(getattr(actual, name)).view(np.uint32),
                                  np.asarray(getattr(fork, name)).view(np.uint32)):
                changed.add(name)
    assert {'snowc', 'soilt'} <= changed, changed
