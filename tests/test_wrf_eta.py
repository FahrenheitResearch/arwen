"""Independent WRF Fortran eta, native ABI, and unchanged explicit-grid door."""
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.cpu_backend import CpuPreprocessBackend

FIXTURE = json.loads((Path(__file__).parent/'fixtures/wrf_eta_v461.json').read_text())


def backend():
    try:
        return CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f'native CPU bridge is not available: {error}')


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c:c['name'])
def test_automatic_eta_matches_independent_wrf_fortran(case):
    native=backend()
    if case['exit_code']:
        with pytest.raises(ValueError,match='eta levels|layer thickness'):
            native.generate_wrf_eta(case['e_vert'],**case['options'])
        return
    actual=native.generate_wrf_eta(case['e_vert'],**case['options'])
    expected=np.array(case['eta_bits'],dtype=np.uint32).view(np.float32)
    # The portable Rust exponential rounds from FP64. GNU expf can differ
    # by one ULP before conversion from pressure to eta; compare the WRF
    # arithmetic within one FP32 epsilon of the unit eta interval. All
    # option1 cases and the ordinary option2 default are also bit-exact.
    np.testing.assert_allclose(actual,expected,rtol=0,atol=np.finfo(np.float32).eps)
    if case['options']['auto_levels_opt']==1 or case['name']=='option2-n80-top5000':
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert actual[0] == 1 and actual[-1] == 0 and np.all(np.diff(actual)<0)


@pytest.mark.parametrize('name,value,message', [
    ('auto_levels_opt',True,'integer'),('auto_levels_opt',3,'1 or 2'),
    ('p_top',0,'p_top'),('p_top',100000,'p_top'),('p_top',1e100,'finite'),
    ('max_dz',0,'max_dz'),('dzbot',-1,'dzbot'),
    ('dzstretch_s',0,'dzstretch_s'),('dzstretch_u',float('nan'),'finite'),
])
def test_automatic_eta_rejects_invalid_actual_algorithm_inputs(name,value,message):
    with pytest.raises(ValueError,match=message):
        backend().generate_wrf_eta(80,**{name:value})


def test_eta_failure_does_not_modify_caller_output():
    native=backend()
    call=native._library.gpuwm_wrf_eta_f32
    native.generate_wrf_eta(80) # bind ABI
    out=np.full(80,12345,dtype=np.float32)
    message=ctypes.create_string_buffer(16)
    code=call(out.ctypes.data,80,2,0,1000,50,1.3,1.1,290,ctypes.addressof(message),len(message))
    assert code and np.all(out==12345) and len(message.value)<=15
    assert call(None,80,2,5000,1000,50,1.3,1.1,290,None,0)==1


def run_without_eta(*, controls=None):
    return SimpleNamespace(controls=controls or {}, experiment=SimpleNamespace(
        vertical=SimpleNamespace(eta_levels=(),p_top=5000.),
        root=SimpleNamespace(run=SimpleNamespace(nz=79,base_temp=290.))))


def test_metgrid_default_materializes_actual_wrf_levels_and_receipt():
    import tomllib
    from gpuwm.metem_forecast import resolve_metem_vertical
    actual,policy,receipt=resolve_metem_vertical(run_without_eta(),'[shared]\nbase_temp = 290.0\n')
    decoded=np.array(tomllib.loads(actual)['shared']['eta_levels'],dtype=np.float32)
    expected=next(c for c in FIXTURE['cases'] if c['name']=='option2-n80-top5000')
    np.testing.assert_array_equal(decoded.view(np.uint32),expected['eta_bits'])
    assert receipt['controls']['auto_levels_opt']==2 and receipt['explicit_controls']==[]
    assert 'WRF automatic' in policy


def test_metgrid_explicit_generator_option_reaches_native_generator():
    import tomllib
    from gpuwm.metem_forecast import resolve_metem_vertical
    run=run_without_eta(controls={'domains':{'auto_levels_opt':[1],'max_dz':[1000.]}})
    actual,_,receipt=resolve_metem_vertical(run,'[shared]\n')
    expected=next(c for c in FIXTURE['cases'] if c['name']=='option1-n80-top5000')
    np.testing.assert_array_equal(np.array(tomllib.loads(actual)['shared']['eta_levels'],np.float32).view(np.uint32),expected['eta_bits'])
    assert receipt['explicit_controls']==['auto_levels_opt','max_dz']


def test_explicit_eta_keeps_exact_text_without_native_library(monkeypatch):
    from gpuwm.metem_forecast import resolve_metem_vertical
    run=run_without_eta(controls={'domains':{'auto_levels_opt':[999]}})
    run.experiment.vertical.eta_levels=(1.,.90000000000003,0.)
    text='[shared]\neta_levels = [1.0, 0.90000000000003, 0.0]\n'
    monkeypatch.setenv('GPUWM_CPU_PREPROCESS_BRIDGE','missing-library')
    assert resolve_metem_vertical(run,text)==(text,'explicit namelist eta_levels',None)
    with pytest.raises(ValueError,match='eta_levels is explicit'):
        resolve_metem_vertical(run,text,vertical_grid='native')


def test_explicit_native_alternative_keeps_existing_profile_without_bridge(monkeypatch):
    import tomllib
    from gpuwm.metem_forecast import resolve_metem_vertical
    from gpuwm.core.grid import resample_eta_levels
    from gpuwm.native_wrf_contract import CERTIFIED_ETA_LEVELS
    monkeypatch.setenv('GPUWM_CPU_PREPROCESS_BRIDGE','missing-library')
    actual,_,receipt=resolve_metem_vertical(run_without_eta(),'[shared]\n',vertical_grid='native')
    assert tomllib.loads(actual)['shared']['eta_levels']==resample_eta_levels(CERTIFIED_ETA_LEVELS,79).tolist()
    assert receipt['algorithm']=='ArWen-native-profile'


def test_metgrid_translator_carries_generator_controls_without_changing_explicit_eta(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair
    from gpuwm.namelist_import import import_namelists
    import re
    text=INPUT_TEXT.replace(' p_top_requested = 5000,', ' p_top_requested = 5000,\n auto_levels_opt = 1,\n max_dz = 1200,')
    before,_=import_namelists(*_pair(tmp_path,inp=INPUT_TEXT),metgrid_initialization=True)
    explicit,report=import_namelists(*_pair(tmp_path,inp=text),metgrid_initialization=True)
    assert explicit==before
    assert 'inert' in next(item.reason for item in report.dropped if item.key=='auto_levels_opt')
    noeta=re.sub(r' eta_levels = .*?0\.0,\n','',text,flags=re.S)
    _,report=import_namelists(*_pair(tmp_path,inp=noeta),metgrid_initialization=True)
    assert next(item.fixed_value for item in report.fixed if item.key=='auto_levels_opt')==1


def test_missing_additive_symbol_names_matching_bridge_without_loading_gpu():
    native=CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    native._library=SimpleNamespace()
    with pytest.raises(ValueError,match='bridge lacks WRF eta generation'):
        native.generate_wrf_eta(80)


@pytest.mark.parametrize('selector', [0, 3, 4294967297, -4294967295, 2**64 + 2])
def test_eta_selector_cannot_wrap_before_native_call(selector):
    native = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    native._library = SimpleNamespace()  # no symbol: validation must precede FFI
    with pytest.raises(ValueError, match='auto_levels_opt must be 1 or 2'):
        native.generate_wrf_eta(80, auto_levels_opt=selector)


@pytest.mark.parametrize('marker', [-1.0, -1.00000005, -0.99999994])
@pytest.mark.parametrize('option', [1, 2])
def test_metgrid_automatic_sentinel_reaches_selected_wrf_generator(tmp_path, marker, option):
    import re
    import tomllib
    from test_namelist_import import INPUT_TEXT, _pair
    from gpuwm.experiment import build_experiment
    from gpuwm.metem_forecast import resolve_metem_vertical
    from gpuwm.namelist_import import import_namelists, parse_namelist_text
    text = INPUT_TEXT.replace('e_vert = 9, 9,', 'e_vert = 80, 80,')
    text = text.replace(' p_top_requested = 5000,',
                        f' p_top_requested = 5000,\n auto_levels_opt = {option},')
    omitted = re.sub(r' eta_levels = .*?0\.0,\n', '', text, flags=re.S)
    sentinel = re.sub(r' eta_levels = .*?0\.0,\n',
                      f' eta_levels = {marker},\n', text, flags=re.S)
    expected_text, _ = import_namelists(*_pair(tmp_path, inp=omitted), metgrid_initialization=True)
    actual_text, report = import_namelists(*_pair(tmp_path, inp=sentinel), metgrid_initialization=True)
    assert actual_text == expected_text
    assert next(item.fixed_value for item in report.fixed if item.key == 'eta_levels') == []
    run = SimpleNamespace(experiment=build_experiment(tomllib.loads(actual_text), source='WRF sentinel fixture'),
                          controls=parse_namelist_text(sentinel))
    resolved, _, receipt = resolve_metem_vertical(run, actual_text)
    expected = next(case for case in FIXTURE['cases']
                    if case['name'] == f'option{option}-n80-top5000')
    actual = np.array(tomllib.loads(resolved)['shared']['eta_levels'], np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected['eta_bits'])
    assert receipt['controls']['auto_levels_opt'] == option


@pytest.mark.parametrize('marker,automatic', [
    ([-1.0, 0.5, 0.0], True), ([-0.99999988], False),
    ([-1.00000012], False), ([1.0, 0.5, 0.0], False), ([], False),
])
def test_wrf_eta_marker_uses_first_real_value_and_original_threshold(marker, automatic):
    from gpuwm.ingest.eta import wrf_automatic_eta_requested
    assert wrf_automatic_eta_requested(marker) is automatic
