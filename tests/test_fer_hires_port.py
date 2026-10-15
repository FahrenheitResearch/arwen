"""Ferrier table, stock reference and carrying-state contracts."""
from pathlib import Path
import hashlib
import re
import numpy as np
import pytest
from gpuwm.core import fer_hires as fh

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests/data/oracles/ferrier_hires'

def test_packaged_table_has_the_pinned_words_and_record_bounds():
    table = fh.load_table()
    words = fh.host_table_words(table)
    assert words.shape == (fh.TABLE_WORDS,)
    assert words.dtype == np.float32
    offset = 0
    for name, n in fh.RECORDS:
        assert len(table[name]) == n
        np.testing.assert_array_equal(words[offset:offset+n], table[name])
        offset += n
    assert offset == fh.OFF_TBPVS

@pytest.mark.parametrize('name,count', [('refute',65),('coverage',22)])
def test_new_stock_fixtures_are_complete_and_pinned(name,count):
    d = FIX / name
    for row in (d/'SHA256SUMS').read_text().splitlines():
        digest, filename = row.split()
        assert hashlib.sha256((d/filename).read_bytes()).hexdigest() == digest
    assert sum(x.startswith('C,') for x in (d/'columns_inputs.csv').read_text().splitlines()) == count
    assert len((d/'levels.csv').read_text().splitlines()) == 1 + count*2*50
    assert len((d/'surface.csv').read_text().splitlines()) == 1 + count*2

def test_restart_inventory_carries_the_ice_history():
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS
    assert set(fh.STATE_FIELDS) <= set(STATE_SERIALIZED_ATTRS)

def test_stock_export_uses_the_real_registry_names():
    from gpuwm.wrf_physics_inventory import _INVENTORIES
    from gpuwm.wrf_direct import _PACKAGE_FIELD_METADATA
    assert _PACKAGE_FIELD_METADATA['CWM'] == ('Total condensate mixing ratio', 'kg kg-1')
    row = _INVENTORIES[5]
    assert [f.netcdf_name for f in row.wrfinput_fields] == ['QVAPOR','QCLOUD','QRAIN','QICE','CWM']
    assert [f.netcdf_name for f in row.runtime_state_not_wrfinput] == ['F_ICE_PHY','F_RAIN_PHY','F_RIMEF_PHY']

def test_mixed_edge_refusal_names_the_missing_rime_closure():
    from gpuwm.core.microphysics_transition import _UNVALIDATED_MIXED_EDGE_REASONS
    assert 'sedimentation' in _UNVALIDATED_MIXED_EDGE_REASONS[5][1]

def test_the_radius_uses_stock_wrf_dum2_order():
    # This is a port contract, backed by the full strict column comparison.
    src = (ROOT/'gpuwm/core/kernels/fer_hires.cu').read_text()
    assert 'Rcw = ARcw * gfk_pow(DUM2, FH_C1);' in src

@pytest.mark.gpu
@pytest.mark.parametrize('name', ['lane','refute','coverage'])
def test_strict_gpu_column_oracle(name,tmp_path):
    import os
    import importlib.util
    import cupy as cp
    if os.environ.get('GPUWM_WRF_EXACT') != '1':
        pytest.skip('the bitwise column proof requires GPUWM_WRF_EXACT=1')
    root=ROOT/'tools/ferrier_hires_wrf461_oracle'
    def load(filename):
        spec=importlib.util.spec_from_file_location(filename,root/(filename+'.py'))
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod
    ref=FIX if name=='lane' else FIX/name
    load('run_woof_ferrier').main(ref/'columns_inputs.csv',tmp_path)
    assert load('compare_ferrier').compare(ref,tmp_path,ref/'columns_inputs.csv')
