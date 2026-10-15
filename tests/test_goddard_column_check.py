"""Check the acceptance checker independently of a GPU implementation."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import struct
import numpy as np
import pytest

spec=spec_from_file_location('goddard_check',Path(__file__).parents[1]/'tools/goddard_wrf461_oracle/check_columns.py')
check=module_from_spec(spec)
spec.loader.exec_module(check)

@pytest.mark.parametrize('band',['sw_net','sw_down','sw_up','sw_surface','lw_net','lw_down','lw_up'])
def test_one_bit_flux_mutation_fails_strict(band):
    reference={name:np.array([[1.,2.]],dtype='<f4').view('<u4') for name in check.FIELDS}
    candidate={name:value.copy() for name,value in reference.items()}
    candidate[band][0,1]^=np.uint32(1)
    passed,receipt=check.compare_fields(reference,candidate,True)
    assert not passed
    assert receipt[band]['different_words']==1
    assert receipt[band]['max_ulp']==1

@pytest.mark.parametrize('band',['sw_down','lw_down'])
def test_default_large_flux_error_fails(band):
    reference={name:np.array([[1.]],dtype='<f4').view('<u4') for name in check.FIELDS}
    candidate={name:value.copy() for name,value in reference.items()}
    candidate[band][:]=np.float32(2.).view(np.uint32)
    assert not check.compare_fields(reference,candidate,False)[0]

def test_signed_zero_is_a_strict_word_difference():
    reference={'lw_net':np.array([[0.]],dtype='<f4').view('<u4')}
    candidate={'lw_net':np.array([[-0.]],dtype='<f4').view('<u4')}
    passed,receipt=check.compare_fields(reference,candidate,True)
    assert not passed
    assert receipt['lw_net']['different_words']==1

def test_nonfinite_even_matching_words_fail():
    reference={'lw_net':np.array([[np.nan]],dtype='<f4').view('<u4')}
    assert not check.compare_fields(reference,reference,True)[0]

def test_missing_field_fails():
    with pytest.raises(ValueError,match='field set'):
        check.compare_fields({'sw_net':np.zeros((1,1),dtype='<u4')},{},True)

def test_wrong_shape_fails():
    with pytest.raises(ValueError,match='shape'):
        check.compare_fields({'sw_net':np.zeros((1,1),dtype='<u4')},
                             {'sw_net':np.zeros((1,2),dtype='<u4')},True)

def test_truncated_fixture_fails(tmp_path):
    file=tmp_path/'columns.bin'
    file.write_bytes(struct.pack('<5i',0x47524435,64,32,4,64))
    with pytest.raises(ValueError,match='truncated'):
        check.read_columns(file)

def test_committed_fixture_is_finite_and_complete(tmp_path):
    import gzip
    root=Path(__file__).parents[1]/'tools/goddard_wrf461_oracle/fixtures'
    file=tmp_path/'reference.bin'
    file.write_bytes(gzip.decompress((root/'defined-reference.bin.gz').read_bytes()))
    columns,kind=check.read_columns(file)
    assert kind==64
    assert sum(v.size for k,v in columns.items() if k in check.FIELDS[:7])==12928
    assert all(v.shape[0]==64 for v in columns.values())
    assert all(np.isfinite(v.view('<f4')).all() for v in columns.values())

def test_real_fixture_mutation_in_each_band_is_caught(tmp_path):
    import gzip
    root=Path(__file__).parents[1]/'tools/goddard_wrf461_oracle/fixtures'
    file=tmp_path/'reference.bin'
    file.write_bytes(gzip.decompress((root/'defined-reference.bin.gz').read_bytes()))
    reference,_=check.read_columns(file)
    for band in ('sw_net','lw_net'):
        candidate={name:words.copy() for name,words in reference.items()}
        candidate[band][19,31]^=np.uint32(1)
        passed,receipt=check.compare_fields(reference,candidate,True)
        assert not passed and receipt[band]['different_words']==1

def test_band_source_regenerates_from_pinned_wrf(tmp_path):
    import os,subprocess,sys
    wrf=os.environ.get('WRF_SOURCE_ROOT')
    if not wrf:pytest.skip('pristine WRF source required')
    root=Path(__file__).parents[1]
    output=tmp_path/'bands.cuh'
    subprocess.run([sys.executable,str(root/'tools/goddard_wrf461_oracle/translate_bands.py'),wrf,str(output)],check=True)
    assert output.read_bytes()==(root/'gpuwm/core/kernels/goddard_bands.cuh').read_bytes()

def test_packaged_cloud_words_are_exact_rust_decoder_output(tmp_path):
    import os,subprocess,gzip
    wrf=os.environ.get('WRF_SOURCE_ROOT')
    if not wrf:pytest.skip('pristine WRF source required')
    root=Path(__file__).parents[1]
    binary=tmp_path/'rw_lookup_table'
    subprocess.run(['rustc','--edition=2021','-O',str(root/'tools/goddard_wrf461_oracle/decode_cloud_table.rs'),'-o',str(binary)],check=True)
    words=subprocess.check_output([str(binary),str(Path(wrf)/'run/BROADBAND_CLOUD_GODDARD.bin')])
    packaged=gzip.decompress((root/'gpuwm/data/wrf_radiation/goddard/cloud-table.le.bin.gz').read_bytes())
    assert words==packaged
    assert len(words)==172656
    assert np.isfinite(np.frombuffer(words,dtype='<f4')).all()

def test_nan_radius_reference_has_zero_cloud_optics(tmp_path):
    import gzip
    root=Path(__file__).parents[1]/'tools/goddard_wrf461_oracle/fixtures'
    file=tmp_path/'reference.bin';file.write_bytes(gzip.decompress((root/'defined-reference.bin.gz').read_bytes()))
    reference,_=check.read_columns(file)
    for name in check.FIELDS[7:]:
        assert not np.any(reference[name][63]),name
