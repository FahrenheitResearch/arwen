"""Unit boundary checks; physical evidence is the native real-case run."""
from dataclasses import replace
import json
import pytest
from gpuwm.ocean.interface import OceanGrid, validate_plane, AtmosphereForcing, validate_fields
from gpuwm.ocean.door import read_config, rows

class Descriptor:
    def __init__(self,shape=(63,99),typestr='<f4',strides=None,readonly=False):
        self.__cuda_array_interface__={'shape':shape,'typestr':typestr,'strides':strides,'data':(4096,readonly),'version':3}

def grid():
    a=Descriptor();return OceanGrid(99,63,25000.,25000.,a,a,a,a)

@pytest.mark.parametrize('field,match',[([], 'CUDA'),(Descriptor((99,63)),'shape'),(Descriptor(typestr='<f8'),'float32'),(Descriptor(strides=(4,252)),'contiguous'),(Descriptor(readonly=True),'writable')])
def test_reject_corrupt_solver_boundary(field,match):
    with pytest.raises((TypeError,ValueError),match=match):validate_plane(field,(63,99),'pressure')

def test_shared_forcing_fields_are_validated():
    g=grid();g.validate();a=Descriptor()
    f=AtmosphereForcing(a,a,a,a,a,a,a,0.)
    validate_fields(f,g)
    with pytest.raises(ValueError,match='float32'):validate_fields(replace(f,tau_y=Descriptor(typestr='<f8')),g)

@pytest.mark.parametrize('values',[{'interval_seconds':0},{'interval_seconds':float('inf')},{'solvers':['waves','waves']},{'solvers':['unregistered']},{'unknown':2}])
def test_invalid_config_has_no_silent_fallback(tmp_path,values):
    # TOML literals for finite and infinite values stay ordinary TOML.
    data={'prepared':'bathymetry',**values}
    lines=[]
    for k,v in data.items():
        value='inf' if isinstance(v,float) and v==float('inf') else json.dumps(v)
        lines.append(f'{k} = {value}')
    p=tmp_path/'ocean.toml';p.write_text('\n'.join(lines))
    with pytest.raises(ValueError):read_config(p)

def test_config_and_product_rows_are_reachable(tmp_path):
    p=tmp_path/'ocean.toml';p.write_text('[ocean]\nprepared="bathy"\nsolvers=["waves", "surge"]\n')
    cfg=read_config(p)
    assert cfg['prepared']==str(tmp_path/'bathy')
    assert set(cfg['solvers'])==set(rows()['solvers'])
    assert {'hs','tp','direction','eta','inundation','elevation','wind_speed','psfc','stress'}<=rows()['products'].keys()


def test_atmosphere_config_anchors_ocean_contract_beside_run_file(tmp_path,monkeypatch):
    from gpuwm.config import _anchor_config_file_paths
    configs=tmp_path/'configs';configs.mkdir()
    contract=configs/'ocean.toml';contract.write_text('[ocean]\nprepared="prepared"\n')
    monkeypatch.chdir(tmp_path)
    merged={'ocean_config':'ocean.toml'}
    _anchor_config_file_paths(merged,configs/'run.toml')
    assert merged['ocean_config']==str(contract)


def test_history_units_describe_stored_si_pressure_not_display_hpa():
    from gpuwm.io.wrf_output_schema import REGISTRY_VAR_META
    assert REGISTRY_VAR_META['OCEAN_HS'][1]=='m'
    assert REGISTRY_VAR_META['OCEAN_ETA'][1]=='m'
    assert REGISTRY_VAR_META['OCEAN_PSFC'][1]=='Pa'
