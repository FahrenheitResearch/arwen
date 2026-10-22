"""Ordinary zero static records pass; active gridded input is never dropped."""
from types import SimpleNamespace
import numpy as np
import pytest
from gpuwm.ingest.wrfinput import URBAN_TABLE_ONLY_INPUTS, urban_static_input_dispositions


class Variable:
    def __init__(self,dimensions,value):
        self.dimensions=dimensions
        self.value=value
    def __getitem__(self,key):
        return self.value


def fixture(name):
    extents={'Time':1,'south_north':2,'west_east':3,
             'urban_wind_directions_stag':4,'month_hour':288}
    dimensions=URBAN_TABLE_ONLY_INPUTS[name]
    value=np.zeros(tuple(extents[d] for d in dimensions),np.float32)
    ds=SimpleNamespace(variables={name:Variable(dimensions,value)},
                       dimensions={d:range(n) for d,n in extents.items()})
    return ds,{'south_north':2,'west_east':3}


@pytest.mark.parametrize('name,generation', [
    pytest.param((tuple(URBAN_TABLE_ONLY_INPUTS))[5], 'wrf_471', id='wrf_471-STDH_URB2D'),
])
def test_zero_source_static_records_keep_the_table_cold_start(name,generation):
    ds,shape=fixture(name)
    cfg=SimpleNamespace(sf_urban_physics=1,urban_ucm_generation=generation)
    result=urban_static_input_dispositions(ds,'wrfinput_d01',cfg,shape)
    assert 'zero optional urban' in result[name]
    ds.variables[name].value.reshape(-1)[0]=np.float32(.25)
    with pytest.raises(ValueError,match='missing gridded urban morphology or external anthropogenic-heat handover'):
        urban_static_input_dispositions(ds,'wrfinput_d01',cfg,shape)
    assert 'inactive' in urban_static_input_dispositions(
        ds,'wrfinput_d01',SimpleNamespace(sf_urban_physics=0),shape)[name]


def test_static_morphology_cannot_borrow_wrong_grid_dimensions():
    ds,shape=fixture('BUILD_HEIGHT')
    ds.variables['BUILD_HEIGHT'].dimensions=('Time','west_east','south_north')
    with pytest.raises(ValueError,match='incompatible dimensions'):
        urban_static_input_dispositions(ds,'wrfinput_d01',SimpleNamespace(sf_urban_physics=1),shape)


def test_active_nonfinite_morphology_is_refused():
    ds,shape=fixture('MH_URB2D')
    ds.variables['MH_URB2D'].value[0,0,0]=np.nan
    with pytest.raises(ValueError,match='nonzero or nonfinite'):
        urban_static_input_dispositions(ds,'wrfinput_d01',SimpleNamespace(sf_urban_physics=1),shape)