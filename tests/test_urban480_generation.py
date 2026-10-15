"""Generation selection, real table controls, state inventory and restart law."""
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import pytest

from gpuwm.checkpoint_identity import urban_algorithm_identity
from gpuwm.config import validate_urban_config
from gpuwm.core.urban_state import option_spec,urban_var_init_host
from gpuwm.core.urban_tables import load_urban_params,urban_category_set
from gpuwm.core.urban_ucm import controlled_params,pack_params,UrbanCanopyError,UCM480_STATE_PLANES
from gpuwm.io.restart import configuration_echo,_require_config_match,_configuration_digest_values
from test_urban_config import _cfg


def test_height_admission_reads_the_selected_generation_and_table(monkeypatch):
    from test_urban_config import _tree
    from gpuwm.experiment import ucm_first_level_refusal
    import gpuwm.core.urban_tables as tables
    experiment = _tree(1, 1)
    run = experiment.domains[0].run
    run.urban_ucm_generation = 'wrf_480'
    run.urban_table_dir = '/selected/urban/table'
    calls = []
    def heights(lcz, **options):
        calls.append((lcz, options))
        return {1: 100.0}
    monkeypatch.setattr(tables, 'ucm_canopy_heights', heights)
    assert '100.0 m' in ucm_first_level_refusal(experiment)
    assert calls == [(1, {'generation': 'wrf_480', 'tbl_dir': run.urban_table_dir}),
                     (0, {'generation': 'wrf_480', 'tbl_dir': run.urban_table_dir})]


@pytest.mark.parametrize('tree',(0,1))
@pytest.mark.parametrize('tt',(0,1))
@pytest.mark.parametrize('fvg',(0.,1.))
def test_source_controls_select_the_new_parameter_layout(tree,tt,fvg):
    cfg=_cfg(sf_urban_physics=1,urban_ucm_generation='wrf_480',urban_treeoption=tree,urban_ttscheme=tt,urban_fvg=fvg)
    validate_urban_config(cfg)
    params=controlled_params(load_urban_params(1,generation='wrf_480'),cfg)
    packed=pack_params(params)
    assert packed.switches[-3:].tolist()==[tree,tt,1]
    assert np.all(packed.table[:,36]==np.float32(fvg))
    assert packed.table.shape==(3,49)
    assert set(n for n,_ in UCM480_STATE_PLANES)<=set(option_spec(1,generation='wrf_480'))


def test_old_generation_state_and_restart_echo_stay_unchanged():
    cfg=_cfg(sf_urban_physics=1,urban_water_balance='wrf_471')
    old=asdict(cfg)
    for key in ('urban_ucm_generation','urban_treeoption','urban_ttscheme','urban_fvg','urban_table_dir'):
        old.pop(key)
        assert key not in configuration_echo(cfg)
    assert _configuration_digest_values(old)==_configuration_digest_values(asdict(cfg))
    _require_config_match(old,cfg,Path('checkpoint'))
    assert not set(n for n,_ in UCM480_STATE_PLANES)&set(option_spec(1))
    assert urban_algorithm_identity(cfg)=='urban-slucm-wrf-v4.7.1-v1'


def test_generation_and_control_flips_refuse_restart():
    old=_cfg(sf_urban_physics=1)
    new=replace(old,urban_ucm_generation='wrf_480')
    with pytest.raises(ValueError,match='urban_ucm_generation'):
        _require_config_match(configuration_echo(old),new,Path('checkpoint'))
    assert urban_algorithm_identity(new)!=urban_algorithm_identity(old)
    _require_config_match(configuration_echo(new),new,Path('checkpoint'))
    for field,value in (('urban_treeoption',0),('urban_ttscheme',1),('urban_fvg',.5)):
        with pytest.raises(ValueError,match=field):
            _require_config_match(configuration_echo(new),replace(new,**{field:value}),Path('checkpoint'))


def test_unsolved_force_restore_vegetation_is_refused():
    params=load_urban_params(1,generation='wrf_480')
    values=dict(params.values);values['TS_SCHEME_DATA']=np.asarray(2,np.int32);values['TREEOPTION']=np.asarray(1,np.int32)
    with pytest.raises(UrbanCanopyError,match='does not solve vegetated ground temperature or moisture'):
        pack_params(replace(params,values=values))


@pytest.mark.parametrize('field,value',(('urban_fvg',-.1),('urban_fvg',1.1),('urban_fvg',float('nan')),('urban_treeoption',2),('urban_ttscheme',2)))
def test_source_control_bounds(field,value):
    with pytest.raises(ValueError,match=field):
        validate_urban_config(_cfg(sf_urban_physics=1,urban_ucm_generation='wrf_480',**{field:value}))


def test_new_registry_arrays_cold_start_from_the_same_lsm_layers():
    params=load_urban_params(1,generation='wrf_480')
    tsk=np.array([[291,301]],np.float32)
    soil=np.array([[[289,299]],[[287,297]],[[285,295]],[[283,293]]],np.float32)
    arrays=urban_var_init_host(option=1,use_wudapt_lcz=0,params=params,categories=urban_category_set(isurban=13),
        ivgtyp=np.array([[13,14]],np.int32),tsk=tsk,tslb=soil,tmn=tsk-10,smois=soil*0+.2,num_urban_hi=15,nz=4,spec=option_spec(1,generation='wrf_480'))
    np.testing.assert_array_equal(arrays['tvg_urb2d'],tsk)
    np.testing.assert_array_equal(arrays['tt_urb2d'],tsk)
    np.testing.assert_array_equal(arrays['tvgl_urb3d'],arrays['trl_urb3d'])
    np.testing.assert_array_equal(arrays['smg_urb3d'],arrays['smr_urb3d'])
