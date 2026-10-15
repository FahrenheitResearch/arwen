"""Driver admission and atmosphere contract, separate from band fixtures."""
def test_column_driver_exists():
    from gpuwm.core.radiation_composition import ColumnTableRadiation
    assert ColumnTableRadiation.publishes_olr


def test_pair_admitted():
    from gpuwm.config import RA_LW_PHYSICS_ACCEPTED, RA_SW_PHYSICS_ACCEPTED
    assert 5 in RA_LW_PHYSICS_ACCEPTED
    assert 5 in RA_SW_PHYSICS_ACCEPTED


def test_registry_pair_and_output_names():
    import json
    from pathlib import Path
    r=json.loads((Path(__file__).parents[1]/'gpuwm/physics_registry_v2.json').read_text())
    row=r['components']['radiation']['options']['new-goddard']
    assert row['selectors']=={'ra_lw_physics':5,'ra_sw_physics':5}
    assert {'GLW','SWDOWN','SWDDNI','TLWDN','TSWUP'} <= set(row['extensions']['column_contract']['wrfout_names'])


def test_driver_fixture_contract():
    import gzip,struct,numpy as np
    from pathlib import Path
    d=Path(__file__).parents[1]/'tools/goddard_wrf461_oracle/fixtures'
    raw=gzip.decompress((d/'driver-reference.bin.gz').read_bytes())
    assert struct.unpack('<4i',raw[:16])==(0x47524436,64,32,4)
    values=np.frombuffer(raw,dtype='<f4',offset=16)
    assert values.size==64*(2*32+15)
    assert np.isfinite(values).all()
    assert values.min()<0 and values.max()>500


def test_gsfc_radius_provider_contract():
    from types import SimpleNamespace
    from gpuwm.core.radiation_composition import radiation_radius_provider_fields
    from pytest import raises
    names=tuple('re_'+s+'_gsfc' for s in ('ice','cloud','rain','snow','graupel','hail'))
    state=SimpleNamespace(**dict(zip(names,range(6))))
    assert radiation_radius_provider_fields(state,7)==(tuple(range(6)),True)
    del state.re_hail_gsfc
    with raises(ValueError,match='re_hail_gsfc'):
        radiation_radius_provider_fields(state,7)


def test_non_gsfc_radius_routing_and_zero_fallback():
    from types import SimpleNamespace
    from gpuwm.core.radiation_composition import radiation_radius_provider_fields
    state=SimpleNamespace(effc=10,effi=30,effs=80)
    assert radiation_radius_provider_fields(state,8)==((30,10,None,80,None,None),False)
    assert radiation_radius_provider_fields(state,8,0)==((None,)*6,False)
    assert radiation_radius_provider_fields(state,0)==((None,)*6,False)


def test_solver_tier_precedes_header_storage_declarations():
    from gpuwm.core.kernels import module_source_int_defines
    source=module_source_int_defines('goddard',(('GDD_MAX_LEVELS',50),))
    assert source.index('#define GDD_MAX_LEVELS 50') < source.index('#ifndef GDD_MAX_LEVELS')
