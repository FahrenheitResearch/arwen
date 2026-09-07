"""Public metgrid launch and its shared initialization contracts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('arguments', [[], ['case.toml','--met-em','met'],
    ['--wrfinput','wrf','--met-em','met']])
def test_exclusive_input_choice_precedes_readiness(monkeypatch, arguments):
    from gpuwm import cli, capabilities
    monkeypatch.setattr(capabilities,'require_for_command',lambda *a:pytest.fail('invalid choice reached readiness'))
    with pytest.raises(SystemExit) as stopped:
        cli.main(['run',*arguments])
    assert stopped.value.code == 2


def test_public_metgrid_dispatch_and_existing_config(monkeypatch):
    from gpuwm import cli, capabilities, provenance_gate, metem_forecast
    calls=[]
    monkeypatch.setattr(capabilities,'require_for_command',lambda *a:None)
    monkeypatch.setattr(provenance_gate,'announce',lambda *a:None)
    monkeypatch.setattr(metem_forecast,'run_metem_forecast',lambda *a,**kw:calls.append((a,kw)) or 0)
    assert cli.main(['run','--met-em','met','--outdir','forecast','--run-seconds','120']) == 0
    assert calls[0][0] == (Path('met'),Path('forecast'))
    assert calls[0][1]['run_seconds'] == 120
    parsed=cli.build_parser().parse_args(['run','case.toml'])
    assert parsed.config == Path('case.toml') and parsed.met_em is None


def test_metgrid_context_does_not_spoof_wrf_boundary_representation(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair
    from gpuwm.namelist_import import import_namelists
    text=INPUT_TEXT.replace(' use_theta_m = 0,',' use_theta_m = 1,')
    paths=_pair(tmp_path,inp=text)
    _, report=import_namelists(*paths,metgrid_initialization=True)
    entry=next(item for item in report.fixed if item.key=='use_theta_m')
    assert 'physical temperature' in entry.reason
    with pytest.raises(ValueError,match='distinct input contracts'):
        import_namelists(*paths,metgrid_initialization=True,wrf_boundary_use_theta_m=1)


def test_shared_deep_soil_formula_preserves_native_bytes():
    from gpuwm.static.build import deep_soil_temperature_at_terrain
    rng=np.random.default_rng(217)
    for dtype in (np.float32,np.float64):
        soil=(280+rng.random((5,7))*15).astype(dtype)
        terrain=(rng.random((5,7))*3000).astype(dtype)
        mask=(rng.random((5,7))>.3).astype(dtype)
        old=np.where(mask>.5,soil-.0065*terrain,soil)
        new=deep_soil_temperature_at_terrain(soil,terrain,mask)
        assert new.dtype == old.dtype
        assert new.tobytes() == old.tobytes()


@pytest.mark.parametrize('value', [1, 'false', None])
def test_soil_public_routers_do_not_coerce_truthy_fractional_flag(value):
    from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil, preprocess_ruc_soil
    for call,kw in ((preprocess_land_surface_soil,{'sf_surface_physics':2}),
                    (preprocess_ruc_soil,{})):
        with pytest.raises(TypeError,match='fractional_seaice must be boolean'):
            call({},soil_type=None,fractional_seaice=value,**kw)


def pressure_case(levels):
    return SimpleNamespace(path=Path('met_em.d01.test'),snapshot=SimpleNamespace(levels_hpa=np.array(levels),fields={}),
        attributes={'FLAG_PSFC':1,'FLAG_SOILHGT':1},geometry={'num_metgrid_levels':len(levels)+1})


def test_hybrid_pressure_representation_resolves_wrf_sfcp_override():
    from gpuwm.metem_door import metgrid_initialization_controls
    run=SimpleNamespace(controls={})
    assert metgrid_initialization_controls(pressure_case([100,300,700,950]),run)['sfcp_to_sfcp'] is True
    descending = pressure_case([950,700,300,100])
    with pytest.raises(ValueError,match='sfcprs3 requires FLAG_SLP'):
        metgrid_initialization_controls(descending,run)
    descending.attributes['FLAG_SLP'] = 1
    with pytest.raises(ValueError,match='declared PMSL field'):
        metgrid_initialization_controls(descending,run)
    descending.snapshot.fields['PMSL'] = np.full((1,1),101100.)
    assert metgrid_initialization_controls(descending,run)['sfcp_to_sfcp'] is False
    run.controls={'domains':{'sfcp_to_sfcp':[True]}}
    assert metgrid_initialization_controls(pressure_case([950,700,300,100]),run)['sfcp_to_sfcp'] is True
    missing=pressure_case([950,700,300,100]);missing.attributes['FLAG_PSFC']=0
    with pytest.raises(ValueError,match='FLAG_PSFC'):
        metgrid_initialization_controls(missing,run)


def test_metgrid_metadata_missing_directory_is_actionable(tmp_path):
    from gpuwm.metem_door import resolve_metem_run
    with pytest.raises(ValueError,match='producing namelist.input'):
        resolve_metem_run(tmp_path)


def test_scalar_soil_coordinate_is_named_refusal(tmp_path):
    from test_metem_ingest import case
    from gpuwm.ingest.metem import read_met_em, MetgridRefusal
    def scalar(ds): ds.createVariable('SOIL_LEVELS','f4',())[...]=1
    with pytest.raises(MetgridRefusal,match='soil depth dimension, not a scalar'):
        read_met_em(case(tmp_path,mutate=scalar))
