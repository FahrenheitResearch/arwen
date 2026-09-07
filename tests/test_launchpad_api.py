"""The visual client authors through the real wizard, never a second fitter."""
import json
from pathlib import Path

import pytest

from gpuwm import launchpad_api as api


def values(tmp_path):
    return dict(directory=str(tmp_path), name="launchpad_proof", lat=35.3, lon=-97.5,
                width=180, height=150, dx=12, nz=49, hours=6, history=720,
                chain="", tiles="off", vram=16, profile="", source="gfs",
                cycle="2026-09-05T00")


def test_catalog_uses_registered_sources_and_profiles():
    from gpuwm.source_adapters import source_adapters
    from gpuwm.physics_menu import shipped_profiles
    result=api.catalog()
    assert {s["id"] for s in result["sources"]}=={s.source_id for s in source_adapters()}
    assert {p["profile_id"] for p in result["profiles"]}==set(shipped_profiles())


def test_real_wizard_creates_editable_draft_without_weather_job(tmp_path):
    from gpuwm.experiment import load_experiment
    result=api.create_draft(values(tmp_path))
    path=Path(result["path"])
    exp=load_experiment(path)
    assert exp.name=="launchpad_proof" and exp.run_seconds==21600
    assert exp.domains[0].run.dx==12000
    assert exp.domains[0].run.nz==49
    assert (exp.domains[0].run.nx,exp.domains[0].run.ny)==(18,16)
    assert result["requested_region"]=={"width_km":180.,"height_km":150.}
    assert "requested_region" not in result["inspection"]["document"]
    assert path.with_name("forecast.namelist.wps").is_file()
    assert "[fetch]" in result["text"]
    assert not (tmp_path/"forcing").exists()
    assert not list(tmp_path.rglob("wrfout*"))


@pytest.mark.parametrize("key,value", [("lat", float("nan")),("width",True),("dx",0)])
def test_invalid_visual_values_refuse_before_authoring(tmp_path,key,value):
    request=values(tmp_path); request[key]=value
    with pytest.raises(ValueError):api.create_draft(request)
    assert not (tmp_path/"forecast.toml").exists()


def test_inspection_retains_unknown_values_without_preset_gate():
    text="# keep this in the editor\n[shared]\nfuture_switch=919\n[[domain]]\nnx=80\n"
    result=api.inspect_text(text)
    assert result["document"]["shared"]["future_switch"]==919


def test_external_plan_rejects_unknown_door(tmp_path):
    with pytest.raises(ValueError,match="Choose WPS"):
        api.external_plan(dict(directory=str(tmp_path),kind="invented"))

@pytest.mark.parametrize("key", ["hours","nz"])
def test_fractional_integer_controls_never_silently_change_request(tmp_path,key):
    request=values(tmp_path);request[key]=6.5
    with pytest.raises(ValueError,match="whole number"):api.create_draft(request)
    assert not (tmp_path/"region.geojson").exists()


def test_slider_convenience_ranges_do_not_limit_exact_values():
    assert api.number({"hours":1001},"hours",integral=True)==1001
    assert api.number({"dx":.005},"dx",positive=True)==.005
    assert api.number({"width":.5},"width",positive=True)==.5


def test_inspection_returns_shared_projected_footprints_and_exact_clock(tmp_path):
    result=api.create_draft(values(tmp_path))
    inspected=result["inspection"]
    assert "validation_error" not in inspected
    assert len(inspected["footprints"])==1
    assert len(inspected["footprints"][0])==132
    assert inspected["timesteps"]==["60"]


def test_each_visible_scientific_control_uses_real_schema(tmp_path):
    import copy
    from gpuwm.experiment import build_experiment_from_config_tables, _DOMAIN_KEYS
    result=api.create_draft(values(tmp_path))
    raw=result["inspection"]["document"]
    raw["shared"]["wrf_rrtmg_compatibility"]="none"
    # Each value is independently validated by the exact file-backed loader's
    # shared table boundary. Compatibility failures remain genuine errors.
    controls={"nx":72,"ny":60,"dx":12500.,"time_step":30,
              "time_step_fract_num":0,"time_step_fract_den":2,
              "history_interval_s":1440,"mp_physics":6,"cu_physics":0,
              "ra_lw_physics":1,"ra_sw_physics":1,"ra_rrtmg_variant":"rrtmg_legacy",
              "wrf_rrtmg_compatibility":"none",
              "bl_pbl_physics":1,"sf_surface_physics":2,"num_soil_layers":4,"sf_sfclay_physics":1,"radt":12.}
    for key,value in controls.items():
        edited=copy.deepcopy(raw)
        target=edited["domain"][0] if key in _DOMAIN_KEYS else edited["shared"]
        target[key]=value
        exp=build_experiment_from_config_tables(edited,source=str(result["path"]),base_dir=tmp_path)
        if key in {"nx","ny","dx","mp_physics","cu_physics","ra_lw_physics","ra_sw_physics","ra_rrtmg_variant","bl_pbl_physics","sf_surface_physics","sf_sfclay_physics","radt","wrf_rrtmg_compatibility","num_soil_layers"}:
            assert getattr(exp.domains[0].run,key)==value,key
        else:assert getattr(exp.domains[0],key)==value,key
    # Exact fractional seconds are all preserved, never rounded to integer dt.
    edited=copy.deepcopy(raw);edited["domain"][0].update(time_step=7,time_step_fract_num=1,time_step_fract_den=2)
    exp=build_experiment_from_config_tables(edited,source=str(result["path"]),base_dir=tmp_path)
    from fractions import Fraction
    assert exp.dt_exact(1)==Fraction(15,2)


def test_readout_uses_resolved_sibling_geometry_and_explicit_spacing(tmp_path):
    import copy
    from gpuwm.experiment_document import render_experiment_document
    request=values(tmp_path);request.update(width=2000,height=1800)
    made=api.create_draft(request)
    raw=copy.deepcopy(made["inspection"]["document"])
    raw.pop("fetch",None)
    raw.pop("tiles",None)
    root=raw["domain"][0]
    root.update(nx=120,ny=120,dx=12000.)
    children=[]
    for gid,ratio,dx in [(2,4,3000.),(3,3,4000.)]:
        child={"grid_id":gid,"parent_id":1,"parent_grid_ratio":ratio,
               "parent_time_step_ratio":ratio,"nx":48,"ny":48,
               "i_parent_start":20,"j_parent_start":20,"dx":dx,
               "history_interval_s":720.}
        children.append(child)
    raw["domain"]=[root,*children]
    def inspect():
        return api.inspect_text(render_experiment_document(raw),tmp_path/"siblings.toml")
    result=inspect()
    assert "validation_error" not in result,result
    assert result["grid_geometry"]==[
        {"grid_id":1,"nx":120,"ny":120,"dx":12000.,"dy":12000.},
        {"grid_id":2,"nx":48,"ny":48,"dx":3000.,"dy":3000.},
        {"grid_id":3,"nx":48,"ny":48,"dx":4000.,"dy":4000.}]
    # An edited root propagates through the actual loader; neither row order
    # nor a previously displayed spacing is an authority for the geometry.
    root["dx"]=24000.;children[0]["dx"]=6000.;children[1]["dx"]=8000.
    assert [d["dx"] for d in inspect()["grid_geometry"]]==[24000.,6000.,8000.]
    children[1]["dx"]=4000.
    rejected=inspect();assert "validation_error" in rejected
    assert "grid_geometry" not in rejected
    children[1]["dx"]=8000.;root["dy"]=12000.
    rejected=inspect();assert "isotropic" in rejected["validation_error"]
    assert "grid_geometry" not in rejected
