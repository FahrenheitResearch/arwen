from gpuwm.config import RunConfig

def run_config(**settings):
    return RunConfig(nx=16, ny=16, nz=8, dx=12000., dy=12000., ztop=10000., dt=30., run_seconds=120., **settings)

def test_ocean_option_is_executable_configuration():
    cfg = run_config(sf_ocean_physics=1)
    assert cfg.oml_hml0 == 50.0
    assert cfg.oml_gamma == 0.14
    assert cfg.oml_relaxation_time == 0.0

import dataclasses
import pytest
from gpuwm.surface_ocean import ocean_fields, ocean_option, validate_ocean_inputs, drop_inert_ocean
from gpuwm.core.physics_inventory import PHYSICS_SLOT_DISPATCH

def test_state_inventory_and_dispatch():
    assert ocean_fields(run_config(sf_ocean_physics=1)) == ("tml", "t0ml", "hml", "h0ml", "huml", "hvml", "tmoml")
    assert PHYSICS_SLOT_DISPATCH["sf_ocean_physics"][1] == "_run_surface_ocean"
    assert ocean_option()["cadence"] == "bldt"
    assert ocean_fields(run_config()) == ()

@pytest.mark.parametrize("depth", [0.0, -1.0])
def test_missing_depth_names_the_breakage(depth):
    with pytest.raises(ValueError, match="mixed-layer depth is uninitialised"):
        validate_ocean_inputs(run_config(sf_ocean_physics=1, oml_hml0=depth), {})

def test_negative_depth_preserves_required_inputs():
    with pytest.raises(ValueError, match="HUML, HVML, TMOML"):
        validate_ocean_inputs(run_config(sf_ocean_physics=1, oml_hml0=-1), {"h0ml": object()})
    validate_ocean_inputs(run_config(sf_ocean_physics=1, oml_hml0=-1),
                         dict.fromkeys(("h0ml", "huml", "hvml", "tmoml"), object()))

def test_off_config_identity_stays_inert():
    values=dataclasses.asdict(run_config()); drop_inert_ocean(values)
    assert not any(name in values for name in ("sf_ocean_physics", "oml_hml0", "oml_gamma", "oml_relaxation_time"))

def test_output_metadata_has_all_seven_fields():
    from gpuwm.io.wrf_output_schema import OCEAN_OUTPUT_FIELDS, SCHEME_OUTPUT_FIELDS
    assert set(OCEAN_OUTPUT_FIELDS) == set(ocean_option()["state_fields"])
    for name, row in OCEAN_OUTPUT_FIELDS.items():
        assert SCHEME_OUTPUT_FIELDS[name] == row
        assert row.netcdf_name == name.upper() and row.dtype == "f4"


def test_wrf_input_axes_are_mass_columns():
    from gpuwm.ingest.wrfinput import WRFINPUT_DIMENSIONS
    for name in ocean_option()["input_fields"]:
        assert WRFINPUT_DIMENSIONS[name] == ("south_north", "west_east")

def test_registry_has_surface_ocean_selector_and_knobs():
    # Registered like the CLM lake: a lower-boundary selector beside the
    # land surface, its three settings read only when it is on.
    from gpuwm.physics_registry import physics_registry
    r = physics_registry()
    assert r["parameters"]["sf_ocean_physics"]["enum"] == [0, 1]
    assert r["parameters"]["sf_ocean_physics"]["default"] == 0
    for name in ("oml_hml0", "oml_gamma", "oml_relaxation_time"):
        assert r["parameters"][name]["component_id"] == "land_surface"
        assert r["parameters"][name]["read_when"] == {"sf_ocean_physics": 1}


def test_real_namelist_import_admits_ocean_key_and_scalars(tmp_path):
    from test_namelist_import import WPS_TEXT, INPUT_TEXT
    from gpuwm.namelist_import import import_namelists
    inp=INPUT_TEXT.replace("&physics", "&physics\n sf_ocean_physics=1, oml_hml0=25., oml_gamma=0.12, oml_relaxation_time=86400.,",1)
    wps=tmp_path/"namelist.wps"; path=tmp_path/"namelist.input"
    wps.write_text(WPS_TEXT); path.write_text(inp)
    text,_=import_namelists(wps,path)
    assert "sf_ocean_physics = 1" in text
    assert "oml_hml0 = 25" in text and "oml_relaxation_time = 86400" in text


def test_word_gate_detects_a_sign_bit_change_at_zero_ulp():
    import numpy as np
    from tools.oml_wrf461_oracle.validate import metrics
    row=metrics(np.array([-0.],dtype=np.float32),np.array([0.],dtype=np.float32))
    assert row["max_ulp"] == 0 and row["changed_words"] == 1
