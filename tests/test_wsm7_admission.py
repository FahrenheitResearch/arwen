"""Hold MP24 admission, state transport and pricing to its WSM-family row."""
from types import SimpleNamespace
import pytest
from gpuwm.config import RunConfig, validate_run_config

def cfg(**changes):
    values = dict(nx=4, ny=3, nz=8, dx=1000., dy=1000., ztop=10000.,
                  dt=10., run_seconds=10., moist=True, mp_physics=24,
                  sf_sfclay_physics=91, sf_surface_physics=2,
                  bl_pbl_physics=1, num_soil_layers=4, cu_physics=0)
    values.update(changes)
    return RunConfig(**values)

def test_admitted_and_hail_has_no_number_moment():
    from gpuwm.core.moist import extra_moist_species
    from gpuwm.core.wsm_family import row
    assert validate_run_config(cfg()).mp_physics == 24
    state = SimpleNamespace(qi=object(), qh=object(), qnh=None)
    assert extra_moist_species(state) == ("qi", "qs", "qg", "qh")
    assert row(24).kernel_species == ("qc", "qi", "qr", "qs", "qg", "qh")

def test_own_reflectivity_unit_has_consistent_pricing():
    from gpuwm.core import preflight as p
    from gpuwm.core.refl import SCHEME_NATIVE_REFL_10CM, REFL_10CM_INPUT_SPECIES
    p._hold_reflectivity_rail_equal_to_the_operator()
    assert 24 in SCHEME_NATIVE_REFL_10CM
    assert 24 not in REFL_10CM_INPUT_SPECIES
    assert "wsm7" in p._MICROPHYSICS_KERNEL_MODULES[24]
    slots = p.scratch_slot_registry(cfg())
    assert "mp_hailnc" in slots and "mp_hailncv" in slots
    assert "wsm7_theta" in slots and "refl_10cm" in slots

def test_tiers_preserve_deeper_column_capacity():
    from gpuwm.core import wsm_family as f
    from gpuwm.physics_vertical_contract import WSM7_VERTICAL_LEVEL_BOUNDS
    assert WSM7_VERTICAL_LEVEL_BOUNDS == f.row(24).vertical_level_bounds
    try:
        f.force_level_tier(64)
        assert f.level_tier(24, 45) == 64
        assert f.level_tier(24, 72) == 80
        f.force_level_tier(80)
        assert f.level_tier(24, 45) == 80
        with pytest.raises(ValueError, match="requires"):
            f.level_tier(24, 81)
    finally:
        f.force_level_tier(None)

def test_four_precipitating_classes_are_bound():
    from gpuwm.core.physics_inventory import microphysics_scratch_slots
    from gpuwm.core.wsm_family import row
    assert row(24).precip == ("rain", "snow", "graupel", "hail")
    slots = microphysics_scratch_slots(24)
    assert ("hailnc", "mp_hailnc") in slots
    assert ("hailncv", "mp_hailncv") in slots

def test_stock_package_inventory_keeps_the_seventh_mass():
    from gpuwm.wrf_physics_inventory import stock_wrf_physics_inventory
    inventory = stock_wrf_physics_inventory(24)
    assert inventory.registry_package == "wsm7scheme"
    assert [f.netcdf_name for f in inventory.wrfinput_fields] == [
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL"]


def test_state_transport_and_scratch_lifetimes_cover_hail():
    from gpuwm.core import preflight as p
    c = cfg(km_opt=4, diff_6th_opt=2)
    shapes = p.state_array_shapes(c)
    assert {"qi", "qs", "qg", "qh", "qh0", "effc", "effi", "effs"} <= shapes.keys()
    assert not {"qnh", "qndrop", "nc", "nr", "ni"} & shapes.keys()
    assert "qh" in p.nest_field_kinds(c)
    slots = p.scratch_slot_registry(c)
    assert "smag_rqh" in slots
    assert not {slot for slot in slots if p.scratch_slot_lifetime(slot) is None}
    for name in ("theta", "rho", "pii", "dz", "z8w"):
        assert p.scratch_slot_uses_arena("wsm7_" + name)
    assert not p.scratch_slot_uses_arena("mp_hailnc")


def test_radar_floor_and_compatibility_metadata_include_mp24():
    from gpuwm.da.obsop import clear_air_floor_dbz
    from gpuwm.wrf461_compatibility import MP_OPTIONS
    assert clear_air_floor_dbz(24) == -35.0
    assert 24 in MP_OPTIONS


def test_digest_and_restart_rebuild_the_family_preparation_buffers():
    from gpuwm.io.restart import classify_scratch_slot
    from gpuwm.core.wsm_family import row
    for field in ("theta", "rho", "pii", "dz", "z8w"):
        assert classify_scratch_slot(row(24).name + "_" + field) == "rebuild"
    assert classify_scratch_slot("mp_hailnc") == "serialize"
    assert classify_scratch_slot("mp_hailncv") == "serialize"
    from gpuwm.io.restart import RestartManifestError
    with pytest.raises(RestartManifestError, match="not classified"):
        classify_scratch_slot(row(24).name + "_future_persistent_field")
