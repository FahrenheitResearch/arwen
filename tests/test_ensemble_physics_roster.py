"""Physics arms use ordinary selectors, validation and prepared owners."""
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.member_variants import normalize_member_variants, variant_configuration
from gpuwm.ensemble.production import PreparedEnsembleSession


def test_named_physics_arm_overrides_explicit_children_without_mutating_control():
    raw = {"shared": {"mp_physics": 28, "bl_pbl_physics": 5},
           "domain": [{"grid_id": 1, "mp_physics": 28},
                      {"grid_id": 2, "mp_physics": 28, "bl_pbl_physics": 5}]}
    arm = normalize_member_variants([
        {"name": "control"}, {"name": "other", "physics": {
            "mp_physics": 8, "bl_pbl_physics": 1, "sf_sfclay_physics": 1,
            "ra_lw_physics": 1, "ra_sw_physics": 1,
            "sf_surface_physics": 2, "num_soil_layers": 4}}], 2)[1]
    selected = variant_configuration(raw, arm)
    assert raw["domain"][1]["mp_physics"] == 28
    assert selected["shared"]["ra_lw_physics"] == 1
    assert selected["domain"][0]["mp_physics"] == 8
    assert selected["domain"][1]["bl_pbl_physics"] == 1


@pytest.mark.parametrize("key,value", [("nx", 50), ("eta_levels", [1, 0]),
    ("starting_time_step", 30), ("spp_pbl", 1), ("inflow_perturbation", True)])
def test_physics_arm_cannot_change_prepared_grid_or_bypass_draw_admission(key, value):
    with pytest.raises(ValueError, match="wrong prepared grid or bypass a refusal"):
        normalize_member_variants([{"name": "bad", "physics": {key: value}}], 1)


def test_noahmp_uses_single_run_soil_geometry_instead_of_a_second_land_menu():
    arm = normalize_member_variants([{"name": "noahmp", "physics": {
        "sf_surface_physics": 4, "num_soil_layers": 4}}], 1)[0]
    assert arm["physics"]["sf_surface_physics"] == 4


def test_provider_cannot_substitute_another_microphysics_and_receipt_records_actual_choice(tmp_path):
    session = PreparedEnsembleSession({"member_variants": [{"name": "other", "physics": {
        "mp_physics": "8", "bl_pbl_physics": 1}}]},
        output_directory=tmp_path, input_provider=lambda **unused: None)
    run = SimpleNamespace(sf_surface_physics=3, num_soil_layers=6,
                          mp_physics=28, bl_pbl_physics=1)
    inputs = {0: SimpleNamespace(experiment=SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, run=run),)))}
    with pytest.raises(ValueError, match="bound mp_physics=28.*wrong physics selection"):
        session._remember_member_land_layouts(inputs)
    run.mp_physics = 8
    session._remember_member_land_layouts(inputs)
    assert session._variant_receipt(0)["resolved_physics"] == [{
        "grid_id": 1, "mp_physics": 8, "bl_pbl_physics": 1}]


def test_old_land_receipt_has_no_new_physics_metadata(tmp_path):
    session = PreparedEnsembleSession({"member_variants": [{"name": "old", "physics": {
        "sf_surface_physics": 3, "num_soil_layers": 6}}]},
        output_directory=tmp_path, input_provider=lambda **unused: None)
    inputs = {0: SimpleNamespace(experiment=SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, run=SimpleNamespace(sf_surface_physics=3,num_soil_layers=6)),)))}
    session._remember_member_land_layouts(inputs)
    assert "resolved_physics" not in session._variant_receipt(0)
