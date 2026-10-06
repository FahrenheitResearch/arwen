"""Named roster requests select the actual member's prepared surface arm."""
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.request import EnsembleRequest
from gpuwm.ensemble.production import PreparedEnsembleSession
from gpuwm.ensemble.seeds import member_seed


def variants():
    return [{"name": "control"},
            {"name": "soil-dry", "surface": {"soil_moisture_scale": 0.8}},
            {"name": "noah-warm", "physics": {"sf_surface_physics": 2, "num_soil_layers": 4},
             "surface": {"sst_offset_k": 1}}]


def test_named_roster_infers_recipe_count_and_records_existing_selectors_only():
    source = variants()
    request = EnsembleRequest.from_mapping({"base_seed": 73, "member_variants": source})
    assert request.members == 3 and request.recipe == "member-roster"
    receipt = request.receipt()
    assert receipt["member_variants"][2]["physics"] == {"sf_surface_physics": 2, "num_soil_layers": 4}
    assert receipt["member_variants"][0]["surface"] == {"soil_moisture_scale": 1, "sst_offset_k": 0}
    source[1]["surface"]["soil_moisture_scale"] = 0.5
    assert request.member_variants[1]["surface"]["soil_moisture_scale"] != 0.5
    receipt["member_variants"][2]["surface"]["sst_offset_k"] = 5
    assert request.member_variants[2]["surface"]["sst_offset_k"] == 1


@pytest.mark.parametrize("options, words", [
    ({"members": 3, "recipe": "member-roster"}, "one named record"),
    ({"members": 2, "recipe": "member-roster", "member_variants": variants()}, "one named record"),
    ({"members": 3, "recipe": "time-lagged", "member_variants": variants()}, "belong to recipe"),
    ({"members": 3, "recipe": "member-roster", "member_variants": variants(),
      "perturbation": {"kind": "surface-state", "sst_offset_k": 1}}, "competing surface states"),
    ({"members": 3, "member_variants": True}, "list of named"),
    ({"member_variants": {"name": "control"}}, "list of named"),
])
def test_unbound_roster_shapes_refuse_before_source_or_gpu(options, words):
    with pytest.raises(ValueError, match=words):
        EnsembleRequest.from_mapping(options)


def test_mutated_roster_is_revalidated_before_execution():
    request = EnsembleRequest.from_mapping({"member_variants": variants()})
    request.member_variants[2]["physics"]["num_soil_layers"] = 6
    with pytest.raises(ValueError, match="wrong land column"):
        EnsembleRequest.from_mapping(request)


def test_ordinary_and_existing_recipe_receipts_do_not_gain_roster_keys():
    assert "member_variants" not in EnsembleRequest(1).receipt()
    assert "member_variants" not in EnsembleRequest(4, recipe="time-lagged").receipt()


def test_session_requires_bound_member_inputs_for_named_land_arms(tmp_path):
    request = EnsembleRequest.from_mapping({"member_variants": variants()})
    with pytest.raises(ValueError, match="no bound member inputs.*one prepared land column"):
        PreparedEnsembleSession(request, output_directory=tmp_path)


def test_roster_control_keeps_the_ordinary_initialization_callback_inactive(tmp_path):
    session = PreparedEnsembleSession({"member_variants": variants()}, output_directory=tmp_path,
                                     input_provider=lambda **unused: None)
    assert session._initialization_callback(0) is None


def test_session_selects_each_members_surface_descriptor_and_original_seed(tmp_path, monkeypatch):
    from gpuwm.ensemble import surface_recipe
    selected = []
    def factory(request, **identity):
        selected.append((request.perturbation, identity))
        identity["record"]({"member_id": identity["member_id"], "seed": identity["seed"], "domains": []})
        return object()
    monkeypatch.setattr(surface_recipe, "surface_initialization_callback", factory)
    session = PreparedEnsembleSession({"base_seed": 73, "member_variants": variants()},
        output_directory=tmp_path, input_provider=lambda **unused: None)
    session._initialization_callback(1)
    session._initialization_callback(2, seed=901)
    assert selected[0][0]["soil_moisture_scale"] == session.request.member_variants[1]["surface"]["soil_moisture_scale"]
    assert selected[0][0]["sst_offset_k"] == 0
    assert selected[0][1]["member_id"] == 1 and selected[0][1]["seed"] == member_seed(73, 1)
    assert selected[1][0]["sst_offset_k"] == 1 and selected[1][1]["seed"] == 901
    assert session._surface_receipts[1]["seed"] == member_seed(73, 1)
    assert session._surface_receipts[2]["seed"] == 901
    assert session._surface_receipts[2]["member_variant"] == {
        "name": "noah-warm", "member_id": 2, "seed": 901,
        "physics": {"sf_surface_physics": 2, "num_soil_layers": 4},
        "surface": {"soil_moisture_scale": 1.0, "sst_offset_k": 1.0}}


def test_selected_prepared_member_seed_remains_the_source_authority(tmp_path, monkeypatch):
    from gpuwm.ensemble import surface_recipe
    selected = []
    monkeypatch.setattr(surface_recipe, "surface_initialization_callback",
                       lambda request, **identity: selected.append(identity) or object())
    session = PreparedEnsembleSession({"base_seed": 73, "member_variants": variants()},
        output_directory=tmp_path, input_provider=lambda **unused: None)
    session._initialization_callback(2, prepared_member=SimpleNamespace(seed=905))
    assert selected[0]["seed"] == 905


def test_unity_control_receipt_records_variant_without_a_state_owner(tmp_path):
    session = PreparedEnsembleSession({"base_seed": 73, "member_variants": variants()},
        output_directory=tmp_path, input_provider=lambda **unused: None)
    assert session._initialization_callback(0) is None
    receipt = session._variant_receipt(0)
    assert receipt["name"] == "control" and receipt["seed"] == member_seed(73, 0)
    assert receipt["physics"] == {} and receipt["surface"] == {"soil_moisture_scale": 1.0, "sst_offset_k": 0.0}
    assert not session._surface_receipts


def test_run_plan_recognizes_implicit_named_roster_before_preparation():
    from gpuwm.runplan import _plan_recipe
    plan = SimpleNamespace(run_options={})
    assert _plan_recipe(plan, {"ensemble": {"member_variants": variants()}}) == "member-roster"


def test_roster_receipt_records_loaded_control_and_overridden_land_layouts(tmp_path):
    session = PreparedEnsembleSession({"base_seed": 73, "member_variants": variants()},
        output_directory=tmp_path, input_provider=lambda **unused: None)
    def inputs(scheme, layers):
        return SimpleNamespace(experiment=SimpleNamespace(domains=tuple(
            SimpleNamespace(grid_id=grid, run=SimpleNamespace(sf_surface_physics=scheme, num_soil_layers=layers))
            for grid in (1, 2))))
    session._remember_member_land_layouts({0: inputs(3, 6), 1: inputs(3, 6), 2: inputs(2, 4)})
    assert session._variant_receipt(0)["resolved_land"] == [
        {"grid_id": grid, "sf_surface_physics": 3, "num_soil_layers": 6} for grid in (1, 2)]
    assert session._variant_receipt(2)["resolved_land"] == [
        {"grid_id": grid, "sf_surface_physics": 2, "num_soil_layers": 4} for grid in (1, 2)]


def test_roster_surface_composes_existing_binding_with_the_same_authoritative_seed(tmp_path, monkeypatch):
    from gpuwm.ensemble import surface_recipe
    bindings = []
    provider = SimpleNamespace(bind_model=lambda **identity: bindings.append(identity))
    def surface_factory(request, *, previous, record, member_id, seed, **unused):
        def initialize(**context):
            previous(**context)
            record({"member_id": member_id, "seed": seed, "domains": []})
        return initialize
    monkeypatch.setattr(surface_recipe, "surface_initialization_callback", surface_factory)
    session = PreparedEnsembleSession({"base_seed": 73, "member_variants": variants()},
        output_directory=tmp_path, input_provider=lambda **unused: None, stochastic_provider=provider)
    selected = SimpleNamespace(seed=905)
    model = object()
    session._initialization_callback(2, prepared_member=selected)(model=model)
    assert bindings[0]["model"] is model and bindings[0]["member_id"] == 2
    assert bindings[0]["seed"] == session._surface_receipts[2]["seed"] == 905
    assert bindings[0]["prepared_member"] is selected


def test_session_refuses_provider_that_ignores_member_land_selectors(tmp_path):
    session = PreparedEnsembleSession({"member_variants": variants()},
        output_directory=tmp_path, input_provider=lambda **unused: None)
    wrong = SimpleNamespace(experiment=SimpleNamespace(domains=(
        SimpleNamespace(grid_id=2, run=SimpleNamespace(sf_surface_physics=3, num_soil_layers=6)),)))
    with pytest.raises(ValueError, match="member 2 domain 2 requests sf_surface_physics=2.*wrong land column"):
        session._remember_member_land_layouts({2: wrong})
