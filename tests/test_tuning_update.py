"""Training-only proposal provenance, categorical isolation and native EKI."""
import os
from pathlib import Path

import pytest

from gpuwm import tuning, tuning_update


def fixtures(tmp_path):
    packet = tmp_path / "packet"
    table = [
        {"name": "land_model", "kind": "categorical", "values": [2, 3, 4], "default": 3},
        {"name": "bl_mynn_mixlength", "kind": "categorical", "values": [2, 3], "default": 2},
        {"name": "urban_option", "kind": "categorical", "values": [0, 1], "default": 0},
        {"name": "soil_moisture_scale", "kind": "continuous", "min": 0.5, "max": 1.5, "default": 1, "schemes": [2, 3, 4]},
        {"name": "canopy_resistance_scale", "kind": "continuous", "min": 0.5, "max": 1.5, "default": 1, "schemes": [3]},
    ]
    members = []
    for identity, land, soil in (("current", 3, 1), ("setting-001", 2, 0.6),
                                 ("setting-002", 2, 1.4), ("setting-003", 4, 1)):
        parameters = {"land_model": land, "bl_mynn_mixlength": 2, "urban_option": 0,
                      "soil_moisture_scale": soil, "canopy_resistance_scale": 1}
        active = {key: value for key, value in parameters.items() if key != "canopy_resistance_scale" or land == 3}
        members.append({"id": identity, "parameters": parameters, "active_parameters": active,
                        "physics": {"sf_urban_physics": 0}})
    round_ = {"schema": tuning.SCHEMA, "product": "city", "train_days": ["2024-05-01", "2024-05-02"],
              "heldout_days": ["2024-05-03", "2024-05-04"], "members": members,
              "parameter_space": table, "baseline_member": "current"}
    tuning.write_json(packet / "round.json", round_)
    tuning.seal_packet(packet)
    # Feature order deliberately differs from training rank and packet order.
    training = {
        "decision": {"winner": "setting-001", "selection_metric": "mean_daily_rmse_training_only",
                     "heldout": None, "heldout_days": [], "heldout_periods": []},
        "ranking": [{"member": identity, "training_score": score, "heldout_score": None, "heldout": None}
                    for identity, score in (("setting-001", 1), ("current", 2), ("setting-002", 3), ("setting-003", 4))],
        "heldout_spread": None,
        "metrics": [], "masks": [],
        "train_features": {"method": "training_only_station_day_period_mean", "units": "same_as_input",
            "members": ["setting-003", "setting-002", "setting-001"],
            "labels": [{"day": "2024-05-01", "station": "station", "period": "day", "count": 12}],
            "predictions": [[281], [282], [280]], "observations": [281]},
    }
    selection = tmp_path / "selection.json"
    tuning.write_json(selection, {"round_sha256": tuning.digest(packet / "round.json"), "training": training})
    return packet, selection, round_, training


def test_ranked_proposals_preserve_settings_and_do_not_ship(tmp_path):
    packet, selection, round_, _ = fixtures(tmp_path)
    output = tmp_path / "proposal.json"
    result = tuning_update.propose(packet, selection, "unused-binary", output, keep=2)
    assert [row["source_member"] for row in result["members"]] == ["setting-001", "setting-002"]
    originals = {row["id"]: row["parameters"] for row in round_["members"]}
    assert all(row["parameters"] == originals[row["source_member"]] for row in result["members"])
    assert result["requires_new_forecasts"] and result["requires_new_unseen_holdout"]
    assert result["source_selection_sha256"] == tuning.digest(selection)
    with pytest.raises(ValueError, match="already exists"):
        tuning_update.propose(packet, selection, "unused-binary", output)


@pytest.mark.parametrize("mutate", [
    lambda training: training["decision"].update(heldout={"rmse_gain": 3}),
    lambda training: training["train_features"]["labels"][0].update(day="2024-05-03"),
    lambda training: training["ranking"][0].update(heldout_score=1),
])
def test_heldout_input_is_rejected_before_numerical_update(tmp_path, mutate, monkeypatch):
    packet, selection, _, training = fixtures(tmp_path)
    mutate(training)
    tuning.write_json(selection, {"round_sha256": tuning.digest(packet / "round.json"), "training": training})
    def unexpected(*args):
        raise AssertionError("held-out input reached numerical update")
    monkeypatch.setattr(tuning_update, "numerical", unexpected)
    with pytest.raises(ValueError, match="held-out"):
        tuning_update.propose(packet, selection, "unused-binary", tmp_path / "proposal.json", method="eki")


def test_seal_selection_and_output_containment_are_verified(tmp_path):
    packet, selection, _, _ = fixtures(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        tuning_update.propose(packet, selection, "unused", packet / "proposal.json")
    altered = tuning.read_json(selection)
    altered["round_sha256"] = "0" * 64
    tuning.write_json(selection, altered)
    with pytest.raises(ValueError, match="different sealed round"):
        tuning_update.propose(packet, selection, "unused", tmp_path / "proposal.json")
    (packet / "round.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed"):
        tuning_update.propose(packet, selection, "unused", tmp_path / "proposal.json")


def test_packet_path_escape_is_rejected(tmp_path):
    packet, selection, _, _ = fixtures(tmp_path)
    seal = tuning.read_json(packet / "packet-sha256.json")
    seal["files"]["../selection.json"] = tuning.digest(selection)
    tuning.write_json(packet / "packet-sha256.json", seal)
    with pytest.raises(ValueError, match="escape"):
        tuning_update.propose(packet, selection, "unused", tmp_path / "proposal.json")


def test_native_eki_alignment_active_parameters_and_singleton(tmp_path):
    binary = os.environ.get("TUNING_HARNESS_NUMERICS")
    if not binary or not Path(binary).is_file():
        pytest.skip("set TUNING_HARNESS_NUMERICS to the built Rust executable")
    packet, selection, round_, _ = fixtures(tmp_path)
    result = tuning_update.propose(packet, selection, binary, tmp_path / "proposal.json", method="eki")
    original = {row["id"]: row["parameters"] for row in round_["members"]}
    proposals = {row["source_member"]: row["parameters"] for row in result["members"]}
    assert proposals["setting-001"]["soil_moisture_scale"] > original["setting-001"]["soil_moisture_scale"]
    assert proposals["setting-002"]["soil_moisture_scale"] < original["setting-002"]["soil_moisture_scale"]
    for identity, parameters in proposals.items():
        for name in ("land_model", "bl_mynn_mixlength", "urban_option", "canopy_resistance_scale"):
            assert parameters[name] == original[identity][name]
    assert proposals["setting-003"] == original["setting-003"]
    assert result["groups"][0]["continuous_parameters"] == ["soil_moisture_scale"]
    assert result["groups"][0]["numerical_receipt"]["solve_dimension"] == 2
    assert result["groups"][1]["status"] == "retained_ranked_no_update"


def test_reference_product_proposals_are_refused(tmp_path):
    packet, selection, round_, _ = fixtures(tmp_path)
    round_["product"] = "hrrr-exact"
    tuning.write_json(packet / "round.json", round_)
    tuning.seal_packet(packet)
    with pytest.raises(ValueError, match="city or research"):
        tuning_update.propose(packet, selection, "unused", tmp_path / "proposal.json")
