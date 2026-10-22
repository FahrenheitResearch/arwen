"""Containment and immutable packet checks for opt-in tuning orchestration."""
import json
from pathlib import Path

import pytest

from gpuwm import tuning


def spec():
    return {"product": "city", "train_days": ["2026-01-01", "2026-01-02"],
            "heldout_days": ["2026-01-03", "2026-01-04"], "member_count": 24,
            "hours": 24, "cycle_hour": 0, "parameter_space": []}


def test_reference_product_and_leaking_split_refuse():
    value = spec()
    value["product"] = "hrrr-exact"
    with pytest.raises(ValueError, match="reference-code"):
        tuning.validate_spec(value)
    value = spec()
    value["heldout_days"][0] = value["train_days"][0]
    with pytest.raises(ValueError, match="disjoint"):
        tuning.validate_spec(value)


def test_packet_tampering_is_detected(tmp_path):
    tuning.write_json(tmp_path / "round.json", spec())
    tuning.seal_packet(tmp_path)
    assert tuning.verify_packet(tmp_path)["product"] == "city"
    (tmp_path / "round.json").write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        tuning.verify_packet(tmp_path)


def test_selector_update_does_not_touch_domain_geometry():
    text = '[shared]\nnum_soil_layers = 6\nbl_mynn_mixlength = 1\n\n[[domain]]\nnx = 288\ndx = 2250.0\n'
    changed = tuning.patch_table(text, "shared", {"num_soil_layers": 4, "bl_mynn_mixlength": 2})
    assert changed.endswith(text[text.index("[[domain]]"):])
    assert "num_soil_layers = 4" in changed
    assert "bl_mynn_mixlength = 2" in changed


def test_atomic_receipt_has_real_newline_and_no_nonfinite(tmp_path):
    destination = tmp_path / "receipt.json"
    tuning.write_json(destination, {"decision": "pending"})
    assert destination.read_bytes().endswith(b"\n")
    assert not destination.with_suffix(".json.partial").exists()
    assert json.loads(destination.read_text())["decision"] == "pending"
    with pytest.raises(ValueError):
        tuning.write_json(destination, {"score": float("nan")})


def test_inactive_ruc_urban_controls_are_explicit():
    rows = [{"name": "urban_fraction_scale", "default": 1, "schemes": [2, 4], "requires_urban": True}]
    parameters = {"land_model": 3, "bl_mynn_mixlength": 1, "urban_option": 1,
                  "soil_moisture_scale": .8, "sst_offset_k": .5,
                  "urban_fraction_scale": .7, "urban_heat_capacity_scale": 1,
                  "urban_conductivity_scale": 1}
    member = tuning.member_descriptor("test", parameters, 17, "city", rows)
    assert member["physics"]["sf_urban_physics"] == 0
    assert member["inactive_parameters"] == ["urban_fraction_scale"]
    assert member["urban"]["fraction_scale"] == 1
