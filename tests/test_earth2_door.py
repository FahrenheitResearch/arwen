"""Fast shared job-form/CLI contract proof; no weights, network or GPU."""
import json

import jsonschema
import pytest

from gpuwm.earth2_door import SCHEMA_FILE, build_spec, canonical_json, load_spec


def test_canonical_spec_and_unknown_names(tmp_path, monkeypatch):
    spec = build_spec(model="fourcastnet3", cycle="2024-09-24T12", hours=24,
                      storm="auto", nest="2.5km", mix="gefs:18,fourcastnet3:18")
    assert spec["model"] == "fcn3"
    assert spec["cycle"] == "2024-09-24T12:00:00Z"
    assert spec["nest_km"] == 2.5
    assert spec["ensemble"] == {"members": 36, "mix": [
        {"source": "fcn3", "members": 18}, {"source": "gefs", "members": 18}]}
    document = canonical_json(spec)
    assert document == json.dumps(spec, sort_keys=True, indent=2) + "\n"
    jsonschema.Draft202012Validator(json.loads(SCHEMA_FILE.read_text())).validate(spec)
    saved = tmp_path / "spec.json"
    saved.write_text(document)
    assert canonical_json(load_spec(saved)) == document
    assert "card" not in spec
    card_spec = build_spec(model="fcn3", cycle="2024-09-24T12", card="16gb")
    assert card_spec["card"] == "16gb"
    saved.write_text(canonical_json(card_spec))
    assert load_spec(saved) == card_spec
    for options, match in ((dict(model="unknown-parent"), "valid names"),
                           (dict(mix="unknown-source:2"), "valid names"),
                           (dict(diagnose=["unknown-diagnostic"]), "valid names"),
                           (dict(ic_source="unknown-ic"), "valid names"),
                           (dict(card="unknown-card"), "card"),
                           (dict(hours=0), "positive integer")):
        with pytest.raises(ValueError, match=match):
            build_spec(**(dict(model="fcn3", cycle="2024-09-24T12") | options))
    spec["table"]["sha256"] = "0" * 64
    saved.write_text(canonical_json(spec))
    with pytest.raises(ValueError, match="SHA256 differs"):
        load_spec(saved)

    # A new table row, alias and cadence require no new builder code.
    from gpuwm import earth2_door, earth2_run
    table = earth2_run.model_table()
    table["models"]["table-added-model"] = dict(table["models"]["fcn3"], aliases=["added-alias"], step_hours=3)
    authority = tmp_path / "earth2-models.v1.json"
    authority.write_text(json.dumps(table))
    monkeypatch.setattr(earth2_run, "TABLE", authority)
    monkeypatch.setattr(earth2_door, "TABLE", authority)
    added = build_spec(model="added-alias", cycle="2024-09-24T12", hours=9)
    assert added["model"] == "table-added-model"
    assert added["hours"] == 9
    assert build_spec(model="fcn3", cycle="2024-09-24T12", hours=1)["hours"] == 1
    assert build_spec(model="fcn3", cycle="2024-09-24T12", diagnose="windgust")["diagnose"] == ["windgust"]
