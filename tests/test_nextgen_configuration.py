"""The arbitrary-realization boundary and licence hold, without GPU work."""
import copy

import pytest

from gpuwm.hydro.config import configure, materialize_reference, resolve_configuration


def test_arbitrary_row_preserves_composition_parameters_and_order():
    row = {
        "id": "regional-research",
        "admission": {"status": "permitted", "licence": "Apache-2.0"},
        "sources": [{"revision": "research-pin", "path": "composition"}],
        "settings": {
            "interval_seconds": 900,
            "stages": [
                {"name": "runoff", "factory": "gpuwm.hydro.experimental:prepare",
                 "options": {"parameter_binding": "fabric.soil_k", "coefficient": 1.25}},
                {"name": "routing", "factory": "gpuwm.hydro.network:prepare"},
            ],
            "routing": {"nudging": False, "method": "research-method"},
        },
    }
    table = {"schema": "gpuwm-nextgen-realizations-v1", "realizations": [row]}
    original = copy.deepcopy(table)
    selected = resolve_configuration(row["id"], table=table)
    with pytest.raises(TypeError):
        selected.row["settings"]["stages"][0]["options"]["coefficient"] = 9
    settings = {"realization": row["id"], "routing": {"custom_boundary": "boundary-pin"}}
    resolved = configure(settings, table=table)
    assert [stage["name"] for stage in resolved["stages"]] == ["runoff", "routing"]
    assert resolved["stages"][0]["options"]["parameter_binding"] == "fabric.soil_k"
    assert resolved["routing"] == {"nudging": False, "method": "research-method",
                                    "custom_boundary": "boundary-pin"}
    assert resolved["numerics"] == "fast"
    assert configure(resolved, table=table) == resolved
    strict = configure(dict(settings, numerics="strict"), table=table)
    assert strict["realization_identity"]["settings_sha256"] != resolved["realization_identity"]["settings_sha256"]
    resolved["stages"][0]["options"]["coefficient"] = 9
    assert table == original
    assert selected.as_dict() == row


def test_named_licence_hold_precedes_factory_preparation():
    table = {"schema": "gpuwm-nextgen-realizations-v1", "realizations": [{
        "id": "held-module",
        "admission": {"status": "blocked", "reason": "upstream source requires a commercial royalty agreement"},
        "sources": [], "settings": {"stages": []},
    }]}
    with pytest.raises(ValueError, match="commercial royalty agreement"):
        configure({"realization": "held-module"}, table=table)


def test_runtime_resource_binding_is_single_pass_and_requires_fresh_rebinding():
    settings = {"realization": "lower-colorado-hyfeatures",
                "settings_bindings": {"fabric_manifest": "/data/${literal}/manifest.json"}}
    original = copy.deepcopy(settings)
    resolved = configure(settings)
    assert settings == original
    assert [(s["name"], s["phase"]) for s in resolved["stages"]] == [
        ("network", "prepare"), ("reservoirs", "prepare"), ("routing", "after")]
    assert resolved["stages"][0]["options"]["manifest"] == "/data/${literal}/manifest.json"
    assert resolved["stages"][2]["options"]["initial_state_component"] == "network"
    # The native short-timestep flag and output labels belong to captured
    # manifest metadata. An omitted captured flag previously selected false in the factory.
    assert "assume_short_ts" not in resolved["stages"][2]["options"]
    assert "output_reach_ids" not in resolved["stages"][2]["options"]
    assert configure(resolved) == resolved
    changed = copy.deepcopy(resolved)
    changed["settings_bindings"]["fabric_manifest"] = "/data/other.json"
    with pytest.raises(ValueError, match="stale stage paths"):
        configure(changed)
    settings["settings_bindings"]["fabric_manifest"] = "/data/other.json"
    rebound = configure(settings)
    assert rebound["stages"][0]["options"]["manifest"] == "/data/other.json"
    assert rebound["realization_identity"] != resolved["realization_identity"]
    for bindings in ({}, {"fabric_manifest": "/a", "typo": "/b"}):
        with pytest.raises(ValueError, match="settings bindings"):
            configure({"realization": settings["realization"], "settings_bindings": bindings})


def test_portable_reference_binding_preserves_native_tokens_and_rebinds_identity():
    row = {
        "id": "portable-research", "admission": {"status": "permitted"},
        "sources": [], "reference_parameters": ["root"],
        "reference": {"realization": {"modules": [
            {"init_config": "${root}/{{id}}.ini", "coefficient": 0.0},
            {"variables_names_map": {"input": "output"}},
        ]}},
    }
    table = {"schema": "gpuwm-nextgen-realizations-v1", "realizations": [row]}
    settings = {"realization": row["id"], "reference_bindings": {"root": "/data/a"}}
    resolved = configure(settings, table=table)
    assert resolved["configuration_reference"]["realization"]["modules"] == [
        {"init_config": "/data/a/{{id}}.ini", "coefficient": 0.0},
        {"variables_names_map": {"input": "output"}},
    ]
    assert configure(resolved, table=table) == resolved
    resolved["reference_bindings"]["root"] = "/data/b"
    rebound = configure(resolved, table=table)
    assert rebound["configuration_reference"]["realization"]["modules"][0]["init_config"] == "/data/b/{{id}}.ini"
    assert rebound["realization_identity"] != resolved["realization_identity"]
    for bindings in ({}, {"root": "/a", "typo": "/b"}):
        with pytest.raises(ValueError, match="reference bindings"):
            materialize_reference(row["id"], bindings, table=table)
    literal = materialize_reference(row["id"], {"root": "${unexpanded}"}, table=table)
    assert literal["realization"]["modules"][0]["init_config"] == "${unexpanded}/{{id}}.ini"
