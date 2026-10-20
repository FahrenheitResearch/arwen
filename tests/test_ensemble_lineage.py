"""Lineage identity and probability mass survive public doors and restarts."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.lineage import (
    SCHEMA, branch_lineage, build_lineage, family_weights, provenance_label, validate_record,
)
from gpuwm.ensemble.request import EnsembleRequest


def base(source="gefs", member=0, source_member="p01"):
    return dict(member=member, source=source, source_member=source_member,
                cycle="2022-06-13T00", seed=42, rung="source-member", physics_arm="base")


def batch():
    return {"schema": "gpuwm-earth2-batch-v1", "model": "fcn3",
            "weights": {"revision": "test-checkpoint-r1", "files": [{"sha256": "a" * 64, "bytes": 10}]},
            "donor": {"source": "era5", "cycle": "2022-06-13T00", "sha256": "b" * 64},
            "members": [{"id": "n000", "seed": 123, "composition": {"surface": "era5"}}]}


def test_ai_receipt_records_weights_noise_and_donor_without_mutating_receipt():
    receipt = batch()
    before = deepcopy(receipt)
    row = build_lineage(base("earth2-fcn3", source_member="n000"), receipt=receipt)
    assert receipt == before
    assert row["parent_family"] == "ai"
    assert row["model_key"] == "earth2-fcn3"
    assert row["weight_revision"] == receipt["weights"]
    assert row["noise_seed"] == 123
    assert row["seed"] == 42
    assert row["donor"] == receipt["donor"]
    assert row["composition"] == {"surface": "era5"}
    assert row["label"] == provenance_label(row)


def test_physics_identity_and_no_donor_are_declared_by_row():
    row = build_lineage(base())
    assert row["parent_family"] == "physics"
    assert row["model_key"] == "gefs"
    assert row["weight_revision"]["upstream_model_id"] == "gefs"
    assert row["donor"] == {}


def test_unrecorded_weights_are_explicit_not_fabricated():
    row = build_lineage(base("earth2-fcn3", source_member="n000"))
    assert row["weight_revision"]["status"] == "runtime-evidence-not-recorded"
    assert "sha256" not in row["weight_revision"]


@pytest.mark.parametrize("weights", ["ai=0.4,physics=0.5", "ai=nan,physics=0", "ai=1,ai=0", "other=1", {"ai": True}, {"ai": -1, "physics": 2}])
def test_invalid_family_weights_refuse(weights):
    with pytest.raises(ValueError, match="family weights"):
        family_weights(weights)


def trajectories():
    return ({"source": "earth2-fcn3", "cycle": "2022-06-13T00", "member": "n000", "parent_receipt": batch()},
            {"source": "gefs", "cycle": "2022-06-13T00", "member": "p01"},
            {"source": "gefs", "cycle": "2022-06-13T00", "member": "p02"})


def test_explicit_mixed_trajectory_defaults_and_override_round_trip():
    request = EnsembleRequest.from_mapping({"trajectories": trajectories()})
    assert request.family_weights == {"ai": 1 / 3, "physics": 2 / 3}
    assert [r["family_weight"] for r in request.provenance] == [1 / 3, 2 / 3, 2 / 3]
    override = EnsembleRequest.from_mapping({**request.receipt(), "family_weights": "ai=0.5,physics=0.5"})
    assert override.family_weights == {"ai": .5, "physics": .5}
    assert EnsembleRequest.from_mapping(override.receipt()) == override
    with pytest.raises(ValueError, match="represented families"):
        EnsembleRequest.from_mapping({**request.receipt(), "family_weights": "ai=1"})


def test_v1_reader_keeps_old_canonical_label_and_receipt_unchanged():
    row = {**base(), "schema": "gpuwm-member-provenance.v1"}
    row["label"] = provenance_label(row)
    request = EnsembleRequest(1, provenance=(row,))
    assert request.provenance == (row,)
    assert EnsembleRequest.from_mapping(request.receipt()).provenance == (row,)
    assert "family_weights" not in request.receipt()


def test_checkpoint_identity_round_trip_seals_lineage(monkeypatch, tmp_path):
    from gpuwm.ensemble.restart_roster import identity_record, atomic_json, verify_roster, CONTRACT
    from gpuwm.core import model
    monkeypatch.setattr(model, "restart_identity_payload", lambda _: {"physics": 8})
    request = EnsembleRequest.from_mapping({"trajectories": trajectories()})
    inputs = {i: SimpleNamespace(experiment=object(), source=r["source"]) for i, r in enumerate(request.provenance)}
    identities = identity_record(request, range(3), inputs, [1, 2, 3])
    assert [r["provenance"] for r in identities["members"]] == list(request.provenance)
    record = {"schema": CONTRACT, "identities": identities,
              "members": [{"member_id": r["member_id"], "identity_sha256": r["identity_sha256"], "status": "pending", "provenance": r["provenance"]} for r in identities["members"]], "files": []}
    path = tmp_path / "ensemble-restart.json"
    atomic_json(path, record)
    restored = verify_roster(path, json.loads(json.dumps(identities)), root=tmp_path)
    assert restored["identities"]["request"]["provenance"] == list(request.provenance)


def test_saved_checkpoint_branch_keeps_ai_parent_identity_and_nested_ancestry():
    from gpuwm.ensemble.mass_launcher import branch_members
    row = build_lineage(base("earth2-fcn3", source_member="n000"), receipt=batch())
    branched = branch_lineage(row, {"kind": "saved-checkpoint", "checkpoint": "parent.npz"})
    result = branch_members([{"parent_id": 0, "checkpoint": "next.npz", "provenance": branched}], 2, surface=True)
    for child in result:
        label = child["provenance"]
        assert label["parent"]["parent"] == row
        for key in ("model_key", "weight_revision", "noise_seed", "donor", "parent_family"):
            assert label[key] == row[key]
        assert label["perturbation_recipe"]["branch"] == label["branch"]
        validate_record(label)


def test_new_source_family_is_only_table_data(monkeypatch):
    from dataclasses import replace
    from gpuwm import source_adapters
    adapter = replace(source_adapters.get_source_adapter("gefs"), source_id="future-source", parent_family="ai", upstream_model_id="future-model", upstream_model_version="v99")
    monkeypatch.setattr(source_adapters, "get_source_adapter", lambda _: adapter)
    row = build_lineage(base("future-source"))
    assert row["parent_family"] == "ai"
    assert row["model_key"] == "future-source"
    assert row["weight_revision"]["version"] == "v99"


def test_family_flag_passes_through_config_door():
    from gpuwm.ensemble.door import request_for_payload
    request = request_for_payload(b"", override={"trajectories": trajectories()}, family_weights="ai=0.5,physics=0.5")
    assert request.family_weights == {"ai": .5, "physics": .5}


def test_completed_batch_receipt_enriches_before_output_identity(tmp_path):
    from gpuwm.ensemble.lineage import enrich_request
    request = EnsembleRequest.from_mapping({"trajectories": [
        {"source": "earth2-fcn3", "cycle": "2022-06-13T00", "member": "n000"}]})
    (tmp_path / "earth2-batch.json").write_text(json.dumps(batch()))
    restored = enrich_request(request, [{"out": str(tmp_path / "n000")}])
    assert restored.provenance[0]["weight_revision"] == batch()["weights"]
    assert restored.provenance[0]["noise_seed"] == 123
    assert request.provenance[0]["weight_revision"]["status"] == "runtime-evidence-not-recorded"


def test_receipt_cannot_relabel_a_different_model():
    receipt = batch()
    receipt["model"] = "pangu"
    with pytest.raises(ValueError, match="differs from selected"):
        build_lineage(base("earth2-fcn3", source_member="n000"), receipt=receipt)


def test_mixed_ladder_weights_follow_member_family_share_and_config_override():
    from datetime import datetime, timedelta, timezone
    from gpuwm.ensemble.wofs_recipe import WofsOptions, plan_families
    start = datetime(2022, 6, 13, tzinfo=timezone.utc)
    ladder = plan_families(sources="earth2-fcn3:18,gefs:18", cycle=start, start=start,
                          end=start + timedelta(hours=6), options=WofsOptions(members=36))
    records, weights = ladder.provenance_records()
    assert weights == {"ai": .5, "physics": .5}
    assert len(records) == 36
    assert {r["model_key"] for r in records[:18]} == {"earth2-fcn3"}
    assert {r["model_key"] for r in records[18:]} == {"gefs"}


def test_parameterized_explicit_trajectory_runs_the_model_it_records():
    request = EnsembleRequest.from_mapping({"trajectories": [
        {"source": "earth2", "model": "fcn3", "cycle": "2022-06-13T00", "member": "n000"}]})
    assert request.trajectories[0]["source"] == "earth2-fcn3"
    assert request.provenance[0]["model_key"] == request.trajectories[0]["source"]


def fetch_receipt():
    return {"schema": "gpuwm-earth2-fetch-v2", "model": "fcn3", "members": [0, 1],
            "donor": batch()["donor"],
            "runner": {"earth2studio": "0.15.0", "torch": "2.8.0", "seed": 333,
                "seeds": [333, 334], "seed_scope": "per-member replay seeds",
                "weights": [{"asset": "checkpoint.pt", "revision": "test-r1", "sha256": "c" * 64}],
                "files": [{"member": member, "seed": 333 + member, "lead_hours": lead}
                          for member in (0, 1) for lead in (0, 6)]}}


def test_numeric_fetch_receipt_keeps_selected_seed_and_structured_weight_hashes(tmp_path):
    from gpuwm.ensemble.lineage import enrich_request
    receipt = fetch_receipt()
    before = deepcopy(receipt)
    request = EnsembleRequest.from_mapping({"trajectories": [
        {"source": "earth2", "model": "fcn3", "cycle": "2022-06-13T00", "member": "1"}]})
    (tmp_path / "earth2-run.json").write_text(json.dumps(receipt))
    enriched = enrich_request(request, [{"model": "fcn3", "out": str(tmp_path)}])
    row = enriched.provenance[0]
    assert row["model_key"] == "earth2-fcn3"
    assert row["noise_seed"] == 334
    assert row["noise_seed_scope"] == "per-member replay seeds"
    assert row["weight_revision"]["files"] == receipt["runner"]["weights"]
    assert row["weight_revision"]["runtime"] == {"earth2studio": "0.15.0", "torch": "2.8.0"}
    assert row["donor"] == receipt["donor"]
    assert EnsembleRequest.from_mapping(enriched.receipt()) == enriched
    assert receipt == before


def test_numeric_fetch_receipt_rejects_absent_member_and_conflicting_seed():
    with pytest.raises(ValueError, match="exactly one source member"):
        build_lineage(base("earth2-fcn3", source_member="2"), receipt=fetch_receipt())
    receipt = fetch_receipt()
    receipt["runner"]["files"][-1]["seed"] = 999
    with pytest.raises(ValueError, match="conflicting member seeds"):
        build_lineage(base("earth2-fcn3", source_member="1"), receipt=receipt)


def test_selected_model_does_not_hide_a_mismatched_receipt():
    receipt = batch()
    receipt["model"] = "pangu"
    with pytest.raises(ValueError, match="differs from selected"):
        build_lineage(base("earth2"), model="fcn3", receipt=receipt)


def test_family_subset_can_keep_global_member_indices():
    from gpuwm.ensemble.lineage import weight_records
    row = build_lineage(base(member=18))
    records, weights = weight_records([row])
    assert records[0]["member"] == 18
    assert weights == {"physics": 1}


def test_v1_checkpoint_identity_keeps_original_hash(monkeypatch):
    import hashlib
    from gpuwm.core import model
    from gpuwm.ensemble.restart_roster import identity_record
    monkeypatch.setattr(model, "restart_identity_payload", lambda _: {"physics": 8})
    legacy = {**base(), "schema": "gpuwm-member-provenance.v1"}
    legacy["label"] = provenance_label(legacy)
    request = EnsembleRequest(1, provenance=(legacy,))
    payload = {"member_id": 0, "seed": 42, "experiment": {"physics": 8},
               "source": "gefs", "prepared_head_sha256": None}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity = identity_record(request, [0], {0: SimpleNamespace(experiment=object(), source="gefs")}, [42])
    assert identity["members"][0]["identity_sha256"] == expected
    assert identity["request"]["provenance"] == [legacy]


def test_null_optional_checkpoint_metadata_remains_toml_safe():
    import tomllib
    from gpuwm.toml_document import emit_experiment_toml
    receipt = batch()
    receipt["weights"]["revision"] = None
    row = build_lineage(base("earth2-fcn3", source_member="n000"), receipt=receipt)
    assert "revision" not in row["weight_revision"]
    assert row["weight_revision"]["files"] == receipt["weights"]["files"]
    assert tomllib.loads(emit_experiment_toml({"member": row}))["member"] == row
    assert receipt["weights"]["revision"] is None


def test_wrfout_attribute_uses_v2_canonical_label(monkeypatch):
    from gpuwm import runtime
    from gpuwm.io import wrfout
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
    monkeypatch.setattr(wrfout, "wrf_global_attrs", lambda *a, **k: {})
    monkeypatch.setattr(wrfout, "initial_condition_global_attrs", lambda *a, **k: {})
    row = build_lineage(base("earth2-fcn3", source_member="n000"), receipt=batch())
    capture = MemberOutputCapture(lambda **k: None, 0, provenance=row)
    with member_output_scope(capture):
        attrs = runtime._global_wrf_attrs(None, None)
    assert attrs["GPUWM_ENSEMBLE_PROVENANCE"] == row["label"]


def test_v1_trajectory_alias_stays_byte_stable_in_receipts():
    row = {**base("gfs-ensemble"), "schema": "gpuwm-member-provenance.v1"}
    row["label"] = provenance_label(row)
    # V1 readers validate retained records without resolving legacy aliases.
    request = EnsembleRequest(1, recipe="multi-model", provenance=(row,),
        trajectories=({"source": "gfs-ensemble", "cycle": "2022-06-13T00", "member": "p01"},))
    assert request.receipt()["trajectories"][0]["source"] == "gfs-ensemble"
    assert request.provenance == (row,)


def test_resolved_native_stochastic_selectors_seal_before_checkpoint_identity():
    from gpuwm.ensemble.lineage import bind_runtime_stochastic
    request = EnsembleRequest.from_mapping({"trajectories": trajectories()})
    selected = {"spp_selectors": [{"grid_id": 1, "selected_flags": {"spp_pbl": 1}}],
                "process_configurations": {"sppt": None, "spp": {"pbl": {"amplitude": .2}}}}
    restored = bind_runtime_stochastic(request, {0: selected})
    record = restored.provenance[0]
    assert record["perturbation_recipe"]["runtime_stochastic"]["spp_selectors"] == selected["spp_selectors"]
    assert record["perturbation_recipe"]["runtime_stochastic"]["process_configurations"] == {"spp": {"pbl": {"amplitude": .2}}}
    assert restored.provenance[1] == request.provenance[1]
    assert "runtime_stochastic" not in request.provenance[0]["perturbation_recipe"]
    validate_record(record)
    assert EnsembleRequest.from_mapping(restored.receipt()) == restored


def test_declared_weight_files_with_null_checksums_remain_toml_safe():
    import tomllib
    from gpuwm.earth2_run import model_table
    from gpuwm.toml_document import emit_experiment_toml
    weights = model_table()["models"]["fcn3"]["weights"]
    assert any(item["sha256"] is None for item in weights["files"])
    row = build_lineage(base("earth2-fcn3", source_member="n0333"))
    files = row["weight_revision"]["files"]
    assert all(item.get("sha256") is not None for item in files if "sha256" in item)
    assert [item["path"] for item in files] == [item["path"] for item in weights["files"]]
    assert tomllib.loads(emit_experiment_toml({"member": row}))["member"] == row
    assert any(item["sha256"] is None for item in weights["files"])
