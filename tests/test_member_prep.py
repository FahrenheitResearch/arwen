"""Member verification and preparation: the fail-closed contract.

Synthetic inventory rows drive the verification logic (the pure half);
the real-bytes battery in ``test_member_identity_real_bytes.py`` drives
the same functions through the Rust bridge on production GRIB2.
"""

from __future__ import annotations

from datetime import datetime
import json

import pytest

from gpuwm.member_grammar import MemberGrammar, MemberIdentityRefusal
from gpuwm.member_prep import (MemberFileEvidence, RECEIPT_NAME,
                               prepare_member, verify_member_rows)


def _grammar() -> MemberGrammar:
    return MemberGrammar({
        "schema": "rw-wps.members.v1",
        "name": "synthetic-members",
        "format": "grib2",
        "layout": "file_per_member",
        "declared_member_count": 4,
        "classes": {
            "control": {
                "ordinals": [0],
                "member_id": "c00",
                "token": "xc00",
                "verification": {
                    "product_definition_templates": [1, 11],
                    "type_of_ensemble_forecast": 1,
                    "perturbation_number": "ordinal",
                    "ensemble_size": 3,
                    "type_of_generating_process": 4,
                    "forecast_generating_process_id": 7,
                },
            },
            "perturbed": {
                "ordinals": {"first": 1, "last": 3},
                "member_id": "p{ordinal:02d}",
                "token": "xp{ordinal:02d}",
                "verification": {
                    "product_definition_templates": [1, 11],
                    "type_of_ensemble_forecast": 3,
                    "perturbation_number": "ordinal",
                    "ensemble_size": 3,
                    "type_of_generating_process": 4,
                    "forecast_generating_process_id": 7,
                },
            },
        },
        "statistics": {
            "xavg": {
                "statistic": "ensemble mean",
                "token": "xavg",
                "product_definition_templates": [2, 12],
                "derived_forecast": 0,
            },
            "xspr": {
                "statistic": "ensemble spread",
                "token": "xspr",
                "product_definition_templates": [2, 12],
                "derived_forecast": 2,
            },
        },
        "products": {
            "main": {
                "relative_path":
                    "cycle.{yyyymmdd}/{hh}/{token}.t{hh}z.f{fff}",
            },
        },
    }, source="synthetic")


def _row(index=0, *, pdt=1, member="1", ensemble_type="3", size="3",
         derived="-", generating="4", process="7"):
    return {
        "index": str(index), "pdt": str(pdt), "member": member,
        "ensemble_type": ensemble_type, "ensemble_size": size,
        "derived_forecast": derived, "generating_process": generating,
        "forecast_generating_process_id": process,
    }


def _verify(member_id, rows):
    grammar = _grammar()
    return verify_member_rows(
        grammar, grammar.member(member_id), rows, source_label="synthetic")


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

def test_a_member_whose_bytes_carry_its_declared_triple_verifies():
    evidence = _verify("p01", [
        _row(0), _row(1, pdt=11),
    ])
    assert isinstance(evidence, MemberFileEvidence)
    assert evidence.messages == 2
    assert evidence.product_definition_templates == (1, 11)
    assert evidence.perturbation_number == 1


def test_an_unflagged_control_verifies_through_its_declared_class():
    """A control stamped exactly like a perturbed member (the AI-ensemble
    convention) is not a defect to the engine: the grammar DECLARES that
    its control carries type 3 with perturbationNumber 0, and the bytes
    verify against the declaration, not against the WMO control codes."""

    grammar = MemberGrammar({
        "schema": "rw-wps.members.v1",
        "name": "unflagged-control",
        "format": "grib2",
        "layout": "file_per_member",
        "declared_member_count": 2,
        "classes": {
            "control": {
                "ordinals": [0], "member_id": "m000", "token": "m000",
                "verification": {
                    "product_definition_templates": [1],
                    "type_of_ensemble_forecast": 3,
                    "perturbation_number": "ordinal",
                    "ensemble_size": 2,
                },
            },
            "perturbed": {
                "ordinals": [1], "member_id": "m001", "token": "m001",
                "verification": {
                    "product_definition_templates": [1],
                    "type_of_ensemble_forecast": 3,
                    "perturbation_number": "ordinal",
                    "ensemble_size": 2,
                },
            },
        },
        "products": {"main": {"relative_path": "{token}.f{fff}"}},
    }, source="synthetic")
    evidence = verify_member_rows(
        grammar, grammar.member("m000"),
        [_row(0, member="0", ensemble_type="3", size="2",
              generating="4", process="9")],
        source_label="synthetic")
    assert evidence.perturbation_number == 0


# ---------------------------------------------------------------------------
# Refusals -- each names the concrete breakage it prevents
# ---------------------------------------------------------------------------

def test_a_statistic_used_as_a_member_refuses_by_name():
    with pytest.raises(MemberIdentityRefusal) as caught:
        _verify("p01", [_row(0, pdt=2, member="-", ensemble_type="-",
                             derived="0")])
    message = str(caught.value)
    assert "STATISTIC" in message
    assert "unweighted mean of all members" in message
    assert "'xavg'" in message          # the declared namespace names it
    assert "refused as a member" in message


def test_the_spread_refuses_as_the_spread_not_as_a_generic_error():
    with pytest.raises(MemberIdentityRefusal) as caught:
        _verify("c00", [_row(0, pdt=2, member="-", ensemble_type="-",
                             derived="2")])
    assert "'xspr'" in str(caught.value)


def test_a_filename_byte_mismatch_names_both_sides():
    """Filename says p01, bytes say perturbationNumber 2."""

    with pytest.raises(MemberIdentityRefusal) as caught:
        _verify("p01", [_row(0, member="2")])
    message = str(caught.value)
    assert "claimed as synthetic-members member p01" in message
    assert "perturbationNumber 1" in message
    assert "perturbationNumber 2" in message
    assert "member p02" in message      # which member the bytes really are


def test_a_wrong_ensemble_flag_refuses_naming_both_values():
    with pytest.raises(MemberIdentityRefusal) as caught:
        _verify("c00", [_row(0, member="0", ensemble_type="3")])
    message = str(caught.value)
    assert "declared typeOfEnsembleForecast 1" in message
    assert "carries typeOfEnsembleForecast 3" in message


def test_a_different_encoded_ensemble_size_refuses_as_foreign_bytes():
    """The triple alone cannot separate two ensembles' member N; the
    declared encoded size (which sources set incompatibly) can."""

    with pytest.raises(MemberIdentityRefusal) as caught:
        _verify("p01", [_row(0, size="30")])
    message = str(caught.value)
    assert "declared encoded ensemble size 3" in message
    assert "numberOfForecastsInEnsemble = 30" in message


def test_a_different_generating_process_refuses():
    with pytest.raises(MemberIdentityRefusal,
                       match="generatingProcessIdentifier 7"):
        _verify("p01", [_row(0, process="107")])


def test_deterministic_bytes_refuse_as_carrying_no_ensemble_identity():
    with pytest.raises(MemberIdentityRefusal) as caught:
        _verify("p01", [_row(0, pdt=0, member="-", ensemble_type="-",
                             size="-")])
    assert "no ensemble identity" in str(caught.value)


def test_an_undeclared_template_refuses_by_number():
    with pytest.raises(MemberIdentityRefusal, match="template 60"):
        _verify("p01", [_row(0, pdt=60)])


def test_one_foreign_message_in_a_correct_file_still_refuses():
    """Fail closed means all messages, not most."""

    rows = [_row(i) for i in range(5)] + [_row(5, member="3")]
    with pytest.raises(MemberIdentityRefusal, match="field 5"):
        _verify("p01", rows)


def test_missing_ensemble_octets_on_a_member_template_refuse():
    with pytest.raises(MemberIdentityRefusal, match="typeOfEnsembleForecast"):
        _verify("p01", [_row(0, ensemble_type="-")])


# ---------------------------------------------------------------------------
# Preparation: the member-addressed tree and its receipt
# ---------------------------------------------------------------------------

def _stage_inputs(tmp_path, monkeypatch, rows_by_member):
    """Fake the bridge: inventory rows come from a lookup, bytes from disk."""

    inputs = tmp_path / "inputs"
    grammar = _grammar()
    cycle = datetime(2026, 8, 17, 0)
    for member_id, _rows in rows_by_member.items():
        relative = grammar.relative_path(member_id, "main", cycle, 0)
        path = inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"payload-of-{member_id}".encode())

    def _fake_rows(source, executable=None):
        for member_id, rows in rows_by_member.items():
            if grammar.relative_path(member_id, "main", cycle, 0).endswith(
                    source.name) and f"-{member_id}".encode() in source.read_bytes():
                return rows
        raise AssertionError(f"unplanned inventory of {source}")

    monkeypatch.setattr(
        "gpuwm.member_prep.member_inventory_rows", _fake_rows)
    return inputs, cycle


def test_preparation_stages_a_member_addressed_tree_with_a_receipt(
        tmp_path, monkeypatch):
    inputs, cycle = _stage_inputs(
        tmp_path, monkeypatch, {"p02": [_row(0, member="2")]})
    grammar_document = tmp_path / "synthetic.members.json"
    grammar_document.write_text(
        json.dumps(_grammar_document()), encoding="utf-8")
    member_dir = prepare_member(
        grammar_path=grammar_document, member_id="p02", cycle=cycle,
        steps=[0], inputs_root=inputs, output_root=tmp_path / "prepared")
    assert member_dir == (
        tmp_path / "prepared" / "synthetic-members" / "20260817T00Z" / "p02")
    staged = member_dir / "main" / "xp02.t00z.f000"
    assert staged.read_bytes() == b"payload-of-p02"
    receipt = json.loads((member_dir / RECEIPT_NAME).read_text())
    assert receipt["member"]["id"] == "p02"
    assert receipt["member"]["ordinal"] == 2
    assert receipt["member_set"]["name"] == "synthetic-members"
    assert receipt["member_set"]["sha256"]
    (entry,) = receipt["files"]
    assert entry["observed"]["perturbation_number"] == 2
    assert entry["sha256"]
    assert "sizing authority" in receipt["counting_rule"]


def test_preparation_refuses_a_missing_declared_input_by_relative_path(
        tmp_path, monkeypatch):
    inputs, cycle = _stage_inputs(
        tmp_path, monkeypatch, {"p02": [_row(0, member="2")]})
    grammar_document = tmp_path / "synthetic.members.json"
    grammar_document.write_text(
        json.dumps(_grammar_document()), encoding="utf-8")
    with pytest.raises(MemberIdentityRefusal) as caught:
        prepare_member(
            grammar_path=grammar_document, member_id="p02", cycle=cycle,
            steps=[0, 3], inputs_root=inputs,
            output_root=tmp_path / "prepared")
    assert "cycle.20260817/00/xp02.t00z.f003" in str(caught.value)
    assert not (tmp_path / "prepared" / "synthetic-members").exists()


def test_a_refused_member_leaves_no_partial_tree(tmp_path, monkeypatch):
    inputs, cycle = _stage_inputs(
        tmp_path, monkeypatch,
        {"p02": [_row(0, member="3")]})     # bytes are a different member
    grammar_document = tmp_path / "synthetic.members.json"
    grammar_document.write_text(
        json.dumps(_grammar_document()), encoding="utf-8")
    with pytest.raises(MemberIdentityRefusal):
        prepare_member(
            grammar_path=grammar_document, member_id="p02", cycle=cycle,
            steps=[0], inputs_root=inputs,
            output_root=tmp_path / "prepared")
    prepared_root = tmp_path / "prepared"
    leftovers = list(prepared_root.rglob("*")) if prepared_root.exists() else []
    assert not [p for p in leftovers if p.is_file()]


def test_an_already_prepared_member_refuses_rather_than_overwriting(
        tmp_path, monkeypatch):
    inputs, cycle = _stage_inputs(
        tmp_path, monkeypatch, {"p02": [_row(0, member="2")]})
    grammar_document = tmp_path / "synthetic.members.json"
    grammar_document.write_text(
        json.dumps(_grammar_document()), encoding="utf-8")
    kwargs = dict(
        grammar_path=grammar_document, member_id="p02", cycle=cycle,
        steps=[0], inputs_root=inputs, output_root=tmp_path / "prepared")
    prepare_member(**kwargs)
    with pytest.raises(MemberIdentityRefusal, match="never overwrites"):
        prepare_member(**kwargs)


def test_preparing_a_statistic_as_a_member_refuses_before_any_decode(
        tmp_path, monkeypatch):
    inputs, cycle = _stage_inputs(
        tmp_path, monkeypatch, {"p02": [_row(0, member="2")]})
    grammar_document = tmp_path / "synthetic.members.json"
    grammar_document.write_text(
        json.dumps(_grammar_document()), encoding="utf-8")
    with pytest.raises(MemberIdentityRefusal, match="ensemble mean"):
        prepare_member(
            grammar_path=grammar_document, member_id="xavg", cycle=cycle,
            steps=[0], inputs_root=inputs,
            output_root=tmp_path / "prepared")


def _grammar_document():
    return {
        "schema": "rw-wps.members.v1",
        "name": "synthetic-members",
        "format": "grib2",
        "layout": "file_per_member",
        "declared_member_count": 4,
        "classes": {
            "control": {
                "ordinals": [0], "member_id": "c00", "token": "xc00",
                "verification": {
                    "product_definition_templates": [1, 11],
                    "type_of_ensemble_forecast": 1,
                    "perturbation_number": "ordinal",
                    "ensemble_size": 3,
                    "type_of_generating_process": 4,
                    "forecast_generating_process_id": 7,
                },
            },
            "perturbed": {
                "ordinals": {"first": 1, "last": 3},
                "member_id": "p{ordinal:02d}", "token": "xp{ordinal:02d}",
                "verification": {
                    "product_definition_templates": [1, 11],
                    "type_of_ensemble_forecast": 3,
                    "perturbation_number": "ordinal",
                    "ensemble_size": 3,
                    "type_of_generating_process": 4,
                    "forecast_generating_process_id": 7,
                },
            },
        },
        "statistics": {
            "xavg": {
                "statistic": "ensemble mean", "token": "xavg",
                "product_definition_templates": [2, 12],
                "derived_forecast": 0,
            },
        },
        "products": {
            "main": {
                "relative_path":
                    "cycle.{yyyymmdd}/{hh}/{token}.t{hh}z.f{fff}",
            },
        },
    }
