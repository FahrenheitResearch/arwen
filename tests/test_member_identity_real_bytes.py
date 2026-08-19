"""The member capability against real production bytes, via the Rust bridge.

Every file under ``tests/fixtures/ensemble-member-identity/`` is
unmodified production GRIB2 (whole envelopes of the 2026-08-17 00Z
cycles; provenance and SHA-256 in the fixture README).  These tests run
the REAL ``grib2_inventory`` executable -- the artifact, not a mock --
so what passes here is the shipped verification path end to end.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest

from gpuwm.member_grammar import MemberIdentityRefusal, load_member_grammar
from gpuwm.member_prep import (RECEIPT_NAME, main, prepare_member,
                               verify_member_file)
from gpuwm.source_authorities import packaged_member_grammar

_TESTS = Path(__file__).resolve().parent
FIXTURES = _TESTS / "fixtures" / "ensemble-member-identity"
CYCLE = datetime(2026, 8, 17, 0)


def _gefs():
    return load_member_grammar(
        packaged_member_grammar("gefs-ensemble-grib2-members-v1"))


def _aigefs():
    return load_member_grammar(
        packaged_member_grammar("aigefs-ensemble-grib2-members-v1"))


def _gefs_file(name: str):
    return FIXTURES / "gefs.20260817" / "00" / "atmos" / "pgrb2ap5" / name


def _aigefs_file(member: str, name: str):
    return (FIXTURES / "aigefs.20260817" / "00" / member / "model" / "atmos"
            / "grib2" / name)


# ---------------------------------------------------------------------------
# Acceptance on real bytes
# ---------------------------------------------------------------------------

def test_the_flagged_control_verifies_with_its_low_res_control_type():
    evidence = verify_member_file(
        _gefs(), "c00", _gefs_file("gec00.t00z.pgrb2a.0p50.f000"))
    assert evidence.perturbation_number == 0
    assert evidence.type_of_ensemble_forecast == 1
    assert evidence.ensemble_size == (30,)


def test_a_perturbed_member_verifies_with_its_ordinal():
    evidence = verify_member_file(
        _gefs(), "p01", _gefs_file("gep01.t00z.pgrb2a.0p50.f000"))
    assert evidence.perturbation_number == 1
    assert evidence.type_of_ensemble_forecast == 3


def test_accumulation_messages_verify_through_the_declared_pdt_11():
    evidence = verify_member_file(
        _gefs(), "c00", _gefs_file("gec00.t00z.pgrb2a.0p50.f003"))
    assert evidence.product_definition_templates == (1, 11)


def test_the_unflagged_control_verifies_through_its_declared_class():
    """The AI ensemble stamps its control exactly like a perturbed
    member (type 3); the grammar declares that fact, so the control
    verifies without any code knowing this ensemble's name."""

    evidence = verify_member_file(
        _aigefs(), "mem000", _aigefs_file("mem000", "aigefs.t00z.sfc.f000.grib2"))
    assert evidence.perturbation_number == 0
    assert evidence.type_of_ensemble_forecast == 3
    assert evidence.ensemble_size == (31,)


# ---------------------------------------------------------------------------
# Refusals on real bytes
# ---------------------------------------------------------------------------

def test_the_real_ensemble_mean_refuses_as_a_member_by_name():
    with pytest.raises(MemberIdentityRefusal) as caught:
        verify_member_file(
            _gefs(), "p01", _gefs_file("geavg.t00z.pgrb2a.0p50.f000"))
    message = str(caught.value)
    assert "STATISTIC" in message
    assert "unweighted mean of all members" in message
    assert "'geavg'" in message


def test_the_real_spread_refuses_naming_the_spread():
    with pytest.raises(MemberIdentityRefusal) as caught:
        verify_member_file(
            _gefs(), "c00", _gefs_file("gespr.t00z.pgrb2a.0p50.f000"))
    message = str(caught.value)
    assert "'gespr'" in message
    assert "standard deviation" in message


def test_a_member_file_claimed_as_another_member_names_both():
    """Filename-level claim p07, bytes carry perturbationNumber 1."""

    with pytest.raises(MemberIdentityRefusal) as caught:
        verify_member_file(
            _gefs(), "p07", _gefs_file("gep01.t00z.pgrb2a.0p50.f000"))
    message = str(caught.value)
    assert "member p07" in message and "perturbationNumber 7" in message
    assert "perturbationNumber 1" in message
    assert "member p01" in message


def test_one_ensembles_bytes_refuse_under_the_others_grammar():
    """GEFS member 1 and AIGEFS member 1 carry the IDENTICAL verification
    triple (PDT 1, type 3, perturbationNumber 1).  The declared encoded
    ensemble size (30 vs 31 -- the octet the two sources set
    incompatibly) is what tells the bytes apart."""

    with pytest.raises(MemberIdentityRefusal) as caught:
        verify_member_file(
            _aigefs(), "mem001", _gefs_file("gep01.t00z.pgrb2a.0p50.f000"))
    message = str(caught.value)
    assert "declared encoded ensemble size 31" in message
    assert "numberOfForecastsInEnsemble = 30" in message


def test_deterministic_bytes_refuse_as_no_ensemble_identity():
    deterministic = (_TESTS / "fixtures" / "gdas-process-id"
                     / "nomads-gdas-20260729t12z-f000.grib2")
    with pytest.raises(MemberIdentityRefusal, match="no ensemble identity"):
        verify_member_file(_gefs(), "p01", deterministic)


def test_asking_to_prepare_the_mean_refuses_before_touching_bytes():
    with pytest.raises(MemberIdentityRefusal, match="ensemble mean"):
        prepare_member(
            grammar_id="gefs-ensemble-grib2-members-v1", member_id="geavg",
            cycle=CYCLE, steps=[0], inputs_root=FIXTURES,
            output_root=FIXTURES)     # never reached


# ---------------------------------------------------------------------------
# End-to-end preparation on real bytes
# ---------------------------------------------------------------------------

def test_a_real_control_prepares_into_a_member_addressed_tree(tmp_path):
    member_dir = prepare_member(
        grammar_id="gefs-ensemble-grib2-members-v1", member_id="c00",
        cycle=CYCLE, steps=[0, 3], products=["pgrb2a"],
        inputs_root=FIXTURES, output_root=tmp_path)
    assert member_dir.name == "c00"
    staged = sorted(p.name for p in (member_dir / "pgrb2a").iterdir())
    assert staged == [
        "gec00.t00z.pgrb2a.0p50.f000", "gec00.t00z.pgrb2a.0p50.f003"]
    receipt = json.loads(
        (member_dir / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["member"]["ordinal"] == 0
    assert receipt["member_set"]["registry_id"] == (
        "gefs-ensemble-grib2-members-v1")
    for entry in receipt["files"]:
        assert entry["observed"]["perturbation_number"] == 0
        assert entry["sha256"]
    steps = sorted(entry["step_hours"] for entry in receipt["files"])
    assert steps == [0, 3]


def test_a_path_addressed_member_prepares_with_identity_in_the_tree(
        tmp_path):
    """The feed whose leaf filenames are identical across members: the
    prepared tree and receipt must carry what the leaf name cannot."""

    member_dir = prepare_member(
        grammar_id="aigefs-ensemble-grib2-members-v1", member_id="mem001",
        cycle=CYCLE, steps=[0], products=["sfc"],
        inputs_root=FIXTURES, output_root=tmp_path)
    staged = member_dir / "sfc" / "aigefs.t00z.sfc.f000.grib2"
    assert staged.is_file()
    assert member_dir.name == "mem001"    # identity lives in the tree
    receipt = json.loads(
        (member_dir / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["member"]["id"] == "mem001"
    assert receipt["files"][0]["observed"]["perturbation_number"] == 1


def test_a_statistic_swapped_under_a_member_path_leaves_no_tree(tmp_path):
    """The worst case end to end: a file at a MEMBER's declared path
    whose bytes are the ensemble mean.  Preparation must refuse at the
    byte gate and leave nothing that looks prepared."""

    inputs = tmp_path / "inputs"
    target = (inputs / "gefs.20260817" / "00" / "atmos" / "pgrb2ap5"
              / "gep01.t00z.pgrb2a.0p50.f000")
    target.parent.mkdir(parents=True)
    target.write_bytes(
        _gefs_file("geavg.t00z.pgrb2a.0p50.f000").read_bytes())
    with pytest.raises(MemberIdentityRefusal, match="STATISTIC"):
        prepare_member(
            grammar_id="gefs-ensemble-grib2-members-v1", member_id="p01",
            cycle=CYCLE, steps=[0], products=["pgrb2a"],
            inputs_root=inputs, output_root=tmp_path / "prepared")
    prepared = tmp_path / "prepared"
    leftovers = list(prepared.rglob("*")) if prepared.exists() else []
    assert not [p for p in leftovers if p.is_file()]


# ---------------------------------------------------------------------------
# The front door
# ---------------------------------------------------------------------------

def test_the_cli_prepares_and_prints_the_member_tree(tmp_path, capsys):
    code = main([
        "--member-set", "gefs-ensemble-grib2-members-v1",
        "--member", "p02", "--cycle", "2026-08-17T00", "--steps", "0",
        "--products", "pgrb2a",
        "--inputs", str(FIXTURES), "--output", str(tmp_path),
    ])
    output = capsys.readouterr().out.strip()
    assert code == 0
    assert output.endswith("p02")
    assert (Path(output) / RECEIPT_NAME).is_file()


def test_the_cli_refusal_reaches_the_user_and_exits_nonzero(
        tmp_path, capsys):
    code = main([
        "--member-set", "gefs-ensemble-grib2-members-v1",
        "--member", "gespr", "--cycle", "2026-08-17T00", "--steps", "0",
        "--inputs", str(FIXTURES), "--output", str(tmp_path),
    ])
    output = capsys.readouterr().out
    assert code == 2
    assert "ensemble standard deviation" in output
    assert "not a member" in output


def test_the_cli_verify_only_reports_evidence_without_staging(
        tmp_path, capsys):
    code = main([
        "--member-set", "aigefs-ensemble-grib2-members-v1",
        "--member", "mem002", "--verify-only",
        str(_aigefs_file("mem002", "aigefs.t00z.sfc.f000.grib2")),
    ])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["observed"]["perturbation_number"] == 2
    assert not list(tmp_path.iterdir())


def test_the_cli_describes_a_packaged_set(capsys):
    code = main(["--describe", "gefs-ensemble-grib2-members-v1"])
    output = capsys.readouterr().out
    assert code == 0
    assert "31 declared members" in output
    assert "gespr" in output and "never members" in output
