"""The envelope measures a spread, flags a degenerate one, and stays home.

Controls, named:

* **config-identity mismatch** -- a receipt whose members are not the run it
  is cited beside makes the check exit non-zero, and a receipt carrying
  another campaign's pins (a different innermost spacing, a different
  microphysics) cannot pass it;
* **admission-gate refusal** -- a member whose consumed-input hashes are not
  the preserved set is refused by the reducer, and the refusal names it;
* **degeneracy** -- an all-identical member set produces a zero envelope, and
  the row says so rather than passing every candidate silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import n5s_delegation_fixture as fixture
from gpuwm.verify import chaos_envelope, spectral
from tools import certify_band_from_ensemble, matched_wrfout_envelope

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "real74_thompson_1218z_rrtmg_legacy_4dom.toml"
DOCUMENT = REPO_ROOT / "docs" / "public" / "VERIFICATION.md"

CADENCE = 600
DURATION = 1800
DOMAINS = ("d01", "d02")
DX = {"d01": 12000.0, "d02": 3000.0}


def _registration(**overrides):
    pins = dict(
        start_time=fixture.FIXTURE_START, domain_dx_m=DX,
        state_fields=("T", "QVAPOR"), leads_seconds=(CADENCE, 2 * CADENCE),
        cadence_seconds=CADENCE, reflectivity_field="REFL_10CM",
        reflectivity_threshold=40.0, low_pass_physical_width_m=6000.0,
        low_pass_interior_exclusion_cells=1, boundary_width_cells=2,
        fss_radius_m=5000.0, object_min_area_km2=25.0, object_connectivity=8,
        evaluator_commit="a" * 40)
    pins.update(overrides)
    return chaos_envelope.make_registration(**pins)


def _run(root: Path, name: str, seed: int, *, object_second: int = CADENCE):
    return fixture.build_run(
        root, name, seed=seed, domains=DOMAINS, cadence_seconds=CADENCE,
        duration_seconds=DURATION, object_second=object_second)


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory):
    root = tmp_path_factory.mktemp("envelope")
    registration = _registration()
    members = [_run(root, f"member-{index:02d}", seed=100 * index + 3)
               for index in range(3)]
    unperturbed = _run(root, "unperturbed", seed=71)
    candidate = _run(root, "candidate", seed=77)
    receipt = chaos_envelope.build_envelope(
        registration=registration, member_directories=members,
        candidate_directory=candidate, unperturbed_directory=unperturbed,
        member_identity=chaos_envelope.config_identity(CONFIG),
        output=root / "envelope.json")
    return root, registration, receipt


# --------------------------------------------------------------------------
# receipt content contract
# --------------------------------------------------------------------------


def test_every_row_carries_the_recorded_contract(ensemble):
    _root, registration, receipt = ensemble
    assert receipt["schema"] == chaos_envelope.ENVELOPE_SCHEMA
    assert receipt["registration_sha256"] == registration["registration_sha256"]
    assert len(receipt["evaluator_commit"]) == 40
    assert receipt["pair_count"] == 3  # three members -> three unordered pairs
    assert receipt["member_identity"]["domains"], receipt["member_identity"]
    seen = set()
    for row in receipt["rows"]:
        assert set(row) >= {
            "metric", "domain", "lead_seconds", "pair_count", "envelope",
            "envelope_degenerate", "candidate_distance", "verdict"}
        assert row["pair_count"] == 3
        assert row["metric"] in chaos_envelope.METRIC_CLASSES
        assert row["domain"] in DOMAINS
        assert row["lead_seconds"] in (CADENCE, 2 * CADENCE)
        assert row["envelope_degenerate"] is (row["envelope"] == 0.0)
        assert row["verdict"] in {
            "degenerate-envelope", "within-envelope", "outside-envelope"}
        seen.add((row["metric"], row["domain"], row["lead_seconds"]))
    # every metric class, on every domain, at every lead
    assert len(seen) == len(chaos_envelope.METRIC_CLASSES) * len(DOMAINS) * 2


def test_the_receipt_carries_the_spectral_column(ensemble):
    _root, _registration, receipt = ensemble
    assert receipt["spectral_pins_sha256"] == spectral.PINS_SHA256
    spectral_rows = [row for row in receipt["rows"]
                     if row["metric"] == "spectral_log_distance"]
    assert len(spectral_rows) == len(spectral.FIELD_PINS) * len(DOMAINS) * 2
    variables = {row["key"].split(":", 1)[1].split("|", 1)[0]
                 for row in spectral_rows}
    assert variables == {pin["variable"] for pin in spectral.FIELD_PINS}
    # a spectral row is a distance in decades: non-negative, and this
    # fixture's members differ, so the spread is not degenerate
    assert all(row["envelope"] > 0.0 for row in spectral_rows), spectral_rows


def test_the_receipt_names_the_class_it_does_not_compute(ensemble):
    """The coverage map carries its own gap rather than implying none."""
    _root, _registration, receipt = ensemble
    unscheduled = receipt["metric_classes_unscheduled"]
    assert [entry["metric_class"] for entry in unscheduled] == ["distributional"]
    assert unscheduled[0]["status"] == "unscheduled"
    assert not set(entry["metric_class"] for entry in unscheduled) & set(
        receipt["metric_classes"])
    assert {row["metric"] for row in receipt["rows"]} == set(
        receipt["metric_classes"])


def test_the_registration_hash_covers_the_pins():
    registration = _registration()
    tampered = json.loads(json.dumps(registration))
    tampered["parameters"]["reflectivity_threshold"] = 35.0
    with pytest.raises(ValueError, match="hash does not match"):
        chaos_envelope.validate_registration(tampered)


def test_nearest_rank_returns_a_value_some_pair_achieved():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert chaos_envelope.nearest_rank(values, 95.0) == 4.0
    assert chaos_envelope.nearest_rank(values, 50.0) == 2.0
    assert chaos_envelope.nearest_rank([2.0, 1.0], 95.0) == 2.0
    for percentile in (5.0, 33.0, 95.0, 100.0):
        assert chaos_envelope.nearest_rank(values, percentile) in values


def test_a_zero_envelope_is_flagged_not_silently_passed():
    """CONTROL: identical members measure nothing, and the row admits it."""
    keys = {"low_pass_state_rmse:T|d01|600": [0.0, 0.0, 0.0]}
    rows = chaos_envelope.envelope_rows(
        keys, {"low_pass_state_rmse:T|d01|600": 4.0}, percentile=95.0)
    assert rows[0]["envelope_degenerate"] is True
    assert rows[0]["verdict"] == "degenerate-envelope"
    rows = chaos_envelope.envelope_rows(
        {"low_pass_state_rmse:T|d01|600": [1.0, 2.0, 3.0]},
        {"low_pass_state_rmse:T|d01|600": 4.0}, percentile=95.0)
    assert rows[0]["verdict"] == "outside-envelope"
    assert rows[0]["envelope"] == 3.0


# --------------------------------------------------------------------------
# the envelope belongs to the run it is cited beside
# --------------------------------------------------------------------------


def test_a_matching_receipt_passes_the_identity_check(ensemble):
    _root, _registration, receipt = ensemble
    assert chaos_envelope.check_config_identity(
        receipt, CONFIG, DOCUMENT) == []
    assert matched_wrfout_envelope.main([
        "check-identity", "--receipt", str(_root / "envelope.json"),
        "--config", str(CONFIG), "--document", str(DOCUMENT)]) == 0


def test_a_receipt_from_other_pins_cannot_be_cited_here(ensemble, tmp_path):
    """CONTROL: another campaign's ladder and microphysics must be refused."""
    _root, _registration, receipt = ensemble
    foreign = json.loads(json.dumps(receipt))
    innermost = foreign["member_identity"]["domains"][-1]
    innermost["dx_m"] = 1000.0 / 3.0
    innermost["dy_m"] = 1000.0 / 3.0
    innermost["mp_physics"] = 10
    issues = chaos_envelope.check_config_identity(foreign, CONFIG, DOCUMENT)
    assert any("dx_m" in issue for issue in issues), issues
    assert any("mp_physics" in issue for issue in issues), issues
    path = tmp_path / "foreign.json"
    path.write_text(json.dumps(foreign), encoding="utf-8")
    assert matched_wrfout_envelope.main([
        "check-identity", "--receipt", str(path), "--config", str(CONFIG),
        "--document", str(DOCUMENT)]) == 1


def test_a_receipt_without_an_identity_is_refused():
    assert chaos_envelope.check_config_identity({}, CONFIG) == [
        "receipt records no member geometry/physics identity"]


def test_the_document_half_reads_the_spacings_off_the_config(ensemble, tmp_path):
    _root, _registration, receipt = ensemble
    quiet = tmp_path / "no-such-ladder.md"
    quiet.write_text("a document that names no domain spacing\n",
                     encoding="utf-8")
    issues = chaos_envelope.check_config_identity(receipt, CONFIG, quiet)
    assert len(issues) == 4, issues
    assert all("states no" in issue for issue in issues)


# --------------------------------------------------------------------------
# reducer admission
# --------------------------------------------------------------------------


def _member(name: str, digests: dict) -> dict:
    return {"id": name, "config_digest": f"digest-{name}",
            "input_sha256": digests}


PRESERVED = {"wrfinput_d01": "a" * 64, "wrfbdy_d01": "b" * 64}


def test_the_reducer_admits_only_the_preserved_input_set():
    admitted = chaos_envelope.admit_members(
        [_member("m0", dict(PRESERVED)), _member("m1", dict(PRESERVED))],
        required_input_sha256=PRESERVED)
    assert [member["id"] for member in admitted] == ["m0", "m1"]


def test_the_reducer_refuses_a_member_from_another_base_state():
    """CONTROL: one wrong input hash, and nothing is reduced."""
    wrong = dict(PRESERVED, wrfbdy_d01="c" * 64)
    with pytest.raises(ValueError, match=r"m1: input hashes differ"):
        chaos_envelope.admit_members(
            [_member("m0", dict(PRESERVED)), _member("m1", wrong)],
            required_input_sha256=PRESERVED)
    with pytest.raises(ValueError, match="records no consumed-input hashes"):
        chaos_envelope.admit_members(
            [{"id": "m2"}], required_input_sha256=PRESERVED)


def test_the_band_block_is_reduced_from_the_receipt(ensemble, tmp_path):
    _root, _registration, receipt = ensemble
    members = [_member(f"m{index}", dict(PRESERVED)) for index in range(3)]
    (tmp_path / "members.json").write_text(
        json.dumps({"members": members}), encoding="utf-8")
    (tmp_path / "required.json").write_text(
        json.dumps(PRESERVED), encoding="utf-8")
    (tmp_path / "pairs.json").write_text("pair scores\n", encoding="utf-8")
    code = certify_band_from_ensemble.main([
        "--receipt", str(_root / "envelope.json"),
        "--members", str(tmp_path / "members.json"),
        "--required-inputs", str(tmp_path / "required.json"),
        "--pair-scores", str(tmp_path / "pairs.json"),
        "--output", str(tmp_path / "band.json")])
    assert code == 0
    band = json.loads((tmp_path / "band.json").read_text(encoding="utf-8"))
    assert band["provenance"] == "wrf-ensemble-envelope"
    assert band["scope"] == "internal"
    assert band["member_count"] == 3
    assert band["registration_sha256"] == receipt["registration_sha256"]
    assert band["interval_statistic"].startswith("E95.0 = nearest-rank")
    assert len(band["rows"]) == len(receipt["rows"])
    assert set(band["member_config_digest"]) == {"m0", "m1", "m2"}


def test_the_reducer_exits_non_zero_when_a_member_is_refused(ensemble, tmp_path):
    _root, _registration, _receipt = ensemble
    members = [_member("m0", dict(PRESERVED)),
               _member("m1", dict(PRESERVED, wrfbdy_d01="d" * 64))]
    (tmp_path / "members.json").write_text(
        json.dumps({"members": members}), encoding="utf-8")
    (tmp_path / "required.json").write_text(
        json.dumps(PRESERVED), encoding="utf-8")
    (tmp_path / "pairs.json").write_text("pair scores\n", encoding="utf-8")
    assert certify_band_from_ensemble.main([
        "--receipt", str(_root / "envelope.json"),
        "--members", str(tmp_path / "members.json"),
        "--required-inputs", str(tmp_path / "required.json"),
        "--pair-scores", str(tmp_path / "pairs.json"),
        "--output", str(tmp_path / "band.json")]) == 1
    assert not (tmp_path / "band.json").exists()


def test_the_generic_modules_name_no_case():
    for module in (chaos_envelope, spectral, matched_wrfout_envelope,
                   certify_band_from_ensemble):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        for token in ("real74", "1974", "ohio", "hrrr", "oklahoma"):
            assert token not in source, (module.__name__, token)
