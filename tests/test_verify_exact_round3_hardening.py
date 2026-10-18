"""Mechanical probes for the round-3 scorer defects. CPU only."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from test_verify_exact_scoring_holes import (
    _frames, _run, _scalar, _write_recording, _write_woof,
)


@pytest.mark.parametrize("error", ["ValueError: shape mismatch", "OSError: failed read",
                                   "NotImplementedError: missing code branch"])
@pytest.mark.parametrize("stored_status", ["CRASH", "REFUSED"])
def test_code_error_before_frame_zero_is_a_crash(tmp_path, error, stored_status):
    _write_recording(tmp_path / "rec", {"A001": {"builds": _scalar(_frames(40))}},
                     summary={"A001": "yes"})
    _write_woof(tmp_path / "out", "A001", {}, status=stored_status)
    path = tmp_path / "out/woof/c/A001/hashes.json"
    doc = json.loads(path.read_text())
    doc["woof"]["error"] = error
    path.write_text(json.dumps(doc))
    code, rows = _run(tmp_path)
    assert rows["A001"]["outcome"] == "WOOF-CRASH"
    assert code == 1


def test_legacy_round3_scalar_refusal_retains_its_class():
    from gpuwm.verify_exact.cli import initialisation_refusal
    from gpuwm.verify_exact.compare import RunRecord
    error = ("NotImplementedError: scalar_pblmix=1 requires bl_pbl_physics=1 "
             "(YSU) or 5 (MYNN) and mp_physics=28; the implemented coupling reads "
             "that scheme's exch_h and the Thompson aerosol scalar fields nc/ni/nwfa/nifa.")
    assert initialisation_refusal("REFUSED", error, RunRecord({}, {}))[0] == "REFUSED"
    assert initialisation_refusal("REFUSED", "ConfigurationRefused: invalid input",
                                  RunRecord({}, {0: {}}))[0] == "CRASH"


@pytest.mark.parametrize("status", ["NON-FINITE", "CRASH", "OK"])
def test_nonfinite_referee_step_two_bounds_scoring_at_one(tmp_path, status):
    from gpuwm.verify_exact.recording import Recording
    _write_recording(tmp_path / "rec", {"A001": {"builds": _scalar(
        _frames(41), status=status, last_good=3)}}, summary={"A001": "yes"})
    path = tmp_path / "rec/scalar/c/A001/hashes.json"
    doc = json.loads(path.read_text())
    doc["nonfinite"] = {"U": [2, 3], "QRAIN": [3], "RQIBLTEN": [0]}
    path.write_text(json.dumps(doc))
    _write_woof(tmp_path / "out", "A001", {s: f for s, f in _frames(41).items() if s < 2},
                status="CRASH")
    ref = Recording(tmp_path / "rec").select()[0].references["scalar"]
    assert ref.last_good_step == 1
    code, rows = _run(tmp_path)
    assert rows["A001"]["outcome"] == "UNSOUND-REF"
    assert "non-finite from step 2" in rows["A001"]["referee_soundness"]
    assert rows["A001"]["comparisons"]["scalar"]["missing_steps"] == []
    assert code == 3


def test_only_ok_exits_three_with_no_proof_line(tmp_path, capsys):
    _write_recording(tmp_path / "rec", {"D001": {"stratum": "D", "builds": {
        "strict": ("WRF-REFUSED", 0, None)}}}, summary={"D001": "no"})
    _write_woof(tmp_path / "out", "D001", _frames(42))
    code, rows = _run(tmp_path)
    assert rows["D001"]["outcome"] == "WOOF-ONLY-OK" and code == 3
    assert "no proof: no run or fixture matched a sound referee" in capsys.readouterr().out


def test_hash_only_rescore_does_not_require_disposable_inputs(tmp_path):
    _write_recording(tmp_path / "rec", {"A001": {"inputs": False,
        "builds": _scalar(_frames(44))}}, summary={"A001": "yes"})
    _write_woof(tmp_path / "out", "A001", _frames(44))
    code, rows = _run(tmp_path)
    assert rows["A001"]["outcome"] == "PASS" and code == 0


def test_identity_changes_when_tracked_bytes_change_unstaged(tmp_path, monkeypatch):
    from gpuwm.verify_exact import cli
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    file = tmp_path / "tracked.txt"
    file.write_bytes(b"before\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "gpuwm/verify_exact/cli.py"))
    before = cli._engine_identity()
    file.write_bytes(b"after\n")
    after = cli._engine_identity()
    assert before["index_tree"] == after["index_tree"]
    assert before["tracked_disk_sha256"] != after["tracked_disk_sha256"]
    file.unlink()
    assert cli._engine_identity()["tracked_disk_sha256"] != after["tracked_disk_sha256"]


def test_scalar_consumer_refusal_has_a_named_valueerror_type():
    from gpuwm.config import validate_scalar_pblmix_consumer
    with pytest.raises(ValueError) as caught:
        validate_scalar_pblmix_consumer(SimpleNamespace(scalar_pblmix=1,
                                                       bl_pbl_physics=5, mp_physics=6))
    assert type(caught.value).__name__ == "UnsupportedConfiguration"
    assert "exch_h" in str(caught.value)
    assert isinstance(caught.value, NotImplementedError)


@pytest.mark.parametrize("first,per_frame,last_good", [(0, 1, -1), (1, 1, 0),
                                                      (2, 2, 1), (3, 1, 2)])
def test_nonfinite_steps_are_model_steps_and_step_zero_is_unsound(
        tmp_path, first, per_frame, last_good):
    from gpuwm.verify_exact.recording import Recording
    _write_recording(tmp_path / "rec", {"A001": {"builds": _scalar(
        _frames(43), status="NON-FINITE", last_good=3)}}, summary={"A001": "no"})
    path = tmp_path / "rec/scalar/c/A001/hashes.json"
    doc = json.loads(path.read_text())
    doc["steps_per_frame"] = per_frame
    doc["nonfinite"] = {"U": [first], "QRAIN": [], "RQIBLTEN": [0]}
    path.write_text(json.dumps(doc))
    ref = Recording(tmp_path / "rec").select()[0].references["scalar"]
    assert ref.last_good_step == last_good
    assert ref.first_nonfinite_step == first
    if first == 0:
        _write_woof(tmp_path / "out", "A001", _frames(43))
        code, rows = _run(tmp_path)
        assert rows["A001"]["outcome"] == "UNSOUND-REF" and code == 3


@pytest.mark.parametrize("turb", ["d1k4", "d2k4"])
def test_generator_forbids_the_wrf_blowup(turb):
    from test_combo_sweep_generator import _gen
    from collections import OrderedDict
    g = _gen()
    space = g.Space(OrderedDict(list(g.CORE.items()) + list(g.DYN.items()) + list(g.SUB.items())))
    assert space.solve({"pbl": 0, "turb": turb, "epssm": 0.1}) is None
    assert space.solve({"pbl": 0, "turb": turb, "epssm": 0.5}) is not None


def test_committed_generator_list_has_no_wrf_blowup():
    from test_combo_sweep_generator import HERE
    doc = json.loads((HERE / "combos.json").read_text())
    for row in doc["combos"]:
        f = row["factors"]
        assert not (f["pbl"] == 0 and f["turb"] in ("d1k4", "d2k4")
                    and f["epssm"] == 0.1), row["id"]
    for name, coverage in doc["coverage"].items():
        if "covered" in coverage:
            assert coverage["covered"] == coverage["required"], name
