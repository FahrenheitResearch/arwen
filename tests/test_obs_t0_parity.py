"""Pins for the campaign's t=0 parity receipt.

The receipt composes the tree's full-state digest and changes one thing:
across two engines the digest's bit-parity ceilings are recorded rather
than applied.  These pins hold that change honest:

* a large cross-engine gap is still a receipt, not a failure -- the gap is
  the finding;
* the bit-parity verdict is carried verbatim and marked non-binding, so it
  can neither be lost nor mistaken for the verdict;
* what *is* gated is that this is a t=0 receipt at all: a frame at the
  wrong lead is refused; and
* the receipt has to name how the two engines came to share a state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.verify.obs.t0_parity import (
    IC_ROUTES,
    SCHEMA_ID,
    build_t0_parity_receipt,
    canonical_json,
    check_initial_frames,
    make_registration,
    registration_sha256,
    render_markdown,
    summarise_gap,
)
from gpuwm.verify.t0_state_digest import CARRIER_GROUPS
from tools.obs_t0_parity_receipt import EXIT_COVERAGE, EXIT_OK
from tools.obs_t0_parity_receipt import main as cli_main

netCDF4 = pytest.importorskip("netCDF4")

COMMIT = "0" * 40

_NY, _NX, _NZ, _NS = 5, 4, 3, 2


def write_state_frame(path: Path, *, valid_time: str,
                      simulation_start: str,
                      title: str = " SYNTHETIC ENGINE",
                      offsets: dict[str, float] | None = None,
                      omit: tuple[str, ...] = ()) -> Path:
    """A frame carrying one carrier from each required group."""
    offsets = offsets or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        dataset.createDimension("Time", None)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("west_east", _NX)
        dataset.createDimension("south_north", _NY)
        dataset.createDimension("bottom_top", _NZ)
        dataset.createDimension("soil_layers_stag", _NS)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.array(list(valid_time), "S1")

        surface = ("Time", "south_north", "west_east")
        column = ("Time", "bottom_top", "south_north", "west_east")
        soil = ("Time", "soil_layers_stag", "south_north", "west_east")
        plan = {
            "T": (column, 300.0),
            "MU": (surface, 8000.0),
            "QVAPOR": (column, 0.008),
            "T2": (surface, 290.0),
            "PSFC": (surface, 98000.0),
            "TSLB": (soil, 285.0),
        }
        for name, (dimensions, base) in plan.items():
            if name in omit:
                continue
            variable = dataset.createVariable(name, "f4", dimensions)
            shape = tuple(1 if d == "Time" else len(dataset.dimensions[d])
                          for d in dimensions)
            variable[:] = np.full(shape, base + offsets.get(name, 0.0),
                                  dtype="f4")

        dataset.setncattr("TITLE", title)
        dataset.setncattr("SIMULATION_START_DATE", simulation_start)
        dataset.setncattr("MAP_PROJ", np.int32(1))
        dataset.setncattr("DX", np.float32(3000.0))
        dataset.setncattr("DY", np.float32(3000.0))
    finally:
        dataset.close()
    return path


def stage_pair(root: Path, *, valid_time: str = "2021-12-10_12:00:00",
               simulation_start: str = "2021-12-10_12:00:00",
               frame_stamp: str = "2021-12-10_12_00_00",
               offsets: dict[str, float] | None = None,
               omit: tuple[str, ...] = ()) -> tuple[Path, Path]:
    candidate = root / "candidate"
    reference = root / "reference"
    write_state_frame(candidate / f"wrfout_d01_{frame_stamp}",
                      valid_time=valid_time,
                      simulation_start=simulation_start,
                      title=" ENGINE A", offsets=offsets, omit=omit)
    write_state_frame(reference / f"wrfout_d01_{frame_stamp}",
                      valid_time=valid_time,
                      simulation_start=simulation_start,
                      title=" ENGINE B")
    return candidate, reference


# --------------------------------------------------------------------------
# the registration
# --------------------------------------------------------------------------


def test_the_registration_says_the_gap_is_not_gated():
    registration = make_registration()
    assert registration["schema"] == SCHEMA_ID
    assert registration["gap"]["gated"] is False
    assert registration["bit_parity_gate"]["binding"] is False
    assert registration["coverage_gate"]["initial_frame_required"] is True


def test_the_registration_requires_every_required_carrier_group():
    required = {group.name for group in CARRIER_GROUPS if group.required}
    listed = set(make_registration()["coverage_gate"]
                 ["required_carrier_groups"])
    assert listed == required


def test_the_registration_hash_moves_when_the_gate_moves():
    baseline = registration_sha256(make_registration())
    moved = make_registration()
    moved["coverage_gate"]["initial_frame_required"] = False
    assert registration_sha256(moved) != baseline


def test_the_registration_is_canonical_json():
    assert json.loads(canonical_json(make_registration())) == \
        make_registration()


# --------------------------------------------------------------------------
# the frames
# --------------------------------------------------------------------------


def test_a_time_is_compared_by_its_meaning_not_its_spelling(tmp_path):
    """``Times`` uses colons and the start attribute may not."""
    candidate, reference = stage_pair(tmp_path)
    frames = check_initial_frames(candidate, reference)
    assert frames["all_initial"] is True
    assert frames["not_initial_domains"] == []
    sides = frames["domains"]["d01"]["sides"]
    assert sides["candidate"]["simulation_start"] == "2021-12-10_12_00_00"
    assert sides["candidate"]["frame_valid_time"] == "2021-12-10_12_00_00"


def test_a_later_frame_is_not_an_initial_frame(tmp_path):
    candidate, reference = stage_pair(
        tmp_path, valid_time="2021-12-10_14:00:00",
        frame_stamp="2021-12-10_14_00_00")
    frames = check_initial_frames(candidate, reference)
    assert frames["all_initial"] is False
    assert frames["not_initial_domains"] == ["d01"]


# --------------------------------------------------------------------------
# the receipt
# --------------------------------------------------------------------------


def test_an_identical_pair_is_measured(tmp_path):
    candidate, reference = stage_pair(tmp_path)
    receipt = build_t0_parity_receipt(candidate, reference,
                                      evaluator_commit=COMMIT,
                                      ic_route="exporter-parity")
    assert receipt["verdict"] == "MEASURED"
    assert receipt["coverage_verdict"] == "PASS"
    assert receipt["coverage_reasons"] == []
    assert receipt["comparison_kind"] == "cross-engine"
    assert receipt["ic_route"] == "exporter-parity"
    assert receipt["bit_parity_gate"]["verdict"] == "PASS"
    assert receipt["bit_parity_gate"]["binding"] is False


def test_a_large_gap_is_still_a_measurement_not_a_failure(tmp_path):
    """The point of the receipt: publish the gap, never grade it."""
    candidate, reference = stage_pair(tmp_path, offsets={"T2": 3.0})
    receipt = build_t0_parity_receipt(candidate, reference,
                                      evaluator_commit=COMMIT,
                                      ic_route="exporter-parity")
    assert receipt["bit_parity_gate"]["verdict"] == "FAIL"
    assert receipt["verdict"] == "MEASURED"
    assert receipt["coverage_verdict"] == "PASS"
    surface = receipt["gap"]["d01"]["surface"]
    assert surface["max_abs_diff"] == pytest.approx(3.0, abs=1e-3)
    assert surface["max_abs_diff_variable"] == "T2"
    assert surface["bit_parity_verdict"] == "FAIL"


def test_the_gap_names_the_variable_that_carried_it(tmp_path):
    candidate, reference = stage_pair(tmp_path,
                                      offsets={"T2": 1.0, "PSFC": 50.0})
    receipt = build_t0_parity_receipt(candidate, reference,
                                      evaluator_commit=COMMIT,
                                      ic_route="exporter-parity")
    surface = receipt["gap"]["d01"]["surface"]
    assert surface["max_abs_diff_variable"] == "PSFC"


def test_a_frame_at_the_wrong_lead_is_refused(tmp_path):
    candidate, reference = stage_pair(
        tmp_path, valid_time="2021-12-10_14:00:00",
        frame_stamp="2021-12-10_14_00_00")
    receipt = build_t0_parity_receipt(candidate, reference,
                                      evaluator_commit=COMMIT,
                                      ic_route="exporter-parity")
    assert receipt["verdict"] == "REFUSED"
    assert receipt["coverage_verdict"] == "FAIL"
    assert any("not the initial frame" in reason
               for reason in receipt["coverage_reasons"])


def test_a_missing_required_carrier_group_is_refused(tmp_path):
    candidate, reference = stage_pair(tmp_path, omit=("TSLB",))
    receipt = build_t0_parity_receipt(candidate, reference,
                                      evaluator_commit=COMMIT,
                                      ic_route="exporter-parity")
    assert receipt["verdict"] == "REFUSED"
    assert any("soil" in reason for reason in receipt["coverage_reasons"])


def test_an_empty_staging_is_refused(tmp_path):
    (tmp_path / "candidate").mkdir()
    (tmp_path / "reference").mkdir()
    receipt = build_t0_parity_receipt(tmp_path / "candidate",
                                      tmp_path / "reference",
                                      evaluator_commit=COMMIT,
                                      ic_route="wps-real")
    assert receipt["verdict"] == "REFUSED"
    assert any("no frame pair" in reason
               for reason in receipt["coverage_reasons"])


@pytest.mark.parametrize("route", IC_ROUTES)
def test_every_registered_route_is_accepted(tmp_path, route):
    candidate, reference = stage_pair(tmp_path)
    receipt = build_t0_parity_receipt(candidate, reference,
                                      evaluator_commit=COMMIT,
                                      ic_route=route)
    assert receipt["ic_route"] == route


def test_a_receipt_refuses_an_unnamed_route(tmp_path):
    candidate, reference = stage_pair(tmp_path)
    with pytest.raises(ValueError, match="ic_route"):
        build_t0_parity_receipt(candidate, reference,
                                evaluator_commit=COMMIT,
                                ic_route="whatever-we-did")


def test_summarise_gap_ignores_a_variable_with_no_measurement():
    digest = {
        "domains": {
            "d01": {
                "groups": {
                    "surface": {
                        "status": "scored",
                        "scored_arrays": 2,
                        "verdict": "FAIL",
                        "variables": {
                            "T2": {"max_abs_diff": 0.25},
                            "XLAND": {"max_abs_diff": None},
                        },
                    },
                },
            },
        },
    }
    summary = summarise_gap(digest)
    assert summary["d01"]["surface"]["max_abs_diff"] == pytest.approx(0.25)
    assert summary["d01"]["surface"]["max_abs_diff_variable"] == "T2"


# --------------------------------------------------------------------------
# the rendering
# --------------------------------------------------------------------------


def test_the_markdown_marks_the_bit_parity_gate_non_binding(tmp_path):
    candidate, reference = stage_pair(tmp_path, offsets={"T2": 3.0})
    receipt = build_t0_parity_receipt(candidate, reference,
                                      evaluator_commit=COMMIT,
                                      ic_route="exporter-parity")
    text = render_markdown(receipt)
    assert "**not binding**" in text
    assert "**MEASURED**" in text
    assert receipt["registration_sha256"] in text


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


def test_the_cli_exits_ok_on_a_gap_it_merely_measured(tmp_path):
    """A large gap must not become a non-zero status."""
    candidate, reference = stage_pair(tmp_path, offsets={"T2": 3.0})
    receipt_path = tmp_path / "out" / "receipt.json"
    status = cli_main([
        "--candidate-dir", str(candidate),
        "--reference-dir", str(reference),
        "--ic-route", "exporter-parity",
        "--out-json", str(receipt_path),
        "--out-md", str(tmp_path / "out" / "receipt.md"),
        "--evaluator-commit", COMMIT,
    ])
    assert status == EXIT_OK
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "MEASURED"
    assert receipt["bit_parity_gate"]["verdict"] == "FAIL"


def test_the_cli_refuses_a_frame_at_the_wrong_lead(tmp_path):
    candidate, reference = stage_pair(
        tmp_path, valid_time="2021-12-10_14:00:00",
        frame_stamp="2021-12-10_14_00_00")
    status = cli_main([
        "--candidate-dir", str(candidate),
        "--reference-dir", str(reference),
        "--ic-route", "exporter-parity",
        "--out-json", str(tmp_path / "receipt.json"),
        "--evaluator-commit", COMMIT,
    ])
    assert status == EXIT_COVERAGE
