"""Pins for the battery's cross-reader receipt.

The battery reads two engines' history tapes through one science core.
These pins hold the properties that make the receipt worth anything:

* the control really is independent -- point the science core at a value
  that is not in the file and the receipt says FAIL;
* the tolerances are in the registration, so moving one moves the hash
  every receipt carries;
* a quantity the frame does not store is reported as unavailable rather
  than silently dropped from a PASS;
* pairing compares what the two writers mean, not how wide they store it,
  and it still refuses two different domains; and
* the reader is pinned by distribution, so a receipt cannot be produced by
  an unnamed reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.verify.obs.cross_reader import (
    PAIRING_FIELD_ABS_TOL_DEGREES,
    REGISTERED_QUANTITIES,
    ROTATION_ADJUDICATION,
    SCIENCE_CORE_VERSION,
    SCOPE_BATTERY_CASE,
    SCOPE_READER_QUALIFICATION,
    STORED_ROTATION_ANGLE,
    ReaderQuantity,
    build_cross_reader_receipt,
    canonical_json,
    check_science_core_pin,
    compare_arrays,
    make_registration,
    registration_sha256,
    render_markdown,
    score_side,
)
from tools.obs_cross_reader_receipt import (
    EXIT_COVERAGE,
    EXIT_OK,
    EXIT_VERDICT,
)
from tools.obs_cross_reader_receipt import main as cli_main

netCDF4 = pytest.importorskip("netCDF4")

COMMIT = "0" * 40

_NY, _NX, _NZ = 6, 5, 4


def write_frame(path: Path, *, valid_time: str = "2021-12-10_12:00:00",
                title: str = " SYNTHETIC WRITER",
                float_attribute_dtype=np.float32,
                lat_offset: float = 0.0, lon_offset: float = 0.0,
                stand_lon: float = -97.0,
                omit: tuple[str, ...] = (),
                seed: int = 0) -> Path:
    """A minimal ARW-convention frame the science core will open.

    ``T`` is present because the science core reads it to find the grid
    dimensions; everything else here is what some registered control or
    pairing check needs.
    """
    dataset = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        dataset.createDimension("Time", None)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("west_east", _NX)
        dataset.createDimension("south_north", _NY)
        dataset.createDimension("bottom_top", _NZ)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.array(list(valid_time), "S1")

        rng = np.random.default_rng(seed)
        surface = ("Time", "south_north", "west_east")
        column = ("Time", "bottom_top", "south_north", "west_east")
        for name in ("T2", "U10", "V10", "SINALPHA", "COSALPHA"):
            if name in omit:
                continue
            variable = dataset.createVariable(name, "f4", surface)
            variable[:] = rng.standard_normal((1, _NY, _NX)).astype("f4")
        latitude = dataset.createVariable("XLAT", "f4", surface)
        latitude[:] = (np.linspace(39.0, 40.0, _NY * _NX)
                       .reshape(1, _NY, _NX).astype("f4") + lat_offset)
        longitude = dataset.createVariable("XLONG", "f4", surface)
        longitude[:] = (np.linspace(-98.0, -97.0, _NY * _NX)
                        .reshape(1, _NY, _NX).astype("f4") + lon_offset)
        for name in ("REFL_10CM", "T"):
            if name in omit:
                continue
            variable = dataset.createVariable(name, "f4", column)
            variable[:] = rng.standard_normal((1, _NZ, _NY, _NX)).astype("f4")

        dataset.setncattr("TITLE", title)
        dataset.setncattr("SIMULATION_START_DATE", valid_time)
        dataset.setncattr("MAP_PROJ", np.int32(1))
        for name, value in (("DX", 3000.0), ("DY", 3000.0),
                            ("TRUELAT1", 30.0), ("TRUELAT2", 60.0),
                            ("STAND_LON", stand_lon)):
            dataset.setncattr(name, float_attribute_dtype(value))
        dataset.setncattr("WEST-EAST_GRID_DIMENSION", np.int32(_NX + 1))
        dataset.setncattr("SOUTH-NORTH_GRID_DIMENSION", np.int32(_NY + 1))
        dataset.setncattr("BOTTOM-TOP_GRID_DIMENSION", np.int32(_NZ + 1))
    finally:
        dataset.close()
    return path


class TruthfulReader:
    """A stand-in science core that answers from the file it was given.

    It exists so the harness's own behaviour can be pinned without a real
    science core, and so :class:`WrongReader` has something to differ from.
    """

    def __init__(self, *, version: str | None = SCIENCE_CORE_VERSION) -> None:
        self._version = version

    def provenance(self) -> dict[str, object]:
        return {
            "import_name": "wrf",
            "distribution": "wrf-rust",
            "distribution_version": self._version,
            "module_file": "<stand-in>",
            "module_version_attribute": self._version,
            "installed_from": None,
            "installed_editable": None,
        }

    def open(self, path: Path):
        dataset = netCDF4.Dataset(str(path))
        fields = {}
        for name in ("T2", "U10", "V10", "SINALPHA", "COSALPHA",
                     "REFL_10CM"):
            variable = dataset.variables.get(name)
            if variable is None:
                continue
            variable.set_auto_mask(False)
            fields[name] = np.asarray(variable[0], dtype=np.float64)
        dataset.close()
        return fields

    def quantity(self, handle, quantity: ReaderQuantity) -> np.ndarray:
        missing = [name for name in quantity.control_inputs
                   if name not in handle]
        if missing:
            raise KeyError(f"frame does not store {missing}")
        if quantity.name == "t2":
            return handle["T2"]
        if quantity.name == "uvmet10_u":
            return (handle["U10"] * handle["COSALPHA"]
                    - handle["V10"] * handle["SINALPHA"])
        if quantity.name == "uvmet10_v":
            return (handle["U10"] * handle["SINALPHA"]
                    + handle["V10"] * handle["COSALPHA"])
        if quantity.name == "refl_10cm_column_max":
            return np.max(handle["REFL_10CM"], axis=0)
        raise KeyError(quantity.name)


class WrongReader(TruthfulReader):
    """A science core that is wrong about one quantity, on purpose."""

    def __init__(self, *, wrong: str, delta: float = 1.0) -> None:
        super().__init__()
        self._wrong = wrong
        self._delta = delta

    def quantity(self, handle, quantity: ReaderQuantity) -> np.ndarray:
        values = super().quantity(handle, quantity)
        if quantity.name == self._wrong:
            perturbed = np.array(values, dtype=np.float64, copy=True)
            perturbed.flat[0] += self._delta
            return perturbed
        return values


@pytest.fixture()
def pair(tmp_path: Path) -> dict[str, Path]:
    return {
        "left": write_frame(tmp_path / "wrfout_d01_2021-12-10_12_00_00",
                            title=" WRITER ONE", seed=1),
        "right": write_frame(tmp_path / "right"
                             / "wrfout_d01_2021-12-10_12_00_00",
                             title=" WRITER TWO", seed=2,
                             float_attribute_dtype=np.float64),
    }


@pytest.fixture(autouse=True)
def _right_directory(tmp_path: Path):
    (tmp_path / "right").mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# the registration
# --------------------------------------------------------------------------


def test_every_registered_quantity_carries_a_tolerance_and_a_reason():
    for quantity in REGISTERED_QUANTITIES:
        assert quantity.abs_tol >= 0.0
        assert quantity.tolerance_reason.strip(), quantity.name
        assert quantity.kind in {"passthrough", "reduction", "derived"}


def test_exact_operations_are_registered_at_zero_tolerance():
    """A passthrough read and a column maximum introduce no arithmetic.

    If either ever acquires a nonzero tolerance, the receipt stops being
    able to tell a reader disagreement from rounding.
    """
    for quantity in REGISTERED_QUANTITIES:
        if quantity.kind in {"passthrough", "reduction"}:
            assert quantity.abs_tol == 0.0, quantity.name


def test_the_registration_hash_moves_when_a_tolerance_moves():
    baseline = registration_sha256()
    moved = make_registration()
    moved["quantities"]["uvmet10_u"]["abs_tol"] = 1.0
    assert registration_sha256(moved) != baseline


def test_the_registration_is_canonical_json():
    text = canonical_json(make_registration())
    assert json.loads(text) == make_registration()


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------


def test_compare_arrays_sees_a_single_perturbed_element():
    left = np.zeros((4, 3))
    right = np.zeros((4, 3))
    right[2, 1] = 0.5
    metrics = compare_arrays(left, right)
    assert metrics["max_abs_diff"] == pytest.approx(0.5)
    assert metrics["differing_elements"] == 1
    assert metrics["elements"] == 12


def test_compare_arrays_counts_a_one_sided_nan_rather_than_averaging_it_away():
    left = np.zeros((3,))
    right = np.array([0.0, np.nan, 0.0])
    metrics = compare_arrays(left, right)
    assert metrics["nonfinite_disagreements"] == 1
    assert metrics["max_abs_diff"] == pytest.approx(0.0)


def test_a_shape_disagreement_is_not_a_pass():
    metrics = compare_arrays(np.zeros((3, 2)), np.zeros((2, 3)))
    assert metrics["shape_agrees"] is False
    assert metrics["max_abs_diff"] is None


# --------------------------------------------------------------------------
# the control is independent: a wrong reader must fail
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wrong", [quantity.name
                                   for quantity in REGISTERED_QUANTITIES])
def test_a_reader_that_is_wrong_about_one_quantity_fails(pair, wrong):
    """The control must be able to catch the thing it exists to catch."""
    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT,
                                         reader=WrongReader(wrong=wrong))
    assert receipt["verdict"] == "FAIL"
    assert receipt["readers_agree"] is False
    for side in receipt["sides"].values():
        assert side["quantities"][wrong]["verdict"] == "FAIL"
        for name, entry in side["quantities"].items():
            if name != wrong:
                assert entry["verdict"] == "PASS", name


def test_a_truthful_reader_passes_and_the_pair_is_recorded(pair):
    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    assert receipt["verdict"] == "PASS"
    assert receipt["pairing"]["same_case"] is True
    assert receipt["pairing"]["distinct_writers"] is True
    assert sorted(receipt["sides"]) == ["left", "right"]
    assert {entry["side"] for entry in receipt["inputs"]} == {"left", "right"}
    for entry in receipt["inputs"]:
        assert len(entry["sha256"]) == 64


def test_an_unnamed_case_makes_the_receipt_say_what_it_is_evidence_of(pair):
    """A receipt on some matched pair must not read as a case receipt."""
    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    assert receipt["battery_case_id"] is None
    assert receipt["scope"] == SCOPE_READER_QUALIFICATION
    assert "claims nothing about a" in render_markdown(receipt)


def test_a_named_case_is_carried_into_the_receipt(pair):
    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT,
                                         reader=TruthfulReader(),
                                         case_id="B-99")
    assert receipt["battery_case_id"] == "B-99"
    assert receipt["scope"] == SCOPE_BATTERY_CASE
    assert "B-99" in render_markdown(receipt)


def test_the_alternate_rotation_is_recorded_and_not_gated(pair):
    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    diagnostic = (receipt["sides"]["left"]["quantities"]["uvmet10_u"]
                  ["diagnostic_alternate_control"])
    assert diagnostic["gated"] is False
    assert diagnostic["metrics"]["max_abs_diff"] > 0.0
    assert receipt["verdict"] == "PASS"


def test_the_sign_control_is_not_carried_as_a_rival_convention(pair):
    """The rotation sign is adjudicated; the control prices an error in it.

    The first receipt this instrument issued described the opposite-sign
    result as an open convention question.  It was not one -- the stored
    angle inside the opposite formula shape is a rotation nothing ships.
    These assertions keep the receipt from drifting back to the old
    reading.
    """
    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    rotation = receipt["registration"]["earth_rotation"]
    assert rotation["status"] == "settled"
    assert rotation["settled_by"] == ROTATION_ADJUDICATION
    assert rotation["stored_angle"] == STORED_ROTATION_ANGLE
    assert "MAP_PROJ = 1" in rotation["scope"]
    for side in receipt["sides"].values():
        for name in ("uvmet10_u", "uvmet10_v"):
            diagnostic = (side["quantities"][name]
                          ["diagnostic_alternate_control"])
            assert diagnostic["is_rival_convention"] is False
            assert diagnostic["settled_by"] == ROTATION_ADJUDICATION
            assert "sign" in diagnostic["means"]
    text = render_markdown(receipt)
    assert "Not a rival convention" in text
    assert "cost of a sign error" in text


def test_the_adjudication_receipt_this_registration_cites_exists():
    """A citation that does not resolve is worse than none."""
    root = Path(__file__).resolve().parents[1]
    assert (root / ROTATION_ADJUDICATION).is_file(), ROTATION_ADJUDICATION


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


def test_a_quantity_the_frame_does_not_store_is_unavailable_not_absent(
        tmp_path):
    (tmp_path / "right").mkdir(exist_ok=True)
    sides = {
        "left": write_frame(tmp_path / "wrfout_d01_2021-12-10_12_00_00",
                            omit=("REFL_10CM",)),
        "right": write_frame(
            tmp_path / "right" / "wrfout_d01_2021-12-10_12_00_00"),
    }
    receipt = build_cross_reader_receipt(sides, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    entry = receipt["sides"]["left"]["quantities"]["refl_10cm_column_max"]
    assert entry["status"] == "unavailable"
    assert "REFL_10CM" in entry["reason"]
    assert receipt["unavailable_quantities"] == [
        "left:refl_10cm_column_max"]
    assert receipt["verdict"] == "FAIL"


def test_a_science_core_refusal_is_recorded_rather_than_raised(tmp_path):
    class RefusingReader(TruthfulReader):
        def quantity(self, handle, quantity):
            raise RuntimeError("unknown variable")

    path = write_frame(tmp_path / "wrfout_d01_2021-12-10_12_00_00")
    side = score_side(path, reader=RefusingReader())
    for entry in side["quantities"].values():
        assert entry["status"] == "unavailable"
        assert "unknown variable" in entry["reason"]


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------


def test_pairing_ignores_the_width_a_writer_stores_a_projection_at(pair):
    """One writer stores FP32 attributes, the other FP64, same projection."""
    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    assert receipt["pairing"]["same_case"] is True
    assert receipt["pairing"]["differing_attributes"] == {}


def test_pairing_refuses_a_different_projection(tmp_path):
    (tmp_path / "right").mkdir(exist_ok=True)
    sides = {
        "left": write_frame(tmp_path / "wrfout_d01_2021-12-10_12_00_00"),
        "right": write_frame(
            tmp_path / "right" / "wrfout_d01_2021-12-10_12_00_00",
            stand_lon=-95.0),
    }
    receipt = build_cross_reader_receipt(sides, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    assert receipt["pairing"]["same_case"] is False
    assert "STAND_LON" in receipt["pairing"]["differing_attributes"]
    assert receipt["verdict"] == "FAIL"


def test_pairing_refuses_a_displaced_domain(tmp_path):
    """A shifted grid is a different case however the attributes read."""
    (tmp_path / "right").mkdir(exist_ok=True)
    sides = {
        "left": write_frame(tmp_path / "wrfout_d01_2021-12-10_12_00_00"),
        "right": write_frame(
            tmp_path / "right" / "wrfout_d01_2021-12-10_12_00_00",
            lat_offset=10 * PAIRING_FIELD_ABS_TOL_DEGREES),
    }
    receipt = build_cross_reader_receipt(sides, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    georeference = receipt["pairing"]["georeference"]
    assert georeference["XLAT"]["verdict"] == "FAIL"
    assert georeference["XLONG"]["verdict"] == "PASS"
    assert receipt["pairing"]["same_case"] is False
    assert receipt["verdict"] == "FAIL"


def test_pairing_refuses_a_different_valid_time(tmp_path):
    (tmp_path / "right").mkdir(exist_ok=True)
    sides = {
        "left": write_frame(tmp_path / "wrfout_d01_2021-12-10_12_00_00"),
        "right": write_frame(
            tmp_path / "right" / "wrfout_d01_2021-12-10_13_00_00",
            valid_time="2021-12-10_13:00:00"),
    }
    receipt = build_cross_reader_receipt(sides, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    assert "valid_time" in receipt["pairing"]["differing_attributes"]
    assert receipt["verdict"] == "FAIL"


# --------------------------------------------------------------------------
# the reader pin
# --------------------------------------------------------------------------


def test_an_unpinned_reader_fails_the_verdict(pair):
    receipt = build_cross_reader_receipt(
        pair, evaluator_commit=COMMIT,
        reader=TruthfulReader(version="0.0.1-not-the-pin"))
    assert receipt["science_core_pin"]["status"] == "mismatch"
    assert receipt["readers_agree"] is True
    assert receipt["verdict"] == "FAIL"


def test_an_unnameable_reader_fails_the_verdict(pair):
    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT,
                                         reader=TruthfulReader(version=None))
    assert receipt["science_core_pin"]["status"] == "unresolved"
    assert receipt["verdict"] == "FAIL"


def test_a_stale_version_attribute_is_noted_and_not_gated():
    pin = check_science_core_pin({
        "distribution_version": SCIENCE_CORE_VERSION,
        "module_version_attribute": "0.0.0",
    })
    assert pin["ok"] is True
    assert "0.0.0" in pin["version_attribute_note"]


def test_a_receipt_refuses_an_evaluator_it_cannot_name(pair):
    with pytest.raises(ValueError, match="40-hex"):
        build_cross_reader_receipt(pair, evaluator_commit="not-a-commit",
                                   reader=TruthfulReader())


# --------------------------------------------------------------------------
# the rendering
# --------------------------------------------------------------------------


def test_the_markdown_carries_the_verdict_and_the_diagnostics(pair):
    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT,
                                         reader=TruthfulReader())
    text = render_markdown(receipt)
    assert "**PASS**" in text
    assert "never gated" in text
    assert receipt["registration_sha256"] in text
    for quantity in REGISTERED_QUANTITIES:
        assert quantity.name in text


# --------------------------------------------------------------------------
# the mandated science core itself
# --------------------------------------------------------------------------


def test_the_mandated_science_core_agrees_with_the_control(pair):
    """The reader the battery will actually use, on a real frame it reads."""
    pytest.importorskip("wrf", reason="the mandated science core, wrf-rust")

    receipt = build_cross_reader_receipt(pair, evaluator_commit=COMMIT)
    for label, side in receipt["sides"].items():
        for name, entry in side["quantities"].items():
            assert entry["status"] == "scored", (label, name)
            assert entry["metrics"]["max_abs_diff"] <= entry["abs_tol"], (
                label, name)
    assert receipt["readers_agree"] is True


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


def test_the_cli_writes_a_receipt_and_reports_agreement(pair, tmp_path):
    pytest.importorskip("wrf", reason="the mandated science core, wrf-rust")

    receipt_path = tmp_path / "out" / "receipt.json"
    table_path = tmp_path / "out" / "receipt.md"
    status = cli_main([
        "--frame", f"left={pair['left']}",
        "--frame", f"right={pair['right']}",
        "--out-json", str(receipt_path),
        "--out-md", str(table_path),
        "--evaluator-commit", COMMIT,
    ])
    assert status == EXIT_OK
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "PASS"
    assert receipt["evaluator_commit"] == COMMIT
    assert table_path.read_text(encoding="utf-8").startswith("# cross-reader")


def test_the_cli_reports_a_failing_verdict_with_its_own_status(tmp_path):
    pytest.importorskip("wrf", reason="the mandated science core, wrf-rust")

    (tmp_path / "right").mkdir(exist_ok=True)
    left = write_frame(tmp_path / "wrfout_d01_2021-12-10_12_00_00")
    right = write_frame(tmp_path / "right"
                        / "wrfout_d01_2021-12-10_13_00_00",
                        valid_time="2021-12-10_13:00:00")
    status = cli_main([
        "--frame", f"left={left}", "--frame", f"right={right}",
        "--out-json", str(tmp_path / "receipt.json"),
        "--evaluator-commit", COMMIT,
    ])
    assert status == EXIT_VERDICT


def test_the_cli_refuses_a_frame_that_is_not_there(tmp_path):
    status = cli_main([
        "--frame", f"left={tmp_path / 'nothing_here'}",
        "--out-json", str(tmp_path / "receipt.json"),
        "--evaluator-commit", COMMIT,
    ])
    assert status == EXIT_COVERAGE
    assert not (tmp_path / "receipt.json").exists()
