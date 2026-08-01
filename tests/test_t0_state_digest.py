"""Pins for the full-state t=0 parity digest.

The published matched-run comparator scores eight carriers.  A t=0 claim
built on it is an inference from the surface to the whole state, and the
point of this digest is that the inference is replaced by a measurement.
These pins hold the properties that make the measurement worth anything:

* it scores strictly more than the published comparator, and its coverage
  is recorded rather than assumed;
* its ULP is the one certified definition, and an assertion pointed at a
  different definition fails;
* its ceilings are imported, not typed, so a receipt cannot calibrate its
  own gate;
* the boundary group is probed and both outcomes are representable;
* the gate can be made to fail by perturbing one carrier; and
* nothing in it knows how many domains a case has.
"""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gpuwm.verify.nest_gates import (
    FP32_STATE_EQUIV_MAX_ULPS,
    FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS,
    FP32_STATE_EQUIV_P99_MAX_ULPS,
)
from gpuwm.verify.state_equiv import fp32_signed_ulp
from gpuwm.verify.t0_state_digest import (
    BOUNDARY_GROUP,
    CARRIER_GROUPS,
    SCHEMA_ID,
    build_t0_receipt,
    canonical_json,
    discover_frame_pairs,
    make_t0_registration,
    receipt_records,
    receipt_variables,
    registered_variables,
    registration_sha256,
    render_markdown,
    score_variables,
)
from tools.matched_wrfout_stream_compare import _FIELDS as PUBLISHED_FIELDS
from tools.matched_wrfout_t0_state_digest import (
    EXIT_COVERAGE,
    EXIT_OK,
    EXIT_VERDICT,
)
from tools.matched_wrfout_t0_state_digest import main as digest_main

REPO_ROOT = Path(__file__).resolve().parents[1]
DIGEST_MODULE = REPO_ROOT / "gpuwm" / "verify" / "t0_state_digest.py"

#: The published receipt, on the path the release snapshot keeps.
PUBLISHED_RECEIPT_JSON = "gpuwm/data/certification/t0_state_parity_digest.json"

#: Any 40-hex string; the tests never need the real one, and taking it from
#: git would make them depend on the checkout rather than on the code.
FAKE_COMMIT = "0" * 40

NZ, NY, NX, NS = 4, 6, 5, 3


# ---------------------------------------------------------------------------
# synthetic staging
# ---------------------------------------------------------------------------


def _fields(seed: int) -> dict[str, np.ndarray]:
    """A frame carrying at least one variable from every carrier group."""
    rng = np.random.default_rng(seed)

    def arr(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    return {
        # dry_dynamics
        "U": arr(NZ, NY, NX + 1), "V": arr(NZ, NY + 1, NX),
        "W": arr(NZ + 1, NY, NX), "PH": arr(NZ + 1, NY, NX),
        "PHB": arr(NZ + 1, NY, NX), "T": arr(NZ, NY, NX),
        "MU": arr(NY, NX), "MUB": arr(NY, NX),
        "P": arr(NZ, NY, NX), "PB": arr(NZ, NY, NX),
        # moisture
        "QVAPOR": arr(NZ, NY, NX), "QCLOUD": arr(NZ, NY, NX),
        "QRAIN": arr(NZ, NY, NX),
        # surface
        "T2": arr(NY, NX), "Q2": arr(NY, NX), "PSFC": arr(NY, NX),
        "U10": arr(NY, NX), "V10": arr(NY, NX), "TSK": arr(NY, NX),
        "HGT": arr(NY, NX), "XLAND": arr(NY, NX),
        # soil
        "TSLB": arr(NS, NY, NX), "SMOIS": arr(NS, NY, NX),
        "SH2O": arr(NS, NY, NX), "TMN": arr(NY, NX),
        # accumulation
        "RAINC": arr(NY, NX), "RAINNC": arr(NY, NX),
        # diagnostic
        "REFL_10CM": arr(NZ, NY, NX), "SWDOWN": arr(NY, NX),
    }


def _boundary_fields(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for field in ("U", "V", "T", "QVAPOR"):
        for side, width in (("XS", NY), ("XE", NY), ("YS", NX), ("YE", NX)):
            out[f"{field}_B{side}"] = rng.standard_normal(
                (width, NZ, 2)).astype(np.float32)
            out[f"{field}_BT{side}"] = rng.standard_normal(
                (width, NZ, 2)).astype(np.float32)
    return out


def _write_netcdf(path: Path, fields: dict[str, np.ndarray]) -> None:
    import netCDF4

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as dataset:
        dataset.createDimension("Time", None)
        seen: dict[int, str] = {}
        for name in sorted(fields):
            array = np.asarray(fields[name], dtype=np.float32)
            dims = ["Time"]
            for size in array.shape:
                label = seen.setdefault(size, f"dim{size}")
                if label not in dataset.dimensions:
                    dataset.createDimension(label, size)
                dims.append(label)
            variable = dataset.createVariable(name, "f4", tuple(dims))
            variable[0] = array


def _stage(tmp_path: Path, *, domains: tuple[str, ...] = ("d01", "d02"),
           stamp: str = "2020-01-01_00_00_00",
           boundary: str | None = "both",
           perturb=None, drop: tuple[str, ...] = ()) -> tuple[Path, Path]:
    """Write a candidate/reference pair and return the two directories.

    ``perturb`` is called with the candidate field table for each domain so
    a test can move exactly one carrier; ``drop`` removes variables from
    both sides, which is how the deleted-group control is built.
    """
    candidate_dir = tmp_path / "candidate"
    reference_dir = tmp_path / "reference"
    for index, domain in enumerate(domains):
        reference = _fields(index)
        for name in drop:
            reference.pop(name, None)
        candidate = {name: array.copy() for name, array in reference.items()}
        if perturb is not None:
            perturb(candidate, domain)
        _write_netcdf(reference_dir / f"wrfout_{domain}_{stamp}", reference)
        _write_netcdf(candidate_dir / f"wrfout_{domain}_{stamp}", candidate)
    if boundary in {"both", "candidate"}:
        _write_netcdf(candidate_dir / "wrfbdy_d01", _boundary_fields(11))
    if boundary in {"both", "reference"}:
        _write_netcdf(reference_dir / "wrfbdy_d01", _boundary_fields(11))
    return candidate_dir, reference_dir


def _advance(value: np.float32, ulps: int) -> np.float32:
    bits = np.asarray([value], dtype=np.float32).view(np.uint32)
    bits[0] = np.uint32(int(bits[0]) + ulps)
    return bits.view(np.float32)[0]


@pytest.fixture()
def staged(tmp_path: Path) -> tuple[Path, Path]:
    return _stage(tmp_path)


@pytest.fixture()
def receipt(staged) -> dict:
    candidate_dir, reference_dir = staged
    return build_t0_receipt(candidate_dir, reference_dir,
                            evaluator_commit=FAKE_COMMIT)


# ---------------------------------------------------------------------------
# AC-1: coverage is strictly wider than the published comparator, and recorded
# ---------------------------------------------------------------------------


def test_the_registration_claims_strictly_more_than_the_published_comparator():
    """The digest exists because eight carriers are not a state."""
    published = set(PUBLISHED_FIELDS)
    registered = set(registered_variables())
    assert published < registered
    assert registered - published


def test_the_receipt_record_set_is_a_strict_superset_of_the_published_fields(
        receipt):
    scored = receipt_variables(receipt)
    assert set(PUBLISHED_FIELDS) < scored
    records = receipt_records(receipt)
    assert {domain for domain, _name in records} == {"d01", "d02"}


def test_the_published_receipt_scores_what_the_staged_frames_carry():
    """AC-1's superset clause, measured on the real pair rather than assumed.

    On a synthetic pair every comparator field is present on both sides and
    the record set is a strict superset.  On the pair that was actually
    staged it is a superset of seven of the eight: the candidate's t=0
    frame does not carry REFL_10CM at all -- it appears from the first
    forecast frame on -- so there is nothing to score at t=0.  That is a
    property of what was written, so it is pinned here as measured and
    recorded in the receipt rather than papered over.  If a later run
    writes the field at t=0 this test fails, which is the correct way to
    find out.
    """
    receipt = json.loads(
        (REPO_ROOT / PUBLISHED_RECEIPT_JSON).read_text(encoding="utf-8"))
    scored = receipt_variables(receipt)
    assert set(PUBLISHED_FIELDS) - scored == {"REFL_10CM"}
    assert len(scored - set(PUBLISHED_FIELDS)) > 20

    for domain, entry in receipt["domains"].items():
        unscored = entry["groups"]["diagnostic"]["unscored"]
        assert unscored.get("REFL_10CM") == "absent from candidate", domain


def test_every_group_whose_variables_exist_scores_at_least_one_array(receipt):
    for domain, entry in receipt["domains"].items():
        for group in CARRIER_GROUPS:
            present = set(group.variables) & receipt_variables(receipt)
            scored = entry["groups"][group.name]["scored_arrays"]
            if present:
                assert scored, f"{domain}/{group.name} scored nothing"
            assert scored == len(
                [name for name in group.variables
                 if (domain, name) in receipt_records(receipt)])


def test_the_receipt_records_the_scored_carrier_count_per_domain(receipt):
    """The count is reported, not asserted: no criterion fixes its value."""
    for entry in receipt["domains"].values():
        counted = sum(group["scored_arrays"]
                      for group in entry["groups"].values())
        assert entry["scored_carriers"] == counted
        assert counted > 0


def test_a_required_group_with_nothing_scored_exits_non_zero(tmp_path: Path):
    """AC-1's coverage branch, driven on a pair with one group deleted."""
    soil = next(g for g in CARRIER_GROUPS if g.name == "soil")
    assert soil.required
    candidate_dir, reference_dir = _stage(tmp_path, drop=soil.variables)
    out_json = tmp_path / "receipt.json"
    status = digest_main([
        "--candidate-dir", str(candidate_dir),
        "--reference-dir", str(reference_dir),
        "--out-json", str(out_json),
        "--evaluator-commit", FAKE_COMMIT,
    ])
    assert status == EXIT_COVERAGE
    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert written["uncovered_required_groups"] == ["soil"]
    assert "soil" not in written["covered_groups"]


def test_a_clean_pair_exits_zero_and_a_perturbed_one_exits_on_the_verdict(
        tmp_path: Path):
    candidate_dir, reference_dir = _stage(tmp_path)
    out_json = tmp_path / "clean.json"
    out_md = tmp_path / "clean.md"
    assert digest_main([
        "--candidate-dir", str(candidate_dir),
        "--reference-dir", str(reference_dir),
        "--out-json", str(out_json), "--out-md", str(out_md),
        "--evaluator-commit", FAKE_COMMIT,
    ]) == EXIT_OK
    assert "t=0 full-state parity digest" in out_md.read_text(encoding="utf-8")

    def perturb(fields: dict[str, np.ndarray], _domain: str) -> None:
        flat = fields["T"].reshape(-1)
        flat[0] = _advance(flat[0], FP32_STATE_EQUIV_MAX_ULPS + 1)

    bad_candidate, bad_reference = _stage(tmp_path / "bad", perturb=perturb)
    assert digest_main([
        "--candidate-dir", str(bad_candidate),
        "--reference-dir", str(bad_reference),
        "--out-json", str(tmp_path / "bad.json"),
        "--evaluator-commit", FAKE_COMMIT,
    ]) == EXIT_VERDICT


def test_no_frame_pair_at_all_exits_non_zero(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    candidate_dir, _reference = _stage(tmp_path)
    assert digest_main([
        "--candidate-dir", str(candidate_dir),
        "--reference-dir", str(empty),
        "--out-json", str(tmp_path / "none.json"),
        "--evaluator-commit", FAKE_COMMIT,
    ]) == EXIT_COVERAGE


# ---------------------------------------------------------------------------
# AC-2: the ULP is the certified one, and a different definition fails
# ---------------------------------------------------------------------------


def _registered_reduction(signed: np.ndarray) -> dict[str, float]:
    """The comparator's own reduction, restated here on purpose.

    Restating it is what makes the check independent: if the digest ever
    reduced differently from the ledger, this would see it.
    """
    flat = np.asarray(signed, dtype=np.int64).reshape(-1)
    magnitude = np.abs(flat)
    rank = math.ceil(0.99 * flat.size) - 1
    return {
        "max_ulp": int(np.max(magnitude)),
        "p99_ulp": int(np.sort(magnitude)[rank]),
        "mean_signed_ulp": float(np.mean(flat, dtype=np.float64)),
    }


def _raw_bit_difference_ulp(candidate, reference) -> np.ndarray:
    """A deliberately different ULP definition: raw bit subtraction.

    It omits the monotone mapping, so it disagrees on every negative value
    and puts the two signed zeros 2**31 apart instead of adjacent.  Nothing
    in the tree computes ULPs this way; it exists only to prove the
    assertion below can fail.
    """
    left = np.ascontiguousarray(candidate, dtype=np.float32).view(np.uint32)
    right = np.ascontiguousarray(reference, dtype=np.float32).view(np.uint32)
    return left.astype(np.int64) - right.astype(np.int64)


def _awkward_pair() -> tuple[np.ndarray, np.ndarray]:
    """Random FP32 plus the encodings that separate ULP definitions."""
    rng = np.random.default_rng(20260731)
    reference = rng.standard_normal(4096).astype(np.float32)
    candidate = np.asarray(
        [_advance(value, int(offset)) for value, offset
         in zip(reference, rng.integers(-3, 4, reference.size))],
        dtype=np.float32)
    tiny = np.float32(np.finfo(np.float32).smallest_subnormal)
    edges_reference = np.asarray(
        [0.0, -0.0, tiny, -tiny, tiny * 3, -1.0, -2.5], dtype=np.float32)
    edges_candidate = np.asarray(
        [-0.0, 0.0, tiny * 2, -tiny * 2, tiny, _advance(np.float32(-1.0), 1),
         -2.5], dtype=np.float32)
    return (np.concatenate([candidate, edges_candidate]),
            np.concatenate([reference, edges_reference]))


def _assert_receipt_ulp_matches(definition) -> None:
    """The AC-2 assertion, parameterized by the ULP definition under test."""
    candidate, reference = _awkward_pair()
    scored = score_variables({"x": candidate}, {"x": reference}, ["x"])
    measured = scored["variables"]["x"]
    expected = _registered_reduction(definition(candidate, reference))
    for key, value in expected.items():
        assert measured[key] == value, key


def test_every_ulp_in_the_receipt_is_the_certified_definition():
    _assert_receipt_ulp_matches(fp32_signed_ulp)


def test_the_ulp_assertion_fails_against_a_different_definition():
    """The negative control.  Without it the assertion above proves nothing."""
    with pytest.raises(AssertionError):
        _assert_receipt_ulp_matches(_raw_bit_difference_ulp)


def test_the_certified_ulp_is_bit_equal_on_signed_zeros_and_subnormals():
    candidate, reference = _awkward_pair()
    signed = fp32_signed_ulp(candidate, reference)
    assert signed.dtype == np.int64
    # Signed zeros are adjacent patterns under the certified definition and
    # 2**31 apart under raw bit subtraction; the pair is in the fixture so
    # the control above has something to disagree about.
    assert abs(int(signed[-7])) == 1 and abs(int(signed[-6])) == 1
    assert abs(int(_raw_bit_difference_ulp(candidate, reference)[-7])) == 2 ** 31
    assert int(signed[-1]) == 0


def test_the_end_to_end_receipt_metrics_recompute_from_the_certified_ulp(
        staged, receipt):
    """AC-2 on the receipt itself, not only on the scoring helper."""
    from gpuwm.verify.t0_state_digest import read_frame

    candidate_dir, reference_dir = staged
    name = "QVAPOR"
    candidate = read_frame(candidate_dir / "wrfout_d01_2020-01-01_00_00_00",
                           [name])[name]
    reference = read_frame(reference_dir / "wrfout_d01_2020-01-01_00_00_00",
                           [name])[name]
    expected = _registered_reduction(fp32_signed_ulp(candidate, reference))
    measured = receipt["domains"]["d01"]["groups"]["moisture"]["variables"][name]
    for key, value in expected.items():
        assert measured[key] == value


# ---------------------------------------------------------------------------
# AC-3: registration hash, and ceilings that are imported rather than typed
# ---------------------------------------------------------------------------


def test_the_receipt_embeds_a_reproducible_registration_hash(receipt):
    assert receipt["schema"] == SCHEMA_ID
    assert re.fullmatch(r"[0-9a-f]{40}", receipt["evaluator_commit"])
    assert receipt["registration"] == make_t0_registration()
    assert receipt["registration_sha256"] == registration_sha256()
    assert receipt["registration_sha256"] == registration_sha256(
        make_t0_registration())


def test_an_edited_registration_changes_the_hash():
    """The pre-registration is only evidence if editing it is visible."""
    edited = make_t0_registration()
    edited["carrier_groups"]["soil"]["required"] = False
    assert registration_sha256(edited) != registration_sha256()


def test_a_non_hex_evaluator_commit_is_refused(tmp_path: Path):
    candidate_dir, reference_dir = _stage(tmp_path)
    with pytest.raises(ValueError, match="40-hex"):
        build_t0_receipt(candidate_dir, reference_dir,
                         evaluator_commit="unavailable")


def test_the_registration_quotes_the_imported_ceilings():
    ceilings = make_t0_registration()["ceilings"]
    assert ceilings["source"] == "gpuwm.verify.nest_gates"
    assert ceilings["max_ulp"] == FP32_STATE_EQUIV_MAX_ULPS
    assert ceilings["p99_ulp"] == FP32_STATE_EQUIV_P99_MAX_ULPS
    assert (ceilings["mean_signed_ulp_max_abs"]
            == FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS)


_CEILING_NAME_RE = re.compile(
    r"ceil|ulp|tolerance|threshold|max_abs|epsilon|bound", re.IGNORECASE)


class _CeilingLiteralScan(ast.NodeVisitor):
    """Numeric constants sitting where a ceiling would sit.

    A grep for the values would be weaker in both directions -- it would
    miss a ceiling spelled ``4 + 4`` and would fire on an unrelated ``8``.
    This looks at the position instead: what a number is assigned to,
    compared against, keyed by, or passed as.
    """

    def __init__(self) -> None:
        self.found: list[tuple[int, str]] = []

    @staticmethod
    def _numbers(node: ast.AST) -> list[ast.Constant]:
        return [child for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, (int, float))
                and not isinstance(child.value, bool)]

    def _flag(self, node: ast.AST, where: str) -> None:
        for number in self._numbers(node):
            self.found.append((number.lineno, f"{where}: {number.value!r}"))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            name = getattr(target, "id", None) or getattr(target, "attr", None)
            if name and _CEILING_NAME_RE.search(name):
                self._flag(node.value, f"assigned to {name}")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        name = getattr(node.target, "id", None)
        if name and _CEILING_NAME_RE.search(name) and node.value is not None:
            self._flag(node.value, f"assigned to {name}")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        mentions = any(
            _CEILING_NAME_RE.search(getattr(child, "id", "")
                                    or getattr(child, "attr", ""))
            for operand in operands for child in ast.walk(operand)
            if isinstance(child, (ast.Name, ast.Attribute)))
        if mentions:
            for operand in operands:
                self._flag(operand, "compared against a ceiling")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and _CEILING_NAME_RE.search(key.value)):
                self._flag(value, f"keyed by {key.value!r}")
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg and _CEILING_NAME_RE.search(node.arg):
            self._flag(node.value, f"passed as {node.arg}")
        self.generic_visit(node)


def _scan_module(source: str) -> list[tuple[int, str]]:
    scan = _CeilingLiteralScan()
    scan.visit(ast.parse(source))
    return scan.found


def test_the_digest_module_holds_no_numeric_ceiling_literal():
    found = _scan_module(DIGEST_MODULE.read_text(encoding="utf-8"))
    assert found == [], found


def test_the_ceiling_literal_scan_can_fail():
    """Control: the scan is not a constant 'clean'."""
    assert _scan_module("MAX_ULP_CEILING = 8\n")
    assert _scan_module("if measured_ulp > 8:\n    pass\n")
    assert _scan_module("d = {'max_ulp': 8}\n")
    assert _scan_module("f(ulp_ceiling=8)\n")
    assert _scan_module("BLOCK_BYTES = 8\n") == []


def test_the_digest_module_imports_the_ceilings_by_name():
    tree = ast.parse(DIGEST_MODULE.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "gpuwm.verify.nest_gates"
        for alias in node.names
    }
    assert imported == {
        "FP32_STATE_EQUIV_MAX_ULPS",
        "FP32_STATE_EQUIV_P99_MAX_ULPS",
        "FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS",
    }


def test_the_ceiling_pin_predates_this_tree():
    """`git log -S` locates the pin, and it is an ancestor of HEAD.

    The other half of AC-3 -- that the pin precedes the *receipt's* commit
    -- is checkable only once the receipt has a committed home, which is
    owner decision D-12.
    """
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    found = subprocess.run(
        ["git", "log", "-S", "FP32_STATE_EQUIV_MAX_ULPS = 8", "--format=%H",
         "--", "gpuwm/verify/nest_gates.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if found.returncode != 0:
        pytest.skip(f"git unavailable: {found.stderr.strip()}")
    commits = found.stdout.split()
    assert commits, "the pin's introducing commit is not in this history"
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commits[-1], "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert ancestor.returncode == 0, commits[-1]


# ---------------------------------------------------------------------------
# AC-4: the boundary group is probed, and both outcomes are representable
# ---------------------------------------------------------------------------


def test_the_boundary_group_is_scored_when_both_sides_staged_one(receipt):
    boundary = receipt["boundary"]
    assert boundary["status"] == "scored"
    assert boundary["absent_sides"] == []
    assert BOUNDARY_GROUP in receipt["covered_groups"]
    tables = boundary["domains"]["d01"]
    assert tables["value_tables"] and tables["tendency_tables"]
    assert set(tables["value_tables"]).isdisjoint(tables["tendency_tables"])
    assert all(name.startswith(tuple("UVTQ")) for name in tables["variables"])
    assert tables["verdict"] == "PASS"


@pytest.mark.parametrize(
    "staged_side, absent", [("candidate", "reference"),
                            ("reference", "candidate")])
def test_the_boundary_group_names_the_absent_side(tmp_path: Path, staged_side,
                                                  absent):
    candidate_dir, reference_dir = _stage(tmp_path / staged_side,
                                          boundary=staged_side)
    built = build_t0_receipt(candidate_dir, reference_dir,
                             evaluator_commit=FAKE_COMMIT)
    boundary = built["boundary"]
    assert boundary["status"] == "unavailable"
    assert boundary["absent_sides"] == [absent]
    assert absent in boundary["reason"]
    assert BOUNDARY_GROUP not in built["covered_groups"]


def test_the_boundary_status_is_exactly_one_of_two_values(tmp_path: Path):
    neither = build_t0_receipt(*_stage(tmp_path / "none", boundary=None),
                               evaluator_commit=FAKE_COMMIT)
    assert neither["boundary"]["status"] == "unavailable"
    assert sorted(neither["boundary"]["absent_sides"]) == [
        "candidate", "reference"]
    declared = make_t0_registration()["boundary_group"]["status_values"]
    assert sorted(declared) == ["scored", "unavailable"]


def test_boundary_tables_absent_on_one_side_are_recorded_not_dropped(
        tmp_path: Path):
    candidate_dir, reference_dir = _stage(tmp_path)
    trimmed = _boundary_fields(11)
    trimmed.pop("U_BXS")
    _write_netcdf(candidate_dir / "wrfbdy_d01", trimmed)
    built = build_t0_receipt(candidate_dir, reference_dir,
                             evaluator_commit=FAKE_COMMIT)
    tables = built["boundary"]["domains"]["d01"]
    assert tables["unscored"] == {"U_BXS": "absent from candidate"}


# ---------------------------------------------------------------------------
# AC-5: the mutation control -- one carrier past the imported ceiling
# ---------------------------------------------------------------------------


def test_perturbing_one_carrier_past_the_ceiling_fails_its_group(
        tmp_path: Path):
    """Clean passes, perturbed fails.  Either half alone proves nothing."""
    clean = build_t0_receipt(*_stage(tmp_path / "clean"),
                             evaluator_commit=FAKE_COMMIT)
    assert clean["domains"]["d01"]["groups"]["soil"]["verdict"] == "PASS"
    assert clean["verdict"] == "PASS"

    def perturb(fields: dict[str, np.ndarray], domain: str) -> None:
        if domain != "d01":
            return
        flat = fields["TSLB"].reshape(-1)
        flat[0] = _advance(flat[0], FP32_STATE_EQUIV_MAX_ULPS + 1)

    perturbed = build_t0_receipt(*_stage(tmp_path / "bad", perturb=perturb),
                                 evaluator_commit=FAKE_COMMIT)
    groups = perturbed["domains"]["d01"]["groups"]
    assert groups["soil"]["verdict"] == "FAIL"
    assert groups["soil"]["variables"]["TSLB"]["max_ulp"] > (
        FP32_STATE_EQUIV_MAX_ULPS)
    assert groups["moisture"]["verdict"] == "PASS"
    assert perturbed["domains"]["d02"]["groups"]["soil"]["verdict"] == "PASS"
    assert perturbed["verdict"] == "FAIL"


def test_a_shape_disagreement_is_a_blocking_failure_not_a_broadcast():
    scored = score_variables(
        {"x": np.ones(4, np.float32)}, {"x": np.ones(5, np.float32)}, ["x"])
    assert scored["verdict"] == "FAIL"
    assert "shape mismatch" in scored["variables"]["x"]["reason"]


# ---------------------------------------------------------------------------
# AC-15: the input inventory
# ---------------------------------------------------------------------------


def test_the_receipt_names_every_file_it_opened_with_a_side_and_a_hash(
        staged, receipt):
    import hashlib

    candidate_dir, reference_dir = staged
    opened = {(record["side"], record["name"]) for record in receipt["inputs"]}
    expected = {("candidate", path.name)
                for path in candidate_dir.iterdir()} | {
        ("reference", path.name) for path in reference_dir.iterdir()}
    assert opened == expected

    roots = {"candidate": candidate_dir, "reference": reference_dir}
    for record in receipt["inputs"]:
        path = roots[record["side"]] / record["name"]
        assert record["sha256"] == hashlib.sha256(
            path.read_bytes()).hexdigest()
        assert record["size_bytes"] == path.stat().st_size
        assert record["role"] in {"wrfout", "wrfbdy"}


def test_the_inventory_and_the_boundary_status_agree(receipt):
    roles = {record["role"] for record in receipt["inputs"]}
    assert ("wrfbdy" in roles) == (receipt["boundary"]["status"] == "scored")


# ---------------------------------------------------------------------------
# AC-7: nothing here knows how many domains a case has
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domains", [("d01",), ("d01", "d02"),
                                     ("d01", "d02", "d03", "d04")])
def test_the_digest_scores_whatever_domain_count_is_staged(tmp_path: Path,
                                                           domains):
    built = build_t0_receipt(
        *_stage(tmp_path / f"n{len(domains)}", domains=domains),
        evaluator_commit=FAKE_COMMIT)
    assert sorted(built["domains"]) == list(domains)
    assert built["verdict"] == "PASS"
    for domain in domains:
        assert built["domains"][domain]["scored_carriers"] > 0


def test_nothing_on_the_digest_import_path_is_a_case_module():
    """The named temptation: a case module reached for a helper.

    ``gpuwm/verify/cases/`` holds one campaign's tables.  A generic digest
    that imported one -- for a hashing helper, say -- would put a case on
    the certification import path, and the leakage gate cannot see an
    import that is legitimate in every other module.
    """
    before = set(sys.modules)
    for name in [n for n in sys.modules if "gpuwm.verify.t0_state_digest" in n]:
        del sys.modules[name]
    import importlib

    importlib.import_module("gpuwm.verify.t0_state_digest")
    reached = {name for name in set(sys.modules) - before
               if name.startswith("gpuwm.verify.cases")}
    assert reached == set()
    tree = ast.parse(DIGEST_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        module = getattr(node, "module", None) or ""
        names = [alias.name for alias in getattr(node, "names", [])]
        assert "cases" not in module.split("."), module
        assert not any("cases" in name.split(".") for name in names), names


# ---------------------------------------------------------------------------
# the receipt is a function of its inputs
# ---------------------------------------------------------------------------


def test_the_receipt_regenerates_byte_identically(staged):
    candidate_dir, reference_dir = staged
    first = canonical_json(build_t0_receipt(
        candidate_dir, reference_dir, evaluator_commit=FAKE_COMMIT))
    second = canonical_json(build_t0_receipt(
        candidate_dir, reference_dir, evaluator_commit=FAKE_COMMIT))
    assert first == second
    assert json.loads(first)["verdict"] == "PASS"


def test_the_cli_writes_lf_endings_on_every_platform(tmp_path: Path):
    """Byte-identity has to survive the platform, or it proves nothing.

    Default text-mode translation would make the same inputs produce a
    CRLF receipt on one machine and an LF receipt on another, so a
    cross-machine byte comparison -- the corruption check this receipt
    form is built for -- would report a difference that is only a line
    ending.
    """
    candidate_dir, reference_dir = _stage(tmp_path)
    out_json = tmp_path / "lf.json"
    out_md = tmp_path / "lf.md"
    digest_main([
        "--candidate-dir", str(candidate_dir),
        "--reference-dir", str(reference_dir),
        "--out-json", str(out_json), "--out-md", str(out_md),
        "--evaluator-commit", FAKE_COMMIT,
    ])
    for path in (out_json, out_md):
        assert b"\r\n" not in path.read_bytes(), path


def test_the_committed_receipt_is_stored_with_lf_endings():
    """The control on the published file, not only on a fresh write."""
    committed = REPO_ROOT / PUBLISHED_RECEIPT_JSON
    assert committed.is_file()
    assert b"\r\n" not in committed.read_bytes()


def test_the_earliest_common_frame_is_the_one_scored(tmp_path: Path):
    candidate_dir, reference_dir = _stage(tmp_path)
    later = "2020-01-01_01_00_00"
    for directory in (candidate_dir, reference_dir):
        _write_netcdf(directory / f"wrfout_d01_{later}", _fields(7))
    pairs = discover_frame_pairs(candidate_dir, reference_dir)
    assert [(pair.domain, pair.valid_time) for pair in pairs] == [
        ("d01", "2020-01-01_00_00_00"), ("d02", "2020-01-01_00_00_00")]
    named = discover_frame_pairs(candidate_dir, reference_dir,
                                 valid_time=later)
    assert [(pair.domain, pair.valid_time) for pair in named] == [
        ("d01", later)]


def test_the_markdown_table_carries_the_same_numbers(receipt):
    rendered = render_markdown(receipt)
    assert receipt["registration_sha256"] in rendered
    assert receipt["evaluator_commit"] in rendered
    for group in CARRIER_GROUPS:
        assert f"| {group.name} |" in rendered
    assert "boundary group: scored" in rendered
