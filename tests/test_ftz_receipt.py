"""The FTZ receipt is derived from bits, and the derivation is checkable.

Every verdict in ``receipt.json`` is recomputed here from ``bitpatterns.csv``
by an independent implementation of the rule in ``classify``'s docstring.
This file names no expected verdict: it imports the vocabulary constants and
``test_this_file_states_no_verdict`` greps itself for their values.  A test
that asserted "R2 flushes" would be a second, uncontrolled authority.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import requires_gpu  # noqa: E402
from tools.ftz_receipt import probe  # noqa: E402
from tools.ftz_receipt import route_inventory as ri  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_DIR = ROOT / "tools" / "ftz_receipt" / "receipt"
FIXTURES = ROOT / "tools" / "ftz_receipt" / "fixtures"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads((RECEIPT_DIR / "receipt.json").read_text(
        encoding="utf-8"))


@pytest.fixture(scope="module")
def bit_rows() -> list[dict]:
    with (RECEIPT_DIR / "bitpatterns.csv").open(encoding="utf-8",
                                                newline="") as handle:
        return list(csv.DictReader(handle))


# ---- the receipt describes itself ----------------------------------------

def test_receipt_records_the_stack_it_measured(receipt):
    device = receipt["device"]
    for key in ("name", "uuid", "pci_bus_id", "compute_capability",
                "cuda_driver_version", "cuda_runtime_version",
                "nvrtc_version", "cupy_version"):
        assert device.get(key) not in (None, ""), key
    site = receipt["cupy_ftz_append_site"]
    assert site["found"] is True
    assert site["module"] == "cupy.cuda.compiler"
    capture = receipt["effective_option_capture"]
    assert capture["method"]
    for route_id, records in capture["per_route"].items():
        assert records, route_id
        for record in records:
            assert record["fired"] is True
            assert record["effective_flags"]


def test_the_probe_ran_twice_and_the_tables_agree(receipt):
    dual = receipt["dual_run"]
    assert dual["runs"] == 2
    assert dual["byte_identical"] is True
    assert len(dual["bit_table_sha256"]) == 2
    assert dual["bit_table_sha256"][0] == dual["bit_table_sha256"][1]
    table = (RECEIPT_DIR / "bitpatterns.csv").read_text(encoding="utf-8")
    assert hashlib.sha256(table.encode("utf-8")).hexdigest() == \
        dual["bit_table_sha256"][0]


def test_every_committed_artifact_hash_matches_disk(receipt):
    checked = 0
    for name, record in receipt["artifacts"].items():
        if "sha256" not in record:
            assert record.get("reason"), name
            continue
        path = RECEIPT_DIR / name
        assert path.exists(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == \
            record["sha256"], name
        checked += 1
    assert checked > 1


def test_artifact_provenance_is_labelled(receipt):
    allowed = {"shipped-path", "mirror", "unavailable", "not-collected"}
    for name, record in receipt["artifacts"].items():
        assert record["provenance"] in allowed, (name, record["provenance"])
        if record["provenance"] == "mirror":
            assert record.get("mirror_of"), name
            assert record.get("description"), name


def test_every_route_mechanism_cell_is_present(receipt):
    routes = set(receipt["routes"])
    mechanisms = list(receipt["mechanisms"])
    assert len(mechanisms) == 6
    seen = {(cell["route"], cell["mechanism"]) for cell in receipt["cells"]}
    assert seen == {(route, mechanism)
                    for route in routes for mechanism in mechanisms}
    for cell in receipt["cells"]:
        if cell["verdict"] == probe.VERDICT_NOT_APPLICABLE:
            assert cell["not_applicable_reason"], cell
        else:
            assert cell["input_count"] > 0, cell


def test_the_arms_are_not_all_the_same(receipt):
    """A pipeline that ignored its options would make every cell vacuous."""
    sensitivity = receipt["flag_sensitivity"]
    assert sensitivity["distinct_arm_bit_tables"] > 1
    assert sensitivity["arms_differ"] is True


# ---- verdicts are derived, not authored ----------------------------------

def _independent_verdict(rows: list[dict]) -> str:
    """Re-derive one cell's verdict from the CSV alone.

    Written from ``classify``'s docstring rather than by calling it; the
    vocabulary is imported so no verdict string is typed in this file.
    """
    if not rows:
        return probe.VERDICT_NOT_APPLICABLE
    separable = 0
    ieee_everywhere = True
    flushed_everywhere = True
    for row in rows:
        device = int(row["device_bits"], 16)
        ieee = int(row["ieee_reference_bits"], 16)
        flushed = int(row["flushed_reference_bits"], 16)
        if ieee != flushed:
            separable += 1
        if device != ieee:
            ieee_everywhere = False
        if device != flushed:
            flushed_everywhere = False
    if ieee_everywhere:
        return probe.VERDICT_IEEE if separable \
            else probe.VERDICT_INCONCLUSIVE
    if flushed_everywhere:
        return probe.VERDICT_FLUSH
    return probe.VERDICT_DIVERGENT


def test_every_verdict_recomputes_from_the_bit_table(receipt, bit_rows):
    for cell in receipt["cells"]:
        subset = [row for row in bit_rows
                  if row["route"] == cell["route"]
                  and row["mechanism"] == cell["mechanism"]]
        assert _independent_verdict(subset) == cell["verdict"], cell


def test_the_host_reference_columns_are_reproducible(bit_rows):
    """The IEEE and flushed columns come from host binary32, not the device."""
    by_id = {row["id"]: row for row in probe.INPUT_ROWS}
    ops = dict(probe.MECHANISMS)
    for row in bit_rows:
        op = ops[row["mechanism"]]
        source = by_id[row["input_id"]]
        assert int(row["ieee_reference_bits"], 16) == probe._reference(
            op, source, flushing=False)
        assert int(row["flushed_reference_bits"], 16) == probe._reference(
            op, source, flushing=True)


def test_this_file_states_no_verdict():
    text = Path(__file__).read_text(encoding="utf-8")
    for verdict in probe.VERDICTS:
        assert verdict not in text, (
            f"{verdict!r} is typed in this file; verdicts must be imported "
            f"by name so the receipt stays the only authority")


def test_the_probe_states_no_verdict_either():
    """``probe.py`` may define the vocabulary but must not assign a cell."""
    text = (ROOT / "tools" / "ftz_receipt" / "probe.py").read_text(
        encoding="utf-8")
    for route in ("R1", "R2", "R3", "R4", "R5"):
        for verdict in probe.VERDICTS:
            assert f"{route}: {verdict}" not in text
            assert f"{route} {verdict}" not in text


# ---- the probe rides the production compile path -------------------------

def test_the_loader_arm_carries_no_option_tuple():
    text = (ROOT / "tools" / "ftz_receipt" / "probe.py").read_text(
        encoding="utf-8")
    assert "-std=c++17" not in text, (
        "the R1 arm must take its options from the loader, not from a "
        "literal in the probe")
    assert "load_module(\"ftz_probe\")" in text


def test_each_arm_matches_the_site_it_claims(receipt):
    inventory = ri.load_inventory(ROOT)
    expected = {
        "R1": ("gpuwm/core/kernels/__init__.py", "cupy.RawModule",
               "load_module"),
        # R2's kind moved with 2358b3c06 (out-of-core merge): NVRTC 13
        # rejects the duplicated --ftz, so the shortwave compile left
        # cupy.RawModule for the direct-NVRTC bypass the longwave site
        # already used.  Same option tuple, new constructor.
        "R2": ("gpuwm/core/rrtmg_sw.py",
               "cupy.cuda.compiler.compile_using_nvrtc", "__init__"),
        "R3": ("gpuwm/core/rrtmg_lw.py",
               "cupy.cuda.compiler.compile_using_nvrtc", None),
        "R4": ("gpuwm/core/mynn_pbl_gpu.py", "cupy.ReductionKernel",
               "_nonpositive"),
    }
    for route_id, (path, kind, enclosing) in expected.items():
        site = ri.find_site(inventory, file=path, constructor_kind=kind,
                            enclosing=enclosing)
        recorded = receipt["routes"][route_id]
        assert recorded["site"].startswith(f"{path}:{site['line']}")
        assert recorded["site_options"] == site["options"], route_id


def test_the_control_arm_is_the_loader_tuple_plus_one_flag(receipt):
    base = receipt["routes"]["R1"]["arm_options"]
    control = receipt["routes"]["R1-ftztrue"]["arm_options"]
    assert control[:len(base)] == base
    assert len(control) == len(base) + 1
    assert control[-1].endswith("ftz=true")


# ---- control 1: the PTX parser discriminates ------------------------------

def test_parser_control_separates_the_two_fixtures():
    present = probe.ptx_ftz_modifiers(
        (FIXTURES / "ftz_modifier_present.ptx").read_text(encoding="utf-8"))
    absent = probe.ptx_ftz_modifiers(
        (FIXTURES / "ftz_modifier_absent.ptx").read_text(encoding="utf-8"))
    assert present["with_ftz"] and not present["without_ftz"]
    assert absent["without_ftz"] and not absent["with_ftz"]
    assert present["f32_instruction_count"] == absent["f32_instruction_count"]


def test_a_constant_answer_parser_fails_the_fixture_pair():
    """The control's own control: a parser that ignores its input fails."""
    def constant(_text):
        return {"with_ftz": {"mul": 1}, "without_ftz": {},
                "f32_instruction_count": 4}

    present = constant((FIXTURES / "ftz_modifier_present.ptx").read_text(
        encoding="utf-8"))
    absent = constant((FIXTURES / "ftz_modifier_absent.ptx").read_text(
        encoding="utf-8"))
    assert present == absent
    with pytest.raises(AssertionError):
        assert absent["without_ftz"] and not absent["with_ftz"]


# ---- control 2: the probe is sensitive to what it measures ---------------

_PTX_MUL = ('asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(r) '
            ': "f"(a), "f"(b));')
_PTX_CVT = 'asm volatile("cvt.rn.f32.f64 %0, %1;" : "=f"(r) : "d"(a));'


@requires_gpu
def test_probe_sensitivity_control_moves_the_bit_column(receipt):
    """Rewrite the inline-PTX helper and the d2f encoder; the bits must move.

    If the mutated arm produced the same words as the shipped one, the probe
    would not be measuring the instruction it names, and every R5 cell would
    be an accident.
    """
    import numpy as np
    import cupy as cp
    from gpuwm.core import kernels as gpuwm_kernels

    source = probe._probe_source()
    mutated = source.replace(_PTX_MUL, "r = a * b;")
    assert mutated != source, "the inline-PTX multiply arm has moved"
    mutated2 = mutated.replace(_PTX_CVT, "r = __double2float_rn(a);")
    assert mutated2 != mutated, "the inline-PTX convert arm has moved"

    options = tuple(receipt["routes"]["R1"]["arm_options"])
    module = cp.RawModule(code=mutated2, options=options)
    module.compile()

    shipped = gpuwm_kernels.load_module("ftz_probe")
    count = len(probe.INPUT_ROWS)
    au = cp.asarray(np.array([probe._f32_bits(row["a"])
                              for row in probe.INPUT_ROWS], dtype=np.uint32))
    bu = cp.asarray(np.array([probe._f32_bits(row["b"])
                              for row in probe.INPUT_ROWS], dtype=np.uint32))
    ad = cp.asarray(np.array([row["d"] for row in probe.INPUT_ROWS],
                             dtype=np.float64))

    def run(mod, name, convert):
        out = cp.zeros(count, dtype=cp.uint32)
        function = mod.get_function(name)
        args = (ad, out, np.int32(count)) if convert else (
            au, bu, out, np.int32(count))
        function((1,), (32,), args)
        cp.cuda.runtime.deviceSynchronize()
        return [int(value) for value in out.get()]

    for name, convert in (("ftz_probe_ptx_mul", False),
                          ("ftz_probe_ptx_cvt", True)):
        assert run(module, name, convert) != run(shipped, name, convert), (
            f"{name} produced identical words after its instruction was "
            f"replaced; the probe cannot see what it claims to measure")


@requires_gpu
def test_the_committed_table_reproduces_on_this_device(receipt):
    inventory = ri.load_inventory(ROOT)
    rows, _meta = probe.run_probe(inventory)
    assert probe.bitpatterns_csv(rows) == (
        RECEIPT_DIR / "bitpatterns.csv").read_text(encoding="utf-8")
