"""gpuwm-vs-WRF parity for the Noah-MP flux-preparation leaves.

``SFCDIF1``, ``RAGRB`` and ``STOMATA`` at WRF v4.6.1 commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``.  The fixture
``gpuwm/data/noahmp/oracle/noahmp-fluxprep.csv`` is produced by
``tools/noahmp_wrf461_oracle/build_fluxprep.sh`` from a *scratch copy* of the
pinned WRF tree carrying only ``patches/noahmp-lsm-leaf-visibility.patch``
(50 ``private ::`` -> ``public ::`` accessibility statements, nothing else).
The bar is bitwise: ``max_ulp 0`` on every live output slot, CPU and CUDA.

Four independent gates, because bitwise parity alone is not evidence that a
port is right:

* the committed CSVs are the pinned bytes and still pass the build-time
  validator, including its branch-coverage assertions;
* every case is reproduced bit for bit on the CPU, and on CUDA;
* the fixture kills every argument mutant on every consumed argument, so it
  can detect a port that ignores an argument;
* the branches the pinned option identity kills are asserted *absent* from the
  port rather than described in a comment.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.noahmp_fluxprep import (
    FLUXPREP_EVALUATORS,
    STOMATA_NITER,
    SfcdifDomainError,
    sfcdif1,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = REPO_ROOT / "gpuwm" / "data" / "noahmp" / "oracle"
FLUXPREP_CSV = ORACLE_DIR / "noahmp-fluxprep.csv"
DISC_CSV = ORACLE_DIR / "noahmp-fluxprep-discrimination.csv"
ATANF_CSV = ORACLE_DIR / "glibc-atanf-fp32.csv"
ORACLE_TOOLS = REPO_ROOT / "tools" / "noahmp_wrf461_oracle"
KERNEL_SOURCE = REPO_ROOT / "gpuwm" / "core" / "kernels" / "noahmp_fluxprep.cu"

PINNED_ASSETS = {
    FLUXPREP_CSV:
        "c8673ef6fc58a89d34756323866415a2a6d975237484fdfaaadeb2807e35d39c",
    DISC_CSV:
        "5c9e8d5abea520723426e7492e44b13d155e1658f6d59568acc4d8782d88d0e7",
    ATANF_CSV:
        "c52ad0e0f9f4b43e695210190a5202d064630817aab3bf8a94cde94b17e85e82",
}

#: 3 leaves x 28 cases, every input/output/topology slot.
PINNED_VALUE_COUNT = 816


def _load_tool(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_tool("noahmp_fluxprep_validator",
                       ORACLE_TOOLS / "validate_fluxprep_oracle.py")


def _f32_from_bits(bits: str) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(bits, 16)),
                         dtype=np.float32)[0]


def _ulp_delta(got: np.float32, want: np.float32) -> int:
    return int(fp32_ulp_distance(got, want).reshape(-1)[0])


def _load_cases():
    with FLUXPREP_CSV.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    packed = defaultdict(lambda: {"int": {}, "in": {}, "out": {}})
    for row in rows:
        packed[(row["leaf"], row["case"])][row["role"]][int(row["slot"])] = row
    cases = {}
    for key, roles in packed.items():
        ints = np.array([int(float(roles["int"][s]["value"]))
                         for s in sorted(roles["int"])], dtype=np.int64)
        x = np.array([_f32_from_bits(roles["in"][s]["bits"])
                      for s in sorted(roles["in"])], dtype=np.float32)
        outs = [(s, roles["out"][s]["name"], int(roles["out"][s]["index"]),
                 roles["out"][s]["live"] == "1", roles["out"][s]["bits"])
                for s in sorted(roles["out"])]
        cases[key] = (ints, x, outs)
    return cases


CASES = _load_cases()
LEAF_NAMES = sorted({leaf for leaf, _ in CASES})


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------

def test_fluxprep_assets_are_the_pinned_bytes():
    for path, digest in PINNED_ASSETS.items():
        payload = path.read_bytes()
        assert b"\r" not in payload, f"{path.name} is not LF-only"
        actual = hashlib.sha256(payload).hexdigest()
        assert actual == digest, f"{path.name}: {actual} != {digest}"


def test_fluxprep_fixture_still_discriminates():
    """Re-run the build-time validator on the committed CSVs.

    This keeps the guarantee without WSL or gfortran: structure, bit
    round-trip, the substitution sweep and every branch-coverage assertion.
    """
    table = VALIDATOR.load_leaves(FLUXPREP_CSV)
    VALIDATOR.check_structure(table)
    nvalues = VALIDATOR.check_bits(table)
    counts = VALIDATOR.check_discrimination(DISC_CSV)
    VALIDATOR.check_branch_coverage(table)
    assert nvalues == PINNED_VALUE_COUNT
    assert sum(counts.values()) == sum(
        spec["n_in"] for spec in VALIDATOR.LEAVES.values())
    assert set(counts) == set(LEAF_NAMES)


def test_every_oracle_leaf_has_a_cpu_reference():
    assert set(LEAF_NAMES) == set(FLUXPREP_EVALUATORS), (
        f"oracle leaves {LEAF_NAMES} vs ports {sorted(FLUXPREP_EVALUATORS)}")


def test_substitution_probe_is_zero_except_where_declared():
    """Only SFCDIF1's ZLVL may be probed with something other than 0.0.

    An undeclared non-zero probe would let a slot look discriminating while
    the substitution it survived was chosen to be harmless.
    """
    with DISC_CSV.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        key = (row["leaf"], row["name"], int(row["index"]))
        probe = _f32_from_bits(row["probe_bits"])
        declared = VALIDATOR.PROBES.get(row["leaf"], {}).get(key[1:])
        if probe != np.float32(0.0):
            assert declared is not None, f"{key} probed with {probe}"
            assert np.float32(declared[0]) == probe
            assert declared[1], f"{key} has no justification"
            seen.add(key)
    assert seen == {("sfcdif1", "zlvl", 0)}


# ---------------------------------------------------------------------------
# CPU parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key", sorted(CASES), ids=lambda key: f"{key[0]}-{key[1]}")
def test_cpu_fluxprep_leaf_is_bitwise(key):
    leaf, case = key
    ints, x, outs = CASES[key]
    got = FLUXPREP_EVALUATORS[leaf](x, ints)
    assert got.dtype == np.float32
    assert got.shape == (len(outs),)
    failures = []
    for slot, name, index, live, bits in outs:
        actual = np.float32(got[slot - 1])
        want = _f32_from_bits(bits) if live else np.float32(0.0)
        distance = _ulp_delta(actual, want)
        if distance:
            failures.append(
                f"{name}[{index}] slot {slot}: got "
                f"{struct.unpack('<I', struct.pack('<f', actual))[0]:08X}"
                f" ({actual!r}) want "
                f"{struct.unpack('<I', struct.pack('<f', want))[0]:08X}"
                f" ({want!r}), max_ulp {distance}")
    assert not failures, f"{leaf}/{case}\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# glibc atanf, which SFCDIF1 needs and noahmp_libm did not previously carry
# ---------------------------------------------------------------------------

def test_atanf_reproduces_glibc_on_the_pinned_sample():
    from gpuwm.core.noahmp_libm import atanf

    with ATANF_CSV.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) >= 4000
    failures = []
    for row in rows:
        x = _f32_from_bits(row["bits"])
        want = int(row["atanf_bits"], 16)
        got = struct.unpack("<I", struct.pack("<f", atanf(x)))[0]
        if got != want:
            failures.append(f"atanf({row['bits']}): {got:08X} != {want:08X}")
    assert not failures, "\n".join(failures[:20])


def test_atanf_is_not_the_fp64_shim():
    """A negative control: FP64-then-round is a *different* function.

    If this ever passes, the pinned sample has stopped covering the region
    where glibc's fdlibm reduction differs from the correctly-rounded result,
    and the atanf test above has become unable to fail.
    """
    with ATANF_CSV.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    disagreements = 0
    for row in rows:
        x = _f32_from_bits(row["bits"])
        shim = np.float32(np.arctan(np.float64(x)))
        if struct.unpack("<I", struct.pack("<f", shim))[0] \
                != int(row["atanf_bits"], 16):
            disagreements += 1
    assert disagreements > 0, (
        "the pinned atanf sample no longer distinguishes glibc from an "
        "FP64-then-round shim, so it cannot detect that substitution")


# ---------------------------------------------------------------------------
# Mutation study
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("leaf", LEAF_NAMES)
def test_fixture_kills_every_argument_mutant(leaf):
    """The fixture must be able to detect a port that ignores an argument.

    For every input slot -- FP32 and integer topology alike -- neither zero nor
    any constant the fixture gives that slot may be substituted and still
    reproduce it, unless the argument is declared inert with a WRF line-number
    justification.
    """
    from gpuwm.core import noahmp_leaf_mutation as mutation

    fixture = mutation.load_leaf_fixture(leaf, FLUXPREP_CSV)
    evaluator = FLUXPREP_EVALUATORS[leaf]
    assert not mutation.baseline_mismatches(fixture, evaluator)
    verdicts = mutation.summarise(
        mutation.run_mutation_study(leaf, fixture, evaluator),
        inert=VALIDATOR.LEAVES[leaf]["inert"], partial={})
    assert len(verdicts) == fixture.x.shape[1] + fixture.ints.shape[1]
    problems = mutation.failures(verdicts)
    assert not problems, "\n".join(problems)
    for verdict in verdicts:
        if verdict.declared_inert:
            assert verdict.justification, (
                f"{leaf}: {verdict.name}[{verdict.index}] has no "
                f"justification")


def test_fixture_distinguishes_the_powisf2_cube_lowering(monkeypatch):
    """``FV**3`` must be ``__powisf2``, and the fixture must be able to say so.

    A negative control on the CUDA half found that it originally could not:
    every case that reached ``FV**3`` either clamped ``MOZ``/``MOZG``
    immediately afterwards or happened to carry a friction velocity where
    ``fl(x * fl(x*x))`` and ``powf(x, 3.0)`` agree.  Four FV values were moved
    onto the ~26% of FP32 friction velocities where they disagree.  This test
    fails if that property is ever lost.
    """
    import gpuwm.core.noahmp_fluxprep as module
    from gpuwm.core.noahmp_libm import powf

    monkeypatch.setattr(
        module, "_powi3",
        lambda x: powf(np.float32(x), np.float32(3.0)))
    moved = 0
    for (leaf, _case), (ints, x, outs) in CASES.items():
        got = FLUXPREP_EVALUATORS[leaf](x, ints)
        for slot, _name, _index, live, bits in outs:
            want = _f32_from_bits(bits) if live else np.float32(0.0)
            moved += bool(_ulp_delta(np.float32(got[slot - 1]), want))
    assert moved > 0, (
        "swapping the __powisf2 expansion of FV**3 for a single-rounding "
        "powf(x, 3.0) moved no pinned output, so the fixture cannot detect "
        "that substitution")


def test_declared_inert_arguments_are_exactly_the_unreferenced_ones():
    """The inert sets are a claim about the WRF source; pin the claim.

    RAGRB's MOZG is the interesting one: it is ``INTENT(INOUT)``, so it is not
    inert by declaration, only by the assignment at :4536 that precedes every
    read.
    """
    assert set(VALIDATOR.LEAVES["ragrb"]["inert"]) == {
        ("tv", 0), ("mozg", 0), ("vegtyp", 0), ("iloc", 0), ("jloc", 0)}
    assert set(VALIDATOR.LEAVES["sfcdif1"]["inert"]) == {
        ("iloc", 0), ("jloc", 0)}
    assert set(VALIDATOR.LEAVES["stomata"]["inert"]) == {
        ("vegtyp", 0), ("iloc", 0), ("jloc", 0)}


# ---------------------------------------------------------------------------
# What the pinned option identity kills, asserted rather than described
# ---------------------------------------------------------------------------

DEAD_UNDER_PINNED_IDENTITY = {
    "sfcdif2": "reachable only from IF(OPT_SFC == 2); opt_sfc is 1",
    "canres": "reachable only from IF(OPT_CRS == 2); opt_crs is 1",
    "calhum": "called only by CANRES, which opt_crs = 1 kills",
    "gecros": "reachable only from IF(opt_crop == 2); opt_crop is 0",
}


def test_dead_branches_are_absent_from_the_port():
    """None of the routines this option identity kills may be ported.

    Comments claiming a branch is dead cost nothing; this makes the claim
    executable on both halves of the port.
    """
    sources = {
        "cpu": (REPO_ROOT / "gpuwm" / "core" / "noahmp_fluxprep.py"),
        "cuda": KERNEL_SOURCE,
    }
    for label, path in sources.items():
        text = path.read_text(encoding="ascii")
        body = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith(("#", "//", "*", '"""'))
        ).lower()
        for name, reason in DEAD_UNDER_PINNED_IDENTITY.items():
            assert name not in body, (
                f"{label} half references {name!r}, which is dead under the "
                f"pinned identity ({reason})")


def test_stomata_iteration_count_is_the_pinned_constant():
    """``DATA NITER /3/`` at :5045 is a constant, not a convergence test."""
    assert STOMATA_NITER == 3
    cuda = KERNEL_SOURCE.read_text(encoding="ascii")
    assert "#define NMP_STOMATA_NITER 3" in cuda


def test_sfcdif1_refuses_the_fatal_domain_instead_of_inventing_a_value():
    """``ZLVL <= ZPD`` stops WRF at :4651; the port must not answer."""
    with pytest.raises(SfcdifDomainError):
        sfcdif1(1, 0, 290.0, 1.18, 120.0, 0.01, 0.65, 0.65, 0.1, 0.1, 3.5,
                1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3)
    # Strictly greater is fine.
    sfcdif1(1, 0, 290.0, 1.18, 120.0, 0.01, 10.0, 0.65, 0.1, 0.1, 3.5,
            1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3)


# ---------------------------------------------------------------------------
# CUDA parity
# ---------------------------------------------------------------------------

def _require_cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as error:                        # pragma: no cover
        pytest.skip(f"CUDA unavailable: {error}")
    return cp


@pytest.mark.gpu
@pytest.mark.parametrize("leaf", LEAF_NAMES)
def test_cuda_fluxprep_leaf_is_bitwise(leaf):
    _require_cuda()
    from gpuwm.core.noahmp_fluxprep_gpu import evaluate_fluxprep_leaf

    keys = sorted(key for key in CASES if key[0] == leaf)
    ints = np.stack([CASES[key][0] for key in keys])
    x = np.stack([CASES[key][1] for key in keys])
    device = evaluate_fluxprep_leaf(leaf, x, ints)
    assert device.dtype == np.float32
    assert device.shape == (len(keys), len(CASES[keys[0]][2]))
    host = device.get()

    failures = []
    for case_index, key in enumerate(keys):
        for slot, name, index, live, bits in CASES[key][2]:
            actual = np.float32(host[case_index, slot - 1])
            want = _f32_from_bits(bits) if live else np.float32(0.0)
            distance = _ulp_delta(actual, want)
            if distance:
                failures.append(
                    f"{key[1]}: {name}[{index}] slot {slot}: got "
                    f"{struct.unpack('<I', struct.pack('<f', actual))[0]:08X}"
                    f" want "
                    f"{struct.unpack('<I', struct.pack('<f', want))[0]:08X}"
                    f", max_ulp {distance}")
    assert not failures, f"cuda {leaf}\n" + "\n".join(failures)


@pytest.mark.gpu
def test_cuda_fluxprep_kernels_are_immune_to_fma_contraction():
    """``-fmad=true`` and ``-fmad=false`` must give bit-identical output.

    If they differ, some ``a*b+c`` boundary in the kernel is still unpinned and
    the parity above holds only by the compiler's current choice.
    """
    cp = _require_cuda()
    from gpuwm.core.kernels import _preamble
    from gpuwm.core.noahmp_fluxprep_gpu import FLUXPREP_LAYOUTS

    source = _preamble() + KERNEL_SOURCE.read_text(encoding="ascii")
    modules = {}
    for label, options in (("fmad_on", ("-std=c++17",)),
                           ("fmad_off", ("-std=c++17", "--fmad=false"))):
        module = cp.RawModule(code=source, options=options)
        module.compile()
        modules[label] = module

    differences = []
    for leaf, layout in sorted(FLUXPREP_LAYOUTS.items()):
        keys = sorted(key for key in CASES if key[0] == leaf)
        host_x = np.stack([CASES[key][1] for key in keys])
        host_ix = np.stack([CASES[key][0] for key in keys]).astype(np.int32)
        results = {}
        for label, module in modules.items():
            out = cp.zeros((len(keys), layout.n_out), dtype=cp.float32)
            module.get_function(f"noahmp_fluxprep_{leaf}")(
                (1,), (64,),
                (cp.asarray(host_x),
                 cp.asarray(np.ascontiguousarray(host_ix)),
                 out, np.int32(len(keys))))
            results[label] = out.get().view(np.uint32)
        if not np.array_equal(results["fmad_on"], results["fmad_off"]):
            moved = np.argwhere(results["fmad_on"] != results["fmad_off"])
            differences.append(f"{leaf}: {len(moved)} slots move under -fmad")
    assert not differences, "\n".join(differences)
