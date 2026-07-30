"""gpuwm-vs-WRF parity for the Noah-MP *thermal* leaf group.

``gpuwm/data/noahmp/oracle/noahmp-thermal.csv`` is produced by
``tools/noahmp_wrf461_oracle/build_thermal.sh`` from a *scratch copy* of the
pinned WRF v4.6.1 tree carrying only
``patches/noahmp-lsm-leaf-visibility.patch`` (50 ``private ::`` ->
``public ::`` accessibility statements, nothing else).  The bar is bitwise:
``max_ulp 0`` on every live output slot, CPU and CUDA.

Four independent things are gated here, because ``max_ulp 0`` alone is not
evidence that a port is right:

* the fixture is the pinned bytes, and it still discriminates (the build-time
  validator is re-run on the committed CSVs, so the guarantee survives without
  WSL or gfortran);
* the CPU and CUDA transcriptions reproduce every live slot bit for bit;
* every argument-drop mutant is killed, or is declared inert with the WRF line
  numbers that make it inert;
* the CUDA kernels give identical output with ``--fmad=true`` and
  ``--fmad=false``, so no ``a*b+c`` boundary is left to the compiler's choice.

The device ROSR12 used by HSTEP is additionally replayed against the *already
pinned* ``rosr12`` leaf fixture in ``noahmp-leaves.csv``, so the second copy of
that routine cannot drift from the first.
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
from gpuwm.core.noahmp_thermal import THERMAL_EVALUATORS


REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = REPO_ROOT / "gpuwm" / "data" / "noahmp" / "oracle"
THERMAL_CSV = ORACLE_DIR / "noahmp-thermal.csv"
THERMAL_DISC_CSV = ORACLE_DIR / "noahmp-thermal-discrimination.csv"
LEAVES_CSV = ORACLE_DIR / "noahmp-leaves.csv"
ORACLE_TOOLS = REPO_ROOT / "tools" / "noahmp_wrf461_oracle"
THERMAL_VALIDATOR = ORACLE_TOOLS / "validate_thermal_oracle.py"

PINNED_THERMAL_ASSETS = {
    THERMAL_CSV:
        "859ea841af8d46e714f9853f0aa4360091fec5ffed7887db16c8a6de5b2b9778",
    THERMAL_DISC_CSV:
        "af1070012c059399a35237f0606064f1b8bf49382d01a919df527a6e724472e4",
}
THERMAL_NVALUES = 2162


def _load_tool(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_tool("noahmp_thermal_validator", THERMAL_VALIDATOR)


def _f32_from_bits(bits: str) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(bits, 16)),
                         dtype=np.float32)[0]


def _ulp_delta(got: np.float32, want: np.float32) -> int:
    return int(fp32_ulp_distance(got, want).reshape(-1)[0])


def _load_cases(path: Path, leaves: set[str] | None = None):
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    packed = defaultdict(lambda: {"int": {}, "in": {}, "out": {}})
    for row in rows:
        if leaves is not None and row["leaf"] not in leaves:
            continue
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


THERMAL_CASES = _load_cases(THERMAL_CSV)
THERMAL_LEAVES = sorted({leaf for leaf, _ in THERMAL_CASES})


def _require_cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as error:                        # pragma: no cover
        pytest.skip(f"CUDA unavailable: {error}")
    return cp


# ---------------------------------------------------------------- fixture --

def test_thermal_oracle_assets_are_the_pinned_bytes():
    for path, digest in PINNED_THERMAL_ASSETS.items():
        payload = path.read_bytes()
        assert b"\r" not in payload, f"{path.name} is not LF-only"
        actual = hashlib.sha256(payload).hexdigest()
        assert actual == digest, f"{path.name}: {actual} != {digest}"


def test_thermal_fixture_still_discriminates():
    """Re-run the build-time validator on the committed CSVs.

    Structure, bit round-trip, the live=0 pre-fill convention, the zero-probe
    sweep and every branch assertion, without needing WSL or gfortran.
    """
    table = VALIDATOR.load_leaves(THERMAL_CSV)
    VALIDATOR.check_structure(table)
    nvalues = VALIDATOR.check_bits(table)
    counts = VALIDATOR.check_discrimination(THERMAL_DISC_CSV)
    VALIDATOR.check_branch_coverage(table)
    assert nvalues == THERMAL_NVALUES
    assert sum(counts.values()) == sum(
        spec["n_in"] for spec in VALIDATOR.LEAVES.values())
    assert set(counts) == set(THERMAL_LEAVES)


def test_every_thermal_oracle_leaf_has_a_cpu_reference():
    assert set(THERMAL_LEAVES) == set(THERMAL_EVALUATORS), (
        f"oracle leaves {THERMAL_LEAVES} vs ports "
        f"{sorted(THERMAL_EVALUATORS)}"
    )


# -------------------------------------------------------------------- CPU --

@pytest.mark.parametrize(
    "key", sorted(THERMAL_CASES), ids=lambda key: f"{key[0]}-{key[1]}")
def test_cpu_thermal_leaf_is_bitwise(key):
    leaf, case = key
    ints, x, outs = THERMAL_CASES[key]
    got = THERMAL_EVALUATORS[leaf](x, ints)
    assert got.dtype == np.float32
    assert got.shape == (len(outs),)
    failures = []
    for slot, name, index, live, bits in outs:
        actual = np.float32(got[slot - 1])
        want = _f32_from_bits(bits) if live else np.float32(0.0)
        distance = _ulp_delta(actual, want)
        if distance:
            failures.append(
                f"{name}[{index}] slot {slot}"
                f"{'' if live else ' (undefined slot, must stay 0.0)'}: got "
                f"{struct.unpack('<I', struct.pack('<f', actual))[0]:08X}"
                f" ({actual!r}) want "
                f"{struct.unpack('<I', struct.pack('<f', want))[0]:08X}"
                f" ({want!r}), max_ulp {distance}")
    assert not failures, f"{leaf}/{case}\n" + "\n".join(failures)


# ------------------------------------------------------------------- CUDA --

@pytest.mark.gpu
@pytest.mark.parametrize("leaf", THERMAL_LEAVES)
def test_cuda_thermal_leaf_is_bitwise(leaf):
    _require_cuda()
    from gpuwm.core.noahmp_thermal_gpu import evaluate_thermal

    keys = sorted(key for key in THERMAL_CASES if key[0] == leaf)
    ints = np.stack([THERMAL_CASES[key][0] for key in keys])
    x = np.stack([THERMAL_CASES[key][1] for key in keys])
    device = evaluate_thermal(leaf, x, ints)
    assert device.dtype == np.float32
    assert device.shape == (len(keys), len(THERMAL_CASES[keys[0]][2]))
    host = device.get()

    failures = []
    for case_index, key in enumerate(keys):
        for slot, name, index, live, bits in THERMAL_CASES[key][2]:
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
def test_device_rosr12_matches_the_already_pinned_rosr12_leaf():
    """HSTEP's device ROSR12 must not drift from the pinned `rosr12` leaf.

    ``noahmp_leaves.cu`` pins ROSR12 only as a kernel body, so HSTEP cannot
    call it and ``noahmp_thermal.cu`` carries a device-function copy.  This
    replays the pinned ``rosr12`` fixture -- built by a different harness, from
    the same WRF source -- through *that copy*.
    """
    cp = _require_cuda()
    from gpuwm.core.noahmp_thermal_gpu import thermal_module

    cases = _load_cases(LEAVES_CSV, leaves={"rosr12"})
    keys = sorted(cases)
    assert keys, "the pinned rosr12 leaf fixture disappeared"
    ints = np.stack([cases[key][0] for key in keys]).astype(np.int32)
    x = np.stack([cases[key][1] for key in keys]).astype(np.float32)
    out = cp.zeros((len(keys), 21), dtype=cp.float32)
    thermal_module().get_function("noahmp_thermal_rosr12_probe")(
        (1,), (64,),
        (cp.asarray(x), cp.asarray(np.ascontiguousarray(ints)), out,
         np.int32(len(keys))))
    host = out.get()

    failures = []
    for case_index, key in enumerate(keys):
        for slot, name, index, live, bits in cases[key][2]:
            actual = np.float32(host[case_index, slot - 1])
            want = _f32_from_bits(bits) if live else np.float32(0.0)
            if _ulp_delta(actual, want):
                failures.append(f"{key[1]}: {name}[{index}] slot {slot}")
    assert not failures, "device rosr12 drift\n" + "\n".join(failures)


@pytest.mark.gpu
def test_thermal_cuda_kernels_are_immune_to_fma_contraction():
    """``--fmad=true`` and ``--fmad=false`` must give bit-identical output.

    HRT is dense in ``a*b - c*d`` and ``-(a+b)``; if contraction moved a slot,
    the parity above would hold only by the compiler's current choice.
    """
    cp = _require_cuda()
    from gpuwm.core.noahmp_thermal_gpu import THERMAL_LAYOUTS, thermal_module

    modules = {
        "fmad_on": thermal_module(("-std=c++17",)),
        "fmad_off": thermal_module(("-std=c++17", "--fmad=false")),
    }
    differences = []
    for leaf, layout in sorted(THERMAL_LAYOUTS.items()):
        keys = sorted(key for key in THERMAL_CASES if key[0] == leaf)
        host_x = np.stack([THERMAL_CASES[key][1] for key in keys])
        host_ix = np.stack(
            [THERMAL_CASES[key][0] for key in keys]).astype(np.int32)
        if layout.n_int == 0:
            host_ix = np.zeros((len(keys), 1), dtype=np.int32)
        results = {}
        for label, module in modules.items():
            out = cp.zeros((len(keys), layout.n_out), dtype=cp.float32)
            module.get_function(f"noahmp_thermal_{leaf}")(
                (1,), (64,),
                (cp.asarray(host_x),
                 cp.asarray(np.ascontiguousarray(host_ix)),
                 out, np.int32(len(keys))))
            results[label] = out.get().view(np.uint32)
        if not np.array_equal(results["fmad_on"], results["fmad_off"]):
            moved = np.argwhere(results["fmad_on"] != results["fmad_off"])
            differences.append(f"{leaf}: {len(moved)} slots move under "
                               f"contraction, first at {tuple(moved[0])}")
    assert not differences, "\n".join(differences)


@pytest.mark.gpu
def test_cuda_thermal_parity_can_fail():
    """Negative control: a one-ULP edit to the kernel must break the gate.

    A passing parity test is only evidence if it could have failed.  This
    recompiles the same translation unit with HRT's 0.5 in the bottom-boundary
    depth (:5444) moved by a single ULP and requires the fixture to reject it.
    """
    cp = _require_cuda()
    from gpuwm.core.noahmp_thermal_gpu import THERMAL_LAYOUTS, thermal_source

    original = "MU(0.5f, AD(zsnso[j - 1], zsnso[j]))"
    perturbed = "MU(0.50000006f, AD(zsnso[j - 1], zsnso[j]))"
    source = thermal_source()
    assert source.count(original) == 1, (
        "the HRT bottom-boundary literal moved; retarget this control")
    module = cp.RawModule(code=source.replace(original, perturbed),
                          options=("-std=c++17",))
    module.compile()

    layout = THERMAL_LAYOUTS["hrt"]
    keys = sorted(key for key in THERMAL_CASES if key[0] == "hrt")
    host_x = np.stack([THERMAL_CASES[key][1] for key in keys])
    host_ix = np.stack(
        [THERMAL_CASES[key][0] for key in keys]).astype(np.int32)
    out = cp.zeros((len(keys), layout.n_out), dtype=cp.float32)
    module.get_function("noahmp_thermal_hrt")(
        (1,), (64,),
        (cp.asarray(host_x), cp.asarray(np.ascontiguousarray(host_ix)),
         out, np.int32(len(keys))))
    host = out.get()

    moved = 0
    for case_index, key in enumerate(keys):
        for slot, name, index, live, bits in THERMAL_CASES[key][2]:
            if not live:
                continue
            if _ulp_delta(np.float32(host[case_index, slot - 1]),
                          _f32_from_bits(bits)):
                moved += 1
    assert moved > 0, (
        "a one-ULP edit to HRT's bottom-boundary constant left every pinned "
        "slot unchanged, so the CUDA parity test cannot fail")


# ---------------------------------------------------------------- mutation --

@pytest.mark.parametrize("leaf", THERMAL_LEAVES)
def test_thermal_fixture_kills_every_argument_mutant(leaf):
    """The fixture must detect a port that ignores an argument.

    For every input slot -- FP32 and integer topology alike -- neither zero
    nor any constant the fixture gives that slot may be substituted and still
    reproduce it, unless the argument is declared inert or partially
    observable with a WRF line-number justification.
    """
    from gpuwm.core import noahmp_leaf_mutation as mutation

    spec = VALIDATOR.LEAVES[leaf]
    evaluator = THERMAL_EVALUATORS[leaf]
    fixture = mutation.load_leaf_fixture(leaf, THERMAL_CSV)
    assert not mutation.baseline_mismatches(fixture, evaluator)
    verdicts = mutation.summarise(
        mutation.run_mutation_study(leaf, fixture, evaluator),
        inert=spec["inert"], partial=spec.get("partial", {}))
    assert len(verdicts) == fixture.x.shape[1] + fixture.ints.shape[1]
    problems = mutation.failures(verdicts)
    assert not problems, "\n".join(problems)
    for verdict in verdicts:
        if verdict.declared_inert or verdict.declared_partial:
            assert verdict.justification, (
                f"{leaf}: {verdict.name}[{verdict.index}] has no "
                f"justification")


def test_frh2o_is_dead_under_the_pinned_option_identity():
    """FRH2O is pinned but unreachable, and nothing may quietly wire it up.

    Its only call site in the module is :5692, inside ``IF (OPT_FRZ == 2)``,
    and the pinned WRF Registry identity is ``opt_frz = 1``.  ``phasechange``
    must therefore implement the :5683-5689 closed form and must never call
    ``frh2o`` -- if it ever does, the port has silently changed which option
    identity it models.
    """
    import inspect

    from gpuwm.core import noahmp_thermal

    body = inspect.getsource(noahmp_thermal.phasechange)
    assert "frh2o" not in body.split('"""')[2], (
        "phasechange calls frh2o; that is the OPT_FRZ == 2 path and the "
        "pinned identity is opt_frz = 1"
    )
    harness = (ORACLE_TOOLS / "run_thermal.F90").read_text(encoding="ascii")
    assert "call noahmp_options(4, 1, 1, 3, 1, 1," in harness, (
        "the harness no longer pins the WRF Registry default identity; "
        "opt_frz is the sixth argument and must stay 1"
    )


def test_every_inert_declaration_is_exercised():
    """No inert declaration may name a slot the fixture does not carry."""
    table = VALIDATOR.load_leaves(THERMAL_CSV)
    for leaf, spec in VALIDATOR.LEAVES.items():
        case = spec["cases"][0]
        present = {(row["name"], int(row["index"]))
                   for row in table[(leaf, case, "in")].values()}
        present |= {(row["name"], 0)
                    for row in table[(leaf, case, "int")].values()}
        for argument, reason in spec["inert"].items():
            assert argument in present, (
                f"{leaf}: inert declaration {argument} names no input slot")
            assert reason.strip(), f"{leaf}: {argument} has no reason"
