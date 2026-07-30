"""gpuwm-vs-WRF parity for the Noah-MP parameter transfer.

The fixture `gpuwm/data/noahmp/oracle/noahmp-parameters.csv` is
`NOAHMP_TABLES::read_mp_*` followed by
`module_sf_noahmpdrv::TRANSFER_MP_PARAMETERS`, run by unmodified WRF v4.6.1
over seven land identities x 181 transferred field/index pairs. These tests
drive `gpuwm.core.noahmp.transfer_mp_parameters` over the same identities and
require bitwise agreement in WRF's own working precision (IEEE binary32).

Every one of the 181 pairs is accounted for: each is classified as a direct
table lookup, an urban-override target, or one of the two quantities WRF
derives inside the transfer. The blocks WRF's pinned option identity leaves
formally undefined -- the crop branch, the irrigation block, the MPTABLE
entries that are read but never transferred, and FRZX for soil type 14 -- are
asserted to be absent from both the fixture and the port, so the remaining gap
is executable rather than a comment.
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
from gpuwm.core.noahmp import (
    FRZX_UNDEFINED_SOILTYPE,
    URBAN_CSOIL_OVERRIDE,
    URBAN_SOIL_OVERRIDE,
    load_noahmp_parameters,
    transfer_mp_parameters,
)
from gpuwm.core.noahmp_leaves import LEAF_EVALUATORS


ORACLE_PATH = (
    Path(__file__).parents[1]
    / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-parameters.csv"
)
ORACLE_SHA256 = "1a2617bad69ff050f728b71c46ce13a4624091a68488d1cb2b59fcda106eaa35"
ORACLE_ROWS = 1267
ORACLE_CASES = 7
ORACLE_FIELDS = 181
NSOIL = 4

# Quantities TRANSFER_MP_PARAMETERS computes rather than looks up
# (module_sf_noahmpdrv.F lines 1692 and 1707-1708).
DERIVED_FIELDS = ("KDT", "FRZX")
# Quantities the URBAN_FLAG branch overwrites after the table lookup
# (module_sf_noahmpdrv.F lines 1696-1702).
URBAN_OVERRIDDEN_FIELDS = tuple(URBAN_SOIL_OVERRIDE) + ("CSOIL",)
DERIVED_PAIRS = 2
URBAN_OVERRIDDEN_PAIRS = 4 * NSOIL + 1
DIRECT_LOOKUP_PAIRS = ORACLE_FIELDS - DERIVED_PAIRS - URBAN_OVERRIDDEN_PAIRS

# Components TRANSFER_MP_PARAMETERS can assign that neither the fixture nor
# gpuwm covers, and why. Names are WRF's noahmp_parameters components.
UNCOVERED_CROP_BRANCH = (
    "PLTDAY", "HSDAY", "PLANTPOP", "IRRI", "GDDTBASE", "GDDTCUT", "GDDS1",
    "GDDS2", "GDDS3", "GDDS4", "GDDS5", "C3C4", "AREF", "PSNRF", "I2PAR",
    "TASSIM0", "TASSIM1", "TASSIM2", "K", "EPSI", "Q10MR", "FOLN_MX",
    "LEFREEZ", "DILE_FC", "DILE_FW", "FRA_GR", "LF_OVRC", "ST_OVRC",
    "RT_OVRC", "LFMR25", "STMR25", "RTMR25", "GRAINMR25", "LFPT", "STPT",
    "RTPT", "GRAINPT", "LFCT", "STCT", "RTCT", "BIO2LAI",
)
UNCOVERED_IRRIGATION_BLOCK = (
    "IRR_FRAC", "IRR_HAR", "IRR_LAI", "IRR_MAD", "FILOSS", "SPRIR_RATE",
    "MICIR_RATE", "FIRTFAC", "IR_RAIN",
)
UNCOVERED_READ_BUT_NEVER_TRANSFERRED = (
    "SLAREA", "EPS1", "EPS2", "EPS3", "EPS4", "EPS5",
)


def _bits(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", np.float32(value)))[0]


def _ulp(want: float, got: float) -> int:
    return abs(_bits(want) - _bits(got))


def _load_rows() -> list[dict[str, str]]:
    with ORACLE_PATH.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _cases(rows):
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    return grouped


@pytest.fixture(scope="module")
def oracle_rows():
    return _load_rows()


@pytest.fixture(scope="module")
def bundle():
    return load_noahmp_parameters()


def _transfer(bundle, meta):
    return transfer_mp_parameters(
        bundle,
        vegtype=int(meta["vegtype"]),
        soiltype=[int(meta["soiltype"])] * NSOIL,
        slopetype=int(meta["slopetype"]),
        soilcolor=int(meta["soilcolor"]),
    )


def test_parameter_oracle_fixture_identity_is_the_pinned_one():
    payload = ORACLE_PATH.read_bytes()
    assert b"\r" not in payload
    assert hashlib.sha256(payload).hexdigest() == ORACLE_SHA256
    rows = _load_rows()
    assert len(rows) == ORACLE_ROWS == ORACLE_CASES * ORACLE_FIELDS
    assert len({row["case"] for row in rows}) == ORACLE_CASES


def test_every_transferred_field_is_reproduced_bitwise(bundle, oracle_rows):
    """gpuwm must match WRF exactly on all 7 x 181 transferred values."""

    worst: dict[tuple[str, int], int] = {}
    compared = 0
    mismatches: list[str] = []
    for case, rows in _cases(oracle_rows).items():
        parameters = _transfer(bundle, rows[0])
        for row in rows:
            field, index = row["field"], int(row["index"])
            want = float(row["value"])
            got = float(parameters.field(field, index))
            distance = _ulp(want, got)
            compared += 1
            worst[(field, index)] = max(worst.get((field, index), 0), distance)
            if distance:
                mismatches.append(
                    f"{case}: {field}({index}) WRF={want!r} gpuwm={got!r} "
                    f"ulp={distance}"
                )
    assert compared == ORACLE_ROWS
    assert not mismatches, "\n".join(mismatches)
    assert max(worst.values()) == 0
    assert len(worst) == ORACLE_FIELDS


def test_field_inventory_is_completely_partitioned(oracle_rows):
    """No field may be silently skipped: the 181 pairs must partition."""

    pairs = {(row["field"], int(row["index"])) for row in oracle_rows}
    assert len(pairs) == ORACLE_FIELDS

    derived = {pair for pair in pairs if pair[0] in DERIVED_FIELDS}
    urban = {pair for pair in pairs if pair[0] in URBAN_OVERRIDDEN_FIELDS}
    direct = pairs - derived - urban

    assert len(derived) == DERIVED_PAIRS
    assert len(urban) == URBAN_OVERRIDDEN_PAIRS
    assert len(direct) == DIRECT_LOOKUP_PAIRS
    assert derived | urban | direct == pairs
    assert not (derived & urban) and not (derived & direct) and not (urban & direct)


def test_direct_lookups_come_from_the_packaged_table_bytes(bundle, oracle_rows):
    """The 162 pure-lookup pairs must be table bytes, not fitted numbers."""

    for case, rows in _cases(oracle_rows).items():
        parameters = _transfer(bundle, rows[0])
        for row in rows:
            field, index = row["field"], int(row["index"])
            if field in DERIVED_FIELDS:
                continue
            if parameters.urban_flag and field in URBAN_OVERRIDDEN_FIELDS:
                continue
            got = np.float32(float(parameters.field(field, index)))
            assert _bits(got) == _bits(float(row["value"])), (
                f"{case}: direct lookup {field}({index}) disagrees with WRF"
            )


def test_derived_kdt_and_frzx_follow_the_wrf_expressions(bundle, oracle_rows):
    """KDT and FRZX must reproduce WRF's own binary32 arithmetic exactly."""

    frzk = np.float32(bundle.general.values["FRZK_DATA"])
    for case, rows in _cases(oracle_rows).items():
        table = {(row["field"], int(row["index"])): np.float32(float(row["value"]))
                 for row in rows}
        parameters = _transfer(bundle, rows[0])

        kdt = np.float32(
            np.float32(table[("REFKDT", 0)] * table[("DKSAT", 1)])
            / table[("REFDK", 0)]
        )
        assert _bits(kdt) == _bits(table[("KDT", 0)]), f"{case}: KDT"
        assert _bits(parameters.field("KDT", 0)) == _bits(table[("KDT", 0)])

        frzfact = np.float32(
            np.float32(table[("SMCMAX", 1)] / table[("SMCREF", 1)])
            * np.float32(np.float32(0.412) / np.float32(0.468))
        )
        frzx = np.float32(frzk * frzfact)
        assert _bits(frzx) == _bits(table[("FRZX", 0)]), f"{case}: FRZX"
        assert _bits(parameters.field("FRZX", 0)) == _bits(table[("FRZX", 0)])


def test_urban_flag_branch_matches_wrf_on_both_sides(bundle, oracle_rows):
    """The URBAN_FLAG override, and its knock-on effect on FRZX."""

    urban_cases = []
    for case, rows in _cases(oracle_rows).items():
        parameters = _transfer(bundle, rows[0])
        table = {(row["field"], int(row["index"])): float(row["value"])
                 for row in rows}
        assert bool(table[("URBAN_FLAG", 0)]) == parameters.urban_flag
        if not parameters.urban_flag:
            continue
        urban_cases.append(case)
        for name, override in URBAN_SOIL_OVERRIDE.items():
            for layer in range(1, NSOIL + 1):
                assert _bits(parameters.field(name, layer)) == _bits(override)
                assert _bits(table[(name, layer)]) == _bits(override)
        assert _bits(parameters.scalar("CSOIL")) == _bits(URBAN_CSOIL_OVERRIDE)
        assert _bits(table[("CSOIL", 0)]) == _bits(URBAN_CSOIL_OVERRIDE)
        # FRZX is formed after the override, so it must use 0.45/0.42.
        expected = np.float32(
            np.float32(bundle.general.values["FRZK_DATA"])
            * np.float32(
                np.float32(np.float32(0.45) / np.float32(0.42))
                * np.float32(np.float32(0.412) / np.float32(0.468))
            )
        )
        assert _bits(parameters.field("FRZX", 0)) == _bits(expected)
    assert urban_cases == ["urban"]


def test_slope_lookup_uses_genparm_slope_table(bundle, oracle_rows):
    for case, rows in _cases(oracle_rows).items():
        parameters = _transfer(bundle, rows[0])
        slopetype = int(rows[0]["slopetype"])
        assert _bits(parameters.scalar("SLOPE")) == _bits(
            bundle.general.slopes[slopetype - 1]
        ), f"{case}: SLOPE_TABLE({slopetype})"


def test_uncovered_transfer_blocks_are_explicitly_absent(bundle, oracle_rows):
    """The gap is executable: these components are NOT yet oracle-validated.

    * the crop branch (`CROPTYPE > 0`) -- `opt_crop = 0` gives `croptype = 0`,
      so WRF never assigns it and the fixture never emits it;
    * the irrigation block -- `iopt_irr = 0` skips `read_mp_irrigation_
      parameters`, leaving `parameters%IRR_*` formally undefined;
    * `SLAREA` and `EPS1..EPS5` -- read from MPTABLE, never transferred.
    """

    assert len(UNCOVERED_CROP_BRANCH) == 41
    assert len(UNCOVERED_IRRIGATION_BLOCK) == 9
    assert len(UNCOVERED_READ_BUT_NEVER_TRANSFERRED) == 6
    emitted = {row["field"] for row in oracle_rows}
    uncovered = (
        UNCOVERED_CROP_BRANCH
        + UNCOVERED_IRRIGATION_BLOCK
        + UNCOVERED_READ_BUT_NEVER_TRANSFERRED
    )
    assert not (emitted & set(uncovered)), "fixture unexpectedly covers a gap"

    parameters = _transfer(bundle, _cases(oracle_rows)["grassland"][0])
    for name in uncovered:
        assert name not in parameters.values, (
            f"{name} became available without an oracle to validate it"
        )
    with pytest.raises(NotImplementedError, match="crop branch"):
        transfer_mp_parameters(
            bundle, vegtype=12, soiltype=[8] * NSOIL, slopetype=1,
            soilcolor=4, croptype=1,
        )


def test_frzx_is_undefined_for_the_water_soil_category(bundle):
    """WRF leaves FRZX unset for SOILTYPE(1) == 14; gpuwm must not invent it."""

    parameters = transfer_mp_parameters(
        bundle, vegtype=10, soiltype=[FRZX_UNDEFINED_SOILTYPE] * NSOIL,
        slopetype=1, soilcolor=4,
    )
    assert "FRZX" not in parameters.values
    with pytest.raises(KeyError):
        parameters.scalar("FRZX")
    # SOILPARM.TBL row 14 is all-zero hydraulics with SMCREF = 0, so FRZFACT
    # would divide by zero -- that is the reason WRF guards the assignment.
    assert parameters.field("SMCREF", 1) == 0.0
    # Everything else still transfers for that soil category.
    assert parameters.scalar("KDT") == 0.0
    assert parameters.scalar("SLOPE") == pytest.approx(0.1)


def test_transfer_rejects_out_of_range_identities(bundle):
    for kwargs, message in (
        (dict(vegtype=21, soiltype=[6] * NSOIL, slopetype=1, soilcolor=4),
         "vegtype"),
        (dict(vegtype=10, soiltype=[20] * NSOIL, slopetype=1, soilcolor=4),
         "soiltype"),
        (dict(vegtype=10, soiltype=[6] * NSOIL, slopetype=10, soilcolor=4),
         "slopetype"),
        (dict(vegtype=10, soiltype=[6] * NSOIL, slopetype=1, soilcolor=9),
         "soilcolor"),
    ):
        with pytest.raises(ValueError, match=message):
            transfer_mp_parameters(bundle, **kwargs)


# ===========================================================================
# Per-leaf solver oracles
#
# `gpuwm/data/noahmp/oracle/noahmp-leaves.csv` is produced by
# `tools/noahmp_wrf461_oracle/build_leaves.sh` from a *scratch copy* of the
# pinned WRF tree carrying only `patches/noahmp-lsm-leaf-visibility.patch`
# (50 `private ::` -> `public ::` accessibility statements, nothing else).
# The bar is bitwise: max_ulp 0 on every live output slot, CPU and CUDA.
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = REPO_ROOT / "gpuwm" / "data" / "noahmp" / "oracle"
LEAVES_CSV = ORACLE_DIR / "noahmp-leaves.csv"
DISC_CSV = ORACLE_DIR / "noahmp-leaves-discrimination.csv"
VISIBILITY_PATCH = REPO_ROOT / "patches" / "noahmp-lsm-leaf-visibility.patch"
ORACLE_TOOLS = REPO_ROOT / "tools" / "noahmp_wrf461_oracle"

PINNED_LEAF_ASSETS = {
    LEAVES_CSV:
        "cb985a6c05ae261e1d8d74c797719cd230be2128a190163b36061e46365903ec",
    DISC_CSV:
        "a82343047b724225cc66ca5398323a491822f6fc2f03c67b821e184b2bf7a218",
    VISIBILITY_PATCH:
        "164885a9ce956a51be2ab189165da8b60217b636a720c994e04dfb862a05aad5",
}


def _load_tool(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _f32_from_bits(bits: str) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(bits, 16)),
                         dtype=np.float32)[0]


def _ulp_delta(got: np.float32, want: np.float32) -> int:
    """Signed-magnitude ULP distance between two FP32 values."""
    return int(fp32_ulp_distance(got, want).reshape(-1)[0])


def _load_leaf_cases():
    with LEAVES_CSV.open(newline="", encoding="ascii") as stream:
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


LEAF_CASES = _load_leaf_cases()
LEAF_NAMES = sorted({leaf for leaf, _ in LEAF_CASES})


def test_leaf_oracle_assets_are_the_pinned_bytes():
    for path, digest in PINNED_LEAF_ASSETS.items():
        payload = path.read_bytes()
        assert b"\r" not in payload, f"{path.name} is not LF-only"
        actual = hashlib.sha256(payload).hexdigest()
        assert actual == digest, f"{path.name}: {actual} != {digest}"


def test_visibility_patch_body_is_visibility_only():
    """Re-audit the exposure patch without touching the pinned WRF tree.

    The full audit (pristine + patched source hashes, line counts, PRIVATE
    entity attributes) runs inside `build_leaves.sh`. Here we check the patch
    body itself: every removed line is a `private ::` accessibility statement,
    every added line is that statement with `public`, and the lifted symbol
    list is exactly the pinned 50.
    """
    checker = _load_tool("noahmp_visibility_check",
                         ORACLE_TOOLS / "check_visibility_patch.py")
    removed, added = [], []
    for line in VISIBILITY_PATCH.read_text(encoding="ascii").splitlines():
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
    assert len(removed) == len(added) == 50
    lifted = []
    for old, new in zip(removed, added):
        old_match = checker._PRIVATE_STMT.match(old)
        new_match = checker._PUBLIC_STMT.match(new)
        assert old_match is not None, old
        assert new_match is not None, new
        assert old_match.group(1) == new_match.group(1), (old, new)
        assert old_match.group(2) == new_match.group(2), (old, new)
        lifted.append(checker._SYMBOL.match(old_match.group(2)).group(1))
    assert tuple(lifted) == checker.LIFTED_SYMBOLS


def test_leaf_fixture_still_discriminates():
    """The fixture must still be able to detect a deleted input.

    This re-runs the build-time validator on the committed CSVs, so the
    guarantee survives without WSL or gfortran: every input slot of every leaf
    either moves an output when zeroed, or is in that leaf's declared inert set
    with a WRF line number attached.
    """
    validator = _load_tool("noahmp_leaf_validator",
                           ORACLE_TOOLS / "validate_leaf_oracle.py")
    table = validator.load_leaves(LEAVES_CSV)
    validator.check_structure(table)
    nvalues = validator.check_bits(table)
    counts = validator.check_discrimination(DISC_CSV)
    validator.check_branch_coverage(table)
    assert nvalues == 1570
    assert sum(counts.values()) == sum(
        spec["n_in"] for spec in validator.LEAVES.values())
    assert set(counts) == set(LEAF_NAMES)


def test_every_oracle_leaf_has_a_cpu_reference():
    assert set(LEAF_NAMES) == set(LEAF_EVALUATORS), (
        f"oracle leaves {LEAF_NAMES} vs ports {sorted(LEAF_EVALUATORS)}"
    )


@pytest.mark.parametrize(
    "key", sorted(LEAF_CASES), ids=lambda key: f"{key[0]}-{key[1]}")
def test_cpu_leaf_is_bitwise(key):
    leaf, case = key
    ints, x, outs = LEAF_CASES[key]
    got = LEAF_EVALUATORS[leaf](x, ints)
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


def _require_cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as error:                        # pragma: no cover
        pytest.skip(f"CUDA unavailable: {error}")
    return cp


@pytest.mark.parametrize("leaf", LEAF_NAMES)
def test_cuda_leaf_is_bitwise(leaf):
    _require_cuda()
    from gpuwm.core.noahmp_leaves_gpu import evaluate_leaf

    keys = sorted(key for key in LEAF_CASES if key[0] == leaf)
    ints = np.stack([LEAF_CASES[key][0] for key in keys])
    x = np.stack([LEAF_CASES[key][1] for key in keys])
    device = evaluate_leaf(leaf, x, ints)
    assert device.dtype == np.float32
    assert device.shape == (len(keys), len(LEAF_CASES[keys[0]][2]))
    host = device.get()

    failures = []
    for case_index, key in enumerate(keys):
        for slot, name, index, live, bits in LEAF_CASES[key][2]:
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


@pytest.mark.parametrize("leaf", LEAF_NAMES)
def test_leaf_fixture_kills_every_argument_mutant(leaf):
    """The fixture must be able to detect a port that ignores an argument.

    Bitwise parity alone does not establish this: a sibling lane in this project
    hit ``max_ulp 0`` on 29 columns and then found 13 of 14 argument-drop
    mutants still reproduced its pinned CSV. For every input slot -- FP32 and
    integer topology alike -- this asserts that neither zero nor any constant
    the fixture gives that slot can be substituted and still reproduce it,
    unless the argument is declared inert or partially observable with a WRF
    line-number justification.
    """
    from gpuwm.core import noahmp_leaf_mutation as mutation

    fixture = mutation.load_leaf_fixture(leaf)
    assert not mutation.baseline_mismatches(fixture)
    verdicts = mutation.summarise(mutation.run_mutation_study(leaf, fixture))
    assert len(verdicts) == fixture.x.shape[1] + fixture.ints.shape[1]
    problems = mutation.failures(verdicts)
    assert not problems, "\n".join(problems)
    # Every declaration must be exercised, so a stale one cannot linger.
    for verdict in verdicts:
        if verdict.declared_inert or verdict.declared_partial:
            assert verdict.justification, (
                f"{leaf}: {verdict.name}[{verdict.index}] has no justification")


def test_cuda_kernels_are_immune_to_fma_contraction():
    """-fmad=true and -fmad=false must give bit-identical output.

    If they differ, some ``a*b+c`` boundary in the kernel is still unpinned and
    the parity above holds only by the compiler's current choice.
    """
    cp = _require_cuda()
    from gpuwm.core.kernels import _preamble
    from gpuwm.core.noahmp_leaves_gpu import LEAF_LAYOUTS

    source = _preamble() + (
        REPO_ROOT / "gpuwm" / "core" / "kernels" / "noahmp_leaves.cu"
    ).read_text(encoding="ascii")
    modules = {}
    for label, options in (("fmad_on", ("-std=c++17",)),
                           ("fmad_off", ("-std=c++17", "--fmad=false"))):
        module = cp.RawModule(code=source, options=options)
        module.compile()
        modules[label] = module

    differences = []
    for leaf, layout in sorted(LEAF_LAYOUTS.items()):
        keys = sorted(key for key in LEAF_CASES if key[0] == leaf)
        host_x = np.stack([LEAF_CASES[key][1] for key in keys])
        host_ix = np.stack([LEAF_CASES[key][0] for key in keys]).astype(np.int32)
        if layout.n_int == 0:
            host_ix = np.zeros((len(keys), 1), dtype=np.int32)
        results = {}
        for label, module in modules.items():
            out = cp.zeros((len(keys), layout.n_out), dtype=cp.float32)
            module.get_function(f"noahmp_leaf_{leaf}")(
                (1,), (64,),
                (cp.asarray(host_x), cp.asarray(np.ascontiguousarray(host_ix)),
                 out, np.int32(len(keys))))
            results[label] = out.get().view(np.uint32)
        if not np.array_equal(results["fmad_on"], results["fmad_off"]):
            moved = np.argwhere(results["fmad_on"] != results["fmad_off"])
            differences.append(f"{leaf}: {len(moved)} slots move under "
                               f"contraction, first at {tuple(moved[0])}")
    assert not differences, "\n".join(differences)


# --------------------------------------------------------------------------
# glibc transcendental identity
# --------------------------------------------------------------------------

GLIBC_LIBM_CSV = ORACLE_DIR / "glibc-libm-fp32.csv"
GLIBC_LIBM_SHA256 = \
    "15a65f24a0fb9c6204298639089b5dceafad03a7d79b0b31f1928e681d828e3e"


def _load_libm_reference():
    with GLIBC_LIBM_CSV.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def test_glibc_libm_reference_is_the_pinned_bytes():
    payload = GLIBC_LIBM_CSV.read_bytes()
    assert b"\r" not in payload
    assert hashlib.sha256(payload).hexdigest() == GLIBC_LIBM_SHA256
    rows = _load_libm_reference()
    assert len(rows) == 30000
    assert {row["call"] for row in rows} == {"logf", "log10f", "expf", "powf"}


def test_cpu_libm_shims_reproduce_glibc_bitwise():
    """The CPU transcendentals must be glibc's, not correctly-rounded ones.

    glibc 2.39's log10f is the 1993 SunPro FP32 reduction and disagrees with
    the correctly-rounded result on 18.47% of the FP32 domain (0.1, 1.0] --
    exactly TDFCND's LOG10(SATRATIO) domain -- so an FP64-then-round shim
    cannot reach max_ulp 0 on a Noah-MP soil column.
    """
    from gpuwm.core import noahmp_libm

    calls = {
        "logf": lambda a, b: noahmp_libm.logf(a),
        "log10f": lambda a, b: noahmp_libm.log10f(a),
        "expf": lambda a, b: noahmp_libm.expf(a),
        "powf": noahmp_libm.powf,
    }
    counts: dict[str, int] = {}
    failures = []
    for row in _load_libm_reference():
        call = row["call"]
        a = _f32_from_bits(row["a"])
        b = _f32_from_bits(row["b"]) if row["b"] else np.float32(0.0)
        got = np.float32(calls[call](a, b))
        want = _f32_from_bits(row["result"])
        counts[call] = counts.get(call, 0) + 1
        if struct.pack("<f", got) != struct.pack("<f", want):
            failures.append(
                f"{call}({row['a']}{',' + row['b'] if row['b'] else ''}) got "
                f"{struct.unpack('<I', struct.pack('<f', got))[0]:08X} "
                f"want {row['result']}")
    assert not failures, "\n".join(failures[:20])
    assert counts == {"logf": 4000, "log10f": 8000, "expf": 6000,
                      "powf": 12000}


def test_cuda_libm_shims_reproduce_glibc_bitwise():
    """The device transcendentals must agree with glibc too, not just with CPU."""
    cp = _require_cuda()
    from gpuwm.core.kernels import _preamble

    probe = """
extern "C" __global__ void nmp_libm_probe(
    const int* __restrict__ call, const unsigned int* __restrict__ a,
    const unsigned int* __restrict__ b, unsigned int* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real x = __uint_as_float(a[i]);
    real y = __uint_as_float(b[i]);
    real r;
    switch (call[i]) {
      case 0: r = r_log(x); break;
      case 1: r = r_log10(x); break;
      case 2: r = r_exp(x); break;
      default: r = r_pow(x, y); break;
    }
    out[i] = __float_as_uint(r);
}
"""
    source = _preamble() + (
        REPO_ROOT / "gpuwm" / "core" / "kernels" / "noahmp_leaves.cu"
    ).read_text(encoding="ascii") + probe
    module = cp.RawModule(code=source, options=("-std=c++17",))
    module.compile()

    order = {"logf": 0, "log10f": 1, "expf": 2, "powf": 3}
    rows = _load_libm_reference()
    call = np.array([order[row["call"]] for row in rows], dtype=np.int32)
    a = np.array([int(row["a"], 16) for row in rows], dtype=np.uint32)
    b = np.array([int(row["b"], 16) if row["b"] else 0 for row in rows],
                 dtype=np.uint32)
    want = np.array([int(row["result"], 16) for row in rows], dtype=np.uint32)

    out = cp.zeros(len(rows), dtype=cp.uint32)
    threads = 128
    module.get_function("nmp_libm_probe")(
        ((len(rows) + threads - 1) // threads,), (threads,),
        (cp.asarray(call), cp.asarray(a), cp.asarray(b), out,
         np.int32(len(rows))))
    got = out.get()
    bad = np.argwhere(got != want).ravel()
    detail = [f"{rows[i]['call']}({rows[i]['a']},{rows[i]['b']}) got "
              f"{got[i]:08X} want {want[i]:08X}" for i in bad[:20]]
    assert bad.size == 0, (
        f"{bad.size} of {len(rows)} device transcendentals disagree with "
        f"glibc 2.39\n" + "\n".join(detail))
