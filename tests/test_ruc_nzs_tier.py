"""The ``RUC_NZS`` tier ladder, and the proof it changed nothing at nine.

``gpuwm/core/kernels/ruc.cu`` sized every per-thread soil scratch array with a
bare literal -- ``zshalf[9]``, ``cotso[9]``, ``dtdzs[14]`` -- and read its
level table from ``__constant__ real ruc_soil_layer_depth[9]``.  That was the
whole nine-level pin on the forecast column.  The lift compiles the module at
a geometry chosen by the caller, through the same
``gpuwm.core.kernels.get_kernel_int_defines`` loader ``acoustic.cu``'s
``WPHI_MAX_LEV`` ladder uses.

The acceptance that matters is NEGATIVE: every nine-level configuration that
ran before must compile the same translation unit it compiled before.  Four
independent instruments say so here, and every one of them is CPU-only:

1. **The mechanism is inert at nine.**  :func:`ruc_module_defines` is EMPTY at
   9, so the launcher takes the unspecialized loader and the string handed to
   NVRTC is byte-identical to ``module_source("ruc")`` -- the exact string the
   pre-ladder launcher produced.  Digested and compared.

2. **The lift is exactly a macro-for-literal substitution, PLUS one named
   fix.**  The shipped source is run BACKWARDS -- the sentinel-delimited
   blocks are removed or restored, and the five macros are replaced by the
   bare literals they expand to at nine -- and the result is hashed against
   :data:`PRE_LIFT_FILE_SHA256`, the digest ``FROZEN_MODULE_DIGESTS['ruc'][0]``
   carried on 29c337754 *before* this lane touched anything.  This test
   authenticates itself against the tree's own record.  If it is red, the lift
   changed something other than a level count and the re-pinned freeze digest
   is no longer justified.

   The "plus one named fix" is the ``RUC_NZS DZSTOP`` block, kept in its own
   sentinel and its own inversion step precisely so that it cannot hide
   inside the substitution.  ``ruc_soil_finalize`` computed ``dzstop = 1 /
   (0.01f - 0.0f)`` -- WRF's NINE-level ``zsmain(2) - zsmain(1)`` written out
   as a literal instead of read from the table -- and the macro sweep walked
   past it because it is a DEPTH, not an extent.  At six levels that divides
   by 0.01 where the geometry is 0.05, and the kernel returned a ground heat
   flux five times too large: measured, before the fix, as grdflx
   -337.1 W m-2 against the host lane's -67.4 on the same column.  It now
   reads ``ruc_soil_layer_depth[1] - [0]`` like every other site in the file.
   At nine those ARE 0.01f and 0.00f, so no number moves --
   ``tests/test_ruc_nzs_device.py`` measures that on the hardware -- but the
   PTX does move, a ``__constant__`` load where an immediate was.  That is
   why the ladder's no-op and identical-PTX claims below are measured
   against a source that already carries the fix, and why the fix has a
   generated-code test of its own.

3. **The generated code is identical**, measured twice on real tools: token
   streams out of a host C preprocessor, and PTX out of ``nvcc -ptx``.  Each
   has a negative control at ``-DRUC_NZS=6`` that must DIFFER, so neither
   comparison passes merely because it cannot fail.

4. **The no-arithmetic rule is mechanized.**  Every macro the ladder defines
   expands to a BARE DECIMAL LITERAL -- ``#define RUC_NZS_M2 7``, never
   ``#define RUC_NZS_M2 (RUC_NZS - 2)``.  The second form is correct
   arithmetic and expands to ``(9 - 2)`` where the pre-lift source had the
   single token ``7``, which is what would make (2) impossible.  Note the
   rule is about the DEFINITIONS: use sites such as ``RUC_NZS - step`` are
   required by the lift and expand to the same ``9 - step`` they always did.

Every test in this file is CPU-only and imports no CuPy -- which is why
:mod:`gpuwm.core.ruc_tier` exists as its own module rather than living in
``gpuwm.core.ruc_gpu``, whose module-scope ``import cupy`` would make this
whole file unimportable on a box with no card.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from gpuwm.core.kernels import module_source
from gpuwm.core.ruc_contract import (NUM_SOIL_LAYERS,
                                     WRF_SUPPORTED_NUM_SOIL_LAYERS)
from gpuwm.core.ruc_tier import ruc_kernel_source, ruc_module_defines

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "gpuwm" / "core" / "kernels" / "ruc.cu"

#: ``sha256`` of ``gpuwm/core/kernels/ruc.cu`` as it stood on 29c337754,
#: BEFORE the RUC_NZS lift.  Written out here rather than imported from
#: ``tests/test_mp8_frozen.py``, so that moving the freeze pin cannot make
#: this file agree with itself.  It is the digest the mp=8 freeze carried,
#: which is what makes the reconstruction below self-authenticating.
PRE_LIFT_FILE_SHA256 = (
    "d446b7462e4952416d3e21482b051823766a6f675163236686c7d9fab7fbbdb7")

#: The four lines the depth table was, before the ladder selected it.
PRE_LIFT_DEPTH_TABLE = """__constant__ real ruc_soil_layer_depth[9] = {
    0.00f, 0.01f, 0.04f, 0.10f, 0.30f,
    0.60f, 1.00f, 1.60f, 3.00f
};
"""

#: Each derived macro and the bare decimal literal it expands to at the
#: shipped geometry.  LONGEST FIRST: substituting ``RUC_NZS`` before
#: ``RUC_NZS_M1`` would turn ``RUC_NZS_M1`` into ``9_M1``.
SHIPPED_MACRO_LITERALS = (
    ("RUC_NZS_M1", "8"),
    ("RUC_NZS_M2", "7"),
    ("RUC_NZS_M3", "6"),
    ("RUC_DTDZS_LEN", "14"),
    ("RUC_NZS", "9"),
)

TIER_LADDER = "RUC_NZS TIER LADDER"
DEPTH_TABLE = "RUC_NZS DEPTH TABLE"
DZSTOP = "RUC_NZS DZSTOP"

#: The one line ``ruc_soil_finalize``'s ``dzstop`` was before the fix.
#: Inverted SEPARATELY from the ladder, because it is the only edit this lane
#: made to ``ruc.cu`` that is not a macro-for-literal substitution: it changes
#: the generated code at nine while leaving every number identical.  Folding
#: it into :func:`_reconstruct_pre_lift` would let the ladder's "preprocessor
#: no-op" and "identical PTX" claims quietly cover a change that is neither.
PRE_FIX_DZSTOP = (
    "    const real dzstop = __fdiv_rn(one, __fsub_rn(0.01f, 0.0f));" + chr(10))


def _shipped() -> str:
    """``ruc.cu`` exactly as it sits on disk, with no newline translation."""
    return KERNEL.read_text(encoding="utf-8", newline="")


def _block(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"// >>> {re.escape(name)} >>>\n.*?// <<< {re.escape(name)} <<<\n",
        re.S)


def _drop_sentinel_block(text: str, name: str) -> str:
    """Remove a sentinel block AND the blank line that separates it."""
    pattern = re.compile(_block(name).pattern + "\n", re.S)
    out, count = pattern.subn("", text)
    assert count == 1, f"expected exactly one {name} block, found {count}"
    return out


def _replace_sentinel_block(text: str, name: str, body: str) -> str:
    out, count = _block(name).subn(lambda _match: body, text)
    assert count == 1, f"expected exactly one {name} block, found {count}"
    return out


def _reconstruct_pre_lift(text: str) -> str:
    """Invert the LIFT only: sentinels out, macros back to bare literals.

    Deliberately does NOT invert the ``RUC_NZS DZSTOP`` fix, so that
    the token-stream and PTX comparisons below compare two sources
    that both carry it.  What they measure is then the ladder, which
    is what they claim to measure.
    """
    text = _drop_sentinel_block(text, TIER_LADDER)
    text = _replace_sentinel_block(text, DEPTH_TABLE, PRE_LIFT_DEPTH_TABLE)
    for macro, literal in SHIPPED_MACRO_LITERALS:
        text = re.sub(rf"\b{macro}\b", literal, text)
    return text


def _reconstruct_pre_fix(text: str) -> str:
    """Undo the one named non-substitution edit, and only it.

    Runs BEFORE :func:`_reconstruct_pre_lift` on any composed inversion:
    the macro pass rewrites every ``RUC_NZS`` token in the file, the
    sentinel names included, so a block looked up by name has to be
    resolved while the name is still spelled the way the source spells it.
    """
    return _replace_sentinel_block(text, DZSTOP, PRE_FIX_DZSTOP)


# ---------------------------------------------------------------------------
# 1. The tier itself
# ---------------------------------------------------------------------------

def test_the_shipped_geometry_injects_no_define_at_all():
    """The emptiness IS the mechanism; if it is not empty, nine has moved."""
    assert ruc_module_defines(NUM_SOIL_LAYERS) == ()
    assert NUM_SOIL_LAYERS == 9


def test_the_six_level_geometry_asks_for_exactly_one_define():
    assert ruc_module_defines(6) == (("RUC_NZS", 6),)


@pytest.mark.parametrize("nzs", [0, 1, 4, 5, 7, 8, 10, 12, -9])
def test_a_geometry_wrf_does_not_define_is_refused(nzs):
    with pytest.raises(ValueError, match="is not one of"):
        ruc_module_defines(nzs)


def test_every_admitted_geometry_has_a_tier():
    for count in WRF_SUPPORTED_NUM_SOIL_LAYERS:
        defines = ruc_module_defines(count)
        assert defines == (() if count == NUM_SOIL_LAYERS
                           else (("RUC_NZS", count),))


def test_the_nine_level_source_is_the_unspecialized_module_byte_for_byte():
    """Leg 1: no nine-level run can see a different translation unit."""
    generated = ruc_kernel_source(NUM_SOIL_LAYERS)
    unspecialized = module_source("ruc")
    assert generated == unspecialized
    assert (hashlib.sha256(generated.encode("utf-8")).hexdigest()
            == hashlib.sha256(unspecialized.encode("utf-8")).hexdigest())
    assert "#define RUC_NZS 6" not in generated


def test_a_six_level_source_adds_exactly_one_line_and_nothing_else():
    """Mutation control: the comparison above CAN fail.

    Removing the one injected define must recover the unspecialized string
    byte for byte, so a tiered compile cannot smuggle in any other edit.
    """
    generated = ruc_kernel_source(6)
    unspecialized = module_source("ruc")
    assert generated != unspecialized
    injected = "#define RUC_NZS 6\n"
    assert generated.count(injected) == 1
    assert generated.replace(injected, "", 1) == unspecialized


# ---------------------------------------------------------------------------
# 2. The lift is exactly a macro-for-literal substitution
# ---------------------------------------------------------------------------

def test_the_lift_is_exactly_a_macro_for_literal_substitution():
    """Reconstruct the pre-lift ruc.cu FROM the shipped one and hash it.

    The lift is a pure textual substitution: five macros, each expanding to
    one bare decimal literal, plus two sentinel-delimited blocks.  Inverting
    it must reproduce the file the mp=8 freeze pinned BEFORE the lift, byte
    for byte.

    If this is red, the lift changed something other than a level count --
    reformatting, re-wrapping, a "while I'm here" fix -- and the re-pinned
    freeze digest is no longer justified by anything.
    """
    reconstructed = _reconstruct_pre_lift(_reconstruct_pre_fix(_shipped()))
    digest = hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
    assert digest == PRE_LIFT_FILE_SHA256, (
        "the shipped ruc.cu does not invert to the pre-lift file.  Either a "
        "non-substitution edit entered the lift without a sentinel of its "
        "own, or a macro maps to a different literal than "
        "SHIPPED_MACRO_LITERALS claims")


def test_the_named_fix_is_the_only_non_substitution_edit():
    """Inverting the ladder ALONE must NOT reach the pre-lift file.

    The digest test above passes through two inversions.  Without this, a
    second undeclared edit could be hiding inside the DZSTOP sentinel and
    the pair would still agree.  This pins the split itself: the ladder
    inversion leaves exactly one difference, and it is the dzstop line.
    """
    shipped = _shipped()
    ladder_only = _reconstruct_pre_lift(shipped)
    assert PRE_FIX_DZSTOP not in ladder_only
    assert hashlib.sha256(
        ladder_only.encode("utf-8")).hexdigest() != PRE_LIFT_FILE_SHA256, (
        "inverting the ladder alone reached the pre-lift file, so the "
        "dzstop fix is not in the shipped source at all")

    fix_only = _reconstruct_pre_fix(shipped)
    assert PRE_FIX_DZSTOP in fix_only
    assert hashlib.sha256(_reconstruct_pre_lift(fix_only).encode(
        "utf-8")).hexdigest() == PRE_LIFT_FILE_SHA256
    # The difference between the shipped file and that one is EXACTLY the
    # sentinel block -- nothing was smuggled in beside it.
    assert shipped.replace(
        _block(DZSTOP).search(shipped).group(0), PRE_FIX_DZSTOP) == fix_only


def test_the_reconstruction_can_fail():
    """Negative control for the test above.

    A one-character change to the shipped source must break the digest.
    Without this, a reconstruction that accidentally normalised the file
    would pass and prove nothing.

    The mutation is deliberately made OUTSIDE both sentinel blocks: a change
    inside one of them is *supposed* to vanish under reconstruction, so
    mutating there would test nothing.
    """
    shipped = _shipped()
    anchor = "const real* zsmain = ruc_soil_layer_depth;"
    assert anchor in shipped
    mutated = shipped.replace(anchor, anchor + " ", 1)
    assert mutated != shipped, "the mutation control mutated nothing"
    digest = hashlib.sha256(_reconstruct_pre_lift(
        _reconstruct_pre_fix(mutated)).encode("utf-8")).hexdigest()
    assert digest != PRE_LIFT_FILE_SHA256


def test_the_pre_lift_depth_table_is_the_ingest_tables_nine_level_row():
    """The literals carried inline above are not a third transcription.

    They are pinned against the table that is oracle-matched to WRF's
    ``init_soil_depth_3``, so the two cannot drift.
    """
    import numpy as np

    from gpuwm.ingest.ruc_soil import ruc_soil_depths

    literals = [float(token) for token in
                re.findall(r"(-?\d+\.\d+)f", PRE_LIFT_DEPTH_TABLE)]
    expected = np.asarray(ruc_soil_depths(9)[0], dtype=np.float32)
    assert len(literals) == 9
    assert np.array_equal(np.asarray(literals, dtype=np.float32), expected)


def _depth_table_arm(count: int) -> list[float]:
    """The float literals of one ``#if`` arm of the depth table."""
    block = _block(DEPTH_TABLE).search(_shipped())
    assert block is not None
    arm = re.search(
        rf"#(?:if|elif) RUC_NZS == {count}\n(.*?)\n#(?:elif|endif)",
        block.group(0), re.S)
    assert arm is not None, f"no depth-table arm for RUC_NZS == {count}"
    body = arm.group(1)
    initializer = re.search(r"=\s*\{(.*?)\}", body, re.S)
    assert initializer is not None, f"arm {count} has no initializer"
    return [float(token)
            for token in re.findall(r"(-?\d+\.\d+)f", initializer.group(1))]


@pytest.mark.parametrize("count", [6, 9])
def test_the_kernel_depth_table_is_the_ingest_tables_row(count):
    """Every arm's literals ARE the oracle-matched table, bit for bit.

    ``gpuwm.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M`` is the one transcription of
    WRF's ``init_soil_depth_3`` that is checked against real.exe's ZS/DZS.
    The device cannot import it -- ``__constant__`` needs literals -- so the
    literals are pinned against it here.  This is strictly stronger than the
    length check it replaces in ``tests/test_soil_layer_geometry.py``: it
    pins the VALUES, not just the count, and a length check would have been
    perfectly happy with six wrong depths.
    """
    import numpy as np

    from gpuwm.ingest.ruc_soil import ruc_soil_depths

    literals = np.asarray(_depth_table_arm(count), dtype=np.float32)
    expected = np.asarray(ruc_soil_depths(count)[0], dtype=np.float32)
    assert len(literals) == count
    assert np.array_equal(literals.view(np.uint32), expected.view(np.uint32)), (
        f"ruc.cu's {count}-level depth table is "
        f"{[hex(v) for v in literals.view(np.uint32)]}, the ingest table is "
        f"{[hex(v) for v in expected.view(np.uint32)]}")


def test_the_depth_table_has_an_arm_for_every_admitted_geometry():
    for count in WRF_SUPPORTED_NUM_SOIL_LAYERS:
        assert len(_depth_table_arm(count)) == count


def test_the_sentinels_are_present_exactly_once_each():
    """The reconstruction slices on these; a duplicate would slice wrong."""
    text = _shipped()
    for name in (TIER_LADDER, DEPTH_TABLE):
        assert text.count(f"// >>> {name} >>>\n") == 1, name
        assert text.count(f"// <<< {name} <<<\n") == 1, name


def test_the_ladder_guards_the_shipped_literal_and_derives_the_rest():
    """The ``#ifndef`` triple and the ``#if`` ladder, read structurally."""
    text = _shipped()
    lines = text.splitlines()
    defines = [i for i, line in enumerate(lines)
               if line.strip().startswith("#define RUC_NZS ")]
    assert len(defines) == 1, "exactly one RUC_NZS definition"
    i = defines[0]
    assert lines[i].strip() == "#define RUC_NZS 9"
    assert lines[i - 1].strip() == "#ifndef RUC_NZS"
    assert lines[i + 1].strip() == "#endif"
    assert lines[i + 2].strip() == "#if RUC_NZS == 9"
    assert "#elif RUC_NZS == 6" in text
    assert "#error" in text


# ---------------------------------------------------------------------------
# 3. The no-arithmetic rule, mechanized
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


#: Every macro the ladder defines, and the bare literal it must expand to at
#: the shipped geometry.
LADDER_DEFINE = re.compile(
    r"^#define (RUC_NZS|RUC_NZS_M[123]|RUC_DTDZS_LEN) +(.*)$", re.M)


def test_every_derived_macro_expands_to_a_bare_decimal_literal():
    """``RUC_NZS_M2`` is defined as ``7``, never as ``RUC_NZS - 2``.

    THIS is the no-arithmetic rule, and it is a rule about the DEFINITIONS,
    not about the use sites.  ``#define RUC_NZS_M2 (RUC_NZS - 2)`` would be
    perfectly correct arithmetic and would expand to ``(9 - 2)`` where the
    pre-lift source had the single token ``7`` -- so the preprocessed
    translation unit at nine would no longer be token-for-token what it was,
    and the cheap reconstruction proof above would die with it.

    Use sites are a different matter and are deliberately NOT restricted: the
    source says ``int kn = RUC_NZS - step;`` where it used to say
    ``int kn = 9 - step;``, and that expands to exactly the same tokens.  A
    guard that banned an operator NEXT TO a macro would forbid the fifteen
    index-arithmetic sites this lift is required to produce, while catching
    nothing the reconstruction test does not already catch exhaustively.
    """
    defines = LADDER_DEFINE.findall(_strip_comments(_shipped()))
    offenders = [(name, body) for name, body in defines
                 if not re.fullmatch(r"\d+", body.strip())]
    assert not offenders, (
        f"these ladder macros are not bare decimal literals: {offenders}.  "
        "An expression here is correct arithmetic and still breaks the "
        "token-identity of the nine-level translation unit")

    # RUC_NZS is defined once, under the #ifndef guard.  The four derived
    # macros are defined once per admitted geometry, so twice each.
    from collections import Counter
    counts = Counter(name for name, _ in defines)
    assert counts == {"RUC_NZS": 1, "RUC_NZS_M1": 2, "RUC_NZS_M2": 2,
                      "RUC_NZS_M3": 2, "RUC_DTDZS_LEN": 2}, counts

    # And each arm's values are the ones the geometry actually implies.
    body = _strip_comments(_shipped())
    for count, expected in ((9, ("8", "7", "6", "14")),
                            (6, ("5", "4", "3", "8"))):
        arm = re.search(
            rf"#(?:if|elif) RUC_NZS == {count}\n(.*?)\n#(?:elif|else|endif)",
            body, re.S)
        assert arm is not None, f"no ladder arm for RUC_NZS == {count}"
        got = tuple(m.group(2).strip() for m in
                    LADDER_DEFINE.finditer(arm.group(1)))
        assert got == expected, (
            f"RUC_NZS == {count} derives {got}, expected {expected}: "
            f"M1/M2/M3 are n-1/n-2/n-3 and DTDZS_LEN is 2*(n-2)")


def test_only_the_ladders_own_macros_are_used_in_the_body():
    """No RUC_NZS-family name may be used that the ladder does not define.

    A typo -- ``RUC_NZS_M4``, ``RUC_NZS_MI`` -- would silently preprocess to
    itself and then fail to compile only for whoever next builds the module.
    """
    defined = {name for name, _ in LADDER_DEFINE.findall(_shipped())}
    used = set(re.findall(r"\bRUC_[A-Z0-9_]+\b",
                          _strip_comments(_shipped())))
    assert used <= defined, f"undefined RUC_ macros used: {sorted(used - defined)}"


# ---------------------------------------------------------------------------
# 4. The generated code is identical, measured on real tools
# ---------------------------------------------------------------------------

def _host_preprocessor() -> list[str] | None:
    for root in (Path("C:/Program Files/Microsoft Visual Studio"),
                 Path("C:/Program Files (x86)/Microsoft Visual Studio")):
        if not root.is_dir():
            continue
        found = sorted(root.glob(
            "*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
        if found:
            return [str(found[-1]), "-nologo", "-EP", "-TP"]
    return None


def _preprocess(source: str, tmp_path: Path, name: str,
                extra: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    command = _host_preprocessor()
    assert command is not None
    path = tmp_path / name
    path.write_text(source, encoding="utf-8", newline="")
    return subprocess.run(command + list(extra) + [str(path)],
                          capture_output=True, text=True, cwd=tmp_path)


def _token_stream(source: str, tmp_path: Path, name: str,
                  extra: tuple[str, ...] = ()) -> list[str]:
    """Preprocessed non-blank lines.

    ``ruc.cu`` has no ``#include``, so the preprocessor needs no header
    search path and its output is a pure macro expansion of this one file.
    ``-EP`` keeps the vertical whitespace a removed directive or comment left
    behind, which is invisible to the compiler, so blank lines are dropped.
    """
    result = _preprocess(source, tmp_path, name, extra)
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.skipif(_host_preprocessor() is None,
                    reason="no host C preprocessor installed")
def test_the_ladder_is_a_preprocessor_no_op_at_nine(tmp_path):
    """The whole claim, run through a real preprocessor."""
    shipped = _shipped()
    assert (_token_stream(shipped, tmp_path, "shipped.cpp")
            == _token_stream(_reconstruct_pre_lift(shipped), tmp_path,
                             "pre_lift.cpp"))


@pytest.mark.skipif(_host_preprocessor() is None,
                    reason="no host C preprocessor installed")
def test_the_preprocessor_comparison_can_fail(tmp_path):
    """Negative control: selecting six levels must move the token stream."""
    shipped = _shipped()
    assert (_token_stream(shipped, tmp_path, "six.cpp", ("-DRUC_NZS=6",))
            != _token_stream(_reconstruct_pre_lift(shipped), tmp_path,
                             "pre_lift.cpp"))


@pytest.mark.skipif(_host_preprocessor() is None,
                    reason="no host C preprocessor installed")
@pytest.mark.parametrize("nzs", [4, 5, 7, 8, 12])
def test_an_unadmitted_geometry_stops_the_compile(tmp_path, nzs):
    """``#error`` fires, so a bad tier is a build failure not a bad column."""
    result = _preprocess(_shipped(), tmp_path, f"bad{nzs}.cpp",
                         (f"-DRUC_NZS={nzs}",))
    assert result.returncode != 0
    assert "RUC_NZS must be 6 or 9" in (result.stdout + result.stderr)


def _nvcc() -> list[str] | None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    command = [nvcc, "-ptx", "-std=c++17", "-arch=sm_120"]
    host = _host_preprocessor()
    if host is not None:
        command += ["-ccbin", str(Path(host[0]).parent)]
    return command


def _ptx(source: str, tmp_path: Path, name: str,
         extra: tuple[str, ...] = ()) -> str:
    command = _nvcc()
    assert command is not None
    src = tmp_path / f"{name}.cu"
    out = tmp_path / f"{name}.ptx"
    src.write_text(source, encoding="utf-8", newline="")
    result = subprocess.run(command + list(extra) + ["-o", str(out), str(src)],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    return re.sub(r"^//.*$", "", out.read_text(encoding="utf-8"), flags=re.M)


def _assembled(cu_text: str) -> str:
    """``module_source``'s preamble in front of an arbitrary ruc.cu body."""
    assembled = module_source("ruc")
    shipped = _shipped()
    assert assembled.endswith(shipped)
    return assembled[:len(assembled) - len(shipped)] + cu_text


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_ladder_generates_identical_ptx_at_nine(tmp_path):
    """Stronger than the token comparison, and still device-free."""
    shipped = _shipped()
    assert (_ptx(_assembled(shipped), tmp_path, "shipped")
            == _ptx(_assembled(_reconstruct_pre_lift(shipped)), tmp_path,
                    "pre_lift"))


#: A kernel that is not in ``ruc.cu``, appended to make a source that must
#: compile to different PTX.  The six-level tier is NOT used as the negative
#: control here: until the depth table grows its ``#elif RUC_NZS == 6`` arm,
#: ``-DRUC_NZS=6`` does not compile at all, and "it failed to build" is not
#: evidence that ``_ptx`` can tell two working sources apart.  The six-level
#: PTX comparison lives in
#: :func:`test_the_six_level_tier_generates_different_ptx` instead.
_PTX_PROBE = """
__global__ void ruc_ptx_instrument_probe(float* out) { out[0] = 1.0f; }
"""


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_ptx_comparison_can_fail(tmp_path):
    """Negative control: the instrument distinguishes two sources.

    Without this, ``test_the_ladder_generates_identical_ptx_at_nine`` could
    be passing because ``_ptx`` returns the same string for everything.
    """
    shipped = _shipped()
    assert (_ptx(_assembled(shipped), tmp_path, "shipped")
            != _ptx(_assembled(shipped + _PTX_PROBE), tmp_path, "probed"))


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_named_fix_is_a_real_change_to_the_generated_code(tmp_path):
    """The dzstop fix is NOT a preprocessor no-op, and must not read as one.

    Everything else in this file argues that the lift generates identical
    code at nine.  The fix does not: reading ``ruc_soil_layer_depth[1]``
    where an immediate stood is a ``__constant__`` load, and if the PTX came
    out identical it would mean the compiler folded the table back into an
    immediate -- which is exactly the ptxas folding this file's own depth
    table is ``__constant__`` to prevent.

    So this asserts the change is visible in the generated code at nine,
    while ``tests/test_ruc_nzs_device.py`` asserts no NUMBER moves there.
    Two different claims, and conflating them is what would let a real fix
    be waved through as "inert".
    """
    shipped = _shipped()
    assert (_ptx(_assembled(shipped), tmp_path, "fixed")
            != _ptx(_assembled(_reconstruct_pre_fix(shipped)), tmp_path,
                    "pre_fix"))


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_six_level_tier_compiles_and_generates_different_ptx(tmp_path):
    """The six-level module is a real translation unit, not a hope.

    Two claims in one compile, both device-free: ``-DRUC_NZS=6`` BUILDS --
    every scratch extent, every derived bound and the ``#elif`` depth-table
    arm agree well enough for nvcc's front end and ptxas -- and what it
    builds is genuinely different code, which is the negative control the
    nine-level identity test needs.
    """
    shipped = _shipped()
    assert (_ptx(_assembled(shipped), tmp_path, "six", ("-DRUC_NZS=6",))
            != _ptx(_assembled(shipped), tmp_path, "nine"))


# ---------------------------------------------------------------------------
# 5. The host lane's geometry closures
# ---------------------------------------------------------------------------

def test_wrfs_default_root_count_is_in_range_at_every_admitted_geometry():
    """``gpuwm.core.ruc``'s ``chosen = 4`` is WRF's :797 nroot fallback.

    It is a fixed level index, not a count that scales with the column, so it
    is only safe while ``4 <= n - 1`` for every admitted ``n``.  That holds at
    6 and at 9.  If the admitted set ever gains a shorter geometry, this fails
    here rather than by ``_root_count_field`` rejecting an nroot the physics
    itself produced.
    """
    assert min(WRF_SUPPORTED_NUM_SOIL_LAYERS) - 1 >= 4


@pytest.mark.parametrize("count", [6, 9])
def test_the_host_and_device_zshalf_derivations_agree(count):
    """One transcription on the host; the same expression on the device.

    ``ruc_zshalf`` replaced three identical loops in ``gpuwm.core.ruc`` and a
    fourth in ``gpuwm.core.ruc_gpu``.  The device builds the same interface
    depths from ``__constant__ ruc_soil_layer_depth`` in seven kernels.  Both
    are ``fadd_rn`` then a multiply by 0.5, which is exact for finite float32,
    so they must agree bit for bit -- and the literals the device uses are the
    ones this file already pins against the ingest table.
    """
    import numpy as np

    from gpuwm.core.ruc import ruc_soil_geometry, ruc_zshalf

    zs, _ = ruc_soil_geometry(count)
    host = ruc_zshalf(zs)

    device_literals = np.asarray(_depth_table_arm(count), dtype=np.float32)
    device = np.zeros(count, dtype=np.float32)
    for level in range(1, count):
        device[level] = np.float32(
            np.float32(device_literals[level - 1] + device_literals[level])
            * np.float32(0.5))

    assert np.array_equal(host.view(np.uint32), device.view(np.uint32))
    assert host[0] == np.float32(0.0)
