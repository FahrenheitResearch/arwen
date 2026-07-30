"""How much the Noah-MP kernels duplicate, and where the copies have drifted.

``gpuwm/core/noahmp_kernel_sources.py`` states that glibc's ``powf``/``expf``/
``logf`` are "transcribed **once** in this tree, in ``noahmp_leaves.cu``,
because two copies of a 32-entry constant table can drift and only one of them
would be audited".  That is the right reason.  It is also no longer true: five
other kernels carry their own transcriptions under a different spelling
(``glibc_powf`` rather than ``r_pow``), in three mutually different
generations.  The drift the docstring was written to prevent has happened, out
of sight of the naming convention that was supposed to prevent it.

This matters beyond tidiness.  The remaining route to the device ceiling is to
compose Noah-MP's ENERGY subsystems into one translation unit, and every one of
these symbols then collides.  The lane's own report recorded that step as
"merging ten proved-identical device copies", i.e. free.  It is not free:

* ``f_min``/``f_max`` come in two orderings, ``(a < b) ? a : b`` and
  ``(b < a) ? b : a``.  These agree on ordinary values and disagree on
  **signed zeros and NaN** -- ``f_min(+0.0, -0.0)`` is ``-0.0`` under the first
  and ``+0.0`` under the second.  ``-ftz=true`` is appended by CuPy
  unconditionally and gfortran does not flush, so signed zeros are exactly the
  input class this project has already been bitten by three times.  Picking
  either spelling for a merged definition changes the other group's arithmetic
  until each call site is shown not to reach a signed zero.
* ``glibc_powf`` comes in three.  ``noahmp_water.cu`` carries two early exits
  (``1**y`` and ``+0 ** finite-positive``) that ``noahmp_bareflux.cu`` lacks,
  and its own comment records why: "CANWATER hits it whenever FWET is exactly
  0, which is every dry-canopy case".  Without them ``bareflux``'s copy returns
  NaN for a zero base.  That is **not** a live defect there -- WRF's SFCDIF1
  (phys/module_sf_noahmplsm.F) computes ``MOZ`` and ``MOZ2`` from the same
  Monin-Obukhov length with positive numerators, so ``MOZ < 0`` implies
  ``MOZ2 < 0`` and both bases exceed one whenever the branch is taken -- but it
  is a difference a merge has to resolve deliberately rather than by whichever
  copy happens to be pasted first.
* ``noahmp_vegeflux.cu`` is an older generation throughout: ``uint32_t``
  spellings, a ``USE_DEVICE_LIBM`` escape (negative-control only, and not
  reached on the shipped path), and **literal float constants** where the newer
  files use ``__constant__`` tables.  ptxas 12.x mis-folds FP32 ties at compile
  time and ``__constant__`` is the only barrier this project has measured to
  work, so that generation is the one to retire, not the one to merge onto.
  Its ``expf`` overflow threshold is the adjacent float to the newer one
  (``88.72283935546875f`` against a double ``88.72283172607422``), which makes
  the two disagree at exactly one input.

So the inventory is pinned here rather than asserted away.  A merge can shrink
it one symbol at a time, with each step re-gated against that consumer's own
oracle; what it must not do is grow, and no new duplicate may appear
unrecorded.
"""

from __future__ import annotations

import collections
import hashlib
import re
from pathlib import Path

import pytest

from gpuwm.core.noahmp_kernel_sources import KERNEL_DIR

_DEFINITION = re.compile(r"^__device__[^\n(]*?([A-Za-z_]\w*)\s*\(", re.M)
_NOT_A_NAME = frozenset({"if", "for", "while", "return", "switch"})


def _normalise(body: str) -> str:
    """Comment- and whitespace-insensitive, token-sensitive."""
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    return re.sub(r"\s+", " ", body).strip()


def device_definitions(text: str) -> dict[str, str]:
    """``{symbol: sha256-prefix of the normalised body}`` for one source."""
    found: dict[str, str] = {}
    for match in _DEFINITION.finditer(text):
        name = match.group(1)
        if name in _NOT_A_NAME:
            continue
        brace = text.find("{", match.start())
        if brace < 0 or ";" in text[match.end():brace]:
            continue  # a forward declaration, not a definition
        depth = 0
        body = None
        for index in range(brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    body = text[match.start():index + 1]
                    break
        if body is None:
            continue
        found.setdefault(
            name,
            hashlib.sha256(_normalise(body).encode()).hexdigest()[:10])
    return found


def duplicate_inventory(kernel_dir: Path) -> dict[str, tuple[tuple[str, ...], ...]]:
    """``{symbol: (group, group, ...)}`` for every symbol defined twice or more.

    Each group is the sorted stems whose bodies are byte-equal after comment
    and whitespace normalisation.  One group means the copies agree; more than
    one means they have drifted.
    """
    by_symbol: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for path in sorted(kernel_dir.glob("noahmp*.cu")):
        stem = path.stem.replace("noahmp_", "")
        for name, digest in device_definitions(
                path.read_text(encoding="utf-8", errors="replace")).items():
            by_symbol[name][stem] = digest

    inventory: dict[str, tuple[tuple[str, ...], ...]] = {}
    for name, per_file in by_symbol.items():
        if len(per_file) < 2:
            continue
        groups: dict[str, list[str]] = collections.defaultdict(list)
        for stem, digest in per_file.items():
            groups[digest].append(stem)
        inventory[name] = tuple(sorted(
            tuple(sorted(stems)) for stems in groups.values()))
    return inventory


#: Every device symbol defined in more than one Noah-MP kernel, and which
#: files agree with which.  A single group means the copies are identical and
#: the merge is mechanical; two or more means they have drifted and the merge
#: needs an adjudication recorded beside it.
RECORDED: dict[str, tuple[tuple[str, ...], ...]] = {
    # --- identical copies: mechanical to merge -----------------------------
    "col_load": (("snow", "water"),),
    "col_store": (("snow", "water"),),
    "d_canwater": (("soilwater", "water"),),
    "d_infil": (("soilwater", "water"),),
    "d_rosr12": (("soilwater", "water"),),
    "d_soilwater": (("soilwater", "water"),),
    "d_srt": (("soilwater", "water"),),
    "d_sstep": (("soilwater", "water"),),
    "d_wdfcnd1": (("soilwater", "water"),),
    "d_wdfcnd2": (("soilwater", "water"),),
    "f_neg": (("soilwater", "water"),),
    "nmp_checkint": (("fluxprep", "leaves"),),
    # Added with the subnormal fix: `__double2float_rn` flushes a subnormal
    # result to zero on this toolchain, so every device expf/powf disagreed
    # with `gpuwm.core.noahmp_libm` for expf arguments in [-103.616, -87.337).
    # It went into ALL EIGHT copies in the same edit precisely so this row
    # stays a single group -- fixing one and not the others is how `r_pow`
    # would have acquired a ninth generation.  The eight are every Noah-MP
    # kernel that transcribes glibc's expf or powf and converts the binary64
    # result with `__double2float_rn`; `vegeflux` is absent because it uses a
    # plain `(float)` cast, which is a separate site and a separate argument.
    # tests/test_noahmp_slab_libm.py sweeps the whole band for leaves and
    # fluxprep, tests/test_noahmp_kernel_subnormals.py for the other six, and
    # both keep a live control showing the hardware conversion still flushing.
    "nmp_d2f_rn": (("bareflux", "fluxprep", "leaves", "radiation", "snow",
                    "soilwater", "vegprecip", "water"),),
    "nmp_exp2_core": (("fluxprep", "leaves"),),
    "nmp_powf_log2": (("fluxprep", "leaves"),),
    "nmp_zeroinfnan": (("fluxprep", "leaves"),),
    "r_exp": (("fluxprep", "leaves"),),
    "r_log": (("fluxprep", "leaves"),),
    "r_pow": (("fluxprep", "leaves"),),
    # --- drifted copies: each needs an adjudication ------------------------
    "combine": (("snow",), ("water",)),
    "combo": (("snow",), ("water",)),
    "compact": (("snow",), ("water",)),
    "divide": (("snow",), ("water",)),
    "snowfall": (("snow",), ("water",)),
    "snowh2o": (("snow",), ("water",)),
    "snowwater": (("snow",), ("water",)),
    "f_abs": (("sflx",), ("soilwater", "water")),
    "f_max": (("bareflux",), ("radiation", "snow", "soilwater", "water")),
    "f_min": (("bareflux",), ("radiation", "snow", "soilwater", "water")),
    "glibc_atanf": (("bareflux",), ("vegeflux",)),
    "glibc_expf": (("radiation", "soilwater", "water"), ("snow",),
                   ("vegeflux",)),
    "glibc_logf": (("bareflux",), ("radiation",), ("vegeflux",)),
    "glibc_powf": (("bareflux", "radiation"), ("soilwater", "water"),
                   ("vegeflux",)),
    "powf_exp2_inline": (("bareflux", "radiation", "soilwater", "water"),
                         ("vegeflux",)),
    "powf_log2_inline": (("bareflux", "radiation", "soilwater", "water"),
                         ("vegeflux",)),
}

#: Symbols whose copies have drifted.  Derived, not typed, so it cannot fall
#: out of step with the table above.
DRIFTED = frozenset(
    name for name, groups in RECORDED.items() if len(groups) > 1)

#: Drifted, but provably equivalent: the bodies differ only in the *spelling*
#: of names, not in the operations or the values behind them.  These can be
#: merged without an arithmetic argument, and separating them from the rest is
#: the difference between "16 problems" and "8 problems and 8 renames".
#:
#: * the seven snow routines are byte-identical between ``noahmp_snow.cu`` and
#:   ``noahmp_water.cu`` except that ``water`` prefixes every constant with
#:   ``SN_`` to avoid colliding with its own.  Both prefixes resolve through
#:   ``C_F32`` and ``C_SN_F32``, which
#:   :func:`test_the_two_snow_constant_tables_hold_the_same_values` shows hold
#:   the same 32 values in the same order.
#: * ``f_abs`` is ``fabsf(x)`` in one copy and an explicit ``& 0x7FFFFFFF`` in
#:   the other.  Fortran's ``ABS`` on ``REAL(4)`` is a sign-bit clear, and so
#:   is ``fabsf``; the two agree on every input including ``-0.0``.
SAFE_RENAMES = frozenset({
    "combine", "combo", "compact", "divide", "snowfall", "snowh2o",
    "snowwater", "f_abs",
})

#: Drifted in a way that changes results on some input.  These are the ones a
#: merge has to adjudicate, each against its consumers' own oracle.
ARITHMETIC_DRIFT = DRIFTED - SAFE_RENAMES


def test_the_duplicate_inventory_is_exactly_what_is_recorded():
    """The whole table, so a merge shows up as a diff rather than silence."""
    assert duplicate_inventory(KERNEL_DIR) == RECORDED


def test_the_drift_has_not_grown():
    """A cheaper assertion that names the number, for a fast read.

    35, not the 34 first recorded: ``nmp_d2f_rn`` was added to recover the
    subnormal ``expf``/``powf`` results the hardware double-to-float
    conversion flushes.  It has since spread from two files to eight, and the
    count is *still* 35 -- the row grew, no row appeared.  That is the shape
    this fix was meant to have: eight byte-identical copies are one group, and
    a group is one merge.  The count of *drifted* symbols is what this test is
    really about, and that is unchanged at 16 -- neither ``glibc_expf`` nor
    ``powf_exp2_inline`` split, because every copy took the identical edit.
    """
    assert len(RECORDED) == 35
    assert len(DRIFTED) == 16, sorted(DRIFTED)
    assert len(SAFE_RENAMES) == 8, sorted(SAFE_RENAMES)
    assert ARITHMETIC_DRIFT == {
        "f_min", "f_max", "glibc_atanf", "glibc_expf", "glibc_logf",
        "glibc_powf", "powf_exp2_inline", "powf_log2_inline",
    }, sorted(ARITHMETIC_DRIFT)


def _constant_table(text: str, name: str) -> list[str]:
    match = re.search(
        r"__constant__[^\n]*\b" + name + r"\s*\[[^\]]*\]\s*=\s*\{(.*?)\}",
        text, re.S)
    assert match is not None, f"{name} not found"
    body = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.S)
    return [v.strip() for v in body.replace("\n", " ").split(",") if v.strip()]


def test_the_two_snow_constant_tables_hold_the_same_values():
    """Why the seven snow routines count as renames rather than as drift.

    ``noahmp_water.cu`` prefixes every snow constant with ``SN_`` so it does
    not collide with its own.  That is only a rename if the tables behind the
    two prefixes agree, so the values are compared here rather than assumed --
    element for element, as raw bit patterns, with the comments stripped
    because the comments are exactly where the two files differ.
    """
    snow = _constant_table(
        (KERNEL_DIR / "noahmp_snow.cu").read_text(encoding="utf-8"), "C_F32")
    water = _constant_table(
        (KERNEL_DIR / "noahmp_water.cu").read_text(encoding="utf-8"),
        "C_SN_F32")
    assert len(snow) == 32 and len(water) == 32
    assert snow == water


def test_the_libm_is_not_transcribed_once_despite_the_docstring():
    """Pin the contradiction until the merge resolves it.

    ``noahmp_kernel_sources`` says the glibc transcription exists once.  It
    exists in ``leaves``/``fluxprep`` as ``r_pow``/``r_exp``/``r_log`` and
    again, differently, as ``glibc_powf``/``glibc_expf``/``glibc_logf``.  When
    that is fixed this test is what tells the author to update the docstring.
    """
    inventory = duplicate_inventory(KERNEL_DIR)
    assert inventory["r_pow"] == (("fluxprep", "leaves"),)
    assert len(inventory["glibc_powf"]) == 3
    assert "bareflux" in {stem for group in inventory["glibc_powf"]
                          for stem in group}


def test_the_min_max_orderings_really_do_disagree_on_a_signed_zero():
    """The divergence is arithmetic, not cosmetic -- shown, not asserted.

    Evaluated in Python on the same IEEE-754 semantics the two C expressions
    have.  ``-ftz`` does not enter: both zeros are representable either way.
    """
    def first(a: float, b: float) -> float:
        return a if a < b else b      # bareflux

    def second(a: float, b: float) -> float:
        return b if b < a else a      # radiation/snow/soilwater/water

    import math
    got_first = first(0.0, -0.0)
    got_second = second(0.0, -0.0)
    assert math.copysign(1.0, got_first) == -1.0
    assert math.copysign(1.0, got_second) == 1.0
    assert math.copysign(1.0, got_first) != math.copysign(1.0, got_second)


# --------------------------------------------------------------------------
# Negative controls
# --------------------------------------------------------------------------

def _scratch(tmp_path: Path, name: str, text: str) -> Path:
    kernels = tmp_path / "kernels"
    kernels.mkdir(exist_ok=True)
    (kernels / name).write_text(text, encoding="utf-8")
    return kernels


def test_two_identical_copies_report_one_group(tmp_path):
    body = "__device__ float f_min(float a, float b) { return a < b ? a : b; }\n"
    kernels = _scratch(tmp_path, "noahmp_a.cu", body)
    (kernels / "noahmp_b.cu").write_text(body, encoding="utf-8")
    assert duplicate_inventory(kernels) == {"f_min": (("a", "b"),)}


def test_a_drifted_copy_reports_two_groups(tmp_path):
    kernels = _scratch(
        tmp_path, "noahmp_a.cu",
        "__device__ float f_min(float a, float b) { return a < b ? a : b; }\n")
    (kernels / "noahmp_b.cu").write_text(
        "__device__ float f_min(float a, float b) { return b < a ? b : a; }\n",
        encoding="utf-8")
    assert duplicate_inventory(kernels) == {"f_min": (("a",), ("b",))}


def test_comments_and_whitespace_do_not_count_as_drift(tmp_path):
    kernels = _scratch(
        tmp_path, "noahmp_a.cu",
        "__device__ float f_min(float a, float b) { return a < b ? a : b; }\n")
    (kernels / "noahmp_b.cu").write_text(
        "__device__ float f_min(float a,\n"
        "                       float b)\n"
        "{\n"
        "    // the same function, laid out differently\n"
        "    return a < b ? a : b;\n"
        "}\n", encoding="utf-8")
    assert duplicate_inventory(kernels) == {"f_min": (("a", "b"),)}


def test_a_forward_declaration_is_not_a_definition(tmp_path):
    kernels = _scratch(
        tmp_path, "noahmp_a.cu",
        "__device__ float f_min(float a, float b);\n"
        "__device__ float other(float a) { return a; }\n")
    (kernels / "noahmp_b.cu").write_text(
        "__device__ float f_min(float a, float b) { return a < b ? a : b; }\n"
        "__device__ float other(float a) { return a; }\n", encoding="utf-8")
    assert duplicate_inventory(kernels) == {"other": (("a", "b"),)}


def test_a_symbol_defined_once_is_not_in_the_inventory(tmp_path):
    kernels = _scratch(
        tmp_path, "noahmp_a.cu",
        "__device__ float only_here(float a) { return a; }\n")
    assert duplicate_inventory(kernels) == {}
