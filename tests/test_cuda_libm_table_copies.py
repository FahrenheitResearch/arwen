"""Every CUDA transcription of a glibc libm table, held to one spelling.

THE BREAKAGE THIS PREVENTS
--------------------------
``0x3FEFD9B0D3158574`` -- row 1 of glibc's ``__exp2f_data`` -- is written out
by hand in **eight** ``.cu`` files and once more, in Python, in
``gpuwm/core/noahmp_libm.py``.  Nine independent transcriptions of one table.

The 2026-08-28 fault-injection audit corrupted that row by ONE ULP and
measured what the estate does about it:

* in ``noahmp_soilwater.cu`` alone -- caught in 4.5 s by
  ``tests/test_noahmp_kernel_source_scans.py``, and only because
  ``noahmp_water.cu`` happens to carry a verbatim copy of the same SECTION,
  so a text-equality check on that pair notices.  That gate is a duplication
  artefact, not a table gate.
* the same ULP applied to ``noahmp_soilwater.cu`` AND ``noahmp_water.cu``
  together, so the pair still agrees -- **NOT CAUGHT** by anything.
* the same ULP in ``noahmp_snow.cu``, one of the six copies no equality check
  covers at all -- **NOT CAUGHT** by anything.

A one-ULP error in an exp2f table is a wrong exponential in a land-surface
kernel on every GPU column, on every step, for the life of the release.
Nothing in the estate compared any ``.cu`` table against the Python spelling,
and nothing compared the copies against each other.  This is that comparison.

WHAT IT ASSERTS
---------------
1. Every ``__constant__ unsigned long long`` table with the same NAME SUFFIX
   (``EXP2F_TAB``, ``LOGF_TAB``, ...) holds identical values in every kernel
   source that declares it.  A single drifted copy names itself.
2. Every such suffix that has a Python counterpart in
   ``gpuwm.core.noahmp_libm`` equals it bit pattern for bit pattern.  This is
   the anchor that survives all eight copies drifting TOGETHER, which is the
   case cross-copy equality cannot see.
3. A suffix appearing in the kernels that this file does not register is a
   failure, so a newly hand-transcribed table cannot arrive unguarded.

The scan is text only: no CUDA device, no nvcc, no kernel import.
"""

from __future__ import annotations

import pathlib
import re
import struct
from collections import defaultdict

import pytest

from gpuwm.core import noahmp_libm

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNELS = REPOSITORY_ROOT / "gpuwm" / "core" / "kernels"

#: ``__constant__ unsigned long long NAME[N] = { ... };``
_TABLE = re.compile(
    r"__constant__\s+unsigned\s+long\s+long\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<n>\d+)\s*\]\s*=\s*"
    r"\{(?P<body>[^}]*)\}\s*;",
    re.S)
_WORD = re.compile(r"0[xX]([0-9A-Fa-f]+)ULL")

#: Prefixes the kernel families use to namespace their private copy of a
#: shared table.  The SUFFIX is the table's identity; the prefix says only
#: which family pasted it.  Extending this is metadata, not a code path.
_PREFIXES = ("C_", "MYNN_", "VP_")

#: A suffix whose copies are the same glibc table, mapped to the attribute of
#: ``gpuwm.core.noahmp_libm`` that spells it.  Value ``None`` means "shared
#: across kernels, but with no single Python spelling to anchor against":
#: cross-copy equality is then the whole of its gate, and saying so here is
#: the point -- an omission that is written down is not an omission.
SHARED_TABLES: dict[str, str | None] = {
    "EXP2F_TAB": "_EXP2F_TAB",
    "EXP2F_POLY": "_EXP2F_POLY",
    "EXP2F_POLY_SCALED": "_EXP2F_POLY_SCALED",
    "LOGF_TAB": "_LOGF_TAB",
    "POWF_LOG2_TAB": "_POWF_TAB",
    "POWF_LOG2_POLY": "_POWF_A",
    "POWF_POLY": "_POWF_A",
    # e_powf_log2_data.c stores one table of (invc, logc) pairs.  The
    # vegprecip kernel splits it into two 16-entry arrays instead of one
    # interleaved 32-entry array; same numbers, different packing, so each
    # half is anchored against its stride of the Python table.
    "POWF_INVC": "_POWF_TAB[::2]",
    "POWF_LOGC": "_POWF_TAB[1::2]",
    # The shift/invln2/ln2 constants each family inlines beside its tables.
    # They are identical across every copy and that is checkable; they have
    # no single tuple in noahmp_libm to name, so cross-copy is their gate.
    "EXP2F_MISC": None,
    "LOGF_MISC": None,
    "POWF_MISC": None,
}

#: Suffixes that are NOT one shared table: the same identifier is reused by
#: different kernels for different content.  Cross-copy equality would be a
#: false failure, so they are excluded here WITH the reason, rather than by
#: the scan quietly not matching them.
PER_KERNEL_TABLES: dict[str, str] = {
    "LIMITS": (
        "each kernel's own branch/clamp limits; declared [3] in "
        "noahmp_soilwater.cu and [4] in noahmp_bareflux.cu and "
        "noahmp_radiation.cu, so the copies are deliberately different"),
    "EXPF_LIMITS": (
        "noahmp_snow.cu only: the expf argument range that kernel clamps to; "
        "one declaration, nothing to compare it against"),
}


def _as_double(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFFFFFFFFFFFFFF))[0]


def _python_values(spelling: str) -> tuple[int, ...]:
    """The Python table as a flat tuple of float64 BIT PATTERNS.

    ``noahmp_libm`` stores most tables as Python floats built from hex-float
    literals and one (``_EXP2F_TAB``) as the raw bit patterns glibc stores.
    Comparing bit patterns rather than floats is deliberate: it is the
    comparison that sees a 1-ULP edit, which comparing with ``==`` on floats
    would also see but comparing with ``math.isclose`` would not.  Pairs are
    flattened because the kernels write the same numbers as a flat array.
    """

    attribute, _, slicing = spelling.partition("[")
    flat: list[int] = []
    for item in getattr(noahmp_libm, attribute):
        parts = item if isinstance(item, tuple) else (item,)
        for part in parts:
            flat.append(part if isinstance(part, int) else
                        struct.unpack("<Q", struct.pack("<d", part))[0])
    if slicing:
        # The slice is over the FLATTENED table, because that is the shape
        # the kernels write: e_powf_log2_data.c stores (invc, logc) pairs and
        # a kernel that splits them takes every other flat entry, not every
        # other pair.  Slicing the pairs instead silently compares invc rows
        # against logc rows, which is a failure that looks like real drift.
        start, _, rest = slicing.rstrip("]").partition(":")
        stop, _, step = rest.partition(":")
        flat = flat[slice(int(start) if start else None,
                          int(stop) if stop else None,
                          int(step) if step else None)]
    return tuple(flat)


def _tables() -> dict[str, dict[str, tuple[int, ...]]]:
    """``{suffix: {relative kernel path: bit patterns}}``."""

    found: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
    sources = sorted(list(KERNELS.rglob("*.cu")) + list(KERNELS.rglob("*.cuh")))
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _TABLE.finditer(text):
            name = match.group("name")
            suffix = name
            for prefix in _PREFIXES:
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    break
            values = tuple(int(word, 16)
                           for word in _WORD.findall(match.group("body")))
            declared = int(match.group("n"))
            rel = path.relative_to(REPOSITORY_ROOT).as_posix()
            assert len(values) == declared, (
                f"{rel}: {name} declares [{declared}] and lists {len(values)} "
                "values; the transcription lost or gained a row")
            found[suffix][rel] = values
    return dict(found)


TABLES = _tables()
_SHARED_FOUND = sorted(s for s in TABLES if s in SHARED_TABLES)
_ANCHORED = sorted(s for s in _SHARED_FOUND if SHARED_TABLES[s])


def test_the_scan_found_the_kernel_tables_at_all() -> None:
    """The instrument, tested in the direction that would go quiet.

    If a refactor renames the declarations or moves the kernel tree, every
    comparison below becomes vacuously true and the file still passes.  These
    are the counts measured at 659962929; they are a floor, not a pin, so
    adding a kernel does not fail them.
    """

    assert TABLES, (
        f"no __constant__ unsigned long long table was found under {KERNELS}; "
        "the scan no longer matches the kernels and every comparison below "
        "is silently vacuous")
    assert len(TABLES.get("EXP2F_TAB", {})) >= 8, (
        f"the glibc __exp2f_data transcription was found in "
        f"{len(TABLES.get('EXP2F_TAB', {}))} kernel source(s); eight carried "
        "it at 659962929, so a copy going unread means the scan stopped "
        "reading a file, not that a copy was removed")
    assert len(_ANCHORED) >= 7, (
        f"only {len(_ANCHORED)} table(s) are anchored to a Python spelling; "
        "seven were at 659962929")


@pytest.mark.parametrize("suffix", _SHARED_FOUND)
def test_every_kernel_copy_of_a_shared_table_agrees(suffix: str) -> None:
    """Copy against copy.  Catches a ULP edited into ONE transcription."""

    copies = TABLES[suffix]
    distinct: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for rel, values in sorted(copies.items()):
        distinct[values].append(rel)
    if len(distinct) == 1:
        return
    groups = sorted(distinct.items(), key=lambda kv: (-len(kv[1]), kv[1]))
    report = [f"    {len(files)} file(s) {files}\n"
              f"      rows 0..3: {[hex(v) for v in values[:4]]}"
              for values, files in groups]
    raise AssertionError(
        f"the {suffix} table is transcribed differently in different kernel "
        f"sources; {len(copies)} copies fall into {len(distinct)} groups:\n"
        + "\n".join(report)
        + "\n  These are all the same glibc table.  One copy was edited.")


@pytest.mark.parametrize("suffix", _ANCHORED)
def test_every_kernel_table_matches_its_python_spelling(suffix: str) -> None:
    """Kernel against Python.  Catches every copy drifting together."""

    spelling = SHARED_TABLES[suffix]
    expected = _python_values(spelling)
    for rel, values in sorted(TABLES[suffix].items()):
        assert len(values) == len(expected), (
            f"{rel}: {suffix} holds {len(values)} rows and "
            f"noahmp_libm.{spelling} holds {len(expected)}; the CUDA and "
            "Python transcriptions of one glibc table are different lengths")
        for index, (got, want) in enumerate(zip(values, expected)):
            assert got == want, (
                f"{rel}: {suffix}[{index}] is {got:#018x} "
                f"({_as_double(got)!r}) and noahmp_libm.{spelling}[{index}] "
                f"is {want:#018x} ({_as_double(want)!r}).  These are the same "
                "glibc table written twice; the CUDA kernel and the Python "
                "reference now disagree on exp2f/logf/powf.")


@pytest.mark.parametrize("suffix", sorted(TABLES))
def test_every_kernel_table_suffix_is_registered(suffix: str) -> None:
    """A newly hand-transcribed table cannot arrive unguarded."""

    assert suffix in SHARED_TABLES or suffix in PER_KERNEL_TABLES, (
        f"a __constant__ unsigned long long table named *{suffix} appears in "
        f"{sorted(TABLES[suffix])} and this file registers neither a shared "
        "spelling nor a per-kernel reason for it.  Add it to SHARED_TABLES "
        "with the noahmp_libm attribute it transcribes (or None), or to "
        "PER_KERNEL_TABLES with the reason its copies legitimately differ.  "
        "An unregistered table is guarded by nothing, which is the state the "
        "2026-08-28 audit measured for six of the eight exp2f copies.")


def test_the_per_kernel_exclusions_still_describe_the_tree() -> None:
    """A stale exclusion is a hole.  Retire it when its reason expires."""

    stale = sorted(name for name in PER_KERNEL_TABLES if name not in TABLES)
    assert not stale, (
        f"PER_KERNEL_TABLES excludes {stale} from cross-copy equality and no "
        "kernel declares them any more; drop the entry so the exclusion list "
        "cannot outlive the reason it was written for")
