"""No routine, interface or transcribed block exists twice in the oracle.

WHY THIS EXISTS.  The harness held three hand-written copies of the
``cu_ntiedtke_pre_run`` transcription, two of ``cu_ntiedtke_run``'s
conversion block, and two of ``foealfa``/``foeewm``.  Exactly one copy of
each was proved; the others carried a comment pointing at the proved one
("run_nt_prep.F90 proves this replication exact").  It does not -- it
proves its own copy, and nothing compared the copies.

That is the port's own recurring failure -- **resolution by apparent
identity instead of by provenance** -- appearing inside the oracle that
exists to prevent it.  The copies did agree, measured: all 52 recorded
CSVs are byte-identical after consolidation.  The exposure was structural,
not numerical, and a structural exposure needs a structural gate.

WHAT SHAPE THIS TAKES, and it is the shape that has worked here.  It does
NOT list the three names that went wrong -- a hand-maintained list is what
failed.  It walks every Fortran file in the oracle directory and fails on
any *second* definition, so the next duplicate lands here without anyone
remembering this happened.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

_DIR = (Path(__file__).resolve().parents[1] / "tools" / "ntiedtke_wrf461_oracle")

#: Fortran sources of the oracle harness, including the .inc files, which
#: are the transcriptions and therefore the whole point.
_SOURCES = sorted(list(_DIR.glob("*.F90")) + list(_DIR.glob("*.inc")))

#: Any function definition, whatever the result-type prefix.  Two ways this
#: line has already been wrong, both silent:
#:
#:   1. it matched only ``real(kind=kind_phys) function`` and so missed
#:      ``logical function wne``, which had three copies -- a gate that
#:      matches only the shapes you thought of finds only what you expected;
#:   2. the pattern's \b reached the file as a literal backspace byte
#:      (0x08), so it matched NOTHING and the duplicate test passed on an
#:      empty set.  Bisected afterwards, in docs/ntiedtke/PORT-RECORD.md section 29:
#:      the tool transport halves a DOUBLED backslash, and a non-raw Python
#:      literal then reads the surviving \b as an escape.  Raw strings
#:      with single backslashes are immune both ways, which is what the two
#:      patterns below use.  (This very comment held a 0x08 for one commit,
#:      written the same way while documenting the problem.)
#:
#: Both were caught by the vacuity guard below, not by review.
_FUNC = re.compile(r"^\s*[\w()=,.\s]*?\bfunction\s+(\w+)\s*\(", re.M | re.I)
_SUBR = re.compile(r"^\s*subroutine\s+(\w+)\s*\(", re.M | re.I)


def test_the_oracle_directory_is_where_this_thinks_it_is():
    """A gate that scans nothing passes vacuously."""
    assert _SOURCES, f"no Fortran sources under {_DIR}"
    names = {p.name for p in _SOURCES}
    for expected in ("nt_cases.F90", "run_nt_prep.F90", "run_nt_cuinin.F90",
                     "run_nt_cumastrn.F90", "nt_run_conversion.inc",
                     "nt_cumastrn_body.inc"):
        assert expected in names, f"{expected} missing from {sorted(names)}"


def _definitions(pattern):
    seen = defaultdict(list)
    for path in _SOURCES:
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            seen[m.group(1).lower()].append(path.name)
    return seen


def test_the_patterns_match_the_things_they_are_meant_to_match():
    """The other half of "a gate that scans nothing passes vacuously".

    The file set is guarded above; this guards the PATTERNS, and it has
    earned its place twice already -- see the note on _FUNC. Both times the
    duplicate test passed while seeing nothing, which is the worst way for
    a gate to fail: it reports the answer you wanted.

    Naming the procedures the regexes must see turns a silent miss into a
    failure. The names are chosen to span the shapes: a plain ``function``,
    a ``logical function``, a ``real(kind=kind_phys) function``, a bind(C)
    interface body and an ordinary module subroutine.
    """
    funcs = set(_definitions(_FUNC))
    subs = set(_definitions(_SUBR))
    for name in ("hexw", "wne", "nt_foealfa", "nt_foeewm"):
        assert name in funcs, f"the function pattern no longer sees {name}"
    for name in ("x_pre_run", "x_post_run", "x_cuinin", "x_cuascn",
                 "nt_build_column"):
        assert name in subs, f"the subroutine pattern no longer sees {name}"


def test_no_procedure_is_defined_or_declared_twice():
    """One name, one body.  Interfaces count -- three copies of an
    interface would reintroduce exactly what three copies of a body did.

    ``end subroutine``/``end function`` lines do not match the patterns, so
    a single definition registers once.
    """
    dupes = {}
    for pattern in (_FUNC, _SUBR):
        for name, files in _definitions(pattern).items():
            if len(files) > 1:
                dupes[name] = files
    assert not dupes, (
        f"defined or declared in more than one place: {dupes}. "
        f"Two copies of a transcription is two things to keep in step, and "
        f"only one of them will be graded.")


# ANCHORED. The first version had no word boundaries, so it matched
# `zzz = kte - ppp` as readily as `zz = kte - pp` -- an OVER-match, the
# failure mode no vacuity guard can see, found by the negative control
# in test_ntiedtke_pattern_controls.py on that file's first run.
_FLIP = re.compile(
    r"\bzz\s*=\s*(?:kte|nz|nz1|kte\s*\+\s*1)\s*[-+]\s*(?:1\s*-\s*)?pp\b", re.I)


def test_the_vertical_flip_is_transcribed_in_exactly_one_file():
    """The flip is the port's one structural inversion (section 2).

    It was hand-copied into three harnesses.  Two are now calls to the real
    routine.  The third is deliberate and must stay: ``run_nt_prep.F90``
    keeps its transcription in order to compare it, field by field, against
    the real ``cu_ntiedtke_pre_run`` -- that comparison IS the direct proof,
    and deleting the transcription would delete the proof.

    So this is not "the flip appears nowhere". It is "the flip appears in
    the one file that grades it", which is a different and checkable claim.
    """
    where = {p.name: len(_FLIP.findall(p.read_text(encoding="utf-8")))
             for p in _SOURCES}
    where = {k: v for k, v in where.items() if v}
    assert set(where) == {"run_nt_prep.F90"}, (
        f"a hand-written pre_run flip survives outside the file that grades "
        f"it: {where}. The real routine is linkable now.")


def test_the_file_that_transcribes_the_flip_also_grades_it():
    """The other half, and the half that would rot silently.

    A transcription kept "in order to be compared" is worth nothing the
    moment the comparison goes. This fails if run_nt_prep stops calling the
    real routine, or stops checking every field against it.
    """
    src = (_DIR / "run_nt_prep.F90").read_text(encoding="utf-8")
    assert "call x_pre_run(" in src, "run_nt_prep no longer calls the real routine"
    assert "call x_post_run(" in src
    # Fifteen pre_run fields and eight post_run fields, named in the
    # failure report so a mismatch says WHICH field.
    for field in ("prsl", "ghtl", "omg", "tf", "qvf", "qcf", "qif", "uf",
                  "vf", "qvftenz", "thftenz", "prsi", "ghti", "slimsk"):
        assert f"r_{field}" in src, f"pre_run field {field} is no longer compared"
    for field in ("raincv", "pratec", "rthcuten", "rqvcuten", "rqccuten",
                  "rqicuten", "rucuten", "rvcuten"):
        assert f"p_{field}" in src, f"post_run field {field} is no longer compared"


def test_the_conversion_block_lives_only_in_its_include():
    """``cu_ntiedtke_run``:228-277 is genuinely unreachable -- it is inside
    a public routine, not a callable one -- so it must be transcribed. Once.

    The signature statement is the vapour-mixing-ratio conversion, which
    appears nowhere else in the scheme.
    """
    sig = re.compile(r"zqp1\(n,\s*k\)\s*=\s*qvf\(n,\s*k\)\s*/\s*\(\s*1\.0\s*\+\s*qvf",
                     re.I)
    where = [p.name for p in _SOURCES
             if sig.search(p.read_text(encoding="utf-8"))]
    assert where == ["nt_run_conversion.inc"], where


@pytest.mark.parametrize("symbol", ["cu_ntiedtke_pre_run",
                                    "cu_ntiedtke_post_run"])
def test_the_globalized_symbols_are_actually_called(symbol):
    """Globalizing without calling would leave the transcription in charge.

    build.sh flips the binding bit; that is worth nothing until a harness
    uses it. Asserted against the interface's bind(C) name so a rename in
    either place shows up here.
    """
    joined = "\n".join(p.read_text(encoding="utf-8") for p in _SOURCES)
    assert f'bind(C,name="__module_cu_ntiedtke_MOD_{symbol}")' in joined
    build = (_DIR / "build.sh").read_text(encoding="utf-8")
    assert f"__module_cu_ntiedtke_MOD_{symbol}" in build, (
        f"{symbol} is declared but build.sh does not globalize it")
