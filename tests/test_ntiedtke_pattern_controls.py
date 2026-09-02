"""Every pattern-driven gate in this port gets a NEGATIVE control.

WHY THIS EXISTS, and it is a class rather than an instance (review).
Five regex-coverage failures in this port. Four were **under-matching** --
the pattern saw less than its corpus:

* ``audit_intent_aliasing``'s ``DECL`` never matched intent-less dummies
* the non-mutation claims matcher was blind to the "never read" family
* the duplicate-procedure pattern saw only ``real(kind=kind_phys) function``
* and then matched **nothing**, its ``\\b`` having become a 0x08 byte

Every vacuity guard retrofitted after those asks one question: *did the
pattern find enough?* Floors, minimum counts, named things that must
appear. **All of them pass more easily as the match set grows.**

The fifth failure was **over-matching**. ``NtWorkspace.resolve`` matched
``"copy of"`` as well as ``"alias of"``, collapsing ``zqq`` onto ``pqte``
and ``ztt`` onto ``ptte`` -- snapshots read as second names for one
storage. The post-conversion computes ``ptte - ztt``; collapsed, it
computes ``ptte - ptte`` and returns **zero convective heating on every
column**, finite and plausible. No floor, no minimum, no vacuity guard
could have seen it.

So the guard family was one-sided. This is the other half: for each
pattern, something it must NOT match, chosen to be the nearest thing that
would be wrong.
"""
from __future__ import annotations

import re

import pytest


# ---------------------------------------------------------------------------
# NtWorkspace.resolve -- the one that was actually wrong
# ---------------------------------------------------------------------------

def test_resolve_does_not_collapse_a_COPY_onto_its_source():
    """THE NEGATIVE CONTROL FOR THE BUG THAT MOTIVATED THIS FILE.

    ``zqq`` and ``ztt`` are snapshots taken before cumastrn runs and read
    afterwards. If they resolve to their sources, the post-conversion's
    ``ptte - ztt`` becomes ``ptte - ptte``.
    """
    pytest.importorskip("cupy")
    from gpuwm.core.ntiedtke import NtWorkspace

    w = NtWorkspace(ncol=2, nz=8)
    for name, source in (("zqq", "pqte"), ("ztt", "ptte")):
        assert w.resolve(name) == name, (
            f"{name} resolves to {w.resolve(name)}; it is a COPY and must "
            f"keep its own storage")
        assert w.bind(name, 1).data.ptr != w.bind(source, 1).data.ptr


def test_resolve_does_not_collapse_a_ZEROED_seed():
    pytest.importorskip("cupy")
    from gpuwm.core.ntiedtke import NtWorkspace

    w = NtWorkspace(ncol=2, nz=8)
    for name in ("ztenu", "ztenv"):
        assert w.resolve(name) == name
    assert w.bind("ztenu", 1).data.ptr != w.bind("ptenu", 1).data.ptr


def test_the_alias_pattern_matches_only_the_alias_class():
    """Checked on the prose directly, so the three classes stay three."""
    from gpuwm.core.ntiedtke import _ALIAS_RE

    assert _ALIAS_RE.match("alias of ztp1")
    for negative in ("copy of pqte before cumastrn (:274)",
                     "zeroed by the assembler (:258-259), copied at :1019",
                     "driver", "driver (pi3d)",
                     "aliased to nothing", "an alias of ztp1"):
        assert not _ALIAS_RE.match(negative), (
            f"the alias pattern matches {negative!r}, which is not an alias")


def test_every_seed_falls_in_exactly_one_class():
    """The classes must partition, or the resolver's job is undefined."""
    from gpuwm.core.ntiedtke import NT_SEEDS

    for name, why in NT_SEEDS.items():
        classes = [c for c in ("alias of", "copy of", "zeroed by", "driver")
                   if why.startswith(c)]
        assert len(classes) == 1, (
            f"{name}: {why!r} falls in {classes}, not exactly one class")


# ---------------------------------------------------------------------------
# the duplicate-procedure gate
# ---------------------------------------------------------------------------

def test_the_procedure_patterns_do_not_match_non_definitions():
    """``end function`` and a call site are not definitions.

    If they matched, every procedure would appear at least twice and the
    duplicate gate would fire constantly -- or, worse, someone would
    "fix" it by loosening the duplicate check.
    """
    from tests.test_ntiedtke_oracle_single_source import _FUNC, _SUBR

    for negative in ("  end function hexw",
                     "  end subroutine cumastrn",
                     "     x = hexw(value)",
                     "     call cumastrn(a, b, c)",
                     "! the function hexw formats a word"):
        assert not _FUNC.search(negative), f"_FUNC matches {negative!r}"
        assert not _SUBR.search(negative), f"_SUBR matches {negative!r}"


def test_the_flip_pattern_does_not_match_an_unrelated_assignment():
    """``zz = kte - pp`` is the flip. ``zz = kte`` is not."""
    from tests.test_ntiedtke_oracle_single_source import _FLIP

    assert _FLIP.search("       zz = kte - pp")
    for negative in ("       zz = kte", "       zz = nz + 1",
                     "       pp = pp + 1", "       zzz = kte - ppp"):
        assert not _FLIP.search(negative), f"_FLIP matches {negative!r}"


# ---------------------------------------------------------------------------
# the non-mutation claims matcher
# ---------------------------------------------------------------------------

def test_the_claim_patterns_do_not_match_ordinary_prose():
    """A denylist that matches everything is a denylist that gets deleted.

    These are sentences a reader would write about the same arrays without
    asserting a non-mutation, and none should be flagged.
    """
    from tests.test_ntiedtke_capture_provenance import _CLAIM_PATTERNS

    for negative in (
            "cudtdqn writes ptent and ptenq",
            "the closure rescales the downdraft arrays at :726-740",
            "measured: 1,926 of 5,292 slots differ",
            "the kernel must leave those slots alone",   # a rule, not a claim
    ):
        low = negative.lower()
        hits = [p for p in _CLAIM_PATTERNS if p in low]
        assert not hits, f"{negative!r} is flagged by {hits}"


# ---------------------------------------------------------------------------
# the stage-signature parser
# ---------------------------------------------------------------------------

def test_the_kernel_parser_does_not_pick_up_device_functions():
    """Only ``extern "C" __global__`` entry points are stages.

    ``__device__ __forceinline__`` helpers -- nt_cuentrn, nt_cubasmcn,
    nt_foeewm -- run INSIDE stages and must not become stages themselves,
    or the assembler would try to launch them.
    """
    from tests.test_ntiedtke_stage_signature import _parse

    got = set(_parse())
    for helper in ("nt_cuentrn", "nt_cubasmcn", "nt_foeewm", "nt_foealfa",
                   "nt_cuadjtqn0", "nt_geometry_ok"):
        assert helper not in got, f"{helper} was parsed as a launchable stage"
    assert all(n.startswith("ntiedtke_") for n in got), sorted(got)


# ---------------------------------------------------------------------------
# and the rule itself
# ---------------------------------------------------------------------------

def test_the_two_halves_are_both_present_in_the_tree():
    """Positive controls exist; this asserts the negative ones do too.

    Named by test function rather than by counting, because a count would
    be satisfied by any five tests in this file.
    """
    from pathlib import Path

    here = Path(__file__)
    text = here.read_text(encoding="utf-8")
    for must in ("does_not_collapse_a_COPY",
                 "do_not_match_non_definitions",
                 "does_not_match_an_unrelated_assignment",
                 "do_not_match_ordinary_prose",
                 "does_not_pick_up_device_functions"):
        assert must in text, f"the negative control {must} is gone"

    vac = (here.parent / "test_ntiedtke_oracle_single_source.py").read_text(
        encoding="utf-8")
    assert "patterns_match_the_things_they_are_meant_to_match" in vac, (
        "the POSITIVE control went away; both halves are needed and "
        "under-matching is still four of the five failures")
