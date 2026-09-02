"""Every field Phase 1 is graded on traces to a named producer.

WHY THIS EXISTS, and it is the structural form of a diagnosis rather than
another list (argued by review).

Three times the remaining-work count measured a smaller thing than it
appeared to: "thirteen of thirteen kernels graded" omitted the assembler,
the assembler omitted ``cu_ntiedtke_post_run``. The general form is

    each count was taken over the artifact in front of me
    rather than over the artifact the end condition names.

Phase 1's end condition names ``nt-levels.csv``. So the count must be taken
over **that file's header**, not over a hand-maintained list of categories
-- because a hand-maintained list is exactly what has failed three times.

THIS WOULD HAVE CAUGHT IT ON DAY ONE. None of the eight tendency fields
traces to anything: ``cu_ntiedtke_run`` produces ``pt``/``pqv``/``pu``/
``pv`` and ``zprecc``, and nothing in the tree turns those into
``rthcuten``/``rucuten``/``raincv``/``pratec``. The gap was visible from the
header the whole time and nothing looked at the header.

It is the same shape as every gate here that has worked: the aliasing audit
drives the class-2 test, the ``.inc`` drives the constant table, the kernel
source drives the stage ids. The ones that failed restated a list by hand.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gpuwm.verify.ntiedtke_oracle import ORACLE_DIR

#: The two files Phase 1's end condition is stated over.
GRADED_FILES = ("nt-levels.csv", "nt-surface.csv")

#: Fields that are the fixture echoing its own DRIVER INPUTS back, so the
#: port does not produce them -- it consumes them.  Listed so that "no
#: producer" cannot be confused with "an input".
_INPUT_ECHO = {
    "case", "dx", "k",
    "t3d", "qv3d", "qc3d", "qi3d", "u3d", "v3d", "pcps", "p8w", "dz8w",
    "rho3d", "pi3d", "w", "qvften", "thften",
    "xland", "hfx", "qfx", "psfc",
}

#: field -> the component that produces it.  ``None`` means NOTHING IN THE
#: TREE PRODUCES IT YET, which is the state this gate exists to surface.
PRODUCERS: dict[str, str | None] = {
    "scale_fac": "ntiedtke_prep",
    "scale_fac2": "ntiedtke_prep",
    "cu_act_flag": "the assembler (set where ldcum survives)",
    # --- the eight, CLOSED 2026-08-29 --------------------------------
    # They traced to nothing for the whole port. This gate is what found
    # that, from the header, and the fix is ntiedtke_post_run: mirror,
    # kernel and CSVs, graded at max_ulp == 0 on all 108 columns.
    "rthcuten": "ntiedtke_post_run",
    "rqvcuten": "ntiedtke_post_run",
    "rqccuten": "ntiedtke_post_run",
    "rqicuten": "ntiedtke_post_run",
    "rucuten": "ntiedtke_post_run",
    "rvcuten": "ntiedtke_post_run",
    "raincv": "ntiedtke_post_run",
    "pratec": "ntiedtke_post_run",
}

#: What the eight WERE waiting on, kept as the record of a closed gap.
#: It named the objcopy and the harness, and that is exactly what closed
#: it -- build.sh globalizes cu_ntiedtke_post_run, run_nt_prep.F90 calls
#: the real routine and records its own entry boundary.
_CLOSED_BY = ("cu_ntiedtke_post_run (module_cu_ntiedtke.F:502-527), reached "
              "through objcopy --globalize-symbol, captured at its own call "
              "site into nt-post-in-*.csv / nt-post-out-*.csv, and ported as "
              "ntiedtke_post_run -- stage 19, 0 B frame, 40 registers")

#: What Phase 1's end condition is waiting on NOW.  One name, so the gap
#: keeps an owner rather than becoming an absence again.
_PENDING = ("the assembler: nothing allocates the column arrays, walks "
            "NT_CALL_ORDER and produces driver-level outputs, so 'the "
            "assembled pipeline reproduces nt-levels.csv bitwise' has no "
            "pipeline to run. Every array a kernel touches is still "
            "allocated by a parity test.")


def _fields(name):
    with open(ORACLE_DIR / name, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def test_every_graded_field_is_classified():
    """A field in the header with no row here is unaccounted for.

    Driven off the HEADER, not a restatement of it -- so a field added to
    the oracle lands here rather than being silently ungraded.
    """
    unclassified = []
    for name in GRADED_FILES:
        for f in _fields(name):
            if f in _INPUT_ECHO or f in PRODUCERS:
                continue
            unclassified.append(f"{name}:{f}")
    assert not unclassified, (
        f"fields in the Phase 1 end-condition files with no producer and "
        f"no input classification: {unclassified}")


def test_no_graded_field_traces_to_nothing():
    """The gap, closed, and stated so it cannot quietly reopen.

    This test used to assert the OPPOSITE -- that exactly eight fields had
    no producer -- and it was written to fail the day they gained one.
    That is what a direction-of-the-gap assertion is for, and it fired on
    schedule. It is inverted rather than deleted, because a field added to
    the oracle with nothing producing it must still land here.
    """
    missing = sorted(f for f, p in PRODUCERS.items() if p is None)
    assert not missing, (
        f"graded fields with no producer: {missing}. Every column of "
        f"nt-levels.csv must trace to a named component.")
    assert _CLOSED_BY and _PENDING, (
        "the record of what closed the gap, and of what Phase 1 waits on "
        "now, must both be present")


def test_phase_one_is_done_and_the_real_gate_owns_it_now():
    """INVERTED. The assembler exists, so this stops guarding an absence.

    Its predecessor failed the moment ``gpuwm/core/ntiedtke.py`` grew a
    walk, which is what a direction-of-the-gap assertion is for: it fired
    on schedule rather than being remembered. What replaces it is a
    pointer, because the end condition is no longer a thing this file can
    check -- it is a bitwise comparison against ``nt-levels.csv`` and it
    lives in the file that runs the pipeline.

    Kept rather than deleted so that if the end-to-end gate is ever
    removed, something still says Phase 1's claim rested on it.
    """
    from pathlib import Path

    here = Path(__file__).resolve().parent
    gate = here / "test_ntiedtke_phase_one_end_condition.py"
    assert gate.is_file(), (
        "the end-to-end gate is gone. Phase 1's claim -- the assembled "
        "pipeline reproduces nt-levels.csv bitwise -- rested entirely on "
        "it, and nothing else in this suite makes that comparison.")
    text = gate.read_text(encoding="utf-8")
    assert "test_the_pipeline_reproduces_nt_levels_bitwise" in text
    assert "test_chunking_does_not_change_the_answer" in text, (
        "the chunking gate is gone; the fixture is one chunk, so without "
        "it the multi-chunk path every real run takes is ungraded")

    launcher = (here.parent / "gpuwm" / "core"
                / "ntiedtke.py").read_text(encoding="utf-8")
    assert "def run_chunk" in launcher and "NT_CALL_ORDER" in launcher, (
        "the assembler no longer walks the declared order from the module")


def test_the_closed_gap_and_the_open_one_are_both_in_the_port_doc():
    """A gap the record does not carry is a gap only this file knows."""
    doc = (Path(__file__).resolve().parents[1]
           / "docs/ntiedtke/PORT-RECORD.md").read_text(encoding="utf-8")
    assert "cu_ntiedtke_post_run" in doc
    assert "502-527" in doc, "the doc must name post_run's real range"
    assert "assembler" in doc, "the remaining Phase 1 gap must be named"
