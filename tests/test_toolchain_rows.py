"""A167: which per-toolchain row a GPU parity table reads, on every Blackwell part.

The breakage this prevents: tests/_toolchain_rows.py keyed its no-fallback
rule on a list of two compute capabilities, 10.0 and 12.0.  A B300 (10.3), a
Thor (11.0) or a GB10 (12.1) compiles constant divisions as reciprocal
multiplies just as they do (measured: NVRTC 12.9.86 and 13.4.92, every
compute_100 to compute_121 target they accept), yet it fell through to the
sm_86/89 default row, so a MYNN table could pass or fail on such a card
against another architecture's reading.  The rule is now keyed on every
compute-capability major of 10 or above.  CPU only: the toolchain is passed
in, no device is opened.
"""
from __future__ import annotations

import pytest

from _toolchain_rows import rewrites_constant_division, toolchain_row

#: Every target NVRTC 12.9.86 or 13.4.92 accepts, by whether it rewrote
#: ``x / 3.0f`` and ``x / 60.0f`` (A167; the measurement is in the module
#: docstring of tests/_toolchain_rows.py).
REWRITTEN = ("100", "101", "103", "110", "120", "121")
KEPT_DIV_RN = ("75", "80", "86", "87", "89", "90")


def test_every_blackwell_capability_is_keyed_and_no_earlier_one_is():
    assert [cc for cc in REWRITTEN if not rewrites_constant_division(cc)] == []
    assert [cc for cc in KEPT_DIV_RN if rewrites_constant_division(cc)] == []


def test_an_unlisted_blackwell_part_skips_naming_itself():
    """Before A167 a 10.3 card read the default row here."""
    with pytest.raises(pytest.skip.Exception, match=r"no row for sm_103"):
        toolchain_row({("120", (13, 4)): "sm_120 row"}, "default row",
                      "TABLE", current=("103", (13, 4)))


def test_a_blackwell_part_with_rows_under_another_compiler_fails_naming_it():
    with pytest.raises(pytest.fail.Exception,
                       match=r"no row for NVRTC 12\.9 on sm_121"):
        toolchain_row({("121", (13, 4)): "sm_121 row"}, "default row",
                      "TABLE", current=("121", (12, 9)))


def test_an_earlier_architecture_reads_the_default_row():
    rows = {("120", (13, 4)): "sm_120 row"}
    assert toolchain_row(rows, "default row", "TABLE",
                         current=("89", (12, 9))) == "default row"
    assert toolchain_row(rows, "default row", "TABLE",
                         current=("90", (13, 4))) == "default row"


def test_a_recorded_toolchain_reads_its_own_row():
    rows = {("120", (13, 4)): "13.4 row", ("120", (12, 9)): "12.9 row"}
    assert toolchain_row(rows, "default row", "TABLE",
                         current=("120", (12, 9))) == "12.9 row"
    assert toolchain_row(rows, "default row", "TABLE",
                         current=("120", (13, 4))) == "13.4 row"
