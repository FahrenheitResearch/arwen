"""``hoststore.host_memory`` answers on every platform the product ships on.

THE DEFECT THIS EXISTS FOR
--------------------------
``tilestream/hoststore.py`` read ``/proc/meminfo`` unconditionally, so on
Windows -- the platform the product's own front door ships on, and the one
the cut box runs -- EVERY ``HostDomainStore`` construction died in
``check_allocatable`` with ``FileNotFoundError: /proc/meminfo`` before a
single byte was pinned.  Seven of ``tilestream/test_hoststore.py``'s eight
gates failed on that one line, and the out-of-core host store -- the module
that makes a larger-than-VRAM domain possible at all -- was unusable on the
machine it was built for.  ``tilestream/autoplan.py`` had already grown the
``GlobalMemoryStatusEx`` branch for exactly this reason (its module
docstring: "a planner that had no host source on Windows did not fall back
to a guess, it RAISED"); the store never inherited it.

This file is SEPARATE from ``test_hoststore.py`` on purpose: that module's
helpers import cupy, so the whole module is auto-marked ``gpu`` and the
stage-1 leg (GPUWM_NO_LOCAL_GPU=1, the Windows cut box) skips it.  The
host-memory read needs no card, and the box that needs this gate most is
exactly the one the gpu shard never runs on -- so the gate lives where the
CPU leg can see it.
"""

from __future__ import annotations

import os

import pytest

from tilestream import hoststore as HS


def test_host_memory_answers_on_this_platform() -> None:
    """The whole defect: this call must not need procfs to exist."""
    mem = HS.host_memory()
    assert set(mem) == {"total", "available", "free"}, sorted(mem)
    for key, value in mem.items():
        assert isinstance(value, int), (key, type(value))


def test_host_memory_values_are_sane() -> None:
    """A source that answered zeros would wave every allocation through
    ``check_allocatable``'s fraction cap (0.47 * 0 = 0 refuses everything)
    or refuse everything (available 0), so the values are held to physical
    sense, not just to existing."""
    mem = HS.host_memory()
    assert mem["total"] >= (1 << 30), (
        f"MemTotal {mem['total']} B is under 1 GiB; no supported box is "
        "that small, so the source is misreading")
    assert 0 < mem["available"] <= mem["total"], mem
    assert 0 <= mem["free"] <= mem["total"], mem


@pytest.mark.skipif(os.name != "nt", reason="the Windows branch is only "
                    "measurable on Windows")
def test_windows_branch_does_not_open_procfs() -> None:
    """On Windows the answer comes from ``GlobalMemoryStatusEx``, full stop.

    Pinned by construction rather than by mocking ``open``: the Windows
    reader is called directly and must agree with ``host_memory`` -- if a
    procfs read ever came back in front of it, the two would still agree,
    but ``host_memory`` would already have raised on the box running this
    test, which has no ``/proc``.
    """
    direct = HS._host_memory_windows()
    assert direct["total"] >= (1 << 30)
    assert 0 < direct["available"] <= direct["total"]
    via_public = HS.host_memory()
    # Same source, sampled twice: totals are constant, available drifts.
    assert via_public["total"] == direct["total"]


def test_linux_parser_still_reads_meminfo_text() -> None:
    """The /proc/meminfo PARSER, held to a fixture so the Linux arm cannot
    regress while the Windows arm is being fixed.  Pure text, no procfs."""
    sample = ("MemTotal:       98467840 kB\n"
              "MemFree:        61906944 kB\n"
              "MemAvailable:   96468992 kB\n"
              "Buffers:              0 kB\n")
    parsed = HS._parse_meminfo(sample.splitlines())
    assert parsed["MemTotal"] == 98467840 * 1024
    assert parsed["MemAvailable"] == 96468992 * 1024
    assert parsed["MemFree"] == 61906944 * 1024


def test_check_allocatable_runs_on_this_platform() -> None:
    """The consumer that actually died on Windows, exercised end to end:
    a tiny request must pass all three gates, an absurd one must be
    refused by the fraction cap -- both without touching procfs and
    without allocating anything."""
    HS.check_allocatable(1 << 20)                     # 1 MiB: always fine
    with pytest.raises(HS.HostMemoryExhausted):
        HS.check_allocatable(1 << 50)                 # 1 PiB: never fine
