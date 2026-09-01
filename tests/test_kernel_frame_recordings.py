"""The per-thread local-frame table is a RECORDING, and it says whose.

``KERNEL_MAX_LOCAL_SIZE_BYTES`` prices the launch-time local-memory
backing store, which is the largest single term in the non-pool device
budget (5.7 GiB for one kf launch on a 170-SM card).  Every row of it is
a number NVRTC produced for ONE target architecture with ONE compiler
build, and until 2026-08-20 the module carried those numbers as if they
were a property of the source.

They are not.  Measured the same NVRTC-plus-driver way on three compile
platforms:

  ======================  =========  =========  =========
  module                  sm_120     sm_120     sm_86
                          13.0.48    13.3.33    13.0.48
  ======================  =========  =========  =========
  gf                       22,416     22,416     23,984
  noah                        176        176        224
  thompson_aerosol_warm         0          0        112
  ysu                       9,232      9,232      7,184
  nssl2_fused_gs              112        216        112
  rrtmgp_cloud                  0         40          0
  shinhong                 14,040     17,160     14,040
  noahmp_leaves               272        208        208
  ======================  =========  =========  =========

Four rows move with the ARCHITECTURE at a fixed compiler, four move with
the COMPILER BUILD at a fixed architecture, and one moves with both.  So
the table is a joint property of (target architecture, NVRTC build), and
the only honest form for it is a set of named recordings plus a ceiling
over them.

These tests need no device: they check the shape of the recording and the
arithmetic of the ceiling.  ``tests/test_preflight.py::
test_the_recorded_local_frames_match_the_driver`` is the leg that puts a
real compiler behind them.
"""

from __future__ import annotations

import re
from pathlib import Path

from gpuwm.core import preflight as pf

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "gpuwm" / "core" / "kernels"


def test_every_recording_names_the_box_and_the_compiler_that_made_it():
    """A frame table with no provenance cannot be told from an assumption.

    The breakage this prevents is the one that opened the item: rows
    measured on a card that has since left the machine were carried in
    generic code with the box named only in prose, so a reader on any
    other machine had no way to see that the numbers were somebody
    else's.
    """
    recordings = pf.KERNEL_LOCAL_FRAME_RECORDINGS
    assert len(recordings) >= 2, "one recording cannot show that rows move"
    for row in recordings:
        assert row.box, "a recording must name the box it was read on"
        assert row.device
        assert re.fullmatch(r"\d+", row.compute_capability), row.compute_capability
        assert re.fullmatch(r"\d+\.\d+\.\d+", row.nvrtc_build), row.nvrtc_build
        assert row.platform_family in ("windows", "linux")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row.measured), row.measured
        assert row.frames, "a recording with no frames is not a recording"


def test_the_shipped_table_is_the_ceiling_over_every_recording():
    """Never below a measurement: under-pricing is what breaches a rail.

    The reservation is ``(frame - stack) x SMs x threads/SM``, so a row
    one byte under what this box's compiler emits under-charges by one
    byte times the whole resident-thread capacity.  Over-pricing costs
    headroom and is the direction the module has always taken; the
    ceiling makes that promise arithmetic instead of prose.
    """
    ceiling = pf.KERNEL_MAX_LOCAL_SIZE_BYTES
    for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        for module, frame in row.frames.items():
            assert module in ceiling, (
                f"{module}: measured on {row.box} and absent from the "
                "shipped table")
            assert ceiling[module] >= frame, (
                f"{module}: shipped {ceiling[module]} B is BELOW the "
                f"{frame} B {row.box} measured")
    derived = {}
    for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        for module, frame in row.frames.items():
            derived[module] = max(derived.get(module, 0), frame)
    assert dict(ceiling) == derived, (
        "the shipped table must be exactly the element-wise maximum over "
        "the recordings, so no row can drift away from every measurement")


def test_every_kernel_source_has_a_row_or_is_declared_unmeasurable():
    """A ``.cu`` with no row is a module the pricing cannot see.

    Found by running the regeneration gate on a second box:
    ``health_tile.cu`` (the tile-streamed health reduction,
    gpuwm/core/streaming.py:1728) had shipped since the out-of-core merge
    with no row in either table, so the gate that is supposed to notice a
    frame moving could not even enumerate it.
    """
    sources = {path.stem for path in KERNEL_DIR.glob("*.cu")}
    priced = (set(pf.KERNEL_MAX_LOCAL_SIZE_BYTES)
              | set(pf.UNMEASURED_KERNEL_MODULES))
    assert sources - priced == set(), sorted(sources - priced)
    assert priced - sources == set(), sorted(priced - sources)


def test_a_recorded_platform_is_recognised_from_its_own_fingerprint():
    """The gate has to know whether THIS box is one of the recorded ones."""
    row = pf.KERNEL_LOCAL_FRAME_RECORDINGS[0]
    fingerprint = {
        "device_compute_capability": row.compute_capability,
        "nvrtc_build": row.nvrtc_build,
    }
    assert pf.kernel_frame_recording_for(fingerprint) is row
    assert pf.kernel_frame_recording_for(
        {"device_compute_capability": "1", "nvrtc_build": "0.0.0"}) is None
    # An unresolved fingerprint is not a match on anything: "unavailable"
    # must never be read as "the reference box".
    assert pf.kernel_frame_recording_for(
        {"device_compute_capability": "unavailable",
         "nvrtc_build": "unavailable"}) is None
    assert pf.kernel_frame_recording_for({}) is None


def test_an_over_wide_frame_is_reported_with_the_bytes_it_under_charges():
    """The refusal names the breakage in the unit that breaks: bytes.

    A module compiling wider than its row does not fail visibly -- the
    run is admitted by a fit gate that under-counted the driver's backing
    store and OOMs later -- so the report has to convert the frame delta
    into the device bytes nobody charged for.
    """
    profile = pf.DeviceLocalMemoryProfile(
        name="probe", multiprocessor_count=68,
        max_threads_per_multiprocessor=1536)
    observed = dict(pf.KERNEL_MAX_LOCAL_SIZE_BYTES)
    observed["thompson"] = observed["thompson"] + 8
    over = pf.under_priced_kernel_frames(observed, profile=profile)
    assert set(over) == {"thompson"}
    assert over["thompson"].shipped_bytes == pf.KERNEL_MAX_LOCAL_SIZE_BYTES[
        "thompson"]
    assert over["thompson"].observed_bytes == observed["thompson"]
    assert over["thompson"].unpriced_device_bytes == 8 * 68 * 1536
    assert pf.under_priced_kernel_frames(
        dict(pf.KERNEL_MAX_LOCAL_SIZE_BYTES), profile=profile) == {}


def test_every_chained_translation_unit_says_why_it_has_no_recording():
    """A launched translation unit outside these tables must be a DECISION.

    The recordings key on ``.cu`` files that compile ALONE, because that is
    what both readers enumerate -- ``tools/vram_reserve_probe.py``
    (``mode_frames``) and ``tests/test_preflight.py::
    test_the_recorded_local_frames_match_the_driver`` glob
    ``gpuwm/core/kernels/*.cu`` and drop whatever NVRTC refuses standalone.
    Three units the model really launches are composed rather than
    standalone -- the two legacy-RRTMG chains and ``p3_composed``, which is
    what an ``mp_physics = 50`` domain loads -- so each is priced on EVERY
    platform from ONE reading, and ``under_priced_kernel_frames`` cannot
    report a drifting one, because its ``observed`` argument comes from
    that same glob.

    The breakage this prevents is a fourth composed unit arriving with no
    sentence anywhere: an unrecorded single-platform price reads exactly
    like a measured cross-platform one, and the difference is a
    reservation the fit gate never charged.  P3 is the case that opened
    it -- ``p3_composed`` shipped priced at 0 B off one sm_120 reading with
    nothing in this module mentioning that it existed.
    """
    from gpuwm.core import kernel_frame_recordings as kfr

    chained = set(pf.CHAINED_TRANSLATION_UNIT_FRAMES)
    assert chained, "no composed units at all means this gate measures air"

    # A composed unit may not live in a recording, and that is enforced
    # upstream rather than here: a chained-unit key in a frames mapping
    # makes frame_ceiling() disagree with KERNEL_MAX_LOCAL_SIZE_BYTES and
    # gpuwm.core.preflight raises at import (preflight.py:1663).
    for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        assert chained.isdisjoint(row.frames), (
            f"{row.box}: {sorted(chained & set(row.frames))} is a composed "
            "translation unit and cannot carry a standalone frame row")

    recorded = kfr.CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW
    assert set(recorded) == chained, (
        "every composed translation unit must say why it has no recording, "
        "and nothing else may claim to be one; unexplained "
        f"{sorted(chained - set(recorded))}, stale "
        f"{sorted(set(recorded) - chained)}.  Add the reason to "
        "CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW in "
        "gpuwm/core/kernel_frame_recordings.py -- do not delete this "
        "assertion")
    for unit, reason in recorded.items():
        assert len(reason.split()) >= 15, f"{unit}: {reason!r} is a label"
        assert ".cu" in reason or ".py" in reason, (
            f"{unit}: the reason must point at the source that composes the "
            f"unit or at the gate that re-audits its frame, got {reason!r}")

    # P3's own fragment, spelled out: it is UNMEASURABLE rather than
    # unmeasured -- p3.cu borrows the tree's one glibc r_pow/r_exp/r_log
    # from noahmp_leaves.cu and fails NVRTC alone -- so no recording may
    # ever grow a row for it, and a row would be a value nothing measured.
    assert "p3" in pf.UNMEASURED_KERNEL_MODULES
    assert (pf.CHAINED_TRANSLATION_UNIT_FRAMES["p3_composed"].covers
            == frozenset({"p3"}))
    for row in pf.KERNEL_LOCAL_FRAME_RECORDINGS:
        assert "p3" not in row.frames, row.box
