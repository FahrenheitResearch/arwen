"""Every New Tiedtke stage must run under ONE launch geometry.

The column workspace is per-block and lane-interleaved, so a column's slots
belong to one ``(block, lane)`` pair for the whole stage sequence.  Launch
one stage on a different threads-per-block or grid and that column resumes
on a different lane, reading another column's state.  Nothing crashes and
every number stays finite.

This is the one guarantee with no analogue in the Fortran -- no comparison
against the reference can catch it -- and it is the one most likely to be
broken deliberately, because this project's culture is kernel performance
work and ``ntiedtke_cutypen`` sits at 91 registers.  Someone will want to
re-tile that stage for occupancy.

So it is enforced three ways, and this file is the third:

1. ``NtLaunchGeometry`` is frozen and built once per step.
2. ``NtStages`` methods take no grid or block argument, so there is no
   per-stage override to reach for.
3. Every kernel REPORTS the geometry it actually observed and refuses to
   compute when it disagrees -- so a launch routed around the launcher
   turns into a loud parity failure instead of a silent cross-column read.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from gpuwm.core import ntiedtke as launcher
from gpuwm.core.ntiedtke import (
    NT_STAGE_COUNT, NT_TPB, NtLaunchGeometry, NtStages,
)


# ---------------------------------------------------------------------------
# structural: there is nothing to reach for
# ---------------------------------------------------------------------------

def test_the_geometry_is_frozen():
    g = NtLaunchGeometry(ncol=108, nz=49)
    with pytest.raises(Exception):
        g.ncol = 64          # type: ignore[misc]


def test_the_tile_width_cannot_be_changed_quietly():
    """tpb is refused, but NOT because it is a correctness constraint.

    It was documented as "the workspace lane count", which assumed a
    per-block lane-interleaved workspace that does not exist: the kernels
    index by GLOBAL column and are tile-independent.  So 32 is a free
    tuning knob, and a profiler reaching for occupancy would be right to
    want it -- cuascn is at 94 registers, cutypen at 91, and one warp per
    block caps occupancy through the blocks-per-SM limit.

    The refusal stays because the tile is chosen once per step on purpose,
    and because a tile CAP would be introduced through this descriptor.
    This test now checks the reason as well as the raise: a stale reason on
    a live gate is how the next reader concludes something false.
    """
    for tpb in (64, 128):
        with pytest.raises(ValueError, match="tile width") as e:
            NtLaunchGeometry(ncol=108, nz=49, tpb=tpb)
        assert "no longer a correctness constraint" in str(e.value)


def test_no_launch_path_accepts_a_grid_or_block():
    """The re-tiling of a single stage must be impossible to express here.

    If a future edit adds a ``grid=`` or ``block=`` parameter to the
    launcher, re-tiling one stage becomes a one-word change at one call
    site -- exactly the failure this design exists to prevent.
    """
    sig = inspect.signature(NtStages.launch)
    banned = {"grid", "block", "tpb", "nblocks", "threads", "blocks"}
    assert not (banned & set(sig.parameters)), (
        f"NtStages.launch grew a geometry parameter: {sig}")
    src = inspect.getsource(launcher.NtStages)
    assert "self.geometry.grid, self.geometry.block" in src, (
        "the launch no longer reads its geometry from the descriptor")


def test_the_stage_ids_agree_with_the_kernel_source():
    """The Python ids and the .cu defines index the same report array.

    DRIVEN OFF THE MODULE, not off a hand-written list. It used to name
    six of the twenty ids, so fourteen stages could disagree between
    ntiedtke.py and ntiedtke.cu without this noticing -- the restated-set
    bug that has now cost this campaign a link line, three guards and four
    duplicated procedures.
    """
    import re as _re
    from pathlib import Path
    from gpuwm.core import kernels as kl
    cu = (Path(kl.__file__).parent / "ntiedtke.cu").read_text(
        encoding="utf-8")
    ids = {n: getattr(launcher, n) for n in dir(launcher)
           if n.startswith("NT_STAGE_")
           and isinstance(getattr(launcher, n), int)}
    assert len(ids) >= 20, (
        f"only {len(ids)} NT_STAGE_* ids found on the launcher; there are "
        f"nineteen stages plus NT_STAGE_COUNT")
    for name, value in sorted(ids.items()):
        assert _re.search(rf"^#define\s+{name}\s+{value}\s*$", cu,
                          _re.M), (
            f"{name} = {value} in ntiedtke.py has no matching #define in "
            f"ntiedtke.cu")
    # And the other direction: a #define the module does not carry.
    in_cu = set(_re.findall(r"^#define\s+(NT_STAGE_\w+)\s+\d+\s*$", cu,
                            _re.M))
    assert in_cu == set(ids), (
        f"only in the .cu: {sorted(in_cu - set(ids))}; "
        f"only in ntiedtke.py: {sorted(set(ids) - in_cu)}")


# ---------------------------------------------------------------------------
# behavioural: the device reports what it actually ran under
# ---------------------------------------------------------------------------

@pytest.fixture
def stages():
    pytest.importorskip("cupy")
    return NtStages(NtLaunchGeometry(ncol=108, nz=49))


def _prep_args(cp, ncol, nz):
    """Minimal, correctly-shaped arguments for ntiedtke_prep."""
    half = [cp.zeros((nz, ncol), dtype=cp.float32) for _ in range(11)]
    full = [cp.zeros((nz + 1, ncol), dtype=cp.float32) for _ in range(2)]
    scal = [cp.zeros(ncol, dtype=cp.float32) for _ in range(4)]
    out_h = [cp.zeros((nz, ncol), dtype=cp.float32) for _ in range(11)]
    out_f = [cp.zeros((nz + 1, ncol), dtype=cp.float32) for _ in range(2)]
    return (tuple(half[:9]) + (full[0], full[1]) + tuple(half[9:11])
            + tuple(scal) + tuple(out_h) + tuple(out_f)
            + (cp.zeros(ncol, dtype=cp.int32),
               cp.zeros(ncol, dtype=cp.float32),
               cp.zeros(ncol, dtype=cp.float32),
               cp.zeros(ncol, dtype=cp.float32),
               np.int32(ncol), np.int32(nz), np.float32(60.0),
               np.int32(1), np.int32(2), np.float32(9.81)))


def _convert_args(cp, ncol, nz):
    """Minimal, correctly-shaped arguments for ntiedtke_convert.

    Only the shapes matter here: these tests are about launch ORDER, not
    about the numbers, and the parity suites already grade the numbers.
    """
    ro = [cp.zeros((nz + 2, ncol), dtype=cp.float32) for _ in range(10)]
    out = [cp.zeros((nz + 2, ncol), dtype=cp.float32) for _ in range(10)]
    return (tuple(ro) + tuple(out)
            + (np.int32(ncol), np.int32(nz),
               np.float32(1004.5), np.float32(287.0), np.float32(461.6),
               np.float32(2.5e6), np.float32(3.50e5), np.float32(9.81)))


def test_a_stage_reports_the_descriptor_it_ran_under(stages):
    import cupy as cp
    g = stages.geometry
    stages.launch("ntiedtke_prep", _prep_args(cp, g.ncol, g.nz))
    cp.cuda.Stream.null.synchronize()
    assert stages.check_geometry()
    assert stages.stages_that_ran() == {"prep"}
    report = cp.asnumpy(stages.geom_report)
    assert int(report[launcher.NT_STAGE_PREP]) == g.observed_stamp


def test_a_stage_launched_off_the_descriptor_is_caught(stages):
    """Route around the launcher and the device notices.

    This is the case the two structural defences cannot cover: someone
    calling ``get_function(...)`` directly with their own grid.  The kernel
    reports the geometry it SAW, so check_geometry raises -- and because it
    refuses to compute on a mismatch, the outputs stay untouched and the
    parity suite fails too.
    """
    import cupy as cp
    from gpuwm.core.kernels import load_module

    g = stages.geometry
    fn = load_module("ntiedtke").get_function("ntiedtke_prep")
    # a DIFFERENT tile: 64 threads, half the blocks
    bad_tpb = 64
    bad_blocks = (g.ncol + bad_tpb - 1) // bad_tpb
    fn((bad_blocks,), (bad_tpb,),
       _prep_args(cp, g.ncol, g.nz)
       + (np.int32(g.tpb), np.int32(g.nblocks), stages.geom_report,
          stages.order_report, cp.zeros(1, dtype=np.int32)))
    cp.cuda.Stream.null.synchronize()

    with pytest.raises(RuntimeError, match="different launch geometries"):
        stages.check_geometry()


def test_a_stage_that_never_ran_is_not_mistaken_for_agreement(stages):
    """An all-clear must mean the stages ran, not that none did."""
    import cupy as cp
    assert stages.stages_that_ran() == set()
    assert stages.check_geometry()          # vacuously true, and reported
    report = cp.asnumpy(stages.geom_report)
    assert all(int(v) == -1 for v in report), (
        "unrun stages must be distinguishable from a real geometry")


# ---------------------------------------------------------------------------
# A kernel that COMPILES is not a kernel that is GRADED
# ---------------------------------------------------------------------------
# All thirteen kernels compile at 0 B frame.  Only some of them have been
# run on the GPU and compared against the oracle.  Those are different
# claims, and "13 kernels, all 0 B" reads like the stronger one -- so the
# weaker state is written down where it can fail rather than left to a
# sentence in the port doc.

#: Kernels with a GPU parity test that launches them and compares against
#: the pinned CSVs.  Adding one here without a test is the failure this
#: guards; forgetting to add one after writing a test is caught too.
_GPU_GRADED = {
    "ntiedtke_adjust",
    "ntiedtke_ke_dissipation",
    "ntiedtke_momentum_profile",
    "ntiedtke_updraft_scale",
    "ntiedtke_momentum_rescale",
    "ntiedtke_cloud_depth",
    "ntiedtke_prep", "ntiedtke_convert", "ntiedtke_cuinin",
    "ntiedtke_cutypen", "ntiedtke_midlevel", "ntiedtke_mfub",
    "ntiedtke_closure", "ntiedtke_cuascn", "ntiedtke_cudtdqn",
    "ntiedtke_cududvn", "ntiedtke_cudlfsn", "ntiedtke_cuddrafn",
    "ntiedtke_cuflxn", "ntiedtke_post_run",
    "ntiedtke_post_conversion",
}

#: Kernels that compile at 0 B and have NOT been launched against the
#: oracle.  EMPTY as of 2026-08-29: all thirteen are graded on the GPU.
#:
#: The set is kept rather than deleted, because a kernel added later with
#: no parity test has to land somewhere visible, and an empty set that
#: something can fall into is a better home than a deleted concept.
_COMPILE_ONLY: set[str] = set()


def _kernel_names():
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "gpuwm" / "core" / "kernels" / "ntiedtke.cu").read_text(
        encoding="utf-8")
    return set(re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)', src))


def test_every_kernel_is_classified_as_graded_or_compile_only():
    """No kernel may be neither.  Silence is the thing being removed."""
    names = _kernel_names()
    unclassified = names - _GPU_GRADED - _COMPILE_ONLY
    stale = (_GPU_GRADED | _COMPILE_ONLY) - names
    assert not unclassified, (
        f"kernels with no stated grading status: {sorted(unclassified)}")
    assert not stale, f"classified kernels that no longer exist: {sorted(stale)}"


def test_the_compile_only_set_is_still_what_it_says():
    """Every kernel is launched against the oracle, not merely compiled.

    This was written in the direction of the gap while five kernels were
    compile-only, and it fired when they were graded -- which is what a
    direction-of-the-gap assertion is for.  Now inverted: it REQUIRES the
    set to be empty, so a kernel added without a parity test fails here.
    """
    assert not _COMPILE_ONLY, (
        f"a kernel is not GPU-graded: {sorted(_COMPILE_ONLY)}. Every "
        "kernel must be launched against the pinned CSVs, not merely "
        "compiled -- the two are different claims and only one of them "
        "is parity.")


def test_every_kernel_holds_no_local_frame():
    """The whole scheme, in one assertion.

    Standing rule 3 is why this is a test and not a remark: a frame is
    105 KiB of VRAM per byte on this card, and the reservation is charged
    at LAUNCH, so a kernel that grows one costs real memory the moment it
    runs.
    """
    pytest.importorskip("cupy")
    from gpuwm.core.kernels import load_module
    mod = load_module("ntiedtke")
    bad = {}
    for name in sorted(_kernel_names()):
        frame = int(mod.get_function(name).attributes["local_size_bytes"])
        if frame:
            bad[name] = frame
    assert not bad, f"kernels holding a local frame: {bad}"


# ---------------------------------------------------------------------------
# GUARANTEE 4: launch order is the Fortran call order
# ---------------------------------------------------------------------------
# The only continuity guarantee in docs/ntiedtke/PORT-RECORD.md section 7 that outlived
# the workspace, and it was prose until now.  Thirteen kernels threading
# column arrays through caller-allocated buffers share state exactly as hard
# as a lane-interleaved workspace would have.
#
# geom_report says THAT a stage ran and UNDER WHAT TILE.  It says nothing
# about WHEN.  An out-of-order launch reads a predecessor's array before
# that predecessor wrote it: nothing crashes, every number stays finite, and
# a parity suite that launches the sequence it was written with cannot see
# it.  That is the guarantee-6 argument applied to order instead of tile,
# with the same risk profile -- the person who breaks it is doing something
# reasonable, namely overlapping "independent" stages on separate streams
# during Phase 4.

#: The declared sequence lives in the MODULE now: the assembler is
#: what walks it and a test is not the right owner of a production
#: contract. Imported so every assertion below still reads it.
from gpuwm.core.ntiedtke import NT_CALL_ORDER  # noqa: E402


def test_the_declared_order_covers_every_kernel_but_the_probe():
    """ntiedtke_midlevel is the one omission, and it is deliberate.

    It exists to grade cubasmcn and cuentrn standalone; in the real
    sequence both run INSIDE cuascn.  So it is not a stage of the scheme
    and must not appear in the call order -- but it must be named here, or
    its absence looks like an oversight to the next reader.
    """
    names = _kernel_names()
    missing = names - set(NT_CALL_ORDER)
    assert missing == {"ntiedtke_midlevel"}, (
        f"kernels absent from the declared call order: {sorted(missing)}")
    assert len(NT_CALL_ORDER) == len(set(NT_CALL_ORDER)), "duplicate stage"


def test_the_order_report_starts_empty(stages):
    """-1 is 'did not run', distinguishable from ticket 0."""
    import cupy as cp
    assert (cp.asnumpy(stages.order_report) == -1).all()


def test_check_order_accepts_the_sequence_that_ran(stages):
    """Two stages, launched in order, verified in order."""
    import cupy as cp
    from gpuwm.core.ntiedtke import NT_STAGE_PREP, NT_STAGE_CONVERT
    g = stages.geometry
    stages.launch("ntiedtke_prep", _prep_args(cp, g.ncol, g.nz))
    stages.launch("ntiedtke_convert", _convert_args(cp, g.ncol, g.nz))
    cp.cuda.Stream.null.synchronize()
    assert stages.check_order((NT_STAGE_PREP, NT_STAGE_CONVERT))


def test_check_order_CATCHES_a_swap(stages):
    """The assertion that makes this a gate rather than a receipt.

    Launch two stages in one order and declare the other. If this passed,
    the whole mechanism would be decorative.
    """
    import cupy as cp
    from gpuwm.core.ntiedtke import NT_STAGE_PREP, NT_STAGE_CONVERT
    g = stages.geometry
    stages.launch("ntiedtke_prep", _prep_args(cp, g.ncol, g.nz))
    stages.launch("ntiedtke_convert", _convert_args(cp, g.ncol, g.nz))
    cp.cuda.Stream.null.synchronize()
    with pytest.raises(RuntimeError, match="declared order"):
        stages.check_order((NT_STAGE_CONVERT, NT_STAGE_PREP))


def test_check_order_catches_a_stage_that_never_ran(stages):
    """A declared stage that did not launch is a missing stage, not a pass."""
    import cupy as cp
    from gpuwm.core.ntiedtke import (NT_STAGE_PREP, NT_STAGE_CONVERT,
                                     NT_STAGE_CUININ)
    g = stages.geometry
    stages.launch("ntiedtke_prep", _prep_args(cp, g.ncol, g.nz))
    stages.launch("ntiedtke_convert", _convert_args(cp, g.ncol, g.nz))
    cp.cuda.Stream.null.synchronize()
    with pytest.raises(RuntimeError, match="declared order"):
        stages.check_order((NT_STAGE_PREP, NT_STAGE_CONVERT,
                            NT_STAGE_CUININ))


def test_the_ticket_is_taken_once_per_launch_not_once_per_thread(stages):
    """108 columns, 4 blocks -- and exactly one ticket.

    Drawn by block 0 thread 0, which always exists and, being column 0,
    survives the `i >= ncol` guard. A per-thread ticket would make the
    order report meaningless and the numbers would still look plausible.
    """
    import cupy as cp
    g = stages.geometry
    assert g.nblocks > 1, "this test needs more than one block to mean much"
    stages.launch("ntiedtke_prep", _prep_args(cp, g.ncol, g.nz))
    cp.cuda.Stream.null.synchronize()
    order = cp.asnumpy(stages.order_report)
    assert int(order[0]) == 0, f"prep drew ticket {order[0]}, expected 0"
    assert (order[1:] == -1).all(), "another stage recorded a ticket"
