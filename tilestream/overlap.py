"""The event chain that lets a sweep's transfers run beside its compute.

:mod:`tilestream.driver` used to issue everything a buffer does -- gather,
ring save, ring patch, step, scatter -- on that buffer's ONE stream.  Inside
a buffer that serialises the copies against the compute by construction, and
the attribution run wrote down what it costs
(``tilestream/OVERLAP-ATTRIBUTION.md``): on the box the build was proven on,
2.3-3.3 s of transfer sat exposed per quiet step, 45% of it at the sweep
seam where the last tiles' scatters drain against an idle GPU and the next
step's first gathers fill against one.

The driver's overlap mode splits each buffer onto three streams -- a copy-in
stream (H2D: geography, boundary tables, carrier gather, ring patch), the
compute stream, and a copy-out stream (D2H: ring save, scatter) -- so the
copy engines and the SMs each have their own queue.  What made the single
stream CORRECT was its program order; take that away and every ordering the
ring scheme needs has to be stated as an event.  This module owns that
statement, in two halves:

* the STRUCTURAL edges, fixed per tile and independent of the plan.  The
  driver enforces these with one event per (tile, operation) class:

  - gather -> save (the save reads window cells the gather writes)
  - gather -> patch (same stream, program order: both write window cells)
  - save -> step (the step overwrites the band the save is reading out)
  - patch -> step (the step reads the cells the patch restores)
  - step -> scatter (the scatter copies the interior the step wrote), with
    the observer / health / post-step hooks recorded BEHIND the step event
    so a hook that reads the buffer is ordered ahead of the scatter too
  - scatter -> the buffer's next gather (the gather overwrites the window
    the scatter is still reading)

* the PLAN-SHAPED edges, which depend on which rectangles actually
  intersect and are the reason this is a module and not four lines in the
  driver.  :class:`OverlapSchedule` computes them from a
  :class:`tilestream.rings.RingPlan`; the driver waits on exactly these
  lists and nothing else.

THE FOUR PLAN-SHAPED HAZARDS, one list each
-------------------------------------------
``patch_waits[j]`` -- RAW on the arena, within a sweep.  patch(j) reads
bands that save(k), k < j, wrote.  This is :attr:`RingPlan.patch_deps`
verbatim; it moves here so the driver has one place to take every wait
from.  Under the single-stream loop the same-buffer case was ordered for
free and skipped; save and patch now sit on DIFFERENT streams even when the
tiles share a buffer, so no case is skipped.

``scatter_waits[k]`` -- WAR on the store, within a sweep.  gather(j),
j < k, reads store cells that scatter(k) overwrites, and there is no patch
in that direction.  :attr:`RingPlan.war_deps` verbatim, waited on the
gather-complete event.

``gather_seam_waits[j]`` -- RAW on the store, ACROSS the sweep seam.
Within one sweep a gather may race an earlier tile's scatter because the
patch overwrites every byte they can disagree on; across the seam that
argument inverts -- sweep s+1's gather must see sweep s's values, the ring
holds time-s-minus-one, and nothing patches it.  The old loop bought this
ordering with ``deviceSynchronize`` at every sweep's end, which is exactly
the exposed fill/drain the attribution measured.  The event form: tile j's
gather waits on the PREVIOUS recording of the scatter event of every tile
whose writes reach j's window -- ``{j}`` (its own interior) union the
within-sweep lists read in both directions.

``save_seam_waits[j]`` -- WAR on the arena, ACROSS the sweep seam.
save(j) at sweep s+1 refills bands that patch(i) at sweep s is still
reading, for every i whose patch list names tile j.  Waited on the
patch-complete (copy-in ready) event's previous recording.

WHY THE SEAM WAITS MAY LAND ON THE WRONG SWEEP'S RECORDING, AND WHY THAT IS
STILL CORRECT: a CUDA event wait takes the event's most recent RECORDED
snapshot at issue time.  The driver issues sweeps in host program order, so
when tile j's gather at sweep s+1 waits on scatter(k): if k has not yet
been re-issued this sweep the snapshot is sweep s's scatter -- the exact
dependency; if k < j has already been re-issued, the snapshot is sweep
s+1's own scatter of k, which FOLLOWS sweep s's on the same stream --
strictly later, so the wait is conservative, never early.  An event that
has never been recorded satisfies the wait immediately, which is the first
sweep's correct answer: there is nothing to wait for, and the driver's
construction-time ``deviceSynchronize`` already fenced the store fill.

:func:`assert_schedule_covers_hazards` re-derives every intersection from
the plan's own gather/scatter rectangles -- the same independence
discipline as :func:`tilestream.rings.assert_ring_covers_reads` -- and
refuses a schedule with a missing edge.  The driver runs it at construction
for ``overlap="on"``, so a hand-edited or regressed dependency list is a
refusal at setup, not a plausible forecast.  ``overlap="unchained"`` skips
the checker and the waits both: it is the negative control that shows the
chain is load-bearing, exactly as ``ring_ordering="submission"`` and
``write_mode="inplace"`` are kept to show for their own orderings.

What is deliberately NOT here: ``_advance_clock`` and
``set_carrier_scalars``.  The attribution run measured both below
0.02 ms/step -- they read and write host-side counters that the step's own
host code maintains synchronously at issue time -- so they need no event
and removing "their" synchronization would be optimizing the noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilestream import rings as _rings


__all__ = [
    "OverlapError",
    "OverlapSchedule",
    "assert_schedule_covers_hazards",
]


class OverlapError(RuntimeError):
    """An overlap schedule is missing an ordering the ring scheme needs."""


@dataclass(frozen=True)
class OverlapSchedule:
    """Every plan-shaped wait of the dedicated-stream sweep, by tile.

    Tuples of tile indices, one entry per tile of the plan.  The driver
    walks these lists verbatim; nothing else in the loop decides an
    ordering.  See the module docstring for what each list orders and which
    event it is waited against.
    """

    ntiles: int
    #: patch(j) waits save(k): RAW on the arena, within a sweep.
    patch_waits: tuple[tuple[int, ...], ...]
    #: scatter(k) waits gather(j): WAR on the store, within a sweep.
    scatter_waits: tuple[tuple[int, ...], ...]
    #: gather(j) at sweep s+1 waits scatter(k) at sweep s: RAW on the store,
    #: across the seam.  Always contains j itself.
    gather_seam_waits: tuple[tuple[int, ...], ...]
    #: save(j) at sweep s+1 waits patch-ready(i) at sweep s: WAR on the
    #: arena, across the seam.
    save_seam_waits: tuple[tuple[int, ...], ...]

    @classmethod
    def from_plan(cls, plan: "_rings.RingPlan") -> "OverlapSchedule":
        """Derive the four lists from a ring plan's own dependency sets.

        ``patch_deps`` and ``war_deps`` are taken verbatim.  The two seam
        lists are compositions of the same sets read in both directions:
        tile k's writes reach tile j's reads exactly when ``k < j`` and
        ``k in patch_deps[j]`` (the band planner saved that overlap), or
        ``k > j`` and ``j in war_deps[k]`` (the planner ordered that read
        ahead of the write), or ``k == j`` (a window contains its own
        interior).  ``save_seam_waits`` is ``patch_deps`` transposed: the
        tiles whose patches read tile j's bands.
        """
        n = plan.ntiles
        reads_writes_of: list[set[int]] = [set() for _ in range(n)]
        patch_readers: list[set[int]] = [set() for _ in range(n)]
        for j in range(n):
            reads_writes_of[j].add(j)
            for k in plan.patch_deps[j]:
                reads_writes_of[j].add(k)
                patch_readers[k].add(j)
        for k in range(n):
            for j in plan.war_deps[k]:
                reads_writes_of[j].add(k)
        return cls(
            ntiles=n,
            patch_waits=tuple(tuple(d) for d in plan.patch_deps),
            scatter_waits=tuple(tuple(d) for d in plan.war_deps),
            gather_seam_waits=tuple(tuple(sorted(s))
                                    for s in reads_writes_of),
            save_seam_waits=tuple(tuple(sorted(s)) for s in patch_readers),
        )


def _write_read_pairs(plan: "_rings.RingPlan"):
    """``(writer, reader)`` tile pairs whose rectangles truly intersect.

    Re-derived from the specs' own ``Transfer`` objects through
    :func:`tilestream.rings._transfer_rects` -- NOT from ``war_deps`` or
    ``patch_deps`` -- so a plan whose dependency sets were built wrong, or a
    schedule assembled from the wrong plan, fails against the geometry
    rather than against a copy of itself.
    """
    n = plan.ntiles
    reads: dict[tuple[int, str], list] = {}
    writes: dict[tuple[int, str], list] = {}
    for i, spec in enumerate(plan.specs):
        for kind in plan.kinds:
            reads[(i, kind)] = _rings._transfer_rects(spec, "gather", kind)
            writes[(i, kind)] = _rings._transfer_rects(spec, "scatter", kind)
    pairs: set[tuple[int, int]] = set()
    for k in range(n):
        for j in range(n):
            if (k, j) in pairs:
                continue
            hit = False
            for kind in plan.kinds:
                for w_full, _wt in writes[(k, kind)]:
                    for r_full, _rt in reads[(j, kind)]:
                        if _rings._intersect(w_full, r_full) is not None:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                pairs.add((k, j))
    return pairs


def assert_schedule_covers_hazards(schedule: OverlapSchedule,
                                   plan: "_rings.RingPlan") -> None:
    """Raise unless every geometric hazard has an edge in the schedule.

    The independent re-check, deliberately built the way
    :func:`tilestream.rings.assert_ring_covers_reads` is built: it walks the
    plan's own gather/scatter rectangles and tests each true intersection
    against the schedule's lists, so a dependency dropped anywhere between
    the geometry and the waits is a refusal at setup.  Removing any one wait
    class from a schedule makes this fire, which is what the hermetic
    revert-check in :mod:`tilestream.test_overlap` asserts.
    """
    n = plan.ntiles
    if schedule.ntiles != n:
        raise OverlapError(
            f"schedule describes {schedule.ntiles} tiles but the plan has "
            f"{n}; the two were built from different plans")
    pairs = _write_read_pairs(plan)
    for k, j in sorted(pairs):
        # gather_seam_waits is indexed by the READER.
        if k not in schedule.gather_seam_waits[j]:
            raise OverlapError(
                f"tile {k}'s writes reach tile {j}'s window but the seam "
                f"list does not order gather({j}) after scatter({k}); the "
                "next sweep's gather could read the previous generation")
        if k == j:
            continue
        if k < j:
            if k not in schedule.patch_waits[j]:
                raise OverlapError(
                    f"tile {j}'s patch reads tile {k}'s saved band but the "
                    f"schedule does not order patch({j}) after save({k})")
        else:
            if j not in schedule.scatter_waits[k]:
                raise OverlapError(
                    f"tile {j} reads store cells tile {k} overwrites but "
                    f"the schedule does not order scatter({k}) after "
                    f"gather({j})")
    for j in range(n):
        if j not in schedule.gather_seam_waits[j]:
            raise OverlapError(
                f"tile {j}'s own interior is missing from its seam list; "
                "the next sweep's gather could read its own stale interior")
        for k in schedule.patch_waits[j]:
            if j not in schedule.save_seam_waits[k]:
                raise OverlapError(
                    f"patch({j}) reads tile {k}'s bands but the schedule "
                    f"does not order the next sweep's save({k}) after it; "
                    "the arena could be refilled under a live read")
