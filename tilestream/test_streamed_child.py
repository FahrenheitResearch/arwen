"""RESIDENT PARENT + TILE-STREAMED CHILD through the executor, both ways.

Run it::

    python -m tilestream.test_streamed_child             # the whole matrix
    python -m tilestream.test_streamed_child --steps 6   # shorter window

THE ROLES-FLIPPED TWIN OF ``tilestream/test_nest_executor.py``
--------------------------------------------------------------
That gate streams the PARENT and keeps the child resident; this one keeps
the parent resident and streams the CHILD -- the other concurrent-nesting
shape of the 2.3 train, the one whose user story is a large fine child
that does not fit beside its resident parent.  Same product loop
(``gpuwm.core.model.execute_experiment``, its FORCE/FEEDBACK edge
dispatch and clock imposition), same static two-domain tree, and the
child's streaming goes through the PRODUCTION builder
(``streaming.prepared_domain_builder`` -> the nest tile hook), not a
harness re-implementation.

The legs:

``A  resident   fb0``   the control.
``B  resident   fb1``   MUST differ from A in the parent (the treatment
                        proof: arms that agree prove the experiment never
                        ran).
``C  child-streamed fb0``  d01 (state) and d02 (store) bit-identical to A.
``D  child-streamed fb1``  bit-identical to B, both domains.  Two-way
                        feedback restricting OUT OF the child's store into
                        the resident parent.
``E  resident solo``    parent alone, no child, same executor and cadence.
                        E's d01 vs C's d01 is the #43 mirror: a one-way
                        parent is bitwise unchanged by its streamed child.
``N1 stale child store``  C with the store UNPUBLISHED from the child
                        state: the coupler's frame pull and the feedback
                        read then see the frozen attach-time child.  d02
                        MUST differ from A (the instrument sees the
                        defect) and d01 MUST NOT move (one-way staleness
                        cannot reach a resident parent).
``N1b stale store, fb1``  the two-way twin: d01 MUST differ from B (the
                        feedback restricted attach-time air).
``N2 stale tile tables``  the launch-time generation reload disarmed
                        after the first FORCE: buffers keep serving
                        FORCE-1 tables while the generation claims
                        currency.  d02 MUST differ from A, d01 MUST equal
                        C's -- the reload is load-bearing, in both
                        directions, before any PASS above is believed.
``W  starved frame``    the frame pull narrowed below the boundary zone
                        (``child_frame_windows`` at width 2 < bdy width
                        5): stale child cells inside the zone
                        ``bdy_interp1`` reads.  d02 MUST differ, d01 MUST
                        equal C's -- the window is really narrowing the
                        read.

Same discipline as the sibling gates: exit code 3 means the CARD WAS
TAKEN, cadence census printed for BOTH sides of every comparison, digests
over the STORE for a streamed domain and over the state for a resident
one.  The fb1 initial restriction runs BEFORE the child is attached to
the streaming transport -- the same order the product takes
(``build_experiment`` runs it at tree assembly; ``steppers_for_tree``
attaches after).

At this gate geometry (72^2 child) the boundary frame is most of the
child, so the corridor's traffic RATIO is modest; the ratio grows with
the child (~16x at the flagship shape, formula in the design doc).  This
gate is about bytes and bits, not budgets -- the same posture as the
inverse gate's 550sq-class shape.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from tilestream.test_moving_nest import (
    NBUFFERS, PARENT_DT, RATIO, VRAM_NEEDED_GIB, build_parent,
    cadence_census, carrier_digest, child_cfg, domain_boundaries,
    parent_cfg, wait_for_vram)
from tilestream.test_nest_executor import (
    DEFAULT_STEPS, assemble, experiment, initial_feedback)

#: The child's pinned tiling: 72^2 at tile 24 with halo 16 is 3x3 tiles
#: whose windows (56^2) sit inside the domain, and 9 tiles over 2 buffers
#: means every buffer serves several tiles per sweep -- the tile hook's
#: re-target path and the launch-time reload are BOTH exercised, not just
#: the hook-once case the reload exists for.
CHILD_TILE = 24


def stream_child(model, *, store="host", tile=CHILD_TILE):
    """Attach a :class:`StreamedDomain` to d02 through the PRODUCTION
    builder, and return ``{2: stepper}``."""
    from gpuwm.core import streaming

    node = model.node(2)
    cfg = node.cfg.run
    options = streaming.StreamingOptions(
        mode="on", tile_nx=int(tile), tile_ny=int(tile),
        nbuffers=NBUFFERS, store=store)
    decision = streaming.decide(cfg, options)
    stepper = streaming.make_stepper(
        node.state, cfg, options, decision=decision,
        build=streaming.prepared_domain_builder(node, check_geography=False))
    if not streaming.is_streaming(stepper):
        raise RuntimeError("the child did not stream")
    return {2: stepper}


def run_leg(mode, *, feedback=0, nested=True, steps=DEFAULT_STEPS,
            unpublish=False, stale_tables=False, frame_width=None,
            tile=None, store="host", validate=True, dump=None,
            verbose=True) -> dict:
    """One executor-driven run.  ``mode`` is ``resident`` or ``streamed``
    (streamed = the CHILD streams; the parent is always resident here).

    ``unpublish=True`` strips the store from the CHILD state, so the
    coupler's frame pull and the feedback read answer "resident" and read
    the frozen attach-time arrays -- the stale-child negative control.
    The integration itself still runs off the store and is still correct;
    what breaks is exactly and only what the corridor's store consult
    repairs.

    ``stale_tables=True`` disarms ``nest_stream._copy_owned_sides`` after
    the first FORCE has landed, so every buffer keeps serving FORCE-1
    tables -- the stale-tile-tables negative control, aimed at the
    launch-time generation reload specifically.

    ``frame_width`` overrides the frame pull's strip width (the mirror of
    the inverse gate's ``force_halo``): ``2`` is below the boundary zone
    the tables are built from, so it MUST move the child and MUST NOT
    move the parent.

    ``validate=False`` is for the two controls that DELIBERATELY
    mis-force the child (stale tables, starved frame): feeding a child
    inconsistent boundary tendencies can push a boundary-zone moisture
    a few 1e-6 negative within a dozen steps, and the full-state health
    gate then kills the leg -- which is the gate doing its job on a leg
    whose whole point is to be wrong.  The control's measurement is its
    DIGESTS; run 1 measured exactly this (qv(46,70,66) = -2.54e-06 on
    the stale-tables leg, evidence/streamed-child/).

    ``store`` reaches ``stream_child``: ``"device"`` makes the store the
    state's own arrays, isolating the TILING arithmetic from the host
    TRANSPORT -- the same discriminator test_moving_nest carries.

    ``dump`` names a directory that receives this leg's FULL result --
    per-field digests included -- as JSON before any comparison runs, so
    a later leg's death cannot take the earlier legs' evidence with it.
    """
    import cupy as cp

    from gpuwm.core import nest_stream
    from gpuwm.core.model import execute_experiment

    tile = CHILD_TILE if tile is None else int(tile)
    wait_for_vram(VRAM_NEEDED_GIB)
    pcfg = parent_cfg()
    ccfg = child_cfg(pcfg) if nested else None
    bnd = domain_boundaries(pcfg, seconds=max(21600.0, steps * PARENT_DT))
    parent_state, geo = build_parent(pcfg, boundaries=bnd)
    exp = experiment(pcfg, ccfg, steps=steps)
    model = assemble(exp, parent_state, geo.grid, feedback=feedback)
    if nested and feedback:
        initial_feedback(model)

    steppers = {}
    streamed = None
    if mode == "streamed":
        steppers = stream_child(model, store=store, tile=tile)
        streamed = steppers[2]
        if unpublish:
            from gpuwm.core.streaming import _STORE_ATTR, domain_store

            delattr(model.node(2).state, _STORE_ATTR)
            assert domain_store(model.node(2).state) is None, (
                "the negative control failed to unpublish the store; "
                "whatever it measures next is not the stale-child path")
    elif mode != "resident":
        raise ValueError(f"unknown mode {mode!r}")

    frames_before = nest_stream.child_frame_windows
    copy_before = nest_stream._copy_owned_sides
    if frame_width is not None:
        from gpuwm.core.streaming import frame_windows

        def starved(run_cfg, _w=int(frame_width)):
            return frame_windows(int(run_cfg.ny), int(run_cfg.nx), _w)

        nest_stream.child_frame_windows = starved
    if stale_tables:
        def gated_copy(specs, source):
            # The FIRST interval's tables land (or every leg would be
            # forced by zeros, which tests allocation, not staleness);
            # everything after serves FORCE-1 forever.
            if source.rolling_generation <= 1:
                return copy_before(specs, source)
            return 0

        nest_stream._copy_owned_sides = gated_copy
    try:
        t0 = time.perf_counter()
        execution = execute_experiment(
            model, steppers=steppers, validate_state=bool(validate),
            pool_trim_per_period=False)
        cp.cuda.runtime.deviceSynchronize()
        wall = time.perf_counter() - t0
    finally:
        nest_stream.child_frame_windows = frames_before
        nest_stream._copy_owned_sides = copy_before

    d01_sha, d01_per = carrier_digest(model.root.state)
    out = {
        "mode": mode, "feedback": int(feedback), "nested": bool(nested),
        "unpublished": bool(unpublish), "stale_tables": bool(stale_tables),
        "frame_width": frame_width, "store": store,
        "validated": bool(validate),
        "steps": int(execution.steps), "forces": int(execution.forces),
        "d01_sha256": d01_sha, "d01_fields": d01_per,
        "cadence": cadence_census(model, steppers),
        "wall_s": round(wall, 2),
    }
    if nested:
        d02_source = streamed.store if streamed is not None \
            else model.node(2).state
        d02_sha, d02_per = carrier_digest(d02_source)
        coupler = model.node(2).coupler
        out.update(
            d02_sha256=d02_sha, d02_fields=d02_per,
            force_count=int(coupler.force_count),
            feedback_count=int(coupler.feedback_count),
            force_sync_bytes=int(coupler.force_sync_bytes))
    if dump is not None:
        import os

        os.makedirs(dump, exist_ok=True)
        tag = "-".join(str(v) for v in (
            mode, store, f"fb{feedback}", "nested" if nested else "solo",
            "unpub" if unpublish else "", "stale" if stale_tables else "",
            f"fw{frame_width}" if frame_width is not None else "")
            if v)
        with open(os.path.join(dump, tag + ".json"), "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    del model, steppers, streamed, parent_state
    import gc

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if verbose:
        print(json.dumps({k: v for k, v in out.items()
                          if not k.endswith("_fields")}, indent=2))
    return out


def _diff_fields(a, b, key) -> list:
    return sorted(n for n in a[key] if a[key][n] != b[key].get(n))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--tile", type=int, default=None)
    args = ap.parse_args(argv)

    import cupy as cp

    def leg(name, **kw):
        print(f"--- {name}")
        try:
            return run_leg(steps=args.steps, tile=args.tile,
                           dump="evidence-legs", **kw)
        except (cp.cuda.memory.OutOfMemoryError,
                cp.cuda.runtime.CUDARuntimeError) as err:
            print(f"CARD TAKEN: {err}")
            sys.exit(3)

    failures = []

    def check(label, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {label}" +
              (f"  ({detail})" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    a = leg("A resident fb0", mode="resident", feedback=0)
    b = leg("B resident fb1", mode="resident", feedback=1)
    c = leg("C child-streamed fb0", mode="streamed", feedback=0)
    # Localization aid, printed the moment both sides exist: run 1 died
    # in a later leg with C-vs-A d02 already diverged and the per-field
    # map filtered out of the printed summary, so WHICH carriers moved
    # was unknowable from the log.  Named immediately, and kept in the
    # evidence-legs dump either way.
    print(json.dumps({"C_vs_A_d02_differing":
                      _diff_fields(c, a, "d02_fields")[:24]}))
    # validate=False: C2 is a pure discriminator (transport vs tiling) and
    # its data must land even if the diverged trajectory grazes a health
    # bound; A's own validated run already guards the reference.
    c2 = leg("C2 child-streamed fb0, DEVICE store", mode="streamed",
             feedback=0, store="device", validate=False)
    print(json.dumps({"C2_vs_A_d02_differing":
                      _diff_fields(c2, a, "d02_fields")[:24]}))
    d = leg("D child-streamed fb1", mode="streamed", feedback=1)
    e = leg("E resident solo (no child)", mode="resident", nested=False)
    n1 = leg("N1 negative (child store unpublished)", mode="streamed",
             feedback=0, unpublish=True)
    n1b = leg("N1b negative (unpublished, fb1)", mode="streamed",
              feedback=1, unpublish=True)
    # validate=False on the two controls that mis-force the child ON
    # PURPOSE; see run_leg.  Their measurement is the digest rows below.
    n2 = leg("N2 negative (stale tile tables)", mode="streamed",
             feedback=0, stale_tables=True, validate=False)
    w = leg("W negative (starved frame)", mode="streamed", feedback=0,
            frame_width=2, validate=False)

    # The treatment proofs: feedback=1 must CHANGE the parent, both modes.
    check("B differs from A in d01 (feedback fired, resident)",
          b["d01_sha256"] != a["d01_sha256"])
    check("D differs from C in d01 (feedback fired, child streamed)",
          d["d01_sha256"] != c["d01_sha256"])

    # The negative controls: the instrument must see every defect class.
    check("N1's d02 differs from A's (a stale child store is visible)",
          n1["d02_sha256"] != a["d02_sha256"])
    check("N1's d01 == A's d01 (one-way staleness cannot reach the parent)",
          n1["d01_sha256"] == a["d01_sha256"],
          f"differing: {_diff_fields(n1, a, 'd01_fields')[:8]}")
    check("N1b's d01 differs from B's (stale feedback reaches the parent "
          "and the gate can see it)",
          n1b["d01_sha256"] != b["d01_sha256"])
    check("N2's d02 differs from A's (frozen tile tables are visible: the "
          "launch-time reload is load-bearing)",
          n2["d02_sha256"] != a["d02_sha256"])
    check("N2's d01 == C's d01 (frozen tables cannot reach the parent)",
          n2["d01_sha256"] == c["d01_sha256"],
          f"differing: {_diff_fields(n2, c, 'd01_fields')[:8]}")
    check("W's d02 differs from A's (a starved frame is visible)",
          w["d02_sha256"] != a["d02_sha256"])
    check("W's d01 == C's d01 (a starved frame cannot reach the parent)",
          w["d01_sha256"] == c["d01_sha256"],
          f"differing: {_diff_fields(w, c, 'd01_fields')[:8]}")

    # The bit-identity claims.
    check("C2 d02 == A d02 (DEVICE-store child: tiling arithmetic alone)",
          c2["d02_sha256"] == a["d02_sha256"],
          f"differing: {_diff_fields(c2, a, 'd02_fields')[:8]}")
    check("C2 d01 == A d01 (device-store child leaves the parent alone)",
          c2["d01_sha256"] == a["d01_sha256"],
          f"differing: {_diff_fields(c2, a, 'd01_fields')[:8]}")
    check("C d01 == A d01 (one-way parent, child streamed vs resident)",
          c["d01_sha256"] == a["d01_sha256"],
          f"differing: {_diff_fields(c, a, 'd01_fields')[:8]}")
    check("C d02 == A d02 (one-way STREAMED child vs resident child)",
          c["d02_sha256"] == a["d02_sha256"],
          f"differing: {_diff_fields(c, a, 'd02_fields')[:8]}")
    check("D d01 == B d01 (two-way parent, child streamed vs resident)",
          d["d01_sha256"] == b["d01_sha256"],
          f"differing: {_diff_fields(d, b, 'd01_fields')[:8]}")
    check("D d02 == B d02 (two-way STREAMED child vs resident child)",
          d["d02_sha256"] == b["d02_sha256"],
          f"differing: {_diff_fields(d, b, 'd02_fields')[:8]}")
    check("E d01 == C d01 (#43 mirror: parent unchanged by its streamed "
          "child)",
          e["d01_sha256"] == c["d01_sha256"],
          f"differing: {_diff_fields(e, c, 'd01_fields')[:8]}")

    for row, ref in (("C", a), ("D", b)):
        got = {"C": c, "D": d}[row]
        check(f"{row} force_count == {ref['mode']} reference",
              got.get("force_count") == ref.get("force_count"))

    # The corridor traffic receipt: the child-side pull per FORCE is
    # bounded by kinds x 4 strips x (field + mup), width spec_bdy_width +
    # 8 (+1 slice widening), with a factor-2 slack.  A whole-child pull
    # cannot pass at any child large enough to stream for real; at THIS
    # child the bound documents the arithmetic and the >0 half documents
    # that the corridor actually moved store bytes.
    from gpuwm.core.nest_stream import NEST_FORCE_FRAME_HALO_CHILD_CELLS
    from tilestream.test_moving_nest import CHILD_NX, CHILD_NY, NZ

    width = 5 + NEST_FORCE_FRAME_HALO_CHILD_CELLS + 1
    strip_cells = 2 * (CHILD_NX + CHILD_NY + 4) * width
    kinds = 16
    window_bytes = strip_cells * 4 * (NZ + 1 + 1)
    bound = 2 * kinds * window_bytes * c["force_count"]
    whole_child = (16 * CHILD_NX * CHILD_NY * NZ * 4 * c["force_count"])
    check(f"C's corridor traffic is O(child frame) "
          f"({c['force_sync_bytes'] / 2**20:.1f} MiB <= bound "
          f"{bound / 2**20:.1f} MiB; whole-child pulls would be "
          f"{whole_child / 2**20:.0f} MiB)",
          0 < c["force_sync_bytes"] <= bound)

    print(json.dumps({"failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
