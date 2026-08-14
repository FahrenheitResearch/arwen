"""STREAMED-PARENT NESTING THROUGH THE EXECUTOR, both coupling directions.

Run it::

    python -m tilestream.test_nest_executor              # the whole matrix
    python -m tilestream.test_nest_executor --steps 6    # shorter window

WHAT THIS GATE ADDS OVER THE TWO THAT EXIST
-------------------------------------------
``tilestream/test_nest.py`` proved the COUPLER: ``force()`` and
``feedback_commit()`` driven by hand around a hand-transcribed schedule are
bit-identical streamed vs resident, one-way and two-way.
``tilestream/test_moving_nest.py`` proved the EXECUTOR moves a nest on a
streamed parent -- one-way only, because until now the executor REFUSED
``feedback = 1`` under a streamed parent, in a dispatch no gate drove: the
refusal and the coupler capability it contradicted arrived from different
lanes of the same three-way merge (2358b3c06), and the coupler gate never
goes through ``execute_experiment``.

This gate drives ``gpuwm.core.model.execute_experiment`` -- the product
loop, with its own FORCE/FEEDBACK edge dispatch, its windowed
``sync_to_state`` at FORCE and its clock imposition -- over a static
two-domain tree, and compares whole carrier sets:

``A  resident   fb0``   the control.
``B  resident   fb1``   MUST differ from A in the parent (the treatment
                        proof: an A/B pair whose arms agree proves the
                        experiment never ran, which this project has
                        shipped as a result five times and will not again).
``C  streamed   fb0``   d01 (store) and d02 (state) bit-identical to A.
``D  streamed   fb1``   bit-identical to B.  This is the leg the executor
                        refused before fix(executor); it is the reason this
                        gate exists.
``E  streamed   solo``  d01 alone, no child in the tree, same executor,
                        same schedule cadence.  E's store vs C's store is
                        the #43 proof under streaming: a one-way parent is
                        bitwise unchanged by the presence of its nest.
``N  negative``         C with the store UNPUBLISHED from the parent state
                        and the executor's windowed FORCE projection
                        disarmed -- today's coupler reads ``node.state``
                        raw, exactly the pre-repair code path.  d02 MUST
                        differ from A's, or this gate cannot see the defect
                        it patrols and every PASS above is vacuous.

Same discipline as the sibling gates: exit code 3 means the CARD WAS TAKEN
(a lost allocation race on a shared box is not a measurement), cadence
census printed for BOTH sides of every comparison, digests over the store
for a streamed domain and over the state for a resident one -- comparing a
streamed leg's ``node.state`` would compare the attach-time snapshot and
fail for a reason that has nothing to do with nesting.

THE INITIAL FEEDBACK TRANSACTION IS PART OF THE TREE, NOT THE LOOP
------------------------------------------------------------------
``build_experiment`` runs WRF's ``med_nest_initial`` equivalent -- prepare/
commit/finalize once, clocks equal at t=0 -- when it assembles a
``feedback = 1`` tree (model.py:660-672).  ``assemble_idealized_tree``
does not, so the fb1 legs run it by hand BEFORE the parent is attached to
the streaming transport, which is the same order the product takes: the
initial restriction lands on the resident prepared state, and ``attach``
copies the post-restriction parent into the store.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from tilestream.test_moving_nest import (
    PARENT_DT, RATIO, START_TIME, TILE, VRAM_NEEDED_GIB, build_parent,
    cadence_census, carrier_digest, child_cfg, domain_boundaries,
    parent_cfg, stream_parent, wait_for_vram)

#: Child placement, from test_moving_nest: a 72^2 child at ratio 3 is a
#: 24-parent-cell footprint, centred, clear of the specified zone.
I_PARENT_START = J_PARENT_START = 116

#: Parent steps in the window.  radt_minutes = 2 at dt = 30 is radiation
#: every 4 steps; cudt_minutes = 1 is cumulus every 2; 12 steps fire the
#: slow cadences 3x and 6x on both sides of every comparison, which the
#: census below prints rather than assumes.
DEFAULT_STEPS = 12


def experiment(pcfg, ccfg=None, *, steps=DEFAULT_STEPS):
    """A STATIC ExperimentConfig: one or two domains, no [relocation].

    ``test_moving_nest.experiment`` always builds the follow block; the legs
    here must not move the child, because a relocation between C and E would
    change which parent cells the coupler reads and turn the #43 comparison
    into a comparison of two different footprints.
    """
    from gpuwm.experiment import (DomainConfig, ExperimentConfig,
                                  ProjectionConfig, VerticalConfig)
    from tilestream import harness

    run_seconds = float(steps) * float(PARENT_DT)
    root_dc = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=run_seconds, run=pcfg, time_step=int(pcfg.dt))
    domains = [root_dc]
    if ccfg is not None:
        domains.append(DomainConfig(
            grid_id=2, parent_id=1,
            i_parent_start=I_PARENT_START, j_parent_start=J_PARENT_START,
            parent_grid_ratio=RATIO, parent_time_step_ratio=RATIO,
            history_interval_s=run_seconds, run=ccfg))
    return ExperimentConfig(
        name="nestexec", start_time=START_TIME, run_seconds=run_seconds,
        vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig("lambert", **{
            k: harness.REAL74_PROJECTION[k]
            for k in ("ref_lat", "ref_lon", "truelat1", "truelat2",
                      "stand_lon")}),
        restart_interval_s=0.0, domains=tuple(domains))


def assemble(exp, parent_state, parent_grid, *, feedback=0):
    """The tree, with the coupler carrying the leg's feedback flag.

    ``assemble_idealized_tree``'s default factory builds the one-way
    production coupler; the fb1 legs need the same coupler told
    ``feedback = 1``, which is exactly what ``build_experiment`` does with
    ``exp.feedback`` (model.py:660).
    """
    from gpuwm.core.nest import NestCoupler

    from tilestream.test_moving_nest import _preparer
    from gpuwm.verify.cases.nest_ideal_common import assemble_idealized_tree

    return assemble_idealized_tree(
        exp, parent_state, grids=[parent_grid] * len(exp.domains),
        coupler_factory=lambda node: NestCoupler(node, feedback=feedback),
        domain_preparer=_preparer)


def initial_feedback(model) -> None:
    """``med_nest_initial``: the t=0 restriction ``build_experiment`` runs."""
    from gpuwm.core.model import FeedbackScratch

    child = model.node(2)
    scratch = FeedbackScratch()
    child.coupler.feedback_prepare(child, scratch)
    child.coupler.feedback_commit(child)
    child.coupler.feedback_finalize(child)


def run_leg(mode, *, feedback=0, nested=True, steps=DEFAULT_STEPS,
            unpublish=False, force_halo=None, tile=None,
            verbose=True) -> dict:
    """One executor-driven run.  ``mode`` is ``resident`` or ``streamed``.

    ``unpublish=True`` is the negative control: the store is stripped from
    the parent state (``domain_store`` answers None, so the coupler reads
    the frozen attach-time arrays exactly as the pre-repair tree did).
    The integration itself still runs off the store and is still correct --
    what breaks is exactly and only what the repair repaired, which is what
    makes it a control rather than a second defect.

    ``force_halo`` overrides ``nest.NEST_FORCE_HALO_PARENT_CELLS`` for the
    leg.  ``0`` is the OTHER negative control, the one that validates the
    windowed FORCE corridor as an instrument: a window with no halo leaves
    the SINT stencil's outer cells stale, which must move the child and
    must NOT move the parent -- a too-small window that changed nothing
    would mean the window is not actually narrowing the read, and every
    windowed PASS above it would be vacuous.
    """
    import cupy as cp

    from gpuwm.core import nest as nest_mod
    from gpuwm.core.model import execute_experiment

    tile = TILE if tile is None else int(tile)
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
        steppers = stream_parent(model, geo, bnd, store="host", tile=tile)
        streamed = steppers[1]
        if unpublish:
            from gpuwm.core.streaming import _STORE_ATTR, domain_store

            delattr(model.root.state, _STORE_ATTR)
            assert domain_store(model.root.state) is None, (
                "the negative control failed to unpublish the store; "
                "whatever it measures next is not the pre-repair path")
            streamed.sync_to_state = lambda *a, **k: 0
    elif mode != "resident":
        raise ValueError(f"unknown mode {mode!r}")

    halo_before = nest_mod.NEST_FORCE_HALO_PARENT_CELLS
    if force_halo is not None:
        nest_mod.NEST_FORCE_HALO_PARENT_CELLS = int(force_halo)
    try:
        t0 = time.perf_counter()
        execution = execute_experiment(
            model, steppers=steppers, validate_state=True,
            pool_trim_per_period=False)
        cp.cuda.runtime.deviceSynchronize()
        wall = time.perf_counter() - t0
    finally:
        nest_mod.NEST_FORCE_HALO_PARENT_CELLS = halo_before

    d01_source = streamed.store if streamed is not None else model.root.state
    d01_sha, d01_per = carrier_digest(d01_source)
    out = {
        "mode": mode, "feedback": int(feedback), "nested": bool(nested),
        "unpublished": bool(unpublish),
        "steps": int(execution.steps), "forces": int(execution.forces),
        "d01_sha256": d01_sha, "d01_fields": d01_per,
        "cadence": cadence_census(model, steppers),
        "wall_s": round(wall, 2),
    }
    if nested:
        d02_sha, d02_per = carrier_digest(model.node(2).state)
        coupler = model.node(2).coupler
        out.update(
            d02_sha256=d02_sha, d02_fields=d02_per,
            force_count=int(coupler.force_count),
            feedback_count=int(coupler.feedback_count),
            force_sync_bytes=int(coupler.force_sync_bytes),
            force_halo=(nest_mod.NEST_FORCE_HALO_PARENT_CELLS
                        if force_halo is None else int(force_halo)))
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
            return run_leg(steps=args.steps, tile=args.tile, **kw)
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
    c = leg("C streamed fb0", mode="streamed", feedback=0)
    d = leg("D streamed fb1", mode="streamed", feedback=1)
    e = leg("E streamed solo (no nest)", mode="streamed", nested=False)
    n = leg("N negative (store unpublished)", mode="streamed", feedback=0,
            unpublish=True)
    w = leg("W negative (halo-starved window)", mode="streamed", feedback=0,
            force_halo=0)

    # The treatment proof: feedback=1 must CHANGE the parent, resident.
    check("B differs from A in d01 (feedback fired, resident)",
          b["d01_sha256"] != a["d01_sha256"])
    check("D differs from C in d01 (feedback fired, streamed)",
          d["d01_sha256"] != c["d01_sha256"])
    # The negative control: the instrument must see the defect.
    check("N's d02 differs from A's (the gate can see a stale coupler)",
          n["d02_sha256"] != a["d02_sha256"])

    # The bit-identity claims.
    check("C d01 == A d01 (one-way parent, streamed vs resident)",
          c["d01_sha256"] == a["d01_sha256"],
          f"differing: {_diff_fields(c, a, 'd01_fields')[:8]}")
    check("C d02 == A d02 (one-way child under a streamed parent)",
          c["d02_sha256"] == a["d02_sha256"],
          f"differing: {_diff_fields(c, a, 'd02_fields')[:8]}")
    check("D d01 == B d01 (two-way parent, streamed vs resident)",
          d["d01_sha256"] == b["d01_sha256"],
          f"differing: {_diff_fields(d, b, 'd01_fields')[:8]}")
    check("D d02 == B d02 (two-way child under a streamed parent)",
          d["d02_sha256"] == b["d02_sha256"],
          f"differing: {_diff_fields(d, b, 'd02_fields')[:8]}")
    check("E d01 == C d01 (#43 under streaming: parent unchanged by nest)",
          e["d01_sha256"] == c["d01_sha256"],
          f"differing: {_diff_fields(e, c, 'd01_fields')[:8]}")

    for row, ref in (("C", a), ("D", b)):
        got = {"C": c, "D": d}[row]
        check(f"{row} force_count == {ref['mode']} reference",
              got.get("force_count") == ref.get("force_count"))

    # The windowed FORCE corridor, as an instrument and as a bound.
    check("W's d02 differs from A's (a starved window is visible)",
          w["d02_sha256"] != a["d02_sha256"])
    check("W's d01 == C's d01 (a starved window cannot reach the parent)",
          w["d01_sha256"] == c["d01_sha256"])
    # O(child footprint), asserted, not narrated: the pull per FORCE is
    # bounded by kinds x (window field + window mup), with the +1 slice
    # widening and a factor-2 slack.  A whole-parent pull (the pre-corridor
    # cost) is ~39x this bound at this geometry and CANNOT pass.
    from tilestream.test_moving_nest import (CHILD_NX, CHILD_NY, NZ,
                                             PARENT_NX, PARENT_NY)
    from gpuwm.core.nest import NEST_FORCE_HALO_PARENT_CELLS as HALO

    span = -(-CHILD_NX // RATIO) + 2 * HALO + 1
    span_j = -(-CHILD_NY // RATIO) + 2 * HALO + 1
    kinds = 16          # full(real74)+KF forcing set; superset is slack
    window_bytes = span * span_j * 4 * (NZ + 1 + 1)   # field + w face + mup
    bound = 2 * kinds * window_bytes * c["force_count"]
    full_parent = (16 * PARENT_NX * PARENT_NY * NZ * 4
                   * c["force_count"])
    check(f"C's corridor traffic is O(child footprint) "
          f"({c['force_sync_bytes'] / 2**20:.1f} MiB <= bound "
          f"{bound / 2**20:.1f} MiB; whole-parent would be "
          f"{full_parent / 2**20:.0f} MiB)",
          0 < c["force_sync_bytes"] <= bound)

    print(json.dumps({"failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
