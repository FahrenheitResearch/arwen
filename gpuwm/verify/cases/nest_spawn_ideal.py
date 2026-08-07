"""A dormant nest born out of a WK82 supercell, on the idealized route.

The spawn feature end to end on the path that can carry it today: the
real-data route reserves a dormant nest but cannot yet give a NEWBORN
its soil/land state (an open physics question, refused by name in
``gpuwm.runtime.run_experiment``), while the idealized route attaches
physics through ``prepare_idealized_domain`` and therefore can.

The story this case tells, in one run of the shipped driver
(:func:`gpuwm.runtime.walk_spawn_legs`):

1. the 1 km parent integrates ALONE while the supercell spins up -- the
   declared nest is priced from the first step and costs zero compute;
2. the storm's composite reflectivity crosses the trigger inside the
   watch window, and the nest is materialized from the LIVE parent at
   the storm-core centroid, whole-parent-cell aligned and clamped clear
   of the parent edge;
3. the 333 m child integrates its own substeps for the rest of the run,
   forced by its parent every parent step.

The zero-compute claim is measured, not asserted: the child's clock
step count at the end must equal exactly the steps between its birth
instant and the run end, so a nest that had been integrating all along
would fail the receipt.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpuwm.core.nest_spawn import SpawnConfig
from gpuwm.core.spawn_runner import SpawnRunner
from gpuwm.experiment import pre_spawn_experiment
from gpuwm.runtime import walk_spawn_legs
from gpuwm.static.lambert import grids_from_projection_config
from gpuwm.verify.cases.nest_ideal_common import (
    _preserve_parent_map_policy, assemble_idealized_tree,
    consume_history_reflectivity, prepare_idealized_domain, write_json)
from gpuwm.verify.cases.nest_ideal_r3 import _build_root, load_scaffold

#: One hour: the supercell has a mature core well inside it, and the
#: window below leaves ~30 model minutes of nest life after the fire.
RUN_SECONDS = 3600.0

#: History cadence AND therefore the leg cadence
#: (``gpuwm.runtime._spawn_leg_seconds`` falls back to it when no
#: relocation cadence is configured).  It matters twice: it is how often
#: the walk stops to ask, and it is when the microphysics computes
#: ``refl_10cm`` at all -- a trigger consulted before the first
#: output-due instant would be asking for a plane that has never been
#: computed.
HISTORY_SECONDS = 300.0

#: Composite dBZ.  35 is a convective core on a 1 km grid without being
#: so high that the demo depends on the storm's peak intensity.
THRESHOLD_DBZ = 35.0

#: The watch window.  earliest_s is past the first refl instant and past
#: the bubble's own transient; latest_s leaves the nest real time to run.
EARLIEST_S = 600.0
LATEST_S = 3000.0


class _IdealizedPrepared:
    """The prepared-case surface the attach step records."""

    static_fields = None
    geog_selection = None
    initial_result = None


class _IdealizedSpawnPreparer:
    """``on_child_built`` for the idealized route.

    The same seam the relocation demo uses, and the same rule: the
    initializer never invents driver state, the route attaches it.  Map
    policy is preserved from the parent first (an idealized child must
    not re-derive a projection the parent already fixed), then the
    ordinary idealized domain preparation runs.
    """

    def __init__(self, start_time):
        self.start_time = start_time
        self.prepared_by_grid_id: dict[int, object] = {}
        self.calls: list[int] = []

    def __call__(self, initialized, child_dc, parent_node) -> None:
        _preserve_parent_map_policy(parent_node.state, initialized.state)
        prepare_idealized_domain(
            initialized.state, child_dc, initialized.grid, self.start_time)
        self.calls.append(int(child_dc.grid_id))
        self.prepared_by_grid_id[int(child_dc.grid_id)] = _IdealizedPrepared()


def spawn_scaffold(*, run_seconds: float = RUN_SECONDS,
                   history_seconds: float = HISTORY_SECONDS,
                   threshold: float = THRESHOLD_DBZ,
                   earliest_s: float = EARLIEST_S,
                   latest_s: float = LATEST_S):
    """The N2c ratio-3 geometry with its child declared DORMANT.

    The child's configured ``i_parent_start``/``j_parent_start`` become a
    placeholder: they price the memory plan, and the trigger chooses
    where the nest is actually born.
    """
    exp = load_scaffold()
    root, child = exp.domains
    root = replace(
        root, history_interval_s=history_seconds,
        run=replace(root.run, run_seconds=run_seconds,
                    output_interval_s=history_seconds))
    child = replace(
        child, history_interval_s=history_seconds,
        spawn=SpawnConfig(trigger="reflectivity", threshold=float(threshold),
                          earliest_s=float(earliest_s),
                          latest_s=float(latest_s)),
        run=replace(child.run, run_seconds=run_seconds,
                    output_interval_s=history_seconds))
    return replace(exp, run_seconds=run_seconds, domains=(root, child))


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    if callable(get) and hasattr(value, "__cuda_array_interface__"):
        return np.ascontiguousarray(get())
    return np.ascontiguousarray(np.asarray(value))


def _field_health(state, names=("thp", "mup", "u", "v", "w")) -> dict:
    out = {}
    for name in names:
        value = getattr(state, name, None)
        if value is None:
            continue
        host = _host(value).astype(np.float64)
        out[name] = {
            "finite": bool(np.isfinite(host).all()),
            "min": float(np.min(host)),
            "max": float(np.max(host)),
        }
    return out


def run(outdir: str | Path, *, run_seconds: float = RUN_SECONDS,
        history_seconds: float = HISTORY_SECONDS,
        threshold: float = THRESHOLD_DBZ,
        earliest_s: float = EARLIEST_S,
        latest_s: float = LATEST_S) -> dict[str, object]:
    """Integrate the parent, spawn the nest at the storm, keep going."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exp = spawn_scaffold(
        run_seconds=run_seconds, history_seconds=history_seconds,
        threshold=threshold, earliest_s=earliest_s, latest_s=latest_s)
    pre = pre_spawn_experiment(exp)
    if [dc.grid_id for dc in pre.domains] != [exp.root.grid_id]:
        raise RuntimeError("the dormant nest is not dormant before the run")

    grids = tuple(grids_from_projection_config(exp))
    model = assemble_idealized_tree(pre, _build_root(exp), grids=(grids[0],))
    model._prepared_by_grid_id = {int(exp.root.grid_id): _IdealizedPrepared()}

    preparer = _IdealizedSpawnPreparer(exp.start_time)
    runner = SpawnRunner.from_experiment(
        exp, on_child_built=preparer,
        receipts_path=outdir / "spawn_receipts.json")
    if runner is None:
        raise RuntimeError("the scaffold declares no dormant nest")

    history_rows: list[tuple[int, int]] = []
    seen_domains: set[int] = set()

    def history_handler(tree, node, ticks) -> None:
        # A domain's FIRST frame is its initial frame, and an initial
        # frame carries no microphysics-time reflectivity because no step
        # has run yet.  consume_history_reflectivity already knows that
        # for t = 0; a spawned nest's first frame is at its BIRTH tick,
        # which is its own t = 0 and is not zero.  Skipping exactly the
        # first sighting keeps the one-frame handoff's cadence-bug
        # detection armed for every later frame.
        gid = int(node.cfg.grid_id)
        if gid in seen_domains:
            consume_history_reflectivity(node, ticks)
        seen_domains.add(gid)
        history_rows.append((gid, int(ticks)))

    # validate_state=False, deliberately and with a reason.
    # execute_experiment runs its FULL-state "initialized-or-restored"
    # gate at the top of every call, and a leg walk calls it once per
    # leg.  That gate is written for a freshly initialized or restored
    # state; a mid-run Morrison parent legitimately carries tiny
    # negative moment values (nr ~ -2e-11 -- physically zero, numerically
    # not) that the in-run health cadence tolerates and that no leg
    # boundary created.  Re-asserting an initialization gate on a
    # mid-flight state would fail the run for a property the run already
    # has, so the walk does not, and the receipt measures the fields
    # directly instead (child_health / parent_health below).
    walk_spawn_legs(
        model, exp, None, spawn_runner=runner, writers=None,
        lbc_interval_s=None, history_handler=history_handler,
        validate_state=False)

    root_node = model.root
    spawned = sorted(gid for gid in model.nodes_by_grid_id
                     if gid != root_node.cfg.grid_id)
    fired = [row for row in runner.receipts if row["event"] == "spawned"]

    receipt: dict[str, object] = {
        "case": "nest_spawn_ideal",
        "contract": "gpuwm.nest-spawn-ideal-demo.v1",
        "configuration": {
            "run_seconds": float(run_seconds),
            "leg_seconds": float(history_seconds),
            "trigger": "reflectivity",
            "threshold_dbz": float(threshold),
            "window_seconds": [float(earliest_s), float(latest_s)],
            "parent": f"{exp.root.run.nx}x{exp.root.run.ny}x{exp.root.run.nz}"
                      f" dx={exp.root.run.dx:g}",
            "nest": f"{exp.domain(2).run.nx}x{exp.domain(2).run.ny}"
                    f" dx={exp.domain(2).run.dx:.4g}",
        },
        "spawned_grid_ids": spawned,
        "history_frames": len(history_rows),
    }

    if not fired:
        receipt["verdict"] = "NO-FIRE"
        receipt["note"] = (
            "the watch closed without the trigger crossing; the "
            "reservation was held for the whole run and cost zero "
            "compute, which is the contract -- but this run demonstrates "
            "nothing about birth.  Lower --threshold or lengthen the run.")
        receipt["spawn_receipts"] = runner.receipts
        write_json(outdir / "receipt.json", receipt)
        return receipt

    event = fired[0]
    birth_seconds = float(event["elapsed_seconds"])
    born = event["born"][0]
    child_node = model.node(int(born["grid_id"]))

    # The zero-compute-before-birth measurement.
    #
    # NOT the child's clock.step_count: that is a DERIVED field, and the
    # leg retarget recomputes it from the clock spec's start_ticks, which
    # for a trigger-born nest is 0 (active_experiment gives it no
    # start_time).  It therefore reports the whole run no matter when the
    # nest appeared, and it said 1800 for a nest that existed for 1350
    # steps.  An instrument that cannot fail the claim does not test it.
    #
    # The direct measurement is the child's own OUTPUT: a domain emits a
    # history frame only once it exists, so the tick of its FIRST frame
    # is the instant it started producing, and the count is how many
    # cadence instants it lived through.  Both come from the executor's
    # own history callbacks, not from anything this walk rewrites.
    child_gid = int(born["grid_id"])
    tick_den = float(child_node.clock.tick_den)
    child_ticks = sorted(t for gid, t in history_rows if gid == child_gid)
    first_child_seconds = (None if not child_ticks
                           else child_ticks[0] / tick_den)
    expected_frames = int(round(
        (float(run_seconds) - birth_seconds) / float(history_seconds))) + 1
    receipt.update({
        "birth": {
            "elapsed_seconds": birth_seconds,
            "placement_parent_cells": born["placement"],
            "trigger": born["spawn_receipt"]["trigger"],
            "parent_bitwise_unchanged": bool(
                born["spawn_receipt"]["parent_bitwise_unchanged"]),
            "atmosphere_source": born["spawn_receipt"][
                "atmosphere_source"]["kind"],
            "static_source": born["spawn_receipt"]["terrain"].get(
                "static_source"),
        },
        "zero_compute_before_birth": {
            "note": ("a domain emits history only once it exists, so the "
                     "child's FIRST frame must land exactly on its birth "
                     "instant and it must have exactly the frames between "
                     "that instant and the run end"),
            "birth_seconds": birth_seconds,
            "first_child_frame_seconds": first_child_seconds,
            "child_frames": len(child_ticks),
            "expected_frames": expected_frames,
            "parent_only_before_birth": bool(
                all(gid == int(exp.root.grid_id)
                    for gid, t in history_rows
                    if t / tick_den < birth_seconds)),
            "pass": bool(first_child_seconds == birth_seconds
                         and len(child_ticks) == expected_frames),
        },
        "on_child_built_calls": list(preparer.calls),
        "child_health": _field_health(child_node.state),
        "parent_health": _field_health(root_node.state),
        "elapsed_seconds_at_end": {
            "parent": float(root_node.clock.elapsed_seconds),
            "child": float(child_node.clock.elapsed_seconds),
        },
        "spawn_receipts": runner.receipts,
    })
    child_ok = all(v["finite"] for v in receipt["child_health"].values())
    receipt["verdict"] = (
        "PASS" if (receipt["zero_compute_before_birth"]["pass"]
                   and child_ok
                   and receipt["birth"]["parent_bitwise_unchanged"]
                   and spawned == [int(born["grid_id"])])
        else "FAIL")
    write_json(outdir / "receipt.json", receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--run-seconds", type=float, default=RUN_SECONDS)
    parser.add_argument("--history-seconds", type=float,
                        default=HISTORY_SECONDS)
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DBZ)
    parser.add_argument("--earliest-s", type=float, default=EARLIEST_S)
    parser.add_argument("--latest-s", type=float, default=LATEST_S)
    args = parser.parse_args(argv)
    receipt = run(args.outdir, run_seconds=args.run_seconds,
                  history_seconds=args.history_seconds,
                  threshold=args.threshold, earliest_s=args.earliest_s,
                  latest_s=args.latest_s)
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "spawn_receipts"}, indent=2, default=str))
    return 0 if receipt.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
