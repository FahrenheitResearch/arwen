"""Prove the DA cycle's loop is CLOSED, and prove it the hard way.

Running is not the proof.  The proof is that the analysis demonstrably
changed the subsequent forecast, and that a null analysis demonstrably
did not.  Two arms, one card, one session.

Every device-touching phase runs in a BRIDGE PROCESS
(``mpas_cycle_bridge.worker``), never in this one.  This module is
``gpuwm.cycle``, and ``gpuwm.cycle`` is gpuwm: the port pins gpuwm by
commit and refuses to load its frozen Arwen if gpuwm is already live
from another tree.  That refusal is exactly where the first version of
this harness died -- in ``seed``, before it ever ran a step.  What
crosses the process boundary is anchors and segments on disk.

``seed``
    Construct the device stack from the init, integrate ``--steps`` real
    dycore steps, and take a boundary download.  Publish that boundary
    as anchor 001 into BOTH arm roots -- prognostics, the
    saved-diagnostics sidecar, and the physics backend's restart payload
    in the anchor's seam slot -- with the arm's increment baked in.  Then
    CONTINUE THE SAME STACK, in the same process, for another ``--steps``
    and save that as ``uninterrupted.npz``.  That second boundary is the
    un-analysed continuation: the thing a closed loop must reproduce when
    the analysis is zero.

``arm``
    A FRESH PROCESS.  Read anchor 001 off disk, apply the increment it
    carries, rehydrate the device stack through the port's rehydration
    seam, read the state back OFF THE DEVICE, integrate ``--steps``.
    Publishes anchor 002, stamped from evidence.

``verdict``
    null arm vs uninterrupted        MUST be bitwise identical
    applied arm vs uninterrupted     MUST differ

**A nonzero delta is not sufficient evidence that the experiment ran.**
Two crashed arms once differed slightly and sailed past an exact-0.0
guard while publishing a fake ``speedup: 1.0073`` and ``meets_gate:
true``.  So the verdict is suppressed field by field unless the positive
evidence is present in the anchors themselves: the requested number of
step receipts, the ingestion receipt's own account of what it saw, an
``mpas-cuda`` stamp, and -- the one that cannot be faked -- the hash of
the state read back OFF THE DEVICE after rehydration matching the
analysed state that was written to disk.  If the device did not take the
analysis, that hash does not match, and no amount of downstream
divergence can rescue the claim.

**The treatment is a flat rho_theta patch, not a LETKF increment.**  It
is a control for the loop MECHANISM and must never be reported as data
assimilation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gpuwm.cycle import mpas_bridge
from gpuwm.cycle.anchor import (anchor_for_cycle, prognostic_sha256,
                                read_anchor, write_anchor)
from gpuwm.cycle.ingestion import verify_ingestion

UNINTERRUPTED_NAME = "uninterrupted.npz"
VERDICT_NAME = "closed-loop-verdict.json"
SUPPRESSED = "SUPPRESSED: evidence absent"
TREATMENT_NOTE = ("flat rho_theta patch -- a control for the loop mechanism, "
                  "NOT a DA increment and NOT an analysis of any observation")


def _port_config(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(Path(args.port_config).read_text(encoding="utf-8"))


def _increment(rho_theta: np.ndarray, mode: str,
               args: argparse.Namespace) -> np.ndarray:
    """The A/B's treatment.  See TREATMENT_NOTE -- this is not DA."""
    delta = np.zeros_like(rho_theta)
    if mode != "applied":
        return delta
    entity_axis = int(np.argmax(delta.shape))
    view = np.moveaxis(delta, entity_axis, -1)
    touched = min(args.cells, view.shape[-1])
    view[..., :touched] += np.asarray(args.amplitude, delta.dtype)
    return np.ascontiguousarray(np.moveaxis(view, -1, entity_axis))


def seed(args: argparse.Namespace) -> int:
    config = _port_config(args)
    out = Path(args.out).expanduser().absolute()
    out.mkdir(parents=True, exist_ok=True)

    mpas_bridge.launch(phase="seed", port_root=args.port_root,
                       port_config=args.port_config, steps=int(args.steps),
                       out=out / "worker-seed", timeout=args.timeout)
    anchor_seg = mpas_bridge.read_segment(out / "worker-seed" / "anchor")
    control_seg = mpas_bridge.read_segment(out / "worker-seed" / "control")
    arrays = mpas_bridge.segment_arrays(anchor_seg)
    control = mpas_bridge.segment_arrays(control_seg)["prognostic"]

    anchor_seconds = float(anchor_seg["model_time_seconds"])
    dt = float(anchor_seg["dt_seconds"])
    rho_theta = arrays["prognostic"]["rho_theta"]

    # The seed boundary carries no APPLIED analysis -- the increment sits
    # in it as PENDING -- so it is stamped mpas-cuda-frames.  Only the
    # boundary an arm publishes, after the analysis has gone back through
    # the dycore, can earn mpas-cuda.
    for root, mode in ((args.null_root, "null"),
                       (args.applied_root, "applied")):
        delta = _increment(rho_theta, mode, args)
        published = write_anchor(
            Path(root), cycle_index=1,
            anchor_ticks=int(round(anchor_seconds * 1000)),
            valid_time=config.get("valid_time", "2026-08-12T06:00:00Z"),
            parent_kind=mpas_bridge.FRAMES_KIND,
            prognostic=arrays["prognostic"], derived=arrays["derived"],
            seam=arrays["seam"], mesh_id=str(config["mesh"]),
            analysis={"state": "PENDING", "arrays": {"rho_theta": delta}},
            extra={"backend_restart": anchor_seg["backend_restart"],
                   "arm": mode, "treatment": TREATMENT_NOTE,
                   "integration": {"steps_executed":
                                   anchor_seg["steps_executed"],
                                   "dt_seconds": dt}})
        print(f"[seed] {mode} arm anchor {published} "
              f"(increment nonzero elements {int(np.count_nonzero(delta))})")

    np.savez(out / UNINTERRUPTED_NAME, **control)
    receipt = {
        "steps_per_segment": int(args.steps),
        "dt_seconds": dt,
        "anchor_seconds": anchor_seconds,
        "control_seconds": float(control_seg["model_time_seconds"]),
        "segment_1_step_receipts": anchor_seg["steps_executed"],
        "segment_2_step_receipts": control_seg["steps_executed"],
        "anchor_state_sha256": anchor_seg["prognostic_sha256"],
        "uninterrupted_state_sha256": prognostic_sha256(control),
        "frozen_execution_sources": anchor_seg["frozen_execution_sources"],
        "arwen_pin": anchor_seg["arwen_pin"],
        "treatment": TREATMENT_NOTE,
    }
    (out / "seed-receipt.json").write_text(
        json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    print("[seed] " + json.dumps(receipt, default=str))
    return 0


def arm(args: argparse.Namespace) -> int:
    """One fresh-process cycle leg: anchor 001 in, anchor 002 out."""
    config = _port_config(args)
    root = Path(args.root).expanduser().absolute()
    source = anchor_for_cycle(root, 1)
    if source is None:
        raise SystemExit(f"no anchor 001 under {root}; run seed first")
    doc = read_anchor(source)
    increment = doc.increment() or {}

    out = Path(args.out).expanduser().absolute()
    segment = mpas_bridge.launch(
        phase="segment", port_root=args.port_root,
        port_config=args.port_config, steps=int(args.steps),
        anchor=source, out=out, timeout=args.timeout)

    # The gate is fed the hash the DEVICE gave back after rehydration,
    # not the host copy that was uploaded: a host-side comparison would
    # pass even if the upload never happened.
    ingestion = verify_ingestion(
        background_sha256=segment["background_sha256"],
        increment=increment,
        analysis_sha256=segment["rehydrated_sha256"],
        label=f"{args.arm_name} arm rehydration")

    stamp = mpas_bridge.stamp_for_segment(segment,
                                          steps_requested=int(args.steps))
    arrays = mpas_bridge.segment_arrays(segment)
    published = write_anchor(
        root, cycle_index=2,
        anchor_ticks=int(round(float(segment["model_time_seconds"]) * 1000)),
        valid_time=config.get("valid_time_2",
                              config.get("valid_time",
                                         "2026-08-12T12:00:00Z")),
        parent_kind=stamp["parent_kind"],
        prognostic=arrays["prognostic"], derived=arrays["derived"],
        seam=arrays["seam"], mesh_id=str(config["mesh"]),
        analysis={"state": "APPLIED", "ingestion": ingestion},
        extra={"backend_restart": segment["backend_restart"],
               "arm": args.arm_name, "stamp": stamp,
               "treatment": TREATMENT_NOTE,
               "integration": {"steps_executed": segment["steps_executed"],
                               "dt_seconds": segment["dt_seconds"]}})
    print(f"[arm:{args.arm_name}] published {published} "
          f"stamped {stamp['parent_kind']}")
    print(json.dumps(stamp, indent=2, default=str))
    return 0


def verdict(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().absolute()
    control = dict(np.load(out / UNINTERRUPTED_NAME))
    control_sha = prognostic_sha256(control)

    report: dict[str, Any] = {"uninterrupted_sha256": control_sha,
                              "treatment": TREATMENT_NOTE, "arms": {}}
    for name, root in (("null", args.null_root),
                       ("applied", args.applied_root)):
        arm_block: dict[str, Any] = {"root": str(root)}
        published = anchor_for_cycle(Path(root), 2)
        if published is None:
            arm_block["verdict"] = SUPPRESSED
            arm_block["reason"] = "the arm published no anchor 002"
            report["arms"][name] = arm_block
            continue
        doc = read_anchor(published)
        manifest = doc.manifest
        integration = manifest.get("integration") or {}
        ingestion = (manifest.get("analysis") or {}).get("ingestion") or {}
        stamp = manifest.get("stamp") or {}
        steps = integration.get("steps_executed")
        arm_block["parent_kind"] = manifest["parent"]["kind"]
        arm_block["steps_executed"] = steps
        arm_block["ingestion_state"] = ingestion.get("state")
        arm_block["increment_nonzero_cells"] = ingestion.get(
            "increment_nonzero_cells")
        arm_block["analysis_sha256"] = ingestion.get("analysis_sha256")
        arm_block["background_sha256"] = ingestion.get("background_sha256")
        arm_block["rehydrated_sha256"] = stamp.get("rehydrated_sha256")
        arm_block["analysis_reentered_dycore"] = stamp.get(
            "analysis_reentered_dycore")
        arm_block["published_sha256"] = manifest["parent"][
            "prognostic_sha256"]
        arm_block["bitwise_identical_to_uninterrupted"] = (
            arm_block["published_sha256"] == control_sha)

        # Positive evidence, demanded before any verdict is emitted.
        evidence: list[str] = []
        if not steps:
            evidence.append("no step receipts: the dycore may not have run")
        if arm_block["parent_kind"] != mpas_bridge.CLOSED_KIND:
            evidence.append(f"parent stamped {arm_block['parent_kind']!r}, "
                            "not mpas-cuda: the loop was not closed")
        if not ingestion:
            evidence.append("no ingestion receipt: the analysis was never "
                            "put through the three-hash gate")
        if stamp.get("analysis_reentered_dycore") is not True:
            evidence.append("no device readback proving the analysed state "
                            "reached the dycore")
        for gap in stamp.get("evidence_gaps") or ():
            evidence.append(f"stamp gap: {gap}")
        arm_block["evidence_gaps"] = evidence
        if evidence:
            arm_block["verdict"] = SUPPRESSED
        else:
            arm_block["verdict"] = (
                "BITWISE MATCH"
                if arm_block["bitwise_identical_to_uninterrupted"]
                else "DIVERGED")
        report["arms"][name] = arm_block

    null = report["arms"].get("null", {})
    applied = report["arms"].get("applied", {})
    if null.get("verdict") == "BITWISE MATCH" and \
            applied.get("verdict") == "DIVERGED":
        report["loop"] = "CLOSED"
        report["claim"] = (
            "the analysis re-entered the dycore: a zero analysis reproduced "
            "the un-analysed continuation bit for bit through a full "
            "anchor-on-disk round trip across a process boundary, and a "
            "nonzero one changed the subsequent forecast. This is a proof "
            "of the CYCLE MECHANISM. The treatment was a flat rho_theta "
            "patch, not DA.")
    else:
        report["loop"] = "NOT PROVEN"
        report["claim"] = SUPPRESSED
    (out / VERDICT_NAME).write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["loop"] == "CLOSED" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cycle_closed_loop_proof",
                                     description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)

    seed_p = sub.add_parser("seed")
    seed_p.add_argument("--port-root", required=True)
    seed_p.add_argument("--port-config", required=True)
    seed_p.add_argument("--steps", type=int, required=True)
    seed_p.add_argument("--null-root", required=True)
    seed_p.add_argument("--applied-root", required=True)
    seed_p.add_argument("--cells", type=int, default=20_000)
    seed_p.add_argument("--amplitude", type=float, default=0.05)
    seed_p.add_argument("--timeout", type=float, default=None)
    seed_p.add_argument("--out", required=True)
    seed_p.set_defaults(func=seed)

    arm_p = sub.add_parser("arm")
    arm_p.add_argument("--port-root", required=True)
    arm_p.add_argument("--port-config", required=True)
    arm_p.add_argument("--steps", type=int, required=True)
    arm_p.add_argument("--root", required=True)
    arm_p.add_argument("--arm-name", required=True, dest="arm_name")
    arm_p.add_argument("--timeout", type=float, default=None)
    arm_p.add_argument("--out", required=True)
    arm_p.set_defaults(func=arm)

    verdict_p = sub.add_parser("verdict")
    verdict_p.add_argument("--null-root", required=True)
    verdict_p.add_argument("--applied-root", required=True)
    verdict_p.add_argument("--out", required=True)
    verdict_p.set_defaults(func=verdict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
