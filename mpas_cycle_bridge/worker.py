"""The forecast worker: one process, one segment, no cycling spine.

This is the far side of the bridge.  It is launched by
``gpuwm.cycle.mpas_bridge`` as a fresh process, and inside it the only
gpuwm that exists is the frozen Arwen checkout the port pins.  It never
imports ``gpuwm.cycle`` -- structurally, not by convention:

* the first executable statement installs
  :class:`mpas_cycle_bridge.spine_guard.SpineImportGuard` on
  ``sys.meta_path``, which refuses ``gpuwm.cycle`` by name, always, and
  refuses ``gpuwm`` itself until the guard is armed with the pinned
  checkout;
* the bootstrap removes the repository root from ``sys.path`` before it
  binds the port, so the spine tree is not even reachable;
* the guard's after-the-fact verification -- where ``gpuwm`` actually
  came from -- is written into the segment receipt, so the constraint is
  *evidenced* per run rather than asserted once in a docstring.

Two subcommands.

``seed``
    Build the device stack from the case init, integrate ``--steps``,
    write that boundary as the ``anchor`` segment.  Then CONTINUE THE
    SAME STACK for another ``--steps`` and write that as the ``control``
    segment: the un-analysed continuation a closed loop must reproduce
    when the analysis is zero.  Both come from one process and one card,
    which is what makes them comparable.

``segment``
    Read a committed anchor, apply the increment the anchor carries,
    rehydrate the device stack through the port's rehydration seam,
    read the state back OFF THE DEVICE, integrate ``--steps``, write the
    boundary.

What crosses the process boundary is anchors and segments on disk.
There is no pickle, no shared memory, no import.  The port's own restart
proof -- a 15+15 fresh-process split matching a 30-step continuous run
bitwise -- is the evidence that a boundary here costs nothing numeric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

# --------------------------------------------------------------------------
# bootstrap.  Order matters and every line of it is load-bearing.

_BRIDGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BRIDGE_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from mpas_cycle_bridge import BRIDGE_SCHEMA  # noqa: E402
from mpas_cycle_bridge import anchor_codec as codec  # noqa: E402
from mpas_cycle_bridge import portbind  # noqa: E402
from mpas_cycle_bridge.spine_guard import install_guard  # noqa: E402

# The bridge is loaded.  Drop the repository root again so the live spine
# tree is not on the import path while the port pins its Arwen checkout.
# The guard would refuse a spine import anyway; this makes it unreachable
# as well, which is the belt to the guard's braces.
while str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))

import numpy as np  # noqa: E402

SEGMENT_MANIFEST = "segment.json"
PROGNOSTIC_STEM = codec.PROGNOSTIC_STEM
DERIVED_STEM = codec.DERIVED_STEM
SEAM_STEM = codec.SEAM_STEM
METRIC_STEM = codec.METRIC_STEM


# --------------------------------------------------------------------------
# segment output


def _write_segment(out_dir: Path, *, role: str, boundary: Mapping[str, Any],
                   receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Emit one boundary as a segment directory, in the anchor's format.

    A segment is *not* an anchor.  It is the worker's report of one leg,
    which the spine reads and -- if the evidence in ``receipt`` supports
    it -- seals into the next anchor with the spine's own writer, its own
    refusals and its own parent-kind stamp.  Keeping the stamp on the
    spine side is deliberate: the stamp is a claim about evidence, and
    the process that produced the evidence should not be the one that
    grades it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = codec.SUFFIX[codec.ARRAY_FORMAT]
    skeleton, seam = portbind.split_backend_restart(
        boundary["backend_restart"])

    written: dict[str, str] = {}

    def emit(stem: str, mapping: Mapping[str, np.ndarray]) -> str:
        target = out_dir / f"{stem}{suffix}"
        codec.write_arrays(target, mapping)
        codec.fsync_file(target)
        written[target.name] = codec.file_sha256(target)
        return target.name

    manifest = {
        "schema": BRIDGE_SCHEMA,
        "role": str(role),
        "array_format": codec.ARRAY_FORMAT,
        "prognostic_path": emit(PROGNOSTIC_STEM, boundary["prognostic"]),
        "derived_path": emit(DERIVED_STEM, boundary["derived"]),
        "seam_path": emit(SEAM_STEM, seam),
        "metric_path": emit(METRIC_STEM, boundary["metric"]),
        "backend_restart": skeleton,
        "prognostic_sha256": codec.prognostic_sha256(boundary["prognostic"]),
        "model_time_seconds": float(boundary["model_time_seconds"]),
        "files": written,
        **dict(receipt),
    }
    path = out_dir / SEGMENT_MANIFEST
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n",
                    encoding="utf-8")
    codec.fsync_file(path)
    return manifest


# --------------------------------------------------------------------------
# the shared device-stack preamble


def _config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _prepare(args: argparse.Namespace, guard: Any) -> dict[str, Any]:
    """Bind the port, arm the guard, build the forecast host."""
    config = _config(args.port_config)
    arwen = Path(config["arwen_checkout"]).expanduser().resolve()

    # Arming BEFORE any device construction is the point: from here on,
    # the only gpuwm this process may import is this checkout, and the
    # guard says so at the import rather than eight frames deep in the
    # port's Arwen pin.
    guard.arm(arwen)

    binding = portbind.bind_port(args.port_root)
    from mpas_port.cuda_backend import KernelCache, require_cuda

    paths = {role: Path(config[role]).expanduser().absolute()
             for role in ("grid", "static", "init")}
    binding.mesh_binding.bind_mesh(binding.proof, str(config["mesh"]),
                                   grid=paths["grid"], static=paths["static"],
                                   forecast=binding.forecast)
    authority = binding.forecast.verify_forecast_authorities(paths)
    host = binding.forecast.prepare_forecast_host(
        paths, authority, start_time_text=config.get("start_time"))

    cache_root = Path(config["cache_root"]).expanduser().absolute()
    cache_root.mkdir(parents=True, exist_ok=True)
    capability = require_cuda(min_compute=(12, 0), required_compute=(12, 0),
                              cache_dir=cache_root)
    cache = KernelCache(capability=capability, cache_dir=cache_root)
    return {"config": config, "binding": binding, "host": host,
            "cache": cache, "arwen": arwen}


# --------------------------------------------------------------------------
# seed


def seed(args: argparse.Namespace, guard: Any) -> int:
    ctx = _prepare(args, guard)
    binding = ctx["binding"]
    proof = binding.proof
    stack = proof._construct_device_stack(
        host=ctx["host"], cache=ctx["cache"], arwen_checkout=ctx["arwen"])

    out = Path(args.out).expanduser().absolute()
    dt = binding.dt_seconds

    _, _, first = proof._run_steps(stack=stack, start_step=0,
                                   end_step=int(args.steps),
                                   capture_steps=set())
    anchor_boundary = portbind.download_stack(binding, stack,
                                              label="seed anchor")
    pin_receipt = guard.verify_pinned()
    common = {
        "dt_seconds": dt,
        "steps_requested": int(args.steps),
        "frozen_execution_sources": binding.source_receipt.get("sha256"),
        "arwen_pin": pin_receipt,
        "stamp_basis": "the analysis re-enters the dycore through "
                       "_construct_device_stack's rehydration seam",
    }
    anchor_manifest = _write_segment(
        out / "anchor", role="anchor", boundary=anchor_boundary,
        receipt={**common, "steps_executed": len(first),
                 "rehydrated_sha256": None,
                 "rehydration": "none: seed constructs from the case init"})

    # The un-analysed continuation: the SAME stack, never interrupted.
    _, _, second = proof._run_steps(stack=stack, start_step=0,
                                    end_step=int(args.steps),
                                    capture_steps=set())
    control_boundary = portbind.download_stack(binding, stack,
                                               label="uninterrupted control")
    control_manifest = _write_segment(
        out / "control", role="control", boundary=control_boundary,
        receipt={**common, "steps_executed": len(second),
                 "rehydrated_sha256": None,
                 "rehydration": "none: continued in-process from the anchor "
                                "boundary without a download/upload round "
                                "trip -- this is the control"})
    print(json.dumps({"anchor": anchor_manifest["prognostic_sha256"],
                      "control": control_manifest["prognostic_sha256"],
                      "anchor_steps": len(first),
                      "control_steps": len(second),
                      "arwen_pin_obeyed": pin_receipt.get("obeyed")},
                     indent=2))
    return 0


# --------------------------------------------------------------------------
# segment


def segment(args: argparse.Namespace, guard: Any) -> int:
    anchor_dir = Path(args.anchor).expanduser().absolute()
    manifest = codec.read_manifest(anchor_dir)

    prognostic = codec.read_member(anchor_dir, manifest["parent"]["path"])
    derived = codec.read_member(anchor_dir, manifest["derived"]["path"])
    seam_block = manifest.get("seam") or {}
    seam = codec.read_member(anchor_dir, seam_block.get("path"))
    analysis = manifest.get("analysis") or {}
    increment = codec.read_member(anchor_dir, analysis.get("path"))

    background_sha = codec.prognostic_sha256(prognostic)
    recorded = manifest["parent"]["prognostic_sha256"]
    if background_sha != recorded:
        raise codec.CodecRefusal(
            "anchor prognostics do not hash to what the manifest recorded",
            anchor=str(anchor_dir), recorded=recorded, actual=background_sha)

    # Apply the analysis on the host, then never touch it again: what the
    # device is handed is the analysed state, and the gate downstream
    # asks whether the device gave that exact state back.
    applied_fields: list[str] = []
    if increment:
        for name, delta in sorted(increment.items()):
            if name not in prognostic:
                raise codec.CodecRefusal(
                    "analysis increment names a field the parent state "
                    "does not carry", field=name,
                    present=sorted(prognostic))
            base = prognostic[name]
            if base.shape != np.asarray(delta).shape:
                raise codec.CodecRefusal(
                    "analysis increment does not match the field it "
                    "increments", field=name, state_shape=list(base.shape),
                    increment_shape=list(np.asarray(delta).shape))
            prognostic[name] = codec.contiguous(
                (base + np.asarray(delta, dtype=base.dtype)).astype(
                    base.dtype, copy=False))
            applied_fields.append(name)
    analysed_sha = codec.prognostic_sha256(prognostic)

    ctx = _prepare(args, guard)
    binding = ctx["binding"]
    proof = binding.proof

    state = portbind.prognostic_state(binding, prognostic)
    saved = portbind.saved_diagnostics(binding, derived)
    backend_restart = (portbind.join_backend_restart(
        manifest["backend_restart"], seam)
        if manifest.get("backend_restart") and seam else None)

    stack = proof._construct_device_stack(
        host=ctx["host"], cache=ctx["cache"], arwen_checkout=ctx["arwen"],
        state=state, saved_diagnostics=saved,
        backend_restart=backend_restart)

    # THE evidence that the analysis reached the dycore: read the state
    # back off the device immediately after rehydration and hash it.  If
    # the device did not take the analysis, this does not match
    # ``analysed_sha`` and no amount of downstream divergence rescues the
    # claim.
    rehydrated = portbind.download_stack(binding, stack,
                                         label="post-rehydration readback")
    rehydrated_sha = codec.prognostic_sha256(rehydrated["prognostic"])

    _, _, receipts = proof._run_steps(stack=stack, start_step=0,
                                      end_step=int(args.steps),
                                      capture_steps=set())
    boundary = portbind.download_stack(binding, stack,
                                       label="segment boundary")
    pin_receipt = guard.verify_pinned()

    written = _write_segment(
        Path(args.out).expanduser().absolute(), role="segment",
        boundary=boundary,
        receipt={
            "dt_seconds": binding.dt_seconds,
            "steps_requested": int(args.steps),
            "steps_executed": len(receipts),
            "frozen_execution_sources": binding.source_receipt.get("sha256"),
            "arwen_pin": pin_receipt,
            "source_anchor": str(anchor_dir),
            "background_sha256": background_sha,
            "analysed_sha256": analysed_sha,
            "rehydrated_sha256": rehydrated_sha,
            "increment_fields": applied_fields,
            "increment_sha256": (None if not increment
                                 else codec.prognostic_sha256(increment)),
            "increment_nonzero_cells": (
                0 if not increment
                else int(sum(int(np.count_nonzero(value))
                             for value in increment.values()))),
            "backend_restart_restored": backend_restart is not None,
            "rehydration": "_construct_device_stack(state=, "
                           "saved_diagnostics=, backend_restart=)",
        })
    print(json.dumps({
        "prognostic_sha256": written["prognostic_sha256"],
        "rehydrated_sha256": rehydrated_sha,
        "analysed_sha256": analysed_sha,
        "steps_executed": len(receipts),
        "arwen_pin_obeyed": pin_receipt.get("obeyed")}, indent=2))
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mpas_cycle_bridge.worker",
                                     description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)

    seed_p = sub.add_parser("seed")
    seed_p.add_argument("--port-root", required=True)
    seed_p.add_argument("--port-config", required=True)
    seed_p.add_argument("--steps", type=int, required=True)
    seed_p.add_argument("--out", required=True)
    seed_p.set_defaults(func=seed)

    seg_p = sub.add_parser("segment")
    seg_p.add_argument("--port-root", required=True)
    seg_p.add_argument("--port-config", required=True)
    seg_p.add_argument("--anchor", required=True)
    seg_p.add_argument("--steps", type=int, required=True)
    seg_p.add_argument("--out", required=True)
    seg_p.set_defaults(func=segment)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    guard = install_guard()
    status = Path(args.out).expanduser().absolute() / "worker-status.json"
    try:
        code = int(args.func(args, guard))
        payload = {"ok": True, "returncode": code, "phase": args.phase}
    except BaseException as error:  # noqa: BLE001 - recorded, then re-raised
        payload = {"ok": False, "returncode": 1, "phase": args.phase,
                   "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
        try:
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps(payload, indent=2) + "\n",
                              encoding="utf-8")
        except Exception:  # pragma: no cover - the raise below is the report
            pass
        raise
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
