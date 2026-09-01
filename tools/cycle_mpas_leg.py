"""Run one leg of the cycle: anchor N -> analysis -> advance -> anchor N+1.

This is a STANDALONE process on purpose.  The MPAS GPU port lives in a
separate repository that pins gpuwm by SHA and re-verifies the checkout
before and after every run, so the cycling spine cannot be a library that
port imports.  What crosses the boundary is anchors on disk.  That
constraint is also what makes the cycle crash-recoverable and what lets a
Rust DA engine read the same artifacts.

The leg, in order:

1. read anchor N (committed only -- a half-written anchor is refused);
2. apply ``analysis_increment.nc`` to the prognostics, if one is carried;
3. **if the derived block is now stale, rebuild it or REFUSE** -- an
   increment rewrites ``rho_theta`` and the carried ``exner`` then
   describes the atmosphere the model had before the radar spoke;
4. hash the rehydrated state IN THE FORECAST PROCESS and put it through
   the three-hash ingestion gate;
5. advance the parent by one cycle;
6. publish anchor N+1 carrying the ingestion receipt and the residual.

Two backends:

``--backend replay --history GLOB``
    A LOUD STUB.  It reads a recorded history series and emits the next
    frame as if it had been integrated.  It stamps ``parent_kind:
    "replay"`` into the anchor and prints ``REPLAY BACKEND: this leg did
    not integrate a dycore`` on every single run.  It exists so the
    spine's plumbing can be exercised end to end on a box with no GPU
    and no port checkout; it is not a forecast and never claims to be.

``--backend mpas-cuda --port-root PATH``
    The real thing, and it runs OUT OF PROCESS.  The port pins its own
    gpuwm checkout by commit and refuses to replace a live gpuwm from
    another tree; ``gpuwm.cycle`` is gpuwm, so a leg that binds the port
    in-process hits that import wall on a real card no matter how green
    its tests are.  This backend therefore does exactly what the front
    door does: it spawns ``mpas_cycle_bridge.worker`` through
    :func:`gpuwm.cycle.mpas_bridge.launch` and talks to it only through
    anchors and segments on disk.  It was in-process until 2026-08-14
    and its tests passed the whole time, because a re-export resolves
    cleanly on a box with no port checkout and no card.

Exit codes: ``0`` published, ``2`` backend unavailable, ``3`` refused.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gpuwm.cycle.anchor import (anchor_for_cycle, latest_anchor,
                                prognostic_sha256, read_anchor, write_anchor)
from gpuwm.cycle import mpas_bridge
from gpuwm.cycle.consistency import (CONSISTENCY_THRESHOLD,
                                     UNIT_VERTICAL_METRIC, derived_is_stale,
                                     rebuild_exner, require_consistent)
from gpuwm.cycle.contracts import (CycleRefusal, seconds_to_ticks,
                                   ticks_to_seconds)
from gpuwm.cycle.ingestion import verify_ingestion
from gpuwm.cycle.mpas_port_adapter import (PORT_FORECAST_RELPATH,
                                           PORT_MESH_BINDING_RELPATH,
                                           PORT_PROOF_RELPATH,
                                           PORT_SRC_RELDIR,
                                           split_backend_restart)

REPLAY_BANNER = "REPLAY BACKEND: this leg did not integrate a dycore"

#: Keys ``--backend mpas-cuda`` needs to reach the port's forecast host.
#: They are a JSON file rather than nine flags because they travel
#: together and a leg that has some of them has none of them.
PORT_CONFIG_KEYS = ("mesh", "grid", "static", "init", "arwen_checkout",
                    "cache_root")

def _vertical_metric(derived: Mapping[str, np.ndarray]) -> Any:
    """The metric this anchor's exner was built on.

    MPAS forms exner as ``(zz*(rd/p0)*rho_theta) ** (rd/cv)``, so a state
    graded without ``zz`` reads ``|zz**(rd/cv) - 1|`` -- 0.4 on a real
    mesh, on a state that is correct.  An anchor that carries the metric
    is graded against it; one that does not is a synthetic unit-metric
    state and says so rather than defaulting silently.
    """
    zz = derived.get("zz")
    return UNIT_VERTICAL_METRIC if zz is None else zz


class ReplayBackend:
    """The loud stub: replays recorded frames instead of integrating.

    Its rebuild capability is deliberately narrow and honest.  Replay
    knows the equation of state, so it can rebuild ``exner`` from
    ``rho_theta``; it knows nothing about the port's discrete curl, so it
    cannot rebuild ``normal_velocity``, ``vertical_velocity`` or the
    perturbation family.  Handed a stale derived block containing any of
    those, it refuses rather than guessing -- which is the whole point of
    the stale-derived guard.
    """

    kind = "replay"
    rebuildable_fields = ("exner",)

    #: The metric the replayed series lives on.  Set by the leg runner
    #: from the anchor; a synthetic series is an explicit unit metric,
    #: never an assumed one.
    vertical_metric: Any = UNIT_VERTICAL_METRIC

    def __init__(self, history_glob: str) -> None:
        self.history_glob = history_glob
        self.frames = sorted(Path(p) for p in globlib.glob(history_glob))
        if not self.frames:
            raise CycleRefusal("replay backend found no history frames",
                               history_glob=history_glob,
                               remedy="point --history at the recorded "
                                      "series, e.g. 'hist/frame_*.npz'")
        print(REPLAY_BANNER)
        print(f"  history: {history_glob} ({len(self.frames)} frames)")

    # -- capability ------------------------------------------------------
    def rebuild_diagnostics(self, prognostic: Mapping[str, np.ndarray],
                            derived: Mapping[str, np.ndarray], *,
                            label: str) -> dict[str, np.ndarray]:
        unrebuildable = sorted(set(derived) - set(self.rebuildable_fields))
        if unrebuildable:
            raise CycleRefusal(
                "cannot resume on stale derived diagnostics",
                label=label, backend=self.kind,
                unrebuildable_fields=unrebuildable,
                rebuildable_fields=list(self.rebuildable_fields),
                remedy="run this leg under --backend mpas-cuda, whose "
                       "_rebuild_saved_diagnostics owns these fields; the "
                       "replay stub must not invent them")
        return {"exner": rebuild_exner(prognostic["rho_theta"],
                                       self.vertical_metric)}

    def rehydrated_sha256(self, prognostic: Mapping[str, np.ndarray]) -> str:
        # The forecast process hashes what it actually loaded, which is
        # the only hash the ingestion gate can trust.
        loaded = {name: np.array(array, copy=True)
                  for name, array in prognostic.items()}
        self._loaded = loaded
        return prognostic_sha256(loaded)

    # -- advance ---------------------------------------------------------
    def advance(self, prognostic: Mapping[str, np.ndarray],
                derived: Mapping[str, np.ndarray], *, target_ticks: int,
                label: str) -> tuple[dict[str, np.ndarray],
                                     dict[str, np.ndarray]]:
        target_seconds = ticks_to_seconds(target_ticks)
        wanted = set(prognostic)
        for frame in self.frames:
            with np.load(frame) as handle:
                fields = {name: np.asarray(handle[name])
                          for name in handle.files}
            time_seconds = float(np.asarray(
                fields.get("time_seconds", np.nan)).reshape(-1)[0])
            if abs(time_seconds - target_seconds) > 0.5e-3:
                continue
            out_prognostic = {name: fields[name] for name in wanted
                              if name in fields}
            missing = sorted(wanted - set(out_prognostic))
            if missing:
                raise CycleRefusal(
                    "replay frame does not carry every prognostic field",
                    label=label, frame=str(frame), missing=missing)
            out_derived = {name: fields[name] for name in fields
                           if name not in wanted}
            print(f"  replayed frame {frame.name} at t={time_seconds:.3f}s")
            return out_prognostic, out_derived
        raise CycleRefusal(
            "replay history has no frame at the target time",
            label=label, target_seconds=target_seconds,
            frames=[f.name for f in self.frames],
            remedy="record the series at the cycle length you are asking "
                   "for, or pass a --cycle-seconds that matches it")


def _bridge_leg(*, root: Path, source: Path, doc: Any, label: str,
                port_root: Path, port_config: Mapping[str, Any] | None,
                port_config_path: Path | None, cycle_seconds: float,
                threshold: float, steps: int | None,
                timeout: float | None) -> Path:
    """One real-dycore leg, run OUT OF PROCESS through the bridge.

    The port pins its own gpuwm checkout and installs an import guard
    that refuses a live gpuwm from any other tree.  ``gpuwm.cycle`` IS
    gpuwm, so there is no in-process arrangement of this leg that works
    on a real card -- the previous shape of this function bound the port
    directly and its tests passed only because a re-export resolves
    cleanly where no port and no card exist.  Everything the forecast
    process runs now lives in ``mpas_cycle_bridge``, which has no gpuwm
    imports in it at all, and this function is the leg runner's view of
    the same route the front door takes.

    The bridge worker owns the whole closed loop for a segment: it reads
    the anchor, applies the increment the anchor carries, rehydrates the
    device through the port's own state seam, hashes what the DEVICE gave
    back, integrates, and reports.  The grading -- consistency, the
    three-hash ingestion gate, the parent-kind stamp -- stays on this
    side, because the process that produced the evidence must not be the
    one that grades it.
    """
    # "the port is not here" stays exit 2 (backend unavailable) rather
    # than exit 3 (refused): an absent checkout is an environment fact,
    # not a judgement about the run.  The bridge cannot make that
    # distinction for us because it only ever sees a subprocess failure.
    looked_in = [str(port_root / relative) for relative in
                 (PORT_SRC_RELDIR, PORT_PROOF_RELPATH, PORT_FORECAST_RELPATH,
                  PORT_MESH_BINDING_RELPATH)]
    absent = [candidate for candidate in looked_in
              if not Path(candidate).exists()]
    if absent:
        raise BackendUnavailable(
            "mpas_port (via mpas_cycle_bridge.worker)", looked_in,
            f"missing from the port root: {sorted(absent)}")
    if port_config_path is None:
        raise CycleRefusal(
            "the bridge worker takes the port case configuration as a FILE, "
            "not a parsed mapping; it runs in another process",
            label=label,
            remedy="pass --port-config PATH")
    manifest = doc.manifest
    dt = float(manifest.get("integration", {}).get("dt_seconds") or 0.0)
    if steps is None:
        if dt <= 0.0:
            raise CycleRefusal(
                "this leg cannot infer how many dycore steps a cycle is",
                label=label, cycle_seconds=float(cycle_seconds),
                remedy="pass --port-steps N; the clock is never rounded to "
                       "fit and the anchor carries no dt to divide by")
        exact = float(cycle_seconds) / dt
        if abs(exact - round(exact)) > 1e-9 or round(exact) <= 0:
            raise CycleRefusal(
                "cycle length is not a whole number of dycore steps",
                label=label, cycle_seconds=float(cycle_seconds),
                dt_seconds=dt, steps=exact)
        steps = int(round(exact))

    out = Path(root) / "bridge" / f"leg-{int(manifest['cycle_index']) + 1:03d}"
    segment = mpas_bridge.launch(
        phase="segment", port_root=str(port_root),
        port_config=str(port_config_path), steps=int(steps),
        anchor=str(source), out=out, timeout=timeout)
    arrays = mpas_bridge.segment_arrays(segment)
    stamp = mpas_bridge.stamp_for_segment(segment, steps_requested=int(steps))

    prognostic = arrays["prognostic"]
    derived = arrays["derived"]
    zz = np.asarray(arrays["metric"]["zz"])

    residual = require_consistent(prognostic, derived, vertical_metric=zz,
                                  threshold=threshold, label=label)
    print(f"  hydrostatic residual {residual['max_relative_residual']:.3e} "
          f"(floor {residual['resolution_floor']:.3e}, "
          f"threshold {threshold:.1e}, metric "
          f"{residual['vertical_metric']})")

    increment = doc.increment() if doc.has_increment() else {}
    ingestion = verify_ingestion(
        background_sha256=segment["background_sha256"],
        increment=increment,
        analysis_sha256=segment["rehydrated_sha256"], label=label)
    print(f"  ingestion {ingestion['state']}: "
          f"{ingestion['increment_nonzero_cells']} nonzero cells over "
          f"{ingestion['fields']}")
    print(f"  integrated {segment.get('steps_executed')} real dycore steps "
          f"to t={segment.get('model_time_seconds')}s -> "
          f"stamped {stamp['parent_kind']}")

    cycle_ticks = seconds_to_ticks(cycle_seconds, label="cycle_seconds")
    target_ticks = int(manifest["anchor_ticks"]) + cycle_ticks
    published = write_anchor(
        root, cycle_index=int(manifest["cycle_index"]) + 1,
        anchor_ticks=target_ticks,
        valid_time=_advance_valid_time(manifest["valid_time"], cycle_seconds),
        parent_kind=stamp["parent_kind"], prognostic=prognostic,
        derived=derived, seam=arrays["seam"],
        mesh_id=manifest["parent"]["mesh_id"],
        analysis={"state": ingestion["state"], "ingestion": ingestion},
        diagnostics_rebuilt=False,
        extra={"consistency": residual, "resumed_from": source.name,
               "stamp": stamp, "requested_parent_kind": "mpas-cuda",
               "backend_restart": segment.get("backend_restart"),
               "integration": {
                   "steps_executed": segment.get("steps_executed"),
                   "steps_requested": int(steps),
                   "dt_seconds": segment.get("dt_seconds"),
                   "model_time_seconds": segment.get("model_time_seconds")}})
    print(f"  published {published.name}")
    return published


class BackendUnavailable(RuntimeError):
    """The backend could not be reached, and says exactly where we looked."""

    def __init__(self, module_name: str, looked_in: list[str],
                 detail: str) -> None:
        super().__init__(
            f"backend module {module_name!r} is not importable; looked in "
            f"{looked_in}; {detail}")
        self.module_name = module_name
        self.looked_in = looked_in


def _apply_increment(prognostic: dict[str, np.ndarray],
                     increment: Mapping[str, np.ndarray], *,
                     label: str) -> dict[str, np.ndarray]:
    unknown = sorted(set(increment) - set(prognostic))
    if unknown:
        raise CycleRefusal("analysis increment names fields the state does "
                           "not carry", label=label, unknown=unknown,
                           state_fields=sorted(prognostic))
    applied = {name: np.array(array, copy=True)
               for name, array in prognostic.items()}
    for name, delta in increment.items():
        applied[name] = applied[name] + np.asarray(delta)
    return applied


def run_leg(*, root: str | Path, backend_name: str, cycle_seconds: float,
            history: str | None = None, port_root: str | Path | None = None,
            port_config: Mapping[str, Any] | None = None,
            port_config_path: str | Path | None = None,
            port_steps: int | None = None,
            port_timeout: float | None = None,
            cycle_index: int | None = None,
            threshold: float = CONSISTENCY_THRESHOLD) -> Path:
    """Advance one cycle and publish the next anchor.  Returns its path."""
    root = Path(root)
    source = (anchor_for_cycle(root, cycle_index) if cycle_index is not None
              else latest_anchor(root))
    if source is None:
        raise CycleRefusal("no committed anchor to resume from",
                           root=str(root), cycle_index=cycle_index,
                           remedy="seed the run with an anchor written by "
                                  "the initialiser before running a leg")
    doc = read_anchor(source)
    manifest = doc.manifest
    label = (f"cycle={manifest['cycle_index']} "
             f"valid={manifest['valid_time']} anchor={source.name}")

    if backend_name == "mpas-cuda":
        # OUT OF PROCESS, through the same bridge the front door uses.
        # There is no in-process arrangement of this that survives the
        # port's Arwen import guard on a real card.
        return _bridge_leg(
            root=root, source=source, doc=doc, label=label,
            port_root=Path(port_root or root), port_config=port_config,
            port_config_path=(Path(port_config_path)
                              if port_config_path else None),
            cycle_seconds=cycle_seconds, threshold=threshold,
            steps=port_steps, timeout=port_timeout)
    if backend_name != "replay":
        raise CycleRefusal("unknown backend", backend=backend_name,
                           known=["replay", "mpas-cuda"])
    backend: Any = ReplayBackend(history or "")

    prognostic = doc.prognostic()
    derived = doc.derived()
    backend.vertical_metric = _vertical_metric(derived)
    background_sha256 = manifest["parent"]["prognostic_sha256"]

    increment = doc.increment() if doc.has_increment() else {}
    applied = _apply_increment(prognostic, increment, label=label)

    # The stale test is taken against the state as it will be integrated,
    # not as it was stored: applying an increment is exactly what makes a
    # carried derived block describe the wrong atmosphere.
    post = {"parent": {"prognostic_sha256": prognostic_sha256(applied)},
            "derived": {"derived_from_sha256":
                        manifest["derived"]["derived_from_sha256"]}}
    diagnostics_rebuilt = False
    if derived_is_stale(post):
        derived = backend.rebuild_diagnostics(applied, derived, label=label)
        diagnostics_rebuilt = True
    elif hasattr(backend, "prepare_stack"):
        # No increment moved the state, so nothing is stale -- but the
        # closed-loop backend still has to rehydrate the device from the
        # anchor before it can hash what it loaded.
        backend.prepare_stack(applied, derived)

    residual = require_consistent(applied, derived, threshold=threshold,
                                  vertical_metric=backend.vertical_metric,
                                  label=label)
    print(f"  hydrostatic residual {residual['max_relative_residual']:.3e} "
          f"(floor {residual['resolution_floor']:.3e}, "
          f"threshold {threshold:.1e}, metric "
          f"{residual['vertical_metric']})")

    analysis_sha256 = backend.rehydrated_sha256(applied)
    ingestion = verify_ingestion(background_sha256=background_sha256,
                                 increment=increment,
                                 analysis_sha256=analysis_sha256,
                                 label=label)
    print(f"  ingestion {ingestion['state']}: "
          f"{ingestion['increment_nonzero_cells']} nonzero cells over "
          f"{ingestion['fields']}")

    cycle_ticks = seconds_to_ticks(cycle_seconds, label="cycle_seconds")
    target_ticks = int(manifest["anchor_ticks"]) + cycle_ticks
    next_prognostic, next_derived = backend.advance(
        applied, derived, target_ticks=target_ticks, label=label)

    valid_time = _advance_valid_time(manifest["valid_time"], cycle_seconds)

    # The physics seam's restart payload is what makes the NEXT leg a
    # resumption rather than a cold start: without it the backend's clock
    # disagrees with the driver's and the port refuses to construct.  Its
    # arrays go in the anchor's seam slot; the rest is JSON in the
    # manifest.  Nothing is pickled across the boundary.
    seam_arrays: dict[str, np.ndarray] | None = None
    extra: dict[str, Any] = {"consistency": residual,
                             "resumed_from": source.name}
    published_restart = getattr(backend, "published_backend_restart", None)
    if published_restart is not None:
        skeleton, seam_arrays = split_backend_restart(published_restart)
        extra["backend_restart"] = skeleton
    if getattr(backend, "step_receipts", None):
        extra["integration"] = {
            "steps_executed": len(backend.step_receipts),
            "dt_seconds": backend.binding.dt_seconds,
            "boundary_fingerprints": backend.boundary_fingerprints,
        }

    published = write_anchor(
        root, cycle_index=int(manifest["cycle_index"]) + 1,
        anchor_ticks=target_ticks, valid_time=valid_time,
        parent_kind=backend.kind, prognostic=next_prognostic,
        derived=next_derived, seam=seam_arrays,
        mesh_id=manifest["parent"]["mesh_id"],
        analysis={"state": ingestion["state"], "ingestion": ingestion},
        diagnostics_rebuilt=diagnostics_rebuilt, extra=extra)
    print(f"  published {published.name}")
    return published


def _advance_valid_time(valid_time: str, cycle_seconds: float) -> str:
    from datetime import datetime, timedelta, timezone
    stamp = datetime.fromisoformat(str(valid_time).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (stamp + timedelta(seconds=float(cycle_seconds))).isoformat() \
        .replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cycle_mpas_leg",
        description="Advance one cycle: anchor N -> analysis -> anchor N+1.")
    parser.add_argument("--root", required=True,
                        help="run root holding anchors/")
    parser.add_argument("--backend", default="replay",
                        choices=["replay", "mpas-cuda"])
    parser.add_argument("--history", default=None,
                        help="glob of recorded frames for --backend replay")
    parser.add_argument("--port-root", default=None,
                        help="checkout of the MPAS GPU port")
    parser.add_argument("--port-config", default=None,
                        help="JSON file of the port case configuration "
                             f"({', '.join(PORT_CONFIG_KEYS)})")
    parser.add_argument("--port-steps", type=int, default=None,
                        help="dycore steps this leg asks the bridge for; "
                             "required unless the anchor records dt")
    parser.add_argument("--port-timeout", type=float, default=None,
                        help="seconds to allow the forecast worker")
    parser.add_argument("--cycle-seconds", type=float, required=True)
    parser.add_argument("--cycle-index", type=int, default=None,
                        help="resume from this anchor instead of the latest")
    parser.add_argument("--consistency-threshold", type=float,
                        default=CONSISTENCY_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = (json.loads(Path(args.port_config).read_text(encoding="utf-8"))
                  if args.port_config else None)
        run_leg(root=args.root, backend_name=args.backend,
                history=args.history, port_root=args.port_root,
                port_config=config, port_config_path=args.port_config,
                port_steps=args.port_steps, port_timeout=args.port_timeout,
                cycle_seconds=args.cycle_seconds,
                cycle_index=args.cycle_index,
                threshold=args.consistency_threshold)
    except BackendUnavailable as error:
        print(f"BACKEND UNAVAILABLE: {error}", file=sys.stderr)
        return 2
    except CycleRefusal as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
