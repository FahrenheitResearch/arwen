"""The integration seam: bind the spine's injected callables to real modules.

Three lanes built the cycle in parallel against a shared contracts module.
Two seams did not meet, and this module is where they are joined:

1. ``gpuwm.cycle.cli`` imports ``build_placement_provider(kind=...,
   child_slots=..., allow_clamp=...)`` from ``gpuwm.cycle.placement`` and
   refuses by name if it is absent.  The placement lane shipped the three
   provider CLASSES and a keyword-only ``plan_children`` that takes a
   ``SlotPool``, but never that factory.  Nothing was wrong with either
   half; they simply never met.

2. The spine's ``advance_parent`` returned ticks and nothing else, so the
   CLI cycled a clock over an empty root: no anchor was written, the
   three-hash ingestion gate never ran and the hydrostatic instrument
   never fired.  Every one of those exists in the anchor lane.  The
   parent engine here is what makes a cycle actually land on disk.

The adapters keep the supervisor's contract exactly as the spine lane
documented it: ``advance_parent(cycle_index, anchor_in) -> {...}`` with
``parent_ticks``/``anchor_ticks`` integers, and ``plan_children(
cycle_index, parent_record) -> [records]`` carrying geographic keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gpuwm.cycle.anchor import (latest_anchor, prognostic_sha256, read_anchor,
                                write_anchor)
from gpuwm.cycle.children import ChildSlot, SlotPool
from gpuwm.cycle.children import plan_children as plan_children_at_boundary
from gpuwm.cycle.consistency import (CONSISTENCY_THRESHOLD,
                                     UNIT_VERTICAL_METRIC, derived_is_stale,
                                     hydrostatic_residual, rebuild_exner)
from gpuwm.cycle.contracts import CycleRefusal
from gpuwm.cycle.ingestion import verify_ingestion
from gpuwm.cycle.placement import (ObsPlacementProvider,
                                   SchedulePlacementProvider,
                                   TrackerPlacementProvider,
                                   parent_geometry_from_fields, rank_and_assign)

#: A child that no longer has this much signal under it is retired and its
#: reservation returns to the pool.  Named, not inlined, because it is the
#: number that decides a despawn.
DEFAULT_RETIRE_BELOW_DBZ = 35.0


#: The observation plane the obs provider places on when a caller names
#: none.  ``gpuwm-obs.radar-grid.v1`` ships ``z_obs``, ``z_max`` and
#: ``z_mean`` side by side and states the choice between them is the
#: CONSUMER's, so this is a configured default and not a fixed field: it
#: is the file's own observation variable, the one every DA lane reads and
#: the one the real three-volume KTLX series was placed against.  It was
#: ``z_max`` here and ``z_obs`` in every producer -- a name that only ever
#: agreed with hand-built test data.
DEFAULT_OBS_PLACEMENT_FIELD = "z_obs"

#: Likewise for the parent plane the tracker places on.
DEFAULT_TRACKER_PLACEMENT_FIELD = "composite_reflectivity"


def build_placement_provider(*, kind: str, child_slots: int,
                             allow_clamp: bool = False,
                             parent_geometry=None,
                             signal_for: Callable | None = None,
                             radar_grid_path: Any = None,
                             radar_grid_paths: Sequence[Any] = (),
                             field: str | None = None,
                             reader: Callable | None = None,
                             schedule: Sequence[Mapping] = (),
                             child_nx: int = 199, child_ny: int = 199,
                             child_dx_m: float = 1000.0,
                             child_dt_seconds: float = 5.0,
                             parent_grid_ratio: int = 3,
                             threshold: float = 40.0,
                             min_separation_km: float = 40.0,
                             retire_below_strength: float
                             = DEFAULT_RETIRE_BELOW_DBZ,
                             state: dict | None = None):
    """The symbol the CLI imports.  Returns a ``plan_children`` callable.

    ``kind`` is one of ``tracker``, ``schedule`` or ``obs``.  The returned
    callable owns the slot pool across cycles, which is what makes a
    retirement give its reservation back rather than leaking it.

    ``radar_grid_paths`` is ONE FILE PER CYCLE, in order, and a single
    entry is reused for every cycle.  A cycling run whose observations
    never change cannot exercise re-placement, retirement or slot reclaim
    -- which is precisely how those four behaviours stayed unexercised
    while every unit test around them passed.
    """

    kinds = ("tracker", "schedule", "obs")
    if kind not in kinds:
        raise CycleRefusal("unknown placement provider",
                           placement_provider=str(kind), known=list(kinds))

    pool = SlotPool([ChildSlot(grid_id=2 + n, nx=int(child_nx),
                               ny=int(child_ny), dx_m=float(child_dx_m),
                               dt_seconds=float(child_dt_seconds),
                               parent_grid_ratio=int(parent_grid_ratio))
                     for n in range(int(child_slots))])

    common = dict(min_separation_km=float(min_separation_km),
                  max_children=int(child_slots), child_nx=int(child_nx),
                  child_ny=int(child_ny), child_dx_m=float(child_dx_m))
    volumes = [item for item in (radar_grid_paths or ()) if item is not None]
    if not volumes and radar_grid_path is not None:
        volumes = [radar_grid_path]

    def _obs_provider(path):
        return ObsPlacementProvider(
            radar_grid_path=path,
            field=str(field or DEFAULT_OBS_PLACEMENT_FIELD),
            threshold_dbz=float(threshold), reader=reader, **common)

    if kind == "tracker":
        provider = TrackerPlacementProvider(
            field=str(field or DEFAULT_TRACKER_PLACEMENT_FIELD),
            threshold=float(threshold), **common)
    elif kind == "obs":
        if not volumes and reader is None:
            raise CycleRefusal(
                "the obs placement provider needs a radar-grid file",
                placement_provider="obs",
                remedy="pass --placement-obs-file PATH (repeatable, one per "
                       "cycle in order), or inject a reader")
        provider = _obs_provider(volumes[0] if volumes else None)
    else:
        provider = SchedulePlacementProvider(entries=tuple(schedule), **common)

    carried = state if state is not None else {}
    carried.setdefault("previous", [])

    def plan(cycle_index: int, parent_record: Mapping) -> list[dict]:
        geometry = parent_geometry
        if geometry is None:
            raise CycleRefusal(
                "placement needs the parent's geometry",
                cycle_index=int(cycle_index),
                remedy="pass parent_geometry= when building the provider; "
                       "at the front door that is --parent-geo-file (or the "
                       "first --placement-obs-file) with --parent-dx-m")
        signal = None if signal_for is None else signal_for(cycle_index,
                                                            parent_record)
        # Each cycle places against ITS OWN volume when a series was given.
        this_cycle = provider
        if kind == "obs" and len(volumes) > 1:
            this_cycle = _obs_provider(
                volumes[min(int(cycle_index) - 1, len(volumes) - 1)])
        requests = this_cycle(cycle_index=int(cycle_index),
                              valid_time=parent_record.get("valid_time"),
                              parent_geometry=geometry, signal=signal)
        records = plan_children_at_boundary(
            cycle_index=int(cycle_index), pool=pool,
            requests=list(requests),
            previous_children=carried["previous"],
            retire_below_strength=float(retire_below_strength),
            min_separation_km=float(min_separation_km),
            parent_geometry=geometry, allow_clamp=bool(allow_clamp))
        carried["previous"] = [dict(item) for item in records
                               if item.get("state") in ("LIVE", "PLANNED")]
        carried["pool"] = pool
        return records

    plan.pool = pool
    plan.provider = provider
    return plan


def build_replay_parent_engine(*, root, clock, history_frames,
                               mesh_id: str,
                               increment_for: Callable | None = None,
                               parent_kind: str = "replay",
                               banner: bool = True):
    """An ``advance_parent`` that publishes a real anchor every cycle.

    This is the leg of ``tools/cycle_mpas_leg.py`` expressed as the
    callable the supervisor injects, so the CLI's cycle and the standalone
    leg runner exercise the same anchor writer, the same ingestion gate
    and the same consistency instrument.

    LOUD STUB, unchanged from the leg runner's contract: with
    ``parent_kind='replay'`` this does NOT integrate a dycore.  It replays
    a recorded series and says so on every leg, and the anchor carries
    ``parent_kind: 'replay'`` so no downstream reader can mistake it.
    """

    root = Path(root)
    frames = list(history_frames)

    def _metric(frame):
        """The vertical metric this replayed series lives on.

        A recorded series that carries ``zz`` is graded against it; a
        synthetic series built on a unit metric says so explicitly.  The
        instrument has no default, which is the point: the omission it
        used to make silently is now a thing a caller has to state.
        """
        zz = frame.get("vertical_metric")
        return UNIT_VERTICAL_METRIC if zz is None else zz

    def advance(cycle_index: int, anchor_in) -> dict:
        if banner and parent_kind == "replay":
            print("REPLAY BACKEND: this leg did not integrate a dycore")
        boundary = clock.boundary_ticks(int(cycle_index))
        metric = _metric(frames[min(int(cycle_index), len(frames) - 1)])

        previous = latest_anchor(root)
        if previous is None:
            frame = frames[0]
            prognostic = {k: np.array(v, copy=True)
                          for k, v in frame["prognostic"].items()}
            derived = {k: np.array(v, copy=True)
                       for k, v in frame["derived"].items()}
            background_sha = prognostic_sha256(prognostic)
            ingestion = None
            rebuilt = None
        else:
            document = read_anchor(previous)
            prognostic = {k: np.array(v, copy=True)
                          for k, v in _mapping(document.prognostic).items()}
            derived = {k: np.array(v, copy=True)
                       for k, v in _mapping(document.derived).items()}
            background_sha = prognostic_sha256(prognostic)

            try:
                increment = _mapping(document.increment)
            except Exception:
                increment = {}
            for name, delta in increment.items():
                if name in prognostic:
                    prognostic[name] = prognostic[name] + np.asarray(delta)

            rebuilt = False
            stale = bool(increment) and prognostic_sha256(
                prognostic) != (document.manifest.get("parent", {})
                                or {}).get("prognostic_sha256")
            if stale:
                derived = _rebuild_derived(prognostic, derived,
                                           vertical_metric=metric)
                rebuilt = True

            analysis_sha = prognostic_sha256(prognostic)
            ingestion = verify_ingestion(
                background_sha256=background_sha, increment=increment,
                analysis_sha256=analysis_sha,
                label=f"cycle={cycle_index} ticks={boundary}")

            index = min(int(cycle_index), len(frames) - 1)
            frame = frames[index]
            for name, array in frame["prognostic"].items():
                if name in prognostic:
                    prognostic[name] = prognostic[name] + (
                        np.asarray(array) - np.asarray(
                            frames[max(index - 1, 0)]["prognostic"][name]))
            derived = _rebuild_derived(prognostic, derived,
                                       vertical_metric=metric)

        residual = hydrostatic_residual(prognostic, derived,
                                        vertical_metric=metric)
        worst = float(residual["max_relative_residual"])
        if worst > CONSISTENCY_THRESHOLD:
            raise CycleRefusal(
                "parent state and its carried diagnostics do not agree",
                cycle_index=int(cycle_index), residual=worst,
                resolution_floor=float(residual["resolution_floor"]),
                vertical_metric=residual["vertical_metric"],
                argmax_index=residual["argmax_index"],
                threshold=float(CONSISTENCY_THRESHOLD))

        prognostic["time_seconds"] = np.asarray(
            boundary / 1000.0, dtype=np.float64)

        analysis_block = None
        if ingestion is not None:
            analysis_block = {"state": ingestion["state"],
                              "ingestion": ingestion}
        next_increment = (None if increment_for is None
                          else increment_for(int(cycle_index), prognostic))
        if next_increment:
            analysis_block = dict(analysis_block or {})
            analysis_block["arrays"] = next_increment
            analysis_block.setdefault("state", "PENDING")

        path = write_anchor(
            root, cycle_index=int(cycle_index), anchor_ticks=int(boundary),
            valid_time=clock.valid_time(int(cycle_index)),
            parent_kind=parent_kind, prognostic=prognostic, derived=derived,
            mesh_id=str(mesh_id), analysis=analysis_block,
            diagnostics_rebuilt=rebuilt)

        return {"kind": parent_kind, "parent_ticks": int(boundary),
                "anchor_ticks": int(boundary), "anchor_path": str(path),
                "valid_time": clock.valid_time(int(cycle_index)),
                # The planes a tracker placement provider reads.  Carried
                # on the record rather than re-read off the anchor: the
                # planner runs at the same boundary that produced them,
                # and a second read is a second chance to disagree.
                "derived_arrays": derived,
                "prognostic_arrays": prognostic,
                "hydrostatic_residual": worst,
                "residual_block": residual,
                "diagnostics_rebuilt": rebuilt,
                "background_sha256": background_sha,
                "ingestion": ingestion,
                "replay_stub": parent_kind == "replay"}

    return advance


def _mapping(value):
    """Anchor members are loader METHODS, not attributes.  Accept both."""

    return dict(value() if callable(value) else (value or {}))


def _rebuild_derived(prognostic: Mapping[str, np.ndarray],
                     previous: Mapping[str, np.ndarray], *,
                     vertical_metric: Any) -> dict[str, np.ndarray]:
    """Recompute exner from rho_theta, the way the port's rebuild does.

    The stale-diagnostics hazard is closed by rebuilding here rather than
    by carrying the pre-analysis exner forward under a new hash.

    This function used to carry its OWN transcription of the equation of
    state, and that transcription omitted the vertical metric ``zz`` in
    exactly the way the grading instrument did.  Two independent copies
    of one equation, wrong the same way, is a rebuild that writes a bad
    exner into the anchor and then grades it as good.  There is now one
    equation -- :func:`gpuwm.cycle.consistency.rebuild_exner` -- and both
    the rebuild and the grade call it.
    """

    out = {k: np.array(v, copy=True) for k, v in previous.items()}
    if "rho_theta" in prognostic and "exner" in out:
        stored = np.asarray(out["exner"]).dtype
        rebuilt = rebuild_exner(prognostic["rho_theta"], vertical_metric)
        out["exner"] = rebuilt.astype(stored) if stored.kind == "f" else rebuilt
    return out


# --------------------------------------------------------------------------
# seam 3: the product door and the closed loop, joined
#
# The front-door lane wrote a refusal that named a symbol for the model
# lane to land -- ``gpuwm.cycle.engine.build_model_parent_engine`` -- and
# the model lane landed the CAPABILITY (an out-of-process bridge to the
# port's device stack, in ``gpuwm.cycle.mpas_bridge``) under a different
# name in a different module.  Both halves were correct, the five-way
# merge was textually clean, and ``gpuwm cycle --parent-kind mpas-cuda``
# still refused with "this tree has no engine adapter", because nothing
# ever defined the name the door looked up.  This is that name.


def build_model_parent_engine(*, root, clock, parent_kind: str, mesh_id: str,
                              port_root, port_config, port_steps: int,
                              history_frames=None,
                              increment_for: Callable | None = None,
                              timeout: float | None = None):
    """An ``advance_parent`` that runs the REAL dycore, out of process.

    Every leg spawns ``mpas_cycle_bridge.worker`` in a fresh interpreter
    and talks to it only through anchors and segments on disk.  That is
    not a style choice: the port pins its own gpuwm checkout and installs
    an import guard, so a spine that constructs the port IN-PROCESS hits
    an import wall on a real card.  :mod:`gpuwm.cycle.mpas_bridge` is the
    only supported way in, and this adapter is the door's route to it.

    THE STAMP IS EARNED, NOT REQUESTED.  ``parent_kind`` here says which
    engine the user ASKED for; what lands on the anchor is whatever
    :func:`gpuwm.cycle.mpas_bridge.stamp_for_segment` grades the segment
    as.  A leg that asked for ``mpas-cuda`` but cannot show step
    receipts, rehydration through the port's state seam, and a device
    readback hash equal to the analysed state is stamped
    ``mpas-cuda-frames`` with its gaps listed on the receipt.  Asking
    louder does not upgrade the evidence.

    ``history_frames`` is accepted and IGNORED: this engine's state comes
    off the device, not out of a recorded series.  It is in the signature
    only so the door can call model and replay kinds identically.
    """
    from gpuwm.cycle import mpas_bridge

    root = Path(root)
    port_steps = int(port_steps)
    if port_steps <= 0:
        raise CycleRefusal(
            "a forecast segment of no steps is not a forecast",
            port_steps=port_steps,
            remedy="pass --port-steps N with N >= 1")

    work = root / "bridge"

    def _publish(cycle_index: int, segment: Mapping[str, Any],
                 *, ingestion, background_sha: str, analysis_state: str):
        stamp = mpas_bridge.stamp_for_segment(segment,
                                              steps_requested=port_steps)
        arrays = mpas_bridge.segment_arrays(segment)
        prognostic = arrays["prognostic"]
        derived = arrays["derived"]
        boundary = clock.boundary_ticks(int(cycle_index))

        # The same consistency instrument the replay engine runs.  Real
        # device output is exactly where a stale-diagnostics bug would do
        # the most damage, so it is not skipped for being "real".
        #
        # It is graded against the metric the DEVICE used, carried out on
        # the segment beside the state, rather than an assumed unit one.
        # That assumption is what halted this engine at cycle 1 with a
        # 0.407 residual on a state three other instruments called
        # correct: MPAS forms exner as (zz*(rd/p0)*rho_theta)**(rd/cv),
        # and on x1.40962 zz spans 0.83..2.35.
        zz = np.asarray(arrays["metric"]["zz"])
        if zz.shape != np.asarray(prognostic["rho_theta"]).shape:
            raise CycleRefusal(
                "the segment's vertical metric does not have the shape of "
                "its own rho_theta; this boundary cannot be graded",
                cycle_index=int(cycle_index), zz_shape=tuple(zz.shape),
                rho_theta_shape=tuple(
                    np.asarray(prognostic["rho_theta"]).shape))
        residual = hydrostatic_residual(prognostic, derived,
                                        vertical_metric=zz)
        worst = float(residual["max_relative_residual"])
        if worst > CONSISTENCY_THRESHOLD:
            raise CycleRefusal(
                "parent state and its carried diagnostics do not agree",
                cycle_index=int(cycle_index), residual=worst,
                resolution_floor=float(residual["resolution_floor"]),
                threshold=float(CONSISTENCY_THRESHOLD),
                vertical_metric=residual["vertical_metric"],
                argmax_index=residual["argmax_index"],
                parent_kind=stamp["parent_kind"])

        analysis_block: dict[str, Any] | None = None
        if ingestion is not None:
            analysis_block = {"state": analysis_state, "ingestion": ingestion}
        staged = (None if increment_for is None
                  else increment_for(int(cycle_index), prognostic))
        if staged:
            analysis_block = dict(analysis_block or {})
            analysis_block["arrays"] = staged
            analysis_block.setdefault("state", "PENDING")

        path = write_anchor(
            root, cycle_index=int(cycle_index), anchor_ticks=int(boundary),
            valid_time=clock.valid_time(int(cycle_index)),
            parent_kind=stamp["parent_kind"], prognostic=prognostic,
            derived=derived, seam=arrays["seam"], mesh_id=str(mesh_id),
            analysis=analysis_block,
            extra={"backend_restart": segment.get("backend_restart"),
                   "stamp": stamp,
                   "requested_parent_kind": str(parent_kind),
                   "integration": {
                       "steps_executed": segment.get("steps_executed"),
                       "steps_requested": port_steps,
                       "dt_seconds": segment.get("dt_seconds"),
                       "model_time_seconds": segment.get(
                           "model_time_seconds")}})

        return {"kind": stamp["parent_kind"], "parent_ticks": int(boundary),
                "anchor_ticks": int(boundary), "anchor_path": str(path),
                "valid_time": clock.valid_time(int(cycle_index)),
                "derived_arrays": derived, "prognostic_arrays": prognostic,
                "hydrostatic_residual": worst, "residual_block": residual,
                "diagnostics_rebuilt": False,
                "background_sha256": background_sha,
                "ingestion": ingestion,
                "stamp": stamp,
                "requested_parent_kind": str(parent_kind),
                # This engine integrated a dycore.  The replay stub flag
                # is what downstream readers use to tell the two apart,
                # and it is False here BECAUSE the bridge ran, not
                # because the kind string looks impressive.
                "replay_stub": False}

    def advance(cycle_index: int, anchor_in) -> dict:
        out = work / f"cycle-{int(cycle_index):03d}"
        previous = latest_anchor(root)

        if previous is None:
            # Cycle 1 has no analysed state to re-enter, so it cannot
            # earn the closed stamp and does not ask for it: the seed
            # phase spins the dycore up and publishes the boundary that
            # the first real analysis will be applied to.
            mpas_bridge.launch(phase="seed", port_root=port_root,
                               port_config=port_config, steps=port_steps,
                               out=out, timeout=timeout)
            segment = mpas_bridge.read_segment(out / "anchor")
            arrays = mpas_bridge.segment_arrays(segment)
            return _publish(cycle_index, segment, ingestion=None,
                            background_sha=prognostic_sha256(
                                arrays["prognostic"]),
                            analysis_state="NONE_YET")

        document = read_anchor(previous)
        try:
            increment = _mapping(document.increment)
        except Exception:
            increment = {}

        segment = mpas_bridge.launch(
            phase="segment", port_root=port_root, port_config=port_config,
            steps=port_steps, anchor=previous, out=out, timeout=timeout)

        # The gate is fed the hash the DEVICE gave back after rehydration,
        # not the host copy that was uploaded.  A host-side comparison
        # passes even when the upload never happened, which is precisely
        # the failure this instrument exists to catch.
        ingestion = verify_ingestion(
            background_sha256=segment["background_sha256"],
            increment=increment,
            analysis_sha256=segment["rehydrated_sha256"],
            label=f"cycle={cycle_index} parent={parent_kind}")

        return _publish(cycle_index, segment, ingestion=ingestion,
                        background_sha=segment["background_sha256"],
                        analysis_state=ingestion["state"])

    return advance
