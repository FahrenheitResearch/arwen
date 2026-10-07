"""EXPERIMENTAL: cycling radar DA over a prepared single-domain case.

Forecast legs on the real GPU dycore, a real LETKF analysis between
them, real NEXRAD volumes -- the loop the ensemble engine cannot host
today and that the v1.2 assembly notes call the engine gap: the engine
refuses domain trees and loads only the ``[case_data]`` route, while the
prepared-tree runners are deterministic and hash-bind their own run
bounds.  Until that gap closes this driver is how a prepared case gets
cycled, and it is deliberately written as the reference implementation
the engine work can absorb rather than as a one-off script.

What it does, per leg:

1. wires the model exactly as
   :func:`gpuwm.prepared_single_domain_forecast.run_prepared_forecast`
   does -- same preflight, same prepared-cache restore, same physics
   initialisation, same boundary clock binding, same
   ``execute_experiment`` -- but per trajectory;
2. restores the trajectory's tree checkpoint set from the end of the
   previous leg through the restart owner
   (:func:`gpuwm.io.restart.restore_tree_restart`), which places every
   domain's clock and carries the atmosphere, the physics driver's
   surface and soil state, the precipitation accumulators and the held
   radiation, boundary-layer and cumulus tendencies, exactly as
   ``gpuwm run --restart`` continues a run;
3. applies the pending analysis and integrates one leg on the GPU, with
   the state-health validator on;
4. writes the trajectory's tree checkpoint set at the leg's end
   (:func:`gpuwm.io.restart.write_tree_restart`), which is what joins
   this leg to the next, and mirrors the serialised prognostic state
   (:data:`gpuwm.state_serialization_contract.STATE_SERIALIZED_ATTRS`)
   to the host for the filter;
5. analyses the members against a ``gpuwm-obs.radar-grid.v1`` file
   through :mod:`gpuwm.da.radar_assimilation`, optionally adding
   :mod:`gpuwm.da.hotstart`'s reflectivity nudge, and applies the result
   through :func:`gpuwm.ensemble.increments.apply_increments` with
   ``refresh_diagnostics`` after it -- the shipped appliers, not a
   private write path.

A control trajectory runs beside the members and is never analysed, so
every number the report carries has a no-DA counterpart taken through
the same code.

**What a leg boundary is.**  A restart.  The join used to carry the
serialised atmosphere alone, so soil, surface, accumulators, held
tendencies and the radiation carriers restarted from the prepared
background at every analysis, for every trajectory, and the driver's
own record said so.  Measured on the card with the tree's own nested
fixture (``tests/test_da_cycle_join_gpu.py``): that join left arrays of
the parent and of the child different from a continuous run at the
same instant, and the restart join leaves none.  A nest that a
trajectory carries rides in the same set and is restored by the same
call; a nest born on a later leg activates at that leg's boundary
(:func:`gpuwm.da.nested_forecast.child_born_at`), so its physics counts
its own first step as step one and its clock is placed on the tree's
tick lattice at birth.

**Deliberate simplifications, stated rather than buried.**

* Each member runs its own lateral boundary forcing
  (:func:`gpuwm.da.perturb.perturbed_lateral_boundaries`, scale
  ``--boundary-perturbation``, default 1): the shared tables ramp over the
  first boundary interval into the member's own draw, so spread no longer
  dies toward the rim.  ``--boundary-perturbation 0`` restores the shared
  boundaries.  After every analysis each member also takes additive
  inflation (:func:`gpuwm.da.perturb.additive_inflation`,
  ``--additive-inflation``, default 0.25 of the initial amplitudes) with
  its ensemble mean removed, so the analysis mean is the filter's and only
  the spread grows.
* The observation grid is supplied per leg and is bound to the
  observation file by digest inside the adapter; a member's own column
  heights differ from that grid by the member's own perturbation, which
  is representativeness error rather than a binding error.

**Cycling past the end of one process.**  ``--save-ensemble`` writes the
leg boundary -- each trajectory's tree checkpoint set plus the analysis
increments the next leg has still to apply -- through
:mod:`tools.da_ensemble_state`, and ``--resume-ensemble`` starts from
one.  That is what a continuous nowcast needs: the next radar volume
does not exist when this one is assimilated, so the ensemble has to
survive the wait without being re-initialised.  The generation is
written at the end of the last OBSERVED leg, so trailing free legs stay
a branch off the cycle rather than becoming it, and the boundary-data
horizon of the prepared case is a refusal rather than a surprise inside
the integrator.

Nothing here is on a default route.  EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import shutil
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from gpuwm.da.iau import IAU_MODES, resolve_iau_mode

try:                                    # python -m tools.da_cycle_prepared
    from tools import da_ensemble_state as ens_state
    from tools import da_solve_ab as ab_bundle
except ImportError:                     # python tools/da_cycle_prepared.py
    import da_ensemble_state as ens_state
    import da_solve_ab as ab_bundle

#: Name of the unassimilated trajectory in every report.
CONTROL = "control"

#: Schema of the report this driver writes.
REPORT_SCHEMA = "gpuwm-da.prepared-cycle-report.v1"

#: The physics-receipt fields that are registry VOCABULARY rather than
#: physics.  A prepared authority written by an older tree names its
#: maturity tier and registry digest in that tree's vocabulary; every
#: selector, component and profile id can still match exactly.
#: ``--tolerate-physics-vocabulary-drift`` allows a mismatch confined to
#: these fields, records it in the report, and refuses any other
#: difference.  Without the flag the preflight is strict.
#:
#: Since A153 the preflight itself resolves a maturity or document-digest
#: difference from 2.8.0 on (it compares the registry's physics parts), so
#: this flag only still matters for an authority written before 2.8.0,
#: whose registry the history does not hold; ``registry_physics`` is here
#: so the flag keeps doing exactly that and nothing more.
PHYSICS_VOCABULARY_FIELDS = ("maturity", "registry_sha256",
                             "registry_physics")


class Iau4dLegRunner:
    """``run_leg`` for ``--iau-mode 4d``: the analysis forced over a window
    centred on its own time, from the member's own trajectory.

    The window is ``[t_a - W/2, t_a + W/2]``.  Its first half lies before
    the analysis, inside the leg that ended there, so that leg is run in two
    parts and leaves a restart at ``t_end - W/2`` (part A) before it carries
    on to its end (part B, whose state the filter analyses); the leg after
    the analysis then starts again from part A's restart -- the member's own
    trajectory half a window before the analysis -- with the increment
    forced from there.  The unforced re-integration is the deterministic
    model on the same restart, so the rewound leg leaves the background
    trajectory exactly where the forcing begins.

    Part A writes under ``<stage>/iau-mid`` (a root the driver's stage does
    not own), and this runner -- the controller, which owns every deletion;
    the member leg deletes nothing -- removes each mid restart once the
    rewound leg has restored it.  The control is never analysed, so it is never split
    or rewound.  Serial route only: the packed transport carries neither
    the mid restart nor the rewind (the driver refuses that pairing).
    """

    MID_ROOT = "iau-mid"

    def __init__(self, run_leg, *, window_seconds: float):
        self.run_leg = run_leg
        self.half = 0.5 * float(window_seconds)
        self.mid: dict = {}

    def _mid_context(self, context, **changes):
        return dataclasses.replace(
            context, stage_root=Path(context.stage_root) / self.MID_ROOT,
            out=Path(context.out) / self.MID_ROOT, **changes)

    def __call__(self, context, name):
        import shutil

        import dataclasses

        rewind = (name != CONTROL and bool(context.pending)
                  and name in self.mid)
        if name != CONTROL and context.pending and not rewind:
            raise RuntimeError(
                f"leg {context.leg} {name}: an analysis is pending and no "
                "restart half a window before it was kept; 4D-IAU needs "
                "the leg that ended at the analysis run in this mode")
        split = (name != CONTROL and context.analysis_due)
        t_mid = float(context.t_end) - self.half
        if split and not t_mid > (float(context.t_start)
                                  - (self.half if rewind else 0.0)):
            raise RuntimeError(
                f"leg {context.leg} {name}: a {2 * self.half:.0f} s window "
                "does not fit a leg of "
                f"{context.t_end - context.t_start:.0f} s")
        first = context
        consumed_mid = None
        if rewind:
            consumed_mid = self.mid.pop(name)
            first = dataclasses.replace(
                context, t_start=float(context.t_start) - self.half,
                restart=consumed_mid, resumed=True, iau_rewound=True)
        records = {}
        if split:
            part_a = self.run_leg(self._mid_context(
                first, t_end=t_mid, analysis_due=False), name)
            records["part_a"] = part_a.record
            self.mid[name] = part_a.restart
            result = self.run_leg(dataclasses.replace(
                context, t_start=t_mid, restart=part_a.restart, pending=None,
                resumed=True, iau_rewound=False), name)
        else:
            result = self.run_leg(first, name)
        if consumed_mid is not None:
            # Restored by the rewound part; nothing else reads it.
            shutil.rmtree(Path(consumed_mid).parent, ignore_errors=True)
        result.record["iau4d"] = {
            "window_half_seconds": self.half,
            "rewound_from_seconds": (float(context.t_start) - self.half
                                     if rewind else None),
            "split_at_seconds": t_mid if split else None,
            **({"part_a": records["part_a"]} if records else {}),
        }
        return result


def run_member_leg_entry():
    """The member leg the roster runs (imported late: the leg module
    imports this one)."""
    from tools.da_member_leg import run_member_leg
    return run_member_leg


def to_host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(np.asarray(value))


#: Schema of the restart-identity components every trajectory's model
#: publishes, so a checkpoint restored into the wrong trajectory is
#: refused with the component named rather than as a bare hash.
TRAJECTORY_IDENTITY_SCHEMA = "gpuwm-da.cycle-trajectory-identity.v1"


def trajectory_identity(identity, name) -> dict:
    """The restart-identity components of one trajectory's model.

    The ensemble identity (:class:`tools.da_ensemble_state.
    EnsembleIdentity`) says which case, grid, scheme and ensemble the
    checkpoint belongs to; the trajectory name says WHICH member.  Both
    are bound, because the restart owner compares fingerprints before it
    reads an array, and member 3's checkpoint restored into member 5's
    model is a plausible-looking ensemble with one member counted twice.
    """
    return {
        "schema": TRAJECTORY_IDENTITY_SCHEMA,
        "ensemble": identity.to_payload(),
        "trajectory": str(name),
    }


def trajectory_fingerprint(identity, name) -> str:
    """SHA-256 of :func:`trajectory_identity`, the model's fingerprint."""
    import hashlib

    payload = json.dumps(trajectory_identity(identity, name),
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_leg_restart(model, directory, *, valid_time, auto_epssm=None) -> Path:
    """Write a trajectory's tree checkpoint set at the end of its leg.

    The restart owner's own writer, on the model the leg just integrated:
    every domain the trajectory carries, the physics driver's surface and
    soil state, the accumulators, the held tendencies, the radiation
    carriers and each clock's exact ticks and boundary accumulator bits.
    Legal only at a period boundary with nothing pending, which is what
    a completed ``execute_experiment`` leaves behind; the owner checks
    that rather than this driver asserting it.  Returns the root member,
    which is what :func:`restore_leg_restart` takes.
    """
    from gpuwm.io.restart import write_tree_restart

    return write_tree_restart(Path(directory), model, valid_time,
                              auto_epssm=auto_epssm)


def restore_leg_restart(model, path, *, expected_seconds: float):
    """Restore a trajectory's checkpoint set into a freshly wired model.

    ``restore_tree_restart`` validates the whole set -- configuration,
    base state, physics setup, array inventory, fingerprint -- before it
    writes a byte, then restores every domain's arrays, driver and clock.
    On top of that, this driver requires the set to stand at the leg
    boundary it is about to integrate from: a generation resumed at the
    wrong elapsed time would otherwise integrate a leg whose boundary
    data belongs to a different hour.
    """
    from gpuwm.io.restart import restore_tree_restart

    info = restore_tree_restart(Path(path), model)
    restored_seconds = info.elapsed_ticks / info.tick_den
    if restored_seconds != float(expected_seconds):
        raise RuntimeError(
            f"the checkpoint set at {path} stands at {restored_seconds:g} "
            f"s of model time and this leg starts at {expected_seconds:g} "
            "s; a leg continues the checkpoint that ended the leg before "
            "it, and this one did not")
    return info


def member_background(result):
    """The background the analysis reads for one forecast result.

    The in-process mirror when the leg ran in this process; otherwise a
    lazy view of the member's own leg-end restart, which holds the same
    bytes (``tools/da_member_leg_proof.py`` checks that on the card) and
    is read one field at a time, only for the fields the analysis uses.
    """
    from gpuwm.da.radar_assimilation import CheckpointStateView

    if result.snapshot is not None:
        return result.snapshot
    return CheckpointStateView(result.restart)


def release_backgrounds(backgrounds: dict) -> None:
    """Close every restart view a finished analysis held, then forget all."""
    for value in backgrounds.values():
        close = getattr(value, "close", None)
        if callable(close):
            close()
    backgrounds.clear()


def write_member_increments(path, increments) -> Path:
    """Member 0's accepted increments as float32, the analysis's own record.

    Written uncompressed: zlib over fourteen whole-domain fields cost
    seconds of one core inside every analysis, for a diagnostic that
    every reader opens with ``np.load`` either way.  The arrays are the
    same float32 bytes.
    """
    path = Path(path)
    np.savez(path, **{key: np.asarray(value).astype(np.float32)
                      for key, value in increments.items()})
    return path


def analysis_backgrounds(snapshots, members: int, *, directory, t_end,
                         files: bool) -> dict:
    """``{member: background}`` for one analysis.

    The backgrounds themselves, with no copy, unless ``files`` asks for
    the legacy staged checkpoints the replay bundle copies: then each
    member's state is saved under ``directory`` exactly as before.
    """
    if not files:
        return {index: snapshots[index] for index in range(int(members))}
    checkpoints = {}
    for index in range(int(members)):
        member_dir = Path(directory) / f"member_{index:03d}"
        member_dir.mkdir(parents=True)
        path = member_dir / f"gpuwmrst_d01_{int(t_end):06d}.npz"
        state = snapshots[index]
        np.savez(path, **{f"state/{key}": state[key] for key in state})
        checkpoints[index] = path
    return checkpoints


class StagedRestarts:
    """Where a run stages each trajectory's leg-end restart set, and its removal.

    A staged set joins one leg to the next inside this process and is
    consumed by exactly one later leg: :meth:`consume` removes it once
    the owner has restored it.  The last leg's sets have no later leg,
    and a run that stops early leaves every set it had staged, so
    :meth:`clear` removes whatever the stage still holds when the run
    ends, however it ends.  A generation under ``--save-ensemble`` is
    copied out of the stage before that and is never touched here: the
    generation is the durable record, the stage is scratch.  The member
    checkpoints an analysis reads (:meth:`analysis_directory`) are
    staged and cleared the same way.

    The default stage is ``<out>/stage`` and is removed with its
    contents.  A directory named by ``--stage-dir`` is the caller's and
    is left in place, emptied of what this run staged.
    """

    #: A set's members, as the restart owner names them.
    MEMBER_GLOB = "gpuwmrst_*.npz"

    def __init__(self, stage_root, *, default: bool, logs_dir=None):
        self.stage_root = Path(stage_root)
        self.root = self.stage_root / "restart"
        self.logs_dir = None if logs_dir is None else Path(logs_dir)
        self.default = bool(default)
        self._analysis_dirs: list[Path] = []
        self._cleared: dict | None = None

    def directory(self, leg_number: int, name) -> Path:
        """Where trajectory ``name`` writes its set at the end of a leg."""
        return (self.root / f"leg{int(leg_number):03d}"
                / ens_state.trajectory_key(name))

    def holds(self, root_member) -> bool:
        """Whether a set's root member lies inside this stage."""
        try:
            return Path(root_member).resolve().is_relative_to(
                self.root.resolve())
        except OSError:
            return False

    def consume(self, root_member) -> bool:
        """Remove a staged set the leg has restored; a set elsewhere stays.

        Returns whether anything was removed, which the leg record
        keeps: a generation's set, resumed from ``--resume-ensemble``,
        lies outside the stage and is the record the resume came from.
        """
        if not self.holds(root_member):
            return False
        directory = Path(root_member).parent
        shutil.rmtree(directory, ignore_errors=True)
        try:
            directory.parent.rmdir()      # the leg's directory, once empty
        except OSError:
            pass
        return True

    def analysis_directory(self, leg_number: int) -> Path:
        """Where one analysis stages the member checkpoints it reads."""
        path = self.stage_root / f"cycle_{int(leg_number):03d}"
        self._analysis_dirs.append(path)
        return path

    def inventory(self) -> dict:
        """How many sets the stage holds now, and their size in bytes."""
        sets = 0
        size = 0
        if self.root.is_dir():
            for trajectory in self.root.glob("leg*/*"):
                members = [member for member in trajectory.glob(
                    self.MEMBER_GLOB) if member.is_file()]
                if members:
                    sets += 1
                    size += sum(member.stat().st_size for member in members)
        return {"restart_sets": sets, "restart_bytes": size}

    def clear(self) -> dict:
        """Remove everything this run staged; the receipt says what went.

        Idempotent: the receipt of the first clearing is returned again
        by every later call, so a run that cleared its stage on the way
        out and is cleared once more by the door reports one clearing.
        """
        if self._cleared is not None:
            return self._cleared
        held = self.inventory()
        shutil.rmtree(self.root, ignore_errors=True)
        # The packed route's per-leg job directories: a member stopped
        # mid-leg leaves its whole restart in its own output root there.
        packed = self.stage_root / "packed"
        packed_bytes = 0
        logs_kept = 0
        if packed.is_dir() and self.logs_dir is not None:
            # The member and control logs are the only record of why a job
            # failed: moved aside (leg/job layout kept), never deleted.
            for log in packed.rglob("*.log"):
                target = self.logs_dir / log.relative_to(packed)
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(log), target)
                    logs_kept += 1
                except OSError:
                    pass
        if packed.is_dir():
            packed_bytes = sum(path.stat().st_size for path in
                               packed.rglob("*") if path.is_file())
            shutil.rmtree(packed, ignore_errors=True)
        analysis_removed = 0
        for path in self._analysis_dirs:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                analysis_removed += 1
        root_removed = False
        if self.default and self.stage_root.is_dir():
            try:
                self.stage_root.rmdir()
                root_removed = True
            except OSError:
                pass
        self._cleared = {
            "restart_sets_removed": held["restart_sets"],
            "restart_bytes_removed": held["restart_bytes"],
            "analysis_directories_removed": analysis_removed,
            "packed_job_bytes_removed": int(packed_bytes),
            "packed_job_logs_kept": int(logs_kept),
            "packed_job_logs_directory": (None if self.logs_dir is None
                                          else str(self.logs_dir)),
            "restart_directory_left": self.root.exists(),
            "stage_root_removed": root_removed,
        }
        return self._cleared


def restart_domain_ids(path) -> tuple[int, ...]:
    """The domain ids a checkpoint set carries, off its own header."""
    from gpuwm.io.restart import read_restart_header

    ids = read_restart_header(Path(path)).get("domain_ids") or ()
    return tuple(int(gid) for gid in ids)


def restart_child_birth_seconds(path, *, grid_id: int, start_time) -> float:
    """When the nest in a checkpoint set was born, in seconds from ``start_time``.

    Read off the child member's own ``domain_start_time``, which is the
    calendar the restart owner checks a resumed child against; the
    resuming leg rebuilds the child's configuration on that instant so
    the check passes and the child keeps the activation epoch it was
    born with.
    """
    import datetime

    from gpuwm.io.restart import read_restart_header, tree_restart_members

    members = tree_restart_members(Path(path))
    if int(grid_id) not in members:
        raise RuntimeError(
            f"the checkpoint set at {path} carries domains "
            f"{sorted(members)} and no d{int(grid_id):02d}")
    header = read_restart_header(members[int(grid_id)])
    stamp = header.get("domain_start_time")
    if not isinstance(stamp, str):
        raise RuntimeError(
            f"the d{int(grid_id):02d} member of {path} records no "
            "domain_start_time; a nest's activation epoch is read from "
            "its checkpoint and this one has none")
    born = datetime.datetime.fromisoformat(stamp)
    return float((born - start_time).total_seconds())



def _reflectivity_floor_argument(text: str):
    """``none`` switches the common reflectivity echo floor off; else a
    finite dBZ value."""

    import argparse
    import math

    if text.strip().lower() == "none":
        return None
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(
            f"expected a finite dBZ value or 'none', got {text!r}")
    return value


def _dispersion_ratio_argument(text: str):
    """``none`` switches the dispersion gate (or its batch condition) off;
    else a finite positive ratio."""

    import argparse
    import math

    if text.strip().lower() == "none":
        return None
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"{text!r}: a positive ratio or 'none'")
    return value


def dispersion_gate_line(ratio, batch_ratio) -> str:
    """The one line every run prints about the dispersion gate: its two
    thresholds, and whether they are the defaults."""

    from gpuwm.da.velocity_dispersion import (
        DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO,
        DEFAULT_VELOCITY_DISPERSION_RATIO)

    if ratio is None:
        return ("velocity dispersion gate: off, so radial velocity updates "
                "theta and vapour wherever it reaches, under-dispersed or "
                "not (gpuwm.da.velocity_dispersion)")
    default = (ratio == DEFAULT_VELOCITY_DISPERSION_RATIO
               and batch_ratio == DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO)
    batch = ("none, every batch gated on its columns alone"
             if batch_ratio is None else f"{batch_ratio:g}")
    return (f"velocity dispersion gate: column {ratio:g}, batch {batch} "
            f"({'default' if default else 'set'})")


def cards_except(gpu_uuid):
    """This process's device ordinals other than the card ``gpu_uuid``.

    ``None`` when the cards cannot be named (no CUDA, or a runtime that
    does not report UUIDs): the analysis then uses every card and shares
    one with the control, which is slower and never wrong.
    """
    try:
        import cupy as cp

        ordinals = []
        for index in range(int(cp.cuda.runtime.getDeviceCount())):
            raw = cp.cuda.runtime.getDeviceProperties(index).get("uuid")
            if not raw:
                return None
            text = bytes(raw).hex()
            name = (f"GPU-{text[:8]}-{text[8:12]}-{text[12:16]}-"
                    f"{text[16:20]}-{text[20:32]}")
            if name.lower() != str(gpu_uuid).lower():
                ordinals.append(index)
        return tuple(ordinals) or None
    except Exception:
        return None


def plan_radar_assimilation(args, mp_physics, *, analysis_fields,
                            cwp: bool):
    """The analysis configuration this cycle will run, built once.

    ONE CONSTRUCTION FOR TWO CALLS, and that is the whole point.  The
    filter's own refusals -- a microphysics scheme the radar operator has
    no H(x) for, a clear-air arm whose floor nobody has read, a
    non-negative field analysed with no positivity policy -- are stated in
    ``gpuwm.da.radar_assimilation.RadarAssimilationConfig.__post_init__``,
    so they fire wherever this configuration is first BUILT.  Until audit
    R-051's follow-up that was inside the leg loop, at the first analysis
    seam, with leg 0's whole ensemble integration already spent; a refusal
    that arrives there is the defect the item named, one seam later rather
    than one member later.

    ``main`` therefore calls this above the leg loop with the field set the
    cycle CAN analyse, and again inside the leg with the set ensemble
    spread left it.  The leg-time set is the plan-time set narrowed (see
    ``analysis_field_selection`` in the report), so the probe never refuses
    a cycle the legs would have run: every refusal it can raise is one the
    leg would have raised, hours later.

    ``mp_physics`` comes from the prepared authority's RunConfig, which is
    the scheme the members actually integrate -- never from a flag, which
    could disagree with the run and would then check the wrong row.
    """

    from gpuwm.da.letkf import Localization
    from gpuwm.da.radar_assimilation import RadarAssimilationConfig
    from gpuwm.da.velocity_dispersion import (
        DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO,
        DEFAULT_VELOCITY_DISPERSION_RATIO)
    from gpuwm.da.reflectivity_echo import (
        DEFAULT_REFLECTIVITY_OUTLIER_SIGMAS, DEFAULT_REFLECTIVITY_FLOOR_DBZ)

    cwp_localization = None
    if cwp:
        cwp_localization = Localization(
            horizontal_m=(args.cwp_horizontal_loc_m
                          if args.cwp_horizontal_loc_m is not None
                          else args.horizontal_loc_m),
            vertical_m=args.cwp_vertical_loc_m)
    return RadarAssimilationConfig(
        localization=Localization(
            horizontal_m=args.horizontal_loc_m,
            vertical_m=args.vertical_loc_m),
        rtps_alpha=args.rtps_alpha, relaxation=args.relaxation,
        analysis_fields=tuple(analysis_fields),
        spread_repair=getattr(args,"spread_repair","off"),
        spread_repair_z_threshold=getattr(args,"spread_repair_z_threshold",25.0),
        spread_repair_seed=int(getattr(args,"seed",0)),
        spread_repair_adaptive=bool(getattr(args,"spread_repair_adaptive",False)),
        velocity=True,
        reflectivity=bool(args.reflectivity_analysis),
        fall_speed=resolved_fall_speed(args, mp_physics),
        velocity_thinning_cells=args.thin_cells,
        velocity_error_inflation=args.err_inflation,
        reflectivity_thinning_cells=args.z_thin_cells,
        reflectivity_error_inflation=args.z_err_inflation,
        clear_air=bool(args.clear_air_analysis),
        clear_air_thinning_cells=args.z0_thin_cells,
        clear_air_error_inflation=args.z0_err_inflation,
        cwp=bool(cwp),
        cwp_localization=cwp_localization,
        cwp_thinning_cells=args.cwp_thin_cells,
        cwp_error_inflation=args.cwp_err_inflation,
        positivity_policy=args.positivity_policy,
        mp_physics=int(mp_physics),
        solve_device=args.solve_device,
        memory_budget_mib=args.memory_budget_mib,
        velocity_dispersion_ratio=getattr(
            args, "velocity_dispersion_gate",
            DEFAULT_VELOCITY_DISPERSION_RATIO),
        velocity_dispersion_batch_ratio=getattr(
            args, "velocity_dispersion_batch_gate",
            DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO),
        precip_analysis=getattr(args, "precip_analysis", "off"),
        reflectivity_floor_dbz=getattr(
            args, "reflectivity_floor_dbz", DEFAULT_REFLECTIVITY_FLOOR_DBZ),
        reflectivity_outlier_sigmas=getattr(
            args, "reflectivity_outlier_sigmas",
            DEFAULT_REFLECTIVITY_OUTLIER_SIGMAS),
        **_field_rule_kwargs(args),
        **_radar_class_kwargs(args))


def _field_rule_kwargs(args) -> dict:
    """The field rules (gpuwm.da.field_rules) from the door's flags; an
    absent flag (an older argv) takes the default."""
    from gpuwm.da import field_rules as fr

    pick = (lambda name, default: getattr(args, name, default))
    return {
        "field_rules": pick("field_rules", fr.DEFAULT_RULES),
        "z_thermo_weight": pick("z_thermo_weight", fr.DEFAULT_Z_THERMO_WEIGHT),
        "z_hydrometeors": pick("z_hydrometeors", fr.DEFAULT_Z_HYDROMETEORS),
        "z_qv_cap": pick("z_qv_cap", fr.DEFAULT_Z_QV_CAP),
        "number_rediagnosis": pick("number_rediagnosis",
                                   fr.DEFAULT_NUMBER_MODE),
    }


def _radar_class_kwargs(args) -> dict:
    """The radar observation class settings (gpuwm.da.radar_classes) from
    the door's flags; an absent flag (an older argv) takes the default."""
    from gpuwm.da import radar_classes as rc

    pick = (lambda name, default: getattr(args, name, default))
    return {
        "z_source": pick("z_source", "z_mean"),
        "reflectivity_clear_floor_dbz": pick(
            "reflectivity_clear_floor_dbz", rc.DEFAULT_CLEAR_FLOOR_DBZ),
        "reflectivity_dead_band": pick(
            "reflectivity_dead_band", rc.DEFAULT_DEAD_BAND),
        "reflectivity_error_dbz": pick(
            "reflectivity_error_dbz", rc.DEFAULT_ECHO_ERROR_DBZ),
        "clear_air_error_dbz": pick(
            "clear_air_error_dbz", rc.DEFAULT_CLEAR_ERROR_DBZ),
        "reflectivity_level_stride": pick(
            "reflectivity_level_stride", rc.DEFAULT_ECHO_LEVEL_STRIDE),
        "clear_air_level_stride": pick(
            "clear_air_level_stride", rc.DEFAULT_CLEAR_LEVEL_STRIDE),
        "radar_top_pa": pick("radar_top_pa", rc.DEFAULT_TOP_PA),
        "reflectivity_huber_c": pick("reflectivity_huber_c", None),
    }


def resolved_fall_speed(args, mp_physics) -> str:
    """The radial-velocity fall-speed closure this cycle runs.

    ``auto`` (the default) is ``reflectivity`` -- ``w - vt`` with each
    species' own fall speed (gpuwm.da.obsop.species_blended_fall_speed) --
    wherever the scheme has a routed H_Z(x), and ``none`` (plain ``w``)
    only where it has none, which the leg record states.  Until this
    default the driver projected plain ``w`` for every scheme, so every
    elevated beam through precipitation was simulated without the
    hydrometeors' fall.
    """
    from gpuwm.da.radar_assimilation import reflectivity_route_available

    stated = getattr(args, "fall_speed", "auto")
    if stated != "auto":
        return stated
    return ("reflectivity" if reflectivity_route_available(mp_physics)
            else "none")


def planned_analysis_fields(args, mp_physics) -> tuple:
    """The fields the plan-time probe checks, before any spread is known.

    ``--hydrometeors`` analyses the scheme's own moment set (the leg then
    drops whole species the ensemble is constant in); without it the
    analysis is the wind pair.  Derived from ``gpuwm.da.moments`` rather
    than typed, so a scheme whose moment set changes cannot leave the
    probe checking a stale list.
    """

    from gpuwm.da import moments

    if not args.hydrometeors:
        return ("u", "v")
    return tuple(moments.analysis_fields(int(mp_physics)))


#: The layer of the column a convective updraft actually occupies.
#: Below the floor the layers are thin because the boundary layer needs
#: them and the vertical velocities there are small; above the ceiling
#: they can be thin again in the anvil and the stratosphere, where they
#: are equally beside the point.  The reference model's own 60-level
#: ladder makes that concrete: its thinnest layer above 2 km sits at
#: 18.7 km.
LIMITER_ONSET_FLOOR_M = 2000.0
LIMITER_ONSET_CEILING_M = 12000.0


def limiter_onset(cfg, floor_m: float = LIMITER_ONSET_FLOOR_M,
                  ceiling_m: float = LIMITER_ONSET_CEILING_M):
    """The updraft at which this ladder and this step start being limited.

    WRF's vertical-velocity limiter (``w_damping = 1``,
    ``gpuwm/core/dycore.py::apply_w_damping``) pushes the w tendency
    against the motion where the vertical Courant number ``w*dt/dz``
    passes 1, which is to say at ``w = dz/dt``.  That number depends on
    nothing but the ladder and the step -- no storm, no assumption -- and
    it is the one line that says whether a run is about to spend its
    updraft on the limiter.  It is reported for the thinnest layer
    BETWEEN ``floor_m`` and ``ceiling_m``, the part of the column a
    convective updraft occupies; outside it the layers can be thin for
    reasons that have nothing to do with updrafts.  It is a floor on
    where limiting can begin, not a promise that it will.

    ``dz`` is differenced from the FULL levels, which is where ``w``
    lives and where a layer starts and stops.  The half-level spacing is
    the same number only where the ladder is uniform; where it stretches
    it is off by the stretch ratio, and at the ground it is half a
    layer.

    WHAT BREAKAGE THIS PREVENTS (the gate law): a ladder refined
    to 200 m layers under the 15 s step that was chosen for 680 m ones.
    Measured on the card, that combination limited above 13 m/s -- an
    ordinary convective updraft -- fired on 606 cells of the parent
    domain, halved the storm's peak w from 33.9 to 18.5 m/s and cost the
    forecast its storm, while the ladder it replaced limited only above
    41 m/s and never fired at all.  Four legs of card time found that
    out; one printed line says it first.

    Returns ``None`` when the limiter is off or the column cannot be
    read, because a diagnostic line is never worth failing a run over.
    """

    try:
        if int(getattr(cfg, "w_damping", 0)) != 1:
            return None
        import numpy as _np
        from gpuwm.core import constants as _c
        from gpuwm.core.grid import (analytic_base_terrain_height,
                                     compute_hybrid_coeffs)
        eta = _np.asarray(cfg.eta_levels, dtype=_np.float64)
        p_top = float(cfg.p_top)
        hy = compute_hybrid_coeffs(eta, int(cfg.hybrid_opt),
                                   float(cfg.etac), _c.P0, p_top)
        pd_half = hy["c3h"] * (_c.P0 - p_top) + hy["c4h"] + p_top
        pd_full = hy["c3f"] * (_c.P0 - p_top) + hy["c4f"] + p_top
        z_half = _np.array([analytic_base_terrain_height(float(v))
                            for v in pd_half])
        z_full = _np.array([analytic_base_terrain_height(float(v))
                            for v in pd_full])
        # Full level to full level: the LAYER the parcel crosses.  The
        # spacing between half levels is a different number wherever the
        # ladder stretches, and half of layer 0 at the ground.
        dz = _np.diff(z_full)
        aloft = ((z_half >= float(floor_m))
                 & (z_half <= float(ceiling_m)))
        if not aloft.any():
            return None
        index = _np.flatnonzero(aloft)
        k = int(index[int(_np.argmin(dz[aloft]))])
        thinnest = float(dz[k])
        return {"thinnest_layer_m": thinnest,
                "thinnest_layer_height_m": float(z_half[k]),
                "w_onset_ms": thinnest / float(cfg.dt)}
    except Exception:
        return None


def bound_child_correction(child_state, correction, *, policy,
                           array_module):
    """Bound a child's correction by the run's policy against the CHILD.

    The parent's analysis is bounded so that the PARENT's background plus
    its increment is non-negative.  The child's background is a different
    field -- its own fine-scale state, evolved since the nest was born --
    and the same correction added to that can put a positive-definite
    species below zero with no arithmetic noise involved at all.  The
    only thing between the correction and the child's pre-leg gate was
    the rounding clamp, which is deliberately held to a millionth of the
    field's own magnitude: right for rounding, and silent about this.

    Measured on the card on an 80-level cycle: water vapour at
    -1.014e-5 kg kg-1 on the child's boundary row at leg 1, four orders
    of magnitude past the clamp's floor, refused by the child's own gate.

    So the correction goes through the SAME policy the parent's analysis
    already went through, against the child's own background, over the
    fields that policy has an opinion about and the child actually
    carries.  A run that stated no policy gets none here either, which is
    the only way to reach this code without one.  Returns
    ``(correction, receipt)``; the receipt is ``None`` when there was no
    policy to apply, and otherwise counts what the bound cost, per field,
    in the child's own receipt.
    """

    if policy is None:
        return correction, None
    from gpuwm.da.positivity import NON_NEGATIVE_FIELDS, apply_positivity
    names = tuple(
        name for name in sorted(correction)
        if name in NON_NEGATIVE_FIELDS
        and getattr(child_state, name, None) is not None)
    if not names:
        return correction, None
    prior = {name: getattr(child_state, name) for name in names}
    bounded, receipt = apply_positivity(
        prior, {name: correction[name] for name in names}, policy=policy)
    out = dict(correction)
    for name in names:
        out[name] = array_module.asarray(bounded[name])
    return out, receipt


#: The threshold used when ``mp_physics`` names no scheme this tree knows
#: the moment structure of.  Morrison's row, which is the same fallback
#: :func:`gpuwm.da.moments.repair_moments` applies when it cannot resolve
#: a scheme, so an unknown scheme gets one answer rather than two.
MOMENT_MASS_THRESHOLD_FALLBACK_KG_KG = 1e-14


def moment_mass_threshold(mp_physics) -> float:
    """The mass above which THIS scheme demands a number moment.

    Read from the scheme rather than fixed, because it is not one number:
    Thompson's activity gate is R1 = 1e-12 (module_mp_thompson.F:183) and
    Morrison's is MQSMALL = 1e-14, two orders of magnitude lower.  A
    single 1e-12 stood here and was described as "the same threshold
    gpuwm.da.moments refuses on", which was Thompson's row read as
    everyone's: under Morrison every cell between 1e-14 and 1e-12 with a
    number at or below zero went unconditioned here and was then refused
    by the moment policy at the next leg, which is precisely the refusal
    this conditioning exists to prevent.
    """

    return resolved_moment_mass_threshold(mp_physics)[0]


def resolved_moment_mass_threshold(mp_physics) -> tuple[float, bool]:
    """``(threshold, whether the scheme answered for it)``.

    The catch names the two ways a scheme fails to resolve and nothing
    else: ``MomentPolicyError`` is the refusal ``scheme_moments`` raises
    for a scheme with no registered moment structure, and ``TypeError``
    or ``ValueError`` is ``int()`` on something that is not a scheme
    number at all.  A catch wider than that would hand Thompson
    Morrison's 1e-14 on any unrelated failure -- an import error, a
    renamed attribute, a typo in a caller -- which is a quieter spelling
    of the single-threshold defect this pair was written to close, with
    nothing to show that it happened.

    The second element is what makes the fallback visible at all: a
    caller that must not condition against a stand-in can ask whether the
    scheme answered instead of comparing the number it got against a
    constant.  :func:`keep_moment_pairs` does not need to -- it refuses
    an unresolvable scheme a few lines later, through the moment policy's
    own named refusal -- so the fallback there is unreachable rather than
    quiet, which a test pins.
    """

    from gpuwm.da import moments as _moments

    try:
        value = _moments.scheme_moments(int(mp_physics)).q_threshold
    except (_moments.MomentPolicyError, TypeError, ValueError):
        return float(MOMENT_MASS_THRESHOLD_FALLBACK_KG_KG), False
    return float(value), True


def keep_moment_pairs(state, increment, *, mp_physics):
    """Condition an increment so it cannot hand on a broken moment pair.

    An ensemble filter writes every field its OWN additive increment.
    Nothing in that arithmetic knows that a species' mass and its number
    moment are one object, so a cell can leave the solve carrying mass
    above the threshold with a number moment at or below zero.  The
    moment policy then refuses the leg that applies it -- correctly,
    because the scheme's own number initialisation is what would have to
    supply the missing moment and inventing an intercept here would be a
    science decision.  Measured on a four-radar hydrometeor analysis that
    is about two thousand cells of half a million; measured on a
    clear-air analysis over one radar it is a hundred.  Either way the
    cycle stops.

    There is a conditioning that invents nothing and that is the
    background's own:

    * where the background already holds that species ACTIVE -- mass
      above the scheme's own threshold, with a positive number -- the
      number takes the SAME multiplicative factor the mass took.  That
      is the background's drop size distribution carried forward, which
      is exactly what the perturbation half of this driver already does
      to the hydrometeor state, and it leaves Z proportional to mass
      rather than manufacturing a distribution;
    * where the background holds none of it, or holds it below the
      threshold, there is no distribution to carry, so the analysis
      declines the whole pair's increment in that cell rather than
      creating condensate it cannot describe.  Declining is not repair:
      the cell keeps exactly the background.

    "Active" is the scheme's own word, and it is the word that matters.
    Any positive background mass used to qualify, so a cloud the scheme
    had already evaporated to 5e-15 kg/kg with a droplet number of 5e4
    left standing was rescaled by an increment of 1e-3 kg/kg: a factor
    of 2e11, a droplet number of 1e16 per kilogram, above the 1e15 the
    pre-leg health gate admits for a number moment, and the next leg
    refused the state.  Below the threshold the scheme reads no number
    at all, so there is no distribution there to carry.

    The rescaled number is then checked against the health gate's own
    ceiling (:func:`gpuwm.core.health.rule_for_field`) and for
    finiteness, and a cell whose result is not representable declines
    the whole pair to the prior.  Never capped: a number held at the
    ceiling is a distribution nobody measured, and the cap would be the
    silent version of the refusal it avoids.

    "Above the threshold" is the SCHEME's threshold
    (:func:`moment_mass_threshold`), not a single number: Thompson
    demands a moment above 1e-12 and Morrison above 1e-14, and using
    Thompson's for both left a Morrison run's smallest broken cells to be
    refused by the next leg instead of conditioned here.

    Only cells that would otherwise break are touched, so a healthy
    increment comes back unchanged and by identity.  Returns
    ``(increment, report)``; the report counts the cells rescaled, the
    cells declined for an inactive background and, separately, the cells
    declined because the rescaled number would not be representable.
    """
    from gpuwm.core.health import rule_for_field
    from gpuwm.da import moments as _moments

    # The threshold goes in the record, so a leg says which mass it
    # conditioned against rather than leaving a reader to re-derive it
    # from the scheme number.  It is always the scheme's own: an
    # unresolvable scheme cannot reach the fallback here, because
    # analysis_fields below refuses it by name first.
    threshold = moment_mass_threshold(mp_physics)

    def _report():
        return {"conditioned": False, "cells_rescaled": 0,
                "cells_declined": 0, "cells_declined_for_ceiling": 0,
                "species": {},
                "mass_threshold_kg_kg": threshold}

    names = tuple(increment)
    available = tuple(
        name for name in set(names)
        | set(_moments.analysis_fields(int(mp_physics)))
        if getattr(state, name, None) is not None)
    pairs = [pair for pair
             in _moments.pairs_present(available, mp_physics=int(mp_physics))
             if set(pair.fields) & set(names)]
    if not pairs:
        return increment, _report()
    out = dict(increment)
    report = _report()
    for pair in pairs:
        q0 = to_host(getattr(state, pair.mass)).astype(np.float64)
        n0 = to_host(getattr(state, pair.number)).astype(np.float64)
        dq = np.asarray(out.get(pair.mass, np.zeros_like(q0)),
                        dtype=np.float64).copy()
        dn = np.asarray(out.get(pair.number, np.zeros_like(n0)),
                        dtype=np.float64).copy()
        q1 = q0 + dq
        n1 = n0 + dn
        bad = (q1 > threshold) & (n1 <= 0.0)
        if not bad.any():
            continue
        # An ACTIVE background: the scheme reads this cell's number, so
        # there is a distribution to carry.  Below the threshold there
        # is none, whatever number was left standing there.
        rescale = bad & (q0 > threshold) & (n0 > 0.0)
        ceiling_rule = rule_for_field(pair.number)
        ceiling = (np.inf if ceiling_rule.upper is None
                   else float(ceiling_rule.upper))
        if rescale.any():
            factor = np.ones_like(q0)
            np.divide(q1, q0, out=factor, where=rescale)
            rescaled = n0 * factor
            # The result has to be a number the next leg's gate admits.
            # Where it is not, the pair declines to the prior; a cap
            # would hand the scheme a distribution nobody measured.
            unrepresentable = rescale & ~(np.isfinite(rescaled)
                                          & (rescaled <= ceiling))
            rescale = rescale & ~unrepresentable
            dn = np.where(rescale, rescaled - n0, dn)
        else:
            unrepresentable = np.zeros_like(bad)
        decline = bad & ~rescale
        if decline.any():
            dq = np.where(decline, 0.0, dq)
            dn = np.where(decline, 0.0, dn)
        extra = {}
        if (pair.volume is not None
                and getattr(state, pair.volume, None) is not None):
            v0 = to_host(getattr(state, pair.volume)).astype(np.float64)
            dv = np.asarray(out.get(pair.volume, np.zeros_like(v0)),
                            dtype=np.float64).copy()
            if rescale.any():
                factor = np.ones_like(q0)
                np.divide(q1, q0, out=factor, where=rescale)
                dv = np.where(rescale, v0 * factor - v0, dv)
            dv = np.where(decline, 0.0, dv)
            extra[pair.volume] = dv
        out[pair.mass] = dq.astype(
            np.asarray(increment.get(pair.mass, dq)).dtype, copy=False)
        out[pair.number] = dn.astype(
            np.asarray(increment.get(pair.number, dn)).dtype, copy=False)
        out.update(extra)
        for_ceiling = int(unrepresentable.sum())
        report["conditioned"] = True
        report["cells_rescaled"] += int(rescale.sum())
        report["cells_declined"] += int(decline.sum())
        report["cells_declined_for_ceiling"] += for_ceiling
        report["species"][pair.species] = {
            "rescaled": int(rescale.sum()),
            "declined": int(decline.sum()),
            "declined_for_ceiling": for_ceiling,
            "number_ceiling": None if not np.isfinite(ceiling) else ceiling}
    if not report["conditioned"]:
        return increment, report
    return out, report


def merge_hotstart_increments(filter_increments, hot_increments, *, prior,
                              positivity_policy, report_overlap: bool = True):
    """The filter increment plus the insertion, bounded as one analysis.

    Both halves are increments to the SAME background from the same
    reflectivity volume, so where they name the same field they are summed
    -- and the sum is what the next leg applies.  Each half is bounded on
    its own: the filter's by the run's positivity policy against this
    background, the insertion's by its own configured caps and by never
    taking more vapour than the column holds.  Neither of those bounds the
    SUM, and two admissible negatives add to an inadmissible one: a real
    cycle put water vapour at -1.02e-4 kg/kg this way and the next leg
    refused the state it was handed.

    So the mapping goes back through the same policy against the same
    background -- every field of it, on every leg, because this mapping
    IS what the generation saves and what the next leg applies, and a
    field left out here is a field that leaves the process unbounded.
    Re-binding a field the filter already bound against this same
    background changes nothing, which is what makes that safe.  A run
    that stated no policy gets no policy here either; that is the only
    way to reach this code without one, and inventing one would be a
    different analysis than the one asked for.

    Refuses before any of that when the mapping carries a field the
    pre-leg health gate floors at zero and the policy has no opinion
    about: those two lists live in different files, and when they
    disagree the run finds out a leg later in another process.

    Returns ``(merged, overlap, positivity_receipt)``.  ``overlap`` is the
    per-field size of each half, for the fields both had an opinion about,
    and is empty when ``report_overlap`` is false.  ``positivity_receipt``
    is ``None`` only when the run stated no policy.
    """

    merged = dict(filter_increments)
    overlap: dict[str, dict] = {}
    for field, values in (hot_increments or {}).items():
        if field not in merged:
            merged[field] = values
            continue
        # BOTH halves have an opinion about this field.  Summing double
        # counts the same reflectivity volume, so each component's size is
        # reported rather than one silently overwriting the other.
        base = np.asarray(merged[field], np.float64)
        added = np.asarray(values, np.float64)
        if report_overlap:
            overlap[field] = {
                "filter_rms": float(np.sqrt(np.mean(base ** 2))),
                "hotstart_rms": float(np.sqrt(np.mean(added ** 2))),
            }
        merged[field] = (base + added).astype(np.float32)

    receipt = None
    if positivity_policy is not None:
        from gpuwm.core.health import rule_for_field
        from gpuwm.da import positivity as _positivity
        from gpuwm.da.positivity import (PositivityError, apply_positivity,
                                         verify_non_negative)
        # The two contracts that decide whether this mapping survives the
        # process boundary are written in different files, and when they
        # disagree the run finds out four minutes later, in another
        # process, as a health gate refusing a number with nothing
        # pointing back here.  ``rule_for_field`` says which fields the
        # pre-leg gate floors at zero; ``NON_NEGATIVE_FIELDS`` says which
        # ones the policy will bound.  A field in the first and not the
        # second is analysed unbounded and then refused -- which is what
        # the aerosol-aware tracers did on a radar cycle whose saved
        # generation reached -2.96e9 kg^-1 in 4,178 cells while the
        # policy's own receipt called them "unconstrained".  Read from
        # the INSTALLED module, so a tree whose list has grown past the
        # package a run imports is caught here rather than on the card.
        unbounded = tuple(
            name for name in sorted(merged)
            if name not in _positivity.NON_NEGATIVE_FIELDS
            and rule_for_field(name).lower == 0.0)
        if unbounded:
            raise PositivityError(
                f"the analysis carries {list(unbounded)}, which the health "
                f"gate floors at zero and the positivity policy "
                f"{positivity_policy!r} has no opinion about: the saved "
                "generation would go out unbounded on those fields and the "
                "next leg's pre-leg health gate would refuse the state it "
                "built from them. The two lists that disagree are "
                "gpuwm.core.health.rule_for_field and "
                f"gpuwm.da.positivity.NON_NEGATIVE_FIELDS in "
                f"{_positivity.__file__}")
        # EVERY field of the mapping, and on every leg -- not only the
        # insertion's fields on the legs that carry an insertion.  This
        # mapping IS what the generation saves and what the next leg
        # applies, so the field that is not re-bound here is the field
        # that leaves this process unbounded.
        merged, receipt = apply_positivity(
            prior, merged, policy=positivity_policy)
        if positivity_policy in _positivity.BOUNDING_POLICIES:
            # The post-condition that catches a policy applied to the
            # wrong mapping, asked of the bytes the next leg will apply.
            verify_non_negative(prior, merged)
    return merged, overlap, receipt


def rain_snapshot(driver) -> dict:
    """The restart-owned surface rain accumulators, in native millimetres.

    The physics output owner supplies both RAINC and RAINNC even when a
    producer is disabled. Neither field is reset at a DA leg boundary.
    A decrease in saved accumulations is therefore a scoring refusal,
    rather than a negative increment that a consumer may clip to zero.
    """
    fields = driver.output_fields()
    return {name: to_host(fields[name]).astype(np.float32)
            for name in ("RAINC", "RAINNC")}


#: The reflectivity a composite frame carries (its npz ``refl_product``).
#: NATIVE is the model's own REFL_10CM from an output-due microphysics call,
#: the field WOOF publishes; OPERATOR is the DA forward operator over the
#: state between steps.  Scores and maps compare like with like.
REFL_PRODUCT_NATIVE = "native_refl_10cm"
REFL_PRODUCT_OPERATOR = "da_forward_operator"


def _column_max_float32(field) -> np.ndarray:
    """``to_host(field).astype(np.float32).max(axis=0)``, the same bytes.

    On the card when the field is there: casting to float32 is monotonic
    and a maximum picks one of its inputs, so the column maximum is exact
    in either order and only the 2-D answer crosses to the host instead of
    the whole 3-D reflectivity.
    """
    if hasattr(field, "__cuda_array_interface__"):
        import cupy as cp

        return to_host(cp.asarray(field).astype(cp.float32).max(axis=0))
    return to_host(field).astype(np.float32).max(axis=0)


def _write_rain_composite(npz_path, *, state, driver, grid, cfg,
                          elapsed_seconds, exp, label, domain=None,
                          reflectivity=None, writer=None):
    """Save a native two-dimensional score frame without modifying state.

    ``reflectivity`` is the scheme-native REFL_10CM an output-due
    microphysics call staged; without one (a frame no step has produced,
    such as an analysis issued before its first step) the DA forward
    operator stands in.  The two differ on one state (CONUS first light,
    20Z control: 1,739 against 1,368 cells at 35 dBZ or more), so every
    frame records which it carries in ``refl_product`` and a scorer must
    not difference one against the other.

    The frame is taken from the live state here; with ``writer`` (an
    executor), the compression and the wrfout copy run on it and this
    returns its future, so a forecast's history alarm costs the step loop
    only the reads.  Breakage: on the 9 km CONUS free leg the members sat
    at about 175 % CPU writing the two-minute rain history while the cards
    idled.
    """
    from gpuwm.da import obsop

    rain = rain_snapshot(driver)
    if reflectivity is None:
        field, product = (obsop.simulated_reflectivity(state, cfg),
                          REFL_PRODUCT_OPERATOR)
    else:
        field, product = reflectivity, REFL_PRODUCT_NATIVE
    composite = _column_max_float32(field)
    del field

    def persist():
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, refl_colmax=composite, **rain,
                            elapsed_seconds=np.float64(elapsed_seconds),
                            rain_reset_id=np.int64(0),
                            refl_product=np.str_(product))
        return _write_composite_wrfout(
            npz_path, composite, grid, cfg, elapsed_seconds, exp,
            label=label, domain=domain, rain=rain)

    if writer is None:
        return persist()
    return writer.submit(persist)


_LATLON_MASS: dict = {}


def _latlon_mass(grid):
    """``grid.latlon_mass()``, once per grid object in a process: every
    history frame of a leg asked the projection for the same arrays."""
    key = id(grid)
    cached = _LATLON_MASS.get(key)
    if cached is None or cached[0] is not grid:
        cached = _LATLON_MASS[key] = (grid, grid.latlon_mass())
    return cached[1]


def _write_composite_wrfout(npz_path, refl_colmax, grid, cfg,
                            elapsed_seconds: float, exp, *, label: str,
                            domain=None, rain=None):
    """A real wrfout beside a composite ``.npz``; the path, or ``None``.

    ``domain`` is the CHILD's ``DomainConfig`` when this frame belongs to
    a nest.  It puts the WRF topology group (GRID_ID, PARENT_ID, the two
    parent starts and the ratio) in the file, so the frame STATES which
    domain it is instead of leaving that to the ``dNN`` token in its
    name.  Omitted for the root domain, which is what a file with no
    topology group already means.

    Why a writer and not a Rust ``.npz`` reader: ``.npz`` carries no
    geolocation contract, so a reader for it would be a per-format adapter
    -- exactly what the arbitrary-acceptance rule forbids -- while this
    lane already holds the grid the composite was computed on and a wrfout
    is the container that states it.

    Additive and non-fatal.  The cycle's product is the ``.npz``; a
    failure to write the second copy is REPORTED and the forecast
    continues, because a rendering convenience must not be able to kill a
    DA cycle.
    """

    import datetime

    from gpuwm.io.surface_wrfout import (SurfaceSnapshotRefusal,
                                         snapshot_wrfout_path,
                                         write_surface_wrfout)
    from gpuwm.io.wrfout import wrf_global_attrs

    try:
        lat, lon = _latlon_mass(grid)
        snapshot = {
            "XLAT": np.asarray(lat, np.float32),
            "XLONG": np.asarray(lon, np.float32),
            "REFL_COMPOSITE": np.asarray(refl_colmax, np.float32),
        }
        if rain is not None:
            snapshot.update(rain)
        start = getattr(exp, "start_time", None)
        if not isinstance(start, datetime.datetime):
            start = datetime.datetime(1970, 1, 1)
        stamp = (start + datetime.timedelta(seconds=float(elapsed_seconds))
                 ).strftime("%Y-%m-%d_%H:%M:%S")
        topology = {} if domain is None else {
            "grid_id": int(domain.grid_id),
            "parent_id": int(domain.parent_id),
            "i_parent_start": int(domain.i_parent_start),
            "j_parent_start": int(domain.j_parent_start),
            "parent_grid_ratio": int(domain.parent_grid_ratio)}
        # ``start`` stays the HEAD grid's start on every domain, which is
        # what WRF itself writes: SIMULATION_START_DATE is the run origin
        # rw_wrfbatch measures each product's lead from, and a child that
        # stamped its own start there would label the same instant with a
        # different lead than the parent frame beside it.
        attrs = wrf_global_attrs(grid, start, dt=float(cfg.dt), **topology)
        if rain is not None:
            attrs["GPUWM_RAIN_RESET_ID"] = 0
            attrs["GPUWM_RAIN_ACCUMULATION_SEMANTICS"] = (
                "cumulative native mm; complete restart joins preserve "
                "RAINC and RAINNC; no accumulator reset in this lineage")
        report = write_surface_wrfout(
            snapshot_wrfout_path(npz_path), snapshot, time_str=stamp,
            dx=float(cfg.dx), dy=float(cfg.dy), global_attrs=attrs,
            grid_id=None if domain is None else int(domain.grid_id),
            title=f"gpuwm DA composite ({label})")
        # Anything the snapshot carried that the file did not get, by
        # name.  Empty for the composite this lane builds, and stated
        # anyway, because the lane that starts carrying a second field
        # must not have to discover it went missing from a blank panel.
        lost = report.skipped_report()
        if lost:
            print(f"    composite wrfout for {label} omits: {lost}")
        return report.path
    except (SurfaceSnapshotRefusal, AttributeError, TypeError,
            ValueError, OSError) as problem:
        print(f"    composite wrfout skipped for {label}: {problem}")
        return None

#: Card memory each packed card keeps back beside its member processes.
PACKED_CARD_RESERVE_BYTES = 4*1024**3


def packed_members_per_card(choice, cards, member_peak_bytes, *,
                            mps_available=None):
    """Member processes per card for a packed roster, or ValueError.

    Any number of cards and members: :func:`gpuwm.da.member_wave.plan_wave`
    runs ``len(cards) x width`` members at once and the rest in later
    waves.  The route used to accept only 32 or 64 members on exactly
    eight cards, which named no breakage (the waves were already
    general) and kept the packed step off every one-to-four-card box.
    ``auto`` takes 4, 2 or 1, the most the priced complete trajectory
    fits on the tightest card after the reserve; a stated width that
    does not fit is refused.
    """
    if not cards:
        raise ValueError("packed DA needs at least one physical card")
    available = min(min(card["free_bytes"], card["total_bytes"])
                    - PACKED_CARD_RESERVE_BYTES for card in cards)
    if choice == "auto":
        # Four members per card share it through an MPS controller
        # (gpuwm.da.member_wave refuses four without one).  auto took 4
        # whenever memory allowed, so on 96 GB cards with no controller
        # running every packed run failed at launch (box K, 2026-10-05,
        # twice).  Without an answering controller auto stops at 2.
        if mps_available is None:
            from gpuwm.da.member_wave import mps_available as _probe
            mps_available = _probe()
        widths = (4, 2, 1) if mps_available else (2, 1)
        for width in widths:
            if width*member_peak_bytes <= available:
                return width
        raise ValueError(
            f"one packed member ({member_peak_bytes/1024**3:.1f} GiB priced) "
            f"does not fit the tightest card ({available/1024**3:.1f} GiB "
            "after the reserve)")
    width = int(choice)
    if width*member_peak_bytes > available:
        raise ValueError(
            f"{width} packed members ({width*member_peak_bytes/1024**3:.1f} "
            "GiB priced) do not fit the tightest card "
            f"({available/1024**3:.1f} GiB after the reserve)")
    return width


class RecoveryRing:
    """A packed run's recovery generations, pruned when the run stops early.

    The packed controller writes a recovery generation at every leg
    boundary into a ring of slots; a run that completes removes the ring
    (its report says so).  A run that stops early keeps exactly ONE: the
    newest generation whose manifest landed, the boundary a resume
    continues from.  Every other slot, and a slot whose write the stop
    interrupted, is removed, and ``recovery-cleanup.json`` beside the ring
    says what went.  Breakage: a stopped CONUS run on box E left 188 GB of
    stage and recovery sets behind and the next run died on a full disk.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.done = None

    def clear(self) -> dict | None:
        if self.done is not None or not self.root.is_dir():
            return self.done
        newest, newest_seconds = None, None
        for slot in sorted(self.root.glob("slot*")):
            manifest = slot / "ensemble-manifest.json"
            try:
                seconds = float(json.loads(manifest.read_text(
                    encoding="utf-8"))["elapsed_seconds"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if newest_seconds is None or seconds > newest_seconds:
                newest, newest_seconds = slot, seconds
        removed = []
        for slot in sorted(self.root.glob("slot*")):
            if slot == newest:
                continue
            size = sum(path.stat().st_size for path in slot.rglob("*")
                       if path.is_file())
            shutil.rmtree(slot, ignore_errors=True)
            removed.append({"slot": slot.name, "bytes": int(size)})
        self.done = {"kept": None if newest is None else newest.name,
                     "kept_elapsed_seconds": newest_seconds,
                     "removed": removed,
                     "rule": "a stopped run keeps its newest complete "
                             "generation only"}
        try:
            (self.root / "recovery-cleanup.json").write_text(
                json.dumps(self.done, indent=1), encoding="utf-8")
        except OSError:
            pass
        return self.done


def control_vr_innovations(document, u_e, v_n, w_m):
    """``(per_radar, pooled)``: the control's radial-velocity innovation
    against every radar of a leg's document (no filter, direct H(x)), one
    row per radar and the residual arrays of the radars that observed.

    Every radar in the file, not radar 0.  Each antenna has its own beam
    geometry, so a radial-velocity innovation is only defined per radar;
    indexing [0] and calling the answer "the" control innovation was
    harmless while every file held one radar and silently wrong the moment
    one held twenty.  The caller pools the residuals, which is the same
    statistic when there is one radar.

    On a v2 file the stored plane covers only the radar's reach window and
    is zero outside it, so the innovation is evaluated on the window: the
    same points in the same (k, j, i) order as on the expanded domain, so
    the same residual array.  Expanding five whole-domain planes per radar
    was about 25 s of every 9 km CONUS analysis (some 140 radars) with
    seven cards idle.  On v1 (no windows) it is the whole-domain path.
    """
    from gpuwm.da.obs_radar import (_radar_window, beam_unit_vectors,
                                    observation_shape,
                                    simulated_radial_velocity)
    from gpuwm.obs.radar_grid import radar_plane

    obs_shape = observation_shape(document)
    per_radar, pooled = [], []
    for index, radar in enumerate(document["radars"]):
        window = _radar_window(document, index, obs_shape)
        if window is None:
            vr_mask = np.asarray(
                radar_plane(document, "vr_mask", index)).astype(bool)
        else:
            j0, j1, i0, i1 = window
            nj, ni = j1 - j0 + 1, i1 - i0 + 1
            crop = (slice(None), slice(j0, j1 + 1), slice(i0, i1 + 1))

            def stored(name, _index=index, _nj=nj, _ni=ni):
                return np.asarray(
                    document["variables"][name][_index])[:, :_nj, :_ni]
            vr_mask = stored("vr_mask").astype(bool)
        points = int(vr_mask.sum())
        row = {"radar": str(radar["id"]), "points": points}
        if points:
            if window is None:
                vr_obs = np.asarray(
                    radar_plane(document, "vr_obs", index), np.float64)
                sim = simulated_radial_velocity(
                    u_e, v_n, w_m, beam_unit_vectors(document, index))
            else:
                missing = [name for name in ("vr_beam_east", "vr_beam_north",
                                             "vr_beam_up")
                           if name not in document["variables"]]
                if missing:
                    # beam_unit_vectors' own refusal, on the window path.
                    beam_unit_vectors(document, index)
                vr_obs = np.asarray(stored("vr_obs"), np.float64)
                sim = simulated_radial_velocity(
                    np.asarray(u_e)[crop], np.asarray(v_n)[crop],
                    np.asarray(w_m)[crop],
                    tuple(np.asarray(stored(name), dtype=np.float64)
                          for name in ("vr_beam_east", "vr_beam_north",
                                       "vr_beam_up")))
            d = vr_obs[vr_mask] - sim[vr_mask]
            pooled.append(d)
            row["innovation_mean_ms"] = float(d.mean())
            row["innovation_rms_ms"] = float(np.sqrt(np.mean(d ** 2)))
        per_radar.append(row)
    return per_radar, pooled


def prune_recovery_ring(kept) -> list:
    """Remove every other slot of the ring ``kept`` belongs to.

    Called once generation ``kept`` has landed (its manifest written), so
    the ring always holds one complete generation -- the newest -- plus,
    while the next one is written, that one in flight.  Breakage: the ring
    kept both slots for the whole run, two complete 32-member CONUS
    generations (76 GB each) on top of the stage and the issuance
    generation, about 250 GB at peak on box N; box L (243 GB free) could not
    host a cycled arm at all.  A resume reads the newest landed generation
    (``latest_generation``), which is the one kept.
    """
    kept = Path(kept)
    removed = []
    for slot in sorted(kept.parent.glob("slot*")):
        if slot == kept or not slot.is_dir():
            continue
        size = sum(path.stat().st_size for path in slot.rglob("*")
                   if path.is_file())
        shutil.rmtree(slot)
        removed.append({"slot": slot.name, "bytes": int(size)})
    return removed


class _Stopped(SystemExit):
    """A termination signal turned into an unwind, so the door's cleanup
    runs (a default SIGTERM ends the process without it)."""


def _unwind_on_termination():
    """While the cycle runs, SIGTERM and SIGHUP raise :class:`_Stopped`.

    Only signals still at their default disposition and only on the main
    thread; the member waves leave a disposition the controller set alone
    and stop their workers on the way out."""
    import signal
    import threading

    previous = {}
    if threading.current_thread() is not threading.main_thread():
        return previous

    def handler(signum, unused_frame):
        raise _Stopped(128 + signum)

    for name in ("SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is not None and signal.getsignal(number) is signal.SIG_DFL:
            previous[number] = signal.signal(number, handler)
    return previous


def main() -> int:
    """The door: run the cycle, and clear its stage however the run ends.

    :func:`cycle` registers the stage it writes under in ``stages`` as
    soon as it knows where that is, so a run that stops on a refusal, a
    device error or a treatment verdict has its staged restart sets
    removed here exactly as a run that reaches its last leg does.  A
    completed run clears the stage itself and writes the receipt into
    its report; the clearing here is the same call again and removes
    nothing more.
    """
    import signal

    stages: list = []
    previous = _unwind_on_termination()
    try:
        return cycle(stages)
    finally:
        for stage in stages:
            stage.clear()
        for number, handler in previous.items():
            signal.signal(number, handler)


def build_parser():
    """The public serial and packed DA doors share one argument table."""
    from gpuwm.da import background

    parser = argparse.ArgumentParser(
        description="cycling radar DA over a prepared single-domain case")
    # -- the prepared authority (all required: this driver reproduces the
    #    front door's own binding rather than inventing a looser one) ----
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--authority-dir", type=Path, default=None,
                        help="defaults to <prepared-root>/../authority")
    # NO `choices=` HERE, deliberately; the refusal is this driver's own
    # sentence, raised while the value is converted so it lands before the
    # required-argument sweep rather than after it.  Two reasons, both
    # worth the lines.  argparse's invalid-choice wording is the
    # interpreter's, not ours: the choice list lost its quotes in 3.12 and
    # got them back in 3.13, so a caller reading it, or a test pinning it,
    # is pinned to a Python version rather than to this tool.
    # And `BACKGROUND_SOURCES` is a LIVE projection of the runnable source
    # table, so a name argparse would have frozen into `choices` at parser
    # build time is a name this driver never explains: a source the table
    # HAS but marks not runnable was told only that it was not in a list.
    def background_source(name: str) -> str:
        if name in background.BACKGROUND_SOURCES:
            return name
        known = ", ".join(sorted(background.BACKGROUND_SOURCES))
        raise argparse.ArgumentTypeError(
            f"{name} has no background registry entry, so this driver "
            "cannot state the cycle cadence, publication lag or forecast "
            "horizon its legs are planned from. The sources it has a "
            f"registry for are {known}. A source gpuwm's adapter table "
            "carries but this registry does not is one the table marks "
            "not runnable; prepare the case on a runnable source, or name "
            "the source the prepared root was actually built on")

    parser.add_argument(
        "--source", default=background.DEFAULT_BACKGROUND_SOURCE,
        type=background_source,
        # The roster stays where `choices=` used to put it -- one
        # unbroken brace list -- because argparse wraps a HELP paragraph
        # at the terminal width and would hyphenate a source id across
        # two lines, which is not a name anyone can copy back in.
        metavar="{" + ",".join(sorted(background.BACKGROUND_SOURCES)) + "}",
        help=("which background the prepared case was built on.  This is "
              "the SELECTION recorded in the report, not a switch that "
              "changes how the case is read: the prepared root already "
              "IS one source's case, and naming a different one here is "
              "refused at the front door.  "
              f"{background.DEFAULT_BACKGROUND_SOURCE} is the default "
              "and its behaviour is unchanged"))
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--prepared-content-sha256", required=True)
    parser.add_argument(
        "--physics-profile", required=True,
        help="the named physics profile the prepared case was bound to, "
             "or 'experiment-config' for a case whose physics the "
             "experiment config itself selects (profile_binding "
             "'experiment-config' in its sim run report); "
             "the preflight then binds the experiment's own physics, as "
             "the sim door does")
    parser.add_argument("--run-seconds", type=float, required=True,
                        help="the hash-bound experiment's own run_seconds")
    parser.add_argument("--history-interval-seconds", type=float,
                        required=True)
    parser.add_argument(
        "--tolerate-physics-vocabulary-drift", action="store_true",
        help=("accept a prepared physics receipt that differs from this "
              f"tree only in {list(PHYSICS_VOCABULARY_FIELDS)} -- registry "
              "vocabulary, not physics.  Any other difference is still a "
              "refusal, and the divergence is recorded in the report"))
    # -- the cycle ------------------------------------------------------
    parser.add_argument("--leg-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--leg-durations-seconds", type=float, nargs="+", default=None,
        help=("explicit duration of every observed and free leg, in order. "
              "A clean no-DA control can use the observed arm's exact "
              "spin-up and issuance clock without assimilating its files. "
              "When present, this replaces the scalar leg-duration flags"))
    parser.add_argument(
        "--final-leg-seconds", type=float, default=None,
        help=("duration of the LAST leg only (default: --leg-seconds). "
              "The last leg's analysis is computed and never applied, so "
              "a longer final leg is a free forecast from the last "
              "applied analysis, verified against the final obs file at "
              "its own end -- cycling at one cadence and verifying at a "
              "longer lead without pretending the driver can restart"))
    parser.add_argument(
        "--free-leg-seconds", type=float, default=None,
        help=("duration of each FREE leg (default: --leg-seconds). A "
              "nowcast that cycles on the radar's own volume times has "
              "an observed-leg length set by the data and a forecast "
              "length set by what is worth looking at; this separates "
              "them instead of making one impersonate the other"))
    parser.add_argument(
        "--free-legs", type=int, default=0,
        help=("number of trailing legs run with NO observations: no "
              "analysis, no verification, composites still saved.  When "
              "nonzero, every obs leg's analysis IS applied (the free "
              "legs are the forecast running past the last observation, "
              "whose verification frames do not exist yet -- receipts "
              "and renders must say so).  Zero keeps the legacy "
              "final-leg-is-verification rule unchanged"))
    parser.add_argument(
        "--save-composites", action="store_true",
        help=("write each trajectory's column-max H_Z(x) at every leg "
              "end to <out>/composites/legNN_<name>.npz (float32 "
              "(ny, nx)).  A quicklook diagnostic, deliberately tiny; "
              "wrfout history for cycled arms remains its own package"))
    parser.add_argument("--rain-history", action="store_true",
        help="write native rain and column echo on the authority history clock during the free forecast")
    parser.add_argument("--rain-forecast-start-seconds", type=float, default=None,
        help="elapsed issuance clock for rain history, identical in DA and clean-start controls")
    parser.add_argument("--packed-smoke", action="store_true",
        help="bounded real-data smoke: four members on one card, at most 65x65 and two hours of model time")
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument(
        "--forecast-members-per-card", choices=("serial", "auto", "1", "2", "4"), default="serial",
        help=("member forecast processes on each physical GPU, over any "
              "number of cards and members (members beyond cards x this "
              "run in later waves); auto selects 4, 2 or 1, the most the "
              "priced complete trajectory fits on every card. "
              "Four requires an active owned CUDA MPS controller"))
    parser.add_argument(
        "--forecast-device-uuids", nargs="+", default=None,
        help="physical GPU UUIDs for packed forecast waves; defaults to the "
             "cards CUDA_VISIBLE_DEVICES names by UUID, else every card")
    parser.add_argument(
        "--forecast-leg-timeout-seconds", type=float, default=1800.0,
        help="bounded wall time for one complete packed forecast roster")
    # -- the fine nest, over every leg ----------------------------------
    #
    # Off unless asked for.  When on it runs on EVERY leg, observed and
    # free: a child attached to the free legs alone is born at the fork
    # between the cycle and whatever consumes its analysis, and a domain
    # minutes old is still growing the fine structure its spacing exists
    # to resolve, so a window comparison across that fork measures the
    # birth as much as the weather.  Everything about the child except
    # these keys is DERIVED -- dx, dy and dt come off the parent through
    # the ratio chain and are never typed here.
    parser.add_argument(
        "--nest-ratio", type=int, default=3,
        help=("parent-to-child refinement ratio, applied to BOTH space "
              "and time (WRF's SINT assumes a square ratio).  3 off a "
              "3 km parent is a 1 km nest at a 5 s step"))
    parser.add_argument(
        "--nest-half-width-km", type=float, default=None,
        help=("half-width of the nest in kilometres, centred in the "
              "parent.  Turning this on is what enables the nest; "
              "mutually exclusive with --nest-nx/--nest-ny"))
    parser.add_argument("--nest-nx", type=int, default=None,
                        help="child extent in CHILD cells (with --nest-ny)")
    parser.add_argument("--nest-ny", type=int, default=None,
                        help="child extent in CHILD cells (with --nest-nx)")
    parser.add_argument("--nest-i-parent-start", type=int, default=None,
                        help="1-based parent cell of the child's origin "
                             "(default: centred)")
    parser.add_argument("--nest-j-parent-start", type=int, default=None)
    parser.add_argument(
        "--nest-members", type=int, default=None,
        help=("how many ensemble members carry a nest, beside the "
              "control (default 0: the control only).  The parent always "
              "carries the FULL ensemble -- the nest is deliberately "
              "cheap, and nest cost scales as ratio^3 per covered parent "
              "cell TIMES this number"))
    parser.add_argument(
        "--nest-history-interval-s", type=float, default=None,
        help="child history cadence (default: the parent's)")
    parser.add_argument(
        "--nest-acknowledge", action="append", default=[],
        help=("acknowledge a nested-domain admissibility refusal by id, "
              "e.g. nested-forecast:sub-gray-zone-pbl"))
    parser.add_argument(
        "--obs", type=Path, action="append", default=[],
        help=("one gpuwm-obs.radar-grid.v1 file per leg, in leg order.  "
              "The last leg's file is read for verification only -- its "
              "analysis is computed and reported but never applied, so "
              "the final state is a forecast nobody has corrected"))
    parser.add_argument(
        "--grid-wrfout", type=Path, action="append", default=[],
        help=("the wrfout whose georeference each leg's observations were "
              "gridded onto, in leg order; one per --obs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, default=None,
                        help="where each trajectory's leg-end restart set "
                             "and the member checkpoints of an analysis "
                             "are staged (a tmpfs is a good choice; budget "
                             "one restart set per trajectory plus one "
                             "analysis's member checkpoints); a set is "
                             "removed once the next leg has restored it, "
                             "and whatever the stage still holds when the "
                             "run ends is removed with it, however the run "
                             "ends; the generation under --save-ensemble "
                             "is the copy that stays; default is a "
                             "directory under --out, removed at the end")
    # -- carrying the ensemble between processes ------------------------
    # A leg boundary inside one process is each trajectory's restart set
    # plus its unapplied increments; these two flags are that same
    # boundary on disk, so a continuous nowcast can assimilate an
    # observation that did not exist when the previous cycle ran.
    # tools/da_ensemble_state.py documents the format and does the
    # identity checking.
    parser.add_argument(
        "--resume-ensemble", type=Path, default=None,
        help=("resume from an ensemble generation written by "
              "--save-ensemble instead of perturbing a fresh ensemble. "
              "Leg 0 restores each trajectory's restart set, applies the "
              "increments the generation carried, and starts at the "
              "generation's own elapsed seconds"))
    parser.add_argument(
        "--save-ensemble", type=Path, default=None,
        help=("write the ensemble generation at the end of the LAST "
              "OBSERVED leg -- before any free legs, because free legs "
              "are a branch off the cycle and must never become the "
              "cycle.  Requires at least one --obs"))
    parser.add_argument(
        "--leg-number-offset", type=int, default=0,
        help=("absolute leg number of this run's first leg, so composite "
              "and increment filenames from consecutive resumed runs do "
              "not collide (default 0)"))
    # -- the ensemble and the analysis ----------------------------------
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--wind-sigma-ms", type=float, default=1.5)
    parser.add_argument("--length-scale-km", type=float, default=150.0)
    parser.add_argument("--horizontal-loc-m", type=float, default=36000.0)
    parser.add_argument("--vertical-loc-m", type=float, default=4000.0)
    parser.add_argument("--rtps-alpha", type=float, default=0.9)
    parser.add_argument("--relaxation", default="rtps",
                        choices=("rtps", "rtpp"),
                        help="which posterior relaxation --rtps-alpha "
                             "drives; see gpuwm.da.letkf.RELAXATION_MODES")
    parser.add_argument("--thin-cells", type=int, default=2)
    parser.add_argument("--err-inflation", type=float, default=1.0)
    parser.add_argument("--memory-budget-mib", type=float, default=6144.0)
    parser.add_argument(
        "--solve-device", default="auto", choices=("auto", "host", "cuda"),
        help=("where the LETKF analysis solves.  'auto' (default) takes "
              "the card when this process can reach one and numpy when it "
              "cannot, and the receipt records which and why.  'host' and "
              "'cuda' are honoured verbatim: pin host to reproduce a "
              "receipt banked on numpy, pin cuda to make a missing card "
              "an error instead of a silent 20x slower run"))
    parser.add_argument(
        "--dump-analysis-bundle", type=Path, default=None,
        help=("copy each leg's ANALYSIS INPUTS -- the staged member "
              "checkpoints, that leg's observation file and the history "
              "file its observations were gridded onto -- into "
              "DIR/leg_NNN, as a bundle tools/da_solve_ab.py can replay.  "
              "That is how the solve-device A/B gets a real leg to "
              "compare on without re-running the forecast that produced "
              "it.  Copies, because the stage directory is usually a "
              "tmpfs the next leg overwrites; budget one ensemble of "
              "checkpoints plus one observation file per leg"))
    parser.add_argument("--no-hotstart", action="store_true")
    parser.add_argument(
        "--no-card-servers", action="store_true",
        help="start a new worker process for every packed member leg instead "
             "of one long-lived card server per card (the default keeps the "
             "interpreter, CUDA context, kernels and proved prepared-cache "
             "arrays across members: gpuwm.da.member_wave.CardServer)")
    # -- spread that does not come from the one initial draw --------------
    parser.add_argument(
        "--spread-repair", choices=("off", "innovation", "blanket"), default="off",
        help="Lane 8 spread maintenance: observed-echo and positive-innovation "
             "gates, targeted missed-echo covariance, and mean-preserving "
             "0.5 K / 0.5 m/s / 5 percent vapour noise. Replaces the two "
             "older noise arms when selected. blanket is a comparison arm")
    parser.add_argument("--spread-repair-z-threshold", type=float, default=25.0)
    parser.add_argument("--spread-repair-adaptive", action="store_true", default=argparse.SUPPRESS,
        help="carry spatial inverse-gamma inflation through accepted ensemble "
             "generations; resume requires the matching saved uncertainty")
    parser.add_argument("--spread-repair-vtsm", action="store_true", default=argparse.SUPPRESS,
        help="retain actual native t-900/t/t+900 forecasts and their observers; "
             "use 3K covariance samples and return K central trajectories")
    parser.add_argument("--spread-vts-ram-gib", type=float, default=argparse.SUPPRESS,
        help="explicit admitted RAM budget for sealed VTSM forecast retention")
    parser.add_argument(
        "--additive-inflation", type=float, default=0.25,
        help="after every analysis add to each member a fresh smooth draw "
             "of the initial perturbation's own fields and length scales at "
             "this fraction of its amplitudes, ensemble mean removed "
             "(gpuwm.da.perturb.additive_inflation). Default 0.25: each "
             "analysis re-adds about 6 percent of the initial variance, "
             "which keeps unobserved regions from collapsing under RTPS "
             "without rivalling the radial-velocity increments. 0 turns "
             "it off")
    parser.add_argument(
        "--echo-noise", type=float, default=1.0,
        help="after every analysis add, where the radar observed echo of "
             "25 dBZ or more (spread over the perturbation's own length "
             "scale), a further smooth draw of the initial perturbation's "
             "fields at this fraction of their amplitudes, ensemble mean "
             "removed (Dowell and Wicker 2009; gpuwm.da.perturb.echo_weight). "
             "The filter can only build echo some member has; this gives "
             "the members differences where the storm is observed. 0 turns "
             "it off")
    parser.add_argument(
        "--boundary-perturbation", type=float, default=1.0,
        help="each member's lateral boundary forcing is perturbed at this "
             "fraction of the initial perturbation's amplitudes "
             "(gpuwm.da.perturb.perturbed_lateral_boundaries), so members "
             "no longer share one boundary and spread does not die toward "
             "the rim. 0 restores the shared boundaries (needed to resume "
             "a generation saved by a tree without member boundaries: the "
             "restart's setup fingerprint hashes the attached tables)")
    parser.add_argument(
        "--boundary-perturbation-hours", type=float, default=6.0,
        help="e-folding time of a member's boundary perturbation between "
             "boundary frames (default 6 h)")
    # -- the moisture / hydrometeor half ---------------------------------
    # All default to zero amplitude, so a caller that does not ask for
    # them gets the wind-only ensemble this driver has always built and
    # the wind-only analysis that goes with it.  The switch that changes
    # the science is --hydrometeors, and it is explicit.
    parser.add_argument(
        "--hydrometeors", action="store_true",
        help="perturb the scheme's moisture and hydrometeor state and "
             "analyse it, instead of u and v alone.  Hydrometeor species "
             "are scaled MULTIPLICATIVELY together with their number and "
             "volume moments, which preserves the drop size distribution "
             "exactly, cannot produce a negative mixing ratio, and cannot "
             "break a moment pair")
    parser.add_argument("--theta-sigma-k", type=float, default=0.5)
    parser.add_argument("--qv-log-sigma", type=float, default=0.05,
                        help="FRACTIONAL, not kg/kg")
    parser.add_argument("--hydro-log-sigma", type=float, default=0.7,
                        help="FRACTIONAL, per species, applied to every "
                             "moment of that species")
    parser.add_argument("--thermo-length-scale-km", type=float, default=60.0)
    parser.add_argument("--thermo-vertical-levels", type=float, default=3.0)
    parser.add_argument("--clip-sigmas", type=float, default=2.5)
    parser.add_argument("--reflectivity-analysis", action="store_true",
                        help="assimilate the merged reflectivity batch "
                             "beside the velocity batches; requires "
                             "--hydrometeors, since reflectivity against "
                             "a wind-only state vector analyses nothing")
    parser.add_argument(
        "--fall-speed", default="auto",
        choices=("auto", "reflectivity", "none"),
        help="radial velocity's vertical term: 'reflectivity' is w minus "
             "the hydrometeor fall speed (rain by Sun and Crook from the "
             "member's own dBZ, snow, graupel and hail at their own Lin "
             "et al. speeds); 'none' is plain w. 'auto' (default) is "
             "reflectivity wherever the scheme has an H_Z(x)")
    parser.add_argument("--z-thin-cells", type=int, default=2)
    parser.add_argument("--z-err-inflation", type=float, default=1.0)
    parser.add_argument("--clear-air-analysis", action="store_true",
                        help="assimilate clear-air 'zero' observations: "
                             "cells the radar measured and found free of "
                             "significant echo. Suppresses spurious "
                             "convection. Requires --hydrometeors and an "
                             "observation file built with a clear-air "
                             "assessment; a file without one is refused "
                             "rather than having zeroes inferred from its "
                             "echo mask")
    parser.add_argument("--precip-analysis", default="off",
                        choices=("off", "clear"),
                        help="the radar half of the GSD cloud analysis on "
                             "the device after the solve "
                             "(gpuwm.da.hydrometeor_analysis), run on every "
                             "ensemble member: 'clear' removes rain, snow "
                             "and graupel and their number moments where "
                             "radar observes no echo, the step NOAA gives "
                             "every ensemble member. The trim and build "
                             "rule is a deterministic analysis's and this "
                             "cycle analyses no deterministic member, so it "
                             "is not offered here. Off by default (opt-in "
                             "until a rain-scored real case); the removed "
                             "water is a declared sink. Needs an "
                             "observation file with a clear-air assessment")
    parser.add_argument("--z0-thin-cells", type=int, default=4,
                        help="clear air is the majority of any volume and "
                             "is smooth, so it starves the filter's rank "
                             "faster than echo does")
    parser.add_argument("--z0-err-inflation", type=float, default=1.0)
    # -- the satellite half ----------------------------------------------
    parser.add_argument(
        "--goes-cwp", type=Path, action="append", default=[],
        help="one gpuwm-obs.goes-grid.v1 file per leg, in leg order, "
             "assimilated as a cloud-water-path batch beside the radar "
             "ones. Build them with tools/obs_goes_grid_build.py. Legs "
             "past the end of this list assimilate radar only. Requires "
             "--hydrometeors and --cwp-vertical-loc-m")
    parser.add_argument("--cwp-thin-cells", type=int, default=2)
    parser.add_argument("--cwp-err-inflation", type=float, default=1.0)
    parser.add_argument("--cwp-horizontal-loc-m", type=float, default=None,
                        help="horizontal localisation for the CWP batch; "
                             "defaults to --horizontal-loc-m")
    parser.add_argument(
        "--cwp-vertical-loc-m", type=float, default=None,
        help="vertical localisation for the CWP batch, in metres. "
             "REQUIRED with --goes-cwp and deliberately has no default: "
             "CWP is a COLUMN INTEGRAL carried at one level, so this "
             "radius is what decides whether the observation acts on the "
             "column it integrated or on a slab. The radar default "
             "(4 km) would assimilate a whole-column measurement as a "
             "4 km-tall one, and in particular would stop a clear-sky "
             "zero from removing model cloud at other heights")
    parser.add_argument(
        "--cwp-ice-species", default="qc,qi,qs",
        help="comma-separated condensate integrated for an ice/mixed "
             "observation. The default is the model's own optical "
             "condensate (gpuwm/core/rrtmgp.py:1097-1098). "
             "'qc,qi' is docs/obs-goes-cwp-operator-spec.md's v1 rule; "
             "which is right is a scoreboard question")
    parser.add_argument("--positivity-policy", default="mean-preserving",
                        choices=("mean-preserving", "clip", "reject",
                                 "none"),
                        help="what a negative analysed mixing ratio "
                             "becomes. Default mean-preserving: the "
                             "undershooting members end at zero and the "
                             "ensemble keeps the filter's mean, where the "
                             "clip added mass at every analysis. "
                             "gpuwm.da.positivity documents what each "
                             "choice costs")
    # -- the radial-velocity dispersion gate (default ON) ------------------
    # gpuwm.da.velocity_dispersion names the breakage (an under-dispersed
    # Vr ensemble writing vapour into theta and vapour) and the measurement
    # behind both defaults; the analysis receipt records the ratios either
    # way.
    from gpuwm.da.velocity_dispersion import (
        DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO,
        DEFAULT_VELOCITY_DISPERSION_RATIO)
    from gpuwm.da.reflectivity_echo import (
        DEFAULT_REFLECTIVITY_OUTLIER_SIGMAS, DEFAULT_REFLECTIVITY_FLOOR_DBZ)
    parser.add_argument(
        "--velocity-dispersion-gate", type=_dispersion_ratio_argument,
        default=DEFAULT_VELOCITY_DISPERSION_RATIO, metavar="RATIO|none",
        help="withhold a radial-velocity batch from theta and vapour in the "
             "columns where its innovation variance exceeds RATIO times its "
             "ensemble plus observation error variance, inside a batch "
             "gated by --velocity-dispersion-batch-gate. Default 2; 'none' "
             "lets Vr update theta and vapour everywhere, which on a "
             "storm-scale first analysis put 6.44 Mt of vapour where the "
             "gate at 2 removes 0.79")
    parser.add_argument(
        "--velocity-dispersion-batch-gate", type=_dispersion_ratio_argument,
        default=DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO, metavar="RATIO|none",
        help="gate a radial-velocity batch only when its innovation "
             "variance over all its gates exceeds RATIO times their "
             "ensemble plus observation error variance. Default 3: the "
             "first analysis's batches measured 4.68 and 3.72, every cycled "
             "one at most 1.97. 'none' gates every batch on its columns "
             "alone, which once cycled withheld Vr from theta and vapour in "
             "a fifth to over half of a storm's columns")
    parser.add_argument(
        "--reflectivity-floor-dbz", type=_reflectivity_floor_argument,
        default=DEFAULT_REFLECTIVITY_FLOOR_DBZ, metavar="DBZ|none",
        help="raise reflectivity observations AND the members' H(x) to this "
             "common floor before differencing (gpuwm.da.reflectivity_echo). "
             "Default 15 dBZ, the engine's echo threshold. 'none' differences "
             "returns from -15 dBZ against the scheme's -35 dBZ H(x) floor, "
             "which on the CONUS 9 km first-light analysis drove +36 dBZ "
             "mean innovations and 2.8x MRMS's strong-echo area one hour "
             "later")
    parser.add_argument(
        "--reflectivity-outlier-sigmas", type=_dispersion_ratio_argument,
        default=DEFAULT_REFLECTIVITY_OUTLIER_SIGMAS, metavar="K|none",
        help="temper reflectivity observations whose innovation lies more "
             "than K standard deviations outside sqrt(spread^2 + sigma_o^2): "
             "their error is raised just enough to sit at K sigma, so the "
             "filter is not asked to extrapolate past the members. Default "
             "3 (DART's outlier threshold); observations inside it are "
             "untouched; 'none' turns it off")
    # -- field rules (gpuwm.da.field_rules, design D1) ------------------------
    from gpuwm.da import field_rules as _fr
    parser.add_argument(
        "--field-rules", choices=_fr.RULES, default=_fr.DEFAULT_RULES,
        help="design: reflectivity and clear air move hydrometeor mass, and "
             "theta/qv only by --z-thermo-weight inside observed echo; no "
             "number or aerosol is analysed. joint: every observation moves "
             "every field (the 10-01 campaign, where radar added 3x more "
             "vapour than condensate)")
    parser.add_argument("--z-thermo-weight", type=float,
                        default=_fr.DEFAULT_Z_THERMO_WEIGHT,
                        help="share of the radar's theta/qv increment kept "
                             "inside observed echo (0 = Z to mass only)")
    parser.add_argument("--z-hydrometeors",
                        action=argparse.BooleanOptionalAction,
                        default=_fr.DEFAULT_Z_HYDROMETEORS,
                        help="reflectivity moves hydrometeor mass "
                             "(--no-z-hydrometeors: KENDA-style)")
    parser.add_argument("--z-qv-cap", type=_reflectivity_floor_argument,
                        default=_fr.DEFAULT_Z_QV_CAP, metavar="KGKG|none",
                        help="cap on |qv| increment from reflectivity, kg/kg")
    parser.add_argument("--number-rediagnosis", choices=_fr.NUMBER_MODES,
                        default=_fr.DEFAULT_NUMBER_MODE,
                        help="rain/ice number after the update: scheme "
                             "relation from analysed mass, background size, "
                             "or off")
    # -- radar observation classes (gpuwm.da.radar_classes, design A1) ------
    from gpuwm.da import radar_classes as _rc
    parser.add_argument(
        "--z-source", choices=("z_mean", "z_obs", "z_max"), default="z_mean",
        help="reflectivity reduction: z_mean (in-cell linear-Z mean, the "
             "default) or z_max/z_obs (the in-cell maximum, which sat 6.6 dB "
             "above the mean on the CONUS case)")
    parser.add_argument(
        "--reflectivity-clear-floor-dbz", type=_reflectivity_floor_argument,
        default=_rc.DEFAULT_CLEAR_FLOOR_DBZ, metavar="DBZ|none",
        help="clear-air floor shared by observation and H(x); weak echo below "
             "it joins the clear class. 'none' keeps the legacy unclassified "
             "batches")
    parser.add_argument(
        "--reflectivity-dead-band", action=argparse.BooleanOptionalAction,
        default=_rc.DEFAULT_DEAD_BAND,
        help="do not assimilate echo between the clear floor and 15 dBZ")
    parser.add_argument(
        "--reflectivity-error-dbz", type=_reflectivity_floor_argument,
        default=_rc.DEFAULT_ECHO_ERROR_DBZ, metavar="DBZ|none",
        help="fixed echo observation error (none = the file's gate-count error)")
    parser.add_argument(
        "--clear-air-error-dbz", type=_reflectivity_floor_argument,
        default=_rc.DEFAULT_CLEAR_ERROR_DBZ, metavar="DBZ|none",
        help="clear-class observation error (none = the file's)")
    parser.add_argument("--reflectivity-level-stride", type=int,
                        default=_rc.DEFAULT_ECHO_LEVEL_STRIDE)
    parser.add_argument("--clear-air-level-stride", type=int,
                        default=_rc.DEFAULT_CLEAR_LEVEL_STRIDE)
    parser.add_argument(
        "--radar-top-pa", type=_reflectivity_floor_argument,
        default=_rc.DEFAULT_TOP_PA, metavar="PA|none",
        help="assimilate no radar observation above this pressure (11 km)")
    parser.add_argument(
        "--reflectivity-huber-c", type=_reflectivity_floor_argument,
        default=None, metavar="C|none",
        help="Huber weighting of echo beyond C standard deviations")
    # -- surface observations (default OFF) -------------------------------
    # METAR/ASOS through the rw_asos seam (gpuwm-obs.asos-surface.v2, and
    # the v1 records written before it).  A quantity is enabled by stating
    # its error standard deviation; a METAR record is hourly-matched, so
    # most sub-hourly cycles legitimately see zero fresh surface reports,
    # and a record decoded with rw_asos --product asos1min carries a report
    # every minute.  Each report enters exactly one analysis, the one
    # nearest the instant it was taken.  gpuwm/da/obs_surface.py documents
    # what the seam can and cannot express.
    parser.add_argument(
        "--surface-obs", type=Path, default=None,
        help="one gpuwm-obs.asos-surface record (v2, or v1) covering the "
             "whole run; each report is routed to the analysis nearest "
             "the instant it was taken.  OFF unless given")
    parser.add_argument(
        "--sfc-t2-sigma-k", type=float, default=None,
        help="assimilate 2 m temperature with this error stddev (K); "
             "representativeness, not instrument precision -- WoFS-like "
             "practice is 1.5-2.5 K at storm-scale grids")
    parser.add_argument(
        "--sfc-td-sigma-k", type=float, default=None,
        help=("enable observed dewpoint via the pinned native Q2/PSFC operator; "
              "stated dewpoint standard deviation in K, with forecast-pressure "
              "proxy recorded. Requires hydrometeor state analysis"))
    parser.add_argument(
        "--sfc-wspd-sigma-ms", type=float, default=None,
        help="assimilate 10 m wind SPEED (the v1 seam carries no "
             "direction) with this error stddev (m/s); H is "
             "hypot(u10, v10) of the member diagnostics")
    parser.add_argument("--sfc-err-inflation", type=float, default=1.0)
    parser.add_argument(
        "--sfc-elev-max-diff-m", type=float, default=200.0,
        help="refuse stations whose table elevation differs from the "
             "model terrain at their gridpoint by more than this")
    parser.add_argument(
        "--sfc-max-age-s", type=float, default=900.0,
        help="refuse reports older (or newer) than this at the analysis")
    parser.add_argument(
        "--sfc-horizontal-loc-m", type=float, default=None,
        help="surface localization override; both --sfc-*-loc-m or "
             "neither (default: the run's --horizontal/vertical-loc-m)")
    parser.add_argument("--sfc-vertical-loc-m", type=float, default=None)
    # -- conventional observations (default OFF) ----------------------------
    # Aircraft, sondes, motion vectors and mesonet as neutral-table rows
    # (gpuwm-obs.table.v2, written by the rw-obs front doors), each row
    # typed by the conventional type table (gpuwm.da.obs_conventional):
    # its operator, batch, error and localization are table data.
    parser.add_argument(
        "--obs-table", type=Path, action="append", default=[],
        help="a gpuwm-obs.table.v2 (or v1) neutral observation table "
             "covering the run; repeatable. Each row is assimilated by the "
             "analysis whose window holds its valid time. OFF unless given")
    parser.add_argument(
        "--obs-type-table", type=Path, default=None,
        help="the conventional type table (default: the shipped "
             "gpuwm/data/da/conventional-obs-types.v1.json)")
    parser.add_argument(
        "--obs-type-override", type=str, default=None,
        help="JSON mapping a batch name (or *) to fields replaced on that "
             "batch's type rows, e.g. horizontal_m or use")
    # -- incremental analysis update (default OFF) --------------------------
    parser.add_argument(
        "--iau-window-seconds", type=float, default=None,
        help="add each member's analysis increment as a forcing over the "
             "first this-many seconds of the next leg (gpuwm.da.iau), "
             "instead of all at once at its start. OFF unless given")
    parser.add_argument(
        "--iau-fields", choices=("dynamics", "all"), default="dynamics",
        help="under --iau-window-seconds: spread only thp, qv, u, v (w, php, "
             "mup) and add hydrometeor increments at the leg start "
             "(dynamics, default), or spread every analysed field (all)")
    parser.add_argument(
        "--iau-mode", choices=IAU_MODES, default=None,
        help="how the increment enters the next leg: oneshot (all at its "
             "start), split (a fraction after each step, gpuwm.da.iau."
             "IncrementalUpdate), 3d (a held tendency inside every RK stage "
             "over --iau-window-seconds from the analysis time, gpuwm.da."
             "iau.IauForcing). Default: split when --iau-window-seconds is "
             "given, else oneshot")
    parser.add_argument(
        "--hydrostatic-rebalance", action=argparse.BooleanOptionalAction,
        default=False,
        help="re-integrate php so the analysed column stays hydrostatic at "
             "its own dry mass (gpuwm.da.hydrostatic); under an IAU mode the "
             "php change is spread with the increment")
    # --balance-perturbations (lane 7) is retired: the member perturbation
    # re-integrates php itself by default (gpuwm.da.perturb mass_balance =
    # "hydrostatic", 2026-10-06), so the leg-0 guard it was has no defect
    # left to cite.
    from gpuwm.da.perturb import WIND_MODES
    parser.add_argument(
        "--wind-perturbation", choices=list(WIND_MODES),
        default="rotational",
        help="how the members' u and v draws relate (gpuwm.da.perturb "
             "wind_mode): 'rotational' (one streamfunction, non-divergent "
             "on the C grid, the default) or 'independent' (the "
             "pre-2026-10-06 unrelated draws, a comparison arm only)")
    parser.add_argument(
        "--maspt-minutes", type=float, default=60.0,
        help="record the mean absolute surface pressure tendency over each "
             "leg's first this-many minutes (gpuwm.da.maspt; reads only); "
             "0 turns it off")
    # -- lane 6 (design E3): windows, members, options, comparison arm ------
    parser.add_argument(
        "--radar-tten-windows", default=None,
        help="with --radar-tten: a tools/radar_tten_windows.py product root "
             "(<root>/<YYYYmmddTHHMMZ>/ref.f32). Each forced leg is split "
             "into 15-minute windows, each heated from its own Level II "
             "volume valid at the window's end (HRRR's four slots per "
             "hour), instead of the leg-end file for the whole leg")
    parser.add_argument(
        "--radar-tten-mode", choices=("heating", "lhn"), default="heating",
        help="heating: NOAA's radar latent heating; lhn: latent heat "
             "nudging, the comparison arm (the model's own heating scaled "
             "by observed over model rain rate, 0.5 to 2)")
    parser.add_argument(
        "--radar-tten-dt-cond", type=float, default=20.0,
        help="minutes over which observed condensate formed (NOAA 20)")
    parser.add_argument(
        "--radar-tten-pbl-extension", action="store_true",
        help="heat down to the PBL top (not NOAA's level-7 floor) where at "
             "least 200 hPa of the column is observed")
    parser.add_argument(
        "--radar-tten-strict-suppression", action="store_true",
        help="exactly zero heating at every point the radar observed as no "
             "echo, after NOAA's smoothing (which bleeds heat into them)")
    parser.add_argument(
        "--radar-tten-perturb-members", action="store_true",
        help="draw dt_cond (15-30 min) and the 28 dBZ thresholds (+/-3 dBZ) "
             "per member, deterministically from --seed")
    parser.add_argument(
        "--radar-tten-control", action="store_true",
        help="force the control trajectory too, with the unperturbed "
             "parameters (design Tier A: every member and the control)")
    # -- radar latent heating (default OFF) ---------------------------------
    # HRRR's mp_tend_radar kernel: on each observed leg every member's
    # theta takes a radar-derived tendency in place of the microphysics
    # heating where the radar covers, built on the card by
    # gpuwm.da.radar_tten from the leg's observation file and the member's
    # own state.  HRRR forces its deterministic pre-forecast and NOT its
    # ensemble; forcing the members here is a declared research
    # divergence, stated in the help, the report and the leg records.
    # Kept LAST in this list so flags added elsewhere merge without
    # touching it.
    parser.add_argument(
        "--radar-tten", action="store_true",
        help="RESEARCH, and the reverse of HRRR's arrangement: force each "
             "ENSEMBLE MEMBER's theta with radar latent heating on every "
             "observed leg whose analysis is applied. HRRR forces only its "
             "deterministic pre-forecast (parm/conus/hrrr_wrfpre.nl:109) "
             "and runs its ensemble members unforced "
             "(parm/hrrrdas/hrrrdas_wrf.nl:101); this cycle has no "
             "analysed deterministic member, so the members are forced. "
             "Each observation is then used twice inside the ensemble "
             "(the forcing, then the analysis at the end of the same "
             "leg), and member spread in theta shrinks where the radar "
             "covers (measured by tools/radar_tten_proof/spread.py). One "
             "slot per leg, built from the leg's observation file (valid "
             "at its end) and the member's state at the start of the leg; "
             "observed clear air gets zero heating, no coverage keeps the "
             "microphysics, which runs under HRRR's companion clamp "
             "mp_tend_lim = 0.07 K/s (hrrr_wrfpre.nl:108) while forced. "
             "The control and the free legs are never forced. OFF unless "
             "given")
    return parser


def _radar_tten_slots(args) -> int:
    """Slots one forced leg holds: one per 15-minute window of the longest
    leg with ``--radar-tten-windows``, else NOAA's one-slot arrangement."""
    if not getattr(args, "radar_tten_windows", None):
        return 1
    durations = (args.leg_durations_seconds
                 or [args.leg_seconds] * max(len(args.obs or ()), 1))
    return max(int(round(float(max(durations)) / 900.0)), 1)


def cycle(stages: list) -> int:
    # Deferred like every other gpuwm import in this driver: the module
    # has to be importable by a bare `python tools/da_cycle_prepared.py
    # --help` from a checkout, and the package lands on sys.path only
    # once the process is actually running the tool.
    import math
    from gpuwm.da import background
    parser = build_parser()
    args = parser.parse_args()
    # ``--physics-profile experiment-config`` stays the string it was
    # given: preflight_prepared_forecast binds that name as the
    # experiment's own physics (EXPERIMENT_CONFIG_PROFILE), and the
    # workers replay this argument record and rebuild the ensemble
    # identity from the same string.  Mapping it to None here made the
    # controller's identity "experiment-config" and every worker's
    # "None"; each member leg was refused (box S 2026-10-06 18:51Z).
    adaptive_spread = bool(getattr(args,"spread_repair_adaptive",False))
    vts_enabled = bool(getattr(args, "spread_repair_vtsm", False))
    if vts_enabled:
        from gpuwm.da.spread_vts import validate_request
        if not args.obs or not args.hydrometeors:
            parser.error("VTSM needs actual observation legs and native hydrometeor covariance fields")
        if args.radar_tten:
            parser.error("VTSM refuses current-volume TTEN because it leaks analysis-time observations into the shifted prior")
        if args.dump_analysis_bundle is not None:
            parser.error("VTSM retained native RAM observers cannot be represented by the central-only replay bundle")
        budget = getattr(args, "spread_vts_ram_gib", None)
        if budget is None or not math.isfinite(budget) or budget <= 0:
            parser.error("VTSM needs --spread-vts-ram-gib with an explicit positive admitted RAM budget")
        args.history_interval_seconds = math.gcd(int(args.history_interval_seconds), 900)
    if adaptive_spread and args.spread_repair == "off":
        parser.error("--spread-repair-adaptive needs --spread-repair so its uncertainty is used and persisted")
    if args.spread_repair != "off" and not (
            args.hydrometeors and args.reflectivity_analysis):
        parser.error("--spread-repair needs --hydrometeors and --reflectivity-analysis "
                     "to distinguish missed observed echo from clear or uncovered air")
    lane6_options = (args.radar_tten_windows, args.radar_tten_mode != "heating",
                     args.radar_tten_dt_cond != 20.0,
                     args.radar_tten_pbl_extension,
                     args.radar_tten_strict_suppression,
                     args.radar_tten_perturb_members, args.radar_tten_control)
    if any(lane6_options) and not args.radar_tten:
        parser.error(
            "a --radar-tten-* option was given without --radar-tten: the "
            "run would be unforced while its command line says how to force")
    if args.radar_tten_windows and not Path(args.radar_tten_windows).is_dir():
        parser.error(f"--radar-tten-windows {args.radar_tten_windows} is not "
                     "a directory")
    if args.radar_tten and not args.obs:
        parser.error(
            "--radar-tten builds its heating from the observation files, "
            "and this run has no --obs: every leg would run unforced while "
            "the report recorded a forced run")
    if args.radar_tten and (args.nest_half_width_km is not None
                            or args.nest_nx is not None
                            or args.nest_ny is not None):
        parser.error(
            "--radar-tten forces the root domain's microphysics heating, "
            "and a nest runs its own: inside the nest the parent would be "
            "heated by radar and the child by its scheme, two answers for "
            "the same columns")
    if args.surface_obs is not None:
        if (args.sfc_t2_sigma_k is None and args.sfc_wspd_sigma_ms is None
                and args.sfc_td_sigma_k is None):
            parser.error(
                "--surface-obs was given but neither --sfc-t2-sigma-k "
                "nor --sfc-wspd-sigma-ms nor --sfc-td-sigma-k states an error; a quantity is "
                "enabled by stating its sigma, and there is no default "
                "sigma on purpose")
        if not args.obs:
            parser.error(
                "--surface-obs joins the radar analysis legs; a "
                "surface-only cycle has no analysis times to join and "
                "is not wired")
    elif (args.sfc_td_sigma_k is not None
          or args.sfc_t2_sigma_k is not None
          or args.sfc_wspd_sigma_ms is not None
          or args.sfc_horizontal_loc_m is not None
          or args.sfc_vertical_loc_m is not None):
        parser.error(
            "surface flags were given without --surface-obs; they would "
            "silently assimilate nothing")
    if args.sfc_td_sigma_k is not None:
        if not args.hydrometeors:
            parser.error("--sfc-td-sigma-k requires --hydrometeors so Q2 can constrain vapor")
        if not np.isfinite(args.sfc_td_sigma_k) or args.sfc_td_sigma_k <= 0:
            parser.error("--sfc-td-sigma-k must be a finite positive standard deviation in K")
        from gpuwm.da.surface_dewpoint import require_native_operator
        require_native_operator()
    if (args.sfc_horizontal_loc_m is None) != (
            args.sfc_vertical_loc_m is None):
        parser.error(
            "--sfc-horizontal-loc-m and --sfc-vertical-loc-m come "
            "together; a localisation is a lens, not a line")
    if args.reflectivity_analysis and not args.hydrometeors:
        parser.error(
            "--reflectivity-analysis needs --hydrometeors: reflectivity "
            "constrains condensate, and against a u/v state vector every "
            "dBZ increment would come from wind-hydrometeor sampling "
            "covariance in a rank-(R-1) ensemble, which is noise")
    if args.clear_air_analysis and not args.hydrometeors:
        parser.error(
            "--clear-air-analysis needs --hydrometeors: a clear-air zero "
            "says condensate is absent, and against a u/v state vector "
            "there is no condensate for it to act on")
    if args.goes_cwp and not args.hydrometeors:
        parser.error(
            "--goes-cwp needs --hydrometeors: cloud water path IS the "
            "column condensate, and against a u/v state vector every CWP "
            "increment would come from wind-condensate sampling covariance "
            "in a rank-(R-1) ensemble, which is noise")
    if args.goes_cwp and args.cwp_vertical_loc_m is None:
        parser.error(
            "--goes-cwp needs an explicit --cwp-vertical-loc-m. CWP is a "
            "column integral carried at one model level, so its vertical "
            "localisation radius is not a tuning detail: it is what decides "
            "whether the observation acts on the column it actually "
            "integrated. Inheriting the radar radius would silently "
            "assimilate a whole-column measurement as a 4 km-tall one, and "
            "would stop an obs-clear zero from removing model cloud that "
            "sits outside that slab")
    if args.goes_cwp and len(args.goes_cwp) > len(args.obs):
        parser.error(
            f"--goes-cwp was given {len(args.goes_cwp)} file(s) but --obs "
            f"has {len(args.obs)}; satellite files are matched to legs by "
            "position and a leg with no radar file runs no analysis at all")

    nest_requested = (args.nest_half_width_km is not None
                      or args.nest_nx is not None
                      or args.nest_ny is not None)
    if nest_requested and args.free_legs <= 0 and not args.obs:
        parser.error(
            "--nest-* needs legs to run on: this command has neither "
            "--obs nor --free-legs")
    if args.nest_members is not None and not nest_requested:
        parser.error("--nest-members without a nest extent "
                     "(--nest-half-width-km or --nest-nx/--nest-ny)")
    if (args.nest_members is not None
            and args.nest_members > args.members):
        parser.error(
            f"--nest-members {args.nest_members} exceeds --members "
            f"{args.members}: the nest is a subset of the parent ensemble")

    # The nested lane still carried the older, stricter form of this rule
    # ("--obs is required", full stop).  The cycling nowcast deliberately
    # relaxed it so a run can be a pure free forecast off a resumed
    # generation, which is how the auto-cycle daemon extends a nowcast past
    # the observations it has.  The relaxed rule is the newer intent and it
    # is a superset, so it wins; the nest refusals above are additive.
    if not args.obs and args.free_legs <= 0:
        parser.error(
            "--obs is required (one observation file per leg). A run "
            "with no observations at all is only meaningful as a free "
            "forecast, which needs --free-legs and, to start from "
            "anything but the prepared background, --resume-ensemble")
    if args.save_ensemble is not None and not args.obs:
        parser.error(
            "--save-ensemble writes the generation at the end of the "
            "last OBSERVED leg, and this run has no --obs: a free-legs "
            "branch is a forecast off the cycle, never the cycle itself")
    if args.leg_number_offset < 0:
        parser.error("--leg-number-offset must be >= 0")
    if len(args.grid_wrfout) != len(args.obs):
        parser.error(
            f"--grid-wrfout was given {len(args.grid_wrfout)} time(s) and "
            f"--obs {len(args.obs)}: every leg's observations are bound to "
            "the georeference they were gridded onto, and pairing them by "
            "position is the caller's statement of which is which")
    legs = len(args.obs) + args.free_legs
    if args.leg_durations_seconds is not None:
        import math
        if (len(args.leg_durations_seconds) != legs or any(
                not math.isfinite(value) or value <= 0
                for value in args.leg_durations_seconds)):
            parser.error(
                "--leg-durations-seconds needs one finite positive duration "
                "per observed or free leg; clocks cannot be dropped or reordered")
    if args.rain_history:
        if not args.save_composites or args.rain_forecast_start_seconds is None:
            parser.error("--rain-history needs --save-composites and an explicit --rain-forecast-start-seconds")
        if not np.isfinite(args.rain_forecast_start_seconds) or args.rain_forecast_start_seconds < 0:
            parser.error("rain issuance clock must be finite and nonnegative")
    if args.packed_smoke and args.forecast_members_per_card == "serial":
        parser.error("--packed-smoke requires the packed member route")
    authority = (args.authority_dir if args.authority_dir is not None
                 else args.prepared_root.parent / "authority")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage_root = Path(args.stage_dir) if args.stage_dir else out / "stage"
    stage = StagedRestarts(stage_root, default=args.stage_dir is None,
                           logs_dir=out / "packed-job-logs")
    stages.append(stage)
    # Registered after the stage: a completed run has removed the ring
    # already (nothing to prune); a stopped one keeps its newest boundary.
    stages.append(RecoveryRing(out / "packed-recovery"))
    report: dict = {"schema": REPORT_SCHEMA, "stability": "experimental",
                    "args": {key: str(value) for key, value
                             in vars(args).items()}, "legs": []}
    report["velocity_dispersion_gate"] = args.velocity_dispersion_gate
    report["velocity_dispersion_batch_gate"] = (
        args.velocity_dispersion_batch_gate)
    print(dispersion_gate_line(args.velocity_dispersion_gate,
                               args.velocity_dispersion_batch_gate),
          flush=True)
    t_total = time.time()

    # ---- imports (CuPy present; model env untouched) -----------------------
    import cupy as cp

    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.preflight import (device_free_and_total_bytes,
                                      local_memory_profile_from_device)
    from gpuwm.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus, execute_experiment)
    from gpuwm.da import (cycle_admission, ensemble_stats, moments,
                          nested_forecast, obsop, perturb, radar_tten)
    from gpuwm.da.hotstart import HotStartConfig, hotstart_increments
    from gpuwm.da.letkf import Localization
    from gpuwm.da.obs_radar import read_document
    from gpuwm.da.obs_surface import (SurfaceObsConfig,
                                      surface_to_gridded_obs)
    from gpuwm.da.obsop_cwp import CwpComposition, checkpoint_cwp_provider
    # RadarAssimilationConfig is NOT imported here: every construction in
    # this driver goes through plan_radar_assimilation, which is what
    # holds the plan-time review and the leg's configuration to one
    # object (audit R-051).
    from gpuwm.da.radar_assimilation import (analysis_device_price,
                                             assimilate_radar_grid,
                                             grid_rotation,
                                             member_earth_winds)
    from gpuwm.da import treatment

    # Which streams this invocation CLAIMS. Derived from the flags once,
    # here, so the proof below is checking the same set the driver acted
    # on. Radial velocity is unconditional: every leg reads a radar grid
    # and the velocity batches are what a cycle is built around.
    enabled_obs_kinds = ["radial_velocity"]
    if args.reflectivity_analysis:
        enabled_obs_kinds.append("reflectivity")
    if args.clear_air_analysis:
        enabled_obs_kinds.append("clear_air_reflectivity")
    if args.surface_obs is not None:
        enabled_obs_kinds.append("surface")
    if args.goes_cwp:
        enabled_obs_kinds.append("cloud_water_path")
    enabled_obs_kinds = tuple(enabled_obs_kinds)
    from datetime import timedelta

    from gpuwm.ensemble.increments import apply_increments
    from gpuwm.ensemble.member import refresh_diagnostics
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.io.restart import (DRIVER_TENDENCY_ATTRS,
                                  RESTART_FORMAT_VERSION,
                                  tree_restart_members)
    from gpuwm.runtime import declared_constant_glw
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.prepared_single_domain_forecast import (
        preflight_prepared_forecast)
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    # The prepared authority was written by the tree that ran the 24 h
    # forecast on this node; between that tree and this branch the physics
    # REGISTRY VOCABULARY moved (maturity "model-validated" ->
    # "wrf-matched-run", and with it the registry digest).  Every physics
    # selector, component and profile id is identical -- verified below
    # field by field -- so this driver tolerates exactly those two
    # vocabulary fields, refuses anything else, and records the
    # divergence in its report.  Driver-side accommodation only; the
    # repo's preflight stays strict.
    import gpuwm.prepared_single_domain_forecast as psdf
    _orig_check = psdf._validate_front_door_physics_proof
    vocabulary_divergence: dict = {}

    def _tolerant_check(proof, *, source, profile, cfg):
        try:
            return _orig_check(proof, source=source, profile=profile,
                               cfg=cfg)
        except ValueError as error:
            if (not args.tolerate_physics_vocabulary_drift
                    or "physics selection differs" not in str(error)):
                raise
            selected = dict(proof["physics"])
            expected = dict(psdf.validate_single_domain_physics_profile(
                profile, config=cfg,
                expert_acknowledgements=tuple(
                    selected["acknowledgements"]),
                acknowledgement_provenance=selected[
                    "acknowledgement_provenance"]))
            diverged = {}
            for key in PHYSICS_VOCABULARY_FIELDS:
                if selected.get(key) != expected.get(key):
                    diverged[key] = {"proof": selected.pop(key, None),
                                     "branch": expected.pop(key, None)}
            if selected != expected:
                raise
            vocabulary_divergence.update(diverged)
            print(f"physics proof vocabulary divergence tolerated: "
                  f"{diverged}", flush=True)
            return dict(proof["physics"])

    psdf._validate_front_door_physics_proof = _tolerant_check

    inputs = preflight_prepared_forecast(
        source=args.source, prepared_root=args.prepared_root,
        proof_sha256=args.proof_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
        prepared_content_sha256=args.prepared_content_sha256,
        experiment_config=authority / "experiment.toml",
        wps_namelist=authority / "namelist.wps",
        physics_profile=args.physics_profile,
        run_seconds=args.run_seconds,
        history_interval_seconds=args.history_interval_seconds)
    psdf._validate_front_door_physics_proof = _orig_check
    report["physics_proof_vocabulary_divergence"] = vocabulary_divergence
    # The contract a leg boundary rides on, recorded beside the
    # per-trajectory restart entries: the state arrays the host mirror
    # hands the filter, and the restart owner's own format and driver
    # inventory that the checkpoint sets carry on top of them.
    report["restart_contract"] = {
        "owner": "gpuwm.io.restart",
        "format_version": int(RESTART_FORMAT_VERSION),
        "state_serialized_attrs": list(STATE_SERIALIZED_ATTRS),
        "driver_tendency_attrs": list(DRIVER_TENDENCY_ATTRS),
    }
    exp = inputs.experiment
    cfg = exp.root.run
    dt = float(cfg.dt)
    leg_seconds = float(args.leg_seconds)
    final_leg_seconds = (float(args.final_leg_seconds)
                         if args.final_leg_seconds is not None
                         else leg_seconds)
    free_leg_seconds = (float(args.free_leg_seconds)
                        if args.free_leg_seconds is not None else None)
    n_obs = len(args.obs)

    def leg_length(index: int) -> float:
        """How long leg ``index`` runs for.

        A free leg with its own declared length uses it; otherwise the
        rule is the one this driver has always had -- every leg is
        ``--leg-seconds`` except the last, which may be longer.
        """

        if args.leg_durations_seconds is not None:
            return float(args.leg_durations_seconds[index])
        if free_leg_seconds is not None and index >= n_obs:
            return free_leg_seconds
        return final_leg_seconds if index == legs - 1 else leg_seconds
    print(f"preflight OK: {cfg.nx}x{cfg.ny}x{cfg.nz} dt={dt} "
          f"mp={cfg.mp_physics} hyps={cfg.hypsometric_opt}", flush=True)
    _onset = limiter_onset(cfg)
    if _onset is not None:
        print(f"vertical limiter: thinnest layer "
              f"{_onset['thinnest_layer_m']:.0f} m at "
              f"{_onset['thinnest_layer_height_m']:.0f} m, so w_damping "
              f"starts limiting above {_onset['w_onset_ms']:.1f} m/s at "
              f"dt={dt}", flush=True)

    # ---- plan review for the DA door ------------------------------------
    # The analysis configuration is BUILT here, before an ensemble member
    # is perturbed and long before leg 0 integrates anything, and every
    # refusal the filter states is therefore raised here: the active
    # scheme's radar H(x) route, the clear-air floor, the positivity
    # policy, the localisation and inflation knobs.  Audit
    # R-051 moved those refusals out of the first analysis and into the
    # configuration; this call is what makes the configuration exist at
    # plan time rather than at the first analysis seam, where a whole
    # ensemble integration has already been spent.  The leg builds its
    # own through the same function, with the fields ensemble spread
    # leaves it (never more than these); the memory admission prices
    # every observed leg's analysis with this one, the widest any leg
    # can run.
    planned_analysis = plan_radar_assimilation(
        args, cfg.mp_physics,
        analysis_fields=planned_analysis_fields(args, cfg.mp_physics),
        cwp=bool(args.goes_cwp))

    perturb_fields = [
        {"name": "u", "amplitude": args.wind_sigma_ms,
         "length_scale_km": args.length_scale_km},
        {"name": "v", "amplitude": args.wind_sigma_ms,
         "length_scale_km": args.length_scale_km},
    ]
    perturb_species: list[dict] = []
    if args.hydrometeors:
        # The species the SCHEME advances, never a list typed here.  qv
        # is every scheme's mass_only field and has no number moment, so
        # it goes through the field path with the multiplicative mode;
        # the rest go through the species path, which scales each one's
        # mass, number and volume moments by a single common factor.
        scheme = moments.scheme_moments(int(cfg.mp_physics))
        perturb_fields += [
            {"name": "theta", "amplitude": args.theta_sigma_k,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels},
            {"name": "qv", "amplitude": args.qv_log_sigma,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels,
             "mode": "lognormal", "clip_sigmas": args.clip_sigmas},
        ]
        perturb_species = [
            {"mass_field": name, "amplitude": args.hydro_log_sigma,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels,
             "clip_sigmas": args.clip_sigmas,
             "threshold_kg_kg": scheme.q_threshold}
            for name in scheme.mass_fields
            if name in perturb.SUPPORTED_SPECIES]
        report["perturbation"] = {
            "scheme": scheme.name, "mp_physics": scheme.mp_physics,
            "species": [spec["mass_field"] for spec in perturb_species]}
    cfg_perturb = perturb.PerturbationConfig.from_mapping({
        "dx_km": float(cfg.dx) / 1000.0, "dy_km": float(cfg.dy) / 1000.0,
        "rim_width": 5,
        "fields": perturb_fields,
        "species": perturb_species,
        "wind_mode": str(getattr(args, "wind_perturbation", "rotational")),
        "hypsometric_opt": int(cfg.hypsometric_opt),
    })
    hot_cfg = HotStartConfig()
    if args.additive_inflation < 0.0 or args.boundary_perturbation < 0.0:
        raise SystemExit("--additive-inflation and --boundary-perturbation "
                         "are fractions of the initial amplitudes and "
                         "cannot be negative")
    # Each member's own lateral boundaries are built and attached by the
    # member leg itself (tools.da_member_leg, the same deterministic draw
    # every leg, so the restart's setup fingerprint is stable); its record
    # comes back in the trajectory's leg record.  The control keeps the
    # shared tables.

    # ---- how each trajectory's background was built --------------------
    # Planned BEFORE any GPU work, from the perturbation configuration
    # this run actually assembled, so an ensemble that would be N
    # identical copies of the control refuses here instead of consuming
    # a card for an hour and reporting zero spread.  The result is one
    # record per trajectory, and it goes into the report verbatim: a
    # skill comparison can then attribute a difference to the background
    # rather than guess at it.
    try:
        member_plan = background.plan_member_backgrounds(
            control_name=CONTROL, members=int(args.members),
            seed=int(args.seed), perturbed_fields=perturb_fields,
            perturbed_species=perturb_species)
    except background.BackgroundError as error:
        # A refusal a caller can act on, in this driver's own idiom --
        # the boundary-horizon refusal below reads the same way.
        raise SystemExit(str(error)) from None
    report["background"] = background.background_receipt(
        source=args.source, cycle=None, members=member_plan,
        prepared_content_sha256=str(args.prepared_content_sha256),
        notes={
            "prepared_root": str(args.prepared_root),
            "forcing_hours": list(inputs.forcing_hours),
            "initial_valid_time": exp.start_time.isoformat(),
            "run_seconds": float(args.run_seconds),
            # Taken from the hash-bound proof, not from a flag: these
            # are what the preparation actually fetched, so a reader can
            # tell how old the first guess was without trusting the
            # command line that started this process.
            "source_cycle": inputs.proof.get("source_cycle"),
            "source_forecast_hours": inputs.proof.get(
                "source_forecast_hours"),
            # Stated rather than implied: the members share this case's
            # lateral boundary conditions, so the rim taper is what keeps
            # the perturbation legal and spread decays toward the rim by
            # construction (gpuwm/da/perturb.py documents the whole list).
            "shared_lateral_boundaries": (
                float(args.boundary_perturbation) == 0.0),
            "member_lateral_boundaries": {
                "scale": float(args.boundary_perturbation),
                "time_scale_hours": float(
                    args.boundary_perturbation_hours),
                "rule": "gpuwm.da.perturb.perturbed_lateral_boundaries; "
                        "the control keeps the shared tables"},
            "additive_inflation_scale": float(args.additive_inflation),
        })
    print("background: " + json.dumps({
        "source": report["background"]["source"],
        "initial_hydrometeors": report["background"]["initial_hydrometeors"],
        "members": report["background"]["ensemble"]["member_count"],
        "construction": report["background"]["ensemble"]["construction"],
    }, sort_keys=True), flush=True)

    trajectories = [CONTROL] + list(range(args.members))
    setup_arrays: dict | None = None
    #: The leg-end HOST MIRROR of each trajectory's serialised state, for
    #: the filter: the analysis reads members from it, prices their
    #: spread from it and bounds increments against it.  It is not the
    #: leg join -- ``restarts`` below is.
    snapshots: dict = {name: None for name in trajectories}
    #: The root member of each trajectory's leg-end tree checkpoint set,
    #: written by the restart owner and restored by it at the next leg's
    #: start.  This is what joins one leg to the next.
    restarts: dict = {name: None for name in trajectories}
    pending: dict = {name: None for name in trajectories}
    hot_pending: dict = {}

    # ---- the ensemble this run starts from ---------------------------------
    # Either a fresh perturbed ensemble off the prepared background (leg 0
    # perturbs), or a generation a previous process wrote (leg 0 restores
    # and applies that generation's unapplied analysis).  The identity is
    # what makes the second safe: everything that changes the meaning of
    # the stored arrays is compared before a single array is read.
    identity = ens_state.EnsembleIdentity(
        members=int(args.members), nx=int(cfg.nx), ny=int(cfg.ny),
        nz=int(cfg.nz), dt_s=dt, mp_physics=int(cfg.mp_physics),
        physics_profile=str(args.physics_profile),
        prepared_content_sha256=str(args.prepared_content_sha256))
    resumed_from: dict | None = None
    #: The trajectories whose resumed checkpoint set carries a child,
    #: keyed as this driver keys its trajectories.  Empty when the
    #: generation has none, which leaves leg 0 building the child from
    #: the analysed parent exactly as a fresh run does.
    resumed_nested: list = []
    resumed_nest_receipt = None
    base_seconds = 0.0
    if args.resume_ensemble is not None:
        restarts, pending, resumed_from = ens_state.read_generation(
            args.resume_ensemble, identity)
        resumed_nested = ens_state.nested_trajectories(resumed_from)
        resumed_nest_receipt = (resumed_from.get("nest")
                                if resumed_nested else None)
        base_seconds = float(resumed_from["elapsed_seconds"])
        report["resumed_from"] = {
            "directory": str(args.resume_ensemble),
            "written": resumed_from["written"],
            "elapsed_seconds": base_seconds,
            "leg_number": resumed_from["leg_number"],
            "valid_time": resumed_from.get("valid_time"),
            "trajectories_with_unapplied_analysis": sorted(
                key for key, entry
                in resumed_from["trajectories"].items()
                if entry.get("pending")),
            "trajectories_with_a_child": sorted(
                str(key) for key in resumed_nested),
        }
        print(f"resumed ensemble from {args.resume_ensemble} at "
              f"{base_seconds:.0f} s elapsed "
              f"(leg {resumed_from['leg_number']})", flush=True)
    report["leg_number_offset"] = int(args.leg_number_offset)
    report["base_elapsed_seconds"] = base_seconds

    # The prepared case's lateral boundary conditions only cover the
    # hash-bound run length.  Integrating past it would read boundary
    # data that does not exist, so it is a refusal here rather than a
    # surprise inside the integrator -- a resumed daemon hits this edge
    # eventually by construction, and the accurate answer is a new case.
    span_end = base_seconds + sum(leg_length(i) for i in range(legs))
    if span_end > float(args.run_seconds) + 1e-6:
        raise SystemExit(
            f"this run would integrate to {span_end:.0f} s but the "
            f"prepared case is bound to {float(args.run_seconds):.0f} s "
            "of boundary data; shorten the legs or prepare a case on a "
            "newer background")
    if vts_enabled:
        last_analysis = base_seconds+sum(leg_length(i) for i in range(len(args.obs)))
        if last_analysis+900 > float(args.run_seconds):
            raise SystemExit("VTSM future continuation exceeds the hash-bound native LBC horizon; prepare a longer forcing root")
    #: The absolute leg number of a within-run leg index.
    def leg_number(index: int) -> int:
        return int(args.leg_number_offset) + index

    #: The last leg that carries an observation; the ensemble generation
    #: is written there and nowhere else.
    save_at_leg = len(args.obs) - 1 if args.save_ensemble else None
    #: Per-member leg-end dBZ, diagnosed on the device from the state
    #: the restart set and the host mirror were taken from.  This is
    #: H_Z(x) for the filter.
    member_dbz: dict = {}
    #: Per-member leg-end 2m/10m diagnostics off the live physics driver
    #: -- the surface H(x), taken at the SAME point as member_dbz, so the
    #: filter sees the diagnostics of the very state it analyses.
    member_sfc: dict = {}
    thb_host = None

    # ---- the fine nest over the free forecast ------------------------
    #
    # The child is DERIVED here, once, so an inadmissible nest is a
    # preflight refusal rather than a crash six legs in.  Which
    # trajectories carry it is a separate, explicit decision: the parent
    # carries the whole ensemble and the nest is deliberately cheap.
    nest_geometry = None
    nest_child_dc = None
    nest_trajectories: tuple = ()
    #: Every leg the child runs on, which is every leg there is.  It was
    #: the free legs alone until 2026-09-18; see nested_forecast.nest_legs
    #: for why that made the child a creature of the fork.
    nest_leg_numbers = nested_forecast.nest_legs(
        observed_legs=len(args.obs), free_legs=int(args.free_legs))
    if nest_requested:
        nest_members = (nested_forecast.DEFAULT_NEST_MEMBERS
                        if args.nest_members is None
                        else int(args.nest_members))
        nest_geometry = nested_forecast.NestGeometry(
            ratio=int(args.nest_ratio),
            nx=args.nest_nx, ny=args.nest_ny,
            half_width_km=args.nest_half_width_km,
            i_parent_start=args.nest_i_parent_start,
            j_parent_start=args.nest_j_parent_start,
            history_interval_s=args.nest_history_interval_s,
            members=nest_members)
        nest_child_dc = nested_forecast.nest_domain_config(
            exp, nest_geometry,
            acknowledgements=tuple(args.nest_acknowledge))
        # The acoustic rule the nest's own ground needs (A181), read off
        # the terrain it integrates before any leg is wired: every leg's
        # child is derived from this configuration.
        nest_child_dc, nest_acoustic = nested_forecast.nest_acoustics(
            exp, nest_child_dc, parent_terrain=inputs.static["HGT_M"],
            parent_grid=inputs.grid)
        nest_admissibility = nested_forecast.validate_nest_admissibility(
            nest_child_dc.run, parent_run=cfg,
            acknowledgements=tuple(args.nest_acknowledge))
        nest_trajectories = tuple(
            [CONTROL] + list(range(nest_members)))
        report["nest"] = nested_forecast.nested_forecast_receipt(
            geometry=nest_geometry,
            exp=nested_forecast.nested_experiment(exp, nest_child_dc),
            child_dc=nest_child_dc, admissibility=nest_admissibility,
            land_receipt={
                "terrain_policy": nested_forecast.TERRAIN_POLICY,
                "land_policy": nested_forecast.LAND_POLICY},
            legs=list(nest_leg_numbers),
            nest_members=nest_members)
        report["nest"]["trajectories"] = [str(name)
                                          for name in nest_trajectories]
        from gpuwm.acoustic_adaptation import acoustic_receipt
        report["nest"]["acoustic_substeps"] = acoustic_receipt(nest_acoustic)
        child_run = nest_child_dc.run
        print(f"nest: d{nest_child_dc.grid_id:02d} {child_run.nx}x"
              f"{child_run.ny}x{child_run.nz} dx={child_run.dx:g} "
              f"dt={child_run.dt:g} over legs "
              f"{nest_leg_numbers[0]}..{nest_leg_numbers[-1]} for "
              f"{len(nest_trajectories)} trajector"
              f"{'y' if len(nest_trajectories) == 1 else 'ies'}", flush=True)

    #: When each nesting trajectory's child was born, in seconds from the
    #: run start.  The child is built from the parent on its FIRST leg
    #: and activates at that leg's boundary; every later leg rebuilds its
    #: configuration on the same instant so the restart owner finds the
    #: calendar it wrote, and restores the child from the trajectory's
    #: own checkpoint set, so the fine-scale structure it develops
    #: survives a leg boundary instead of being flattened back to a
    #: parent interpolation every fifteen minutes.  ``None`` until born.
    nest_birth: dict = {name: None for name in nest_trajectories}
    if resumed_nested:
        if nest_child_dc is None:
            raise SystemExit(
                "the resumed generation carries a child and this run "
                "passed no --nest-*: continuing without it would throw "
                "away a child that has been running since "
                f"{resumed_from['written']}, silently")
        differences = [
            f"{field}: generation "
            f"{(resumed_nest_receipt or {}).get(field)} vs run {value}"
            for field, value in (
                ("nx", int(nest_child_dc.run.nx)),
                ("ny", int(nest_child_dc.run.ny)),
                ("nz", int(nest_child_dc.run.nz)),
                ("i_parent_start", int(nest_child_dc.i_parent_start)),
                ("j_parent_start", int(nest_child_dc.j_parent_start)),
                ("parent_grid_ratio",
                 int(nest_child_dc.parent_grid_ratio)))
            if int((resumed_nest_receipt or {}).get(field, -1)) != value]
        if differences:
            raise SystemExit(
                "the resumed generation's child is not this run's "
                "child, and restoring its arrays onto a differently "
                "placed domain would produce a plausible-looking "
                "forecast of nowhere: " + "; ".join(differences))
        for name in resumed_nested:
            if name not in nest_birth:
                raise SystemExit(
                    f"the resumed generation carries a child on "
                    f"trajectory {name!r} and this run's nest covers "
                    f"{sorted(str(n) for n in nest_trajectories)}; "
                    "--nest-members has to be at least what the "
                    "generation was written with")
            # The child's activation epoch is read off its own
            # checkpoint member, the one place the calendar the restart
            # owner will check it against is written.
            nest_birth[name] = restart_child_birth_seconds(
                restarts[name], grid_id=int(nest_child_dc.grid_id),
                start_time=exp.start_time)
        report["nest"]["resumed_trajectories"] = sorted(
            str(name) for name in resumed_nested)
        report["nest"]["resumed_birth_seconds"] = {
            str(name): nest_birth[name] for name in resumed_nested}

    def nests_this_leg(leg: int, name) -> bool:
        return (nest_child_dc is not None and leg in nest_leg_numbers
                and name in nest_trajectories)

    def release_device_memory() -> None:
        """Collect a finished trajectory and return its pool blocks.

        Releases only what nothing references any more: a trajectory's
        owners are unreachable here because they were locals of
        ``run_trajectory``, which has returned.
        """
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    # ---- the legs ------------------------------------------------------------

    leg_starts = []
    _cursor = base_seconds
    for _index in range(legs):
        leg_starts.append(_cursor)
        _cursor += leg_length(_index)

    # ---- surface observations (rw_asos seam), default OFF -------------------
    surface_cfg = None
    surface_schedule = None
    if args.surface_obs is not None:
        from datetime import datetime as _datetime  # noqa: PLC0415
        from datetime import timedelta as _timedelta  # noqa: PLC0415

        if not isinstance(exp.start_time, _datetime):
            raise SystemExit(
                "--surface-obs needs the experiment's absolute start time "
                f"to route reports to analyses, and exp.start_time is "
                f"{type(exp.start_time).__name__}")
        surface_localization = None
        if args.sfc_horizontal_loc_m is not None:
            surface_localization = Localization(
                horizontal_m=float(args.sfc_horizontal_loc_m),
                vertical_m=float(args.sfc_vertical_loc_m))
        surface_cfg = SurfaceObsConfig(
            temperature_error_k=args.sfc_t2_sigma_k,
            wind_speed_error_ms=args.sfc_wspd_sigma_ms,
            dewpoint_error_k=args.sfc_td_sigma_k,
            error_inflation=float(args.sfc_err_inflation),
            elevation_max_diff_m=float(args.sfc_elev_max_diff_m),
            max_age_seconds=float(args.sfc_max_age_s),
            temperature_localization=surface_localization,
            wind_localization=surface_localization,
            dewpoint_localization=surface_localization)
        #: One analysis time per OBSERVED leg -- the same t_end the radar
        #: analysis runs at.  The adapter routes each hourly report to
        #: exactly one of these.
        surface_schedule = [
            exp.start_time + _timedelta(seconds=leg_starts[i]
                                        + leg_length(i))
            for i in range(len(args.obs))]
        report["surface_observations"] = {
            "record": str(args.surface_obs),
            "t2_sigma_k": args.sfc_t2_sigma_k,
            "td_sigma_k": args.sfc_td_sigma_k,
            "wspd_sigma_ms": args.sfc_wspd_sigma_ms,
            "analysis_times": [t.isoformat() for t in surface_schedule],
        }

    # ---- conventional observations (neutral tables), default OFF ---------
    conv_table = conv_rows = conv_schedule = None
    if args.obs_table:
        from datetime import datetime as _datetime  # noqa: PLC0415
        from datetime import timedelta as _timedelta  # noqa: PLC0415

        from gpuwm.da.obs_conventional import read_tables, read_type_table

        if not isinstance(exp.start_time, _datetime):
            raise SystemExit(
                "--obs-table needs the experiment's absolute start time to "
                "route reports to analyses, and exp.start_time is "
                f"{type(exp.start_time).__name__}")
        overrides = (json.loads(args.obs_type_override)
                     if args.obs_type_override else None)
        conv_table = read_type_table(args.obs_type_table,
                                     overrides=overrides)
        conv_rows = read_tables(args.obs_table)
        conv_schedule = [
            exp.start_time + _timedelta(seconds=leg_starts[i]
                                        + leg_length(i))
            for i in range(len(args.obs))]
        report["conventional_observations"] = {
            "tables": list(conv_rows.receipts),
            "rows": len(conv_rows),
            "type_table": {"path": conv_table.path,
                           "sha256": conv_table.sha256},
            "overrides": overrides,
            "analysis_times": [t.isoformat() for t in conv_schedule],
        }
    if args.iau_window_seconds is not None:
        if not (args.iau_window_seconds > 0):
            raise SystemExit("--iau-window-seconds must be positive")
        report["iau"] = {"window_seconds": float(args.iau_window_seconds),
                         "fields": args.iau_fields,
                         "form": "forward, from each analysis time"}
    try:
        args.iau_mode = resolve_iau_mode(args.iau_mode,
                                         args.iau_window_seconds)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if "iau" in report:
        report["iau"]["mode"] = args.iau_mode
    report["hydrostatic_rebalance"] = bool(args.hydrostatic_rebalance)
    if args.maspt_minutes < 0:
        raise SystemExit("--maspt-minutes must be zero or positive")

    # ---- the fit decision, before the first upload ----------------------
    # One decision for the whole run, taken before the first observation
    # upload and the first restore: every trajectory of every leg is the
    # same forecast, a nesting one is the largest, and each is released
    # before the next is wired, so the largest trajectory is what the
    # card has to hold (gpuwm.da.cycle_admission says what it counts).
    # Each observed leg's analysis is priced from its own observation
    # file, with the plan-time configuration reviewed above.
    analysis_prices = []
    for obs_leg, obs_name in enumerate(args.obs):
        obs_file = Path(obs_name)
        if not obs_file.is_file():
            raise FileNotFoundError(
                f"leg {obs_leg}: no observation file at {obs_file}")
        obs_grid = TargetGrid.from_wrfout(Path(args.grid_wrfout[obs_leg]))
        analysis_prices.append(analysis_device_price(
            planned_analysis, members=int(args.members), grid=obs_grid,
            document=read_document(obs_file, expected_grid=obs_grid),
            extra_localizations=(
                (() if surface_cfg is None
                 else tuple(surface_cfg.batch_localizations()))
                + (() if conv_table is None
                   else conv_table.batch_localizations()))))
    analysis_price = cycle_admission.worst_analysis(analysis_prices)
    unsolvable = cycle_admission.unsolvable_analysis_message(analysis_price)
    if unsolvable is not None:
        raise SystemExit(unsolvable)
    if args.radar_tten and int(cfg.mp_physics) == 0:
        raise SystemExit(
            "--radar-tten replaces the microphysics heating increment, and "
            "this case runs mp_physics = 0: no microphysics call would read "
            "the heating, and every forced member would refuse at its first "
            "leg after the ensemble was already built")
    mass_shape = (int(cfg.nz), int(cfg.ny), int(cfg.nx))
    from gpuwm.da.hydrostatic import loading_masses_for_scheme
    admission = cycle_admission.price_cycle(
        (nested_forecast.nested_experiment(exp, nest_child_dc)
         if nest_trajectories else exp),
        forcing_intervals=max(1, len(inputs.forcing_hours) - 1),
        observation_points=(int(cfg.nz) * int(cfg.ny) * int(cfg.nx)
                            if args.obs and not args.no_hotstart else 0),
        perturbation_bytes=(
            perturb.device_working_bytes(
                cfg_perturb, mass_shape,
                plan_work_bytes=perturb.fft_plan_work_bytes(
                    cfg_perturb, mass_shape, cp),
                loading_masses=loading_masses_for_scheme(cfg.mp_physics))
            if resumed_from is None and int(args.members) > 0 else 0),
        profile=local_memory_profile_from_device(cp),
        analysis=analysis_price,
        radar_tten_points=(int(cfg.nz) * int(cfg.ny) * int(cfg.nx)
                           if args.radar_tten else 0),
        radar_tten_slots=_radar_tten_slots(args))
    free_bytes, _total_bytes = device_free_and_total_bytes()
    try:
        admission = cycle_admission.admit_cycle(admission,
                                                free_bytes=free_bytes)
    except cycle_admission.CycleMemoryRefused as error:
        report["memory_admission"] = error.admission.receipt()
        (out / "cycle-report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        raise SystemExit(str(error)) from None
    report["memory_admission"] = admission.receipt()
    print(f"memory admission: {admission.required_bytes:,} bytes for the "
          f"largest trajectory and the analysis "
          f"({admission.analysis_route or 'none on the card'}) within "
          f"{admission.budget_bytes:,} of {admission.free_bytes:,} free",
          flush=True)
    if args.radar_tten:
        report["radar_tten"] = {
            "arrangement": "ensemble members forced on observed legs whose "
                           "analysis is applied; the control and free legs "
                           "unforced",
            "declared_divergence": (
                "HRRR forces only its deterministic pre-forecast "
                "(parm/conus/hrrr_wrfpre.nl:109, mp_tend_radar = 1) and runs "
                "its ensemble members unforced (parm/hrrrdas/"
                "hrrrdas_wrf.nl:101, mp_tend_radar = 0). This cycle forces "
                "the members instead: each observation is used twice inside "
                "the ensemble (the forcing, then the analysis at the end of "
                "the same leg) and member spread in theta shrinks where the "
                "radar covers. Research line, not HRRR's product."),
            "spread_measurement": "tools/radar_tten_proof/spread.py",
            "mp_tend_lim_k_per_s": radar_tten.HRRR_MP_TEND_LIM,
            "case_mp_tend_lim_k_per_s": float(cfg.mp_tend_lim),
            "mp_tend_lim_rule": (
                "while a member is forced its microphysics runs under "
                "HRRR's companion clamp (parm/conus/hrrr_wrfpre.nl:108); "
                "the control and unforced legs keep the case's"),
            "mode": args.radar_tten_mode,
            "windows": (
                {"root": str(args.radar_tten_windows), "minutes": 15,
                 "rule": "each window heated from its own gridded Level II "
                         "volume valid at the window's end"}
                if args.radar_tten_windows else
                "one slot per leg from the leg-end observation file"),
            "dt_cond_min": args.radar_tten_dt_cond,
            "pbl_extension": bool(args.radar_tten_pbl_extension),
            "strict_suppression": bool(args.radar_tten_strict_suppression),
            "members_perturbed": (
                dict(radar_tten.MEMBER_PERTURBATION, seed=args.seed)
                if args.radar_tten_perturb_members else False),
            "control_forced": bool(args.radar_tten_control),
        }
        if args.radar_tten_control:
            report["radar_tten"]["arrangement"] = (
                "every member and the control forced on observed legs whose "
                "analysis is applied (design Tier A); free legs unforced")

    packed_forecasts = args.forecast_members_per_card != "serial"
    run_leg_route = run_member_leg_entry()
    if args.iau_mode == "4d":
        # Each refusal names what 4D-IAU would silently lose.
        if packed_forecasts:
            raise SystemExit(
                "--iau-mode 4d runs on --forecast-members-per-card serial: "
                "the packed transport carries neither the mid-window "
                "restart nor the rewound leg start, so a packed member "
                "would get a forward window labelled centred")
        if nest_requested:
            raise SystemExit(
                "--iau-mode 4d with a nest: the child's carry-down of a "
                "rewound, two-part leg is not wired, and the child would "
                "keep a trajectory its parent left")
        if args.resume_ensemble is not None:
            raise SystemExit(
                "--iau-mode 4d with --resume-ensemble: the generation "
                "carries no restart half a window before its analysis, "
                "so the first leg could not be centred")
        if args.save_ensemble is not None:
            raise SystemExit(
                "--iau-mode 4d with --save-ensemble: a generation written "
                "here would carry no mid-window restart for the next run")
        run_leg_route = Iau4dLegRunner(
            run_leg_route, window_seconds=float(args.iau_window_seconds))
        report["iau"]["form"] = ("centred on each analysis time: the leg "
                                 "before it is split at t - W/2 and the "
                                 "leg after it restarts there")
    forecast_cards, forecast_pack, forecast_peak = [], 1, 0
    if packed_forecasts:
        from gpuwm.da.member_wave import query_cards, GIB
        if args.packed_smoke:
            if int(args.members) != 4 or max(int(cfg.nx), int(cfg.ny)) > 65 or sum(leg_length(i) for i in range(legs)) > 7200:
                raise SystemExit("packed smoke requires four members, at most 65x65 cells and two hours of model time")
        forecast_cards = query_cards(args.forecast_device_uuids)
        if args.packed_smoke and len(forecast_cards) != 1:
            raise SystemExit("packed smoke runs on one physical card; name it with --forecast-device-uuids")
        forecast_only = cycle_admission.price_cycle(
            (nested_forecast.nested_experiment(exp, nest_child_dc)
             if nest_trajectories else exp),
            forcing_intervals=max(1, len(inputs.forcing_hours) - 1),
            observation_points=(int(cfg.nz)*int(cfg.ny)*int(cfg.nx)
                                if args.obs and not args.no_hotstart else 0),
            perturbation_bytes=admission.perturbation_bytes,
            profile=local_memory_profile_from_device(cp), analysis=None,
            radar_tten_points=(int(cfg.nz)*int(cfg.ny)*int(cfg.nx)
                               if args.radar_tten else 0),
            radar_tten_slots=_radar_tten_slots(args))
        forecast_peak = forecast_only.required_bytes
        try:
            forecast_pack = packed_members_per_card(
                args.forecast_members_per_card, forecast_cards, forecast_peak)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        report["packed_forecast_admission"] = {
            "forecast_only": forecast_only.receipt(), "cards": forecast_cards,
            "members_per_card": forecast_pack, "mps_required": forecast_pack == 4,
            "decode_threads": 12, "decode_total_cap_bytes": 160*GIB,
            "host_reserve_bytes": (4 if args.packed_smoke else 48)*GIB,
            "smoke": bool(args.packed_smoke)}

    from tools.da_member_leg import MemberLegContext, run_member_leg
    from gpuwm.da.member_roster import ControlLaunch, forecast_roster

    leg_input_cache: dict = {}

    def leg_inputs(leg):
        """One leg's clocks and observation-side inputs, read once.

        A function rather than the top of the leg loop because the packed
        route starts the NEXT leg's control during this leg's analysis,
        and that control needs the next leg's inputs then.
        """
        if leg in leg_input_cache:
            return leg_input_cache[leg]
        t_start = leg_starts[leg]
        t_end = t_start + leg_length(leg)
        has_obs = leg < len(args.obs)
        goes_path = None
        if leg < len(args.goes_cwp):
            goes_path = Path(args.goes_cwp[leg])
            if not goes_path.is_file():
                raise FileNotFoundError(
                    f"leg {leg}: no GOES CWP file at {goes_path}")
        if has_obs:
            obs_path = Path(args.obs[leg])
            if not obs_path.is_file():
                raise FileNotFoundError(
                    f"leg {leg}: no observation file at {obs_path}")
            # Without --free-legs: the last leg's analysis is computed and
            # reported but NEVER applied -- it is the verification the run
            # is judged by, and applying it would leave the final state
            # corrected by the very observations used to score it.
            # With --free-legs: EVERY obs leg's analysis is applied, and
            # the trailing free legs are the forecast that runs past the
            # observations -- which is exactly why they cannot verify
            # anything: their verification frames do not exist yet.  The
            # override is this explicit flag condition, never a silent
            # change to the legacy rule.
            # --save-ensemble means this run is one link in a chain that
            # keeps cycling, so its last analysis is applied for the
            # same reason a free forecast's is: the run does not END at
            # this observation, and leaving the analysis uncomputed-into
            # the carried state would silently drop one cycle.
            if args.free_legs > 0 or args.save_ensemble is not None:
                analysis_due = True
                verification_only = False
            else:
                analysis_due = leg < legs - 1
                verification_only = leg == legs - 1

            grid_h = TargetGrid.from_wrfout(Path(args.grid_wrfout[leg]))
            document = read_document(obs_path, expected_grid=grid_h) \
                if obs_path.is_file() else None
        else:
            obs_path = None
            grid_h = None
            document = None
            analysis_due = False
            verification_only = False
        # Radar latent heating: the leg's reflectivity in NOAA's convention,
        # once per leg; each member builds its own slot from it.  Only on a
        # leg whose analysis is applied: a verification-only leg's file is
        # the score, and forcing the forecast with it would grade the
        # forecast against the observations that drove it.
        tten_reflectivity = tten_source = None
        tten_note = None
        if args.radar_tten and document is not None and analysis_due:
            tten_reflectivity, tten_source = \
                radar_tten.reflectivity_from_document(document)
            tten_note = tten_source
        elif args.radar_tten:
            tten_note = (
                "none: a free leg" if document is None else
                "none: this leg's observations verify the run and are "
                "not used to force it")
        echo_columns = None
        if (document is not None and analysis_due
                and float(args.echo_noise) > 0.0):
            # Host weight for the echo-located noise (audit S5), from the
            # leg's own observation file.
            echo_columns = perturb.echo_weight(
                document["variables"]["z_obs"],
                document["variables"]["z_mask"],
                dx_km=float(cfg_perturb.dx_km),
                dy_km=float(cfg_perturb.dy_km),
                length_scale_km=float(min(
                    spec.length_scale_km for spec in cfg_perturb.fields)))
        leg_input_cache[leg] = SimpleNamespace(
            t_start=t_start, t_end=t_end, has_obs=has_obs,
            goes_path=goes_path, obs_path=obs_path, grid_h=grid_h,
            document=document, analysis_due=analysis_due,
            verification_only=verification_only,
            tten_reflectivity=tten_reflectivity, tten_note=tten_note,
            echo_columns=echo_columns)
        return leg_input_cache[leg]

    def contexts_for(leg, li):
        """``context_for(name)`` for one leg, from the driver's state now."""
        def context_for(name):
            return MemberLegContext(
                args=args, inputs=inputs, identity=identity,
                cfg_perturb=cfg_perturb, hot_cfg=hot_cfg,
                leg=leg, absolute_leg_number=leg_number(leg),
                t_start=li.t_start, t_end=li.t_end, stage_root=stage.stage_root,
                out=out, resumed=resumed_from is not None,
                restart=restarts.get(name), pending=pending.get(name),
                nest_child_dc=nest_child_dc, nested=nests_this_leg(leg, name),
                nest_birth=nest_birth.get(name), document=li.document,
                analysis_due=li.analysis_due,
                tten_reflectivity=li.tten_reflectivity,
                obs_path=li.obs_path,
                surface_enabled=(surface_cfg is not None
                                 or (conv_table is not None
                                     and conv_table.needs_surface())),
                setup_arrays=setup_arrays, thb_host=thb_host)
        return context_for

    def packed_workdir(leg):
        return stage.stage_root / "packed" / f"leg{leg_number(leg):03d}"

    # -- the packed route's overlaps ------------------------------------------
    # The control's next leg runs on the last card while this leg's analysis
    # runs on the others, and the recovery generation is written while the
    # next leg's members start: each member's request links the pending file
    # the generation wrote for it (gpuwm.da.member_roster).  On the 9 km
    # CONUS case the control (39 s a leg, 155 s at leg 0) and the
    # generation (30 s) were serial with everything else.
    from concurrent.futures import ThreadPoolExecutor as _Background
    import threading as _threading
    background = _Background(max_workers=2, thread_name_prefix="da-overlap")
    next_control = None
    control_card = forecast_cards[-1]["uuid"] if packed_forecasts else None
    analysis_cards = None
    if packed_forecasts:
        analysis_cards = cards_except(control_card)
    generation_job = None
    card_servers: dict = {}
    if packed_forecasts and not getattr(args, "no_card_servers", False):
        from gpuwm.da.member_wave import start_card_servers

        card_servers = start_card_servers(
            forecast_cards, stage.stage_root / "card-servers",
            per_card=forecast_pack)

        class _CloseServers:
            def clear(self):
                for server in card_servers.values():
                    server.close()
                return None

        stages.append(_CloseServers())
        report["card_servers"] = {
            "cards": sorted(card_servers),
            "servers_per_card": forecast_pack,
            "rule": "one long-lived member worker per member slot for the run "
                    "(gpuwm.da.member_wave.CardServer); --no-card-servers "
                    "starts a process per member leg"}

    def join_generation():
        """Wait for the generation being written; record it; raise its error."""
        nonlocal generation_job
        if generation_job is None:
            return
        job, generation_record, generation_dir, t_written = generation_job
        generation_job = None
        generation, removed = job["future"].result()
        generation_record["written"] = generation["written"]
        generation_record["background_wall_seconds"] = round(
            time.monotonic() - t_written, 3)
        if removed is not None:
            generation_record["older_slots_removed"] = removed
        print(f"ensemble generation written to "
              f"{generation_dir} at {generation_record['elapsed_seconds']:.0f}"
              " s elapsed", flush=True)

    def pending_file_from_generation(name):
        if generation_job is None:
            return None
        job = generation_job[0]
        while not job["events"][name].wait(0.25):
            if job["future"].done():
                job["future"].result()
                if not job["events"][name].is_set():
                    raise RuntimeError(
                        f"the generation finished without writing {name}'s "
                        "pending increments")
        return job["paths"][name]

    spread_controller = None
    spread_grid_sha = None
    if (resumed_from is not None and "spread_repair" in resumed_from
            and not adaptive_spread):
        from gpuwm.da.spread_restart import SpreadRestartError
        raise SpreadRestartError("this generation carries adaptive spread; disabling it on resume would silently discard its saved uncertainty")
    if args.spread_repair != "off":
        from gpuwm.da.spread_repair import SpreadRepairConfig, SpreadRepairController
        spread_config = SpreadRepairConfig(
            policy=args.spread_repair, observed_threshold_dbz=args.spread_repair_z_threshold,
            adaptive=adaptive_spread)
        spread_state = None
        spread_metadata = None
        if adaptive_spread:
            from gpuwm.da.spread_restart import read_sidecar, clock_valid_time
            first_grid = leg_inputs(0).grid_h
            if first_grid is None:
                raise ValueError("adaptive spread needs the observation grid to bind its spatial uncertainty")
            spread_grid_sha = first_grid.identity_sha256()
            if resumed_from is not None:
                spread_state, spread_metadata = read_sidecar(
                    args.resume_ensemble, resumed_from, identity=identity.to_payload(),
                    config=spread_config, seed=args.seed,
                    grid_identity_sha256=spread_grid_sha,
                    valid_time=clock_valid_time(exp.start_time, base_seconds))
        spread_controller = SpreadRepairController(spread_config, seed=args.seed, state=spread_state)
        if spread_metadata is not None:
            spread_controller.last_receipt = {"analysis_valid_time": spread_metadata["analysis_valid_time"]}
    vts_cycle = None
    if vts_enabled:
        import psutil
        from gpuwm.da.spread_vts import VtsCycle
        retained_budget = int(args.spread_vts_ram_gib * 1024**3)
        if psutil.virtual_memory().available < retained_budget+48*1024**3:
            raise MemoryError("VTSM admission needs its explicit retained RAM budget plus the cycle's 48 GiB analysis reserve")
        vts_cycle = VtsCycle(stage.stage_root / "vts.sock", max_bytes=retained_budget)
        stages.append(vts_cycle)
        args.spread_vts_socket = str(vts_cycle.server.path)
    for leg in range(legs):
        li = leg_inputs(leg)
        for done_leg in [k for k in leg_input_cache if k < leg]:
            del leg_input_cache[done_leg]      # its observation document
        t_start, t_end = li.t_start, li.t_end
        if vts_enabled and li.analysis_due:
            validate_request(analysis_seconds=t_end, origin_seconds=t_start,
                             history_seconds=args.history_interval_seconds)
            args.spread_vts_grid_identity = li.grid_h.identity_sha256()
        if adaptive_spread and li.grid_h is not None:
            if li.grid_h.identity_sha256() != spread_grid_sha:
                raise ValueError("adaptive spread target grid changed between cycles; uncertainty would move to different cells")
        leg_record: dict = {"leg": leg_number(leg), "leg_in_run": leg,
                            "start_s": t_start, "end_s": t_end,
                            "trajectories": {}}
        # The complete restart and pending increments own the leg join.
        # Previous host mirrors have already served analysis and are not input
        # to the next forecast. Keeping them beside a new 32-member roster
        # would double the host-state residency of large regional domains.
        release_backgrounds(snapshots)
        # Pending owns the accepted float32 increments; the solver result
        # from the previous analysis is not another leg input.
        increments = None
        member_dbz.clear()
        member_sfc.clear()
        goes_path, obs_path = li.goes_path, li.obs_path
        echo_columns = li.echo_columns
        grid_h, document = li.grid_h, li.document
        analysis_due, verification_only = li.analysis_due, li.verification_only
        if args.radar_tten:
            leg_record["radar_tten_observations"] = li.tten_note
        # The public serial and packed routes use the same complete-tree leg.
        # Every result is accepted before any input is consumed or analysis runs.
        forecast_started = time.monotonic()
        control = None
        if next_control is not None:
            control = next_control.result()
            next_control = None
        results, forecast_receipt = forecast_roster(
            trajectories, context_for=contexts_for(leg, li),
            packed=packed_forecasts, devices=forecast_cards,
            members_per_card=forecast_pack, member_peak_bytes=forecast_peak,
            workdir=packed_workdir(leg),
            stage=stage, run_leg=run_leg_route,
            release=release_device_memory,
            timeout_seconds=args.forecast_leg_timeout_seconds,
            host_reserve_bytes=(4 if args.packed_smoke else 48)*1024**3,
            control=control,
            pending_files=(pending_file_from_generation if packed_forecasts
                           else None),
            before_consume=join_generation,
            servers=card_servers or None)
        for name, result in results.items():
            restarts[name] = result.restart
            # A packed member publishes no host mirror: its leg-end restart
            # (already adopted into this run's stage) IS that state, and the
            # analysis reads the few fields it needs from it on demand.
            snapshots[name] = member_background(result)
            pending[name] = None
            if result.H_Z is not None:
                member_dbz[int(name)] = result.H_Z
            if result.H_surface is not None:
                member_sfc[int(name)] = result.H_surface
            if result.hot_pending is not None:
                hot_pending[name] = result.hot_pending
            if result.nest_birth is not None:
                nest_birth[name] = result.nest_birth
            leg_record["trajectories"][str(name)] = result.record
            if "member_lateral_boundaries" in result.record:
                report.setdefault("member_lateral_boundaries", {})[
                    str(name)] = result.record["member_lateral_boundaries"]
            if setup_arrays is None:
                setup_arrays = result.setup_arrays
            if thb_host is None:
                thb_host = result.thb_host
        leg_record["forecast_barrier"] = forecast_receipt
        leg_record["forecast_plus_io_wall_seconds"] = time.monotonic() - forecast_started
        analysis_started = time.monotonic()
        # Wall clock of the driver's own stages around the solve, in run
        # order (the solve splits its own in stage_wall_seconds).
        driver_clock: dict = {}
        driver_mark = [time.monotonic()]

        def _driver_stage(name):
            now = time.monotonic()
            driver_clock[name] = round(driver_clock.get(name, 0.0)
                                       + now - driver_mark[0], 3)
            driver_mark[0] = now
        if packed_forecasts and leg + 1 < legs:
            # The control is unanalysed: its next leg needs only the restart
            # it just wrote, so it starts now, beside this leg's analysis.
            next_context = contexts_for(leg + 1, leg_inputs(leg + 1))("control")
            next_control = background.submit(
                ControlLaunch, next_context, workdir=packed_workdir(leg + 1),
                gpu_uuid=control_card,
                server=card_servers.get(control_card))

        # -- analysis at t_end ------------------------------------------------
        # The additive-inflation and echo-noise draws depend only on the
        # members' leg-end states, the seed and the leg, never on the
        # filter, so they are drawn on a background thread from here, beside
        # the solve, and joined where they are added.  Drawn after the
        # filter they were 118-128 s of every 32-member 9 km CONUS analysis
        # (box L check1, 2026-10-06) with seven cards idle.  The same calls
        # with the same arguments: the same noise, byte for byte.
        inflation_jobs: dict = {}
        if analysis_due:
            wants_additive = float(args.additive_inflation) > 0.0
            wants_echo = (echo_columns is not None
                          and bool(np.any(echo_columns > 0.0)))
            if wants_additive or wants_echo:
                inflation_pool = _Background(
                    max_workers=1, thread_name_prefix="da-inflation")
                inflation_priors = {index: snapshots[index]
                                    for index in range(args.members)}
                if wants_additive:
                    inflation_jobs["additive"] = inflation_pool.submit(
                        perturb.additive_inflation, inflation_priors,
                        cfg_perturb, seed=args.seed,
                        leg=int(leg_number(leg)),
                        scale=float(args.additive_inflation))
                if wants_echo:
                    inflation_jobs["echo"] = inflation_pool.submit(
                        perturb.additive_inflation, inflation_priors,
                        cfg_perturb, seed=args.seed,
                        leg=int(leg_number(leg)),
                        scale=float(args.echo_noise), weight=echo_columns,
                        stream="echo-noise")
                inflation_pool.shutdown(wait=False)
        if analysis_due or verification_only:
            # The members' backgrounds go to the analysis as they are: the
            # in-process mirror, or a lazy view of the member's own leg-end
            # restart.  Saving a whole-state copy of every member to a
            # second npz and reading it back whole was three passes of the
            # ensemble through host memory and disk per analysis
            # (gpuwm.da.radar_assimilation.CheckpointStateView).  Files are
            # staged only for the replay bundle, which copies files.
            shm_leg = stage.analysis_directory(leg_number(leg))
            if shm_leg.exists():
                shutil.rmtree(shm_leg)
            vts_roster = None
            analysis_snapshots = snapshots
            analysis_members = int(args.members)
            if vts_cycle is not None and analysis_due:
                from gpuwm.da.spread_vts_observers import pool_retained_slots
                vts_roster = pool_retained_slots(vts_cycle.slots(results),
                    members=int(args.members), analysis_seconds=t_end,
                    grid_identity_sha256=grid_h.identity_sha256())
                analysis_snapshots = vts_roster.snapshots
                analysis_members = 3*int(args.members)
            checkpoints = analysis_backgrounds(
                analysis_snapshots, analysis_members, directory=shm_leg,
                t_end=t_end, files=(args.dump_analysis_bundle is not None
                                    and obs_path is not None))

            analysis_fields = ("u", "v")
            provider = None
            cfg_fall_speed = resolved_fall_speed(args, cfg.mp_physics)
            leg_record["fall_speed"] = cfg_fall_speed
            if args.hydrometeors:
                # Derived from the scheme, then intersected with what the
                # checkpoints actually carry: a scheme may advance a
                # moment this configuration does not (Morrison's nc is
                # prognostic only under progn=1), and naming a field the
                # background has not got is a refusal rather than a zero.
                # Then any field the ensemble is constant in is dropped
                # by WHOLE SPECIES, so no moment pair is truncated -- the
                # filter refuses a spreadless field, correctly, and a
                # species the model has not made anywhere is a species
                # with nothing to update.
                carried = set(analysis_snapshots[0])
                candidates = tuple(
                    name for name in moments.analysis_fields(
                        int(cfg.mp_physics)) if name in carried)
                spreads = {}
                for name in candidates:
                    # The whole-stack float64 expressions, on every core
                    # (gpuwm.da.ensemble_stats: the same bytes).
                    scale, widest = ensemble_stats.field_spread(
                        [analysis_snapshots[i][name] for i in range(analysis_members)])
                    spreads[name] = {"max_abs": scale,
                                     "max_spread": widest,
                                     "usable": widest > 1e-12 * scale}
                dropped = {name for name, entry in spreads.items()
                           if not entry["usable"]}
                for pair in moments.pairs_present(
                        tuple(carried), mp_physics=int(cfg.mp_physics)):
                    if dropped & set(pair.fields):
                        dropped |= set(pair.fields) & set(candidates)
                analysis_fields = tuple(name for name in candidates
                                        if name not in dropped)
                leg_record["analysis_field_selection"] = {
                    "candidates": list(candidates),
                    "dropped_for_no_ensemble_spread": sorted(dropped),
                    "spreads": spreads,
                }
                if not analysis_fields:
                    raise RuntimeError(
                        f"leg {leg}: no analysed field has ensemble "
                        f"spread ({spreads})")
                # Clear air is differenced against the same H_Z(x) echo is,
                # so it needs the same provider -- a clear-air-only cycle
                # would otherwise reach the filter with none.
                if args.reflectivity_analysis or args.clear_air_analysis:
                    def provider(index, state, _t=dict(member_dbz)):
                        """H_Z(x) from the device diagnostic this driver
                        already computed for the very state that was
                        snapshotted -- the product's own authority, and
                        the same one the leg-end diagnostics and the hot
                        start read.  The float64 column mirror would be
                        one Python call per column and a second Z
                        authority in the same cycle."""
                        return _t[int(index)]
            if provider is None and cfg_fall_speed == "reflectivity":
                # The fall speed reads the member's own dBZ, the same device
                # diagnostic the reflectivity batch would difference.
                def provider(index, state, _t=dict(member_dbz)):
                    """H_Z(x) from the leg-end device diagnostic."""
                    return _t[int(index)]
            if vts_roster is not None:
                provider = vts_roster.reflectivity_provider
            cwp_provider = None
            if goes_path is not None:
                if setup_arrays is None:
                    raise RuntimeError(
                        f"leg {leg}: the CWP operator needs c1h/c2h/dnw/"
                        "mub2d off a live state and none was captured; "
                        "this leg ran no trajectory")
                composition = CwpComposition(
                    ice=tuple(name.strip() for name
                              in args.cwp_ice_species.split(",")
                              if name.strip()),
                    clear=tuple(name.strip() for name
                                in args.cwp_ice_species.split(",")
                                if name.strip()))
                cwp_provider = checkpoint_cwp_provider(
                    cfg, composition=composition, **setup_arrays)
                if vts_roster is not None:
                    from gpuwm.da.spread_vts import setup_identity
                    from gpuwm.da.spread_vts_observers import make_cwp_provider
                    setup_digest = setup_identity({**setup_arrays,
                        "phb": vts_roster.central_states[0]["phb"],
                        "thb": vts_roster.central_states[0]["thb"]})
                    cwp_provider = make_cwp_provider(vts_roster, cfg,
                        setup_arrays=setup_arrays, setup_identity_sha256=setup_digest,
                        composition=composition)
            # The SAME function the plan-time probe above the leg loop
            # called, with this leg's measured field set: one construction,
            # so the configuration that was reviewed before leg 0 and the
            # configuration that runs cannot differ by a knob.
            _driver_stage("backgrounds_and_field_selection")
            cfg_da = plan_radar_assimilation(
                args, cfg.mp_physics, analysis_fields=analysis_fields,
                cwp=goes_path is not None)
            if next_control is not None and analysis_cards:
                # Every card but the one the next control runs on.
                cfg_da = dataclasses.replace(cfg_da,
                                             solve_cards=analysis_cards)
            # -- the replayable copy of this leg's analysis inputs -------
            # Written BEFORE the solve, from the same objects the solve is
            # about to consume, so a bundle can never describe a different
            # analysis than the one this leg ran.  Radar-only: a leg whose
            # analysis also needs a reflectivity or CWP forward operator
            # cannot be replayed from files alone (the operator needs the
            # scheme's setup state, which is not in a checkpoint), and a
            # bundle that silently dropped those batches would compare two
            # arms on an analysis neither of them performed.
            if args.dump_analysis_bundle is not None and obs_path is not None:
                if cfg_da.cwp or cfg_da.reflectivity or cfg_da.clear_air \
                        or cfg_da.fall_speed == "reflectivity" \
                        or surface_cfg is not None:
                    print(f"leg {leg}: analysis bundle NOT dumped -- this "
                          "analysis carries batches whose forward operator "
                          "is not reconstructable from files "
                          f"(reflectivity={cfg_da.reflectivity}, "
                          f"clear_air={cfg_da.clear_air}, "
                          f"cwp={cfg_da.cwp}, "
                          f"fall_speed={cfg_da.fall_speed}, "
                          f"surface={surface_cfg is not None})", flush=True)
                else:
                    bundle_dir = (Path(args.dump_analysis_bundle)
                                  / f"leg_{leg_number(leg):03d}")
                    ab_manifest = ab_bundle.dump_real_bundle(
                        bundle_dir, checkpoints=checkpoints,
                        obs_path=obs_path,
                        grid_wrfout=Path(args.grid_wrfout[leg]),
                        grid=grid_h, cfg=cfg_da,
                        note=("Analysis inputs of a real cycling DA leg, "
                              "copied at the analysis seam by "
                              "tools/da_cycle_prepared.py."),
                        extra={"driver": "tools/da_cycle_prepared.py",
                               "leg": int(leg),
                               "leg_number": int(leg_number(leg)),
                               "elapsed_seconds": float(t_end),
                               "solve_device_of_the_dumping_run":
                                   args.solve_device})
                    leg_record["analysis_bundle"] = {
                        "path": str(bundle_dir),
                        "members": len(ab_manifest["members"]),
                        "grid_identity_sha256":
                            ab_manifest["grid"]["identity_sha256"]}
                    print(f"leg {leg}: analysis bundle -> {bundle_dir}",
                          flush=True)
            surface_batches = None
            surface_prov = None
            if surface_cfg is not None and vts_roster is None:
                simulated = {"t2": None, "u10": None, "v10": None, "q2": None, "psfc": None}
                stacks = [member_sfc[i] for i in range(args.members)]
                if surface_cfg.temperature:
                    simulated["t2"] = np.stack(
                        [entry["t2"] for entry in stacks])
                if surface_cfg.wind_speed:
                    simulated["u10"] = np.stack(
                        [entry["u10"] for entry in stacks])
                    simulated["v10"] = np.stack(
                        [entry["v10"] for entry in stacks])
                if surface_cfg.dewpoint:
                    simulated["q2"] = np.stack([entry["q2"] for entry in stacks])
                    simulated["psfc"] = np.stack([entry["psfc"] for entry in stacks])
                surface_batches, surface_prov = surface_to_gridded_obs(
                    args.surface_obs, target_grid=grid_h,
                    analysis_time=surface_schedule[leg],
                    analysis_times=surface_schedule,
                    config=surface_cfg,
                    simulated_t2=simulated["t2"],
                    simulated_u10=simulated["u10"],
                    simulated_v10=simulated["v10"],
                    simulated_q2=simulated["q2"],
                    simulated_psfc=simulated["psfc"])
            conv_batches = None
            conv_prov = None
            if conv_table is not None and vts_roster is None:
                from gpuwm.da.obs_conventional import conventional_batches
                t_conv = time.time()
                conv_surface = None
                if conv_table.needs_surface():
                    conv_surface = {
                        key: np.stack([member_sfc[i][key]
                                       for i in range(args.members)])
                        for key in ("t2", "q2", "psfc", "u10", "v10")
                        if key in member_sfc[0]}
                conv_batches, conv_prov = conventional_batches(
                    conv_rows, conv_table,
                    states=[snapshots[i] for i in range(args.members)],
                    thb=thb_host, grid=grid_h,
                    analysis_time=conv_schedule[leg],
                    surface=conv_surface,
                    rotation=grid_rotation(grid_h))
                conv_prov["seconds"] = round(time.time() - t_conv, 2)
                print(f"leg {leg}: conventional "
                      + ", ".join(f"{b['batch']} {b['superobservations']}"
                                  f"/{b['reports']}"
                                  for b in conv_prov["batches"])
                      + f" ({conv_prov['seconds']} s)", flush=True)
            extra_batches = list(surface_batches or ()) + list(
                conv_batches or ())
            extra_prov = surface_prov
            if conv_prov is not None:
                extra_prov = {
                    "batches": list((surface_prov or {}).get("batches")
                                    or ()) + list(conv_prov["batches"]),
                    "surface": surface_prov,
                    "conventional": conv_prov,
                }
            if vts_roster is not None:
                from gpuwm.da.spread_vts_observers import build_extra_observations
                extra_batches, extra_prov = build_extra_observations(
                    vts_roster, target_grid=grid_h,
                    analysis_time=(surface_schedule[leg] if surface_cfg is not None
                                   else conv_schedule[leg] if conv_table is not None
                                   else exp.start_time+timedelta(seconds=t_end)),
                    surface_source=args.surface_obs, surface_config=surface_cfg,
                    analysis_times=surface_schedule if surface_cfg is not None else None,
                    conventional_rows=conv_rows if conv_table is not None else None,
                    conventional_table=conv_table, thb=thb_host,
                    rotation=grid_rotation(grid_h))
                surface_prov = extra_prov.get("surface")
                conv_prov = extra_prov.get("conventional")
            _driver_stage("bundle_and_surface_batches")
            t_solve = time.time()
            # The leg's document was read and grid-bound once already
            # (leg_inputs); handing the path again re-read the whole radar
            # file inside the analysis, 12.5 s of every 9 km CONUS
            # analysis on box E with seven cards idle.
            analysis_spread_controller = spread_controller
            if spread_controller is not None and (vts_roster is not None or (adaptive_spread and not analysis_due)):
                from gpuwm.da.spread_restart import fork_controller
                analysis_spread_controller = fork_controller(spread_controller)
            increments, prov = assimilate_radar_grid(
                checkpoints,
                li.document if li.document is not None else obs_path,
                grid_h, cfg_da,
                reflectivity_provider=provider,
                extra_obs=extra_batches or None,
                extra_obs_provenance=extra_prov,
                cwp_observations=goes_path,
                cwp_provider=cwp_provider,
                analysis_runner=(analysis_spread_controller.analysis_runner(
                    pressure=tuple(analysis_snapshots[i]["p"] for i in range(analysis_members)))
                    if analysis_spread_controller is not None
                    and (analysis_due or adaptive_spread) else None))
            if vts_roster is not None:
                from gpuwm.da.spread_vts import reduce_and_bound
                increments, vts_bounds = reduce_and_bound(vts_roster, increments, cfg_da)
                prov["valid_time_shifting"] = {**vts_roster.receipt,
                                              "central_rebind": vts_bounds}
                if spread_controller is not None:
                    spread_controller.state = analysis_spread_controller.state
                    spread_controller.last_receipt = analysis_spread_controller.last_receipt
                checkpoints = None
                analysis_snapshots = snapshots
                provider = cwp_provider = None
                vts_roster = None
                vts_cycle.release()
            leg_record["analysis"] = {
                "applied": bool(analysis_due),
                "solve_seconds": round(time.time() - t_solve, 1),
                "stage_wall_seconds": prov.get("stage_wall_seconds"),
                "analysis_fields": list(analysis_fields),
                "innovations": prov["innovations"],
                "filter": prov["filter"],
                "velocity_thinning": prov["velocity_thinning"],
                "reflectivity_thinning": prov["reflectivity_thinning"],
                "reflectivity_echo_conditioning": prov.get(
                    "reflectivity_echo_conditioning"),
                "moment_policy": prov["moment_policy"],
                "positivity": prov["positivity"],
                **({"valid_time_shifting": prov["valid_time_shifting"]}
                   if "valid_time_shifting" in prov else {}),
                "surface": surface_prov,
                "conventional": conv_prov,
                # What was actually assimilated this leg, COUNTED off the
                # batches the filter solved rather than inferred from
                # which flags were on. A leg whose satellite file held no
                # usable observation must not read the same as one that
                # had none to read, and a stream that was enabled but
                # contributed nothing must not read the same as one that
                # contributed. See gpuwm.da.treatment.
                "assimilated": treatment.cycle_record(
                    prov["innovations"],
                    adapter_provenance=prov["observations"],
                    extra_obs_provenance=prov["extra_observations"],
                    cwp_provenance=prov["cwp_observations"]),
                "goes_cwp_file": (None if goes_path is None
                                  else goes_path.name),
                "cwp_thinning": prov["cwp_thinning"],
                "cwp_error_inflation": prov["cwp_error_inflation"],
                "cwp_localization_horizontal_m": prov[
                    "cwp_localization_horizontal_m"],
                "cwp_localization_vertical_m": prov[
                    "cwp_localization_vertical_m"],
                "cwp_observations": prov["cwp_observations"],
                "cwp_composition": (
                    None if cwp_provider is None
                    else composition.to_payload()),
            }
            _driver_stage("solve")
            inc_stats = {}
            for field in analysis_fields:
                inc_stats[field] = ensemble_stats.increment_stats(
                    [increments[i][field] for i in sorted(increments)])
                if field in ("u", "v"):
                    inc_stats[field]["max_abs_ms"] = inc_stats[field][
                        "max_abs"]
            leg_record["analysis"]["increments"] = inc_stats
            # Every mass-shaped analysed field must be nonzero on exactly
            # the same gridpoints: with prior_inflation = 1 the increment
            # outside a localisation lens is bitwise zero, so two fields
            # disagreeing about where the analysis reached would mean the
            # localisation is not doing what it claims.
            support = None
            support_ok = True
            for field in analysis_fields:
                if field in ("u", "v", "w"):
                    continue
                touched = ensemble_stats.support(
                    [increments[i][field] for i in sorted(increments)])
                if support is None:
                    support = touched
                elif not np.array_equal(support, touched):
                    support_ok = False
            leg_record["analysis"]["structural_zero"] = {
                "mass_fields_share_one_support": bool(support_ok),
                "support_points": (0 if support is None
                                   else int(support.sum())),
                "filter_active_points": int(
                    prov["filter"]["active_points"]),
            }

            _driver_stage("increment_record_and_support")
            # control-member Vr innovation (no filter, direct H(x)).
            control_state = snapshots[CONTROL]
            u_e, v_n, w_m = member_earth_winds(
                control_state, grid_rotation(grid_h),
                where="control snapshot")
            # Every radar in the file, not radar 0 (control_vr_innovations).
            radars = list(document["radars"])
            per_radar, pooled = control_vr_innovations(document, u_e, v_n, w_m)
            alld = (np.concatenate(pooled) if pooled
                    else np.zeros(0, np.float64))
            leg_record["analysis"]["control_vr"] = {
                # The count that says how much of the network this leg
                # actually saw, beside the innovation it produced.
                "radars": len(radars),
                "radars_with_points": len(pooled),
                "points": int(alld.size),
                "innovation_mean_ms": (float(alld.mean()) if alld.size
                                       else None),
                "innovation_rms_ms": (float(np.sqrt(np.mean(alld ** 2)))
                                      if alld.size else None),
                "per_radar": per_radar,
            }

            _driver_stage("control_vr")
            if spread_controller is not None and analysis_due:
                leg_record["analysis"]["spread_repair"] = spread_controller.last_receipt
            elif adaptive_spread:
                leg_record["analysis"]["spread_repair"] = analysis_spread_controller.last_receipt
            if analysis_due and spread_controller is None and float(args.additive_inflation) > 0.0:
                # After the filter, before the merge that bounds the sum:
                # fresh smooth noise per member, its ensemble mean removed,
                # so the analysis mean stays the filter's and only spread
                # is added (gpuwm.da.perturb.additive_inflation).
                noise, noise_record = inflation_jobs["additive"].result()
                for index, fields in noise.items():
                    increments[index] = dict(increments[index])
                    for field, values in fields.items():
                        if field in increments[index]:
                            base = np.asarray(increments[index][field])
                            increments[index][field] = (
                                base.astype(np.float64) + values).astype(
                                    base.dtype)
                        else:
                            increments[index][field] = values
                leg_record["analysis"]["additive_inflation"] = noise_record
            if (analysis_due and spread_controller is None and echo_columns is not None
                    and bool(np.any(echo_columns > 0.0))):
                # Where the radar sees echo, a further echo-weighted draw on
                # its own stream (audit S5), ensemble mean removed.
                echo_noise, echo_record = inflation_jobs["echo"].result()
                for index, fields in echo_noise.items():
                    increments[index] = dict(increments[index])
                    for field, values in fields.items():
                        if field in increments[index]:
                            base = np.asarray(increments[index][field])
                            increments[index][field] = (
                                base.astype(np.float64) + values).astype(
                                    base.dtype)
                        else:
                            increments[index][field] = values
                echo_record["echo_columns"] = int(
                    np.count_nonzero(echo_columns >= 1.0))
                leg_record["analysis"]["echo_noise"] = echo_record
            _driver_stage("additive_inflation_and_echo_noise")
            if analysis_due:
                overlap: dict[str, dict] = {}
                bound_points = 0
                bound_mass = 0.0
                bound_fields: set[str] = set()
                bounded_members = 0
                def merge_one(index):
                    hot = (hot_pending.get(index) or {}
                           if not args.no_hotstart else {})
                    return merge_hotstart_increments(
                        increments[index], hot,
                        prior=snapshots[index],
                        positivity_policy=args.positivity_policy,
                        report_overlap=(index == 0))

                # Members are independent (each its own increments and its
                # own background), so they are bounded side by side; the
                # results are taken in member order as before.  One after
                # another this was about 1.5 s per member on the CONUS case.
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max(1, min(
                        32, int(args.members)))) as merge_pool:
                    merged_all = list(merge_pool.map(
                        merge_one, range(args.members)))
                for index in range(args.members):
                    merged, member_overlap, positivity = merged_all[index]
                    if index == 0:
                        overlap = member_overlap
                    if positivity is not None:
                        bounded_members += 1
                        bound_points += int(positivity["negative_points"])
                        bound_mass += float(positivity.get(
                            "mass_added",
                            positivity.get("mass_added_by_clip", 0.0)))
                        bound_fields.update(
                            positivity["constrained_fields"])
                    pending[index] = merged
                del merged_all
                if overlap or bounded_members:
                    leg_record["analysis"]["hotstart_overlap"] = {
                        "fields": sorted(overlap),
                        "member_000_rms": overlap,
                        "rule": "summed, then the SUM is put back through "
                                "the run's positivity policy against the "
                                "same background; both are increments to "
                                "the same background from the same "
                                "reflectivity volume, so this double "
                                "counts that volume in the overlapping "
                                "fields",
                        "positivity_after_merge": {
                            "policy": args.positivity_policy,
                            "members_bounded": bounded_members,
                            "constrained_fields": sorted(bound_fields),
                            "negative_points": bound_points,
                            "mass_added_by_clip": bound_mass,
                        },
                    }
                hot_pending.clear()
            _driver_stage("merge_and_positivity_rebind")
            if pending.get(0):
                write_member_increments(
                    out / f"increments_m000_leg{leg_number(leg)}.npz",
                    pending[0])
            shutil.rmtree(shm_leg, ignore_errors=True)
            # The analysis is over: ``pending`` owns the accepted increments.
            # The solver's whole-ensemble output and the members' background
            # arrays are not inputs to anything after this point, and kept
            # until the next leg's top they sat in host memory through the
            # recovery write and the whole next forecast leg, beside the
            # workers' decoders.
            increments = None
            checkpoints = None
            release_backgrounds(snapshots)

        if analysis_due or verification_only:
            _driver_stage("member0_record_and_release")
            leg_record["analysis"]["driver_stage_wall_seconds"] = dict(
                driver_clock)
        leg_record["analysis_plus_io_wall_seconds"] = time.monotonic() - analysis_started
        recovery_started = time.monotonic()
        # -- carry the cycle across the process boundary ------------------
        # At the end of the last OBSERVED leg the staged state is exactly
        # what the next leg would have consumed: each trajectory's
        # restart set plus the analysis increments waiting to be applied.
        # Writing it here (before any free legs run) is what lets the
        # next observation -- which does not exist yet -- be assimilated
        # by a different process without re-initialising the ensemble.
        if (packed_forecasts or adaptive_spread
                or (save_at_leg is not None and leg == save_at_leg)):
            # The child travels in the generation, inside its
            # trajectory's own set.  A cycling generation is where the
            # next process picks the ensemble up, and a child dropped
            # there is a child born again at the next process boundary
            # -- the same defect as being born at the fork, one seam
            # further along.
            nest_receipt = None
            if nest_child_dc is not None:
                child_run = nest_child_dc.run
                nest_receipt = {
                    "grid_id": int(nest_child_dc.grid_id),
                    "parent_id": int(nest_child_dc.parent_id),
                    "nx": int(child_run.nx), "ny": int(child_run.ny),
                    "nz": int(child_run.nz),
                    "dx_m": float(child_run.dx),
                    "dt_s": float(child_run.dt),
                    "parent_grid_ratio": int(
                        nest_child_dc.parent_grid_ratio),
                    "i_parent_start": int(nest_child_dc.i_parent_start),
                    "j_parent_start": int(nest_child_dc.j_parent_start),
                    "terrain_policy": nested_forecast.TERRAIN_POLICY,
                    "land_policy": nested_forecast.LAND_POLICY,
                    "trajectories": [
                        str(n) for n in nest_trajectories
                        if nest_birth.get(n) is not None],
                    "birth_seconds": {
                        str(n): nest_birth[n] for n in nest_trajectories
                        if nest_birth.get(n) is not None},
                }
            generation_dir = (args.save_ensemble if save_at_leg is not None and leg == save_at_leg
                              else ens_state.slot_dir(out / "packed-recovery", leg_number(leg)))
            generation_kwargs = dict(
                identity=identity,
                elapsed_seconds=t_end, leg_number=leg_number(leg),
                # Copies of the two mappings: the next leg rebinds their
                # entries while a background write may still be reading.
                restarts=dict(restarts), pending=dict(pending),
                nest=nest_receipt,
                note=("written at an accepted complete-roster boundary; "
                      "any free legs after it are a branch and are not "
                      "part of this ensemble's history.  Each nesting "
                      "trajectory carries the child it has been running "
                      "since the cycle started, so the next process "
                      "continues it rather than building a new one"))
            if adaptive_spread:
                from gpuwm.da.spread_restart import capture, clock_valid_time
                generation_kwargs["spread_repair"] = capture(
                    spread_controller, identity=identity.to_payload(),
                    grid_identity_sha256=spread_grid_sha,
                    valid_time=clock_valid_time(exp.start_time, t_end),
                    elapsed_seconds=t_end, leg_number=leg_number(leg))
            generation_record = {
                "directory": str(generation_dir),
                "elapsed_seconds": t_end,
                "leg_number": leg_number(leg),
            }
            leg_record["ensemble_generation"] = generation_record
            if packed_forecasts:
                # Written beside the next leg: the next members' requests
                # link the pending files as they land, and the next barrier
                # waits for the whole generation before any restart it
                # links is consumed.
                join_generation()
                events = {name: _threading.Event() for name in trajectories}
                paths: dict = {}

                def written(name, entry, _dir=Path(generation_dir),
                            _paths=paths, _events=events):
                    _paths[name] = (_dir / entry["pending"]
                                    if "pending" in entry else None)
                    _events[name].set()

                def write_and_prune(_dir=Path(generation_dir),
                                    _kwargs=generation_kwargs,
                                    _written=written):
                    # The older ring slot goes the moment this generation's
                    # manifest lands, not at the next barrier: box L's
                    # 3-cycle run held both 76 GB slots through a whole
                    # leg and the disk fell to 37 GB free.
                    generation = ens_state.write_generation(
                        _dir, on_written=_written, **_kwargs)
                    removed = (prune_recovery_ring(_dir)
                               if _dir.parent == out / "packed-recovery"
                               else None)
                    return generation, removed

                job = {"events": events, "paths": paths}
                job["future"] = background.submit(write_and_prune)
                generation_record["written"] = "in the background"
                generation_job = (job, generation_record, generation_dir,
                                  time.monotonic())
            else:
                generation = ens_state.write_generation(
                    generation_dir, **generation_kwargs)
                generation_record["written"] = generation["written"]
                print(f"ensemble generation written to "
                      f"{generation_dir} at {t_end:.0f} s elapsed",
                      flush=True)

        leg_record["recovery_io_wall_seconds"] = time.monotonic() - recovery_started
        leg_record["cycle_wall_seconds"] = (leg_record["forecast_plus_io_wall_seconds"]
            + leg_record["analysis_plus_io_wall_seconds"] + leg_record["recovery_io_wall_seconds"])
        report["legs"].append(leg_record)

        # ---- the treatment proof --------------------------------------
        # Counted, not claimed. An enabled stream that assimilated nothing
        # over the opening cycles stops the run here, before six hours of
        # card time produce a headline naming a stream that never touched
        # the state. The verdict is written into the record either way, so
        # a completed run carries its own proof and a stopped one carries
        # its own reason.
        analyses = [leg["analysis"]["assimilated"]
                    for leg in report["legs"]
                    if isinstance(leg.get("analysis"), dict)
                    and leg["analysis"].get("assimilated") is not None]
        try:
            report["treatment"] = treatment.verify_treatment(
                enabled_obs_kinds, analyses)
        except treatment.TreatmentNotApplied as error:
            report["treatment"] = {
                "label": "full-stack",
                "verdict": "silent_stream",
                "enabled": list(enabled_obs_kinds),
                "error": str(error),
            }
            (out / "cycle-report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8")
            raise SystemExit(f"TREATMENT_NOT_APPLIED: {error}") from error

        (out / "cycle-report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")

    # -- the stage: nothing this run staged outlives it ------------------
    # The last leg's sets have no next leg to consume them.  A generation
    # written above already copied every set it needed, so what is
    # removed here is scratch, and the receipt of the removal is in the
    # report beside the sizes the legs recorded.
    join_generation()
    if next_control is not None:
        next_control.result().stop()
    background.shutdown(wait=True)
    report["staging"] = stage.clear()
    if packed_forecasts and (out / "packed-recovery").is_dir():
        recovery = out / "packed-recovery"
        report["packed_recovery_cleanup"] = {
            "bytes": sum(p.stat().st_size for p in recovery.rglob("*") if p.is_file()),
            "reason": "successful run; failed runs retain the last accepted boundary"}
        shutil.rmtree(recovery)
    report["total_wall_seconds"] = round(time.time() - t_total, 1)
    (out / "cycle-report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("CYCLE_DRIVER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
