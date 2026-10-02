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

* Members share the deterministic run's lateral boundary conditions --
  the perturbation lane's documented setting, and the reason ensemble
  spread is suppressed near the rim.
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

    def __init__(self, stage_root, *, default: bool):
        self.stage_root = Path(stage_root)
        self.root = self.stage_root / "restart"
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
        velocity=True,
        reflectivity=bool(args.reflectivity_analysis),
        fall_speed="none",
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
            DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO))


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
        if positivity_policy in ("clip", "reject"):
            # The post-condition that catches a policy applied to the
            # wrong mapping, asked of the bytes the next leg will apply.
            verify_non_negative(prior, merged)
    return merged, overlap, receipt


def _write_composite_wrfout(npz_path, refl_colmax, grid, cfg,
                            elapsed_seconds: float, exp, *, label: str,
                            domain=None):
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
        lat, lon = grid.latlon_mass()
        snapshot = {
            "XLAT": np.asarray(lat, np.float32),
            "XLONG": np.asarray(lon, np.float32),
            "REFL_COMPOSITE": np.asarray(refl_colmax, np.float32),
        }
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
    stages: list = []
    try:
        return cycle(stages)
    finally:
        for stage in stages:
            stage.clear()


def cycle(stages: list) -> int:
    # Deferred like every other gpuwm import in this driver: the module
    # has to be importable by a bare `python tools/da_cycle_prepared.py
    # --help` from a checkout, and the package lands on sys.path only
    # once the process is actually running the tool.
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
    parser.add_argument("--physics-profile", required=True)
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
    parser.add_argument("--members", type=int, default=10)
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
    parser.add_argument("--positivity-policy", default=None,
                        choices=("clip", "reject", "none"),
                        help="required once a non-negative field is "
                             "analysed; gpuwm.da.positivity documents "
                             "what each choice costs")
    # -- the radial-velocity dispersion gate (default ON) ------------------
    # gpuwm.da.velocity_dispersion names the breakage (an under-dispersed
    # Vr ensemble writing vapour into theta and vapour) and the measurement
    # behind both defaults; the analysis receipt records the ratios either
    # way.
    from gpuwm.da.velocity_dispersion import (
        DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO,
        DEFAULT_VELOCITY_DISPERSION_RATIO)
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
    args = parser.parse_args()
    if args.surface_obs is not None:
        if args.sfc_t2_sigma_k is None and args.sfc_wspd_sigma_ms is None:
            parser.error(
                "--surface-obs was given but neither --sfc-t2-sigma-k "
                "nor --sfc-wspd-sigma-ms states an error; a quantity is "
                "enabled by stating its sigma, and there is no default "
                "sigma on purpose")
        if not args.obs:
            parser.error(
                "--surface-obs joins the radar analysis legs; a "
                "surface-only cycle has no analysis times to join and "
                "is not wired")
    elif (args.sfc_t2_sigma_k is not None
          or args.sfc_wspd_sigma_ms is not None
          or args.sfc_horizontal_loc_m is not None
          or args.sfc_vertical_loc_m is not None):
        parser.error(
            "surface flags were given without --surface-obs; they would "
            "silently assimilate nothing")
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
    if args.hydrometeors and args.positivity_policy is None:
        parser.error(
            "--hydrometeors analyses physically non-negative fields and "
            "--positivity-policy is unstated. clip / reject / none are not "
            "equivalent: clipping at zero ADDS mass and is biased wetward, "
            "rejecting conserves the background and invents gradients, and "
            "none lets the microphysics meet the negatives")

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
    authority = (args.authority_dir if args.authority_dir is not None
                 else args.prepared_root.parent / "authority")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage_root = Path(args.stage_dir) if args.stage_dir else out / "stage"
    stage = StagedRestarts(stage_root, default=args.stage_dir is None)
    stages.append(stage)
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
    from gpuwm.da import (cycle_admission, moments, nested_forecast, obsop,
                          perturb)
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
    })
    hot_cfg = HotStartConfig()

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
            "shared_lateral_boundaries": True,
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

    def child_config_for(name, born_at: float):
        """This trajectory's child, activating at ``born_at`` seconds.

        One configuration per birth instant rather than one for the run:
        the instant is the child's calendar, and it is what the restart
        header carries and checks.
        """
        return nested_forecast.child_born_at(nest_child_dc, exp, born_at)

    # ---- model wiring, per member-leg ---------------------------------------

    def wire(run_seconds_total: float, *, child_dc=None):
        """Build one leg's parent model, and the clocks the child needs.

        With ``child_dc`` the clock and the schedule are the TWO-domain
        ones, but the child itself is NOT built here.  It is derived from
        the parent's state by SINT, and which parent state that is
        depends on the leg: a child born this leg is built from the
        ANALYSED parent after the restart set is restored and the
        increment applied, and a child the trajectory already carries is
        built before the restore so the restore can fill it.  Building a
        newborn from a pre-analysis parent would hand the nest a
        forecast nobody corrected, which is the very failure this whole
        design exists to avoid.  :func:`assemble` closes the model once
        the child is real.

        The root's external boundary clock is BOUND here, as
        ``run_prepared_forecast`` binds it, so a leg's Davies relaxation
        consumes WRF's post-increment ``dtbc`` recurrence from the clock
        the restart owner places rather than the retired elapsed-seconds
        calculation, and the checkpoint header records that semantic.
        """
        from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock

        exp_leg = dataclasses.replace(exp,
                                      run_seconds=float(run_seconds_total))
        live_born = ()
        if child_dc is not None:
            exp_leg = nested_forecast.nested_experiment(exp_leg, child_dc)
            live_born = (int(child_dc.grid_id),)
        restored = restore_prepared_cache(
            inputs.prepared_cache_path, expected_identity=inputs.cache_identity,
            cfg=cfg, static=inputs.static)
        driver = initialize_prepared_physics(
            restored.initial_result, cfg, restored.met, restored.surface,
            inputs.static, inputs.landuse_identity, inputs.grid,
            exp.start_time,
            constant_glw_wm2=declared_constant_glw(exp))
        tick = resolve_clock(
            exp_leg, lbc_interval_s=float(inputs.boundary_interval_seconds),
            live_born_children=live_born)
        schedule = build_schedule(exp_leg, tick)
        clocks = tick.clocks()
        node = DomainNode(exp.root, inputs.grid,
                          restored.initial_result.state, clocks[1],
                          None, [], None)
        if getattr(cfg, "specified", False):
            bind_lateral_boundary_clock(node.state, node.clock)
        return SimpleNamespace(node=node, restored=restored, driver=driver,
                               clocks=clocks, schedule=schedule,
                               child_dc=child_dc)

    def assemble(wired, *, name, child_node=None):
        """Turn the wired pieces into the ExperimentState the executor runs.

        The fingerprint is the TRAJECTORY's (:func:`trajectory_fingerprint`),
        so the restart owner refuses a checkpoint set that belongs to
        another member, another ensemble or another case before it reads
        an array, and names the component that differs.
        """
        nodes = {1: wired.node}
        if child_node is not None:
            nodes[child_node.cfg.grid_id] = child_node
        model = ExperimentState(wired.node, MappingProxyType(nodes),
                                wired.schedule, None,
                                trajectory_fingerprint(identity, name))
        model._experiment_fingerprint_components = trajectory_identity(
            identity, name)
        model._runtime_status = ModelRuntimeStatus()
        model._resumed = False
        model._resume_committed_history_grid_ids = frozenset()
        # The shared scratch arena hands every domain a prefix VIEW of one
        # backing buffer, on the premise that the schedule steps exactly
        # one domain at a time; the shared dycore workspace is the same
        # bargain.  Both stay None on this route -- with a nest attached
        # that is no longer a memory question but a correctness one, and
        # it is asserted rather than assumed.
        model._scratch_arena = None
        model._dycore_state_workspace = None
        if model._scratch_arena is not None \
                or model._dycore_state_workspace is not None:
            raise RuntimeError(
                "the DA cycling path requires per-domain scratch: a "
                "shared arena would let parent and child write the same "
                "bytes with no exception raised")
        model._io_manager = None
        model._last_checkpoint = None
        model._prepared_by_grid_id = MappingProxyType({
            1: SimpleNamespace(static_fields=inputs.static,
                               geog_selection=None,
                               initial_result=wired.restored.initial_result)})
        return model

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
            error_inflation=float(args.sfc_err_inflation),
            elevation_max_diff_m=float(args.sfc_elev_max_diff_m),
            max_age_seconds=float(args.sfc_max_age_s),
            temperature_localization=surface_localization,
            wind_localization=surface_localization)
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
            "wspd_sigma_ms": args.sfc_wspd_sigma_ms,
            "analysis_times": [t.isoformat() for t in surface_schedule],
        }

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
            extra_localizations=(() if surface_cfg is None
                                 else surface_cfg.batch_localizations())))
    analysis_price = cycle_admission.worst_analysis(analysis_prices)
    unsolvable = cycle_admission.unsolvable_analysis_message(analysis_price)
    if unsolvable is not None:
        raise SystemExit(unsolvable)
    mass_shape = (int(cfg.nz), int(cfg.ny), int(cfg.nx))
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
                    cfg_perturb, mass_shape, cp))
            if resumed_from is None and int(args.members) > 0 else 0),
        profile=local_memory_profile_from_device(cp),
        analysis=analysis_price)
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

    for leg in range(legs):
        t_start = leg_starts[leg]
        t_end = t_start + leg_length(leg)
        leg_record: dict = {"leg": leg_number(leg), "leg_in_run": leg,
                            "start_s": t_start, "end_s": t_end,
                            "trajectories": {}}
        member_dbz.clear()
        member_sfc.clear()
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
        z_obs_cp = z_mask_cp = None
        if document is not None and not args.no_hotstart:
            z_obs_cp = cp.asarray(np.asarray(
                document["variables"]["z_obs"], np.float32))
            z_mask_cp = cp.asarray(np.asarray(
                document["variables"]["z_mask"]).astype(bool))

        def run_trajectory(name) -> None:
            """One trajectory's leg, in a scope that ends with it.

            Every device owner the leg builds -- the restored state, the
            physics driver, the model, the child and its driver, the
            leg-end diagnostics -- is a local of this call, so returning
            drops the last reference to each of them.  What the leg hands
            on leaves as host data only, through the run's own tables:
            the restart set, the host mirror, the leg-end H(x), the hot
            start increments and the leg record.  The trajectory loop body
            used to run in :func:`cycle`'s own scope, where those names
            stayed bound after the old ``teardown`` deleted only its loop
            variable, so the previous trajectory's whole model was still
            alive while the next one was restored beside it, and a domain
            whose one trajectory fits the card ran out of memory building
            its second.
            """
            nonlocal setup_arrays, thb_host
            t_leg = time.time()
            nested_leg = nests_this_leg(leg, name)
            # -- what this leg starts from ---------------------------------
            #
            # Leg 0 of a fresh run starts from the prepared background
            # (perturbed for a member).  Every other leg starts from the
            # trajectory's own restart set: the one the previous leg
            # wrote, or the one the resumed generation carried.
            source = None
            child_in_checkpoint = False
            born_at = None
            if not (leg == 0 and resumed_from is None):
                source = restarts[name]
                if source is None:
                    raise RuntimeError(
                        f"leg {leg} {name}: no restart set to continue "
                        "from; the previous leg wrote none")
                stored_ids = restart_domain_ids(source)
                if nested_leg:
                    child_in_checkpoint = (
                        int(nest_child_dc.grid_id) in stored_ids)
                elif len(stored_ids) > 1:
                    raise RuntimeError(
                        f"leg {leg} {name}: the restart set carries "
                        f"domains {list(stored_ids)} and this leg runs "
                        "the root alone; a child cannot be dropped at "
                        "a leg boundary silently")
            if nested_leg:
                born_at = (nest_birth[name] if child_in_checkpoint
                           else float(t_start))
            child_dc_leg = (child_config_for(name, born_at)
                            if nested_leg else None)
            wired = wire(t_end, child_dc=child_dc_leg)
            node, restored, driver = wired.node, wired.restored, wired.driver
            state = node.state
            if setup_arrays is None:
                # The eta coordinate arrays and the base column mass a
                # checkpoint does not serialize (STATE_SETUP_ARRAYS in
                # gpuwm/state_serialization_contract.py). The CWP operator
                # integrates in the model's own mass measure and cannot
                # rebuild them from the npz, so they are captured here off
                # a live state, once, exactly as the reflectivity provider
                # would need thb.
                setup_arrays = {
                    "c1h": to_host(state.c1h).astype(np.float64),
                    "c2h": to_host(state.c2h).astype(np.float64),
                    "dnw": to_host(state.dnw).astype(np.float64),
                    "mub2d": to_host(state.mub2d).astype(np.float64),
                }
            child_node = child_driver = None
            nest_entry = None
            if nested_leg:
                nest_entry = leg_record["trajectories"].setdefault(
                    str(name), {}).setdefault("nest", {})
                nest_entry["grid_id"] = nest_child_dc.grid_id
                nest_entry["born_at_seconds"] = float(born_at)

            def _build_child():
                """The child object, from the parent's live state.

                Its clock is placed at the child's own birth first, and
                the builder refreshes the model time from it before the
                physics is attached, so the child's driver counts its
                ITIMESTEP from its activation.
                """
                clock = wired.clocks[nest_child_dc.grid_id]
                nested_forecast.place_newborn_clock(clock)
                return nested_forecast.build_nested_child(
                    node, child_dc_leg,
                    static=inputs.static, surface=restored.surface,
                    landuse_identity=inputs.landuse_identity,
                    valid_time=exp.start_time, clock=clock,
                    parent_driver=driver,
                    constant_glw_wm2=declared_constant_glw(exp))

            # A child the trajectory already carries is built BEFORE the
            # restore, because the restart owner restores the whole set
            # into the whole tree: the SINT below only builds the object
            # and its base state, and every array it holds is then
            # overwritten from the checkpoint.
            if child_in_checkpoint:
                child_node, child_driver, _land_receipt = _build_child()
            # `resumed` drives model._resumed below, which stops the
            # resumed leg from rewriting history the previous process
            # already committed.  `resumed_from is None` is the separate
            # and more dangerous condition: a generation carried in from
            # --resume-ensemble is ALREADY perturbed, and perturbing it
            # again at leg 0 would throw away the analysis it carried with
            # no error raised anywhere.  Both guards are required; they are
            # not the same question.
            resumed = False
            #: The parent's fields the pending increment names, as the
            #: restart set restored them and BEFORE the increment: the
            #: background side of the correction a carried child takes.
            restored_background: dict = {}
            if source is None:
                if name != CONTROL:
                    perturb.apply_perturbations(
                        state, args.seed + int(name), cfg_perturb)
                    refresh_diagnostics(
                        state, hypsometric_opt=cfg.hypsometric_opt)
            else:
                model = assemble(wired, name=name, child_node=child_node)
                restart_info = restore_leg_restart(
                    model, source, expected_seconds=t_start)
                resumed = True
                entry_restore = leg_record["trajectories"].setdefault(
                    str(name), {})
                entry_restore["restored_from"] = {
                    "root_member": Path(source).name,
                    "domain_ids": list(restart_domain_ids(source)),
                    "elapsed_seconds": (restart_info.elapsed_ticks
                                        / restart_info.tick_den),
                }
                # A staged set is consumed by exactly one leg; a
                # generation's set is the durable record and stays.
                entry_restore["staged_set_removed"] = stage.consume(source)
                if pending[name]:
                    if child_in_checkpoint:
                        for field in sorted(pending[name]):
                            live = getattr(state, field, None)
                            if live is None or getattr(
                                    child_node.state, field, None) is None:
                                continue
                            restored_background[field] = to_host(live)
                    to_apply, pair_report = keep_moment_pairs(
                        state, pending[name], mp_physics=cfg.mp_physics)
                    receipt = apply_increments(
                        state, to_apply, mp_physics=cfg.mp_physics)
                    leg_record["trajectories"].setdefault(str(name), {})[
                        "apply_fields"] = receipt["field_count"]
                    if pair_report["conditioned"]:
                        leg_record["trajectories"].setdefault(str(name), {})[
                            "moment_pairs_kept"] = pair_report
                    refresh_diagnostics(
                        state, hypsometric_opt=cfg.hypsometric_opt)
            health = StateHealthValidator(state).validate(
                phase=f"leg{leg}.{name}")
            if not health.ok:
                raise FloatingPointError(
                    f"leg {leg} {name}: pre-leg health failed: "
                    f"{vars(health)}")

            # -- the fine nest ------------------------------------------------
            #
            # A child born this leg is built AFTER the restore, the
            # increment application and the health gate: the parent state
            # it is derived from has to be the ANALYSED one, or the nest
            # would inherit a forecast nobody corrected and the whole
            # exercise would be worth less than interpolating the output.
            if nested_leg:
                if child_node is None:
                    child_node, child_driver, _land_receipt = _build_child()
                    nest_birth[name] = float(born_at)
                    nest_entry["initialization"] = "parent-live-state-sint"
                else:
                    # A later nested leg: the child already has fine
                    # structure of its own, restored above from the
                    # trajectory's own checkpoint set; flattening it back
                    # to a parent interpolation every leg boundary would
                    # throw away exactly what the nest is for.
                    nest_entry["initialization"] = "restart-set"
                    # The child is not analysed itself -- the filter runs
                    # on the parent ensemble, and a 1 km member set is a
                    # different (and much larger) experiment.  What the
                    # child gets is the correction the analysis made to
                    # its parent, carried down by the operator it was
                    # born through: SINT(analysed) - SINT(background),
                    # differenced rather than interpolated as one
                    # increment because SINT is monotonicity-limited and
                    # therefore not linear.  Both sides are raw
                    # interpolations of the PARENT -- the restored parent
                    # before the increment, and the live parent after it.
                    # Over exactly the fields the increment names: a
                    # moment the applier repaired on a field the analysis
                    # did not name stays on the parent, and a leg with no
                    # analysis leaves the child bitwise alone.
                    child_state = child_node.state
                    analysed_parent = {
                        field: getattr(state, field)
                        for field in sorted(restored_background)}
                    carried = {}
                    if restored_background:
                        correction = nested_forecast.nest_down_analysis(
                            restored_background, analysed_parent,
                            child_dc_leg, cfg, array_module=cp)
                        # The parent's analysis was bounded against the
                        # PARENT's background; the child's background is
                        # its own evolved state, so the same correction
                        # can drive a positive-definite species below
                        # zero there.  Same policy, child's background.
                        correction, child_positivity = (
                            bound_child_correction(
                                child_state, correction,
                                policy=args.positivity_policy,
                                array_module=cp))
                        if child_positivity is not None:
                            nest_entry["positivity_on_correction"] = (
                                child_positivity)
                        for field, delta in correction.items():
                            largest = float(cp.abs(delta).max())
                            if largest == 0.0:
                                continue
                            target = getattr(child_state, field)
                            target[...] = target + delta.astype(
                                target.dtype)
                            carried[field] = largest
                        del correction
                        refresh_diagnostics(
                            child_state,
                            hypsometric_opt=child_dc_leg.run.hypsometric_opt)
                    del analysed_parent
                    nest_entry["analysis_carried_down"] = {
                        "how": ("sint(analysed parent) - sint(restored "
                                "parent), the child's own state kept"),
                        "fields": sorted(carried),
                        "max_abs_correction": {
                            field: carried[field]
                            for field in sorted(carried)},
                    }
                child_state = child_node.state
                child_health = StateHealthValidator(child_state).validate(
                    phase=f"leg{leg}.{name}.nest")
                if not child_health.ok:
                    raise FloatingPointError(
                        f"leg {leg} {name}: nest pre-leg health failed: "
                        f"{vars(child_health)}")
            del restored_background

            model = assemble(wired, name=name, child_node=child_node)
            if resumed:
                model._resumed = True
                model._resume_committed_history_grid_ids = frozenset(
                    model.nodes_by_grid_id)

            execute_experiment(model, history_handler=None,
                               progress_callback=None, validate_state=True,
                               skip_feedback_path=True)
            cp.cuda.Stream.null.synchronize()

            # -- the leg join: this trajectory's restart set --------------
            #
            # Written FIRST, before any diagnostic reads the state, so the
            # set is the model exactly as the integration left it.  The
            # writer refuses anything but a period boundary with nothing
            # pending, which is what a completed integration is.
            restart_root = stage.directory(leg_number(leg), name)
            restarts[name] = write_leg_restart(
                model, restart_root,
                auto_epssm=(nested_forecast.nested_experiment(
                    exp, nest_child_dc).auto_epssm
                    if len(model.nodes_by_grid_id) > 1 else exp.auto_epssm),
                valid_time=exp.start_time + timedelta(seconds=float(
                    node.clock.ticks / node.clock.tick_den)))
            written_members = tree_restart_members(restarts[name])
            entry = leg_record["trajectories"].setdefault(str(name), {})
            entry["pool_trim"] = model._pool_trim_policy
            entry["restart"] = {
                "root_member": restarts[name].name,
                "domain_ids": sorted(int(gid) for gid in written_members),
                "bytes": int(sum(member.stat().st_size
                                 for member in written_members.values())),
                "elapsed_seconds": float(node.clock.elapsed_seconds),
            }

            thb_live = getattr(state, "thb", None)
            thb_snapshot = to_host(thb_live) if thb_live is not None \
                else None

            # -- leg-end diagnostics on the live device state ---------------
            refl = obsop.simulated_reflectivity(state, cfg)
            refl_host = to_host(refl).astype(np.float32)
            if name != CONTROL:
                member_dbz[int(name)] = refl_host.astype(np.float64)
            if surface_cfg is not None and name != CONTROL:
                # Leg-END surface diagnostics off the live driver: the
                # 2m/10m fields the surface layer diagnosed for the very
                # state that was mirrored, taken here beside member_dbz
                # and nowhere else.
                absent = [key for key in ("t2", "u10", "v10")
                          if key not in driver.fields]
                if absent:
                    raise RuntimeError(
                        f"leg {leg} {name}: --surface-obs needs the "
                        f"driver's {absent} diagnostics and this physics "
                        "profile does not allocate them")
                member_sfc[int(name)] = {
                    key: to_host(driver.fields[key]).astype(np.float64)
                    for key in ("t2", "u10", "v10")}
            if args.save_composites:
                comp_dir = out / "composites"
                comp_dir.mkdir(parents=True, exist_ok=True)
                composite_npz = (comp_dir /
                                 f"leg{leg_number(leg):02d}_{name}.npz")
                np.savez_compressed(
                    composite_npz,
                    refl_colmax=refl_host.max(axis=0),
                    elapsed_seconds=np.float64(
                        node.clock.elapsed_seconds))
                # ...and the same composite as a real wrfout beside it, so
                # `gpuwm render --engine rust` -- the renderer the render
                # law names -- can draw this member.  The npz is what the
                # cycle's own analysis reads and is untouched; this is a
                # second copy in the format a different tool reads.
                _write_composite_wrfout(
                    composite_npz, refl_host.max(axis=0), inputs.grid, cfg,
                    node.clock.elapsed_seconds, exp,
                    label=f"leg {leg_number(leg):02d} member {name}")
            # -- the nest's own leg-end product ------------------------------
            #
            # Written under a d02 name beside the parent's rather than
            # replacing it: the point of the exercise is a fine view OF the
            # parent's forecast, and a reader has to be able to see both.
            refl_nest = None
            if child_node is not None:
                refl_nest = obsop.simulated_reflectivity(
                    child_node.state, child_dc_leg.run)
                refl_nest_host = to_host(refl_nest).astype(np.float32)
                nest_entry["refl_max_dbz"] = float(refl_nest_host.max())
                nest_entry["elapsed_seconds"] = float(
                    child_node.clock.elapsed_seconds)
                nest_entry["step_count"] = int(child_node.clock.step_count)
                nest_entry["domain_start_offset_seconds"] = float(
                    child_node.state.domain_start_offset)
                if args.save_composites:
                    # ``leg_number(leg)``, the same author as every other
                    # leg-named artifact this driver writes.  The child's
                    # frame and the parent's frame of the SAME leg have to
                    # carry the same number, or a run given a leg-number
                    # offset writes a nest under a leg the parent frames
                    # never use and nothing finds it.
                    nest_npz = (comp_dir /
                                f"leg{leg_number(leg):02d}_{name}_d"
                                f"{nest_child_dc.grid_id:02d}.npz")
                    nest_colmax = refl_nest_host.max(axis=0)
                    np.savez_compressed(
                        nest_npz,
                        refl_colmax=nest_colmax,
                        elapsed_seconds=np.float64(
                            child_node.clock.elapsed_seconds),
                        dx_m=np.float64(nest_child_dc.run.dx),
                        i_parent_start=np.int32(
                            nest_child_dc.i_parent_start),
                        j_parent_start=np.int32(
                            nest_child_dc.j_parent_start),
                        parent_grid_ratio=np.int32(
                            nest_child_dc.parent_grid_ratio))
                    # ...and the child's own wrfout beside it, for the
                    # same reason the parent gets one and by the same
                    # call.  Without it the nest's product had no route
                    # to the renderer the render law names: an ``.npz``
                    # states no geolocation, so rw_wrfbatch cannot read
                    # one, and a door that can ask for a nest whose
                    # picture nobody can draw is not a shipped door.
                    _write_composite_wrfout(
                        nest_npz, nest_colmax, child_node.grid,
                        nest_child_dc.run,
                        child_node.clock.elapsed_seconds, exp,
                        label=(f"leg {leg_number(leg):02d} member {name} "
                               f"d{nest_child_dc.grid_id:02d}"),
                        domain=nest_child_dc)
            entry["wall_seconds"] = round(time.time() - t_leg, 1)
            entry["elapsed_seconds"] = float(node.clock.elapsed_seconds)
            if document is not None:
                z_mask = np.asarray(
                    document["variables"]["z_mask"]).astype(bool)
                z_obs = np.asarray(document["variables"]["z_obs"],
                                   np.float64)
                inside = z_mask
                model_in_mask = refl_host[inside].astype(np.float64)
                entry["z_obs_space"] = {
                    "points": int(inside.sum()),
                    "obs_mean_dbz": float(z_obs[inside].mean()),
                    "model_mean_dbz": float(model_in_mask.mean()),
                    "model_max_dbz": float(model_in_mask.max()),
                    "model_cols_gt35_in_echo": int(
                        (refl_host.max(axis=0) >= 35.0)[
                            z_mask.any(axis=0)].sum()),
                    "obs_cols_gt35": int(
                        ((z_obs * z_mask).max(axis=0) >= 35.0).sum()),
                    "innovation_mean_dbz": float(
                        (z_obs[inside] - model_in_mask).mean()),
                }
            if (analysis_due and name != CONTROL
                    and not args.no_hotstart):
                increments_hot, hot_prov = hotstart_increments(
                    state, z_obs_cp, z_mask_cp, hot_cfg,
                    simulated_dbz=refl)
                hot_pending[name] = {
                    field: to_host(values).astype(np.float32)
                    for field, values in increments_hot.items()}
                entry["hotstart"] = {
                    key: hot_prov[key] for key in hot_prov
                    if isinstance(hot_prov[key], (int, float, str))}

            # The host mirror the FILTER reads.  Not the leg join: that
            # is the restart set above, which carries this and everything
            # the mirror does not.
            snapshot = {}
            for field in STATE_SERIALIZED_ATTRS:
                value = getattr(state, field, None)
                if value is not None:
                    snapshot[field] = to_host(value)
            snapshots[name] = snapshot
            entry["analysis_state_fields"] = sorted(snapshot)
            if thb_host is None and thb_snapshot is not None:
                thb_host = thb_snapshot
            pending[name] = None
            print(f"leg {leg} {name}: {entry['wall_seconds']} s, "
                  f"elapsed {entry['elapsed_seconds']:.0f} s", flush=True)

        for name in trajectories:
            run_trajectory(name)
            # The trajectory's owners died with its scope; the collection
            # and the pool drain hand their blocks back before the next
            # trajectory is wired, and before the analysis below.
            release_device_memory()

        # -- analysis at t_end ------------------------------------------------
        if analysis_due or verification_only:
            shm_leg = stage.analysis_directory(leg_number(leg))
            if shm_leg.exists():
                shutil.rmtree(shm_leg)
            checkpoints = {}
            for index in range(args.members):
                member_dir = shm_leg / f"member_{index:03d}"
                member_dir.mkdir(parents=True)
                path = member_dir / f"gpuwmrst_d01_{int(t_end):06d}.npz"
                np.savez(path, **{f"state/{k}": v
                                  for k, v in snapshots[index].items()})
                checkpoints[index] = path

            analysis_fields = ("u", "v")
            provider = None
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
                carried = set(snapshots[0])
                candidates = tuple(
                    name for name in moments.analysis_fields(
                        int(cfg.mp_physics)) if name in carried)
                spreads = {}
                for name in candidates:
                    stack = np.stack([snapshots[i][name] for i in
                                      range(args.members)]).astype(
                                          np.float64)
                    scale = float(np.abs(stack).max())
                    widest = float(stack.std(axis=0, ddof=1).max())
                    spreads[name] = {"max_abs": scale,
                                     "max_spread": widest,
                                     "usable": widest > 1e-12 * scale}
                    del stack
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
            # The SAME function the plan-time probe above the leg loop
            # called, with this leg's measured field set: one construction,
            # so the configuration that was reviewed before leg 0 and the
            # configuration that runs cannot differ by a knob.
            cfg_da = plan_radar_assimilation(
                args, cfg.mp_physics, analysis_fields=analysis_fields,
                cwp=goes_path is not None)
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
                        or surface_cfg is not None:
                    print(f"leg {leg}: analysis bundle NOT dumped -- this "
                          "analysis carries batches whose forward operator "
                          "is not reconstructable from files "
                          f"(reflectivity={cfg_da.reflectivity}, "
                          f"clear_air={cfg_da.clear_air}, "
                          f"cwp={cfg_da.cwp}, "
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
            if surface_cfg is not None:
                simulated = {"t2": None, "u10": None, "v10": None}
                stacks = [member_sfc[i] for i in range(args.members)]
                if surface_cfg.temperature:
                    simulated["t2"] = np.stack(
                        [entry["t2"] for entry in stacks])
                if surface_cfg.wind_speed:
                    simulated["u10"] = np.stack(
                        [entry["u10"] for entry in stacks])
                    simulated["v10"] = np.stack(
                        [entry["v10"] for entry in stacks])
                surface_batches, surface_prov = surface_to_gridded_obs(
                    args.surface_obs, target_grid=grid_h,
                    analysis_time=surface_schedule[leg],
                    analysis_times=surface_schedule,
                    config=surface_cfg,
                    simulated_t2=simulated["t2"],
                    simulated_u10=simulated["u10"],
                    simulated_v10=simulated["v10"])
            t_solve = time.time()
            increments, prov = assimilate_radar_grid(
                checkpoints, obs_path, grid_h, cfg_da,
                reflectivity_provider=provider,
                extra_obs=surface_batches,
                extra_obs_provenance=surface_prov,
                cwp_observations=goes_path,
                cwp_provider=cwp_provider)
            leg_record["analysis"] = {
                "applied": bool(analysis_due),
                "solve_seconds": round(time.time() - t_solve, 1),
                "analysis_fields": list(analysis_fields),
                "innovations": prov["innovations"],
                "filter": prov["filter"],
                "velocity_thinning": prov["velocity_thinning"],
                "reflectivity_thinning": prov["reflectivity_thinning"],
                "moment_policy": prov["moment_policy"],
                "positivity": prov["positivity"],
                "surface": surface_prov,
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
            inc_stats = {}
            for field in analysis_fields:
                stack = np.stack([increments[i][field]
                                  for i in sorted(increments)])
                inc_stats[field] = {
                    "finite": bool(np.isfinite(stack).all()),
                    "max_abs": float(np.abs(stack).max()),
                    "rms_where_nonzero": float(np.sqrt(np.mean(
                        stack[stack != 0.0] ** 2)))
                    if np.any(stack != 0.0) else 0.0,
                    "nonzero_fraction": float(np.mean(stack != 0.0)),
                }
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
                stack = np.stack([increments[i][field]
                                  for i in sorted(increments)])
                touched = np.any(stack != 0.0, axis=0)
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

            # control-member Vr innovation (no filter, direct H(x)).
            control_state = snapshots[CONTROL]
            u_e, v_n, w_m = member_earth_winds(
                control_state, grid_rotation(grid_h),
                where="control snapshot")
            from gpuwm.da.obs_radar import (beam_unit_vectors,
                                            simulated_radial_velocity)
            # Every radar in the file, not radar 0.  Each antenna has its
            # own beam geometry, so a radial-velocity innovation is only
            # defined per radar; indexing [0] and calling the answer "the"
            # control innovation was harmless while every file held one
            # radar and silently wrong the moment one held twenty.  The
            # aggregate below pools the residuals, which is the same
            # statistic when there is one radar -- so single-radar
            # receipts keep the numbers they had.
            radars = list(document["radars"])
            per_radar = []
            pooled = []
            from gpuwm.obs.radar_grid import radar_plane
            for index, radar in enumerate(radars):
                # radar_plane, not variables[...][index]: on a v2 file the
                # stored plane covers only this radar's reach window, and
                # the control innovation is computed against a whole-domain
                # wind field.  On v1 it is the same array it always was.
                vr_mask = np.asarray(
                    radar_plane(document, "vr_mask", index)).astype(bool)
                points = int(vr_mask.sum())
                row = {"radar": str(radar["id"]), "points": points}
                if points:
                    vr_obs = np.asarray(
                        radar_plane(document, "vr_obs", index), np.float64)
                    sim = simulated_radial_velocity(
                        u_e, v_n, w_m, beam_unit_vectors(document, index))
                    d = vr_obs[vr_mask] - sim[vr_mask]
                    pooled.append(d)
                    row["innovation_mean_ms"] = float(d.mean())
                    row["innovation_rms_ms"] = float(
                        np.sqrt(np.mean(d ** 2)))
                per_radar.append(row)
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

            if analysis_due:
                overlap: dict[str, dict] = {}
                bound_points = 0
                bound_mass = 0.0
                bound_fields: set[str] = set()
                bounded_members = 0
                for index in range(args.members):
                    hot = (hot_pending.get(index) or {}
                           if not args.no_hotstart else {})
                    merged, member_overlap, positivity = (
                        merge_hotstart_increments(
                            increments[index], hot,
                            prior=snapshots[index],
                            positivity_policy=args.positivity_policy,
                            report_overlap=(index == 0)))
                    if index == 0:
                        overlap = member_overlap
                    if positivity is not None:
                        bounded_members += 1
                        bound_points += int(positivity["negative_points"])
                        bound_mass += float(
                            positivity.get("mass_added_by_clip", 0.0))
                        bound_fields.update(
                            positivity["constrained_fields"])
                    pending[index] = merged
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
            if pending.get(0):
                np.savez_compressed(
                    out / f"increments_m000_leg{leg_number(leg)}.npz",
                    **{k: v.astype(np.float32) for k, v
                       in pending[0].items()})
            shutil.rmtree(shm_leg, ignore_errors=True)

        # -- carry the cycle across the process boundary ------------------
        # At the end of the last OBSERVED leg the staged state is exactly
        # what the next leg would have consumed: each trajectory's
        # restart set plus the analysis increments waiting to be applied.
        # Writing it here (before any free legs run) is what lets the
        # next observation -- which does not exist yet -- be assimilated
        # by a different process without re-initialising the ensemble.
        if save_at_leg is not None and leg == save_at_leg:
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
            generation = ens_state.write_generation(
                args.save_ensemble, identity=identity,
                elapsed_seconds=t_end, leg_number=leg_number(leg),
                restarts=restarts, pending=pending,
                nest=nest_receipt,
                note=("written at the end of the last observed leg; "
                      "any free legs after it are a branch and are not "
                      "part of this ensemble's history.  Each nesting "
                      "trajectory carries the child it has been running "
                      "since the cycle started, so the next process "
                      "continues it rather than building a new one"))
            leg_record["ensemble_generation"] = {
                "directory": str(args.save_ensemble),
                "elapsed_seconds": t_end,
                "leg_number": leg_number(leg),
                "written": generation["written"],
            }
            print(f"ensemble generation written to "
                  f"{args.save_ensemble} at {t_end:.0f} s elapsed",
                  flush=True)

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
    report["staging"] = stage.clear()
    report["total_wall_seconds"] = round(time.time() - t_total, 1)
    (out / "cycle-report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("CYCLE_DRIVER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
