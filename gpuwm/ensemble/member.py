"""Running one ensemble member (EXPERIMENTAL).

This is a *wrapper*, not a second runner.  It calls exactly the
functions ``gpuwm.runtime.run_experiment`` calls for a single-domain
experiment, in the same order, and adds one seam between preparation and
integration where the perturbation hook sees the initial state:

    quarantine -> catalog -> prepare ->[perturb]->[re-diagnose]-> integrate

The integration loop, the physics, and the I/O are untouched.

The ``[re-diagnose]`` step is the perturbation contract's caller
post-condition, not an optimisation.  ``state.p``, ``state.al`` and
``state.alt`` are diagnostics of ``(thp, php, mup, qv)``; a hook that
perturbs theta or moisture leaves them describing the state *before* the
perturbation.  ``gpuwm.da.perturb`` cannot refresh them itself -- the
diagnostic path is CUDA-first and importing it would make a NumPy-backed
perturbation impossible -- so it records the requirement in
``provenance["post_conditions"]`` and hands it to whoever calls it.  That
is this module.

Multi-domain experiments are refused here rather than half-supported:
the perturbation seam for a domain tree has to decide what a member
perturbation means on nests, and that decision is not this lane's.

**A restart leg is not perturbed, and its baseline is its checkpoint --
the whole baseline, hash AND clock.**  ``restore_restart`` runs inside
the integrator and overwrites the whole prepared state, so anything this
seam wrote first is discarded.  A cycling leg therefore skips the hook
outright, records ``initial_state_sha256`` as the sha of the checkpoint
it restored, and compares its final state AND its final clock to that
checkpoint's -- the four facts that were previously all about a state the
leg never integrated.  Reading the clock baseline off the prepared state
was the residual half of that bug: a no-op integrator then reported a
successful 60 s advance with a prepared clock that never moved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from gpuwm.ensemble.perturbation import resolve_perturbation
from gpuwm.ensemble.state_sha import (checkpoint_elapsed_seconds,
                                      checkpoint_state_sha256,
                                      hash_state_arrays,
                                      live_state_sha_receipt,
                                      live_state_sha256)
from gpuwm.ensemble.wrfout_inventory import member_inventory

#: The state attributes ``gpuwm.core.diagnostics.update_diagnostics``
#: rewrites.  Hashed before and after the refresh so the manifest can
#: show that the post-condition was load-bearing rather than ceremonial.
DIAGNOSED_ATTRS = ("p", "al", "alt")


def diagnostics_sha256(state) -> str:
    """Content hash of the diagnosed ``(p, al, alt)`` triple.

    Same reduction as :func:`gpuwm.ensemble.state_sha.live_state_sha256`,
    over the diagnostics instead of the prognostics, so the two numbers
    are comparable and neither can be mistaken for the other.
    """

    arrays = [(name, getattr(state, name)) for name in DIAGNOSED_ATTRS
              if getattr(state, name, None) is not None]
    if not arrays:
        raise ValueError(
            "state carries none of "
            f"{', '.join(DIAGNOSED_ATTRS)}; there are no diagnostics to "
            "refresh and the perturbation post-condition cannot be met")
    return hash_state_arrays(arrays)


def refresh_diagnostics(state, *, hypsometric_opt: int) -> dict:
    """Honour the perturbation contract's caller post-condition.

    Recomputes ``p``/``al``/``alt`` from the perturbed
    ``(thp, php, mup, qv)`` and returns a receipt saying whether they
    moved.  ``moved: false`` is not a failure -- a perturbation confined
    to ``u``/``v`` leaves the equation of state alone -- but it is worth
    recording, because ``moved: false`` on a theta perturbation means the
    refresh did not reach the state the integrator is about to advance.
    """

    from gpuwm.core.diagnostics import update_diagnostics

    before = diagnostics_sha256(state)
    update_diagnostics(state, hypsometric_opt)
    after = diagnostics_sha256(state)
    return {
        "contract": "gpuwm.da.perturb/post_conditions/update_diagnostics",
        "ran": True,
        "hypsometric_opt": int(hypsometric_opt),
        "attributes": list(DIAGNOSED_ATTRS),
        "sha256_before": before,
        "sha256_after": after,
        "moved": after != before,
    }


@dataclass(frozen=True)
class MemberOutcome:
    """Everything the manifest needs about one finished member."""

    index: int
    seed: int
    member_dir: Path
    initial_state_sha256: str
    final_state_sha256: str
    wall_seconds: float
    sim_seconds: float
    wrfout_count: int
    last_checkpoint: str | None
    perturbation: Mapping[str, object] = field(default_factory=dict)
    #: Per-wrfout frame inventory -- relative path, domain, valid times
    #: with their frame indices, size and sha256 -- as
    #: :func:`gpuwm.ensemble.wrfout_inventory.member_inventory` builds it.
    #: ``None`` from a runner that produces none; the manifest records it
    #: verbatim so "no inventory" is never mistaken for "inventory of no
    #: files", and a consumer admitting this member past the default
    #: statuses can refuse rather than believe a bare count.
    wrfout_inventory: tuple | None = None
    #: Which restart-contract attributes ``final_state_sha256`` covered.
    #: ``None`` from a runner that does not produce one; the manifest
    #: records it verbatim so a partial hash is never mistaken for a
    #: whole-state one.
    final_state_inventory: Mapping[str, object] | None = None


def run_member(*, base_config, member_dir, index: int, seed: int,
               perturbation: str, perturbation_options: Mapping | None = None,
               run_seconds: float | None = None,
               restart=None, progress_callback=None) -> MemberOutcome:
    """Prepare, perturb, and integrate one member into ``member_dir``."""
    from gpuwm import runtime
    from gpuwm.case_data import load_experiment_case
    from gpuwm.ingest.preflight import build_input_catalog
    from gpuwm.io.wrfout import quarantine_orphan_wrfouts

    outdir = Path(member_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    hook = resolve_perturbation(perturbation)

    exp, data = load_experiment_case(base_config)
    if len(exp.domains) != 1:
        raise ValueError(
            f"{base_config} declares {len(exp.domains)} domains; the "
            "ensemble engine wraps the single-domain runner only. A "
            "domain tree needs a stated policy for what a member "
            "perturbation means on a nest, which this engine does not "
            "invent.")
    dc = runtime.single_domain(exp)
    length = float(exp.run_seconds if run_seconds is None else run_seconds)
    if length <= 0.0:
        raise ValueError(f"run_seconds must be positive, got {length}")

    started = time.perf_counter()
    quarantine_orphan_wrfouts(outdir)
    catalog = build_input_catalog(data)
    snapshots = runtime.forcing_snapshots(data, catalog)
    runtime.forcing_schedule(exp, data, snapshots)
    prepared = runtime.prepare_experiment_case(
        exp, data, input_catalog=catalog, forcing_by_time=snapshots)

    state = prepared.initial_result.state
    prepared_sha = live_state_sha256(state)
    spacing_km = (dc.run.dx / 1000.0, dc.run.dy / 1000.0)

    # A leg that restarts from a checkpoint is NOT perturbed.  Whatever
    # this seam wrote onto the prepared state, ``restore_restart`` inside
    # the integrator overwrites it before the first step, so perturbing
    # here produced a receipt describing a state that never existed: a
    # recorded increment that did not reach the forecast, an
    # ``initial_state_sha256`` that was not the state the leg started
    # from, and an unchanged-state guard comparing the final state to an
    # unrelated pre-restore draw.  Initial-condition perturbations belong
    # to leg 0; a cycling leg begins at its analysis, which is the point.
    if restart is not None:
        report = {}
        changed = False
        perturbed_sha = prepared_sha
        diagnostics = {
            "contract": "gpuwm.da.perturb/post_conditions/update_diagnostics",
            "ran": False,
            "reason": "this leg restarts from a checkpoint, so the prepared "
                      "state is overwritten before the first step and was "
                      "not perturbed",
        }
        skipped_reason = (
            "restart legs are not perturbed: restore_restart overwrites the "
            "prepared state, so an initial-condition perturbation applied "
            "here would be recorded and then discarded")
    else:
        report = dict(hook(state, seed, dict(perturbation_options or {}),
                           grid_spacing_km=spacing_km) or {})
        perturbed_sha = live_state_sha256(state)
        changed = perturbed_sha != prepared_sha
        # The caller post-condition, honoured exactly when there is
        # something to honour it for.  A hook that left the prognostics
        # byte-identical cannot have staled the diagnostics
        # ``prepare_experiment_case`` just computed from them, and
        # re-running the EOS kernel over an untouched state would be work
        # whose only effect is to make the "did nothing" path differ from
        # the one the preparation left behind.
        diagnostics = (
            refresh_diagnostics(state,
                                hypsometric_opt=dc.run.hypsometric_opt)
            if changed else
            {"contract":
                "gpuwm.da.perturb/post_conditions/update_diagnostics",
             "ran": False,
             "reason": "the hook left the prognostic state byte-identical, "
                       "so p/al/alt are still the ones prepared from it"})
        skipped_reason = None

    detail = {
        "provenance": hook.provenance,
        "resolved": dict(hook.detail),
        "state_sha256_before": prepared_sha,
        "state_sha256_after": perturbed_sha,
        "changed_state": changed,
        "applied": restart is None,
        "post_conditions": diagnostics,
        "hook_report": report,
    }
    if skipped_reason is not None:
        detail["skipped_reason"] = skipped_reason

    # The baseline the unchanged-state guard compares against is the state
    # the integrator is about to advance -- AFTER the diagnostic refresh,
    # because p/al/alt are inside the hashed inventory and the refresh
    # moves them.  Hashing before the refresh made the guard pass
    # automatically on exactly the thermodynamic perturbations it exists
    # to protect.
    #
    # On a restart leg the integrator replaces the state wholesale, so
    # BOTH halves of the baseline come from the RESTORED object -- the
    # checkpoint's own hash and the checkpoint's own clock.  Taking the
    # clock from the prepared state instead was the residual hole: a no-op
    # integrator left the prepared state untouched with its clock at 0,
    # the guard compared the prepared hash against the checkpoint hash,
    # found them different, and recorded a member claiming a 60 s advance
    # over a state nothing had restored, let alone integrated.
    pre_restore_sha = live_state_sha256(state)
    if restart is not None:
        baseline_sha = checkpoint_state_sha256(restart)
        baseline_what = f"the checkpoint {Path(restart).name} it restored"
        elapsed_before = checkpoint_elapsed_seconds(restart)
        clock_what = f"the checkpoint {Path(restart).name}'s clock"
    else:
        baseline_sha = pre_restore_sha
        baseline_what = "the state it was handed"
        elapsed_before = _elapsed_seconds(state)
        clock_what = "its own clock before the leg"
    initial_sha = baseline_sha

    summary = runtime.integrate_prepared_case(
        outdir, prepared, start_time=exp.start_time,
        output_title=data.output_title, domain_id=data.output_domain,
        run_seconds=length, history_interval_s=dc.history_interval_s,
        restart_interval_s=exp.restart_interval_s, restart_path=restart,
        progress_callback=progress_callback)
    wall_seconds = time.perf_counter() - started

    final = live_state_sha_receipt(state)
    final_sha = final["sha256"]
    elapsed_after = _elapsed_seconds(state)
    clock_advanced = (elapsed_before is not None
                      and elapsed_after is not None
                      and elapsed_after > elapsed_before)
    if final_sha == baseline_sha and not clock_advanced:
        raise RuntimeError(
            "the state object this engine hashed is byte-identical to "
            f"{baseline_what} after integrating {length:g} s, and its "
            f"clock did not advance past {clock_what} either, so it is not "
            "the object the runner advanced; the member's "
            "final_state_sha256 would be a lie. Refusing to record it.")
    # A restart leg has a second thing to prove: that the restore reached
    # THIS object at all.  ``restore_restart`` overwrites every serialised
    # array, so a state still byte-identical to its pre-restore self is a
    # state the restore never touched -- and hashing it produces a
    # final_state_sha256 for a forecast that did not happen.  The one
    # legitimate way to observe it is a checkpoint that already equals the
    # prepared state, in which case the clock has to carry the proof.
    if restart is not None and final_sha == pre_restore_sha \
            and pre_restore_sha != baseline_sha and not clock_advanced:
        raise RuntimeError(
            "this leg was told to restart from "
            f"{Path(restart).name}, but after integrating {length:g} s the "
            "state object this engine hashed is still byte-identical to "
            "the PRE-RESTORE prepared state and differs from the "
            "checkpoint, and its clock did not advance past the "
            "checkpoint's. The restore never reached this object, so "
            "sim_seconds and final_state_sha256 would describe a forecast "
            "that did not happen. Refusing to record it.")
    # The frame inventory is taken from the paths the integrator itself
    # reports, not from a glob: a glob would inventory whatever is in the
    # directory, which is precisely the thing the inventory exists to
    # distinguish the run's own output from.
    inventory = member_inventory(summary.wrfout_paths, member_dir=outdir)
    return MemberOutcome(
        index=index,
        seed=seed,
        member_dir=outdir,
        initial_state_sha256=initial_sha,
        final_state_sha256=final_sha,
        wall_seconds=wall_seconds,
        sim_seconds=float(summary.completed_seconds),
        wrfout_count=len(summary.wrfout_paths),
        last_checkpoint=None,
        perturbation=detail,
        final_state_inventory=final["inventory"],
        wrfout_inventory=inventory,
    )


def _elapsed_seconds(state):
    """``state.elapsed_seconds`` as a float, or ``None`` if it has none.

    An exact steady state advanced through a valid integration can leave
    every hashed array untouched -- ``perturbation="none"`` over a state
    at rest is the honest example -- and rejecting that as "the runner did
    not advance this object" was a false refusal.  The clock is the
    independent witness, and it is not in the hashed inventory.
    """
    value = getattr(state, "elapsed_seconds", None)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
