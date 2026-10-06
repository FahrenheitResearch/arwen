"""Bounded checkpoint recovery for static prepared nested forecasts.

Only a typed full-state health failure enters this path. The original gate
and complete restart identity checks remain unchanged. Fixed clocks, moving
topologies and ensemble capture fail closed because their rollback has not
been qualified here.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
import json
import re

from gpuwm.core.health import HealthCheckError

MAX_HEALTH_RETRIES = 2
RECOVERY_RECEIPT = "stability-recovery.json"


class RecoveryRefused(ValueError):
    """No proven checkpoint or safe adaptive policy is available."""


def _interval(whole, denominator):
    return (None if int(whole) < 0 else
            Fraction(int(whole), int(denominator) or 1))


def reduced_clock_policy(run, *, failed_dt: Fraction,
                         checkpoint_dt: Fraction):
    """Halve the effective step and hard cap, on the 0.01 s clock lattice."""
    from gpuwm.core.adaptive_clock import wrf_default_clamps

    if not run.use_adaptive_time_step:
        raise RecoveryRefused("health recovery requires an adaptive clock; "
                              "a fixed-clock step remains restart identity")
    old_max = _interval(run.max_time_step, run.max_time_step_den)
    if old_max is None:
        old_max = Fraction(wrf_default_clamps(run.dx, run.dy)[1])
    proposed = min(old_max, failed_dt, checkpoint_dt) / 2
    cap = Fraction(int(proposed * 100), 100)
    if cap <= 0:
        raise RecoveryRefused("halving the step leaves no positive 0.01 s "
                              "adaptive clock interval")
    old_min = _interval(run.min_time_step, run.min_time_step_den)
    if old_min is None:
        old_min = Fraction(wrf_default_clamps(run.dx, run.dy)[2])
    floor = min(old_min, cap)
    return replace(
        run, starting_time_step=cap.numerator,
        starting_time_step_den=cap.denominator,
        max_time_step=cap.numerator, max_time_step_den=cap.denominator,
        min_time_step=floor.numerator, min_time_step_den=floor.denominator,
        target_cfl=run.target_cfl / 2, target_hcfl=run.target_hcfl / 2), {
            "failed_step_s": float(failed_dt),
            "checkpoint_step_s": float(checkpoint_dt),
            "maximum_step_s": [float(old_max), float(cap)],
            "minimum_step_s": [float(old_min), float(floor)],
            "target_cfl": [run.target_cfl, run.target_cfl / 2],
            "target_hcfl": [run.target_hcfl, run.target_hcfl / 2],
            "maximum_step_exact": {"numerator": cap.numerator,
                                   "denominator": cap.denominator},
        }


class NestedHealthRecovery:
    """Execute at most three legs, keeping every decision in a small receipt."""

    def __init__(self, *, model, experiment, output_directory, writers=None,
                 history=None, observer=None, before_rewind=None,
                 sealed_forcing_extension=False):
        self.model = model
        self.experiment = experiment
        self.output_directory = Path(output_directory)
        self.writers = writers
        self.history = history if history is not None else []
        self.observer = observer
        self.before_rewind = before_rewind
        self.sealed_forcing_extension = sealed_forcing_extension
        self.receipt = {
            "schema": "gpuwm.nested-health-recovery/v1",
            "enabled_by_default": True, "maximum_retries": MAX_HEALTH_RETRIES,
            "step_reduction_factor": 0.5, "status": "NOT_NEEDED",
            "attempts": [], "failures": [],
        }

    @property
    def receipt_path(self):
        return self.output_directory / "evidence" / RECOVERY_RECEIPT

    def _publish(self):
        from gpuwm.supervisor import atomic_write_json
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.receipt_path, self.receipt)

    def _announce_attempt(self, attempt, reason, *, notify_observer=True):
        event = {"event": "stability_retry", **attempt,
                 "phase": attempt["status"].lower(),
                 "receipt": str(self.receipt_path)}
        print("gpuwm stability recovery: " + json.dumps(event, sort_keys=True),
              flush=True)
        warn = getattr(self.observer, "warn", None)
        if notify_observer and warn is not None:
            warn("stability_retry", reason, recovery=event)

    def run(self, execute):
        """``execute`` receives the effective experiment for the current leg."""
        self._publish()
        while True:
            try:
                result = execute(self.experiment)
            except HealthCheckError as error:
                phase = error.report.phase or ""
                domains = re.findall(r"d[0-9]+", phase)
                self.receipt["failures"].append({
                    "error_type": type(error).__name__, "error": str(error),
                    "phase": phase, "variable": error.report.first_bad_field,
                    "domain": None if not domains else domains[-1],
                    "failed_model_seconds": float(
                        self.model.root.clock.elapsed_seconds),
                })
                # Preserve the original gate before checkpoint validation or
                # observers can fail. Every retry has a visible start and end.
                self._publish()
                if len(self.receipt["attempts"]) >= MAX_HEALTH_RETRIES:
                    self.receipt["status"] = "EXHAUSTED"
                    self._publish()
                    error.add_note("nested health recovery exhausted its two "
                                   f"retries; receipt: {self.receipt_path}")
                    raise
                previous_attempts = len(self.receipt["attempts"])
                try:
                    self._recover(error)
                except BaseException as refusal:
                    interrupted = not isinstance(refusal, Exception)
                    status = "INTERRUPTED" if interrupted else "REFUSED"
                    detail_key = "terminal_error" if interrupted else "refusal"
                    self.receipt["status"] = status
                    self.receipt[detail_key] = (
                        f"{type(refusal).__name__}: {refusal}")
                    attempt = None
                    if len(self.receipt["attempts"]) > previous_attempts:
                        attempt = self.receipt["attempts"][-1]
                        attempt["status"] = status
                        attempt[detail_key] = self.receipt[detail_key]
                    self._publish()
                    if attempt is not None:
                        self._announce_attempt(
                            attempt, f"nested health recovery {status.lower()}: "
                            f"{refusal}", notify_observer=False)
                    if interrupted:
                        refusal.add_note("nested health recovery interrupted; "
                                         f"original gate: {error}; "
                                         f"receipt: {self.receipt_path}")
                        raise
                    error.add_note("nested health recovery refused: "
                                   f"{refusal}; receipt: {self.receipt_path}")
                    raise error from refusal
            except BaseException as error:
                self.receipt["status"] = (
                    "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED")
                self.receipt["terminal_error"] = (
                    f"{type(error).__name__}: {error}")
                self._publish()
                raise
            else:
                self.receipt["status"] = (
                    "RECOVERED" if self.receipt["attempts"] else "NOT_NEEDED")
                self._publish()
                return result

    def _recover(self, error):
        from gpuwm.core.health import health_validator_for_domain
        from gpuwm.core.model import publish_declared_experiment
        from gpuwm.ensemble.runtime_context import current_capture
        from gpuwm.io.restart import (
            tree_restart_members, read_restart_header, restore_tree_restart)
        from gpuwm.supervisor import restart_attempt, validate_manifest_checkpoint

        exp = self.experiment
        if (exp.relocation.enabled or any(
                getattr(dc, "spawn", None) is not None
                or getattr(dc, "follow", None) is not None
                or getattr(dc, "retire", None) is not None
                or (getattr(dc, "start_time", None) is not None
                    and dc.start_time != exp.start_time)
                for dc in exp.domains)):
            raise RecoveryRefused("automatic rollback of moving, spawned or "
                                  "delayed domain topology is not qualified")
        if current_capture() is not None:
            raise RecoveryRefused("automatic rollback of an ensemble capture "
                                  "would retain failed-leg aggregates")
        checkpoint = getattr(self.model, "_last_checkpoint", None)
        if checkpoint is None:
            raise RecoveryRefused("no last committed checkpoint; hourly "
                                  "restarts must be configured before failure")
        checkpoint = Path(checkpoint).resolve(strict=True)
        if checkpoint.parent != self.output_directory.resolve():
            raise RecoveryRefused("last checkpoint is outside this run's "
                                  "output directory")
        paths = tree_restart_members(checkpoint)
        if set(paths) != set(self.model.nodes_by_grid_id):
            raise RecoveryRefused("checkpoint topology differs from this live "
                                  "tree; automatic topology rollback is not qualified")
        headers = {gid: read_restart_header(validate_manifest_checkpoint(path))
                   for gid, path in paths.items()}
        root_header = headers[int(self.model.root.cfg.grid_id)]
        checkpoint_seconds = Fraction(int(root_header["elapsed_ticks"]),
                                      int(root_header["tick_den"]))
        if checkpoint_seconds > Fraction(
                int(self.model.root.clock.ticks), self.model.root.clock.tick_den):
            raise RecoveryRefused("checkpoint is later than the failing "
                                  "model step; refusing to skip model time")
        new_domains, changes = [], {}
        for dc in exp.domains:
            gid = int(dc.grid_id)
            node = self.model.nodes_by_grid_id[gid]
            adaptive = headers[gid].get("adaptive_clock")
            if adaptive is None:
                raise RecoveryRefused(f"d{gid:02d} checkpoint has no adaptive "
                                      "clock state")
            checkpoint_dt = Fraction(int(adaptive["step_ticks"]),
                                     int(headers[gid]["tick_den"]))
            failed_dt = Fraction(int(node.clock.step_ticks), node.clock.tick_den)
            new_run, changes[f"d{gid:02d}"] = reduced_clock_policy(
                node.cfg.run, failed_dt=failed_dt, checkpoint_dt=checkpoint_dt)
            new_domains.append(replace(dc, run=new_run))
        effective = replace(exp, domains=tuple(new_domains))
        attempt = {
            "retry": len(self.receipt["attempts"]) + 1,
            "checkpoint": str(checkpoint),
            "checkpoint_model_seconds": float(checkpoint_seconds),
            "cause": str(error), "clock_policy_changes": changes,
            "status": "VALIDATING_RESTORE", "discarded_outputs": [],
            "discarded_leg_force_observations": {
                f"d{int(node.cfg.grid_id):02d}": dict(node.coupler.transition_receipt())
                for node in self.model.walk_parent_first()
                if node.coupler is not None
                and callable(getattr(node.coupler, "transition_receipt", None))},
        }
        self.receipt["attempts"].append(attempt)
        self.receipt["status"] = "RETRYING"
        self._publish()
        reason = (f"full-state health failure; retry {attempt['retry']}/"
                  f"{MAX_HEALTH_RETRIES} from {float(checkpoint_seconds):g} s "
                  "with half-sized adaptive step caps")
        self._announce_attempt(attempt, reason)
        if self.writers is not None:
            self.writers.drain()
        # Policy alone may change. Full state/setup/physics identity validation
        # remains in restore_tree_restart and validates the whole set first.
        old_configs = {gid: node.cfg for gid, node in
                       self.model.nodes_by_grid_id.items()}
        try:
            for dc in effective.domains:
                self.model.nodes_by_grid_id[int(dc.grid_id)].cfg = dc
            info = restore_tree_restart(
                checkpoint, self.model,
                sealed_forcing_extension=self.sealed_forcing_extension)
        except BaseException:
            for gid, cfg in old_configs.items():
                self.model.nodes_by_grid_id[gid].cfg = cfg
            raise
        for node in self.model.walk_parent_first():
            health_validator_for_domain(self.model, node).require_healthy(
                phase=f"stability-retry-restored.d{int(node.cfg.grid_id):02d}")
        # The observer stops render consumers before any output is rewound.
        restart_attempt(self.observer, reason)
        if self.before_rewind is not None:
            self.before_rewind(checkpoint)
        cutoff = exp.start_time + timedelta(seconds=float(checkpoint_seconds))
        if self.writers is not None:
            def record_deletion_plan(records):
                attempt["discarded_outputs"] = records
                attempt["output_deletion_status"] = "PLANNED"
                self._publish()

            attempt["discarded_outputs"] = self.writers.rewind_to_checkpoint(
                cutoff, marker_directory=self.output_directory / "ready",
                before_delete=record_deletion_plan)
            attempt["output_deletion_status"] = "DELETED"
        self.history[:] = [row for row in self.history
                           if int(row["ticks"]) <= info.elapsed_ticks]
        self.experiment = effective
        publish_declared_experiment(self.model, effective)
        attempt["status"] = "RESUMED"
        self._publish()
        self._announce_attempt(attempt, reason)
