"""Retirement and re-arm policy for trigger-spawned nests.

This module deliberately owns policy only.  Tree mutation remains a leg-boundary
operation in :mod:`gpuwm.runtime`, the same schedule-surgery seam used by spawn.
A retirement decision therefore never skips a STEP op from an already-built
schedule: it changes the domain set used to build the *next* leg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gpuwm.core import uh_diag
from gpuwm.core.storm_tracking import (FollowConfig, NestFootprint,
                                       _plane_from_state, build_follow_config)

RETIRE_CONTRACT = "gpuwm-nest-retire.v1"
REARM_CONTRACT = "gpuwm-nest-rearm.v1"
RETIRE_TRIGGERS = ("uh", "reflectivity", "time")
RETIRE_KEYS = frozenset({
    "trigger", "threshold", "sustained_s", "min_lifetime_s", "at_s",
})
REARM_KEYS = frozenset({"max_firings", "cooldown_s"})
DOMAIN_FOLLOW_EXTRA_KEYS = frozenset({
    "cadence_seconds", "max_move_parent_cells", "min_overlap_fraction",
})


@dataclass(frozen=True)
class RetireConfig:
    """When one live spawned nest stops participating in the next leg.

    Field-triggered retirement means the maximum signal under the *live child
    footprint* stays at or below ``threshold`` continuously for ``sustained_s``
    after ``min_lifetime_s`` has elapsed.  ``time`` is deterministic test/manual
    policy and retires after ``at_s`` seconds of that episode's active lifetime.
    """

    trigger: str
    threshold: float | None = None
    sustained_s: float = 0.0
    min_lifetime_s: float = 0.0
    at_s: float | None = None

    def __post_init__(self) -> None:
        if self.trigger not in RETIRE_TRIGGERS:
            raise ValueError(
                f"retire trigger must be one of {RETIRE_TRIGGERS}, got "
                f"{self.trigger!r}")
        for name in ("sustained_s", "min_lifetime_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"retire {name} must be finite and >= 0, got {value!r}")
        if self.trigger == "time":
            if self.at_s is None or not math.isfinite(float(self.at_s)) or float(self.at_s) < 0.0:
                raise ValueError("retire trigger='time' requires finite at_s >= 0 (episode age seconds)")
            if self.threshold is not None:
                raise ValueError("retire trigger='time' refuses threshold; it reads no field")
        else:
            if self.at_s is not None:
                raise ValueError(f"retire trigger={self.trigger!r} refuses at_s; field decay chooses the instant")
            if self.threshold is None or not math.isfinite(float(self.threshold)):
                raise ValueError(f"retire trigger={self.trigger!r} requires a finite threshold")

    def to_json(self) -> dict[str, object]:
        out: dict[str, object] = {
            "contract": RETIRE_CONTRACT,
            "trigger": self.trigger,
            "sustained_s": float(self.sustained_s),
            "min_lifetime_s": float(self.min_lifetime_s),
        }
        if self.threshold is not None:
            out["threshold"] = float(self.threshold)
        if self.at_s is not None:
            out["at_s"] = float(self.at_s)
        return out


@dataclass(frozen=True)
class RearmConfig:
    """How many episodes one declared spawn slot may serve."""

    max_firings: int = 1
    cooldown_s: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_firings, bool) or int(self.max_firings) != self.max_firings or int(self.max_firings) < 1:
            raise ValueError(f"rearm max_firings must be an integer >= 1, got {self.max_firings!r}")
        if not math.isfinite(float(self.cooldown_s)) or float(self.cooldown_s) < 0.0:
            raise ValueError(f"rearm cooldown_s must be finite and >= 0, got {self.cooldown_s!r}")

    def to_json(self) -> dict[str, object]:
        return {
            "contract": REARM_CONTRACT,
            "max_firings": int(self.max_firings),
            "cooldown_s": float(self.cooldown_s),
        }


def _unknown(table: dict, allowed, label: str, source: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ValueError(f"{label} of {source} does not have key(s) {unknown}; lifecycle keys are honored or refused, never ignored")


def build_retire_config(table: dict, source: str, *, grid_id: int) -> RetireConfig:
    label = f"[[domain]] grid_id={grid_id} retire"
    _unknown(table, RETIRE_KEYS, label, source)
    if "trigger" not in table:
        raise ValueError(f"{label} of {source} is missing required key 'trigger'")
    kwargs = dict(table)
    try:
        return RetireConfig(**kwargs)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{label} of {source}: {err}") from None


def build_rearm_config(table: dict, source: str, *, grid_id: int) -> RearmConfig:
    label = f"[[domain]] grid_id={grid_id} rearm"
    _unknown(table, REARM_KEYS, label, source)
    try:
        return RearmConfig(**dict(table))
    except (TypeError, ValueError) as err:
        raise ValueError(f"{label} of {source}: {err}") from None


@dataclass(frozen=True)
class DomainFollowConfig:
    """Per-domain follow policy plus relocation cadence/bounds."""

    tracker: FollowConfig
    cadence_seconds: float
    max_move_parent_cells: int | None = None
    min_overlap_fraction: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.cadence_seconds)) or float(self.cadence_seconds) <= 0.0:
            raise ValueError("follow cadence_seconds must be finite and > 0")
        if self.max_move_parent_cells is not None and int(self.max_move_parent_cells) < 1:
            raise ValueError("follow max_move_parent_cells must be >= 1 when present")
        if self.min_overlap_fraction is not None and not 0.0 <= float(self.min_overlap_fraction) <= 1.0:
            raise ValueError("follow min_overlap_fraction must lie in [0, 1]")

    def to_json(self) -> dict[str, object]:
        return {
            **self.tracker.to_json(),
            "cadence_seconds": float(self.cadence_seconds),
            "max_move_parent_cells": self.max_move_parent_cells,
            "min_overlap_fraction": self.min_overlap_fraction,
        }


def build_domain_follow_config(table: dict, source: str, *, grid_id: int) -> DomainFollowConfig:
    label = f"[[domain]] grid_id={grid_id} follow"
    extras = {key: table[key] for key in DOMAIN_FOLLOW_EXTRA_KEYS if key in table}
    tracker_table = {key: value for key, value in table.items()
                     if key not in DOMAIN_FOLLOW_EXTRA_KEYS}
    unknown = set(tracker_table) - set(__import__("gpuwm.core.storm_tracking", fromlist=["FOLLOW_KEYS"]).FOLLOW_KEYS)
    if unknown:
        raise ValueError(f"{label} of {source} has unknown key(s) {sorted(unknown)}")
    if "cadence_seconds" not in extras:
        raise ValueError(f"{label} of {source} requires cadence_seconds; each follower owns its own evaluation window")
    tracker = build_follow_config(tracker_table, source)
    try:
        return DomainFollowConfig(
            tracker=tracker, cadence_seconds=float(extras["cadence_seconds"]),
            max_move_parent_cells=(None if extras.get("max_move_parent_cells") is None else int(extras["max_move_parent_cells"])),
            min_overlap_fraction=(None if extras.get("min_overlap_fraction") is None else float(extras["min_overlap_fraction"])))
    except ValueError as err:
        raise ValueError(f"{label} of {source}: {err}") from None


class RetirementWatch:
    """Deterministic episode-local decay timer for one live child."""

    def __init__(self, config: RetireConfig) -> None:
        self.config = config
        self.quiet_since: float | None = None
        self.receipts: list[dict] = []

    def reset(self) -> None:
        self.quiet_since = None

    def evaluate(self, parent_state, child_cfg, *, t: float, born_t: float) -> dict:
        cfg = self.config
        t = float(t)
        age = t - float(born_t)
        base = {"contract": RETIRE_CONTRACT, "grid_id": int(child_cfg.grid_id),
                "t": t, "episode_age_s": age}
        if age < float(cfg.min_lifetime_s):
            self.quiet_since = None
            row = {**base, "decision": "hold:min-lifetime"}
            self.receipts.append(row)
            return row
        if cfg.trigger == "time":
            retire = age >= float(cfg.at_s)
            row = {**base, "decision": "retire" if retire else "hold:time",
                   "retire": retire, "at_s": float(cfg.at_s)}
            self.receipts.append(row)
            return row

        plane = _plane_from_state(
            parent_state, cfg.trigger,
            uh_slot=uh_diag.UH_SPAWN_WINDOW_SLOT)
        fp = NestFootprint.coerce(child_cfg)
        box = fp.search_box(plane.shape, 0)
        sample = np.asarray(plane[box])
        maximum = float(np.nanmax(sample)) if sample.size else float("-inf")
        quiet = maximum <= float(cfg.threshold)
        if quiet:
            if self.quiet_since is None:
                self.quiet_since = t
        else:
            self.quiet_since = None
        quiet_for = 0.0 if self.quiet_since is None else t - self.quiet_since
        retire = bool(quiet and quiet_for >= float(cfg.sustained_s))
        row = {**base, "decision": "retire" if retire else ("hold:quiet" if quiet else "hold:signal"),
               "retire": retire, "field": cfg.trigger,
               "threshold": float(cfg.threshold), "max_value": maximum,
               "quiet_for_s": quiet_for, "sustained_s": float(cfg.sustained_s)}
        self.receipts.append(row)
        return row

    def drain_receipts(self) -> list[dict]:
        out, self.receipts = self.receipts, []
        return out


def declares_lifecycle(domain_cfg) -> bool:
    """Whether this domain opts into episode-numbered output.

    ``retire`` and ``rearm`` are the tables that can put a SECOND episode
    through one slot, which is the only thing an episode number
    distinguishes.  A plain one-shot ``spawn`` is not a lifecycle episode:
    it fires once, and numbering its single history run would move every
    existing spawn config's output for no reader's benefit.

    ``follow`` is deliberately excluded.  It relocates a nest within one
    episode -- placement policy, not episode policy -- and the relocated
    frames belong in the same series as the frames before the move.
    """
    return (getattr(domain_cfg, "retire", None) is not None
            or getattr(domain_cfg, "rearm", None) is not None)


def output_episode(domain_cfg, episode: int) -> int:
    """The episode number the history writers should use for one domain.

    ``SpawnRunner.episodes`` counts every firing from 1 because the re-arm
    bound (``max_firings``) is stated in firings.  That counter is NOT the
    writers' pathname input: a domain with no declared lifecycle reports 0
    and keeps the flat ``wrfout_dNN_*`` name it has always written, while
    a declared one reports its true episode and gets ``dNN/episode-NNN/``
    from its FIRST episode, so its layout is the same shape throughout.
    """
    return int(episode) if declares_lifecycle(domain_cfg) else 0


def admit_restart_with_lifecycle(exp, restart) -> bool:
    """Whether this resume must carry lifecycle policy across the split.

    ``True`` when a checkpoint is being restored into an experiment that
    declares ``retire``, ``rearm`` or ``follow``: the run does not merely
    reload arrays, it reloads the POLICY STATE that decides what the tree
    does next -- which slots have fired, which are closed, how long a
    signal has been quiet, where a follower last moved and how many hops
    ago.  The checkpoint now persists every one of those
    (``gpuwm-nest-lifecycle-restart.v1``), and the resume rebuilds the
    tree the block describes before a single array lands.

    This function no longer refuses.  It used to, and the refusal named
    exactly the breakage that has since been built out: a resume that
    re-fired a spent slot, retired on a timer that restarted at zero, or
    moved a nest from hysteresis it did not have.  The refusals that
    remain are the ones a checkpoint can still be WRONG about, and they
    live where the checkpoint is read --
    :func:`gpuwm.io.restart.read_tree_lifecycle_header`:

    * no lifecycle block under a declaring experiment (a checkpoint that
      predates persistence cannot say which slots fired);
    * a lifecycle block under a lifecycle-free run (every fired slot and
      move history would be dropped on the floor);
    * an unknown contract, or a block whose key set this build does not
      read whole;
    * a block naming a domain the member set does not carry, or naming a
      retired domain that still owns a member;
    * a leg cadence other than the one the resuming run stops on, which
      would evaluate the same policy at different instants.

    Kept as a named seam, and returned rather than merely computed, so
    the admission is greppable, reachable in a test without building a
    model, and reported to the user who asked for it.
    """
    if restart is None:
        return False
    return any(getattr(dc, "retire", None) is not None
               or getattr(dc, "rearm", None) is not None
               or getattr(dc, "follow", None) is not None
               for dc in exp.domains)


__all__ = [
    "REARM_CONTRACT", "REARM_KEYS", "RETIRE_CONTRACT", "RETIRE_KEYS",
    "RETIRE_TRIGGERS", "DomainFollowConfig", "RearmConfig", "RetireConfig",
    "RetirementWatch", "admit_restart_with_lifecycle",
    "build_domain_follow_config", "build_rearm_config",
    "build_retire_config", "declares_lifecycle", "output_episode",
]
