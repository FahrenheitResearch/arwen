"""Activate spawn-triggered nests at leg boundaries.

The companion to :class:`gpuwm.core.relocation_runner.RelocationRunner`,
and deliberately a DIFFERENT cadence granularity, for a structural
reason worth stating once:

A relocation changes a live child's ``i_parent_start``/``j_parent_start``
and nothing the schedule depends on, so the runner mutates the node in
place inside one :func:`gpuwm.core.model.execute_experiment` call.  A
SPAWN adds a domain to the tree, and
:func:`gpuwm.core.clock.build_schedule` bakes each domain's activation
tick into precomputed per-period op lists
(``active = {gid for gid in step if starts[gid] <= start_ticks}``) --
a trigger-driven domain has no activation tick when that expansion runs,
so it cannot be in the schedule, and no in-place hook can add it.
Activation is therefore leg-boundary schedule surgery, exactly as
``docs/nest-spawn-at-trigger.md`` specifies: the leg before a fire
integrates ``active_experiment(exp)``, the leg after integrates
``active_experiment(exp, {gid: (i, j)})``.

This module is the mechanism for that boundary.  Construction is the
route's job because the route owns the two things neither the mechanism
nor the policy can invent: the ``on_child_built`` preparer that gives a
newborn child its physics/land driver, and the assembler that rebuilds
the next leg's tree.  The runner supplies the assembler's missing half
through :meth:`SpawnRunner.child_initializer`.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from gpuwm.core.uh_diag import (UH_SPAWN_WINDOW_SLOT,
                               reset_tracker_window)

SPAWN_RUNNER_CONTRACT = "gpuwm.spawn-runner.v1"

#: The keys of the checkpoint header's ``nest_lifecycle.spawn`` block, in
#: the order ``docs/nest-lifecycle.md`` states them.  This is the whole
#: surface :meth:`SpawnRunner.state_json` writes and
#: :meth:`SpawnRunner.restore_state` reads: every one of them decides the
#: NEXT leg boundary, so the block is honored in full or refused, never
#: partially defaulted.
SPAWN_STATE_KEYS = ("watches", "spawned", "retired", "episodes",
                    "quiet_since", "spawns_executed")

#: What each field prevents, quoted back in the refusal so a truncated or
#: hand-edited block says which decision it would have silently changed.
_SPAWN_STATE_BREAKAGE = {
    "watches": "a fired slot re-fires and a closed window reopens",
    "spawned": "a live episode loses its fired placement and its birth time",
    "retired": "the re-arm cooldown restarts from the resume instant",
    "episodes": "max_firings miscounts and a one-shot slot fires again",
    "quiet_since": "a sustained-decay timer holds instead of retiring",
    "spawns_executed": "the run's own account of what it did is wrong",
}

#: How many HELD boundary records the in-memory ledger keeps.
#:
#: A held record says the tree did not change, and there is one per leg
#: boundary for the whole run -- ~4,600 of them at 384 h.  The FILE keeps
#: every one of them (that is the audit trail); memory keeps a recent
#: window, so the ledger a caller embeds in its own receipt is bounded by
#: the CONFIG rather than by the forecast length.  Decisions -- a spawn,
#: a retirement, a re-arm, the closing summary -- are never evicted:
#: their count is bounded by slots x max_firings, so keeping them all
#: costs nothing that grows.
#:
#: 64 is a window, not a budget: enough that a failure's recent history
#: is in hand without opening the file, small enough that the ledger is
#: the same size at hour 400 as at hour 4.
HELD_RECEIPTS_KEPT = 64

#: The record shape of the receipts stream: ONE COMPLETE JSON OBJECT PER
#: LINE, appended and flushed as each boundary is decided.
#:
#: It replaces a whole-file rewrite.  ``_atomic_json`` re-serialised the
#: entire ledger at every boundary, so the bytes a run wrote grew as the
#: SQUARE of its length -- at 384 h that is tens of gigabytes of writes
#: to a diagnostic nobody reads until the run ends.  It also republished
#: through a temporary and ``os.replace``, which is the exact pattern
#: that killed a 6 h run on Windows when a user tailed the file
#: (WinError 5); the track writer already moved off it for that reason
#: (gpuwm.core.storm_track_writer._Stream.open) and this follows it.
#:
#: Every line carries its own ``contract``, so a line stands alone: a
#: killed process loses nothing earlier, and a reader never has to parse
#: a truncated array.
RECEIPTS_SUFFIX = ".jsonl"

_LOG = logging.getLogger(__name__)


class SpawnRunnerRefusal(RuntimeError):
    """A spawn runner was asked for something it must not invent."""


def _finite(value, what: str) -> float:
    """A lifecycle timer that survives the header's ``allow_nan=False``.

    Refused HERE, where the message can name the timer, rather than inside
    ``json.dumps`` at the end of ``write_tree_restart`` -- there it aborts
    the whole checkpoint write mid-run and names only the encoder.
    """
    number = float(value)
    if not math.isfinite(number):
        raise SpawnRunnerRefusal(
            f"{what} is {value!r}; the checkpoint header is written with "
            "allow_nan=False, so a non-finite lifecycle timer would abort "
            "the whole checkpoint write and name only the JSON encoder")
    return number


def _placement(value, what: str) -> tuple[int, int]:
    try:
        pair = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        raise SpawnRunnerRefusal(
            f"{what} must be a two-element parent-cell placement, got "
            f"{value!r}") from None
    if len(pair) != 2:
        raise SpawnRunnerRefusal(
            f"{what} must be a two-element parent-cell placement, got "
            f"{value!r}")
    return pair


class SpawnRunner:
    """Consult every dormant nest at each leg boundary; activate the fired.

    ``statics_provider`` is the per-footprint own-grid static builder
    (:func:`gpuwm.ingest.nest_spawn_init.prepare_spawn_statics` bound to
    the route's catalog), or ``None`` for the statics-free branch, where
    the newborn keeps the parent-SINT ground exactly as every idealized
    cold start does.  It is a parameter rather than a config key for the
    same reason ``staging`` is on the relocation runner: it changes what
    the route must have on disk and nothing a config can assert.
    """

    def __init__(self, *, experiment, on_child_built, controller=None,
                 statics_provider=None, receipts_path=None,
                 array_module=None):
        from gpuwm.core.nest_spawn import SpawnController

        if on_child_built is None:
            raise SpawnRunnerRefusal(
                "SpawnRunner requires the route's on_child_built preparer: "
                "the spawn path attaches no physics driver, and a nest born "
                "without one integrates nothing")
        if controller is None:
            controller = SpawnController.from_experiment(experiment)
        if controller is None:
            raise SpawnRunnerRefusal(
                "this experiment declares no dormant ([[domain]] spawn = "
                "{...}) nest; a spawn runner with nothing to watch is a "
                "silent no-op, so this refuses.  Use "
                "SpawnRunner.from_experiment, which returns None instead")
        self.experiment = experiment
        self.controller = controller
        self.on_child_built = on_child_built
        self.statics_provider = statics_provider
        self.array_module = array_module
        self.receipts_path = (None if receipts_path is None
                              else Path(receipts_path))
        #: The append-only stream, opened on the first record.  Lazy so a
        #: runner that is constructed and never consulted leaves no file
        #: behind claiming a boundary happened.
        self._receipt_stream = None
        #: Whether this runner has ever opened its stream.  Latches the
        #: truncate to the FIRST record so a reopen appends.
        self._receipts_opened = False
        #: How many HELD records the in-memory ledger currently holds.
        self._held_kept = 0
        #: grid_id -> fired (i_parent_start, j_parent_start)
        self.spawned: dict[int, tuple[int, int]] = {}
        #: grid_id -> the live ChildInitResult carried across legs
        self._child_results: dict[int, object] = {}
        self.receipts: list[dict] = []
        self.spawns_executed = 0
        # Episode lifecycle.  The declared experiment remains immutable; these
        # maps are the deterministic live view used to build successive legs.
        from gpuwm.core.nest_lifecycle import RetirementWatch
        self.retired: set[int] = set()
        self.episodes: dict[int, int] = {int(g): 0 for g in controller.watches}
        self.birth_times: dict[int, float] = {}
        self.retired_times: dict[int, float] = {}
        self._retirement = {
            int(dc.grid_id): RetirementWatch(dc.retire)
            for dc in experiment.domains if getattr(dc, "retire", None) is not None}
        #: grid_id -> the CURRENT (post-move) placement a restored block
        #: reported, for the caller's one-hop relocation rebuild.  Empty
        #: on a runner that was not restored.
        self.restored_current_placements: dict[int, tuple[int, int]] = {}

    @classmethod
    def from_experiment(cls, exp, *, on_child_built, statics_provider=None,
                        receipts_path=None,
                        array_module=None) -> "SpawnRunner | None":
        """The route's one-line hookup; ``None`` when nothing is dormant."""
        from gpuwm.core.nest_spawn import SpawnController

        controller = SpawnController.from_experiment(exp)
        if controller is None:
            return None
        return cls(experiment=exp, on_child_built=on_child_built,
                   controller=controller, statics_provider=statics_provider,
                   receipts_path=receipts_path, array_module=array_module)

    # -- the leg views ----------------------------------------------------

    @property
    def active(self):
        """The experiment view the NEXT leg integrates.

        Before any fire this is ``pre_spawn_experiment(exp)`` (dormant
        nests absent); after each fire the fired nests join at their
        trigger-chosen placements.
        """
        from gpuwm.experiment import active_experiment

        return active_experiment(
            self.experiment, dict(self.spawned), retired=self.retired,
            birth_times=dict(self.birth_times))

    @property
    def pending(self) -> tuple[int, ...]:
        return self.controller.pending

    @property
    def needs_boundaries(self) -> bool:
        """Whether later leg boundaries can still change the active tree."""
        active_retire = any(gid in self.spawned for gid in self._retirement)
        can_rearm = any(
            getattr(self.experiment.domain(gid), "rearm", None) is not None
            and self.episodes.get(gid, 0) < int(self.experiment.domain(gid).rearm.max_firings)
            for gid in self.retired)
        return bool(self.controller.pending or active_retire or can_rearm)

    def next_decision_time(self, t: float) -> float | None:
        """The earliest model time a leg boundary could change the tree.

        ``None`` means "the next boundary, whenever it is": something
        here reads the LIVE FIELD, so the instant is unknowable and the
        walk must keep asking.  A number means nothing can happen before
        it, and :func:`gpuwm.runtime.spawn_leg_boundary` may run the leg
        straight to it.

        THE DEFECT THIS CLOSES.  :attr:`needs_boundaries` stays true for
        the whole run whenever any retired slot can still re-arm, and
        every boundary it grants costs one full schedule rebuild
        (:func:`gpuwm.runtime.walk_spawn_legs`).  A 384 h run with a
        re-armable slot therefore rebuilt its schedule ~4,600 times to
        discover ~4,600 times that a cooldown had not elapsed.  A
        cooldown is a KNOWN instant; so is a window that has not opened,
        and so is a manual trigger.  Each of those contributes its own
        instant here instead of a boundary.

        Deliberately EARLY where it cannot be exact: a slot whose re-arm
        also waits on a parent episode reports its cooldown anyway.
        Asking too soon costs one rebuild; asking too late would miss a
        decision, and this is a cost fix, not a policy change.

        A STASH-BACKED WATCH FORFEITS THE WHOLE OPTIMISATION, and that is
        the one place skipping a boundary would change an ANSWER rather
        than a cost.  ``uh`` and ``reflectivity`` are consumer-owned
        windows that this runner ZEROES at every boundary it takes: the
        signal a watch reads is "the strongest since I last looked".
        Skip six hours of boundaries and the next look sees six hours of
        accumulation, so a slot coming off its cooldown would fire
        immediately on rotation that happened while it was spent -- the
        same stale-signal episode the re-arm's own "not at this
        boundary" rule exists to prevent.  Whenever any slot that can
        still act reads one of those planes, every boundary is taken.
        Pressure is exempt because it is reduced from the live column and
        carries no window at all (STASH_BACKED_FIELDS).
        """
        from gpuwm.core.storm_tracking import STASH_BACKED_FIELDS

        t = float(t)
        horizons: list[float] = []
        for gid in sorted(self.controller.watches):
            watch = self.controller.watches[gid]
            spent = (watch.closed and gid not in self.spawned
                     and gid not in self.retired)
            if spent:
                continue
            triggers = [watch.config.trigger]
            retire = self._retirement.get(gid)
            if retire is not None:
                triggers.append(retire.config.trigger)
            if any(name in STASH_BACKED_FIELDS for name in triggers):
                return None
        for gid in sorted(self.controller.watches):
            watch = self.controller.watches[gid]
            if gid in self.spawned:
                # LIVE.  Only a retirement policy can end it.
                retire = self._retirement.get(gid)
                if retire is None:
                    continue
                cfg = retire.config
                born = float(self.birth_times.get(gid, t))
                floor_t = born + float(cfg.min_lifetime_s)
                if cfg.trigger == "time":
                    horizons.append(max(floor_t, born + float(cfg.at_s)))
                    continue
                # A field decay reads the plane.  Before min_lifetime_s
                # every evaluation is a guaranteed hold; after it the
                # instant belongs to the weather.
                if t < floor_t:
                    horizons.append(floor_t)
                    continue
                return None
            if gid in self.retired:
                rearm = getattr(self.experiment.domain(gid), "rearm", None)
                if rearm is None or self.episodes.get(gid, 0) >= int(
                        rearm.max_firings):
                    continue                      # the slot is spent
                retired_t = self.retired_times.get(gid)
                if retired_t is None:
                    return None
                horizons.append(float(retired_t) + float(rearm.cooldown_s))
                continue
            if watch.fired or watch.closed:
                continue
            cfg = watch.config
            if cfg.trigger == "time":
                horizons.append(float(cfg.at_s))
                continue
            if t < float(cfg.earliest_s):
                horizons.append(float(cfg.earliest_s))
                continue
            # Inside its window, a field trigger can fire at any
            # boundary; past latest_s it still owes one boundary, to
            # record that the window closed.
            return None
        if not horizons:
            return None
        return min(horizons)

    def _descendants(self, grid_id: int) -> set[int]:
        out = {int(grid_id)}
        changed = True
        while changed:
            changed = False
            for dc in self.experiment.domains:
                if int(dc.parent_id) in out and int(dc.grid_id) not in out:
                    out.add(int(dc.grid_id)); changed = True
        return out

    def _rearm_ready(self, t: float) -> list[int]:
        ready = []
        for gid in sorted(self.retired):
            dc = self.experiment.domain(gid)
            cfg = getattr(dc, "rearm", None)
            if cfg is None or self.episodes.get(gid, 0) >= int(cfg.max_firings):
                continue
            retired_t = self.retired_times.get(gid)
            if retired_t is None or float(t) - retired_t < float(cfg.cooldown_s):
                continue
            # A nested slot cannot re-arm while its parent episode is absent.
            if int(dc.parent_id) != int(self.experiment.root.grid_id) and int(dc.parent_id) not in self.spawned:
                continue
            ready.append(gid)
        return ready

    # -- the restart state ------------------------------------------------
    #
    # PURE STATE, both directions: no I/O, no model mutation, no
    # checkpoint vocabulary.  ``gpuwm.io.restart`` owns where this block
    # lives in the header; the runner owns only what is in it, which is
    # every piece of policy state a later leg boundary reads and cannot
    # recompute from the fields.

    def state_json(self, model=None) -> dict:
        """The ``spawn`` block of the lifecycle restart header.

        ``model`` is optional and supplies exactly one thing: the CURRENT
        placement of each live spawned child.  ``self.spawned`` keeps the
        FIRED placement, because that is what
        :func:`gpuwm.experiment.validate_spawn_placement` must adjudicate
        again on the way back in, while a follower's post-move position
        lives on ``node.cfg`` and nowhere else (``relocate_child`` mutates
        it in place).  Without a model the two are equal, which is exactly
        true for a run with no follower.
        """
        current = {gid: (int(pos[0]), int(pos[1]))
                   for gid, pos in self.spawned.items()}
        if model is not None:
            for gid in list(current):
                try:
                    node = model.node(int(gid))
                except Exception:
                    continue
                cfg = getattr(node, "cfg", None)
                if cfg is None:
                    continue
                current[gid] = (int(cfg.i_parent_start),
                                int(cfg.j_parent_start))

        spawned: dict[str, dict] = {}
        for gid in sorted(self.spawned):
            fired = _placement(self.spawned[gid], f"d{gid:02d} fired placement")
            if gid not in self.birth_times:
                raise SpawnRunnerRefusal(
                    f"d{gid:02d} is spawned but carries no birth time; a "
                    "retirement policy measures episode age from it, so "
                    "writing the block without one would checkpoint a nest "
                    "that can never reach its min_lifetime_s")
            spawned[str(gid)] = {
                "fired": [fired[0], fired[1]],
                "current": [current[gid][0], current[gid][1]],
                "episode": int(self.episodes.get(gid, 0)),
                "born_t": _finite(self.birth_times[gid],
                                  f"d{gid:02d} born_t"),
            }

        retired: dict[str, dict] = {}
        for gid in sorted(self.retired):
            if gid not in self.retired_times:
                raise SpawnRunnerRefusal(
                    f"d{gid:02d} is retired but carries no retirement time; "
                    "the re-arm cooldown is measured from it, so the slot "
                    "could never re-arm after a resume")
            retired[str(gid)] = {
                "retired_t": _finite(self.retired_times[gid],
                                     f"d{gid:02d} retired_t"),
                "episode": int(self.episodes.get(gid, 0)),
            }

        return {
            "watches": {
                str(gid): {"fired": bool(watch.fired),
                           "closed": bool(watch.closed)}
                for gid, watch in sorted(self.controller.watches.items())},
            "spawned": spawned,
            "retired": retired,
            # Every declared slot, including the never-fired ones at 0, so
            # a restored runner cannot read "absent" as "never fired".
            "episodes": {str(gid): int(self.episodes.get(gid, 0))
                         for gid in sorted(self.controller.watches)},
            "quiet_since": {
                str(gid): (None if watch.quiet_since is None
                           else _finite(watch.quiet_since,
                                        f"d{gid:02d} quiet_since"))
                for gid, watch in sorted(self._retirement.items())},
            "spawns_executed": int(self.spawns_executed),
        }

    def restore_state(self, block) -> dict[int, tuple[int, int]]:
        """Seed this runner from a checkpoint's ``spawn`` block.

        Returns ``grid_id -> CURRENT placement`` for every restored
        episode -- the input to the caller's one-hop relocation rebuild.
        The runner itself is left holding the FIRED placements, which is
        what the next ``active`` view must present.

        The whole block is validated before ANY of it is applied, the same
        candidate posture :meth:`on_leg_boundary` keeps: a refusal must not
        leave a half-seeded runner that then integrates.
        """
        if not isinstance(block, dict):
            raise SpawnRunnerRefusal(
                "the checkpoint's spawn block must be a mapping, got "
                f"{type(block).__name__}")
        unknown = sorted(set(block) - set(SPAWN_STATE_KEYS))
        if unknown:
            raise SpawnRunnerRefusal(
                f"the checkpoint's spawn block carries key(s) {unknown} "
                "this build does not know; seeding the runner from a policy "
                "state it can only read in part is how a resumed run "
                "silently re-fires or holds a slot, so this refuses")
        missing = sorted(set(SPAWN_STATE_KEYS) - set(block))
        if missing:
            named = "; ".join(f"{key}: {_SPAWN_STATE_BREAKAGE[key]}"
                              for key in missing)
            raise SpawnRunnerRefusal(
                f"the checkpoint's spawn block is missing {missing}. Each "
                f"decides the next leg boundary -- {named} -- so a partial "
                "block is refused rather than defaulted")

        declared = set(self.controller.watches)

        def slot(raw, where: str) -> int:
            gid = int(raw)
            if gid not in declared:
                raise SpawnRunnerRefusal(
                    f"the checkpoint's spawn {where} names d{gid:02d}, "
                    "which this experiment does not declare as a dormant "
                    f"spawn slot (declared: {sorted(declared)}); the "
                    "checkpoint and the config describe different trees")
            return gid

        def only(row, allowed, where: str) -> dict:
            if not isinstance(row, dict):
                raise SpawnRunnerRefusal(
                    f"the checkpoint's spawn {where} must be a mapping, got "
                    f"{type(row).__name__}")
            extra = sorted(set(row) - set(allowed))
            short = sorted(set(allowed) - set(row))
            if extra or short:
                raise SpawnRunnerRefusal(
                    f"the checkpoint's spawn {where} has key(s) {extra} this "
                    f"build does not know and is missing {short}; the entry "
                    "is honored in full or refused")
            return row

        watches: dict[int, tuple[bool, bool]] = {}
        for raw, row in dict(block["watches"]).items():
            gid = slot(raw, "watches")
            only(row, ("fired", "closed"), f"watches d{gid:02d}")
            watches[gid] = (bool(row["fired"]), bool(row["closed"]))
        absent = sorted(declared - set(watches))
        if absent:
            raise SpawnRunnerRefusal(
                f"the checkpoint's spawn watches say nothing about "
                f"{['d%02d' % gid for gid in absent]}; a slot restored by "
                "omission defaults to un-fired and re-fires at the next "
                "boundary, so every declared slot must appear")

        episodes = {gid: 0 for gid in declared}
        for raw, value in dict(block["episodes"]).items():
            episodes[slot(raw, "episodes")] = int(value)
        for gid, count in episodes.items():
            if count < 0:
                raise SpawnRunnerRefusal(
                    f"d{gid:02d} reports {count} episodes; the count bounds "
                    "rearm max_firings and cannot be negative")

        spawned: dict[int, tuple[int, int]] = {}
        current: dict[int, tuple[int, int]] = {}
        born: dict[int, float] = {}
        for raw, row in dict(block["spawned"]).items():
            gid = slot(raw, "spawned")
            only(row, ("fired", "current", "episode", "born_t"),
                 f"spawned d{gid:02d}")
            if int(row["episode"]) != episodes[gid]:
                raise SpawnRunnerRefusal(
                    f"the checkpoint says d{gid:02d} is in episode "
                    f"{int(row['episode'])} but its episode count is "
                    f"{episodes[gid]}; the two disagree, and the count is "
                    "what bounds rearm max_firings, so this refuses rather "
                    "than picking one")
            spawned[gid] = _placement(row["fired"],
                                      f"d{gid:02d} fired placement")
            current[gid] = _placement(row["current"],
                                      f"d{gid:02d} current placement")
            born[gid] = _finite(row["born_t"], f"d{gid:02d} born_t")

        retired: dict[int, float] = {}
        for raw, row in dict(block["retired"]).items():
            gid = slot(raw, "retired")
            only(row, ("retired_t", "episode"), f"retired d{gid:02d}")
            if gid in spawned:
                raise SpawnRunnerRefusal(
                    f"the checkpoint lists d{gid:02d} as both live and "
                    "retired; one slot serves one episode at a time, and "
                    "restoring both would put a retired nest in the next "
                    "leg's active tree")
            if int(row["episode"]) != episodes[gid]:
                raise SpawnRunnerRefusal(
                    f"the checkpoint says d{gid:02d} retired out of episode "
                    f"{int(row['episode'])} but its episode count is "
                    f"{episodes[gid]}; a miscounted episode re-arms a slot "
                    "past its declared max_firings")
            retired[gid] = _finite(row["retired_t"], f"d{gid:02d} retired_t")

        quiet: dict[int, float | None] = {}
        for raw, value in dict(block["quiet_since"]).items():
            gid = int(raw)
            if gid not in self._retirement:
                raise SpawnRunnerRefusal(
                    f"the checkpoint's spawn quiet_since names d{gid:02d}, "
                    "which declares no retire table in this experiment; the "
                    "checkpoint and the config describe different policy")
            quiet[gid] = (None if value is None
                          else _finite(value, f"d{gid:02d} quiet_since"))

        executed = int(block["spawns_executed"])
        if executed < 0:
            raise SpawnRunnerRefusal(
                f"spawns_executed is {executed}; the run cannot have "
                "un-spawned a nest")

        # Validated in full; commit.
        for gid, (fired, closed) in watches.items():
            watch = self.controller.watches[gid]
            watch.fired = fired
            watch.closed = closed
        self.spawned = dict(spawned)
        self.episodes = dict(episodes)
        self.birth_times = dict(born)
        self.retired = set(retired)
        self.retired_times = dict(retired)
        for gid, watch in self._retirement.items():
            watch.quiet_since = quiet.get(gid)
        self.spawns_executed = executed
        self.restored_current_placements = dict(current)
        return dict(current)

    # -- receipts ---------------------------------------------------------

    def _append_to_stream(self, entry: dict) -> None:
        """One line, flushed.  See :data:`RECEIPTS_SUFFIX` for why."""
        if self.receipts_path is None:
            return
        if self._receipt_stream is None:
            self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate ONCE, on this runner's first record, and append
            # for the rest of its life -- one file per run segment, never
            # renamed.  Each segment owns its own ledger
            # (docs/nest-lifecycle.md), and the runner already refuses an
            # --outdir that holds a previous run, so a fresh directory is
            # a fresh file.
            #
            # The mode is latched because close_receipt() closes the
            # handle: a caller that records anything after it -- a second
            # close, a driver that summarises twice -- would otherwise
            # truncate the whole ledger to that one line.
            mode = "a" if self._receipts_opened else "w"
            self._receipts_opened = True
            self._receipt_stream = self.receipts_path.open(
                mode, encoding="utf-8", newline="\n")
        self._receipt_stream.write(
            json.dumps(entry, sort_keys=True, default=str) + "\n")
        # Flushed per record: a killed process loses nothing, because the
        # bytes are already with the OS and every record is one complete
        # line, so nothing earlier can be corrupted.
        self._receipt_stream.flush()

    def _retain(self, entry: dict) -> None:
        """Keep the entry in memory, bounded in the run's LENGTH.

        Every decision is kept -- their count is bounded by the config.
        Held boundaries are the stream that grows with the forecast, so
        the oldest is evicted once the window is full.  The window is
        the RECENT end, because ``receipts[-1]`` is what every caller
        reads it for.
        """
        self.receipts.append(entry)
        if entry.get("event") != "held":
            return
        self._held_kept += 1
        while self._held_kept > HELD_RECEIPTS_KEPT:
            for index, row in enumerate(self.receipts):
                if row.get("event") == "held":
                    del self.receipts[index]
                    self._held_kept -= 1
                    break
            else:                                     # pragma: no cover
                self._held_kept = 0

    def _record(self, entry: dict) -> dict:
        entry = {"contract": SPAWN_RUNNER_CONTRACT, **entry}
        rows = self.controller.drain_receipts()
        if rows:
            entry["watch_receipts"] = rows
        self._append_to_stream(entry)
        self._retain(entry)
        return entry

    # -- the leg-boundary hook --------------------------------------------

    def refresh_from_model(self, model) -> None:
        """Carry each live spawned child forward into the next leg.

        A relocation between two legs REPLACES the child's state object
        (``relocate_child`` rebuilds it), so the adoption source is the
        model's live node, never the object the spawn receipt captured.
        """
        from gpuwm.ingest.nest_init import ChildInitResult

        for gid in list(self._child_results):
            try:
                node = model.node(gid)
            except Exception:
                continue
            if node is None or not bool(getattr(node, "_started", True)):
                continue
            self._child_results[gid] = ChildInitResult(
                state=node.state, grid=node.grid, coord=None, real=None,
                static_fields=None, horizontal=None, soil=None,
                domain=node.cfg)

    def on_leg_boundary(self, model, *, t=None) -> dict | None:
        """One spawn opportunity; the activation record, or ``None``.

        Returns ``None`` when nothing fired (the tree for the next leg is
        unchanged); otherwise a record whose ``"experiment"`` is the view
        the next leg must assemble and whose ``"child_initializer"`` is
        the adoption seam that attaches the newborn's node/clock/coupler.
        """
        from gpuwm.core.storm_tracking import NestFootprint
        from gpuwm.ingest.nest_spawn_init import spawn_child_from_parent

        self.refresh_from_model(model)
        if t is None:
            t = float(model.root.clock.elapsed_seconds)
        t = float(t)

        # Re-arm only slots retired on an EARLIER boundary.  Even with a zero
        # cooldown, retire+spawn at the same instant would let stale signal
        # create two episodes with no inactive interval.
        rearmed = []
        for gid in self._rearm_ready(t):
            self.retired.remove(gid)
            self.controller.watches[gid].rearm(
                t=t, episode=self.episodes.get(gid, 0) + 1)
            self._retirement.get(gid) and self._retirement[gid].reset()
            rearmed.append(gid)

        started = [node for node in model.walk_parent_first()
                   if bool(getattr(node, "_started", True))]
        # Retirement decisions are made from the SAME pre-reset leg window as
        # spawning.  They affect only the next leg; no op from the schedule
        # that just completed is skipped.
        retired_roots: list[int] = []
        lifecycle_rows: list[dict] = []
        for gid in sorted(self._retirement):
            if gid not in self.spawned or gid not in model.nodes_by_grid_id:
                continue
            node = model.node(gid)
            if node.parent is None:
                continue
            row = self._retirement[gid].evaluate(
                node.parent.state, node.cfg, t=t,
                born_t=self.birth_times.get(gid, t))
            lifecycle_rows.append(row)
            if row.get("retire"):
                retired_roots.append(gid)

        retired_grid_ids: set[int] = set()
        for gid in retired_roots:
            retired_grid_ids.update(self._descendants(gid))
        # Only declared spawn slots participate in the active-experiment map;
        # runtime detachment below also receives static descendants.
        for gid in sorted(retired_grid_ids):
            if gid in self.spawned:
                self.spawned.pop(gid, None)
                self.retired.add(gid)
                self.retired_times[gid] = t
                watch = self._retirement.get(gid)
                if watch is not None:
                    watch.reset()

        parent_states = {int(node.cfg.grid_id): node.state
                         for node in started
                         if int(node.cfg.grid_id) not in retired_grid_ids}

        # A watch must not read signal that already lives under another
        # nest, so every LIVE child's footprint is excluded; evaluate_all
        # chains the ones fired at this same boundary into the set too.
        active_footprints = tuple(
            NestFootprint.coerce(node.cfg) for node in started
            if node.parent is not None
            and int(node.cfg.grid_id) not in retired_grid_ids)

        events = self.controller.evaluate_all(
            parent_states, t, active_footprints=active_footprints)
        # Every watch has now looked at every parent it watches, so every
        # spawn window is zeroed -- fired, held or excluded alike (Drew's
        # ruling, 2026-08-07: the window is "max since I last looked").
        # This is the spawn consumer's OWN slot; the relocation runner
        # resets a different one on its own cadence, so neither can blind
        # the other at a boundary they happen to share.
        for parent_state in parent_states.values():
            reset_tracker_window(parent_state, UH_SPAWN_WINDOW_SLOT)
        if not events:
            recorded = self._record({
                "event": "lifecycle" if (retired_grid_ids or rearmed) else "held",
                "elapsed_seconds": t, "pending": list(self.controller.pending),
                "retired_grid_ids": sorted(retired_grid_ids),
                "retired_roots": sorted(retired_roots),
                "rearmed_grid_ids": sorted(rearmed),
                "lifecycle_receipts": lifecycle_rows})
            if retired_grid_ids or rearmed:
                record = dict(recorded)
                record["experiment"] = self.active
                return record
            return None

        # active_experiment FIRST: it is what carries the fired placement
        # onto the DomainConfig (a field trigger's placement is nothing
        # like the declared placeholder) and it re-runs the clearance rule
        # on it.  Materialization then reads that adjudicated config.
        #
        # Computed on a CANDIDATE map and committed only once every nest
        # of this boundary has materialized, so a refusal mid-way leaves
        # self.spawned describing what actually exists.  (The watch itself
        # has already recorded the fire -- the controller owns that state
        # -- so this bounds the damage rather than undoing it.)
        candidate = dict(self.spawned)
        for event in events:
            candidate[int(event.grid_id)] = (
                int(event.i_parent_start), int(event.j_parent_start))
        from gpuwm.experiment import active_experiment

        # The births of THIS boundary join the epoch map before the view
        # is built: the newborn's own clock is minted off this object, so
        # an epoch recorded only after the fire would leave the first
        # child clock -- the one it is born holding -- reading t = 0.
        candidate_births = dict(self.birth_times)
        for event in events:
            candidate_births[int(event.grid_id)] = float(t)
        active = active_experiment(self.experiment, candidate,
                                   birth_times=candidate_births)

        born = []
        for event in events:
            gid = int(event.grid_id)
            child_dc = active.domain(gid)
            parent_node = model.node(int(self.controller.parent_of[gid]))
            statics = (None if self.statics_provider is None
                       else self.statics_provider(child_dc, parent_node))
            receipt = spawn_child_from_parent(
                child_dc, parent_node,
                static_fields=statics,
                blend_width=int(getattr(self.experiment, "blend_width", 5)),
                scratch_arena=getattr(model, "_scratch_arena", None),
                dycore_state_workspace=getattr(
                    model, "_dycore_state_workspace", None),
                array_module=self.array_module,
                on_child_built=self.on_child_built,
                trigger_receipt=event.receipt)
            self._child_results[gid] = receipt["child_result"]
            self.spawned[gid] = candidate[gid]
            self.retired.discard(gid)
            self.retired_times.pop(gid, None)
            self.episodes[gid] = self.episodes.get(gid, 0) + 1
            self.birth_times[gid] = t
            self.spawns_executed += 1
            born.append({
                "episode": int(self.episodes[gid]),
                "grid_id": gid,
                "placement": [int(event.i_parent_start),
                              int(event.j_parent_start)],
                "spawn_receipt": {key: value
                                  for key, value in receipt.items()
                                  if key != "child_result"},
            })

        recorded = self._record({
            "event": "spawned",
            "elapsed_seconds": t,
            "grid_ids": [int(e.grid_id) for e in events],
            "born": born,
            "active_grid_ids": [int(dc.grid_id) for dc in active.domains],
            "pending_after": list(self.controller.pending),
            "retired_grid_ids": sorted(retired_grid_ids),
            "retired_roots": sorted(retired_roots),
            "rearmed_grid_ids": sorted(rearmed),
            "lifecycle_receipts": lifecycle_rows,
            "episode_by_grid_id": {str(g): int(self.episodes[g])
                                   for g in sorted(self.episodes)},
        })
        # The RETURNED record carries live objects for the driver; the
        # RECORDED one must stay JSON-safe, because self.receipts is what
        # gets written to receipts_path and what callers serialise into
        # their own receipts.  Copy rather than mutate in place, or an
        # ExperimentConfig and a closure end up inside the run's
        # durable evidence.
        record = dict(recorded)
        record["experiment"] = active
        record["child_initializer"] = self.child_initializer()
        # The live newborns, for a driver that attaches them to an
        # EXISTING tree in place (the real-data leg walk) rather than
        # re-assembling the tree around them.  Keyed by grid_id, the
        # ``segment_state``/``child_result`` precedent again: a live
        # object beside the JSON, never inside it.
        record["child_results"] = {
            int(e.grid_id): self._child_results[int(e.grid_id)]
            for e in events}
        return record

    # -- the leg-boundary rebuild's missing half ---------------------------

    def child_initializer(self, default=None):
        """Adopt every already-spawned child into the next leg's assembly.

        Hand this to the route's assembler as ``child_initializer``: a
        spawned domain is returned as-is (its live state, grid and coord),
        so the assembler builds the node, binds the clock the rebuilt
        schedule minted for it, and attaches the coupler -- the same
        node/clock/coupler attachment every leg boundary performs.  Any
        other domain falls through to ``default``.
        """
        from gpuwm.ingest.nest_init import ChildInitResult, parent_only_init

        if default is None:
            default = parent_only_init
        results = self._child_results

        def initialize(child_dc, parent_node, **kwargs):
            gid = int(child_dc.grid_id)
            if gid not in results:
                return default(child_dc, parent_node, **kwargs)
            carried = results[gid]
            return ChildInitResult(
                state=carried.state, grid=carried.grid,
                coord=getattr(carried, "coord", None), real=None,
                static_fields=None, horizontal=None, soil=None,
                domain=child_dc)

        return initialize

    # -- the closing summary ----------------------------------------------

    def close_receipt(self, model=None) -> dict:
        """Summarize the run, naming every watch that never fired."""
        unfired = []
        for gid in sorted(self.controller.watches):
            watch = self.controller.watches[gid]
            if int(gid) in self.spawned:
                continue
            unfired.append({
                "grid_id": int(gid),
                "closed": bool(getattr(watch, "closed", False)),
                "note": ("the window closed without the trigger firing; the "
                         "reservation was held for the whole run and cost "
                         "zero compute -- that is the contract, not a leak"
                         if getattr(watch, "closed", False) else
                         "the run ended with this watch still open"),
            })
        entry = self._record({
            "event": "closed",
            "spawns_executed": int(self.spawns_executed),
            "spawned": {str(gid): list(pos)
                        for gid, pos in sorted(self.spawned.items())},
            "never_fired": unfired,
            "episodes": {str(g): int(v) for g, v in sorted(self.episodes.items())},
            "retired": sorted(self.retired),
        })
        if model is not None and getattr(
                model, "_spawn_receipts", None) is not self.receipts:
            model._spawn_receipts = self.receipts
        if self._receipt_stream is not None:
            self._receipt_stream.close()
            self._receipt_stream = None
        return entry


__all__ = ["HELD_RECEIPTS_KEPT", "RECEIPTS_SUFFIX", "SPAWN_RUNNER_CONTRACT",
           "SPAWN_STATE_KEYS", "SpawnRunner", "SpawnRunnerRefusal"]
