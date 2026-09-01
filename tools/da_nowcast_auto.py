"""A nowcast that keeps going: one site, one gallery, no further command.

``python -m tools.da_nowcast_auto start --site XXXX --out DIR`` launches
a detached daemon that, from then on, does by itself what
:mod:`tools.da_nowcast` does once:

    bootstrap   the front door up to its georeference forecast
                (``da_nowcast run --stop-after forecast``): survey,
                domain, fetch, prepare, forecast.  ONE prepared case per
                epoch, and that case's boundary data is what bounds the
                epoch.
    catch up    assimilate every volume between the model's init hour
                and now, oldest first -- a real spin-up on real data
                rather than an hour of free integration
    cycle       once current, take each new volume as it lands, on the
                radar's own rhythm (4-6 minutes in a precipitation VCP,
                not a 15-minute clock)
    forecast    refresh the free legs from the newest analysis
    render      redraw the gallery IN PLACE at one stable path

and then waits.  Refreshing that one page is the whole user experience.

**The ensemble is alive across cycles.**  Each cycle resumes the
generation the last one wrote (:mod:`tools.da_ensemble_state`) and
writes its own, so the covariance is the one the cycling built, not a
fresh perturbation every few minutes.  Free legs branch off that state
and never become it.

**The free forecast refreshes when the analysis is current.**  While the
daemon is still catching up, a 90-minute forecast off a state that is
about to be superseded by the next volume is work nobody will look at;
those cycles run observations only, and the gallery says which mode it
is in.

**Honest degradation, out loud.**  A late volume, a skipped cycle, a
failed stage: each is written into the status file AND stamped on the
gallery page through ``auto-notice.json``.  Nothing is padded, no volume
is re-used as a new one, and the model is never advanced past data it
does not have.

**Epochs end.**  A prepared case carries a finite window of lateral
boundary data.  Approaching it, the daemon says so, boots a new epoch on
a newer background and re-initialises the ensemble -- a generation
written against one prepared case cannot be restored into another, and
the identity check exists to stop anyone pretending otherwise.

**Backgrounds age (EXPERIMENTAL: overlap handover, default OFF).**  A
daemon left running holds one prepared case until its boundary data is
spent, so its background gets older by the hour while the analysis stays
current.  Currency is the whole point of this nowcast, so
``--background-max-age`` opts into the conservative fix: past that age
the daemon prepares a SECOND case from the newest available cycle, spins
a second ensemble up on it by assimilating the same volumes the first one
is assimilating, and once that ensemble has ``--spinup-cycles`` cycles
behind it the gallery and the free forecast switch to it and the old
ensemble is retired -- its state left on disk, nothing deleted.

No state crosses cases and no identity semantics move: the fresh
ensemble is initialised on the fresh case and cycled up from scratch,
exactly as a bootstrap is.  What it costs is transient double compute
during the overlap, and the spin-up lane pays the cheaper half of it --
it runs observations only, never a free forecast, because nobody is
looking at its forecast until it becomes the primary.  Both lanes'
cycles are ordinary child processes run one at a time from this one
loop, so the card sees the same one-run-at-a-time admission it always
has; the spin-up mostly fills the primary's idle time between volumes.

If the fresh case fails to prepare, or its spin-up fails, the daemon
retires the attempt, says so on the gallery, waits out a cooldown and
goes on cycling the old ensemble.  A fresher background is never worth
killing a working nowcast for.

**Runs and commits belong in different worktrees.**  This daemon
fingerprints the run root's git HEAD at start and stops -- loudly, case
and gallery intact -- if it moves.  Point ``--run-root`` at a worktree
nobody commits into.

Site ids are arguments here as everywhere: no station name belongs in
this file, its defaults or its identifiers (standing owner rule).

HONESTY: demo-grade nowcast.  UNSCORED, outside any registered campaign,
EXPERIMENTAL like every tool it drives.  No skill claim is made or
implied; the gallery says so on every figure.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:                                    # python -m tools.da_nowcast_auto
    from tools import da_ensemble_state as ens_state
    from tools.da_nowcast import (NOWCAST_DEFAULT_PHYSICS_PROFILE,
                                  DealiasChoice, RadarSelection, cycle_cmd,
                                  iso, obs_cmd, parse_iso, render_cmd,
                                  resolvable_length_scale_km,
                                  spawn_detached, validate_site)
except ImportError:                     # python tools/da_nowcast_auto.py
    import da_ensemble_state as ens_state
    from da_nowcast import (NOWCAST_DEFAULT_PHYSICS_PROFILE,
                            DealiasChoice, RadarSelection, cycle_cmd,
                            iso, obs_cmd, parse_iso, render_cmd,
                            resolvable_length_scale_km,
                            spawn_detached, validate_site)

#: The status file a page or a person polls.
SCHEMA = "gpuwm-da.nowcast-auto.v1"

#: The banner the renderer puts on the gallery when the daemon has news.
NOTICE_SCHEMA = "gpuwm-da.nowcast-auto-notice.v1"

#: The durable record of every gallery source swap this daemon made.
HANDOVER_SCHEMA = "gpuwm-da.nowcast-handover.v1"

#: The marker left in a lane's directory when it stops being cycled.
RETIRED_SCHEMA = "gpuwm-da.nowcast-retired-lane.v1"

#: Where the daemon can be.  ``spinning-up`` is the overlap-handover
#: lane's cycle: the daemon is on the card for a SECOND ensemble that is
#: not yet the gallery's source.
STATES = ("starting", "bootstrapping", "catching-up", "assimilating",
          "spinning-up", "rendering", "waiting", "stopped", "failed")

#: How often to ask what volumes exist.  Well under the volume cadence,
#: so a new volume is picked up within a fraction of a cycle.
POLL_SECONDS = 45

#: The nominal spacing between volumes in a precipitation VCP.  Used
#: only to decide when a quiet feed has been quiet long enough to say so.
EXPECTED_VOLUME_INTERVAL_SECONDS = 360.0

#: How far past the expected arrival a volume may be before the daemon
#: calls it late ON THE FIGURE.  Two nominal intervals: one missed
#: volume is a hiccup, two is something the viewer has to be told.
LATE_AFTER_SECONDS = 720.0

#: Bounds on one assimilation leg, in seconds of model time.  The floor
#: keeps a near-duplicate volume from asking the model to integrate a
#: handful of steps; the ceiling stops a feed outage from being papered
#: over with one enormous leg nobody observed.
MIN_LEG_SECONDS = 120.0
MAX_LEG_SECONDS = 1800.0

#: How far a georeference wrfout may sit from a cycle's valid time and
#: still be used to grid its observations.  Two georeference intervals:
#: the nearest file on a lattice of that spacing is never more than half
#: an interval away, so anything past this means the lattice is not the
#: one the bootstrap asked for and the staleness is worth refusing over.
GRID_OFFSET_CEILING_SECONDS = 600.0

#: How far a radar volume may sit from the cycle's snapped model time
#: and still be the observation for it.  The snap itself moves the time
#: by at most half a model step; this is slack for the archive's own
#: stamping, not licence to grab a different volume.
OBS_MATCH_CEILING_SECONDS = 240.0

#: How many cycles may fail in a row before the daemon stops.  A cycle
#: that fails does not advance the analysis clock, so retrying forever
#: means every later volume eventually exceeds the leg ceiling and the
#: daemon refuses each one in turn -- busy, loud, and going nowhere.
MAX_CONSECUTIVE_FAILURES = 3

#: Whole hours of boundary data a bootstrap asks for: how long one
#: prepared case can be cycled on before a new epoch is needed.
EPOCH_HOURS = 4

#: Bootstrap cadence.  ``cycles x cycle-seconds`` is what the front door
#: subtracts from the window end to get the model init, and it insists
#: that init lands on a whole hour -- so one 300 s cycle puts init on the
#: most recent whole hour at or before the newest volume, which is the
#: shortest catch-up the front door's rules allow.
BOOTSTRAP_CYCLES = 1
BOOTSTRAP_CYCLE_SECONDS = 300

#: How many cycles a spin-up ensemble must have behind it before the
#: gallery is allowed to switch to it.  From the measured spin-up on the
#: two live cases this daemon has run: analyses became competitive with
#: the running ensemble's by cycle 3-4.  Four is the conservative end of
#: that measurement, and it is an argument because the measurement is
#: two cases, not a law.
HANDOVER_SPINUP_CYCLES = 4

#: How long after a failed handover attempt before another is tried.  A
#: fresh case that will not prepare is usually a fetch or a card problem
#: that lasts minutes, and retrying it every poll would spend the
#: primary's idle time on the same failure forever.
HANDOVER_COOLDOWN_MINUTES = 30.0


class AutoError(SystemExit):
    """A refusal with its reason."""

    def __init__(self, message: str) -> None:
        super().__init__(f"da_nowcast_auto: {message}")


class StageFailure(AutoError):
    """A child process failed.

    Separate from a plain refusal because the two mean opposite things
    about whether to keep going.  A refusal is a statement about the
    DATA -- this volume is too close, that gap is an outage -- and the
    next volume may well be fine.  A stage failure is the machinery not
    working, and machinery that is not working does not usually start
    working because it was asked again forty-five seconds later.
    """


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def log(text: str) -> None:
    print(f"[{now_utc():%Y-%m-%dT%H:%M:%S}Z] {text}", flush=True)


def _py() -> str:
    return sys.executable or "python"


# ---------------------------------------------------------------------------
# pure planning (unit-tested)
# ---------------------------------------------------------------------------
def snap_to_step(seconds: float, dt_s: float) -> float:
    """The nearest model step boundary.

    The integrator only stops on a step and the leg-boundary clock is
    placed by an integer step count, so a volume time is snapped here
    rather than rounded somewhere downstream.  At a 15 s step this moves
    a cycle by at most 7.5 s, and the offset is receipted.
    """

    if dt_s <= 0:
        raise AutoError("model timestep must be positive")
    return round(float(seconds) / float(dt_s)) * float(dt_s)


@dataclass(frozen=True)
class LegPlan:
    """One assimilation leg: how far to advance, and to what time."""

    leg_seconds: float
    end_elapsed_s: float
    valid: datetime
    snap_offset_s: float

    def to_payload(self) -> dict:
        return {"leg_seconds": self.leg_seconds,
                "end_elapsed_s": self.end_elapsed_s,
                "valid": iso(self.valid),
                "snap_offset_seconds": round(self.snap_offset_s, 3)}


def plan_leg(*, init: datetime, elapsed_s: float, volume_time: datetime,
             dt_s: float, min_leg_s: float = MIN_LEG_SECONDS,
             max_leg_s: float = MAX_LEG_SECONDS) -> LegPlan:
    """Advance to this volume's time, or say why that is not a leg.

    The refusals are the honest half: a volume the cycle has already
    passed, one so close to the last that the leg would be a few steps,
    or a gap so long that one leg would swallow it -- that last is a feed
    outage and has to be handled as one, not as a very long cycle.
    """

    raw = (volume_time - init).total_seconds()
    end = snap_to_step(raw, dt_s)
    leg = end - float(elapsed_s)
    if leg <= 0.0:
        raise AutoError(
            f"volume at {iso(volume_time)} is at or before the cycle's "
            f"own clock ({elapsed_s:.0f} s elapsed); nothing to advance")
    if leg < min_leg_s:
        raise AutoError(
            f"volume at {iso(volume_time)} is only {leg:.0f} s past the "
            f"last cycle (floor {min_leg_s:.0f} s); waiting for the "
            "next volume rather than cycling on a sliver")
    if leg > max_leg_s:
        raise AutoError(
            f"volume at {iso(volume_time)} is {leg / 60:.1f} min past "
            f"the last cycle (ceiling {max_leg_s / 60:.0f} min); that "
            "is a feed gap, not a cycle")
    return LegPlan(leg_seconds=leg, end_elapsed_s=end,
                   valid=init + timedelta(seconds=end),
                   snap_offset_s=end - raw)


def usable_volumes(listing: dict) -> list[dict]:
    """Real volumes from a listing, oldest first.

    ``MDM`` keys are metadata companions, not volumes; taking one for a
    volume is a decode failure several minutes later instead of a skip
    here.
    """

    return sorted((v for v in listing.get("volumes", [])
                   if not str(v["filename"]).endswith("MDM")),
                  key=lambda v: v["valid_time"])


def pick_next_volume(volumes: list[dict], *, after: datetime,
                     min_gap_s: float) -> dict | None:
    """The OLDEST volume at least ``min_gap_s`` past ``after``.

    Oldest rather than newest is what makes the daemon catch up on real
    data: after a bootstrap the model's init hour is behind the feed, and
    every volume in between is an observation nobody has used yet.
    Skipping to the newest would throw that spin-up away.
    """

    for volume in volumes:
        stamp = parse_iso(volume["valid_time"])
        if (stamp - after).total_seconds() >= min_gap_s:
            return volume
    return None


def volumes_behind(volumes: list[dict], *, after: datetime) -> int:
    """How many volumes the cycle has still to work through."""

    return sum(1 for v in volumes
               if parse_iso(v["valid_time"]) > after)


def epoch_has_room(*, elapsed_s: float, run_seconds: float,
                   free_forecast_s: float, margin_s: float) -> bool:
    """Can another cycle plus its free forecast fit in this case?

    The prepared case carries boundary data to ``run_seconds`` and not
    one second further, so this is asked BEFORE a cycle starts.
    """

    return (float(elapsed_s) + float(margin_s) + float(free_forecast_s)
            <= float(run_seconds))


def free_forecast_seconds(free_legs: int, free_leg_seconds: float
                          ) -> float:
    return max(0, int(free_legs)) * float(free_leg_seconds)


# ---------------------------------------------------------------------------
# overlap handover: when a background is too old to keep cycling on
# ---------------------------------------------------------------------------
def background_age_seconds(*, init: datetime, now: datetime) -> float:
    """How old the case a lane is cycling on has become.

    Measured from the prepared case's model init -- the hour whose
    background and lateral boundaries the whole epoch is anchored to --
    and NOT from the analysis time, which the cycling keeps current by
    construction and would therefore always report zero age.  The gap
    between those two numbers is the entire problem this exists for.
    """

    return (now - init).total_seconds()


def handover_due(*, init: datetime, now: datetime, max_age_s: float,
                 cooldown_until: datetime | None = None) -> bool:
    """Is it time to prepare a fresher case beside the running one?

    ``max_age_s <= 0`` is off, and off means the daemon behaves exactly
    as it did before this capability existed: one case per epoch until
    its boundary data is spent.
    """

    if float(max_age_s) <= 0.0:
        return False
    if cooldown_until is not None and now < cooldown_until:
        return False
    return background_age_seconds(init=init, now=now) >= float(max_age_s)


def handover_ready(*, spinup_cycles: int, required_cycles: int,
                   spinup_analysis: datetime,
                   primary_analysis: datetime) -> bool:
    """May the gallery switch to the spin-up ensemble yet?

    Two conditions, and the second is the one that is easy to forget.
    The ensemble needs its spin-up cycles -- a covariance built by
    assimilating, not a fresh perturbation.  It ALSO has to have caught
    up to the ensemble it is replacing: a fresher background whose
    analysis is half an hour behind the running one is not an
    improvement, it is a newer model of an older sky, and handing the
    gallery to it would trade the one advantage this nowcast has for the
    appearance of freshness.
    """

    if int(spinup_cycles) < max(1, int(required_cycles)):
        return False
    return spinup_analysis >= primary_analysis


def overlap_summary(*, enabled: bool, max_age_s: float,
                    required_cycles: int, state: str,
                    background_age_s: float | None,
                    spinup: dict | None,
                    cooldown_until: datetime | None,
                    handovers: int) -> dict:
    """What the status file says about the overlap, every publish.

    Published even when the capability is off, because the number it
    turns on for -- how old the running background is -- is worth
    reading whether or not anything is being done about it.
    """

    return {
        "capability": "overlap-handover",
        "stability": "experimental",
        "enabled": bool(enabled),
        "state": state,
        "background_age_seconds": (None if background_age_s is None
                                   else round(background_age_s, 1)),
        "background_max_age_seconds": (float(max_age_s)
                                       if enabled else None),
        "spinup_cycles_required": int(required_cycles),
        "spinup": spinup,
        "cooldown_until": (None if cooldown_until is None
                           else iso(cooldown_until)),
        "handovers": int(handovers),
    }


def handover_record(*, at: datetime, reason: str, retiring: dict,
                    promoted: dict, overlap_started: datetime,
                    primary_cycles_during_overlap: int) -> dict:
    """The durable receipt for one gallery source swap.

    Both case digests, the overlap window, and the cycle counts each
    ensemble contributed across it, so a frame on the gallery can always
    be traced to the case it came from.
    """

    return {
        "schema": HANDOVER_SCHEMA,
        "stability": "experimental",
        "at": iso(at),
        "reason": reason,
        "retired": retiring,
        "promoted": promoted,
        "overlap": {
            "started": iso(overlap_started),
            "seconds": round((at - overlap_started).total_seconds(), 1),
            "primary_cycles": int(primary_cycles_during_overlap),
            "spinup_cycles": int(promoted.get("cycles", 0)),
        },
        "note": ("no ensemble state crossed cases: the promoted "
                 "ensemble was initialised on its own prepared case and "
                 "cycled up from scratch. The retired case, its "
                 "ensemble generations and its figures are left on disk "
                 "exactly as they were."),
    }


def wrfout_time(name: str) -> datetime | None:
    """The valid time a wrfout's own filename declares, or None."""

    marker = "_d01_"
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    try:
        return datetime.strptime(
            tail[:19], "%Y-%m-%d_%H_%M_%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def nearest_grid_wrfout(paths, valid: datetime, *,
                        ceiling_s: float = GRID_OFFSET_CEILING_SECONDS
                        ) -> tuple[Path, float]:
    """The georeference file closest in time to a cycle's valid time.

    Observations are gridded onto a wrfout's georeference and the model
    is analysed against them on that same grid, so the two must be the
    SAME file.  Layer heights drift with the mass field, which is why
    the nearest is taken and the offset receipted rather than assumed to
    be zero -- and why an offset past the ceiling is a refusal.
    """

    best: tuple[Path, float] | None = None
    for path in paths:
        stamp = wrfout_time(Path(path).name)
        if stamp is None:
            continue
        offset = abs((stamp - valid).total_seconds())
        if best is None or offset < best[1]:
            best = (Path(path), offset)
    if best is None:
        raise AutoError(
            "no georeference wrfout to grid observations onto; the "
            "bootstrap's forecast stage produced none")
    if best[1] > ceiling_s:
        raise AutoError(
            f"nearest georeference wrfout {best[0].name} is "
            f"{best[1] / 60:.1f} min from {iso(valid)} (ceiling "
            f"{ceiling_s / 60:.0f} min); this epoch's forecast does not "
            "cover this cycle")
    return best


def assemble_view_report(cycle_reports: list[dict], *,
                         free_legs: int) -> dict:
    """One cycle-report the gallery renderer can read, from many runs.

    Each completed cycle contributes its OBSERVED leg; the newest
    contributes its free legs too.  That is the same rule the composite
    files follow on disk -- cycle k writes leg k (observed) and legs
    k+1.. (free), and cycle k+1 overwrites the first of those with its
    own observed leg -- so the report and the figures cannot disagree
    about which leg is which.

    A superseded free leg is genuinely gone from the view.  It was a
    forecast for a time the model has since seen data for, and showing
    it beside the analysis of that same time as though both were current
    is not something this project does.
    """

    if not cycle_reports:
        raise AutoError("no cycles have completed yet")
    newest = cycle_reports[-1]
    legs: list[dict] = []
    for report in cycle_reports:
        observed = [leg for leg in report["legs"]
                    if leg.get("analysis") is not None]
        if not observed:
            raise AutoError(
                "a cycle run carried no observed leg; the view cannot "
                "be assembled from it")
        legs.append(observed[0])
    free = [leg for leg in newest["legs"]
            if leg.get("analysis") is None][:max(0, int(free_legs))]
    legs.extend(free)
    args = dict(newest["args"])
    args["free_legs"] = len(free)
    return {
        "schema": newest["schema"],
        "stability": "experimental",
        "assembled_by": "tools/da_nowcast_auto.py",
        "assembly_rule": ("one observed leg per completed cycle, in "
                          "cycle order, then the newest cycle's free "
                          "legs; superseded free legs are dropped"),
        "args": args,
        "cycles": len(cycle_reports),
        "legs": legs,
    }


def notice_payload(*, level: str, headline: str, detail: str) -> dict:
    """The banner the gallery carries when the daemon has news."""

    if level not in ("info", "warn"):
        raise AutoError(f"unknown notice level {level!r}")
    return {"schema": NOTICE_SCHEMA, "level": level,
            "headline": headline, "detail": detail,
            "updated": iso(now_utc())}


def volume_is_late(*, last_valid: datetime | None, now: datetime,
                   expected_interval_s: float = (
                       EXPECTED_VOLUME_INTERVAL_SECONDS),
                   late_after_s: float = LATE_AFTER_SECONDS) -> bool:
    """Has the feed been quiet long enough to say so on the figure?"""

    if last_valid is None:
        return False
    return ((now - last_valid).total_seconds()
            > float(expected_interval_s) + float(late_after_s))


def should_render(*, cycle_index: int, caught_up: bool,
                  render_every: int) -> bool:
    """Redraw now, or let the catch-up run on?

    Rendering costs real seconds and a catch-up cycle's gallery is
    superseded within one more cycle.  A current analysis always
    redraws; a catch-up one redraws often enough to show progress.
    """

    if caught_up:
        return True
    if render_every <= 0:
        return False
    return (cycle_index + 1) % render_every == 0


# ---------------------------------------------------------------------------
# the status file
# ---------------------------------------------------------------------------
def status_path(out: Path) -> Path:
    return Path(out) / "auto-status.json"


def stop_path(out: Path) -> Path:
    return Path(out) / "stop-requested"


def write_status(out: Path, payload: dict) -> None:
    """Rewrite the status atomically; it is somebody else's poll target."""

    path = status_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def read_status(out: Path) -> dict:
    path = status_path(out)
    if not path.is_file():
        raise AutoError(f"{path} does not exist; nothing has started "
                        "here")
    return json.loads(path.read_text(encoding="utf-8"))


def verdict_line(state: str, *, cycles: int, site: str,
                 behind: int = 0, notice: dict | None = None) -> str:
    if state == "stopped":
        return (f"{site}: stopped after {cycles} cycle(s); the gallery "
                "holds the last analysis and forecast it produced")
    if state == "failed":
        why = (notice["headline"] if notice
               else "stopped by an error")
        return (f"{site}: {why} after {cycles} cycle(s); the case and "
                "the gallery are intact")
    if state == "catching-up":
        return (f"{site}: catching up on real data, {behind} volume(s) "
                f"still ahead of the analysis ({cycles} done)")
    if notice is not None and notice.get("level") == "warn":
        return f"{site}: {notice['headline']}"
    if state == "waiting":
        return (f"{site}: {cycles} cycle(s) assimilated; waiting for "
                "the next volume")
    return f"{site}: {state}, {cycles} cycle(s) assimilated"


# ---------------------------------------------------------------------------
# the one place that asks what radar data exists
# ---------------------------------------------------------------------------
def list_site_volumes(*, site: str, start: datetime, end: datetime,
                      bucket: str | None) -> list[dict]:
    """Every usable volume for a site in a window, oldest first.

    THE single call site that asks an archive what exists.  A live
    Level-II feed replaces the body of this function and nothing else:
    everything above it consumes ``{"filename", "key", "valid_time"}``
    and never learns where that came from.
    """

    from gpuwm.obs.nexrad import (find_nexrad_bin, nexrad_remedy,
                                  run_list)

    binary = find_nexrad_bin()
    if binary is None:
        raise AutoError(f"no rw_nexrad front door: {nexrad_remedy()}")
    return usable_volumes(run_list(binary, site=site, start=iso(start),
                                   end=iso(end), bucket=bucket))


# ---------------------------------------------------------------------------
# stage commands
# ---------------------------------------------------------------------------
def dealias_argv_tail(args) -> list[str]:
    """The dealias switches this daemon passes on, always spelled out.

    Every front door in this pipeline states the choice rather than
    letting the callee default, for the reason
    :class:`~tools.da_nowcast.DealiasChoice` gives: whatever built the
    assimilated observations must also build the verification
    composites.  ``DealiasChoice.argv_tail`` is empty when dealiasing is
    off, which is what an unflagged daemon does -- the same as the front
    door's ``run`` with no ``--dealias``.
    """

    return DealiasChoice.from_args(args).argv_tail()


def obs_argv_kwargs(*, site: str, valid: datetime, grid_wrfout: Path,
                    out_nc: Path, work_dir: Path, args) -> dict:
    """The obs stage's keyword arguments, as one inspectable mapping.

    The stage builds its argv through this rather than spelling the
    keywords at the call site so a test can bind the SAME mapping the
    daemon passes against ``inspect.signature(obs_cmd)``.  That is not
    ceremony: ``obs_cmd`` grew a required ``dealias`` argument and this
    call site did not, so every cycle of the daemon died with a
    ``TypeError`` on its first observation stage while three tests that
    name ``Daemon`` went on passing -- they read its source as text.
    See tests/test_da_nowcast_daemon_argv.py.
    """

    return {
        # Single-radar, deliberately, and this is the whole reason.
        #
        # The front door (``tools/da_nowcast.py run``) takes --sites and
        # --discover-sites and records the selection in its receipt, so
        # its rolling verifier recovers the same radars from disk.  This
        # daemon has no such seam: it re-execs itself across epochs by
        # REBUILDING its own argv (see the relaunch in main), so a radar
        # flag that did not survive that reconstruction would silently
        # thin a long-running daemon from multi-radar to single-radar at
        # the first epoch roll -- and every gallery after it would say
        # multi-radar because the first one did.
        #
        # A half-done passthrough here is worse than none.  Wiring it
        # properly means the flags, the selection, the bootstrap argv
        # (``bootstrap_cmd``), and the re-exec argv all agreeing, with a
        # test that an epoch roll preserves the radars.  Until that
        # exists this stays explicit: one radar, named, no default.
        "selection": RadarSelection(anchor=site),
        "valid": valid,
        "grid_wrfout": grid_wrfout,
        "out_nc": out_nc,
        "work_dir": work_dir,
        "bucket": args.bucket,
        # The dealias choice, by contrast, IS wired the whole way: the
        # flags, the bootstrap argv and the re-exec argv all agree, and
        # tests/test_da_nowcast_daemon_argv.py holds the epoch roll to it.
        "dealias": DealiasChoice.from_args(args),
    }


def bootstrap_cmd(*, site: str, out: Path, args) -> list[str]:
    """The front door, stopped at its georeference forecast.

    Everything before the observations is the front door's job and stays
    the front door's job: survey, siting, fetch, prepare, forecast.  This
    daemon owns the cycling that comes after, and nothing else.
    """

    argv = [
        _py(), "-m", "tools.da_nowcast", "run",
        "--site", site,
        "--window-end", "latest",
        "--cycles", str(BOOTSTRAP_CYCLES),
        "--cycle-seconds", str(BOOTSTRAP_CYCLE_SECONDS),
        "--free-legs", "0",
        "--run-hours", str(int(args.epoch_hours)),
        "--members", str(int(args.members)),
        "--out", str(out),
        "--source", args.source,
        "--dx-km", f"{args.dx_km:g}",
        "--box-half-km", f"{args.box_half_km:g}",
        "--physics-profile", args.physics_profile,
        "--solve-device", args.solve_device,
        "--stop-after", "forecast",
        "--no-verify",
    ]
    if args.polygon is not None:
        argv.extend(("--domain-polygon", str(args.polygon)))
    if args.vram_gib is not None:
        argv.extend(("--vram-gib", f"{float(args.vram_gib):g}"))
    if args.bucket:
        argv.extend(("--bucket", args.bucket))
    if args.geog_root:
        argv.extend(("--geog-root", str(args.geog_root)))
    if args.bridge:
        argv.extend(("--bridge", str(args.bridge)))
    if args.allow_stale:
        argv.append("--allow-stale")
    # The georeference forecast the front door builds here feeds the
    # same verification composites the cycles are graded against, so
    # its dealias choice is this daemon's choice, spelled out.
    argv.extend(dealias_argv_tail(args))
    return argv


#: Lines a dying interpreter repeats while it tears itself down.  They
#: are not the failure; they are what the failure knocked over on its
#: way out, and there can be hundreds of them AFTER the traceback that
#: says what actually happened.
_TEARDOWN_NOISE = ("Original exception was:", "Error in sys.excepthook:",
                   "Exception ignored in:")


def failure_excerpt(stderr: str, stdout: str, *, limit: int = 14) -> str:
    """The lines that say what went wrong, not the ones that came last.

    A CuPy process that dies mid-analysis floods stderr with excepthook
    cascade for as long as it takes to unwind, so the tail of stderr is
    reliably the LEAST informative part of it.  The first real traceback
    is what a person needs, and it is near the top.
    """

    lines = [line for line in stderr.splitlines()
             if line.strip() and not any(line.startswith(noise)
                                         for noise in _TEARDOWN_NOISE)]
    for index, line in enumerate(lines):
        if line.startswith("Traceback ("):
            return "\n".join(lines[index:index + limit])
    for index, line in enumerate(lines):
        if "Error" in line or "Exception" in line:
            start = max(0, index - 2)
            return "\n".join(lines[start:start + limit])
    if lines:
        return "\n".join(lines[:limit])
    return "\n".join(stdout.splitlines()[-limit:]) or "(no output)"


def run_step(name: str, argv: list[str], *, cwd: Path, log_dir: Path,
             index: int) -> dict:
    """Run one child process, receipt it, and refuse loudly on failure."""

    started = now_utc()
    t0 = time.monotonic()
    log(f"--- {name} ---")
    proc = subprocess.run(argv, cwd=str(cwd), capture_output=True,
                          text=True, errors="replace")
    wall = time.monotonic() - t0
    stderr_lines = proc.stderr.splitlines()
    receipt = {
        "schema": "gpuwm-da.nowcast-stage.v1",
        "stage": name, "argv": argv, "returncode": proc.returncode,
        "wall_seconds": round(wall, 1), "started": iso(started),
        "stdout_tail": proc.stdout.splitlines()[-40:],
        # BOTH ends of stderr: a traceback is at the top and a teardown
        # cascade is at the bottom, and keeping only the bottom is how
        # the real error gets thrown away.
        "stderr_head": stderr_lines[:80],
        "stderr_tail": stderr_lines[-40:],
        "stderr_lines": len(stderr_lines),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{index:05d}-{name}.json").write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    if proc.returncode != 0:
        raise StageFailure(
            f"{name} failed (exit {proc.returncode}):\n"
            + failure_excerpt(proc.stderr, proc.stdout))
    log(f"--- {name} done in {wall:.0f}s ---")
    return receipt


# ---------------------------------------------------------------------------
# the run root must not move underneath a run
# ---------------------------------------------------------------------------
def worktree_fingerprint(root: Path) -> dict:
    """HEAD and dirty-file count of the tree the model runs out of.

    Not a purity claim -- a fingerprint.  A commit landing in the tree a
    forecast is executing from is the failure this exists to catch, and
    catching it needs only 'did this change'.
    """

    def git(*argv: str) -> str:
        try:
            proc = subprocess.run(["git", *argv], cwd=str(root),
                                  capture_output=True, text=True,
                                  errors="replace")
        except OSError:
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"head": head,
            "dirty_paths": len([ln for ln in status.splitlines() if ln]),
            "is_git": bool(head)}


def fingerprint_changed(before: dict, after: dict) -> str | None:
    """What moved, in one line, or None if nothing did."""

    if not before.get("is_git") or not after.get("is_git"):
        return None
    if before["head"] != after["head"]:
        return (f"the run root's HEAD moved from {before['head'][:8]} to "
                f"{after['head'][:8]} while the daemon was running")
    if before["dirty_paths"] != after["dirty_paths"]:
        return ("the run root's uncommitted-file count changed from "
                f"{before['dirty_paths']} to {after['dirty_paths']} "
                "while the daemon was running")
    return None


# ---------------------------------------------------------------------------
# argument surface
# ---------------------------------------------------------------------------
def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``start`` and ``loop``: one definition, one help."""

    parser.add_argument("--site", required=True, type=validate_site,
                        help="four-letter radar station id (argument, "
                             "never a default)")
    parser.add_argument("--out", type=Path, required=True,
                        help="daemon root: epochs, ensemble, gallery "
                             "and the status file live here")
    # Default and justification live in tools.da_nowcast.DEFAULT_MEMBERS
    # (measured 2026-08-05 on a 32 GB and a 16 GB card).  Repeated as a
    # literal here only because the daemon must parse without importing
    # the front door.
    parser.add_argument("--members", type=int, default=10,
                        help="ensemble size (default 10). The "
                             "trajectory advance is exactly linear in "
                             "it -- 3.06 s per member-leg at N=10, 20 "
                             "and 36 -- but the LETKF solve is not, so "
                             "total wall clock grows faster than N. "
                             "See tools.da_nowcast.DEFAULT_MEMBERS for "
                             "the skill measurements behind 10")
    parser.add_argument("--free-legs", type=int, default=6,
                        help="free-forecast legs refreshed once the "
                             "analysis is current (default 6)")
    parser.add_argument("--free-leg-seconds", type=float, default=900.0,
                        help="length of one free-forecast leg "
                             "(default 900)")
    parser.add_argument("--source", default="hrrr",
                        choices=("hrrr", "gfs"),
                        help="background source for every epoch's "
                             "prepared case. Default hrrr, permanently "
                             "(Drew ruling, 2026-08-06); gfs is "
                             "retained for archival reproduction only. "
                             "The roster is gpuwm.da.background's "
                             "registry -- the next source (RRFS) is a "
                             "registry entry, not a new branch")
    parser.add_argument("--dx-km", type=float, default=3.0)
    parser.add_argument("--box-half-km", type=float, default=198.0)
    parser.add_argument("--polygon", type=Path, default=None,
                        help="a GeoJSON polygon to use as the domain "
                             "instead of siting one on the echo -- the "
                             "box a caller drew")
    parser.add_argument("--physics-profile",
                        default=NOWCAST_DEFAULT_PHYSICS_PROFILE,
                        help="shipped physics profile for every stage "
                             f"(default {NOWCAST_DEFAULT_PHYSICS_PROFILE})")
    parser.add_argument("--solve-device", default="cuda",
                        choices=("cuda", "host"))
    parser.add_argument("--epoch-hours", type=int, default=EPOCH_HOURS,
                        help="boundary-data hours one prepared case is "
                             f"built with (default {EPOCH_HOURS}); when "
                             "the cycle approaches it the daemon boots "
                             "a new epoch on a newer background")
    parser.add_argument(
        "--background-max-age", type=float, default=0.0,
        metavar="MINUTES",
        help="EXPERIMENTAL overlap handover, default 0 = OFF. Minutes "
             "of age the prepared background may reach before the "
             "daemon prepares a fresher case beside the running one, "
             "spins a second ensemble up on it, and switches the "
             "gallery to it. 0 keeps the old behaviour exactly: one "
             "case per epoch until its boundary data is spent. The "
             "cost while both run is a second cycle per pass, mostly "
             "in time the card would have spent waiting for a volume")
    parser.add_argument(
        "--spinup-cycles", type=int, default=HANDOVER_SPINUP_CYCLES,
        help=("cycles the fresh ensemble must complete, AND be caught "
              "up to the running one, before the gallery switches to "
              f"it (default {HANDOVER_SPINUP_CYCLES}; measured "
              "spin-up on this daemon's two live cases had analyses "
              "competitive by cycle 3-4). Only read when "
              "--background-max-age is set"))
    parser.add_argument(
        "--handover-cooldown-minutes", type=float,
        default=HANDOVER_COOLDOWN_MINUTES,
        help=("after a failed handover attempt, wait this long before "
              f"trying another (default {HANDOVER_COOLDOWN_MINUTES:g}). "
              "The daemon goes on cycling the old ensemble throughout"))
    parser.add_argument("--poll-seconds", type=int,
                        default=POLL_SECONDS,
                        help=f"archive poll interval (default "
                             f"{POLL_SECONDS})")
    parser.add_argument("--render-every", type=int, default=4,
                        help="while catching up, redraw the gallery "
                             "every N cycles (a current analysis always "
                             "redraws; default 4)")
    parser.add_argument("--min-leg-seconds", type=float,
                        default=MIN_LEG_SECONDS)
    parser.add_argument("--max-leg-seconds", type=float,
                        default=MAX_LEG_SECONDS)
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="stop after this many cycles (0 = forever, "
                             "the default; this is a daemon)")
    parser.add_argument("--max-epochs", type=int, default=0,
                        help="stop after this many prepared cases "
                             "(0 = unlimited)")
    parser.add_argument(
        "--max-consecutive-failures", type=int,
        default=MAX_CONSECUTIVE_FAILURES,
        help=("stop after this many cycles fail in a row (default "
              f"{MAX_CONSECUTIVE_FAILURES}). A transient failure is "
              "worth retrying; a persistent one is worth stopping for, "
              "because the analysis clock does not advance through it "
              "and every later volume becomes an unreachable gap"))
    parser.add_argument("--vram-gib", type=float, default=None,
                        help="size every epoch's memory preflight "
                             "against this card, so a run matches the "
                             "verdict its caller was shown")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--geog-root", type=Path, default=None)
    parser.add_argument("--bridge", type=Path, default=None)
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument("--horizontal-loc-m", type=float, default=12000.0)
    parser.add_argument("--vertical-loc-m", type=float, default=3000.0)
    parser.add_argument(
        "--length-scale-km", type=float, default=None,
        help="perturbation length scale in km (default: capped to what "
             "the fitted domain carries; a smaller box cannot represent "
             "a larger scale, and the cap is said out loud)")
    parser.add_argument("--seed", type=int, default=None)
    # Dealiasing, spelled the way the front door spells it -- same flag
    # names, same shipped default -- because this daemon BUILDS the front
    # door's argv and its own obs stage's argv, and two front doors that
    # describe the same solver differently is how a reader ends up
    # believing they ran different ones.
    parser.add_argument("--dealias", action="store_true",
                        help="unfold radial velocity per sweep instead of "
                             "masking every gate that might be folded, "
                             "for the assimilated observations AND the "
                             "verification composites. The choice is "
                             "carried across every epoch roll, so a long "
                             "daemon does not quietly stop dealiasing at "
                             "the first one")
    # The same two options tools.obs_radar_grid_build takes, defined by
    # that tool and added here rather than restated.
    from tools.obs_radar_grid_build import add_dealias_engine_arguments
    add_dealias_engine_arguments(parser)
    parser.add_argument("--run-root", type=Path, default=None,
                        help="the worktree the model runs out of "
                             "(default: this file's repo). Point it at "
                             "a tree nobody commits into: a commit "
                             "landing mid-run is what the runtime "
                             "integrity checking reacts to")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_nowcast_auto",
        description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    add_run_arguments(sub.add_parser(
        "start", help="launch the daemon detached and return at once"))
    add_run_arguments(sub.add_parser(
        "loop", help="BE the daemon in this process (what start spawns; "
                     "run it by hand to watch it work)"))

    status = sub.add_parser("status", help="print the status file")
    status.add_argument("--out", type=Path, required=True)
    status.add_argument("--json", action="store_true")

    stopper = sub.add_parser(
        "stop", help="ask the daemon to stop after its current cycle "
                     "(writes a request file; kills nothing)")
    stopper.add_argument("--out", type=Path, required=True)
    return parser


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_run_root(args) -> Path:
    root = args.run_root
    if root is None:
        env = os.environ.get("GPUWM_DA_RUN_ROOT", "").strip()
        root = Path(env) if env else repo_root()
    root = Path(root).resolve()
    if not (root / "tools" / "da_nowcast_auto.py").is_file():
        raise AutoError(
            f"--run-root {root} is not a gpuwm worktree (no "
            "tools/da_nowcast_auto.py in it)")
    return root


def run_argument_dests() -> list[str]:
    """Every ``dest`` :func:`add_run_arguments` defines.

    Read off the parser rather than listed here, so it cannot fall
    behind the parser it is describing.  This is what
    :func:`loop_argv` is checked for completeness against.
    """

    probe = argparse.ArgumentParser(add_help=False)
    add_run_arguments(probe)
    return [action.dest for action in probe._actions
            if action.dest != "help"]


def loop_argv(args, run_root: Path) -> list[str]:
    """The ``loop`` command line equivalent to this ``start``.

    Spawning a documented mode rather than an internal entry point means
    the status file can name a command a person can run by hand.

    THE TRAP IN THIS FILE.  ``start`` does not become the daemon; it
    REBUILDS its own command line here and spawns that.  A flag added to
    :func:`add_run_arguments` and not added here parses fine, is
    accepted at the command line, is echoed in the start-mode status
    file -- and is silently gone from the process that actually does the
    work.  Nothing crashes.  The daemon simply runs on the default for
    the rest of its life while every receipt says otherwise.

    So this list is not free-form.
    ``tests/test_da_nowcast_auto.py::TestReExecArgvSurvival`` walks
    :func:`run_argument_dests` and fails if any of them cannot be
    recovered by parsing what this function emits, whatever the value.
    Add a flag above, add it here, or the suite says so.
    """

    argv = [_py(), "-m", "tools.da_nowcast_auto", "loop",
            "--site", args.site, "--out", str(Path(args.out).resolve()),
            "--members", str(args.members),
            "--free-legs", str(args.free_legs),
            "--free-leg-seconds", str(args.free_leg_seconds),
            "--dx-km", f"{args.dx_km:g}",
            "--box-half-km", f"{args.box_half_km:g}",
            "--physics-profile", args.physics_profile,
            # Which background every later epoch is prepared against.
            # Without it the re-exec drops back to the parser default, so
            # a daemon started on one background would silently prepare
            # its next epoch on the other and report that one as the
            # source -- the same class of break the dealias tail below
            # was added for.
            "--source", args.source,
            "--solve-device", args.solve_device,
            "--epoch-hours", str(args.epoch_hours),
            "--background-max-age", str(args.background_max_age),
            "--spinup-cycles", str(args.spinup_cycles),
            "--handover-cooldown-minutes",
            str(args.handover_cooldown_minutes),
            "--poll-seconds", str(args.poll_seconds),
            "--render-every", str(args.render_every),
            "--min-leg-seconds", str(args.min_leg_seconds),
            "--max-leg-seconds", str(args.max_leg_seconds),
            "--max-cycles", str(args.max_cycles),
            "--max-epochs", str(args.max_epochs),
            "--max-consecutive-failures",
            str(args.max_consecutive_failures),
            "--horizontal-loc-m", str(args.horizontal_loc_m),
            "--vertical-loc-m", str(args.vertical_loc_m),
            "--run-root", str(run_root)]
    if args.length_scale_km is not None:
        argv.extend(("--length-scale-km", str(args.length_scale_km)))
    if args.polygon is not None:
        argv.extend(("--polygon", str(Path(args.polygon).resolve())))
    if args.bucket:
        argv.extend(("--bucket", args.bucket))
    if args.geog_root:
        argv.extend(("--geog-root", str(args.geog_root)))
    if args.bridge:
        argv.extend(("--bridge", str(args.bridge)))
    if args.allow_stale:
        argv.append("--allow-stale")
    if args.vram_gib is not None:
        argv.extend(("--vram-gib", f"{float(args.vram_gib):g}"))
    if args.seed is not None:
        argv.extend(("--seed", str(args.seed)))
    # An epoch roll rebuilds this command line and execs it.  A dealias
    # choice missing here would unfold velocity until the first roll and
    # then quietly stop, with every gallery after it still saying the run
    # was dealiased because the first one was.
    argv.extend(dealias_argv_tail(args))
    return argv


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "start":
        return start_mode(args)
    if args.mode == "loop":
        return Daemon(args).run()
    if args.mode == "status":
        return status_mode(args)
    return stop_mode(args)


def start_mode(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_root = resolve_run_root(args)
    if stop_path(out).exists():
        # A stop request from a previous daemon must not stop this one on
        # its first tick.  It is moved aside, never removed.
        os.replace(stop_path(out), stop_path(out).with_name(
            f"stop-requested.cleared-{now_utc():%Y%m%d%H%M%S}"))
    argv = loop_argv(args, run_root)
    log_file = out / "auto.log"
    pid = spawn_detached(argv, cwd=run_root, log_path=log_file)
    write_status(out, {
        "schema": SCHEMA, "state": "starting", "site": args.site,
        "members": int(args.members),
        "free_legs": int(args.free_legs),
        "out": str(out), "gallery": str(out / "gallery"),
        "gallery_page": str(out / "gallery" / "index.html"),
        "run_root": str(run_root), "pid": pid, "argv": argv,
        "log": str(log_file), "started": iso(now_utc()),
        "updated": iso(now_utc()), "cycles_completed": 0,
        "epoch": None, "last_cycle": None, "history": [],
        "notice": None, "stop_requested": False,
        "verdict": f"{args.site}: starting",
        "honesty": ("demo-grade nowcast; UNSCORED, outside any "
                    "registered campaign; no skill claim is made or "
                    "implied"),
    })
    print(f"daemon started, pid {pid}")
    print(f"  status : {status_path(out)}")
    print(f"  gallery: {out / 'gallery' / 'index.html'}")
    print(f"  log    : {log_file}")
    print(f"  stop   : python -m tools.da_nowcast_auto stop --out {out}")
    return 0


def status_mode(args) -> int:
    payload = read_status(args.out)
    if args.json:
        print(json.dumps(payload, indent=1))
        return 0
    print(payload["verdict"])
    print(f"  state   : {payload['state']}")
    print(f"  cycles  : {payload['cycles_completed']}")
    print(f"  gallery : {payload['gallery_page']}")
    if payload.get("notice"):
        print(f"  notice  : {payload['notice']['headline']}")
    if payload.get("last_cycle"):
        last = payload["last_cycle"]
        print(f"  last    : valid {last['valid']} in "
              f"{last['wall_seconds']} s")
    return 1 if payload["state"] == "failed" else 0


def stop_mode(args) -> int:
    out = Path(args.out)
    if not out.is_dir():
        raise AutoError(f"{out} is not a daemon directory")
    stop_path(out).write_text(iso(now_utc()), encoding="utf-8")
    print("stop requested; the daemon finishes its current cycle and "
          f"exits. Nothing is killed: {stop_path(out)}")
    return 0


@dataclass(frozen=True)
class _RunLength:
    """The three plan attributes :func:`tools.da_nowcast.cycle_cmd` reads.

    Not a second plan type: it exposes exactly what that builder touches,
    so a daemon whose cadence comes from the radar and the one-shot front
    door go on emitting the same command line from the same code.
    """

    run_seconds: float
    cycle_seconds: float
    free_legs: int


@dataclass
class Lane:
    """One ensemble, on one prepared case, with its own everything.

    A lane is the unit the identity binding already forces this daemon
    into: a prepared case, the ensemble generations written against it,
    the cycles it produced and the view assembled from them.  Nothing in
    a lane can be restored into another one, which is why the overlap
    handover runs two lanes side by side and swaps which of them the
    gallery reads instead of trying to move state between them.

    Before the overlap capability there was exactly one lane and these
    were plain attributes of the daemon.  They are collected here so a
    second one can exist without a second copy of the cycling code.
    """

    number: int
    role: str
    epoch: dict
    root: Path
    started: datetime
    cycle_reports: list = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return self.root / "epochs" / f"epoch{self.number:04d}"

    @property
    def view(self) -> Path:
        return self.dir / "view"

    @property
    def init(self) -> datetime:
        return parse_iso(self.epoch["init"])

    @property
    def analysis(self) -> datetime:
        return self.init + timedelta(
            seconds=float(self.epoch["elapsed_seconds"]))

    @property
    def cycles(self) -> int:
        return int(self.epoch["cycles"])

    def age_seconds(self, now: datetime) -> float:
        return background_age_seconds(init=self.init, now=now)

    def summary(self, now: datetime | None = None) -> dict:
        """What a receipt says about this lane, without the bulk."""

        payload = {
            "epoch": self.number,
            "role": self.role,
            "case_name": self.epoch["case_name"],
            "prepared_content_sha256": self.epoch[
                "prepared_content_sha256"],
            "init": self.epoch["init"],
            "analysis": iso(self.analysis),
            "cycles": self.cycles,
            "elapsed_seconds": float(self.epoch["elapsed_seconds"]),
            "dir": str(self.dir),
            "ensemble_root": str(self.dir / "ensemble"),
            "started": iso(self.started),
        }
        if now is not None:
            payload["background_age_seconds"] = round(
                self.age_seconds(now), 1)
        return payload


# ---------------------------------------------------------------------------
# the daemon
# ---------------------------------------------------------------------------
class Daemon:
    """One site, cycled until asked to stop."""

    def __init__(self, args) -> None:
        self.args = args
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.run_root = resolve_run_root(args)
        self.gallery = self.out / "gallery"
        self.site = args.site
        self.state = "starting"
        self.cycles = 0
        self.behind = 0
        self.epoch_number = -1
        self.notice: dict | None = None
        self.last_cycle: dict | None = None
        self.history: list[dict] = []
        self.failures = 0
        self.started = now_utc()
        self.fingerprint = worktree_fingerprint(self.run_root)
        # -- overlap handover (experimental; off unless asked for) ------
        self.primary: Lane | None = None
        self.spinup: Lane | None = None
        self.retired: list[dict] = []
        self.handovers: list[dict] = []
        self.spinup_cycles_total = 0
        self.overlap_started: datetime | None = None
        self.overlap_primary_cycles = 0
        self.spinup_failures = 0
        self.cooldown_until: datetime | None = None

    # -- what used to be plain attributes, now the primary lane's -------
    # Kept as properties so everything that reads them -- the status
    # file, the renderer's view directory, the epoch-room check -- goes
    # on meaning "the lane the gallery is showing" without knowing that
    # a second one may exist beside it.
    @property
    def epoch(self) -> dict | None:
        return None if self.primary is None else self.primary.epoch

    @property
    def cycle_reports(self) -> list[dict]:
        return [] if self.primary is None else self.primary.cycle_reports

    @property
    def overlap_enabled(self) -> bool:
        return float(getattr(self.args, "background_max_age", 0.0)) > 0.0

    @property
    def background_max_age_s(self) -> float:
        return float(getattr(self.args, "background_max_age", 0.0)) * 60.0

    def overlap_state(self) -> str:
        if not self.overlap_enabled:
            return "off"
        if self.spinup is not None:
            return "spinning-up"
        if (self.cooldown_until is not None
                and now_utc() < self.cooldown_until):
            return "cooldown"
        return "idle"

    # -- status ----------------------------------------------------------
    def publish(self) -> None:
        now = now_utc()
        write_status(self.out, {
            "schema": SCHEMA,
            "state": self.state,
            "site": self.site,
            "members": int(self.args.members),
            "free_legs": int(self.args.free_legs),
            "free_leg_seconds": float(self.args.free_leg_seconds),
            "out": str(self.out),
            "gallery": str(self.gallery),
            "gallery_page": str(self.gallery / "index.html"),
            "run_root": str(self.run_root),
            "run_root_fingerprint": self.fingerprint,
            "pid": os.getpid(),
            "log": str(self.out / "auto.log"),
            "started": iso(self.started),
            "updated": iso(now_utc()),
            "cycles_completed": self.cycles,
            "consecutive_failures": self.failures,
            "volumes_behind": self.behind,
            "epoch": self.epoch,
            "background_age_seconds": (
                None if self.primary is None
                else round(self.primary.age_seconds(now), 1)),
            "overlap": overlap_summary(
                enabled=self.overlap_enabled,
                max_age_s=self.background_max_age_s,
                required_cycles=int(self.args.spinup_cycles),
                state=self.overlap_state(),
                background_age_s=(None if self.primary is None
                                  else self.primary.age_seconds(now)),
                spinup=(None if self.spinup is None
                        else self.spinup.summary(now)),
                cooldown_until=self.cooldown_until,
                handovers=len(self.handovers)),
            "spinup_cycles_total": self.spinup_cycles_total,
            "handovers": self.handovers[-8:],
            "retired_lanes": self.retired[-8:],
            "last_cycle": self.last_cycle,
            "history": self.history[-24:],
            "notice": self.notice,
            "stop_requested": stop_path(self.out).exists(),
            "verdict": verdict_line(self.state, cycles=self.cycles,
                                    site=self.site, behind=self.behind,
                                    notice=self.notice),
            "honesty": ("demo-grade nowcast; UNSCORED, outside any "
                        "registered campaign; no skill claim is made "
                        "or implied"),
        })

    def announce(self, level: str, headline: str, detail: str) -> None:
        """Say it in the status file AND on the gallery page."""

        self.notice = notice_payload(level=level, headline=headline,
                                     detail=detail)
        view = self.view_dir()
        if view is not None:
            view.mkdir(parents=True, exist_ok=True)
            (view / "auto-notice.json").write_text(
                json.dumps(self.notice, indent=1), encoding="utf-8")
        self.publish()
        log(f"{level.upper()}: {headline} -- {detail}")

    def finish(self, state: str, headline: str, detail: str) -> int:
        """Reach a terminal state, and leave the gallery saying so.

        Announcing into the status file is not enough: the gallery is
        the whole user experience, and a page that goes on showing the
        last cycle with no word that the daemon has stopped is a page
        that looks live and is not.  So the notice is written and the
        figures are redrawn once, if there is anything to redraw.
        """

        self.state = state
        self.announce("warn" if state == "failed" else "info",
                      headline, detail)
        if self.cycle_reports:
            try:
                self.render_view()
            except Exception as error:      # a picture, not the point
                log(f"final render failed: {error}")
        self.state = state
        self.publish()
        return 1 if state == "failed" else 0

    # -- layout ----------------------------------------------------------
    def epoch_dir(self) -> Path | None:
        """The PRIMARY lane's directory: what the gallery is drawn from."""

        return None if self.primary is None else self.primary.dir

    def view_dir(self) -> Path | None:
        """The view the GALLERY reads: always the primary lane's.

        A spin-up lane stages its own view every cycle so it is ready to
        be drawn the moment it is promoted, but it is never the page.
        """

        if self.primary is None:
            return None
        return self.primary.view

    # -- epochs ----------------------------------------------------------
    def build_lane(self, *, role: str) -> Lane:
        """Build one prepared case, its georeference forecast, its lane.

        Returns rather than installs.  The caller decides whether the
        lane it gets back becomes the gallery's source now (a bootstrap
        or a fresh epoch) or cycles beside one that already is (an
        overlap spin-up), and a failure in here therefore leaves
        whatever is already running untouched.
        """

        import tomllib

        self.epoch_number += 1
        number = self.epoch_number
        base = self.out / "epochs" / f"epoch{number:04d}"
        boot = base / "bootstrap"
        log(f"epoch {number} ({role}): bootstrapping into {boot}")
        run_step("bootstrap",
                 bootstrap_cmd(site=self.site, out=boot, args=self.args),
                 cwd=self.run_root, log_dir=base / "receipts", index=0)
        bindings = json.loads(
            (boot / "receipts" / "06-bindings.json")
            .read_text(encoding="utf-8"))
        plan = bindings["plan"]
        init = parse_iso(plan["init"])
        authority = Path(bindings["authority_dir"])
        experiment = tomllib.loads(
            (authority / "experiment.toml").read_text(encoding="utf-8"))
        epoch = {
            "number": number,
            "role": role,
            "init": iso(init),
            "run_seconds": float(plan["run_hours"]) * 3600.0,
            "dt_s": float(experiment["domain"][0]["time_step"]),
            # Quoted back to the cycle driver verbatim.  The prepared
            # experiment is authoritative about its own history cadence,
            # and telling it a different number earns a warning on every
            # cycle for no reason.
            "history_interval_s": float(
                experiment["domain"][0]["history_interval_s"]),
            "nx": int(experiment["domain"][0]["nx"]),
            "ny": int(experiment["domain"][0]["ny"]),
            "nz": int(experiment["shared"]["nz"]),
            "case_name": bindings["case_name"],
            # Quoted back to the cycle driver on every cycle: the
            # prepared case IS one source's case, and the bindings
            # receipt is the authority on which.
            "source": bindings["source"],
            "bootstrap": str(boot),
            "authority": str(authority),
            "prepared_root": bindings["prepared_root"],
            "run_dir": bindings["run_dir"],
            "proof_sha256": bindings["proof_sha256"],
            "source_manifest_sha256": bindings[
                "source_manifest_sha256"],
            "prepared_content_sha256": bindings[
                "prepared_content_sha256"],
            "seed": bindings["seed"],
            "elapsed_seconds": 0.0,
            "cycles": 0,
        }
        # What this domain can actually carry.  A caller who drew a
        # smaller box did not ask for different science, so the scale is
        # capped to the box and the cap is announced rather than left to
        # surface as a refusal inside the first perturbation.
        scale, note = resolvable_length_scale_km(
            nx=epoch["nx"], ny=epoch["ny"],
            dx_m=float(experiment["domain"][0]["dx"]),
            dy_m=float(experiment["domain"][0].get(
                "dy", experiment["domain"][0]["dx"])),
            requested=self.args.length_scale_km)
        epoch["length_scale_km"] = scale
        epoch["length_scale_note"] = note
        if note:
            log(f"note: {note}")
        return Lane(number=number, role=role, epoch=epoch, root=self.out,
                    started=now_utc())

    def bootstrap(self) -> None:
        """Build the gallery's lane and make it the one being cycled."""

        self.state = "bootstrapping"
        self.publish()
        first = self.primary is None
        lane = self.build_lane(role="primary")
        self.primary = lane
        started = (f"nowcast started on the {lane.init:%H:%M}Z background"
                   if first else f"new background at {lane.init:%H:%M}Z")
        self.announce(
            "info", started,
            ("the ensemble is initialised fresh on this prepared case: "
             "a generation written against a different case cannot be "
             "restored into it, and pretending otherwise is exactly "
             "what the identity check refuses"))

    def grid_candidates(self, lane: Lane):
        return sorted((Path(lane.epoch["run_dir"]) / "wrfout").glob(
            "wrfout_d01_*"))

    def analysis_time(self) -> datetime:
        return self.primary.analysis

    # -- one cycle -------------------------------------------------------
    def cycle(self, lane: Lane, volume: dict, *,
              caught_up: bool) -> None:
        args = self.args
        primary = lane is self.primary
        init = lane.init
        plan = plan_leg(init=init,
                        elapsed_s=float(lane.epoch["elapsed_seconds"]),
                        volume_time=parse_iso(volume["valid_time"]),
                        dt_s=float(lane.epoch["dt_s"]),
                        min_leg_s=args.min_leg_seconds,
                        max_leg_s=args.max_leg_seconds)
        grid, offset = nearest_grid_wrfout(self.grid_candidates(lane),
                                           plan.valid)
        base = lane.dir
        view = lane.view
        index = lane.cycles
        receipts = base / "receipts"
        # A free forecast off a state the next volume is about to
        # supersede is work nobody will look at.  It is refreshed when
        # the analysis is current, and the gallery says which mode it
        # is in -- not silently skipped.
        #
        # A SPIN-UP lane never refreshes one at all, however current it
        # is: it is not the gallery's source yet, so its free forecast
        # would be drawn by nobody and thrown away at the handover.  It
        # is also the whole reason the overlap is affordable -- the
        # observed leg is the cheap half of a cycle.
        free_legs = int(args.free_legs) if (caught_up and primary) else 0
        if not primary:
            self.state = "spinning-up"
        else:
            self.state = "assimilating" if caught_up else "catching-up"
        self.publish()

        t0 = time.monotonic()
        obs_dir = view / "obs"
        obs_dir.mkdir(parents=True, exist_ok=True)
        out_nc = obs_dir / (f"obs-{self.site.lower()}-"
                            f"{plan.valid:%Y%m%d%H%M}.nc")
        # Single-radar, deliberately, and this is the whole reason.
        #
        # The front door (``tools/da_nowcast.py run``) takes --sites and
        # --discover-sites and records the selection in its receipt, so
        # its rolling verifier recovers the same radars from disk.  This
        # daemon has no such seam: it re-execs itself across epochs by
        # REBUILDING its own argv (see the relaunch in main), so a radar
        # flag that did not survive that reconstruction would silently
        # thin a long-running daemon from multi-radar to single-radar at
        # the first epoch roll -- and every gallery after it would say
        # multi-radar because the first one did.
        #
        # A half-done passthrough here is worse than none.  Wiring it
        # properly means the flags, the selection, the bootstrap argv
        # (``bootstrap_cmd``), and the re-exec argv all agreeing, with a
        # test that an epoch roll preserves the radars.  Until that
        # exists this stays explicit: one radar, named, no default.
        argv = obs_cmd(**obs_argv_kwargs(
            site=self.site, valid=plan.valid, grid_wrfout=grid,
            out_nc=out_nc, work_dir=base / "vols", args=args))
        argv.extend(("--max-offset-seconds",
                     str(OBS_MATCH_CEILING_SECONDS)))
        run_step(f"obs-{index:04d}", argv, cwd=self.run_root,
                 log_dir=receipts, index=index * 10 + 1)

        ensemble_root = base / "ensemble"
        resume = None
        if index > 0:
            found = ens_state.latest_generation(ensemble_root)
            if found is None:
                raise AutoError(
                    f"cycle {index} has no ensemble generation to "
                    "resume; the previous cycle did not finish writing "
                    "one")
            resume = found[0]
        save = ens_state.slot_dir(ensemble_root, index)

        cycle_out = base / "cycles" / f"cycle{index:04d}"
        seed = (args.seed if args.seed is not None
                else int(lane.epoch["seed"]))
        argv = cycle_cmd(
            prepared_root=Path(lane.epoch["prepared_root"]),
            authority_dir=Path(lane.epoch["authority"]),
            profile=args.physics_profile,
            plan=_RunLength(
                run_seconds=lane.epoch["run_seconds"],
                cycle_seconds=lane.epoch["history_interval_s"],
                free_legs=free_legs),
            members=args.members, obs_files=[out_nc],
            grid_wrfouts=[grid], cycle_out=cycle_out,
            proof_sha=lane.epoch["proof_sha256"],
            manifest_sha=lane.epoch["source_manifest_sha256"],
            content_sha=lane.epoch["prepared_content_sha256"],
            seed=seed, solve_device=args.solve_device,
            horizontal_loc_m=args.horizontal_loc_m,
            vertical_loc_m=args.vertical_loc_m,
            length_scale_km=lane.epoch["length_scale_km"],
            # The LANE's epoch, because a handover runs two of them, and
            # the epoch's own source rather than a literal: the bindings
            # receipt is the authority on which source the prepared case
            # is, and a hardcoded one would quote the wrong background
            # back to the cycle driver for every case not prepared from it.
            source=lane.epoch["source"],
            leg_seconds=plan.leg_seconds, free_legs=free_legs,
            free_leg_seconds=args.free_leg_seconds,
            resume_ensemble=resume, save_ensemble=save,
            leg_number_offset=index)
        run_step(f"cycle-{index:04d}", argv, cwd=self.run_root,
                 log_dir=receipts, index=index * 10 + 2)

        report = json.loads((cycle_out / "cycle-report.json")
                            .read_text(encoding="utf-8"))
        lane.cycle_reports.append(report)
        wall = time.monotonic() - t0
        lane.epoch["elapsed_seconds"] = plan.end_elapsed_s
        lane.epoch["cycles"] = index + 1
        solve = next((leg["analysis"]["solve_seconds"]
                      for leg in report["legs"]
                      if leg.get("analysis")), None)
        record = {
            "index": index, "epoch": lane.number,
            # Which ensemble produced this cycle, and which prepared
            # case it was produced against.  During an overlap the
            # history carries both lanes' cycles interleaved, and
            # without these two fields no one could tell them apart.
            "lane": lane.role,
            "case_digest": str(
                lane.epoch["prepared_content_sha256"])[:12],
            "volume": volume["filename"],
            **plan.to_payload(),
            "grid_wrfout": grid.name,
            "grid_offset_seconds": round(offset, 1),
            "obs_file": out_nc.name,
            "free_legs": free_legs,
            "forecast_refreshed": bool(free_legs),
            "solve_seconds": solve,
            "wall_seconds": round(wall, 1),
            "resumed_from": None if resume is None else str(resume),
            "ensemble_generation": str(save),
        }
        self.history.append(record)
        if primary:
            self.cycles += 1
            # ``last_cycle`` is what the status file and the launcher
            # quote as "the analysis".  A spin-up cycle is not that --
            # it belongs to an ensemble nobody is being shown -- so it
            # goes into the history and no further.
            self.last_cycle = record
            if self.spinup is not None:
                self.overlap_primary_cycles += 1
        else:
            self.spinup_cycles_total += 1
        # Staging is EVERY cycle; drawing is not.  A cycle whose
        # composites were never copied is a hole in the view that the
        # next render walks into, so the two are separate decisions.
        # A spin-up lane stages too, so that the frame drawn right
        # after a handover is its whole history and not just its last
        # cycle.
        self.stage_view(lane, cycle_out, free_legs=free_legs)
        if primary and should_render(cycle_index=index,
                                     caught_up=caught_up,
                                     render_every=args.render_every):
            self.render_view()

    def stage_view(self, lane: Lane, cycle_out: Path, *,
                   free_legs: int) -> None:
        """Fold this cycle's composites and legs into the lane's view."""

        composites = lane.view / "cycle" / "composites"
        composites.mkdir(parents=True, exist_ok=True)
        for npz in sorted((cycle_out / "composites").glob("leg*.npz")):
            shutil.copy2(npz, composites / npz.name)
        merged = assemble_view_report(lane.cycle_reports,
                                      free_legs=free_legs)
        (lane.view / "cycle" / "cycle-report.json").write_text(
            json.dumps(merged, indent=1, default=str), encoding="utf-8")

    def render_view(self) -> None:
        """Redraw the gallery in place, from the PRIMARY lane's view.

        A failed render is a failed PICTURE.  The analysis behind it is
        finished, receipted and carried forward, so this says what
        happened and lets the cycling go on rather than throwing away a
        good cycle over a drawing.
        """

        self.state = "rendering"
        self.publish()
        try:
            run_step(f"render-{self.epoch['cycles']:04d}",
                     render_cmd(
                         case_dir=self.view_dir(), gallery=self.gallery,
                         authority_dir=Path(self.epoch["authority"])),
                     cwd=self.run_root,
                     log_dir=self.epoch_dir() / "receipts",
                     index=self.epoch["cycles"] * 10 + 3)
        except AutoError as failure:
            self.announce(
                "warn", "the gallery did not redraw this cycle",
                str(failure).replace("da_nowcast_auto: ", "")
                + " -- the analysis itself completed and was carried "
                  "forward; the figures below are from the last render "
                  "that succeeded")

    # -- overlap handover (EXPERIMENTAL; nothing below runs unless
    #    --background-max-age was given) ---------------------------------
    def handover_receipt_path(self) -> Path:
        return self.out / "overlap-handovers.json"

    def write_handover_receipt(self, record: dict) -> None:
        """Append one handover to the durable record.

        The status file keeps only a tail of these; this file keeps all
        of them, because "which prepared case produced the frame I am
        looking at" has to stay answerable for a daemon that has been up
        for days.
        """

        path = self.handover_receipt_path()
        existing: list[dict] = []
        if path.is_file():
            try:
                existing = json.loads(
                    path.read_text(encoding="utf-8")).get("handovers", [])
            except (OSError, ValueError, AttributeError):
                existing = []
        payload = {"schema": HANDOVER_SCHEMA, "stability": "experimental",
                   "site": self.site, "updated": iso(now_utc()),
                   "handovers": [*existing, record]}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str),
                       encoding="utf-8")
        os.replace(tmp, path)

    def retire_lane(self, lane: Lane, *, reason: str) -> dict:
        """Stop cycling a lane and leave everything it made where it is.

        Retiring is bookkeeping, not cleanup.  The prepared case, every
        ensemble generation written against it, the cycle receipts and
        the figures all stay exactly as they are -- a retired lane is
        the last thing anyone would want deleted if a handover turns out
        to have been a mistake.
        """

        lane.role = "retired"
        generation = None
        try:
            found = ens_state.latest_generation(lane.dir / "ensemble")
            generation = None if found is None else str(found[0])
        except Exception:               # bookkeeping, never a failure
            generation = None
        payload = {
            "schema": RETIRED_SCHEMA,
            "stability": "experimental",
            "retired_at": iso(now_utc()),
            "reason": reason,
            **lane.summary(),
            "last_ensemble_generation": generation,
            "note": ("nothing was deleted. The prepared case, its "
                     "ensemble generations, its cycle receipts and its "
                     "figures are on disk exactly as this lane left "
                     "them; they simply stop being added to."),
        }
        lane.dir.mkdir(parents=True, exist_ok=True)
        (lane.dir / "retired.json").write_text(
            json.dumps(payload, indent=1, default=str), encoding="utf-8")
        self.retired.append(payload)
        log(f"epoch {lane.number} retired: {reason}")
        return payload

    def start_cooldown(self) -> None:
        self.cooldown_until = now_utc() + timedelta(
            minutes=float(self.args.handover_cooldown_minutes))

    def abandon_spinup(self, detail: str, *, stage: str) -> None:
        """THE FAIL-SAFE.  Drop the fresh lane, keep the working one.

        Reached from the three places a handover can come to nothing:
        the fresh case would not prepare, its ensemble would not cycle,
        or what got prepared was not actually any fresher.  None of them
        is allowed to touch the running ensemble, the primary's failure
        budget or the daemon's exit -- a fresher background is never
        worth killing a working nowcast for.  The attempt is retired,
        the reason goes on the gallery, and a cooldown stops the daemon
        spending the primary's idle time on the same dead end every
        poll.
        """

        lane = self.spinup
        self.spinup = None
        self.spinup_failures = 0
        self.overlap_started = None
        self.overlap_primary_cycles = 0
        self.start_cooldown()
        if lane is not None:
            self.retire_lane(lane, reason=stage)
        wait = float(self.args.handover_cooldown_minutes)
        self.announce(
            "warn",
            "stayed on the current background — the fresher one did "
            "not come up",
            f"{stage}: {detail} -- the running ensemble is "
            "untouched and still cycling, the panels below are its own, "
            f"and another attempt is made in about {wait:g} min. "
            "Nothing from the abandoned attempt was deleted.")

    def maybe_start_spinup(self, now: datetime, *,
                           primary_current: bool) -> bool:
        """Prepare a fresher case beside the running one, if it is time."""

        if self.primary is None or self.spinup is not None:
            return False
        if not primary_current:
            # The running ensemble is still working through a backlog.
            # Its background being old is real, but putting a second
            # ensemble on the card before the first one has caught up
            # makes BOTH of them late, and late is the thing this
            # daemon has to avoid above all.
            return False
        if not handover_due(init=self.primary.init, now=now,
                            max_age_s=self.background_max_age_s,
                            cooldown_until=self.cooldown_until):
            return False
        if (self.args.max_epochs
                and self.epoch_number + 1 >= self.args.max_epochs):
            return False
        age_min = self.primary.age_seconds(now) / 60.0
        self.announce(
            "info",
            f"preparing a fresher background — this one is "
            f"{age_min:.0f} min old",
            "the running ensemble goes on cycling throughout and the "
            "panels below stay its own. A second ensemble is being "
            "started on the newest available cycle and spun up on the "
            "same volumes; the gallery only switches once that one has "
            f"{int(self.args.spinup_cycles)} cycles behind it and has "
            "caught up to this one.")
        try:
            lane = self.build_lane(role="spinup")
        except (AutoError, Exception) as error:
            self.abandon_spinup(
                str(error).replace("da_nowcast_auto: ", ""),
                stage="preparing the fresher case failed")
            return True
        self.spinup = lane
        if lane.init <= self.primary.init:
            # THE CHURN GUARD.  A bootstrap's init is the most recent
            # whole hour at or before the newest volume, so a threshold
            # shorter than the interval between available cycles gets
            # handed back the SAME background it is trying to replace --
            # and, with nothing to stop it, would hand over to it, find
            # the new lane just as old, and prepare another, forever.
            # There is nothing fresher; say so and wait out the
            # cooldown rather than burn the card discovering it again.
            self.abandon_spinup(
                f"the newest case the archive can build still "
                f"initialises at {lane.init:%H:%M}Z, which is the "
                "background the running ensemble is already on",
                stage="there was nothing fresher to hand over to")
            return True
        self.overlap_started = now_utc()
        self.overlap_primary_cycles = 0
        self.spinup_failures = 0
        self.announce(
            "info",
            f"spinning up a {lane.init:%H:%M}Z background beside the "
            f"{self.primary.init:%H:%M}Z one",
            "no state crosses between them: the new ensemble is "
            "initialised on its own prepared case and assimilates the "
            "same volumes from scratch. The panels below are still the "
            "running ensemble's until it has caught up and the handover "
            "is announced.")
        return True

    def advance_spinup(self, volumes: list[dict]) -> bool:
        """One spin-up cycle, on the same feed the primary is reading."""

        lane = self.spinup
        if lane is None:
            return False
        args = self.args
        volume = pick_next_volume(
            volumes, after=lane.analysis,
            min_gap_s=(args.min_leg_seconds + float(lane.epoch["dt_s"])))
        if volume is None:
            return False
        caught_up = (parse_iso(volume["valid_time"])
                     >= parse_iso(volumes[-1]["valid_time"]))
        try:
            self.cycle(lane, volume, caught_up=caught_up)
        except StageFailure as failure:
            self.spinup_failures += 1
            detail = str(failure).replace("da_nowcast_auto: ", "")
            if self.spinup_failures >= args.max_consecutive_failures:
                self.abandon_spinup(
                    detail,
                    stage="spinning the fresh ensemble up failed")
            else:
                log(f"spin-up cycle failed ({self.spinup_failures} of "
                    f"{args.max_consecutive_failures}): {detail}")
            return True
        except AutoError as refusal:
            # A statement about the data, exactly as on the primary
            # lane, and no reason to throw a spin-up away.
            log("spin-up cycle skipped: "
                + str(refusal).replace("da_nowcast_auto: ", ""))
            return False
        except Exception as error:      # never the primary's problem
            self.abandon_spinup(
                f"{type(error).__name__}: {error}",
                stage="spinning the fresh ensemble up failed")
            return True
        self.spinup_failures = 0
        return True

    def handover(self, *, reason: str) -> None:
        """Switch the gallery and the free forecast to the fresh lane."""

        now = now_utc()
        old, new = self.primary, self.spinup
        record = handover_record(
            at=now, reason=reason, retiring=old.summary(now),
            promoted=new.summary(now),
            overlap_started=self.overlap_started or new.started,
            primary_cycles_during_overlap=self.overlap_primary_cycles)
        new.role = "primary"
        self.primary = new
        self.spinup = None
        self.overlap_started = None
        self.overlap_primary_cycles = 0
        self.spinup_failures = 0
        self.handovers.append(record)
        self.write_handover_receipt(record)
        self.retire_lane(
            old, reason=f"handed over to epoch {new.number}: {reason}")
        # Said AFTER the swap, so the notice is written into the view
        # the very next render draws -- the frame that first shows the
        # new case is the frame that says it is a new case.
        self.announce(
            "info",
            f"switched to a {new.init:%H:%M}Z background "
            f"(was {old.init:%H:%M}Z)",
            f"the panels below are now a different prepared case, "
            f"assimilated from scratch over {new.cycles} cycle(s) "
            "beside the one they replace; no ensemble state crossed "
            "between them. The retired case and its ensemble are kept "
            "on disk. The free forecast returns with this ensemble's "
            "next cycle.")
        try:
            self.render_view()
        except Exception as error:      # a picture, not the point
            log(f"post-handover render failed: {error}")

    def maybe_handover(self) -> bool:
        if self.spinup is None or self.primary is None:
            return False
        if not handover_ready(
                spinup_cycles=self.spinup.cycles,
                required_cycles=int(self.args.spinup_cycles),
                spinup_analysis=self.spinup.analysis,
                primary_analysis=self.primary.analysis):
            return False
        self.handover(reason="the running background reached "
                             f"--background-max-age "
                             f"({self.args.background_max_age:g} min)")
        return True

    def overlap_step(self, volumes: list[dict], now: datetime, *,
                     primary_current: bool = True) -> bool:
        """The whole capability, in the order it has to happen.

        Returns whether the card did any work here, so the loop knows
        not to sleep through a spin-up's backlog.
        """

        if not self.overlap_enabled or self.primary is None:
            return False
        if self.failures:
            # The primary's own machinery is not working.  Putting a
            # second ensemble on the card in the middle of that makes
            # the failure harder to read and the recovery slower.
            return False
        if self.spinup is None:
            return self.maybe_start_spinup(
                now, primary_current=primary_current)
        worked = self.advance_spinup(volumes)
        self.maybe_handover()
        return worked

    def roll_primary(self) -> None:
        """The primary's case is spent: promote or bootstrap.

        A spin-up already exists only when the overlap capability is on
        and has not yet reached its handover bar.  Promoting it anyway
        beats a cold bootstrap -- it has a prepared case AND however
        many assimilation cycles it managed -- and the receipt says
        plainly that this handover was forced by boundary exhaustion
        rather than earned at the spin-up bar.
        """

        if self.spinup is not None and self.primary is not None:
            self.handover(
                reason=("the running case's boundary data is spent; the "
                        f"spin-up had {self.spinup.cycles} of "
                        f"{int(self.args.spinup_cycles)} cycles"))
            return
        self.bootstrap()

    # -- the loop --------------------------------------------------------
    def stop_requested(self) -> bool:
        return stop_path(self.out).exists()

    def integrity_ok(self) -> bool:
        moved = fingerprint_changed(self.fingerprint,
                                    worktree_fingerprint(self.run_root))
        if moved is None:
            return True
        self.finish(
            "failed", "stopped: the run root moved underneath the run",
            moved + ". Runs and commits belong in different worktrees. "
            "The case and the gallery are intact; start a new daemon "
            "against a tree nobody commits into.")
        return False

    def needs_new_epoch(self) -> bool:
        if self.epoch is None:
            return True
        return not epoch_has_room(
            elapsed_s=self.epoch["elapsed_seconds"],
            run_seconds=self.epoch["run_seconds"],
            free_forecast_s=free_forecast_seconds(
                self.args.free_legs, self.args.free_leg_seconds),
            margin_s=self.args.max_leg_seconds)

    def run(self) -> int:
        args = self.args
        log(f"daemon up: site {self.site}, N={args.members}, run root "
            f"{self.run_root}")
        self.publish()
        announced_late = False
        try:
            while True:
                if self.stop_requested():
                    return self.finish(
                        "stopped", "stopped on request",
                        "the gallery holds the last analysis and free "
                        "forecast the daemon produced; nothing after "
                        "this is newer than what is shown")
                if not self.integrity_ok():
                    return 1
                if self.needs_new_epoch():
                    # The ceiling counts PREPARED CASES, so it only
                    # bites when a new one would be prepared.  Promoting
                    # a spin-up that already exists prepares nothing.
                    if (self.spinup is None and args.max_epochs
                            and self.epoch_number + 1 >= args.max_epochs):
                        return self.finish(
                            "stopped", "stopped at the epoch ceiling",
                            f"--max-epochs {args.max_epochs} reached; "
                            "this case's boundary data is spent and "
                            "nothing below will get any newer")
                    self.roll_primary()
                    announced_late = False

                analysis_at = self.analysis_time()
                now = now_utc()
                # ONE listing per pass, wide enough for both lanes: a
                # spin-up starts behind the primary and would otherwise
                # need its own call to see the volumes it has to work
                # through.
                listing_from = analysis_at
                if self.spinup is not None:
                    listing_from = min(listing_from, self.spinup.analysis)
                volumes = list_site_volumes(
                    site=self.site,
                    start=listing_from - timedelta(seconds=600),
                    end=now, bucket=args.bucket)
                self.behind = volumes_behind(volumes, after=analysis_at)
                # One step of slack over the leg floor.  Snapping a
                # volume time back to the step lattice can shorten the
                # leg by up to half a step, and a volume picked at
                # exactly the floor would then be refused BY the floor
                # -- and picked again on the next poll, forever.  The
                # margin makes that impossible rather than unlikely.
                volume = pick_next_volume(
                    volumes, after=analysis_at,
                    min_gap_s=(args.min_leg_seconds
                               + float(self.epoch["dt_s"])))

                worked = False
                primary_current = True
                if volume is None:
                    self.state = "waiting"
                    newest = (parse_iso(volumes[-1]["valid_time"])
                              if volumes else None)
                    if volume_is_late(last_valid=newest or analysis_at,
                                      now=now):
                        if not announced_late:
                            stamp = iso(newest) if newest else "(none)"
                            self.announce(
                                "warn",
                                "next volume is late — the panels below "
                                "are running on older data",
                                f"nothing newer than {stamp} has "
                                "appeared for this site. Nothing is "
                                "padded or repeated: the figures are "
                                "exactly what the last real volume "
                                "produced, and the daemon keeps asking.")
                            announced_late = True
                    else:
                        self.publish()
                else:
                    caught_up = (parse_iso(volume["valid_time"])
                                 >= parse_iso(volumes[-1]["valid_time"]))
                    primary_current = caught_up
                    try:
                        self.cycle(self.primary, volume,
                                   caught_up=caught_up)
                    except StageFailure as failure:
                        # Machinery, not data.  A failed cycle leaves
                        # the analysis clock where it was, so retrying
                        # is right for a transient fault and wrong
                        # forever.
                        self.failures += 1
                        detail = str(failure).replace(
                            "da_nowcast_auto: ", "")
                        if self.failures >= args.max_consecutive_failures:
                            return self.finish(
                                "failed",
                                f"stopped after {self.failures} cycles "
                                "failed in a row", detail
                                + " -- the case and the gallery are "
                                  "intact; the analysis clock has not "
                                  "moved since the last cycle that "
                                  "worked")
                        self.state = "waiting"
                        self.announce(
                            "warn",
                            f"cycle failed ({self.failures} of "
                            f"{args.max_consecutive_failures} in a row)",
                            detail + " -- retrying on the next poll; the "
                            "figures below are from the last cycle that "
                            "worked")
                        time.sleep(args.poll_seconds)
                        continue
                    except AutoError as refusal:
                        # A refusal here is a statement about the data,
                        # not a crash: say it and keep waiting.  It does
                        # not count against the failure budget.
                        self.state = "waiting"
                        self.announce("warn", "cycle skipped",
                                      str(refusal).replace(
                                          "da_nowcast_auto: ", ""))
                        time.sleep(args.poll_seconds)
                        continue

                    worked = True
                    self.failures = 0
                    announced_late = False
                    if self.last_cycle["forecast_refreshed"]:
                        self.state = "waiting"
                        self.announce(
                            "info",
                            "analysis current to "
                            f"{self.last_cycle['valid'][11:16]}Z",
                            f"cycle {self.cycles} took "
                            f"{self.last_cycle['wall_seconds']} s; the "
                            "free forecast below was refreshed from it "
                            "and runs past every observation the model "
                            "has seen")
                    else:
                        self.state = "catching-up"
                        self.announce(
                            "info",
                            f"catching up — {self.behind - 1} volume(s) "
                            "still ahead of the analysis",
                            "observations only while the analysis is "
                            "behind the feed; the free forecast is "
                            "refreshed once it reaches the newest "
                            "volume")

                # The overlap runs AFTER the primary, every pass, one
                # child process at a time out of this one loop -- the
                # card never sees two of this daemon's runs at once.
                # Mostly it fills the wait between volumes, which is
                # what makes the double compute affordable.
                worked = self.overlap_step(
                    volumes, now,
                    primary_current=primary_current) or worked

                if args.max_cycles and self.cycles >= args.max_cycles:
                    return self.finish(
                        "stopped", "stopped at the cycle ceiling",
                        f"--max-cycles {args.max_cycles} reached; "
                        "nothing below will get any newer")

                if not worked:
                    time.sleep(args.poll_seconds)
        except AutoError as error:
            return self.finish("failed", "stopped by an error",
                               str(error).replace(
                                   "da_nowcast_auto: ", ""))
        except KeyboardInterrupt:
            return self.finish("stopped", "interrupted",
                               "stopped by a signal; the figures below "
                               "are the last ones it produced")


if __name__ == "__main__":
    raise SystemExit(main())
