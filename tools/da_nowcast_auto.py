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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:                                    # python -m tools.da_nowcast_auto
    from tools import da_ensemble_state as ens_state
    from tools.da_nowcast import (RadarSelection, cycle_cmd, iso, obs_cmd,
                                  parse_iso, render_cmd,
                                  resolvable_length_scale_km,
                                  spawn_detached, validate_site)
except ImportError:                     # python tools/da_nowcast_auto.py
    import da_ensemble_state as ens_state
    from da_nowcast import (RadarSelection, cycle_cmd, iso, obs_cmd,
                            parse_iso, render_cmd,
                            resolvable_length_scale_km,
                            spawn_detached, validate_site)

#: The status file a page or a person polls.
SCHEMA = "gpuwm-da.nowcast-auto.v1"

#: The banner the renderer puts on the gallery when the daemon has news.
NOTICE_SCHEMA = "gpuwm-da.nowcast-auto-notice.v1"

#: Where the daemon can be.
STATES = ("starting", "bootstrapping", "catching-up", "assimilating",
          "rendering", "waiting", "stopped", "failed")

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
                        default="wsm6-ysu-mm5-noah-no-radiation-v1")
    parser.add_argument("--solve-device", default="cuda",
                        choices=("cuda", "host"))
    parser.add_argument("--epoch-hours", type=int, default=EPOCH_HOURS,
                        help="boundary-data hours one prepared case is "
                             f"built with (default {EPOCH_HOURS}); when "
                             "the cycle approaches it the daemon boots "
                             "a new epoch on a newer background")
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


def loop_argv(args, run_root: Path) -> list[str]:
    """The ``loop`` command line equivalent to this ``start``.

    Spawning a documented mode rather than an internal entry point means
    the status file can name a command a person can run by hand.
    """

    argv = [_py(), "-m", "tools.da_nowcast_auto", "loop",
            "--site", args.site, "--out", str(Path(args.out).resolve()),
            "--members", str(args.members),
            "--free-legs", str(args.free_legs),
            "--free-leg-seconds", str(args.free_leg_seconds),
            "--dx-km", f"{args.dx_km:g}",
            "--box-half-km", f"{args.box_half_km:g}",
            "--physics-profile", args.physics_profile,
            "--solve-device", args.solve_device,
            "--epoch-hours", str(args.epoch_hours),
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
        self.epoch: dict | None = None
        self.notice: dict | None = None
        self.last_cycle: dict | None = None
        self.history: list[dict] = []
        self.failures = 0
        self.cycle_reports: list[dict] = []
        self.started = now_utc()
        self.fingerprint = worktree_fingerprint(self.run_root)

    # -- status ----------------------------------------------------------
    def publish(self) -> None:
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
    def epoch_dir(self) -> Path:
        return self.out / "epochs" / f"epoch{self.epoch_number:04d}"

    def view_dir(self) -> Path | None:
        if self.epoch_number < 0:
            return None
        return self.epoch_dir() / "view"

    # -- epochs ----------------------------------------------------------
    def bootstrap(self) -> None:
        """Build one prepared case and the georeference forecast on it."""

        import tomllib

        self.epoch_number += 1
        self.cycle_reports = []
        self.state = "bootstrapping"
        self.publish()
        base = self.epoch_dir()
        boot = base / "bootstrap"
        log(f"epoch {self.epoch_number}: bootstrapping into {boot}")
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
        self.epoch = {
            "number": self.epoch_number,
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
            nx=self.epoch["nx"], ny=self.epoch["ny"],
            dx_m=float(experiment["domain"][0]["dx"]),
            dy_m=float(experiment["domain"][0].get(
                "dy", experiment["domain"][0]["dx"])),
            requested=self.args.length_scale_km)
        self.epoch["length_scale_km"] = scale
        self.epoch["length_scale_note"] = note
        if note:
            log(f"note: {note}")
        started = (f"nowcast started on the {init:%H:%M}Z background"
                   if self.epoch_number == 0
                   else f"new background at {init:%H:%M}Z")
        self.announce(
            "info", started,
            ("the ensemble is initialised fresh on this prepared case: "
             "a generation written against a different case cannot be "
             "restored into it, and pretending otherwise is exactly "
             "what the identity check refuses"))

    def grid_candidates(self):
        return sorted((Path(self.epoch["run_dir"]) / "wrfout").glob(
            "wrfout_d01_*"))

    def analysis_time(self) -> datetime:
        return (parse_iso(self.epoch["init"])
                + timedelta(seconds=self.epoch["elapsed_seconds"]))

    # -- one cycle -------------------------------------------------------
    def cycle(self, volume: dict, *, caught_up: bool) -> None:
        args = self.args
        init = parse_iso(self.epoch["init"])
        plan = plan_leg(init=init,
                        elapsed_s=float(self.epoch["elapsed_seconds"]),
                        volume_time=parse_iso(volume["valid_time"]),
                        dt_s=float(self.epoch["dt_s"]),
                        min_leg_s=args.min_leg_seconds,
                        max_leg_s=args.max_leg_seconds)
        grid, offset = nearest_grid_wrfout(self.grid_candidates(),
                                           plan.valid)
        base = self.epoch_dir()
        view = self.view_dir()
        index = int(self.epoch["cycles"])
        receipts = base / "receipts"
        # A free forecast off a state the next volume is about to
        # supersede is work nobody will look at.  It is refreshed when
        # the analysis is current, and the gallery says which mode it
        # is in -- not silently skipped.
        free_legs = int(args.free_legs) if caught_up else 0
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
        argv = obs_cmd(selection=RadarSelection(anchor=self.site),
                       valid=plan.valid,
                       grid_wrfout=grid, out_nc=out_nc,
                       work_dir=base / "vols", bucket=args.bucket)
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
                else int(self.epoch["seed"]))
        argv = cycle_cmd(
            prepared_root=Path(self.epoch["prepared_root"]),
            authority_dir=Path(self.epoch["authority"]),
            profile=args.physics_profile,
            plan=_RunLength(
                run_seconds=self.epoch["run_seconds"],
                cycle_seconds=self.epoch["history_interval_s"],
                free_legs=free_legs),
            members=args.members, obs_files=[out_nc],
            grid_wrfouts=[grid], cycle_out=cycle_out,
            proof_sha=self.epoch["proof_sha256"],
            manifest_sha=self.epoch["source_manifest_sha256"],
            content_sha=self.epoch["prepared_content_sha256"],
            seed=seed, solve_device=args.solve_device,
            horizontal_loc_m=args.horizontal_loc_m,
            vertical_loc_m=args.vertical_loc_m,
            length_scale_km=self.epoch["length_scale_km"],
            source=self.epoch["source"],
            leg_seconds=plan.leg_seconds, free_legs=free_legs,
            free_leg_seconds=args.free_leg_seconds,
            resume_ensemble=resume, save_ensemble=save,
            leg_number_offset=index)
        run_step(f"cycle-{index:04d}", argv, cwd=self.run_root,
                 log_dir=receipts, index=index * 10 + 2)

        report = json.loads((cycle_out / "cycle-report.json")
                            .read_text(encoding="utf-8"))
        self.cycle_reports.append(report)
        wall = time.monotonic() - t0
        self.epoch["elapsed_seconds"] = plan.end_elapsed_s
        self.epoch["cycles"] = index + 1
        self.cycles += 1
        solve = next((leg["analysis"]["solve_seconds"]
                      for leg in report["legs"]
                      if leg.get("analysis")), None)
        self.last_cycle = {
            "index": index, "epoch": self.epoch_number,
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
        self.history.append(self.last_cycle)
        # Staging is EVERY cycle; drawing is not.  A cycle whose
        # composites were never copied is a hole in the view that the
        # next render walks into, so the two are separate decisions.
        self.stage_view(cycle_out, free_legs=free_legs)
        if should_render(cycle_index=index, caught_up=caught_up,
                         render_every=args.render_every):
            self.render_view()

    def stage_view(self, cycle_out: Path, *, free_legs: int) -> None:
        """Fold this cycle's composites and legs into the view."""

        view = self.view_dir()
        composites = view / "cycle" / "composites"
        composites.mkdir(parents=True, exist_ok=True)
        for npz in sorted((cycle_out / "composites").glob("leg*.npz")):
            shutil.copy2(npz, composites / npz.name)
        merged = assemble_view_report(self.cycle_reports,
                                      free_legs=free_legs)
        (view / "cycle" / "cycle-report.json").write_text(
            json.dumps(merged, indent=1, default=str), encoding="utf-8")

    def render_view(self) -> None:
        """Redraw the gallery in place.

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
                    if (args.max_epochs
                            and self.epoch_number + 1 >= args.max_epochs):
                        return self.finish(
                            "stopped", "stopped at the epoch ceiling",
                            f"--max-epochs {args.max_epochs} reached; "
                            "this case's boundary data is spent and "
                            "nothing below will get any newer")
                    self.bootstrap()
                    announced_late = False

                analysis_at = self.analysis_time()
                now = now_utc()
                volumes = list_site_volumes(
                    site=self.site,
                    start=analysis_at - timedelta(seconds=600),
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
                    time.sleep(args.poll_seconds)
                    continue

                caught_up = (parse_iso(volume["valid_time"])
                             >= parse_iso(volumes[-1]["valid_time"]))
                try:
                    self.cycle(volume, caught_up=caught_up)
                except StageFailure as failure:
                    # Machinery, not data.  A failed cycle leaves the
                    # analysis clock where it was, so retrying is right
                    # for a transient fault and wrong forever.
                    self.failures += 1
                    detail = str(failure).replace(
                        "da_nowcast_auto: ", "")
                    if self.failures >= args.max_consecutive_failures:
                        return self.finish(
                            "failed",
                            f"stopped after {self.failures} cycles "
                            "failed in a row", detail
                            + " -- the case and the gallery are intact; "
                              "the analysis clock has not moved since "
                              "the last cycle that worked")
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
                    # A refusal here is a statement about the data, not
                    # a crash: say it and keep waiting.  It does not
                    # count against the failure budget.
                    self.state = "waiting"
                    self.announce("warn", "cycle skipped",
                                  str(refusal).replace(
                                      "da_nowcast_auto: ", ""))
                    time.sleep(args.poll_seconds)
                    continue

                self.failures = 0
                announced_late = False
                if self.last_cycle["forecast_refreshed"]:
                    self.state = "waiting"
                    self.announce(
                        "info",
                        "analysis current to "
                        f"{self.last_cycle['valid'][11:16]}Z",
                        f"cycle {self.cycles} took "
                        f"{self.last_cycle['wall_seconds']} s; the free "
                        "forecast below was refreshed from it and runs "
                        "past every observation the model has seen")
                else:
                    self.state = "catching-up"
                    self.announce(
                        "info",
                        f"catching up — {self.behind - 1} volume(s) "
                        "still ahead of the analysis",
                        "observations only while the analysis is behind "
                        "the feed; the free forecast is refreshed once "
                        "it reaches the newest volume")

                if args.max_cycles and self.cycles >= args.max_cycles:
                    return self.finish(
                        "stopped", "stopped at the cycle ceiling",
                        f"--max-cycles {args.max_cycles} reached; "
                        "nothing below will get any newer")
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
