"""Downscale a finished forecast from its page: a finer forecast inside it, driven by its saved output.

The page's Downscale button runs the command the controller builds for the same answers (``tools/arwen-tui``:
``downscale_guide`` then ``Guide::request``), token for token and in that order::

    gpuwm downscale HISTORY --point=LAT,LON --parent-restart=latest --parent-domain=N --ratio=R
        [--child-size=NX,NY] [--max-boundary-interval-seconds=B] [--hours=H] [--output-interval-seconds=S]
        [--card=TIER] --tiles=auto --out DIR [--dry-run] [--auto-vram] --render-products=SPEC

with these defaults: the parent's newest complete checkpoint set, the finest grid the run saved, a refinement of 3,
the parent's output interval, streaming when it does not fit, the card measured (``--auto-vram``), and every product
drawn (``all``, the engine's own default).

The boundary ceiling ``B`` is the cadence the chosen grid's frames actually have, read from their names and shown on
the page, so the engine refuses frames that are not at that cadence and its plan and report name the ceiling the
page asked for.  ``--accept-parent-cadence`` records a person's acceptance of the archive's cadence; the page asks
no one for that, so it never sends it.  Over frames coarser than the engine's downscaling guidance the engine still
says so, and that is true: where a finer forecast's storms form depends on how often its edges are driven.

A drawn box is sized in child cells exactly as the desktop panel sizes it (:func:`box_cells`), so the same box
asks the engine for the same grid from either door.  A card size named on the page is the engine's own tier
(``--card``) in place of measuring this computer's card.

Everything here reads file names and the run's own event log.  The GUI server never opens a NetCDF file and never
imports the engine's modules: the engine reads the frames itself when the command runs.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import math
from pathlib import Path
import re
from typing import Any

from . import runs
from .files import read_json, scrub, within

#: The panel's refinement when none is chosen (``DownscalePanel::default``: ``ratio: 3``).
DEFAULT_RATIO = 3
RATIO_RANGE = (2, 99)
#: What the finished child is drawn as when the page names nothing: the engine's own default
#: (``gpuwm.first_products.DEFAULT_RENDER_PRODUCTS``), the one the terminal's plot selection starts from.
DEFAULT_PRODUCTS = "all"
#: The panel's streaming choice: stream in tiles when the child does not fit the card resident.
TILES = "auto"
#: Every key the page may send; any other is refused rather than dropped, as the controller refuses one, so a
#: caller never believes it set something nothing read.
KEYS = frozenset({"mode", "lat", "lon", "box", "domain", "ratio", "dx_km", "hours", "output_minutes", "card",
                  "products", "name"})
MODES = ("plan", "run")

_FRAME = re.compile(r"^wrfout_d(\d{2})_(\d{4}-\d{2}-\d{2})[_:](\d{2})[_:](\d{2})[_:](\d{2})$")
_CHECKPOINT = re.compile(r"^gpuwmrst_d(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})((?:__.*)?)\.npz$")
PRODUCTS_RE = re.compile(r"^[a-z0-9_]+(,[a-z0-9_]+)*$")
#: How deep under a run folder its frames are looked for when its event log does not name them
#: (``chain/<run>/run/wrfout`` on the prepared route is four folders down).
FIND_DEPTH = 5


# ------------------------------------------------------------------ the parent's saved output

def frame_listing(history: Path | None) -> dict[int, list[datetime]]:
    """The valid times of the frames a history folder holds, per grid id, oldest first."""

    times: dict[int, set[datetime]] = {}
    if history is None:
        return {}
    try:
        names = [entry.name for entry in history.iterdir()]
    except OSError:
        return {}
    for name in names:
        found = _FRAME.match(name)
        if not found:
            continue
        domain = int(found.group(1))
        try:
            valid = datetime.strptime(" ".join(found.group(2, 3, 4, 5)), "%Y-%m-%d %H %M %S")
        except ValueError:
            continue
        if 1 <= domain <= 99:
            times.setdefault(domain, set()).add(valid.replace(tzinfo=timezone.utc))
    return {domain: sorted(found) for domain, found in sorted(times.items())}


def cadence_seconds(times: list[datetime]) -> float | None:
    """How often a grid saved a frame: its most common gap, the smaller of two equally common ones."""

    gaps = Counter(int((b - a).total_seconds()) for a, b in zip(times, times[1:]) if b > a)
    if not gaps:
        return None
    most = max(gaps.values())
    return float(min(gap for gap, count in gaps.items() if count == most))


def checkpoint_sets(folder: Path | None) -> dict[int, int]:
    """How many checkpoint sets a folder holds per grid id (``gpuwmrst_dNN_<time>[__<id>].npz``)."""

    sets: dict[int, set[str]] = {}
    if folder is None:
        return {}
    try:
        names = [entry.name for entry in folder.iterdir()]
    except OSError:
        return {}
    for name in names:
        found = _CHECKPOINT.match(name)
        if found:
            sets.setdefault(int(found.group(1)), set()).add(found.group(2) + found.group(3))
    return {domain: len(found) for domain, found in sets.items()}


def _here(rundir: Path, recorded: str) -> Path | None:
    """A folder the event log recorded, found in this run folder.

    The log keeps the absolute path the run had when it wrote it; a folder copied to another place or another
    computer keeps the names under the run's own, so the tail after the run's name is looked for here first.
    """

    candidate = Path(recorded)
    if within(rundir, candidate) and candidate.is_dir():
        return candidate
    parts = [part for part in re.split(r"[\\/]", recorded) if part]
    # the run's name can recur below it (chain/NAME/run on the prepared route): the longest tail that is here wins
    for at, part in enumerate(parts):
        if part == rundir.name:
            here = rundir.joinpath(*parts[at + 1:])
            if here.is_dir():
                return here
    return candidate if candidate.is_dir() else None


def saved_output(rundir: Path) -> tuple[Path | None, Path | None]:
    """Where a run saved its frames and its checkpoint sets.

    From its own event log first: any grid's ``output_committed`` path names the frames folder, and
    ``model_progress.last_checkpoint`` the checkpoints (the prepared routes keep frames under ``wrfout/`` and
    checkpoints one folder up; a downscaled child keeps both beside its manifest).  A log that names neither is
    answered from the folder itself: the first folder under it holding frames.
    """

    history = checkpoint = None
    events = rundir / runs.EVENTS
    if events.is_file():
        for record in reversed(runs.event_records(events)):
            if history is None and record.get("event") == "output_committed" and record.get("domain") is not None \
                    and record.get("path"):
                parent = re.split(r"[\\/]", str(record["path"]))[:-1]
                history = _here(rundir, "/".join(parent)) if parent else None
            if checkpoint is None and record.get("last_checkpoint"):
                parent = re.split(r"[\\/]", str(record["last_checkpoint"]))[:-1]
                checkpoint = _here(rundir, "/".join(parent)) if parent else None
            if history is not None and checkpoint is not None:
                break
    if history is None:
        for depth in range(FIND_DEPTH):
            found = next(iter(sorted(rundir.glob("/".join(["*"] * depth + ["wrfout_d[0-9][0-9]_*"])))), None)
            if found is not None and _FRAME.match(found.name):
                history = found.parent
                break
    if checkpoint is None and history is not None:
        # the prepared routes: checkpoints in the run folder above wrfout/
        checkpoint = history.parent if checkpoint_sets(history.parent) else history
    return history, checkpoint


def grid_spacings(root: Path, rundir: Path) -> dict[int, float]:
    """Each grid's spacing in km, from the run's own resolved plan (a downscale's from its plan)."""

    events = rundir / runs.EVENTS
    grid = (runs.event_facts(events).get("grid") or {}) if events.is_file() else {}
    domains = list(grid.get("domains") or [])
    if not domains:
        child = runs.downscale_grid(root, rundir)
        domains = [row for row in (child or {}).get("domains") or [] if not row.get("context")]
        if not domains:
            plan = runs.downscale_plan(rundir) or {}
            try:
                domains = [{"grid_id": int(plan["child_grid_id"]),
                            "dx_km": float((plan.get("child_grid") or {})["dx"]) / 1000.0}]
            except (KeyError, TypeError, ValueError):
                domains = []
    out = {}
    for row in domains:
        try:
            if row.get("grid_id") is not None and float(row.get("dx_km") or 0) > 0:
                out[int(row["grid_id"])] = float(row["dx_km"])
        except (TypeError, ValueError):
            continue
    return out


def parent_facts(root: Path, rundir: Path, info: dict[str, Any]) -> dict[str, Any]:
    """What a run offers a finer forecast, and in words why not when it offers nothing.

    ``info`` is the run's derived state (:func:`runs.status`).  Eligibility is read from the run's own receipts,
    as the desktop panel reads it: a local run that finished or was stopped, with at least two saved frames of the
    chosen grid to drive the finer forecast between and one checkpoint set to read the parent's physics from.
    """

    history, checkpoint = saved_output(rundir)
    frames = frame_listing(history)
    sets = checkpoint_sets(checkpoint)
    if checkpoint != history:
        for domain, count in checkpoint_sets(history).items():
            sets[domain] = max(sets.get(domain, 0), count)
    spacing = grid_spacings(root, rundir)
    domains = [{"id": domain, "frames": len(times), "restart_sets": sets.get(domain, 0),
                "interval_s": cadence_seconds(times), "dx_km": spacing.get(domain),
                "first": times[0].strftime("%Y-%m-%dT%H:%M:%SZ"), "last": times[-1].strftime("%Y-%m-%dT%H:%M:%SZ")}
               for domain, times in frames.items()]
    # The finest grid the run saved: a finer forecast of a coarser grid is no finer than the nest the run already
    # has.  A downscaled run saved only its own grid (d02, d03, ...), so it is offered as a parent too.
    default = max(frames) if frames else None
    reason, why = eligibility(info, history, domains, default)
    return {"eligible": reason is None, "reason": reason, "why": why, "history": history, "checkpoint": checkpoint,
            "domains": domains, "domain": default}


def eligibility(info: dict[str, Any], history: Path | None, domains: list[dict[str, Any]],
                domain: int | None) -> tuple[str | None, str]:
    """``(reason, why)`` a run cannot be downscaled, or ``(None, "")``; the reasons are the desktop panel's."""

    state = info.get("state")
    if info.get("machine"):
        return (f"This forecast ran on {info['machine']}, so the output a finer forecast starts from is there, not "
                "on this computer.", "Downscale it on that machine.")
    if state not in ("finished", "stopped"):
        # a run still writing has a newest frame and checkpoint that may be half written
        return (f"This forecast is {state}. A finer forecast starts from a finished or stopped one, whose saved "
                "output is complete.", "Wait for it to finish, or stop it.")
    if history is None or not domains:
        return ("This forecast saved no output frames, so it cannot be downscaled.", "")
    row = next((item for item in domains if item["id"] == domain), None)
    if row is None:
        return (f"This forecast saved no frames of grid d{domain:02d}.", "Pick a grid it saved.")
    if row["frames"] < 2:
        return ("Only one saved output frame, so this run cannot be downscaled.",
                "A finer forecast is driven between two saved times; save output more often in the next forecast.")
    if row["restart_sets"] < 1:
        return ("No restart checkpoints saved, so this run cannot be downscaled.",
                "The engine reads the parent's physics from its checkpoint; keep checkpoints in the next forecast.")
    return None, ""


# ------------------------------------------------------------------ a drawn box, sized as the desktop sizes it

def _rem(value: float, modulus: float) -> float:
    return value % modulus  # Python's % is Rust's rem_euclid for a positive modulus


def longitude_span(west: float, east: float) -> float:
    span = east - west
    return span + 360.0 if span < 0 else span


def centre_longitude(west: float, east: float) -> float:
    value = _rem(west + longitude_span(west, east) * 0.5 + 180.0, 360.0) - 180.0
    return 180.0 if value == -180.0 else value


def continuous_box(south: float, west: float, north: float, east: float) -> tuple[float, float, float, float]:
    start = _rem(west + 180.0, 360.0) - 180.0
    return min(south, north), start, max(south, north), start + longitude_span(west, east)


def _distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1])
    h = math.sin(dlat * 0.5) ** 2 + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) \
        * math.sin(dlon * 0.5) ** 2
    return 2.0 * 6371.0 * math.asin(min(math.sqrt(h), 1.0))


def box_centre(south: float, west: float, north: float, east: float) -> tuple[float, float]:
    s, w, n, e = continuous_box(south, west, north, east)
    return (s + n) * 0.5, centre_longitude(w, e)


def box_extent_m(south: float, west: float, north: float, east: float) -> tuple[float, float]:
    """East-west along the great circle at the box's middle latitude, north-south along its centre meridian."""

    s, w, n, e = continuous_box(south, west, north, east)
    mid = (s + n) * 0.5
    lon = centre_longitude(w, e)
    return _distance_km((mid, w), (mid, e)) * 1000.0, _distance_km((s, lon), (n, lon)) * 1000.0


def round_half_away(value: float) -> float:
    """Rust's ``f64::round``: halves away from zero (Python's ``round`` goes to even)."""

    return math.copysign(math.floor(abs(value) + 0.5), value)


def box_cells(south: float, west: float, north: float, east: float, ratio: int,
              parent_spacing_m: float) -> tuple[int, int]:
    """The drawn box as child cells the engine places exactly (the desktop's ``box_cells``).

    The extent over the child spacing, rounded, then snapped to the nearest whole multiple of the ratio and never
    below two parent cells: the engine's own unit, so the grid it plans is the box as asked.
    """

    child = parent_spacing_m / max(ratio, 1)
    width, height = box_extent_m(south, west, north, east)

    def cells(metres: float) -> float:
        if not (math.isfinite(child) and child > 0):
            return 0.0
        return min(max(round_half_away(metres / child), 0.0), 2.0 ** 32 - 1)

    unit = float(max(ratio, 1))

    def snap(count: float) -> int:
        return int(min(max(round_half_away(count / unit) * unit, 2.0 * unit), 2.0 ** 32 - 1))

    return snap(cells(width)), snap(cells(height))


def box_refusal(south: float, west: float, north: float, east: float) -> str | None:
    """The desktop's words for a box that is not one area of the Earth, or None."""

    values = (south, west, north, east)
    if not all(math.isfinite(v) for v in values) or south == north or min(south, north) < -90 or max(south, north) > 90:
        return "The box has invalid latitude or longitude values. Draw a box with some area inside the map."
    s, _, n, _ = continuous_box(south, west, north, east)
    if s <= -90.0 or n >= 90.0:
        return "The box reaches a pole. Draw it away from the poles."
    span = longitude_span(west, east)
    if not 0 < span <= 180:
        return "The box spans more than half the globe from west to east. Draw a narrower box."
    return None


# ------------------------------------------------------------------ the command

def number(value: float) -> str:
    """A number as Rust writes an f64 with ``{}``, the way the terminal's guide spells every answer."""

    value = float(value)
    if value == 0.0:
        return "-0" if math.copysign(1.0, value) < 0 else "0"
    if value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    text = repr(value)
    return format(Decimal(text), "f") if "e" in text or "E" in text else text


def command_args(*, history: Path, lat: float, lon: float, domain: int, ratio: int,
                 child_size: tuple[int, int] | None, boundary_seconds: float | None, hours: float | None,
                 output_seconds: float | None, card: str | None, out: Path, plan: bool,
                 products: str) -> list[str]:
    """``gpuwm downscale``'s arguments, in the order the terminal's Downscale guide writes them.

    ``boundary_seconds`` is the cadence of the parent grid's saved frames (:func:`cadence_seconds`), asked for as
    the boundary ceiling; None leaves the engine to take the archive's own and say so.
    """

    args = [str(history), f"--point={number(lat)},{number(lon)}", "--parent-restart=latest",
            f"--parent-domain={domain}", f"--ratio={ratio}"]
    if child_size is not None:
        args.append(f"--child-size={child_size[0]},{child_size[1]}")
    if boundary_seconds is not None:
        args.append(f"--max-boundary-interval-seconds={number(boundary_seconds)}")
    if hours is not None:
        args.append(f"--hours={number(hours)}")
    if output_seconds is not None:
        args.append(f"--output-interval-seconds={number(output_seconds)}")
    if card is not None:
        # the capacity question's place in the guide: a declared card turns measuring off
        args.append(f"--card={card}")
    args += [f"--tiles={TILES}", "--out", str(out)]
    if plan:
        args.append("--dry-run")
    if card is None:
        args.append("--auto-vram")
    args.append(f"--render-products={products}")
    return args


def minutes_text(seconds: float) -> str:
    """The desktop's output-interval answer: the parent's cadence in minutes as its panel writes it."""

    minutes = seconds / 60.0
    return f"{minutes:.0f}" if abs(minutes - round(minutes)) < 1e-9 else f"{minutes:.2f}"


def default_name(parent: str, child_domain: int, ratio: int, taken) -> str:
    """``<parent>-d02-x3``, with ``-2``, ``-3`` when that name is taken, within the 64 characters a name has."""

    tail = f"-d{child_domain:02d}-x{ratio}"
    head = re.sub(r"[^A-Za-z0-9._-]", "-", parent)[: 64 - len(tail) - 4].rstrip("-.") or "run"
    name = base = f"{head}{tail}"
    n = 2
    while taken(name):
        name = f"{base}-{n}"
        n += 1
    return name


def name_taken(root: Path, name: str) -> bool:
    return (root / name).exists() or runs.start_record_path(root / name).exists()


def _corner(value: Any) -> list[float] | None:
    try:
        lat, lon = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return [lat, lon] if math.isfinite(lat) and math.isfinite(lon) else None


def plan_outline(plan: dict[str, Any]) -> list[list[float]] | None:
    """The engine's own child outline as ``[lat, lon]`` corners around its edge, or None.

    Read as the desktop panel reads it (``plan_outline``): the engine writes an object keyed by compass corner
    (``sw``, ``se``, ``ne``, ``nw``, beside ``parent_cells`` and ``basis``), read in that order; an array of
    corners is taken as it is.  Three finite corners are the least that outline anything.
    """

    outline = plan.get("child_outline")
    if isinstance(outline, dict):
        corners = [_corner(outline.get(key)) for key in ("sw", "se", "ne", "nw")]
    elif isinstance(outline, list):
        corners = [_corner(item) for item in outline]
    else:
        corners = []
    ring = [corner for corner in corners if corner is not None]
    return ring if len(ring) >= 3 else None


def plan_warnings(plan: dict[str, Any]) -> list[str]:
    """The engine's warning sentences: each record is ``{"action": sentence, "why": mechanism}`` (as
    :func:`gpuwm.explain.warn` writes it), read as the desktop panel reads it; a bare string is the sentence."""

    sentences = []
    for item in plan.get("warnings") or []:
        if isinstance(item, dict):
            text = next((item[key] for key in ("action", "message", "text")
                         if isinstance(item.get(key), str) and item[key].strip()), None)
        else:
            text = item if isinstance(item, str) else None
        if text and text.strip():
            sentences.append(scrub(text.strip().splitlines()[0]))
    if isinstance(plan.get("memory_note"), str) and plan["memory_note"].strip():
        sentences.append(scrub(plan["memory_note"].strip()))
    return sentences


def plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    """The engine's review of the child as the page shows it: the grid, its price, the streaming verdict, the
    engine's own warnings, and its outline on the parent map."""

    grid = plan.get("child_grid") or {}
    memory = plan.get("memory") or {}
    streaming = plan.get("streaming") or {}
    words = []
    try:
        dx_km = float(grid["dx"]) / 1000.0
        words.append(f"{int(grid['nx'])} by {int(grid['ny'])} points at {dx_km:g} km, {int(grid['nz'])} levels.")
    except (KeyError, TypeError, ValueError):
        dx_km = None
    hours = None
    if grid.get("run_seconds"):
        hours = round(float(grid["run_seconds"]) / 3600.0, 3)
        words.append(f"{hours:g} hour{'' if hours == 1 else 's'} of forecast.")
    peak, budget = memory.get("peak_envelope_bytes"), memory.get("budget_bytes")
    if peak and budget:
        words.append(f"Needs {peak / 2**30:.1f} GiB of the {budget / 2**30:.1f} GiB the card allows.")
    disk = plan.get("disk") if isinstance(plan.get("disk"), dict) else {}
    total, free = disk.get("total_bytes"), disk.get("free_bytes")
    if total is not None:
        if free is None:
            words.append(f"Writes about {total / 2**30:.1f} GiB to disk.")
        elif total > free:
            words.append(f"Writes about {total / 2**30:.1f} GiB to disk, more than the "
                         f"{free / 2**30:.1f} GiB free there.")
        else:
            words.append(f"Writes about {total / 2**30:.1f} GiB to disk, with "
                         f"{free / 2**30:.1f} GiB free there.")
    if streaming.get("mode") == "resident":
        words.append("Runs resident on the card.")
    elif streaming.get("mode") == "streamed":
        words.append("Streams through the card in tiles.")
    regime = plan.get("les_regime")
    return {
        "child_grid_id": plan.get("child_grid_id"), "parent_domain": plan.get("parent_domain"),
        "nx": grid.get("nx"), "ny": grid.get("ny"), "nz": grid.get("nz"), "dx_km": dx_km, "hours": hours,
        "output_interval_s": grid.get("output_interval_s"), "ratio": grid.get("ratio"),
        "i_parent_start": grid.get("i_parent_start"), "j_parent_start": grid.get("j_parent_start"),
        "outline": plan_outline(plan), "memory": memory or None,
        "disk": {key: disk.get(key) for key in (
            "total_bytes", "free_bytes", "fits", "history_bytes", "checkpoint_bytes", "picture_bytes",
            "pictures_per_frame", "keep_checkpoints")} if disk else None,
        "streaming": {key: scrub(str(streaming[key])) for key in ("mode", "why") if streaming.get(key) is not None}
        or None,
        "render_products": plan.get("render_products"), "warnings": plan_warnings(plan),
        "regime": scrub(str(regime.get("statement"))) if isinstance(regime, dict) and regime.get("statement") else None,
        "words": " ".join(words),
    }


def review_plan(folder: Path, name: str) -> dict[str, Any] | None:
    """The plan ``--dry-run`` wrote beside ``--out`` (``<name>.downscale-plan.json``)."""

    document = read_json(folder / f"{name}.{runs.DOWNSCALE_PLAN}", default=None)
    if isinstance(document, dict) and str(document.get("schema", "")).startswith(runs.DOWNSCALE_SCHEMA):
        return document
    return None


__all__ = ["DEFAULT_PRODUCTS", "DEFAULT_RATIO", "KEYS", "MODES", "PRODUCTS_RE", "RATIO_RANGE", "TILES",
           "box_cells", "box_centre", "box_refusal", "cadence_seconds", "checkpoint_sets", "command_args",
           "default_name", "eligibility", "frame_listing", "grid_spacings", "minutes_text",
           "name_taken", "number", "parent_facts", "plan_outline", "plan_summary", "plan_warnings", "review_plan",
           "round_half_away", "saved_output"]
