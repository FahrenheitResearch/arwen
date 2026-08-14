"""Draw a box on a map, see what it costs, start the nowcast.

``python -m tools.da_nowcast_launcher serve --work-root DIR`` opens a
local page on 127.0.0.1.  Drag a box over the map, pick a grid spacing
and an ensemble size, and the page answers -- BEFORE anything is
launched -- with the grid the box fits, the memory verdict, the radar
sites that cover it, and an estimate of what one assimilation cycle will
cost in wall-clock seconds.  Press the button and
:mod:`tools.da_nowcast_auto` starts on that box.

**A thin driver, not a second implementation.**  Every answer the page
gives comes from the shipped thing that owns it:

    grid + memory   ``gpuwm domain`` (the wizard fits the box, sizes the
                    ladder and runs the memory preflight; if it refuses,
                    the page shows its refusal and offers no override)
    radar sites     ``rw_nexrad sites`` -- the vendored site table, one
                    subprocess, never a list typed into this file
    the run itself  ``tools.da_nowcast_auto start``, exactly the command
                    line a person could type, printed on the page
    progress        the daemon's own ``gpuwm-da.nowcast-auto.v1`` status
                    file, polled and passed through unchanged

The only thing this module computes for itself is the wall-clock
estimate, and it is one measurement scaled by cell-steps with its basis
printed beside it.

**Offline.**  The page is one self-contained document: the basemap is
inlined from the same vendored Natural Earth assets the gallery draws
from, and there is no CDN, no external font, no external script.  It
loads with the network unplugged.

**Refusals are polite and specific.**  A box too small to hold a storm
and the ground it will cover, a box whose grid the card cannot hold, a
box whose cycle would take longer than the radar takes to produce the
next volume: each is refused with the number that made it so.

No radar-site names in this file, its defaults or its identifiers: the
site table is data read at runtime, and the site is an argument the
caller chooses from what covers their box (standing owner rule).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

#: The plan a caller is shown before they commit.
SCHEMA = "gpuwm-da.nowcast-plan.v1"

KM_PER_DEG_LAT = 111.2
EARTH_RADIUS_KM = 6371.0

#: The measured point every wall-clock estimate is scaled from.  One
#: trajectory advancing one leg, on this box, on the card in it.  It is
#: a MEASUREMENT with a date on it, not a model coefficient: the free
#: legs of the 2026-08-05 live run averaged 3.15 s per trajectory for a
#: 900 s leg on a 132x132x49 grid at dt 15 s (the observed legs of the
#: same run averaged 3.6 s, because they also diagnose against the
#: observation).  Scaling is by cell-steps.
REFERENCE = {
    "nx": 132, "ny": 132, "nz": 49, "dt_s": 15.0,
    "leg_seconds": 900.0, "seconds_per_trajectory_leg": 3.15,
    "members": 10, "solve_seconds": 7.0,
    "process_overhead_seconds": 20.0,
    "device": "NVIDIA GeForce RTX 5090",
    "measured": "2026-08-05 live cycled run, tools/da_cycle_prepared.py",
    "caveat": ("one measured configuration scaled by cell-steps; a "
               "short leg costs more than this predicts, because the "
               "per-leg model wiring is folded into the reference "
               "rather than separated from it"),
}

#: Nominal spacing between radar volumes in a precipitation VCP.  The
#: cycle has to fit inside it or the analysis falls further behind every
#: time round.
VOLUME_INTERVAL_SECONDS = 300.0

#: How many times the volume interval a cycle may take before the box is
#: refused rather than warned about.
CYCLE_BUDGET_REFUSE_FACTOR = 3.0

#: Fewest cells across the short side.  A supercell is tens of km wide
#: and travels 30-60 km in the 90 minutes a nowcast covers, so a domain
#: below this has the storm leaving through the boundary before the
#: forecast it is being run for is over.
MIN_CELLS_PER_SIDE = 60

#: The range authority observations are gridded within.  Read from the
#: superob defaults so this page and the obs builder cannot disagree.
def default_range_km() -> float:
    from gpuwm.obs.superob import SuperobParams

    return float(SuperobParams().max_range_km)


#: Where a run may be launched: a simple directory name, so a value
#: arriving from a browser can never climb out of the work root.
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class LauncherError(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"da_nowcast_launcher: {message}")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _py() -> str:
    return sys.executable or "python"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def length_scale_for(grid: dict, requested: float | None = None):
    """The perturbation scale this grid carries, through the front door.

    One implementation, in the module that owns the front door's own
    planning, so the number the page shows is the number the run uses.
    """

    from tools.da_nowcast import resolvable_length_scale_km

    return resolvable_length_scale_km(
        nx=grid["nx"], ny=grid["ny"], dx_m=grid["dx_km"] * 1000.0,
        dy_m=grid["dx_km"] * 1000.0, requested=requested)


# ---------------------------------------------------------------------------
# the box (pure)
# ---------------------------------------------------------------------------
def normalize_box(west: float, south: float, east: float,
                  north: float) -> dict:
    """A drawn rectangle, ordered and checked, or a refusal.

    Drag direction is not information: a box dragged up-left means the
    same rectangle as one dragged down-right, so the corners are sorted
    rather than trusted.  What IS refused is a rectangle with no area, a
    rectangle off the Earth, and one crossing the antimeridian -- the
    last because the wizard's polygon fitting has its own antimeridian
    handling and quietly handing it a box that means the long way round
    is not something to guess about.
    """

    west, east = sorted((float(west), float(east)))
    south, north = sorted((float(south), float(north)))
    if not (-90.0 < south < north < 90.0):
        raise LauncherError(
            f"latitudes {south:.4f}..{north:.4f} are not a box between "
            "the poles")
    if not (-180.0 <= west < east <= 180.0):
        raise LauncherError(
            f"longitudes {west:.4f}..{east:.4f} are not a box; a box "
            "crossing the antimeridian has to be given to the wizard "
            "directly")
    return {"west": west, "south": south, "east": east, "north": north,
            "center_lat": (south + north) / 2.0,
            "center_lon": (west + east) / 2.0}


def box_span_km(box: dict) -> tuple[float, float]:
    """(west-east, south-north) extent in km at the box's mid-latitude."""

    mid = math.radians(box["center_lat"])
    return ((box["east"] - box["west"]) * KM_PER_DEG_LAT
            * math.cos(mid),
            (box["north"] - box["south"]) * KM_PER_DEG_LAT)


def geojson_from_box(box: dict) -> dict:
    w, s, e, n = box["west"], box["south"], box["east"], box["north"]
    return {"type": "Polygon", "coordinates": [
        [[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def great_circle_km(lat1: float, lon1: float, lat2: float,
                    lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_lat = p2 - p1
    d_lon = math.radians(lon2 - lon1)
    h = (math.sin(d_lat / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def site_coverage(sites: list[dict], box: dict, *,
                  range_km: float) -> list[dict]:
    """Which radars can see this box, best first.

    ``full`` means every corner is inside the range authority, so the
    whole domain is observed; ``partial`` means the box centre is but a
    corner is not, which is a real and usable case (a domain biased
    downstream of a radar often is) and is labelled rather than hidden.
    A site that cannot reach the centre is not offered.
    """

    corners = [(box["south"], box["west"]), (box["south"], box["east"]),
               (box["north"], box["west"]), (box["north"], box["east"])]
    out = []
    for site in sites:
        lat = float(site["lat_deg"])
        lon = float(site["lon_deg"])
        centre = great_circle_km(lat, lon, box["center_lat"],
                                 box["center_lon"])
        farthest = max(great_circle_km(lat, lon, c_lat, c_lon)
                       for c_lat, c_lon in corners)
        if centre > range_km:
            continue
        out.append({
            "id": site["id"], "name": site.get("name", ""),
            "lat_deg": lat, "lon_deg": lon,
            "center_km": round(centre, 1),
            "farthest_corner_km": round(farthest, 1),
            "coverage": "full" if farthest <= range_km else "partial",
        })
    out.sort(key=lambda s: (s["coverage"] != "full", s["center_km"]))
    return out


def read_site_table() -> list[dict]:
    """The vendored NEXRAD site table, through its own front door."""

    from gpuwm.obs.nexrad import (find_nexrad_bin, nexrad_remedy,
                                  run_sites)

    binary = find_nexrad_bin()
    if binary is None:
        raise LauncherError(f"no rw_nexrad front door: "
                            f"{nexrad_remedy()}")
    return list(run_sites(binary).get("sites", []))


# ---------------------------------------------------------------------------
# what it will cost (pure, one measurement scaled)
# ---------------------------------------------------------------------------
def trajectory_leg_seconds(*, nx: int, ny: int, nz: int, dt_s: float,
                           leg_seconds: float) -> float:
    """Wall-clock for one trajectory to advance one leg, estimated."""

    ref = REFERENCE
    ref_work = (ref["nx"] * ref["ny"] * ref["nz"]
                * ref["leg_seconds"] / ref["dt_s"])
    work = nx * ny * nz * float(leg_seconds) / float(dt_s)
    return ref["seconds_per_trajectory_leg"] * work / ref_work


def cost_estimate(*, nx: int, ny: int, nz: int, dt_s: float,
                  members: int, cycle_seconds: float, free_legs: int,
                  free_leg_seconds: float) -> dict:
    """What one assimilation cycle and one forecast refresh will cost.

    The ADVANCE half is linear in the ensemble size, because the
    trajectories are advanced one after another on one card -- which is
    exactly why the ensemble size is a control on the page and not a
    constant in a file.  Measured 2026-08-05 and exact to three figures:
    3.06 s per member-leg at N=10, at N=20 and at N=36
    (evidence/da-demo/ensemble-size-sweep/n{10,20,36}/cycle-report.json).

    The SOLVE half below is modelled linear in N and is NOT.  The same
    sweep measured 6.3 s / 10.1 s / 71.9 s at N=10/20/36: the
    eigendecomposition is O(N^3) per active point and hides behind
    launch overhead until N is large, and --memory-budget-mib pins the
    chunk's bytes so the batch count rises with N on top of that.  This
    estimate therefore UNDER-predicts large ensembles; a linear model
    would have said ~1250 s for the N=36 run that took 1826 s.

    The coefficient is deliberately left alone rather than refitted:
    those timings were taken with cuSOLVER, and this tree now factors
    the LETKF's matrices with its own batched Jacobi kernel by default,
    so the superlinear term's SIZE here is not yet measured.  Its
    direction is.
    """

    trajectories = int(members) + 1          # the control runs too
    advance = trajectory_leg_seconds(
        nx=nx, ny=ny, nz=nz, dt_s=dt_s, leg_seconds=cycle_seconds)
    forecast_one = trajectory_leg_seconds(
        nx=nx, ny=ny, nz=nz, dt_s=dt_s,
        leg_seconds=free_legs * float(free_leg_seconds))
    cells = nx * ny * nz
    ref_cells = REFERENCE["nx"] * REFERENCE["ny"] * REFERENCE["nz"]
    solve = (REFERENCE["solve_seconds"] * (cells / ref_cells)
             * (int(members) / REFERENCE["members"]))
    assimilation = trajectories * advance + solve
    refresh = trajectories * forecast_one
    overhead = REFERENCE["process_overhead_seconds"]
    return {
        "trajectories": trajectories,
        "advance_seconds_per_trajectory": round(advance, 2),
        "assimilation_seconds": round(assimilation + overhead, 1),
        "forecast_refresh_seconds": round(refresh, 1),
        "cycle_seconds_total": round(assimilation + refresh + overhead,
                                     1),
        "solve_seconds": round(solve, 1),
        "process_overhead_seconds": overhead,
        "free_forecast_minutes": round(
            free_legs * float(free_leg_seconds) / 60.0),
        "basis": REFERENCE,
    }


def cycle_budget_verdict(cycle_total_s: float, *,
                         volume_interval_s: float = (
                             VOLUME_INTERVAL_SECONDS)) -> dict:
    """Can a cycle finish before the next volume lands?

    Slower than the feed is not automatically wrong -- the daemon works
    forwards through what it missed and says so -- but a cycle several
    times the volume interval never catches up, and that is a property
    of the box worth refusing before the card spends an hour proving it.
    """

    ratio = cycle_total_s / float(volume_interval_s)
    if ratio <= 1.0:
        return {"level": "ok", "ratio": round(ratio, 2),
                "message": (f"one cycle fits inside the "
                            f"{volume_interval_s / 60:.0f}-minute "
                            "volume interval with room to spare")}
    if ratio <= CYCLE_BUDGET_REFUSE_FACTOR:
        return {"level": "warn", "ratio": round(ratio, 2),
                "message": (f"one cycle is about {ratio:.1f}x the "
                            f"{volume_interval_s / 60:.0f}-minute "
                            "volume interval, so the analysis will run "
                            "behind the feed and the daemon will work "
                            "forwards through what it missed")}
    return {"level": "refuse", "ratio": round(ratio, 2),
            "message": (f"one cycle is about {ratio:.1f}x the "
                        f"{volume_interval_s / 60:.0f}-minute volume "
                        "interval; the analysis would fall further "
                        "behind every cycle and never catch up. Use a "
                        "smaller box, a coarser grid, or fewer members")}


def size_verdict(nx: int, ny: int, dx_km: float) -> dict | None:
    """Refuse a box too small to hold the thing it is nowcasting."""

    if min(nx, ny) >= MIN_CELLS_PER_SIDE:
        return None
    return {
        "level": "refuse",
        "message": (
            f"this box fits a {nx}x{ny} grid at {dx_km:g} km, and the "
            f"short side is under {MIN_CELLS_PER_SIDE} cells "
            f"({min(nx, ny) * dx_km:.0f} km). A storm is tens of km "
            "across and moves 30-60 km in the 90 minutes a nowcast "
            "covers, so it would leave through the boundary before the "
            "forecast was over. Draw a bigger box or coarsen the grid"),
    }


# ---------------------------------------------------------------------------
# the wizard is the sizing authority
# ---------------------------------------------------------------------------
def wizard_cmd(*, polygon: Path, out_toml: Path, dx_km: float,
               profile: str, hours: int, cycle: str, name: str,
               vram_gib: float | None) -> list[str]:
    argv = [_py(), "-m", "gpuwm.cli", "domain",
            "--polygon", str(polygon),
            "--root-dx", f"{dx_km:g}",
            "--physics-profile", profile,
            "--source", "gfs",
            "--cycle", cycle,
            "--hours", str(int(hours)),
            "--name", name,
            "--out", str(out_toml),
            "--explain"]
    if vram_gib is not None:
        argv.extend(("--vram-gib", f"{vram_gib:g}"))
    return argv


def detected_vram_gib() -> float | None:
    """This machine's card, if there is one; the wizard's own probe."""

    try:
        from gpuwm.domain_interactive import detected_vram_gib as probe

        return probe()
    except Exception:                       # no card, no driver, no CuPy
        return None


def fit_box(box: dict, *, dx_km: float, profile: str, hours: int,
            vram_gib: float | None, work_dir: Path) -> dict:
    """Ask the wizard what grid this box makes, and whether it fits.

    Its answer is taken whole.  A refusal is passed through with its own
    wording and no override, because a second opinion about memory from
    this file would be a second sizing path -- exactly what the
    interactive door was built not to be.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    polygon = work_dir / "domain-box.geojson"
    polygon.write_text(json.dumps(geojson_from_box(box)),
                       encoding="utf-8")
    out_toml = work_dir / "plan.toml"
    argv = wizard_cmd(polygon=polygon, out_toml=out_toml, dx_km=dx_km,
                      profile=profile, hours=hours,
                      cycle=f"{now_utc():%Y-%m-%d}T00", name="plan",
                      vram_gib=vram_gib)
    proc = subprocess.run(argv, cwd=str(repo_root()),
                          capture_output=True, text=True,
                          errors="replace")
    if proc.returncode != 0 or not out_toml.is_file():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        return {"ok": False, "argv": argv,
                "refusal": "\n".join(tail)
                           or f"gpuwm domain exited {proc.returncode}"}
    import tomllib

    payload = tomllib.loads(out_toml.read_text(encoding="utf-8"))
    domain = payload["domain"][0]
    return {
        "ok": True, "argv": argv,
        "nx": int(domain["nx"]), "ny": int(domain["ny"]),
        "nz": int(payload["shared"]["nz"]),
        "dx_km": float(domain["dx"]) / 1000.0,
        "dt_s": float(domain["time_step"]),
        "polygon": str(polygon),
        "explain_tail": proc.stdout.strip().splitlines()[-14:],
    }


def plan_box(box: dict, *, dx_km: float, members: int, free_legs: int,
             free_leg_seconds: float, cycle_seconds: float,
             profile: str, epoch_hours: int, range_km: float,
             vram_gib: float | None, work_dir: Path,
             sites: list[dict] | None = None) -> dict:
    """Everything a caller is shown before they commit to a box."""

    fitted = fit_box(box, dx_km=dx_km, profile=profile,
                     hours=epoch_hours, vram_gib=vram_gib,
                     work_dir=work_dir)
    span_ew, span_ns = box_span_km(box)
    plan = {
        "schema": SCHEMA,
        "generated": iso(now_utc()),
        "box": box,
        "box_span_km": {"west_east": round(span_ew, 1),
                        "south_north": round(span_ns, 1)},
        "requested": {"dx_km": dx_km, "members": members,
                      "free_legs": free_legs,
                      "free_leg_seconds": free_leg_seconds,
                      "cycle_seconds": cycle_seconds,
                      "physics_profile": profile,
                      "epoch_hours": epoch_hours,
                      "range_km": range_km},
        "refusals": [], "warnings": [],
        "honesty": ("demo-grade nowcast; UNSCORED, outside any "
                    "registered campaign; the cost figure is an "
                    "estimate scaled from one measurement"),
    }
    if not fitted["ok"]:
        plan["grid"] = None
        plan["refusals"].append({
            "level": "refuse", "source": "gpuwm domain",
            "message": fitted["refusal"]})
        plan["ok"] = False
        return plan

    plan["grid"] = {k: fitted[k] for k in
                    ("nx", "ny", "nz", "dx_km", "dt_s")}
    plan["grid"]["cells"] = fitted["nx"] * fitted["ny"] * fitted["nz"]
    plan["sizing_tail"] = fitted["explain_tail"]
    plan["polygon"] = fitted["polygon"]

    scale, scale_note = length_scale_for(plan["grid"])
    plan["length_scale_km"] = scale
    if scale_note:
        plan["warnings"].append({
            "level": "warn", "source": "perturbation scale",
            "message": scale_note})

    too_small = size_verdict(fitted["nx"], fitted["ny"], dx_km)
    if too_small:
        plan["refusals"].append({**too_small, "source": "domain size"})

    plan["cost"] = cost_estimate(
        nx=fitted["nx"], ny=fitted["ny"], nz=fitted["nz"],
        dt_s=fitted["dt_s"], members=members,
        cycle_seconds=cycle_seconds, free_legs=free_legs,
        free_leg_seconds=free_leg_seconds)
    budget = cycle_budget_verdict(plan["cost"]["cycle_seconds_total"])
    plan["cycle_budget"] = budget
    if budget["level"] == "refuse":
        plan["refusals"].append({**budget, "source": "cycle budget"})
    elif budget["level"] == "warn":
        plan["warnings"].append({**budget, "source": "cycle budget"})

    table = sites if sites is not None else read_site_table()
    plan["sites"] = site_coverage(table, box, range_km=range_km)
    if not plan["sites"]:
        plan["refusals"].append({
            "level": "refuse", "source": "radar coverage",
            "message": (
                f"no radar in the vendored site table is within "
                f"{range_km:g} km of this box's centre, so there are no "
                "observations to assimilate. Draw the box over a site")})
    elif not any(s["coverage"] == "full" for s in plan["sites"]):
        plan["warnings"].append({
            "level": "warn", "source": "radar coverage",
            "message": (
                "no single radar covers the whole box; the corners "
                f"beyond {range_km:g} km will carry no observations and "
                "the model runs free there")})
    plan["ok"] = not plan["refusals"]
    return plan


# ---------------------------------------------------------------------------
# the launch command (the same one a person could type)
# ---------------------------------------------------------------------------
def launch_argv(*, site: str, out: Path, polygon: Path, plan: dict,
                run_root: Path, vram_gib: float | None = None
                ) -> list[str]:
    """The daemon command line for a plan a caller accepted.

    The card is carried through, because the whole point of showing a
    memory verdict before the button is that pressing the button gets a
    run sized against the same card.
    """

    requested = plan["requested"]
    argv = [
        _py(), "-m", "tools.da_nowcast_auto", "start",
        "--site", site,
        "--out", str(out),
        "--polygon", str(polygon),
        "--members", str(int(requested["members"])),
        "--free-legs", str(int(requested["free_legs"])),
        "--free-leg-seconds", str(float(requested["free_leg_seconds"])),
        "--dx-km", f"{float(requested['dx_km']):g}",
        "--physics-profile", requested["physics_profile"],
        "--epoch-hours", str(int(requested["epoch_hours"])),
        "--run-root", str(run_root),
    ]
    if plan.get("length_scale_km") is not None:
        argv.extend(("--length-scale-km",
                     f"{float(plan['length_scale_km']):g}"))
    if vram_gib is not None:
        argv.extend(("--vram-gib", f"{float(vram_gib):g}"))
    return argv


def run_name(stamp: datetime | None = None) -> str:
    return f"nowcast-{(stamp or now_utc()):%Y%m%d-%H%M%S}"


def safe_run_dir(work_root: Path, name: str) -> Path:
    """A run directory named by an untrusted caller, kept inside root."""

    if not _RUN_NAME.match(str(name)):
        raise LauncherError(f"{name!r} is not a run name")
    return Path(work_root).resolve() / name


# ---------------------------------------------------------------------------
# the page: vendored basemap, inlined
# ---------------------------------------------------------------------------
def decimate(points, *, step: int):
    """Every ``step``-th vertex, keeping the last one.

    A browser drawing a whole continent's state borders does not need
    10 m vertices; keeping the endpoint means a decimated ring still
    closes where it started.
    """

    if step <= 1 or len(points) <= 2:
        return [list(p) for p in points]
    kept = [list(p) for p in points[::step]]
    if kept[-1] != list(points[-1]):
        kept.append(list(points[-1]))
    return kept


def basemap_polylines(extent: tuple[float, float, float, float], *,
                      step: int = 2) -> dict:
    """Coastline, national and state lines for one extent.

    From the SAME vendored assets the gallery renders on, at the 110 m
    resolution a screen-sized map wants -- so the page a caller draws on
    and the figures they get back are the same geography.
    """

    # Named refusal before the import; see tools/da_nowcast_render.py's
    # Basemap for the failure this replaces.
    from gpuwm.rustwx import basemap_dir, require_pyshp

    try:
        require_pyshp()
    except ImportError as error:
        raise LauncherError(str(error)) from error
    import shapefile

    assets = basemap_dir()
    if not assets.is_dir():
        raise LauncherError(
            f"vendored basemap assets missing at {assets}; the map "
            "cannot be drawn without them")
    x0, x1, y0, y1 = extent
    layers: dict[str, list] = {}
    for key, rel in (
            ("coast", "natural_earth_110m/ne_110m_coastline"),
            ("nation",
             "natural_earth_110m/ne_110m_admin_0_boundary_lines_land"),
            ("state",
             "natural_earth_110m/ne_110m_admin_1_states_provinces_lines")
    ):
        base = assets / rel
        reader = shapefile.Reader(shp=open(f"{base}.shp", "rb"),
                                  shx=open(f"{base}.shx", "rb"))
        segments = []
        for shape in reader.shapes():
            if getattr(shape, "shapeTypeName", "") == "NULL":
                continue
            bx0, by0, bx1, by1 = shape.bbox
            if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                continue
            pts = list(shape.points)
            idx = list(shape.parts) + [len(pts)]
            for k in range(len(idx) - 1):
                part = pts[idx[k]:idx[k + 1]]
                if len(part) >= 2:
                    segments.append(decimate(part, step=step))
        layers[key] = segments
    return layers


PAGE_CSS = """
:root{--bg:#14161a;--panel:#1c1f26;--ink:#e8e8e4;--dim:#9aa3ad;
--line:#2c313a;--accent:#4fa3e3;--ok:#3fae6a;--warn:#e08a2b;
--bad:#e0553c;--land:#EEEDE6;--water:#E0EAF2}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}
header{padding:.9rem 1.2rem;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:1.15rem;font-weight:600}
.sub{color:var(--dim);font-size:.83rem;margin-top:.2rem}
main{display:grid;grid-template-columns:minmax(0,1fr) 400px;
gap:1rem;padding:1rem;align-items:start}
@media (max-width:1000px){main{grid-template-columns:1fr}}
#mapwrap{position:relative;background:var(--water);border-radius:8px;
overflow:hidden;border:1px solid var(--line)}
canvas{display:block;width:100%;cursor:crosshair;touch-action:none}
.panel{background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:.9rem 1rem;margin-bottom:.8rem}
.panel h2{margin:0 0 .6rem;font-size:.82rem;letter-spacing:.06em;
text-transform:uppercase;color:var(--dim);font-weight:600}
label{display:block;font-size:.8rem;color:var(--dim);margin:.5rem 0 .15rem}
input,select,button{font:inherit;color:var(--ink);background:#12151a;
border:1px solid var(--line);border-radius:6px;padding:.4rem .55rem;
width:100%}
button{background:var(--accent);color:#07121c;border:0;font-weight:600;
cursor:pointer;padding:.55rem}
button:disabled{background:#39424d;color:#7d868f;cursor:not-allowed}
button.ghost{background:#232833;color:var(--ink);font-weight:500}
.row{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}
table{width:100%;border-collapse:collapse;font-size:.82rem}
td{padding:.22rem 0;border-bottom:1px solid #23272f}
td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.msg{padding:.5rem .6rem;border-radius:6px;font-size:.82rem;
margin:.4rem 0;border-left:3px solid}
.msg.refuse{background:#2a1614;border-color:var(--bad);color:#f0bdb2}
.msg.warn{background:#2a2314;border-color:var(--warn);color:#f0dcb2}
.msg.ok{background:#14261b;border-color:var(--ok);color:#b6e6c6}
.hint{color:var(--dim);font-size:.78rem}
pre{background:#0e1116;border:1px solid var(--line);border-radius:6px;
padding:.5rem;font-size:.72rem;overflow-x:auto;white-space:pre-wrap;
word-break:break-all;color:#b9c3cd}
a{color:var(--accent)}
.pill{display:inline-block;font-size:.7rem;padding:.05rem .4rem;
border-radius:99px;border:1px solid currentColor;margin-left:.35rem}
.full{color:var(--ok)}.partial{color:var(--warn)}
"""


#: Canvas width in device pixels; the height follows from the extent.
PAGE_CANVAS_WIDTH = 1200


def canvas_size(extent, *, width: int = PAGE_CANVAS_WIDTH
                ) -> tuple[int, int]:
    """Canvas pixels for an extent, at the extent's true proportions.

    The page projects equirectangularly with a cos(mid-latitude) scale
    on longitude, which cancels out of the pixel mapping -- so the ONLY
    thing keeping the map from being stretched is the canvas having the
    aspect ratio the ground does.  A box drawn on a stretched map is a
    box the caller did not mean.
    """

    west, east, south, north = extent
    mid = math.radians((south + north) / 2.0)
    ground_w = (east - west) * math.cos(mid)
    ground_h = (north - south)
    if ground_w <= 0 or ground_h <= 0:
        raise LauncherError(f"{extent} is not a map extent")
    return width, max(1, round(width * ground_h / ground_w))


def build_page(*, basemap: dict, sites: list[dict], extent, defaults: dict
               ) -> str:
    """One self-contained document: no CDN, no external font, no script
    that is not in this string."""

    width, height = canvas_size(extent)
    payload = json.dumps({"basemap": basemap, "sites": sites,
                          "extent": list(extent),
                          "defaults": defaults},
                         separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ArWen radar-DA nowcast — draw a box</title>
<style>{PAGE_CSS}</style></head><body>
<header>
  <h1>Radar-DA nowcast — draw a box, then run it</h1>
  <div class="sub">Drag a rectangle over the map. The box becomes the
  model domain. Nothing starts until you press Start.
  <b>Demo-grade, UNSCORED</b>, outside any registered campaign.</div>
</header>
<main>
  <div id="mapwrap"><canvas id="map" width="{width}" height="{height}"></canvas></div>
  <div>
    <div class="panel">
      <h2>The box</h2>
      <div id="boxinfo" class="hint">Drag on the map to draw one.</div>
      <button class="ghost" id="clear" style="margin-top:.6rem">
        Clear the box</button>
    </div>
    <div class="panel">
      <h2>Settings</h2>
      <div class="row">
        <div><label for="dx">Grid spacing (km)</label>
          <input id="dx" type="number" step="0.5" min="0.5" max="27"></div>
        <div><label for="members">Ensemble size (N)</label>
          <input id="members" type="number" step="1" min="2" max="96"></div>
      </div>
      <div class="row">
        <div><label for="freelegs">Forecast legs</label>
          <input id="freelegs" type="number" step="1" min="0" max="24"></div>
        <div><label for="freelegsec">Leg length (min)</label>
          <input id="freelegsec" type="number" step="5" min="5"
                 max="60"></div>
      </div>
      <div class="hint" id="horizon"></div>
      <button id="check" style="margin-top:.7rem" disabled>
        Check this box</button>
    </div>
    <div class="panel" id="planpanel" style="display:none">
      <h2>What you would get</h2>
      <div id="planbody"></div>
    </div>
    <div class="panel" id="sitepanel" style="display:none">
      <h2>Radar</h2>
      <label for="site">Sites that cover this box</label>
      <select id="site"></select>
      <button id="start" style="margin-top:.7rem" disabled>
        Start the nowcast</button>
      <div class="hint" style="margin-top:.5rem">This runs the command
      below. It keeps cycling until you stop it.</div>
      <pre id="cmd"></pre>
    </div>
    <div class="panel" id="runpanel" style="display:none">
      <h2>Running</h2>
      <div id="runbody"></div>
    </div>
  </div>
</main>
<script id="data" type="application/json">{payload}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const cv = document.getElementById('map'), cx = cv.getContext('2d');
const EXT = DATA.extent;               // [w,e,s,n]
const lat0 = (EXT[2] + EXT[3]) / 2;
const kx = Math.cos(lat0 * Math.PI / 180);
let box = null, drag = null, plan = null, run = null, timer = null;

function px(lon, lat) {{
  const x = (lon - EXT[0]) * kx, w = (EXT[1] - EXT[0]) * kx;
  const y = (EXT[3] - lat), h = (EXT[3] - EXT[2]);
  return [x / w * cv.width, y / h * cv.height];
}}
function geo(x, y) {{
  const w = (EXT[1] - EXT[0]) * kx, h = (EXT[3] - EXT[2]);
  return [EXT[0] + (x / cv.width) * w / kx,
          EXT[3] - (y / cv.height) * h];
}}
function poly(segs, color, width) {{
  cx.strokeStyle = color; cx.lineWidth = width; cx.beginPath();
  for (const seg of segs) {{
    let first = true;
    for (const p of seg) {{
      const q = px(p[0], p[1]);
      if (first) {{ cx.moveTo(q[0], q[1]); first = false; }}
      else cx.lineTo(q[0], q[1]);
    }}
  }}
  cx.stroke();
}}
function draw() {{
  cx.fillStyle = '#dfe7ef'; cx.fillRect(0, 0, cv.width, cv.height);
  poly(DATA.basemap.state, '#9fb0c0', 1);
  poly(DATA.basemap.nation, '#5d6c7c', 1.4);
  poly(DATA.basemap.coast, '#3c4a58', 1.6);
  for (const s of DATA.sites) {{
    const q = px(s.lon_deg, s.lat_deg);
    cx.fillStyle = '#2a6ea8'; cx.beginPath();
    cx.arc(q[0], q[1], 2.4, 0, 6.2832); cx.fill();
  }}
  if (box) {{
    const a = px(box.west, box.north), b = px(box.east, box.south);
    cx.fillStyle = 'rgba(79,163,227,.20)';
    cx.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    cx.strokeStyle = '#1d6fb8'; cx.lineWidth = 2;
    cx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    if (plan && plan.sites) for (const s of plan.sites) {{
      const q = px(s.lon_deg, s.lat_deg);
      cx.fillStyle = s.coverage === 'full' ? '#1c7a3f' : '#c07a1e';
      cx.beginPath(); cx.arc(q[0], q[1], 4.5, 0, 6.2832); cx.fill();
      cx.strokeStyle = '#0d1117'; cx.lineWidth = 1; cx.stroke();
    }}
  }}
}}
function at(ev) {{
  const r = cv.getBoundingClientRect();
  return [(ev.clientX - r.left) * cv.width / r.width,
          (ev.clientY - r.top) * cv.height / r.height];
}}
cv.addEventListener('pointerdown', e => {{
  drag = at(e); cv.setPointerCapture(e.pointerId);
}});
cv.addEventListener('pointermove', e => {{
  if (!drag) return;
  const p = at(e), a = geo(drag[0], drag[1]), b = geo(p[0], p[1]);
  box = {{west: Math.min(a[0], b[0]), east: Math.max(a[0], b[0]),
          south: Math.min(a[1], b[1]), north: Math.max(a[1], b[1])}};
  plan = null; showBox(); draw();
}});
cv.addEventListener('pointerup', () => {{ drag = null; }});
document.getElementById('clear').onclick = () => {{
  box = null; plan = null; showBox(); draw();
  document.getElementById('planpanel').style.display = 'none';
  document.getElementById('sitepanel').style.display = 'none';
}};
function km(box) {{
  const mid = (box.south + box.north) / 2 * Math.PI / 180;
  return [(box.east - box.west) * 111.2 * Math.cos(mid),
          (box.north - box.south) * 111.2];
}}
function showBox() {{
  const el = document.getElementById('boxinfo');
  document.getElementById('check').disabled = !box;
  if (!box) {{ el.className = 'hint';
    el.textContent = 'Drag on the map to draw one.'; return; }}
  const d = km(box);
  el.className = '';
  el.innerHTML = `<table>
    <tr><td>west / east</td><td>${{box.west.toFixed(2)}}° /
      ${{box.east.toFixed(2)}}°</td></tr>
    <tr><td>south / north</td><td>${{box.south.toFixed(2)}}° /
      ${{box.north.toFixed(2)}}°</td></tr>
    <tr><td>span</td><td>${{d[0].toFixed(0)}} × ${{d[1].toFixed(0)}}
      km</td></tr></table>`;
}}
function horizon() {{
  const n = +document.getElementById('freelegs').value;
  const m = +document.getElementById('freelegsec').value;
  document.getElementById('horizon').textContent =
    n ? `free forecast reaches +${{n * m}} min past the last volume`
      : 'no free forecast: analyses only';
}}
for (const id of ['freelegs', 'freelegsec'])
  document.getElementById(id).addEventListener('input', horizon);
function settings() {{
  return {{
    dx_km: +document.getElementById('dx').value,
    members: +document.getElementById('members').value,
    free_legs: +document.getElementById('freelegs').value,
    free_leg_seconds: +document.getElementById('freelegsec').value * 60
  }};
}}
function msg(level, text) {{
  return `<div class="msg ${{level}}">${{text}}</div>`;
}}
async function post(url, body) {{
  const r = await fetch(url, {{method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(body)}});
  return r.json();
}}
document.getElementById('check').onclick = async () => {{
  const btn = document.getElementById('check');
  btn.disabled = true; btn.textContent = 'Fitting the grid…';
  try {{
    plan = await post('/api/plan', Object.assign({{box}}, settings()));
  }} catch (err) {{
    plan = {{ok: false, refusals: [{{message: String(err)}}]}};
  }}
  btn.disabled = false; btn.textContent = 'Check this box';
  renderPlan(); draw();
}};
function renderPlan() {{
  const p = document.getElementById('planpanel');
  const b = document.getElementById('planbody');
  p.style.display = 'block';
  let h = '';
  if (plan.grid) {{
    const g = plan.grid, c = plan.cost;
    h += `<table>
      <tr><td>model grid</td><td>${{g.nx}} × ${{g.ny}} × ${{g.nz}}</td></tr>
      <tr><td>grid spacing</td><td>${{g.dx_km}} km</td></tr>
      <tr><td>timestep</td><td>${{g.dt_s}} s</td></tr>
      <tr><td>ensemble</td><td>N = ${{plan.requested.members}}
        (+1 control)</td></tr>
      <tr><td>one assimilation</td><td>~${{c.assimilation_seconds}} s</td></tr>
      <tr><td>forecast refresh</td><td>~${{c.forecast_refresh_seconds}} s
        (+${{c.free_forecast_minutes}} min)</td></tr>
      <tr><td><b>one full cycle</b></td>
        <td><b>~${{c.cycle_seconds_total}} s</b></td></tr></table>`;
    h += `<div class="hint" style="margin-top:.4rem">Estimate scaled
      from one measurement: ${{c.basis.seconds_per_trajectory_leg}} s per
      trajectory for a ${{c.basis.leg_seconds}} s leg on
      ${{c.basis.nx}}×${{c.basis.ny}}×${{c.basis.nz}},
      ${{c.basis.device}}. ${{c.basis.caveat}}</div>`;
  }}
  for (const r of (plan.refusals || [])) h += msg('refuse',
    `<b>Refused (${{r.source || 'check'}}):</b> ${{r.message}}`);
  for (const w of (plan.warnings || [])) h += msg('warn',
    `<b>Heads up:</b> ${{w.message}}`);
  if (plan.ok) h += msg('ok', plan.cycle_budget.message);
  b.innerHTML = h;
  const sp = document.getElementById('sitepanel');
  const sel = document.getElementById('site');
  sel.innerHTML = '';
  if (plan.ok && plan.sites && plan.sites.length) {{
    for (const s of plan.sites) {{
      const o = document.createElement('option');
      o.value = s.id;
      o.textContent = `${{s.id}} — ${{s.name}} · ${{s.center_km}} km ` +
        `· ${{s.coverage}} coverage`;
      sel.appendChild(o);
    }}
    sp.style.display = 'block';
    document.getElementById('start').disabled = false;
    showCmd();
    sel.onchange = showCmd;
  }} else {{
    sp.style.display = 'none';
  }}
}}
function showCmd() {{
  const site = document.getElementById('site').value;
  const s = settings();
  document.getElementById('cmd').textContent =
    `python -m tools.da_nowcast_auto start --site ${{site}} ` +
    `--out <run dir> --polygon <the box you drew> ` +
    `--members ${{s.members}} --free-legs ${{s.free_legs}} ` +
    `--free-leg-seconds ${{s.free_leg_seconds}} --dx-km ${{s.dx_km}}`;
}}
document.getElementById('start').onclick = async () => {{
  const btn = document.getElementById('start');
  btn.disabled = true; btn.textContent = 'Starting…';
  const site = document.getElementById('site').value;
  run = await post('/api/launch',
    Object.assign({{box, site}}, settings()));
  btn.textContent = 'Started';
  document.getElementById('runpanel').style.display = 'block';
  if (timer) clearInterval(timer);
  timer = setInterval(poll, 5000); poll();
}};
async function poll() {{
  if (!run || !run.run) return;
  const r = await fetch('/api/status?run=' +
    encodeURIComponent(run.run));
  const s = await r.json();
  const b = document.getElementById('runbody');
  if (s.error) {{ b.innerHTML = msg('warn', s.error); return; }}
  let h = `<table>
    <tr><td>state</td><td>${{s.state}}</td></tr>
    <tr><td>cycles</td><td>${{s.cycles_completed}}</td></tr>
    <tr><td>site</td><td>${{s.site}}</td></tr>
    <tr><td>ensemble</td><td>N = ${{s.members}}</td></tr></table>`;
  h += msg(s.state === 'failed' ? 'refuse' : 'ok', s.verdict);
  if (s.notice) h += msg(s.notice.level === 'warn' ? 'warn' : 'ok',
    `<b>${{s.notice.headline}}</b><br>${{s.notice.detail}}`);
  h += `<p><a href="/runs/${{encodeURIComponent(run.run)}}/gallery/"
    target="_blank">Open the gallery →</a> (it redraws itself every
    cycle; refresh the page)</p>`;
  h += `<button class="ghost" onclick="stopRun()">Stop after this
    cycle</button>`;
  b.innerHTML = h;
}}
async function stopRun() {{
  await post('/api/stop', {{run: run.run}});
  poll();
}}
const D = DATA.defaults;
document.getElementById('dx').value = D.dx_km;
document.getElementById('members').value = D.members;
document.getElementById('freelegs').value = D.free_legs;
document.getElementById('freelegsec').value = D.free_leg_seconds / 60;
horizon(); showBox(); draw();
</script></body></html>
"""


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------
class LauncherState:
    """Everything the request handlers are allowed to touch."""

    def __init__(self, *, work_root: Path, run_root: Path,
                 defaults: dict, extent, vram_gib: float | None,
                 range_km: float) -> None:
        self.work_root = Path(work_root).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.run_root = Path(run_root).resolve()
        self.defaults = defaults
        self.extent = extent
        self.vram_gib = vram_gib
        self.range_km = range_km
        self.lock = threading.Lock()
        self.sites = read_site_table()
        self.page = build_page(
            basemap=basemap_polylines(
                (extent[0], extent[1], extent[2], extent[3])),
            sites=[{"id": s["id"], "lat_deg": s["lat_deg"],
                    "lon_deg": s["lon_deg"]} for s in self.sites],
            extent=extent, defaults=defaults)


def make_handler(state: LauncherState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "gpuwm-da-launcher/1"

        def log_message(self, fmt, *args):
            print(f"[{now_utc():%H:%M:%S}Z] {fmt % args}", flush=True)

        # -- helpers ---------------------------------------------------
        def send_json(self, payload, code: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_bytes(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1 << 20:
                raise LauncherError("request body missing or too large")
            return json.loads(self.rfile.read(length))

        def box_from(self, payload: dict) -> dict:
            box = payload.get("box") or {}
            return normalize_box(box["west"], box["south"], box["east"],
                                 box["north"])

        # -- routes ----------------------------------------------------
        def do_GET(self):                    # noqa: N802 (stdlib name)
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self.send_bytes(state.page.encode("utf-8"),
                                       "text/html; charset=utf-8")
            if url.path == "/api/sites":
                return self.send_json({"schema":
                                       "gpuwm-obs.nexrad-sites.v1",
                                       "sites": state.sites})
            if url.path == "/api/status":
                query = parse_qs(url.query)
                name = (query.get("run") or [""])[0]
                try:
                    path = (safe_run_dir(state.work_root, name)
                            / "auto-status.json")
                except SystemExit as refusal:
                    return self.send_json({"error": str(refusal)}, 400)
                if not path.is_file():
                    return self.send_json(
                        {"error": "no status file yet; the daemon "
                                  "writes one as it starts"}, 404)
                return self.send_json(
                    json.loads(path.read_text(encoding="utf-8")))
            if url.path.startswith("/runs/"):
                return self.serve_gallery(url.path)
            return self.send_json({"error": "no such route"}, 404)

        def serve_gallery(self, path: str):
            parts = [p for p in path.split("/") if p]
            # /runs/<name>/gallery[/<file>] -- nothing else, and the
            # file is a basename, so no request can walk out of it.
            if len(parts) < 3 or parts[2] != "gallery":
                return self.send_json({"error": "no such route"}, 404)
            try:
                gallery = (safe_run_dir(state.work_root, parts[1])
                           / "gallery")
            except SystemExit as refusal:
                return self.send_json({"error": str(refusal)}, 400)
            name = parts[3] if len(parts) > 3 else "index.html"
            target = gallery / Path(name).name
            if not target.is_file():
                return self.send_json(
                    {"error": f"{name} does not exist yet; the gallery "
                              "appears after the first render"}, 404)
            ctype = ("text/html; charset=utf-8"
                     if target.suffix == ".html"
                     else "image/png" if target.suffix == ".png"
                     else "application/json"
                     if target.suffix == ".json" else
                     "application/octet-stream")
            return self.send_bytes(target.read_bytes(), ctype)

        def do_POST(self):                   # noqa: N802 (stdlib name)
            url = urlparse(self.path)
            try:
                payload = self.read_json()
                if url.path == "/api/plan":
                    return self.send_json(self.plan(payload))
                if url.path == "/api/launch":
                    return self.send_json(self.launch(payload))
                if url.path == "/api/stop":
                    return self.send_json(self.stop(payload))
            except SystemExit as refusal:
                return self.send_json(
                    {"ok": False,
                     "refusals": [{"source": "launcher",
                                   "message": str(refusal)}]}, 400)
            except Exception as error:       # a bug, said plainly
                return self.send_json(
                    {"ok": False,
                     "refusals": [{"source": "launcher",
                                   "message": f"{type(error).__name__}:"
                                              f" {error}"}]}, 500)
            return self.send_json({"error": "no such route"}, 404)

        def plan(self, payload: dict) -> dict:
            box = self.box_from(payload)
            with tempfile.TemporaryDirectory(
                    prefix="da-nowcast-plan-") as tmp:
                return plan_box(
                    box, dx_km=float(payload["dx_km"]),
                    members=int(payload["members"]),
                    free_legs=int(payload["free_legs"]),
                    free_leg_seconds=float(payload["free_leg_seconds"]),
                    cycle_seconds=state.defaults["cycle_seconds"],
                    profile=state.defaults["physics_profile"],
                    epoch_hours=state.defaults["epoch_hours"],
                    range_km=state.range_km, vram_gib=state.vram_gib,
                    work_dir=Path(tmp), sites=state.sites)

        def launch(self, payload: dict) -> dict:
            box = self.box_from(payload)
            site = str(payload["site"]).strip().upper()
            with state.lock:
                name = run_name()
                out = safe_run_dir(state.work_root, name)
                out.mkdir(parents=True, exist_ok=True)
                plan = plan_box(
                    box, dx_km=float(payload["dx_km"]),
                    members=int(payload["members"]),
                    free_legs=int(payload["free_legs"]),
                    free_leg_seconds=float(payload["free_leg_seconds"]),
                    cycle_seconds=state.defaults["cycle_seconds"],
                    profile=state.defaults["physics_profile"],
                    epoch_hours=state.defaults["epoch_hours"],
                    range_km=state.range_km, vram_gib=state.vram_gib,
                    work_dir=out, sites=state.sites)
                if not plan["ok"]:
                    # The page checked before offering the button; this
                    # is the same check on the way in, because a client
                    # is not a gate.
                    return {"ok": False, "refusals": plan["refusals"]}
                if not any(s["id"] == site for s in plan["sites"]):
                    return {"ok": False, "refusals": [{
                        "source": "radar coverage",
                        "message": f"{site} does not cover this box"}]}
                polygon = Path(plan["polygon"])
                argv = launch_argv(site=site, out=out, polygon=polygon,
                                   plan=plan, run_root=state.run_root,
                                   vram_gib=state.vram_gib)
                (out / "launch-plan.json").write_text(
                    json.dumps({**plan, "site": site, "argv": argv},
                               indent=1, default=str),
                    encoding="utf-8")
                proc = subprocess.run(argv, cwd=str(state.run_root),
                                      capture_output=True, text=True,
                                      errors="replace")
                if proc.returncode != 0:
                    return {"ok": False, "refusals": [{
                        "source": "da_nowcast_auto start",
                        "message": (proc.stderr or proc.stdout
                                    ).strip()[-800:]}]}
                return {"ok": True, "run": name, "out": str(out),
                        "argv": argv, "site": site,
                        "gallery": f"/runs/{name}/gallery/",
                        "stdout": proc.stdout.strip().splitlines()}

        def stop(self, payload: dict) -> dict:
            out = safe_run_dir(state.work_root, str(payload["run"]))
            proc = subprocess.run(
                [_py(), "-m", "tools.da_nowcast_auto", "stop",
                 "--out", str(out)], cwd=str(state.run_root),
                capture_output=True, text=True, errors="replace")
            return {"ok": proc.returncode == 0,
                    "message": (proc.stdout or proc.stderr).strip()}

    return Handler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_EXTENT = (-126.0, -66.0, 23.0, 50.5)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dx-km", type=float, default=3.0,
                        help="grid spacing the page opens with")
    # Default and justification live in tools.da_nowcast.DEFAULT_MEMBERS
    # (measured 2026-08-05 on a 32 GB and a 16 GB card).
    parser.add_argument("--members", type=int, default=10,
                        help="ensemble size the page opens with "
                             "(default 10). Measured: N=20 buys "
                             "+0.0018 FSS inside a 0.0062-0.0074 "
                             "across-member scatter, N=36 scores below "
                             "N=10, N=4 costs 0.007 and saves no VRAM")
    parser.add_argument("--free-legs", type=int, default=6)
    parser.add_argument("--free-leg-seconds", type=float, default=900.0)
    parser.add_argument("--cycle-seconds", type=float, default=300.0,
                        help="the cycle length the cost estimate is "
                             "quoted for; the daemon itself cycles on "
                             "the radar's own volume times")
    # One owner for the default, so this door and `da_nowcast run` cannot
    # disagree about which suite an unnamed nowcast gets.
    from tools.da_nowcast import NOWCAST_DEFAULT_PHYSICS_PROFILE

    parser.add_argument("--physics-profile",
                        default=NOWCAST_DEFAULT_PHYSICS_PROFILE,
                        help="shipped physics profile for every stage "
                             f"(default {NOWCAST_DEFAULT_PHYSICS_PROFILE})")
    parser.add_argument("--epoch-hours", type=int, default=4)
    parser.add_argument("--range-km", type=float, default=None,
                        help="radar range authority for coverage "
                             "(default: the superob default)")
    parser.add_argument("--vram-gib", type=float, default=None,
                        help="size against this card instead of the "
                             "one in this machine")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_nowcast_launcher",
        description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    serve = sub.add_parser(
        "serve", help="open the local page and drive the CLI from it")
    serve.add_argument("--work-root", type=Path, required=True,
                       help="where launched runs are written")
    serve.add_argument("--run-root", type=Path, default=None,
                       help="the worktree runs execute out of; point it "
                            "at a tree nobody commits into")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--host", default="127.0.0.1",
                       help="bind address; local by default and there "
                            "is no authentication, so anything else is "
                            "a deliberate choice")
    serve.add_argument("--open", action="store_true",
                       help="open a browser at the page")
    add_common(serve)

    plan = sub.add_parser(
        "plan", help="the same check the page runs, on the command line")
    plan.add_argument("--box", required=True,
                      help="WEST,SOUTH,EAST,NORTH in degrees")
    plan.add_argument("--json", action="store_true")
    add_common(plan)

    page = sub.add_parser(
        "page", help="write the self-contained page to a file and stop")
    page.add_argument("--out", type=Path, required=True)
    add_common(page)
    return parser


def defaults_from(args) -> dict:
    return {"dx_km": args.dx_km, "members": args.members,
            "free_legs": args.free_legs,
            "free_leg_seconds": args.free_leg_seconds,
            "cycle_seconds": args.cycle_seconds,
            "physics_profile": args.physics_profile,
            "epoch_hours": args.epoch_hours}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    range_km = (args.range_km if args.range_km is not None
                else default_range_km())
    if args.mode == "page":
        page = build_page(
            basemap=basemap_polylines(DEFAULT_EXTENT),
            sites=[{"id": s["id"], "lat_deg": s["lat_deg"],
                    "lon_deg": s["lon_deg"]} for s in read_site_table()],
            extent=DEFAULT_EXTENT, defaults=defaults_from(args))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(page, encoding="utf-8")
        print(f"wrote {args.out} ({len(page) / 1024:.0f} KiB, "
              "self-contained)")
        return 0

    if args.mode == "plan":
        west, south, east, north = (
            float(v) for v in str(args.box).split(","))
        box = normalize_box(west, south, east, north)
        with tempfile.TemporaryDirectory(
                prefix="da-nowcast-plan-") as tmp:
            plan = plan_box(
                box, dx_km=args.dx_km, members=args.members,
                free_legs=args.free_legs,
                free_leg_seconds=args.free_leg_seconds,
                cycle_seconds=args.cycle_seconds,
                profile=args.physics_profile,
                epoch_hours=args.epoch_hours, range_km=range_km,
                vram_gib=args.vram_gib, work_dir=Path(tmp))
        if args.json:
            print(json.dumps(plan, indent=1, default=str))
            return 0 if plan["ok"] else 2
        print_plan(plan)
        return 0 if plan["ok"] else 2

    state = LauncherState(
        work_root=args.work_root,
        run_root=args.run_root or repo_root(),
        defaults=defaults_from(args), extent=DEFAULT_EXTENT,
        vram_gib=args.vram_gib, range_km=range_km)
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"launcher on {url}")
    print(f"  runs go to  : {state.work_root}")
    print(f"  runs execute: {state.run_root}")
    print("                (point --run-root at a worktree nobody "
          "commits into: a commit landing mid-run stops the daemon)")
    print(f"  sites known : {len(state.sites)}")
    print("  the page is self-contained; nothing is fetched from the "
          "network to draw it")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("launcher stopped")
    return 0


def print_plan(plan: dict) -> None:
    grid = plan.get("grid")
    span = plan["box_span_km"]
    print(f"box {span['west_east']:.0f} x {span['south_north']:.0f} km")
    if grid:
        cost = plan["cost"]
        print(f"grid {grid['nx']}x{grid['ny']}x{grid['nz']} at "
              f"{grid['dx_km']:g} km, dt {grid['dt_s']:g} s")
        print(f"cost ~{cost['assimilation_seconds']:.0f} s per "
              f"assimilation + ~{cost['forecast_refresh_seconds']:.0f} s "
              f"per forecast refresh = "
              f"~{cost['cycle_seconds_total']:.0f} s per cycle "
              f"(N={plan['requested']['members']})")
        print(f"     {cost['basis']['caveat']}")
    for site in plan.get("sites", [])[:6]:
        print(f"site {site['id']}  {site['center_km']:.0f} km from the "
              f"centre, {site['coverage']} coverage — {site['name']}")
    for entry in plan["refusals"]:
        print(f"REFUSED ({entry['source']}): {entry['message']}")
    for entry in plan["warnings"]:
        print(f"warning ({entry['source']}): {entry['message']}")
    print("GO" if plan["ok"] else "NO-GO")


if __name__ == "__main__":
    raise SystemExit(main())
