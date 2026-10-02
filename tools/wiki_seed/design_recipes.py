"""Design the best simulation per seed event and card class, then check every one with the engine.

CPU only: every check is `gpuwm run-plan --resolve` or `gpuwm cyclone-setup` for a declared card.
Writes design-result.json beside a fit cache; build_recipes.py turns it into wiki documents.

Rungs are ordered best first. For each card the first rung the engine accepts on that card wins, provided
its priced peak is within HEADROOM_FRACTION of the card's budget, no grid finer than 4 km carries a
cumulus scheme, every ERA5-driven 12 km grid reaches at least RING_MIN_KM beyond the grid inside it, and
everything the run writes (its download, its preparation, its history, its checkpoints and its pictures)
fits the card's disk budget at one of the allowed output intervals (all read back from what the engine
generated).

Every buffer_km value is measured by the engine from the box itself (gpuwm.domain_wizard._buffer_cells), not
from the next inner grid, so a ring of R km beyond a grid whose own buffer is B is written B + R.
"""
from __future__ import annotations
import json, math, os, re, subprocess, sys, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

S = Path(__file__).resolve().parent
#: The source tree whose engine checks every recipe (it must be installed in PY's environment).
W = os.environ.get("RECIPE_ENGINE_TREE", str(S.parents[1]))
#: The tree whose gpuwm/gui/seed/wiki-seed.json lists the events.
SEED = os.environ.get("RECIPE_SEED_TREE", W)
#: The interpreter the engine runs under.
PY = os.environ.get("RECIPE_PYTHON", sys.executable)
sys.path.insert(0, W)
from gpuwm.gui.api import region_polygon  # noqa: E402

CARDS = [8, 12, 16, 24, 32]
#: HRRR's archive on the public S3 mirror, which the engine reads for any past cycle, starts US tornadoes
#: from this date. Every other event starts from ERA5 through the keyless ARCO archive: one consistent
#: analysis for any date, where archived GFS exists only for recent years.
HRRR_FROM = "2014-10-01T00"
WORK = Path(os.environ.get("RECIPE_WORK", S / "work"))
TMP = WORK / "tmp"
CACHE = WORK / "fitcache"
TMP.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, GPUWM_NO_LOCAL_GPU="1", PYTHONPATH=W)

_NEEDS = re.compile(r"forecast needs ([\d.]+) GiB; whole-process budget ([\d.]+) GiB")
_EXCEEDS = re.compile(r"at ([\d.]+) GiB peak envelope.*?EXCEEDS the ([\d.]+) GiB budget", re.S)
_NEEDS2 = re.compile(r"needs ([\d.]+) GiB peak envelope.*?EXCEEDS the ([\d.]+) GiB budget", re.S)


def memory(text: str):
    """The memory a REFUSED fit needed, from the words it printed, for an engine tree older than the refusal
    document (RECIPE_ENGINE_TREE may name one)."""
    m = _NEEDS.search(text) or _EXCEEDS.search(text) or _NEEDS2.search(text)
    if not m:
        return None
    need, budget = float(m.group(1)), float(m.group(2))
    return {"need_gib": need, "budget_gib": budget, "fits": need <= budget}


def fitted_memory(resolved: dict):
    """The memory a fit was priced at, from its document's ``memory`` record, in bytes: the resolve document of
    an accepted fit, or the memory refusal document of a fit too big for its card."""
    record = resolved.get("memory") or {}
    need, budget = record.get("peak_envelope_bytes"), record.get("budget_bytes")
    if not need or not budget:
        return None
    gib = 1024 ** 3
    return {"need_gib": round(need / gib, 2), "budget_gib": round(budget / gib, 2), "fits": need <= budget}


def run(argv, cwd=None, timeout=900):
    done = subprocess.run(argv, cwd=cwd, capture_output=True, env=ENV, timeout=timeout, stdin=subprocess.DEVNULL)
    return done.returncode, done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace")


def cached(key: str, fn):
    import hashlib
    path = CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    value = fn()
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


# ------------------------------------------------------------------ engine doors

def _measured_key():
    """The selected writer schema, checkpoint measurements and disk tables."""
    import hashlib
    from gpuwm import disk_budget, download_budget
    history_modules = ("disk_budget.py", "config.py", "core/clock.py",
                       "io/history_layout.py", "io/history_selection.py",
                       "io/wrf_output_schema.py", "core/device_inventory.py",
                       "core/physics_inventory.py", "core/preflight.py")
    package = Path(disk_budget.__file__).parent
    history_key = hashlib.sha256(b"".join(
        (package / name).read_bytes() for name in history_modules)).hexdigest()[:16]
    return [history_key, disk_budget.CHECKPOINT_BYTES_PER_CELL,
            disk_budget.ROOT_CHECKPOINT_BYTES_PER_CELL,
            hashlib.sha256(disk_budget.picture_table_path().read_bytes()).hexdigest()[:16],
            hashlib.sha256(download_budget.table_path().read_bytes()).hexdigest()[:16]]


def fit_domain(intent: dict, box: dict, card: int) -> dict:
    # The resolve answer carries the engine's disk projection, so a cached fit is only good for the
    # bytes-per-cell figures it was projected with; new measurements must re-resolve, or every row
    # would be compared with a projection made from the old figures.
    measured = _measured_key()
    key = json.dumps({"i": intent, "b": box, "c": card, "m": measured}, sort_keys=True)

    def go():
        folder = Path(tempfile.mkdtemp(prefix="fit-", dir=TMP))
        try:
            (folder / "region.geojson").write_text(json.dumps(
                region_polygon(box["lat"], box["lon"], box["width_km"], box["height_km"])))
            full = dict(intent, polygon=str(folder / "region.geojson"), card=f"{card}gb")
            route = "experiment" if full["source"] == "era5" else "prepared"
            plan = {"schema": "gpuwm.run-plan.v1", "name": "fit", "route": route,
                    "config": {"intent": full}, "output_root": str(folder)}
            (folder / "plan.json").write_text(json.dumps(plan))
            rc, out, err = run([PY, "-m", "gpuwm", "run-plan", str(folder / "plan.json"), "--resolve"], cwd=str(folder))
            if rc != 0:
                lines = [l for l in err.splitlines() if l.strip() and "installed wheel" not in l]
                try:
                    refusal = json.loads(out)
                except ValueError:
                    refusal = None
                priced = fitted_memory(refusal) if isinstance(refusal, dict) else None
                return {"ok": False, "memory": priced or memory(err), "error": lines[-6:]}
            resolved = json.loads(out)
            mem = fitted_memory(resolved)
            exp = (resolved.get("configuration") or {}).get("experiment") or {}
            doms = []
            for d in exp.get("domains") or []:
                r = d.get("run") or {}
                nx, ny, dx = int(r.get("nx") or 0), int(r.get("ny") or 0), float(r.get("dx") or 0) / 1000
                doms.append({"grid_id": d.get("grid_id"), "parent_id": d.get("parent_id"), "nx": nx, "ny": ny,
                             "nz": r.get("nz"), "dx_km": round(dx, 4), "width_km": round((nx - 1) * dx, 1),
                             "height_km": round((ny - 1) * dx, 1), "parent_grid_ratio": d.get("parent_grid_ratio"),
                             "i_parent_start": d.get("i_parent_start"), "j_parent_start": d.get("j_parent_start"),
                             "following": bool(d.get("follow"))})
            physics = ""
            import tomllib
            generated = str(resolved.get("generated_config") or "")
            tables = tomllib.loads(generated).get("domain") or [] if generated else []
            for row, table in zip(doms, tables):
                row["cu_physics"] = int(table.get("cu_physics", 0))
            for line in generated.splitlines():
                if line.startswith("# PHYSICS:"):
                    physics = line[len("# PHYSICS:"):].strip()
                    break
            for row, d in zip(doms, exp.get("domains") or []):
                row["history_interval_s"] = float(d.get("history_interval_s") or 0)
            proj = exp.get("projection") or {}
            disk = resolved.get("disk") or {}
            return {"ok": True, "memory": mem, "domains": doms, "physics": physics,
                    "disk": disk, "restart_interval_s": exp.get("restart_interval_s"),
                    # What the run downloads and which preparation it takes, so the designer's own
                    # projection at other output intervals counts them the way the engine does.
                    "fetch": (tomllib.loads(generated).get("fetch") if generated else None),
                    "chain": (disk.get("preparation") or {}).get("chain"),
                    "start_time": exp.get("start_time"), "run_seconds": exp.get("run_seconds"),
                    "projection": {k: proj.get(k) for k in ("map_proj", "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon")},
                    "warnings": [str(w) for w in resolved.get("warnings") or []]}
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    return cached("d" + key, go)


def fit_cyclone(args: dict, card: int) -> dict:
    key = json.dumps({"a": args, "c": card, "m": _measured_key()}, sort_keys=True)

    def go():
        folder = Path(tempfile.mkdtemp(prefix="cyc-", dir=TMP))
        argv = [PY, "-m", "gpuwm", "cyclone-setup", "--source", args["source"], "--cycle", args["cycle"],
                f"--advisory-position={args['advisory_position']}", "--hours", str(args["hours"]),
                "--card", f"{card}gb", "--out", str(folder / "cyclone.toml"), "--json"]
        if args.get("nest_budget_gib"):
            argv[-1:-1] = ["--nest-budget-gib", str(args["nest_budget_gib"])]
        for key in ("isftcflx", "history_interval", "nest_history_interval"):
            if args.get(key) is not None:
                argv[-1:-1] = ["--" + key.replace("_", "-"), str(args[key])]
        rc, out, err = run(argv, cwd=str(TMP))
        if rc != 0:
            shutil.rmtree(folder, ignore_errors=True)
            lines = [l for l in err.splitlines() if l.strip() and "installed wheel" not in l]
            return {"ok": False, "error": lines[-6:]}
        d = json.loads(out)
        import tomllib
        try:
            document = tomllib.loads((folder / "cyclone.toml").read_text(encoding="utf-8"))
            tables = document.get("domain") or []
            restart = float((document.get("experiment") or {}).get("restart_interval_s") or 0)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        m = d.get("memory") or {}
        gib = 1024 ** 3
        doms = [{"grid_id": x["grid_id"], "parent_id": x["parent_id"], "nx": x["nx"], "ny": x["ny"], "nz": x["nz"],
                 "dx_km": x["dx_m"] / 1000, "width_km": round((x["nx"] - 1) * x["dx_m"] / 1000, 1),
                 "height_km": round((x["ny"] - 1) * x["dy_m"] / 1000, 1), "following": x["following"]}
                for x in d.get("domains") or []]
        for row, table in zip(doms, tables):
            row["cu_physics"] = int(table.get("cu_physics", 0))
            row["history_interval_s"] = float(table.get("history_interval_s") or 0)
        need, budget = m.get("peak_envelope_bytes", 0) / gib, m.get("budget_bytes", 0) / gib
        return {"ok": True, "memory": {"need_gib": round(need, 2), "budget_gib": round(budget, 2), "fits": need <= budget},
                "domains": doms, "physics": d.get("profile"), "start_time": d.get("start_time"),
                "restart_interval_s": restart, "run_seconds": float(args["hours"]) * 3600,
                # a following-nest config runs on the experiment route and downloads what its [fetch] names
                "fetch": document.get("fetch"), "chain": "experiment",
                "fit_id": (d.get("fitting") or {}).get("fit_id"), "follow": d.get("follow"),
                "streaming": (d.get("streaming") or {}).get("road"),
                "follow_statics": (d.get("follow_statics") or {}).get("note")}
    return cached("c" + key, go)


# ------------------------------------------------------------------ geometry

def t(s: str) -> datetime:
    return datetime.strptime(s.replace("Z", "").replace(" ", "T")[:16], "%Y-%m-%dT%H:%M")


def km_offset(lat0, lon0, lat, lon):
    dlon = (lon - lon0 + 180) % 360 - 180
    return dlon * 111.32 * math.cos(math.radians(lat0)), (lat - lat0) * 111.32


def from_km(lat0, lon0, x, y):
    lat = lat0 + y / 111.32
    lon = lon0 + x / (111.32 * math.cos(math.radians(lat0)))
    return round(lat, 2), round((lon + 180) % 360 - 180, 2)


def track_at(track, when: datetime):
    pts = [(t(p[3]), p[1], p[0]) for p in track]
    if when <= pts[0][0]:
        return pts[0][1], pts[0][2]
    for (ta, la, lo), (tb, lb, lob) in zip(pts, pts[1:]):
        if ta <= when <= tb:
            f = (when - ta).total_seconds() / max((tb - ta).total_seconds(), 1)
            dlon = (lob - lo + 180) % 360 - 180
            return la + f * (lb - la), (lo + f * dlon + 180) % 360 - 180
    return pts[-1][1], pts[-1][2]


def floor6(when: datetime) -> datetime:
    return when.replace(hour=when.hour // 6 * 6, minute=0, second=0)


def ceil6(when: datetime) -> datetime:
    f = floor6(when)
    return f if f == when else f + timedelta(hours=6)


def cyc(when) -> str:
    if isinstance(when, str):
        when = datetime.fromisoformat(when)
    return when.strftime("%Y-%m-%dT%H")


def source_for(start: datetime, kind: str) -> str:
    c = cyc(start)
    if kind == "tornado" and c >= HRRR_FROM:
        return "hrrr"
    return "era5"


MORRISON = ("morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1",
            "Morrison two-moment microphysics with rimed ice falling as hail, YSU boundary layer, full "
            "RTE+RRTMGP radiation: the one suite matched against WRF, and the engine's own default for this "
            "source.")
PHYSICS = {
    "era5": MORRISON,
    "gfs": MORRISON,
    "hrrr": ("thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1",
             "HRRR's own microphysics family (Thompson), so the storms and ice already in its analysis carry "
             "straight over, with full radiation for the evening and night hours; the engine's own default for "
             "HRRR."),
}
#: Spacing (km) below which the recipe expects the cumulus scheme off: the engine's own
#: convection-permitting bound (gpuwm.domain_wizard.CUMULUS_CONVECTION_PERMITTING_DX_KM).
CUMULUS_OFF_BELOW_KM = 4.0
#: A layout is accepted on a card only when its priced peak is at most this share of the card's
#: budget, so a desktop or a browser on the same card does not turn the first press into a refusal.
HEADROOM_FRACTION = 0.95


def cumulus_words(domains) -> str:
    def spell(values):
        names = [f"{v:g}" for v in sorted(values, reverse=True)]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    on = {d["dx_km"] for d in domains if d.get("cu_physics")}
    off = {d["dx_km"] for d in domains if not d.get("cu_physics")}
    if not on:
        return "No cumulus scheme on any grid: every grid here resolves deep convection itself."
    words = f"Kain-Fritsch on the {spell(on)} km grid"
    if off:
        words += (f" only; off on the {spell(off)} km grid" + ("s" if len(off) > 1 else "")
                  + ", where the storms are resolved and a scheme would convect the same air twice")
    return words + "."


# ------------------------------------------------------------------ tropical cyclones

R_CIRC = 300.0     # km of circulation kept at 3 km around the centre
ROOT_PAD = 700.0   # km of 12 km environment beyond the 3 km grid
ROOT_TRACK_PAD = 400.0  # km of 12 km grid kept around every track point the 3 km grid does not hold
TC_LEAD_H = 36     # hours of spin-up before the peak
TC_AFTER_PEAK_H = 18      # hours kept after the peak
TC_AFTER_LANDFALL_H = 12  # hours kept after the key landfall, so the storm is seen coming ashore
TC_MAX_H = 120     # longest cyclone window; past it the start moves later
KEY_BEFORE_H = 12  # the 1 km grid holds the centre from this long before the key time ...
KEY_AFTER_H = 12   # ... to this long after it
#: km the best-track centre must stay inside the 1 km grid's edge: the eyewall, which is tens of km across,
#: has to be on the 1 km grid, not the centre alone.
EYE_MARGIN_KM = 60.0
#: km of 1 km grid around the key hours' track, largest first; every one is at least EYE_MARGIN_KM.
ONE_KM_MARGINS = (240.0, 180.0, 130.0, 90.0, 60.0)
KEY3_BEFORE_H = 24 # a key-hours 3 km grid starts holding the storm this long before the key time
LATE_LEAD_H = 30   # a late-start fallback begins this long before the key time
LATE_SHORT_LEAD_H = 18  # ... or, for a fast mover on a small card, this long
KEY_RUN_AFTER_H = 6     # a key-hours-only run stops this long after the key time
LATE_ROOT_PAD = 500  # km of 12 km grid beyond a late-start 3 km grid
#: The cyclone surface-flux option (WRF isftcflx, gpuwm domain --isftcflx) every cyclone row sets.
TC_ISFTCFLX = 1
TC_ISFTCFLX_WHY = (
    "Over the sea the surface layer uses Donelan drag, which levels off in strong wind as measured in "
    "hurricanes, with a constant heat and moisture roughness (WRF's tropical cyclone option, isftcflx 1). "
    "The default roughness keeps raising the drag as the wind rises, which holds a strong storm's peak "
    "wind down.")
#: The landfall table (landfall_table.py over the IBTrACS rows the seed was built from).
LANDFALL = json.loads((S / "landfall.json").read_text(encoding="utf-8")) if (S / "landfall.json").is_file() else {}


def _bbox(xy):
    return (min(p[0] for p in xy), max(p[0] for p in xy), min(p[1] for p in xy), max(p[1] for p in xy))


def hourly(track, first: datetime, last: datetime):
    """Best-track centre every hour from first to last, linear between the IBTrACS rows."""
    out, when = [], first
    while when <= last:
        out.append((when, *track_at(track, when)))
        when += timedelta(hours=1)
    return out


def coverage(track, box, start: datetime, end: datetime, key: datetime, margin: float = EYE_MARGIN_KM):
    """Hours before and after ``key`` the centre stays at least ``margin`` km inside ``box`` without a break."""
    lat0, lon0 = box["lat"], box["lon"]
    hw, hh = box["width_km"] / 2 - margin, box["height_km"] / 2 - margin

    def inside(when):
        x, y = km_offset(lat0, lon0, *track_at(track, when))
        return abs(x) <= hw and abs(y) <= hh

    if not inside(key):
        return None
    before = 0
    while key - timedelta(hours=before + 1) >= start and inside(key - timedelta(hours=before + 1)):
        before += 1
    after = 0
    while key + timedelta(hours=after + 1) <= end and inside(key + timedelta(hours=after + 1)):
        after += 1
    return before, after


def tc_plan(ev):
    track = ev["geometry"]["track"]
    peak = t(ev["when"]["peak"])
    first, last = t(track[0][3]), t(track[-1][3])
    land = (LANDFALL.get(ev["id"]) or {}).get("landfall")
    landfall = t(land["time"]) if land else None
    # The key time is the moment the page is about: the landfall when it comes after the peak, else the peak.
    key = landfall if landfall and landfall > peak else peak
    end = ceil6(peak + timedelta(hours=TC_AFTER_PEAK_H))
    if landfall and landfall > peak:
        end = max(end, ceil6(landfall + timedelta(hours=TC_AFTER_LANDFALL_H)))
    end = min(end, floor6(last))
    start = max(floor6(peak - timedelta(hours=TC_LEAD_H)), ceil6(first))
    if (end - start).total_seconds() > TC_MAX_H * 3600:
        start = floor6(end - timedelta(hours=TC_MAX_H))
    hours = int((end - start).total_seconds() // 3600)
    source = source_for(start, "tc")
    klat, klon = track_at(track, key)
    slat, slon = track_at(track, start)

    def xy(points):
        return [km_offset(klat, klon, la, lo) for _, la, lo in points]

    whole = xy(hourly(track, start, end))
    bb = _bbox(whole)
    k1_first, k1_last = max(start, key - timedelta(hours=KEY_BEFORE_H)), min(end, key + timedelta(hours=KEY_AFTER_H))
    k1 = _bbox(xy(hourly(track, k1_first, k1_last)))
    k3 = xy(hourly(track, max(start, key - timedelta(hours=KEY3_BEFORE_H)), end))
    kb3 = _bbox(k3)

    def one_km_box(margin):
        # The key hours' track, with ``margin`` km of 1 km grid around every hourly centre on it.
        cx, cy = (k1[0] + k1[1]) / 2, (k1[2] + k1[3]) / 2
        w = math.ceil(k1[1] - k1[0] + 2 * margin)
        h = math.ceil(k1[3] - k1[2] + 2 * margin)
        clat, clon = from_km(klat, klon, cx, cy)
        return cx, cy, w, h, {"lat": clat, "lon": clon, "width_km": w, "height_km": h}

    def reach(points, cx, cy, w, h, pad):
        # km beyond a w by h box centred at (cx, cy) a grid must reach to hold every point with ``pad`` around it
        return max(max(abs(x - cx) for x, _ in points) + pad - w / 2,
                   max(abs(y - cy) for _, y in points) + pad - h / 2)

    def one_km_rung(margin, circ, holds):
        cx, cy, w, h, box = one_km_box(margin)
        points = whole if holds == "whole" else k3
        b3 = max(math.ceil(reach(points, cx, cy, w, h, circ)), 150)
        # The 12 km grid reaches ROOT_PAD beyond the 3 km grid, and ROOT_TRACK_PAD around any hour the 3 km
        # grid does not hold; both measured from the 1 km box, the way the engine reads buffer_km.
        b12 = max(b3 + int(ROOT_PAD), math.ceil(reach(whole, cx, cy, w, h, ROOT_TRACK_PAD)))
        tag = "3" if holds == "whole" else "3key"
        return {"rung": f"static-12-{tag}-1-m{int(margin)}", "door": "domain", "box": box,
                "intent": {"root_dx_km": 12, "chain": "4,3", "buffer_km": f"{int(b12)},{b3},0"},
                "finest_km": 1.0, "margin_km": margin, "inner_km": max(w, h), "circ_km": circ,
                "three_km_holds": holds, "ring_km": int(ROOT_PAD)}

    rungs = []
    for margin in ONE_KM_MARGINS[:3]:
        rungs.append(one_km_rung(margin, R_CIRC, "whole"))
    for margin in ONE_KM_MARGINS[:3]:
        rungs.append(one_km_rung(margin, R_CIRC, "key"))
    for margin in ONE_KM_MARGINS[3:]:
        rungs.append(one_km_rung(margin, 250.0, "whole"))
        rungs.append(one_km_rung(margin, 250.0, "key"))
    cx, cy = (bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2
    clat, clon = from_km(klat, klon, cx, cy)
    w3 = math.ceil(bb[1] - bb[0] + 2 * R_CIRC)
    h3 = math.ceil(bb[3] - bb[2] + 2 * R_CIRC)
    rungs.append({"rung": "static-12-3-track", "door": "domain",
                  "box": {"lat": clat, "lon": clon, "width_km": w3, "height_km": h3},
                  "intent": {"root_dx_km": 12, "chain": "4", "buffer_km": f"{int(ROOT_PAD)},0"},
                  "finest_km": 3.0, "three_km_holds": "whole", "ring_km": int(ROOT_PAD)})
    disp = [km_offset(slat, slon, la, lo) for _, la, lo in hourly(track, start, end)]
    for grow in (True, False):
        rungs.append({"rung": "following-12-3" + ("" if grow else "-small"), "door": "cyclone-setup",
                      "args": {"source": source, "cycle": cyc(start), "advisory_position": f"{slat:.2f},{slon:.2f}",
                               "hours": hours, "isftcflx": TC_ISFTCFLX}, "grow": grow, "finest_km": 3.0,
                      "disp_km": disp, "three_km_holds": "whole"})
    kx, ky = (kb3[0] + kb3[1]) / 2, (kb3[2] + kb3[3]) / 2
    kclat, kclon = from_km(klat, klon, kx, ky)
    for name, circ in (("static-12-3-key", 250), ("static-12-3-key-small", 200)):
        w = math.ceil(kb3[1] - kb3[0] + 2 * circ)
        h = math.ceil(kb3[3] - kb3[2] + 2 * circ)
        b12 = max(int(ROOT_PAD), math.ceil(reach(whole, kx, ky, w, h, ROOT_TRACK_PAD)))
        rungs.append({"rung": name, "door": "domain",
                      "box": {"lat": kclat, "lon": kclon, "width_km": w, "height_km": h},
                      "intent": {"root_dx_km": 12, "chain": "4", "buffer_km": f"{b12},0"},
                      "finest_km": 3.0, "three_km_holds": "key", "ring_km": int(ROOT_PAD)})

    # A later start: only the hours around the key time a small card can carry. It gives up the earlier
    # hours, and the peak when the key time is a later landfall, and says so.
    def late_window(lead_h, after_h=None, circ=R_CIRC):
        late = max(start, floor6(key - timedelta(hours=lead_h)))
        late_end = end if after_h is None else min(end, ceil6(key + timedelta(hours=after_h)))
        late_hours = int((late_end - late).total_seconds() // 3600)
        lb = _bbox(xy(hourly(track, late, late_end)))
        llat, llon = from_km(klat, klon, (lb[0] + lb[1]) / 2, (lb[2] + lb[3]) / 2)
        return (late, late_end, late_hours, llat, llon,
                math.ceil(lb[1] - lb[0] + 2 * circ), math.ceil(lb[3] - lb[2] + 2 * circ))

    late_3 = []
    for name, lead, after, circ in (("late-12-3", LATE_LEAD_H, None, R_CIRC),
                                    ("late-12-3-short", LATE_SHORT_LEAD_H, None, R_CIRC),
                                    ("key-12-3", LATE_SHORT_LEAD_H, KEY_RUN_AFTER_H, 200.0),
                                    ("key-12-3-small", LATE_SHORT_LEAD_H, KEY_RUN_AFTER_H, 150.0)):
        late, late_end, late_hours, llat, llon, lw, lh = late_window(lead, after, circ)
        if (late > start or late_end < end) and late_hours >= 12:
            late_3.append({"rung": name, "door": "domain", "start": late, "hours": late_hours, "lead_h": lead,
                           "after_h": int((late_end - key).total_seconds() // 3600), "circ_km": circ,
                           "box": {"lat": llat, "lon": llon, "width_km": lw, "height_km": lh},
                           "intent": {"root_dx_km": 12, "chain": "4", "buffer_km": f"{LATE_ROOT_PAD},0"},
                           "finest_km": 3.0, "three_km_holds": "whole", "ring_km": LATE_ROOT_PAD})
    rungs.extend(late_3)
    for name, pad in (("single-12", ROOT_PAD), ("single-12-small", ROOT_TRACK_PAD)):
        rungs.append({"rung": name, "door": "domain",
                      "box": {"lat": clat, "lon": clon, "width_km": w3 + 2 * pad, "height_km": h3 + 2 * pad},
                      "intent": {"root_dx_km": 12}, "finest_km": 12.0})
    late, late_end, late_hours, llat, llon, lw, lh = late_window(LATE_LEAD_H)
    if late > start:
        for name, pad in (("late-12", ROOT_TRACK_PAD), ("late-12-small", 250)):
            rungs.append({"rung": name, "door": "domain", "start": late, "hours": late_hours, "lead_h": LATE_LEAD_H,
                          "box": {"lat": llat, "lon": llon, "width_km": lw + 2 * pad, "height_km": lh + 2 * pad},
                          "intent": {"root_dx_km": 12}, "finest_km": 12.0})
    return {"start": start, "hours": hours, "end": end, "source": source, "rungs": rungs,
            "peak_time": peak, "key_time": key, "key_is": "landfall" if key != peak else "peak",
            "landfall": land, "key": (round(klat, 2), round(klon, 2)), "start_pos": (round(slat, 2), round(slon, 2)),
            "track_bbox_km": [round(v) for v in bb], "track": track}


# ------------------------------------------------------------------ tornadoes

UPSTREAM = 120.0  # km the parent storm is followed back along its own motion before the tornado
#: km of 12 km grid beyond the 3 km grid when the analysis is a coarse reanalysis (ERA5, about 31 km):
#: a 12 km ring carries the synoptic flow so the 3 km grid is not driven straight from 31 km boundaries.
SYNOPTIC_PAD = 500
#: The thinnest 12 km ring an ERA5-driven grid may have, read back from the fitted grids.
RING_MIN_KM = 400


def tornado_plan(ev):
    (lo0, la0), (lo1, la1) = ev["geometry"]["path"]
    tstart, tend = t(ev["when"]["start"]), t(ev["when"]["end"])
    start = floor6(tstart - timedelta(hours=6))
    hours = max(12, math.ceil((tend + timedelta(hours=3) - start).total_seconds() / 3600))
    source = source_for(start, "tornado")
    mlat, mlon = (la0 + la1) / 2, lo0 + ((lo1 - lo0 + 180) % 360 - 180) / 2
    a = km_offset(mlat, mlon, la0, lo0)
    b = km_offset(mlat, mlon, la1, lo1)
    pts = [a, b]
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    if L > 5:
        ux, uy = (a[0] - b[0]) / L, (a[1] - b[1]) / L
        pts.append((a[0] + UPSTREAM * ux, a[1] + UPSTREAM * uy))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bb = (min(xs), max(xs), min(ys), max(ys))
    cx, cy = (bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2
    clat, clon = from_km(mlat, mlon, cx, cy)
    # HRRR is itself a 3 km analysis, so a 3 km root takes its boundaries directly; a reanalysis
    # gets a 12 km outer grid first.
    coarse = source != "hrrr"
    rungs = []
    for margin, floor, outer in ((250, 600, 400), (200, 480, 350), (150, 400, 300), (110, 320, 250),
                                 (80, 240, 220), (60, 200, 180)):
        w = max(math.ceil(bb[1] - bb[0] + 2 * margin), floor)
        h = max(math.ceil(bb[3] - bb[2] + 2 * margin), floor)
        # buffer_km is measured from the box: the 12 km grid reaches SYNOPTIC_PAD beyond the 3 km grid.
        intent = ({"root_dx_km": 12, "chain": "4,3", "buffer_km": f"{outer + SYNOPTIC_PAD},{outer},0"} if coarse else
                  {"root_dx_km": 3, "chain": "3", "buffer_km": f"{outer},0"})
        rungs.append({"rung": f"nest-{'12-' if coarse else ''}3-1-{w}x{h}", "door": "domain",
                      "box": {"lat": clat, "lon": clon, "width_km": w, "height_km": h},
                      "intent": intent, "finest_km": 1.0, "inner_w": w, "inner_h": h, "outer_pad": outer,
                      "margin": margin, "synoptic_pad": SYNOPTIC_PAD if coarse else None,
                      "ring_km": SYNOPTIC_PAD if coarse else None})
    singles = ((3, 1100, 250), (3, 800, 200), (3, 650, 150), (4, 800, 200), (4, 650, 150))
    if coarse:
        # a 3 km grid inside the 12 km ring, at every size, before any grid driven straight from 31 km
        for dx, size, pad in singles[:3]:
            w = max(size, math.ceil(bb[1] - bb[0] + 2 * pad))
            h = max(size, math.ceil(bb[3] - bb[2] + 2 * pad))
            rungs.append({"rung": f"nest-12-3-{w}x{h}", "door": "domain",
                          "box": {"lat": clat, "lon": clon, "width_km": w, "height_km": h},
                          "intent": {"root_dx_km": 12, "chain": "4", "buffer_km": f"{SYNOPTIC_PAD},0"},
                          "finest_km": 3.0, "inner_w": w, "inner_h": h, "synoptic_pad": SYNOPTIC_PAD,
                          "ring_km": SYNOPTIC_PAD})
    for dx, size, pad in singles:
        w = max(size, math.ceil(bb[1] - bb[0] + 2 * pad))
        h = max(size, math.ceil(bb[3] - bb[2] + 2 * pad))
        rungs.append({"rung": f"single-{dx}-{w}x{h}", "door": "domain",
                      "box": {"lat": clat, "lon": clon, "width_km": w, "height_km": h},
                      "intent": {"root_dx_km": dx}, "finest_km": float(dx), "inner_w": w, "inner_h": h})
    return {"start": start, "hours": hours, "end": start + timedelta(hours=hours), "source": source, "rungs": rungs,
            "tornado_start": tstart, "tornado_end": tend,
            "track_mid": (round(mlat, 3), round(mlon, 3)), "path_km": round(L, 1), "nest_centre": (clat, clon)}


# ------------------------------------------------------------------ disk

#: Disk a row may need, per card size in GB: the download, the preparation, the history files, the one
#: checkpoint set run-plan keeps (and the one written before it goes) and the pictures. A first-press button
#: that needs more than this stops partway on a typical disk; the page still compares the row's own number
#: with the free disk before it starts.
DISK_BUDGET_GIB = {8: 40, 12: 50, 16: 60, 24: 80, 32: 100}
#: Output intervals a row may take, densest first, as (root seconds, nest seconds). A tornado's nests keep
#: at most half-hourly output, since a supercell changes in minutes; a cyclone's keep at most three-hourly.
TORNADO_INTERVALS = ((3600, 900), (3600, 1200), (3600, 1800))
TC_INTERVALS = ((3600, 3600), (10800, 3600), (3600, 7200), (10800, 7200), (10800, 10800))
#: Checkpoint sets run-plan keeps by default (gpuwm.resume.DEFAULT_KEEP_CHECKPOINTS).
KEEP_CHECKPOINTS = 1


def fit_profile(fit) -> str:
    """The physics suite a fit runs: cyclone-setup names it, and run-plan's emitted config opens its
    "# PHYSICS:" line with it (a shipped row keeps it as physics.profile)."""
    physics = fit.get("physics")
    if isinstance(physics, dict):
        physics = physics.get("profile")
    profile = str(physics or "").split(":")[0].strip()
    if not profile:
        # Without the suite there is no writer inventory to price: a grid size alone priced a moist run's
        # history as a dry one, under half of what it writes, and picked output intervals that overrun the
        # card's disk budget.
        raise ValueError("a recipe fit names no physics suite, so its history cannot be priced")
    return profile


def domain_run(fit, d, run_seconds):
    """The run settings the engine's emitted config gives one grid of a fit, which its history is priced from.

    The suite's [shared] block (gpuwm.domain_wizard.shared_physics, the one the engine emits for both doors)
    with the grid's own cumulus switch as the fit read it back. The step is a placeholder: it changes neither
    the history inventory nor the frame count of a run that writes from its start to its end.
    """
    from dataclasses import fields
    from gpuwm.config import RunConfig
    from gpuwm.domain_wizard import shared_physics
    shared = shared_physics(fit_profile(fit))
    names = {row.name for row in fields(RunConfig)}
    settings = {key: value for key, value in shared.items() if key in names}
    dx = float(d.get("dx_km") or 3.0) * 1000.0
    if d.get("cu_physics") is not None:
        settings["cu_physics"] = int(d["cu_physics"])
    nz = int(d.get("nz") or settings.get("nz") or 49)
    if settings.get("eta_levels") is not None and len(settings["eta_levels"]) != nz + 1:
        # The emitted ladder belongs to the suite's own level count; a grid with another
        # count carries its own, and the ladder does not change the history inventory.
        settings.pop("eta_levels")
    settings.update(nx=int(d["nx"]), ny=int(d["ny"]), nz=nz,
                    dx=dx, dy=dx, dt=max(1.0, 6.0 * dx / 1000.0), run_seconds=float(run_seconds))
    return RunConfig(**settings)


def projection(fit, run_seconds, root_s, nest_s) -> dict:
    """The engine's own projection (gpuwm.disk_budget) for a fit's grids at these output intervals.

    The download and the preparation are counted from the fit's own [fetch] request and chain, as the
    engine counts them; they do not change with the output intervals. The history is the writer's
    inventory for each grid's own run settings (:func:`domain_run`). The pictures are counted too: the
    page starts every recipe with the standard picture set (an intent plan draws it by default, and the
    storm-following layout's plan names it), so every recipe run draws them.
    """
    from types import SimpleNamespace
    from gpuwm import disk_budget
    domains = fit["domains"]
    exp = SimpleNamespace(run_seconds=float(run_seconds), restart_interval_s=float(fit.get("restart_interval_s") or 0),
                          domains=[
        SimpleNamespace(grid_id=i + 1, history_interval_s=float(root_s if i == 0 else nest_s),
                        run=domain_run(fit, d, run_seconds))
        for i, d in enumerate(domains)])
    return disk_budget.projected_run_bytes(exp, keep_checkpoints=KEEP_CHECKPOINTS, fetch=fit.get("fetch"),
                                           chain=fit.get("chain"), render=True)


def projected_disk_gib(fit, run_seconds, root_s, nest_s) -> float:
    return projection(fit, run_seconds, root_s, nest_s)["total_bytes"] / 1024 ** 3


def pick_intervals(fit, run_seconds, card, kind):
    """The densest allowed output intervals whose projected disk fits the card's budget, or None."""
    ladder = TC_INTERVALS if kind == "tc" else TORNADO_INTERVALS
    if len(fit["domains"]) == 1:
        ladder = tuple(dict.fromkeys((root, root) for root, _ in ladder))
    for root_s, nest_s in ladder:
        gib = projected_disk_gib(fit, run_seconds, root_s, nest_s)
        if gib <= DISK_BUDGET_GIB[card]:
            return root_s, nest_s, round(gib, 1)
    root_s, nest_s = ladder[-1]
    return None, None, round(projected_disk_gib(fit, run_seconds, root_s, nest_s), 1)


# ------------------------------------------------------------------ choose per card

def intent_for(rung, plan, intervals=None):
    # No physics_profile: naming the suite makes the engine emit it verbatim, cumulus included, even on a
    # 3 km root. Left out, the engine derives the same suite and switches cumulus off on every grid finer
    # than its convection-permitting bound.
    intent = dict(rung["intent"], source=plan["source"], cycle=cyc(rung.get("start") or plan["start"]),
                  hours=rung.get("hours") or plan["hours"])
    if plan["source"] == "era5":
        intent["era5_provider"] = "arco"  # Google's public ERA5 archive: no key needed
    if "peak_time" in plan:
        intent["isftcflx"] = TC_ISFTCFLX
    if intervals is not None:
        intent["history_interval_s"] = intervals[0]
        if intent.get("chain"):
            intent["nest_history_interval_s"] = intervals[1]
    return intent


def ring_km(domains):
    """km from each nested grid's edge to its parent's edge, on the thinnest side."""
    rings = []
    by_id = {d["grid_id"]: d for d in domains}
    for d in domains:
        parent = by_id.get(d.get("parent_id"))
        if parent is None or d.get("i_parent_start") is None:
            continue
        dxp, dxc = parent["dx_km"], d["dx_km"]
        west = (d["i_parent_start"] - 1) * dxp
        south = (d["j_parent_start"] - 1) * dxp
        east = (parent["nx"] - 1) * dxp - west - (d["nx"] - 1) * dxc
        north = (parent["ny"] - 1) * dxp - south - (d["ny"] - 1) * dxc
        rings.append({"grid_id": d["grid_id"], "parent_id": parent["grid_id"],
                      "km": round(min(west, east, south, north))})
    return rings


def _self_checks(res, card, rung, plan):
    """Refuse a fit this recipe would describe wrongly, or one that leaves no headroom on the card."""
    if not res.get("ok"):
        return res
    doubled = [d["dx_km"] for d in res.get("domains", [])
               if d["dx_km"] < CUMULUS_OFF_BELOW_KM and d.get("cu_physics")]
    if doubled:
        return dict(res, ok=False, error=[f"cumulus is on at {doubled} km, below the convection-permitting bound"])
    mem = res.get("memory") or {}
    if mem.get("need_gib") and mem.get("budget_gib") and mem["need_gib"] > HEADROOM_FRACTION * mem["budget_gib"]:
        return dict(res, ok=False, error=[
            f"priced at {mem['need_gib']:g} GiB of a {mem['budget_gib']:g} GiB budget: under "
            f"{100 - 100 * HEADROOM_FRACTION:g}% headroom, so a desktop on the same card could refuse it"])
    if rung["door"] == "domain" and plan["source"] == "era5":
        rings = ring_km(res["domains"])
        res = dict(res, rings=rings)
        outer = [r for r in rings if r["parent_id"] == res["domains"][0]["grid_id"]]
        if outer and res["domains"][0]["dx_km"] >= 12 and outer[0]["km"] < RING_MIN_KM:
            return dict(res, ok=False, error=[
                f"the 12 km grid reaches only {outer[0]['km']} km beyond the grid inside it, under the "
                f"{RING_MIN_KM} km that keeps that grid off the 31 km boundaries"])
    if rung.get("finest_km") == 1.0 and "peak_time" in plan and rung["door"] == "domain":
        start = rung.get("start") or plan["start"]
        end = start + timedelta(hours=rung.get("hours") or plan["hours"])
        held = coverage(plan["track"], rung["box"], start, end, plan["key_time"])
        need_before = int(min(KEY_BEFORE_H, (plan["key_time"] - start).total_seconds() // 3600))
        need_after = int(min(KEY_AFTER_H, (end - plan["key_time"]).total_seconds() // 3600))
        res = dict(res, key_cover_h=held)
        if held is None or held[0] < need_before or held[1] < need_after:
            return dict(res, ok=False, error=[
                f"the 1 km box holds the centre {EYE_MARGIN_KM:g} km inside its edge only "
                f"{held and held[0]} h before and {held and held[1]} h after the key time, under the "
                f"{need_before} and {need_after} h this rung promises"])
    return res


def check(rung, plan, card):
    kind = "tc" if "peak_time" in plan else "tornado"
    if rung["door"] == "cyclone-setup":
        args = dict(rung["args"], nest_budget_gib=card if rung["grow"] else None)
        res = fit_cyclone(args, card)
        if not res.get("ok"):
            return res
        parent, nest = res["domains"][0], res["domains"][1]
        room_x = parent["width_km"] / 2 - nest["width_km"] / 2 - 120
        room_y = parent["height_km"] / 2 - nest["height_km"] / 2 - 120
        far_x = max(abs(x) for x, _ in rung["disp_km"])
        far_y = max(abs(y) for _, y in rung["disp_km"])
        room = {"room_km": [round(room_x), round(room_y)], "moves_km": [round(far_x), round(far_y)]}
        if far_x > room_x or far_y > room_y:
            return dict(res, ok=False, **room, error=[
                f"the storm moves {round(far_x)} km east-west and {round(far_y)} km north-south in the window; "
                f"the following nest has room for {round(room_x)} and {round(room_y)} km inside its fixed "
                "12 km parent"])
        root_s, nest_s, gib = pick_intervals(res, res["run_seconds"], card, kind)
        if root_s is None:
            return dict(res, ok=False, disk_gib=gib, error=[
                f"writes {gib:g} GiB even at the sparsest allowed output, over this card's "
                f"{DISK_BUDGET_GIB[card]} GiB disk budget"])
        args.update(history_interval=root_s, nest_history_interval=nest_s)
        res = _self_checks(fit_cyclone(args, card), card, rung, plan)
        if not res.get("ok"):
            return res
        return dict(res, **room, intervals=[root_s, nest_s], args=args,
                    disk_gib=round(projected_disk_gib(res, res["run_seconds"], root_s, nest_s), 1))
    res = _self_checks(fit_domain(intent_for(rung, plan), rung["box"], card), card, rung, plan)
    if not res.get("ok"):
        return res
    run_s = float(rung.get("hours") or plan["hours"]) * 3600
    root_s, nest_s, gib = pick_intervals(res, run_s, card, kind)
    if root_s is None:
        return dict(res, ok=False, disk_gib=gib, error=[
            f"writes {gib:g} GiB even at the sparsest allowed output, over this card's "
            f"{DISK_BUDGET_GIB[card]} GiB disk budget"])
    final = _self_checks(fit_domain(intent_for(rung, plan, (root_s, nest_s)), rung["box"], card), card, rung, plan)
    if not final.get("ok"):
        return final
    engine_gib = (final.get("disk") or {}).get("total_bytes", 0) / 1024 ** 3
    if abs(engine_gib - gib) > 0.2:
        return dict(final, ok=False, error=[
            f"the engine projects {engine_gib:.1f} GiB where the designer projected {gib:g} GiB for the same grids"])
    return dict(final, intervals=[root_s, nest_s], disk_gib=round(engine_gib, 1))


def pick_error(lines):
    for line in lines:
        if ("does not fit" in line or "EXCEEDS" in line or "storm moves" in line or "REFUS" in line
                or "headroom" in line or "cumulus is on" in line or "disk budget" in line
                or "reaches only" in line or "holds the centre" in line):
            return line[:600]
    return lines[-1][:600]


def choose(ev):
    plan = tc_plan(ev) if ev["type"] == "tropical-cyclone" else tornado_plan(ev)
    cards = {}
    for card in CARDS:
        tried = []
        for i, rung in enumerate(plan["rungs"]):
            res = check(rung, plan, card)
            tried.append({"rung": rung["rung"], "ok": res["ok"], "memory": res.get("memory"),
                          "disk_gib": res.get("disk_gib"),
                          "error": None if res["ok"] else pick_error(res.get("error") or [""])})
            if res["ok"]:
                cards[card] = {"rung_index": i, "rung": rung, "fit": res, "tried": tried}
                break
        else:
            cards[card] = {"rung_index": None, "tried": tried}
    return plan, cards


if __name__ == "__main__":
    seed = json.load(open(Path(W) / "gpuwm/gui/seed/wiki-seed.json", encoding="utf-8"))
    only = sys.argv[1:]
    events = [e for e in seed["events"] if not only or e["id"] in only]
    with ThreadPoolExecutor(max_workers=int(os.environ.get("RECIPE_WORKERS", "6"))) as pool:
        results = list(pool.map(choose, events))
    prev = WORK / "design-result.json"
    out = json.loads(prev.read_text(encoding="utf-8")) if prev.is_file() else {}
    for ev, (plan, cards) in zip(events, results):
        p = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in plan.items() if k != "track"}
        out[ev["id"]] = {"plan": p, "cards": cards}
        print(ev["id"], plan["source"], cyc(plan["start"]), plan["hours"])
        for c, v in cards.items():
            f = v.get("fit") or {}
            print("   ", c, v.get("rung", {}).get("rung"), (f.get("memory") or {}).get("need_gib"), f.get("disk_gib"),
                  f.get("intervals"), [(d["dx_km"], d["nx"], d["ny"]) for d in f.get("domains", [])],
                  [x["rung"] + ":" + str((x.get("memory") or {}).get("need_gib")) for x in v["tried"] if not x["ok"]])
    prev.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
