#!/usr/bin/env python3
"""CONUS observations for a first-look WOOF DA cycle, on the deck's own route.

The recent regional deck (tools/da_recent_observations.py) prepares its
observations for eleven named radars and seven state ASOS networks.  This
tool keeps that route and its rules unchanged and widens only the roster:

* radar: every NEXRAD site whose 250 km disc touches the analysis grid
  (discovered from the decoder's own site table, never a hardcoded list),
  the same causal volume choice (newest volume that ENDS at or before the
  analysis, at most 900 s stale, archive LastModified not after it), the
  same decode (REF, VEL, RHO, censor flags), the same region-global
  dealias, dual-pol QC and clear-air-from-censor superob, merged into one
  ``gpuwm-obs.radar-grid`` file per cycle on the grid the analysis runs.
  Sites run in parallel processes; one site's bad hour is a refusal row,
  never the cycle's failure.
* surface: METAR through ``rw_asos`` (stations, fetch, decode, verify) for
  every ASOS network whose extent overlaps the grid.
* count: the observation counts the analysis would assimilate, after the
  deck controller's own thinning (radial velocity 2 cells, reflectivity 2,
  clear air 4) applied with the analysis' own thinning functions.
* roster: the da-speed obs-ops neighbour roster
  (gpuwm.da.neighbor_index.build_neighbor_index) built on a card from the
  real thinned observation masks, at the prepared grid or at a refined
  (3 km) copy of it, with the localisation the deck uses or a wider one.

Observations are archive replays: object LastModified is metadata, not a
proof of first publication.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UTC = dt.timezone.utc
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
RADAR_BUCKET = "unidata-nexrad-level2"
RANGE_KM = 250.0
MAX_ELEVATION_DEG = 20.0
MAX_AGE_S = 900
#: The deck controller's defaults (tools/da_cycle_prepared.py --thin-cells,
#: --z-thin-cells, --z0-thin-cells), which the deck's argv leaves in force.
DECK_THINNING = {"velocity": 2, "reflectivity": 2, "clear_air": 4}


def utc(value):
    stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def iso(stamp):
    return stamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def object_time(key):
    match = re.search(r"(\d{8})[-_](\d{6})", key)
    if not match:
        return None
    return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def list_objects(bucket, prefix, *, timeout=60):
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": 1000}
        if token:
            query["continuation-token"] = token
        url = f"https://{bucket}.s3.amazonaws.com/?" + urllib.parse.urlencode(query)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    tree = ET.fromstring(response.read())
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        for row in tree.findall("s:Contents", NS):
            yield {"bucket": bucket, "key": row.findtext("s:Key", namespaces=NS),
                   "bytes": int(row.findtext("s:Size", namespaces=NS)),
                   "etag": row.findtext("s:ETag", namespaces=NS),
                   "last_modified": row.findtext("s:LastModified", namespaces=NS)}
        token = tree.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            break


def download(row, root, *, timeout=120, cache=None):
    """The object under ``root``; an identical-size copy under the read-only
    ``cache`` (another lane's fetch tree, same bucket/key layout) is used in
    place and never written to."""
    rel = Path(row["bucket"]) / Path(*row["key"].split("/"))
    if cache is not None:
        cached = Path(cache) / rel
        if cached.is_file() and cached.stat().st_size == row["bytes"]:
            return cached, 0.0, False
    path = Path(root) / rel
    if path.is_file() and path.stat().st_size == row["bytes"]:
        return path, 0.0, False
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://{row['bucket']}.s3.amazonaws.com/" + urllib.parse.quote(row["key"])
    started = time.perf_counter()
    for attempt in range(4):
        try:
            tmp = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as sink:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    sink.write(block)
            if tmp.stat().st_size != row["bytes"]:
                raise OSError("short download")
            tmp.replace(path)
            return path, time.perf_counter() - started, True
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# radar
# ---------------------------------------------------------------------------

_GRID = None


def _worker_init(grid_path):
    # Two threads per site process: the box is shared, sites are the
    # parallel axis.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    global _GRID
    from gpuwm.obs.target_grid import TargetGrid
    _GRID = TargetGrid.from_wrfout(Path(grid_path))


def _deck_params():
    from gpuwm.obs.cc_qc import CcQcParams
    from gpuwm.obs.dealias import DealiasParams
    from gpuwm.obs.superob import SuperobParams
    params = SuperobParams(max_range_km=RANGE_KM, max_elevation_deg=MAX_ELEVATION_DEG,
                           dealias=DealiasParams(engine="region-global"), cc_qc=CcQcParams())
    params.validate()
    return params


def site_cycle(task):
    """One site at one analysis: the deck's causal choice, decode and superob.

    Never raises: returns (contribution or None, record).
    """
    from tools.da_recent_observations import (causal_object, causal_volume,
                                              validate_contribution)
    from gpuwm.obs.nexrad import find_nexrad_bin, run_decode, run_verify
    from gpuwm.obs.superob import superob_volume
    from gpuwm.obs.sweeps import read_sweep_pack
    site, analysis, candidates, raw_root, work, cache, *rest = task
    #: "last-modified" (the deck's rule: the archive object's LastModified
    #: must not be after the analysis) or "measured-end" (an archive
    #: replay: the volume's own measured end clock, checked by
    #: causal_volume below, is the causal gate; LastModified on a years-old
    #: archive object is its upload or re-upload time, and refused every
    #: 2024-05-21 volume for a 2024-05-21 19Z analysis).
    publication_gate = rest[0] if rest else "last-modified"
    analysis = utc(analysis)
    record = {"site": site, "analysis_time": iso(analysis), "refusals": [],
              "seconds": {"fetch": 0.0, "decode": 0.0, "superob": 0.0}, "bytes_fetched": 0}
    params = _deck_params()
    binary = find_nexrad_bin()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for row in candidates:
        if publication_gate == "last-modified":
            try:
                causal_object(row, analysis)
            except ValueError as error:
                record["refusals"].append({"key": row["key"], "reason": str(error)})
                continue
        try:
            t0 = time.perf_counter()
            path, _, fetched = download(row, raw_root, cache=cache)
            record["seconds"]["fetch"] += time.perf_counter() - t0
            if fetched:
                record["bytes_fetched"] += row["bytes"]
            t0 = time.perf_counter()
            pack = work / (Path(row["key"]).name + ".pack")
            run_decode(binary, volume=path, out=pack, moments=("REF", "VEL", "RHO"),
                       max_range_km=RANGE_KM, max_elevation_deg=MAX_ELEVATION_DEG,
                       censor_flags=True)
            verified = run_verify(binary, pack=pack)
            if verified.get("status") != "PASS":
                raise ValueError("native radar pack integrity did not PASS")
            volume = read_sweep_pack(pack)
            record["seconds"]["decode"] += time.perf_counter() - t0
            end = causal_volume(volume, analysis, MAX_AGE_S)
            t0 = time.perf_counter()
            contribution = superob_volume(volume, _GRID, params=params, clear_air_from_censor=True)
            record["seconds"]["superob"] += time.perf_counter() - t0
            validate_contribution(contribution)
        except Exception as error:  # noqa: BLE001 - a site's failure is a row
            record["refusals"].append({"key": row["key"], "reason": f"{type(error).__name__}: {error}"})
            try:
                pack.unlink()
            except (OSError, NameError, UnboundLocalError):
                pass
            continue
        record.update(status="contributed", key=row["key"], volume_sha256=sha(path),
                      pack_sha256=sha(pack), measured_end_utc=iso(end),
                      last_modified=row["last_modified"], dealias=contribution.dealias.get("totals"),
                      cc_qc=bool(contribution.cc_qc),
                      cells={"z": int((contribution.z_count > 0).sum()),
                             "vr": int((contribution.vr_count > 0).sum())})
        pack.unlink()  # regenerable; its hash is in the record
        return contribution, record
    record["status"] = "no-causal-volume" if candidates else "no-candidates"
    return None, record


def radar(args):
    from gpuwm.obs.coverage import read_site_table, sites_covering
    from gpuwm.obs.dealias import engine_unavailable_reason
    from gpuwm.obs.nexrad import find_nexrad_bin
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import merge_contributions
    from gpuwm.obs.target_grid import TargetGrid
    t_all = time.perf_counter()
    binary = find_nexrad_bin()
    reason = engine_unavailable_reason("region-global")
    if binary is None or reason is not None:
        raise SystemExit("required native radar/dealias door unavailable: " + str(reason))
    # The superob and merge run in rw-superob; the numpy reference runs only
    # when GPUWM_SUPEROB_PYTHON=1 says so, and the slots receipt records it.
    from gpuwm.obs import superob_bridge
    superob_engine = ("python-reference (GPUWM_SUPEROB_PYTHON=1)"
                      if superob_bridge.python_reference_requested() else "rust:rw-superob")
    if not superob_bridge.python_reference_requested():
        why = superob_bridge.unavailable_reason()
        if why is not None:
            raise SystemExit("required native superob unavailable: " + why)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    grid_path = Path(args.grid_wrfout).resolve()
    grid = TargetGrid.from_wrfout(grid_path)
    analyses = [utc(c) for c in args.cycles]
    table = read_site_table(binary)
    found = sites_covering(grid, table, max_range_km=RANGE_KM, min_coverage_fraction=0.0)
    sites = [entry.site.id for entry in found]
    if args.limit_sites:
        sites = sites[:args.limit_sites]
    discovery = {"table_sites": len(table), "covering_sites": len(sites),
                 "range_km": RANGE_KM, "sites": [entry.to_payload() for entry in found
                                                 if entry.site.id in set(sites)]}
    save(out / "radar-discovery.json", discovery)
    print(f"radar: {len(sites)} of {len(table)} sites reach the grid", flush=True)

    # Inventory: one listing per site-day, in parallel.
    t0 = time.perf_counter()
    days = sorted({(a - dt.timedelta(seconds=MAX_AGE_S)).date() for a in analyses} | {a.date() for a in analyses})
    listing = {}
    with cf.ThreadPoolExecutor(max_workers=48) as pool:
        jobs = {pool.submit(lambda s, d: list(list_objects(RADAR_BUCKET, f"{d:%Y/%m/%d}/{s}/")), s, d): (s, d)
                for s in sites for d in days}
        for job in cf.as_completed(jobs):
            s, d = jobs[job]
            try:
                listing.setdefault(s, []).extend(job.result())
            except OSError as error:
                listing.setdefault(s, [])
                print(f"listing {s} {d}: {error}", flush=True)
    inventory_seconds = time.perf_counter() - t0
    tasks = []
    for index, analysis in enumerate(analyses):
        for s in sites:
            candidates = [row for row in listing.get(s, []) if object_time(row["key"]) is not None
                          and analysis - dt.timedelta(seconds=MAX_AGE_S) <= object_time(row["key"]) <= analysis
                          and not row["key"].endswith("_MDM")]
            candidates.sort(key=lambda row: object_time(row["key"]), reverse=True)
            tasks.append((index, (s, iso(analysis), candidates, str(out / "raw"),
                                  str(out / f"slot-{index:02d}" / "packs" / s), args.raw_cache,
                                  getattr(args, "publication_gate", "last-modified"))))
    save(out / "radar-inventory.json", {"bucket": RADAR_BUCKET, "days": [str(d) for d in days],
                                        "objects_listed": sum(len(v) for v in listing.values()),
                                        "tasks": [{"slot": i, "site": t[0], "analysis": t[1],
                                                   "candidates": [r["key"] for r in t[2]]} for i, t in tasks]})
    print(f"radar: inventory {inventory_seconds:.1f} s, {len(tasks)} site-cycles", flush=True)

    t0 = time.perf_counter()
    results = {i: [] for i in range(len(analyses))}
    # spawn, not fork: the listing threads above leave locks a forked
    # child can inherit held (measured: every forked worker parked on a
    # futex with zero CPU).
    import multiprocessing
    with cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                                initargs=(str(grid_path),),
                                mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {pool.submit(site_cycle, task): (index, task[0]) for index, task in tasks}
        done = 0
        for job in cf.as_completed(jobs):
            index, s = jobs[job]
            try:
                contribution, record = job.result()
            except Exception as error:  # noqa: BLE001
                contribution, record = None, {"site": s, "status": "worker-failed", "reason": repr(error)}
            results[index].append((s, contribution, record))
            done += 1
            if done % 50 == 0:
                print(f"radar: {done}/{len(tasks)} site-cycles, {time.perf_counter() - t0:.0f} s", flush=True)
    ingest_seconds = time.perf_counter() - t0

    params = _deck_params()
    slots = []
    for index, analysis in enumerate(analyses):
        t0 = time.perf_counter()
        rows = sorted(results[index], key=lambda r: sites.index(r[0]))
        contributions = [c for _, c, _ in rows if c is not None]
        records = [r for _, _, r in rows]
        work = out / f"slot-{index:02d}"
        work.mkdir(parents=True, exist_ok=True)
        save(work / "radar-attempts.json", {"sites": records})
        if len(contributions) < 2:
            raise SystemExit(f"slot {index}: only {len(contributions)} radars contributed")
        merged = merge_contributions(contributions, grid, params=params)
        sources = [{k: r.get(k) for k in ("site", "key", "volume_sha256", "pack_sha256",
                                         "measured_end_utc", "last_modified")}
                   for r in records if r.get("status") == "contributed"]
        refusals = [{"site": r["site"], "status": r.get("status"), "refusals": r.get("refusals")}
                    for r in records if r.get("status") != "contributed"]
        context = {"builder": "tools/conus_da_observations.py", "requested_valid_time": iso(analysis),
                   "sources": sources, "refusals": refusals,
                   "publication_policy": "archive-replay; LastModified is object metadata, not first publication proof",
                   "publication_gate": getattr(args, "publication_gate", "last-modified"),
                   "causal_measurement_cutoff_utc": iso(analysis)}
        receipt = write_radar_grid(work / "radar.nc", merged, grid, valid_time=iso(analysis),
                                   params=params, provenance=context, overwrite=True)
        merge_seconds = time.perf_counter() - t0
        save(work / "radar-receipt.json", receipt)
        slot = {"slot": index, "analysis_time": iso(analysis), "obs": str(work / "radar.nc"),
                "obs_sha256": receipt.get("sha256"), "bytes": receipt.get("bytes"),
                "radars_contributing": len(contributions), "radars_reaching": len(sites),
                "merge_write_seconds": round(merge_seconds, 2),
                "site_seconds_sum": {k: round(sum(r.get("seconds", {}).get(k, 0.0) for r in records), 1)
                                     for k in ("fetch", "decode", "superob")},
                "bytes_fetched": sum(r.get("bytes_fetched", 0) for r in records)}
        slots.append(slot)
        print(f"radar slot {index}: {len(contributions)}/{len(sites)} radars, merge+write {merge_seconds:.1f} s", flush=True)
    doc = {"schema": "conus-da.radar-slots.v1", "grid_wrfout": str(grid_path),
           "grid_wrfout_sha256": sha(grid_path), "grid_identity_sha256": grid.identity_sha256(),
           "superob_parameters": params.to_payload(), "superob_engine": superob_engine,
           "workers": args.workers,
           "seconds": {"inventory": round(inventory_seconds, 1), "ingest": round(ingest_seconds, 1),
                       "total": round(time.perf_counter() - t_all, 1)},
           "native_radar_binary_sha256": sha(binary), "slots": slots}
    save(out / "radar-slots.json", doc)
    print(json.dumps(doc["seconds"]), flush=True)


# ---------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------

def run_json(command, log):
    import subprocess
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, timeout=1800)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    Path(log).write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"native command failed {result.returncode}; see {log}")
    return json.loads(result.stdout), time.perf_counter() - started


def grid_bbox(grid):
    import numpy as np
    lat = np.asarray(grid.lat)
    lon = np.asarray(grid.lon)
    return [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]


def surface(args):
    from tools.da_recent_observations import combine_surfaces, validate_surface
    from gpuwm.obs.frontdoor import ASOS
    from gpuwm.obs.surface_networks import networks_for_domain
    from gpuwm.obs.target_grid import TargetGrid
    t_all = time.perf_counter()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    grid = TargetGrid.from_wrfout(Path(args.grid_wrfout))
    bbox = grid_bbox(grid)
    networks = list(networks_for_domain(*bbox))
    native = ASOS.find()
    if native is None:
        raise SystemExit("native rw_asos is required")
    table = out / "stations.json"
    stations, t_stations = run_json([str(native), "stations", "--networks", ",".join(networks),
                                     "--bbox", ",".join(str(v) for v in bbox), "--out", str(table)],
                                    out / "logs" / "stations.txt")
    print(f"surface: {stations.get('stations')} stations in {len(networks)} networks ({t_stations:.1f} s)", flush=True)
    fmt = lambda stamp: stamp.strftime("%Y-%m-%dT%H:%M:%S")
    seams, timings = [], []
    for index, cycle in enumerate(args.cycles):
        analysis = utc(cycle)
        work = out / f"slot-{index:02d}"
        work.mkdir(parents=True, exist_ok=True)
        csv, seam = work / "surface.csv", work / "surface.json"
        _, t_fetch = run_json([str(native), "fetch", "--stations", str(table),
                               "--start", fmt(analysis - dt.timedelta(seconds=MAX_AGE_S)),
                               "--end", fmt(analysis), "--out", str(csv), "--product", "metar"],
                              work / "surface-fetch.txt")
        decoded, t_decode = run_json([str(native), "decode", "--stations", str(table), "--obs", str(csv),
                                      "--start", fmt(analysis), "--end", fmt(analysis),
                                      "--match-seconds", str(MAX_AGE_S), "--min-report-rate", "0.8",
                                      "--out", str(seam), "--product", "metar"], work / "surface-decode.txt")
        record = json.loads(seam.read_text())
        validate_surface(record, analysis)
        run_json([str(native), "verify", "--file", str(seam)], work / "surface-verify.txt")
        seams.append(record)
        timings.append({"slot": index, "analysis_time": iso(analysis), "fetch_seconds": round(t_fetch, 1),
                        "decode_seconds": round(t_decode, 1), "reports": len(record["reports"])})
        print(f"surface slot {index}: {len(record['reports'])} reports (fetch {t_fetch:.1f} s, decode {t_decode:.1f} s)", flush=True)
    combined = combine_surfaces(seams)
    combined["provenance"]["product"] = "hourly-causal-metar-union"
    save(out / "surface.json", combined)
    run_json([str(native), "verify", "--file", str(out / "surface.json")], out / "logs" / "surface-union-verify.txt")
    doc = {"schema": "conus-da.surface-slots.v1", "networks": networks, "bbox": bbox,
           "stations_in_table": stations.get("stations"), "slots": timings,
           "surface_obs": str(out / "surface.json"), "surface_sha256": sha(out / "surface.json"),
           "native_surface_binary_sha256": sha(native),
           "seconds": {"stations": round(t_stations, 1), "total": round(time.perf_counter() - t_all, 1)}}
    save(out / "surface-slots.json", doc)
    print(json.dumps(doc["seconds"]), flush=True)


# ---------------------------------------------------------------------------
# raob (counts only: the deck has no upper-air operator)
# ---------------------------------------------------------------------------

IEM = "https://mesonet.agron.iastate.edu"


def _get_json(url, timeout=60):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read())
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def raob(args):
    """Radiosonde profiles valid in each cycle's window, on the grid.

    Synoptic launches are valid at 00Z and 12Z; a cycle at any other hour
    has none in its window.  Counted from the IEM RAOB archive (an archive
    replay, like everything here).
    """
    import numpy as np
    from gpuwm.obs.target_grid import TargetGrid
    t_all = time.perf_counter()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    grid = TargetGrid.from_wrfout(Path(args.grid_wrfout))
    network = _get_json(f"{IEM}/geojson/network/RAOB.geojson")
    sites = []
    for feature in network["features"]:
        props, (lon, lat) = feature["properties"], feature["geometry"]["coordinates"][:2]
        if props.get("archive_end"):
            continue
        i, j = grid.mass_index(np.array([lat]), np.array([lon]))
        if 0 <= float(i[0]) <= grid.nx - 1 and 0 <= float(j[0]) <= grid.ny - 1:
            sites.append(props["sid"])
    rows = []
    for index, cycle in enumerate(args.cycles):
        analysis = utc(cycle)
        row = {"slot": index, "analysis_time": iso(analysis), "stations_on_grid": len(sites)}
        if analysis.hour % 12:
            row.update(profiles=0, note="no synoptic launch is valid in this hour")
            rows.append(row)
            continue
        stamp = analysis.strftime("%Y%m%d%H%M")

        def one(sid):
            doc = _get_json(f"{IEM}/json/raob.py?ts={stamp}&station={sid}")
            return sid, doc.get("profiles", [])

        t0 = time.perf_counter()
        profiles, levels = 0, {"temperature": 0, "dewpoint": 0, "wind": 0}
        with cf.ThreadPoolExecutor(max_workers=16) as pool:
            for sid, found in pool.map(one, sites):
                for prof in found:
                    if utc(prof["valid"]) != analysis or not prof.get("profile"):
                        continue
                    profiles += 1
                    for level in prof["profile"]:
                        levels["temperature"] += level.get("tmpc") is not None
                        levels["dewpoint"] += level.get("dwpc") is not None
                        levels["wind"] += level.get("sknt") is not None and level.get("drct") is not None
        row.update(profiles=profiles, levels=levels, fetch_seconds=round(time.perf_counter() - t0, 1))
        rows.append(row)
        print(json.dumps(row), flush=True)
    save(out / "raob-counts.json", {"schema": "conus-da.raob-counts.v1", "source": f"{IEM}/json/raob.py",
                                    "assimilated_by_deck": False, "slots": rows,
                                    "seconds": round(time.perf_counter() - t_all, 1)})


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def thinned_document(document, thinning):
    from types import SimpleNamespace
    from gpuwm.da.radar_assimilation import (_thinned_clear_air_document,
                                             _thinned_reflectivity_document,
                                             _thinned_velocity_document)
    cfg = SimpleNamespace(velocity_thinning_cells=thinning["velocity"], velocity_error_inflation=1.0,
                          reflectivity_thinning_cells=thinning["reflectivity"],
                          reflectivity_error_inflation=1.0,
                          clear_air_thinning_cells=thinning["clear_air"], clear_air_error_inflation=1.0)
    document, _ = _thinned_velocity_document(document, cfg)
    document, _ = _thinned_reflectivity_document(document, cfg)
    document, _ = _thinned_clear_air_document(document, cfg)
    return document


def station_points(surface_path, grid, analysis):
    """(variable -> list of flat level-0 column indices) for one analysis."""
    import numpy as np
    record = json.loads(Path(surface_path).read_text())
    stations = {s["station_id"]: s for s in record["stations"]}
    per_var = {}
    for report in record["reports"]:
        if utc(report["valid_time"] + ("" if report["valid_time"].endswith("Z") else "Z")) != analysis:
            continue
        s = stations[report["station_id"]]
        i, j = grid.mass_index(np.array([s["latitude"]]), np.array([s["longitude"]]))
        i, j = int(round(float(i[0]))), int(round(float(j[0])))
        if not (0 <= i < grid.nx and 0 <= j < grid.ny):
            continue
        for name, value in report["values"].items():
            if value is not None and np.isfinite(value):
                per_var.setdefault(name, []).append(j * grid.nx + i)
    return per_var


def count(args):
    import numpy as np
    from gpuwm.da.obs_radar import read_document
    from gpuwm.obs.target_grid import TargetGrid
    grid = TargetGrid.from_wrfout(Path(args.grid_wrfout))
    thinning = dict(DECK_THINNING)
    rows = []
    for index, radar_path in enumerate(args.radar):
        t0 = time.perf_counter()
        document = read_document(Path(radar_path), expected_grid=grid,
                                 expected_grid_identity=grid.identity_sha256())
        raw = document["variables"]
        before = {"z": int(np.asarray(raw["z_mask"]).astype(bool).sum()),
                  "z0": int(np.asarray(raw["z0_mask"]).astype(bool).sum()) if "z0_mask" in raw else 0,
                  "vr": int(np.asarray(raw["vr_mask"]).astype(bool).sum())}
        thin = thinned_document(document, thinning)["variables"]
        z = np.asarray(thin["z_mask"]).astype(bool)
        z0 = np.asarray(thin["z0_mask"]).astype(bool) if "z0_mask" in thin else np.zeros_like(z)
        vr = np.asarray(thin["vr_mask"]).astype(bool)
        radars = [str(r["id"]) for r in document["radars"]]
        windows = document.get("radar_windows")
        after = {"z": int(z.sum()), "z0": int(z0.sum()), "vr": int(vr.sum())}
        row = {"slot": index, "radar": str(radar_path), "valid_time": document.get("valid_time"),
               "grid": [grid.nz, grid.ny, grid.nx], "radars": len(radars),
               "before_thinning": before, "after_deck_thinning": after,
               "vr_by_radar": {r: int(vr[k].sum()) for k, r in enumerate(radars)},
               "count_seconds": round(time.perf_counter() - t0, 1)}
        if args.surface:
            analysis = utc(args.cycles[index]) if args.cycles else utc(document["valid_time"])
            per_var = station_points(args.surface, grid, analysis)
            row["surface"] = {k: len(v) for k, v in per_var.items()}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("slot", "radars", "before_thinning", "after_deck_thinning")}
                         | ({"surface": row["surface"]} if "surface" in row else {})), flush=True)
        if args.save_masks:
            Path(args.save_masks).mkdir(parents=True, exist_ok=True)
            payload = {"shape": np.array(z.shape), "z_raw": np.flatnonzero(np.asarray(raw["z_mask"]).astype(bool)),
                       "z0_raw": np.flatnonzero(np.asarray(raw["z0_mask"]).astype(bool)) if "z0_mask" in raw else np.zeros(0, np.int64),
                       "z": np.flatnonzero(z), "z0": np.flatnonzero(z0)}
            for k, r in enumerate(radars):
                if windows:
                    j0, j1, i0, i1 = (int(v) for v in windows[k])
                else:
                    j0, j1, i0, i1 = 0, grid.ny - 1, 0, grid.nx - 1
                payload[f"vrwin_{k}"] = np.array([j0, j1, i0, i1])
                payload[f"vr_{k}"] = np.flatnonzero(vr[k][:, :j1 - j0 + 1, :i1 - i0 + 1])
                payload[f"vrraw_{k}"] = np.flatnonzero(np.asarray(raw["vr_mask"][k]).astype(bool)[:, :j1 - j0 + 1, :i1 - i0 + 1])
            if args.surface:
                for name, cols in per_var.items():
                    payload[f"sfc_{name}"] = np.asarray(cols, dtype=np.int64)
            np.savez_compressed(Path(args.save_masks) / f"masks-slot-{index:02d}.npz", radars=np.array(radars), **payload)
    save(args.out, {"schema": "conus-da.obs-counts.v1", "thinning": thinning, "slots": rows})


# ---------------------------------------------------------------------------
# roster (GPU)
# ---------------------------------------------------------------------------

def _refine(field, factor):
    """Bilinear refinement of a (ny, nx) or (nz, ny, nx) field by ``factor``."""
    import numpy as np
    from scipy.ndimage import zoom
    field = np.asarray(field, dtype=np.float64)
    zf = (1,) * (field.ndim - 2) + (factor, factor)
    # grid_mode=False keeps the corner points and never repeats a
    # coordinate (grid_mode=True with "nearest" duplicated the edge
    # columns, which the localisation metric rightly refuses).
    return zoom(field, zf, order=1, grid_mode=False)


def _refine_mask_flat(flat, shape, factor, cells):
    """Observed 3-D cells -> every refined cell under them, deck-thinned."""
    import numpy as np
    from gpuwm.da.radar_assimilation import thin_mask
    nz, ny, nx = shape
    mask = np.zeros(nz * ny * nx, bool)
    mask[flat] = True
    mask = mask.reshape(nz, ny, nx).repeat(factor, axis=1).repeat(factor, axis=2)
    if cells > 1:
        mask = thin_mask(mask, np.ones(mask.shape, np.float32), np.ones(mask.shape, np.float32), cells)
    return mask


def roster(args):
    import numpy as np
    import cupy as cp
    from gpuwm.da import letkf as L
    import importlib
    build_neighbor_index = importlib.import_module(args.roster_module).build_neighbor_index
    from gpuwm.da.obs_radar import letkf_grid_geometry
    from gpuwm.da.radar_assimilation import thin_mask
    from gpuwm.obs.target_grid import TargetGrid
    t_all = time.perf_counter()
    tg = TargetGrid.from_wrfout(Path(args.grid_wrfout))
    geom9 = letkf_grid_geometry(tg)
    masks = np.load(args.masks, allow_pickle=False)
    nz, ny, nx = (int(v) for v in masks["shape"])
    f = int(args.refine)
    if f == 1:
        geom = geom9
    else:
        geom = L.GridGeometry(dx_m=float(tg.dx_m) / f, dy_m=float(tg.dy_m) / f,
                              heights_m=np.ascontiguousarray(_refine(geom9.heights_m, f)),
                              lat_deg=_refine(geom9.lat_deg, f), lon_deg=_refine(geom9.lon_deg, f))
    gny, gnx = ny * f, nx * f
    shape = (nz, gny, gnx)
    members = int(args.members)
    radar_loc = L.Localization(args.radar_h_m, args.radar_v_m)
    sfc_loc = L.Localization(args.sfc_h_m, args.sfc_v_m)
    t0 = time.perf_counter()
    batches = []   # (name, mask (nz, nj, ni) bool, window or None, loc)
    if f == 1:
        def full(flat):
            m = np.zeros(nz * ny * nx, bool)
            m[flat] = True
            return m.reshape(shape)
        batches.append(("z", full(masks["z"]), None, radar_loc))
        batches.append(("z0", full(masks["z0"]), None, radar_loc))
    else:
        batches.append(("z", _refine_mask_flat(masks["z_raw"], (nz, ny, nx), f, DECK_THINNING["reflectivity"]), None, radar_loc))
        batches.append(("z0", _refine_mask_flat(masks["z0_raw"], (nz, ny, nx), f, DECK_THINNING["clear_air"]), None, radar_loc))
    radars = [str(r) for r in masks["radars"]]
    for k, r in enumerate(radars):
        j0, j1, i0, i1 = (int(v) for v in masks[f"vrwin_{k}"])
        wshape = (nz, j1 - j0 + 1, i1 - i0 + 1)
        if f == 1:
            m = np.zeros(int(np.prod(wshape)), bool)
            m[masks[f"vr_{k}"]] = True
            m = m.reshape(wshape)
        else:
            m = _refine_mask_flat(masks[f"vrraw_{k}"], wshape, f, DECK_THINNING["velocity"])
        window = (j0 * f, (j1 + 1) * f - 1, i0 * f, (i1 + 1) * f - 1)
        if window == (0, gny - 1, 0, gnx - 1):
            window = None
        batches.append((f"vr:{r}", m, window, radar_loc))
    for key in masks.files:
        # mslp is decoded but the deck assimilates T2, Td2 and wind only.
        if key.startswith("sfc_") and key != "sfc_mslp":
            cols = np.asarray(masks[key], dtype=np.int64)
            j, i = cols // nx, cols % nx
            m = np.zeros(shape, bool)
            m[0, j * f + f // 2, i * f + f // 2] = True
            batches.append((key[4:], m, None, sfc_loc))
    t_masks = time.perf_counter() - t0

    stencils = []
    for name, m, window, loc in batches:
        dj, di = L._horizontal_stencil(loc, geom, gnx, gny)
        dk = L._vertical_stencil(loc, geom)
        mflat = m.reshape(-1)
        if args.obs_stride > 1:
            # Every Nth observed point: the roster's size is linear in the
            # observation count, so a subset prices the whole.
            keep = np.flatnonzero(mflat)[::int(args.obs_stride)]
            mflat = np.zeros_like(mflat)
            mflat[keep] = True
        nobs = int(mflat.sum())
        # Synthetic member H(x) and values: the roster gathers only the
        # observed points, so a lazily zeroed array costs nothing until read.
        sim = np.zeros((members, mflat.size), dtype=np.float64)
        values = np.zeros(mflat.size, dtype=np.float64)
        err2 = np.ones(mflat.size, dtype=np.float64)
        j0, i0 = (window[0], window[2]) if window is not None else (0, 0)
        stencils.append({"name": name, "values": values, "err2": err2, "sim": sim, "mask": mflat,
                         "dk": dk, "dj": dj, "di": di, "hcut": float(loc.horizontal_m),
                         "vcut": float(loc.vertical_m), "j0": j0, "i0": i0,
                         "nj": m.shape[1], "ni": m.shape[2], "nobs": nobs})
    obs_total = sum(st["nobs"] for st in stencils)
    print(f"roster: grid {shape}, {len(stencils)} batches, {obs_total} observations "
          f"(masks {t_masks:.1f} s)", flush=True)

    host_z = np.asarray(geom.height_field(gny, gnx), dtype=np.float64).reshape(-1)
    lat = np.radians(geom.lat_deg).reshape(-1).astype(np.float64)
    lon = np.radians(geom.lon_deg).reshape(-1).astype(np.float64)

    def distance(a, b):
        return L._geodesic_m(lat[a], lon[a], lat[b], lon[b], float(geom.earth_radius_m))

    # The roster's horizontal weight table, one per distinct stencil,
    # float64 on the host and on the card, priced before anything is built.
    # Since 5de1c2a9b the table has one row per OBSERVED column
    # (observed_column_weight_table); a roster module without it still
    # builds the full (columns x offsets) table and is priced that way.
    observed_rows = hasattr(importlib.import_module(args.roster_module), "observed_column_weight_table")
    tables = {}
    for st in stencils:
        key = (tuple(np.asarray(st["dj"]).tolist()), tuple(np.asarray(st["di"]).tolist()), st["hcut"])
        if observed_rows:
            local = np.flatnonzero(st["mask"]) % (st["nj"] * st["ni"])
            cols = (local // st["ni"] + st["j0"]) * gnx + (local % st["ni"] + st["i0"])
            tables.setdefault(key, []).append(np.unique(cols))
        else:
            tables[key] = gny * gnx * int(np.asarray(st["dj"]).size) * 8
    if observed_rows:
        tables = {key: int(np.unique(np.concatenate(parts)).size) * len(key[0]) * 8
                  for key, parts in tables.items()}
    table_bytes = int(sum(tables.values()))
    pool = cp.get_default_memory_pool()
    free0, total = cp.cuda.runtime.memGetInfo()
    if table_bytes > args.max_table_gib * 2**30:
        result = {"schema": "conus-da.roster-scale.v1", "grid": list(shape), "refine": f,
                  "members": members, "observations": obs_total,
                  "localization": {"radar_h_m": args.radar_h_m, "radar_v_m": args.radar_v_m,
                                   "sfc_h_m": args.sfc_h_m, "sfc_v_m": args.sfc_v_m},
                  "horizontal_table_bytes": table_bytes, "card_total_bytes": int(total),
                  "batches": [{"name": st["name"], "observations": st["nobs"],
                               "h_offsets": int(np.asarray(st["dj"]).size)} for st in stencils],
                  "error": (f"not built: the horizontal weight tables need {table_bytes / 2**30:.1f} GiB "
                            f"on the host and again on the card (limit {args.max_table_gib} GiB)")}
        save(args.out, result)
        print(json.dumps({k: v for k, v in result.items() if k != "batches"}), flush=True)
        return
    peak = {"bytes": 0}
    error = None
    index = None
    t0 = time.perf_counter()
    try:
        index = build_neighbor_index(stencils, shape=shape, zflat=host_z, horizontal_distance=distance,
                                     members=members, solve_dtype=np.float64, xp=cp,
                                     gaspari_cohn=L.gaspari_cohn)
        peak["bytes"] = int(pool.total_bytes())
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        import traceback
        error = f"{type(exc).__name__}: {exc}"
        where = traceback.extract_tb(exc.__traceback__)[-3:]
        error += " at " + " <- ".join(f"{Path(fr.filename).name}:{fr.lineno} {fr.line}" for fr in reversed(where))
        peak["bytes"] = int(pool.total_bytes())
    build = time.perf_counter() - t0
    result = {"schema": "conus-da.roster-scale.v1", "grid": list(shape), "refine": f,
              "members": members, "batches": [{"name": st["name"], "observations": st["nobs"],
                                               "h_offsets": int(np.asarray(st["dj"]).size),
                                               "v_offsets": int(np.asarray(st["dk"]).size)}
                                              for st in stencils],
              "observations": obs_total,
              "localization": {"radar_h_m": args.radar_h_m, "radar_v_m": args.radar_v_m,
                               "sfc_h_m": args.sfc_h_m, "sfc_v_m": args.sfc_v_m},
              "horizontal_table_bytes": table_bytes,
              "card_total_bytes": int(total), "card_free_before_bytes": int(free0),
              "build_wall_seconds": round(build, 2), "error": error,
              "obs_stride": int(args.obs_stride), "pool_bytes_at_end": peak["bytes"]}
    if index is not None:
        counts = index.counts
        active = int((counts > 0).sum())
        if args.digest:
            # The roster's exact content (row offsets, observation ids,
            # weights), so two builds can be compared bit for bit.
            h = hashlib.sha256()
            for arr in (index.row_ptr, index.obs, index.weight):
                arr = arr if isinstance(arr, np.ndarray) else cp.asnumpy(arr)
                for lo in range(0, int(arr.size), 1 << 27):
                    h.update(np.ascontiguousarray(arr[lo:lo + (1 << 27)]).tobytes())
            result["roster_sha256"] = h.hexdigest()
        result.update(receipt=index.receipt, active_points=active, max_local_obs=int(counts.max()),
                      mean_local_obs_active=round(float(counts.sum() / max(1, active)), 1),
                      pool_bytes_after_build=peak["bytes"], index_bytes=index.nbytes)
        save(args.out, result)
        # One transform chunk packed from the roster, the downstream use,
        # sized to a packed-array budget the way the transform sizes its
        # chunks (members + 3 arrays of local_max float64 per point).
        local_max = int(counts.max())
        per_point = (members + 3) * local_max * 8
        npack = max(1, min(int(args.pack_points), int(args.pack_budget_gib * 2**30 // per_point)))
        gpts = np.flatnonzero(counts > 0)[:npack]
        try:
            t1 = time.perf_counter()
            packed = index.pack(gpts, local_max, members, np.float64)
            cp.cuda.runtime.deviceSynchronize()
            result.update(pack_points=int(gpts.size), pack_local_max=local_max,
                          pack_seconds=round(time.perf_counter() - t1, 3),
                          pack_bytes=int(sum(a.nbytes for a in packed)))
            del packed
        except Exception as exc:  # noqa: BLE001
            result["pack_error"] = f"{type(exc).__name__}: {exc}"
    result["total_seconds"] = round(time.perf_counter() - t_all, 1)
    save(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "batches"}, default=str), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("radar", help="every reaching NEXRAD, deck route, one radar-grid file per cycle")
    p.add_argument("--grid-wrfout", required=True)
    p.add_argument("--cycles", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit-sites", type=int, default=0, help="smoke only")
    p.add_argument("--raw-cache", help="read-only tree of already fetched objects (bucket/key layout)")
    p.add_argument("--publication-gate", choices=("last-modified", "measured-end"), default="last-modified",
                   help="last-modified (default): an archive object whose LastModified is after the analysis is "
                        "refused (the deck's near-real-time rule). measured-end: an archive replay; the volume's own "
                        "measured end clock (not after the analysis, at most the deck's staleness) is the causal gate, "
                        "because LastModified on a years-old object is its upload time")
    p = sub.add_parser("surface", help="METAR through rw_asos for every overlapping ASOS network")
    p.add_argument("--grid-wrfout", required=True)
    p.add_argument("--cycles", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("raob", help="radiosonde profile counts on the grid (no deck operator)")
    p.add_argument("--grid-wrfout", required=True)
    p.add_argument("--cycles", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("count", help="what the analysis would assimilate after the deck's thinning")
    p.add_argument("--grid-wrfout", required=True)
    p.add_argument("--radar", nargs="+", required=True)
    p.add_argument("--surface")
    p.add_argument("--cycles", nargs="*")
    p.add_argument("--out", required=True)
    p.add_argument("--save-masks")
    p = sub.add_parser("roster", help="build the neighbour roster on a card from real masks")
    p.add_argument("--grid-wrfout", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--refine", type=int, default=1)
    p.add_argument("--members", type=int, default=32)
    p.add_argument("--radar-h-m", type=float, default=12000.0)
    p.add_argument("--radar-v-m", type=float, default=6000.0)
    p.add_argument("--sfc-h-m", type=float, default=12000.0)
    p.add_argument("--sfc-v-m", type=float, default=3000.0)
    p.add_argument("--pack-points", type=int, default=200000)
    p.add_argument("--pack-budget-gib", type=float, default=6.0)
    p.add_argument("--max-table-gib", type=float, default=24.0)
    p.add_argument("--obs-stride", type=int, default=1, help="keep every Nth observation (pricing runs)")
    p.add_argument("--roster-module", default="gpuwm.da.neighbor_index")
    p.add_argument("--digest", action="store_true", help="sha256 of the built roster")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"radar": radar, "surface": surface, "raob": raob, "count": count,
     "roster": roster}[args.command](args)


if __name__ == "__main__":
    main()
