#!/usr/bin/env python3
"""Unsigned recent-case inventory, one fetch controller, and native observations.

Inventory reads only public object metadata. Fetch and prepare belong on the box.
Historical object timestamps do not prove original publication latency.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import threading
import time
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.da_object_fetch import timed
from gpuwm.filesystem_paths import replace_file_with_retry
UTC = dt.timezone.utc
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
BUCKETS = {"unidata-nexrad-level2", "noaa-nexrad-level2", "noaa-mrms-pds",
           "noaa-hrrr-bdp-pds", "noaa-rap-pds", "noaa-gfs-bdp-pds"}
TRUTH_PRODUCTS = ("PrecipRate_00.00", "MergedReflectivityQCComposite_00.50",
                  "RadarQualityIndex_00.00")


def utc(value):
    stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() != dt.timedelta(0):
        raise ValueError("explicit UTC time required")
    return stamp.astimezone(UTC)


def iso(value):
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def seam_utc(value):
    """Native seam clock strings without a zone are documented UTC."""
    stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return utc(stamp.isoformat())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def document_sha(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def model_request_rows(value):
    if isinstance(value, dict):
        if value.get("schema") != "da-rerun.public-object-requests.v1":
            raise ValueError("wrong public model request wrapper schema")
        value = value["objects"]
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("model requests must be a bucket/key list or native case request wrapper")
    return value


def surface_networks_argument(value):
    rows = value.split(",") if isinstance(value, str) else value
    if (not isinstance(rows, list) or not rows or any(not isinstance(row, str) for row in rows)):
        raise ValueError("surface_networks must be a nonempty list or CSV string")
    names = [row.strip() for row in rows]
    if len(set(names)) != len(names) or any(re.fullmatch(r"[A-Za-z0-9_-]+", row) is None for row in names):
        raise ValueError("surface_networks contains empty, repeated or invalid network ids")
    return ",".join(names)


def case_surface_networks(case):
    """The case's named networks, or the domain's own from its bbox.

    A case that names no networks gets every network whose extent meets its
    bbox plus a small margin, from the same function the authoring step
    uses; a bbox no network reaches is refused there by name.
    """
    named = case.get("surface_networks")
    if named is not None:
        return named
    from gpuwm.obs.surface_networks import networks_for_domain
    return list(networks_for_domain(*case["bbox"]))


def save(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="."+path.name+".", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        # A Windows reader holding the receipt refuses a plain os.replace.
        replace_file_with_retry(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_case(doc):
    if doc.get("schema") != "da-rerun.recent-observations.v1":
        raise ValueError("wrong recent observation case schema")
    origin = utc(doc["model_start_utc"])
    analyses = [utc(value) for value in doc["analysis_times_utc"]]
    if len(analyses) not in (3, 4, 5, 6):
        raise ValueError("recent case requires three to six hourly observed legs")
    if analyses != [origin + dt.timedelta(hours=i+1) for i in range(len(analyses))]:
        raise ValueError("analyses must be positive hourly legs from model_start_utc")
    cutoff = utc(doc["observation_cutoff_utc"])
    if analyses[-1] > cutoff:
        raise ValueError("analysis is beyond frozen observation cutoff")
    bbox = doc["bbox"]
    if (len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox)
            or not -180 <= bbox[0] < bbox[2] <= 180
            or not -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("bbox must be finite W,S,E,N")
    sites = doc["radar_sites"]
    if len(sites) < 2 or len(set(sites)) != len(sites) or any(
            re.fullmatch(r"K[A-Z0-9]{3}", site) is None for site in sites):
        raise ValueError("name at least two distinct full NEXRAD site ids")
    minimum = doc.get("min_radars", 2)
    if type(minimum) is not int or not 2 <= minimum <= len(sites):
        raise ValueError("min_radars must be an integer in 2..declared roster size")
    if doc.get("radar_bucket", "unidata-nexrad-level2") not in (
            "unidata-nexrad-level2", "noaa-nexrad-level2"):
        raise ValueError("unsupported public radar archive")
    age = doc.get("radar_max_age_seconds", 900)
    if type(age) is not int or not 300 <= age <= 1800:
        raise ValueError("radar_max_age_seconds must be an integer in 300..1800")
    return origin, analyses


def truth_bbox(doc):
    """A separately frozen truth extent, never cropped to forecast coverage."""
    bounds = doc.get("truth_bbox")
    if (not isinstance(bounds, list) or len(bounds) != 4
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bounds)
            or not -180 <= bounds[0] < bounds[2] <= 180
            or not -90 <= bounds[1] < bounds[3] <= 90):
        raise ValueError("explicit independently widened truth_bbox W,S,E,N required")
    model = doc["bbox"]
    if not all((bounds[0] < model[0], bounds[1] < model[1], bounds[2] > model[2], bounds[3] > model[3])):
        raise ValueError("truth_bbox must contain the complete model bbox with independent support padding")
    padding = doc.get("truth_padding_m")
    if not isinstance(padding, (int, float)) or not math.isfinite(padding) or padding < 75000:
        raise ValueError("truth_bbox needs a frozen native geometric padding receipt of at least 75000 m")
    return bounds


def list_objects(bucket, prefix, *, timeout=30):
    if bucket not in BUCKETS:
        raise ValueError("not an allowed anonymous public bucket")
    from tools.da_object_fetch import events
    for event in events("inventory", {"sources": [{"bucket": bucket, "prefix": prefix}], "timeout_seconds": timeout}):
        if event["phase"] == "object":
            yield {key: event[key] for key in ("bucket", "key", "bytes", "etag", "last_modified")}
        elif event["phase"] == "source" and event["status"] == "unavailable":
            raise ValueError(event["error"])


def object_time(key):
    # NEXRAD site prefix abuts its date. MRMS uses YYYYMMDD-HHMMSS.
    match = re.search(r"(\d{8})[-_](\d{6})", key)
    if not match:
        return None
    return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=UTC)


@timed("fetch.inventory")
def inventory(doc, *, listing=list_objects, first_slot_only=False, model_requests=None, on_object=None):
    origin, analyses = validate_case(doc)
    if first_slot_only:
        analyses = analyses[:1]
    rows, sources, slots = {}, [], []
    bucket = doc.get("radar_bucket", "unidata-nexrad-level2")
    age = doc.get("radar_max_age_seconds", 900)
    dates = sorted({stamp.date() for stamp in [origin, *analyses]})
    radar_rows = {site: [] for site in doc["radar_sites"]}
    model_requests = model_request_rows(([] if first_slot_only else doc.get("model_requests", []))
                                       if model_requests is None else model_requests)
    if listing is list_objects:
        from tools.da_object_fetch import events
        plans = {(bucket, f"{day:%Y/%m/%d}/{site}/") for day in dates for site in radar_rows}
        if not first_slot_only:
            for day in {(analyses[-1]+dt.timedelta(hours=h)).date() for h in range(7)}:
                plans.update(("noaa-mrms-pds", f"CONUS/{product}/{day:%Y%m%d}/") for product in TRUTH_PRODUCTS)
        for request in model_requests:
            raw_path(Path("."), request)
            key = request["key"]
            prefix = key.rsplit("/", 1)[0] + "/" + ".".join(Path(key).name.split(".")[:2]) + "."
            plans.add((request["bucket"], prefix))
        cached = {identity: [] for identity in plans}
        faults = {}
        for event in events("inventory", {"sources": [{"bucket": b, "prefix": p} for b, p in sorted(plans)], "timeout_seconds": 30}):
            if event["phase"] == "object":
                row = {key: event[key] for key in ("bucket", "key", "bytes", "etag", "last_modified")}
                cached[(event["bucket"], event["source"])].append(row)
                if on_object is not None:
                    on_object(row)
            elif event["phase"] == "source" and event["status"] == "unavailable":
                faults[(event["bucket"], event["prefix"])] = event["error"]
        def listing(bucket, prefix):
            if (bucket, prefix) in faults:
                raise ValueError(faults[(bucket, prefix)])
            return iter(cached[(bucket, prefix)])
    for day in dates:
        for site in radar_rows:
            prefix = f"{day:%Y/%m/%d}/{site}/"
            try:
                found = list(listing(bucket, prefix))
                sources.append({"bucket": bucket, "prefix": prefix, "objects": len(found), "status": "listed"})
                radar_rows[site].extend(found)
            except (urllib.error.URLError, OSError, ValueError) as error:
                sources.append({"bucket": bucket, "prefix": prefix, "status": "unavailable", "error": str(error)})
    for analysis in analyses:
        selected = {}
        for site, found in radar_rows.items():
            candidates = [row for row in found if object_time(row["key"]) is not None
                          and analysis-dt.timedelta(seconds=age) <= object_time(row["key"]) <= analysis
                          and not row["key"].endswith("_MDM")]
            candidates.sort(key=lambda row: object_time(row["key"]), reverse=True)
            # Decode in this order until a COMPLETE volume ends at/before analysis.
            selected[site] = [row["key"] for row in candidates]
            for row in candidates:
                rows[(row["bucket"], row["key"])] = row
        slots.append({"analysis_time": iso(analysis), "radar_candidates": selected,
                      "status": "candidate-metadata" if all(selected.values()) else "missing-radar-metadata"})
    # Three one-hour scoring windows and temporal brackets, rather than all six hours.
    windows = [(analyses[-1]+dt.timedelta(hours=lead-1, seconds=-180),
                analyses[-1]+dt.timedelta(hours=lead, seconds=180)) for lead in (1, 3, 6)]
    truth_days = [] if first_slot_only else sorted({(analyses[-1]+dt.timedelta(hours=h)).date() for h in range(7)})
    for day in truth_days:
        for product in TRUTH_PRODUCTS:
            prefix = f"CONUS/{product}/{day:%Y%m%d}/"
            try:
                found = [row for row in listing("noaa-mrms-pds", prefix)
                         if object_time(row["key"]) is not None and any(
                             start <= object_time(row["key"]) <= stop for start, stop in windows)]
                sources.append({"bucket": "noaa-mrms-pds", "prefix": prefix,
                                "selected_objects": len(found), "status": "listed"})
                for row in found:
                    rows[(row["bucket"], row["key"])] = row
            except (urllib.error.URLError, OSError, ValueError) as error:
                sources.append({"bucket": "noaa-mrms-pds", "prefix": prefix, "status": "unavailable", "error": str(error)})
    grouped = {}
    for request in model_requests:
        bucket, key = request["bucket"], request["key"]
        raw_path(Path("."), request)
        prefix = key.rsplit("/", 1)[0] + "/" + ".".join(Path(key).name.split(".")[:2]) + "."
        grouped.setdefault((bucket, prefix), set()).add(key)
    for (bucket, prefix), required in grouped.items():
        try:
            found = {row["key"]: row for row in listing(bucket, prefix)}
            for key in required & found.keys():
                rows[(bucket, key)] = found[key]
            sources.append({"bucket": bucket, "prefix": prefix, "status": "listed",
                            "required_objects": len(required), "missing": sorted(required-found.keys())})
        except (urllib.error.URLError, OSError, ValueError) as error:
            sources.append({"bucket": bucket, "prefix": prefix, "status": "unavailable", "error": str(error)})
    return {"schema": "da-rerun.recent-inventory.v1", "case": doc,
            "model_requests": model_requests,
            "scope": "smoke-first-slot" if first_slot_only else "full-case",
            "audited_at_utc": iso(dt.datetime.now(UTC)), "slots": slots,
            "objects": sorted(rows.values(), key=lambda row: (row["bucket"], row["key"])),
            "bytes": sum(row["bytes"] for row in rows.values()), "sources": sources,
            "publication_policy": "archive-replay; LastModified is object metadata, not first publication proof",
            "raw_payloads_fetched": False, "scientific_gate": "pending"}


def raw_path(root, row):
    key = PurePosixPath(row["key"])
    if (row.get("bucket") not in BUCKETS or key.is_absolute()
            or any(part in (".", "..") for part in key.parts)
            or "\\" in row["key"] or not key.parts or str(key) != row["key"]
            or any(char in row["key"] for char in ":\r\n\x00")):
        raise ValueError("unsafe public object identity")
    root = Path(root).resolve()
    path = root / row["bucket"] / Path(*key.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("public object escaped owned destination")
    return path


@timed("fetch.object")
def _download(row, path):
    from tools.da_object_fetch import download
    result = list(download([{**row, "destination": str(path)}]))
    if len(result) != 1:
        raise ValueError("Rust fetch did not verify exactly one frozen object")
    return result[0]


@timed("fetch.inputs")
def fetch(doc, root, receipt, max_gib, *, reuse_receipt=None, reuse_receipt_sha256=None, streams=1):
    if not math.isfinite(max_gib) or max_gib <= 0:
        raise ValueError("positive finite size cap required")
    all_objects = doc["objects"]
    forecast = bool(doc.get("slots") and doc.get("case"))
    radar_keys = {(doc["case"].get("radar_bucket", "unidata-nexrad-level2"), key)
                  for slot in doc.get("slots", []) for keys in slot["radar_candidates"].values() for key in keys}
    deferred = [row for row in all_objects if forecast and (row["bucket"], row["key"]) in radar_keys]
    truth_rows = [row for row in all_objects if forecast and row["bucket"] == "noaa-mrms-pds"]
    objects = [row for row in all_objects if row not in deferred and row not in truth_rows]
    if type(streams) is not int or not 1 <= streams <= 64:
        raise ValueError("fetch streams must be 1..64; Rust chooses the bounded transfer count")
    if any(type(row.get("bytes")) is not int or row["bytes"] < 1 for row in all_objects):
        raise ValueError("invalid object size")
    if sum(row["bytes"] for row in all_objects) > max_gib * 2**30:
        raise ValueError("inventory exceeds fetch size cap")
    root = Path(root).resolve()
    all_paths = [raw_path(root, row) for row in all_objects]
    paths = [raw_path(root, row) for row in objects]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("duplicate public object destination")
    ancestors = {}
    if reuse_receipt is not None:
        if (reuse_receipt.get("schema") not in ("da-rerun.recent-fetch.v1", "da-rerun.recent-fetch.v2") or reuse_receipt.get("status") != "complete"
                or not isinstance(reuse_receipt_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", reuse_receipt_sha256) is None):
            raise ValueError("reuse needs a COMPLETE ancestor fetch receipt and its file SHA256")
        for row in reuse_receipt.get("objects", []):
            key = (row["bucket"], row["key"])
            if key in ancestors:
                raise ValueError("ancestor receipt has a duplicate object roster")
            if (Path(row["path"]).resolve() != raw_path(root, row)
                    or not isinstance(row.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                    or not row.get("fetched_at_utc")):
                raise ValueError("ancestor source path/hash/time identity is invalid")
            utc(row["fetched_at_utc"])
            ancestors[key] = row
    elif reuse_receipt_sha256 is not None:
        raise ValueError("ancestor SHA supplied without its receipt")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".recent-observation-fetch.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    rows, reused, downloaded = [], 0, 0
    order = {(row["bucket"], row["key"]): index for index, row in enumerate(objects)}
    truth = None
    deferred_rows = [{**row, "destination": str(raw_path(root, row))} for row in deferred]
    def validate_deferred():
        for row in deferred_rows:
            path = Path(row["destination"])
            old = ancestors.get((row["bucket"], row["key"]))
            if path.exists():
                if (old is None or path.is_symlink() or path.stat().st_size != row["bytes"]
                        or any(old.get(field) != row.get(field) for field in ("bytes", "etag", "last_modified"))
                        or sha(path) != old.get("sha256")):
                    raise ValueError("deferred radar cache is not verified by a complete ancestor")
                row.update(path=str(path), sha256=old["sha256"], fetched_at_utc=old["fetched_at_utc"])
            elif old is not None:
                raise ValueError("tracked ancestor radar object is missing")
    def write_fetch(status):
        # Rows keep the frozen inventory order whatever order streams finish.
        rows.sort(key=lambda row: order[(row["bucket"], row["key"])])
        value = {"schema": "da-rerun.recent-fetch.v2" if forecast else "da-rerun.recent-fetch.v1", "objects": rows,
                       "status": status, "fetch_processes": 1, "fetch_streams": min(32, len(objects)),
                       "inventory_sha256": document_sha(doc),
                       "ancestor_receipt_sha256": reuse_receipt_sha256,
                       "reused_objects": reused, "downloaded_objects": downloaded}
        if forecast:
            value.update(scope="forecast", deferred_objects=deferred_rows,
                         deferred_truth=truth_rows, truth=truth,
                         raw_root=str(root), receipt_path=str(Path(receipt).resolve()))
        save(receipt, value)
    pending = []
    try:
        validate_deferred()
        if truth_rows:
            from tools.da_object_fetch import start_truth
            truth = start_truth(doc, root, Path(receipt).with_name(Path(receipt).name + ".truth.json"), max_gib)
        for row, path in zip(objects, paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                old = ancestors.get((row["bucket"], row["key"]))
                if old is None:
                    raise ValueError("fresh destination required; existing raw object is not tracked by a COMPLETE ancestor " + str(path))
                if (any(old.get(field) != row.get(field) for field in ("bytes", "etag", "last_modified"))
                        or path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != old["sha256"]):
                    raise ValueError("ancestor metadata or actual cached bytes differ from frozen inventory")
                rows.append({**row, "path": str(path), "sha256": old["sha256"],
                             "fetched_at_utc": old["fetched_at_utc"], "fetch_action": "reused-verified"})
                reused += 1
                write_fetch("in-progress")
                continue
            if (row["bucket"], row["key"]) in ancestors:
                raise ValueError("tracked ancestor raw object is missing; refusing to fabricate reuse")
            pending.append((row, path))
        if pending:
            from tools.da_object_fetch import download
            for done in download([{**row, "destination": str(path)} for row, path in pending]):
                rows.append(done)
                downloaded += 1
                write_fetch("in-progress")
        write_fetch("complete")
    finally:
        lock.unlink()


def causal_volume(volume, analysis, max_age):
    if (getattr(volume, "complete", None) is not True
            or getattr(volume, "sweeps_incomplete", None) != 0
            or not getattr(volume, "sweeps_in_volume", 0)):
        raise ValueError("radar volume is not a complete measured sweep roster")
    end = getattr(volume, "end_time", None)
    if end is None:
        raise ValueError("radar pack has no measured radial end clock")
    end = seam_utc(end)
    if end > analysis:
        raise ValueError("radar volume finishes after analysis cutoff")
    if (analysis-end).total_seconds() > max_age:
        raise ValueError("radar measured end is stale")
    return end


#: Archive re-stamps: a bucket whose objects were re-uploaded wholesale, so
#: their S3 LastModified is the re-upload date and says nothing about when
#: the scan was first published.  One row per re-stamp; adding a bucket or
#: a second migration is a row here, never a code path.
#:
#: ``restamped_from``: LastModified on or after this is the re-upload's.
#: ``scanned_before``: only objects whose KEY scan time is before this were
#: re-stamped; anything scanned later was published after the migration and
#: keeps its LastModified as the publication bound.
#:
#: Breakage this prevents: 2024 archive objects re-stamped 2025-07-15 by
#: bucket migration; every pre-migration replay case refused (every 2024
#: unidata-nexrad-level2 object reads LastModified 2025-07-15T22:59:04Z,
#: e.g. 2024/05/21/KDMX/KDMX20240521_175821_V06, so causal_object refused
#: all of them and radar_tten_windows wrote "no volumes to grid").
ARCHIVE_RESTAMPS = (
    # The migration re-uploaded the archive oldest first over about six weeks, so the row spans it: measured
    # LastModified of whole site-days (2026-10-12) -- 1998-05-03 KTLX 2025-06-10, 2008-05-10 KTLX 2025-06-25,
    # 2015-06-01 KDMX 2025-07-02, 2022-06-13 KIND 2025-07-11, 2024-05-21 KDMX 2025-07-15, 2025-07-16 KTLX
    # 2025-07-18.  The old row (from 2025-07-15) refused every volume of a 2022 replay case as "LastModified is
    # after analysis cutoff", so radar_tten_windows stopped with "no volumes to grid".
    {"bucket": "unidata-nexrad-level2",
     "restamped_from": "2025-06-01T00:00:00Z",
     "scanned_before": "2025-07-17T00:00:00Z",
     "reason": "bucket migration re-uploaded the archive scanned before 2025-07-17, oldest first, 2025-06 to 2025-07-18"},
)

#: ``causal_object``'s return values: which clock bounded publication.
CAUSAL_BY_LAST_MODIFIED = "last_modified"
CAUSAL_BY_SCAN_TIME = "archive-replay-scan-time"


def archive_restamp(row):
    """The ``ARCHIVE_RESTAMPS`` row that re-stamped ``row``, or ``None``.

    An object qualifies only when its bucket has a row, its LastModified is
    on or after that row's re-upload date, and its key scan time is before
    the migration: then the LastModified is the migration's, not the
    publication's.  A post-migration scan uploaded late does not qualify.
    """
    modified, scanned = row.get("last_modified"), object_time(str(row.get("key", "")))
    if modified is None or scanned is None:
        return None
    for restamp in ARCHIVE_RESTAMPS:
        if (row.get("bucket") == restamp["bucket"]
                and utc(modified) >= utc(restamp["restamped_from"])
                and scanned < utc(restamp["scanned_before"])):
            return restamp
    return None


def causal_object(row, analysis):
    """Refuse an object not provably published by ``analysis``.

    LastModified is the publication upper bound, except for an object an
    ``ARCHIVE_RESTAMPS`` row re-stamped: its LastModified is the migration's,
    so causality is judged by the scan time in its key (archive replay).
    Returns the clock used.  A genuinely late object -- LastModified after
    the cutoff and not re-stamped -- is still refused.
    """
    modified = row.get("last_modified")
    if modified is None:
        raise ValueError("radar archive object has no publication upper-bound metadata")
    if utc(modified) <= analysis:
        return CAUSAL_BY_LAST_MODIFIED
    if archive_restamp(row) is not None:
        if object_time(str(row["key"])) > analysis:
            raise ValueError("radar archive object scan time is after analysis cutoff")
        return CAUSAL_BY_SCAN_TIME
    raise ValueError("radar archive object LastModified is after analysis cutoff")


def validate_contribution(contribution):
    account = contribution.dealias
    totals = account.get("totals", {})
    if (account.get("params", {}).get("engine") != "region-global"
            or totals.get("engine") != "region-global"
            or totals.get("sweeps_dealiased", 0) < 1 or totals.get("gates_offered", 0) < 1
            or totals.get("accounting_balances") is not True
            or not any(sweep.get("native", {}).get("library") for sweep in account.get("sweeps", []))):
        raise ValueError("named region-global native solver did not produce a balanced finite-gate account")
    if not contribution.z_count.any() or not contribution.vr_count.any():
        raise ValueError("radar contributed no usable Z or Vr on the prepared grid")
    if not contribution.cc_qc:
        raise ValueError("requested dual-pol QC did not produce its account")


def validate_fetched(inv, fetched, *, purpose="forecast"):
    if fetched.get("scope") == "truth" and purpose != "truth":
        raise ValueError("truth-only fetch receipt cannot feed forecast observations; required radar/model sources are absent")
    if (fetched.get("schema") not in ("da-rerun.recent-fetch.v1", "da-rerun.recent-fetch.v2")
            or fetched.get("status") != "complete"
            or fetched.get("inventory_sha256") != document_sha(inv)):
        raise ValueError("complete hash-bound public fetch receipt required")
    expected = {(row["bucket"], row["key"]): row for row in inv["objects"]
                if fetched.get("scope") != "truth" or row["bucket"] == "noaa-mrms-pds"}
    actual = {(row["bucket"], row["key"]): row for row in fetched["objects"]}
    planned = fetched.get("deferred_objects", []) + fetched.get("deferred_truth", [])
    if planned:
        if fetched.get("schema") != "da-rerun.recent-fetch.v2" or fetched.get("scope") != "forecast":
            raise ValueError("deferred observations require the explicit forecast receipt contract")
        allowed_radar = {(inv["case"].get("radar_bucket", "unidata-nexrad-level2"), key)
                         for slot in inv["slots"] for keys in slot["radar_candidates"].values() for key in keys}
        for row in planned:
            key = row["bucket"], row["key"]
            if key in actual or key not in expected or any(row.get(field) != expected[key].get(field)
                    for field in ("bytes", "etag", "last_modified")):
                raise ValueError("deferred source metadata differs from frozen inventory")
            if row in fetched.get("deferred_objects", []):
                if key not in allowed_radar or Path(row["destination"]).resolve() != raw_path(fetched["raw_root"], row):
                    raise ValueError("deferred radar destination or candidate roster differs from frozen inventory")
            elif row["bucket"] != "noaa-mrms-pds":
                raise ValueError("only scoring truth may be deferred outside radar candidates")
            actual[key] = row
    if len(actual) != len(fetched["objects"])+len(planned) or actual.keys() != expected.keys():
        raise ValueError("fetch receipt object roster differs from frozen inventory")
    for key, row in actual.items():
        if row in planned and "path" not in row:
            continue
        path = Path(row["path"])
        if (any(row.get(field) != expected[key].get(field) for field in ("bytes", "etag", "last_modified"))
                or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]):
            raise ValueError("raw source identity changed after frozen fetch: " + row["key"])


def run_json(command, log):
    started = time.monotonic()
    result = subprocess.run(command, text=True, capture_output=True, timeout=600)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    Path(log).write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"native command failed {result.returncode}; see {log}")
    return json.loads(result.stdout), time.monotonic()-started


def _site_volume(site, keys, raw, case, analysis, work, binary, grid, params, decode, verify, read_pack,
                 superob):
    """Newest complete causal volume of one site: (contribution, source, refusals)."""
    refusals = []
    for key in sorted(keys, key=lambda value: object_time(value) or dt.datetime.min.replace(tzinfo=UTC), reverse=True):
        row = raw[(case.get("radar_bucket", "unidata-nexrad-level2"), key)]
        try:
            causal_object(row, analysis)
        except ValueError as error:
            refusals.append({"site": site, "key": key, "reason": str(error)})
            continue
        if "path" not in row:
            row.update(_download(row, Path(row["destination"])))
        pack = work / (Path(key).name + ".pack")
        decode(binary, volume=Path(row["path"]), out=pack, moments=("REF", "VEL", "RHO"),
               max_range_km=250.0, max_elevation_deg=20.0, censor_flags=True)
        verified = verify(binary, pack=pack)
        if verified.get("status") != "PASS":
            raise ValueError("native radar pack integrity did not PASS")
        volume = read_pack(pack)
        try:
            end = causal_volume(volume, analysis, case.get("radar_max_age_seconds", 900))
        except ValueError as error:
            refusals.append({"site": site, "key": key, "reason": str(error)})
            continue
        contribution = superob(volume, grid, params=params, clear_air_from_censor=True)
        try:
            validate_contribution(contribution)
        except ValueError as error:
            refusals.append({"site": site, "key": key, "reason": str(error)})
            continue
        source = {"site": site, "key": key, "sha256": row["sha256"],
                  "pack_sha256": sha(pack), "measured_end_utc": iso(end),
                  "last_modified": row["last_modified"], "dealias": contribution.dealias,
                  "cc_qc": contribution.cc_qc,
                  "qc_counts": contribution.counts.to_payload()}
        return contribution, source, refusals
    refusals.append({"site": site, "reason": "no complete causal volume"})
    return None, None, refusals


def available_memory_bytes():
    """Available host RAM, bounded by the current cgroup's remaining RAM."""
    values = []
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                values.append(int(line.split()[1]) * 1024)
        # A v2 container can have less memory than the host advertises.
        group = next(line.split(':', 2)[2] for line in
                     Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::'))
        mount = Path('/sys/fs/cgroup')
        roots = [mount]
        root = (mount / group.lstrip('/')).resolve()
        while root.is_relative_to(mount) and root != mount:
            roots.append(root)
            root = root.parent
        for root in roots:
            try:
                limit = (root / 'memory.max').read_text().strip()
                if limit != 'max':
                    values.append(max(0, int(limit) - int((root / 'memory.current').read_text())))
            except (OSError, ValueError):
                continue
    except (OSError, StopIteration, ValueError):
        pass
    if not values:
        raise ValueError('cannot measure available RAM; unbounded observation workers could exhaust memory')
    return min(values)


def observation_workers(site_count, slot_count=1, *, requested=None,
                        available_bytes=None, worker_bytes=2 * 2**30,
                        retained_slot_bytes=0):
    """Size native work, including results retained until the ordered merge.

    worker_bytes is a conservative scratch allowance, replaceable by a measured
    peak in callers. Retained outputs are separately reserved for every slot.
    """
    if site_count < 1 or slot_count < 1 or worker_bytes < 1 or retained_slot_bytes < 0:
        raise ValueError('positive site, slot and worker sizes required')
    if requested is not None and requested < 1:
        raise ValueError('workers must be positive; zero would suppress observations')
    available = available_memory_bytes() if available_bytes is None else int(available_bytes)
    budget = available * 4 // 5
    slots = min(slot_count, budget // (retained_slot_bytes + worker_bytes))
    if slots < 1:
        raise ValueError('available RAM cannot hold one observation slot and native worker; prevents OOM and incomplete observations')
    wanted = min(site_count, requested if requested is not None else site_count)
    count = min(wanted, (budget - slots * retained_slot_bytes) // (slots * worker_bytes))
    return {'workers': int(count), 'slot_workers': int(slots),
            'total_workers': int(count * slots),
            'site_count': site_count, 'slot_count': slot_count,
            'available_bytes': available, 'budget_bytes': budget,
            'worker_scratch_bytes': worker_bytes, 'retained_slot_bytes': retained_slot_bytes,
            'reserve_fraction': 0.2}


def native_observation_threads(site_count, slot_count=1, requested=None):
    from gpuwm.obs import superob_bridge
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    per_worker = max(1, int(os.environ.get('RAYON_NUM_THREADS', '2')))
    wanted = min(site_count, requested if requested is not None else site_count)
    return superob_bridge.configure_threads(min(cores, max(1, wanted * slot_count * per_worker)))


def observation_memory(grid, sites, params):
    """Reserve native accumulator and merge buffers from Rust reach windows."""
    from gpuwm.obs.superob import horizontal_window
    cells = []
    for site in sites:
        j0, j1, i0, i1 = horizontal_window(grid, site, params)
        cells.append(grid.nz * (j1 - j0 + 1) * (i1 - i0 + 1))
    # 16 eight-byte accumulator planes plus windowed merge outputs and copies.
    # Full-grid output/writer buffers are also live during the merge.
    retained = 256 * sum(cells) + 256 * grid.nz * grid.ny * grid.nx
    return retained


def require_native_observations():
    from gpuwm.obs import superob_bridge
    if superob_bridge.python_reference_requested():
        raise ValueError('parallel observation preparation requires Rust superob; the Python reference cannot be pooled')
    superob_bridge.library()


def process_tree_rss_bytes():
    """RSS of this process and its children, without reading command lines."""
    pending, seen, total = [os.getpid()], set(), 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        root = Path('/proc') / str(pid)
        try:
            for line in (root / 'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1]) * 1024
            # A subprocess can belong to any thread, not just the main thread.
            for task in (root / 'task').iterdir():
                try:
                    pending.extend(int(value) for value in (task / 'children').read_text().split())
                except (OSError, ValueError):
                    pass
        except (OSError, ValueError):
            pass
    return total


class ObservationRss:
    """Sample the native process tree. Short-lived peaks can fall between samples."""
    def __enter__(self):
        self.before = self.peak = process_tree_rss_bytes()
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.05):
                self.peak = max(self.peak, process_tree_rss_bytes())
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, process_tree_rss_bytes())

    def receipt(self):
        return {'before_tree_rss_bytes': self.before, 'peak_tree_rss_bytes': self.peak,
                'incremental_peak_bytes': max(0, self.peak - self.before),
                'sample_seconds': .05, 'basis': 'sampled RSS; shared pages counted per process'}


def observation_stage(stage, out, **attrs):
    from gpuwm.timeline.ledger import Ledger, span
    thread = threading.get_ident()
    attrs = {**attrs, 'thread': thread,
             'span_id': f"{stage}:{attrs.get('slot', '')}:{attrs.get('site', '')}:{thread}"}
    return span(stage, ledger=Ledger(Path(out) / 'timeline.jsonl', product='da'), attrs=attrs)


def _truth_status(out, status, **fields):
    """Publish the completion marker only after truth.json is complete."""
    target = Path(out) / 'truth-job-status.json'
    temporary = target.with_suffix('.tmp')
    save(temporary, {'schema': 'da-rerun.truth-job.v1', 'status': status, **fields})
    replace_file_with_retry(temporary, target)


def start_truth(inv, fetched, out, center_lon, center_lat, *, wait_claim=True):
    """Start optional verification work without making forecast wait for it.

    The scorer must call join_truth on this directory before reading truth.json.
    A failed decode leaves a terminal receipt, not a forecast failure.
    """
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    job = out / 'truth-job.json'
    from tools.da_object_fetch import fetch_binding
    identity = {'inventory_sha256': document_sha(inv),
                'fetch_receipt_sha256': fetch_binding(fetched),
                'center_lon': center_lon, 'center_lat': center_lat}
    # An exclusive job claim prevents duplicate writers to the same packs.
    try:
        with job.open('x', encoding='utf-8') as stream:
            json.dump(identity, stream, sort_keys=True)
    except FileExistsError:
        # Another creator can hold the exclusive file open before its buffered
        # JSON reaches disk. Do not mistake that short startup window for stale
        # truth, and never launch a second writer while waiting for its claim.
        deadline = time.monotonic() + 5
        while True:
            try:
                existing = json.loads(job.read_text())
                break
            except json.JSONDecodeError:
                if not wait_claim:
                    raise ValueError('truth input claim is incomplete; duplicate pack writers are refused')
                if time.monotonic() >= deadline:
                    raise ValueError('truth input claim is incomplete; duplicate pack writers are refused')
                time.sleep(.01)
        if existing != identity:
            raise ValueError('truth output belongs to different inputs; prevents scoring against stale truth')
        return {'out': str(out), **identity}
    save(out / 'background-inventory.json', inv)
    save(out / 'background-fetch.json', fetched)
    command = [sys.executable, str(Path(__file__).resolve()), 'truth',
        '--inventory', str(out / 'background-inventory.json'),
        '--fetch-receipt', str(out / 'background-fetch.json'), '--out', str(out),
        '--center-lon', str(center_lon), '--center-lat', str(center_lat), '--background-worker']
    _truth_status(out, 'RUNNING', command=command)
    try:
        with (out / 'background-truth.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=(os.name != 'nt'))
        # The child publishes terminal status. Do not race it by rewriting RUNNING.
        save(out / 'truth-process.json', {'pid': process.pid, 'command': command})
    except OSError as error:
        _truth_status(out, 'FAILED', error=f'{type(error).__name__}: {error}')
        print(f'truth preparation failed to start; verification unavailable: {error}; see {out}', file=sys.stderr)
    return {'out': str(out), **identity}


def join_truth(out, *, timeout_seconds=1800):
    """Scoring-only join. Never interpret a partial pack tree as truth."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('positive finite truth join timeout required')
    from gpuwm.ingest.boundary_stream import read_replaced_json
    out = Path(out).resolve()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status_path = out / 'truth-job-status.json'
        # The truth job replaces this receipt while it runs; on Windows a
        # read that meets the replace is refused for a moment.
        status = read_replaced_json(status_path) if status_path.exists() else {}
        if status.get('status') == 'COMPLETE':
            path = out / 'truth.json'
            if not path.is_file() or sha(path) != status.get('truth_sha256'):
                raise ValueError('completed truth hash mismatch; prevents scoring partial or changed truth')
            return json.loads(path.read_text())
        if status.get('status') == 'FAILED':
            raise RuntimeError(f"truth decode failed; verification unavailable: {status.get('error')}; see {out / 'background-truth.log'}")
        time.sleep(0.1)
    raise TimeoutError(f'truth decode did not complete; scoring partial truth is refused; see {out}')


def prepare(inv, fetched, grid_path, out, *, workers=None, truth_out=None):
    with ObservationRss() as rss:
        result = _prepare(inv, fetched, grid_path, out, workers=workers, truth_out=truth_out)
    save(Path(out) / 'peak-rss.json', rss.receipt())
    return result


def _prepare(inv, fetched, grid_path, out, *, workers=None, truth_out=None):
    from gpuwm.obs.cc_qc import CcQcParams
    from gpuwm.obs.dealias import DealiasParams, engine_unavailable_reason
    from gpuwm.obs.frontdoor import ASOS
    from gpuwm.obs.nexrad import find_nexrad_bin, run_decode, run_verify
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import SuperobParams, merge_contributions, superob_volume
    from gpuwm.obs.sweeps import read_sweep_pack
    from gpuwm.obs.target_grid import TargetGrid
    case = inv["case"]
    origin, analyses = validate_case(case)
    smoke = inv.get("scope") == "smoke-first-slot"
    if smoke:
        analyses = analyses[:1]
    if len(inv["slots"]) != len(analyses):
        raise ValueError("inventory slot roster differs from declared full/smoke scope")
    validate_fetched(inv, fetched)
    require_native_observations()
    binary = find_nexrad_bin()
    reason = engine_unavailable_reason("region-global")
    if binary is None or reason is not None:
        raise ValueError("required native radar/dealias door unavailable: " + str(reason))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not smoke:
        from tools.da_object_fetch import launch_truth_for_forecast
        west, south, east, north = case['bbox']
        definition = case.get('truth_padding_provenance', {}).get('model_grid_definition', {})
        launch_truth_for_forecast(inv, fetched, truth_out or out / 'truth',
                    case.get('center_lon', definition.get('ref_lon', (west + east) / 2)),
                    case.get('center_lat', definition.get('ref_lat', (south + north) / 2)))
    grid_path = Path(grid_path).resolve()
    grid = TargetGrid.from_wrfout(grid_path)
    params = SuperobParams(max_range_km=250.0, max_elevation_deg=20.0,
                          dealias=DealiasParams(engine="region-global"), cc_qc=CcQcParams())
    params.validate()
    raw = {(row["bucket"], row["key"]): row for row in fetched["objects"]+fetched.get("deferred_objects", [])}
    native_asos = ASOS.find()
    if native_asos is None:
        raise ValueError("native rw_asos is required")
    bbox = ",".join(str(v) for v in case["bbox"])
    table_path = out / "stations.json"
    networks = surface_networks_argument(case_surface_networks(case))
    station_ready = Future()
    from gpuwm.obs.coverage import read_site_table
    table = read_site_table(binary)
    native_threads = native_observation_threads(len(case['radar_sites']), len(analyses), workers)
    retained = observation_memory(grid, [table[site] for site in case['radar_sites']], params)

    def surface_slot(index, analysis, work):
        with observation_stage('prep.obs.surface', out, slot=index):
            if index == 0:
                try:
                    run_json([str(native_asos), 'stations', '--networks', networks,
                              '--bbox', bbox, '--out', str(table_path)], out/'logs/stations.txt')
                except BaseException as error:
                    station_ready.set_exception(error)
                    raise
                station_ready.set_result(True)
            else:
                station_ready.result()
            csv = work / 'surface.csv'
            seam = work / 'surface.json'
            before = analysis - dt.timedelta(seconds=900)
            format_asos = lambda stamp: stamp.strftime('%Y-%m-%dT%H:%M:%S')
            run_json([str(native_asos), 'fetch', '--stations', str(table_path), '--start', format_asos(before),
                      '--end', format_asos(analysis), '--out', str(csv), '--product', 'metar'], work/'surface-fetch.txt')
            run_json([str(native_asos), 'decode', '--stations', str(table_path), '--obs', str(csv),
                      '--start', format_asos(analysis), '--end', format_asos(analysis), '--match-seconds', '900',
                      '--min-report-rate', '0.8', '--out', str(seam), '--product', 'metar'], work/'surface-decode.txt')
            validate_surface(json.loads(seam.read_text()), analysis)
            run_json([str(native_asos), 'verify', '--file', str(seam)], work/'surface-verify.txt')
            return seam

    def radar_site(index, site, metadata, analysis, work):
        with observation_stage('prep.obs.radar', out, slot=index, site=site):
            return _site_volume(
                site, metadata['radar_candidates'][site], raw, case, analysis, work, binary, grid, params,
                run_decode, run_verify, read_sweep_pack, superob_volume)

    def slot_work(item):
        index, (metadata, analysis) = item
        with observation_stage('prep.obs.slot', out, slot=index):
            return build_slot(index, metadata, analysis)

    def build_slot(index, metadata, analysis):
        if metadata["analysis_time"] != iso(analysis):
            raise ValueError("inventory slot clock differs from case")
        contributions, sources, refusals = [], [], []
        work = out / f"slot-{index:02d}"
        work.mkdir(exist_ok=True)
        # Sites are independent until the merge: decode is a subprocess and the
        # superob/dealias kernels are native, so threads overlap them.  Results
        # are merged in roster order, so the merged grid does not depend on
        # which site finished first.
        surface_job = initial_surface if index == 0 else surface_pool.submit(surface_slot, index, analysis, work)
        jobs = [initial_radar if index == 0 and site == case['radar_sites'][0]
                else radar_pool.submit(radar_site, index, site, metadata, analysis, work)
                for site in case['radar_sites']]
        for job in jobs:
            contribution, source, site_refusals = job.result()
            refusals.extend(site_refusals)
            if contribution is not None:
                contributions.append(contribution)
                sources.append(source)
        save(work/"radar-attempts.json", {"sources": sources, "refusals": refusals})
        if len(contributions) < case.get("min_radars", 2):
            raise ValueError("insufficient complete causal radar contributions; see slot refusals " + repr(refusals))
        obs = work / "radar.nc"
        context = {"builder": "tools/da_recent_observations.py", "requested_valid_time": iso(analysis),
                   "sources": sources, "refusals": refusals, "publication_policy": inv["publication_policy"],
                   "causal_measurement_cutoff_utc": iso(analysis)}
        with observation_stage('prep.obs.merge', out, slot=index):
            receipt = write_radar_grid(obs, merge_contributions(contributions, grid, params=params), grid,
                                      valid_time=iso(analysis), params=params, provenance=context)
        save(work/"radar-receipt.json", receipt)
        seam = surface_job.result()
        return {"obs": str(obs), "grid_wrfout": str(grid_path), "leg_seconds": 3600,
                      "analysis_time": iso(analysis), "obs_sha256": sha(obs),
                      "grid_wrfout_sha256": sha(grid_path), "surface_seam": str(seam),
                      "surface_sha256": sha(seam), "radar_sources": sources, "refusals": refusals}

    slots, surface_paths = [], []
    with ThreadPoolExecutor(max_workers=len(analyses)) as surface_pool:
        initial_work = out / 'slot-00'
        initial_work.mkdir(exist_ok=True)
        # This required site is processed once and reused. A first-volume RSS
        # measurement can increase the conservative scratch allowance before
        # any site fan-out. ASOS already runs beside the calibration volume.
        observation_workers(len(case['radar_sites']), retained_slot_bytes=retained)
        initial_surface = surface_pool.submit(surface_slot, 0, analyses[0], initial_work)
        with ObservationRss() as calibration:
            first = radar_site(0, case['radar_sites'][0], inv['slots'][0], analyses[0], initial_work)
        initial_radar = Future()
        initial_radar.set_result(first)
        first = None
        scratch = max(2 * 2**30, calibration.receipt()['incremental_peak_bytes'] * 2)
        sizing = observation_workers(len(case['radar_sites']), len(analyses),
                                     requested=workers, worker_bytes=scratch,
                                     retained_slot_bytes=retained)
        sizing['calibration'] = calibration.receipt()
        sizing['native_threads'] = native_threads
        sizing['calibration_policy'] = 'first required site reused; twice measured incremental RSS, minimum 2 GiB'
        save(out / 'worker-receipt.json', sizing)
        with ThreadPoolExecutor(max_workers=sizing['total_workers']) as radar_pool, \
                ThreadPoolExecutor(max_workers=sizing['slot_workers']) as slot_pool:
            for slot in slot_pool.map(slot_work, enumerate(zip(inv['slots'], analyses))):
                slots.append(slot)
                if len(slots) == 1:
                    # The calibration contribution has already been merged.
                    # Its completed Future must not retain it across later slots.
                    initial_radar = None
                surface_paths.append(slot['surface_seam'])
                save(out/'slots.json', {'schema': 'da-rerun.recent-slots.v1', 'status': 'preparing',
                                       'slots': slots, 'surface_seams': surface_paths})
    # Merge metadata rows only. The native decoder owns all quantities and QC.
    combined = combine_surfaces([json.loads(Path(path).read_text()) for path in surface_paths])
    source_manifest = out/"surface-source-manifest.json"
    save(source_manifest, {"native_seams": [{"path": path, "sha256": sha(path)} for path in surface_paths],
                           "native_csv": [{"path": str(Path(path).with_name("surface.csv")),
                                          "sha256": sha(Path(path).with_name("surface.csv"))} for path in surface_paths]})
    combined["provenance"].update(uri=str(source_manifest), sha256=sha(source_manifest),
                                  product="hourly-causal-metar-union")
    save(out/"surface.json", combined)
    run_json([str(native_asos), "verify", "--file", str(out/"surface.json")], out/"logs/surface-union-verify.txt")
    doc = {"schema": "da-rerun.recent-slots.v1", "status": "prepared-smoke-first-slot" if smoke else "prepared", "slots": slots,
           "scope": inv.get("scope", "full-case"), "requested_full_case_fork_utc": iso(utc(case["analysis_times_utc"][-1])),
           "surface_obs": str(out/"surface.json"), "surface_sha256": sha(out/"surface.json"),
           "station_table_sha256": sha(table_path), "grid_wrfout_sha256": sha(grid_path),
           "model_start_utc": iso(origin), "forecast_fork_utc": iso(analyses[-1]),
           "publication_policy": inv["publication_policy"], "scientific_gate": "pending"}
    if fetched.get("schema") == "da-rerun.recent-fetch.v2":
        landed = [row for row in fetched["deferred_objects"] if "sha256" in row]
        fetched["objects"].extend(landed)
        fetched["deferred_objects"] = [row for row in fetched["deferred_objects"] if "sha256" not in row]
        save(fetched["receipt_path"], fetched)
    doc.update(inventory_sha256=document_sha(inv), fetch_receipt_sha256=document_sha(fetched),
               observation_builder_sha256=sha(Path(__file__)), superob_parameters=params.to_payload(),
               native_radar_binary_sha256=sha(binary), native_surface_binary_sha256=sha(native_asos))
    save(out/"slots.json", doc)
    return doc


def prepare_truth(inv, fetched, out, center_lon, center_lat):
    from gpuwm.obs.frontdoor import MRMS
    from tools.regional_rain_manifest import truth as build_truth
    from types import SimpleNamespace
    if fetched.get("schema") == "da-rerun.recent-fetch.v2":
        from tools.da_object_fetch import truth_receipt
        fetched = truth_receipt(inv, fetched, wait=True)
    validate_fetched(inv, fetched, purpose="truth")
    if inv.get("scope") == "smoke-first-slot":
        raise ValueError("first-slot smoke has no six-hour truth inventory")
    out = Path(out).resolve()
    binary = MRMS.require()
    dirs = {product: out / label for product, label in zip(TRUTH_PRODUCTS, ("rate", "echo", "rqi"))}
    truth_bounds = truth_bbox(inv["case"])
    bbox = ",".join(str(value) for value in truth_bounds)
    first_rate = None
    receipts, jobs, records = [], [], []
    for row in fetched["objects"]:
        if row["bucket"] != "noaa-mrms-pds":
            continue
        product = row["key"].split("/")[1]
        if product not in dirs:
            continue
        folder = dirs[product]
        folder.mkdir(parents=True, exist_ok=True)
        pack = folder / (Path(row["key"]).name + ".obspack")
        jobs.append(['decode', '--product', product, '--file', row['path'],
                     '--bbox', bbox, '--out', str(pack)])
        if row.get('fetched_at_utc'):
            jobs[-1] += ['--fetched-at', row['fetched_at_utc']]
        records.append((row, pack))
        if product == TRUTH_PRODUCTS[0] and first_rate is None:
            first_rate = row["path"]
    if first_rate is None:
        raise ValueError("truth inventory contains no primary rate product")
    geo = out/"grid.geopack"
    jobs.append(['grid', '--product', TRUTH_PRODUCTS[0], '--file', first_rate,
                 '--bbox', bbox, '--out', str(geo)])
    batch_path = out / 'decode-jobs.json'
    save(batch_path, jobs)
    batch, elapsed = run_json([str(binary), 'decode-batch', '--files', str(batch_path)],
                               out / 'logs/decode-batch.txt')
    if (batch.get('schema') != 'gpuwm-obs.mrms-decode-batch.v1'
            or batch.get('status') != 'COMPLETE' or len(batch.get('records', [])) != len(jobs)):
        raise ValueError('native MRMS batch did not complete every pinned product; partial truth cannot be scored')
    if (batch['records'][-1].get('out') != str(geo)
            or batch['records'][-1].get('operation') != 'grid'):
        raise ValueError('native MRMS batch geometry mismatch; wrong coordinates cannot be scored')
    for index, ((row, pack), native) in enumerate(zip(records, batch['records'])):
        if native.get('index') != index or native.get('out') != str(pack):
            raise ValueError('native MRMS batch order/output mismatch; prevents binding the wrong product to truth')
        receipts.append({'key': row['key'], 'source_sha256': row['sha256'], 'pack_sha256': sha(pack),
                         'native_receipt': native['native_receipt'],
                         'decode_wall_seconds': native['decode_wall_seconds']})
    manifest = build_truth(SimpleNamespace(rate_packs=dirs[TRUTH_PRODUCTS[0]], echo_packs=dirs[TRUTH_PRODUCTS[1]],
        geo_pack=geo, analysis_end=inv["case"]["analysis_times_utc"][-1], center_lon=center_lon, center_lat=center_lat))
    manifest["inventory_sha256"] = document_sha(inv)
    manifest["fetch_receipt_sha256"] = document_sha(fetched)
    manifest["rqi_sensitivity_packs"] = str(dirs[TRUTH_PRODUCTS[2]])
    manifest["rqi_policy"] = "native RQI retained separately; no unstated quality threshold applied"
    manifest["truth_bbox"] = truth_bounds
    manifest["truth_padding_m"] = inv["case"]["truth_padding_m"]
    manifest["footprint_coverage_policy"] = "native observed footprint before model intersection; cropped or uncovered objects remain pending"
    save(out/"truth.json", manifest)
    save(out/"decode-receipts.json", {"status": "decoded-primary-fields", "records": receipts,
                                     "scientific_gate": "pending", "native_binary_sha256": sha(binary),
                                     'native_processes': 1, 'batch_wall_seconds': elapsed})
    return manifest


def validate_surface(surface, analysis):
    if (surface.get("schema") != "gpuwm-obs.asos-surface.v2" or surface.get("status") != "READY"
            or surface.get("provenance", {}).get("is_stub") or not surface.get("reports")):
        raise ValueError("native surface seam has no v2 observation clocks/reports")
    for report in surface["reports"]:
        observed = seam_utc(report["observation_time"])
        if observed > analysis or (analysis-observed).total_seconds() > 900:
            raise ValueError("surface report is future or stale")
        if seam_utc(report["valid_time"]) != analysis:
            raise ValueError("surface valid-time slot differs from analysis")


def combine_surfaces(records):
    first = records[0]
    if any(record["station_table_sha256"] != first["station_table_sha256"] for record in records):
        raise ValueError("surface station identity changed between hourly slots")
    combined = dict(first)
    combined["valid_times"] = [stamp for record in records for stamp in record["valid_times"]]
    combined["reports"] = [row for record in records for row in record["reports"]]
    stations = {}
    for record in records:
        for station in record["stations"]:
            old = stations.setdefault(station["station_id"], station)
            if old != station:
                raise ValueError("surface station metadata changed between hourly slots")
    combined["stations"] = [stations[key] for key in sorted(stations)]
    combined["min_report_rate"] = 0.0
    combined["screen"] = dict(first["screen"])
    for name in ("dewpoint_above_temperature_drops", "range_drops"):
        combined["screen"][name] = sum(record["screen"][name] for record in records)
    for name in ("stations_dropped_by_screen", "stations_dropped_by_completeness"):
        combined["screen"][name] = sorted({station for record in records for station in record["screen"][name]})
    keys = [(row["station_id"], row["observation_time"]) for row in combined["reports"]]
    if len(keys) != len(set(keys)):
        raise ValueError("surface report reused across analyses")
    combined["provenance"] = {**first["provenance"], "builder": "tools/da_recent_observations.py",
                              "hourly_native_provenance": [record["provenance"] for record in records],
                              "completeness_policy": "native QC per hourly slot; station union has no across-window completeness claim",
                              "publication_policy": "archive replay; original first availability unknown"}
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    audit = sub.add_parser("inventory")
    audit.add_argument("--case", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--first-slot-only", action="store_true", help="smoke only: first hourly radar slot; no model or verification downloads")
    audit.add_argument("--model-requests", type=Path, help="explicit public bucket/key request list to include in this one fetch inventory")
    get = sub.add_parser("fetch")
    get.add_argument("--inventory", type=Path, required=True)
    get.add_argument("--root", type=Path, required=True)
    get.add_argument("--receipt", type=Path, required=True)
    get.add_argument("--max-gib", type=float, default=40)
    get.add_argument("--reuse-receipt", type=Path, help="reuse only unchanged raw files from this COMPLETE ancestor receipt")
    get.add_argument("--streams", type=int, default=1, help="compatibility value; Rust uses min(32, objects) streams")
    prep = sub.add_parser("prepare")
    prep.add_argument("--inventory", type=Path, required=True)
    prep.add_argument("--fetch-receipt", type=Path, required=True)
    prep.add_argument("--grid-wrfout", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--workers", type=int, default=None, help="native site workers (default: site count bounded by available RAM)")
    prep.add_argument('--truth-out', type=Path, help='background truth directory (default: <out>/truth); scorer joins this directory')
    truth = sub.add_parser("truth")
    truth.add_argument("--inventory", type=Path, required=True)
    truth.add_argument("--fetch-receipt", type=Path, required=True)
    truth.add_argument("--out", type=Path, required=True)
    truth.add_argument("--center-lon", type=float, required=True)
    truth.add_argument("--center-lat", type=float, required=True)
    truth.add_argument('--background-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.mode == "inventory":
        requests = json.loads(args.model_requests.read_text(encoding="utf-8-sig")) if args.model_requests else None
        result = inventory(json.loads(args.case.read_text(encoding="utf-8-sig")),
                           first_slot_only=args.first_slot_only, model_requests=requests,
                           on_object=lambda row: print(json.dumps({"phase": "object", **row}), flush=True))
        save(args.out, result)
        print(json.dumps({"objects": len(result["objects"]), "bytes": result["bytes"], "out": str(args.out)}))
    elif args.mode == "fetch":
        if args.reuse_receipt and args.reuse_receipt.resolve() == args.receipt.resolve():
            raise ValueError("new fetch receipt must preserve its COMPLETE ancestor file")
        ancestor = json.loads(args.reuse_receipt.read_text(encoding="utf-8-sig")) if args.reuse_receipt else None
        fetch(json.loads(args.inventory.read_text()), args.root, args.receipt, args.max_gib,
              reuse_receipt=ancestor, reuse_receipt_sha256=sha(args.reuse_receipt) if args.reuse_receipt else None,
              streams=args.streams)
    elif args.mode == "prepare":
        result = prepare(json.loads(args.inventory.read_text()), json.loads(args.fetch_receipt.read_text()), args.grid_wrfout, args.out,
                         workers=args.workers, truth_out=args.truth_out)
        print(json.dumps({"status": result["status"], "slots": len(result["slots"])}))
    else:
        if (args.out / 'truth-job.json').exists() and not args.background_worker:
            from tools.da_object_fetch import fetch_binding
            identity = json.loads((args.out / 'truth-job.json').read_text())
            expected = {'inventory_sha256': document_sha(json.loads(args.inventory.read_text())),
                'fetch_receipt_sha256': fetch_binding(json.loads(args.fetch_receipt.read_text())),
                'center_lon': args.center_lon, 'center_lat': args.center_lat}
            if identity != expected:
                raise ValueError('background truth identity differs from scoring request; prevents stale truth scoring')
            result = join_truth(args.out)
        else:
            try:
                with observation_stage('prep.obs.truth', args.out):
                    result = prepare_truth(json.loads(args.inventory.read_text()), json.loads(args.fetch_receipt.read_text()),
                                           args.out, args.center_lon, args.center_lat)
                if args.background_worker:
                    _truth_status(args.out, 'COMPLETE', truth_sha256=sha(args.out / 'truth.json'))
            except Exception as error:
                if args.background_worker:
                    _truth_status(args.out, 'FAILED', error=f'{type(error).__name__}: {error}')
                raise
        print(json.dumps({"status": "decoded-primary-fields", "truth": str(args.out/"truth.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
