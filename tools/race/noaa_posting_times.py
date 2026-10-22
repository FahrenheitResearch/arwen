#!/usr/bin/env python3
"""Measure posting timestamps using public listings only; never download GRIBs.

Default: seven complete UTC days ending yesterday. --today collects the partial
current day. Output includes absent objects explicitly, raw HTTP evidence,
per-file rows, and paired race-window statistics. Sources are declarative JSON.
NOMADS listing times have minute precision and are interpreted as UTC. Their
mtimes are not first-observed HTTP availability; no polling history is inferred.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import statistics
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, unquote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

UTC = timezone.utc
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def stamp(value):
    return value.isoformat().replace("+00:00", "Z")


def parse(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def minutes(later, earlier):
    return (parse(later) - parse(earlier)).total_seconds() / 60 if later and earlier else None


def stats(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


class Listings:
    def __init__(self, raw, timeout):
        self.raw, self.timeout = raw, timeout
        raw.mkdir(parents=True, exist_ok=True)

    def fetch(self, url, suffix):
        name = hashlib.sha256(url.encode()).hexdigest() + suffix
        start = datetime.now(UTC)
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "posting-times/1.0"}),
                             timeout=self.timeout) as response:
                    body = response.read()
                    status, headers = response.status, dict(response.headers)
                break
            except HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    body, status, headers = exc.read(), exc.code, dict(exc.headers)
                    break
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        (self.raw / name).write_bytes(body)
        receipt = {"url": url, "status": status, "headers": headers,
                   "started_at": stamp(start), "finished_at": stamp(datetime.now(UTC)),
                   "body_sha256": hashlib.sha256(body).hexdigest(), "body_file": name}
        (self.raw / (name + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
        return body, receipt

    def objects(self, source, prefix):
        objects, receipts, token = {}, [], None
        while True:
            query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                query["continuation-token"] = token
            url = f'https://{source["bucket"]}.s3.amazonaws.com/?{urlencode(query)}'
            body, receipt = self.fetch(url, ".xml")
            receipts.append(receipt)
            if receipt["status"] != 200:
                raise RuntimeError(f'{url}: HTTP {receipt["status"]}')
            root = ET.fromstring(body)
            for item in root.findall("s:Contents", NS):
                objects[item.findtext("s:Key", namespaces=NS)] = {
                    "mtime": item.findtext("s:LastModified", namespaces=NS),
                    "size": int(item.findtext("s:Size", namespaces=NS)),
                    "raw_file": receipt["body_file"]}
            if root.findtext("s:IsTruncated", namespaces=NS) != "true":
                return objects, receipts
            token = root.findtext("s:NextContinuationToken", namespaces=NS)
            if not token:
                raise RuntimeError("Truncated S3 listing without continuation token")

    def directory(self, url):
        body, receipt = self.fetch(url, ".html")
        entries = {}
        if receipt["status"] == 200:
            text = body.decode("utf-8")
            pattern = r'<a\s+href="([^"]+)"[^>]*>.*?</a>\s+(\d{2}-[A-Za-z]{3}-\d{4}\s+\d{2}:\d{2})'
            for name, mtime in re.findall(pattern, text):
                entries[unquote(html.unescape(name))] = stamp(
                    datetime.strptime(mtime, "%d-%b-%Y %H:%M").replace(tzinfo=UTC))
        return entries, receipt


def summarize(rows, sources):
    analysis, boundary = sources["analysis"], sources["boundary"]
    lookup = {(r["model"], r["cycle"], r["product"], r["lead"]): r for r in rows}
    cases = []
    for row in rows:
        cycle = parse(row["cycle"])
        if (row["model"] != analysis["model"] or row["product"] != analysis["initial_product"]
                or row["lead"] != 0 or cycle.hour not in analysis["extended_cycle_hours"]):
            continue
        last = [lookup.get((analysis["model"], row["cycle"], p, analysis["extended_lead"]))
                for p in analysis["products"]]
        available = [r for r in last if r and r["s3_last_modified"]]
        finish = min(available, key=lambda r: r["s3_last_modified"]) if available else None
        bc = stamp(cycle - timedelta(hours=boundary["lag_hours"]))
        frames = [lookup.get((boundary["model"], bc, boundary["products"][0], lead))
                  for lead in range(boundary["last_lead"] + 1)]
        needed = frames[boundary["lag_hours"]:]
        ready = (max(r["s3_last_modified"] for r in needed)
                 if all(r and r["s3_last_modified"] for r in needed) else None)
        ready_idx = (max(max(r["s3_last_modified"], r["s3_index_last_modified"]) for r in needed)
                     if all(r and r["s3_last_modified"] and r["s3_index_last_modified"] for r in needed)
                     else None)
        f51 = frames[-1]
        case = {"cycle": row["cycle"], "cycle_hour": cycle.hour,
                "f00": row["s3_last_modified"], "f00_minutes": row["s3_minutes_after_cycle"],
                "f48": finish["s3_last_modified"] if finish else None,
                "f48_product": finish["product"] if finish else None,
                "f48_minutes": finish["s3_minutes_after_cycle"] if finish else None,
                "window_minutes": minutes(finish["s3_last_modified"], row["s3_last_modified"]) if finish else None,
                "boundary_cycle": bc,
                "boundary_f51": f51["s3_last_modified"] if f51 else None,
                "boundary_f51_relative_f00_minutes": minutes(f51["s3_last_modified"], row["s3_last_modified"]) if f51 else None,
                "boundary_complete": ready, "boundary_complete_with_indexes": ready_idx,
                "boundary_complete_relative_f00_minutes": minutes(ready, row["s3_last_modified"]),
                "boundary_indexes_relative_f00_minutes": minutes(ready_idx, row["s3_last_modified"]),
                "boundary_missing_leads": [i for i, r in enumerate(frames) if not r or not r["s3_last_modified"]]}
        cases.append(case)
    by_hour = {}
    for hour in analysis["extended_cycle_hours"]:
        paired = [c for c in cases if c["cycle_hour"] == hour and c["window_minutes"] is not None]
        by_hour[f"{hour:02d}"] = {name: stats(c[name] for c in paired) for name in (
            "f00_minutes", "f48_minutes", "window_minutes", "boundary_f51_relative_f00_minutes",
            "boundary_complete_relative_f00_minutes", "boundary_indexes_relative_f00_minutes")}
    return {"deadline": "earliest S3 f48 among listed products; paired cycles only",
            "boundary_required_leads": [boundary["lag_hours"], boundary["last_lead"]],
            "by_cycle_hour": by_hour, "cycles": cases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--end-date", type=date.fromisoformat, help="inclusive UTC day; default yesterday")
    p.add_argument("--today", action="store_true", help="current partial UTC day only")
    p.add_argument("--output", type=Path, default=Path("posting-times.json"))
    p.add_argument("--sources", type=Path, default=Path(__file__).with_name("noaa-posting-sources.json"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=30)
    args = p.parse_args()
    if args.days < 1 or args.workers < 1 or args.timeout <= 0:
        p.error("days, workers and timeout must be positive")
    if args.today and args.end_date:
        p.error("--today and --end-date are mutually exclusive")
    started = datetime.now(UTC)
    end = started.date() if args.today else args.end_date or started.date() - timedelta(days=1)
    days = 1 if args.today else args.days
    start = end - timedelta(days=days - 1)
    sources = json.loads(args.sources.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_name(args.output.stem + "-raw")
    listing = Listings(raw, args.timeout)
    targets, tasks, errors, requests = [], {}, [], []
    analysis, boundary = sources["analysis"], sources["boundary"]
    for offset in range(days * 24):
        cycle = datetime.combine(start, datetime.min.time(), UTC) + timedelta(hours=offset)
        if cycle > started:
            continue
        for role in ("analysis", "boundary"):
            source = sources[role]
            if role == "boundary" and cycle.hour not in analysis["extended_cycle_hours"]:
                continue
            source_cycle = cycle - timedelta(hours=boundary["lag_hours"]) if role == "boundary" else cycle
            directory = source["directory"].format(date=source_cycle.strftime("%Y%m%d"))
            prefix = directory + source["prefix"].format(hour=source_cycle.hour)
            tasks[(role, stamp(source_cycle))] = (source, prefix)
            leads = (range(source["last_lead"] + 1) if role == "boundary" else
                     source["leads"] + ([source["extended_lead"]]
                     if cycle.hour in source["extended_cycle_hours"] else []))
            for product in source["products"]:
                for lead in leads:
                    name = source["file"].format(hour=source_cycle.hour, product=product, lead=lead)
                    targets.append({"model": source["model"], "role": role, "cycle": stamp(source_cycle),
                                    "product": product, "lead": lead, "file": name,
                                    "key": directory + name, "directory": directory})
    objects = {}
    stage_start = stamp(datetime.now(UTC))
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(listing.objects, s, prefix): key for key, (s, prefix) in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                objects[key], receipts = future.result()
                requests.extend(receipts)
            except Exception as exc:
                errors.append({"stage": "s3", "task": key, "error": str(exc)})
    stages = [{"stage": "s3_listings", "started_at": stage_start, "finished_at": stamp(datetime.now(UTC))}]
    directories, roots = {}, {}
    stage_start = stamp(datetime.now(UTC))
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(listing.directory, s["nomads_root"]): role for role, s in sources.items()}
        for future in as_completed(futures):
            role = futures[future]
            try:
                roots[role], receipt = future.result()
                requests.append(receipt)
                if receipt["status"] != 200:
                    errors.append({"stage": "nomads_root", "task": role, "error": f'HTTP {receipt["status"]}'})
            except Exception as exc:
                errors.append({"stage": "nomads_root", "task": role, "error": str(exc)})
        wanted = {(r["role"], r["directory"]) for r in targets}
        futures = {pool.submit(listing.directory, sources[role]["nomads_root"] + directory): (role, directory)
                   for role, directory in wanted if directory.split("/")[0] + "/" in roots.get(role, {})}
        for future in as_completed(futures):
            key = futures[future]
            try:
                directories[key], receipt = future.result()
                requests.append(receipt)
                if receipt["status"] != 200:
                    errors.append({"stage": "nomads_directory", "task": key, "error": f'HTTP {receipt["status"]}'})
            except Exception as exc:
                errors.append({"stage": "nomads_directory", "task": key, "error": str(exc)})
    stages.append({"stage": "nomads_listings", "started_at": stage_start, "finished_at": stamp(datetime.now(UTC))})
    for row in targets:
        key = (row["role"], row["cycle"])
        entry = objects.get(key, {}).get(row["key"], {})
        index = objects.get(key, {}).get(row["key"] + ".idx", {})
        row["s3_last_modified"] = entry.get("mtime")
        row["s3_index_last_modified"] = index.get("mtime")
        row["s3_size"] = entry.get("size")
        row["s3_raw_file"] = entry.get("raw_file")
        row["s3_status"] = "present" if entry else "absent" if key in objects else "listing_error"
        row["nomads_mtime"] = directories.get((row["role"], row["directory"]), {}).get(row["file"])
        row["nomads_index_mtime"] = directories.get((row["role"], row["directory"]), {}).get(row["file"] + ".idx")
        root_known = row["role"] in roots
        day_present = row["directory"].split("/")[0] + "/" in roots.get(row["role"], {})
        row["nomads_status"] = ("present" if row["nomads_mtime"] else
                                "day_not_listed" if root_known and not day_present else
                                "absent" if (row["role"], row["directory"]) in directories else "listing_error")
        row["s3_minutes_after_cycle"] = minutes(row["s3_last_modified"], row["cycle"])
        row["nomads_minutes_after_cycle"] = minutes(row["nomads_mtime"], row["cycle"])
        row["s3_minus_nomads_minutes"] = minutes(row["s3_last_modified"], row["nomads_mtime"])
    targets.sort(key=lambda r: (r["cycle"], r["model"], r["file"]))
    result = {"schema_version": 1, "started_at": stamp(started), "finished_at": stamp(datetime.now(UTC)),
              "start_date": start.isoformat(), "end_date": end.isoformat(), "partial_today": args.today,
              "sources": sources, "raw_directory": raw.name,
              "semantics": {"s3": "ListObjectsV2 LastModified, UTC, current object version",
                            "nomads": "directory mtime interpreted as UTC, minute precision; not first-seen time",
                            "missing": "null is unobserved, never a schedule estimate",
                            "hydro": "not applicable: listings only, no forecast"},
              "stages": stages, "requests": requests, "errors": errors, "rows": targets,
              "summary": summarize(targets, sources)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(targets), "errors": errors,
                      "windows": result["summary"]["by_cycle_hour"]}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
