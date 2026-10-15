"""Read-only project storage preflight. Limits are decimal bytes, configurable per project.

Public JSON counts all files in all releases, including yanked files. An exact
existing file costs no additional space on a retry. Invalid or unavailable
inventory refuses; no credentials, upload or deletion is used here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

PROJECTS = ("gpuwm", "gpuwm-data")
LIMITS = Path(__file__).with_name("pypi_limits.json")


class CapacityError(ValueError):
    pass


def positive_bytes(value, label):
    if type(value) is not int or value < 0:
        raise CapacityError(f"{label}: expected nonnegative integer bytes, got {value!r}")
    return value


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def inventory(document, project):
    if normalize(document.get("info", {}).get("name", "")) != project:
        raise CapacityError(f"{project}: public JSON names a different project")
    releases = document.get("releases")
    if not isinstance(releases, dict):
        raise CapacityError(f"{project}: public JSON has no complete releases inventory")
    files = {}
    for version, rows in releases.items():
        if not isinstance(rows, list):
            raise CapacityError(f"{project}: invalid release {version}")
        for row in rows:
            try:
                name, sha = row["filename"], row["digests"]["sha256"]
                size = positive_bytes(row["size"], f"{project}/{name}")
            except (KeyError, TypeError) as exc:
                raise CapacityError(f"{project}: incomplete file inventory") from exc
            if not isinstance(name, str) or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise CapacityError(f"{project}: invalid file identity")
            identity = (size, sha)
            if name in files and files[name] != identity:
                raise CapacityError(f"{project}: conflicting public file {name}")
            files[name] = identity
    return files


def evaluate(project, document, new_files, limit):
    limit = positive_bytes(limit, f"{project} limit")
    if not limit:
        raise CapacityError(f"{project}: limit must be positive")
    files = inventory(document, project)
    added, seen = 0, set()
    for row in new_files:
        name = row["filename"]
        if name in seen:
            raise CapacityError(f"{project}: duplicate proposed file {name}")
        seen.add(name)
        size = positive_bytes(row["bytes"], f"{project}/{name}")
        sha = row["sha256"]
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise CapacityError(f"{project}: invalid proposed hash for {name}")
        if name in files:
            if files[name] != (size, sha):
                raise CapacityError(f"{project}: existing file differs: {name}")
        else:
            added += size
    current = sum(size for size, _ in files.values())
    projected = current + added
    return {"project": project, "current_bytes": current, "new_bytes": added,
            "projected_bytes": projected, "limit_bytes": limit,
            "headroom_bytes": limit - projected, "existing_files": len(files),
            "releases": len(document["releases"]), "status": "PASS" if projected <= limit else "REFUSED"}


def fetch_project(project):
    if project not in PROJECTS:
        raise CapacityError(f"unknown project {project}")
    url = f"https://pypi.org/pypi/{project}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except Exception as exc:
        raise CapacityError(f"{project}: cannot read complete public JSON inventory: {exc}") from exc


def refusal(row):
    return (f"REFUSED: {row['project']} project storage: current={row['current_bytes']} bytes + "
            f"new={row['new_bytes']} bytes = {row['projected_bytes']} bytes; "
            f"limit={row['limit_bytes']} bytes; exceeds by {-row['headroom_bytes']} bytes")


def check_plan(plan, *, fetch=fetch_project, limits=None):
    limits = json.loads(LIMITS.read_text(encoding="utf-8")) if limits is None else limits
    rows = [evaluate(project, fetch(project), plan[project], limits[project]) for project in PROJECTS]
    return {"schema": "arwen.pypi-capacity.v1", "checked_at": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "REFUSED", "projects": rows}


def packet_plan(packet):
    manifest = packet / "PUBLICATION-ASSETS.json"
    data = manifest.read_bytes()
    rows = json.loads(data)["pypi"]["artifacts"]
    plan = {project: [] for project in PROJECTS}
    for row in rows:
        name = row["filename"]
        if Path(name).name != name or not name.startswith(("gpuwm-", "gpuwm_data-")):
            raise CapacityError(f"unexpected PyPI packet filename {name}")
        project = "gpuwm-data" if name.startswith("gpuwm_data-") else "gpuwm"
        path = packet / name
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != row["bytes"] or sha != row["sha256"]:
            raise CapacityError(f"{project}: packet file differs: {name}")
        plan[project].append(row)
    if not all(plan.values()):
        raise CapacityError("packet must include both gpuwm and gpuwm-data")
    return plan, hashlib.sha256(data).hexdigest()


def check_packet(packet, *, fetch=fetch_project, limits=None, out=None):
    plan, manifest_sha = packet_plan(packet)
    receipt = check_plan(plan, fetch=fetch, limits=limits)
    receipt["manifest_sha256"] = manifest_sha
    if out:
        out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    failed = [refusal(row) for row in receipt["projects"] if row["status"] != "PASS"]
    if failed:
        raise CapacityError("\n".join(failed))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--limits", type=Path, default=LIMITS)
    parser.add_argument("--fixture-dir", type=Path, help="recorded public JSON, never a live approval")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    fetch = (lambda p: json.loads((args.fixture_dir / f"{p}.json").read_text())) if args.fixture_dir else fetch_project
    try:
        check_packet(args.packet, fetch=fetch, limits=json.loads(args.limits.read_text()), out=args.out)
    except (CapacityError, KeyError, OSError, ValueError) as exc:
        print(str(exc))
        return 1
    print(f"PASS: both projects fit; receipt {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
