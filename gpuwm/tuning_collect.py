"""Collect tuning histories through the existing native station scorers.

Python checks receipt and filename metadata and schedules native requests.
Rust reads weather arrays, matches observations and computes scores. The
collector reads completed forecasts in place and never deletes histories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

from gpuwm.tuning import digest, read_json, verify_packet, write_json

FRAME = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2})[_T](\d{2})[:_-](\d{2})[:_-](\d{2})(?:\.nc)?$")
SCHEMA = "gpuwm.tuning-collection/v1"


def label(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("member labels must be path-safe ASCII identifiers")
    return value


def expected_times(day, hours):
    start = datetime.strptime(day, "%Y-%m-%d")
    return [(start + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M:%S")
            for hour in range(hours + 1)]


def domain_table(packet):
    """Read the declared grid hierarchy, without opening weather fields."""
    jobs = read_json(Path(packet) / "run-list.json")
    if not jobs:
        raise ValueError("the packet has no forecast jobs")
    config = Path(packet) / jobs[0]["config"]
    with config.open("rb") as stream:
        rows = tomllib.load(stream)["domain"]
    domains = {}
    pending = list(rows)
    while pending:
        progress = False
        for row in list(pending):
            identifier = int(row["grid_id"])
            parent = int(row.get("parent_id", 0))
            if parent and parent not in domains:
                continue
            if identifier <= 0 or identifier > 99 or identifier in domains:
                raise ValueError("invalid or duplicate declared domain")
            if "dx" in row:
                spacing = float(row["dx"])
            elif parent:
                ratio = int(row["parent_grid_ratio"])
                if ratio <= 0:
                    raise ValueError("a child domain has no positive grid ratio")
                spacing = domains[parent]["dx_m"] / ratio
            else:
                raise ValueError("the parent domain has no declared grid spacing")
            if not 0 < spacing < float("inf"):
                raise ValueError("a declared domain has no finite positive spacing")
            domains[identifier] = {"id": f"d{identifier:02d}", "dx_m": spacing}
            pending.remove(row)
            progress = True
        if not progress:
            raise ValueError("the declared domain hierarchy has a missing parent or a cycle")
    ordered = sorted(domains.values(), key=lambda row: row["id"])
    if any(not left["dx_m"] > right["dx_m"] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("domain IDs must have strictly decreasing spacing; the importer would otherwise select a coarser grid as the finest domain")
    return ordered


def history_files(directory, day, domains, hours):
    """Inventory exact hourly filenames. Native scorers prove encoded times."""
    expected = set(expected_times(day, hours))
    result = {row["id"]: {} for row in domains}
    for path in sorted(Path(directory).rglob("wrfout_*")):
        if not path.is_file():
            continue
        match = FRAME.fullmatch(path.name)
        if match is None:
            continue
        domain, date, hour, minute, second = match.groups()
        if minute != "00" or second != "00":
            continue
        valid = f"{date}T{hour}:{minute}:{second}"
        if domain not in result or valid not in expected:
            raise ValueError(f"unexpected hourly history: {path}")
        if valid in result[domain]:
            raise ValueError(f"duplicate {domain} history at {valid}: {path}")
        result[domain][valid] = str(path.resolve())
    for domain, files in result.items():
        missing = sorted(expected - files.keys())
        if missing:
            raise ValueError(f"{directory}: missing {domain} histories {missing}")
    return result


def completed_run(runs, day, member, domains, hours, expected_member=None):
    directory = Path(runs) / day / label(member)
    if (directory / "DONE").read_text(encoding="ascii").strip() != "0":
        raise ValueError(f"{directory} has no successful completion marker")
    receipt = read_json(directory / "tuning-run.json")
    if (receipt.get("day"), receipt.get("member_id"), receipt.get("status"), receipt.get("exit_code")) != (day, member, "succeeded", 0):
        raise ValueError(f"{directory} is not a successful receipt for this member and day")
    if receipt.get("job_sha256") != digest(directory / "job.json"):
        raise ValueError(f"{directory} job bytes differ from the successful run receipt")
    job = read_json(directory / "job.json")
    proof = read_json(directory / "member-receipt.json")
    if job["member"]["id"] != member or proof.get("status") != "PASS" or proof["preflight"]["member"]["id"] != member:
        raise ValueError(f"{directory} has no matching successful member receipt")
    if expected_member is not None:
        for key in ("seed", "product", "physics", "surface", "urban"):
            if job["member"].get(key) != expected_member.get(key):
                raise ValueError(f"{directory} completed {key} differs from the frozen setting")
        expected_set = expected_member.get("physics_params_file")
        actual_set = job["member"].get("physics_params_file")
        if bool(expected_set) != bool(actual_set) or expected_set and digest(expected_set) != digest(actual_set):
            raise ValueError(f"{directory} completed physics parameter bytes differ from the frozen setting")
    return history_files(directory / "run", day, domains, hours)


def surface_record(observations, day, hours):
    root = Path(observations)
    candidates = [root / f"surface-{day}.json", root / f"{day}.json", root / day / "surface.json"]
    found = {path.resolve() for path in candidates if path.is_file()}
    if len(found) != 1:
        raise ValueError(f"{day} needs exactly one decoded surface observation record")
    path = found.pop()
    record = read_json(path)
    if (record.get("schema"), record.get("status"), record.get("provenance", {}).get("is_stub")) != ("gpuwm-obs.asos-surface.v2", "READY", False):
        raise ValueError(f"{path} is not a verified native ASOS record")
    if record.get("valid_times") != expected_times(day, hours):
        raise ValueError(f"{path} does not contain the exact initialization and forecast valid-time table")
    source = Path(record["provenance"]["uri"])
    if digest(source) != record["provenance"]["sha256"]:
        raise ValueError(f"{path} raw observation provenance bytes changed")
    return path, record


def native_environment(output):
    # A unique leaf stays inside the lane's node scratch, never the system /tmp.
    anchor = Path(os.environ.get("TMPDIR", str(Path.home() / ".gpuwm" / "tuning-harness" / "tmp")))
    if anchor.resolve() == Path("/tmp"):
        raise ValueError("native collection needs an owned scratch TMPDIR, not the system /tmp")
    identity = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()[:16]
    temporary = anchor / ("collect-" + identity)
    temporary.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1",
               TMPDIR=str(temporary), PYTHONDONTWRITEBYTECODE="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[name] = "1"
    return env


def native_call(command, log, environment, timeout):
    prefix = ["nice", "-n", "10"] if os.name != "nt" else []
    with Path(log).open("w", encoding="utf-8") as stream:
        result = subprocess.run(prefix + [str(value) for value in command], stdout=stream,
            stderr=subprocess.STDOUT, env=environment, timeout=timeout, check=False)
    if result.returncode:
        raise ValueError(f"native station scoring exited {result.returncode}; see {log}")


def verify_collected_scores(directory, *, round_sha256=None):
    """Bind returned station CSV bytes before submitting them to Rust."""
    root = Path(directory).resolve()
    receipt = read_json(root / "collection-receipt.json")
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "COLLECTED":
        raise ValueError("returned scores have no successful collection receipt")
    if round_sha256 is not None and receipt.get("round_sha256") != round_sha256:
        raise ValueError("returned scores belong to different frozen round bytes")
    request_path = root / "import-request.json"
    if digest(request_path) != receipt["import_request_sha256"]:
        raise ValueError("returned import request bytes changed")
    request = read_json(request_path)
    declared = {row["path"] for row in request["files"]}
    if len(declared) != len(request["files"]) or declared != set(receipt["score_csv_sha256"]):
        raise ValueError("returned import paths differ from the scored CSV manifest")

    def verify(name, expected):
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("returned score paths must stay relative to the collection directory")
        if digest(path) != expected:
            raise ValueError(f"returned score bytes changed: {name}")

    for name, expected in receipt["score_csv_sha256"].items():
        verify(name, expected)
    for row in receipt["station_scores"]:
        if receipt["score_csv_sha256"].get(row["hourly_path"]) != row["hourly_sha256"]:
            raise ValueError("returned score hash tables disagree")
        verify(row["scores_path"], row["scores_sha256"])
        verify(row["request_path"], row["request_sha256"])
    scoreboard_audits = receipt.get("scoreboard_audits", [])
    if receipt.get("scoreboard_audit"):
        scoreboard_audits = [*scoreboard_audits, receipt["scoreboard_audit"]]
    for row in scoreboard_audits:
        verify(row.get("score_path", "scoreboard-audit/score.json"), row["score_sha256"])
    for part in receipt.get("collection_parts", []):
        verify(part["receipt_path"], part["receipt_sha256"])
        part_root = (root / part["receipt_path"]).resolve().parent
        if part_root == root:
            raise ValueError("combined collection parts must have their own subdirectories")
        verify_collected_scores(part_root, round_sha256=receipt["round_sha256"])
    return request


def combine_collections(roots, output, round_sha256, *, expected_days=None):
    """Copy verified station summaries into one portable returned packet.

    Completed histories are not opened. A day can therefore be retired after
    its collection has passed, before the next day consumes disk space.
    """
    roots = [Path(root).resolve() for root in roots]
    output = Path(output).resolve()
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("merge needs distinct successful day-collection directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("merged output already contains artifacts; use a fresh directory")
    if any(output == root or output.is_relative_to(root) for root in roots):
        raise ValueError("merged output must be separate from the source collections")
    parts = []
    days = set()
    for root in roots:
        request = verify_collected_scores(root, round_sha256=round_sha256)
        receipt = read_json(root / "collection-receipt.json")
        if receipt.get("collection_parts"):
            raise ValueError("merge takes original day collections, rather than already merged packets")
        if (root / "DONE").read_text(encoding="ascii").strip() != "0":
            raise ValueError(f"{root} has no successful collection marker")
        part_days = receipt["days"]
        if not part_days or len(set(part_days)) != len(part_days) or days.intersection(part_days):
            raise ValueError("collection parts repeat a day or contain no scored day")
        days.update(part_days)
        if request["expected_members"] != receipt["members"]:
            raise ValueError("a collection's members differ from its import request")
        expected = {(day, domain["id"]) for day in part_days for domain in receipt["domains"]}
        actual = {(row["day"], row["domain"]) for row in request["files"]}
        audits = {(row["day"], row["domain"]) for row in receipt["station_scores"]}
        if (actual != expected or audits != expected or len(actual) != len(request["files"])
                or len(audits) != len(receipt["station_scores"])):
            raise ValueError("a collection is missing a complete day/domain station score")
        parts.append((root, request, receipt))
    if expected_days is not None and days != set(expected_days):
        raise ValueError("collection parts do not cover every sealed training and held-out day")
    first_request, first = parts[0][1:]
    signatures = {key: (value["sha256"], value.get("abi")) for key, value in first["native_executables"].items()}
    for _, request, receipt in parts[1:]:
        signature = {key: (value["sha256"], value.get("abi")) for key, value in receipt["native_executables"].items()}
        if (receipt["members"] != first["members"] or receipt["domains"] != first["domains"]
                or signature != signatures or request["timezone_offset_hours"] != first_request["timezone_offset_hours"]):
            raise ValueError("collection parts use different members, geometry, scorer bytes or local-time rules")
    output.mkdir(parents=True, exist_ok=True)
    files, audits, scoreboard_audits, manifests = [], [], [], []
    for index, (root, request, receipt) in enumerate(parts):
        prefix = Path("parts") / f"{index:03d}"
        needed = {"collection-receipt.json", "import-request.json", "DONE"}
        for row in receipt["station_scores"]:
            needed.update(row[key] for key in ("hourly_path", "scores_path", "request_path"))
            log = Path(row["request_path"]).parent / "native.log"
            if (root / log).is_file():
                needed.add(log.as_posix())
        if (root / "scoreboard-audit").is_dir():
            needed.update(path.relative_to(root).as_posix() for path in (root / "scoreboard-audit").rglob("*") if path.is_file())
        for name in sorted(needed):
            source = (root / name).resolve()
            if not source.is_relative_to(root) or Path(name).is_absolute():
                raise ValueError("a collection artifact lies outside its source directory")
            destination = output / prefix / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        for row in request["files"]:
            files.append({**row, "path": (prefix / row["path"]).as_posix()})
        for row in receipt["station_scores"]:
            audits.append({**row, **{key: (prefix / row[key]).as_posix() for key in ("hourly_path", "scores_path", "request_path")}})
        if receipt.get("scoreboard_audit"):
            row = receipt["scoreboard_audit"]
            scoreboard_audits.append({**row, "score_path": (prefix / row.get("score_path", "scoreboard-audit/score.json")).as_posix()})
        manifests.append({"receipt_path": (prefix / "collection-receipt.json").as_posix(),
                          "receipt_sha256": digest(root / "collection-receipt.json"), "days": receipt["days"]})
    imported = {"files": files, "expected_members": first["members"],
                "timezone_offset_hours": first_request["timezone_offset_hours"]}
    write_json(output / "import-request.json", imported)
    combined = {"schema": SCHEMA, "status": "COLLECTED", "round_sha256": round_sha256,
        "native_executables": first["native_executables"], "domains": first["domains"], "days": sorted(days),
        "members": first["members"], "station_scores": audits, "scoreboard_audits": scoreboard_audits,
        "score_csv_sha256": {row["hourly_path"]: row["hourly_sha256"] for row in audits},
        "import_request_sha256": digest(output / "import-request.json"), "collection_parts": manifests,
        "primary_method": first["primary_method"], "nested_selection": first["nested_selection"],
        "histories_deleted": False, "merge_history_reads": False, "best_maps": "pending frozen selection and Rust renderer"}
    write_json(output / "collection-receipt.json", combined)
    verify_collected_scores(output, round_sha256=round_sha256)
    (output / "DONE").write_text("0\n", encoding="ascii")
    return combined


def collect(packet, runs, observations, scorer, scoreboard, output, *, timezone_offset_hours=None, timeout=1800, days=None):
    if os.name == "nt":
        raise ValueError("actual native weather collection runs on the assigned node or box")
    packet, runs, observations, output = map(lambda value: Path(value).resolve(),
                                            (packet, runs, observations, output))
    if any(output == anchor or output.is_relative_to(anchor) for anchor in (packet, runs, observations)):
        raise ValueError("collection output must be separate from packet, histories and observations")
    if output.exists() and any(output.iterdir()):
        raise ValueError("collection output already contains artifacts; use a fresh directory")
    round_ = verify_packet(packet)
    hours = int(round_["hours"])
    if hours != 24 or int(round_["cycle_hour"]) != 0:
        raise ValueError("collection expects complete f00 to f24 histories of 00 UTC starts")
    offset = timezone_offset_hours if timezone_offset_hours is not None else round_.get("timezone_offset_hours")
    if isinstance(offset, bool) or not isinstance(offset, int) or not -12 <= offset <= 14:
        raise ValueError("declare timezone_offset_hours in the recipe or pass --timezone-offset-hours")
    if timeout <= 0:
        raise ValueError("native scoring needs a positive timeout")
    members = [label(member["id"]) for member in round_["members"]]
    if len(set(members)) != len(members) or round_["baseline_member"] not in members:
        raise ValueError("the packet has duplicate members or no current-product member")
    domains = domain_table(packet)
    sealed_days = round_["train_days"] + round_["heldout_days"]
    if days is None:
        days = sealed_days
    elif not days or len(set(days)) != len(days) or any(day not in sealed_days for day in days):
        raise ValueError("--days must name a unique, nonempty subset of the sealed round days")
    # Complete the inventory before opening an output directory or scoring.
    surface = {day: surface_record(observations, day, hours) for day in days}
    settings = {member["id"]: dict(member) for member in round_["members"]}
    for member in settings.values():
        if member.get("physics_params_file"):
            member["physics_params_file"] = str(packet / member["physics_params_file"])
    histories = {day: {member: completed_run(runs, day, member, domains, hours, settings[member])
                       for member in members} for day in days}
    binaries = {"surface_score": Path(scorer).resolve(), "rw_scoreboard": Path(scoreboard).resolve()}
    identities = {name: {"path": str(path), "sha256": digest(path)} for name, path in binaries.items()}
    output.mkdir(parents=True, exist_ok=True)
    env = native_environment(output)
    abi = subprocess.run([str(binaries["rw_scoreboard"]), "--abi"], capture_output=True,
                         text=True, env=env, timeout=60, check=False)
    if abi.returncode or "scoreboard-v2" not in abi.stdout:
        raise ValueError("the scoreboard executable does not implement the frozen v2 surface method")
    identities["rw_scoreboard"]["abi"] = abi.stdout.strip()
    files, audits = [], []
    try:
        for day in days:
            observation_path, record = surface[day]
            for domain in domains:
                directory = output / "station-scores" / day / domain["id"]
                directory.mkdir(parents=True, exist_ok=True)
                request = {"surface": str(observation_path),
                    "arms": [{"id": member, "files": histories[day][member][domain["id"]]} for member in members],
                    "identical_geography_pairs": [], "provenance": {"round_sha256": digest(packet / "round.json"),
                        "day": day, "domain": domain["id"], "domain_dx_m": domain["dx_m"], "product": round_["product"]}}
                request_path = directory / "request.json"
                write_json(request_path, request)
                native_call([binaries["surface_score"], request_path, directory], directory / "native.log", env, timeout)
                score = read_json(directory / "scores.json")
                if (score.get("schema") != "urban-heat-surface-score-v1" or score.get("status") != "SCORED"
                        or score.get("request_sha256") != digest(request_path)
                        or score.get("surface_sha256") != digest(observation_path)):
                    raise ValueError(f"{directory} native score receipt does not bind this request")
                hourly_path = (directory / "hourly.csv").relative_to(output).as_posix()
                files.append({"path": hourly_path, "day": day,
                              "domain": domain["id"], "method": "bilinear"})
                audits.append({"day": day, "domain": domain["id"], "hourly_path": hourly_path,
                    "request_path": request_path.relative_to(output).as_posix(),
                    "scores_path": (directory / "scores.json").relative_to(output).as_posix(),
                    "request_sha256": digest(request_path),
                    "scores_sha256": digest(directory / "scores.json"), "hourly_sha256": digest(directory / "hourly.csv"),
                    "observations_sha256": digest(observation_path), "raw_observations_sha256": record["provenance"]["sha256"]})
        # Exercise the scoreboard's actual raw-T2 extractor on the finest grid
        # of the current product. The primary scores above retain urban support.
        day, member = days[0], round_["baseline_member"]
        domain = min(domains, key=lambda row: row["dx_m"])["id"]
        valid = expected_times(day, hours)[1]
        observation_path, record = surface[day]
        stations = [{"station_id": row["station_id"], "lat": row["latitude"],
                     "lon": row["longitude"], "elevation_m": row["elevation_m"]} for row in record["stations"]]
        audit_root = output / "scoreboard-audit"
        audit_root.mkdir()
        common = {"schema": "woof.scoreboard.request.v1", "method_id": "scoreboard-v2", "domain": domain,
                  "valid_time": valid + "Z", "stations": stations}
        requests = [dict(common, action="extract", format="wrf", init_time=day + "T00:00:00Z",
                         input=histories[day][member][domain][valid], output=str(audit_root / "forecast.json")),
                    dict(common, action="observe", format="iem_csv", observation_source="surface",
                         input=record["provenance"]["uri"], output=str(audit_root / "observations.json")),
                    {"schema": "woof.scoreboard.request.v1", "method_id": "scoreboard-v2", "action": "score_observations",
                     "forecast": str(audit_root / "forecast.json"), "observations": str(audit_root / "observations.json"),
                     "variable": "t2_raw", "verification_kind": "point", "observation_source": "surface",
                     "output": str(audit_root / "score.json")}]
        for index, request in enumerate(requests):
            path = audit_root / f"{index}.request.json"
            write_json(path, request)
            native_call([binaries["rw_scoreboard"], "--request", path], audit_root / f"{index}.log", env, timeout)
        imported = {"files": files, "expected_members": members, "timezone_offset_hours": offset}
        write_json(output / "import-request.json", imported)
        returned_hashes = {audit["hourly_path"]: audit["hourly_sha256"] for audit in audits}
        receipt = {"schema": SCHEMA, "status": "COLLECTED", "round_sha256": digest(packet / "round.json"),
                   "native_executables": identities, "domains": domains, "days": days, "members": members,
                   "station_scores": audits, "import_request_sha256": digest(output / "import-request.json"),
                   "score_csv_sha256": returned_hashes,
                   "scoreboard_audit": {"day": day, "member": member, "domain": domain, "valid": valid,
                                        "score_path": "scoreboard-audit/score.json", "score_sha256": digest(audit_root / "score.json")},
                   "primary_method": "urban native bilinear; five-cell interior; common land and terrain support across members",
                   "nested_selection": "Rust importer selects the finest common valid domain for each station-hour",
                   "histories_deleted": False, "best_maps": "pending frozen selection and Rust renderer"}
        write_json(output / "collection-receipt.json", receipt)
        (output / "DONE").write_text("0\n", encoding="ascii")
        return receipt
    finally:
        # Native tools use this owned directory only; remove it when empty.
        try:
            Path(env["TMPDIR"]).rmdir()
        except OSError:
            pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "merge":
        parser = argparse.ArgumentParser(description="Merge hash-bound native day scores without reopening histories")
        parser.add_argument("--packet", required=True)
        parser.add_argument("--parts", nargs="+", required=True)
        parser.add_argument("--out", required=True)
        args = parser.parse_args(argv[1:])
        try:
            round_ = verify_packet(args.packet)
            result = combine_collections(args.parts, args.out, digest(Path(args.packet) / "round.json"),
                                         expected_days=round_["train_days"] + round_["heldout_days"])
            print(json.dumps({"status": result["status"], "days": result["days"], "members": len(result["members"])}, indent=2))
            return 0
        except (ValueError, OSError, KeyError) as error:
            print(f"tuning collect merge: {error}", file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("packet", "runs", "observations", "scorer", "scoreboard", "out"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--timezone-offset-hours", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--days", help="comma-separated subset of sealed round dates; collect one completed day at a time")
    args = parser.parse_args(argv)
    try:
        result = collect(args.packet, args.runs, args.observations, args.scorer, args.scoreboard, args.out,
                         timezone_offset_hours=args.timezone_offset_hours, timeout=args.timeout_seconds,
                         days=None if args.days is None else [day.strip() for day in args.days.split(",")])
        print(json.dumps({"status": result["status"], "days": result["days"], "members": len(result["members"])}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"tuning collect: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
