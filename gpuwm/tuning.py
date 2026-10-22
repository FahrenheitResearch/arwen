"""Opt-in product physics tuning. Python schedules; Rust computes the search and scores."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SCHEMA = "gpuwm.tuning-round/v1"
PRODUCTS = {"city", "research"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def numerical(binary, operation, value):
    result = subprocess.run([str(binary), operation], input=json.dumps(value, allow_nan=False),  # text-pipe-ok: a one-shot JSON value
                            encoding="utf-8", capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Rust {operation} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def validate_spec(spec):
    if spec.get("product") not in PRODUCTS:
        raise ValueError("tuning applies to city or research products; a reference-code product cannot change physics")
    train, hold = spec["train_days"], spec["heldout_days"]
    if len(train) < 2 or len(hold) < 2 or len(set(train + hold)) != len(train + hold):
        raise ValueError("training and held-out days must be disjoint, unique, and contain at least two days each")
    for day in train + hold:
        datetime.strptime(day, "%Y-%m-%d")
    if not 16 <= spec["member_count"] <= 24:
        raise ValueError("the first-round recipe requires 16 to 24 settings including controls")
    if spec.get("hours") != 24 or spec.get("cycle_hour") != 0:
        raise ValueError("this recipe scores a complete, observed UTC day using f01 to f24 of a 00 UTC start")
    names = [row["name"] for row in spec["parameter_space"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate parameter table names")


def patch_table(text, section, updates):
    marker = f"[{section}]"
    position = text.find(marker + "\n")
    if position < 0:
        raise ValueError(f"base experiment lacks {marker}")
    start = position + len(marker) + 1
    match = re.search(r"^\[", text[start:], re.M)
    end = start + match.start() if match else len(text)
    body = text[start:end]
    for key, value in updates.items():
        spelling = value if isinstance(value, str) else json.dumps(value)
        line = f"{key} = {spelling}"
        pattern = rf"^{re.escape(key)}\s*=.*$"
        if re.search(pattern, body, re.M):
            body = re.sub(pattern, lambda _: line, body, flags=re.M)
        else:
            body += line + "\n"
    return text[:start] + body + text[end:]


def member_descriptor(identity, settings, seed, product, rows):
    land = int(settings["land_model"])
    physics = {"sf_surface_physics": land, "num_soil_layers": 6 if land == 3 else 4,
               "bl_mynn_mixlength": int(settings["bl_mynn_mixlength"]),
               "sf_urban_physics": 0 if land == 3 else int(settings["urban_option"])}
    active, inactive = {}, []
    for row in rows:
        key = row["name"]
        applies = land in row.get("schemes", [2, 3, 4])
        applies = applies and (not row.get("requires_urban") or physics["sf_urban_physics"] == 1)
        if applies:
            active[key] = settings[key]
        else:
            inactive.append(key)
            settings[key] = row["default"]
    surface = {key: settings[key] for key in ("soil_moisture_scale", "sst_offset_k")}
    urban = {"fraction_scale": settings["urban_fraction_scale"],
             "heat_capacity_scale": settings["urban_heat_capacity_scale"],
             "conductivity_scale": settings["urban_conductivity_scale"]}
    return {"id": identity, "seed": seed, "product": product, "parameters": settings,
            "active_parameters": active, "inactive_parameters": inactive,
            "physics": physics, "surface": surface, "urban": urban}


def plan(spec_path, binary, output):
    spec_path, output = Path(spec_path), Path(output)
    spec = read_json(spec_path)
    validate_spec(spec)
    if output.exists() and any(output.iterdir()):
        raise ValueError("round directory already has artifacts; create a new round to preserve its receipt")
    output.mkdir(parents=True, exist_ok=True)
    base_path = (spec_path.parent / spec["base_config"]).resolve()
    base = base_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    if "[ensemble]" in base or "[physics_params]" in base:
        raise ValueError("base must be an ordinary product experiment without a second ensemble or parameter set")
    rows = spec["parameter_space"]
    sampled = numerical(binary, "sample", {"parameters": rows, "count": spec["member_count"] - 3, "seed": spec["seed"]})
    defaults = {row["name"]: row["default"] for row in rows}
    members = []
    for identity, land in (("current", 3), ("land-2-control", 2), ("land-4-control", 4)):
        settings = {**defaults, "land_model": land}
        members.append(member_descriptor(identity, settings, spec["seed"], spec["product"], rows))
    for index, sample in enumerate(sampled["members"]):
        members.append(member_descriptor(f"setting-{index + 1:03d}", sample["parameters"],
                       spec["seed"] + index + 1, spec["product"], rows))
    groups, runs = {}, []
    for member in members:
        physics = member["physics"]
        group = "land-{sf_surface_physics}-mix-{bl_mynn_mixlength}-urban-{sf_urban_physics}".format(**physics)
        member["preparation_group"] = group
        groups[group] = physics
        if physics["sf_surface_physics"] == 3 and member["parameters"]["canopy_resistance_scale"] != 1:
            relative = f"parameter-sets/{member['id']}.toml"
            member["physics_params_file"] = relative
            path = output / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text(f'[physics_params]\nname = "{member["id"]}"\n[physics_params.values]\n"ruc.rs" = {member["parameters"]["canopy_resistance_scale"]}\n', encoding="utf-8")
        for day in spec["train_days"] + spec["heldout_days"]:
            runs.append({"id": f"{day}/{member['id']}", "day": day, "member_id": member["id"],
                         "split": "train" if day in spec["train_days"] else "heldout", "group": group,
                         "config": f"configs/{day}/{group}.toml", "forecast_hours": list(range(1, 25))})
    for day in spec["train_days"] + spec["heldout_days"]:
        for group, physics in groups.items():
            text = patch_table(base, "shared", physics)
            text = patch_table(text, "experiment", {"name": json.dumps("product-tuning"),
                "start_time": day + "T00:00:00", "run_seconds": 86400.0})
            text = patch_table(text, "fetch", {"cycle": json.dumps(day + "T00"), "hours": 24})
            # Runtime path is made relative to the operator's working directory, never a machine path.
            text = patch_table(text, "static.highres", {"cache_root": json.dumps("shared-prepared/highres-cache")})
            path = output / f"configs/{day}/{group}.toml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    write_json(output / "round.json", {**spec, "schema": SCHEMA, "members": members,
        "baseline_member": "current", "base_config_sha256": digest(base_path),
        "search": "seeded Latin hypercube with three unperturbed land controls",
        "decision": "pending real forecasts and frozen held-out check"})
    write_json(output / "run-list.json", runs)
    seal_packet(output)
    return {"settings": len(members), "forecasts": len(runs), "preparation_groups_per_day": len(groups)}


def seal_packet(packet):
    files = {str(path.relative_to(packet)).replace("\\", "/"): digest(path)
             for path in sorted(Path(packet).rglob("*")) if path.is_file() and path.name != "packet-sha256.json"}
    write_json(Path(packet) / "packet-sha256.json", {"schema": "gpuwm.tuning-packet/v1", "files": files})


def verify_packet(packet):
    packet = Path(packet).resolve()
    seal = read_json(packet / "packet-sha256.json")
    for name, expected in seal["files"].items():
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts or not (packet / candidate).resolve().is_relative_to(packet):
            raise ValueError("sealed packet paths must remain inside the packet")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            raise ValueError("sealed packet hash is invalid")
        if digest(packet / candidate) != expected:
            raise ValueError(f"packet bytes changed: {name}")
    return read_json(packet / "round.json")


def cpu_limit():
    from gpuwm.ingest.preparation_workers import cpu_budget
    return min(4, cpu_budget()["available_cpus"])


def prepare(packet, shared, geog, day=None):
    packet, shared = Path(packet).resolve(), Path(shared).resolve()
    verify_packet(packet)
    from gpuwm.tuning_prepare import prepare_group
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["GPUWM_NO_LOCAL_GPU"] = "1"
    os.environ.pop("GPUWM_PHYSICS_PARAMS", None)
    budget = cpu_limit()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS",
                "GPUWM_PREPROCESS_THREADS", "GPUWM_MAPPED_ENGINE_THREADS"):
        os.environ[key] = str(budget)
    temporary = Path.home() / ".gpuwm" / "tuning-harness" / "tmp" / "preparation"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    for config in sorted((packet / "configs").glob("*/*.toml")):
        config_day, group = config.parent.name, config.stem
        if day is not None and config_day != day:
            continue
        location = shared / "prepared" / config_day / group
        receipt = location / "tuning-preparation.json"
        if receipt.exists():
            existing = read_json(receipt)
            if existing.get("status") != "PASS":
                raise ValueError("the previous preparation attempt did not pass; preserve it and use a fresh shared root")
            if existing.get("packet_config_sha256") != digest(config):
                raise ValueError("prepared group belongs to different configuration bytes")
            continue
        prepared = prepare_group(config, shared_raw=shared / "raw" / config_day, geog_root=Path(geog),
                                 output=location, cache_root=shared / "highres-cache", workers=budget)
        write_json(receipt, prepared)


def run(packet, shared, output, devices, timeout, split, day=None):
    packet, shared, output = Path(packet).resolve(), Path(shared).resolve(), Path(output).resolve()
    round_ = verify_packet(packet)
    if os.name == "nt":
        raise ValueError("GPU forecasts run on the assigned remote box")
    device_list = [value.strip() for value in devices.split(",") if value.strip()]
    if not device_list or len(set(device_list)) != len(device_list):
        raise ValueError("assign unique card indices to the runner")
    if timeout <= 0:
        raise ValueError("each forecast needs a positive hard time bound")
    jobs = [job for job in read_json(packet / "run-list.json") if job["split"] == split and (day is None or job["day"] == day)]
    if not jobs:
        raise ValueError("no jobs in the named day and split")
    members = {member["id"]: member for member in round_["members"]}
    output.mkdir(parents=True, exist_ok=True)
    results = []
    def worker(card, assigned):
        local = []
        for job in assigned:
            destination = output / job["day"] / job["member_id"]
            receipt = destination / "tuning-run.json"
            if receipt.exists():
                raise ValueError(f"existing run receipt: {receipt}; use a new output directory")
            destination.mkdir(parents=True, exist_ok=True)
            prepared_record = read_json(shared / "prepared" / job["day"] / job["group"] / "tuning-preparation.json")
            if prepared_record.get("packet_config_sha256") != digest(packet / job["config"]):
                raise ValueError("preparation belongs to different sealed packet configuration bytes")
            prepared = {key: value for key, value in prepared_record.items() if key in {
                "prepared_root", "experiment_config", "experiment_config_sha256", "preparation_receipt_sha256", "prepared_head_sha256"}}
            member = {key: value for key, value in members[job["member_id"]].items()
                      if key in {"id", "seed", "product", "physics", "surface", "urban", "physics_params_file"}}
            if member.get("physics_params_file"):
                member["physics_params_file"] = str(packet / member["physics_params_file"])
            request = {"member": member, "prepared": prepared,
                       "output_directory": str(destination / "run"), "receipt_path": str(destination / "member-receipt.json")}
            job_path = destination / "job.json"
            write_json(job_path, request)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=card, GPUWM_NO_LOCAL_GPU="0")
            environment.pop("GPUWM_PHYSICS_PARAMS", None)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
                environment[key] = "1"
            environment["TMPDIR"] = str(Path.home() / ".gpuwm" / "tuning-harness" / "tmp" / job["day"] / job["member_id"])
            Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            with (destination / "forecast.log").open("w", encoding="utf-8") as log:
                try:
                    process = subprocess.run([sys.executable, "-B", "-m", "gpuwm.tuning_member", "--job", str(job_path)],
                        stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=timeout, check=False)
                    code, status = process.returncode, "succeeded" if process.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    code, status = 124, "timed-out"
            result = {**job, "status": status, "exit_code": code, "card": card,
                      "wall_seconds": time.monotonic() - started, "job_sha256": digest(job_path)}
            write_json(receipt, result)
            (destination / "DONE").write_text(str(code) + "\n", encoding="ascii")
            local.append(result)
        return local
    with ThreadPoolExecutor(max_workers=len(device_list)) as pool:
        futures = [pool.submit(worker, card, jobs[index::len(device_list)]) for index, card in enumerate(device_list)]
        for future in futures:
            results.extend(future.result())
    stage_name = split if day is None else f"{split}-{day}"
    write_json(output / f"{stage_name}-scheduler.json", {"schema": SCHEMA, "runs": results,
        "all_succeeded": all(result["status"] == "succeeded" for result in results)})
    (output / f"{stage_name}-DONE").write_text("0\n" if all(r["status"] == "succeeded" for r in results) else "1\n")
    return results


def score(packet, binary, import_request, output):
    round_ = verify_packet(packet)
    output = Path(output)
    if (output / "round-receipt.json").exists():
        raise ValueError("a round was already scored; preserve its receipt and use a new receipt directory")
    import_path = Path(import_request).resolve()
    from gpuwm.tuning_collect import verify_collected_scores
    verify_collected_scores(import_path.parent, round_sha256=digest(Path(packet) / "round.json"))
    import_value = read_json(import_path)
    for record in import_value["files"]:
        source = Path(record["path"])
        record["path"] = str(source if source.is_absolute() else import_path.parent / source)
    rows = numerical(binary, "import", import_value)
    config = {key: round_[key] for key in ("train_days", "heldout_days", "baseline_member")}
    config.update(expected_members=[member["id"] for member in round_["members"]],
        ensemble_members=[member["id"] for member in round_["members"] if member["id"].startswith("setting-")],
        minimum_pairs=round_.get("minimum_pairs", 24))
    train_rows = [row for row in rows["rows"] if row["day"] in config["train_days"]]
    training = numerical(binary, "score", {"rows": train_rows, "config": {**config, "heldout_days": []}})
    selection_path = output / "selection.json"
    if selection_path.exists():
        frozen = read_json(selection_path)
        if (frozen["round_sha256"] != digest(Path(packet) / "round.json")
                or frozen["input_sha256"] != digest(import_request) or frozen["training"] != training):
            raise ValueError("the pending frozen selection belongs to different input scores or settings")
    else:
        write_json(selection_path, {"round_sha256": digest(Path(packet) / "round.json"),
            "input_sha256": digest(import_request), "training": training,
            "frozen_at": datetime.now(timezone.utc).isoformat()})
    result = numerical(binary, "score", {"rows": rows["rows"],
        "config": {**config, "frozen_winner": training["decision"]["winner"]}})
    write_json(output / "round-receipt.json", {"schema": SCHEMA, "settings": round_["members"],
        "selection_sha256": digest(output / "selection.json"), "support": {k: v for k, v in rows.items() if k != "rows"},
        "scores": result, "decision_scope": "candidate for product review; no default or deployment changed"})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--spec", required=True); p.add_argument("--numerics", required=True); p.add_argument("--out", required=True)
    for name in ("prepare", "run", "score", "verify"):
        p = commands.add_parser(name); p.add_argument("--packet", required=True)
        if name in ("prepare", "run"):
            p.add_argument("--shared-root", required=True)
        if name == "prepare":
            p.add_argument("--geog-root", required=True)
            p.add_argument("--day")
        if name == "run":
            p.add_argument("--out", required=True); p.add_argument("--devices", required=True)
            p.add_argument("--timeout-seconds", type=int, default=5400)
            p.add_argument("--split", choices=("train", "heldout"), required=True)
            p.add_argument("--day")
        if name == "score":
            p.add_argument("--numerics", required=True); p.add_argument("--import-request", required=True); p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan": result = plan(args.spec, args.numerics, args.out)
        elif args.command == "prepare": result = prepare(args.packet, args.shared_root, args.geog_root, args.day)
        elif args.command == "run": result = run(args.packet, args.shared_root, args.out, args.devices, args.timeout_seconds, args.split, args.day)
        elif args.command == "score": result = score(args.packet, args.numerics, args.import_request, args.out)
        else: result = {"verified": bool(verify_packet(args.packet))}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if args.command == "run" and any(row["status"] != "succeeded" for row in result) else 0
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(f"tuning: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
