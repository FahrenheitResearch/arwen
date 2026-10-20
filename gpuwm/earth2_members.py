"""Generated parent members: one isolated GPU job, shared donor, parallel Rust staging.

The source row names this selection owner. Model-specific RNG and perturbation
facts live in earth2-models.v1.json. ``python -m gpuwm.earth2_members`` is also a
standalone batch acquisition door, suitable for staging before a regional run.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import time

RECEIPT = "earth2-batch.json"
SCHEMA = "gpuwm-earth2-batch-v1"
FORECAST = "earth2.nc"


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_receipt(expected, assets):
    if isinstance(assets, dict):
        return assets
    # Cache locations and loader open order do not change package identity.
    content = sorted(({key: asset[key] for key in ("asset", "sha256", "bytes")}
                      for asset in assets), key=lambda asset: (
                          asset["asset"], asset["sha256"], asset["bytes"]))
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return dict(expected or {}, files=assets,
                checksum=hashlib.sha256(encoded).hexdigest(),
                checksum_kind="sha256-canonical-asset-manifest")


def provider_of(fetch):
    from gpuwm.earth2_run import model_table
    model = fetch.get("model")
    source = str(fetch.get("source", ""))
    return model or (source.removeprefix("earth2-") if source.startswith("earth2-")
                     else model_table()["runtime"]["default_model"])


def provider_flag(args):
    return provider_of(vars(args))


def wizard_arguments(args):
    model = provider_flag(args)
    return ["--model", model] if model else []


def policy_for(model=None):
    from gpuwm.earth2_run import model_table, resolve_model
    key, row = resolve_model(model or model_table()["runtime"]["default_model"])
    policy = row.get("members")
    if not policy:
        raise ValueError(f"Earth-2 model {key} has no member policy in its table row")
    return key, row, policy


def noise_id(seed):
    return f"n{int(seed):04d}"


def validate_member(member, *, cycle=None):
    if not isinstance(member, str) or not re.fullmatch(r"n[0-9]{4,10}", member):
        raise ValueError("Earth-2 member must be a noise id n0333 (nonnegative integer seed)")
    seed = int(member[1:])
    if seed >= 2**32 or noise_id(seed) != member:
        raise ValueError("Earth-2 noise id must encode a canonical uint32 seed")
    return member


def ensemble_members(*, provider=None, source=None, members=None, base_seed=None,
                     index_offset=0):
    model = provider_of({"source": source}) if source and not provider else provider
    _, _, policy = policy_for(model)
    count = int(policy["population"] if members is None else members)
    first = int(policy["first_seed"] if base_seed is None else base_seed) + int(index_offset)
    if count <= 0 or first < 0 or first + count > 2**32:
        raise ValueError("Earth-2 population must contain positive count and uint32 seeds")
    return tuple(noise_id(first + index) for index in range(count))


def member_forcing():
    _, row, _ = policy_for()
    return "earth2*.nc", int(row["step_hours"]) * 3600


def member_fetch_hints(fetch, member, *, window_seconds, root):
    key, row, _ = policy_for(provider_of(fetch))
    member = validate_member(member)
    return dict(fetch, model=key, member=member,
                hours=math.ceil(float(window_seconds) / (3600 * row["step_hours"]))
                      * row["step_hours"],
                earth2_batch_root=str(Path(root)),
                out=str(Path(root) / member))


def batch_fetch_hints(fetches):
    """Attach the complete selected roster before any member can acquire it."""
    members = ",".join(dict.fromkeys(fetch["member"] for fetch in fetches))
    return [dict(fetch, earth2_members=members) for fetch in fetches]


def _stage_member(job):
    """One process per member, with Rust owning decode, composition and output."""
    from gpuwm.mapped_authoring import author_input_manifest
    from gpuwm import mapped_engine_bridge as engine
    from gpuwm.source_authorities import (
        packaged_authorities, packaged_contributing_mappings, packaged_provenance_files)

    started = time.perf_counter()
    root, member, seed, profile, donor, threads, model, cycle, *paths = job
    folder = Path(root) / member
    folder.mkdir(parents=True, exist_ok=True)
    forecasts = tuple(map(Path, paths[0])) if paths else (folder / FORECAST,)
    forecast = forecasts[0]
    authorities = packaged_authorities(profile)
    contributing = dict(packaged_contributing_mappings(profile))
    provenance = dict(packaged_provenance_files(profile))
    supplements = {"physical_analysis_surface_data": (Path(donor),)}
    manifest = folder / "inputs.json"
    author_input_manifest(manifest, mapping_path=authorities["mapping"],
        composition_path=authorities["composition"], primary_files=forecasts,
        supplement_files=supplements, provenance_files=provenance,
        contributing_mappings=contributing, member=member,
        member_identity="earth2-noise-seed-v1")
    output = folder / "composed"
    execution = engine.run_engine("compose", mapping=authorities["mapping"],
        composition=authorities["composition"], files=forecasts, output=output,
        supplements=supplements, provenance=provenance,
        contributing_mappings=contributing, input_manifest=manifest,
        input_manifest_sha256=sha256(manifest), threads=threads)
    evidence = engine.read_composition_evidence(output)
    # The native frameset validates stream extent and every consumed field hash.
    frames = engine.open_frameset(output)
    valid_times = [frame.valid_time.isoformat() for frame in frames]
    frames.close()
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    link = analysis / Path(donor).name
    if not link.exists():
        link.symlink_to(os.path.relpath(donor, analysis))
    publish_handoff(folder, forecasts, donor, source=f"earth2-{model}",
                    cycle=cycle,
                    member=member, batch_receipt=Path(root) / RECEIPT)
    receipt = {"id": member, "seed": seed, "noise_id": member,
        "forecast": str(forecast), "forecast_sha256": sha256(forecast),
        "forecasts": [{"path": str(path), "sha256": sha256(path)} for path in forecasts],
        "composition": str(output), "input_manifest": str(manifest),
        "valid_times": valid_times, "composition_evidence": evidence,
        "staging_seconds": round(time.perf_counter() - started, 3),
        "worker_pid": os.getpid(), "native_threads": threads,
        "bytes": sum(p.stat().st_size for p in folder.rglob("*")
                     if p.is_file() and not p.is_symlink()), "execution": dict(execution, output=str(execution["output"]))}
    _json(folder / "member.json", receipt)
    return receipt


def acquire_batch(*, model, cycle, hours, area, out, members, batch_size=None,
                  weights_package=None, workers=None):
    """Acquire all members atomically under a cross-process batch lock.

    Later member fetches verify the same acquisition identity and reuse the
    completed batch. A partial batch never masquerades as a completed one.
    """
    import fcntl
    from gpuwm.earth2_run import (model_table, earth2_python, producer_attributes,
                                 resolve_donor_policy, _donor)

    key, row, policy = policy_for(model)
    ids = tuple(validate_member(member) for member in members)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Earth-2 batch needs a nonempty, unique noise-id roster")
    seeds = [int(member[1:]) for member in ids]
    step = int(row["step_hours"])
    if hours <= 0 or hours % step:
        raise ValueError(f"Earth-2 hours must be a positive multiple of {step}")
    batch_size = policy["batch_size"] if batch_size is None else batch_size
    if batch_size != "auto":
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Earth-2 batch size must be positive")
    if len(area) != 4:
        raise ValueError("Earth-2 area needs south, west, north, east")
    south, west, north, east = map(float, area)
    crop = [max(-90., south - .5), (west - .5) % 360,
            min(90., north + .5), (east + .5) % 360]
    if not south < north or crop[1] > crop[3]:
        raise ValueError("Earth-2 area must be ordered and fit one longitude interval")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected_donor = resolve_donor_policy(row, cycle)
    table = model_table()
    ic = table["runtime"]["initial_condition_sources"][row.get("initial_condition_source", "era5")]
    identity = {"model": key, "cycle": cycle.isoformat(), "hours": hours,
                "area": crop, "members": list(ids), "seeds": seeds,
                "batch_size": batch_size, "member_policy": policy,
                "weights": row.get("weights"), "weights_package": weights_package,
                "runtime_env": row.get("runtime_env", {}),
                "loader_kwargs": row.get("load_model_kwargs", {}) | row.get("loader_kwargs", {}),
                "ic_kwargs": dict(ic.get("kwargs", {}), **row.get("ic_kwargs", {})),
                "ic_class": ic["earth2studio_class"], "output_grid": row.get("output_grid"),
                "profile": selected_donor["profile"], "donor_policy": selected_donor}
    with (out / "earth2-batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt_path = out / RECEIPT
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("identity") != identity:
                raise ValueError(f"{out} holds a different Earth-2 batch identity")
            if sha256(receipt["donor"]["path"]) != receipt["donor"]["sha256"]:
                raise ValueError("Earth-2 shared donor digest changed")
            for member in receipt["members"]:
                for frame in member.get("forecasts", [{"path": member["forecast"],
                        "sha256": member["forecast_sha256"]}]):
                    if sha256(frame["path"]) != frame["sha256"]:
                        raise ValueError(f"Earth-2 member {member['id']} forecast digest changed")
            return receipt
        started = time.perf_counter()
        from gpuwm.source_authorities import packaged_authorities
        mapping = packaged_authorities(selected_donor["profile"])["mapping"]
        request = dict(identity, model_class=row["earth2studio_class"],
            ic_class=ic["earth2studio_class"], init=cycle.strftime("%Y-%m-%dT%H:%M:%S"),
            steps=hours // step, output=str(out / "ensemble.nc"),
            member_outputs=[str(out / member / FORECAST) for member in ids],
            output_attributes=producer_attributes(mapping))
        request.update(members=list(range(len(ids))), seed=seeds[0],
            pinned_version=model_table()["runtime"]["pinned_version"],
            weight_cache=str(out / "runtime-cache" / "weights"),
            data_cache=str(out / "runtime-cache" / "data"),
            expected_weights=row.get("weights", {}),
            model_package=row.get("weights", {}).get("uri"),
            loader_preparations=row.get("loader_preparations", []),
            model_calls=row.get("model_calls", []))
        Path(request["weight_cache"]).mkdir(parents=True, exist_ok=True)
        request_path = out / "earth2-request.json"
        _json(request_path, request)
        runner_receipt = out / "earth2-runner.json"
        # Fetch shared analysis before admission. The scheduler prefix wraps only
        # inference/cropping; native composition starts after it has released.
        donor_started = time.perf_counter()
        donor_report = _donor(table=table, row=row, cycle=cycle, area=crop, out=out, force=False)
        donor = Path(donor_report["analysis"])
        donor_seconds = round(time.perf_counter() - donor_started, 3)
        if not runner_receipt.exists():
            runner = Path(__file__).with_name("earth2_runner.py")
            prefixes = json.loads(os.environ.get("GPUWM_EARTH2_STAGE_PREFIX", "{}"))
            if not isinstance(prefixes, dict):
                raise ValueError("GPUWM_EARTH2_STAGE_PREFIX needs an object of stage argv prefixes")
            prefix = prefixes.get("model", [])
            if not isinstance(prefix, list) or any(not isinstance(arg, str) for arg in prefix):
                raise ValueError("GPUWM_EARTH2_STAGE_PREFIX model needs a string argv list")
            original = [earth2_python(), str(runner), str(request_path)]
            command = [*prefix, *original]
            runner_started = time.perf_counter()
            with (out / "runner.stdout.log").open("w") as stdout, \
                 (out / "runner.stderr.log").open("w") as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr)
            _json(out / "earth2-admission.json", {
                "command": command, "unwrapped_command": original,
                "scheduler_prefix": prefix, "returncode": result.returncode,
                "wall_with_queue_seconds": round(time.perf_counter() - runner_started, 3)})
            if result.returncode:
                raise RuntimeError(f"Earth-2 ensemble runner exited {result.returncode}; "
                                   f"see {out / 'runner.stderr.log'}")
            report = json.loads((out / "runner.stdout.log").read_text().splitlines()[-1])
            report["request_sha256"] = sha256(request_path)
            _json(runner_receipt, report)
        else:
            report = json.loads(runner_receipt.read_text())
            if report["request_sha256"] != sha256(request_path):
                raise ValueError("partial Earth-2 batch belongs to another request")
        stage_started = time.perf_counter()
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        count = min(len(ids), int(workers or cores))
        if count < 1:
            raise ValueError("Earth-2 staging workers must be positive")
        # Allocate the remainder too: 18 members on 24 cores receive six
        # two-thread native workers and twelve one-thread workers.
        widths = [cores // count + int(index < cores % count) for index in range(count)]
        jobs = [(str(out), member, seed, selected_donor["profile"], str(donor),
                 widths[index % count], key, cycle,
                 [frame["path"] for frame in sorted(report.get("files", []),
                      key=lambda frame: frame["lead_hours"]) if frame["member"] == index]
                 or [str(out / member / FORECAST)])
                for index, (member, seed) in enumerate(zip(ids, seeds))]
        with ProcessPoolExecutor(max_workers=count,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            staged = list(pool.map(_stage_member, jobs))
        assets = report["weights"]
        weights = weight_receipt(row.get("weights"), assets)
        receipt = {"schema": SCHEMA, "identity": identity, "model": key,
            "weights": weights, "member_policy": policy,
            "members": staged, "batch_size": report["batch_size"],
            "batch_memory": report["batch_memory"], "source": f"earth2-{key}",
            "profile": selected_donor["profile"],
            "peak_gpu_gib": report["peak_gpu_gib"],
            **({"peak_reserved_gpu_gib": report["peak_reserved_gpu_gib"]}
               if "peak_reserved_gpu_gib" in report else {}),
            "inference_seconds": report["inference_seconds"],
            "staging_seconds": round(time.perf_counter() - stage_started, 3),
            "staging_workers": count, "available_cores": cores,
            "donor_seconds": donor_seconds,
            "donor": {**donor_report, "cycle": cycle.isoformat(),
                      "path": str(donor), "sha256": sha256(donor)},
            "runner": report, "wall_seconds": round(time.perf_counter()-started, 3)}
        _json(receipt_path, receipt)
        return receipt


def fetch_member(args, *, model, cycle, area):
    key, _, policy = policy_for(model)
    member = validate_member(args.member)
    members = getattr(args, "earth2_members", None)
    ids = members.split(",") if members else [member]
    if member not in ids:
        raise ValueError(f"selected member {member} is absent from Earth-2 batch roster")
    out = Path(args.out).resolve()
    root = Path(getattr(args, "earth2_batch_root", None) or out.parent)
    receipt = acquire_batch(model=key, cycle=cycle, hours=int(args.hours), area=area,
        out=root, members=ids, batch_size=getattr(args, "earth2_batch_size", None),
        weights_package=getattr(args, "earth2_weights_package", None))
    selected = next(row for row in receipt["members"] if row["id"] == member)
    publish_handoff(out, [frame["path"] for frame in selected.get("forecasts", [])]
                    or selected["forecast"], receipt["donor"]["path"],
                    source=args.source, cycle=cycle, member=member,
                    batch_receipt=root.resolve() / RECEIPT)
    return receipt


def publish_handoff(out, forecast, donor, *, source, cycle, member=None,
                    batch_receipt=None):
    """The ordinary mapped preparation consumes this explicit fetch handoff."""
    from gpuwm.fetch_routes import (
        INPUT_LIST_NAME, PREP_ARGUMENTS_NAME, PREP_ARGUMENTS_SCHEMA)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    listing = out / INPUT_LIST_NAME
    forecasts = forecast if isinstance(forecast, (list, tuple)) else (forecast,)
    listing.write_text("".join(str(Path(path).resolve()) + "\n" for path in forecasts))
    argv = ["--source", str(source), "--input-list", str(listing),
            "--supplement", f"physical_analysis_surface_data={Path(donor).resolve()}",
            "--author-input-manifest", str(out / "inputs.json")]
    handoff = {"schema": PREP_ARGUMENTS_SCHEMA, "source": source,
               "prep_source": source, "cycle": cycle.strftime("%Y-%m-%dT%H"),
               "argv": argv, "unbound_supplement_roles": [],
               "caller_supplies": ["--wps-namelist", "--experiment-config",
                                   "--geog-root", "--output-root"]}
    if member is not None:
        handoff.update(member=member, member_identity="earth2-noise-seed-v1",
                       parent_receipt=str(batch_receipt))
    _json(out / PREP_ARGUMENTS_NAME, handoff)
    return handoff


def verify_handoff(hints, handoff, *, out):
    """Seal exactly one noise member, its donor and its recorded bytes."""
    from concurrent.futures import ThreadPoolExecutor
    from gpuwm.fetch import parse_cycle
    from gpuwm.member_grammar import MemberIdentityRefusal
    from gpuwm.source_adapters import canonical_source_token
    member = validate_member(hints["member"])
    source = canonical_source_token(hints["source"])
    def refuse(message):
        raise MemberIdentityRefusal(message)
    if handoff.get("source") != source or handoff.get("member") != member:
        refuse("Generated source/member differs from the selected handoff")
    cycle = parse_cycle(hints["cycle"], "era5")
    if parse_cycle(handoff["cycle"], "era5") != cycle:
        refuse("Generated forcing cycle differs from the selected cycle")
    batch_path = Path(handoff["parent_receipt"]).resolve(strict=True)
    if hints.get("earth2_batch_root") and batch_path != (Path(hints["earth2_batch_root"]) / RECEIPT).resolve():
        refuse("Generated handoff names another batch root")
    batch = json.loads(batch_path.read_text())
    if batch.get("schema") != SCHEMA or batch["model"] != provider_of(hints):
        refuse("Generated receipt model differs from the selected source")
    if datetime.fromisoformat(batch["identity"]["cycle"]).replace(tzinfo=None) != cycle.replace(tzinfo=None):
        refuse("Generated receipt cycle differs from the selected cycle")
    selected = next((row for row in batch["members"] if row["id"] == member), None)
    if selected is None or selected["seed"] != int(member[1:]):
        refuse("Generated receipt has no matching member seed")
    arguments = handoff["argv"]
    if arguments.count("--input-list") != 1:
        refuse("Generated forcing needs exactly one bound input-list")
    listing = Path(arguments[arguments.index("--input-list") + 1]).resolve(strict=True)
    paths = [str(Path(line).resolve(strict=True)) for line in listing.read_text().splitlines() if line.strip()]
    expected = selected["forecasts"]
    if paths != [frame["path"] for frame in expected] or not paths or len(set(paths)) != len(paths):
        refuse("Generated input-list differs from the member's recorded trajectory")
    donor = batch["donor"]
    if f"physical_analysis_surface_data={donor['path']}" not in arguments:
        refuse("Generated handoff donor differs from the batch receipt")
    from netCDF4 import Dataset
    index = batch["identity"]["members"].index(member)
    for frame in expected:
        with Dataset(frame["path"]) as data:
            if (int(data.getncattr("earth2_member")) != index or
                    int(data.getncattr("earth2_generation_seed")) != selected["seed"]):
                refuse("Generated frame member/seed differs from the selected trajectory")
    def check(frame):
        if sha256(frame["path"]) != frame["sha256"]:
            refuse("Generated forcing hash differs: " + frame["path"])
        return frame
    with ThreadPoolExecutor(max_workers=min(32, len(expected) + 1)) as pool:
        checked = list(pool.map(check, [*expected, donor]))
    result = {"schema": "arwen.forcing-member.v1", "source": source,
              "member": member, "cycle": handoff["cycle"], "files": checked[:-1],
              "dependency_files": [donor], "parent_receipt": str(batch_path),
              "weights": batch["weights"], "seed": selected["seed"], "donor": donor}
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    target = root / f"member-inputs-{key}.txt"
    text = "".join(path + "\n" for path in paths)
    try:
        with target.open("x") as stream:
            stream.write(text)
    except FileExistsError:
        if target.read_text() != text:
            refuse("Verified generated-member input-list is corrupt")
    result["input_list"] = str(target.resolve())
    _json(root / f"member-verification-{key}.json", result)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--hours", required=True, type=int)
    parser.add_argument("--area", required=True, help="south,west,north,east")
    parser.add_argument("--members", required=True, help="comma-separated noise ids")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--weights-package")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    from gpuwm.fetch import parse_cycle
    receipt = acquire_batch(model=args.model, cycle=parse_cycle(args.cycle, "era5"),
        hours=args.hours, area=list(map(float, args.area.split(","))), out=args.out,
        members=args.members.split(","), batch_size=args.batch_size,
        weights_package=args.weights_package, workers=args.workers)
    print(json.dumps({"receipt": str(Path(args.out) / RECEIPT),
                      "members": len(receipt["members"]),
                      "inference_seconds": receipt["inference_seconds"],
                      "staging_seconds": receipt["staging_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
