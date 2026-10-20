"""Table-driven Earth2Studio acquisition and its bound preparation handoff.

The runtime stays in GPUWM_EARTH2_PYTHON. Models, loader arguments, licences,
profiles and donor policy are data, shared by fetch and the go launch route.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TABLE = Path(__file__).with_name("authorities") / "earth2-models.v1.json"
PYTHON_ENV = "GPUWM_EARTH2_PYTHON"
RECEIPT = "earth2-run.json"
PREP_HANDOFF_SCHEMA = "gpuwm-fetch-prep-arguments-v1"
PUBLISHES_PREP_HANDOFF = True
FETCH_ACCEPTS_AREA = True


def model_table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))


def resolve_donor_policy(row: dict, cycle: datetime, mode: str = "auto", *, now=None) -> dict:
    """Resolve one same-cycle physical donor and its sealed profile from table data.

    ``forecast`` is explicit even for an archived forecast replay. ``auto``
    uses the table's age boundary; a row may override either mode or name
    one donor for both. Acquisition and model inference remain runner-owned.
    """
    table = model_table()
    policy = table["donor_policy"]
    if mode == "auto":
        current = now or datetime.now(cycle.tzinfo)
        mode = ("hindcast" if cycle < current - timedelta(days=policy["auto_hindcast_age_days"])
                else "forecast")
    if mode not in policy["defaults"]:
        raise ValueError(f"donor mode {mode!r} is not declared in donor_policy.defaults")
    override = row.get(policy["row_override_key"])
    if override is None:
        override = next((rule["donor"] for rule in policy.get("row_defaults", ())
                         if all(row.get(key) == value for key, value in rule["match"].items())), {})
    donor_id = (override if isinstance(override, str) else
                override.get(mode, override.get("source", policy["defaults"][mode])))
    donor = dict(policy["donors"][donor_id])
    profile = row["profile"]
    variants = policy["profile_variants"].get(profile)
    if variants is None:
        variants = next((values for values in policy["profile_variants"].values()
                         if profile in values.values()), None)
    if variants is None or donor_id not in variants:
        raise ValueError(f"profile {profile!r} has no declared {donor_id!r} donor variant")
    return {**donor, "id": donor_id, "mode": mode, "profile": variants[donor_id],
            "cycle": cycle.isoformat(), "hours": 0}


def source_aliases() -> tuple[str, ...]:
    """Unclaimed runnable model shorthands from the table, without I/O."""
    aliases = []
    for key, row in model_table()["models"].items():
        if row.get("kind") == "diagnostic":
            continue
        try:
            resolve_model(key)
        except ValueError:
            continue
        aliases.extend((key, *row.get("aliases", ())))
    return tuple(aliases)


def acquisition_hints(source: str, model: str | None = None) -> dict:
    """Pure planning hook; never imports a model runtime or downloads bytes."""
    named = None if source in ("earth2", "earth2studio") else model_declaration(source)[0]
    if model is not None and named is not None and model_declaration(model)[0] != named:
        raise ValueError("--model conflicts with the model named by --source")
    selected = model or named
    return {"model": model_declaration(selected)[0]} if selected is not None else {}


def model_declaration(name: str | None) -> tuple[str, dict]:
    """Read identity for planning; acquisition separately enforces runtime policy."""
    table = model_table()
    models = table["models"]
    wanted = (name or "").strip().lower().replace("_", "-")
    if wanted in ("", "earth2", "earth2studio"):
        wanted = table["runtime"].get("default_model", "")
    if wanted.startswith("earth2-"):
        wanted = wanted[len("earth2-"):]
    for model_id, row in models.items():
        if wanted == model_id or wanted in (alias.lower() for alias in row.get("aliases", ())):
            return model_id, row
    raise ValueError("choose a model row: " + ", ".join(sorted(models)))


def resolve_model(name: str | None) -> tuple[str, dict]:
    model_id, row = model_declaration(name)
    from gpuwm.earth2_licensing import require_licence
    require_licence(model_id, row)
    if not row.get("profile"):
        raise ValueError(f"{model_id}: {row.get('needs', 'no packaged profile')}")
    refusal = row.get("runtime_refusal") or row.get("refusal") or row.get("licence_refusal")
    if refusal:
        raise ValueError(f"{model_id}: {refusal}")
    weights = row.get("weights") or {}
    if weights.get("account_needed") and weights.get("host") != "huggingface":
        raise ValueError(f"{model_id}: weights need an account at {weights.get('host')}; "
                         "only anonymous public weights are acquired")
    return model_id, row


def acquisition_facts(source: str) -> dict:
    """The inventory and readiness doors describe the same table policy."""
    from gpuwm.source_adapters import get_source_adapter

    adapter = get_source_adapter(source)
    return {"generation": "generated on the card at run time",
            "model": adapter.model_key or model_table()["runtime"]["default_model"],
            "readiness_basis": "the same-cycle donor analysis readiness",
            "donor_policy": model_table()["donor_policy"]}


def readiness(source, cycle, hours, *, now=None, **options):
    """Ask for the donor analysis, without loading models or acquiring bytes."""
    from gpuwm import fetch, source_readiness as readiness_api
    from gpuwm.source_adapters import get_source_adapter

    checked = readiness_api._naive(now) if now else readiness_api._utc_now()
    named_cycle = None
    try:
        adapter = get_source_adapter(source)
        model, row = resolve_model(adapter.model_key)
        named_cycle = (fetch.parse_cycle(cycle, source)
                       if isinstance(cycle, str) and cycle != "latest" else cycle)
        donor = resolve_donor_policy(
            row, checked if named_cycle == "latest" else named_cycle, now=checked)
    except ValueError as error:
        return (readiness_api._refusal_document(
            source, named_cycle if isinstance(named_cycle, datetime) else None,
            "named", hours, options.get("as_posted", True), str(error), checked),
            readiness_api.REFUSED_EXIT)
    # The donor is an analysis at lead zero, not an AI-member forecast
    # window. In particular, the AI member number is not a donor member.
    donor_document, code = readiness_api.readiness(
        donor["source"], named_cycle, donor["hours"], start_hour=0,
        provider=donor["provider"], transport=options.get("transport"),
        no_probe=options.get("no_probe", False), probe=options.get("probe"),
        as_posted=options.get("as_posted", True), now=checked)
    document = {**donor_document, "source": source,
                "member": options.get("member"),
                "window": {"start_lead": options.get("start_hour") or 0,
                           "hours": hours, "cadence": options.get("cadence"),
                           "final_lead": (options.get("start_hour") or 0) + hours},
                "acquisition_owner": __name__,
                "generation": {"description": "generated on the card at run time",
                               "model": model, "donor": donor,
                               "readiness": donor_document}}
    return document, code


def require_inference_vram(model_id, row):
    """Check a row's measured capacity floor only when generating new output."""
    minimum = float(row.get("min_vram_gib", 0))
    if minimum <= 0:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        raise ValueError(f"{model_id}: inference needs {minimum:g} GiB of GPU memory; CUDA devices are hidden")
    selector = visible.split(",")[0] if visible else "0"
    command = ["nvidia-smi", "--query-gpu=memory.total,memory.free",
               "--format=csv,noheader,nounits", "--id=" + selector]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        lines = result.stdout.strip().splitlines()
        if len(lines) != 1:
            raise ValueError("expected one selected GPU")
        total, free = (float(v.strip()) for v in lines[0].split(","))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError(f"{model_id}: cannot verify the row's {minimum:g} GiB memory requirement: {error}") from error
    measured = dict(minimum_gib=minimum, total_mib=total, free_mib=free, command=command,
                    note=row.get("vram_note"))
    if min(total, free) < minimum * 1024:
        raise ValueError(f"{model_id}: inference needs {minimum:g} GiB of free GPU memory; "
                         f"card has {total/1024:.2f} GiB total, {free/1024:.2f} GiB free. "
                         + row.get("vram_note", "The measured table requirement is not met."))
    return measured


def earth2_python() -> str:
    """Require the table's version, before any model or donor download."""
    runtime = model_table()["runtime"]
    candidate = os.environ.get(PYTHON_ENV) or sys.executable
    try:
        probe = subprocess.run([candidate, "-c", "from importlib.metadata import version; "
                                "print(version('earth2studio'))"], capture_output=True, text=True)
    except OSError:
        probe = None
    if probe is not None and probe.returncode == 0 and probe.stdout.strip() == runtime["pinned_version"]:
        return candidate
    raise ValueError(f"{candidate} needs earth2studio=={runtime['pinned_version']}. "
                     f"Use runtime.install_recipes in {TABLE}; set {PYTHON_ENV} to that venv's python")


def members_from_flag(member) -> list[int]:
    raw = [0] if member is None else (member if isinstance(member, (tuple, list)) else str(member).split(","))
    try:
        values = [int(value) for value in raw]
    except (ValueError, TypeError):
        raise ValueError("--member must be a nonnegative integer or comma-separated member list") from None
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("--member needs distinct nonnegative IDs")
    return values


def _donor(*, table, row, cycle, area, out, force):
    """A donor acquisition owner may supply the policy's bound supplements."""
    from importlib import import_module
    policy = table.get("donor_policy", {})
    owner = policy.get("acquisition_owner")
    if owner:
        return import_module(owner).acquire(row=row, cycle=cycle, area=area, out=out,
                                            force=force, policy=policy)
    selected = resolve_donor_policy(row, cycle)
    workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    os.environ.setdefault("GPUWM_RW_ZARR_WORKERS", str(workers))
    analysis = out / "analysis"
    south, west, north, east = area or (-90, -180, 90, 179.75)
    # The public reader accepts signed longitudes, while E2S crops in 0..360.
    signed = lambda value: (float(value) + 180) % 360 - 180
    recipe = table["runtime"]["donor_acquisition"][selected["provider"]]
    if recipe["kind"] == "arco":
        from gpuwm.era5_arco import retrieve_era5_arco
        retrieve_era5_arco(cycle=cycle, hours=0,
                       area=f"{south},{signed(west)},{north},{signed(east)}",
                       out=analysis, cadence=6, force=force)
        path = analysis / recipe["output"]
    else:
        # Reuse the public fetch route, including its inventory checks and
        # endpoint policy. A donor is an analysis, so only lead zero is read.
        tokens = [sys.executable, "-m", "gpuwm", "fetch", "--source", recipe["source"],
                  "--cycle", cycle.strftime("%Y-%m-%dT%H"), "--hours", "0", "--out", str(analysis)]
        if recipe.get("area"):
            tokens += ["--area", f"{south},{signed(west)},{north},{signed(east)}"]
        tokens += recipe.get("arguments", [])
        if force:
            tokens.append("--force-refetch")
        subprocess.run(tokens, check=True)
        # Some GRIB analyses publish atmosphere and soil in separate objects.
        # Their ordered record streams are one legal GRIB file; declare the
        # input patterns and output name in the acquisition table.
        if recipe.get("inputs"):
            inputs = []
            for pattern in recipe["inputs"]:
                matches = sorted(analysis.glob(pattern))
                if len(matches) != 1:
                    raise ValueError(f"donor recipe expected one {pattern}, found {len(matches)}")
                inputs += matches
            import shutil
            with (analysis / recipe["output"]).open("wb") as merged:
                for item in inputs:
                    with item.open("rb") as source:
                        shutil.copyfileobj(source, merged, length=8 * 1024 * 1024)
        candidates = list(analysis.glob(recipe["output"]))
        if len(candidates) != 1:
            raise ValueError(f"donor recipe expected one {recipe['output']}, found {len(candidates)}")
        path = candidates[0]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {**selected, "analysis": str(path.resolve()),
            "sha256": digest,
            "parallelism": f"{os.environ['GPUWM_RW_ZARR_WORKERS']} threads / network",
            "supplements": {"physical_analysis_surface_data": str(path.resolve())}}


def preparation_profile(args, adapter):
    """Select only a sealed donor variant, from this fetch's own receipt."""
    listing = getattr(args, "input_list", None)
    candidates = []
    if listing is not None:
        listing = Path(listing)
        candidates.append(listing.parent / RECEIPT)
        # Noise-member handoffs bind the shared batch receipt explicitly.
        handoff_path = listing.parent / "prep-arguments.json"
        if handoff_path.is_file():
            handoff = json.loads(handoff_path.read_text())
            if handoff.get("member_identity") == "earth2-noise-seed-v1":
                candidates.insert(0, Path(handoff["parent_receipt"]))
        if listing.is_file():
            candidates += [Path(path).parent / RECEIPT
                           for path in listing.read_text().splitlines() if path.strip()]
    candidates += [Path(path).parent / RECEIPT
                   for path in (getattr(args, "mapped_inputs", None) or ())]
    receipt = next((path for path in candidates if path.is_file()), None)
    if receipt is None:
        return adapter.packaged_profile
    report = json.loads(receipt.read_text())
    if report.get("schema") == "gpuwm-earth2-fetch-v1":
        # Legacy single-trajectory receipts bind only the model's base profile.
        # They have no generated source/donor fields and remain immutable.
        name = report.get("model")
        if not isinstance(name, str) or not name:
            raise ValueError("legacy preparation receipt has no model identity")
        model_id, row = resolve_model(name)
        if adapter.source_id not in ("earth2", f"earth2-{model_id}"):
            raise ValueError("preparation receipt belongs to a different generated source")
        recorded = report.get("model_row", {})
        if not isinstance(recorded, dict) or any(
            recorded.get(key) != row.get(key)
            for key in ("earth2studio_class", "profile", "step_hours")
        ):
            raise ValueError("legacy preparation model declaration differs from the authority")
        if report.get("profile") != row["profile"]:
            raise ValueError("legacy preparation receipt must use its declared base profile")
        return row["profile"]
    model_id, row = resolve_model(report["source"] if adapter.source_id == "earth2"
                                  else adapter.source_id)
    if report["source"] != f"earth2-{model_id}":
        raise ValueError("preparation receipt belongs to a different generated source")
    selected = report["donor"]
    declared = model_table()["donor_policy"]["profile_variants"][row["profile"]]
    if declared.get(selected["id"]) != report["profile"]:
        raise ValueError("preparation donor/profile pair is absent from the source table")
    return report["profile"]


def publish_handoffs(receipt, out):
    """Use only forecast and donor paths recorded by this completed fetch."""
    # The generic acquisition handoff already owns every field needed by go.
    from gpuwm import fetch_routes
    from gpuwm.source_adapters import canonical_source_token
    source = receipt["source"]
    prep_source = canonical_source_token(source)
    for member in receipt["members"]:
        paths = [entry["path"] for entry in receipt["runner"]["files"]
                 if entry["member"] == member and entry["lead_hours"] >= receipt["forecast_start_hour"]]
        listing = out / f"inputs-m{member:04d}.txt"
        listing.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
        tokens = ["--source", prep_source, "--input-list", str(listing),
                  "--author-input-manifest", str(out / f"manifest-m{member:04d}.json")]
        for role, path in sorted(receipt["donor"]["supplements"].items()):
            tokens += ["--supplement", f"{role}={path}"]
        # Every member keeps its own handoff; go consumes one selected
        # trajectory. Fetching a list never merges members into one state.
        target = out if len(receipt["members"]) == 1 else out / f"member-{member:04d}"
        target.mkdir(exist_ok=True)
        fetch_routes.write_prep_arguments(out=target, source=source, prep_source=prep_source,
                                          cycle=datetime.fromisoformat(receipt["cycle"]), tokens=tokens,
                                          member=str(member))


def producer_attributes(mapping):
    """Carry profile-declared field and lead-coordinate units to the writer."""
    document = json.loads(Path(mapping).read_text())
    attributes = document.get("producer_attributes", {})
    lead = document.get("coordinates", {}).get("time", {}).get("lead", {})
    for name in lead.get("selector", {}).get("name", []):
        attributes.setdefault(name, {})["units"] = lead["units"]
    return attributes


def run_model(*, model: str | None, cycle: datetime, hours: int, area, out: Path,
              ic_source: str | None = None, force: bool = False, member=None,
              forecast_start_hour: int = 0, cadence=None) -> dict:
    from gpuwm.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu
    if no_local_gpu():
        raise ValueError(f"{NO_LOCAL_GPU_ENV} is set; Earth2 generation cannot open the local GPU. "
                         "Run generation in an admitted GPU environment with this switch unset")
    model_id, row = resolve_model(model)
    table = model_table()
    runtime = table["runtime"]
    ic_source = ic_source or row.get("initial_condition_source", "era5")
    step = int(row["step_hours"])
    if hours <= 0 or hours % step or forecast_start_hour < 0 or forecast_start_hour % step:
        raise ValueError(f"hours and forecast_start_hour must follow the row's {step} h clock")
    if cadence is not None and int(cadence) != step:
        raise ValueError(f"this model produces every {step} h; cadence={cadence} would misstate its output")
    members = members_from_flag(member)
    population = row.get("members", {}).get("population")
    if population is not None and max(members) >= int(population):
        raise ValueError(f"member ID must be below the row population {population}")
    if len(members) > 1 and not row.get("stochastic", row.get("ensemble", row.get("members", {}).get("kind") in ("native_noise", "native_ensemble", "perturbation"))):
        raise ValueError("this row declares no stochastic ensemble; a member list would repeat one trajectory")
    ic = runtime["initial_condition_sources"].get(ic_source)
    if ic is None:
        raise ValueError(f"initial-condition source {ic_source!r} is absent from the runtime table")
    memory_admission = require_inference_vram(model_id, row)
    python = earth2_python()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    crop = None
    if area is not None:
        south, west, north, east = (float(value) for value in area)
        crop = [max(-90., south - .5), (west - .5) % 360,
                min(90., north + .5), (east + .5) % 360]
        if crop[1] > crop[3]:
            raise ValueError("the selected crop crosses 0 degrees; split it into two areas")
    forecast = out / f"earth2-{model_id}-{cycle:%Y%m%dT%H}.nc"
    existing = list(out.glob(forecast.stem + "-m*-f*.nc"))
    if existing and not force:
        raise ValueError(f"{out} already holds forecast frames; use another --out or --force-refetch")
    if force:
        for path in existing:
            path.unlink()
    cache_policy = runtime["weight_cache"]
    cache = Path(os.environ.get(cache_policy["environment"], cache_policy["default"])).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    request = {"memory_admission": memory_admission,
               "model_class": row["earth2studio_class"], "ic_class": ic["earth2studio_class"],
               "init": cycle.isoformat(), "steps": (hours + forecast_start_hour) // step,
               "output": str(forecast), "area": crop, "members": members,
               "seed": int(row.get("seed", row.get("member", {}).get("seed", 0))) + members[0], "weight_cache": str(cache),
               "pinned_version": runtime["pinned_version"], "data_cache": str(out / "runtime-cache"),
               "loader_kwargs": row.get("load_model_kwargs", {}) | row.get("loader_kwargs", {}),
               "model_package": row.get("weights", {}).get("uri"),
               "expected_weights": row.get("weights", {}),
               "loader_preparations": row.get("loader_preparations", []),
               "model_calls": row.get("model_calls", []),
               "output_grid": row.get("output_grid"),
               "ic_kwargs": dict(ic.get("kwargs", {}), **row.get("ic_kwargs", {})),
               "runtime_env": row.get("runtime_env", {})}
    if row.get("members"):
        policy = row["members"]
        request["member_policy"] = policy
        request["seeds"] = [int(policy["first_seed"]) + member for member in members]
        request["seed"] = request["seeds"][0]
        request["batch_size"] = policy.get("batch_size", 1)
    from gpuwm.source_authorities import packaged_authorities
    mapping = packaged_authorities(row["profile"])["mapping"]
    request["output_attributes"] = producer_attributes(mapping)
    if (row.get("weights") or {}).get("host") == "huggingface":
        from gpuwm.hf_fetch import fetch as fetch_weights
        request["package_root"] = fetch_weights(model_id)["package_root"]
    request_path = out / "earth2-request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    log = out / "earth2-runner.log"
    print(f"fetch: generating {model_id} members {members}; log {log}", flush=True)
    runner = Path(__file__).with_name("earth2_runner.py")
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([python, str(runner), str(request_path)], stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Earth2 runtime exited {completed.returncode}; see {log}")
    report = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    donor_start = time.perf_counter()
    donor = _donor(table=table, row=row, cycle=cycle, area=crop, out=out, force=force)
    receipt = {"schema": "gpuwm-earth2-fetch-v2", "source": f"earth2-{model_id}",
               "model": model_id, "model_row": row,
               "model_table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(),
               "initial_condition_source": ic_source, "cycle": cycle.isoformat(),
               "hours": hours, "forecast_start_hour": forecast_start_hour,
               "members": members, "forecast": [item["path"] for item in report["files"]],
               "analysis": donor.get("analysis"), "donor": donor, "profile": donor["profile"],
               "runner": report, "stages": report["stages"] + [
                   {"stage": "donor", "seconds": round(time.perf_counter() - donor_start, 3),
                    "parallelism": donor.get("parallelism", "native threads / network; see donor receipt")}],
               "wall_seconds": round(time.perf_counter() - started, 3)}
    (out / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    publish_handoffs(receipt, out)
    (out / "fetch-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def fetch(args) -> int:
    from gpuwm.fetch import _resolve_area, parse_cycle
    if args.cycle in (None, "latest") or args.hours is None or args.out is None:
        raise ValueError("Earth2 generation needs an explicit --cycle, --hours and --out")
    area = _resolve_area(args)
    if area is not None:
        area = (area.lat_south, area.lon_west, area.lat_north, area.lon_east)
    from gpuwm.source_adapters import get_source_adapter
    if str(args.source).startswith("earth2-") and getattr(args, "model", None):
        if resolve_model(args.source)[0] != resolve_model(args.model)[0]:
            raise ValueError("--source and --model select different model rows")
    if getattr(args, "earth2_members", None) or str(getattr(args, "member", "") or "").startswith("n"):
        from gpuwm.earth2_members import fetch_member
        fetch_member(args, model=getattr(args, "model", None) or get_source_adapter(args.source).model_key,
                     cycle=parse_cycle(args.cycle, "era5"), area=area)
        return 0
    source = str(args.source)
    model = getattr(args, "model", None) or get_source_adapter(source).model_key

    run_model(model=model, cycle=parse_cycle(args.cycle, "era5"), hours=int(args.hours), area=area,
              out=Path(args.out), force=bool(args.force_refetch), member=getattr(args, "member", None),
              forecast_start_hour=getattr(args, "forecast_start_hour", None) or 0,
              cadence=getattr(args, "cadence", None))
    return 0


def prepare_launch(args):
    """Create one ordinary regional CONFIG for the chosen generated member."""
    import argparse
    import shutil
    import tomllib
    from gpuwm import domain_wizard
    from gpuwm.toml_document import emit_experiment_toml
    config = getattr(args, "config", None)
    if config is None:
        if not all(getattr(args, key, None) for key in ("source", "init", "center")):
            raise ValueError("generation needs --source, --init and --center, or CONFIG")
        model_id, _ = resolve_model(args.source)
        values = members_from_flag(getattr(args, "member", None))
        if len(values) != 1:
            raise ValueError("go launches one trajectory; generate a member list with fetch and use its member handoffs")
        identity = {key: getattr(args, key, None) for key in
                    ("source", "init", "center", "hours", "extent_km", "card", "member", "history_interval", "geog_root")}
        identity = {key: str(value) if isinstance(value, Path) else value
                    for key, value in identity.items()}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        root = Path(getattr(args, "outdir", None) or "earth2-forecast").resolve() / "configurations" / key
        root.mkdir(parents=True, exist_ok=True)
        config = root / "forecast.toml"
        parser = argparse.ArgumentParser()
        domain_wizard.register_cli(parser.add_subparsers(dest="command"))
        argv = ["domain", "--point=" + args.center, "--source", "earth2-" + model_id,
                "--cycle", args.init, "--hours", str(args.hours), "--root-dx", "3",
                "--point-extent-km", str(args.extent_km), "--history-interval",
                str(getattr(args, "history_interval", None) or 3600),
                "--nz", "49", "--out", str(config)]
        for flag in ("card", "geog_root"):
            if getattr(args, flag, None):
                argv += ["--" + flag.replace("_", "-"), str(getattr(args, flag))]
        code = domain_wizard.domain_main(parser.parse_args(argv))
        if code:
            raise ValueError(f"domain creation exited {code}")
        args.source = args.init = args.center = None
    elif getattr(args, "source", None) or getattr(args, "init", None) or getattr(args, "center", None):
        raise ValueError("CONFIG already owns its source, cycle and domain; use --cycle to re-time it")
    elif getattr(args, "history_interval", None) is not None:
        raise ValueError("CONFIG owns its history interval; edit the configuration")
    config = Path(config)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    forcing = raw.get("fetch", {})
    declared = forcing.get("source")
    if declared in ("earth2", "earth2studio"):
        key, _ = resolve_model(forcing.get("model"))
        forcing["source"] = "earth2-" + key
        forcing.pop("model", None)
        target = config.with_name(config.stem + ".generated.toml")
        target.write_text(emit_experiment_toml(raw), encoding="utf-8")
        namelist = config.with_name(config.stem + ".namelist.wps")
        if namelist.is_file():
            shutil.copyfile(namelist, target.with_name(target.stem + ".namelist.wps"))
        config = target
    elif forcing.get("model") and resolve_model(declared)[0] != resolve_model(forcing["model"])[0]:
        raise ValueError("CONFIG source and model select different model rows")
    if getattr(args, "member", None) is not None:
        members = members_from_flag(args.member)
        if len(members) != 1:
            raise ValueError("go requires one source member; use fetch for a GPU member batch")
        raw = tomllib.loads(config.read_text(encoding="utf-8"))
        raw["fetch"]["member"] = str(members[0])
        target = config.with_name(config.stem + f".member-{members[0]:04d}.toml")
        target.write_text(emit_experiment_toml(raw), encoding="utf-8")
        namelist = config.with_name(config.stem + ".namelist.wps")
        if namelist.is_file():
            shutil.copyfile(namelist, target.with_name(target.stem + ".namelist.wps"))
        config = target
    args.config = config


def validate_member(source, member):
    if isinstance(member, str) and member.startswith("n"):
        from gpuwm.earth2_members import validate_member as validate_noise
        return [validate_noise(member)]
    members = members_from_flag(member)
    if source != "earth2":
        _, row = resolve_model(source)
        if len(members) > 1 and not row.get("stochastic", row.get("ensemble", row.get("members", {}).get("kind") in ("native_noise", "native_ensemble", "perturbation"))):
            raise ValueError("the selected row declares no stochastic ensemble")
    return members


def verify_handoff(hints, handoff, *, out):
    """Bind generated-member provenance and actual bytes before preparation."""
    if isinstance(hints.get("member"), str) and hints["member"].startswith("n"):
        from gpuwm.earth2_members import verify_handoff as verify_noise
        return verify_noise(hints, handoff, out=out)
    from concurrent.futures import ThreadPoolExecutor
    from gpuwm.fetch import parse_cycle
    from gpuwm.member_grammar import MemberIdentityRefusal
    from gpuwm.source_adapters import get_source_adapter
    source = get_source_adapter(hints["source"]).source_id
    member, = members_from_flag(hints.get("member"))
    if handoff.get("source") != source or handoff.get("member") != str(member):
        raise MemberIdentityRefusal("Generated source/member differs from the selected handoff")
    if parse_cycle(handoff["cycle"], "era5") != parse_cycle(hints["cycle"], "era5"):
        raise MemberIdentityRefusal("Generated forcing cycle differs from the selected cycle")
    arguments = list(handoff["argv"])
    if arguments.count("--input-list") != 1:
        raise MemberIdentityRefusal("Generated forcing needs exactly one bound input-list")
    listing = Path(arguments[arguments.index("--input-list") + 1]).resolve(strict=True)
    fetched = json.loads((listing.parent / RECEIPT).read_text())
    if (fetched["source"] != source or
            parse_cycle(fetched["cycle"], "era5") != parse_cycle(hints["cycle"], "era5")):
        raise MemberIdentityRefusal("Generated receipt source/cycle differs from the request")
    expected = [item for item in fetched["runner"]["files"]
                if item["member"] == member and item["lead_hours"] >= fetched["forecast_start_hour"]]
    paths = [str(Path(line).resolve(strict=True)) for line in listing.read_text().splitlines() if line.strip()]
    if not paths or paths != [item["path"] for item in expected] or len(set(paths)) != len(paths):
        raise MemberIdentityRefusal("Generated input-list differs from the member's recorded trajectory")
    def check(item):
        with Path(item["path"]).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise MemberIdentityRefusal("Generated forcing hash differs: " + item["path"])
        return {"path": item["path"], "sha256": digest, "member": member,
                "lead_hours": item["lead_hours"]}
    # HDF5 metadata is inspected serially because its C library is not thread
    # safe; the full-file reads and hashes use independent thread workers.
    from netCDF4 import Dataset
    for item in expected:
        with Dataset(item["path"]) as frame:
            if (int(frame.getncattr("earth2_member")) != member or
                    int(frame.getncattr("earth2_generation_seed")) != item.get("seed", fetched["runner"]["seed"])):
                raise MemberIdentityRefusal("Generated frame's member attribute differs: " + item["path"])
    workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(check, expected))
    receipt = {"schema": "arwen.forcing-member.v1", "source": source, "member": str(member),
               "cycle": handoff["cycle"], "model_table_sha256": fetched["model_table_sha256"],
               "weights": fetched["runner"]["weights"], "seed": expected[0].get("seed", fetched["runner"]["seed"]),
               "files": files}
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(encoded.encode()).hexdigest()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"member-inputs-{key}.txt"
    text = "".join(path + "\n" for path in paths)
    try:
        with target.open("x") as stream:
            stream.write(text)
    except FileExistsError:
        if target.read_text() != text:
            raise MemberIdentityRefusal("Verified generated-member input-list is corrupt")
    receipt["input_list"] = str(target.resolve())
    (out / f"member-verification-{key}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
