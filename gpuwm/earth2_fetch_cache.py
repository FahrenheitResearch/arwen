"""Reuse sealed acquisition receipts without copying another run's arrays.

The index is content keyed. GPUWM_EARTH2_FETCH_RECEIPTS may name additional
read-only earth2-run.json files, separated by the platform path separator.
Fetch-v2 seals member/lead files. The original single-member contract is
accepted only after its installed defaults and native file inventory are checked.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(spec):
    from gpuwm.cyclone_search import resolve_region
    identity = {key: spec[key] for key in ("model", "cycle", "hours", "area", "ic_source", "table")}
    identity["search_region"] = resolve_region(spec["area"], automatic=spec["storm"] == "auto")
    return identity


def _index(spec):
    from gpuwm.earth2_run import model_table
    policy = model_table()["runtime"]["weight_cache"]
    weights = Path(os.environ.get(policy["environment"], policy["default"])).expanduser()
    root = Path(os.environ.get("GPUWM_EARTH2_FETCH_CACHE", str(weights / "fetch-receipts")))
    key = hashlib.sha256(json.dumps(_identity(spec), sort_keys=True).encode()).hexdigest()
    return root / (key + ".json")



def _relocate(receipt, request, directory):
    """Resolve a copied acquisition tree only when its original files are gone."""
    output = request.get("output")
    if not output:
        return
    original_root = Path(output).parent
    changes = {}

    def local(value):
        original = Path(value)
        if original.is_file():
            return str(original)
        try:
            relative = original.relative_to(original_root)
        except ValueError as error:
            raise ValueError(f"fetch artifact is outside the producer output tree: {original}") from error
        candidate = directory / relative
        if not candidate.is_file():
            raise ValueError(f"copied fetch artifact is missing: {candidate} (producer {original})")
        changes[str(original)] = str(candidate)
        return str(candidate)

    if isinstance(receipt.get("forecast"), str):
        receipt["forecast"] = local(receipt["forecast"])
    for entry in receipt.get("runner", {}).get("files", []):
        entry["path"] = local(entry["path"])
    if receipt.get("analysis"):
        receipt["analysis"] = local(receipt["analysis"])
    donor = receipt.get("donor", {})
    if donor.get("analysis"):
        donor["analysis"] = local(donor["analysis"])
    for role, value in donor.get("supplements", {}).items():
        donor["supplements"][role] = local(value)
    if changes:
        receipt["artifact_relocation"] = dict(producer_output_root=str(original_root),
            local_output_root=str(directory), paths=changes,
            verification="original receipt unchanged; forecast hashes and current artifact hashes checked")


def covers_area(available, wanted):
    """Containment of south,west,north,east rectangles, including 0..360 crops."""
    if available is None:
        return True
    if wanted is None:
        return False
    if any(len(area) != 4 or not all(math.isfinite(float(v)) for v in area)
           for area in (available, wanted)):
        raise ValueError("fetch area needs four finite bounds")
    south, west, north, east = map(float, available)
    s, w, n, e = map(float, wanted)
    width = (east - west) % 360 if abs(east - west) < 360 else 360
    needed = (e - w) % 360 if abs(e - w) < 360 else 360
    offset = (w - west) % 360
    return south <= s and n <= north and (width == 360 or offset + needed <= width + 1e-8)


def check_domain_coverage(receipt, config):
    """Check the authored parent and moving corridor before allocating prep work."""
    import tomllib
    coverage = receipt.get("cache_reuse", {}).get("coverage")
    if coverage is None or coverage["available_area"] is None:
        return
    planned = tomllib.loads(Path(config).read_text())
    area = planned.get("fetch", {}).get("area")
    if area is None:
        # Cyclone configurations whose acquisition owner is global omit a
        # fetch hint. Reuse the wizard's projected root envelope and margin;
        # the following nest is already constrained to that parent corridor.
        from gpuwm.domain_wizard import experiment_from_text, fetch_area_hint
        experiment = experiment_from_text(Path(config).read_text(), source=str(config))
        parent = experiment.domains[0].run
        area = fetch_area_hint(planned["projection"], parent.nx, parent.ny,
                               source=receipt["source"], root_dx_m=parent.dx)
    if isinstance(area, str):
        area = [float(v) for v in area.split(",")]
    if area is None or not covers_area(coverage["available_area"], area):
        raise ValueError(f"earth2 input area {coverage['available_area']} does not cover "
                         f"the authored storm domain/corridor {area}; supply a larger completed fetch")
    coverage["checked_domain_area"] = area


def _verify(spec, path, *, explicit_input=False):
    """A longer forecast or a spatial superset is reusable after identity checks."""
    from gpuwm.earth2_run import resolve_model
    path = Path(path).resolve()
    receipt = json.loads(path.read_text())
    if receipt.get("schema") not in ("gpuwm-earth2-fetch-v1", "gpuwm-earth2-fetch-v2"):
        raise ValueError("fetch cache needs a completed gpuwm-earth2-fetch-v1 or v2 receipt")
    _, row = resolve_model(spec["model"])
    cycle = datetime.fromisoformat(receipt["cycle"].replace("Z", "+00:00"))
    wanted = datetime.fromisoformat(spec["cycle"].replace("Z", "+00:00"))
    if cycle.replace(tzinfo=cycle.tzinfo or timezone.utc) != wanted:
        raise ValueError("fetch cache cycle differs")
    legacy = receipt["schema"] == "gpuwm-earth2-fetch-v1"
    checks = {"model": spec["model"], "model_row": row,
              "initial_condition_source": spec["ic_source"], "members": [0],
              "forecast_start_hour": 0}
    if legacy:
        checks = {key: checks[key] for key in ("model", "initial_condition_source")}
    for key, value in checks.items():
        if receipt.get(key) != value:
            raise ValueError(f"fetch cache {key} differs")
    if receipt["hours"] < spec["hours"]:
        raise ValueError("fetch cache forecast is shorter than the requested window")
    request_path = path.parent / "earth2-request.json"
    request = json.loads(request_path.read_text())
    from gpuwm.cyclone_search import resolve_region
    region = resolve_region(spec["area"], automatic=spec["storm"] == "auto")
    if spec["storm"] == "auto" and spec["area"] is None and request["area"] is not None:
        from gpuwm.cyclone_search import parse_area
        region = dict(id="forecast-area", area=parse_area(request["area"]),
                      basis="receipt-checked regional forecast area")
    area = region["area"]
    # Explicit regional jobs retain their preparation halo. Configured search
    # bounds select cycle positions; the authored parent is checked separately.
    crop = area
    if spec["area"] is not None:
        crop = [max(-90., area[0] - .5), (area[1] - .5) % 360,
                min(90., area[2] + .5), (area[3] + .5) % 360]
    if not (explicit_input and area is None) and not covers_area(request["area"], crop):
        raise ValueError("fetch cache area differs; available area does not cover the requested domain")
    receipt["input_coverage"] = dict(available_area=request["area"], requested_area=crop,
        search_region=region,
        search_domain="explicit input coverage" if explicit_input and area is None else region["basis"])
    _relocate(receipt, request, path.parent)
    if legacy:
        from gpuwm.earth2_legacy_cache import verify
        receipt["legacy_compatibility"] = verify(spec, receipt, request,
                                                allow_regional=explicit_input or area is not None)
        artifacts = {str(request_path): _hash(request_path)}
        for artifact in (receipt["forecast"], receipt["analysis"]):
            artifacts[str(artifact)] = _hash(artifact)
        return receipt, None, artifacts
    files = [entry for entry in receipt["runner"]["files"]
             if entry["member"] == 0 and 0 <= entry["lead_hours"] <= spec["hours"]]
    expected = list(range(0, spec["hours"] + 1, int(row["step_hours"])))
    if sorted(entry["lead_hours"] for entry in files) != expected:
        raise ValueError("fetch cache has missing or repeated member/lead frames")
    artifacts = {str(request_path): _hash(request_path)}
    for entry in files:
        if _hash(entry["path"]) != entry["sha256"]:
            raise ValueError(f"fetch cache forecast hash differs: {entry['path']}")
        artifacts[entry["path"]] = entry["sha256"]
    for donor in set(receipt["donor"]["supplements"].values()):
        artifacts[donor] = _hash(donor)
    return receipt, files, artifacts


def remember(spec, path):
    _, _, artifacts = _verify(spec, path)
    index = _index(spec)
    index.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(identity=_identity(spec), receipt=str(Path(path).resolve()),
                 receipt_sha256=_hash(path), artifacts=artifacts)
    temporary = index.with_suffix(f".{os.getpid()}.new")
    temporary.write_text(json.dumps(entry, sort_keys=True, indent=2) + "\n")
    temporary.replace(index)


def reuse(spec, out, input_dir=None):
    """Return a bound local receipt, or None when no cache was offered.

    A named but incompatible or corrupt cache refuses before another expensive
    model run. The caller can remove the cache entry or choose another output.
    """
    from gpuwm.earth2_run import publish_handoffs
    started = time.monotonic()
    index = _index(spec)
    candidates = ([Path(input_dir).expanduser() / "earth2-run.json"] if input_dir is not None else
                  [Path(p) for p in os.environ.get("GPUWM_EARTH2_FETCH_RECEIPTS", "").split(os.pathsep) if p])
    sealed = None
    if input_dir is None and index.is_file():
        sealed = json.loads(index.read_text())
        if sealed["identity"] != _identity(spec):
            raise ValueError("fetch cache index identity differs")
        # Raw forecasts are disposable. A retained index whose producer or
        # arrays have been retired is an ordinary miss, not a broken install.
        if all(Path(path).is_file() for path in [sealed["receipt"], *sealed["artifacts"]]):
            if _hash(sealed["receipt"]) != sealed["receipt_sha256"]:
                raise ValueError("fetch cache receipt hash differs")
            candidates.insert(0, Path(sealed["receipt"]))
        else:
            sealed = None
    if not candidates:
        return None
    errors = []
    for path in candidates:
        try:
            receipt, files, artifacts = _verify(spec, path, explicit_input=input_dir is not None)
            if sealed and path == Path(sealed["receipt"]) and artifacts != sealed["artifacts"]:
                raise ValueError("fetch cache donor or forecast hash differs")
        except (ValueError, KeyError, OSError) as error:
            errors.append(f"{path}: {error}")
            continue
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        receipt["cache_reuse"] = dict(parent_receipt=str(path.resolve()), parent_sha256=_hash(path),
                                     verified_artifacts=artifacts,
                                     coverage=receipt.pop("input_coverage"),
                                     wall_seconds=round(time.monotonic() - started, 3))
        receipt["hours"] = spec["hours"]
        if files is not None:
            receipt["runner"]["files"] = files
            receipt["forecast"] = [entry["path"] for entry in files]
        else:
            receipt.update(source="earth2-" + spec["model"], members=[0], forecast_start_hour=0)
            receipt["donor"] = dict(source=spec["ic_source"], cycle=spec["cycle"],
                analysis=receipt["analysis"], supplements={"physical_analysis_surface_data": receipt["analysis"]})
        (out / "earth2-run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if files is not None:
            publish_handoffs(receipt, out)
        else:
            from gpuwm.fetch_routes import write_prep_arguments
            from gpuwm.source_adapters import canonical_source_token
            source = receipt["source"]
            tokens = ["--source", canonical_source_token(source), "--input", receipt["forecast"],
                      "--supplement", "physical_analysis_surface_data=" + receipt["analysis"],
                      "--author-input-manifest", str(out / "manifest-m0000.json")]
            write_prep_arguments(out=out, source=source, prep_source=canonical_source_token(source),
                                 cycle=datetime.fromisoformat(spec["cycle"]), tokens=tokens, member="0")
        return receipt
    raise ValueError("No compatible completed fetch cache: " + "; ".join(errors))
