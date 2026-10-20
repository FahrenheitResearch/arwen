"""Shared member ancestry. V2 retains v1 fields and canonical JSON labels."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

SCHEMA = "gpuwm-member-provenance.v2"
READABLE_SCHEMAS = ("gpuwm-member-provenance.v1", SCHEMA)
FAMILIES = ("ai", "physics")


def provenance_label(row):
    return json.dumps({k: v for k, v in row.items() if k != "label"}, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def validate_record(row, member=None):
    if (not isinstance(row, dict) or row.get("schema") not in READABLE_SCHEMAS
            or (member is not None and row.get("member") != member)
            or row.get("label") != provenance_label(row)):
        raise ValueError("member provenance must retain its member ID and canonical JSON label")
    if row["schema"] == SCHEMA:
        if row.get("parent_family") not in FAMILIES or not row.get("model_key"):
            raise ValueError("v2 lineage requires a parent family and model key")
        weight = row.get("family_weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("v2 lineage family_weight must be finite and in [0, 1]")
        for key in ("weight_revision", "perturbation_recipe"):
            if not isinstance(row.get(key), dict):
                raise ValueError(f"v2 lineage requires {key}")
    return deepcopy(row)


def family_weights(value=None, *, families=()):
    """Validate mass or default to each family's fraction of members."""
    counts = Counter(families)
    if value is None or value == {}:
        total = sum(counts.values())
        return {key: count / total for key, count in sorted(counts.items())} if total else {}
    if isinstance(value, str):
        parsed = {}
        for item in value.split(","):
            key, separator, number = item.strip().partition("=")
            if not separator or key in parsed:
                raise ValueError("family weights need distinct ai=...,physics=... entries")
            try:
                parsed[key] = float(number)
            except ValueError as error:
                raise ValueError("family weights must be finite numbers") from error
        value = parsed
    if not isinstance(value, dict) or not value or set(value) - set(FAMILIES):
        raise ValueError("family weights take ai and physics entries")
    result = {}
    for key, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("family weights must be finite numbers in [0, 1]")
        result[key] = float(number)
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("family weights must sum to 1")
    if counts and (set(counts) - result.keys() or any(result[k] > 0 and k not in counts for k in result)):
        raise ValueError("family weights must cover represented families; absent families cannot carry mass")
    return result


def _defined_values(value):
    # JSON receipts use null for absent optional metadata; TOML has no null.
    if isinstance(value, dict):
        return {key: _defined_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_defined_values(item) for item in value]
    return deepcopy(value)


def parent_receipt(value):
    if value is None:
        return {}
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("parent_receipt must be a receipt mapping or JSON path")
    return _defined_values(value)


def model_identity(source, model=None):
    """Resolve model aliases through the source registry's declared table."""
    from gpuwm.source_adapters import canonical_source_token, get_source_adapter, model_rows
    adapter = get_source_adapter(source)
    row = adapter.model_row or {}
    key = adapter.model_key
    if key is not None:
        if model is not None and model not in (key, *row.get("aliases", ())):
            raise ValueError("parent receipt model differs from selected source model")
        return adapter, canonical_source_token(source), row
    if model is not None:
        matches = [(name, item) for name, item in model_rows(adapter).items()
                   if model == name or model in item.get("aliases", ())]
        if len(matches) != 1:
            raise ValueError(f"unknown model {model!r} in {adapter.model_table}")
        selected, row = matches[0]
        token = canonical_source_token(adapter.source_id + "-" + selected)
        return get_source_adapter(token), token, row
    return adapter, canonical_source_token(source), row


def _receipt_member(receipt, source_member):
    members = receipt.get("members")
    if not members:
        return receipt
    if all(isinstance(member, dict) for member in members):
        candidates = [member for member in members if str(member.get("id")) == str(source_member)]
        if len(candidates) != 1:
            raise ValueError("parent batch receipt must identify exactly one source member")
        return {**receipt, **candidates[0]}
    # Fetch receipts list numeric member IDs. Their runner records the seed on
    # every lead file; a batch-global RNG seed cannot stand in for another member.
    selected = [index for index, member in enumerate(members) if str(member) == str(source_member)]
    if not selected and source_member is None and len(members) == 1:
        selected = [0]
    if len(selected) != 1:
        raise ValueError("parent fetch receipt must identify exactly one source member")
    index = selected[0]
    runner = receipt.get("runner") or {}
    seeds = {frame["seed"] for frame in runner.get("files", ())
             if str(frame["member"]) == str(members[index]) and "seed" in frame}
    if runner.get("seeds"):
        seeds.add(runner["seeds"][index])
    if not seeds and len(members) == 1 and "seed" in runner:
        seeds.add(runner["seed"])
    if len(seeds) > 1:
        raise ValueError("parent fetch receipt records conflicting member seeds")
    return {**receipt, **({"seed": seeds.pop()} if seeds else {})}


def build_lineage(base, *, model=None, receipt=None, family_weight=1.0, stochastic=None):
    """Build a TOML-safe v2 row. Missing runtime hashes are stated explicitly."""
    record = deepcopy(base)
    record.pop("label", None)
    receipt = parent_receipt(receipt)
    adapter, model_key, model_row = model_identity(record["source"], model or receipt.get("model"))
    if model is not None and receipt.get("model") is not None:
        _, receipt_key, _ = model_identity(record["source"], receipt["model"])
        if receipt_key != model_key:
            raise ValueError("parent receipt model differs from selected source model")
    receipt = _receipt_member(receipt, record.get("source_member"))
    runner = receipt.get("runner") or {}
    revision = receipt.get("weight_revision") or receipt.get("weights") or runner.get("weight_revision") or runner.get("weights")
    if revision is None:
        weights = model_row.get("weights") or {}
        revision = {k: weights[k] for k in ("package", "revision", "sha256", "hash", "files", "uri", "checkpoint") if k in weights}
        revision["upstream_model_id"] = adapter.upstream_model_id or model_key
        if adapter.upstream_model_version:
            revision["version"] = adapter.upstream_model_version
        revision["status"] = "declared" if adapter.parent_family == "physics" else "runtime-evidence-not-recorded"
    elif isinstance(revision, list):
        revision = {"files": deepcopy(revision)}
    elif not isinstance(revision, dict):
        revision = {"revision": str(revision)}
    runtime = {key: runner[key] for key in adapter.lineage_runtime_keys if key in runner}
    if runtime:
        revision = {**revision, "runtime": runtime}
    record.update(schema=SCHEMA, parent_family=adapter.parent_family, model_key=model_key,
                  weight_revision=_defined_values(revision), family_weight=float(family_weight))
    for key in ("noise_seed", "noise_vector_id", "noise_id"):
        if key in receipt or key in runner:
            record[key] = deepcopy(receipt.get(key, runner.get(key)))
    if "seed" in receipt:
        record["noise_seed"] = receipt["seed"]
    if runner.get("seed_scope"):
        record["noise_seed_scope"] = runner["seed_scope"]
    donor = receipt.get("donor")
    if donor is None and adapter.lineage_donor_source:
        donor = {"source": adapter.lineage_donor_source, "cycle": receipt.get("cycle", record["cycle"])}
    record["donor"] = deepcopy(donor) if donor else {}
    if "composition" in receipt:
        record["composition"] = deepcopy(receipt["composition"])
    recipe = {key: deepcopy(record[key]) for key in
              ("rung", "seed", "lag_hours", "physics_arm", "physics", "surface", "perturbation", "branch") if key in record}
    if stochastic:
        recipe["stochastic"] = deepcopy(stochastic)
    record["perturbation_recipe"] = recipe
    record["label"] = provenance_label(record)
    return validate_record(record)


def weight_records(records, weights=None):
    """Assign family mass while preserving legacy v1 records on load."""
    records = [validate_record(row) for row in records]
    families = [row["parent_family"] for row in records if row["schema"] == SCHEMA]
    selected = family_weights(weights, families=families)
    for row in records:
        if row["schema"] == SCHEMA:
            row["family_weight"] = selected[row["parent_family"]]
            row["label"] = provenance_label(row)
    return tuple(records), selected


def trajectory_records(trajectories, *, base_seed=0, variants=(), stochastic=None, weights=None):
    from gpuwm.ensemble.seeds import member_seed
    records = []
    for i, item in enumerate(trajectories):
        variant = variants[i] if variants else {}
        base = {"member": i, "source": item["source"], "cycle": str(item["cycle"]),
                "rung": "source-member", "seed": member_seed(base_seed, i),
                "physics_arm": variant.get("name", "base"), "physics": variant.get("physics", {}),
                "surface": variant.get("surface", {})}
        if item.get("member") is not None:
            base["source_member"] = item["member"]
        if variant.get("wind"):
            base["perturbation"] = {"kind": "uniform-wind", "amplitude_m_s": variant["wind"]}
        records.append(build_lineage(base, model=item.get("model"), receipt=item.get("parent_receipt"), stochastic=stochastic))
    return weight_records(records, weights)


def lineage_summary(records):
    return {"schema": SCHEMA, "family_weights": {r["parent_family"]: r["family_weight"] for r in records if r.get("schema") == SCHEMA},
            "members": [deepcopy(r) for r in records]}


def member_display(row, fallback):
    if not row or row.get("schema") != SCHEMA:
        return fallback
    return f"{fallback} ({row['parent_family']}: {row['model_key']})"


def branch_lineage(parent, branch, *, member=None, seed=None):
    """Keep the complete ancestor and append this branch."""
    row = validate_record(parent)
    row["parent"] = deepcopy(parent)
    row["branch"] = deepcopy(branch)
    if member is not None:
        row["member"] = member
    if seed is not None:
        row["seed"] = seed
    if row["schema"] == SCHEMA:
        row["perturbation_recipe"] = {**row["perturbation_recipe"], "branch": deepcopy(branch),
                                      **({"seed": seed} if seed is not None else {})}
    row["label"] = provenance_label(row)
    return row


def enrich_request(request, fetches):
    """Bind completed acquisition receipts before identities and wrfout are sealed.

    Receipt filenames are registry data. Fetch directories come from the exact
    member configuration, not a recursive search or a model-name convention.
    Reads each small shared batch receipt once, retaining no weather arrays.
    """
    from dataclasses import replace
    from gpuwm.source_adapters import get_source_adapter
    if not request.provenance:
        return request
    cache, records = {}, []
    for row, fetch in zip(request.provenance, fetches, strict=True):
        adapter = get_source_adapter(row["source"])
        if row["schema"] != SCHEMA or not adapter.lineage_receipts:
            records.append(row)
            continue
        root = Path(next((fetch[key] for key in adapter.lineage_root_keys if fetch.get(key)), fetch["out"]))
        candidates = [folder / name for folder in (root, root.parent) for name in adapter.lineage_receipts]
        selected = next((path for path in candidates if path.is_file()), None)
        if selected is None:
            records.append(row)
            continue
        if selected not in cache:
            cache[selected] = parent_receipt(selected)
        records.append(build_lineage(row, model=fetch.get("model"), receipt=cache[selected],
                                     family_weight=row["family_weight"], stochastic=request.stochastic))
    return replace(request, provenance=tuple(records))


def record_stochastic(records, stochastic):
    if not stochastic:
        return records
    result = []
    for row in records:
        row = deepcopy(row)
        if row["schema"] == SCHEMA:
            row["perturbation_recipe"]["stochastic"] = deepcopy(stochastic)
            row["label"] = provenance_label(row)
        result.append(row)
    return tuple(result)


def bind_runtime_stochastic(request, authorities):
    """Seal resolved native selectors and process settings before restart IDs."""
    from dataclasses import replace
    if not request.provenance or not authorities:
        return request
    records = []
    for row in request.provenance:
        row = deepcopy(row)
        authority = authorities.get(row["member"])
        if row["schema"] == SCHEMA and authority:
            selected = {key: _defined_values(authority[key]) for key in
                        ("spp_selectors", "process_configurations", "wrf_seed_labels") if key in authority}
            if selected:
                row["perturbation_recipe"]["runtime_stochastic"] = selected
                row["label"] = provenance_label(row)
        records.append(row)
    return replace(request, provenance=tuple(records))
