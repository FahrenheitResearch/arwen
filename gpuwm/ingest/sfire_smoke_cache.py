"""Explicit zero-background initialization of endogenous SFIRE bulk smoke.

This creates a new prepared cache. It does not relax the normal reader's
identity rules, change meteorology or reinterpret a model checkpoint.
"""
from __future__ import annotations

from dataclasses import MISSING, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from collections.abc import Mapping

import numpy as np


def augment_sfire_smoke_cache(source, target, domain_config):
    from gpuwm.config import RunConfig, validate_chem_config
    from gpuwm.chem_table import load
    from gpuwm.core.chem_context import LEDGER_BUCKETS, allocation_shape, chem_processes
    from gpuwm.core.chem_state import LEDGER_TOTALS, ledger_attr, process_attr
    from gpuwm.ingest.prepared_cache import (
        PREPARED_CACHE_SCHEMA, PreparedCacheReader, PreparedCacheMismatchError,
        _BundleWriter, _canonical, _json_copy, _prepared_cache_staging_path,
        compare_prepared_domain_config, prepared_domain_config_identity,
    )
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("bulk smoke augmentation requires a distinct cache root")
    if target.exists():
        raise FileExistsError("preserve the existing derived cache and choose a new target")
    raw = (source / "header.json").read_bytes()
    original = json.loads(raw)
    reader = PreparedCacheReader(source, expected_identity=original["identity"])
    document = (_json_copy(domain_config) if isinstance(domain_config, Mapping)
                else prepared_domain_config_identity(domain_config))
    cfg = RunConfig(**document["run"])
    if document["run"].get("chem_sets") != cfg.chem_sets:
        raise ValueError("bulk augmentation requires the canonical prepared domain configuration")
    validate_chem_config(cfg)
    table = load(cfg)
    if (table is None or len(table.rows) != 1 or table.enabled_sources
            or set(table.processes) - {"emission.sfire", "mixing.vertmx"}
            or "emission.sfire" not in table.processes):
        raise ValueError("cache augmentation initializes only the endogenous SFIRE bulk row")
    row = table.rows[0]
    if row.emissions or row.boundary or row.units != "ug kg-1" or row.default_inflow != 0:
        raise ValueError("bulk cache augmentation cannot fabricate external chemistry or inflow")
    prior_domain = original["identity"].get("domain_config", {})
    prior_run = prior_domain.get("run", {})
    if prior_run.get("chem_sets") or prior_run.get("chem_sources") or "chem_table_identity" in prior_domain:
        raise PreparedCacheMismatchError("bulk cache augmentation requires a chemistry-off predecessor")
    if any(key.startswith(("state/chem_", "state/chemdiag_")) for key in reader.arrays):
        raise PreparedCacheMismatchError("predecessor already carries chemistry state")
    cold = _json_copy(document)
    cold.pop("chem_table_identity", None)
    cold["run"]["chem_sets"] = ""
    cold["run"]["chem_sources"] = ""
    cold["run"]["fire_smoke"] = False
    defaults = {"run." + field.name: _json_copy(field.default)
                for field in fields(RunConfig) if field.default is not MISSING}
    defaults["start_time"] = document.get("start_time")
    _, differences = compare_prepared_domain_config(prior_domain, cold, not_in_use=defaults)
    if differences:
        raise PreparedCacheMismatchError("bulk augmentation cannot change meteorology or geometry: "
                                         + ", ".join(differences))
    reader.verify_all()
    identity = _json_copy(original["identity"])
    identity["domain_config"] = document
    identity["domain_config"]["chem_table_identity"] = _json_copy(table.identity)
    # Only serialize what the common ChemState allocation class persists.
    zeros = {row.state_attr: ((cfg.nz, cfg.ny, cfg.nx), "float32")}
    for name in (*LEDGER_TOTALS, *LEDGER_BUCKETS):
        zeros[ledger_attr(name)] = ((1,), "float64")
    zeros[ledger_attr("started")] = ((1,), "int32")
    for _, module in chem_processes(table):
        for alloc in module.ALLOCATES:
            if alloc.restart == "serialize":
                zeros[process_attr(alloc)] = (allocation_shape(alloc, 1, cfg.nz, cfg.ny, cfg.nx), alloc.dtype)
    metadata = _json_copy(original["metadata"])
    metadata["state_names"] += sorted(zeros)
    transform = {"schema": "sfire-bulk-zero-cache-transform-v1",
        "original_header_sha256": hashlib.sha256(raw).hexdigest(),
        "original_content_sha256": reader.content_sha256,
        "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_bulk_smoke": "explicit zero background and zero external inflow",
        "source": "endogenous completed-step SFIRE fuel consumption",
        "met_source_identities_preserved": True, "pm25_fraction_defined": False,
        "added_zero_fields": sorted(zeros)}
    metadata["user"]["sfire_smoke_initialization"] = transform
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _prepared_cache_staging_path(target)
    temporary.mkdir()
    try:
        writer = _BundleWriter(temporary)
        writer.expect_reuse(reader, reader.arrays)
        for key in reader.arrays:
            writer.link_verified(key, reader)
        for name in sorted(zeros):
            shape, dtype = zeros[name]
            writer.add("state/" + name, np.zeros(shape, dtype=dtype))
        basis = {"schema": PREPARED_CACHE_SCHEMA, "identity": identity, "metadata": metadata,
                 "arrays": writer.manifest, "payload_bytes": writer.payload_bytes}
        header = {**basis, "status": "READY", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "content_sha256": hashlib.sha256(_canonical(basis).encode()).hexdigest()}
        from gpuwm import __version__
        from gpuwm.ingest.prepared_cache import CACHE_WRITER_KEY
        header[CACHE_WRITER_KEY] = {"gpuwm_version": __version__}
        (temporary / "header.json").write_text(json.dumps(header, indent=2, sort_keys=True, allow_nan=False) + "\n")
        verified = PreparedCacheReader(temporary, expected_identity=identity)
        verified.verify_all()
        for key, spec in reader.arrays.items():
            if {k: v for k, v in spec.items() if k != "file"} != {
                    k: v for k, v in verified.arrays[key].items() if k != "file"}:
                raise ValueError("bulk augmentation changed a predecessor array")
        if hashlib.sha256((source / "header.json").read_bytes()).hexdigest() != transform["original_header_sha256"]:
            raise ValueError("the predecessor header changed during augmentation")
        os.replace(temporary, target)
    except BaseException:
        # This directory was created above with a unique owned basename.
        if temporary.resolve().parent != target.parent:
            raise ValueError("cache staging cleanup escaped the named target directory")
        shutil.rmtree(temporary)
        raise
    return {**transform, "status": "PASS", "path": str(target),
        "derived_header_sha256": hashlib.sha256((target / "header.json").read_bytes()).hexdigest(),
        "content_sha256": header["content_sha256"], "array_count": len(writer.manifest),
        "payload_bytes": writer.payload_bytes, "meteorological_array_sha256": {
            key: spec["sha256"] for key, spec in reader.arrays.items()}}
