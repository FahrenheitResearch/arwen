"""Acquire table-declared runtime surface GRIB records through the Rust fetcher."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import subprocess
import csv
from io import StringIO

EXTRACT_ABI = b"gpuwm-grib2-extract-index-v1"


def extract_runtime_record(path, output, numeric_selector):
    """Select using Rust inventory metadata and copy its original Rust envelope."""
    from gpuwm import bridges
    inventory = bridges.find_bridge("grib2_inventory")
    if inventory is None or EXTRACT_ABI not in inventory.read_bytes():
        raise RuntimeError("the GRIB2 inventory reader predates exact record "
                           "extraction; a full-file fetch would append unrelated "
                           "soil records. Rebuild the current reader.\n" +
                           bridges.artifact_remedy(
                               env_var=bridges.BRIDGE_ENV["grib2_inventory"],
                               filename=bridges.executable_name("grib2_inventory"),
                               subject="the GRIB2 inventory reader"))
    result = subprocess.run([os.fspath(inventory), os.fspath(path)],
                            capture_output=True, text=True, check=True)
    rows = csv.DictReader(StringIO("\n".join(line for line in result.stdout.splitlines()
                                             if not line.startswith("#"))), delimiter="\t")
    requested = dict(numeric_selector)
    candidates = []
    for row in rows:
        if all(key in row and float(row[key]) == float(value)
               for key, value in requested.items()):
            candidates.append(int(row["index"]))
    if len(candidates) != 1:
        raise ValueError(f"runtime surface selector {requested} found "
                         f"{len(candidates)} GRIB records; selecting any one "
                         "would leave its quantity ambiguous")
    subprocess.run([os.fspath(inventory), os.fspath(path),
                    f"--extract-index={candidates[0]}", f"--output={output}"],
                   capture_output=True, text=True, check=True)


def require_runtime_surface_fields(met, adapter):
    """Refuse a cache that dropped a source-declared initial surface field."""
    missing = [row[0] for row in adapter.runtime_surface_fields if row[0] not in met.fields]
    if missing:
        raise ValueError(f"the source declares analyzed runtime fields {missing}, "
                         "but this prepared state does not carry them; using "
                         "climatology would initialize a different vegetated area. "
                         "Fetch and prepare the source again")


def append_runtime_surface_records(path, *, adapter, cycle, lead, host,
                                   binary=None, cache_dir=None, progress=print,
                                   streams=None):
    """Append complete selected GRIB messages, without decoding them in Python."""
    rows = adapter.runtime_surface_fields
    if not rows:
        return []
    from gpuwm import rustwx_fetch
    if binary is None:
        from gpuwm.fetch import select_fetch_engine
        binary = select_fetch_engine("rust", progress=progress).binary
    from gpuwm.fetch import RW_FETCH_SOURCES, count_grib2_messages
    evidence = []
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix=".runtime-surface-", dir=path.parent) as scratch:
        root = Path(scratch)
        combined = root / "combined.grib2"
        shutil.copyfile(path, combined)
        for name, product, selector, units, numeric_selector in rows:
            folder = root / name
            folder.mkdir()
            patterns = folder / "selectors.txt"
            rustwx_fetch.write_pattern_file(patterns, (selector,))
            record = rustwx_fetch.run_fetch(
                binary, model=adapter.upstream_model_id,
                date=f"{cycle:%Y%m%d}", cycle=cycle.hour, hours=(lead,),
                product=product, source=RW_FETCH_SOURCES[host], mode="auto",
                out=folder, pattern_file=patterns, cache_dir=cache_dir,
                keep_idx=True, streams=streams)
            if len(record["files"]) != 1:
                raise ValueError(f"runtime surface {name} fetched another file inventory")
            entry = record["files"][0]
            original_file = folder / entry["name"]
            # An envelope may carry several fields even when its envelope
            # count is one. The numeric Rust selector checks both transports.
            field = folder / "selected.grib2"
            extract_runtime_record(original_file, field, numeric_selector)
            if count_grib2_messages(field) != 1:
                raise ValueError(f"runtime surface {name} did not select one GRIB record")
            with field.open("rb") as source, combined.open("ab") as target:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
                source.seek(0)
                shutil.copyfileobj(source, target)
            evidence.append({"field": name, "units": units, "selector": selector,
                             "numeric_selector": dict(numeric_selector),
                             "source_file": entry["name"], "url": entry["grib_url"],
                             "sha256": digest, "bytes": field.stat().st_size,
                             "fetch_mode": entry["mode"],
                             "source_file_sha256": entry["sha256"]})
        os.replace(combined, path)
    return evidence
