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

#: What a start reads in place of a runtime surface field that its cycle
#: does not publish.  Keyed by the field, never by a source: any adapter
#: whose table declares one of these fields gets the same fallback.  The
#: id and sentence are what the fetch receipt records.  A field with no
#: row here has no fallback and its absence is refused by name.
RUNTIME_SURFACE_FALLBACKS = {
    "VEGFRA": ("static-greenfrac-monthly-climatology",
               "the static GREENFRAC monthly climatology interpolated to "
               "the start date, the green fraction every start read before "
               "analyzed vegetation was retained"),
}


#: The canonical name a mapped composition binds each fallback field
#: under (``fields.vegetation_fraction`` of the HRRR surface vegetation
#: mapping is the VEGFRA a start reads).  Keyed by field, like the
#: fallback table: any composition that binds one of these fields from a
#: contributing source gets the same fallback when that source's files
#: publish no record for it.
CANONICAL_RUNTIME_SURFACE_FIELDS = {"vegetation_fraction": "VEGFRA"}


def composition_unpublished_fallbacks(bindings):
    """``{canonical field: fallback id}`` a composition's decode may take.

    ``bindings`` is a composition's ``field_sources``.  Only fields that a
    contributing source supplies and that RUNTIME_SURFACE_FALLBACKS
    covers are named; the engine records such a binding as unpublished
    only when its files carry no record for any of its fields AND every
    field it binds is named here, and refuses it otherwise.
    """
    found = {}
    for binding in (bindings or {}).values():
        for name in binding.get("fields", ()):
            legacy = CANONICAL_RUNTIME_SURFACE_FIELDS.get(str(name))
            if legacy in RUNTIME_SURFACE_FALLBACKS:
                found[str(name)] = RUNTIME_SURFACE_FALLBACKS[legacy][0]
    return found


def unpublished_binding_fallbacks(entry):
    """The recorded fallbacks of an UNPUBLISHED contributing-source receipt.

    ``{VEGFRA-style field: rows}`` in the shape the native route's
    ``require_runtime_surface_fields`` returns, or ``None`` when ``entry``
    is not an unpublished binding.  A receipt whose fallback is not
    exactly the table's for every bound field raises: the start would
    read a field nobody recorded where it came from.
    """
    alignment = entry.get("alignment") if isinstance(entry, dict) else None
    if not isinstance(alignment, dict) or alignment.get("status") != "UNPUBLISHED":
        return None
    named = alignment.get("fallback")
    fields = entry.get("fields")
    expected = {str(name): RUNTIME_SURFACE_FALLBACKS.get(
                    CANONICAL_RUNTIME_SURFACE_FIELDS.get(str(name)), (None,))[0]
                for name in (fields if isinstance(fields, list) else ())}
    if (not isinstance(named, dict) or not expected or named != expected
            or None in expected.values()):
        raise ValueError(
            f"contributing source {entry.get('binding')!r} is recorded as "
            f"unpublished with fallback {named!r}, which is not the "
            "field-keyed fallback table's for its fields; the start would "
            "read a field without a recorded origin")
    rows = {}
    for name, fallback_id in expected.items():
        legacy = CANONICAL_RUNTIME_SURFACE_FIELDS[name]
        rows[legacy] = [{
            "field": legacy, "fallback_id": fallback_id,
            "fallback": RUNTIME_SURFACE_FALLBACKS[legacy][1],
            "reason": str(alignment.get("reason")),
            "binding": str(entry.get("binding")),
            "source_files": [str(row.get("path")) for row in entry.get("data", ())
                             if isinstance(row, dict)],
        }]
    return rows


class RuntimeSurfaceRecordAbsent(ValueError):
    """The cycle's file carries no record matching a runtime selector."""


def declared_runtime_surface_fields(value, adapter=None):
    """The runtime surface fields a configuration declares required.

    ``value`` is the ``[fetch] runtime_surface`` hint (or the
    ``--runtime-surface`` flag): a comma-separated list of field names.
    A declared field must be one the source's table carries; naming one
    it does not carry would be accepted and read by nothing.
    """
    if value in (None, ""):
        return frozenset()
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",")]
    else:
        names = [str(part).strip() for part in value]
    if any(not name for name in names):
        raise ValueError(f"runtime_surface = {value!r} must be comma-separated "
                         "field names such as 'VEGFRA'")
    declared = frozenset(names)
    if adapter is not None:
        offered = {row[0] for row in adapter.runtime_surface_fields}
        unknown = sorted(declared - offered)
        if unknown:
            raise ValueError(
                f"runtime_surface declares {unknown}, which the "
                f"{adapter.source_id} source table does not carry "
                f"(it carries {sorted(offered) or 'none'}); a declaration "
                "no fetch reads would be silently ignored")
    return declared


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
    if not candidates:
        raise RuntimeSurfaceRecordAbsent(
            f"runtime surface selector {requested} matches no GRIB record "
            f"in {Path(path).name}")
    if len(candidates) != 1:
        raise ValueError(f"runtime surface selector {requested} found "
                         f"{len(candidates)} GRIB records; selecting any one "
                         "would leave its quantity ambiguous")
    subprocess.run([os.fspath(inventory), os.fspath(path),
                    f"--extract-index={candidates[0]}", f"--output={output}"],
                   capture_output=True, text=True, check=True)


def recorded_runtime_surface_fallbacks(source_manifest):
    """The fallback rows a fetch receipt recorded, keyed by field.

    Read from the fetch manifest's per-file ``runtime_surface`` lists.
    Returns ``{}`` when no manifest is given or it recorded none.
    """
    import json
    if source_manifest is None or not Path(source_manifest).is_file():
        return {}
    try:
        document = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError):
        # Not a fetch receipt (a checksum list, say): it records nothing,
        # so a dropped field stays refused.
        return {}
    found = {}
    pending = [document]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            rows = node.get("runtime_surface")
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("fallback_id"):
                        found.setdefault(str(row.get("field")), []).append(row)
            pending.extend(value for value in node.values()
                           if isinstance(value, (dict, list)))
        elif isinstance(node, list):
            pending.extend(node)
    return found


def require_runtime_surface_fields(met, adapter, *, source_manifest=None,
                                   refuse_dropped=True):
    """Refuse a cache that dropped a source-declared initial surface field.

    A field the cycle itself does not publish is not a dropped field: the
    fetch receipt records its fallback (id, reason, cycle) and the start
    reads that fallback.  Returns the recorded fallbacks keyed by field so
    the caller's receipt carries them; ``{}`` when none applied.
    ``refuse_dropped=False`` only reports: a start whose static fields do
    not come from the source's own static table never read the analyzed
    field in place of a climatology it lacks.
    """
    missing = [row[0] for row in adapter.runtime_surface_fields if row[0] not in met.fields]
    recorded = (recorded_runtime_surface_fallbacks(source_manifest)
                if missing else {})
    dropped = [name for name in missing if name not in recorded]
    if dropped and refuse_dropped:
        raise ValueError(f"the source declares analyzed runtime fields {dropped}, "
                         "but this prepared state does not carry them and its "
                         "fetch receipt records no fallback for them; using "
                         "climatology would initialize a different vegetated area. "
                         "Fetch and prepare the source again")
    return {name: recorded[name] for name in missing if name in recorded}


def _index_lists_no_record(error, selector):
    """Whether an ``rw_fetch`` refusal says the index lists no ``selector``.

    ``rw_fetch`` refuses an index subset whose selector matches no index
    line (rw-fetch net.rs ``select``) before any payload moves.  For the
    one selector a runtime surface row asks, that is the cycle saying it
    publishes no such record (HRRR wrfsfc before 2020-12-02 lists no
    VEG), the same answer a full file with no matching message gives.
    """
    from gpuwm import rustwx_fetch
    text = str(error)
    return (isinstance(error, rustwx_fetch.RwFetchError)
            and error.returncode == rustwx_fetch.EXIT_REFUSED
            and "matched no index record" in text and selector in text)


def _probed_entry(binary, *, adapter, cycle, lead, product, source, cache_dir):
    """The object a refused index subset named, from ``rw_fetch probe``."""
    from gpuwm import rustwx_fetch
    report = rustwx_fetch.run_probe(
        binary, model=adapter.upstream_model_id, date=f"{cycle:%Y%m%d}",
        cycle=cycle.hour, hours=(lead,), product=product, source=source,
        mode="auto", cache_dir=cache_dir)
    hours = report.get("hours") or [{}]
    hour = hours[0] if isinstance(hours[0], dict) else {}
    url = hour.get("grib_url")
    probe = hour.get("probe") if isinstance(hour.get("probe"), dict) else {}
    return {"name": str(url).rsplit("/", 1)[-1] if url else None,
            "grib_url": url, "sha256": None, "idx_url": hour.get("idx_url"),
            "idx_record_count": probe.get("idx_record_count")}


def _absent_record(name, selector, adapter, cycle, lead, entry, declared):
    """The receipt row for a cycle that publishes no record for ``name``."""
    reason = f"this cycle publishes no {selector} record"
    where = (f"{adapter.source_id} cycle {cycle:%Y-%m-%dT%H}Z f{int(lead):02d} "
             f"({entry['name']})")
    if name in declared:
        raise ValueError(
            f"runtime surface {name}: {reason} ({where}), and the "
            f"configuration declares runtime_surface {name}, so the start "
            "must read it from this cycle.  Remove it from [fetch] "
            "runtime_surface to start from its fallback, or choose a cycle "
            "that publishes it")
    fallback = RUNTIME_SURFACE_FALLBACKS.get(name)
    if fallback is None:
        raise ValueError(
            f"runtime surface {name}: {reason} ({where}), and "
            "RUNTIME_SURFACE_FALLBACKS declares no fallback for this field, "
            "so the start would have no value for it")
    row = {"field": name, "selector": selector, "fallback_id": fallback[0],
           "fallback": fallback[1], "reason": reason,
           "cycle": f"{cycle:%Y-%m-%dT%H}Z", "lead": int(lead),
           "source_file": entry["name"], "url": entry["grib_url"],
           "source_file_sha256": entry["sha256"]}
    if entry.get("idx_url") is not None:
        # Decided from the object's index, so no payload was downloaded
        # and there is no file digest; the index and its line count are
        # the evidence.
        row.update(evidence="index", idx_url=entry["idx_url"],
                   idx_record_count=entry.get("idx_record_count"))
    return row


def append_runtime_surface_records(path, *, adapter, cycle, lead, host,
                                   binary=None, cache_dir=None, progress=print,
                                   streams=None, declared=frozenset()):
    """Append complete selected GRIB messages, without decoding them in Python.

    ``declared`` names the fields the configuration requires from the cycle
    itself (``[fetch] runtime_surface``).  A table-declared field that the
    cycle does not publish and the configuration does not declare is
    recorded with its fallback instead of appended; a declared one is
    refused by name.
    """
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
            try:
                record = rustwx_fetch.run_fetch(
                    binary, model=adapter.upstream_model_id,
                    date=f"{cycle:%Y%m%d}", cycle=cycle.hour, hours=(lead,),
                    product=product, source=RW_FETCH_SOURCES[host], mode="auto",
                    out=folder, pattern_file=patterns, cache_dir=cache_dir,
                    keep_idx=True, streams=streams)
            except rustwx_fetch.RwFetchError as error:
                # The index lists no such record: the cycle publishes none
                # (HRRR wrfsfc before 2020-12-02 has no VEG).  The index
                # subset refuses before any payload moves, so the absence
                # is recorded from the index instead of from a file.
                if not _index_lists_no_record(error, selector):
                    raise
                entry = _probed_entry(
                    binary, adapter=adapter, cycle=cycle, lead=lead,
                    product=product, source=RW_FETCH_SOURCES[host],
                    cache_dir=cache_dir)
                row = _absent_record(name, selector, adapter, cycle, lead,
                                     entry, declared)
                progress(f"fetch {adapter.source_id} f{int(lead):02d}: "
                         f"{row['reason']}; {name} starts from "
                         f"{row['fallback_id']} (recorded in the fetch receipt)")
                evidence.append(row)
                continue
            if len(record["files"]) != 1:
                raise ValueError(f"runtime surface {name} fetched another file inventory")
            entry = record["files"][0]
            original_file = folder / entry["name"]
            # An envelope may carry several fields even when its envelope
            # count is one. The numeric Rust selector checks both transports.
            field = folder / "selected.grib2"
            try:
                extract_runtime_record(original_file, field, numeric_selector)
            except RuntimeSurfaceRecordAbsent:
                row = _absent_record(name, selector, adapter, cycle, lead,
                                     entry, declared)
                progress(f"fetch {adapter.source_id} f{int(lead):02d}: "
                         f"{row['reason']}; {name} starts from "
                         f"{row['fallback_id']} (recorded in the fetch receipt)")
                evidence.append(row)
                continue
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
