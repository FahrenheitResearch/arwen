"""Prepared forcing and diagnostic-history allocation metadata, without CUDA.

Payloads remain Rust-owned. These hooks read JSON declarations only; the
normal loader still verifies numerical payload bytes before uploading them.
Unknown source axes or kernel resources stay explicit admission gaps.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from .memory import ArrayMemory, StageMemory


def _evidence(context, **pins):
    from dataclasses import asdict
    return json.dumps(dict(schema="arwen.hydro.data-memory/v1",
                           context=asdict(context), **pins), sort_keys=True)


def fabric_memory(*, settings, options, context, components):
    """Price load_cuda's complete independent upload, including unused files.

    The driver checkpoints prepare-only resource arrays too. Borrowers may
    alias these exact names; equal hashes in other factory calls never alias.
    Only manifest and roster JSON are read, never a numerical payload.
    """
    import sys
    from gpuwm import hydrofabric

    if sys.byteorder != "little":
        raise ValueError("hydrofabric memory requires the loader's little-endian host")
    path = Path(options["manifest"]).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != hydrofabric.SCHEMA:
        raise ValueError("unsupported hydrofabric memory manifest schema")
    def digest(value):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("hydrofabric memory needs prepared SHA256 declarations")
        return value

    roster_hash = digest(manifest.get("auxiliary", {}).get("network.json", {}).get("sha256"))
    roster_raw = (path.parent / "network.json").read_bytes()
    if hashlib.sha256(roster_raw).hexdigest() != roster_hash:
        raise ValueError("hydrofabric network.json metadata SHA256 mismatch")
    roster = json.loads(roster_raw)
    axes = {}
    for name, count in (("node_ids", "node_count"), ("catchment_ids", "catchment_count")):
        ids = roster[name]
        if (not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids) or len(ids) != manifest[count]):
            raise ValueError(f"hydrofabric memory needs the unique ordered {name} axis")
        axes[name] = ids
    # These are graph order boundaries, not numerical state. Preserve their
    # exact order rather than inventing a routing/output axis from counts.
    for name in ("level_offsets", "reverse_level_offsets"):
        offsets = roster[name]
        if (not isinstance(offsets, list) or not offsets or offsets[0] != 0
                or offsets[-1] != len(axes["node_ids"])
                or any(type(n) is not int or n < 0 for n in offsets)
                or any(a > b for a, b in zip(offsets, offsets[1:]))):
            raise ValueError(f"hydrofabric memory needs valid {name} metadata")
        axes[name] = offsets
    dtypes = {"<f4": "float32", "<f8": "float64", "<i4": "int32", "<i8": "int64",
              "<u4": "uint32", "<u8": "uint64", "|u1": "uint8", "|b1": "bool"}
    arrays, payloads = [], {}
    for name, field in manifest["files"].items():
        array = ArrayMemory(name, tuple(field["shape"]), dtypes.get(field["dtype"], field["dtype"]),
                            checkpointed=True)
        if type(field["bytes"]) is not int or field["bytes"] != array.nbytes:
            raise ValueError(f"hydrofabric shape/dtype byte length mismatch: {name}")
        if not (path.parent / field["path"]).resolve().is_relative_to(path.parent):
            raise ValueError(f"hydrofabric payload escapes prepared directory: {name}")
        payloads[name] = digest(field["sha256"])
        arrays.append(array)
    root = Path(__file__).parents[1]
    sources = (Path(__file__), root / "hydrofabric.py", root / "hydro/driver.py")
    return StageMemory(tuple(arrays), _evidence(context, factory="gpuwm.hydrofabric:prepare_resource",
        manifest=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()),
        network_sha256=roster_hash, payload_sha256=payloads, input=manifest["input"],
        request_sha256=manifest["request_sha256"], node_order=manifest["node_order"],
        upstream_order=manifest["upstream_order"], prepared_numerics=manifest["numerics"],
        grid=manifest.get("grid"), parent_grid_contract=manifest.get("parent_grid_contract"),
        sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        resource_scope="pool uploads only; host pinned transport excluded; no owned CUDA kernels; "
                       "context and assembly resources remain framework-owned", **axes),
        non_pool_bytes=0)


def series_memory(*, settings, options, context, components):
    """Price the complete uploaded packet and its four checkpoint carriers."""
    if context.mode != "standalone":
        raise ValueError("external forcing series requires standalone hydro mode")
    path = Path(options["manifest"])
    raw = path.read_bytes()
    packet = json.loads(raw)
    if packet.get("schema") != "arwen.nextgen-forcing.v1":
        raise ValueError("unsupported Rust forcing packet schema")
    shape = tuple(packet["shape"])
    if len(shape) != 3 or shape[1] != 8 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid Rust forcing packet shape")
    ids = packet["catchment_ids"]
    if len(ids) != shape[2] or len(set(ids)) != len(ids):
        raise ValueError("forcing memory needs the packet's unique ordered catchment axis")
    native = options.get("native_precision", False)
    filename, dtype, wire = (("forcing.f64", "float64", "<f8") if native else
                             ("forcing.f32", "float32", "<f4"))
    if packet["files"][filename] != wire:
        raise ValueError("Rust forcing packet dtype differs from selected precision")
    digest = packet["data_sha256"][filename]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("forcing memory needs a prepared payload SHA256")
    interval = float(packet["interval_seconds"])
    start = float(options.get("start_seconds", 0.0))
    if not math.isfinite(interval) or interval <= 0 or not 0 <= start < shape[0] * interval:
        raise ValueError("forcing requires positive cadence and an in-range start")
    if options.get("time_origin") and options["time_origin"] != packet["time_origin"]:
        raise ValueError("forcing packet epoch differs from the realization")
    table_pin = None
    if options.get("source"):
        from .forcing import source_profile
        from gpuwm.source_adapters import get_source_adapter
        table, row = source_profile(options["source"])
        table_pin = hashlib.sha256(table.read_bytes()).hexdigest()
        if (packet.get("source_id") != get_source_adapter(options["source"]).source_id
                or packet["profile_row"] != row
                or packet["source_profile_sha256"] != table_pin):
            raise ValueError("forcing packet profile differs from the selected init-source table")
    gaps = []
    # Scheduled extra advances can change coverage; never shrink the packet
    # or capacity to make admission appear to fit.
    if context.duration_seconds is not None and start + context.duration_seconds > shape[0] * interval:
        gaps.append("prepared forcing packet does not cover the requested duration")
    if settings.get("after_schedule"):
        gaps.append("forcing clock coverage under after_schedule needs its resolved advance roster")
    n = shape[2]
    arrays = (ArrayMemory("frames", shape, dtype, checkpointed=True),
              ArrayMemory("fields", (8, n), dtype, checkpointed=True),
              ArrayMemory("clock_seconds", (n,), "float64", checkpointed=True),
              ArrayMemory("status", (n,), "int32", checkpointed=True, status=True))
    return StageMemory(arrays, _evidence(context, manifest_sha256=hashlib.sha256(raw).hexdigest(),
        payload_sha256=digest, filename=filename, catchment_ids=ids, source_table_sha256=table_pin,
        source_profile_sha256=packet["source_profile_sha256"],
        area_weights_sha256=packet["area_weights_sha256"], time_origin=packet["time_origin"],
        interval_seconds=interval, start_seconds=start), gaps=tuple(gaps))


def _source_array(settings, components, component, field):
    """Find declared arrays or the known forcing slab's field view.

    A field exposed only by another owner's runtime ``fields`` dictionary
    needs that owner's metadata declaration; its dtype is never guessed.
    """
    source = components.get(component)
    if source is None:
        return None, source
    direct = next((a for a in source.arrays if a.name == field), None)
    if direct is not None:
        return direct, source
    row = next((r for r in settings.get("stages", ()) if r["name"] == component), {})
    if row.get("factory") == "gpuwm.hydro.forcing_inputs:prepare_series":
        from .forcing import FIELDS
        slab = next((a for a in source.arrays if a.name == "fields"), None)
        if field in FIELDS and slab is not None and len(slab.shape) == 2 and slab.shape[0] == 8:
            return ArrayMemory(field, slab.shape[1:], slab.dtype), source
    return None, source


def module_history_memory(*, settings, options, context, components):
    """Price typed capture slabs plus the three persistent pointer tables."""
    capacity, ids = options["capacity_steps"], options["catchment_ids"]
    if type(capacity) is not int or capacity < 1 or not ids or len(set(ids)) != len(ids):
        raise ValueError("module history needs capacity and unique ordered catchment labels")
    fields = options["fields"]
    if not fields:
        raise ValueError("module history needs at least one field")
    arrays = [ArrayMemory("step", (), "uint64", checkpointed=True),
              ArrayMemory("overflow", (), "uint64", checkpointed=True)]
    gaps, sources, names = [], {}, {"step", "overflow"}
    for field in fields:
        name = field["name"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in names:
            raise ValueError("module history field names must be unique NetCDF identifiers")
        names.add(name)
        array, source = _source_array(settings, components, field["component"], field["field"])
        if array is None:
            gaps.append(f"{name}: source {field['component']}/{field['field']} lacks prepared shape/dtype")
            continue
        if array.shape != (len(ids),) or array.dtype not in ("float32", "float64", "int32", "int64"):
            raise ValueError(f"{name} needs a native-typed catchment vector")
        arrays.append(ArrayMemory(name, (capacity, len(ids)), array.dtype, checkpointed=True))
        # Source evidence is opaque for arbitrary owners. Preserve its bytes
        # and require an ordered-axis declaration before claiming coverage.
        sources[field["component"]] = source.evidence
        try:
            axis = json.loads(source.evidence).get("catchment_ids")
        except (ValueError, AttributeError):
            axis = None
        if axis is None:
            gaps.append(f"{name}: source ordered catchment axis is not declared")
        elif axis != list(ids):
            raise ValueError(f"{name}: source catchment axis differs from history labels")
    # Immutable device addresses and extents are rebuilt at prepare, never
    # included in portable checkpoints. Retain them even with source gaps.
    arrays.extend(ArrayMemory(f"workspace/{name}", (len(fields),), "uint64")
                  for name in ("sources", "targets", "sizes"))
    if context.duration_seconds is not None and math.ceil(context.duration_seconds / context.interval_seconds) > capacity:
        gaps.append("module history capacity does not cover requested duration at effective cadence")
    if settings.get("after_schedule"):
        gaps.append("module history capacity under after_schedule needs its resolved capture roster")
    return StageMemory(tuple(arrays), _evidence(context, catchment_ids=list(ids), capacity_steps=capacity,
        origin=options["origin"], fields=fields, sources=sources), gaps=tuple(gaps))
