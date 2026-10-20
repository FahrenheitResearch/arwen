"""Routing factory allocation metadata; never read numerical payload bytes.

The manifest describes allocations, not a verified hydraulic realization.
The runtime Rust loader still validates payloads. Unknown provider inventories
and compiled resources remain explicit gaps in the construction envelope.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from .memory import ArrayMemory, StageMemory
from .terrain_memory import _Bindings

_DTYPES = {"<f4": "float32", "<f8": "float64", "<i4": "int32", "<i8": "int64",
           "<u4": "uint32", "<u8": "uint64", "|u1": "uint8", "|b1": "bool"}


class _Network:
    def __init__(self, settings, options, context, components):
        if ("network_component" in options) == ("fabric_manifest" in options):
            raise ValueError("routing needs exactly one network_component or fabric_manifest")
        self.bindings = _Bindings(components, context)
        self.arrays, self.manifest, self.seal = [], None, None
        self.component = options.get("network_component")
        path = options.get("fabric_manifest")
        if self.component is not None:
            row = next((row for row in settings.get("stages", ())
                        if row["name"] == self.component), {})
            # Do not reinterpret an arbitrary custom provider as load_cuda.
            if row.get("factory") == "gpuwm.hydrofabric:prepare_resource":
                path = row.get("options", {}).get("manifest")
        if path is None:
            self.bindings.gaps.append("network manifest metadata unavailable for the selected provider")
            return
        path = Path(path).resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if manifest.get("schema") != "arwen.nextgen.hydrofabric.v1":
            raise ValueError("unsupported routing memory hydrofabric manifest schema")
        if not manifest.get("auxiliary", {}).get("network.json", {}).get("sha256"):
            raise ValueError("routing manifest needs its Rust network roster hash")
        self.manifest = manifest
        self.seal = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        self.files = {}
        for key, field in manifest["files"].items():
            dtype = _DTYPES.get(field["dtype"], field["dtype"])
            array = ArrayMemory("fabric/" + key, tuple(field["shape"]), dtype)
            if type(field["bytes"]) is not int or field["bytes"] != array.nbytes:
                raise ValueError(f"fabric shape/dtype byte length mismatch: {key}")
            if not (path.parent / field["path"]).resolve().is_relative_to(path.parent):
                raise ValueError(f"fabric payload escapes prepared directory: {key}")
            if not field.get("sha256"):
                raise ValueError(f"fabric payload lacks its Rust validation hash: {key}")
            self.files[key] = array
            if self.component is None:
                # load_cuda uploads ALL files. Even unused uploads can remain
                # in the pool: conservatively reserve each as resident here.
                self.arrays.append(array)
            else:
                value = self.bindings.get(self.component, key, "inputs/fabric/" + key)
                if value is not None:
                    _spec(value, array.shape, array.dtype)
                    self.arrays.append(value)

    def get(self, key, name):
        declared = self.files[key]
        if self.component is None:
            value = replace(declared, name=name, alias_of=declared.name,
                            checkpointed=True)
            # The stage name is assigned by the plan. Same-stage aliases need
            # its qualified name, found from this factory's exact options.
            return value
        value = self.bindings.get(self.component, key, name, checkpointed=True)
        if value is not None:
            _spec(value, declared.shape, declared.dtype)
        return value


def _spec(array, shape, dtype):
    if array is not None and (array.shape != tuple(shape) or array.dtype != dtype):
        raise ValueError(f"{array.name} needs {dtype}{tuple(shape)}")


def _finish(network, arrays, settings, options, context, factory, **details):
    if network.component is None and network.manifest is not None:
        rows = [r for r in settings.get("stages", ()) if r.get("factory") == factory
                and r.get("options", {}) == options]
        if len(rows) != 1:
            # Local uploads still priced without inventing an alias namespace.
            network.bindings.gaps.append("local fabric checkpoint aliases require a unique stage row")
            arrays = [a for a in arrays if not (a.alias_of or "").startswith("fabric/")]
        else:
            arrays = [replace(a, alias_of=rows[0]["name"] + "/" + a.alias_of)
                      if (a.alias_of or "").startswith("fabric/") else a for a in arrays]
    root = Path(__file__).parent
    sources = [Path(__file__), root / "troute_mc.py", root / "reservoirs.py",
               root / "kernels/troute_mc.cu", root.parent / "core/kernels/nextgen_reservoirs.cu",
               root.parent / "core/kernels/nextgen_reservoirs.cuh"]
    evidence = json.dumps(dict(schema="arwen.hydro.routing-memory/v1", context=asdict(context),
        factory=factory, options=options, manifest=network.seal,
        providers=network.bindings.sources,
        sources={str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sources}, **details), sort_keys=True)
    return StageMemory(tuple(arrays), evidence, non_pool_bytes=None,
                       gaps=tuple(dict.fromkeys(network.bindings.gaps)))


def reservoir_memory(*, settings, options, context, components):
    from .reservoirs import PARAMETERS, CONTROLS, SOURCE_PIN
    network = _Network(settings, options, context, components)
    arrays = network.arrays
    if network.manifest is None:
        return _finish(network, arrays, settings, options, context, "gpuwm.hydro.reservoirs:prepare")
    declaration = network.manifest["reservoirs"]
    if (tuple(declaration["parameter_columns"]) != PARAMETERS
            or tuple(declaration["control_columns"]) != CONTROLS
            or declaration.get("source_pin") != SOURCE_PIN or not declaration.get("validated_csr")):
        raise ValueError("reservoir column/source/CSR declaration differs from the runtime contract")
    mapped = declaration["arrays"]
    if len(network.files[mapped["height"]].shape) != 1:
        raise ValueError("reservoir height needs a vector")
    n = network.files[mapped["height"]].shape[0]
    specs = {"parameters": ((9, n), "float32"), "offsets": ((n + 1,), "int64"),
             "controls": ((8, n), "float64"), "carriers": ((3, n), "float64")}
    specs.update({k: ((n,), "float32") for k in ("previous_inflow", "inflow", "lateral", "height",
                  "initial_height", "levelpool_release", "levelpool_height")})
    specs.update({k: ((n,), "int32") for k in ("types", "indices", "dynamic_type", "active_source", "fault")})
    specs.update({k: ((n,), "float64") for k in ("clock", "release", "assimilated")})
    series = network.files[mapped["series"]]
    if len(series.shape) != 1:
        raise ValueError("reservoir series needs a vector")
    specs.update({k: (series.shape, "float64") for k in ("series", "times")})
    for name, (shape, dtype) in specs.items():
        _spec(network.files[mapped[name]], shape, dtype)
    for name, key in mapped.items():
        array = network.get(key, name)
        if array is not None:
            arrays.append(replace(array, status=name == "fault" and n > 0))
    return _finish(network, arrays, settings, options, context, "gpuwm.hydro.reservoirs:prepare",
                   reservoirs=n, series_values=series.shape[0])


def mc_memory(*, settings, options, context, components):
    network = _Network(settings, options, context, components)
    arrays, bindings = network.arrays, network.bindings
    dt = options.get("dt_seconds", context.interval_seconds)
    if type(dt) not in (int, float) or not 0 < dt or (dt <= context.interval_seconds and context.interval_seconds % dt) or (dt > context.interval_seconds and dt % context.interval_seconds):
        raise ValueError("MC dt_seconds must divide or span whole parent intervals")
    if options.get("time_origin", settings.get("time_origin")) is None:
        raise ValueError("MC needs the realization time_origin for restart identity")
    if "lateral_component" not in options:
        raise ValueError("MC needs lateral_component")
    nudging = options.get("streamflow_nudging", "nudge_component" in options)
    if not isinstance(nudging, bool) or nudging != ("nudge_component" in options):
        raise ValueError("MC nudging boolean and component binding disagree")
    if network.manifest is None:
        return _finish(network, arrays, settings, options, context, "gpuwm.hydro.troute_mc:prepare")
    routing, files = network.manifest.get("routing", {}), network.files
    ordered = "mc_ordered_nodes" if "mc_ordered_nodes" in files else "ordered_nodes"
    output_ids = routing.get("output_reach_ids", routing.get("native_output_node_ids", "node_ids"))
    if len(files["node_ids"].shape) != 1 or len(files[ordered].shape) != 1:
        raise ValueError("routing node and ordered-node metadata need vectors")
    n, m = files["node_ids"].shape[0], files[ordered].shape[0]
    if n > 2147483647 // 3:
        raise ValueError("MC state exceeds int32 indexing")
    has_lakes = any(x["role"] == "lakes" and x["row_count"] for x in network.manifest.get("layers", ()))
    if has_lakes and not (routing.get("validated_reservoir_mapping") and "reservoir_component" in options
            and "mc_ordered_nodes" in files and "mc_level_offsets" in routing and routing.get("reservoir_levels")):
        raise ValueError("fabric lakes require collapsed reservoir nodes and level selections")
    for name, key, shape, dtype in (
        ("parameters", "routing_parameters", (n, 8), "float32"),
        ("upstream_offsets", "upstream_offsets", (n + 1,), "int32"),
        ("upstream_indices", "upstream_indices", (files["upstream_indices"].shape[0],), "int32"),
        ("ordered_nodes", ordered, (m,), "int32"), ("reach_id", output_ids, (n,), "int64")):
        _spec(files[key], shape, dtype)
        value = network.get(key, name)
        if value is not None:
            arrays.append(value)
    def borrow(component, key, name, shape, dtype):
        value = bindings.get(component, key, name, checkpointed=True)
        _spec(value, shape, dtype)
        if value is not None:
            arrays.append(value)
    borrow(options["lateral_component"], options.get("lateral_array", "routing_lateral"), "lateral", (n,), "float32")
    if "initial_state_component" in options:
        borrow(options["initial_state_component"], options.get("initial_state_array", "routing_state"), "state", (n, 3), "float32")
    elif "initial_state_array" in routing:
        key = routing["initial_state_array"]
        if not isinstance(key, str) or not key or key not in files:
            raise ValueError("fabric initial_state_array must name a prepared, hashed array")
        _spec(files[key], (n, 3), "float32")
        value = network.get(key, "state")
        if value is not None:
            arrays.append(value)
    else:
        arrays.append(ArrayMemory("state", (n, 3), "float32", checkpointed=True))
    if nudging:
        borrow(options["nudge_component"], options.get("nudge_array", "nudge_m3_s"), "nudge_m3_s", (n,), "float32")
    else:
        arrays.append(ArrayMemory("nudge_m3_s", (n,), "float32", checkpointed=True))
    arrays.extend(ArrayMemory(name, shape, "float32", checkpointed=True) for name, shape in (
        ("boundary_previous", (n,)), ("boundary_current", (n,)),
        ("previous", (n, 3)), ("diagnostics", (n, 3)), ("published", (3, n))))
    if options.get("publish_channels", False):
        arrays.extend((ArrayMemory("channel_ids", (m,), "int64", checkpointed=True),
                       ArrayMemory("channel_published", (3, m), "float32", checkpointed=True)))
    if "reservoir_component" in options:
        owner = options["reservoir_component"]
        source = components.get(owner)
        selections = routing["reservoir_levels"]
        if selections and not routing.get("validated_reservoir_mapping"):
            raise ValueError("reservoir mapping needs Rust validation")
        if selections and source is None:
            bindings.gaps.append("embedded reservoir checkpoint roster lacks provider memory metadata")
        elif selections:
            for item in source.arrays:
                if item.checkpointed:
                    value = bindings.get(owner, item.name, "reservoir/" + item.name, checkpointed=True)
                    if value is not None:
                        arrays.append(value)  # fault is MC checkpoint state, not MC status.
        unique = {}
        def retain(name, key):
            if name in unique and unique[name] != key:
                raise ValueError("conflicting reservoir checkpoint name over different payload backings")
            unique[name] = key
        for selection in selections:
            key = selection["rows"]
            retain("reservoir_selection/" + network.manifest["files"][key]["sha256"], key)
            key = routing["reservoir_node_indices"]
            retain("reservoir_node_map/" + network.manifest["files"][key]["sha256"], key)
        for name, key in unique.items():
            if len(files[key].shape) != 1 or files[key].dtype != "int32":
                raise ValueError("reservoir selection and node map need int32 vectors")
            value = network.get(key, name)
            if value is not None:
                arrays.append(value)
    if options.get("continuous_initialization"):
        replay = routing.get("initial_replay", {}).get("payload")
        if replay is None:
            raise ValueError("continuous routing initialization requires a sealed replay payload")
        shape = tuple(replay["shape"])
        if replay["dtype"] != "<f8" or replay["bytes"] != math.prod(shape) * 8:
            raise ValueError("continuous routing replay extent or dtype differs")
        arrays.append(ArrayMemory("workspace/initial_runoff_replay", shape, "float64", lifetime="prepare"))
        bindings.gaps.append("continuous replay has a pinned host staging buffer of " + str(replay["bytes"]) + " bytes (host, not VRAM)")
    if options.get("publish_channels", False):
        bindings.gaps.append("MC channel-ID gather implementation workspace is unpriced")
    return _finish(network, arrays, settings, options, context, "gpuwm.hydro.troute_mc:prepare",
                   nodes=n, mc_nodes=m, dt_seconds=dt,
                   reach_output={
                       "reach_id": dict(array="reach_id", shape=[n], dtype="int64"),
                       **{field: dict(array="published", shape=[n], dtype="float32", offset_bytes=i*n*4)
                          for i, field in enumerate(("discharge_m3_s", "velocity_m_s", "depth_m"))},
                       "nudge_m3_s": dict(array="nudge_m3_s", shape=[n], dtype="float32")},
                   output_id_payload_sha256=network.manifest["files"][output_ids]["sha256"],
                   channel_order_payload_sha256=network.manifest["files"][ordered]["sha256"])
