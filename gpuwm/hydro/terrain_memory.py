"""Terrain allocation metadata; no numerical payloads, CUDA imports or launches.

Scratch omitted from a component's checkpoint roster still belongs to its
resident envelope. Unknown providers and kernel bounds remain named gaps.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from .memory import ArrayMemory, StageMemory

_INPUTS = ("surface_depth", "groundwater_storage", "elevation", "soil_depth",
           "specific_yield", "manning", "conductivity", "active",
           "surface_source", "recharge", "outlet_stage", "outlet_width")
_OUTPUTS = ("surface_outlet_volume", "groundwater_outlet_volume", "exfiltration_volume",
            "surface_interval_volume", "groundwater_interval_volume")


class _Bindings:
    def __init__(self, components, context):
        self.components, self.context = components, context
        self.gaps, self.sources = [], {}
        # A continental fabric can expose thousands of reservoir selections.
        # Its ordered axes make the evidence large; decode its numerics once
        # per immutable provider, rather than once for every borrowed array.
        self._provider_numerics = {}

    def evidence_references(self):
        """Seal earlier stages without recursively embedding their rosters.

        HydroMemoryPlan.inventory retains every stage's original evidence.
        Copying those strings into every borrower repeatedly JSON-escapes a
        continental roster, taking minutes on one host core at each admission.
        References bind the exact same UTF-8 bytes to their earlier stage;
        local providers without a named inventory entry remain inline.
        """
        result = {}
        for name, evidence in self.sources.items():
            raw = evidence.encode('utf-8')
            result[name] = dict(stage=name, bytes=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest())
            if name not in self.components:
                result[name]['inline'] = evidence
        return result

    def get(self, component, field, name, *, checkpointed=False):
        source = self.components.get(component)
        if source is None:
            self.gaps.append(f"{name}: provider {component} lacks prepared memory metadata")
            return None
        if component not in self._provider_numerics:
            self.sources[component] = source.evidence
            try:
                evidence = json.loads(source.evidence)
                mode = evidence.get("context", {}).get("numerics")
            except (ValueError, AttributeError, TypeError):
                mode = None
            self._provider_numerics[component] = mode
        mode = self._provider_numerics[component]
        if mode is None:
            self.gaps.append(f"{name}: provider {component} has no declared numerics context")
        elif mode != self.context.numerics:
            raise ValueError(f"{name}: provider numerics differs from terrain memory context")
        array = next((a for a in source.arrays if a.name == field), None)
        if array is None:
            self.gaps.append(f"{name}: provider {component}/{field} lacks prepared shape/dtype")
            return None
        if array.lifetime != "resident":
            raise ValueError(f"{name}: retained terrain input requires resident storage")
        # ArrayMemory aliases address roots, never other views. Flatten without
        # losing subview offsets; reject malformed metadata before construction.
        root, offset, seen = f"{component}/{field}", 0, set()
        target = array
        while target.alias_of is not None:
            if root in seen:
                raise ValueError(f"{name}: cyclic provider memory alias")
            seen.add(root)
            offset += target.offset_bytes
            root = target.alias_of
            owner, separator, key = root.partition("/")
            provider = self.components.get(owner)
            target = (next((a for a in provider.arrays if a.name == key), None)
                      if provider is not None and separator else None)
            if target is None:
                raise ValueError(f"{name}: missing provider backing {root}")
        if target.lifetime != "resident" or offset + array.nbytes > (
                target.nbytes if target.backing_bytes is None else target.backing_bytes):
            raise ValueError(f"{name}: retained view escapes its resident backing")
        return ArrayMemory(name, array.shape, array.dtype, alias_of=root,
                           offset_bytes=offset, checkpointed=checkpointed)

    def finish(self, arrays, context, options, *, kernel, **pins):
        root = Path(__file__).parents[1]
        source = root / "nextgen" / ("inundation.py" if kernel == "inundation" else "terrain.py")
        cuda = root / "core/kernels" / f"nextgen_{kernel}.cu"
        evidence = json.dumps(dict(schema="arwen.hydro.terrain-memory/v1",
            context=asdict(context), options=options, providers=self.evidence_references(),
            metadata_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            factory_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            kernel_source_sha256=hashlib.sha256(cuda.read_bytes()).hexdigest(), **pins), sort_keys=True)
        return StageMemory(tuple(arrays), evidence, non_pool_bytes=None,
                           gaps=tuple(dict.fromkeys(self.gaps)))


def _grid(array, name, dtype):
    if array is not None and (len(array.shape) != 2 or min(array.shape) < 1
                             or math.prod(array.shape) > 2**31 - 1 or array.dtype != dtype):
        raise ValueError(f"{name} needs a nonempty int32-indexed grid of {dtype}")


def _same_grid(arrays, shape):
    if shape is not None and any(a.shape != shape for a in arrays):
        raise ValueError("terrain bound arrays must share the prepared grid shape")


def terrain_memory(*, settings, options, context, components):
    if set(options) != {"dx", "dt", "period_seconds", "identity", "fields"}:
        raise ValueError("terrain memory requires the factory's exact options")
    if not isinstance(options["identity"], dict) or not options["identity"]:
        raise ValueError("terrain identity must name grid/parameters/forcing ordering")
    dx, dt, period = (options[k] for k in ("dx", "dt", "period_seconds"))
    if not all(math.isfinite(x) and x > 0 for x in (dx, dt, period)):
        raise ValueError("terrain controls must be finite and positive")
    steps = round(period / dt)
    if steps < 1 or not math.isclose(steps * dt, period, rel_tol=1e-7):
        raise ValueError("terrain period must contain integer substeps")
    fields = options["fields"]
    if not set(_INPUTS).issubset(fields) or set(fields) - set(_INPUTS + _OUTPUTS):
        raise ValueError("terrain fields must bind all required and only known fields")
    bindings, arrays = _Bindings(components, context), []
    for name, binding in fields.items():
        if set(binding) != {"component", "array"}:
            raise ValueError(f"{name} must explicitly name component and array")
        array = bindings.get(binding["component"], binding["array"], name, checkpointed=True)
        _grid(array, name, "int32" if name == "active" else
              "float64" if name in _OUTPUTS else "float32")
        if array is not None:
            arrays.append(array)
    shape = arrays[0].shape if arrays else None
    _same_grid(arrays, shape)
    if shape is None:
        bindings.gaps.append("terrain grid shape is unavailable; flux and unbound outputs are unpriced")
    else:
        arrays.extend(ArrayMemory(name, shape, "float64", checkpointed=True)
                      for name in _OUTPUTS if name not in fields)
        # This live buffer is intentionally absent from Terrain2D.arrays:
        # rebuilt flux is resident across steps, never portable checkpoint state.
        arrays.append(ArrayMemory("workspace/flux", (14, *shape), "float32"))
    arrays.append(ArrayMemory("health", (1,), "uint32", checkpointed=True, status=True))
    mutable = {"surface_depth", "groundwater_storage", *_OUTPUTS}
    borrowed = [a for a in arrays if a.alias_of is not None]
    for a in borrowed:
        if a.name in mutable and any(b.name != a.name and a.alias_of == b.alias_of
                and a.offset_bytes < b.offset_bytes + b.nbytes
                and b.offset_bytes < a.offset_bytes + a.nbytes for b in borrowed):
            raise ValueError(f"mutable terrain {a.name} overlaps another bound field")
    depth = next((a for a in arrays if a.name == "surface_depth"), None)
    if depth is not None:
        arrays.append(replace(depth, name="depth_m", checkpointed=False))
    if period != context.interval_seconds:
        bindings.gaps.append("terrain prepared period differs from parent cadence; resolved advance roster required")
    return bindings.finish(arrays, context, options, kernel="terrain", shape=shape, substeps=steps)


def lateral_memory(*, settings, options, context, components):
    if set(options) != {"terrain_component", "mapping_component", "offsets_array", "indices_array", "identity"}:
        raise ValueError("terrain lateral memory requires the factory's exact options")
    if not isinstance(options["identity"], dict) or not options["identity"]:
        raise ValueError("lateral identity must seal grid, node axis and inverse cell map")
    bindings, arrays = _Bindings(components, context), []
    offsets = bindings.get(options["mapping_component"], options["offsets_array"],
                           "routing_cell_offsets", checkpointed=True)
    indices = bindings.get(options["mapping_component"], options["indices_array"],
                           "routing_cell_indices", checkpointed=True)
    for array, dtype in ((offsets, "int64"), (indices, "int32")):
        if array is not None:
            if len(array.shape) != 1 or array.dtype != dtype:
                raise ValueError("terrain inverse cell map requires int64 offsets and int32 indices vectors")
            arrays.append(array)
    if offsets is not None:
        nodes = offsets.shape[0] - 1
        if not 0 < nodes <= 2**31 - 1:
            raise ValueError("inverse cell map needs nonempty int32-addressable nodes")
        arrays.append(ArrayMemory("terrain_lateral", (nodes,), "float32", checkpointed=True))
    for field in ("surface_interval_volume", "groundwater_interval_volume"):
        array = bindings.get(options["terrain_component"], field, f"inputs/{field}")
        _grid(array, field, "float64")
        if array is not None:
            arrays.append(array)
    _same_grid([a for a in arrays if a.name.startswith("inputs/")],
               next((a.shape for a in arrays if a.name.startswith("inputs/")), None))
    arrays.append(ArrayMemory("health", (1,), "uint32", checkpointed=True, status=True))
    return bindings.finish(arrays, context, options, kernel="terrain")


def inundation_memory(*, settings, options, context, components):
    bindings, arrays = _Bindings(components, context), []
    shape = None
    for field, dtype in (("surface_depth", "float32"), ("active", "int32"), ("health", "uint32")):
        array = bindings.get(options["terrain_component"], field, f"inputs/{field}")
        if array is None:
            continue
        if field == "health":
            if array.shape != (1,) or array.dtype != dtype:
                raise ValueError("terrain source health must be uint32(1)")
        else:
            _grid(array, field, dtype)
            if shape is not None and array.shape != shape:
                raise ValueError("inundation inputs must share the terrain grid")
            shape = array.shape
        arrays.append(array)
    if shape is None:
        bindings.gaps.append("terrain grid shape is unavailable; diagnostic outputs are unpriced")
    else:
        arrays.extend(ArrayMemory(name, shape, "float32", checkpointed=True)
                      for name in ("depth_m", "maximum_depth_m"))
    arrays.append(ArrayMemory("health", (1,), "uint32", checkpointed=True, status=True))
    return bindings.finish(arrays, context, options, kernel="inundation", shape=shape)
