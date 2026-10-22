"""Parent exchange and native reach-history allocation metadata, without CUDA.

Borrowed device views retain their owner's backing and offset. Parent physics
carriers and compiled resources need their own contracts; these hooks never
turn an allocation lower bound into a complete admission verdict.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from .memory import ArrayMemory, StageMemory
from .terrain_memory import _Bindings


def _finish(arrays, bindings, context, options, *, sources, **pins):
    root = Path(__file__).parents[1]
    paths = [Path(__file__), *(root / name for name in sources)]
    evidence = json.dumps(dict(schema="arwen.hydro.coupling-memory/v1",
        context=asdict(context), options=options, providers=bindings.sources,
        sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in paths}, **pins), sort_keys=True)
    return StageMemory(tuple(arrays), evidence, non_pool_bytes=None,
                       gaps=tuple(dict.fromkeys(bindings.gaps)))


def exchange_memory(*, settings, options, context, components):
    if context.mode != "coupled":
        raise ValueError("parent runoff exchange requires coupled hydro mode")
    bindings, arrays = _Bindings(components, context), []
    if "network_component" in options:
        owner = options["network_component"]
        mapping = {}
        for name in ("offsets", "indices", "weights"):
            value = bindings.get(owner, "exchange_" + name, name, checkpointed=True)
            if value is not None:
                mapping[name] = value
                arrays.append(value)
    else:
        # The legacy direct-manifest factory uploads every payload, even those
        # not used by the exchange. Price those separate backings as well.
        from .data_memory import fabric_memory
        source = fabric_memory(settings=settings,
            options={"manifest": options["hydrofabric_manifest"]}, context=context, components={})
        bindings.sources["local_fabric"] = source.evidence
        rows = [r for r in settings.get("stages", ()) if
                r.get("factory") == "gpuwm.hydro.noahowp.factory:prepare_exchange"
                and r.get("options", {}) == options]
        if len(rows) != 1:
            raise ValueError("local exchange upload requires a unique stage row")
        stage = rows[0]["name"]
        arrays.extend(replace(a, name="fabric/" + a.name, checkpointed=False) for a in source.arrays)
        declarations = {a.name: a for a in source.arrays}
        mapping = {}
        for name in ("offsets", "indices", "weights"):
            a = declarations["exchange_" + name]
            mapping[name] = replace(a, name=name, alias_of=f"{stage}/fabric/{a.name}")
            arrays.append(mapping[name])
    ids = None
    for evidence in bindings.sources.values():
        try:
            ids = json.loads(evidence).get("catchment_ids")
        except (ValueError, AttributeError):
            continue
        if ids is not None:
            break
    if len(mapping) == 3:
        offsets, indices, weights = (mapping[k] for k in ("offsets", "indices", "weights"))
        if (len(offsets.shape) != 1 or offsets.shape[0] < 2 or offsets.dtype != "int64"
                or len(indices.shape) != 1 or indices.dtype != "int32"
                or weights.shape != indices.shape or weights.dtype != "float32"):
            raise ValueError("exchange needs int64 row offsets and matching int32/f32 overlap vectors")
        n = offsets.shape[0] - 1
        if ids is not None and (not isinstance(ids, list) or len(ids) != n or len(set(ids)) != n):
            raise ValueError("exchange catchment axis differs from its CSR row count")
        arrays.extend(ArrayMemory(name, (n,), "float32", checkpointed=True) for name in
            ("surface_runoff_m_s", "subsurface_runoff_m_s", "recharge_m_s",
             "evapotranspiration_m_s", "runoff_depth_m"))
        arrays.extend((ArrayMemory("status", (n,), "int32", checkpointed=True),
            ArrayMemory("sticky_status", (n,), "int32", checkpointed=True,
                        status=True, status_name="exchange")))
    else:
        bindings.gaps.append("exchange destination count is unavailable; output allocations are unpriced")
    # _parent_fields can allocate either of these two buffers. Their presence
    # is not in MemoryContext, so reserve the worst case without guessing a
    # portable checkpoint roster or claiming ownership of existing fields.
    if len(context.parent_shape) == 3 and min(context.parent_shape) > 0:
        arrays.extend(ArrayMemory("workspace/optional_parent/" + name,
                      context.parent_shape[-2:], "float32") for name in ("sfcheadrt", "soldrain"))
    else:
        bindings.gaps.append("parent grid shape unavailable; optional parent allocations are unpriced")
    bindings.gaps.append("borrowed parent Noah-MP flux/soil backing, layers and checkpoint aliases "
                         "require the parent physics allocation contract")
    return _finish(arrays, bindings, context, options,
        sources=("hydro/noahowp/factory.py", "hydro/noahowp/coupling.py", "hydro/noahowp/coupling.cu"),
        catchment_ids=ids, period_seconds=context.interval_seconds,
        parent_allocation_policy="reserve both optional parent buffers; borrowed carriers unpriced")



def accumulated_exchange_memory(*, settings, options, context, components):
    """The Noah/RUC exchange: the routed exchange plus its per-call accumulator buffers.

    Two previous accumulators (checkpointed model state), two step depths and
    one zero carrier, each one parent cell plane.
    """
    base = exchange_memory(settings=settings, options=options, context=context,
                           components=components)
    if len(context.parent_shape) < 2 or min(context.parent_shape) <= 0:
        return replace(base, gaps=base.gaps + ("parent grid shape unavailable; accumulator buffers are unpriced",))
    plane = tuple(context.parent_shape[-2:])
    extra = (ArrayMemory("previous_sfcrunoff", plane, "float32", checkpointed=True),
             ArrayMemory("previous_udrunoff", plane, "float32", checkpointed=True),
             ArrayMemory("workspace/surface_step", plane, "float32"),
             ArrayMemory("workspace/subsurface_step", plane, "float32"),
             ArrayMemory("workspace/zero", plane, "float32"))
    return replace(base, arrays=base.arrays + extra)

def history_memory(*, settings, options, context, components):
    bindings, arrays = _Bindings(components, context), []
    steps, cadence = options["capacity_steps"], options.get("cadence_steps", 1)
    if type(steps) is not int or type(cadence) is not int or cadence < 1 or steps < cadence:
        raise ValueError("history capacity must contain at least one chosen output interval")
    capacity = steps // cadence
    arrays.extend(ArrayMemory(name, (), "uint64", checkpointed=True) for name in ("step", "overflow"))
    owner = options["source"]
    source = components.get(owner)
    try:
        fields = json.loads(source.evidence)["reach_output"]
    except (AttributeError, ValueError, KeyError, TypeError):
        fields = None
    if not isinstance(fields, dict) or not {"reach_id", "discharge_m3_s"}.issubset(fields):
        bindings.gaps.append("history source has no prepared native reach-output view contract")
    else:
        views = {}
        for field in ("reach_id", "discharge_m3_s", "velocity_m_s", "depth_m", "nudge_m3_s"):
            if field not in fields:
                continue
            spec = fields[field]
            offset = spec.get("offset_bytes", 0)
            if type(offset) is not int or offset < 0:
                raise ValueError("history source view requires a nonnegative integer byte offset")
            backing = bindings.get(owner, spec["array"], "inputs/" + field)
            if backing is None:
                continue
            view = replace(backing, shape=tuple(spec["shape"]), dtype=spec["dtype"],
                           offset_bytes=backing.offset_bytes + offset)
            # Reject a declared view outside the logical source allocation,
            # even if the owner's larger root backing would accommodate it.
            if (view.dtype != backing.dtype or offset + view.nbytes > backing.nbytes
                    or offset % (8 if view.dtype in ("int64", "float64") else 4)):
                raise ValueError("history source view escapes or reinterprets its declared allocation")
            views[field] = view
        if {"reach_id", "discharge_m3_s"}.issubset(views):
            ids, flow = views["reach_id"], views["discharge_m3_s"]
            if (len(ids.shape) != 1 or ids.shape[0] < 1 or ids.dtype != "int64"
                    or flow.shape != ids.shape or flow.dtype not in ("float32", "float64")):
                raise ValueError("history requires ordered int64 IDs and native-real reach vectors")
            arrays.append(replace(ids, name="reach_id", checkpointed=True))
            for name, field in (("flow", "discharge_m3_s"), ("velocity", "velocity_m_s"),
                                ("depth", "depth_m"), ("nudge", "nudge_m3_s")):
                if field not in views:
                    continue
                value = views[field]
                if value.shape != ids.shape or value.dtype != flow.dtype:
                    raise ValueError("history source fields differ in order, shape or precision")
                arrays.append(value)
                arrays.append(ArrayMemory(name, (capacity, *ids.shape), flow.dtype, checkpointed=True))
    if context.duration_seconds is not None and math.ceil(context.duration_seconds / options.get("interval_seconds", context.interval_seconds)) > steps:
        bindings.gaps.append("history capacity does not cover requested duration at effective cadence")
    if settings.get("after_schedule"):
        bindings.gaps.append("history capacity under after_schedule needs its resolved capture roster")
    return _finish(arrays, bindings, context, options, sources=("hydro/output.py",),
                   capacity_frames=capacity)
