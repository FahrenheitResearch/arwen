"""Resident catchment-depth to reach-discharge transfer, with GPU-only numerics.

Preparation binds an explicit catchment area convention and receiver axis.
This does not derive physical parameters, initial water stores or source partitions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import MappingProxyType


def compile_unit(strict):
    """Load this unit's module; also the kernel pre-warm's capture
    (gpuwm.kernel_prewarm hydro jobs), so both name one source and option set."""
    from gpuwm.core.ieee_module import compile_module
    flags = ["--std=c++17", "--ftz=false"]
    if strict:
        flags += ["--fmad=false", "--prec-div=true", "--prec-sqrt=true"]
    return compile_module(Path(__file__).with_suffix(".cu").read_text(), flags,
                          "nextgen.runoff.routing")


class PreparedRunoffRouting:
    """One ordered gather per receiver; no floating-point atomic reductions.

    Depth is m per producer period. Area is explicitly m2 or km2. Each term
    uses ``(depth * (area * scale)) / period`` before ascending-catchment-order
    addition, matching the native CFE publication arithmetic when area is km2.
    The f64 discharge and volume diagnostics precede the receiver's f32 cast.
    Empty receivers get zero; invalid runtime input produces NaN and sticky
    status (1 invalid depth/area, 2 overflow, 4 corrupt CSR). The driver blocks
    publication/checkpointing at its normal health boundary.

    GPU setup builds the reverse CSR once. IDs, original sealed input identity,
    area convention, period and reduction order belong to restart identity.
    Every persistent array is checkpointed. Source and geometry providers must
    separately establish matching physical catchments and disjoint source use.
    """

    def __init__(self, *, depth, area, catchment_to_node, node_count,
                 period_seconds, area_units, identity, routing_period_seconds=None):
        import cupy as cp
        import numpy as np
        from gpuwm.numerics import is_strict

        if type(node_count) is not int or not 0 < node_count < 2147483647:
            raise ValueError("runoff receiver count must fit positive int32")
        if (type(period_seconds) not in (int, float)
                or not math.isfinite(period_seconds) or period_seconds <= 0):
            raise ValueError("runoff producer period must be finite and positive")
        if area_units not in ("m2", "km2"):
            raise ValueError("runoff area units must explicitly be m2 or km2")
        routing_period = period_seconds if routing_period_seconds is None else routing_period_seconds
        if (type(routing_period) not in (int, float) or not math.isfinite(routing_period)
                or routing_period < period_seconds or routing_period % period_seconds):
            raise ValueError("runoff routing window must contain whole producer periods")
        self._window = int(routing_period / period_seconds)
        self._steps = 0
        self.device_id = cp.cuda.runtime.getDevice()
        for name, value, dtypes in (("depth", depth, (cp.float32, cp.float64)),
                                     ("area", area, (cp.float64,)),
                                     ("catchment_to_node", catchment_to_node, (cp.int32,))):
            if (not isinstance(value, cp.ndarray) or value.device.id != self.device_id
                    or value.ndim != 1 or not value.flags.c_contiguous
                    or value.dtype not in dtypes):
                raise ValueError(f"runoff {name} must be a contiguous same-card vector of {dtypes}")
        n = depth.size
        if not 0 < n < 2147483647 or area.shape != depth.shape or catchment_to_node.shape != depth.shape:
            raise ValueError("runoff depth, area and receiver map need the same nonempty catchment axis")
        # Setup-only validation/reverse ordering, entirely on the CUDA card.
        if bool(cp.any((catchment_to_node < 0) | (catchment_to_node >= node_count)).item()):
            raise ValueError("runoff catchment receiver is outside the selected network")
        if bool(cp.any(~cp.isfinite(area) | (area <= 0)).item()):
            raise ValueError("runoff catchment area must be finite and positive")
        keys = cp.stack((cp.arange(n, dtype=cp.int32), catchment_to_node))
        indices = cp.lexsort(keys).astype(cp.int32)
        offsets = cp.empty(node_count + 1, dtype=cp.int64)
        offsets[0] = 0
        cp.cumsum(cp.bincount(catchment_to_node, minlength=node_count), out=offsets[1:])
        outputs = dict(routing_lateral=cp.zeros(node_count, cp.float32),
                       routing_discharge_m3_s=cp.zeros(node_count, cp.float64),
                       routing_volume_m3=cp.zeros(node_count, cp.float64),
                       status=cp.zeros(node_count, cp.int32))
        self.arrays = MappingProxyType(dict(depth=depth, area=area,
            catchment_to_node=catchment_to_node, offsets=offsets, indices=indices, **outputs))
        self.status_arrays = MappingProxyType({"runoff": outputs["status"]})
        self.period_seconds = float(period_seconds)
        self.restart_identity = json.loads(json.dumps(dict(
            algorithm="arwen.catchment-depth-to-reach-discharge/v1",
            numerics="strict" if is_strict() else "fast", depth_units="m per producer period",
            area_units=area_units, period_seconds=self.period_seconds,
            catchments=n, receivers=node_count, source_dtype=str(depth.dtype),
            receiver_dtype="float32", reduction="ascending catchment index; per-term divide",
            inputs=identity), allow_nan=False))
        self.restart_identity["routing_period_seconds"] = float(routing_period)
        self.restart_identity["temporal_reduction"] = "sum producer volumes; divide by routing period"
        self._module = compile_unit(is_strict())
        self.kernel = self._module.get_function("route_depth_f32" if depth.dtype == cp.float32 else "route_depth_f64")
        self._args = (depth, area, offsets, indices, *outputs.values(), np.int32(n),
                      np.int32(node_count), np.float64(1e6 if area_units == "km2" else 1),
                      np.float64(routing_period), np.int32(self._window > 1))
        self._reset_args = self._args + (np.int32(1),)
        self._continue_args = self._args + (np.int32(0),)
        self._grid, self._block = ((node_count + 127) // 128,), (128,)

    def advance(self, dt_seconds, *, stream=None):
        if dt_seconds != self.period_seconds:
            raise ValueError("runoff transfer interval differs from its prepared producer period")
        args = self._reset_args if self._steps % self._window == 0 else self._continue_args
        self.kernel(self._grid, self._block, args, stream=stream)
        self._steps += 1

    launch = advance

    def checkpoint_metadata(self):
        return {"producer_steps": self._steps}

    def validate_checkpoint(self, metadata):
        if (set(metadata) != {"producer_steps"} or type(metadata["producer_steps"]) is not int
                or metadata["producer_steps"] < 0):
            raise ValueError("runoff continuation needs its nonnegative producer step")

    def restore_metadata(self, metadata):
        self._steps = metadata["producer_steps"]


class _CatchmentAreas:
    """Area provider on a prepared network's catchment axis (no host data work)."""

    def __init__(self, network, array):
        self.catchment_ids = tuple(network.catchment_ids)
        self.arrays = MappingProxyType({"areasqkm": network.arrays[array]})
        self.restart_identity = dict(source="prepared network payload", array=array,
                                     units="km2", sha256=network.restart_identity["payload_sha256"][array],
                                     network=network.restart_identity["request_sha256"])


def prepare_areas(*, settings, options, state, physics, device_id, mode, components):
    """Resource factory (phase=prepare): catchment areas from a prepared network.

    The Rust routing preparer seals the fabric's areasqkm as a payload on the
    network's catchment axis (prepare-routing-world: catchment_areasqkm).
    """
    return _CatchmentAreas(components[options["network_component"]],
                           options.get("array", "catchment_areasqkm"))


def prepare(*, settings, options, state, physics, device_id, mode, components):
    """Realization factory; MC consumes this component's routing_lateral array.

    The source's catchment metadata path is table-selected. For CFE it is
    ["preparation", "catchments"]. Area providers expose catchment_ids and
    arrays on that same axis; no implicit replacement by polygon area occurs.
    """
    import cupy as cp

    source = components[options["source_component"]]
    network = components[options["network_component"]]
    areas = components[options["area_component"]]
    source_ids = source.restart_identity
    for key in options.get("source_catchments_path", ["catchment_ids"]):
        source_ids = source_ids[key]
    if (not source_ids or tuple(source_ids) != tuple(network.catchment_ids)
            or tuple(areas.catchment_ids) != tuple(network.catchment_ids)):
        raise ValueError("runoff source, area provider and network catchment ordering differ")
    period = source.period_seconds
    fields = getattr(source, "fields", source.arrays)
    depth = fields[options["depth_field"]]
    if depth.size != len(source_ids):
        raise ValueError("runoff depth extent differs from its declared catchment ordering")
    with cp.cuda.Device(device_id):
        return PreparedRunoffRouting(
            depth=depth, area=areas.arrays[options["area_array"]],
            catchment_to_node=network.arrays["catchment_to_node"], node_count=len(network.node_ids),
            period_seconds=period, area_units=options["area_units"],
            routing_period_seconds=options.get("routing_period_seconds"),
            identity=dict(source=source.restart_identity, network=network.restart_identity,
                          area=areas.restart_identity, catchment_ids=list(source_ids),
                          depth_field=options["depth_field"], area_array=options["area_array"],
                          source_partition=options.get("source_partition")))


def prepare_areas_memory(*, settings, options, context, components):
    """Describe the prepare-only area view without constructing a CUDA resource."""
    from .runoff_memory import areas_memory
    return areas_memory(settings=settings, options=options, context=context, components=components)


def prepare_memory(*, settings, options, context, components):
    """Describe gather carriers and known setup buffers before CUDA construction."""
    from .runoff_memory import routing_memory
    return routing_memory(settings=settings, options=options, context=context, components=components)
