"""Prepared depth diagnostic consuming the fresh terrain owner's SI contract.

SPDX-License-Identifier: Apache-2.0
Original diagnostic; no UCAR source, reference program, fixture or port is used.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from types import MappingProxyType


@lru_cache(maxsize=None)
def _kernel(device_id):
    import cupy as cp
    from gpuwm import numerics
    from gpuwm.certify.kernel_manifest import record_module

    source = (Path(__file__).parents[1] / "core/kernels/nextgen_inundation.cu").read_text()
    options = numerics.effective_options(("--std=c++17",))
    with cp.cuda.Device(device_id):
        module = cp.RawModule(code=source, options=options,
                              name_expressions=("inundation_depth",))
        record_module("nextgen_inundation", source=source, options=options, module=module)
        return module.get_function("inundation_depth")


class InundationDepth:
    """Copy valid active-cell depth and track its sampled maximum on the parent card.

    Bind terrain.arrays['surface_depth'], ['active'], ['health'], and its
    restart_identity. Inputs remain owned/checkpointed by terrain. The three
    output arrays are this diagnostic's cross-step state. Construction observes
    the initial state on the supplied setup stream, so initial water contributes
    to the maximum. Only diagnostic invocations add later samples; intermediate
    terrain substeps are not observed, so peaks between samples can be missed.
    Driver must order setup and later launches on the parent stream. Check
    health off-path: 1 source failure, 2 invalid depth, 4 bad mask.
    Any nonzero health invalidates the output, even if later values look finite.
    """

    def __init__(self, *, surface_depth, active, source_health, source_identity,
                 period_seconds, stream=None):
        import cupy as cp
        import numpy as np  # fixed CUDA launch-control scalar only
        from gpuwm import numerics

        if not isinstance(source_identity, dict) or not source_identity:
            raise ValueError("source_identity must identify the prepared terrain grid and ordering")
        if not math.isfinite(period_seconds) or period_seconds <= 0:
            raise ValueError("diagnostic period_seconds must be finite and positive")
        if not isinstance(surface_depth, cp.ndarray):
            raise ValueError("surface_depth must be a CUDA array")
        shape = tuple(surface_depth.shape)
        if len(shape) != 2 or min(shape) < 1 or math.prod(shape) > 2**31-1:
            raise ValueError("inundation needs a nonempty (ny,nx) int32-indexed grid")
        device_id = surface_depth.device.id
        for name, array, dtype, wanted in (("surface_depth", surface_depth, "float32", shape),
                ("active", active, "int32", shape), ("source_health", source_health, "uint32", (1,))):
            if (not isinstance(array, cp.ndarray) or array.device.id != device_id
                    or array.dtype.name != dtype or tuple(array.shape) != wanted
                    or not array.flags.c_contiguous):
                raise ValueError(f"{name} must be same-card C-contiguous {dtype}, shape {wanted}")
        self._count = np.int32(math.prod(shape))
        self._grid, self._block = ((math.prod(shape)+255)//256,), (256,)
        self._period = float(period_seconds)
        with cp.cuda.Device(device_id), (stream if stream is not None else cp.cuda.get_current_stream()):
            depth = cp.empty(shape, dtype="float32")
            maximum = cp.zeros(shape, dtype="float32")
            health = cp.zeros((1,), dtype="uint32")
            self._observe = _kernel(device_id)
            self._args = (self._count, surface_depth, active, source_health, depth, maximum, health)
            self.arrays = {"depth_m": depth, "maximum_depth_m": maximum, "health": health}
            self.status_arrays = MappingProxyType({"health": health})
            self.restart_identity = {"algorithm": "arwen.terrain.inundation-depth/v1",
                                     "source": source_identity, "shape": list(shape),
                                     "numerics": numerics.mode(),
                                     "units": "m", "order": "c", "inactive": "NaN",
                                     "period_seconds": self._period,
                                     "boundary": "terrain-owned; one-way stage drains cannot represent channel backwater flooding"}
            self._observe(self._grid, self._block, self._args, stream=stream)

    def step(self, dt_seconds, *, stream=None):
        """One prepared CUDA grid launch; no allocations, host fields or sync."""
        if dt_seconds != self._period:
            raise ValueError("inundation cadence differs from its prepared period_seconds")
        self._observe(self._grid, self._block, self._args, stream=stream)

    advance = step  # Prepared assembly component convention; identical launch.


def prepare_memory(*, settings, options, context, components):
    from gpuwm.hydro.terrain_memory import inundation_memory
    return inundation_memory(settings=settings, options=options, context=context, components=components)


def prepare(*, settings, options, state, physics, device_id, mode, components):
    """Bind the driver's explicit prior-component row during preparation."""
    import cupy as cp

    name = options["terrain_component"]
    terrain = components[name]
    arrays, identity = terrain.arrays, terrain.restart_identity
    if arrays["surface_depth"].device.id != device_id:
        raise ValueError("inundation terrain must belong to the parent hydro card")
    with cp.cuda.Device(device_id):
        return InundationDepth(surface_depth=arrays["surface_depth"], active=arrays["active"],
                               source_health=arrays["health"], source_identity=identity,
                               period_seconds=identity["period_seconds"],
                               stream=cp.cuda.get_current_stream())
