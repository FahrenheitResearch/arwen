"""Resident literature-based terrain routing; Python only binds CUDA launches.

SPDX-License-Identifier: Apache-2.0
See docs/technical/NEXTGEN-TERRAIN.md for equations and the fresh field ABI.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from types import MappingProxyType


@lru_cache(maxsize=None)
def _kernels(device_id: int):
    import cupy as cp
    from gpuwm import numerics
    from gpuwm.certify.kernel_manifest import record_module

    source = (Path(__file__).parents[1] / "core/kernels/nextgen_terrain.cu").read_text()
    options = numerics.effective_options(("--std=c++17", "--fmad=false",
                                          "--prec-div=true", "--prec-sqrt=true"))
    with cp.cuda.Device(device_id):
        module = cp.RawModule(code=source, options=options,
                              name_expressions=("terrain_flux", "terrain_update", "terrain_lateral"))
        record_module("nextgen_terrain", source=source, options=options, module=module)
        return tuple(module.get_function(name) for name in
                     ("terrain_flux", "terrain_update", "terrain_lateral"))


class Terrain2D:
    """Prepare fixed-duration advances on one card, with stable state pointers.

    Arrays are CUDA C-contiguous (ny,nx). Float fields are float32 except the
    float64 volume outputs; active is int32 (0 sealed, 1 active).
    dx, dt and period_seconds are setup/launch controls, not host model fields.
    External source arrays and receiving outlet stages may be updated in place
    on the parent stream. Check .health off the hot path: any nonzero bit is an
    invalid run, never an accepted repaired field.
    """

    def __init__(self, *, dx: float, dt: float, period_seconds: float,
                 surface_depth, groundwater_storage, elevation, soil_depth,
                 specific_yield, manning, conductivity, active,
                 surface_source, recharge, outlet_stage, outlet_width,
                 identity: dict, surface_outlet_volume=None,
                 groundwater_outlet_volume=None, exfiltration_volume=None,
                 surface_interval_volume=None, groundwater_interval_volume=None):
        import cupy as cp
        import numpy as np  # scalar CUDA launch controls only; no host field arrays
        from gpuwm import numerics

        if not all(math.isfinite(v) and v > 0 for v in (dx, dt, period_seconds)):
            raise ValueError("dx, dt and period_seconds must be finite and positive")
        count = round(period_seconds / dt)
        if count < 1 or not math.isclose(count * dt, period_seconds, rel_tol=1e-7):
            raise ValueError("period_seconds must contain an integer number of terrain substeps")
        if not isinstance(identity, dict) or not identity:
            raise ValueError("identity must name the grid/parameters/forcing ordering provenance")
        shape = tuple(surface_depth.shape)
        if len(shape) != 2 or min(shape) < 1 or math.prod(shape) > 2**31-1:
            raise ValueError("terrain needs a nonempty (ny,nx) grid with int32 cell indices")
        device_id = surface_depth.device.id
        fields = dict(surface_depth=surface_depth, groundwater_storage=groundwater_storage,
                      elevation=elevation, soil_depth=soil_depth, specific_yield=specific_yield,
                      manning=manning, conductivity=conductivity, active=active,
                      surface_source=surface_source, recharge=recharge,
                      outlet_stage=outlet_stage, outlet_width=outlet_width)

        def check(name, array, dtype):
            if not isinstance(array, cp.ndarray) or array.device.id != device_id:
                raise ValueError(f"{name} must be a same-card CUDA array")
            if tuple(array.shape) != shape or array.dtype.name != dtype or not array.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous {dtype}, shape {shape}")

        for name, array in fields.items():
            check(name, array, "int32" if name == "active" else "float32")
        outputs = dict(surface_outlet_volume=surface_outlet_volume,
                       groundwater_outlet_volume=groundwater_outlet_volume,
                       exfiltration_volume=exfiltration_volume,
                       surface_interval_volume=surface_interval_volume,
                       groundwater_interval_volume=groundwater_interval_volume)
        with cp.cuda.Device(device_id):
            for name, array in outputs.items():
                if array is None:
                    array = cp.zeros(shape, dtype="float64")
                    outputs[name] = array
                check(name, array, "float64")
            # Mutable outputs must not alias any input or each other. Read-only
            # parameter/source arrays may share storage (e.g. shared zero forcing).
            all_arrays = {**fields, **outputs}
            mutable = ("surface_depth", "groundwater_storage", *outputs)
            for name in mutable:
                a = all_arrays[name]
                lo, hi = a.data.ptr, a.data.ptr + a.nbytes
                for other, b in all_arrays.items():
                    if other != name and lo < b.data.ptr+b.nbytes and b.data.ptr < hi:
                        raise ValueError(f"mutable {name} overlaps {other}")
            self._flux = cp.empty((14, *shape), dtype="float32")
            self.health = cp.zeros((1,), dtype="uint32")
            self._flow, self._update, _ = _kernels(device_id)
        ny, nx = shape
        self._grid = ((nx * ny + 127) // 128,)
        self._block = (128,)
        self._period = float(period_seconds)
        self._count = count
        controls = (np.int32(nx), np.int32(ny), np.float32(dx), np.float32(dt))
        self._flow_args = (*controls, surface_depth, groundwater_storage, elevation,
                           soil_depth, specific_yield, manning, conductivity, active,
                           surface_source, recharge, outlet_stage, outlet_width,
                           self._flux, self.health)
        update_args = (*controls, surface_depth, groundwater_storage, soil_depth,
                             specific_yield, active, surface_source, recharge, self._flux,
                             outputs["surface_outlet_volume"],
                             outputs["groundwater_outlet_volume"],
                             outputs["exfiltration_volume"],
                             outputs["surface_interval_volume"],
                             outputs["groundwater_interval_volume"])
        self._update_args = (*update_args, np.int32(0), self.health)
        self._reset_update_args = (*update_args, np.int32(1), self.health)
        self.arrays = {**fields, **outputs, "health": self.health}
        self.status_arrays = MappingProxyType({"health": self.health})
        self.depth_m = surface_depth
        self.outputs = {"depth_m": surface_depth, **outputs}
        self.restart_identity = dict(identity, algorithm="arwen.terrain.diffusive-d4.darcy-d8/v1",
                                     numerics=numerics.mode(), shape=list(shape),
                                     dx_m=float(controls[2]), dt_seconds=float(controls[3]),
                                     period_seconds=self._period, substeps=count,
                                     boundary="sealed-exterior+one-way-stage-drains",
                                     interval_export="direct-f64-reset-first-update/v1")

    def flux(self, *, stream=None):
        """Enqueue one grid of donor-limited transfers; no physical state update."""
        self._flow(self._grid, self._block, self._flow_args, stream=stream)

    def update(self, *, stream=None, reset_interval=False):
        """Enqueue conservative gather and exfiltration; requires preceding flux."""
        self._update(self._grid, self._block,
                     self._reset_update_args if reset_interval else self._update_args,
                     stream=stream)

    def advance(self, dt_seconds: float, *, stream=None):
        """Driver Stage launch: the period is fixed at preparation, plain CUDA only."""
        if dt_seconds != self._period:
            raise ValueError("terrain advance differs from the prepared period_seconds")
        for substep in range(self._count):
            self.flux(stream=stream)
            self.update(stream=stream, reset_interval=substep == 0)

    step = advance


def prepare_memory(*, settings, options, context, components):
    from gpuwm.hydro.terrain_memory import terrain_memory
    return terrain_memory(settings=settings, options=options, context=context, components=components)


def prepare(*, settings, options, state, physics, device_id, mode, components):
    """Bind explicit prior-stage arrays through the resident driver's table.

    Each options.fields entry names {component, array}. Providers own loading,
    parameterization and source updates; this factory never invents fields or
    reads raw files. The prepared terrain state includes the bound carriers.
    """
    import cupy as cp

    allowed = {"dx", "dt", "period_seconds", "identity", "fields"}
    if set(options) != allowed:
        raise ValueError(f"terrain options require exactly {sorted(allowed)}")
    required = {"surface_depth", "groundwater_storage", "elevation", "soil_depth",
                "specific_yield", "manning", "conductivity", "active",
                "surface_source", "recharge", "outlet_stage", "outlet_width"}
    optional = {"surface_outlet_volume", "groundwater_outlet_volume", "exfiltration_volume",
                "surface_interval_volume", "groundwater_interval_volume"}
    bindings = options["fields"]
    if not required.issubset(bindings) or set(bindings) - required - optional:
        raise ValueError("terrain fields must bind all required and only known fields")
    fields, identities = {}, {}
    for name, binding in bindings.items():
        if set(binding) != {"component", "array"}:
            raise ValueError(f"{name} must explicitly name component and array")
        provider = components[binding["component"]]
        fields[name] = provider.arrays[binding["array"]]
        if fields[name].device.id != device_id:
            raise ValueError(f"{name} must belong to the parent hydro card")
        identities[binding["component"]] = provider.restart_identity
    if not isinstance(options["identity"], dict) or not options["identity"]:
        raise ValueError("terrain identity must name grid/parameters/forcing ordering")
    identity = dict(options["identity"], providers=identities, bindings=bindings)
    with cp.cuda.Device(device_id):
        return Terrain2D(dx=options["dx"], dt=options["dt"],
                         period_seconds=options["period_seconds"], identity=identity,
                         **fields)


class TerrainLateral:
    """Ordered channel-cell interval export, separate from other lateral sources.

    This producer preserves all mapped node rows, including reservoirs. Binding
    it to a receiver requires an explicit policy for those rows and composition
    with other runoff. It does not create drains, widths, stages or drainage maps.
    """

    def __init__(self, *, terrain, offsets, indices, identity):
        import cupy as cp
        import numpy as np  # prepared launch controls only

        if not isinstance(identity, dict) or not identity:
            raise ValueError("lateral identity must seal grid, node axis and inverse cell map")
        device_id = terrain.depth_m.device.id
        for name, array, dtype in (("offsets", offsets, "int64"),
                                   ("indices", indices, "int32")):
            if (not isinstance(array, cp.ndarray) or array.device.id != device_id or
                    array.ndim != 1 or array.dtype.name != dtype or
                    not array.flags.c_contiguous):
                raise ValueError(f"{name} must be same-card C-contiguous {dtype} vector")
        nodes = offsets.size - 1
        if not 0 < nodes <= 2**31-1:
            raise ValueError("inverse map needs int32-addressable nonempty node axis")
        with cp.cuda.Device(device_id):
            self.terrain_lateral = cp.zeros((nodes,), dtype="float32")
            self.health = cp.zeros((1,), dtype="uint32")
            _, _, self._reduce = _kernels(device_id)
        self._period = terrain._period
        self._grid, self._block = ((nodes+127)//128,), (128,)
        self._args = (np.int32(nodes), np.int32(terrain.depth_m.size),
                      np.int64(indices.size), np.float64(self._period), offsets, indices,
                      terrain.arrays["surface_interval_volume"],
                      terrain.arrays["groundwater_interval_volume"],
                      self.terrain_lateral, self.health)
        self.arrays = {"terrain_lateral": self.terrain_lateral, "health": self.health,
                       "routing_cell_offsets": offsets, "routing_cell_indices": indices}
        self.status_arrays = MappingProxyType({"health": self.health})
        self.outputs = {"terrain_lateral": self.terrain_lateral}
        self.restart_identity = dict(identity, algorithm="arwen.terrain.channel-cell-mean/v1",
                                     terrain=terrain.restart_identity, units="m3/s",
                                     nodes=nodes, period_seconds=self._period,
                                     composition="isolated-contribution",
                                     reservoir_policy="receiver-unbound")

    def advance(self, dt_seconds, *, stream=None):
        if dt_seconds != self._period:
            raise ValueError("lateral advance differs from prepared terrain period")
        self._reduce(self._grid, self._block, self._args, stream=stream)

    step = advance


def prepare_lateral_memory(*, settings, options, context, components):
    from gpuwm.hydro.terrain_memory import lateral_memory
    return lateral_memory(settings=settings, options=options, context=context, components=components)


def prepare_lateral(*, settings, options, state, physics, device_id, mode, components):
    """Prepare the terrain-owned producer after terrain and before its receiver."""
    allowed = {"terrain_component", "mapping_component", "offsets_array", "indices_array", "identity"}
    if set(options) != allowed:
        raise ValueError(f"terrain lateral options require exactly {sorted(allowed)}")
    terrain = components[options["terrain_component"]]
    mapping = components[options["mapping_component"]]
    if terrain.depth_m.device.id != device_id:
        raise ValueError("terrain must belong to the parent hydro card")
    if not isinstance(options["identity"], dict) or not options["identity"]:
        raise ValueError("lateral identity must seal grid, node axis and inverse cell map")
    identity = dict(options["identity"], mapping=mapping.restart_identity,
                    terrain_component=options["terrain_component"],
                    mapping_component=options["mapping_component"],
                    offsets_array=options["offsets_array"], indices_array=options["indices_array"])
    return TerrainLateral(terrain=terrain, identity=identity,
                          offsets=mapping.arrays[options["offsets_array"]],
                          indices=mapping.arrays[options["indices_array"]])
