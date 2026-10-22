"""Resident t-route reservoirs. Rust supplies decoded arrays; CUDA advances them.

Apache-2.0 source pin/notice: licenses/NOTICE-t-route-reservoirs.txt.
No native library is imported by this runtime.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

SOURCE_PIN = "12a8eae0cdfed437143c590659fa7077605a5e70"
PARAMETERS = ("area_km2", "weir_m", "maximum_m", "weir_coefficient",
              "weir_length_m", "dam_length_m", "orifice_m",
              "orifice_coefficient", "orifice_area_m2")
CONTROLS = ("lookback_hours", "observation_interval_seconds",
            "persistence_interval_seconds", "rfc_enabled",
            "rfc_interval_seconds", "rfc_persist_until_seconds",
            "rfc_total_counts", "source_filename_id")
CARRIERS = ("persisted_release", "update_time", "persistence_time")
_MODULES = {}


def prepare(*, settings, options, state, physics, device_id, mode, components):
    """Bind a resource row to Rust-prepared reservoir payloads at setup.

    The routing component owns ordered subset launches. Select ``phase=prepare``
    in the realization so assembly does not also advance this resource. A
    manifest's ``reservoirs.arrays`` maps canonical array names to payload keys;
    Rust supplies initial state, observations, controls and validation together.
    """
    import cupy as cp
    if ("network_component" in options) == ("fabric_manifest" in options):
        raise ValueError("reservoirs need exactly one network_component or fabric_manifest")
    if "network_component" in options:
        source = components[options["network_component"]]
    else:
        from gpuwm.hydrofabric import load_cuda
        source = load_cuda(options["fabric_manifest"], device_id=device_id)
    declaration = source.manifest["reservoirs"]
    bindings = declaration["arrays"]
    arrays = {name: source.arrays[key] for name, key in bindings.items()}
    payload_hashes = source.restart_identity["payload_sha256"]
    metadata = dict(declaration)
    metadata.update(
        network_sha256=source.restart_identity["source_sha256"],
        parameters_sha256=payload_hashes[bindings["parameters"]],
        series_sha256=payload_hashes[bindings["series"]],
        # All mapped inputs affect restart compatibility: controls, type codes,
        # offsets and times must not disappear behind a values-only series hash.
        prepared_inputs={name: payload_hashes[key] for name, key in bindings.items()},
        source_identity=source.restart_identity,
    )
    with cp.cuda.Device(device_id):
        return Reservoirs(arrays, metadata)


def prepare_memory(*, settings, options, context, components):
    """Describe every mapped reservoir carrier without uploading or compiling."""
    from .routing_memory import reservoir_memory
    return reservoir_memory(settings=settings, options=options, context=context, components=components)


def _resolve_strict(strict):
    from gpuwm import numerics
    selected = numerics.is_strict()
    if strict is not None and bool(strict) != selected:
        raise ValueError(f"reservoir strict={strict!r} disagrees with process numerics={numerics.mode()!r}")
    return selected


def kernel_source(strict=None):
    strict = _resolve_strict(strict)
    root = Path(__file__).resolve().parents[1] / "core" / "kernels"
    prefix = ""
    if strict:
        prefix = (root / "glibc_flt32.cuh").read_text() + "\n#define GPUWM_NEXTGEN_RESERVOIR_STRICT 1\n"
    return prefix + (root / "nextgen_reservoirs.cuh").read_text() + "\n" + (root / "nextgen_reservoirs.cu").read_text()


def compilation_receipt(module, *, strict=None):
    """Record loaded arithmetic controls, including global compiler overrides.

    Composed routing kernels can use this with their loaded module, too.
    ``options`` are the effective image-cache/compile request, not an inference
    from the environment or a claim about NVRTC's internal architecture flags.
    """
    from gpuwm import numerics
    selected = _resolve_strict(strict)
    return {"numerics": numerics.receipt(),
            "options": list(numerics.effective_options(module.options)),
            "reservoir_power": "gfk_pow" if selected else "powf"}


def frame_module():
    """The module this process's numerics launch and its compiled source,
    for the local-frame regenerator and gate (preflight
    PRODUCTION_LOADED_KERNEL_UNITS)."""
    import cupy as cp
    return _module(None, cp.cuda.Device().id), kernel_source(None)


def _module(strict, device):
    strict = _resolve_strict(strict)
    key = (device, strict)
    if key not in _MODULES:
        from gpuwm.core.ieee_module import compile_module
        options = ["--std=c++17", "--ftz=false"]
        if strict:
            options += ["--fmad=false", "--prec-div=true", "--prec-sqrt=true"]
        _MODULES[key] = compile_module(kernel_source(strict), tuple(options), "nextgen_reservoirs")
    return _MODULES[key]


class Reservoirs:
    """Bind fixed GPU views once; launch two kernels per reservoir step.

    All views are C-contiguous and same-card. Rust initializes parameters,
    controls, clocks, carried indices and NaN persistence baselines. Filename
    strings live in the immutable Rust table identified by source_filename_id;
    no string processing occurs on device or in the time step.

    `previous_inflow` is owned by the network caller. This stage never rewrites
    it: MC and diffusive routers must preserve their native inflow convention.
    """

    def __init__(self, arrays, metadata, *, strict=None):
        strict = _resolve_strict(strict)
        import cupy as cp
        from numpy import int32, uint64
        n = arrays["height"].size
        specs = {
            "parameters": ("float32", (9, n)),
            "previous_inflow": ("float32", (n,)),
            "inflow": ("float32", (n,)), "lateral": ("float32", (n,)),
            "height": ("float32", (n,)), "initial_height": ("float32", (n,)),
            "levelpool_release": ("float32", (n,)), "levelpool_height": ("float32", (n,)),
            "types": ("int32", (n,)), "offsets": ("int64", (n + 1,)),
            "controls": ("float64", (8, n)), "carriers": ("float64", (3, n)),
            "clock": ("float64", (n,)), "indices": ("int32", (n,)),
            "release": ("float64", (n,)), "assimilated": ("float64", (n,)),
            "dynamic_type": ("int32", (n,)), "active_source": ("int32", (n,)),
            "fault": ("int32", (n,)),
        }
        device = cp.cuda.runtime.getDevice()
        for name, (dtype, shape) in specs.items():
            a = arrays[name]
            if not isinstance(a, cp.ndarray) or a.dtype.name != dtype or a.shape != shape or not a.flags.c_contiguous:
                raise ValueError(f"reservoir view {name} must be contiguous {dtype}{shape}")
            if a.device.id != device:
                raise ValueError(f"reservoir view {name} is on another CUDA device")
        for name in ("series", "times"):
            a = arrays[name]
            if not isinstance(a, cp.ndarray) or a.dtype.name != "float64" or a.ndim != 1 or not a.flags.c_contiguous or a.device.id != device:
                raise ValueError(f"reservoir {name} must be a same-card contiguous float64 vector")
        if arrays["series"].shape != arrays["times"].shape:
            raise ValueError("reservoir times and series must have equal CSR storage lengths")
        if tuple(metadata["parameter_columns"]) != PARAMETERS or tuple(metadata["control_columns"]) != CONTROLS:
            raise ValueError("Rust reservoir parameter/control column order differs from the CUDA contract")
        if metadata.get("source_pin") != SOURCE_PIN:
            raise ValueError("reservoir bundle must identify the audited t-route source pin")
        if not metadata.get("validated_csr"):
            raise ValueError("Rust must validate reservoir CSR bounds, type codes and increasing observation times before upload")
        self.arrays = dict(arrays)
        # RFC bounds failures are sticky checkpoint carriers. Publish their
        # existing vector to the driver's output/checkpoint health boundary.
        self.status_arrays = MappingProxyType({"fault": self.arrays["fault"]} if n else {})
        self.restart_identity = {
            "algorithm": "t-route-reservoirs-v1", "source_pin": SOURCE_PIN,
            "numerics": "strict" if strict else "fast",
            "network_sha256": metadata["network_sha256"],
            "parameters_sha256": metadata["parameters_sha256"],
            "series_sha256": metadata["series_sha256"],
            "time_origin": metadata["time_origin"],
            "reservoir_order_sha256": metadata["reservoir_order_sha256"],
        }
        if "prepared_inputs" in metadata:
            self.restart_identity["prepared_inputs"] = metadata["prepared_inputs"]
            self.restart_identity["source_identity"] = metadata["source_identity"]
        if "initialization" in metadata:
            self.restart_identity["initialization"] = dict(metadata["initialization"])
        module = _module(strict, device)
        self.compile_receipt = compilation_receipt(module, strict=strict)
        self._levelpool = module.get_function("ngen_reservoir_levelpool")
        self._assimilate = module.get_function("ngen_reservoir_assimilate")
        self._n, self._grid, self._block = int32(n), ((n + 127) // 128,), (128,)
        self._selection_args = (self._n, uint64(0), self._n)
        a = self.arrays
        self._lp_args = tuple(a[k] for k in ("parameters", "previous_inflow", "inflow", "lateral"))
        self._lp_output = tuple(a[k] for k in ("height", "initial_height", "levelpool_release", "levelpool_height"))
        self._da_input = tuple(a[k] for k in ("parameters", "inflow", "initial_height", "levelpool_release", "levelpool_height", "types", "offsets", "series", "times", "controls"))
        self._da_output = tuple(a[k] for k in ("clock", "carriers", "indices", "height", "release", "assimilated", "dynamic_type", "active_source", "fault"))

    def launch(self, dt_seconds, *, stream=None):
        from numpy import float32, float64
        if not self._grid[0]:
            return
        self._levelpool(self._grid, self._block,
                        (*self._selection_args, *self._lp_args, float32(dt_seconds), *self._lp_output), stream=stream)
        self._assimilate(self._grid, self._block,
                         (*self._selection_args, *self._da_input, float64(dt_seconds), *self._da_output), stream=stream)

    def subset(self, rows, *, selection_sha256):
        """Prepare a network-level launch once; selected rows must be unique.

        Rust verifies bounds and uniqueness when deriving topological levels.
        The router supplies inflows before this call and scatters release
        afterward, before its downstream level. No selection is built per step.
        """
        import copy
        import cupy as cp
        from numpy import int32
        if not isinstance(rows, cp.ndarray) or rows.dtype.name != "int32" or rows.ndim != 1 or not rows.flags.c_contiguous or rows.device.id != self.arrays["height"].device.id:
            raise ValueError("reservoir subset must be a same-card contiguous int32 row vector")
        view = copy.copy(self)
        view.arrays = dict(self.arrays, selection=rows)
        view.restart_identity = dict(self.restart_identity, selection_sha256=selection_sha256)
        view._selection_args = (int32(rows.size), rows, self._n)
        view._grid = ((rows.size + 127) // 128,)
        return view
