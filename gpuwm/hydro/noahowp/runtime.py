"""Prepared resident Noah-OWP numeric state, batched across catchments.

Python handles setup and enqueues one CUDA kernel per update. The generated
kernel carries out Utilities/Forcing/Interception/Energy/Water on the card.
Initialization/table decoding is outside the numeric update boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from types import MappingProxyType

from . import SOURCE_REVISION

_STAGES = {"utilities": 0, "forcing": 1, "interception": 2,
           "energy": 3, "water": 4, "full": 5, "vegeflux": 6, "bareflux": 7}


def schema():
    return json.loads(Path(__file__).with_name("schema.json").read_text())


_RAW_ABI_START = "// Preserve the original raw module ABI for native replay callers."
_RAW_ABI_END = 'extern "C" __global__ void noahowp_step_monitored('


def physics_options(numerics):
    options = ["--std=c++17", "--ftz=false"]
    if numerics == "strict":
        options += ["--fmad=false", "--prec-div=true", "--prec-sqrt=true"]
    return options


def physics_source(numerics):
    """The resident module's source: physics.cu without the raw ``noahowp_step`` kernel.

    The door launches noahowp_step_monitored, noahowp_export_bmi and
    noahowp_copy_forcing. The raw ABI kernel is a second, fully inlined copy
    of the column physics that no door path launches (tools/noahowp_replay
    compiles the whole file for it), and it doubled the cold compile: 54.4 s
    against 32.2 s on one core (NVRTC 13.4, sm_120, RTX 5070 Ti, 2026-10-18).
    """
    from .math_source import preamble
    text = Path(__file__).with_name("physics.cu").read_text()
    start, end = text.find(_RAW_ABI_START), text.find(_RAW_ABI_END)
    if 0 <= start < end:
        text = text[:start] + text[end:]
    return preamble(numerics) + text


_PREBUILDS = {}
_PREBUILD_LOCK = threading.Lock()


def prebuild(numerics, device_id):
    """Start the physics compile on its own thread now, beside the rest of preparation.

    The cold compile of this one module is the longest host stage of a door
    run (the other hydro kernels compile in under a second each). Started
    before the network, forcing and PET stages prepare, it runs on a second
    core while they do; :class:`PreparedNoahOWP` joins it and then reads the
    image it wrote to the direct image cache. On a warm cache it is a lookup.
    """
    key = (str(numerics), int(device_id))
    with _PREBUILD_LOCK:
        if key in _PREBUILDS:
            return _PREBUILDS[key]
        record = {"numerics": key[0], "device": key[1], "seconds": None, "error": None}

        def run():
            started = time.perf_counter()
            try:
                import cupy as cp
                from gpuwm.core.ieee_module import compile_image
                with cp.cuda.Device(key[1]):
                    compile_image(physics_source(key[0]), physics_options(key[0]))
            except Exception as exc:  # the foreground compile_module reports a real failure itself
                record["error"] = repr(exc)
            record["seconds"] = time.perf_counter() - started

        record["thread"] = threading.Thread(target=run, name="noahowp-prebuild", daemon=True)
        _PREBUILDS[key] = record
        record["thread"].start()
        return record


def _join_prebuild(numerics, device_id):
    record = _PREBUILDS.get((str(numerics), int(device_id)))
    if record is not None:
        record["thread"].join()


class PreparedNoahOWP:
    """Stable state slabs and a bound direct launch; no allocation/sync in step.

    Slabs use field/level-major, catchment-minor order for coalesced device
    access. ``arrays`` exposes both the three storage slabs and every typed
    field view; profile views have shape (catchments, levels). Forcing writers
    must respect these views' strides. ``status`` is an i32 diagnostic bitmask:
    1 upstream diagnostic, 2 fatal condition, 4 unsupported dimensions/options.
    ``status_arrays`` exposes the checkpointed sticky OR of these masks, so a
    healthy step cannot erase an earlier error. Inspect only at diagnostic
    boundaries; launch never clears or reads these masks on the host.
    """

    def __init__(self, fields=None, *, slabs=None, numerics="fast", stage="full",
                 advance=True, parameter_sha256, fabric_sha256, time_origin,
                 forcing=None):
        import cupy as cp
        import numpy as np
        from gpuwm.core.ieee_module import compile_module

        if numerics not in {"fast", "strict"} or stage not in _STAGES:
            raise ValueError("unknown numeric mode or physics stage")
        if (fields is None) == (slabs is None):
            raise ValueError("supply either complete typed fields or packed slabs")
        metadata = schema()
        if fields is not None:
            required = {f["name"] for f in metadata["fields"]}
            if set(fields) != required:
                raise ValueError(f"field inventory differs: missing={sorted(required-set(fields))}, extra={sorted(set(fields)-required)}")
            first = fields["domain.dt"]
            n, device = first.shape[0], first.device.id
            storage = {t: cp.empty((count, n), dtype=t.replace("f32", "float32").replace("i32", "int32").replace("f64", "float64"))
                       for t, count in metadata["slab_fields"].items()}
        else:
            storage = dict(slabs)
            if set(storage) != {"f32", "i32", "f64"}:
                raise ValueError("packed slabs must contain f32, i32, f64")
            n, device = storage["f32"].shape[1], storage["f32"].device.id
        if n < 1:
            raise ValueError("empty catchment batch")
        for t, count in metadata["slab_fields"].items():
            array = storage[t]
            dtype = {"f32": "float32", "i32": "int32", "f64": "float64"}[t]
            if array.shape != (count, n) or str(array.dtype) != dtype or array.device.id != device or not array.flags.c_contiguous:
                raise ValueError(f"{t}: require contiguous {dtype}[{count},{n}] on parent card")
        views = {}
        for f in metadata["fields"]:
            view = storage[f["dtype"]][f["offset"]:f["offset"]+f["count"]].T
            if not f["bounds"]:
                view = view[:, 0]
            views[f["name"]] = view
            if fields is not None:
                supplied = fields[f["name"]]
                if supplied.shape != view.shape or supplied.dtype != view.dtype or supplied.device.id != device:
                    raise ValueError(f"{f['name']}: wrong shape, dtype, or card")
                cp.copyto(view, supplied)  # setup-only CUDA copy
        status = cp.zeros(n, cp.int32)
        sticky_status = cp.zeros(n, cp.int32)
        _join_prebuild(numerics, device)
        module = compile_module(physics_source(numerics), physics_options(numerics),
                                "nextgen.noahowp.physics")
        self._kernel = module.get_function("noahowp_step_monitored")
        self._grid, self._block = ((n+63)//64,), (64,)
        self._args = (np.int32(n), storage["f32"], storage["i32"], storage["f64"],
                      status, np.int32(_STAGES[stage]), np.int32(advance), sticky_status)
        from .bmi import OUTPUTS
        bmi_converted = cp.empty((3, n), cp.float32)
        conversions = {"ETRAN": 0, "QSEVA": 1, "ECAN": 2}
        self._bmi_outputs = MappingProxyType({name: bmi_converted[conversions[name]]
            if conversion else views[field] for name, field, units, conversion in OUTPUTS})
        self.bmi_output_units = MappingProxyType({name: units for name, field, units, conversion in OUTPUTS})
        self._bmi_kernel = module.get_function("noahowp_export_bmi")
        self._bmi_args = (np.int32(n), storage["f32"], bmi_converted)
        forcing_keys = ("U2D", "V2D", "T2D", "PSFC", "Q2D", "LWDOWN", "SWDOWN", "RAINRATE")
        forcing_inputs = {}
        self._forcing_args = None
        if forcing is not None:
            if set(forcing) != set(forcing_keys):
                raise ValueError("forcing must supply the eight canonical catchment fields")
            for key in forcing_keys:
                array = forcing[key]
                if array.shape != (n,) or str(array.dtype) != "float32" or array.device.id != device or not array.flags.c_contiguous:
                    raise ValueError(f"{key}: require contiguous same-card float32 catchment field")
                forcing_inputs["forcing_input/"+key] = array
            self._forcing_kernel = module.get_function("noahowp_copy_forcing")
            self._forcing_args = (np.int32(n), storage["f32"], *(forcing[key] for key in forcing_keys))
        self.arrays = MappingProxyType(views | {"slab/"+k:v for k,v in storage.items()} |
                                      {"status": status, "sticky_status": sticky_status} | forcing_inputs)
        self.status_arrays = MappingProxyType({"lsm": sticky_status})
        # Native diagnostic bit1 is advisory. Fatal2, unsupported4 and every
        # unknown bit still prevent publication unless explicitly overridden.
        self.status_fatal_masks = MappingProxyType({"lsm": 0xfffffffe})
        self.restart_identity = dict(algorithm="noah-owp-modular.cuda/v1", source_revision=SOURCE_REVISION,
            numerics=numerics, catchments=n, nsoil=metadata["nsoil"], nsnow=metadata["nsnow"],
            parameter_sha256=parameter_sha256, fabric_sha256=fabric_sha256,
            time_origin=time_origin, stage=stage, advance=bool(advance), order="hydrofabric catchment order",
            diagnostics={"schema": "arwen.resident-sticky-status/v1",
                         "current": "status", "sticky": "sticky_status",
                         "bits": {"1": "upstream diagnostic", "2": "fatal condition",
                                  "4": "unsupported dimensions or options"}})

    def launch(self, dt_seconds, *, stream=None):
        # domain.dt is resident and checkpointed; supplied dt is driver metadata.
        if self._forcing_args is not None:
            self._forcing_kernel(self._grid, self._block, self._forcing_args, stream=stream)
        self._kernel(self._grid, self._block, self._args, stream=stream)

    advance = launch
