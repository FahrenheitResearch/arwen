"""Typed catchment diagnostics: CUDA capture and the existing Rust NetCDF writer.

These are module histories, with their original units and catchment labels.
No area or discharge conversion is implied. A separate full-state checkpoint
uses the shared Rust envelope; diagnostic archives themselves are not restarts.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time
from types import MappingProxyType

_CAPTURE = r'''
extern "C" __global__ void capture(
    const unsigned long long* sources, const unsigned long long* targets,
    const unsigned long long* sizes, const unsigned long long* step,
    unsigned long long* overflow, unsigned long long capacity, int fields) {
    unsigned long long frame = *step;
    if (frame >= capacity) { if (!blockIdx.x && !threadIdx.x) *overflow=1; return; }
    unsigned long long byte = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (int f=0; f<fields; ++f) {
        if (byte < sizes[f]) {
            const unsigned char* src = (const unsigned char*)sources[f];
            unsigned char* dst = (unsigned char*)targets[f];
            dst[frame*sizes[f]+byte] = src[byte];
        }
    }
}
extern "C" __global__ void tick(unsigned long long* step) { ++*step; }
'''


class ModuleHistory:
    def __init__(self, *, settings, options, components, device_id):
        import cupy as cp
        import numpy as np

        self.capacity = options["capacity_steps"]
        ids = options["catchment_ids"]
        if (type(self.capacity) is not int or self.capacity < 1 or not ids
                or len(set(ids)) != len(ids)):
            raise ValueError("module history needs capacity and unique ordered catchment labels")
        self.interval = float(settings["interval_seconds"])
        arrays = {"step": cp.zeros((), cp.uint64), "overflow": cp.zeros((), cp.uint64)}
        sources, targets, sizes, fields = [], [], [], []
        for row in options["fields"]:
            name = row["name"]
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in arrays:
                raise ValueError("module history field names must be unique NetCDF identifiers")
            component = components[row["component"]]
            available = dict(component.arrays) | dict(getattr(component, "fields", {}))
            source = available[row["field"]]
            if (source.shape != (len(ids),) or not source.flags.c_contiguous
                    or source.device.id != device_id
                    or str(source.dtype) not in ("float32", "float64", "int32", "int64")):
                raise ValueError(f"{name} needs a contiguous native-typed vector on the parent card")
            target = cp.zeros((self.capacity, len(ids)), dtype=source.dtype)
            arrays[name] = target
            sources.append(source)
            targets.append(target)
            sizes.append(source.nbytes)
            fields.append(dict(row, dtype=str(source.dtype), source_identity=component.restart_identity))
        if not fields:
            raise ValueError("module history needs at least one field")
        # Pointer/extent tables are immutable setup metadata, rebuilt on preparation.
        self._sources = tuple(sources)
        self._tables = (cp.asarray([v.data.ptr for v in sources], cp.uint64),
                        cp.asarray([v.data.ptr for v in targets], cp.uint64),
                        cp.asarray(sizes, cp.uint64))
        module = cp.RawModule(code=_CAPTURE, options=("--std=c++11",))
        from gpuwm.certify.kernel_manifest import record_module
        record_module("nextgen_module_history", source=_CAPTURE, options=("--std=c++11",), module=module)
        self._capture = module.get_function("capture")
        self._tick = module.get_function("tick")
        self._grid = ((max(sizes)+127)//128,)
        self._args = (*self._tables, arrays["step"], arrays["overflow"],
                      np.uint64(self.capacity), np.int32(len(fields)))
        self._tick_args = (arrays["step"],)
        self.arrays = MappingProxyType(arrays)
        self.restart_identity = {"schema": "arwen.nextgen.module-history/v1",
                                 "catchment_ids": list(ids), "capacity_steps": self.capacity,
                                 "interval_seconds": self.interval, "origin": options["origin"],
                                 "sample_position": "interval_end", "fields": fields}

    def advance(self, dt_seconds, *, stream=None):
        if dt_seconds != self.interval:
            raise ValueError("module history interval differs from its prepared driver")
        self._capture(self._grid, (128,), self._args, stream=stream)
        self._tick((1,), (1,), self._tick_args, stream=stream)


def prepare(*, settings, options, state, physics, device_id, mode, components):
    effective = dict(settings)
    if mode == "coupled":
        effective["interval_seconds"] = physics.bldt_seconds
    return ModuleHistory(settings=effective, options=options, components=components,
                         device_id=device_id)


def prepare_memory(*, settings, options, context, components):
    from .data_memory import module_history_memory
    return module_history_memory(settings=settings, options=options, context=context, components=components)


def _text(name, value):
    return {"name": name, "dtype": "char", "text": value}


class Writer:
    def __init__(self, *, settings, driver, outdir):
        from .output import _library, _select_output_directory, _report_output_directory
        self.requested_outdir = Path(outdir)
        self.outdir = _select_output_directory(self.requested_outdir)
        self.histories = [(name, component) for name, component in driver.components.items()
                          if isinstance(component, ModuleHistory)]
        if not self.histories:
            raise ValueError("module archive output requires a prepared module history")
        _library()
        if self.outdir != self.requested_outdir:
            _report_output_directory(self.outdir)

    def restore(self, path, driver):
        from .output import restore_checkpoint
        return restore_checkpoint(path, driver)

    def finish(self, driver, receipt):
        from .output import _snapshot, _call, _select_output_directory, _report_output_directory, _history_steps
        # Files may have arrived after preparation. Select again before D2H
        # or publishing any archive; preserve the occupied path in place.
        directory = _select_output_directory(self.outdir)
        if directory != self.outdir:
            self.outdir = directory
            _report_output_directory(self.outdir)
        import cupy as cp
        start = time.perf_counter_ns()
        captures = []
        for name, history in self.histories:
            steps = _history_steps(history)
            if steps > history.capacity:
                raise ValueError("module history buffer exhausted; increase realization capacity_steps before running")
            if steps == 0:
                raise ValueError("module archive needs at least one completed history capture")
            captures.append((name, history, steps))
        self.outdir.mkdir(parents=True, exist_ok=True)
        with cp.cuda.Device(driver.device_id):
            roster, _, payload = _snapshot(driver.arrays, cp.cuda.get_current_stream())
        checkpoint, _ = _call({"operation": "checkpoint-write",
                              "path": str(self.outdir / "restart.nc"),
                              "metadata": driver.checkpoint_metadata(), "arrays": roster}, payload)
        products = []
        for name, history, steps in captures:
            fields = {f["name"]: f for f in history.restart_identity["fields"]}
            completed = {key: value[:steps] if key in fields else value
                         for key, value in history.arrays.items()}
            with cp.cuda.Device(driver.device_id):
                roster, offsets, payload = _snapshot(completed, cp.cuda.get_current_stream())
            dims = [{"name": "time", "len": steps},
                    {"name": "catchment", "len": len(history.restart_identity["catchment_ids"])}]
            variables = []
            for row in roster:
                key = row["name"]
                attrs = [] if key not in fields else [_text("units", fields[key]["units"]),
                    _text("source_component", fields[key]["component"]),
                    _text("source_field", fields[key]["field"])]
                variables.append({"name": key, "dtype": row["dtype"],
                                  "dimensions": ["time", "catchment"] if key in fields else [],
                                  "attributes": attrs, "offset": offsets[key]})
            schema = {"format": "cdf5", "dimensions": dims, "variables": variables,
                      "attributes": [_text("arwen_schema", "arwen.nextgen.module-history/v1"),
                                     _text("history_identity", json.dumps(history.restart_identity)),
                                     _text("scope", "catchment module diagnostics; not a full-state restart")]}
            result, _ = _call({"operation": "dataset-write",
                               "path": str(self.outdir / (name+".nc")), "schema": schema}, payload)
            products.append(result)
        receipt.update(output_directory=str(self.outdir),
                       requested_output_directory=str(self.requested_outdir),
                       checkpoint=checkpoint, products=products,
                       output_total_ms=(time.perf_counter_ns()-start)/1e6,
                       output_scope="native typed catchment histories and separate complete resident checkpoint; D2H/Rust NetCDF at finish")
        (self.outdir/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
        return receipt


def prepare_writer(*, settings, driver, outdir):
    return Writer(settings=settings, driver=driver, outdir=outdir)
