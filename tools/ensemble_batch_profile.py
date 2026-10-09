"""Short source-backed kernel/family profiles, original N=1 versus batch N.

Run only on a claimed remote GPU. This diagnostic instruments launches without
changing their arrays, argument order or numerical source. Event spans include
stream work and may include host submission gaps. They are not a replacement
for uninstrumented forecast throughput or hardware counters from Nsight.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import functools
import gc
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def kernel_family(module, entry):
    if module == "dycore":
        if entry in ("slow_pgf", "slow_buoyancy", "slow_geopotential"):
            return "big_step"
        if entry.startswith(("small_step_init", "small_step_finish")):
            return "bookkeeping"
    if module == "acoustic":
        if entry == "calc_coefs":
            return "acoustic_coefficients"
        if entry.startswith("advance_w_phi"):
            return "acoustic_column_solve"
        if entry.startswith("advance_mu_th"):
            return "acoustic_mass_theta"
        if entry.startswith("advance_uv"):
            return "acoustic_horizontal"
        return "acoustic_other"
    if module == "big_step" and "emdiv" in entry:
        return "acoustic_filter"
    return {
        "advection": "advection", "diagnostics": "diagnostics",
        "face_mass": "stage_fluxes", "ensemble_fluxes": "stage_fluxes",
        "bandwidth_glue": "array_glue", "rk_bookkeeping": "bookkeeping",
        "ensemble_bookkeeping": "bookkeeping", "coriolis_map": "rotation",
        "big_step": "big_step", "openbc": "open_boundaries",
        "lbc_state": "lateral_boundaries", "lbc_flow": "lateral_boundaries",
        "lbc_time": "lateral_boundaries", "diffusion": "mixing", "smag2d": "mixing",
    }.get(module, "other")


class EventRecorder:
    """Preallocated event pairs with hierarchical, same-stream spans.

    Events are created outside the profiled interval. There is no synchronization
    or elapsed-time query at a numerical launch. One synchronization collects
    each arm. Exhaustion refuses before submitting an unrecorded operation.
    """

    def __init__(self, cp, *, capacity=4096, nvtx=False):
        if capacity < 1:
            raise ValueError("event capacity must be positive")
        self.cp, self.capacity, self.nvtx = cp, capacity, bool(nvtx)
        self.events = tuple((cp.cuda.Event(), cp.cuda.Event()) for _ in range(capacity))
        self.records, self.stack, self.handles, self.sources = [], [], {}, {}
        self.enabled = False
        self._wrappers = {}

    def wrap(self, function, label, family, *, kind="raw", metadata=None):
        if (isinstance(function, TimedCallable) and function.recorder is self
                and (function.label, function.family, function.kind) == (label, family, kind)):
            return function
        key = (id(function), label, family, kind)
        if key not in self._wrappers:
            self._wrappers[key] = TimedCallable(self, function, label, family, kind, metadata or {})
        return self._wrappers[key]

    def source(self, source, origin):
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        row = self.sources.setdefault(digest, {"source": source, "origins": []})
        if origin not in row["origins"]:
            row["origins"].append(origin)
        return digest

    def invoke(self, function, label, family, kind, metadata, args, kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        index = len(self.records)
        if index == self.capacity:
            raise RuntimeError("profile event capacity exhausted; increase --max-events for a complete receipt")
        stream = kwargs.get("stream") or self.cp.cuda.get_current_stream()
        start, end = self.events[index]
        row = {"label": label, "family": family, "kind": kind,
               "parent": self.stack[-1] if self.stack else None,
               "stream_pointer": int(getattr(stream, "ptr", 0)), **metadata}
        if kind == "raw" and len(args) >= 2:
            row["grid"] = tuple(args[0])
            row["block"] = tuple(args[1])
        self.records.append(row)
        self.handles.setdefault(label, function)
        self.stack.append(index)
        if self.nvtx:
            self.cp.cuda.nvtx.RangePush(label)
        start.record(stream)
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            row["failure_type"] = type(error).__name__
            raise
        finally:
            end.record(stream)
            if self.nvtx:
                self.cp.cuda.nvtx.RangePop()
            if self.stack.pop() != index:
                raise RuntimeError("profile span hierarchy was corrupted")

    def summarize(self):
        if self.stack:
            raise RuntimeError("profile collection attempted with unfinished spans")
        self.cp.cuda.get_current_stream().synchronize()
        elapsed = [float(self.cp.cuda.get_elapsed_time(*self.events[index]))
                   for index in range(len(self.records))]
        children = [0.0] * len(elapsed)
        for index, row in enumerate(self.records):
            parent = row["parent"]
            if parent is not None:
                if row["stream_pointer"] != self.records[parent]["stream_pointer"]:
                    raise RuntimeError("nested profile moved CUDA streams; same-stream subtraction is invalid")
                children[parent] += elapsed[index]
        groups = {}
        for index, row in enumerate(self.records):
            key = (row["kind"], row["family"], row["label"])
            group = groups.setdefault(key, {"kind": key[0], "family": key[1], "label": key[2],
                "calls": 0, "inclusive_ms": 0.0, "exclusive_ms": 0.0,
                "min_ms": elapsed[index], "max_ms": elapsed[index], "grids": set(), "blocks": set(),
                "source_sha256": set()})
            group["calls"] += 1
            group["inclusive_ms"] += elapsed[index]
            group["exclusive_ms"] += elapsed[index] - children[index]
            group["min_ms"] = min(group["min_ms"], elapsed[index])
            group["max_ms"] = max(group["max_ms"], elapsed[index])
            if "grid" in row:
                group["grids"].add(row["grid"])
                group["blocks"].add(row["block"])
            if "source_sha256" in row:
                group["source_sha256"].add(row["source_sha256"])
        result = []
        for key, group in groups.items():
            group["mean_ms"] = group["inclusive_ms"] / group["calls"]
            group["grids"], group["blocks"] = sorted(group["grids"]), sorted(group["blocks"])
            group["source_sha256"] = sorted(group["source_sha256"])
            if key[0] in ("raw", "elementwise"):
                try:
                    group["kernel_attributes"] = dict(self.handles[key[2]].attributes)
                except (AttributeError, TypeError):
                    group["kernel_attributes"] = None
                if key[0] == "elementwise":
                    try:
                        codes = dict(self.handles[key[2]].cached_codes)
                    except (AttributeError, TypeError):
                        codes = {}
                    group["elementwise_compiled_source_sha256"] = [
                        self.source(source, "CuPy cached generated Elementwise source: " + key[2])
                        for source in codes.values()]
            result.append(group)
        result.sort(key=lambda row: row["exclusive_ms"], reverse=True)
        return {"events_used": len(self.records), "event_capacity": self.capacity,
                "rows": result,
                "raw_launch_span_ms": sum(elapsed[i] for i, row in enumerate(self.records)
                                           if row["kind"] in ("raw", "elementwise")),
                "driver_span_ms": sum(elapsed[i] for i, row in enumerate(self.records) if row["kind"] == "driver"),
                "uncategorized_driver_span_ms": sum(elapsed[i] - children[i] for i, row in enumerate(self.records)
                                                     if row["kind"] == "driver")}


class TimedCallable:
    def __init__(self, recorder, function, label, family, kind, metadata):
        self.recorder, self.function, self.label = recorder, function, label
        self.family, self.kind, self.metadata = family, kind, metadata

    def __getattr__(self, name):
        return getattr(self.function, name)

    def __call__(self, *args, **kwargs):
        return self.recorder.invoke(self.function, self.label, self.family, self.kind,
                                    self.metadata, args, kwargs)


def _import_optional_provider(name):
    """Observe installed experimental providers without requiring their files."""
    qualified = "gpuwm.ensemble." + name
    try:
        return importlib.import_module(qualified)
    except ModuleNotFoundError as error:
        if error.name != qualified:
            raise
        return None


@contextmanager
def instrument(recorder):
    """Patch diagnostic process aliases only, restoring them on every exit.

    The raw/Elementwise objects receive the original call arguments untouched.
    The ensemble executor and its mathematical source files are never edited.
    """
    from gpuwm.core import kernels, dycore, ieva
    from gpuwm.core.state import DomainState
    from gpuwm.ensemble import batch_kernel, batch_fluxes, batch_glue
    # Import providers before alias discovery; future imports get the patched
    # provider module attributes through their ordinary from-import statements.
    for name in ("batch_dycore", "batch_operators", "batch_acoustic", "batch_bigstep",
                 "batch_bookkeeping", "batch_diagnostics", "batch_mixing", "batch_boundaries"):
        importlib.import_module("gpuwm.ensemble." + name)
    _import_optional_provider("batch_openbc")
    restorations = []
    replacements = []
    source_cache = {}

    def raw_source(module, defines=()):
        key = ("original", module, tuple(defines))
        if key not in source_cache:
            source = (kernels.module_source_int_defines(module, tuple(defines))
                      if defines else kernels.module_source(module))
            source_cache[key] = recorder.source(source, "original raw module: " + module)
        return source_cache[key]

    def generated_source(spec, members, defines=(), supplied=None):
        key = ("batch", spec, members, tuple(defines), supplied)
        if key not in source_cache:
            source = supplied if supplied is not None else (
                kernels.module_source_int_defines(spec.module, tuple(defines)) if defines
                else kernels.module_source(spec.module))
            compiled = batch_kernel.generate_batch_source(source, spec, members,
                audit_options=batch_kernel._runtime_audit_options(spec))
            source_cache[key] = recorder.source(compiled, "prepared batch raw module: " + spec.module)
        return source_cache[key]

    def replace_aliases(original, replacement):
        replacements.append((original, replacement))
        for module in tuple(sys.modules.values()):
            namespace = getattr(module, "__dict__", {})
            if not str(namespace.get("__name__", "")).startswith("gpuwm."):
                continue
            for attribute, value in tuple(namespace.items()):
                if value is original:
                    restorations.append((module, attribute, value))
                    setattr(module, attribute, replacement)

    def raw_provider(provider):
        @functools.wraps(provider)
        def wrapped(module, entry, *args, **kwargs):
            function = provider(module, entry, *args, **kwargs)
            defines = args[0] if args else kwargs.get("defines", ())
            return recorder.wrap(function, module + ":" + entry, kernel_family(module, entry),
                                 metadata={"module": module, "entry": entry,
                                           "source_sha256": raw_source(module, defines)})
        return wrapped

    def batch_provider(provider, *, supplied=False):
        @functools.wraps(provider)
        def wrapped(*args, **kwargs):
            result = provider(*args, **kwargs)
            spec = args[1] if supplied else args[0]
            members = args[2] if supplied else args[1]
            function = result[0] if supplied else result
            defines = () if supplied else (args[2] if len(args) > 2 else kwargs.get("defines", ()))
            digest = generated_source(spec, members, defines, args[0] if supplied else None)
            timed = recorder.wrap(function, spec.module + ":" + spec.entry,
                kernel_family(spec.module, spec.entry),
                metadata={"module": spec.module, "entry": spec.entry, "members": members,
                          "source_sha256": digest})
            return (timed,) + result[1:] if supplied else timed
        return wrapped

    def flux_provider(provider):
        @functools.wraps(provider)
        def wrapped(kind, members, has_msf, reciprocal=False):
            key = ("flux", kind, members, has_msf, reciprocal)
            if key not in source_cache:
                source, spec, _ = batch_fluxes._raw_source(kind, has_msf, reciprocal)
                compiled = batch_kernel.generate_batch_source(source, spec, members)
                source_cache[key] = recorder.source(compiled, "prepared raw flux: " + kind)
            return recorder.wrap(provider(kind, members, has_msf, reciprocal),
                "ensemble_fluxes:" + kind, "stage_fluxes",
                metadata={"module": "ensemble_fluxes", "entry": kind, "members": members,
                          "source_sha256": source_cache[key]})
        return wrapped

    def elementwise_provider(provider, entry):
        @functools.wraps(provider)
        def wrapped(*args, **kwargs):
            function = provider(*args, **kwargs)
            digest = recorder.source(function.operation, "original Elementwise operation fragment: " + entry)
            return recorder.wrap(function, "ensemble_fluxes:" + entry,
                                 "stage_fluxes", kind="elementwise", metadata={"source_sha256": digest})
        return wrapped

    def family_function(function, label, family):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            return recorder.invoke(function, label, family, "family", {}, args, kwargs)
        return wrapped

    def family_factory(factory, label, family):
        @functools.wraps(factory)
        def wrapped(*args, **kwargs):
            return recorder.wrap(factory(*args, **kwargs), label, family, kind="family")
        return wrapped

    try:
        for provider in (kernels.get_kernel, kernels.get_kernel_int_defines):
            replace_aliases(provider, raw_provider(provider))
        replace_aliases(batch_kernel._compiled, batch_provider(batch_kernel._compiled))
        replace_aliases(batch_kernel._compiled_source, batch_provider(batch_kernel._compiled_source, supplied=True))
        replace_aliases(batch_fluxes._compiled, flux_provider(batch_fluxes._compiled))
        for factory, entry in ((dycore._couple_momentum_kernel, "momentum"),
                               (dycore._omega_column_kernel, "omega")):
            replace_aliases(factory, elementwise_provider(factory, entry))
        for factory, label in ((batch_glue.prepare_total_mass, "family:total_mass"),
                               (batch_glue.prepare_face_masses, "family:face_masses"),
                               (batch_glue.prepare_total_theta, "family:total_theta"),
                               (batch_glue.prepare_periodic_alias, "family:periodic_alias")):
            replace_aliases(factory, family_factory(factory, label, "array_glue"))
        for function, label, family in ((ieva.stage_face_masses, "family:face_masses", "array_glue"),
                                        (dycore.stage_fluxes, "family:stage_fluxes", "stage_fluxes"),
                                        (dycore.close_periodic_alias, "family:periodic_alias", "array_glue")):
            replace_aliases(function, family_function(function, label, family))
        total_mu = DomainState.total_mu
        restorations.append((DomainState, "total_mu", total_mu))
        DomainState.total_mu = family_function(total_mu, "family:total_mass", "array_glue")
        yield
    finally:
        # Include aliases imported while the instrumented step was running.
        for original, replacement in reversed(replacements):
            for module in tuple(sys.modules.values()):
                namespace = getattr(module, "__dict__", {})
                if str(namespace.get("__name__", "")).startswith("gpuwm."):
                    for attribute, value in tuple(namespace.items()):
                        if value is replacement:
                            setattr(module, attribute, original)
        for owner, name, value in reversed(restorations):
            setattr(owner, name, value)


def _write(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_sources(path, recorder):
    folder = path.with_name(path.stem + ".sources")
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for digest, row in sorted(recorder.sources.items()):
        target = folder / (digest + ".cu")
        target.write_text(row["source"], encoding="utf-8", newline="")
        rows.append({"source_sha256": digest, "file": str(target.relative_to(path.parent)),
                     "bytes": len(row["source"].encode("utf-8")), "origins": row["origins"]})
    return rows


def _revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _comparison(arms):
    if len(arms) != 2:
        return []
    grouped = []
    for arm in arms:
        families = {}
        for row in arm["rows"]:
            if row["kind"] == "driver":
                continue
            family = families.setdefault(row["family"], {"ms": 0.0, "kernel_calls": 0})
            family["ms"] += row["exclusive_ms_per_step"]
            if row["kind"] in ("raw", "elementwise"):
                family["kernel_calls"] += row["calls"]
        grouped.append(families)
    result = []
    for name in sorted(grouped[0].keys() | grouped[1].keys()):
        single, batch = grouped[0].get(name, {}), grouped[1].get(name, {})
        single_ms, batch_ms = single.get("ms", 0.0), batch.get("ms", 0.0)
        result.append({"family": name, "original_single_exclusive_ms_per_step": single_ms,
            "batch_exclusive_ms_per_step": batch_ms,
            "batch_over_single": batch_ms / single_ms if single_ms else None,
            "batch_ms_per_member": batch_ms / arms[1]["members"],
            "scope": "instrumented family spans; hardware kernel time and forecast speedup are not inferred"})
    result.sort(key=lambda row: row["batch_exclusive_ms_per_step"], reverse=True)
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--control-steps", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--revision", help="source snapshot revision when the remote copy has no usable .git")
    args = parser.parse_args(argv)
    import math
    if min(args.nx, args.ny, args.nz, args.steps, args.members, args.max_events) < 1:
        parser.error("grid, members, profile steps and event capacity must be positive")
    if args.members <= 1:
        parser.error("--members must exceed 1 for an original-versus-batch profile")
    if min(args.warmup, args.control_steps, args.reserve_mib) < 0:
        parser.error("warmup, control steps and reserve must be nonnegative")
    if not all(math.isfinite(value) and value > 0 for value in (args.dx, args.dt)):
        parser.error("spacing and timestep must be finite and positive")
    return args


def profile_configuration(args):
    """Keep the fixed clock's declared interval inclusive of its short control."""
    from tools import ensemble_batch_step_probe as probe
    return replace(probe.configuration(args),
                   run_seconds=(args.warmup + args.control_steps + args.steps) * args.dt)


def run(args):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.certify.kernel_manifest import kernel_manifest
    from gpuwm.ensemble.batch_dycore import prepare_dry_step
    from gpuwm.ensemble.batch_state import BatchedDomainState
    from tools import ensemble_batch_step_probe as probe

    receipt = {"schema": "gpuwm/ensemble-kernel-family-profile/v1", "status": "running",
        "revision": args.revision or _revision(), "platform": platform.platform(), "python": sys.version,
        "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workload": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "scope": "short periodic dry acoustic RK3 profile; original single versus resident member batch",
        "timing_scope": "same-stream CUDA event spans, including possible CPU submission gaps; no forecast throughput claim",
        "arithmetic": "original callable, arrays, arguments and CUDA arithmetic sources unchanged",
        "identity": "not checked by this timing tool; use independent component and full-history receipts",
        "layout": "current member_outermost; no layout transformation enabled", "arms": []}
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props["name"]
    receipt.update(gpu=name.decode() if isinstance(name, bytes) else name,
                   cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                   cuda_driver=cp.cuda.runtime.driverGetVersion(), cupy=cp.__version__)
    if args.nvtx and not all(hasattr(cp.cuda.nvtx, name) for name in ("RangePush", "RangePop")):
        raise RuntimeError("--nvtx requested but the installed CuPy NVTX range API is unavailable")
    cfg = profile_configuration(args)
    plan, shared, extras, slots = probe.allocation_plan(cfg, args.reserve_mib * 1024**2)
    _write(args.receipt, receipt)
    try:
        for count in (1, args.members):
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            available, total = cp.cuda.runtime.memGetInfo()
            plan.admit(count, available_bytes=available)
            inputs = probe.host_members(cfg, count, extras)
            recorder = EventRecorder(cp, capacity=args.max_events, nvtx=args.nvtx)
            with instrument(recorder):
                if count == 1:
                    state = probe.scalar_state(inputs[0])
                    advance = lambda: dycore.step(state, cfg, acoustic=True)
                else:
                    state = BatchedDomainState.from_prepared(inputs, array_module=cp, available_bytes=available,
                        shared_fields=shared, extra_specs=extras, scratch_slots={slot: "float32" for slot in slots},
                        reserved_bytes=plan.reserved_bytes)
                    advance = prepare_dry_step(state)
                for _ in range(args.warmup):
                    advance()
                control = None
                if args.control_steps:
                    wall, device = probe.time_steps(advance, args.control_steps)
                    control = {"steps": args.control_steps, "wall_seconds": wall, "device_seconds": device,
                               "scope": "recording disabled; forwarding wrappers remain bound and add host overhead; preceding trajectory only, not a throughput benchmark"}
                cp.cuda.get_current_stream().synchronize()
                recorder.enabled = True
                started = time.perf_counter()
                profiled = recorder.wrap(advance, "driver:step", "driver", kind="driver")
                for _ in range(args.steps):
                    profiled()
                cp.cuda.get_current_stream().synchronize()
                profile_wall = time.perf_counter() - started
                recorder.enabled = False
                arm = recorder.summarize()
                arm.update(members=count, implementation="original_single" if count == 1 else "member_outermost_batch",
                           profile_steps=args.steps, profile_wall_seconds=profile_wall, control=control,
                           pool_live_bytes=cp.get_default_memory_pool().used_bytes(), total_bytes=total,
                           declared_batch_bytes=plan.required_bytes(count),
                           source_manifest=kernel_manifest(),
                           source_manifest_scope="process-cumulative compiler manifest snapshot",
                           source_files=_write_sources(args.receipt, recorder))
                for row in arm["rows"]:
                    row["exclusive_ms_per_step"] = row["exclusive_ms"] / args.steps
                    row["inclusive_ms_per_step"] = row["inclusive_ms"] / args.steps
                receipt["arms"].append(arm)
                _write(args.receipt, receipt)
                print(json.dumps({"members": count, "events_used": arm["events_used"],
                                  "driver_span_ms": arm["driver_span_ms"],
                                  "raw_launch_span_ms": arm["raw_launch_span_ms"],
                                  "uncategorized_driver_span_ms": arm["uncategorized_driver_span_ms"]}), flush=True)
            state = None  # rebound, not deleted: a closure above reads it
            del profiled, advance, recorder, inputs
        receipt["status"] = "measured"
        receipt["family_comparison"] = _comparison(receipt["arms"])
    except BaseException as error:
        receipt.update(status="failed", failure_type=type(error).__name__, failure_message=str(error))
        _write(args.receipt, receipt)
        raise
    _write(args.receipt, receipt)
    return receipt


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    run(arguments())
