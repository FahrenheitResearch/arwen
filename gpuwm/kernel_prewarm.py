"""Compile the uncached dycore diffusion modules while the input is read.

The defect (NV recipe dry run, D5): the first step of NCAR's CONUS 12 km
restart leg took 22 to 37 s on a WARM kernel cache, against a 50 to 57 ms
steady step.  A wall-clock sampler of the worker put 32 s of the 33.5 s
first step inside one NVRTC call: ``compile_diffusion_source`` compiling
``smag2d`` and ``diff_opt1`` on the first ``launch_wrf_smag2d_km``.  The
diffusion units compile through CuPy's direct NVRTC entry point (it keeps
``--ftz=false``, which ``RawModule`` would override), and that entry point
has no disk cache, so every run pays the compile on step 1.

Two changes remove it.  ``compile_diffusion_source`` now keeps its image in
the direct route's persistent exact-image cache
(:mod:`gpuwm.nvrtc_ptx_cache`, the store the no-FTZ units already use), so
a warm kernel cache no longer compiles these units at all.  And on a cold
cache, the worker starts these exact loader calls on a background thread
before it reads the input (the restart read is 10 to 42 s with the card
idle), so step 1 finds the module built or being built.  The image is the one the
loader would have built on step 1: same function, same source, same
options, same process-wide cache entry (``cuda_cache`` keys by device and
serialises first use, so step 1 waits for a compile still running instead
of starting a second one).

EVERY KERNEL, EVERY CORE (2026-10-16).  The diffusion units are three of the
160-220 modules a cold first step compiles one after another on one core.
:func:`start_plan` hands the configuration's whole recorded kernel set to
:mod:`gpuwm.kernel_plan`, which compiles it on a pool as wide as the usable
cores, longest first, through the one NVRTC funnel every route ends in; each
route still writes its own cache entry from the same bytes.

Only modules that carry no physics-parameter literals are warmed here:
their source cannot depend on a parameter set declared later
(:func:`gpuwm.physics_params.declare` refuses a set that arrives after an
affected module compiled, and none of these is affected).

The dedicated cache-warming process also executes tiny real CUDA
preparations beside that pool. CuPy's generated array kernels are keyed by
their source, so static RawModule loaders alone cannot warm them.

THE FORECAST'S OWN VARIANTS (2026-10-20, HRRR on G12).  The static plan names
modules, not the variants a run's selectors build: the NOAA-WRF39 HRRR suite's
step 1 still compiled 98 s on one core (Thompson-aerosol wrf39 units, MYNN
gsd41 and its validators, RUC, the RRTMG legacy adapter, the pd_*/flux_div
advection variants, the upper-wind limiter, CuPy's reductions) with the static
plan fully loaded.  The same process therefore also steps a 16 x 16 synthetic
column under the run's OWN configuration (:func:`gpuwm.warm_kernels.
forecast_column_config`) twice, through the production physics init and
dycore step, so every loader the forecast reaches compiles here with the
run's own selectors and physics-parameter set, beside the fetch and the
preparation instead of inside step 1.
"""

from __future__ import annotations

import os
import threading
import time

#: The units the direct, uncached NVRTC route compiles for the WRF
#: dycore, in the order the first step asks for them.
DIFFUSION_ROUTE_MODULES = ("smag2d", "diff_opt1", "diff6")

#: ``GPUWM_KERNEL_PREWARM=0`` turns the background compile off (step 1
#: then compiles, as before).  Default on.
ENV = "GPUWM_KERNEL_PREWARM"


def modules_for(experiment) -> tuple[str, ...]:
    """The diffusion-route units this experiment's dycore launches."""

    domains = tuple(getattr(experiment, "domains", ()) or ())
    runs = [getattr(domain, "run", None) for domain in domains]
    runs = [run for run in runs if run is not None]
    if not runs:
        return ()
    wanted = []
    if any(int(getattr(run, "diff_opt", 0) or 0) >= 1
           or int(getattr(run, "km_opt", 0) or 0) != 0 for run in runs):
        wanted.append("smag2d")
    if any(int(getattr(run, "diff_opt", 0) or 0) == 1 for run in runs):
        wanted.append("diff_opt1")
    if any(int(getattr(run, "diff_6th_opt", 0) or 0) != 0 for run in runs):
        wanted.append("diff6")
    return tuple(name for name in DIFFUSION_ROUTE_MODULES if name in wanted)


class Prewarm:
    """One background compile of ``modules``; :meth:`join` reports it."""

    def __init__(self, modules):
        self.modules = tuple(modules)
        self.seconds: dict[str, float] = {}
        self.errors: dict[str, str] = {}
        self.started = time.perf_counter()
        self.finished: float | None = None
        self._thread = threading.Thread(target=self._run, name="kernel-prewarm",
                                        daemon=True)

    def start(self) -> "Prewarm":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            import cupy as cp
            from gpuwm.core import kernels
            from gpuwm.physics_params import _affected_modules

            affected = set(_affected_modules())
            unknown = set(self.modules) - set(kernels._DIFFUSION_MODULES)
            if unknown:
                raise ValueError(f"not diffusion-route units: {sorted(unknown)}")
            # The worker's card is device 0 of its mask, the device the run
            # opens; the cache entry is keyed by it.
            with cp.cuda.Device(0):
                for name in self.modules:
                    if name in affected:
                        self.errors[name] = "carries physics-parameter literals; not warmed"
                        continue
                    began = time.perf_counter()
                    try:
                        # The diffusion loader itself, not load_module: the
                        # cache lock is per loader, and holding load_module's
                        # through a 31 s compile stalled every other module
                        # the restore loads (V12 probe with a 10 s read:
                        # init 3.9 s became 36.6 s).  load_module(name) for
                        # these units is this call, so step 1 hits its entry.
                        kernels._load_diffusion_module(name)
                    except Exception as error:  # noqa: BLE001 - step 1 reports it
                        self.errors[name] = f"{type(error).__name__}: {error}"
                    self.seconds[name] = round(time.perf_counter() - began, 3)
        except Exception as error:  # noqa: BLE001 - nothing here may fail a run
            self.errors["*"] = f"{type(error).__name__}: {error}"
        finally:
            self.finished = time.perf_counter()

    def join(self, timeout: float | None = None) -> dict:
        self._thread.join(timeout)
        return self.receipt()

    def receipt(self) -> dict:
        return {"modules": list(self.modules), "seconds": dict(self.seconds),
                "errors": dict(self.errors),
                "wall_seconds": (None if self.finished is None
                                 else round(self.finished - self.started, 3))}


#: Labels do not change kernels. Grid extents remain in the shape key.
_PLAN_TAG_SKIP = frozenset({"case", "grid_id"})


def plan_tag(experiment) -> str:
    """The configuration as kernels see it: each domain's integer, boolean and
    string run options (physics selections, vertical levels, numerics)."""

    import dataclasses
    from gpuwm import kernel_plan

    parts = ["gpuwm-regional"]
    for domain in tuple(getattr(experiment, "domains", ()) or ()):
        run = getattr(domain, "run", None)
        if run is None or not dataclasses.is_dataclass(run):
            continue
        parts.append(";".join(
            f"{field.name}={value!r}" for field in dataclasses.fields(run)
            if field.name not in _PLAN_TAG_SKIP
            and type(value := getattr(run, field.name, None)) in (int, float, bool, str)))
    parts.append(kernel_plan.default_tag_text(()))
    return "|".join(parts)


#: Integer tiers a forecast loads beside a module's plain build, keyed by
#: module: (loader name, defines).  smag2d's scalar flux pair takes the
#: 32-bit index tier on every grid ``dycore.scalar_index32_fits`` admits
#: (gpuwm/core/dycore.py ``_wrf_smag_scalar_kernel``), which is every
#: regional grid a card can hold; the plain build serves the rest of smag2d.
STATIC_TIERS = {"smag2d": (("_load_diffusion_module", (("GPUWM_SMAG_INDEX32", 1),)),)}


def static_modules(experiment) -> tuple[str, ...]:
    """The kernel modules this configuration launches, read from its selectors.

    The set ``gpuwm check`` prices (:func:`gpuwm.core.preflight.
    physics_kernel_modules`: the dycore's core modules plus each domain's
    physics rows), without the fragments that never compile alone.  The
    dycore's core set alone when the selectors cannot be priced.
    """

    from gpuwm.core import preflight

    try:
        modules = set(preflight.physics_kernel_modules(experiment))
    except Exception:  # noqa: BLE001 - an unpriceable selection still warms the dycore
        modules = set(preflight.CORE_KERNEL_MODULES)
    # The frame inventory omits zero-frame helpers from some selector rows.
    # They still compile on first use, so the cold-cache plan needs them.
    modules.update({"rk_bookkeeping", "held_heating", "surface_w", "real_init", "real_init_math"})
    if "diff6" in modules:
        modules.add("diff6_seam")
    if "rrtmgp_rte" in modules:
        modules.update({"rrtmgp_validation", "rrtmg_legacy_solar"})
    if any(getattr(getattr(dc, "run", None), "nwp_diagnostics", 0)
           for dc in getattr(experiment, "domains", ())):
        modules.add("uh_diag")
    modules -= set(preflight.UNMEASURED_KERNEL_MODULES)
    return tuple(sorted(modules))


def _source_bytes(name: str) -> int:
    """A compile-cost proxy that reads no source: the unit's and its headers' file sizes."""

    from gpuwm.core import kernels

    total = 0
    for file_name in (f"{name}.cu", *kernels.EXTRA_HEADERS.get(name, ())):
        try:
            total += (kernels._KDIR / file_name).stat().st_size
        except OSError:
            pass
    return total


def static_jobs(experiment) -> list[tuple[str, object, tuple]]:
    """(label, loader factory, arguments) for every module :func:`static_modules` names.

    Each factory is the loader's own uncached body (``__wrapped__``: the
    process-wide ``cuda_cache`` lock it holds would let only one capture run
    at a time) -- the route the run's ``get_kernel`` reaches for that module.
    The measured long sfclay compile first, then largest source first, so
    the long compiles start together on a narrow box.
    """

    from gpuwm.core import kernels

    jobs = []
    # The sfclay translation unit's optimization costs more than its file
    # size suggests. Starting it tenth left an 8 s serial pool tail on the
    # cold go route, so start that long compile beside the largest units.
    for name in sorted(static_modules(experiment), key=lambda n: (n != "sfclay", -_source_bytes(n), n)):
        if name in _COMPOSED_LOADERS:
            jobs.append((name, _warm_composed, (name,)))
        elif name in _DEVICE_LOADERS:
            jobs.append((name, _warm_device_loader, (name,)))
        elif name == "kf":
            jobs.append((name, _warm_kf, ()))
        elif name in kernels._DIFFUSION_MODULES:
            jobs.append((name, kernels._load_diffusion_module.__wrapped__, (name,)))
        else:
            jobs.append((name, kernels.load_module.__wrapped__, (name,)))
        for loader, defines in STATIC_TIERS.get(name, ()):
            tier = ",".join(f"{key}={value}" for key, value in defines)
            jobs.append((f"{name}[{tier}]", getattr(kernels, loader).__wrapped__, (name, defines)))
    # RTE specializes both above-model columns, using the same layer-count
    # function as the radiation adapter. A plain build cannot warm these.
    modules = set(static_modules(experiment))
    if "rrtmgp_rte" in modules:
        from gpuwm.core.rrtmgp import rrtmgp_above_model_layer_counts

        p_top = getattr(getattr(experiment, "vertical", None), "p_top", 0.0)
        upper = rrtmgp_above_model_layer_counts(p_top)
        tiers = {int(dc.run.nz) + extra for dc in experiment.domains for extra in upper}
        for nlay in sorted(tiers):
            defines = (("RRTMGP_MAX_LAYERS", nlay),)
            jobs.append((f"rrtmgp_rte[RRTMGP_MAX_LAYERS={nlay}]",
                         kernels.load_module_int_defines.__wrapped__, ("rrtmgp_rte", defines)))
    jobs.extend(_inline_jobs(modules))
    jobs.extend(_variant_jobs(experiment, modules))
    return sorted(jobs, key=lambda job: (
        job[0].split("[")[0] != "sfclay", -_source_bytes(job[0].split("[")[0]), job[0]))


def _variant_jobs(experiment, modules):
    """Compile selector and dimension variants before their first launch.

    The tiny execution column cannot warm a different addressing shape.
    Submit both its shape and the declared domain shapes without allocating
    those domains. Integer dimensions stay part of the compiled source/key.
    """
    from gpuwm.core import kernels
    from gpuwm.core.kernel_shape import FUNCTIONS, shape_defines
    from gpuwm.core.thompson_aerosol_launch import AEROSOL_KERNEL_MODULES, WRF39_DEFINES
    from gpuwm.core.mynn_sfclay import MYNN_SFCLAY_DEFINES

    jobs, seen = [], set()
    runs = [dc.run for dc in getattr(experiment, "domains", ()) if hasattr(dc, "run")]
    # Simple selector-only views used by callers need no synthetic geometry.
    if all(hasattr(run, "nx") for run in runs):
        runs.extend(forecast_columns(experiment))
    for run in runs:
        for name in sorted(modules):
            defines = ()
            if name in AEROSOL_KERNEL_MODULES and getattr(run, "thompson_version", "") == "wrf_39_noaa":
                defines = WRF39_DEFINES
            elif name == "mynn_pbl" and getattr(run, "bl_mynn_version", "") == "gsd_41":
                defines = (("MYNN_GSD41", 1),)
            elif name == "mynn_surface":
                defines = MYNN_SFCLAY_DEFINES[getattr(run, "mynn_sfclay_variant", "wrf_461")]
            tiers = [defines] if defines else []
            if name in FUNCTIONS and all(hasattr(run, dim) for dim in ("nz", "ny", "nx")):
                nz, ny, nx = (int(getattr(run, dim)) for dim in ("nz", "ny", "nx"))
                shape = ((1, 1, ny * nx) if name == "mynn_surface" else
                         (nz, 1, ny * nx) if name.startswith("mynn_") else (nz, ny, nx))
                tiers.append(defines + shape_defines(name, FUNCTIONS[name][0], shape))
            for tier in tiers:
                if (name, tier) in seen:
                    continue
                seen.add((name, tier))
                loader = (kernels._load_diffusion_module if name in kernels._DIFFUSION_MODULES
                          else kernels.load_module_int_defines)
                label = name + "[" + ",".join(f"{key}={value}" for key, value in tier) + "]"
                jobs.append((label, loader.__wrapped__, (name, tier)))
    return jobs


# Preflight names assembled translation units, not source files. Use the
# forecast factories, including their options and symbol/constant setup.
_COMPOSED_LOADERS = {
    "noahmp_driver_composed": ("noahmp_driver_gpu", "driver_module"),
    "noahmp_energy_composed": ("noahmp_energy_gpu", "energy_module"),
    "noahmp_glacier_composed": ("noahmp_glacier_gpu", "glacier_module"),
    "noahmp_thermal_composed": ("noahmp_thermal_gpu", "thermal_module"),
    "noahmp_libm_slab_composed": ("noahmp_slab_libm", "_module"),
    "noahmp_vegeflux_runtime": ("noahmp_vegeflux_gpu", "_module"),
    "rrtmg_lw_legacy_chain": ("rrtmg_lw", "_gpu_module"),
    "rrtmg_sw_legacy": ("rrtmg_legacy", "_cuda_sw"),
    # The chained translation units preflight prices
    # (gpuwm/core/preflight.py CHAINED_TRANSLATION_UNIT_FRAMES).  None has a
    # .cu of its own, so load_module refused each one: the HRRR v4 recipe's
    # gwd_opt = 3 put terrain_drag_composed in the plan and the pre-warm
    # child exited 1 with "No such file ... terrain_drag_composed.cu" (V59
    # 8x5090, 2026-10-21).
    "p3_composed": ("p3_device", "p3_module"),
    "urban_bem_composed": ("urban_bem", "_bem_module"),
    "noah_mosaic_unit": ("noah_mosaic", "_mosaic_module"),
    "noah_mosaic_ucm_unit": ("noah_mosaic", "_mosaic_ucm_module"),
}

#: Chained units whose forecast factory takes the device it loads on.
_DEVICE_LOADERS = {
    "terrain_drag_composed": ("terrain_drag", "_module"),
    "thompson_fork_unit": ("thompson_fork", "_module"),
}


def _warm_composed(name):
    import importlib

    module_name, factory_name = _COMPOSED_LOADERS[name]
    module = importlib.import_module("gpuwm.core." + module_name)
    factory = getattr(module, factory_name)
    if name == "rrtmg_sw_legacy":
        return factory(module._sw_tables())
    return factory()


def _warm_device_loader(name):
    """A chained unit whose loader is keyed by device, on the current one."""
    import importlib

    import cupy as cp

    module_name, factory_name = _DEVICE_LOADERS[name]
    module = importlib.import_module("gpuwm.core." + module_name)
    return getattr(module, factory_name)(cp.cuda.runtime.getDevice())


def _warm_kf():
    """KF runs on its own unflushed direct-NVRTC route, not load_module."""
    import cupy as cp
    from gpuwm.core.kf import _kf_module

    return _kf_module(cp.cuda.runtime.getDevice())


def _inline_jobs(modules):
    """Compile inline translation units through the forecast's own factories."""
    from gpuwm.core import tendency_coupling

    jobs = [("gpuwm.core.tendency_coupling", tendency_coupling._kernel, ("couple_mass_rates",))]
    if "noah" in modules:
        from gpuwm.core import noah_forcing, noah_sfcdiags

        jobs.extend([
            ("gpuwm.core.noah_forcing", noah_forcing._kernel, ("noah_forcing",)),
            ("gpuwm.core.noah_sfcdiags", noah_sfcdiags._kernel, ())])
    if "kf" in modules:
        from gpuwm.core import cumulus_clock

        jobs.extend([
            ("gpuwm.core.cumulus_clock:advance", cumulus_clock._advance_kernel, ()),
            ("gpuwm.core.cumulus_clock:clear", cumulus_clock._clear_kernel, ())])
    if "rrtmgp_rte" in modules:
        from gpuwm.core import noahmp_slab_libm

        jobs.append(("gpuwm.core.noahmp_slab_libm:noahmp_libm_slab",
                     noahmp_slab_libm._module, ()))
    return jobs


#: Plan tags whose static jobs this process already submitted: once per
#: configuration (a warm-kernels worker rebinds a tag per profile).
_static_started: set[str] = set()


def single_domain_experiment(run, *, grid_id: int = 1):
    """One-domain experiment view of a bare run configuration, for the static plan.

    ``gpuwm warm-kernels`` builds each profile from a :class:`RunConfig`
    with no experiment around it; :func:`static_modules` reads the same
    selectors either way.  ``history_interval_s`` 0 prices the
    reflectivity rail (every module the profile can launch is wanted).
    """

    from types import SimpleNamespace

    return SimpleNamespace(
        domains=(SimpleNamespace(run=run, grid_id=int(grid_id), history_interval_s=0.0),),
        run_seconds=float(getattr(run, "run_seconds", 0.0) or 0.0))


def submit_static(experiment) -> int:
    """Submit the static plan's captures to the plan pool; the number submitted.

    Once per plan tag in a process (the pool compiles into one process-wide
    table).  Zero when the plan has no pool open, this tag already took its
    static jobs, or the configuration cannot be read.
    """

    from gpuwm import kernel_plan

    tag = kernel_plan._state.get("tag")
    if tag is None or tag in _static_started or kernel_plan._state.get("pool") is None:
        return 0
    _static_started.add(tag)
    try:
        jobs = static_jobs(experiment)
    except Exception:  # noqa: BLE001 - the recorded plan still runs
        return 0
    for _label, factory, arguments in jobs:
        kernel_plan.submit(kernel_plan.ahead, factory, *arguments)
    return len(jobs)


#: How long the forecast column waits for ``gpuwm go``'s own table fetch to
#: stage a physics table it reads (the child starts before that fetch).
FORECAST_TABLE_WAIT_S = 180.0


def forecast_columns(experiment) -> list:
    """One synthetic-column configuration per distinct domain configuration."""

    from gpuwm.warm_kernels import forecast_column_config

    configs, seen = [], set()
    for domain in tuple(getattr(experiment, "domains", ()) or ()):
        run = getattr(domain, "run", None)
        if run is None:
            continue
        cfg = forecast_column_config(run)
        key = repr(cfg)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def warm_forecast_columns(experiment, *, wait_s: float = FORECAST_TABLE_WAIT_S) -> int:
    """Step each domain's synthetic column twice; the number stepped.

    A physics table not on disk yet (``gpuwm go`` stages them in its own
    fetch, after this child starts) is waited for, up to ``wait_s``.
    """

    import cupy as cp
    from gpuwm.warm_kernels import run_column

    stepped = 0
    deadline = time.monotonic() + float(wait_s)
    with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True):
        for cfg in forecast_columns(experiment):
            _warm_column_reductions(cfg)
            while True:
                try:
                    run_column(cfg, reflectivity=True)
                    break
                except FileNotFoundError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(2.0)
            stepped += 1
    return stepped


def _warm_column_reductions(cfg):
    """Warm independent generated validators concurrently, at the column's sizes.

    CuPy CUB chooses distinct compile-time block sizes for vertical profiles,
    surface fields and volume fields. Warming just a scalar misses these.
    Each job owns its stream and arrays; no forecast state is shared.
    """
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor
    from gpuwm import kernel_plan

    sizes = (cfg.nz, cfg.ny * cfg.nx, cfg.nz * cfg.ny * cfg.nx)
    device = cp.cuda.runtime.getDevice()
    jobs = [("any", "bool"), ("all", "bool"), ("count_nonzero", "bool"),
            ("any", "uint8"), ("sum", "float32")]
    if int(cfg.bl_pbl_physics) == 5:
        jobs.append(("nonpositive", "float32"))

    def warm(job):
        operation, dtype = job
        with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True):
            if operation == "nonpositive":
                from gpuwm.core.mynn_pbl_gpu import _nonpositive
                function = _nonpositive()
            else:
                function = getattr(cp, operation)
            for size in sizes:
                values = cp.ones(size, dtype=dtype)
                function(values)

    with ThreadPoolExecutor(min(len(jobs), kernel_plan._workers()),
                            thread_name_prefix="column-reductions") as pool:
        list(pool.map(warm, jobs))


def warm_static(experiment, *, say=None, preparation=False, forecast=False) -> dict:
    """Compile and CACHE ON DISK every module of the static plan, on every core.

    For a process that exists only to warm a kernel cache (the ``gpuwm go``
    parent's spawn during its fetch and preparation, ``python -m
    gpuwm.kernel_prewarm CONFIG``): the plan pool compiles the static
    captures side by side, then each loader runs for real, one after
    another, served from the pool's table, and writes its own disk entry
    (CuPy's cache or the direct-NVRTC store) exactly as a forecast's loader
    would.  The forecast that follows finds every image on disk.

    ``preparation`` also executes the preparation backends on tiny synthetic
    snapshots in independent threads. The dedicated ``main`` enables this
    by default; callers warming only forecast profiles retain their scope.
    ``forecast`` also steps the run's own configuration on a synthetic
    column (:func:`warm_forecast_columns`) after the selected static variants
    load. Its independent generated reductions run concurrently; first-use
    compiler requests keep using the CPU process pool throughout this phase.
    """

    from gpuwm import kernel_plan
    from concurrent.futures import ThreadPoolExecutor

    began = time.perf_counter()
    receipt: dict = {"modules": 0, "loaded": 0, "errors": {}, "plan": None}
    if not kernel_plan.install(tag_text=plan_tag(experiment)):
        receipt["wall_seconds"] = round(time.perf_counter() - began, 3)
        return receipt
    record = kernel_plan.start()
    # Real CuPy-generated preparation kernels are not RawModule captures.
    # Execute their independent tiny preparations beside the static compile
    # pool, before fetch finishes and root_initialize opens its cold cache.
    prep_pool = None
    prep_jobs = {}
    if preparation:
        from gpuwm.ingest.preparation_prewarm import preparation_jobs, warm_preparation_job

        jobs = preparation_jobs(experiment)
        prep_pool = ThreadPoolExecutor(min(len(jobs), kernel_plan._workers()),
                                       thread_name_prefix="preparation-prewarm")
        prep_jobs = {label: prep_pool.submit(warm_preparation_job, options)
                     for label, options in jobs}
    receipt["modules"] = submit_static(experiment)
    jobs_in_order = static_jobs(experiment)
    receipt["static_jobs"] = len(jobs_in_order)
    for label, factory, arguments in jobs_in_order:
        started = time.perf_counter()
        try:
            factory(*arguments)
            receipt["loaded"] += 1
        except Exception as error:  # noqa: BLE001 - a module that refuses is the run's to report
            receipt["errors"][label] = f"{type(error).__name__}: {error}"
        if say is not None:
            say(f"kernel-prewarm: {label} {time.perf_counter() - started:.1f} s")
    if prep_pool is not None:
        receipt["preparation"] = {"groups": len(prep_jobs), "loaded": 0}
        for label, future in prep_jobs.items():
            try:
                future.result()
                receipt["preparation"]["loaded"] += 1
            except Exception as error:  # noqa: BLE001 - preparation owns refusals
                receipt["errors"][label] = f"{type(error).__name__}: {error}"
            if say is not None:
                say(f"kernel-prewarm: {label} {'failed' if label in receipt['errors'] else 'cached'}")
        prep_pool.shutdown(wait=True)
    if forecast:
        # All selected translation units are now available. Starting this
        # execution before their captures registered let foreground first
        # use steal the expensive compile from the parallel plan.
        column_started = time.perf_counter()
        try:
            receipt["forecast_columns"] = warm_forecast_columns(experiment)
        except Exception as error:  # noqa: BLE001 - the forecast owns refusals
            receipt["errors"]["forecast:column"] = f"{type(error).__name__}: {error}"
        if say is not None:
            say(f"kernel-prewarm: forecast:column "
                f"{'failed' if 'forecast:column' in receipt['errors'] else 'cached'} "
                f"(waited {time.perf_counter() - column_started:.1f} s after the static loads)")
    # Dynamic array factories also use the process compiler: closing it after
    # static capture forced CuPy's first encounters back onto the main core.
    kernel_plan.close(wait=True)
    # Publish before declaring the box warm: a following process or pack
    # export must see completed images, not wait for this process's atexit.
    kernel_plan.save()
    receipt["plan"] = kernel_plan.receipt()
    receipt["first_step_ready"] = first_step_ready(
        receipt, preparation=preparation, forecast=forecast)
    receipt["wall_seconds"] = round(time.perf_counter() - began, 3)
    return receipt


def first_step_ready(receipt, *, preparation=False, forecast=False) -> bool:
    """Whether this warm left every kernel the first step needs on disk.

    True only when every static module loaded, every preparation group
    asked for ran, the forecast's own column stepped, and nothing errored.
    A pre-warm that misses any of them leaves that compile to the run.
    """

    if receipt.get("errors"):
        return False
    if not receipt.get("loaded") or receipt.get("loaded") != receipt.get("static_jobs"):
        return False
    if preparation:
        groups = receipt.get("preparation") or {}
        if not groups.get("groups") or groups.get("loaded") != groups.get("groups"):
            return False
    if forecast and not receipt.get("forecast_columns"):
        return False
    return True


def main(argv=None) -> int:
    """``python -m gpuwm.kernel_prewarm CONFIG [--physics-profile ID]``: warm CONFIG's kernels, print the receipt."""

    import argparse
    import json

    parser = argparse.ArgumentParser(prog="gpuwm.kernel_prewarm",
                                     description="Compile and cache a configuration's kernel set on every core.")
    parser.add_argument("config")
    # Accepted for symmetry with the forecast runner's command line; the
    # config is hash-bound to its physics, the profile only names it.
    parser.add_argument("--physics-profile", default=None)
    args = parser.parse_args(argv)
    from gpuwm.experiment import load_experiment
    from gpuwm.local_gpu import no_local_gpu

    if no_local_gpu():
        print(json.dumps({"skipped": "no local GPU"}))
        return 0
    experiment = load_experiment(args.config)
    receipt = warm_static(experiment, say=lambda line: print(line, flush=True),
                          preparation=True, forecast=True)
    print(json.dumps(receipt, sort_keys=True, default=str))
    return 1 if receipt["errors"] else 0


def start_plan(experiment) -> dict | None:
    """Compile this configuration's kernel set now, on every core.

    Two sources feed one pool as wide as the usable cores: the recorded plan
    (every request an earlier run of this configuration made) and the static
    plan (:func:`static_jobs`: the modules this configuration's selectors
    launch, captured through their own loaders by :func:`gpuwm.kernel_plan.
    ahead`).  The static plan is what a FIRST run has: before it, a fresh box
    compiled noah, smag2d (twice) and ysu one after another on one core inside
    step 1 (about 45 s on an RTX PRO 6000 box, 2026-10-17).  Either source
    leaves its images in the single-flight table the run's own requests read;
    nothing about a compile's inputs or cache entries changes.

    Idempotent: the first call in a process installs the funnel and starts
    the pool; later calls return the same record.  ``None`` when there is no
    local card, the plan is turned off (``GPUWM_KERNEL_PLAN=off``) or
    anything about it fails -- the run then compiles as it always did.
    """

    try:
        from gpuwm import kernel_plan
        from gpuwm.local_gpu import no_local_gpu

        if no_local_gpu():
            return None
        if not kernel_plan.install(tag_text=plan_tag(experiment)):
            return None
        record = kernel_plan.start()
        if record is not None:
            submitted = submit_static(experiment)
            if submitted:
                record["static_jobs"] = submitted
        kernel_plan.close()
        return record
    except Exception:  # noqa: BLE001 - nothing here may fail a run
        return None


def start_for(experiment) -> Prewarm | None:
    """Start the background compile for ``experiment``, or return ``None``.

    ``None`` when it is turned off, when no unit applies, when CuPy or a
    card is absent, or when the worker sees more than one card (the run's
    device is then not known to be device 0).
    """

    if os.environ.get(ENV, "1").strip().lower() in ("0", "false", "off", "no"):
        return None
    modules = modules_for(experiment)
    if not modules:
        return None
    try:
        from gpuwm.local_gpu import no_local_gpu

        if no_local_gpu():
            return None
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() != 1:
            return None
    except Exception:  # noqa: BLE001 - no card, no warm-up
        return None
    return Prewarm(modules).start()


__all__ = ["DIFFUSION_ROUTE_MODULES", "ENV", "FORECAST_TABLE_WAIT_S", "STATIC_TIERS", "Prewarm",
           "first_step_ready", "forecast_columns", "main", "modules_for", "plan_tag", "warm_forecast_columns",
           "single_domain_experiment", "start_for", "start_plan", "static_jobs", "static_modules",
           "submit_static", "warm_static"]


if __name__ == "__main__":
    raise SystemExit(main())
