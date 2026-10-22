"""Table-composed preparation of GPU NextGen components.

Rows select factories and their options. Rust prepares hydrofabric/parameters;
the factory binds kernels and pointer-stable state before this function returns.
"""
from __future__ import annotations

import importlib
from types import MappingProxyType

from .driver import ResidentHydroDriver, Stage


def prebuild_kernels(rows, *, settings, device_id):
    """Start every stage's slow kernel compile on a background thread before any stage prepares.

    A factory ``F`` may publish ``F_prebuild(settings=, options=, device_id=)``
    beside it. Without this, a cold kernel cache compiled the Noah-OWP physics
    on the one preparing thread while every other core and the card waited
    (about 56 s at one core per cold door run; the hydro-world VERIFY2
    calibration finding of 2026-10-18).
    """
    for row in rows:
        module, separator, entry = str(row.get("factory", "")).partition(":")
        if not separator or not module.startswith("gpuwm."):
            continue  # prepare() refuses the row with its own message
        hook = getattr(importlib.import_module(module), entry + "_prebuild", None)
        if hook is not None:
            hook(settings=settings, options=row.get("options", {}), device_id=device_id)


def _whole_calls(period, lsm_interval):
    """LSM calls in one stage period, or None when the period is not a whole number of them."""
    if not (lsm_interval > 0 and period > 0):
        return None
    ratio = period / lsm_interval
    whole = round(ratio)
    return whole if whole >= 1 and abs(ratio - whole) <= 1e-9 * max(1.0, ratio) else None


def prepare(*, settings, state, physics, device_id, mode):
    rows = settings.get("stages", ())
    if not rows:
        raise ValueError("NextGen realization has no prepared stage rows; select a runoff/routing realization")
    from .memory import construction_plan, admit_plan, verify_driver
    memory_admission = admit_plan(construction_plan(
        settings, physics=physics, state=state, mode=mode), device_id=device_id)
    prebuild_kernels(rows, settings=settings, device_id=device_id)
    before, after, resources, components, status_fatal_masks = [], [], [], {}, {}
    interval = physics.bldt_seconds if mode == "coupled" else settings["interval_seconds"]
    stage_clocks = False
    for row in rows:
        name = row["name"]
        if name in components:
            raise ValueError(f"duplicate NextGen stage {name!r}")
        module, separator, entry = row["factory"].partition(":")
        if not separator or not module.startswith("gpuwm."):
            raise ValueError(f"NextGen stage {name} needs a gpuwm module:factory")
        phase = row.get("phase", "after")
        if phase not in ("before", "after", "prepare"):
            raise ValueError(f"NextGen stage {name} has unknown phase {phase!r}")
        factory = getattr(importlib.import_module(module), entry)
        component = factory(settings=settings, options=row.get("options", {}),
                            state=state, physics=physics, device_id=device_id,
                            mode=mode, components=MappingProxyType(components))
        components[name] = component
        for key, mask in row.get("status_fatal_masks", {}).items():
            status_fatal_masks[f"{name}/{key}"] = mask
        # Reservoir state used by level-ordered reach hooks must be prepared
        # and checkpointed without also advancing outside those hooks.
        if phase == "prepare":
            resources.append((name, component))
            continue
        cadence = row.get("cadence_steps", 1)
        if row.get("interval_seconds") is not None and phase == "after":
            # The cadence is counted in the forecast's ACTUAL LSM calls, not the
            # prepared ones: go may lower the root step after preparation (the
            # terrain clock's CFL bound), and bldt follows it. A period that is
            # no whole number of LSM calls runs on its own clock instead
            # (stage_clocks). Breakage it prevents (2.9.4 WOOF readiness g3/u6,
            # 2026-10-22): d01 prepared at 60 s ran 30 s steps, the rows kept
            # cadence 300/60 = 5, and attach died ("stage interval must span its
            # whole number of parent LSM calls"); 2.8.7 finished that run.
            cadence = _whole_calls(float(row["interval_seconds"]), float(interval))
            if cadence is None:
                cadence, stage_clocks = 1, True
        stage = Stage(name, component, getattr(component, row.get("launch_method", "advance")),
                      parallelism=row.get("parallelism", "GPU"),
                      cadence_steps=cadence,
                      interval_seconds=row.get("interval_seconds"))
        (before if phase == "before" else after).append(stage)
    output = channel_output = None
    launched = {id(stage.component) for stage in (*before, *after)}
    for component in components.values():
        provider = getattr(component, "reach_output", None) if id(component) in launched else None
        if provider is not None:
            if output is not None:
                raise ValueError("multiple routing stages claim the reach output; select one in the realization")
            output = provider()
        provider = getattr(component, "channel_output", None) if id(component) in launched else None
        sample = provider() if provider is not None else None
        if sample is not None:
            if channel_output is not None:
                raise ValueError("multiple routing stages claim channel output; select one in the realization")
            channel_output = sample
    driver = ResidentHydroDriver(device_id=device_id, interval_seconds=interval,
                               identity={"realization": settings.get("realization"), "mode": mode,
                                         "realization_identity": settings.get("realization_identity"),
                                         "numerics": settings.get("numerics", "fast"),
                                         "rows": rows}, before=before, after=after,
                               resources=resources,
                               reach_output=output, channel_output=channel_output,
                               status_fatal_masks=status_fatal_masks,
                               after_schedule=settings.get("after_schedule"),
                               stage_clocks=stage_clocks)
    driver.memory_admission = memory_admission
    verify_driver(memory_admission, driver)
    return driver
