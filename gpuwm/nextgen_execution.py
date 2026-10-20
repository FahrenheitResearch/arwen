"""Resident NextGen launch sequence and off-path CUDA timing.

Module owners bind kernels and device arguments once. Their callbacks enqueue
onto the caller's current CUDA stream without allocation, copies or waits.
Production enqueue inserts no instrumentation. Timing uses a separate run and
never captures CUDA graphs. No hydrologic numerics live in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class Stage:
    name: str
    enqueue: Callable[[], None]
    parallelism: str


class ResidentSequence:
    """One physical hydro interval, in dependency order, on the parent stream.

    Construction never advances state. ``keepalive`` owns every kernel, device
    workspace and argument binding needed by the callbacks. Device-side clocks
    and schedules belong to those callbacks; Python never reads them per step.
    """

    def __init__(self, stages: Sequence[Stage], *, step_seconds: float,
                 keepalive: Any, metadata: Mapping[str, Any],
                 enqueue_interval: Callable[[], None] | None = None):
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("a hydro interval needs at least one stage")
        names = [stage.name for stage in self.stages]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("timing stages need unique nonempty names")
        if not math.isfinite(step_seconds) or not step_seconds > 0:
            raise ValueError("the physical hydro interval must be positive")
        if any(not stage.parallelism for stage in self.stages):
            raise ValueError("each stage must name its parallelism")
        self.step_seconds = float(step_seconds)
        self.keepalive = keepalive
        self.metadata = json.loads(json.dumps(dict(metadata), allow_nan=False))
        # A driver's complete call preserves orchestration/continuation clocks.
        # The ordered callbacks must enqueue those same kernels for profiling.
        self._enqueue_interval = enqueue_interval or self._enqueue_stages

    def enqueue(self) -> None:
        self._enqueue_interval()

    def _enqueue_stages(self) -> None:
        for stage in self.stages:
            stage.enqueue()


def bind_driver(driver: Any, *, parallelism: Mapping[str, str],
                metadata: Mapping[str, Any], parent_lsm: Stage | None = None
                ) -> ResidentSequence:
    """Prepare timing around the public resident-driver ABI, once at setup.

    Coupled factories supply the actual parent LSM callback. Standalone factories
    bind their runoff modules in the driver's after stages. The timed batch calls
    the complete driver interval, preserving continuation bookkeeping; the fresh
    event profile enqueues the same numerical stages individually.
    """
    before, after = driver.before_stages, driver.after_stages
    dt = driver.interval_seconds
    def bind(stage):
        # Owners may expose the exact prepared launch order for off-path events.
        # Production still calls the complete driver, without these events.
        profile = getattr(stage.component, "profile_stages", None)
        if profile is not None:
            return [Stage(f"{stage.name}/{launch.name}",
                          partial(launch.launch, stream=None), launch.parallelism)
                    for launch in profile]
        return [Stage(stage.name, partial(stage.launch, dt, stream=None),
                      parallelism[stage.name])]

    bound = [launch for stage in before for launch in bind(stage)]
    if parent_lsm is not None:
        bound.append(parent_lsm)
    bound.extend(launch for stage in after for launch in bind(stage))
    details = dict(metadata)
    mode = details.setdefault("mode", "coupled" if parent_lsm else "standalone")
    if mode not in ("coupled", "standalone") or (mode == "coupled") != (parent_lsm is not None):
        raise ValueError("coupled timing needs its actual parent LSM callback")
    details["driver_identity"] = driver.restart_identity

    def whole_interval():
        driver.before_lsm()
        if parent_lsm is not None:
            parent_lsm.enqueue()
        driver.advance()

    return ResidentSequence(bound, step_seconds=dt, keepalive=(driver, parent_lsm),
                            metadata=details, enqueue_interval=whole_interval)


def prepare_standalone(settings: Mapping[str, Any], *, device_id: int = 0
                       ) -> ResidentSequence:
    """Bind executable owner factory rows through the actual resident driver.

    Settings contain prepared stage rows, including any preparation-only
    reservoir resources. They may be expanded by the configuration resolver
    beforehand. No case adapter, hydraulic defaults or synthetic state is added.
    Coupled timing continues to use bind_driver with the actual parent LSM.
    """
    from gpuwm import numerics

    prepared = json.loads(json.dumps(dict(settings), allow_nan=False))
    if prepared.get("numerics", numerics.mode()) != numerics.mode():
        raise ValueError("prepared settings differ from selected process numerics")
    prepared["numerics"] = numerics.mode()
    from gpuwm.hydro.assembly import prepare

    driver = prepare(settings=prepared, state=None, physics=None,
                     device_id=device_id, mode="standalone")
    stages = (*driver.before_stages, *driver.after_stages)
    return bind_driver(
        driver, parallelism={stage.name: stage.parallelism for stage in stages},
        metadata={"numerics": numerics.mode(), "mode": "standalone",
                  "scope": prepared.get("timing_scope", "prepared standalone stages")})


def measure(factory: Callable[[], ResidentSequence], *, steps: int,
            warmup: int = 1) -> dict:
    """Complete a plain batch; profile one interval on independently fresh state.

    Factory construction, compilation and warmup are outside the timed batch.
    Warmup advances state and is reported explicitly. The two stream waits are
    batch boundaries, never module/substep boundaries. Event intervals from the
    separate fresh profile include host enqueue gaps and are not throughput.
    Callers set strict/fast mode in their factory and report it in metadata.
    """
    if steps < 1 or warmup < 0:
        raise ValueError("steps must be positive and warmup nonnegative")
    import cupy as cp

    stream = cp.cuda.get_current_stream()
    device = cp.cuda.runtime.getDevice()
    properties = cp.cuda.runtime.getDeviceProperties(device)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    sequence = factory()
    if cp.cuda.get_current_stream().ptr != stream.ptr:
        raise ValueError("factory changed the caller's CUDA stream")
    for _ in range(warmup):
        sequence.enqueue()
    stream.synchronize()
    start = time.perf_counter_ns()
    for _ in range(steps):
        sequence.enqueue()
    stream.synchronize()
    wall_ns = time.perf_counter_ns() - start

    # Fresh state prevents profiling a later (possibly dry or converged) period
    # and silently attributing that period's stage times to the timed batch.
    profile = factory()
    if cp.cuda.get_current_stream().ptr != stream.ptr:
        raise ValueError("profile factory changed the caller's CUDA stream")
    if (profile.step_seconds != sequence.step_seconds
            or [(s.name, s.parallelism) for s in profile.stages]
            != [(s.name, s.parallelism) for s in sequence.stages]
            or profile.metadata != sequence.metadata):
        raise ValueError("factory changed its physical interval, stages or metadata")
    for _ in range(warmup):
        profile.enqueue()
    events = [cp.cuda.Event() for _ in range(len(profile.stages) + 1)]
    events[0].record(stream)
    for stage, event in zip(profile.stages, events[1:]):
        stage.enqueue()
        event.record(stream)
    events[-1].synchronize()
    stage_rows = [
        {"name": stage.name, "parallelism": stage.parallelism,
         "elapsed_ms": cp.cuda.get_elapsed_time(events[i], events[i + 1])}
        for i, stage in enumerate(profile.stages)
    ]
    return {
        "schema": "gpuwm.nextgen-speed.v1",
        "device": {"ordinal": device, "name": name},
        "metadata": sequence.metadata,
        "physical_step_seconds": sequence.step_seconds,
        "batch": {
            "steps": steps, "warmup_steps": warmup,
            "warmup_state_retained": True,
            "simulated_seconds": steps * sequence.step_seconds,
            "wall_ns": wall_ns, "ms_per_step": wall_ns / steps / 1e6,
            "stream_waits": 2, "events": 0, "cuda_graphs": False,
            "first_enqueue_included": warmup == 0,
        },
        "profile": {
            "fresh_factory": True, "warmup_steps": warmup,
            "physical_steps": 1, "final_event_waits": 1,
            "includes_enqueue_gaps": True,
            "stages": stage_rows,
            "elapsed_ms": cp.cuda.get_elapsed_time(events[0], events[-1]),
        },
    }
