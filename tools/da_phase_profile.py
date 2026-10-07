"""Phase timing and memory for one DA analysis step, without editing it.

Measurement tooling only: wraps named callables (inclusive and exclusive
wall seconds, call counts, RSS on entry and exit) and samples the process
RSS and per-thread CPU from a side thread, then runs a module or script
under those wraps.  Nothing it wraps changes behaviour: each wrapper calls
the original with the same arguments and returns its result untouched.

    python -m tools.da_phase_profile --out prof.json [--cprofile prof.pstats] \
        [--targets default|<file>] -- -m tools.da_cycle_prepared <args...>

Targets name the attribute where the CALLER looks it up (``module:attr``),
because ``from x import f`` binds at import time.  ``default`` covers the
analysis seam of tools/da_cycle_prepared.py and gpuwm/da/radar_assimilation.py
and the solve internals of gpuwm/da/letkf.py.
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

DEFAULT_TARGETS = (
    # obs read and QC
    "gpuwm.da.radar_assimilation:read_document",
    "gpuwm.da.radar_assimilation:_thinned_velocity_document",
    "gpuwm.da.radar_assimilation:_thinned_reflectivity_document",
    "gpuwm.da.radar_assimilation:_thinned_clear_air_document",
    "gpuwm.da.radar_assimilation:validate_analysis_fields",
    "gpuwm.da.obs_surface:surface_to_gridded_obs",
    # member state in and out
    "gpuwm.da.radar_assimilation:read_checkpoint_state",
    "numpy:savez",
    "numpy:savez_compressed",
    # forward operators
    "gpuwm.da.radar_assimilation:member_earth_winds",
    "gpuwm.da.radar_assimilation:_member_fall_speed",
    "gpuwm.da.radar_assimilation:simulated_radial_velocity",
    "gpuwm.da.radar_assimilation:radar_grid_to_gridded_obs",
    "gpuwm.da.radar_assimilation:innovation_summary",
    # gates around the filter
    "gpuwm.da.radar_assimilation:velocity_dispersion",
    "gpuwm.da.radar_assimilation:withhold",
    # the filter
    "gpuwm.da.radar_assimilation:_execute_analysis",
    "gpuwm.da.radar_assimilation:_analysis_attempt",
    "gpuwm.da.radar_assimilation:analyze",
    "gpuwm.da.letkf:_validate_prior",
    "gpuwm.da.letkf:_validate_obs",
    "gpuwm.da.letkf:_batch_reach_box",
    "gpuwm.da.letkf:reachable_slots_estimate",
    "gpuwm.da.letkf:_eigendecompose",
    "gpuwm.da.letkf:_finish",
    # after the filter
    "gpuwm.da.radar_assimilation:apply_positivity",
    "gpuwm.da.radar_assimilation:verify_non_negative",
    "gpuwm.da.radar_assimilation:_precip_analysis",
    "gpuwm.da.radar_assimilation:_saturation_bound",
    "gpuwm.da.radar_assimilation:assimilate_radar_grid",
    "tools.da_cycle_prepared:merge_hotstart_increments",
    "tools.da_cycle_prepared:plan_radar_assimilation",
)


def rss_bytes() -> int:
    """Current resident set size, from /proc on Linux, else 0."""
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return 0


class PhaseProfiler:
    """Inclusive/exclusive wall time per wrapped callable, plus RSS."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self._stack = threading.local()
        self.restore: list = []
        self.missing: list[str] = []

    def _frames(self):
        if not hasattr(self._stack, "frames"):
            self._stack.frames = []
        return self._stack.frames

    def wrap(self, name, fn):
        row = self.rows.setdefault(name, {"calls": 0, "inclusive_s": 0.0,
                                          "exclusive_s": 0.0,
                                          "rss_growth_max_bytes": 0,
                                          "rss_exit_max_bytes": 0})

        @functools.wraps(fn)
        def timed(*args, **kwargs):
            frames = self._frames()
            frames.append(0.0)
            rss0 = rss_bytes()
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                children = frames.pop()
                if frames:
                    frames[-1] += dt
                rss1 = rss_bytes()
                row["calls"] += 1
                row["inclusive_s"] += dt
                row["exclusive_s"] += dt - children
                row["rss_growth_max_bytes"] = max(row["rss_growth_max_bytes"],
                                                  rss1 - rss0)
                row["rss_exit_max_bytes"] = max(row["rss_exit_max_bytes"], rss1)

        for attr in ("supports_host_staging",):
            if hasattr(fn, attr):
                setattr(timed, attr, getattr(fn, attr))
        timed.__wrapped_by_phase_profile__ = True
        return timed

    def install(self, targets):
        for target in targets:
            module_name, attr = target.split(":")
            try:
                module = importlib.import_module(module_name)
                original = getattr(module, attr)
            except (ImportError, AttributeError):
                self.missing.append(target)
                continue
            if getattr(original, "__wrapped_by_phase_profile__", False):
                continue
            setattr(module, attr, self.wrap(target, original))
            self.restore.append((module, attr, original))

    def uninstall(self):
        for module, attr, original in reversed(self.restore):
            setattr(module, attr, original)
        self.restore.clear()


class Sampler(threading.Thread):
    """RSS and process CPU every ``period`` seconds, plus the peak."""

    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period = period
        self.samples: list[tuple[float, int, float]] = []
        self.peak = 0
        self._stop_event = threading.Event()
        self.t0 = time.perf_counter()

    def run(self):
        while not self._stop_event.is_set():
            rss = rss_bytes()
            self.peak = max(self.peak, rss)
            self.samples.append((time.perf_counter() - self.t0, rss,
                                 time.process_time()))
            self._stop_event.wait(self.period)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)


def load_targets(spec: str):
    if spec == "default":
        return list(DEFAULT_TARGETS)
    lines = Path(spec).read_text().splitlines()
    return [line.strip() for line in lines
            if line.strip() and not line.startswith("#")]


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cprofile", type=Path, default=None)
    parser.add_argument("--targets", default="default")
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        parser.error("name what to run: -- -m module args... or -- script.py args...")
    if rest[0] == "-m":
        kind, target, target_args = "module", rest[1], rest[2:]
    else:
        kind, target, target_args = "path", rest[0], rest[1:]

    profiler = PhaseProfiler()
    profiler.install(load_targets(args.targets))
    sampler = Sampler(args.sample_seconds)
    sampler.start()
    prof = None
    if args.cprofile is not None:
        import cProfile
        prof = cProfile.Profile()
    saved_argv = sys.argv
    sys.argv = [target, *target_args]
    exit_code = 0
    t0 = time.perf_counter()
    try:
        if prof is not None:
            prof.enable()
        if kind == "module":
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(target, run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        if prof is not None:
            prof.disable()
            prof.dump_stats(str(args.cprofile))
        wall = time.perf_counter() - t0
        sys.argv = saved_argv
        sampler.stop()
        profiler.uninstall()
        report = {
            "schema": "gpuwm-da.phase-profile.v1",
            "command": rest,
            "exit_code": exit_code,
            "wall_seconds": wall,
            "peak_rss_bytes": sampler.peak,
            "missing_targets": profiler.missing,
            "phases": dict(sorted(profiler.rows.items(),
                                  key=lambda kv: -kv[1]["exclusive_s"])),
            "rss_samples": sampler.samples,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
