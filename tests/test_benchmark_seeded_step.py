from argparse import Namespace
import importlib


def _arguments(**overrides):
    values = {
        "nx": 64,
        "ny": 48,
        "nz": 32,
        "microphysics": 0,
        "warmup_steps": 2,
        "steps": 5,
        "profile_steps": 5,
        "profiler_capture": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_seeded_step_estimate_is_deterministic_and_conservative():
    benchmark = importlib.import_module("tools.benchmark_seeded_step")
    result = benchmark.estimate_run(_arguments())

    assert result["cells"] == 64 * 48 * 32
    assert result["estimated_device_bytes"] > result["cells"] * 4
    assert result["estimated_trace_bytes"] == 0
    assert result["total_executed_steps"] == 14


def test_seeded_step_trace_estimate_scales_with_profile_steps():
    benchmark = importlib.import_module("tools.benchmark_seeded_step")
    result = benchmark.estimate_run(
        _arguments(profile_steps=3, profiler_capture=True))

    assert result["estimated_trace_bytes"] == 3 * 24 * 1024**2
    assert result["total_executed_steps"] == 12
