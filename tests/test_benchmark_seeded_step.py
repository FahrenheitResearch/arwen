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


def test_seeded_step_microphysics_choices_include_mp8(monkeypatch):
    """A wheel user could select 0/1/6/10/18 and not the scheme the rest
    of the product had just promoted to first class; 8 is accepted now
    rather than rejected as an invalid choice."""
    import sys

    benchmark = importlib.import_module("tools.benchmark_seeded_step")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--microphysics", "8"])
    assert benchmark._arguments().microphysics == 8


def test_mp8_sizing_carries_its_uploaded_table_set(monkeypatch):
    """mp8 uploads a pinned 362 MiB table set that scales with nothing in
    the grid; reusing the moist cell-equivalent default for it would
    under-report a small run by more than the whole fixed allowance."""
    benchmark = importlib.import_module("tools.benchmark_seeded_step")
    moist = benchmark.estimate_run(_arguments(microphysics=10))
    thompson = benchmark.estimate_run(_arguments(microphysics=8))

    assert moist["estimated_microphysics_table_bytes"] == 0
    tables = thompson["estimated_microphysics_table_bytes"]
    assert tables > 300 * 1024**2
    assert (thompson["estimated_device_bytes"]
            - moist["estimated_device_bytes"]) == tables
    # Derived from the contract's own record inventory, not written down.
    from gpuwm.core.thompson_contract import (AUXILIARY_TABLE_RECORDS,
                                              GENERATED_TABLE_FILES)
    expected = sum(
        int(record.payload_bytes)
        for group in GENERATED_TABLE_FILES.values()
        for record in group) + sum(
        int(record.payload_bytes) for record in AUXILIARY_TABLE_RECORDS)
    assert tables == expected


def test_dry_and_kessler_sizing_is_unchanged_by_the_table_allowance():
    benchmark = importlib.import_module("tools.benchmark_seeded_step")
    for selector in (0, 1, 6, 18):
        sizing = benchmark.estimate_run(_arguments(microphysics=selector))
        assert sizing["estimated_microphysics_table_bytes"] == 0


def test_seeded_step_provenance_uses_the_shared_install_identity(monkeypatch):
    """`git rev-parse` under site-packages exits 128 for every wheel, and
    it ran AFTER the timed lanes -- so a field run's whole measurement was
    thrown away for a provenance line.  One shared resolver now answers,
    the same one the real HRRR benchmark binds."""
    benchmark = importlib.import_module("tools.benchmark_seeded_step")
    wheel = {
        "identity_source": "installed-wheel-record",
        "git_commit": None, "git_tree": None, "git_status_short": None,
        "distribution_manifest_sha256": None,
        "installed_wheel": {"distribution_name": "gpuwm",
                            "distribution_version": "1.5.0"},
    }
    monkeypatch.setattr("gpuwm.runtime_manifest.provenance",
                        lambda root, **kwargs: dict(wheel))
    identity = benchmark._source_identity()

    assert identity["identity_source"] == "installed-wheel-record"
    assert identity["installed_wheel"]["distribution_version"] == "1.5.0"
    # The v1 key survives, honestly empty rather than absent or invented.
    assert identity["commit"] is None


def test_seeded_step_provenance_still_binds_a_real_checkout(monkeypatch):
    benchmark = importlib.import_module("tools.benchmark_seeded_step")
    checkout = {
        "identity_source": "git", "git_commit": "b" * 40,
        "git_tree": "c" * 40, "git_status_short": [],
        "distribution_manifest_sha256": None, "installed_wheel": None,
    }
    monkeypatch.setattr("gpuwm.runtime_manifest.provenance",
                        lambda root, **kwargs: dict(checkout))
    identity = benchmark._source_identity()
    assert identity["commit"] == "b" * 40
    assert identity["git_commit"] == "b" * 40
