"""The reviewed estimate uses one device observation and the real time window."""
from datetime import timedelta
import dataclasses
import tomllib

import pytest

from gpuwm import runplan
from gpuwm.core import preflight, rrtmg_lw
from gpuwm.starter_template import render_tables
from test_case_data import make_case_toml


@pytest.fixture
def observed_device(monkeypatch):
    # A recorded-shaped observation, supplied at the subprocess boundary.
    # Opening CUDA in the estimating process is forbidden by this tripwire.
    def no_in_process_device():
        pytest.fail("the CPU estimate queried CUDA for radiation workspace width")

    monkeypatch.setattr(rrtmg_lw, "_device_resident_threads", no_in_process_device)
    profile = preflight.DeviceLocalMemoryProfile(
        "fixture 68-SM device", 68, 1536, bare_context_bytes=174 * 1024 ** 2)
    probe = {"free_bytes": 8 * 1024 ** 3, "total_bytes": 10 * 1024 ** 3,
             "profile": dataclasses.asdict(profile)}
    calls = []
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess",
                        lambda: calls.append("probe") or probe)
    return profile, calls


def _case(tmp_path):
    path = make_case_toml(tmp_path)
    raw = tomllib.loads(path.read_text())
    raw["experiment"]["run_seconds"] = 6 * 3600
    raw["shared"].update(ra_lw_physics=4, ra_sw_physics=4,
                         wrf_rrtmg_compatibility="wrf-rrtmg-4-4-legacy-v1",
                         ra_rrtmg_variant="rrtmg_legacy")
    raw["domain"][0].update(nx=120, ny=100, specified=True)
    raw["case_data"]["forcing_interval_s"] = 10800
    path.write_text(render_tables(raw))
    return path


def _plan(tmp_path, path, *, inline=False, route="experiment"):
    return runplan.build_plan(
        {"schema": runplan.PLAN_SCHEMA, "name": "device-estimate", "route": route,
         "config": ({"inline": path.read_text()} if inline else {"path": str(path)}),
         "output_root": str(tmp_path / "run")},
        source="plan.json", base_dir=tmp_path, sha256="0" * 64)


@pytest.mark.parametrize("inline", [False, True])
def test_review_uses_measured_device_and_all_retained_intervals(
        tmp_path, monkeypatch, observed_device, inline):
    from gpuwm.ingest import grib

    path = _case(tmp_path)
    plan = _plan(tmp_path, path, inline=inline)
    _, exp, _ = runplan.resolve_plan(plan, require_inputs=False)
    # Twelve supplied hours remain in memory although only six are forecast.
    times = tuple(exp.start_time + timedelta(hours=3 * index) for index in range(5))
    monkeypatch.setattr(grib, "inspect_era5_forcing_times", lambda *args: times)
    profile, calls = observed_device
    expected = preflight.estimate_phases(
        exp, source=None, forcing_interval_seconds=10800, forcing_intervals=4,
        profile=profile, vram_gib=10).forecast
    short_window = preflight.estimate_phases(
        exp, source=None, forcing_interval_seconds=10800,
        profile=profile, vram_gib=10).forecast
    reference = preflight.estimate_phases(
        exp, source=None, forcing_interval_seconds=10800, forcing_intervals=4).forecast

    result = runplan.estimate_plan(plan)["vram"]

    assert calls == ["probe"]
    assert result["peak_envelope_bytes"] == expected.peak_envelope_bytes
    assert expected.peak_envelope_bytes > short_window.peak_envelope_bytes
    assert expected.peak_envelope_bytes < reference.peak_envelope_bytes
    assert result["device_profile"] == dataclasses.asdict(profile)
    assert result["device_basis"] == "measured local device"
    assert result["forcing_interval_seconds"] == 10800
    assert result["retained_forcing_intervals"] == 4
    assert result["phase_scope"] == "forecast"
    assert not plan.run_dir.exists()


def test_unmeasured_device_keeps_conservative_workspace_without_cuda(
        tmp_path, monkeypatch, observed_device):
    path = _case(tmp_path)
    # A planning config whose download has not happened has an unknown count.
    (tmp_path / "forcing/era5_a.grb").unlink()
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", lambda: None)
    result = runplan.estimate_plan(_plan(tmp_path, path))["vram"]
    assert result["device_profile"] is None
    assert result["device_basis"].startswith("conservative reference")
    assert result["retained_forcing_intervals"] is None
    assert result["peak_envelope_bytes"] > 0


def test_native_plan_prices_declared_fetch_cadence(tmp_path, observed_device):
    path = _case(tmp_path)
    raw = tomllib.loads(path.read_text())
    raw.pop("case_data")
    raw["fetch"] = {"source": "hrrr", "cycle": "1999-05-03T12", "hours": 6,
                    "cadence": 1}
    path.write_text(render_tables(raw))
    plan = _plan(tmp_path, path, route="prepared")
    _, exp, _ = runplan.resolve_plan(plan, require_inputs=False)
    profile, _ = observed_device
    expected = preflight.estimate_experiment(
        exp, forcing_interval_seconds=3600, profile=profile, vram_gib=10)
    result = runplan.estimate_plan(plan)["vram"]
    assert result["forcing_interval_seconds"] == 3600
    assert result["retained_forcing_intervals"] is None
    assert result["peak_envelope_bytes"] == expected.peak_envelope_bytes


def test_explicit_radiation_capacity_matches_the_runtime_chunk_width(monkeypatch):
    from gpuwm.core.rrtmg_legacy import legacy_radiation_vram_bytes

    def no_device():
        pytest.fail("explicit host pricing queried CUDA")

    monkeypatch.setattr(rrtmg_lw, "_device_resident_threads", no_device)
    args = dict(ncol=12000, nz=65, p_top=5000.0)
    explicit = legacy_radiation_vram_bytes(**args, resident_threads=68 * 1536)
    unknown = legacy_radiation_vram_bytes(**args, resident_threads=0)
    assert explicit < unknown
    monkeypatch.setattr(rrtmg_lw, "_device_resident_threads", lambda: 68 * 1536)
    assert explicit == legacy_radiation_vram_bytes(**args)
