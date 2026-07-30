"""Native HRRR single-domain benchmark controller regressions."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

import tools.hrrr_single_domain_benchmark as hrrr_runner
from gpuwm.static.lambert import LambertGrid
from gpuwm.experiment import VerticalConfig
from gpuwm.ingest.hrrr_target import HrrrTargetDomain
from gpuwm.ingest.prepared_cache import prepared_cache_identity
from gpuwm.physics_compat import (
    MORRISON_PROFILE_ID,
    NSSL2_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    WRF_RRTMG_TO_RTE_RRTMGP,
    WSM6_PROFILE_ID,
)
from tools.hrrr_single_domain_benchmark import (
    _PreprocessWorkerBudget, _boundary_mapping_targets, _experiment,
    _initial_hrrr_microphysics_receipt,
    main as hrrr_runner_main,
    _map_boundary_snapshot, _map_snapshot, _parse_args,
    _partition_preprocess_worker_budget,
    _validate_native_hrrr_physics_profile,
    runner_capabilities,
)
from tools.prepare_hrrr_wrf import _validated_worker_receipts


def _required_args():
    return [
        "--bridge", "bridge",
        "--valid-time", "2026-07-20_06:00:00",
        "--manifest-sha256", "manifest-sha256",
        "--static-cache", "static.npz",
        "--static-receipt", "static.json",
        "--namelist-input", "namelist.input",
        "--run-seconds", "3600",
        "--outdir", "output",
    ]


def test_hrrr_runner_capability_query_is_side_effect_free_without_run_args(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert hrrr_runner_main(["--show-capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == runner_capabilities()
    assert payload["schema"] == "gpuwm-runner-capabilities-v1"
    assert payload["supported_sources"] == ["hrrr"]
    assert payload["physics_profile_ids"] == [
        WSM6_PROFILE_ID, THOMPSON_PROFILE_ID, MORRISON_PROFILE_ID,
        NSSL2_PROFILE_ID]
    assert payload["report_schema"] == "gpuwm-native-hrrr-benchmark-v2"
    assert payload["window"]["maximum_source_forecast_hour"] == 48
    assert payload["window"]["maximum_run_seconds"] == 172_800
    assert payload["window"]["run_seconds"]["whole_hour_required"] is False
    assert payload["output"]["io_modes"] == ["none", "history"]
    assert payload["output"]["configurable_cadence"] is True
    cadence = payload["output"]["history_interval_seconds"]
    assert cadence["required_for_mode"] == "history"
    assert cadence["accepted_for_prepare_only_cache_identity"] is True
    assert cadence["must_be_whole_model_steps"] is True
    assert cadence["must_evenly_divide_run"] is False
    assert cadence["last_scheduled_frame_may_precede_run_end"] is True
    assert payload["readiness"] \
        == "FORECAST_IMPLEMENTATION_PRESENT_RUNTIME_PREFLIGHT_REQUIRED"
    assert payload["modes"]["prepare-only"]["available"] is True
    assert payload["modes"]["forecast"]["available"] is True
    assert payload["standalone_rw_wps_wheel"] == {
        "runner_included": True,
        "prepare_only_available": True,
        "forecast_executor_included": False,
    }
    thompson = payload["physics_profiles"][THOMPSON_PROFILE_ID]
    assert thompson["readiness"] == "MODEL_VALIDATED_EXPERIMENTAL_RUNTIME"
    assert thompson["runtime_guard"]["environment"] \
        == "GPUWM_EXPERIMENTAL_THOMPSON_MP8"
    assert thompson["explicit_expert_consent_required"] is False
    assert thompson["table_root"]["environment"] \
        == "GPUWM_THOMPSON_TABLE_ROOT"
    assert len(thompson["table_authority"]["assets"]) == 4
    assert payload["physics_profiles"][MORRISON_PROFILE_ID] == {
        "selector": 10,
        "readiness": "MODEL_VALIDATED_RUNTIME_PROFILE",
        "explicit_expert_consent_required": False,
        "runtime_guards": [],
    }
    nssl2 = payload["physics_profiles"][NSSL2_PROFILE_ID]
    assert nssl2["selector"] == 18
    assert nssl2["readiness"] == "VALIDATION_CANDIDATE"
    assert nssl2["explicit_expert_consent_required"] is False
    assert nssl2["resolved_fixed_preset"] is True
    assert payload["capability_query"]["requires_cupy"] is False
    assert list(tmp_path.iterdir()) == []


def test_hrrr_history_consumes_native_reflectivity_for_every_stashed_scheme():
    for mp_physics in (1, 6, 8, 10, 18):
        assert hrrr_runner._history_uses_native_reflectivity(
            ticks=1, has_moisture=True, mp_physics=mp_physics)
    assert not hrrr_runner._history_uses_native_reflectivity(
        ticks=0, has_moisture=True, mp_physics=18)
    assert not hrrr_runner._history_uses_native_reflectivity(
        ticks=1, has_moisture=False, mp_physics=18)


def test_hrrr_capabilities_fail_closed_when_forecast_executor_is_absent(
        monkeypatch):
    monkeypatch.setattr(
        hrrr_runner,
        "_missing_forecast_executor_modules",
        lambda: ["gpuwm.core.model", "gpuwm.io.wrfout"],
    )

    payload = hrrr_runner.runner_capabilities()

    assert payload["readiness"] == "PREPARATION_ONLY_FORECAST_EXECUTOR_OMITTED"
    assert payload["modes"]["prepare-only"]["available"] is True
    assert payload["modes"]["forecast"] == {
        "available": False,
        "availability_scope": "executor-module-presence-only",
        "launch_ready": None,
        "launch_readiness_check": (
            "validate CuPy, CUDA, GPU allocation, inputs, and profile guards "
            "before launch"),
        "requires_cupy": True,
        "requires_compatible_cuda_gpu": True,
        "missing_executor_modules": [
            "gpuwm.core.model", "gpuwm.io.wrfout"],
        "included_in_standalone_rw_wps_wheel": False,
        "unavailable_reason": (
            "standalone RW-WPS omits the GPUWM forecast executor"),
    }


def test_native_prepare_workers_defaults_to_validated_two_worker_path():
    assert _parse_args(_required_args()).prepare_workers == 2


def test_native_source_window_and_physics_profile_are_retained():
    args = _parse_args(_required_args() + [
        "--forecast-start-hour", "12",
        "--forecast-end-hour", "13",
        "--physics-profile", THOMPSON_PROFILE_ID,
    ])
    assert args.forecast_start_hour == 12
    assert args.forecast_end_hour == 13
    assert args.physics_profile == THOMPSON_PROFILE_ID


@pytest.mark.parametrize("workers", (1, 2, 4, 8))
def test_native_prepare_worker_override_is_retained(workers):
    args = _required_args() + ["--prepare-workers", str(workers)]
    assert _parse_args(args).prepare_workers == workers


def test_native_hrrr_cpu_preprocess_selector_is_retained():
    args = _parse_args(_required_args() + [
        "--preprocess-backend", "cpu",
        "--preprocess-workers", "8",
        "--cpu-preprocess-bridge", "libgpuwm_preprocess_cpu.so",
    ])
    assert args.preprocess_backend == "cpu"
    assert args.preprocess_workers == 8
    assert args.cpu_preprocess_bridge.name == "libgpuwm_preprocess_cpu.so"


def test_native_hrrr_rejects_cuda_workers():
    with pytest.raises(SystemExit):
        _parse_args(_required_args() + [
            "--preprocess-backend", "cuda",
            "--preprocess-workers", "8",
        ])


def test_native_preprocess_budget_is_partitioned_not_multiplied_per_job():
    budget = _PreprocessWorkerBudget(
        backend="cpu", requested_total=8, effective_total=8,
        requested_job_slots=3, future_job_count=3, clock_origin=0.0)
    assert _partition_preprocess_worker_budget(8, 3) == (3, 3, 2)
    assert budget.slot_workers == (3, 3, 2)
    assert [budget.allocation_for_hour(hour) for hour in (1, 2, 3)] == [
        (0, 3), (1, 3), (2, 2)]

    budget.record(
        forecast_hour=0, phase="full_domain_initialization", slot=None,
        native_workers=8, started=0.1, finished=1.0)
    budget.record(
        forecast_hour=1, phase="boundary_mapping", slot=0,
        native_workers=3, started=1.1, finished=1.2)
    budget.record(
        forecast_hour=1, phase="boundary_initialization", slot=0,
        native_workers=3, started=1.3, finished=5.0)
    budget.record(
        forecast_hour=2, phase="boundary_mapping", slot=1,
        native_workers=3, started=1.4, finished=2.0)
    budget.record(
        forecast_hour=2, phase="boundary_initialization", slot=1,
        native_workers=3, started=2.1, finished=5.0)
    budget.record(
        forecast_hour=3, phase="boundary_mapping", slot=2,
        native_workers=2, started=2.2, finished=2.4)
    budget.record(
        forecast_hour=3, phase="boundary_initialization", slot=2,
        native_workers=2, started=2.5, finished=5.0)

    receipt = budget.receipt()
    assert receipt["requested_total_native_workers"] == 8
    assert receipt["effective_total_native_workers"] == 8
    assert receipt["peak_active_native_workers"] == 8
    assert receipt["peak_active_preprocessing_jobs"] == 3
    assert len(receipt["effective_allocation_per_job"]) == 7
    assert receipt["pipeline_decoder_workers_included"] is False


def test_native_preprocess_budget_caps_concurrent_jobs_to_total_threads():
    budget = _PreprocessWorkerBudget(
        backend="cpu", requested_total=2, effective_total=2,
        requested_job_slots=8, future_job_count=4, clock_origin=0.0)
    assert budget.slot_workers == (1, 1)
    assert [budget.allocation_for_hour(hour) for hour in (1, 2, 3, 4)] == [
        (0, 1), (1, 1), (0, 1), (1, 1)]


def test_public_hrrr_receipt_keeps_decoder_and_native_budgets_separate():
    report = {
        "status": "PASS",
        "preparation": {
            "preprocess_backend": {"backend": "cpu", "workers": 8},
            "preprocess_worker_budget": {
                "schema": "gpuwm-preprocess-worker-budget-v1",
                "backend": "cpu",
                "applicable": True,
                "requested_total_native_workers": 8,
                "effective_total_native_workers": 8,
                "effective_allocation_per_job": [
                    {"effective_native_workers": 8},
                    {"effective_native_workers": 4},
                    {"effective_native_workers": 4},
                ],
                "peak_active_native_workers": 8,
                "pipeline_decoder_workers_included": False,
            },
        },
        "pipeline": {
            "workers": {"requested": "4", "selected": 4},
        },
    }

    _, native_budget, decoder_budget = _validated_worker_receipts(
        report, selected_backend="cpu", requested_preprocess_workers=8,
        requested_pipeline_workers="4", final_hour=1)

    assert native_budget["peak_active_native_workers"] == 8
    assert decoder_budget["selected"] == 4
    report["preparation"]["preprocess_worker_budget"][
        "peak_active_native_workers"] = 9
    with pytest.raises(RuntimeError, match="budget was not honored"):
        _validated_worker_receipts(
            report, selected_backend="cpu", requested_preprocess_workers=8,
            requested_pipeline_workers="4", final_hour=1)


def test_pipeline_mode_accepts_launch_ready_prepared_cache():
    common = _required_args()
    manifest_index = common.index("--manifest-sha256")
    del common[manifest_index:manifest_index + 2]
    args = _parse_args(common + [
        "--pipeline-series", "series.tsv",
        "--pipeline-decoder", "decoder",
        "--pipeline-signals", "signals",
        "--source-root", "source",
        "--source-manifest", "SHA256SUMS",
        "--source-manifest-sha256", "source-manifest-sha256",
        "--prepared-cache", "prepared-cache",
        "--prepare-only",
        "--history-interval-seconds", "3600",
    ])
    assert args.pipeline_series.name == "series.tsv"
    assert args.prepare_only
    assert args.prepared_cache.name == "prepared-cache"
    assert args.io_mode == "none"
    assert args.history_interval_seconds == 3600.0


def test_prepare_only_cadence_is_bound_into_cache_identity_without_output():
    args = _parse_args(_required_args() + [
        "--source-manifest-sha256", "a" * 64,
        "--prepared-cache", "prepared-cache",
        "--prepare-only",
        "--history-interval-seconds", "3600",
    ])
    vertical = VerticalConfig(
        eta_levels=(1.0, 0.75, 0.5, 0.25, 0.0),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(
        HrrrTargetDomain.legacy_500x500(), nz=4)
    exp = _experiment(
        vertical, run_seconds=3600.0, target=target,
        history_interval_seconds=args.history_interval_seconds)

    def identity(domain):
        return prepared_cache_identity(
            bridge_manifest_sha256="b" * 64,
            source_manifest_sha256="a" * 64,
            static_cache_sha256="c" * 64,
            namelist_sha256="d" * 64,
            domain_config=domain,
            forcing_hours=(0, 1),
            source_identity={"source": "fixture"},
        )

    prepared_identity = identity(exp.root)
    forecast_exp = _experiment(
        vertical, run_seconds=3600.0, target=target,
        history_interval_seconds=3600.0)
    legacy_default_exp = _experiment(
        vertical, run_seconds=3600.0, target=target,
        history_interval_seconds=300.0)

    assert args.prepare_only is True
    assert args.io_mode == "none"
    assert identity(forecast_exp.root) == prepared_identity
    assert identity(legacy_default_exp.root) != prepared_identity
    assert prepared_identity["domain_config"]["history_interval_s"] == 3600.0
    assert prepared_identity["domain_config"]["run"][
        "output_interval_s"] == 3600.0


def test_prepare_only_main_summary_does_not_require_forecast_cadence(
        tmp_path, monkeypatch, capsys):
    report = {
        "status": "PASS",
        "run_seconds": 43_200.0,
        "io_mode": "prepare-only",
        "memory": {"gpu_peak_used_bytes_observed": 0},
        "prepared_cache": {"content_sha256": "a" * 64},
    }
    monkeypatch.setattr(
        "tools.hrrr_single_domain_benchmark._parse_args",
        lambda _argv: type("Args", (), {"outdir": tmp_path})())
    monkeypatch.setattr(
        "tools.hrrr_single_domain_benchmark.run", lambda _args: report)

    assert hrrr_runner_main([]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "gpu_peak_used_bytes": 0,
        "history_interval_seconds": None,
        "io_mode": "prepare-only",
        "prepared_cache_content_sha256": "a" * 64,
        "run_seconds": 43_200.0,
        "status": "PASS",
    }


def test_boundary_mapping_targets_are_bitwise_full_grid_slices():
    ny, nx, width = 13, 17, 3
    grid = LambertGrid(
        35.5, -98.0, 30.0, 60.0, -97.0,
        1000.0, 1000.0, nx + 1, ny + 1)
    landmask = (np.arange(ny * nx).reshape(ny, nx) % 2).astype(np.float64)
    targets = _boundary_mapping_targets(
        grid, {"LANDMASK": landmask}, SimpleNamespace(nx=nx, ny=ny),
        width=width)
    rectangles = {
        "west": (0, ny, 0, width),
        "east": (0, ny, nx - width, nx),
        "south": (0, width, 0, nx),
        "north": (ny - width, ny, 0, nx),
    }
    full_mass = grid.latlon_mass()
    full_u = grid.latlon_u()
    full_v = grid.latlon_v()
    for side, (y0, y1, x0, x1) in rectangles.items():
        subgrid, target_landmask = targets[side]
        for actual, expected in zip(
                subgrid.latlon_mass(),
                (value[y0:y1, x0:x1] for value in full_mass)):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(
                subgrid.latlon_u(),
                (value[y0:y1, x0:x1 + 1] for value in full_u)):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(
                subgrid.latlon_v(),
                (value[y0:y1 + 1, x0:x1] for value in full_v)):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            target_landmask, landmask[y0:y1, x0:x1])
        assert target_landmask.flags.c_contiguous


def test_f00_and_boundary_mapping_forward_explicit_target_radius(monkeypatch):
    observed = []

    def fake_interpolate(
            snapshot, grid, *, target_landmask, soil_mapping_report,
            surface_fallback_radius, backend):
        observed.append((surface_fallback_radius, backend))
        soil_mapping_report.update({
            "land_stencil": {
                "fallback_radius_cells": surface_fallback_radius,
                "unresolved_target_count": 0,
                "cross_surface_donor_count": 0,
            },
        })
        return SimpleNamespace(
            valid_time="2026-07-20T06:00:00",
            levels_hpa=np.array([1.0]),
            fields={"Q2": np.ones(np.asarray(target_landmask).shape)},
        )

    monkeypatch.setattr(
        "gpuwm.ingest.hrrr.interpolate_hrrr_to_lambert",
        fake_interpolate)
    grid = object()
    static = {"LANDMASK": np.ones((2, 2))}
    f00_report = {}
    _map_snapshot(
        object(), grid, static, f00_report, surface_fallback_radius=10,
        preprocess_backend="cpu-backend")
    targets = {
        side: (grid, np.ones((2, 2)))
        for side in ("west", "east", "south", "north")
    }
    boundary_report = {}
    _map_boundary_snapshot(
        object(), targets, boundary_report, surface_fallback_radius=10,
        preprocess_backend="cpu-backend")

    assert observed == [(10, "cpu-backend")] * 5
    assert f00_report["land_stencil"]["fallback_radius_cells"] == 10
    assert all(
        report["land_stencil"]["fallback_radius_cells"] == 10
        for report in boundary_report["sides"].values())


@pytest.mark.parametrize("nz", [4, 17, 49, 80, 113])
def test_native_hrrr_experiment_threads_arbitrary_explicit_vertical_grid(nz):
    target = dataclasses.replace(HrrrTargetDomain.legacy_500x500(), nz=nz)
    vertical = VerticalConfig(
        eta_levels=tuple(float(value)
                         for value in np.linspace(1.0, 0.0, nz + 1)),
        p_top=12_345.0,
        hybrid_opt=2,
        etac=0.37,
    )
    exp = _experiment(vertical, run_seconds=300.0, target=target)
    assert exp.root.run.nz == nz
    assert exp.vertical == vertical


@pytest.mark.parametrize(
    ("run_seconds", "cadence_seconds", "expected_offsets", "final_suffix",
     "last_equals_run_end"),
    (
        (3600.0, 900.0, [0.0, 900.0, 1800.0, 2700.0, 3600.0],
         "01_00_00", True),
        (14400.0, 7200.0, [0.0, 7200.0, 14400.0], "04_00_00", True),
        (10800.0, 7200.0, [0.0, 7200.0], "02_00_00", False),
    ),
)
def test_native_hrrr_history_uses_floor_schedule_for_any_due_output(
        run_seconds, cadence_seconds, expected_offsets, final_suffix,
        last_equals_run_end):
    vertical = VerticalConfig(
        eta_levels=(1.0, 0.75, 0.5, 0.25, 0.0),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(
        HrrrTargetDomain.legacy_500x500(), nz=4)
    exp = _experiment(
        vertical, run_seconds=run_seconds, target=target,
        history_interval_seconds=cadence_seconds)

    receipt = hrrr_runner._validate_history_output_cadence(
        exp, cadence_seconds)
    schedule = hrrr_runner._history_output_schedule(
        start_time=exp.start_time, run_seconds=run_seconds,
        cadence_seconds=cadence_seconds)

    assert [record[0] for record in schedule] == expected_offsets
    assert schedule[-1][2].endswith(final_suffix)
    assert receipt["expected_frame_count"] == len(schedule)
    assert receipt["last_scheduled_offset_seconds"] == expected_offsets[-1]
    assert receipt["last_scheduled_valid_time"] == schedule[-1][1].isoformat()
    assert receipt["run_end_offset_seconds"] == run_seconds
    assert receipt["last_scheduled_equals_run_end"] is last_equals_run_end
    assert receipt["run_end_frame_scheduled"] is last_equals_run_end
    assert exp.root.run.output_interval_s == cadence_seconds


def test_native_hrrr_history_rejects_mismatch_and_nonstep_cadence():
    vertical = VerticalConfig(
        eta_levels=(1.0, 0.75, 0.5, 0.25, 0.0),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(
        HrrrTargetDomain.legacy_500x500(), nz=4)
    exp = _experiment(
        vertical, run_seconds=10800.0, target=target,
        history_interval_seconds=7200.0)

    with pytest.raises(ValueError, match="exactly match the generated"):
        hrrr_runner._validate_history_output_cadence(exp, 3600.0)
    domain = dataclasses.replace(
        exp.root, history_interval_s=71.0,
        run=dataclasses.replace(exp.root.run, output_interval_s=71.0))
    nonstep = dataclasses.replace(exp, domains=(domain,))
    with pytest.raises(ValueError, match="whole number of exact model"):
        hrrr_runner._validate_history_output_cadence(nonstep, 71.0)


def test_native_hrrr_cli_requires_explicit_history_cadence():
    args = _parse_args(_required_args() + [
        "--io-mode", "history", "--history-interval-seconds", "900"])
    assert args.io_mode == "history"
    assert args.history_interval_seconds == 900.0

    with pytest.raises(SystemExit):
        _parse_args(_required_args() + ["--io-mode", "history"])
    with pytest.raises(SystemExit):
        _parse_args(_required_args() + [
            "--io-mode", "history", "--history-interval-seconds", "nan"])
    with pytest.raises(SystemExit):
        _parse_args(_required_args() + [
            "--io-mode", "none", "--history-interval-seconds", "900"])


def _write_native_physics_namelist(
        path, *, mp_physics=6, include_diff_factor=True,
        full_real74_suite=False, morr_rimed_ice=1, cu_physics=1,
        nssl_controls=""):
    diff_factor = (" diff_6th_factor = 0.10, 0.08,\n"
                   if include_diff_factor else "")
    if full_real74_suite:
        diff_factor = (" diff_6th_factor = 0.10, 0.12,\n"
                       if include_diff_factor else "")
        radiation = """ ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 12, 12,
 sf_sfclay_physics = 91, 91,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 1, 1,
 cu_physics = 1, {cu_physics},
 cudt = 5, 5,
 num_soil_layers = 4, 4,
""".format(cu_physics=cu_physics)
        dynamics = """ epssm = 0.1, 0.5,
 top_lid = .true., .false.,
"""
        morrison = (
            f" morr_rimed_ice = 1, {morr_rimed_ice},\n"
            if mp_physics == 10 else "")
    else:
        radiation = """ ra_lw_physics = 0, 0,
 ra_sw_physics = 1, 1,
 radt = 3, 1,
 sf_sfclay_physics = 91, 91,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 1, 1,
 cu_physics = 0, 0,
"""
        dynamics = ""
        morrison = ""
    path.write_text(f"""&physics
 mp_physics = 6, {mp_physics},
 {morrison}{nssl_controls}{radiation}\
/
&dynamics
{dynamics}\
 km_opt = 4, 4,
 diff_6th_opt = 2, 2,
{diff_factor} diff_6th_slopeopt = 1, 1,
/
""", encoding="ascii")


def test_native_hrrr_fixed_physics_profile_validates_selected_inner_domain(
        tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path)
    receipt = _validate_native_hrrr_physics_profile(path)
    assert receipt["profile"] == WSM6_PROFILE_ID
    assert receipt["selection"] == "last-or-only WRF domain value"
    assert receipt["validated_namelist"]["physics"]["radt"] == 1.0
    assert receipt["validated_namelist"]["dynamics"][
        "diff_6th_factor"] == 0.08


def test_native_hrrr_fixed_physics_profile_rejects_scheme_drift(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, mp_physics=8)
    with pytest.raises(ValueError, match="mp_physics.*must be 6"):
        _validate_native_hrrr_physics_profile(path)


def test_native_hrrr_fixed_physics_profile_rejects_missing_control(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, include_diff_factor=False)
    with pytest.raises(ValueError, match="requires.*diff_6th_factor"):
        _validate_native_hrrr_physics_profile(path)


def test_native_hrrr_thompson_profile_is_guarded_and_table_bound(
        tmp_path, monkeypatch):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, mp_physics=8)
    with pytest.raises(RuntimeError, match="GPUWM_EXPERIMENTAL_THOMPSON_MP8=1"):
        _validate_native_hrrr_physics_profile(path, THOMPSON_PROFILE_ID)

    monkeypatch.setenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", "1")
    with pytest.raises(RuntimeError, match="GPUWM_THOMPSON_TABLE_ROOT"):
        _validate_native_hrrr_physics_profile(path, THOMPSON_PROFILE_ID)

    table_root = tmp_path / "tables"
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(table_root))
    asset = SimpleNamespace(
        filename="freezeH2O.dat", bytes=123, sha256="a" * 64)
    monkeypatch.setattr(
        "gpuwm.core.thompson_contract.validate_table_assets",
        lambda root: (asset,) if root == table_root.resolve() else ())
    receipt = _validate_native_hrrr_physics_profile(
        path, THOMPSON_PROFILE_ID)
    assert receipt["profile"] == THOMPSON_PROFILE_ID
    assert receipt["readiness"] == "MODEL_VALIDATED_EXPERIMENTAL"
    assert receipt["resolved"]["mp_physics"] == 8
    assert receipt["resolved"]["moist"] is True
    assert receipt["resolved"]["moist_cq"] is True
    assert receipt["resolved"]["top_lid"] is False
    assert receipt["thompson_contract"]["assets"] == [{
        "filename": "freezeH2O.dat", "bytes": 123, "sha256": "a" * 64}]


@pytest.mark.parametrize(
    ("profile", "selector"),
    ((MORRISON_PROFILE_ID, 10), (NSSL2_PROFILE_ID, 18)),
)
def test_native_hrrr_real74_suite_profiles_bind_exact_namelist_and_runtime(
        tmp_path, profile, selector):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=selector, full_real74_suite=True)

    receipt = _validate_native_hrrr_physics_profile(path, profile)

    assert receipt["profile"] == profile
    assert receipt["resolved"] == {
        "moist": True,
        "moist_cq": True,
        "mp_physics": selector,
        "top_lid": False,
        "epssm": 0.5,
        "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0,
        "ra_physics": 0,
        "ra_lw_physics": 4,
        "ra_sw_physics": 4,
        "radt": 12.0,
        "wrf_rrtmg_compatibility": WRF_RRTMG_TO_RTE_RRTMGP,
        "sf_sfclay_physics": 91,
        "sf_surface_physics": 2,
        "bl_pbl_physics": 1,
        "cu_physics": 1,
        "cudt_minutes": 5.0,
        "num_soil_layers": 4,
        "terrain_opt": 1,
        "km_opt": 4,
        "diff_6th_opt": 2,
        "diff_6th_factor": 0.12,
        "diff_6th_slopeopt": 1,
        "radiation_scheme_ids": [4, 4],
    }
    assert receipt["radiation_substitution"]["contract"] \
        == WRF_RRTMG_TO_RTE_RRTMGP
    if profile == MORRISON_PROFILE_ID:
        assert receipt["readiness"] == "MODEL_VALIDATED_RUNTIME_PROFILE"
        assert receipt["morrison_contract"]["morr_rimed_ice"] == 1
    else:
        assert receipt["readiness"] == "VALIDATION_CANDIDATE"
        assert receipt["nssl2_contract"]["selector"] == 18
        assert receipt["nssl2_contract"]["resolved_default_mode"] == {
            "two_moment": True,
            "hail": True,
            "predicted_ccn": True,
            "density_moments": 2,
            "sixth_moments": 0,
        }


def test_native_hrrr_morrison_profile_rejects_rimed_ice_drift(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=10, full_real74_suite=True, morr_rimed_ice=0)
    with pytest.raises(ValueError, match="morr_rimed_ice.*must be 1"):
        _validate_native_hrrr_physics_profile(path, MORRISON_PROFILE_ID)


def test_native_hrrr_nssl2_profile_rejects_surrounding_suite_drift(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=18, full_real74_suite=True, cu_physics=0)
    with pytest.raises(ValueError, match="cu_physics.*must be 1"):
        _validate_native_hrrr_physics_profile(path, NSSL2_PROFILE_ID)


def test_native_hrrr_nssl2_accepts_equivalent_explicit_optional_selectors(
        tmp_path):
    path = tmp_path / "namelist.input"
    controls = """ nssl_2moment_on = -1, 1,
 nssl_hail_on = -1, 1,
 nssl_ccn_on = -1, 1,
 nssl_density_on = -1, 2,
 nssl_3moment = 0, 0,
"""
    _write_native_physics_namelist(
        path, mp_physics=18, full_real74_suite=True,
        nssl_controls=controls)

    receipt = _validate_native_hrrr_physics_profile(
        path, NSSL2_PROFILE_ID)

    assert receipt["validated_namelist"]["physics"]["nssl_hail_on"] == 1
    assert receipt["validated_namelist"]["physics"][
        "nssl_density_on"] == 2


@pytest.mark.parametrize("controls,match", (
    (" nssl_hail_on = -1, 0,\n", "fixed preset does not implement"),
    (" nssl_rho_qhl = 900, 800,\n", "nssl_rho_qhl=900.0"),
))
def test_native_hrrr_nssl2_rejects_silently_substituted_optional_controls(
        tmp_path, controls, match):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=18, full_real74_suite=True,
        nssl_controls=controls)

    with pytest.raises(ValueError, match=match):
        _validate_native_hrrr_physics_profile(path, NSSL2_PROFILE_ID)


@pytest.mark.parametrize(
    ("profile", "selector"),
    ((MORRISON_PROFILE_ID, 10), (NSSL2_PROFILE_ID, 18)),
)
def test_native_hrrr_real74_suite_experiment_switches_are_exact(
        profile, selector):
    vertical = VerticalConfig(
        eta_levels=tuple(float(value) for value in np.linspace(1.0, 0.0, 50)),
        p_top=5000.0,
        hybrid_opt=2,
        etac=0.2,
    )
    cfg = _experiment(
        vertical, run_seconds=3600.0, physics_profile=profile).root.run

    assert cfg.mp_physics == selector
    assert cfg.moist is True
    assert cfg.moist_cq is True
    assert cfg.top_lid is False
    assert cfg.epssm == 0.5
    assert cfg.morr_rimed_ice == 1
    assert cfg.sf_sfclay_physics == 91
    assert cfg.sf_surface_physics == 2
    assert cfg.bl_pbl_physics == 1
    assert cfg.cu_physics == 1
    assert cfg.cudt_minutes == 5.0
    assert cfg.ra_physics == 0
    assert (cfg.ra_lw_physics, cfg.ra_sw_physics) == (4, 4)
    assert cfg.wrf_rrtmg_compatibility == WRF_RRTMG_TO_RTE_RRTMGP
    assert cfg.radt == 12.0
    assert cfg.diff_6th_factor == 0.12


@pytest.mark.parametrize("profile", (
    WSM6_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    MORRISON_PROFILE_ID,
    NSSL2_PROFILE_ID,
))
@pytest.mark.parametrize("hybrid_opt", (1, 2))
def test_native_hrrr_profile_preserves_explicit_vertical_hybrid_option(
        profile, hybrid_opt, monkeypatch):
    monkeypatch.setenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", "1")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", "config-only-fixture")
    vertical = VerticalConfig(
        eta_levels=tuple(float(value) for value in np.linspace(1.0, 0.0, 50)),
        p_top=5000.0,
        hybrid_opt=hybrid_opt,
        etac=0.2,
    )
    exp = _experiment(
        vertical, run_seconds=3600.0, physics_profile=profile)
    receipt = {
        "resolved": hrrr_runner._native_hrrr_runtime_switches(profile),
    }

    hrrr_runner._validate_resolved_hrrr_profile(exp, receipt)

    assert exp.root.run.hybrid_opt == hybrid_opt


def test_hrrr_thompson_initialization_requires_mass_and_exact_zero_numbers():
    shape = (3, 2, 4)
    state = SimpleNamespace(**{
        name: np.full(shape, index * 1.0e-6, dtype=np.float32)
        for index, name in enumerate(("qc", "qr", "qi", "qs", "qg"), 1)
    })
    state.ni = np.zeros(shape, dtype=np.float32)
    state.nr = np.zeros(shape, dtype=np.float32)
    receipt = _initial_hrrr_microphysics_receipt(state, THOMPSON_PROFILE_ID)
    assert receipt["state_number_fields"] == {
        "ni": {"all_exact_zero": True},
        "nr": {"all_exact_zero": True},
    }
    state.nr.flat[0] = np.float32(1.0)
    with pytest.raises(ValueError, match="number moment nr must initialize"):
        _initial_hrrr_microphysics_receipt(state, THOMPSON_PROFILE_ID)


def _analyzed_hrrr_mass_state(shape=(3, 2, 4)):
    return SimpleNamespace(**{
        name: np.full(shape, index * 1.0e-6, dtype=np.float32)
        for index, name in enumerate(("qc", "qr", "qi", "qs", "qg"), 1)
    })


def test_hrrr_morrison_initialization_receipts_every_source_absent_moment():
    state = _analyzed_hrrr_mass_state()
    for name in ("nc", "nr", "ni", "ns", "ng"):
        setattr(state, name, np.zeros((3, 2, 4), dtype=np.float32))

    receipt = _initial_hrrr_microphysics_receipt(
        state, MORRISON_PROFILE_ID)

    assert receipt["source_absent_wrf_fields"] == [
        "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL"]
    assert receipt["state_number_fields"] == {
        name: {"all_exact_zero": True}
        for name in ("nc", "nr", "ni", "ns", "ng")
    }
    assert all(
        item["expected_uint32_bits"] == 0
        and item["all_exact_expected"] is True
        for item in receipt["state_source_absent_fields"].values())

    state.ng.flat[0] = np.float32(1.0)
    with pytest.raises(ValueError, match="source-absent state ng"):
        _initial_hrrr_microphysics_receipt(state, MORRISON_PROFILE_ID)


def test_hrrr_nssl2_initialization_receipts_zero_state_and_background_ccn():
    state = _analyzed_hrrr_mass_state()
    zero_fields = (
        "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh",
        "qvolg", "qvolh",
    )
    for name in zero_fields:
        setattr(state, name, np.zeros((3, 2, 4), dtype=np.float32))
    state.qnn = np.full(
        (3, 2, 4), np.float32(408163264.0), dtype=np.float32)

    receipt = _initial_hrrr_microphysics_receipt(state, NSSL2_PROFILE_ID)

    exact = receipt["state_source_absent_fields"]
    assert set(exact) == {*zero_fields, "qnn"}
    assert exact["qnn"] == {
        "expected_float32": 408163264.0,
        "expected_uint32_bits": int(
            np.float32(408163264.0).view(np.uint32)),
        "all_exact_expected": True,
    }
    assert receipt["state_number_fields"]["qnn"] == {
        "expected_float32": 408163264.0,
        "all_exact_expected": True,
    }
    assert exact["qh"]["expected_uint32_bits"] == 0

    state.qnn.flat[0] = np.nextafter(
        np.float32(408163264.0), np.float32(np.inf), dtype=np.float32)
    with pytest.raises(ValueError, match="source-absent state qnn"):
        _initial_hrrr_microphysics_receipt(state, NSSL2_PROFILE_ID)


@pytest.mark.parametrize(
    ("profile", "expected_fields"),
    (
        (MORRISON_PROFILE_ID, {"nc", "nr", "ni", "ns", "ng"}),
        (NSSL2_PROFILE_ID, {
            "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh",
            "qnn", "qvolg", "qvolh",
        }),
    ),
)
def test_real_domain_state_cold_start_satisfies_profile_receipt(
        profile, expected_fields):
    from gpuwm.core.state import DomainState

    target = dataclasses.replace(
        HrrrTargetDomain.legacy_500x500(), nx=16, ny=16, nz=4)
    vertical = VerticalConfig(
        eta_levels=(1.0, 0.75, 0.5, 0.25, 0.0),
        p_top=5000.0,
        hybrid_opt=2,
        etac=0.2,
    )
    cfg = _experiment(
        vertical, run_seconds=3600.0, target=target,
        physics_profile=profile).root.run
    state = DomainState(cfg, array_module=np)

    receipt = _initial_hrrr_microphysics_receipt(state, profile)

    assert set(receipt["state_source_absent_fields"]) == expected_fields
    assert receipt["state_source_absent_fields"].get(
        "qnn", {"expected_float32": 0.0})["expected_float32"] in (
            0.0, 408163264.0)
