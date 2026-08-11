"""Native HRRR single-domain benchmark controller regressions."""

import copy
import dataclasses
from datetime import datetime
import json
from types import SimpleNamespace

import numpy as np
import pytest

import tools.hrrr_single_domain_benchmark as hrrr_runner
from gpuwm.static.lambert import LambertGrid
from gpuwm.experiment import VerticalConfig
from gpuwm.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
from gpuwm.ingest.hrrr_target import HrrrTargetDomain
from gpuwm.ingest.prepared_cache import prepared_cache_identity
from gpuwm.physics_compat import (
    KESSLER_PROFILE_ID,
    MORRISON_PROFILE_ID,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    MYNN_PROFILE_ID,
    MYNN_RTE_RRTMGP_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    RUC_PROFILE_ID,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_TO_RTE_RRTMGP,
    WSM6_PROFILE_ID,
    downward_longwave_disposition,
    single_domain_runtime_switches,
)
from tools.hrrr_single_domain_benchmark import (
    _PreprocessWorkerBudget, _boundary_mapping_targets, _experiment,
    _initial_hrrr_microphysics_receipt,
    main as hrrr_runner_main,
    _map_boundary_snapshot, _map_snapshot, _parse_args,
    _partition_preprocess_worker_budget,
    _validated_namelist_extension_identity,
    _validate_native_hrrr_physics_profile,
    runner_capabilities,
)
from tools.prepare_hrrr_wrf import _validated_worker_receipts
from gpuwm.stream import _materialize_input_namelist


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
        WSM6_PROFILE_ID, KESSLER_PROFILE_ID,
        THOMPSON_PROFILE_ID, THOMPSON_LEGACY_RRTMG_PROFILE_ID,
        THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
        MORRISON_PROFILE_ID,
        NSSL2_PROFILE_ID, NSSL2_LEGACY_RRTMG_PROFILE_ID,
        MYNN_PROFILE_ID, MYNN_RTE_RRTMGP_PROFILE_ID,
        RUC_PROFILE_ID,
        MYNN_RUC_PROFILE_ID, MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
        NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID,
        MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID]
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
    assert thompson["readiness"] == "WRF_MATCHED_RUN_EXPERIMENTAL_RUNTIME"
    assert thompson["runtime_guard"]["environment"] \
        == "GPUWM_EXPERIMENTAL_THOMPSON_MP8"
    assert thompson["explicit_expert_consent_required"] is False
    assert thompson["table_root"]["environment"] \
        == "GPUWM_THOMPSON_TABLE_ROOT"
    assert len(thompson["table_authority"]["assets"]) == 4
    assert payload["physics_profiles"][MORRISON_PROFILE_ID] == {
        "selector": 10,
        "readiness": "WRF_MATCHED_RUN_RUNTIME_PROFILE",
        "explicit_expert_consent_required": False,
        "runtime_guards": [],
    }
    assert payload["physics_profiles"][KESSLER_PROFILE_ID] == {
        "selector": 1,
        "readiness": "IMPLEMENTED_UNVERIFIED",
        "explicit_expert_consent_required": False,
        "runtime_guards": [],
        "source_scope": ["hrrr"],
        "frozen_species_policy": (
            "retain QC/QR; discard source QI/QS/QG with a receipt"),
    }
    nssl2 = payload["physics_profiles"][NSSL2_PROFILE_ID]
    assert nssl2["selector"] == 18
    assert nssl2["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
    assert nssl2["explicit_expert_consent_required"] is False
    legacy = payload["physics_profiles"][NSSL2_LEGACY_RRTMG_PROFILE_ID]
    assert legacy["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
    assert legacy["radiation_solver"] == "legacy RRTMG"
    assert nssl2["resolved_fixed_preset"] is True
    assert payload["physics_profiles"][MYNN_PROFILE_ID]["readiness"] \
        == "IMPLEMENTED_UNVERIFIED"
    assert payload["physics_profiles"][RUC_PROFILE_ID]["readiness"] \
        == "IMPLEMENTED_UNVERIFIED"
    assert payload["physics_profiles"][MYNN_RUC_PROFILE_ID]["readiness"] \
        == "IMPLEMENTED_UNVERIFIED"
    noahmp = payload["physics_profiles"][NOAHMP_PROFILE_ID]
    assert noahmp["explicit_expert_consent_required"] is True
    assert noahmp["expert_acknowledgement_id"] \
        == "noahmp-host-column-throughput-v1"
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


def _extension_namelist(tmp_path, *, lead: int):
    cycle = datetime(2026, 7, 20, 6)
    template = tmp_path / "namelist.template"
    template.write_text(
        "&time_control\n/\n&domains\n max_dom = 1,\n/\n",
        encoding="utf-8")
    output = tmp_path / f"namelist-f{lead:03d}.input"
    _materialize_input_namelist(
        template, output, cycle=cycle, lead=lead,
        domain_starts=[cycle])
    return cycle, output


def test_sealed_prepare_recomputes_namelist_invariant_from_bytes(tmp_path):
    cycle, namelist = _extension_namelist(tmp_path, lead=1)
    common = _required_args() + [
        "--source-manifest-sha256", "a" * 64,
        "--prepared-cache", "prepared-cache",
        "--namelist-input", str(namelist),
        "--prepare-only",
        "--history-interval-seconds", "3600",
        "--sealed-prepared-cache",
    ]
    args = _parse_args(common)
    invariant = _validated_namelist_extension_identity(args, cycle=cycle)
    assert invariant["schema"] == "gpuwm-namelist-extension-invariant-v1"
    assert len(invariant["sha256"]) == 64

    payload = namelist.read_text(encoding="utf-8")
    assert " end_hour = 7," in payload
    namelist.write_text(
        payload.replace(" end_hour = 7,", " end_hour = 8,"),
        encoding="utf-8")
    with pytest.raises(ValueError, match="end time differs"):
        _validated_namelist_extension_identity(args, cycle=cycle)

    with pytest.raises(SystemExit):
        _parse_args(common + [
            "--namelist-extension-invariant-sha256", "b" * 64,
        ])


def test_suffix_prepare_recomputes_full_horizon_namelist_invariant(tmp_path):
    cycle, namelist = _extension_namelist(tmp_path, lead=2)
    args = _parse_args(_required_args() + [
        "--source-manifest-sha256", "a" * 64,
        "--prepared-cache", "prepared-cache",
        "--namelist-input", str(namelist),
        "--forecast-start-hour", "1",
        "--forecast-end-hour", "2",
        "--prepare-only",
        "--history-interval-seconds", "3600",
        "--namelist-extension-suffix",
    ])
    assert args.sealed_prepared_cache is False
    assert _validated_namelist_extension_identity(
        args, cycle=cycle)["schema"] \
        == "gpuwm-namelist-extension-invariant-v1"


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
            surface_fallback_radius, backend, target_name):
        observed.append((surface_fallback_radius, backend, target_name))
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

    assert [entry[:2] for entry in observed] == [(10, "cpu-backend")] * 5
    # Each of the five mappings names ITSELF.  A soil refusal from one of
    # four independently mapped boundary strips used to name neither the
    # strip nor the domain, which on a coastal domain left four
    # identical-looking suspects.
    assert [entry[2] for entry in observed] == [
        "domain 1",
        "the west boundary strip of domain 1",
        "the east boundary strip of domain 1",
        "the south boundary strip of domain 1",
        "the north boundary strip of domain 1",
    ]
    assert f00_report["land_stencil"]["fallback_radius_cells"] == 10
    assert all(
        report["land_stencil"]["fallback_radius_cells"] == 10
        for report in boundary_report["sides"].values())


@pytest.mark.parametrize("nz", [4, 17, 49, 80])
def test_native_hrrr_experiment_threads_admitted_explicit_vertical_grid(nz):
    # The 1.8 route default runs the legacy-RRTMG shortwave port, whose
    # transcribed wrapper caps TOTAL layers at 64 -- comfortably above
    # HRRR's own 51-level native vertical, below this case's 80.  The
    # question here is whether an explicit vertical grid THREADS, so the
    # deep case picks a suite that admits its depth; the ceiling itself
    # is asserted in both directions by the test below.
    profile = (WSM6_PROFILE_ID if nz > 60
               else ROUTE_DEFAULT_PHYSICS_PROFILE)
    target = dataclasses.replace(HrrrTargetDomain.legacy_500x500(), nz=nz)
    vertical = VerticalConfig(
        eta_levels=tuple(float(value)
                         for value in np.linspace(1.0, 0.0, nz + 1)),
        p_top=12_345.0,
        hybrid_opt=2,
        etac=0.37,
    )
    exp = _experiment(vertical, run_seconds=300.0, target=target,
                      physics_profile=profile)
    assert exp.root.run.nz == nz
    assert exp.vertical == vertical


def test_the_route_defaults_legacy_shortwave_layer_ceiling_is_named():
    """The one real limit the 1.8 default carries, both directions.

    The legacy-RRTMG shortwave port is a transcription of WRF's, and its
    wrapper caps total layers at 64.  HRRR's native vertical is 51
    levels, so the ceiling does not bite on this source's own grid --
    but a hand-authored deep vertical passes it, and the refusal has to
    name the number rather than fail somewhere inside radiation setup
    after a preparation has been paid for.
    """
    from gpuwm.physics_compat import PhysicsVerticalPreflightError

    def experiment_at(nz):
        return _experiment(
            VerticalConfig(
                eta_levels=tuple(
                    float(value)
                    for value in np.linspace(1.0, 0.0, nz + 1)),
                p_top=12_345.0, hybrid_opt=2, etac=0.37),
            run_seconds=300.0,
            target=dataclasses.replace(
                HrrrTargetDomain.legacy_500x500(), nz=nz),
            physics_profile=ROUTE_DEFAULT_PHYSICS_PROFILE)

    # HRRR's own native depth builds.
    assert experiment_at(49).root.run.nz == 49
    # A deeper one is refused by number, before anything is paid for.
    with pytest.raises(PhysicsVerticalPreflightError,
                       match="legacy RRTMG shortwave"):
        experiment_at(80)


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
        nssl_controls="", sf_sfclay_physics=91,
        sf_surface_physics=2, bl_pbl_physics=1,
        num_soil_layers=None):
    # d01 comes FIRST in every per-domain column here, because this
    # preparer prepares the root: the profile-conforming value is d01's and
    # the second entry is a nest value that must NOT be selected.  The
    # columns stay deliberately non-uniform, which is what makes reading
    # the wrong end of a WRF array a visible failure rather than a
    # coincidence.
    diff_factor = (" diff_6th_factor = 0.08, 0.10,\n"
                   if include_diff_factor else "")
    if full_real74_suite:
        diff_factor = (" diff_6th_factor = 0.12, 0.10,\n"
                       if include_diff_factor else "")
        radiation = """ ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 12, 12,
 sf_sfclay_physics = 91, 91,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 1, 1,
 cu_physics = {cu_physics}, 1,
 cudt = 5, 5,
 num_soil_layers = 4, 4,
""".format(cu_physics=cu_physics)
        dynamics = """ epssm = 0.5, 0.1,
 top_lid = .false., .true.,
"""
        morrison = (
            f" morr_rimed_ice = {morr_rimed_ice}, 1,\n"
            if mp_physics == 10 else "")
    else:
        soil = (
            f" num_soil_layers = {num_soil_layers}, 4,\n"
            if num_soil_layers is not None else "")
        radiation = f""" ra_lw_physics = 0, 0,
 ra_sw_physics = 1, 1,
 radt = 1, 3,
 sf_sfclay_physics = {sf_sfclay_physics}, 91,
 sf_surface_physics = {sf_surface_physics}, 2,
 bl_pbl_physics = {bl_pbl_physics}, 1,
 cu_physics = 0, 0,
{soil}"""
        dynamics = ""
        morrison = ""
    path.write_text(f"""&physics
 mp_physics = {mp_physics}, 6,
 {morrison}{nssl_controls}{radiation}\
/
&dynamics
{dynamics}\
 km_opt = 4, 4,
 diff_6th_opt = 2, 2,
{diff_factor} diff_6th_slopeopt = 1, 1,
/
""", encoding="ascii")


def test_native_hrrr_fixed_physics_profile_validates_the_root_domain(
        tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path)
    # Named explicitly since 1.8: the route default is the full-radiation
    # suite, and this namelist describes WSM6.  The wsm6 family stays
    # fully selectable -- as a stated choice, which is the point.
    receipt = _validate_native_hrrr_physics_profile(path, WSM6_PROFILE_ID)
    assert receipt["profile"] == WSM6_PROFILE_ID
    assert receipt["selection"] == "d01 (first-or-only) WRF domain value"
    assert receipt["validated_namelist"]["physics"]["radt"] == 1.0
    assert receipt["validated_namelist"]["dynamics"][
        "diff_6th_factor"] == 0.08


def test_native_hrrr_root_preparer_reads_d01_and_names_the_nest_it_ignored(
        tmp_path):
    """The exact v1.3.1 field report, both halves.

    A user handed this single-domain root preparer their two-domain
    namelist.  d01 stated the WSM6 profile's diff_6th_factor = 0.08; d02
    stated a nest's 0.10.  The preparer read the LAST column, refused with
    "selected last-domain value must be 0.08, got 0.1", and the user
    changed d02 -- which is not what they meant and not what the root
    runs.  d01 is now what binds, and a refusal that happens to land on a
    per-domain column says which domain it read and how to fix it.
    """

    good = tmp_path / "d01-conforming.input"
    _write_native_physics_namelist(good)
    assert " diff_6th_factor = 0.08, 0.10,\n" in good.read_text()
    receipt = _validate_native_hrrr_physics_profile(good, WSM6_PROFILE_ID)
    assert receipt["validated_namelist"]["dynamics"][
        "diff_6th_factor"] == 0.08

    drifted = tmp_path / "d01-drifted.input"
    drifted.write_text(
        good.read_text().replace(
            " diff_6th_factor = 0.08, 0.10,",
            " diff_6th_factor = 0.10, 0.08,"),
        encoding="ascii")
    with pytest.raises(ValueError) as refusal:
        _validate_native_hrrr_physics_profile(drifted, WSM6_PROFILE_ID)
    message = str(refusal.value)
    assert "d01's value must be 0.08" in message
    assert ("multi-domain namelist passed to a single-domain root preparer"
            in message)
    assert "&dynamics/diff_6th_factor" in message


def test_native_hrrr_fixed_physics_profile_rejects_scheme_drift(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, mp_physics=8)
    with pytest.raises(ValueError, match="mp_physics.*must be 6"):
        _validate_native_hrrr_physics_profile(path, WSM6_PROFILE_ID)


def test_native_hrrr_fixed_physics_profile_rejects_missing_control(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, include_diff_factor=False)
    with pytest.raises(ValueError, match="requires.*diff_6th_factor"):
        _validate_native_hrrr_physics_profile(path, WSM6_PROFILE_ID)


@pytest.mark.parametrize(
    ("profile", "sfclay", "surface", "pbl", "soil_layers"),
    (
        (MYNN_PROFILE_ID, 5, 2, 5, 4),
        (RUC_PROFILE_ID, 91, 3, 1, 9),
        (MYNN_RUC_PROFILE_ID, 5, 3, 5, 9),
        (NOAHMP_PROFILE_ID, 91, 4, 1, 4),
        (MYNN_NOAHMP_PROFILE_ID, 5, 4, 5, 4),
    ),
)
def test_native_hrrr_new_front_door_families_prepare_exact_profile(
        tmp_path, profile, sfclay, surface, pbl, soil_layers):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, sf_sfclay_physics=sfclay,
        sf_surface_physics=surface, bl_pbl_physics=pbl,
        num_soil_layers=soil_layers)
    acknowledgements = (
        ("noahmp-host-column-throughput-v1",)
        if profile in (NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID) else ()
    )

    receipt = _validate_native_hrrr_physics_profile(
        path, profile, expert_acknowledgements=acknowledgements)

    assert receipt["profile"] == profile
    assert receipt["readiness"] == "IMPLEMENTED_UNVERIFIED"
    assert receipt["resolved"]["sf_sfclay_physics"] == sfclay
    assert receipt["resolved"]["sf_surface_physics"] == surface
    assert receipt["resolved"]["bl_pbl_physics"] == pbl
    assert receipt["resolved"]["num_soil_layers"] == soil_layers
    assert receipt["front_door_selection"]["profile"] == profile


def test_native_hrrr_noahmp_refuses_without_registry_acknowledgement(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, sf_surface_physics=4, num_soil_layers=4)

    with pytest.raises(
            ValueError, match="noahmp-host-column-throughput-v1"):
        _validate_native_hrrr_physics_profile(path, NOAHMP_PROFILE_ID)


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
    # 1.8: the route stages microphysics tables for EVERY mp8 profile,
    # before this profile's own environment launch contract, and staging
    # answers presence first (cheap) and bytes second.  A stand-in root
    # therefore has to exist on disk; its BYTES stay monkeypatched below,
    # which is what this test is actually about.
    table_root.mkdir()
    from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS
    for pinned in CLASSIC_TABLE_ASSETS:
        (table_root / pinned.filename).write_bytes(b"")
    asset = SimpleNamespace(
        filename="freezeH2O.dat", bytes=123, sha256="a" * 64)
    monkeypatch.setattr(
        "gpuwm.core.thompson_contract.validate_table_assets",
        lambda root: (asset,) if root == table_root.resolve() else ())
    receipt = _validate_native_hrrr_physics_profile(
        path, THOMPSON_PROFILE_ID)
    assert receipt["profile"] == THOMPSON_PROFILE_ID
    assert receipt["readiness"] == "WRF_MATCHED_RUN_EXPERIMENTAL"
    assert receipt["resolved"]["mp_physics"] == 8
    assert receipt["resolved"]["moist"] is True
    assert receipt["resolved"]["moist_cq"] is True
    assert receipt["resolved"]["top_lid"] is False
    assert receipt["thompson_contract"]["assets"] == [{
        "filename": "freezeH2O.dat", "bytes": 123, "sha256": "a" * 64}]


def test_the_guarded_mp8_refusal_names_both_variables_with_values(
        tmp_path, monkeypatch):
    """One refusal carries the whole launch contract, not half of it.

    A field run of the shipped 1.5.0 wheel set the first variable, was
    refused for the second, and then had to work out the table root by
    hand -- having already downloaded it.  The guard itself is unchanged:
    both variables are still required and the root is still byte-checked.
    """
    from gpuwm.physics_compat import thompson_guard_exports

    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, mp_physics=8)
    monkeypatch.delenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT", raising=False)

    for preset in ({}, {"GPUWM_EXPERIMENTAL_THOMPSON_MP8": "1"}):
        for name, value in preset.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(RuntimeError) as raised:
            _validate_native_hrrr_physics_profile(path, THOMPSON_PROFILE_ID)
        message = str(raised.value)
        assert "GPUWM_EXPERIMENTAL_THOMPSON_MP8" in message
        assert "GPUWM_THOMPSON_TABLE_ROOT" in message
        # Pasteable, and platform-correct: an `export` line in PowerShell
        # is a syntax error and the reader then debugs the instructions.
        for line in thompson_guard_exports():
            assert line in message


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
            "ra_rrtmg_variant": "rte-rrtmgp",
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
        assert receipt["readiness"] == "WRF_MATCHED_RUN_RUNTIME_PROFILE"
        assert receipt["morrison_contract"]["morr_rimed_ice"] == 1
    else:
        assert receipt["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
        assert receipt["nssl2_contract"]["selector"] == 18
        assert receipt["nssl2_contract"]["is_default_lane"] is True
        assert receipt["nssl2_contract"]["resolved_mode"] == {
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


def test_native_hrrr_nssl2_legacy_profile_binds_exact_solver_identity(
        tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=18, full_real74_suite=True)

    receipt = _validate_native_hrrr_physics_profile(
        path, NSSL2_LEGACY_RRTMG_PROFILE_ID)

    assert receipt["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
    assert receipt["resolved"]["wrf_rrtmg_compatibility"] == WRF_RRTMG_LEGACY
    assert receipt["resolved"]["ra_rrtmg_variant"] == "rrtmg_legacy"
    assert receipt["radiation_identity"] == {
        "contract": WRF_RRTMG_LEGACY,
        "requested_wrf_scheme_ids": [4, 4],
        "resolved_gpuwm_scheme_ids": [4, 4],
        "resolved_gpuwm_solver": "legacy RRTMG",
    }
    assert "radiation_substitution" not in receipt


def _profile_experiment(profile: str, **kwargs):
    """One HRRR experiment per profile, on a grid every profile accepts.

    nz=12 clears the Kain-Fritsch (>=8) and MYNN (>=5) vertical floors, so
    a refusal below is about the switch forward and nothing else.

    ``kwargs`` reaches :func:`_experiment` untouched, which is how the
    window tests below choose a start time: the default one is
    all-daylight over this target and the declaration rule differs
    between the two windows.
    """

    nz = 12
    vertical = VerticalConfig(
        eta_levels=tuple(float(value)
                         for value in np.linspace(1.0, 0.0, nz + 1)),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(HrrrTargetDomain.legacy_500x500(), nz=nz)
    return _experiment(
        vertical, run_seconds=3600.0, target=target,
        physics_profile=profile, **kwargs)


#: Every shipped profile whose downward longwave is fabricated: no
#: longwave scheme, and a land-surface scheme that reads GLW every step.
#: Derived from the product's own classifier rather than listed by hand,
#: so a ninth profile joining the class arrives as coverage rather than
#: as a field report.
_FABRICATED_GLW_PROFILES = sorted(
    profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
    if downward_longwave_disposition(
        ra_lw_physics=int(single_domain_runtime_switches(profile).get(
            "ra_lw_physics", 0)),
        ra_sw_physics=int(single_domain_runtime_switches(profile).get(
            "ra_sw_physics", 0)),
        sf_surface_physics=int(single_domain_runtime_switches(profile).get(
            "sf_surface_physics", 0)))[0] in ("consumed", "published"))

#: 18Z over the route's reference point (-98 E) is early afternoon: an
#: hour from here reaches no local night at all.  03Z is the small hours.
_ALL_DAYLIGHT_START = datetime(2026, 7, 18, 18)
_NOCTURNAL_START = datetime(2026, 7, 18, 3)


@pytest.mark.parametrize("profile", _FABRICATED_GLW_PROFILES)
def test_an_all_daylight_window_still_declares_the_fabricated_longwave(
        profile):
    """Daylight does not make a frozen 300 W m-2 into a computed flux.

    The 1.8.8 assembly wrote BOTH radiation tokens below the night
    check, so every one of these eight profiles refused at config BUILD
    on an all-daylight window -- and this route synthesizes its config in
    code, so there was no file for an operator to write the declaration
    into.  Thirteen nodes of this file were red on it.

    Two claims, two conditions: the constant-longwave token rides the
    GLW disposition (which the sun has nothing to do with), the
    nocturnal token rides the window.  Asserting the nocturnal token is
    ABSENT here is the half that keeps the fix from degenerating into
    "declare everything always", which would make a published
    declaration mean nothing.
    """

    from gpuwm.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                      CONSTANT_DOWNWARD_LONGWAVE_ACK)

    experiment = _profile_experiment(profile,
                                     start_time=_ALL_DAYLIGHT_START)

    declared = tuple(experiment.acknowledgements or ())
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in declared, (
        f"{profile} fabricates its downward longwave on any window; the "
        f"declaration cannot be conditional on nightfall")
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in declared, (
        f"{profile} on an all-daylight window has no nocturnal claim to "
        f"make, and a token written by reflex declares nothing")


@pytest.mark.parametrize("profile", _FABRICATED_GLW_PROFILES)
def test_a_nocturnal_window_declares_both_radiation_claims(profile):
    """The night window carries both tokens, in the order it always did."""

    from gpuwm.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                      CONSTANT_DOWNWARD_LONGWAVE_ACK)

    experiment = _profile_experiment(profile, start_time=_NOCTURNAL_START)

    assert tuple(experiment.acknowledgements or ()) == (
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK, CONSTANT_DOWNWARD_LONGWAVE_ACK)


def test_a_symmetric_profile_declares_nothing_on_either_window():
    """The control: the route's own default carries no token at all.

    Without this the two tests above pass just as well on a route that
    declares both tokens unconditionally, which is the failure mode the
    1.8 default change was made to end.
    """

    for start_time in (_ALL_DAYLIGHT_START, _NOCTURNAL_START):
        experiment = _profile_experiment(ROUTE_DEFAULT_PHYSICS_PROFILE,
                                         start_time=start_time)
        assert tuple(experiment.acknowledgements or ()) == ()


@pytest.mark.parametrize("profile", sorted(SINGLE_DOMAIN_PHYSICS_PROFILES))
def test_the_hrrr_experiment_forwards_every_switch_its_profile_declares(
        profile):
    """The forwarded set covers the profile's whole declared switch set.

    F1: ``_experiment`` copied ``wrf_rrtmg_compatibility`` and dropped
    ``ra_rrtmg_variant``, so the ratified
    ``nssl2-...-rrtmg-legacy-...-v1`` profile -- declared for the HRRR
    route and offered in this runner's ``--physics-profile`` choices --
    could not be built at all: the RunConfig kept the ``"rte-rrtmgp"``
    default and the paired-switch validator refused it in 0.7 s.  Its RTE
    sibling built only because its declared value equalled that default,
    which is exactly why a single-profile test would not have caught this.
    Parametrizing over every shipped profile kills the class.
    """

    switches = single_domain_runtime_switches(profile)
    run = _profile_experiment(profile).root.run

    missing = sorted(set(switches) - set(hrrr_runner._PROFILE_SWITCH_HOMES))
    assert missing == []
    observed = {name: getattr(run, name, "<absent from RunConfig>")
                for name in switches}
    assert observed == switches


def test_the_hrrr_switch_forward_refuses_a_profile_switch_it_has_no_home_for(
):
    """A profile that grows a switch fails loudly, not silently."""

    raw = {"shared": {}, "domain": [{}]}
    with pytest.raises(ValueError, match="no declared home"):
        hrrr_runner._forward_profile_switches(
            raw, {"ra_rrtmg_variant": "rrtmg_legacy",
                  "some_future_switch": 1})


def test_the_hrrr_route_builds_the_ratified_legacy_rrtmg_profile():
    """The exact instance F1 reported, and its RTE sibling beside it."""

    legacy = _profile_experiment(NSSL2_LEGACY_RRTMG_PROFILE_ID).root.run
    rte = _profile_experiment(NSSL2_PROFILE_ID).root.run

    assert legacy.wrf_rrtmg_compatibility == WRF_RRTMG_LEGACY
    assert legacy.ra_rrtmg_variant == "rrtmg_legacy"
    assert rte.wrf_rrtmg_compatibility == WRF_RRTMG_TO_RTE_RRTMGP
    assert rte.ra_rrtmg_variant == "rte-rrtmgp"


def test_native_hrrr_nssl2_profile_rejects_surrounding_suite_drift(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=18, full_real74_suite=True, cu_physics=0)
    with pytest.raises(ValueError, match="cu_physics.*must be 1"):
        _validate_native_hrrr_physics_profile(path, NSSL2_PROFILE_ID)


def test_native_hrrr_nssl2_accepts_equivalent_explicit_optional_selectors(
        tmp_path):
    path = tmp_path / "namelist.input"
    controls = """ nssl_2moment_on = 1, -1,
 nssl_hail_on = 1, -1,
 nssl_ccn_on = 1, -1,
 nssl_density_on = 2, -1,
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
    (" nssl_hail_on = 0, -1,\n", "fixed preset does not implement"),
    (" nssl_rho_qhl = 800, 900,\n", "nssl_rho_qhl=900.0"),
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


class _ReferenceVerticalPlan:
    def __init__(self, source, surface, target):
        self.source = np.asarray(source, dtype=np.float32)
        self.surface = np.asarray(surface, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def apply(self, field, surface_value, **options):
        from gpuwm.verify.npref import np_wrf_real_vert_interp

        options.pop("values_are_finite", None)
        return np.asarray(np_wrf_real_vert_interp(
            field, surface_value, self.source, self.surface, self.target,
            **options), dtype=np.float32)


class _ReferencePreprocessBackend:
    """Independent CPU implementation of the production preprocessing ABI."""

    name = "cpu-reference-test"
    array_module = np

    @staticmethod
    def float32(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def regular_plan(*args, **kwargs):
        raise AssertionError("regular horizontal preprocessing is unused here")

    masked_nearest = regular_plan
    rotate_earth_to_grid = regular_plan
    era5_rh_to_water = regular_plan

    @staticmethod
    def prepare_wrf_vertical(source, surface, target):
        return _ReferenceVerticalPlan(source, surface, target)

    @staticmethod
    def receipt():
        return {"backend": "cpu-reference-test"}


def _decoded_native_hrrr_initialization(
        mp_physics, *, analyzed_species=None,
        analyzed_source_levels=None):
    """Drive decoded native fields through real initialization, never a fake state.

    ``analyzed_species`` selects which of QC/QR/QI/QS/QG carry mass in the
    decoded source.  It defaults to all five; passing a subset (or none)
    reproduces the meteorologically empty analysis the reopening battery hit,
    where four of five species had nothing to retain and the retention gate
    could not fire.
    """

    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.ingest.horiz import HorizontalSnapshot
    from gpuwm.ingest.real import initialize_real

    ny, nx, nz = 2, 3, 8
    levels = np.array(
        [100.0, 300.0, 500.0, 700.0, 850.0, 1000.0],
        dtype=np.float64)
    pressure = np.broadcast_to(
        levels[:, None, None] * 100.0, (levels.size, ny, nx)).copy()
    temperature = np.broadcast_to(
        215.0 + 75.0 * (pressure / 100000.0) ** 0.22,
        pressure.shape).copy()
    height = np.broadcast_to(
        -7900.0 * np.log(pressure / 100000.0), pressure.shape).copy()
    level_index = np.arange(levels.size, dtype=np.float32)[:, None, None]
    row_index = np.arange(ny, dtype=np.float32)[None, :, None]
    column_index = np.arange(nx, dtype=np.float32)[None, None, :]
    carried = (("QC", "QR", "QI", "QS", "QG") if analyzed_species is None
               else tuple(analyzed_species))
    if analyzed_source_levels is not None:
        carried = tuple(analyzed_source_levels)
    analyzed = {}
    for species_index, name in enumerate(("QC", "QR", "QI", "QS", "QG"), 1):
        if name not in carried:
            analyzed[name] = np.zeros(pressure.shape, dtype=np.float32)
            continue
        if analyzed_source_levels is not None:
            value = np.zeros(pressure.shape, dtype=np.float32)
            value[int(analyzed_source_levels[name])] = np.float32(
                species_index * 1.0e-6)
            analyzed[name] = value
            continue
        value = np.asarray(
            species_index * 1.0e-6
            * (1.0 + level_index + 0.25 * row_index
               + 0.125 * column_index),
            dtype=np.float32)
        value[0, 0, 0] = np.float32(0.0)
        analyzed[name] = value
    fields = {
        "PRES": pressure.astype(np.float32),
        "SPFH": np.full(pressure.shape, 0.004, dtype=np.float32),
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, dtype=np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, dtype=np.float32),
        "PSFC": np.full((ny, nx), 100000.0, dtype=np.float32),
        "T2": np.full((ny, nx), 289.0, dtype=np.float32),
        "Q2": np.full((ny, nx), 0.004, dtype=np.float32),
        "U10": np.full((ny, nx + 1), 7.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -1.0, dtype=np.float32),
        **analyzed,
    }
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 7, 20, 6),
        levels_hpa=levels,
        fields=fields,
    )
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0,
        ztop=18000.0, dt=30.0, run_seconds=60.0,
        hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        mp_physics=mp_physics,
    )
    eta = np.linspace(1.0, 0.0, nz + 1)
    terrain = np.zeros((ny, nx), dtype=np.float64)
    return initialize_real(
        snapshot, cfg,
        make_vertical_coord(
            nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
        terrain,
        source_orography=terrain,
        p_top=10000.0,
        use_sh_qv=True,
        preprocess_backend=_ReferencePreprocessBackend(),
        state_backend="cpu",
    )


@pytest.mark.parametrize(
    ("mp_physics", "retained", "discarded"),
    (
        (6, {"QC", "QR", "QI", "QS", "QG"}, set()),
        (18, {"QC", "QR", "QI", "QS", "QG"}, set()),
        (1, {"QC", "QR"}, {"QI", "QS", "QG"}),
    ),
)
def test_native_hrrr_hydrometeor_admission_is_scheme_aware(
        mp_physics, retained, discarded):
    result = _decoded_native_hrrr_initialization(mp_physics)
    evidence = result.hydrometeor_initialization
    assert set(evidence["retained_correspondence"]) == retained
    assert set(evidence["discarded_source_species"]) == discarded
    assert set(evidence["decoded_source_species"]) == {
        "QC", "QR", "QI", "QS", "QG"}
    assert all(
        evidence["initialized_state_species"][state_name][
            "nonzero_count"] > 0
        for state_name in evidence["retained_correspondence"].values()
    )
    if mp_physics == 1:
        assert getattr(result.state, "qi", None) is None
        for policy in evidence["discarded_source_species"].values():
            assert policy["policy"] == (
                "discard-source-species-absent-from-active-moist-package")
            assert policy["wrf_commit"] == (
                "d66e442fccc04111067e29274c9f9eaccc3cef28")


def test_the_retention_receipt_states_when_it_proved_nothing():
    """The battery's vacuous-receipt lesson, bound to the producer.

    The first NSSL/RTE probe passed this receipt on a domain where four of
    five analyzed species were identically zero: the gate is ``source
    nonzero > 0 and live nonzero == 0 -> raise``, so it could not fire, and
    the receipt did not say so.  A green receipt on that domain was not
    evidence that B-H1 was fixed -- it would have passed with the fix
    reverted for QR/QI/QS/QG.  Now the receipt grades its own evidence.
    """

    # A cloud-free analysis: nothing to retain for any species.
    empty = _decoded_native_hrrr_initialization(18, analyzed_species=())
    receipt = _initial_hrrr_microphysics_receipt(
        empty.state, NSSL2_PROFILE_ID, empty.hydrometeor_initialization)
    assert receipt["retention_evidence_summary"]["strength"] == "VACUOUS"
    assert receipt["retention_evidence_summary"]["vacuous_species"] == [
        "QC", "QG", "QI", "QR", "QS"]
    assert receipt["retention_evidence_summary"]["proven_species"] == []
    assert "proves nothing" in \
        receipt["retention_evidence_summary"]["statement"]
    for species in ("QC", "QG", "QI", "QR", "QS"):
        evidence = receipt["retention_evidence"][species]
        assert evidence["strength"] == "VACUOUS"
        assert evidence["source_nonzero_count"] == 0
        assert "not proven" in evidence["reason"]

    # The battery's actual Ohio domain: QC alone carried mass.
    partial = _decoded_native_hrrr_initialization(
        18, analyzed_species=("QC",))
    receipt = _initial_hrrr_microphysics_receipt(
        partial.state, NSSL2_PROFILE_ID, partial.hydrometeor_initialization)
    summary = receipt["retention_evidence_summary"]
    assert summary["strength"] == "PARTIALLY_WRF_EXCLUDED"
    assert summary["proven_species"] == []
    assert summary["partially_excluded_species"] == ["QC"]
    assert summary["vacuous_species"] == ["QG", "QI", "QR", "QS"]
    assert receipt["retention_evidence"]["QC"]["strength"] == (
        "PARTIALLY_WRF_EXCLUDED")

    # The control: all five species carry target-influencing mass, while the
    # surface-close source row is explicitly excluded by WRF for each.
    full = _decoded_native_hrrr_initialization(18)
    receipt = _initial_hrrr_microphysics_receipt(
        full.state, NSSL2_PROFILE_ID, full.hydrometeor_initialization)
    assert receipt["retention_evidence_summary"]["strength"] == (
        "PARTIALLY_WRF_EXCLUDED")
    assert receipt["retention_evidence_summary"]["vacuous_species"] == []
    assert all(
        evidence["strength"] == "PARTIALLY_WRF_EXCLUDED"
        and evidence["source_nonzero_count"] > 0
        and evidence["state_nonzero_count"] > 0
        for evidence in receipt["retention_evidence"].values())


def test_sparse_wrf_exclusion_is_accepted_only_with_complete_v2_evidence():
    result = _decoded_native_hrrr_initialization(
        18, analyzed_source_levels={"QC": 5})

    receipt = _initial_hrrr_microphysics_receipt(
        result.state, NSSL2_PROFILE_ID, result.hydrometeor_initialization)

    qc = receipt["retention_evidence"]["QC"]
    assert qc["strength"] == "WRF_EXCLUDED"
    assert qc["source_nonzero_count"] == 6
    assert qc["target_influencing_source_count"] == 0
    assert qc["wrf_excluded_source_count"] == 6
    assert qc["state_nonzero_count"] == 0
    assert receipt["retention_evidence_summary"]["excluded_species"] == [
        "QC"]

    missing = copy.deepcopy(result.hydrometeor_initialization)
    missing.pop("vertical_disposition")
    with pytest.raises(ValueError, match="disposition evidence is missing"):
        _initial_hrrr_microphysics_receipt(
            result.state, NSSL2_PROFILE_ID, missing)

    # Schema v1 retains the historical broad predicate by design.  The same
    # sparse source/state pair is refused when the exhaustive v2 partition is
    # absent; old receipts are not silently reinterpreted under new rules.
    legacy = copy.deepcopy(result.hydrometeor_initialization)
    legacy["schema"] = "gpuwm-real-hydrometeor-correspondence-v1"
    legacy.pop("vertical_disposition")
    with pytest.raises(ValueError, match="lost all nonzero mass"):
        _initial_hrrr_microphysics_receipt(
            result.state, NSSL2_PROFILE_ID, legacy)


def test_native_hrrr_mp_off_refuses_unfaithful_analyzed_cloud_state():
    with pytest.raises(
            ValueError,
            match="cannot faithfully retain analyzed QC/QR/QI/QS/QG"):
        _decoded_native_hrrr_initialization(0)


def test_hrrr_thompson_initialization_requires_mass_and_exact_zero_numbers():
    result = _decoded_native_hrrr_initialization(8)
    state = result.state
    receipt = _initial_hrrr_microphysics_receipt(
        state, THOMPSON_PROFILE_ID, result.hydrometeor_initialization)
    assert receipt["state_number_fields"] == {
        "ni": {"all_exact_zero": True},
        "nr": {"all_exact_zero": True},
    }
    state.nr.flat[0] = np.float32(1.0)
    with pytest.raises(ValueError, match="number moment nr must initialize"):
        _initial_hrrr_microphysics_receipt(
            state, THOMPSON_PROFILE_ID,
            result.hydrometeor_initialization)


def test_hrrr_morrison_initialization_receipts_every_source_absent_moment():
    result = _decoded_native_hrrr_initialization(10)
    state = result.state

    receipt = _initial_hrrr_microphysics_receipt(
        state, MORRISON_PROFILE_ID, result.hydrometeor_initialization)

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
        _initial_hrrr_microphysics_receipt(
            state, MORRISON_PROFILE_ID,
            result.hydrometeor_initialization)


def test_hrrr_nssl2_initialization_receipts_zero_state_and_background_ccn():
    result = _decoded_native_hrrr_initialization(18)
    state = result.state
    zero_fields = (
        "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh",
        "qvolg", "qvolh",
    )

    receipt = _initial_hrrr_microphysics_receipt(
        state, NSSL2_PROFILE_ID, result.hydrometeor_initialization)

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
        _initial_hrrr_microphysics_receipt(
            state, NSSL2_PROFILE_ID,
            result.hydrometeor_initialization)


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
    mp_physics = 10 if profile == MORRISON_PROFILE_ID else 18
    result = _decoded_native_hrrr_initialization(mp_physics)

    receipt = _initial_hrrr_microphysics_receipt(
        result.state, profile, result.hydrometeor_initialization)

    assert set(receipt["state_source_absent_fields"]) == expected_fields
    assert receipt["state_source_absent_fields"].get(
        "qnn", {"expected_float32": 0.0})["expected_float32"] in (
            0.0, 408163264.0)


def test_hrrr_receipt_rejects_zeroed_state_and_pre_fix_mp18_behavior():
    result = _decoded_native_hrrr_initialization(18)
    state = result.state
    _initial_hrrr_microphysics_receipt(
        state, NSSL2_PROFILE_ID, result.hydrometeor_initialization)

    for name in ("qc", "qr", "qi", "qs", "qg"):
        getattr(state, name)[...] = np.float32(0.0)
    with pytest.raises(ValueError, match="source-to-state correspondence"):
        _initial_hrrr_microphysics_receipt(
            state, NSSL2_PROFILE_ID, result.hydrometeor_initialization)

    # Regression anchor for the old mp_physics not-in-(6,8,10) path: it
    # produced the same all-zero MP18 state and no source correspondence.
    with pytest.raises(ValueError, match="lacks decoded-source"):
        _initial_hrrr_microphysics_receipt(
            state, NSSL2_PROFILE_ID, None)


def _outdir_args(outdir, *extra):
    args = [token for token in _required_args()]
    args[args.index("--outdir") + 1] = str(outdir)
    return args + list(extra)


def test_an_existing_outdir_is_refused_in_a_sentence_not_a_traceback(
        tmp_path, capsys):
    """`mkdir(exist_ok=False)` fired mid-setup as a bare FileExistsError,
    after the target domain and the source window had been resolved -- a
    field run of the shipped 1.5.0 wheel met exactly that.  The same file
    created its FAILURE report with exist_ok=True, so the two halves
    disagreed about whether an existing directory was allowed."""

    outdir = tmp_path / "already-here"
    outdir.mkdir()
    with pytest.raises(SystemExit) as raised:
        _parse_args(_outdir_args(outdir))
    assert raised.value.code == 2
    message = capsys.readouterr().err
    assert str(outdir) in message
    assert "already exists" in message
    assert "--allow-existing" in message
    assert "Traceback" not in message


def test_a_fresh_outdir_is_accepted_unchanged(tmp_path):
    args = _parse_args(_outdir_args(tmp_path / "fresh"))
    assert args.outdir == tmp_path / "fresh"
    assert args.allow_existing is False
    assert not (tmp_path / "fresh").exists()  # parsing creates nothing


def test_allow_existing_reuses_a_directory_that_holds_unrelated_files(
        tmp_path):
    outdir = tmp_path / "mine"
    outdir.mkdir()
    (outdir / "notes.txt").write_text("kept", encoding="utf-8")
    args = _parse_args(_outdir_args(outdir, "--allow-existing"))
    assert args.allow_existing is True
    assert (outdir / "notes.txt").read_text(encoding="utf-8") == "kept"


@pytest.mark.parametrize(
    "name", ("report.json", "progress.json", "wrfout_d01_2026-08-03_12_00_00"))
def test_allow_existing_still_refuses_to_overwrite_a_previous_result(
        tmp_path, capsys, name):
    """"Warn, don't block" is house style; silently replacing a benchmark
    result is not warning, it is losing the measurement."""

    outdir = tmp_path / "reused"
    outdir.mkdir()
    (outdir / name).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _parse_args(_outdir_args(outdir, "--allow-existing"))
    assert raised.value.code == 2
    message = capsys.readouterr().err
    assert str(outdir) in message and name in message
    assert "not" in message and "overwritten" in message
    # The file is still there: a refusal that deleted the thing it was
    # protecting would be worse than the clobber.
    assert (outdir / name).exists()


def test_an_outdir_that_is_a_file_is_named_as_such(tmp_path, capsys):
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        _parse_args(_outdir_args(occupied, "--allow-existing"))
    assert "is not a directory" in capsys.readouterr().err


def test_the_device_name_is_decoded_text_not_a_bytes_repr():
    """A 1.5.0 field report carries `b'NVIDIA GeForce RTX 5070 Ti'` --
    quotes, b prefix and all -- in the field whose whole job is to say
    which card produced the numbers."""
    from tools.hrrr_single_domain_benchmark import _device_name

    class _Runtime:
        @staticmethod
        def getDeviceProperties(index):
            return {"name": b"NVIDIA GeForce RTX 5070 Ti"}

    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=_Runtime))
    assert _device_name(cp) == "NVIDIA GeForce RTX 5070 Ti"
    assert not _device_name(cp).startswith("b'")


def test_an_undecodable_device_name_degrades_instead_of_raising():
    from tools.hrrr_single_domain_benchmark import _device_name

    class _Runtime:
        @staticmethod
        def getDeviceProperties(index):
            return {"name": b"RTX \xff\xfe"}

    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=_Runtime))
    assert _device_name(cp).startswith("RTX ")


def test_an_already_decoded_device_name_passes_through():
    from tools.hrrr_single_domain_benchmark import _device_name

    class _Runtime:
        @staticmethod
        def getDeviceProperties(index):
            return {"name": "NVIDIA GeForce RTX 5090"}

    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=_Runtime))
    assert _device_name(cp) == "NVIDIA GeForce RTX 5090"


def test_the_cold_first_step_is_reported_apart_from_the_warmed_rate():
    """Reproduces the shipped-wheel field run: 900 simulated seconds in
    60 steps, 41.2476 s of integration, a 22.5562 s first step because
    step one pays for every NVRTC compilation the forecast needs.  The
    whole-run average is 2.7498 wall s per simulated minute; the same run
    excluding step one is 1.2672, and it is the second number that
    predicts what an hour of forecast costs."""
    from tools.hrrr_single_domain_benchmark import _warmed_integration_rates

    rates = _warmed_integration_rates(
        integration_seconds=41.247564, run_seconds=900.0, steps=60,
        step_samples=[22.556234, 0.283061, 0.283068])

    assert rates["cold_first_step_seconds"] == pytest.approx(22.556234)
    assert rates["warmed_integration_excluded_steps"] == 1
    assert rates["warmed_integration_seconds"] == pytest.approx(18.69133)
    # 1.2672 wall s per simulated MINUTE, as the field report computed it.
    per_minute = rates[
        "warmed_integration_wall_seconds_per_simulated_hour"] / 60.0
    assert per_minute == pytest.approx(1.2672, abs=5e-5)
    # And the whole-run figure this sits beside is the larger one.
    assert (900.0 / 41.247564) / 60.0 == pytest.approx(0.3637, abs=5e-5)
    assert rates["warmed_integration_simulated_seconds_per_wall_second"] \
        == pytest.approx(885.0 / 18.69133)


@pytest.mark.parametrize(
    ("steps", "samples", "integration"),
    ((1, [3.0], 3.0),          # no warmed population at all
     (60, [], 41.0),           # no per-step samples collected
     (60, [99.0], 41.0)),      # cold step exceeds the whole integration
)
def test_a_warmed_rate_with_no_meaning_is_none_not_a_fabricated_number(
        steps, samples, integration):
    from tools.hrrr_single_domain_benchmark import _warmed_integration_rates

    rates = _warmed_integration_rates(
        integration_seconds=integration, run_seconds=900.0, steps=steps,
        step_samples=samples)
    assert rates["warmed_integration_simulated_seconds_per_wall_second"] \
        is None
    assert rates["warmed_integration_wall_seconds_per_simulated_hour"] is None


def test_the_printed_summary_carries_both_rates(tmp_path, monkeypatch, capsys):
    """stdout is what a short run gets extrapolated from, so the split
    has to reach it -- and the whole-run field keeps its name and its
    value beside the new ones."""
    outdir = tmp_path / "run"

    def fake_run(args):
        args.outdir.mkdir(parents=True, exist_ok=True)
        return {
            "schema": "gpuwm-native-hrrr-benchmark-v2", "status": "PASS",
            "run_seconds": 900.0, "io_mode": "history",
            "history_interval_seconds": 900.0,
            "memory": {"gpu_peak_used_bytes_observed": 9_304_276_992},
            "integration_simulated_seconds_per_wall_second": 900.0 / 41.247564,
            "cold_first_step_seconds": 22.556234,
            "warmed_integration_simulated_seconds_per_wall_second":
                885.0 / 18.69133,
            "prepared_cache": None,
        }

    monkeypatch.setattr(hrrr_runner, "run", fake_run)
    assert hrrr_runner_main(_outdir_args(
        outdir, "--io-mode", "history",
        "--history-interval-seconds", "900")) == 0
    summary = json.loads(capsys.readouterr().out)

    assert summary["simulated_seconds_per_wall_second"] == pytest.approx(
        900.0 / 41.247564)
    assert summary["cold_first_step_seconds"] == pytest.approx(22.556234)
    assert summary["warmed_simulated_seconds_per_wall_second"] \
        == pytest.approx(885.0 / 18.69133)
    # The warmed rate is the larger one; a reader extrapolating from the
    # whole-run figure alone under-counts this machine by a factor of 2.
    assert (summary["warmed_simulated_seconds_per_wall_second"]
            > summary["simulated_seconds_per_wall_second"])


@pytest.mark.parametrize("profile", sorted(SINGLE_DOMAIN_PHYSICS_PROFILES))
def test_every_shipped_profile_resolves_every_per_profile_table(profile):
    """B-04's refusal class, closed for the 13th profile too.

    The battery's first case run passed every static gate and every prior
    stage, then refused at root preparation: SINGLE_DOMAIN_PHYSICS_PROFILES
    had grown to 12 entries while _NATIVE_HRRR_NAMELIST_CONTRACTS held 10,
    and only the NSSL-2 legacy twin had an alias -- at three separate
    inline sites the Thompson twin missed.  This walk resolves EVERY
    shipped profile through every per-profile table this runner (and the
    root-preparation wrapper) keys by profile id, so the next profile
    cannot repeat the class.

    The value leg guards the opposite regression: the namelist contract is
    a hard per-field equality gate on the supplied namelist, so its values
    must equal the profile's own runtime switches -- a Thompson twin
    "simplified" to a bare alias of its base row would pin radt 1.0
    against the profile's 12.0 and refuse the battery's namelist one gate
    later.
    """

    from tools import prepare_hrrr_wrf

    contract = hrrr_runner._native_hrrr_profile_contract(profile)
    assert set(contract) == {"physics", "dynamics"}
    switches = hrrr_runner._native_hrrr_runtime_switches(profile)

    init_profile = hrrr_runner._initialization_contract_profile(profile)
    defaults = hrrr_runner._HRRR_SOURCE_ABSENT_STATE_DEFAULTS[init_profile]
    fields = hrrr_runner._HRRR_SOURCE_ABSENT_WRF_FIELDS[init_profile]
    assert isinstance(list(fields), list)
    assert isinstance(dict(defaults), dict)
    assert profile in prepare_hrrr_wrf._HRRR_COLD_START_CONTRACT
    assert profile in hrrr_runner.runner_capabilities()["physics_profiles"]

    for section, entries in contract.items():
        for key, expected in entries.items():
            name = {"cudt": "cudt_minutes"}.get(key, key)
            assert name in switches, (profile, section, key)
            observed = switches[name]
            if isinstance(expected, bool):
                assert observed is expected, (profile, section, key)
            else:
                assert float(observed) == float(expected), (
                    profile, section, key, observed, expected)


def test_battery_composition_namelist_passes_the_profile_contract(tmp_path):
    """The exact gate the 2026-08-04 case run refused at, CPU-side.

    The battery composition's rendered native namelist, validated through
    _validate_native_hrrr_physics_profile under the legacy-RRTMG Thompson
    profile -- the call tools/prepare_hrrr_wrf.py makes at root
    preparation.  Before the twin's own contract row this raised
    'unsupported native HRRR physics profile'; a bare alias to the base
    Thompson row would instead refuse on radt/diff_6th_factor drift.
    """

    from pathlib import Path

    from tools import battery_wrf_node_plan as node_plan

    repo = Path(__file__).resolve().parents[1]
    config = (repo / "configs" / "battery"
              / "shape_3km_thompson_rrtmg_legacy.toml")
    outdir = tmp_path / "node"
    node_plan.build(config, outdir, ranks=24, repository_root=repo)

    receipt = hrrr_runner._validate_native_hrrr_physics_profile(
        outdir / "namelist.native.input",
        THOMPSON_LEGACY_RRTMG_PROFILE_ID)
    assert receipt["profile"] == THOMPSON_LEGACY_RRTMG_PROFILE_ID
    assert receipt["resolved"]["ra_rrtmg_variant"] == "rrtmg_legacy"
    assert receipt["resolved"]["radiation_scheme_ids"] == [4, 4]
    assert receipt["validated_namelist"]["physics"]["radt"] == 12.0
    assert receipt["validated_namelist"]["dynamics"][
        "diff_6th_factor"] == 0.12
    # The initialization contract is the Thompson one, verbatim -- a
    # microphysics property, radiation-variant-independent.
    contract = receipt["hrrr_initialization_contract"]
    assert contract["source_absent_fields"] == ["QNICE", "QNRAIN"]
    assert contract["source_absent_state_defaults_fp32"] == {
        "ni": 0.0, "nr": 0.0}
