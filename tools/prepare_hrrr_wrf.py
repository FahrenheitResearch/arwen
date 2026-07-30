"""Cross-platform native HRRR preparation and stock-WRF export wrapper."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from gpuwm.physics_compat import (
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    MORRISON_PROFILE_ID,
    MYNN_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    NSSL2_PROFILE_ID,
    RUC_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    WSM6_PROFILE_ID,
)
from gpuwm.hrrr_forecast import hrrr_source_window


REPO = Path(__file__).resolve().parents[1]
MAX_PIPELINE_WORKERS = 64

_HRRR_COLD_START_CONTRACT = {
    WSM6_PROFILE_ID: ((), {}),
    MYNN_PROFILE_ID: ((), {}),
    RUC_PROFILE_ID: ((), {}),
    NOAHMP_PROFILE_ID: ((), {}),
    THOMPSON_PROFILE_ID: (
        ("QNICE", "QNRAIN"),
        {"ni": (0.0, 0), "nr": (0.0, 0)},
    ),
    MORRISON_PROFILE_ID: (
        ("QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL"),
        {
            "nc": (0.0, 0),
            "nr": (0.0, 0),
            "ni": (0.0, 0),
            "ns": (0.0, 0),
            "ng": (0.0, 0),
        },
    ),
    NSSL2_PROFILE_ID: (
        (
            "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
            "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
        ),
        {
            "qh": (0.0, 0),
            "qndrop": (0.0, 0),
            "qnr": (0.0, 0),
            "qni": (0.0, 0),
            "qns": (0.0, 0),
            "qng": (0.0, 0),
            "qnh": (0.0, 0),
            "qnn": (408163264.0, 1304600734),
            "qvolg": (0.0, 0),
            "qvolh": (0.0, 0),
        },
    ),
}


def _run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, cwd=REPO, env=env)


def _decoder(env: dict[str, str]) -> Path:
    requested = env.get("GPUWM_HRRR_DECODER")
    if requested:
        path = Path(requested).resolve()
    else:
        name = "hrrr_grib2_bridge.exe" if os.name == "nt" else "hrrr_grib2_bridge"
        path = REPO / "tools" / "grib1_bridge" / "target" / "release" / name
        if not path.is_file():
            _run([
                "cargo", "build", "--release", "--manifest-path",
                str(REPO / "tools" / "grib1_bridge" / "Cargo.toml"),
                "--bin", "hrrr_grib2_bridge",
            ], env)
    if not path.is_file():
        raise FileNotFoundError(f"HRRR decoder is missing: {path}")
    return path


def _valid_time(raw: str) -> datetime:
    value = datetime.strptime(raw, "%Y-%m-%d_%H:%M:%S")
    if value.minute or value.second:
        raise ValueError("valid time must be an exact hourly HRRR cycle")
    return value


def _positive_finite_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "history-interval-seconds must be a number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            "history-interval-seconds must be finite and positive")
    return value


def _validated_worker_receipts(
        preparation_report: dict[str, object], *, selected_backend: str,
        requested_preprocess_workers: int | None,
        requested_pipeline_workers: str,
        final_hour: int) -> tuple[dict, dict, dict]:
    """Validate the public CPU budget and independent decoder receipt."""

    try:
        preprocess_receipt = preparation_report["preparation"][
            "preprocess_backend"]
        preprocess_worker_budget = preparation_report["preparation"][
            "preprocess_worker_budget"]
        pipeline_worker_receipt = preparation_report["pipeline"]["workers"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "HRRR preparation omitted a worker/backend receipt") from exc
    if (preparation_report.get("status") != "PASS"
            or not isinstance(preprocess_receipt, dict)
            or preprocess_receipt.get("backend") not in {"cpu", "cuda"}):
        raise RuntimeError("HRRR preparation backend receipt is malformed")
    if (selected_backend != "auto"
            and preprocess_receipt["backend"] != selected_backend):
        raise RuntimeError(
            "HRRR preparation backend differs from the public selector")
    if (requested_preprocess_workers is not None
            and preprocess_receipt["backend"] == "cpu"
            and preprocess_receipt.get("workers")
            != requested_preprocess_workers):
        raise RuntimeError(
            "HRRR CPU preparation worker receipt differs from the request")
    if (not isinstance(preprocess_worker_budget, dict)
            or preprocess_worker_budget.get("schema")
            != "gpuwm-preprocess-worker-budget-v1"
            or preprocess_worker_budget.get("backend")
            != preprocess_receipt["backend"]
            or preprocess_worker_budget.get("applicable")
            is not (preprocess_receipt["backend"] == "cpu")
            or preprocess_worker_budget.get(
                "pipeline_decoder_workers_included") is not False):
        raise RuntimeError("HRRR preprocessing worker-budget receipt is malformed")
    if preprocess_receipt["backend"] == "cpu":
        effective_total = preprocess_worker_budget.get(
            "effective_total_native_workers")
        peak_native = preprocess_worker_budget.get(
            "peak_active_native_workers")
        jobs = preprocess_worker_budget.get("effective_allocation_per_job")
        if (isinstance(effective_total, bool)
                or not isinstance(effective_total, int)
                or effective_total < 1
                or isinstance(peak_native, bool)
                or not isinstance(peak_native, int)
                or peak_native < 1
                or effective_total != preprocess_receipt.get("workers")
                or not isinstance(jobs, list)
                or len(jobs) != 1 + 2 * final_hour
                or any(
                    not isinstance(job, dict)
                    or isinstance(job.get("effective_native_workers"), bool)
                    or not isinstance(
                        job.get("effective_native_workers"), int)
                    or not 1 <= job["effective_native_workers"] <= effective_total
                    for job in jobs)
                or peak_native > effective_total):
            raise RuntimeError(
                "HRRR CPU preprocessing worker budget was not honored")
        if (requested_preprocess_workers is not None
                and preprocess_worker_budget.get(
                    "requested_total_native_workers")
                != requested_preprocess_workers):
            raise RuntimeError(
                "HRRR CPU preprocessing budget differs from the request")
    elif preprocess_worker_budget.get("peak_active_native_workers") != 0:
        raise RuntimeError("CUDA preprocessing reported native CPU workers")
    selected_pipeline_workers = (
        pipeline_worker_receipt.get("selected")
        if isinstance(pipeline_worker_receipt, dict) else None)
    if (not isinstance(pipeline_worker_receipt, dict)
            or pipeline_worker_receipt.get("requested")
            != str(requested_pipeline_workers).strip().lower()
            or isinstance(selected_pipeline_workers, bool)
            or selected_pipeline_workers not in range(
                1, MAX_PIPELINE_WORKERS + 1)):
        raise RuntimeError("HRRR decoder worker receipt is malformed")
    return (preprocess_receipt, preprocess_worker_budget,
            pipeline_worker_receipt)


def _validated_physics_receipt(
        preparation_report: dict[str, object], *,
        requested_profile: str) -> dict[str, object]:
    """Retain one exact runner profile and its cold-start evidence."""

    physics = preparation_report.get("physics")
    if (not isinstance(physics, dict)
            or physics.get("schema") != "gpuwm-prepared-physics-profile-v1"
            or physics.get("profile") != requested_profile):
        raise RuntimeError(
            "HRRR preparation physics receipt differs from the request")
    initialization = physics.get("hrrr_initialization")
    expected_wrf_fields, expected_state_fields = \
        _HRRR_COLD_START_CONTRACT[requested_profile]
    if (not isinstance(initialization, dict)
            or initialization.get("schema")
            != "gpuwm-hrrr-microphysics-initialization-v1"
            or initialization.get("source_absent_wrf_fields")
            != list(expected_wrf_fields)
            or not isinstance(
                initialization.get("state_source_absent_fields"), dict)
            or set(initialization["state_source_absent_fields"])
            != set(expected_state_fields)):
        raise RuntimeError(
            "HRRR preparation omitted deterministic cold-start evidence")
    for name, (expected_float32, expected_uint32_bits) \
            in expected_state_fields.items():
        observed = initialization["state_source_absent_fields"][name]
        if (not isinstance(observed, dict)
                or set(observed) != {
                    "expected_float32", "expected_uint32_bits",
                    "all_exact_expected"}
                or isinstance(observed.get("expected_float32"), bool)
                or not isinstance(
                    observed.get("expected_float32"), (int, float))
                or float(observed["expected_float32"]) != expected_float32
                or isinstance(observed.get("expected_uint32_bits"), bool)
                or observed.get("expected_uint32_bits")
                != expected_uint32_bits
                or observed.get("all_exact_expected") is not True):
            raise RuntimeError(
                "HRRR preparation omitted deterministic cold-start evidence")
    return physics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    static = parser.add_mutually_exclusive_group(required=True)
    static.add_argument("--geog-root", type=Path)
    static.add_argument("--static-cache", type=Path)
    parser.add_argument("--static-receipt", type=Path)
    parser.add_argument("--domain-spec", type=Path)
    parser.add_argument("--namelist-input", type=Path, required=True)
    parser.add_argument(
        "--physics-profile",
        choices=SINGLE_DOMAIN_PHYSICS_PROFILES,
        default=WSM6_PROFILE_ID,
        help="explicit GPUWM HRRR physics/runtime contract",
    )
    parser.add_argument(
        "--expert-acknowledgement", action="append", default=[],
        help="registry-owned expert physics acknowledgement id; repeatable")
    parser.add_argument("--valid-time", required=True)
    parser.add_argument(
        "--forecast-start-hour", type=int, default=0,
        help="absolute HRRR cycle-relative source lead used for model time zero")
    parser.add_argument(
        "--forecast-end-hour", type=int,
        help="inclusive absolute source lead; must cover --run-seconds exactly")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-seconds", type=int, default=43_200)
    parser.add_argument(
        "--history-interval-seconds", type=_positive_finite_seconds,
        default=300.0,
        help=("future GPUWM history cadence bound into the prepared-cache "
              "identity; preparation itself writes no history frames"),
    )
    parser.add_argument("--pipeline-workers", default="8")
    parser.add_argument("--prepare-workers", type=int)
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="cuda")
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    valid_time = _valid_time(args.valid_time)
    source_forecast_hours = hrrr_source_window(
        cycle=valid_time, start_hour=args.forecast_start_hour,
        run_seconds=args.run_seconds, end_hour=args.forecast_end_hour)
    model_forcing_hours = tuple(range(len(source_forecast_hours)))
    final_hour = model_forcing_hours[-1]
    model_start_time = valid_time + timedelta(
        hours=source_forecast_hours[0])
    if args.prepare_workers is not None and args.prepare_workers not in range(1, 33):
        raise ValueError("prepare-workers must be between 1 and 32")
    if args.preprocess_workers is not None and args.preprocess_workers < 1:
        raise ValueError("preprocess-workers must be positive")
    if args.preprocess_backend == "cuda" and args.preprocess_workers is not None:
        raise ValueError(
            "preprocess-workers requires preprocess-backend cpu or auto")
    if (args.preprocess_backend != "cpu"
            and args.cpu_preprocess_bridge is not None):
        raise ValueError(
            "cpu-preprocess-bridge requires preprocess-backend cpu")
    try:
        pipeline_workers = int(args.pipeline_workers)
    except ValueError:
        if args.pipeline_workers.lower() != "auto":
            raise ValueError(
                f"pipeline-workers must be 1..{MAX_PIPELINE_WORKERS} or auto"
            ) from None
    else:
        if pipeline_workers not in range(1, MAX_PIPELINE_WORKERS + 1):
            raise ValueError(
                f"pipeline-workers must be 1..{MAX_PIPELINE_WORKERS} or auto")
    if len(args.source_manifest_sha256) != 64:
        raise ValueError("source-manifest-sha256 must contain 64 hexadecimal digits")
    int(args.source_manifest_sha256, 16)
    source_root = args.source_root.resolve()
    source_manifest = args.source_manifest.resolve()
    namelist_input = args.namelist_input.resolve()
    for path in (source_root, source_manifest, namelist_input):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.geog_root is not None and args.domain_spec is None:
        raise ValueError("domain-spec is required with geog-root")
    if args.static_cache is not None and args.static_receipt is None:
        raise ValueError("static-receipt is required with static-cache")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    output.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    decoder = _decoder(env)
    started = time.perf_counter()

    geometry_receipt = output / "native-geometry-receipt.json"
    if args.geog_root is not None:
        static_cache = output / "native-static.npz"
        static_receipt = output / "native-static-receipt.json"
        _run([
            sys.executable, str(REPO / "tools" / "hrrr_build_native_static.py"),
            "--geog-root", str(args.geog_root.resolve()),
            "--domain-spec", str(args.domain_spec.resolve()),
            "--output", str(static_cache), "--receipt", str(static_receipt),
        ], env)
    else:
        static_cache = args.static_cache.resolve()
        static_receipt = args.static_receipt.resolve()
        for path in (static_cache, static_receipt):
            if not path.is_file():
                raise FileNotFoundError(path)
    geometry_command = [
        sys.executable,
        str(REPO / "tools" / "write_hrrr_native_geometry_receipt.py"),
        "--static-cache", str(static_cache),
        "--hrrr-static-receipt", str(static_receipt),
        "--output", str(geometry_receipt),
    ]
    if args.domain_spec is not None:
        geometry_command.extend(("--domain-spec", str(args.domain_spec.resolve())))
    _run(geometry_command, env)
    static_seconds = time.perf_counter() - started

    native = output / "native"
    native.mkdir()
    series = native / (
        f"hrrr-f{source_forecast_hours[0]:02d}-"
        f"f{source_forecast_hours[-1]:02d}-series.tsv")
    rows = []
    for hour in source_forecast_hours:
        atmosphere = source_root / (
            f"hrrr.t{valid_time:%H}z.wrfnatf{hour:02d}.grib2")
        soil = source_root / f"hrrr.t{valid_time:%H}z.soilf{hour:02d}.grib2"
        for path in (atmosphere, soil):
            if not path.is_file():
                raise FileNotFoundError(path)
        rows.append(f"{hour}\t{atmosphere}\t{soil}\n")
    series.write_text("".join(rows), encoding="utf-8")
    benchmark = [
        sys.executable, str(REPO / "tools" / "hrrr_single_domain_benchmark.py"),
        "--bridge", str(native / "native-bridge"),
        "--valid-time", args.valid_time,
        "--forecast-start-hour", str(source_forecast_hours[0]),
        "--forecast-end-hour", str(source_forecast_hours[-1]),
        "--pipeline-series", str(series),
        "--pipeline-decoder", str(decoder),
        "--pipeline-signals", str(native / "pipeline-signals"),
        "--pipeline-workers", str(args.pipeline_workers),
        "--source-root", str(source_root),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", args.source_manifest_sha256.lower(),
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist_input),
        "--physics-profile", args.physics_profile,
        "--prepared-cache", str(native / "prepared-cache"),
        "--prepare-only", "--run-seconds", str(args.run_seconds),
        "--history-interval-seconds", str(args.history_interval_seconds),
        "--outdir", str(native / "preparation-report"),
    ]
    for acknowledgement in args.expert_acknowledgement:
        benchmark.extend((
            "--expert-acknowledgement", acknowledgement))
    if args.prepare_workers is not None:
        benchmark.extend(("--prepare-workers", str(args.prepare_workers)))
    benchmark.extend(("--preprocess-backend", args.preprocess_backend))
    if args.preprocess_workers is not None:
        benchmark.extend(("--preprocess-workers", str(args.preprocess_workers)))
    if args.cpu_preprocess_bridge is not None:
        benchmark.extend((
            "--cpu-preprocess-bridge",
            str(args.cpu_preprocess_bridge.resolve())))
    if args.domain_spec is not None:
        benchmark.extend(("--domain-spec", str(args.domain_spec.resolve())))
    prepare_started = time.perf_counter()
    _run(benchmark, env)
    prepare_seconds = time.perf_counter() - prepare_started
    preparation_report_path = native / "preparation-report" / "report.json"
    preparation_report = json.loads(
        preparation_report_path.read_text(encoding="utf-8"))
    observed_history_interval = preparation_report.get(
        "history_interval_seconds")
    if (isinstance(observed_history_interval, bool)
            or not isinstance(observed_history_interval, (int, float))
            or float(observed_history_interval)
            != float(args.history_interval_seconds)):
        raise RuntimeError(
            "HRRR prepared-cache history cadence differs from the request")
    (preprocess_receipt, preprocess_worker_budget,
     pipeline_worker_receipt) = _validated_worker_receipts(
         preparation_report,
         selected_backend=args.preprocess_backend,
         requested_preprocess_workers=args.preprocess_workers,
         requested_pipeline_workers=args.pipeline_workers,
         final_hour=final_hour)
    physics_receipt = _validated_physics_receipt(
        preparation_report, requested_profile=args.physics_profile)

    export_started = time.perf_counter()
    export_command = [
        sys.executable, "-m", "gpuwm.wrf_direct",
        "--prepared-cache", str(native / "prepared-cache"),
        "--static-cache", str(static_cache),
        "--geometry-receipt", str(geometry_receipt),
        "--output", str(output / "wrf-native-input"),
        "--valid-time", model_start_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "--physics-profile", args.physics_profile,
    ]
    for acknowledgement in args.expert_acknowledgement:
        export_command.extend((
            "--expert-acknowledgement", acknowledgement))
    _run(export_command, env)
    export_seconds = time.perf_counter() - export_started
    result = {
        "status": "PASS",
        "wrf_input": str(output / "wrf-native-input"),
        "source_cycle": valid_time.isoformat(),
        "model_start_time": model_start_time.isoformat(),
        "source_forecast_hours": list(source_forecast_hours),
        "model_forcing_hours": list(model_forcing_hours),
        # Backward-compatible alias: exporter/cache forcing is model-relative.
        "forcing_hours": list(model_forcing_hours),
        "history_interval_seconds": float(args.history_interval_seconds),
        "static_seconds": static_seconds,
        "prepare_seconds": prepare_seconds,
        "export_seconds": export_seconds,
        "total_seconds": time.perf_counter() - started,
        "preprocess_backend": preprocess_receipt,
        "preprocess_worker_budget": preprocess_worker_budget,
        "pipeline_decoder_workers": pipeline_worker_receipt,
        "preparation_report": str(preparation_report_path),
        "physics": physics_receipt,
    }
    (output / "public-wrapper-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
