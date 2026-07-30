"""Public HRRR wrapper source-window orchestration regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuwm.physics_compat import (
    MORRISON_PROFILE_ID,
    NSSL2_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    WSM6_PROFILE_ID,
)
from tools import prepare_hrrr_wrf as prepare


def _cold_start_receipt(profile: str) -> dict[str, object]:
    wrf_fields, state_values = {
        WSM6_PROFILE_ID: ([], {}),
        THOMPSON_PROFILE_ID: (
            ["QNICE", "QNRAIN"],
            {"ni": (0.0, 0), "nr": (0.0, 0)},
        ),
        MORRISON_PROFILE_ID: (
            ["QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL"],
            {name: (0.0, 0) for name in ("nc", "nr", "ni", "ns", "ng")},
        ),
        NSSL2_PROFILE_ID: (
            [
                "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
                "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
            ],
            {
                **{
                    name: (0.0, 0)
                    for name in (
                        "qh", "qndrop", "qnr", "qni", "qns", "qng",
                        "qnh", "qvolg", "qvolh",
                    )
                },
                "qnn": (408163264.0, 1304600734),
            },
        ),
    }[profile]
    return {
        "schema": "gpuwm-hrrr-microphysics-initialization-v1",
        "source_absent_wrf_fields": wrf_fields,
        "state_source_absent_fields": {
            name: {
                "expected_float32": value,
                "expected_uint32_bits": bits,
                "all_exact_expected": True,
            }
            for name, (value, bits) in state_values.items()
        },
    }


def test_public_wrapper_accepts_64_decoder_workers_independent_of_native_budget():
    report = {
        "status": "PASS",
        "preparation": {
            "preprocess_backend": {"backend": "cuda"},
            "preprocess_worker_budget": {
                "schema": "gpuwm-preprocess-worker-budget-v1",
                "backend": "cuda",
                "applicable": False,
                "pipeline_decoder_workers_included": False,
                "peak_active_native_workers": 0,
            },
        },
        "pipeline": {"workers": {"requested": "64", "selected": 64}},
    }

    _preprocess, native_budget, decoder = prepare._validated_worker_receipts(
        report, selected_backend="cuda", requested_preprocess_workers=None,
        requested_pipeline_workers="64", final_hour=12)

    assert decoder["selected"] == 64
    assert native_budget["pipeline_decoder_workers_included"] is False


@pytest.mark.parametrize(
    "physics_profile",
    (
        WSM6_PROFILE_ID,
        THOMPSON_PROFILE_ID,
        MORRISON_PROFILE_ID,
        NSSL2_PROFILE_ID,
    ),
)
def test_public_wrapper_preserves_absolute_source_leads_and_rebases_model_time(
        tmp_path: Path, monkeypatch, physics_profile: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for hour in (12, 13):
        (source / f"hrrr.t05z.wrfnatf{hour:02d}.grib2").write_bytes(
            f"atmos-{hour}".encode("ascii"))
        (source / f"hrrr.t05z.soilf{hour:02d}.grib2").write_bytes(
            f"soil-{hour}".encode("ascii"))
    source_manifest = source / "SHA256SUMS"
    source_manifest.write_text("fixture\n", encoding="ascii")
    static_cache = tmp_path / "static.npz"
    static_receipt = tmp_path / "static.json"
    namelist = tmp_path / "namelist.input"
    for path in (static_cache, static_receipt, namelist):
        path.write_bytes(b"fixture")
    decoder = tmp_path / "hrrr_grib2_bridge"
    decoder.write_bytes(b"decoder")
    monkeypatch.setattr(prepare, "_decoder", lambda _env: decoder)

    commands: list[list[str]] = []

    def fake_run(command: list[str], _env: dict[str, str]) -> None:
        commands.append(command)
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command):
            outdir = Path(command[command.index("--outdir") + 1])
            outdir.mkdir(parents=True)
            (outdir / "report.json").write_text(json.dumps({
                "status": "PASS",
                "history_interval_seconds": 3600.0,
                "physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": physics_profile,
                    "hrrr_initialization": _cold_start_receipt(
                        physics_profile),
                },
                "preparation": {
                    "preprocess_backend": {"backend": "cuda"},
                    "preprocess_worker_budget": {
                        "schema": "gpuwm-preprocess-worker-budget-v1",
                        "backend": "cuda",
                        "applicable": False,
                        "pipeline_decoder_workers_included": False,
                        "peak_active_native_workers": 0,
                    },
                },
                "pipeline": {"workers": {"requested": "8", "selected": 8}},
            }), encoding="utf-8")

    monkeypatch.setattr(prepare, "_run", fake_run)
    output = tmp_path / "output"
    assert prepare.main([
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", "0" * 64,
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist),
        "--physics-profile", physics_profile,
        "--valid-time", "2026-07-18_05:00:00",
        "--forecast-start-hour", "12",
        "--forecast-end-hour", "13",
        "--run-seconds", "3600",
        "--history-interval-seconds", "3600",
        "--output-root", str(output),
    ]) == 0

    series = output / "native" / "hrrr-f12-f13-series.tsv"
    assert [line.split("\t", 1)[0]
            for line in series.read_text(encoding="utf-8").splitlines()] \
        == ["12", "13"]
    benchmark = next(
        command for command in commands
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command))
    assert benchmark[benchmark.index("--forecast-start-hour") + 1] == "12"
    assert benchmark[benchmark.index("--forecast-end-hour") + 1] == "13"
    assert benchmark[benchmark.index("--physics-profile") + 1] \
        == physics_profile
    assert benchmark[benchmark.index("--history-interval-seconds") + 1] \
        == "3600.0"
    assert "--io-mode" not in benchmark
    export = next(command for command in commands if "gpuwm.wrf_direct" in command)
    assert export[export.index("--valid-time") + 1] == "2026-07-18_17:00:00"

    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert receipt["source_cycle"] == "2026-07-18T05:00:00"
    assert receipt["model_start_time"] == "2026-07-18T17:00:00"
    assert receipt["source_forecast_hours"] == [12, 13]
    assert receipt["model_forcing_hours"] == [0, 1]
    assert receipt["forcing_hours"] == [0, 1]
    assert receipt["history_interval_seconds"] == 3600.0
    assert receipt["physics"]["profile"] == physics_profile


def test_public_wrapper_rejects_mismatched_or_incomplete_physics_receipt():
    complete = {
        "physics": {
            "schema": "gpuwm-prepared-physics-profile-v1",
            "profile": MORRISON_PROFILE_ID,
            "hrrr_initialization": _cold_start_receipt(
                MORRISON_PROFILE_ID),
        },
    }
    assert prepare._validated_physics_receipt(
        complete, requested_profile=MORRISON_PROFILE_ID) \
        == complete["physics"]

    with pytest.raises(RuntimeError, match="differs from the request"):
        prepare._validated_physics_receipt(
            complete, requested_profile=NSSL2_PROFILE_ID)
    incomplete = json.loads(json.dumps(complete))
    del incomplete["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            incomplete, requested_profile=MORRISON_PROFILE_ID)

    wrong_fields = json.loads(json.dumps(complete))
    wrong_fields["physics"]["hrrr_initialization"][
        "source_absent_wrf_fields"] = []
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            wrong_fields, requested_profile=MORRISON_PROFILE_ID)

    false_exact = json.loads(json.dumps(complete))
    false_exact["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]["ng"]["all_exact_expected"] = False
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            false_exact, requested_profile=MORRISON_PROFILE_ID)

    wrong_bits = json.loads(json.dumps(complete))
    wrong_bits["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]["ng"]["expected_uint32_bits"] = 1
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            wrong_bits, requested_profile=MORRISON_PROFILE_ID)
