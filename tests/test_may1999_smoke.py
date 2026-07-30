"""Frozen Phase-5 Task-6 ARC-A second-case acceptance gate.

The CPU test validates the complete declared-input catalog and the public
``gpuwm check`` path without requiring CUDA.  The controller-owned GPU test
then exercises static, ingest, and two full integrations (hourly and
30-minute output) from this TOML with no case-specific runtime code.
"""

from __future__ import annotations

import os

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.case_data import load_experiment_case
from gpuwm.ingest.preflight import (build_lbc_records, preflight_report)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "may1999_d01_smoke.toml"
STAGED = Path(os.environ.get("GPUWM_TEST_MAY99_DATA",
                    "gpuwm-fixture-unset/may99-data"))
BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
FORCING = (
    STAGED / "era5_may1999_pl.grib",
    STAGED / "era5_may1999_sl.grib",
)
EXPECTED_HASHES = {
    "era5_may1999_pl.grib":
        "c695442394b0154cb219951f81194f0842bf68000d8c99b35f7aec6ba8f1d1f7",
    "era5_may1999_sl.grib":
        "2f4af209c8daeea05c4a1bcad516dda3a2652f01be7aa692b75d24e7b6e64f4b",
}
VALID_TIMES = (
    datetime(1999, 5, 3, 12),
    datetime(1999, 5, 3, 18),
    datetime(1999, 5, 4, 0),
)
EXPECTED_INVENTORY = {
    "Z", "T", "U", "V", "RH", "U10", "V10", "T2", "D2",
    "LANDSEA", "PSFC", "PMSL", "SKINTEMP", "SST", "SEAICE",
    "SNOW_EC", "SOILGEO", "ST000007", "ST007028", "ST028100",
    "ST100289", "SM000007", "SM007028", "SM028100", "SM100289",
}

requires_inputs = pytest.mark.skipif(
    not (
        all(path.is_file() for path in FORCING)
        and (STAGED / "SHA256SUMS.txt").is_file()
        and (BUNDLE / "era5_grib" / "Vtable.ERA5_CDO").is_file()
        and (BUNDLE / "namelists" / "namelist.wps").is_file()
        and (BUNDLE / "static" / "WPS_GEOG").is_dir()
    ),
    reason="staged May-1999 ERA5 or shared ERA5/WPS resources are absent",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_numeric_finite(archive) -> None:
    checked = 0
    for name in archive.files:
        value = np.asarray(archive[name])
        if np.issubdtype(value.dtype, np.number):
            assert np.isfinite(value).all(), name
            checked += 1
    assert checked > 0


def _assert_wrf_tooling(path: Path, *, ny: int, nx: int) -> None:
    import wrf

    pathname_backend = hasattr(wrf, "WrfFile")
    source = (wrf.WrfFile(str(path)) if pathname_backend
              else netCDF4.Dataset(path))
    try:
        for name in ("slp", "T2"):
            value = np.asarray(wrf.getvar(source, name, meta=False))
            assert value.shape == (ny, nx), (path.name, name, value.shape)
            assert np.isfinite(value).all(), (path.name, name)
    finally:
        if not pathname_backend:
            source.close()


def _expected_output_names(start: datetime, run_seconds: float,
                           interval_s: float) -> list[str]:
    from gpuwm.io.wrfout import wrfout_filename

    return [
        wrfout_filename(start + timedelta(seconds=offset), domain_id=1)
        for offset in range(0, int(run_seconds) + 1, int(interval_s))
    ]


@requires_inputs
@pytest.mark.slow_acceptance
def test_may1999_config_and_gpuwm_check_are_complete_and_case_clean(capsys):
    """CPU-only frozen config/catalog/provenance half of the ARC-A gate."""
    from gpuwm import runtime
    from gpuwm.cli import main as gpuwm_main
    from gpuwm.core.rrtmgp import trace_gases

    exp, data = load_experiment_case(CONFIG)
    dc = exp.domain(1)
    assert exp.name == "may1999_d01_smoke"
    assert exp.start_time == VALID_TIMES[0]  # offset-free TOML means UTC
    assert exp.run_seconds == 21660.0
    assert (dc.run.nx, dc.run.ny, dc.run.nz, dc.run.dt, dc.run.dx) == (
        250, 200, 49, 60.0, 12000.0)
    assert exp.vertical.p_top == 10000.0
    assert len(exp.vertical.eta_levels) == 50
    assert dc.history_interval_s == 3600.0
    assert data.forcing == FORCING
    assert data.forcing_interval_s == 21600.0
    assert data.sfcp_to_sfcp is True
    assert data.source_orography is None  # era5_z_invariant provider
    assert data.co2_vmr is None  # date-indexed table, not a case override
    assert trace_gases(exp.start_time)["co2"] == pytest.approx(367.80e-6)
    assert data.output_domain == 1
    assert data.output_title == "gpuwm May 3 1999 d01 smoke"

    # The staged bundle is immutable input: pin both delivered bytes here.
    assert {path.name: _sha256(path) for path in data.forcing} == EXPECTED_HASHES

    report = preflight_report(exp, data)
    assert report.ok, report.format()
    catalog = report.catalog
    assert catalog.valid_times == VALID_TIMES
    assert len(catalog.levels_hpa) == 37
    assert catalog.levels_hpa[0] == 1.0
    assert catalog.levels_hpa[-1] == 1000.0
    assert set(catalog.inventory) == EXPECTED_INVENTORY
    assert catalog.spatial_coverage is not None
    assert catalog.spatial_coverage.shape == (149, 253)
    assert catalog.run_ceiling_seconds == 43200.0
    assert catalog.product_id == "ERA5"
    assert catalog.provenance["encoding"].startswith("native GRIB1")
    forcing_records = {
        item.path.name: item.sha256
        for item in catalog.files if item.role == "forcing"
    }
    assert forcing_records == EXPECTED_HASHES
    assert {item.path.parent for item in catalog.files
            if item.role in {"forcing", "forcing_provenance"}} == {STAGED}
    assert {item.path.name for item in catalog.files
            if item.role == "forcing_provenance"} == {
                "retrieve.py", "retrieve.log", "SHA256SUMS.txt"}

    records = build_lbc_records(catalog.valid_times)
    run_end = exp.start_time + timedelta(seconds=exp.run_seconds)
    assert records[0].end_time == records[1].start_time == VALID_TIMES[1]
    assert records[1].start_time < run_end < records[1].end_time

    # Exercise the public composed command.  A supplied CPU budget evaluates
    # the estimator-side memory leg without allocating a device.
    assert gpuwm_main(["check", str(CONFIG), "--budget-gib", "64"]) == 0
    check_log = capsys.readouterr().out
    assert "gpuwm input preflight: PASS" in check_log
    assert "valid_times=1999-05-03T12:00:00,1999-05-03T18:00:00,1999-05-04T00:00:00" in check_log
    assert "run_ceiling_seconds=43200" in check_log
    assert "alloc_estimate_le_wddm_budget: PASS" in check_log
    assert "FAIL" not in check_log

    resolved = runtime.resolved_config_report(
        exp, data, forcing_times=catalog.valid_times)
    provenance = json.dumps(catalog.run_provenance, default=str,
                            sort_keys=True)
    audit_log = (resolved + "\n" + provenance).lower()
    assert "era5_z_invariant from forcing soilgeo" in audit_log
    assert "date-indexed noaa annual policy" in audit_log
    for forbidden in (
        "wrfout_reference", "reference_path", "persistence",
        "snow mask", "snow_mask", "warm-sector", "warm_sector",
        "gpuwm phase 3 april 3 1974 d01",
    ):
        assert forbidden not in audit_log


@requires_inputs
@requires_gpu
@pytest.mark.gpu
@pytest.mark.slow_acceptance
def test_may1999_controller_run_crosses_18z_and_variant_is_tooling_readable(
        tmp_path, monkeypatch):
    """Controller GPU gate: generic pipeline, health, and output tooling."""
    from gpuwm import runtime
    from gpuwm.cli import main as gpuwm_main
    from gpuwm.verify.profiles import HealthProfile, OutputSchema

    exp, data = load_experiment_case(CONFIG)
    cfg = exp.domain(1).run
    # Deliberately NO OracleProfile: the ARC-A gate is oracle-free by
    # design -- generic health + output schema only (plan Task 7).
    health_profile = HealthProfile.generic_real_case(
        name="arc-a-generic-health", output_schema=OutputSchema())

    def assert_generic_health(run_summary, experiment):
        domain = experiment.domain(1)
        report = health_profile.evaluate(
            run_summary, start_time=experiment.start_time,
            expected_completed_seconds=experiment.run_seconds,
            forcing_times=VALID_TIMES,
            history_interval_s=domain.history_interval_s,
            domain_id=domain.grid_id, nx=domain.run.nx, ny=domain.run.ny,
            nz=domain.run.nz)
        report.require_ok()

    static_path = tmp_path / "static" / "may1999_static.npz"
    assert gpuwm_main([
        "static", str(CONFIG), "--output", str(static_path)]) == 0
    with np.load(static_path) as static:
        _assert_numeric_finite(static)
        for name in ("HGT_M", "LANDMASK", "LU_INDEX"):
            assert static[name].shape[-2:] == (cfg.ny, cfg.nx), name

    ingest_path = tmp_path / "ingest" / "may1999_initial.npz"
    assert gpuwm_main([
        "ingest", str(CONFIG), "--output", str(ingest_path)]) == 0
    with np.load(ingest_path) as initial:
        _assert_numeric_finite(initial)
        assert initial["u"].shape == (cfg.nz, cfg.ny, cfg.nx + 1)
        assert initial["v"].shape == (cfg.nz, cfg.ny + 1, cfg.nx)
        assert initial["w"].shape == (cfg.nz + 1, cfg.ny, cfg.nx)
        for name in ("thp", "qv", "qc", "qr"):
            assert initial[name].shape == (cfg.nz, cfg.ny, cfg.nx), name
        for name in ("surface_pressure", "surface_qv", "dry_mass"):
            assert initial[name].shape == (cfg.ny, cfg.nx), name

    hourly_dir = tmp_path / "hourly"
    captured = []
    real_run_experiment = runtime.run_experiment

    def capture_run(*args, **kwargs):
        summary = real_run_experiment(*args, **kwargs)
        captured.append(summary)
        return summary

    monkeypatch.setattr(runtime, "run_experiment", capture_run)
    assert gpuwm_main([
        "run", str(CONFIG), "--outdir", str(hourly_dir),
        "--no-supervise",
    ]) == 0
    summary = captured.pop()
    assert_generic_health(summary, exp)
    assert summary.dynamics_substeps == 1
    assert np.isfinite(summary.swdown_peak_wm2)
    hourly_names = [path.name for path in summary.wrfout_paths]
    assert hourly_names == _expected_output_names(
        exp.start_time, exp.run_seconds, 3600.0)
    assert all(name.startswith("wrfout_d01_1999-05-03_")
               and ":" not in name for name in hourly_names)
    assert all(path.is_file() and path.stat().st_size > 0
               for path in summary.wrfout_paths)
    _assert_wrf_tooling(summary.wrfout_paths[-1], ny=cfg.ny, nx=cfg.nx)

    # Frozen odd-cadence variant: only the declared domain history interval
    # changes.  The case, inputs, physics, run length, and source tree do not.
    base_text = CONFIG.read_text(encoding="utf-8")
    assert base_text.count("history_interval_s = 3600.0") == 1
    variant_path = tmp_path / "may1999_d01_smoke_1800s.toml"
    variant_path.write_text(
        base_text.replace("history_interval_s = 3600.0",
                          "history_interval_s = 1800.0"),
        encoding="utf-8")
    variant_exp, variant_data = load_experiment_case(variant_path)
    assert variant_exp.domain(1).history_interval_s == 1800.0

    variant_dir = tmp_path / "half_hourly"
    assert gpuwm_main([
        "run", str(variant_path), "--outdir", str(variant_dir),
        "--no-supervise",
    ]) == 0
    variant = captured.pop()
    assert_generic_health(variant, variant_exp)
    variant_names = [path.name for path in variant.wrfout_paths]
    assert variant_names == _expected_output_names(
        variant_exp.start_time, variant_exp.run_seconds, 1800.0)
    assert all(":" not in name for name in variant_names)
    subhourly = [path for path in variant.wrfout_paths
                 if path.name.endswith("_30_00")]
    assert len(subhourly) == 6
    for path in subhourly:
        _assert_wrf_tooling(path, ny=cfg.ny, nx=cfg.nx)
