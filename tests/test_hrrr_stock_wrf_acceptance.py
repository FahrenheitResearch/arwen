from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from gpuwm.ingest.hrrr_target import HrrrTargetDomain
from tools.run_hrrr_stock_wrf_acceptance import (
    AcceptanceFailure,
    _expected_identity_attrs,
    _finite_variable,
    template_symlink_plan,
    validate_export,
    validate_wrf_logs,
)


def _target_payload(**updates):
    payload = {
        "schema": "gpuwm-hrrr-target-domain-v1",
        "name": "stock_wrf_test",
        "map_proj": "lambert",
        "nx": 13,
        "ny": 13,
        "nz": 49,
        "dx_m": 3000.0,
        "dy_m": 3000.0,
        "ref_lat": 35.5,
        "ref_lon": -98.0,
        "truelat1": 30.0,
        "truelat2": 60.0,
        "stand_lon": -97.0,
        "time_step_seconds": 15,
        "spec_bdy_width": 5,
        "spec_zone": 1,
        "relax_zone": 4,
    }
    payload.update(updates)
    return payload


def test_direct_export_global_attributes_match_each_file_contract():
    target = HrrrTargetDomain(**{
        key: value for key, value in _target_payload().items()
        if key != "schema"})
    valid = "2026-07-18_00:00:00"
    initial = _expected_identity_attrs("wrfinput_d01", target, valid)
    boundary = _expected_identity_attrs("wrfbdy_d01", target, valid)
    assert initial["SIMULATION_START_DATE"] == valid
    assert "SIMULATION_START_DATE" not in boundary
    assert boundary["WEST-EAST_GRID_DIMENSION"] == target.nx + 1


@pytest.mark.skipif(os.name == "nt", reason="Windows test user cannot create symlinks")
def test_template_plan_copies_only_clean_symlink_forest(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    table = assets / "LANDUSE.TBL"
    table.write_text("sealed")
    template = tmp_path / "template"
    template.mkdir()
    (template / "LANDUSE.TBL").symlink_to(table)
    (template / "wrfinput_d01").write_text("old input")
    (template / "wrfbdy_d01").write_text("old boundary")
    (template / "namelist.input").write_text("old namelist")
    (template / "wrfout_d01_old").write_text("old output")

    links, skipped = template_symlink_plan(template)
    assert [item["name"] for item in links] == ["LANDUSE.TBL"]
    assert skipped == [
        "namelist.input", "wrfbdy_d01", "wrfinput_d01", "wrfout_d01_old"]

    (template / "unsealed-table").write_text("not a symlink")
    with pytest.raises(AcceptanceFailure, match="non-symlink"):
        template_symlink_plan(template)


def test_log_gates_are_exact_and_end_time_is_dynamic(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    stdout = run / "wrf.stdout.log"
    stdout.write_text(
        "d01 Input data is acceptable to use: wrfinput_d01\n"
        "d01 Input data is acceptable to use: wrfbdy_d01\n"
        "Timing for main: time 2026-07-18_00:00:15 on domain   1: 0.1 elapsed\n"
        "Timing for main: time 2026-07-18_00:00:30 on domain   1: 0.1 elapsed\n"
        "wrf: SUCCESS COMPLETE WRF\n"
    )
    result = validate_wrf_logs(
        run_dir=run,
        stdout_path=stdout,
        valid_time="2026-07-18_00:00:00",
        run_seconds=30,
        returncode=0,
    )
    assert result["expected_end_time"] == "2026-07-18_00:00:30"
    assert result["input_acceptance"]["wrfinput_d01"]["exact_marker"].endswith(
        "wrfinput_d01")
    assert result["success"]["exact_marker"] == "wrf: SUCCESS COMPLETE WRF"

    stdout.write_text(stdout.read_text().replace(
        "Input data is acceptable to use: wrfbdy_d01",
        "Input data is acceptable to use: something_else"))
    with pytest.raises(AcceptanceFailure, match="did not accept wrfbdy_d01"):
        validate_wrf_logs(
            run_dir=run,
            stdout_path=stdout,
            valid_time="2026-07-18_00:00:00",
            run_seconds=30,
            returncode=0,
        )


def test_finite_scanner_is_bounded_and_rejects_nan(tmp_path):
    path = tmp_path / "values.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("x", 100)
        dataset.createDimension("y", 100)
        variable = dataset.createVariable("field", "f4", ("x", "y"))
        variable[:] = 1.0
    with netCDF4.Dataset(path) as dataset:
        largest = _finite_variable(
            dataset.variables["field"], budget_bytes=1000)
        assert largest <= 1200

    with netCDF4.Dataset(path, "a") as dataset:
        dataset.variables["field"][25, 25] = np.nan
    with netCDF4.Dataset(path) as dataset:
        with pytest.raises(AcceptanceFailure, match="non-finite"):
            _finite_variable(dataset.variables["field"], budget_bytes=1000)


def test_export_manifest_and_target_are_bound_before_reopen(tmp_path, monkeypatch):
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(_target_payload()), encoding="utf-8")
    export = tmp_path / "export"
    export.mkdir()
    specs = {}
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        path = export / name
        path.write_bytes(name.encode("ascii"))
        specs[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": "gpuwm-native-direct-wrf-export-v2",
        "status": "READY",
        "valid_time": "2026-07-18_00:00:00",
        "dimensions": {"nx": 13, "ny": 13, "nz": 49},
        "boundary_interval_seconds": 3600,
        "boundary_record_count": 2,
        "boundary_times": [
            "2026-07-18_00:00:00", "2026-07-18_01:00:00"],
        "next_boundary_times": [
            "2026-07-18_01:00:00", "2026-07-18_02:00:00"],
        "files": specs,
    }
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fake_validate(path: Path, _contract, **_kwargs):
        return {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(
        "tools.run_hrrr_stock_wrf_acceptance._validate_dataset", fake_validate)
    restored, evidence = validate_export(
        export, target_path, "2026-07-18_00:00:00")
    assert restored == manifest
    assert evidence["status"] == "PASS"
    assert evidence["target_domain"]["name"] == "stock_wrf_test"

    manifest["dimensions"]["nx"] = 14
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AcceptanceFailure, match="dimensions mismatch"):
        validate_export(export, target_path, "2026-07-18_00:00:00")
