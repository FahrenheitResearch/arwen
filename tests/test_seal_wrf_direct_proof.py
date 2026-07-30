from __future__ import annotations

from fractions import Fraction

import netCDF4
import numpy as np
import pytest

from tools.seal_wrf_direct_proof import (
    _domain_paths,
    _expected_hierarchy_step_times,
    _expected_step_times,
    _hash_manifest,
    _history_identity,
    _validate_export_authority,
    _validate_export_file_inventory,
)


def test_direct_proof_step_times_are_case_independent():
    assert _expected_step_times("2026-07-20_00:00:00", 10, 5) == [
        "2026-07-20_00:00:05",
        "2026-07-20_00:00:10",
    ]
    assert _expected_step_times("1974-04-03_23:59:55", 10, 5) == [
        "1974-04-04_00:00:00",
        "1974-04-04_00:00:05",
    ]
    assert _expected_step_times(
        "2026-07-20_00:00:00", 5, Fraction(5, 4),
    ) == [
        "2026-07-20_00:00:01",
        "2026-07-20_00:00:02",
        "2026-07-20_00:00:03",
        "2026-07-20_00:00:05",
    ]


@pytest.mark.parametrize(
    "valid_time,run_seconds,time_step_seconds",
    [
        ("not-a-time", 10, 5),
        ("2026-07-20_00:00:00", 0, 5),
        ("2026-07-20_00:00:00", 10, 0),
        ("2026-07-20_00:00:00", 10, 6),
    ],
)
def test_direct_proof_step_times_fail_closed(
    valid_time, run_seconds, time_step_seconds,
):
    with pytest.raises(SystemExit):
        _expected_step_times(valid_time, run_seconds, time_step_seconds)


def test_direct_proof_hash_manifest_is_strict(tmp_path):
    manifest = tmp_path / "hashes.txt"
    manifest.write_text(
        f"{'a' * 64}  /sealed/wrf.exe\n"
        f"{'b' * 64}  wrfinput_d01\n",
        encoding="utf-8",
    )
    assert _hash_manifest(manifest) == {
        "wrf.exe": "a" * 64,
        "wrfinput_d01": "b" * 64,
    }

    manifest.write_text(f"{'a' * 64}  one/wrf.exe\n{'b' * 64}  two/wrf.exe\n")
    with pytest.raises(SystemExit, match="duplicate basename"):
        _hash_manifest(manifest)


def test_hierarchy_step_times_follow_exact_parent_ratios():
    hierarchy = [
        {
            "grid_id": 1,
            "parent_id": 0,
            "parent_time_step_ratio": 1,
            "dt_s": 5.0,
        },
        {
            "grid_id": 2,
            "parent_id": 1,
            "parent_time_step_ratio": 4,
            "dt_s": 1.25,
        },
    ]
    assert _expected_hierarchy_step_times(
        "2026-07-20_00:00:00", 10, 5, hierarchy,
    ) == {
        1: ["2026-07-20_00:00:05", "2026-07-20_00:00:10"],
        2: [
            "2026-07-20_00:00:01",
            "2026-07-20_00:00:02",
            "2026-07-20_00:00:03",
            "2026-07-20_00:00:05",
            "2026-07-20_00:00:06",
            "2026-07-20_00:00:07",
            "2026-07-20_00:00:08",
            "2026-07-20_00:00:10",
        ],
    }


def test_hierarchy_step_times_reject_manifest_dt_drift():
    hierarchy = [
        {
            "grid_id": 1,
            "parent_id": 0,
            "parent_time_step_ratio": 1,
            "dt_s": 5.0,
        },
        {
            "grid_id": 2,
            "parent_id": 1,
            "parent_time_step_ratio": 4,
            "dt_s": 1.5,
        },
    ]
    with pytest.raises(SystemExit, match="differs from the parent-ratio"):
        _expected_hierarchy_step_times(
            "2026-07-20_00:00:00", 10, 5, hierarchy,
        )


def test_hierarchy_history_path_parser_is_strict(tmp_path):
    assert _domain_paths([
        f"d01={tmp_path / 'root'}",
        f"d02={tmp_path / 'child'}",
    ]) == {
        1: tmp_path / "root",
        2: tmp_path / "child",
    }
    with pytest.raises(SystemExit, match="dNN"):
        _domain_paths([f"1={tmp_path / 'root'}"])
    with pytest.raises(SystemExit, match="duplicate"):
        _domain_paths([
            f"d01={tmp_path / 'root'}",
            f"d01={tmp_path / 'again'}",
        ])

    shared = tmp_path / "shared"
    shared.write_bytes(b"history")
    with pytest.raises(SystemExit, match="same file"):
        _domain_paths([
            f"d01={shared}",
            f"d02={shared.parent / '.' / shared.name}",
        ])


def test_hierarchy_export_inventory_is_exact():
    complete = {
        "wrfbdy_d01": {},
        "wrfinput_d01": {},
        "wrfinput_d02": {},
    }
    assert _validate_export_file_inventory(complete, (1, 2)) is complete
    with pytest.raises(SystemExit, match="file inventory"):
        _validate_export_file_inventory(
            {"wrfbdy_d01": {}, "wrfinput_d01": {}}, (1, 2),
        )
    with pytest.raises(SystemExit, match="file inventory"):
        _validate_export_file_inventory(
            {**complete, "unsealed-extra": {}}, (1, 2),
        )


def test_export_authority_binds_schema_shape_and_valid_time():
    single = {
        "schema": "gpuwm-native-direct-wrf-export-v2",
        "status": "READY",
        "valid_time": "2026-07-20_00:00:00",
    }
    assert _validate_export_authority(
        single, "2026-07-20_00:00:00",
    ) is None
    hierarchy = {
        **single,
        "schema": "gpuwm-native-direct-wrf-hierarchy-export-v1",
        "hierarchy": [{"grid_id": 1}],
    }
    assert _validate_export_authority(
        hierarchy, "2026-07-20_00:00:00",
    ) == [{"grid_id": 1}]
    with pytest.raises(SystemExit, match="schema"):
        _validate_export_authority(
            {**hierarchy, "schema": single["schema"]},
            "2026-07-20_00:00:00",
        )
    with pytest.raises(SystemExit, match="valid time"):
        _validate_export_authority(hierarchy, "2026-07-20_03:00:00")


def _write_history_identity(path, *, grid_id: int, dt_s: float = 1.25):
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(
            list("2026-07-20_00:00:00"), dtype="S1",
        )
        dataset.GRID_ID = grid_id
        dataset.PARENT_ID = 1
        dataset.I_PARENT_START = 90
        dataset.J_PARENT_START = 80
        dataset.PARENT_GRID_RATIO = 4
        dataset.setncattr("WEST-EAST_GRID_DIMENSION", 241)
        dataset.setncattr("SOUTH-NORTH_GRID_DIMENSION", 161)
        dataset.setncattr("BOTTOM-TOP_GRID_DIMENSION", 50)
        dataset.DX = 3000.0
        dataset.DY = 3000.0
        dataset.DT = dt_s


def test_history_identity_binds_domain_time_and_geometry(tmp_path):
    history = tmp_path / "wrfout_d02"
    _write_history_identity(history, grid_id=2)
    row = {
        "grid_id": 2,
        "parent_id": 1,
        "i_parent_start": 90,
        "j_parent_start": 80,
        "parent_grid_ratio": 4,
        "nx": 240,
        "ny": 160,
        "nz": 49,
        "dx_m": 3000.0,
        "dy_m": 3000.0,
        "dt_s": 1.25,
    }
    identity = _history_identity(
        history,
        domain_id=2,
        valid_time="2026-07-20_00:00:00",
        hierarchy_row=row,
    )
    assert identity["grid_id"] == 2
    assert identity["parent_id"] == 1
    assert identity["i_parent_start"] == 90
    assert identity["dt_s"] == 1.25

    with pytest.raises(SystemExit, match="GRID_ID"):
        _history_identity(
            history,
            domain_id=1,
            valid_time="2026-07-20_00:00:00",
            hierarchy_row=None,
        )
    with pytest.raises(SystemExit, match="does not contain"):
        _history_identity(
            history,
            domain_id=2,
            valid_time="2026-07-20_01:00:00",
            hierarchy_row=row,
        )

    drifted = dict(row, i_parent_start=91)
    with pytest.raises(SystemExit, match="geometry differs"):
        _history_identity(
            history,
            domain_id=2,
            valid_time="2026-07-20_00:00:00",
            hierarchy_row=drifted,
        )
