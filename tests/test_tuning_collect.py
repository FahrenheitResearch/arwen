"""Metadata failures must not mislabel a partial or different run as scored."""
from pathlib import Path
import json

import pytest

from gpuwm.tuning import digest
from gpuwm.tuning_collect import completed_run, domain_table, expected_times, history_files, surface_record


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def frames(root, day="2026-09-28", hours=24):
    for valid in expected_times(day, hours):
        filename = "wrfout_d01_" + valid.replace("T", "_").replace(":", "_")
        path = root / "chain" / "run" / "wrfout" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque native history")
    return root


def test_missing_last_hour_refuses_complete_day(tmp_path):
    root = frames(tmp_path / "histories")
    latest = next(root.rglob("wrfout_d01_2026-09-29_00_00_00"))
    latest.unlink()
    with pytest.raises(ValueError, match="missing d01 histories.*2026-09-29T00:00:00"):
        history_files(root, "2026-09-28", [{"id": "d01"}], 24)


def test_duplicate_encoded_filename_candidate_is_ambiguous(tmp_path):
    root = frames(tmp_path / "histories")
    original = next(root.rglob("wrfout_d01_2026-09-28_01_00_00"))
    duplicate = root / original.name
    duplicate.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="duplicate d01 history"):
        history_files(root, "2026-09-28", [{"id": "d01"}], 24)


def test_changed_job_cannot_reuse_success_receipt(tmp_path):
    directory = tmp_path / "2026-09-28" / "current"
    frames(directory / "run")
    (directory / "DONE").write_text("0\n")
    write(directory / "job.json", {"member": {"id": "current", "seed": 1}})
    write(directory / "tuning-run.json", {"day": "2026-09-28", "member_id": "current",
        "status": "succeeded", "exit_code": 0, "job_sha256": digest(directory / "job.json")})
    write(directory / "member-receipt.json", {"status": "PASS", "preflight": {"member": {"id": "current"}}})
    assert len(completed_run(tmp_path, "2026-09-28", "current", [{"id": "d01"}], 24)["d01"]) == 25
    with pytest.raises(ValueError, match="completed seed differs from the frozen setting"):
        completed_run(tmp_path, "2026-09-28", "current", [{"id": "d01"}], 24,
                      expected_member={"id": "current", "seed": 2})
    write(directory / "job.json", {"member": {"id": "current", "seed": 2}})
    with pytest.raises(ValueError, match="job bytes differ"):
        completed_run(tmp_path, "2026-09-28", "current", [{"id": "d01"}], 24)


def test_surface_requires_actual_hashed_source(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_bytes(b"actual opaque public source")
    record = {"schema": "gpuwm-obs.asos-surface.v2", "status": "READY",
        "valid_times": expected_times("2026-09-28", 24),
        "provenance": {"is_stub": False, "uri": str(raw), "sha256": digest(raw)}}
    write(tmp_path / "surface-2026-09-28.json", record)
    assert surface_record(tmp_path, "2026-09-28", 24)[1] == record
    raw.write_bytes(b"different source")
    with pytest.raises(ValueError, match="provenance bytes changed"):
        surface_record(tmp_path, "2026-09-28", 24)


def test_nested_grid_spacing_uses_actual_declared_hierarchy(tmp_path):
    config = tmp_path / "configs" / "day" / "group.toml"
    config.parent.mkdir(parents=True)
    config.write_text("[[domain]]\ngrid_id=1\nparent_id=0\ndx=2250.0\n"
                      "[[domain]]\ngrid_id=2\nparent_id=1\nparent_grid_ratio=3\n"
                      "[[domain]]\ngrid_id=3\nparent_id=2\nparent_grid_ratio=3\n")
    write(tmp_path / "run-list.json", [{"config": "configs/day/group.toml"}])
    assert domain_table(tmp_path) == [{"id": "d01", "dx_m": 2250.0},
                                     {"id": "d02", "dx_m": 750.0}, {"id": "d03", "dx_m": 250.0}]
