"""The DA data-movement benches run end to end on a tiny case (CPU)."""

from __future__ import annotations

import json


def test_member_io_bench_runs_the_packed_roster_path(tmp_path, capsys):
    from tools import da_member_io_bench

    da_member_io_bench.main(["--root", str(tmp_path / "b"), "--members", "4",
                             "--nx", "12", "--ny", "10", "--nz", "4",
                             "--extra-3d", "1", "--obs-mib", "1"])
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert row["tree"] == "new" and row["members"] == 4
    for key in ("controller_roster_s", "barrier_validate_s",
                "stage_analysis_inputs_s", "recovery_generation_s"):
        assert row[key] >= 0.0
    assert not (tmp_path / "b").exists()


def test_analysis_host_bench_fingerprints_the_accepted_increments(
        tmp_path, capsys):
    from tools import da_analysis_host_bench

    root = str(tmp_path / "ah")
    da_analysis_host_bench.main(["setup", "--root", root, "--small"])
    capsys.readouterr()
    rows = []
    for label in ("a", "b"):
        da_analysis_host_bench.main(["run", "--root", root, "--small",
                                     "--label", label])
        rows.append(json.loads(capsys.readouterr().out.strip().splitlines()[-1]))
    assert rows[0]["pending_sha256"] == rows[1]["pending_sha256"]
    assert rows[0]["withheld_solves"] >= 1
    assert rows[0]["positivity"]["negative_points"] > 0
