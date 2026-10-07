"""Processes stepping the real fork Thompson adapter lock the tables once each.

The CUDA half of tests/test_thompson_fork_tables_lock.py: four processes
sharing one TMPDIR, lock root and staged fork table root each step
gpuwm.core.microphysics_aerosol._apply_thompson_aerosol under
thompson_version = "wrf_39_noaa" on an oracle column, and each takes the
cross-process "fetch-tables" lock once (its first verification), not once
per step.  The defect it pins: the lock was taken on every step, and about
70 DA member processes sharing one TMPDIR queued on it with their GPUs near
0% (2026-10-06).
"""
from __future__ import annotations

from pathlib import Path

import pytest

cp = pytest.importorskip("cupy")

from gpuwm import fetch_guard  # noqa: E402
from test_thompson_fork_tables_lock import _run_children  # noqa: E402

_PROCESSES = 4
_STEPS = 50


def test_concurrent_processes_stepping_fork_thompson_lock_once_each(tmp_path):
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA runtime")
    from gpuwm.core.thompson_contract import FORK_TABLE_ASSETS
    from gpuwm.physics_compat import thompson_fork_table_root
    root = Path(thompson_fork_table_root())
    if not all((root / asset.filename).is_file()
               for asset in FORK_TABLE_ASSETS):
        pytest.skip(f"fork tables are not staged at {root} "
                    "(gpuwm fetch-tables --thompson-fork)")
    shared_tmp = tmp_path / "tmp"
    shared_tmp.mkdir()
    results = _run_children(tmp_path, "gpu", _PROCESSES, _STEPS, {
        "TMPDIR": str(shared_tmp),
        fetch_guard.LOCK_ROOT_ENV: str(shared_tmp / "gpuwm" / "locks"),
        "GPUWM_THOMPSON_FORK_TABLE_ROOT": str(root),
    })
    assert [result["holds"] for result in results] == [1] * _PROCESSES, results
