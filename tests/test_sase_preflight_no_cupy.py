"""SASE CPU estimates must not load the CUDA launch module or its runtime."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_sase_cpu_accounting_runs_with_device_imports_blocked():
    # A fresh process cannot inherit a previously imported device module.
    # The finder records attempts even if a future import path absorbs its
    # exception, so a fallback cannot hide a request for the CUDA runtime.
    program = r'''
import importlib.abc
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

attempted = []

class RejectDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"cupy", "cupyx"} or fullname == "gpuwm.core.sase":
            attempted.append(fullname)
            raise AssertionError("CPU accounting requested a device import: " + fullname)
        return None

sys.meta_path.insert(0, RejectDeviceImports())
from gpuwm.config import RunConfig, SASE_PBL_SCHEME, validate_run_config
from gpuwm.core import preflight
from gpuwm.core.sase_limits import THREADS_PER_BLOCK

assert THREADS_PER_BLOCK == 128
for nx, ny, nz in ((8, 4, 4), (9, 5, 4), (16, 8, 4)):
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0,
                    ztop=10000.0, dt=1.0, run_seconds=10.0,
                    bl_pbl_physics=SASE_PBL_SCHEME, moist=True,
                    sf_sfclay_physics=1, km_opt=0)
    validate_run_config(cfg)
    ncell = nx * ny * nz
    nblocks = (ncell + 127) // 128
    phases = preflight.sase_workspace_phases(cfg)
    assert phases["solve"]["partials"] == ((5, nblocks), 8)
    assert phases["apply"]["partials_mom"] == ((3, (nx * ny + 127) // 128), 8)
    shapes = preflight.sase_workspace_shapes(cfg)
    workspace_bytes = sum(math.prod(shape) * size for shape, size in shapes.values())
    assert workspace_bytes == 59 * 4 * ncell + 2 * 8 * 5 * nblocks + 2 * 4 * nx * ny
    estimate = preflight.estimate_domain(SimpleNamespace(run=cfg, grid_id=1, parent_id=0))
    baseline = preflight.estimate_domain(SimpleNamespace(
        run=replace(cfg, bl_pbl_physics=0), grid_id=1, parent_id=0))
    assert estimate.category_bytes("sase") == workspace_bytes
    assert estimate.transient_bytes == baseline.transient_bytes + workspace_bytes

assert not attempted, attempted
assert not {"cupy", "cupyx", "gpuwm.core.sase"}.intersection(sys.modules)
print("SASE CPU accounting: 3 grids checked, no device imports")
'''
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1"),
        text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 grids checked, no device imports" in result.stdout
