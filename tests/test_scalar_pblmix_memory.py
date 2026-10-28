"""Keep local scalar rates and solve workspace visible to VRAM admission."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from gpuwm.config import RunConfig
from gpuwm.core.preflight import mynn_scalar_transient_shapes


def test_scalar_diffusion_off_adds_no_memory_charge():
    cfg = RunConfig(nx=1797, ny=1057, nz=50, bl_pbl_physics=5,
                    dx=3000.0, dy=3000.0, ztop=20000.0, dt=15.0,
                    run_seconds=60.0)
    assert mynn_scalar_transient_shapes(cfg) == {}
    # Existing MYNN EDMF scalar estimates are not changed by this option.
    assert mynn_scalar_transient_shapes(replace(cfg, bl_mynn_mixscalars=1)) == {}


def test_scalar_diffusion_prices_both_live_grid_banks_and_chunk_work(monkeypatch):
    import gpuwm.core.preflight as preflight
    monkeypatch.setattr(preflight, "mynn_pbl_column_chunk", lambda cfg: 7)
    # mp=28: the family MYNN's wrapper solves itself.  Every other scheme's
    # scalars take the shared post-PBL arm (the test below).
    cfg = RunConfig(nx=8, ny=6, nz=50, bl_pbl_physics=5, scalar_pblmix=1,
                    mp_physics=28, dx=3000.0, dy=3000.0, ztop=20000.0,
                    dt=15.0, run_seconds=60.0)
    values = sum(math.prod(shape) for shape in mynn_scalar_transient_shapes(cfg).values())
    # Coupling can retain last step's four fields through the composed
    # tendency slot while creating four fresh fields beside four raw rates.
    assert values >= 13 * 50 * 48
    # The active solve needs only seven columns of work, not all 48.
    assert values < 13 * 50 * 48 + (8 * 50 + 2) * 7 + 1
    monkeypatch.setattr(preflight, "mynn_pbl_column_chunk", lambda cfg: 48)
    small_grid_values = sum(math.prod(shape) for shape in
                            mynn_scalar_transient_shapes(cfg).values())
    assert small_grid_values > values


def test_scalar_diffusion_after_mynn_prices_the_schemes_own_scalars():
    """U-E4: under MYNN a scheme other than mp=28 is mixed by the shared arm,
    one raw and one coupled rate per diffused scalar plus the mass
    coefficient; a scheme with none (WSM6) keeps only the coefficient."""
    base = dict(nx=8, ny=6, nz=50, bl_pbl_physics=5, scalar_pblmix=1,
                dx=3000.0, dy=3000.0, ztop=20000.0, dt=15.0, run_seconds=60.0)
    for mp, species in ((8, 1), (18, 4), (50, 3), (6, 0)):
        shapes = mynn_scalar_transient_shapes(RunConfig(mp_physics=mp, **base))
        chunk = 48 if species else 0
        assert shapes == {"pbl_scalar/local_diffusion_peak":
                          ((2 * species + 1) * 50 * 48 + (11 * 50 + 1) * chunk,)}, mp


def test_scalar_diffusion_launcher_allocations_match_the_priced_chunk():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="-1")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(root / "tests/physics_allocation_census.py"),
         "--scalar-pblmix"], cwd=root, env=env, text=True,
        capture_output=True, check=True, timeout=60)
    rows = json.loads(result.stdout)
    assert rows["runtime"] == [[[7, 50], 1400], [[7, 50], 1400]]
    assert rows["standalone"] == rows["runtime"] + [[[7, 251], 7028]]
