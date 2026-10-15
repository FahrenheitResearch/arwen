"""Actual scalar and packed generation paths must write identical words."""
import os
from pathlib import Path

import numpy as np
import pytest
import subprocess
import sys
from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize("version", ("wrf_480", "wrf_461"))
def test_generation_reaches_packed_shared_column_route(version):
    directory = os.environ.get("YSU_COLUMN_FIXTURE")
    if not directory:
        pytest.skip("requires the regenerated source-version column fixture")
    import cupy as cp
    from gpuwm.verify.ysu_wide_oracle import load_wide_oracle
    from gpuwm.core.ysu import launch_ysu
    from gpuwm.ensemble.batch_physics import prepare_ysu_column_batch
    cols = [c for c in load_wide_oracle(Path(directory)) if c.topo == 0 and c.topdown == 1]
    c = next(c for c in cols if np.any(c.levels["qc"] > 0))
    level_map = {"exner": "pi"}
    inputs = {name: cp.asarray(np.tile(c.levels[level_map.get(name, name)][:, None, None], (1, 2, 1)))
              for name in ("u", "v", "theta", "qv", "qc", "qi", "p", "exner", "dz", "rthraten")}
    inputs["p_interface"] = cp.asarray(np.tile(c.p_interface[:, None, None], (1, 2, 1)))
    surfaces = ("psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim", "psih", "xland", "u10", "v10")
    inputs.update({name: cp.full((1, 1) if name == "xland" else (2, 1), c.scalars[name], dtype=cp.float32)
                   for name in surfaces})
    batch = prepare_ysu_column_batch(inputs, members=2, ny=1, nx=1,
        dt=float(c.scalars["dt"]), topdown=1, available_bytes=64 * 1024 * 1024,
        shared_fields=("xland",), ysu_version=version)
    got = batch()
    assert batch.receipt["ysu_version"] == version
    for member in range(2):
        lev = {name: cp.ascontiguousarray(value[:, member:member+1]) for name, value in inputs.items() if value.ndim == 3}
        sfc = {name: inputs[name] if name == "xland" else cp.ascontiguousarray(inputs[name][member:member+1]) for name in surfaces}
        want = launch_ysu(*(lev[name] for name in ("u", "v", "theta", "qv", "qc", "qi", "p", "p_interface", "exner", "dz")),
            rthraten=lev["rthraten"], dt=float(c.scalars["dt"]), ysu_topdown_pblmix=1,
            ysu_version=version, **sfc)
        for name in want:
            a = got[name][:, member:member+1] if got[name].ndim == 3 else got[name][member:member+1]
            np.testing.assert_array_equal(cp.asnumpy(a).view(np.uint32), cp.asnumpy(want[name]).view(np.uint32), err_msg=name)


@requires_gpu
def test_shallow_column_matches_declared_defined_reference(tmp_path):
    directory = os.environ.get("YSU_COLUMN_FIXTURE")
    if not directory:
        pytest.skip("requires the regenerated source-version column fixture")
    script = Path(__file__).parents[1] / "tools/ysu_generations_oracle/check_shallow.py"
    subprocess.run([sys.executable, str(script), str(Path(directory).parent),
                    str(tmp_path / "shallow.json")], check=True)
