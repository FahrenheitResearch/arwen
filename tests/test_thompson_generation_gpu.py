"""Live source-column gates, enabled by a named box oracle/input bank."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("bank",("v4.4_bench_conus12km","v4.4_bench_conus2.5km"))
@pytest.mark.parametrize("kind",("input","restart"))
@pytest.mark.parametrize("strict",(False,True))
def test_every_output_word_equals_the_unmodified_generation_oracle(tmp_path,bank,kind,strict):
    lane = os.environ.get("THOMPSON_GENERATION_ORACLE_LANE")
    if not lane:
        pytest.skip("set THOMPSON_GENERATION_ORACLE_LANE to the owned source-oracle and Rust column bank")
    root = Path(__file__).resolve().parents[1]
    source = Path(lane)/"columns"/(bank+"-"+kind+".npz")
    env = dict(os.environ,GPUWM_THOMPSON_GENERATION_TABLE_ROOT=str(Path(lane)/"oracle44/run"))
    env["GPUWM_WRF_EXACT"] = "1" if strict else "0"
    command = [sys.executable,str(root/"tools/thompson_generation_oracle/benchmark_columns.py"),"compare",str(source),str(tmp_path),"--lane",lane,"--version","wrf_44"]
    subprocess.run(command,check=True,env=env)
    receipt = json.loads((tmp_path/(source.stem+".json")).read_text())[0]
    assert receipt["backend"] == "GPU"
    assert receipt["different"] == 0,receipt
    assert receipt["words"] == 164864
    assert all(r["gpu_sha256"] == r["wrf_sha256"] for r in receipt["fields"].values())


@pytest.mark.parametrize("capacity", (64,128,256))
def test_generation_frame_reservation_covers_the_production_driver(capacity):
    from datetime import datetime
    from gpuwm.config import RunConfig
    from gpuwm.experiment import experiment_from_run_config
    from gpuwm.core.kernels import get_kernel, get_kernel_int_defines
    from gpuwm.core.preflight import kernel_local_frame_bytes
    kernel = (get_kernel("thompson_column","thompson_column_step") if capacity == 64 else
              get_kernel_int_defines("thompson_column","thompson_column_step",(("THOMPSON_COLUMN_KMAX",capacity),)))
    cfg = RunConfig(nx=8,ny=8,nz=capacity,dx=1000.,dy=1000.,ztop=20000.,dt=1.,run_seconds=2.,mp_physics=8,thompson_version="wrf_44")
    priced = kernel_local_frame_bytes(experiment_from_run_config(cfg,datetime(2020,1,1)))
    import cupy as cp
    assert cp.cuda.driver.funcGetAttribute(3,kernel.ptr) <= priced["thompson_column"]
