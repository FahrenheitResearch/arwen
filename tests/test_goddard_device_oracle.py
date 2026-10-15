"""New Goddard radiation (ra_lw/ra_sw_physics = 5) graded on the card against WRF 4.6.1.

tools/goddard_wrf461_oracle/run_driver_gpu.py runs the port's four device
kernels (column preparation, cloud optics, both bands, driver publication)
over the 64-column, 32-layer WRF 4.6.1 driver fixture and compares 5,056
output words in 17 fields: zero ULP under GPUWM_WRF_EXACT=1, the README's
flux and heating limits in default arithmetic.  On the strict leg the two
planted one-bit band mutations must be caught.

Breakage this prevents: the port joined on no battery leg.  The CPU files
(test_goddard_column_check.py, test_goddard_driver.py) check the checker and
the registry/driver wiring without a device, so a Goddard kernel change that
moved its words against compiled WRF would have passed every leg.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "goddard_wrf461_oracle" / "run_driver_gpu.py"


def _run(tmp_path, mutate_band=None):
    import cupy as cp

    assert cp.cuda.runtime.getDeviceCount() > 0
    # The runner is a second CUDA process on the same card.  Inside a GPU
    # shard this process's CuPy pools still hold every earlier test's blocks,
    # and on a 16 GB card (node-4's RTX 5070 Ti, 2.9.0 round 2) the runner's
    # band kernel then could not reserve its local memory at launch
    # (CUDA_ERROR_OUT_OF_MEMORY).  Hand the cached blocks back first.
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    receipt = tmp_path / f"driver-{mutate_band or 'clean'}.json"
    command = [sys.executable, str(RUNNER), "--receipt", str(receipt)]
    if mutate_band:
        command += ["--mutate-band", str(mutate_band)]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    process = subprocess.run(command, cwd=ROOT, env=environment,
                             capture_output=True, text=True, timeout=1200)
    assert receipt.is_file(), process.stdout[-2000:] + process.stderr[-4000:]
    return process.returncode, json.loads(receipt.read_text())


def test_the_driver_matches_wrf461_in_this_process_arithmetic(tmp_path):
    from gpuwm.wrf_exact import ENABLED

    code, receipt = _run(tmp_path)
    failed = {name: row for name, row in receipt["fields"].items() if not row["pass"]}
    assert code == 0 and receipt["full_driver_pass"] and not failed, failed
    assert receipt["strict"] == ENABLED and receipt["columns"] == 64
    assert sum(row["words"] for row in receipt["fields"].values()) == 5056
    if ENABLED:
        assert all(row["different_words"] == 0 for row in receipt["fields"].values())


@pytest.mark.parametrize("band", [1, 2], ids=["sw", "lw"])
def test_a_one_bit_band_mutation_is_caught(tmp_path, band):
    from gpuwm.wrf_exact import ENABLED

    if not ENABLED:
        pytest.skip("a one-bit flip sits inside the default arithmetic limits; "
                    "the strict leg (zero ULP) is the one that must catch it")
    code, receipt = _run(tmp_path, band)
    assert code == 0, "the runner did not detect its planted mutation"
    assert receipt["mutation_band"] == band and not receipt["full_driver_pass"]
