"""The benchmark command in docs/public/WRF-INPUTS.md, run as the page writes it.

Named breakage: 2.9.0 exited 2 on NCAR's public WRF v4.4 CONUS 12 km
benchmark folder ("&physics ... cu_physics=5 is required"), so the page's own
command did not run.  This leg reads the command from the page, points it at
the unpacked folder and runs it.  Set GPUWM_NCAR_CONUS12KM to the
v4.4_bench_conus12km folder (NCAR's v4.4_bench_conus12km.tar.gz), or retain
the folder beside the GPU legs' test inputs (the parent of
GPUWM_TEST_WRF_REAL_DIRECTORY, which every GPU leg exports), where a leg
finds it without another variable.

Named breakage (2.9.2, RULING-8): the command exited 2 in
apply_warm_start_physics because the exact chain's history rows had become
cross-step carriers with no warm-start rule, and this leg, the one test of
that command, skipped on every GPU leg because no leg set the variable.
"""
from __future__ import annotations

import gc
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "public" / "WRF-INPUTS.md"


def _page_command() -> list[str]:
    commands = re.findall(r"(?m)^gpuwm run --benchmark .*$", PAGE.read_text())
    assert len(commands) == 1, commands
    return shlex.split(commands[0])


def _case_folder() -> Path | None:
    named = os.environ.get("GPUWM_NCAR_CONUS12KM")
    if named:
        return Path(named)
    real = os.environ.get("GPUWM_TEST_WRF_REAL_DIRECTORY")
    if real:
        beside = Path(real).parent / "v4.4_bench_conus12km"
        if (beside / "namelist.input").is_file():
            return beside
    return None


def test_wrf_inputs_page_benchmark_command_runs_on_ncar_conus12km(tmp_path):
    folder = _case_folder()
    if folder is None:
        pytest.skip("set GPUWM_NCAR_CONUS12KM to NCAR's unpacked v4.4_bench_conus12km folder, "
                    "or retain it beside GPUWM_TEST_WRF_REAL_DIRECTORY")
    argv = _page_command()
    restart = argv.index("--restart-from") + 1
    outdir = argv.index("--outdir") + 1
    assert argv[restart].startswith("v4.4_bench_conus12km/")
    argv[restart] = str(folder / argv[restart].split("/", 1)[1])
    argv[outdir] = str(tmp_path / argv[outdir])
    from gpuwm.supervisor import priced_reservation_bytes
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    from gpuwm.wrfinput_forecast import wrfinput_urban_columns

    run = resolve_wrfinput_run(folder, restart_from=argv[restart])
    required = priced_reservation_bytes(run.experiment, urban_columns=wrfinput_urban_columns(run))
    assert required is not None, "the NCAR benchmark must have a priced GPU reservation"

    # R6's serial shard held 18.5 GiB beside its own subprocess. Pool flushing alone cannot
    # release arrays in unreachable reference cycles left by earlier tests. Finish device work,
    # collect those arrays, then return their blocks and the pinned pool before admission.
    cupy = pytest.importorskip("cupy")
    cupy.cuda.runtime.deviceSynchronize()
    gc.collect()
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    cupy.cuda.runtime.deviceSynchronize()
    free, total = cupy.cuda.runtime.memGetInfo()
    if total < required:
        pytest.skip(f"this card ({total / 1024**3:.2f} GiB) is smaller than the NCAR CONUS 12 km "
                    f"run's priced reservation ({required / 1024**3:.2f} GiB)")
    assert free >= required, (
        f"NCAR subprocess needs {required / 1024**3:.2f} GiB device-wide free, "
        f"has {free / 1024**3:.2f} GiB after parent cleanup; "
        f"parent CuPy pool still has {pool.used_bytes() / 1024**3:.2f} GiB in live arrays "
        f"and {pool.free_bytes() / 1024**3:.2f} GiB cached")
    done = subprocess.run([sys.executable, "-m", "gpuwm", *argv[1:]], cwd=tmp_path,
                          capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    timing = json.loads((Path(argv[outdir]) / "benchmark-timing.json").read_text())
    # NCAR's 1 h restart leg: 50 steps of 72 s from step 550, as the page says.
    assert timing["status"] == "SUCCESS"
    assert (timing["first_step"], timing["last_step"], timing["steps"]) == (551, 600, 50)
