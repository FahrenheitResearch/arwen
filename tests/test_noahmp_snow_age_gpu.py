"""Release column hashes through the shipped device path and a changed-term control."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]


@requires_gpu
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("generation", [None, "wrf_461"])
def test_complete_release_column_hashes(tmp_path, strict, generation):
    tool = ROOT / "tools/noahmp_snow_age_oracle/profile.py"
    inp, out = tmp_path / "input.bin", tmp_path / "output.bin"
    subprocess.run([sys.executable, str(tool), "gen", str(inp)], check=True)
    env = dict(os.environ)
    env.pop("GPUWM_WRF_EXACT", None)
    if strict:
        env["GPUWM_WRF_EXACT"] = "1"
    command = [sys.executable, str(tool), "run", str(inp), str(out)]
    if generation is not None:
        command += ["--generation", generation]
    subprocess.run(command, env=env, check=True)
    expected = ROOT / "tests/fixtures/noahmp-snow-age" / (
        "expected461.json" if generation == "wrf_461" else "expected470.json")
    subprocess.run([sys.executable, str(ROOT / "tools/noahmp_snow_age_oracle/full_hashes.py"),
                    str(inp), str(out), str(tmp_path / "receipt.json"),
                    "--expected", str(expected)], check=True)
    result = json.loads((tmp_path / "receipt.json").read_text())
    assert result["passed"]
    assert result["words"] == 19614


@requires_gpu
def test_one_bit_changed_night_age_term_is_rejected():
    from tools.noahmp_snow_age_oracle.compare import initial, pack
    from gpuwm.core.noahmp_kernel_sources import runtime_unit, compile_runtime_unit
    from gpuwm.core.noahmp_radiation import albedo
    from gpuwm.core.fp32_ulp import assert_bit_exact

    kw = initial()
    kw.update(cosz=np.float32(-0.3), sneqv=np.float32(50),
              sneqvo=np.float32(50), tauss=np.float32(0.8))
    want = np.float32(albedo(**kw)["tauss"])
    source = runtime_unit("noahmp_radiation").source
    term = "tauss = sa_out[0]; fage = sa_out[1];"
    assert source.count(term) == 1
    source = source.replace(term, "tauss = __uint_as_float(__float_as_uint(sa_out[0]) ^ 1u); fage = sa_out[1];")
    module = compile_runtime_unit("noahmp_radiation", module_key="snow-age:one-bit", source=source)
    inputs = np.frombuffer(pack(kw, False), np.float32, offset=4)
    inputs = cp.asarray(np.concatenate([inputs, np.zeros(4, np.float32)]))
    out = cp.empty(19, dtype=cp.float32)
    module.get_function("noahmp_radiation_batch_night_age")(
        (1,), (1,), (inputs, cp.asarray([1], dtype=cp.int32), out, np.int32(1)))
    got = cp.asnumpy(out)[18:19]
    assert got.view(np.uint32)[0] == (want.view(np.uint32) ^ 1)
    with pytest.raises(AssertionError):
        assert_bit_exact(got, np.array([want], np.float32), "planted TAUSS")
