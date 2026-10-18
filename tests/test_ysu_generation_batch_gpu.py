"""YSU generations against the unmodified WRF releases, on every GPU leg.

The references are committed (tests/data/oracles/ysu_generations): every
word unmodified WRF 4.8.0 and 4.6.1 ``bl_ysu_run`` wrote on the 1,188-column
corpus over three steps, recorded by ``tools/ysu_generations_oracle/gate.py
--fixture record`` from the pinned public sources (PROVENANCE.md there).  So
the strict leg and the GPU shard leg grade the default PBL against WRF in
their own arithmetic mode without a Fortran toolchain.  With
``YSU_COLUMN_FIXTURE`` set to a live build's ``wrf_480-step0`` (the source
leg), the gate rebuilds every reference from the oracles and also requires
the committed words.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import subprocess
import sys
from conftest import requires_gpu
from gpuwm.wrf_exact import ENABLED as WRF_EXACT

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/data/oracles/ysu_generations"
LIVE = os.environ.get("YSU_COLUMN_FIXTURE")
#: wrf_461 under default arithmetic is the 2.8.9 reproduction (its identity
#: arm grades it against 2.8.9's words); under GPUWM_WRF_EXACT=1 it is the
#: source-ordered WRF 4.6.1 arm and is graded against WRF 4.6.1 here.
GRADED = ("wrf_480", "wrf_461") if WRF_EXACT else ("wrf_480",)


def _gate(tmp_path, version, *extra):
    if LIVE:
        root, mode = Path(LIVE).parent, "verify"
    else:
        root, mode = tmp_path / "work", "replay"
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))))
    result = subprocess.run(
        [sys.executable, "-m", "tools.ysu_generations_oracle.gate", "--root", str(root),
         "--version", version, "--out", str(out), "--fixture", mode, *extra],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.stdout.strip(), result.stderr[-4000:]
    return result.returncode, json.loads(result.stdout.strip().splitlines()[-1]), out


@pytest.mark.gpu  # the card work is in the gate subprocess, which the AST marker cannot see
@requires_gpu
@pytest.mark.parametrize("version", GRADED)
def test_generation_is_unmodified_wrf_word_for_word(tmp_path, version):
    rc, line, out = _gate(tmp_path, version)
    assert line["strict_arithmetic"] == bool(WRF_EXACT)
    assert rc == 0 and line["pass"], line
    summary = json.loads((out / "summary.json").read_text())
    assert [s["columns"] for s in summary["steps"]] == [1188] * 3
    if version == "wrf_480":
        # The corpus reaches both sides of WRF 4.8's top-down cap.
        for step in summary["steps"]:
            d = step["topdown_diagnostics"]
            assert d["capped"] > 0 and d["below_cap"] > 0, d


@pytest.mark.gpu  # the card work is in the gate subprocess, which the AST marker cannot see
@requires_gpu
def test_gate_catches_a_one_bit_cap_error(tmp_path):
    rc, line, _ = _gate(tmp_path, "wrf_480", "--plant")
    assert rc == 1 and not line["pass"]
    assert all(n > 0 for n in line["unexplained_words"]), line


def _step0(tmp_path):
    if LIVE:
        return Path(LIVE)
    import lzma
    step0 = tmp_path / "wrf_480-step0"
    step0.mkdir()
    (step0 / "ysu-wide-inputs.bin").write_bytes(lzma.decompress((FIXTURE / "columns.bin.xz").read_bytes()))
    (step0 / "ysu-wide-wrf.bin").write_bytes(
        lzma.decompress((FIXTURE / "wrf_480/step0/ysu-wide-wrf.bin.xz").read_bytes()))
    return step0


@requires_gpu
@pytest.mark.parametrize("version", ("wrf_480", "wrf_461"))
def test_generation_reaches_packed_shared_column_route(tmp_path, version):
    """Actual scalar and packed generation paths must write identical words."""
    import cupy as cp
    from gpuwm.verify.ysu_wide_oracle import load_wide_oracle
    from gpuwm.core.ysu import launch_ysu
    from gpuwm.ensemble.batch_physics import prepare_ysu_column_batch
    cols = [c for c in load_wide_oracle(_step0(tmp_path)) if c.topo == 0 and c.topdown == 1]
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


@pytest.mark.gpu  # the card work is in the gate subprocess, which the AST marker cannot see
@requires_gpu
def test_shallow_column_matches_declared_defined_reference(tmp_path):
    root, mode = (Path(LIVE).parent, "verify") if LIVE else (tmp_path / "work", "replay")
    script = ROOT / "tools/ysu_generations_oracle/check_shallow.py"
    # Graded against WRF's words, so the port runs strict: a default process
    # bounds the explicit entrainment flux by its donor Courant number
    # (ysu_480.cu, hexeng/hex040-fix-global-ysu 1edfbfe95), which moves this
    # shallow column's theta and vapour by design.
    env = dict(os.environ, GPUWM_WRF_EXACT="1", PYTHONPATH=os.pathsep.join(
        filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))))
    subprocess.run([sys.executable, str(script), str(root), str(tmp_path / "shallow.json"),
                    "--fixture", mode], check=True, cwd=ROOT, env=env)
