"""km_opt=2 (1.5-order prognostic TKE) against compiled WRF v4.6.1, bit for bit.

THE BREAKAGE THIS PREVENTS
--------------------------
The km_opt=2 package (cal_deform_and_div, calculate_N2, tke_km, tke_rhs)
had only float64 mirrors and LES statistics behind it.  The column oracle
(tools/tke_km2_wrf461_oracle) ran the unmodified WRF v4.6.1 routines and
WOOF's production launcher on the same float32 words and found, under the
strict build, 35,533 of 1,158,624 output words different (K coefficients up
to 53,930 ULP, BN2 up to 77,538 ULP, the TKE tendency up to 110,978 ULP):

* CUDA's powf/expf/logf in place of the libm power, exponential and log
  (calculate_N2's saturated predicate flipped on cold and tropical columns;
  ``(...)**0.5`` and ``tketmp**1.5`` are libm powers, not SQRT forms);
* ``dx/msftx * dy/msfty`` evaluated as ``(dx/m)*(dy/m)`` instead of
  Fortran's left-to-right ``((dx/m)*dy)/m`` (mlen_h, deltas, the
  mix_upper_bound caps);
* phy_prep's half-level height formed before dividing by g;
* calculate_N2's surface moisture extrapolated from the summed qtot instead
  of species by species;
* tke_shear's surface drag added after both defor13 and defor23 instead of
  between them;
* the positivity limiter's -0.0 lost by adding into the zeroed buffer.

All are fixed default-on in gpuwm/core/kernels/smag2d.cu.  This file holds
the device path to the oracle on 576 columns in 9 regimes (the dry LES
convective boundary layer among them) x 4 namelist arms.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/oracles/tke_km2"
TOOLS = ROOT / "tools/tke_km2_wrf461_oracle"
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"


def _manifest() -> dict:
    return json.loads((FIXTURE / "manifest.json").read_text())


def _tools_module(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"tke_km2_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOLS))
    return module


def test_fixture_is_sealed_to_the_pinned_wrf_build():
    manifest = _manifest()
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name
    build = manifest["wrf_build"]
    assert build["wrf_release"] == "4.6.1" and build["wrf_commit"] == WRF_COMMIT
    assert build["variant"] == "noopt"
    flags = build["commands"][0]
    for flag in ("-O0", "-ffp-contract=off", "-fno-tree-vectorize"):
        assert flag in flags, flag
    pins = _tools_module("build").SOURCE_PINS
    assert build["source_sha256"] == pins
    assert [tuple(a) for a in manifest["arms"]] == list(_tools_module("arms").ARMS)
    assert sum(c["columns"] for c in manifest["cases"]) == 576
    assert len(manifest["cases"]) == 9


def test_the_comparison_sees_one_flipped_word():
    """The instrument can fail: one changed bit is one different word."""
    word_stats = _tools_module("compare").word_stats
    with np.load(FIXTURE / "wrf461-O0-edge_cases.npz") as wrf:
        ref = wrf["iso0_sfx0/rtke"]
    got = ref.copy()
    flat = got.reshape(-1).view(np.uint32)
    flat[123] ^= np.uint32(1)
    stats = word_stats(got, ref)
    assert (stats["different"], stats["max_ulp"]) == (1, 1)
    # A sign flip of zero is a different word, not agreement.
    zero = np.zeros(4, np.float32)
    assert word_stats(-zero, zero)["different"] == 4


@pytest.mark.gpu
def test_strict_build_matches_compiled_wrf_on_every_output_word():
    import cupy  # noqa: F401  (device test; the run itself is a subprocess)
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - CPU box
        pytest.skip(f"no CUDA device: {exc}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPUWM_WRF_EXACT")}
    # The strict build selects arithmetic before gpuwm is imported, so it
    # runs in its own process.  The diffusion stage carries WRF's metric and
    # deformation operation order (gpuwm/wrf_exact.py).
    env.update(GPUWM_WRF_EXACT="1", GPUWM_WRF_EXACT_DIFFUSION="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "tests")] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    out = subprocess.run([sys.executable, str(TOOLS / "check_fixture.py"), str(FIXTURE)],
                         env=env, cwd=ROOT, check=True, capture_output=True, text=True)
    total = json.loads(out.stdout.strip().splitlines()[-1])
    words = sum(t["words"] for t in total.values())
    assert words == 1_158_624
    different = {f: t for f, t in total.items() if t["different"]}
    assert not different, different
