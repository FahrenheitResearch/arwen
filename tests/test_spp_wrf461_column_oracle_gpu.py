"""WOOF's SPP consumers on the GPU are WRF v4.6.1 word for word.

The oracle and its coverage are described in
tests/test_spp_wrf461_column_oracle.py.  Here every output word is graded:

* MYNN with ``spp_pbl = 1``, the CUDA driver ``mynn_bl_driver_cuda`` over
  72 columns x 8 steps x mixing lengths 1 and 2, both replayed (each step
  from WRF's recorded inputs) and free-running (WOOF integrates steps 2-8
  itself, so a rounding difference would compound): 27 output fields,
  366,912 words per run, 0 ULP required.
* GF with a per-column closure pattern, the SPP-specialised GF unit's
  ``gf_gfdrv_stage`` over 216 columns x two pattern sets, graded at the
  CUP_gf seam (out* tendencies, pret, prets, kbcon, ktop): 0 ULP required.
  As in tests/test_gf_gfdrv_cuda.py, WRF's captured fzu words fill the
  kernel's override slots: WOOF's gamma is its own correctly rounded one
  and is graded separately (tests/test_gf_gamma_correctly_rounded.py).

Measured on an RTX PRO 6000 Blackwell (sm_120, NVRTC 12.9) under both
default arithmetic and GPUWM_WRF_EXACT=1: 0 differing words.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "spp_wrf461_column_oracle"))

import spp_column_fixtures as columns  # noqa: E402
import spp_column_grade as grade  # noqa: E402

FIXTURES = columns.FIXTURE_DIR


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mixlength", (1, 2))
@pytest.mark.parametrize("replay", (True, False), ids=("replay", "free"))
def test_device_mynn_driver_with_spp_is_wrf_word_for_word(mixlength, replay):
    pytest.importorskip("cupy")
    graded = grade.grade_mynn(FIXTURES / f"mynn-spp-ml{mixlength}.npz", mixlength,
                              replay=replay, backend="gpu")
    worst = {name: (field["max_ulp"], field["per_group"]) for name, field
             in graded["fields"].items() if field["max_ulp"]}
    assert worst == {}, worst
    assert graded["words"] == 366912


@pytest.mark.gpu
@requires_gpu
def test_device_mynn_spp_moves_the_answer_and_a_zero_pattern_does_not():
    """The oracle is sensitive: SPP changes every perturbed column, and a
    zero pattern under spp_pbl=1 is the deterministic answer bitwise."""
    import cupy as cp
    import _mynn_families as F
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda
    cases, steps = columns.mynn_steps(FIXTURES / "mynn-spp-ml1.npz")
    blocks = steps[1]
    values = F.step_inputs(blocks)
    rstoch = np.asarray([[row["rstoch"] for row in block] for block in blocks], np.float32)
    kwargs = dict(initflag=1, delt=np.float32(blocks[0][0]["delt"]), flag_qs=True,
                  bl_mynn_mixlength=1)

    def run(spp, pattern):
        inputs = {name: cp.asarray(np.ascontiguousarray(value)) for name, value in values.items()}
        if spp:
            inputs["rstoch"] = cp.asarray(pattern)
        out = mynn_bl_driver_cuda(inputs, spp_pbl=spp, **kwargs)
        return {name: cp.asnumpy(out[name]) for name in ("exch_h", "exch_m", "rthblten", "qke")}

    on, off = run(1, rstoch), run(0, None)
    zero = np.all(rstoch == 0, axis=1)
    assert zero.sum() == 18 and (~zero).sum() == 54
    for name in on:
        a = on[name].reshape(72, -1).view(np.uint32)
        b = off[name].reshape(72, -1).view(np.uint32)
        np.testing.assert_array_equal(a[zero], b[zero], err_msg=name)
    moved = np.any(on["exch_h"].reshape(72, -1) != off["exch_h"].reshape(72, -1), axis=1)
    assert np.all(moved[~zero])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("offset", (0, 9))
def test_device_gf_closure_spp_is_wrf_word_for_word(offset):
    pytest.importorskip("cupy")
    graded = grade.grade_gf(FIXTURES / f"gf-spp-o{offset}.npz")
    worst = {name: (field["max_ulp"], field["per_group"]) for name, field
             in graded["fields"].items() if field["max_ulp"]}
    assert worst == {}, worst
    assert graded["columns"] == 216 and graded["converging_deep"] >= 50


@pytest.mark.gpu
@requires_gpu
def test_device_gf_zero_closure_pattern_is_the_plain_unit():
    pytest.importorskip("cupy")
    assert grade.gf_zero_pattern_identity() == 0
