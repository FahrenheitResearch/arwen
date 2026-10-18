"""New Tiedtke's default arithmetic never asks a layer for more water than it holds over the step.

The defect: on the 120 km globe at 600 s (x1.40962, GFS 2026-10-11 12Z, +18 h, cell 398) the scheme
took 4.44 g/kg from a layer holding 2.18 g/kg and put 3.82 g/kg and 2.6 K into the layer above, a
+7.9 K single hot layer.  The first column carries that column's own rates and state at the bad step
(levels 12 to 17); the second is a column the scheme already leaves non-negative.
"""
import numpy as np
import pytest

# The bound is array-module generic (it takes the module of its inputs), so this control runs on
# NumPy and needs no card; a forecast runs the same function on CuPy arrays.
_asnumpy = np.asarray

from gpuwm.core.ntiedtke import bound_donor_depletion  # noqa: E402

DT = 600.0


def _columns():
    qv = np.array([[2.22e-3, 2.22e-3], [1.85e-3, 1.85e-3], [2.18e-3, 2.18e-3],
                   [0.63e-3, 0.63e-3], [1.78e-3, 1.78e-3], [1.47e-3, 1.47e-3]], dtype=np.float32)
    rqv = (np.array([[-0.0489e-3, -0.0489e-3], [-0.0489e-3, -0.0489e-3], [-4.44e-3, -0.5e-3],
                     [3.82e-3, 0.4e-3], [0.0, 0.0], [0.0, 0.0]]) / DT).astype(np.float32)
    rth = (np.array([[0.0681, 0.0681], [0.0656, 0.0656], [-0.0109, -0.0109],
                     [2.61, 0.3], [0.0, 0.0], [0.0, 0.0]]) / DT).astype(np.float32)
    zero = np.zeros_like(qv)
    out_lev = {"rthcuten": np.array(rth), "rqvcuten": np.array(rqv),
               "rqccuten": np.array(zero), "rqicuten": np.array(zero),
               "rucuten": np.array(rth), "rvcuten": np.array(rth)}
    out_sfc = {"raincv": np.array(np.array([0.5, 0.2], dtype=np.float32)),
               "pratec": np.array((np.array([0.5, 0.2]) / DT).astype(np.float32))}
    donors = {"rqvcuten": np.array(qv), "rqccuten": np.array(zero),
              "rqicuten": np.array(zero)}
    return out_lev, out_sfc, donors, qv


def test_no_layer_loses_more_water_than_it_holds_and_the_column_scales_as_one():
    out_lev, out_sfc, donors, qv = _columns()
    before = {k: _asnumpy(v).copy() for k, v in {**out_lev, **out_sfc}.items()}
    bound_donor_depletion(out_lev, out_sfc, donors, DT)
    after_qv = qv + _asnumpy(out_lev["rqvcuten"]) * np.float32(DT)
    assert after_qv.min() >= -1e-9
    alpha = float(_asnumpy(out_sfc["raincv"])[0] / before["raincv"][0])
    assert alpha == pytest.approx(2.18 / 4.44, rel=1e-4)
    for name in out_lev:
        np.testing.assert_allclose(_asnumpy(out_lev[name])[:, 0], before[name][:, 0] * alpha,
                                   rtol=1e-5)
    # The column the scheme already leaves non-negative does not move by a single word.
    for name, value in {**out_lev, **out_sfc}.items():
        got = _asnumpy(value)[..., 1]
        assert np.array_equal(got.view(np.uint32), before[name][..., 1].view(np.uint32))
