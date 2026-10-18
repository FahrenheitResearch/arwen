"""ACSNOM is accumulated in WRF's left-to-right order (CPU, no device).

``module_sf_noahmpdrv.F:1243`` is
``ACSNOM = ACSNOM + QMELT*DT + PONDING + PONDING1 + PONDING2``: four binary32
roundings on a running total.  WOOF summed ``ACSNOM + (QMELT*DT +
PONDINGXY)``, borrowing the grouping PONDINGXY has at :1296, and the WRF
v4.6.1 column oracle (tests/test_noahmp_wrf461_column_oracle.py) measured
1 ULP on 8 snowmelt values.  These tests pin the order of the shared helper
every write-back path now calls, and that the helper is what those paths
call; the device kernel's spelling is held to the same oracle on the GPU.
"""

from __future__ import annotations

import inspect

import numpy as np

from gpuwm.core import noahmp_runtime
from gpuwm.core.noahmp_runtime import acsnom_accumulate


def _arrays(seed=7, n=20000):
    rng = np.random.default_rng(seed)
    f = np.float32
    acsnom = rng.uniform(0.0, 40.0, n).astype(f)
    qmelt = rng.uniform(0.0, 2.0e-3, n).astype(f)
    ponding = rng.uniform(0.0, 0.5, n).astype(f)
    ponding1 = rng.uniform(0.0, 0.2, n).astype(f)
    ponding2 = rng.uniform(0.0, 0.2, n).astype(f)
    return acsnom, qmelt, f(300.0), ponding, ponding1, ponding2


def test_acsnom_is_summed_left_to_right_in_binary32():
    acsnom, qmelt, dt, p, p1, p2 = _arrays()
    got = acsnom_accumulate(acsnom, qmelt, dt, p, p1, p2)
    want = np.empty_like(acsnom)
    for k in range(acsnom.size):
        total = np.float32(acsnom[k] + np.float32(qmelt[k] * dt))
        total = np.float32(total + p[k])
        total = np.float32(total + p1[k])
        want[k] = np.float32(total + p2[k])
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32))


def test_the_grouped_order_really_differs_so_the_pin_can_fail():
    acsnom, qmelt, dt, p, p1, p2 = _arrays()
    grouped = acsnom + (qmelt * dt + ((p + p1) + p2))
    got = acsnom_accumulate(acsnom, qmelt, dt, p, p1, p2)
    assert (grouped.view(np.int32) != got.view(np.int32)).sum() > 100


def test_every_python_write_back_path_calls_the_helper():
    source = inspect.getsource(noahmp_runtime)
    assert 'r["qmelt"] * dt + ponding' not in source
    assert 'dev("qmelt") * dt32f + ponding' not in source
    for function in (noahmp_runtime._write_back_slab,
                     noahmp_runtime._write_back_batch,
                     noahmp_runtime._glacier_lsm_step):
        assert "acsnom_accumulate(" in inspect.getsource(function), function


def test_the_device_write_back_kernel_spells_the_same_order():
    from gpuwm.core import noahmp_slab_libm
    source = inspect.getsource(noahmp_slab_libm.write_slab_arithmetic)
    assert ("o_acsnom[p]=A(A(A(A(o_acsnom[p],M(v_qmelt[c],dt)),v_ponding[c]),"
            "v_ponding1[c]),v_ponding2[c]);") in source
