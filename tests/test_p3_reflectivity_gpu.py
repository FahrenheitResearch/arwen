"""P3's radar observation operator H_Z(x) on the device (audit S14).

P3 was refused by the reflectivity operator because its Z was fused with a
state update inside the scheme's final diagnostics.  The Z half now runs on
local copies (gpuwm/core/kernels/p3_zdiag.cu).  These gates hold it to
the four things an observation operator must be: the scheme's own Z, pure,
one clear-air value, and reachable from the DA operator table.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _stepped_case(steps=3):
    """The P3 CUDA gate's mixed-phase slab, integrated ``steps`` times."""
    import cupy as cp

    from gpuwm.core import p3_device as PD
    try:
        from tests.test_p3_cuda_gpu import _tiny_column_case
    except ImportError:                                  # rootdir-less run
        from test_p3_cuda_gpu import _tiny_column_case

    host, nk, ncol = _tiny_column_case()
    fields = {k: cp.asarray(v) for k, v in host.items()}
    diag = {d: cp.zeros((nk, ncol), dtype=cp.float32) for d in PD.DIAG_SLOTS}
    surf = {s: cp.zeros(ncol, dtype=cp.float32) for s in PD.SURF_SLOTS}
    ws = PD.make_workspace(ncol, nk)
    for it in range(1, steps + 1):
        PD.run_p3_device(fields, diag, surf, workspace=ws, dt=20.0, it=it)
    return fields, diag, nk, ncol


def _operator_inputs(fields):
    from gpuwm.core import p3_device as PD

    return {name: fields[name] for name in PD.REFLECTIVITY_FIELDS}


def test_the_operator_is_the_z_the_step_wrote():
    """On a state that has just left a P3 step, H_Z(x) is the step's own
    zdbz wherever the step diagnosed one.  The only difference allowed is
    the air density: the step forms it from its ENTRY temperature, the
    operator from the state it is given, so the tolerance is a few
    hundredths of a dB, not a scheme difference."""
    import cupy as cp

    from gpuwm.core import p3_device as PD

    fields, diag, nk, ncol = _stepped_case()
    out = cp.empty((nk, ncol), dtype=cp.float32)
    PD.run_p3_reflectivity(_operator_inputs(fields), out)
    step = cp.asnumpy(diag["zdbz"])
    got = cp.asnumpy(out)
    diagnosed = step > -98.0
    assert diagnosed.any() and np.any(step[diagnosed] > 0.0)
    assert np.abs(got[diagnosed] - step[diagnosed]).max() < 0.05


def test_evaluating_the_operator_moves_nothing():
    import cupy as cp

    from gpuwm.core import p3_device as PD

    fields, _diag, nk, ncol = _stepped_case()
    before = {k: cp.asnumpy(v).tobytes() for k, v in fields.items()}
    out = cp.empty((nk, ncol), dtype=cp.float32)
    PD.run_p3_reflectivity(_operator_inputs(fields), out)
    after = {k: cp.asnumpy(v).tobytes() for k, v in fields.items()}
    assert before == after


def test_device_and_host_replay_agree_and_repeat_bit_for_bit():
    import cupy as cp

    from gpuwm.core import p3_device as PD

    fields, _diag, nk, ncol = _stepped_case()
    inputs = _operator_inputs(fields)
    first = cp.empty((nk, ncol), dtype=cp.float32)
    second = cp.empty((nk, ncol), dtype=cp.float32)
    PD.run_p3_reflectivity(inputs, first)
    PD.run_p3_reflectivity(inputs, second)
    assert cp.asnumpy(first).tobytes() == cp.asnumpy(second).tobytes()
    host = PD.p3_reflectivity_host(
        {k: cp.asnumpy(v) for k, v in inputs.items()})
    assert np.abs(host - cp.asnumpy(first)).max() < 1e-3


def test_clear_air_reads_the_one_floor_exactly():
    import cupy as cp

    from gpuwm.core import p3_device as PD
    from gpuwm.da.obsop import CLEAR_AIR_FLOOR_DBZ

    fields, _diag, nk, ncol = _stepped_case()
    inputs = dict(_operator_inputs(fields))
    for name in ("qr", "nr", "qi", "qir", "ni", "qib"):
        inputs[name] = cp.zeros_like(inputs[name])
    out = cp.empty((nk, ncol), dtype=cp.float32)
    PD.run_p3_reflectivity(inputs, out)
    values = np.unique(cp.asnumpy(out))
    assert values.tolist() == [np.float32(PD.CLEAR_AIR_DBZ)]
    assert CLEAR_AIR_FLOOR_DBZ[50] == float(PD.CLEAR_AIR_DBZ)


def test_the_da_operator_table_reaches_p3():
    """simulated_reflectivity dispatches mp_physics=50 by its table row."""
    import cupy as cp

    from gpuwm.da import obsop

    fields, _diag, nk, ncol = _stepped_case()
    shape = (nk, ncol, 1)
    slots = {}

    def scratch(shape_, name):
        slots[name] = cp.empty(shape_, dtype=cp.float32)
        return slots[name]

    th = fields["th"].reshape(shape)
    state = SimpleNamespace(
        p=fields["pres"].reshape(shape), thb=cp.zeros(nk, cp.float32),
        thp=th, scratch=scratch,
        **{n: fields[n].reshape(shape)
           for n in ("qv", "qr", "nr", "qi", "qir", "ni", "qib")})
    got = obsop.simulated_reflectivity(state, SimpleNamespace(mp_physics=50))
    assert got.shape == shape
    assert float(cp.asnumpy(got).max()) > 0.0
