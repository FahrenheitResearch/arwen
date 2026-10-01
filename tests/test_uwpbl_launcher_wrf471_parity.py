"""The UW PBL through its PRODUCT launcher, against WRF v4.7.1, bit for bit.

``gpuwm.core.uwpbl.uwpbl_step`` is what ``PhysicsDriver._run_uwpbl`` calls:
the kernel as the loader assembles it (``_EXTRA_HEADERS["uwpbl"]``), the
saturation table uploaded from ``gpuwm.core.uwpbl_constants``, the
workspace pools and the column chunking.  This module drives it on every
column of every step of the oracle's fixtures of record (see
tests/test_uwpbl_wrf471_parity.py, which pins those fixtures and grades the
CPU reference on them) and compares every float32 output word, and the
int32 KPBL, for EQUALITY.

The one licence-bound exception is stated where it lives: WRF's binary64
cos/acos are glibc's LGPL IBM code, which is not transcribed; the kernel
uses correctly rounded ones.  A column whose output differs from the oracle
must be traced to a cos/acos argument where the two disagree; on the
fixtures of record there is none, so any difference fails here.
Split from tests/test_uwpbl_wrf471_parity.py because the launch helper
imports cupy, which marks a whole module ``gpu`` (tests/conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.uwpbl_oracle import (MASS_INPUTS, MASS_OUTPUTS, dims,
                                       step_arrays)
from test_uwpbl_wrf471_parity import GRIDS, _fixture, _mismatched_columns


def _launch(a, nk, ncol, *, budget_bytes=None):
    import cupy as cp

    from gpuwm.core import uwpbl

    def mass(name):
        return cp.asarray(np.ascontiguousarray(a[name].T[:, None, :]))

    inputs = {name: mass(name) for name in MASS_INPUTS}
    for name in ("p8w", "z_at_w"):
        inputs[name] = mass(name)
    for name in ("hfx", "qfx", "ust", "ht"):
        inputs[name] = cp.asarray(a[name][None, :])
    carried = {"kvm3d": mass("kvm3d_in"), "kvh3d": mass("kvh3d_in"),
               "tauresx2d": cp.asarray(a["tauresx2d_in"][None, :]),
               "tauresy2d": cp.asarray(a["tauresy2d_in"][None, :])}
    kwargs = {} if budget_bytes is None else {"budget_bytes": budget_bytes}
    out = uwpbl.uwpbl_step(inputs, carried, dt=float(a["dt"]),
                           itimestep=int(a["itimestep"]), **kwargs)
    got = {}
    for name in MASS_OUTPUTS + ("tke_pbl", "turbtype3d", "smaw3d"):
        got[name] = cp.asnumpy(out[name])[:, 0, :].T
    got["kvm3d"] = cp.asnumpy(carried["kvm3d"])[:, 0, :].T
    got["kvh3d"] = cp.asnumpy(carried["kvh3d"])[:, 0, :].T
    got["tauresx2d"] = cp.asnumpy(carried["tauresx2d"])[0]
    got["tauresy2d"] = cp.asnumpy(carried["tauresy2d"])[0]
    for name in ("tpert2d", "qpert2d", "wpert2d", "pblh2d", "kpbl2d"):
        got[name] = cp.asnumpy(out[name])[0]
    return got


@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
def test_every_step_of_every_column_is_wrfs_words(grid):
    import cupy  # noqa: F401  (marks this test for -m "not gpu")

    fx = _fixture(grid)
    ncol, nk, nsteps = dims(fx)
    for step in range(1, nsteps + 1):
        want = step_arrays(fx, step)
        got = _launch(want, nk, ncol)
        bad = _mismatched_columns(got, want)
        assert not bad, (f"{grid} step {step}: columns {sorted(bad)} "
                         "differ from the oracle")


@pytest.mark.gpu
def test_a_chunked_launch_equals_a_single_one():
    import cupy  # noqa: F401

    fx = _fixture("g44")
    ncol, nk, _ = dims(fx)
    a = step_arrays(fx, 2)
    whole = _launch(a, nk, ncol)
    from gpuwm.core.uwpbl import uwpbl_workspace_slots
    r8, i4 = uwpbl_workspace_slots(nk)
    five = 5 * (8 * r8 + 4 * i4)           # five columns per launch
    chunked = _launch(a, nk, ncol, budget_bytes=five)
    for name, value in whole.items():
        value = np.asarray(value)
        other = np.asarray(chunked[name])
        if value.dtype != np.int32:
            value, other = value.view(np.uint32), other.view(np.uint32)
        assert np.array_equal(value, other), name
