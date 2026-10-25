"""Grell-Freitas per-column active mask (``cumulus_active_column``).

A variable-resolution mesh runs GF in its coarse cells and no cumulus in its
convection-allowing region.  The adapter launches only the active columns
and zeroes the rest.  The kernel is one thread per column, so:

* an all-True mask is bitwise the unmasked seam, every output, every step;
* with a partial mask, every active column is bitwise the same column of
  the unmasked seam, every output, every step (column physics never reads a
  neighbour), and the GF outputs of every inactive column are zero.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.core import gf as gf_module  # noqa: E402
from gpuwm.core import mpas_column_batch as mcb  # noqa: E402

import test_mpas_column_batch_gpu as base  # noqa: E402

STEPS = 3


def _record_gf(monkeypatch):
    """Record every GF result (rthcuten, rqvcuten, rainc) the seam gets."""
    calls = []
    original = gf_module.GrellFreitas.__call__

    def wrapped(self, **kwargs):
        result = original(self, **kwargs)
        calls.append({name: cp.asnumpy(getattr(result, name)).copy()
                      for name in ("rthcuten", "rqvcuten", "rainc")})
        return result

    monkeypatch.setattr(gf_module.GrellFreitas, "__call__", wrapped)
    return calls


def _run(monkeypatch, mask):
    extra = {} if mask is None else {"cumulus_active_column": mask}
    seam = base._seam(cumulus_scheme="gf", cumulus_seconds=base.DT, **extra)
    calls = _record_gf(monkeypatch)
    inputs = base._profile(seed=3)
    state2 = base._phase2_state(inputs)
    outputs = []
    for _ in range(STEPS):
        result, _ = base._step(seam, inputs, state2)
        outputs.append({name: cp.asnumpy(getattr(result, name)).copy()
                        for name in mcb._OUTPUT_BUFFERS})
    monkeypatch.undo()
    return seam, calls, outputs


def _columns(array, keep):
    return array[..., keep]


def test_all_true_mask_is_bitwise_the_unmasked_seam(monkeypatch):
    _, gf_a, out_a = _run(monkeypatch, None)
    seam_b, gf_b, out_b = _run(monkeypatch, np.ones(base.NCOL, dtype=bool))
    assert len(gf_a) == len(gf_b) == STEPS
    for step in range(STEPS):
        for name, value in out_a[step].items():
            assert np.array_equal(value, out_b[step][name], equal_nan=True), (step, name)
        for name, value in gf_a[step].items():
            assert np.array_equal(value, gf_b[step][name]), (step, name)
    assert seam_b._identity["gf_active_column"] == f"per-column: {base.NCOL} of {base.NCOL} active"


def test_partial_mask_keeps_active_columns_and_zeroes_the_rest(monkeypatch):
    mask = np.zeros(base.NCOL, dtype=bool)
    mask[::2] = True
    _, gf_a, out_a = _run(monkeypatch, None)
    seam_c, gf_c, out_c = _run(monkeypatch, mask)
    off = ~mask
    # The test is not vacuous: GF made rates in the unmasked run.
    assert any(np.any(call["rthcuten"] != 0.0) for call in gf_a)
    for step in range(STEPS):
        for name, value in out_a[step].items():
            assert np.array_equal(_columns(value, mask),
                                  _columns(out_c[step][name], mask),
                                  equal_nan=True), (step, name)
        for name, value in gf_c[step].items():
            assert np.all(_columns(value, off) == 0.0), (step, name)
            assert np.array_equal(_columns(value, mask),
                                  _columns(gf_a[step][name], mask)), (step, name)
    assert seam_c._identity["gf_active_column"] == (
        f"per-column: {int(mask.sum())} of {base.NCOL} active")


def test_no_active_column_launches_nothing_and_rates_are_zero(monkeypatch):
    _, gf_d, out_d = _run(monkeypatch, np.zeros(base.NCOL, dtype=bool))
    for call in gf_d:
        for value in call.values():
            assert np.all(value == 0.0)
    for step in out_d:
        for name, value in step.items():
            assert np.isfinite(value).all(), name


def test_unmasked_identity_has_no_mask_key():
    seam = base._seam(cumulus_scheme="gf", cumulus_seconds=base.DT)
    assert "gf_active_column" not in seam._identity


def test_mask_refusals_name_the_breakage():
    with pytest.raises(ValueError, match="Grell-Freitas adapter only"):
        base._seam(cumulus_scheme="kf",
                   cumulus_active_column=np.ones(base.NCOL, dtype=bool))
    with pytest.raises(ValueError, match="must be boolean"):
        base._seam(cumulus_scheme="gf", cumulus_seconds=base.DT,
                   cumulus_active_column=np.ones(base.NCOL, dtype=np.int32))
    with pytest.raises(ValueError, match="one value per column"):
        base._seam(cumulus_scheme="gf", cumulus_seconds=base.DT,
                   cumulus_active_column=np.ones(base.NCOL + 1, dtype=bool))
