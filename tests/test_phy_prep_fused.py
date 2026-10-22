"""The fused phy_prep launches give the unfused CuPy chain's words.

gpuwm.core.physics._prepare_atmosphere builds the physics atmosphere every
surface/PBL and radiation step.  Its mass-point chain (theta, temperature,
qtot, the hydrostatic layer weight, rho) and its per-level pressure loop now
run as two launches (_prepare_atmosphere_fused).  These gates run both paths
on the same state and require every returned field, and the qtot scratch,
uint32-identical, dual-run -- on states carrying subnormal and signed-zero
condensate (the flush behaviour of the ufuncs the fused statements replace),
1-D and 3-D base-state profiles, an absent species and an extra one.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                                     # pragma: no cover
    _HAS_GPU = False

from gpuwm.core import physics  # noqa: E402

requires_device = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA GPU")
DUAL_RUNS = 2
F = np.float32


class _State(SimpleNamespace):
    def scratch(self, shape, slot, dtype=None):
        buf = self._scratch.get(slot)
        if buf is None:
            buf = cp.zeros(shape, dtype=np.float32 if dtype is None else dtype)
            self._scratch[slot] = buf
        return buf

    def total_theta(self):
        thb = self.thb
        return (thb if thb.ndim == 3 else thb[:, None, None]) + self.thp

    def total_mu(self):
        return self.mub2d + self.mup


def _state(nz, ny, nx, seed, *, base3d, species):
    rng = np.random.default_rng(seed)

    def r(lo, hi, shape):
        return cp.asarray(rng.uniform(lo, hi, shape).astype(F))

    def q(shape, scale):
        a = (rng.random(shape) ** 4 * scale).astype(F)
        flat = a.reshape(-1)
        pick = rng.choice(flat.size, size=flat.size // 8, replace=False)
        specials = np.array([0.0, -0.0, 1.0e-40, 1.4e-45, 1.17e-38, 3.0e-39],
                            F)
        flat[pick] = specials[rng.integers(0, specials.size, pick.size)]
        return cp.asarray(a)

    shape = (nz, ny, nx)
    st = _State(_scratch={})
    st.p = r(5000.0, 101000.0, shape)
    st.thb = r(280.0, 420.0, shape if base3d else (nz,))
    st.thp = r(-8.0, 8.0, shape)
    st.mub2d = r(50000.0, 96000.0, (ny, nx))
    st.mup = r(-900.0, 900.0, (ny, nx))
    st.c1h = r(0.0, 1.0, (nz,))
    st.c2h = r(0.0, 4000.0, (nz,))
    st.dnw = r(-0.05, -0.001, (nz,))
    st.alt = r(0.8, 6.0, shape)
    st.u = r(-40.0, 40.0, (nz, ny, nx + 1))
    st.v = r(-40.0, 40.0, (nz, ny + 1, nx))
    st.phb = r(0.0, 2.0e5, (nz + 1, ny, nx) if base3d else (nz + 1,))
    st.php = r(-50.0, 50.0, (nz + 1, ny, nx))
    st.p_top = 5000.0
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
        setattr(st, name, None)
    for name in species:
        setattr(st, name, q(shape, 2.0e-2 if name == "qv" else 3.0e-3))
    if "qv" in species:
        # Columns with no vapour: qtot starts from a subnormal or a signed
        # zero, where a flushing and a non-flushing add part ways.
        st.qv[:, 0, :2] = 0.0
        st.qv[:, 1, :2] = F(-0.0)
    return st


@requires_device
@pytest.mark.parametrize("base3d", [False, True])
@pytest.mark.parametrize("species", [
    ("qv", "qc", "qr", "qi", "qs", "qg"),
    ("qv", "qc", "qr"),
    ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
    (),
])
def test_fused_phy_prep_equals_the_cupy_chain(monkeypatch, base3d, species):
    nz, ny, nx = 13, 9, 11
    st = _state(nz, ny, nx, seed=len(species) * 10 + base3d,
                base3d=base3d, species=species)
    unfused_state = _state(nz, ny, nx, seed=len(species) * 10 + base3d,
                           base3d=base3d, species=species)
    with monkeypatch.context() as m:
        m.setattr(physics, "_prepare_atmosphere_fused", lambda state: None)
        want = {k: cp.asnumpy(v) for k, v in
                physics._prepare_atmosphere(unfused_state).items()}
    want_qtot = cp.asnumpy(unfused_state._scratch["physics_qtot"])
    for run in range(DUAL_RUNS):
        got_d = physics._prepare_atmosphere_fused(st)
        assert got_d is not None, "fused path declined a float32 device state"
        got = {k: cp.asnumpy(v) for k, v in got_d.items()}
        assert set(got) == set(want)
        for k in want:
            assert got[k].shape == want[k].shape, k
            diff = int(np.count_nonzero(
                got[k].view(np.uint32) != want[k].view(np.uint32)))
            assert diff == 0, f"{k}: {diff} of {want[k].size} words (run {run})"
        qtot = cp.asnumpy(st._scratch["physics_qtot"])
        assert np.array_equal(qtot.view(np.uint32), want_qtot.view(np.uint32))


class _WrfOracleState(_State):
    """The compiled-WRF 4.7.1 phy_prep oracle's state shape
    (gpuwm/verify/bigstep_prep_oracle.py): total theta is WRF's T + 300 and
    there is no thb/thp at all, so the fused path must read the state's own
    total_theta()/total_mu() rather than reassemble them."""

    def total_theta(self):
        return self.t_wrf + F(300)


@requires_device
@pytest.mark.parametrize("species", [("qv", "qc", "qr", "qi", "qs", "qg"), ()])
def test_fused_phy_prep_reads_the_states_own_total_theta_and_mu(
        monkeypatch, species):
    nz, ny, nx = 11, 7, 9

    def build():
        st = _state(nz, ny, nx, seed=77, base3d=False, species=species)
        oracle = _WrfOracleState(**vars(st))
        oracle.t_wrf = oracle.thp * F(3.0)
        del oracle.thb, oracle.thp
        return oracle

    unfused_state, st = build(), build()
    with monkeypatch.context() as m:
        m.setattr(physics, "_prepare_atmosphere_fused", lambda state: None)
        want = {k: cp.asnumpy(v) for k, v in
                physics._prepare_atmosphere(unfused_state).items()}
    for run in range(DUAL_RUNS):
        got_d = physics._prepare_atmosphere_fused(st)
        assert got_d is not None, "fused path declined the oracle-shaped state"
        for k in want:
            got = cp.asnumpy(got_d[k])
            diff = int(np.count_nonzero(
                got.view(np.uint32) != want[k].view(np.uint32)))
            assert diff == 0, f"{k}: {diff} of {want[k].size} words (run {run})"


def test_fused_paths_decline_when_the_module_runs_on_numpy(monkeypatch):
    """The CPU shard rebinds physics.cp to NumPy (tests/test_preflight.py
    runs _prepare_atmosphere on a NumPy DomainState).  NumPy arrays are then
    ``cp.ndarray`` and NumPy has no ElementwiseKernel, so both fused paths
    must decline before touching the state, and the unfused statements run
    on the NumPy arrays.  No device is needed."""
    monkeypatch.setattr(physics, "cp", np)
    rng = np.random.default_rng(5)
    nz, ny, nx = 4, 3, 5
    st = _State(_scratch={})
    requested = []
    st.scratch = lambda shape, slot, dtype=None: requested.append(slot)
    for name, shape in (("p", (nz, ny, nx)), ("thb", (nz,)),
                        ("thp", (nz, ny, nx)), ("mub2d", (ny, nx)),
                        ("mup", (ny, nx)), ("c1h", (nz,)), ("c2h", (nz,)),
                        ("dnw", (nz,)), ("alt", (nz, ny, nx)),
                        ("qv", (nz, ny, nx))):
        setattr(st, name, rng.uniform(0.5, 2.0, shape).astype(F))
    assert physics._prepare_atmosphere_fused(st) is None
    assert requested == []
    a, b, c = (rng.standard_normal((nz, ny, nx)).astype(F) for _ in range(3))
    out = np.full((nz, ny, nx), np.nan, F)
    physics._compose_three(out, a, b, c)
    want = a.copy()
    want += b
    want += c
    assert np.array_equal(out.view(np.uint32), want.view(np.uint32))


@requires_device
def test_fused_phy_prep_declines_non_float32_inputs():
    st = _state(4, 3, 5, seed=1, base3d=False, species=("qv",))
    st.c1h = st.c1h.astype(cp.float64)
    assert physics._prepare_atmosphere_fused(st) is None


@requires_device
@pytest.mark.parametrize("alias", [False, True])
def test_compose_three_equals_the_unfused_statements(alias):
    """_compose_tendencies' PBL + radiation + cumulus sum in one launch
    against the copy and two in-place adds it replaces, with signed zeros
    and subnormals in every operand, and with the target aliasing the PBL
    array (the bldt = 0 composition)."""
    rng = np.random.default_rng(31 + alias)
    shape = (17, 9, 12)

    def field():
        a = (rng.standard_normal(shape) * 1e-4).astype(F)
        flat = a.reshape(-1)
        pick = rng.choice(flat.size, size=flat.size // 5, replace=False)
        specials = np.array([0.0, -0.0, 1.0e-40, -1.0e-40, 1.4e-45,
                             -1.17e-38], F)
        flat[pick] = specials[rng.integers(0, specials.size, pick.size)]
        return a

    pbl, rad, cu = field(), field(), field()
    want = cp.asarray(pbl)
    want[...] = cp.asarray(pbl)
    want += cp.asarray(rad)
    want += cp.asarray(cu)
    want = cp.asnumpy(want)
    for run in range(DUAL_RUNS):
        pbl_d = cp.asarray(pbl)
        out = pbl_d if alias else cp.full(shape, np.nan, cp.float32)
        physics._compose_three(out, pbl_d, cp.asarray(rad), cp.asarray(cu))
        got = cp.asnumpy(out)
        diff = int(np.count_nonzero(got.view(np.uint32) != want.view(np.uint32)))
        assert diff == 0, f"{diff} of {want.size} words (run {run})"


@requires_device
def test_the_compiled_wrf_phy_prep_oracle_runs_the_fused_path(monkeypatch):
    """tests/test_bigstep_prep_wrf471_parity.py holds _prepare_atmosphere to
    compiled WRF 4.7.1's phy_prep word for word.  That check only covers the
    fused launches if the oracle's state (float32 device fields, WRF's T + 300
    as total theta, no thb/thp) is one the fused path accepts; this pins that
    it is, so the WRF oracle stays the direct check of the fused code."""
    from gpuwm.verify import bigstep_prep_oracle as oracle
    fixture = oracle.load_prep_fixture("real-periodic2")
    taken = []
    fused = physics._prepare_atmosphere_fused

    def spy(state):
        result = fused(state)
        taken.append(result is not None)
        return result

    monkeypatch.setattr(physics, "_prepare_atmosphere_fused", spy)
    oracle.prep_port_outputs(fixture)
    assert taken and all(taken), taken
