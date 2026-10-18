"""Small device contracts for finite transfers and final vapor ownership.

The unit is the literal WRF transcription graded bitwise by
tests/test_morrison_wrf461_parity.py; these contracts exercise the
owner-ruled row-37 corrections end to end (the wide freezing rates where
WRF's float exponential overflows, the nucleation vapour cap) and the
column water budget through the public launcher and the state adapter.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core import constants as c


MASS = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBER = ("nc", "nr", "ni", "ns", "ng")
SURFACE = ("rainnc", "rainncv", "snownc", "snowncv",
           "graupelnc", "graupelncv", "sr")
EFFECTIVE = ("effc", "effr", "effi", "effs")


def _column(temp=156.0, rain=1.2e-13):
    shape = (2, 1, 1)
    host = {name: np.zeros(shape, np.float32)
            for name in MASS + NUMBER}
    host["pressure"] = np.array([300.0, 250.0], np.float32).reshape(shape)
    host["pii"] = ((host["pressure"] / np.float32(c.P0))
                   ** np.float32(c.RCP))
    host["theta"] = np.float32(temp) / host["pii"]
    host["rho"] = host["pressure"] / (np.float32(c.RD)
                                         * host["theta"] * host["pii"])
    host["dz"] = np.full(shape, 3000.0, np.float32)
    host["qv"][:, 0, 0] = (4.4e-8, 5.28e-8)
    host["qr"].fill(rain)
    host["nr"].fill(0.00475)
    return host


def _launch(host, dt, mode, *, adapter=False, monkeypatch=None):
    """Run the real launcher, optionally through the actual state adapter."""
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core import morrison
    from gpuwm.core.state import DomainState

    shape = host["theta"].shape
    stages = []
    if adapter:
        cfg = RunConfig(nx=shape[2], ny=shape[1], nz=shape[0],
                        dx=1000.0, dy=1000.0, ztop=6000.0, dt=dt,
                        run_seconds=0.0,
                        moist=True, mp_physics=10, morr_rimed_ice=mode)
        state = DomainState(cfg)
        state.p[...] = cp.asarray(host["pressure"])
        state.thb[...] = cp.asarray(host["theta"][:, 0, 0])
        z = np.concatenate(([0.0], np.cumsum(host["dz"][:, 0, 0])))
        state.phb[...] = cp.asarray(z * c.G, dtype=cp.float32)
        for name in MASS + NUMBER:
            getattr(state, name)[...] = cp.asarray(host[name])
        morrison.apply(state, cfg, dt)
        fields = {name: getattr(state, name)
                  for name in MASS + NUMBER + EFFECTIVE}
        fields.update({name: state.scratch(shape[1:], "mp_" + name)
                       for name in SURFACE})
        fields["theta"] = state.thb[:, None, None] + state.thp
        fields["h_diabatic"] = state.h_diabatic
        density = state.scratch(shape, "morr_rho")
    else:
        fields = {name: cp.asarray(value) for name, value in host.items()}
        fields.update({name: cp.zeros(shape[1:], cp.float32)
                       for name in SURFACE})
        fields.update({name: cp.zeros(shape, cp.float32)
                       for name in EFFECTIVE})
        if monkeypatch is not None:
            original = morrison.get_kernel

            def observed(module, name):
                actual = original(module, name)

                def call(*args, **kwargs):
                    actual(*args, **kwargs)
                    stages.append((name, {key: cp.asnumpy(value)
                                          for key, value in fields.items()}))

                return call

            monkeypatch.setattr(morrison, "get_kernel", observed)
        morrison.launch_morrison(**fields, dt=dt, morr_rimed_ice=mode)
        # The launcher returns the scheme's own density in rho (:1325).
        density = fields["rho"]
    got = {name: cp.asnumpy(value) for name, value in fields.items()}
    return got, cp.asnumpy(density).astype(np.float64), stages


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", (0, 1))
@pytest.mark.parametrize("rain", (1.2e-13, 5.0e-4))
@pytest.mark.parametrize("adapter", (False, True))
@pytest.mark.parametrize("temp", (156.0, 135.0))
def test_cold_rain_keeps_finite_state_and_condensed_water(mode, rain, adapter,
                                                          temp):
    """At 135 K WRF's float Bigg exponential overflows (exp of 91.8); the
    row-37 wide rain rate keeps the state finite and the water closed."""
    host = _column(temp=temp, rain=rain)
    got, density, _ = _launch(host, 50.0, mode, adapter=adapter)
    for name, values in got.items():
        assert np.isfinite(values).all(), name
    for name in MASS + NUMBER + SURFACE:
        assert (got[name] >= 0.0).all(), name
    # At 156 K the vapor is untouched; at 135 K the column is ice
    # supersaturated and deposition moves vapor into ice, so the budget
    # there counts vapor too.  Use entry dry density for both inventories;
    # rainncv is total surface water, including frozen precipitation.
    species = MASS[1:] if temp == 156.0 else MASS
    if temp == 156.0:
        np.testing.assert_array_equal(got["qv"], host["qv"])
    weight = density * host["dz"].astype(np.float64)
    initial = float(np.sum(sum(host[name].astype(np.float64)
                               for name in species) * weight))
    condensed = sum(got[name].astype(np.float64) for name in species)
    final = float(np.sum(condensed * weight) + np.sum(got["rainncv"]))
    # These two levels need at most eight substeps at the scheme's speed
    # caps. Allow eight rounded operations per level/substep and 32 for
    # transfer, density conversion and surface summation.
    unit_roundoff = np.finfo(np.float32).eps / 2.0
    nops = 8 * 2 * 8 + 32
    gamma = nops * unit_roundoff / (1.0 - nops * unit_roundoff)
    assert abs(final - initial) <= gamma * initial


@pytest.mark.gpu
@requires_gpu
def test_vapor_only_cold_control_has_no_spurious_water_or_heating():
    host = _column(rain=0.0)
    got, _, _ = _launch(host, 50.0, 1, adapter=True)
    for name in MASS:
        np.testing.assert_array_equal(got[name], host[name])
    np.testing.assert_array_equal(got["h_diabatic"], 0.0)
    np.testing.assert_array_equal(got["rainncv"], 0.0)


def _settling_column():
    host = _column(temp=280.0, rain=0.0)
    host["pressure"][:, 0, 0] = (90000.0, 85000.0)
    host["pii"][:] = ((host["pressure"] / np.float32(c.P0))
                       ** np.float32(c.RCP))
    host["theta"][:] = np.float32(280.0) / host["pii"]
    host["rho"][:] = host["pressure"] / (np.float32(c.RD)
                                            * host["theta"] * host["pii"])
    host["qv"][:, 0, 0] = (1.0e-5, 0.007339452393352985)
    host["qr"][1, 0, 0] = 1.0e-7
    return host


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", (0, 1))
@pytest.mark.parametrize("temperature", (138.0, 156.0))
def test_cold_cloud_is_finite_through_adapter_and_heating(mode, temperature):
    host = _column(temp=temperature, rain=0.0)
    host["qc"].fill(1.2e-13)
    if temperature == 138.0:
        host["qv"][:, 0, 0] = (1.12e-11, 1.344e-11)
    got, density, _ = _launch(host, 50.0, mode, adapter=True)
    for name, values in got.items():
        assert np.isfinite(values).all(), name
    for name in MASS + NUMBER:
        assert (got[name] >= 0.0).all(), name
    weight = density * host["dz"].astype(np.float64)
    initial = float(np.sum(host["qc"].astype(np.float64) * weight))
    final = float(np.sum(sum(got[name].astype(np.float64)
                             for name in MASS[1:]) * weight)
                  + np.sum(got["rainncv"]))
    np.testing.assert_array_equal(got["qv"], host["qv"])
    assert abs(final - initial) <= 128 * np.finfo(np.float32).eps * initial


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", (0, 1))
def test_settled_trace_rain_returns_to_vapor_and_closes_column(mode, monkeypatch):
    host = _settling_column()
    got, density, stages = _launch(host, 1.0, mode, monkeypatch=monkeypatch)
    assert [name for name, _ in stages] == [
        "mo_process", "mo_sediment_64", "mo_finalize"]
    # Rain is absent at entry in the dry lower cell, arrives through
    # sedimentation, then evaporates in the final loop (WRF :3732-3743).
    assert host["qr"][0, 0, 0] == 0.0
    assert got["qr"][0, 0, 0] == 0.0
    assert got["qv"][0, 0, 0] > host["qv"][0, 0, 0]
    np.testing.assert_array_equal(got["qv"][1], host["qv"][1])
    weight = density * host["dz"].astype(np.float64)
    delta = sum(got[name].astype(np.float64) - host[name] for name in MASS)
    residual = float(np.sum(delta * weight) + np.sum(got["rainncv"]))
    # Charge one stored-vapor ULP in the lower cell and 64 rounded
    # operations on transported condensate.
    floor = float(np.spacing(host["qv"][0, 0, 0]) * weight[0, 0, 0])
    condensate = float(np.sum(host["qr"].astype(np.float64) * weight))
    assert abs(residual) <= floor + 64 * np.finfo(np.float32).eps * condensate
    for name in MASS + NUMBER + EFFECTIVE + ("theta",):
        assert np.isfinite(got[name]).all(), name
