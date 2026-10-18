"""Cumulus history retains inactive producer rates and completed-mass words."""
from types import SimpleNamespace

import numpy as np


def test_inactive_ice_stays_a_diagnostic_without_creating_transport(monkeypatch):
    from gpuwm.core import physics
    from gpuwm.core.physics_inventory import CUMULUS_HISTORY_RATES

    monkeypatch.setattr(physics, "cp", np)
    shape = (2, 1, 1)
    state = SimpleNamespace(p=np.zeros(shape, np.float32),
                            c1h=np.array([.5, .25], np.float32),
                            c2h=np.array([1., 2.], np.float32),
                            total_mu=lambda: np.array([[4.]], np.float32),
                            qv=np.zeros(shape, np.float32), qc=np.zeros(shape, np.float32),
                            qr=np.zeros(shape, np.float32), qi=None, qs=None)
    driver = physics.PhysicsDriver.__new__(physics.PhysicsDriver)
    driver.fields = {name: np.zeros(shape, np.float32) for name in CUMULUS_HISTORY_RATES}
    rate = np.array([.125, .25], np.float32).reshape(shape)
    ice = np.array([.0071, -.0003], np.float32).reshape(shape)
    result = physics.CumulusResult(rthcuten=rate, rqvcuten=rate, rqccuten=rate,
                                   history_rates={"rqicuten": ice})
    cfg = SimpleNamespace(cu_physics=3, bl_pbl_physics=0)
    driver.retain_cumulus_history(result, state)
    driver.couple_cumulus_history(state, cfg)
    driver.finish_history_mass(state, cfg, muts=np.array([[8.]], np.float32))
    # WRF calculate_phy_tend then phy_prep_part2: .125*3/5 and .25*3/4.
    expected = np.array([0x3d99999a, 0x3e400000], np.uint32).reshape(shape)
    for name in ("rthcuten", "rqvcuten", "rqccuten"):
        np.testing.assert_array_equal(driver.fields[name].view("u4"), expected)
    np.testing.assert_array_equal(driver.fields["rqicuten"].view("u4"), ice.view("u4"))
    np.testing.assert_array_equal(result.rthcuten.view("u4"), rate.view("u4"))
    assert result.rqicuten is None and state.qi is None
    assert "rqicuten" not in getattr(driver, "cu_rates", {})
