"""The held stock total decouples with MUTS, not a rebuilt column mass."""
from types import SimpleNamespace as NS
import numpy as np
import pytest


def test_coupled_radiation_cache_uses_the_carried_mass(monkeypatch):
    from gpuwm.core import dycore
    from gpuwm.core.physics import PhysicsDriver
    monkeypatch.setattr(dycore,"WRF_EXACT",True)
    driver=PhysicsDriver.__new__(PhysicsDriver)
    driver.state=NS(c1h=np.ones(1,np.float32),c2h=np.zeros(1,np.float32),
        existing_scratch=lambda slot:np.full((1,1),8,np.float32),
        total_mu=lambda:np.full((1,1),4,np.float32))
    driver._wrf_rthraten=np.full((1,1,1),6,np.float32)
    driver._wrf_rthraten_coupled=True
    np.testing.assert_array_equal(driver.radiation_theta_forcing(),np.float32(.75))
    np.testing.assert_array_equal(driver._wrf_rthraten,np.float32(6))
    driver._wrf_rthraten_coupled=False
    assert driver.radiation_theta_forcing() is driver._wrf_rthraten


def test_restart_requires_the_rounded_total_before_mutating_carriers():
    from gpuwm.io.restart import (
        RestartMismatchError, _validate_strict_radiation_total)
    total = np.full((1, 2, 3), 6, np.float32)
    driver = NS(_wrf_rthraten=total)
    with pytest.raises(RestartMismatchError, match="zero held heating"):
        _validate_strict_radiation_total({}, driver)
    np.testing.assert_array_equal(total, np.float32(6))
    _validate_strict_radiation_total({"diag/_wrf_rthraten": total.copy()}, driver)
    _validate_strict_radiation_total({}, NS(_wrf_rthraten=None))
