"""Held radiation fluxes integrate every step with stock float32 ordering."""
import numpy as np
import pytest
cp=pytest.importorskip("cupy")

from gpuwm.config import RunConfig
from gpuwm.core.physics import PhysicsDriver
from gpuwm.core.physics_inventory import radiation_boundary_flux_names


def test_held_flux_energy_and_night_reset():
    cfg=RunConfig(nx=3,ny=2,nz=4,dx=3000.,dy=3000.,ztop=12000.,dt=15.,
                  run_seconds=60.,ra_lw_physics=4,ra_sw_physics=4,
                  ra_rrtmg_variant="rrtmg_legacy")
    driver=object.__new__(PhysicsDriver)
    names=radiation_boundary_flux_names(cfg)
    flux=np.array([[1.003,500.123,1360.71],[0.,-1.,35.17]],np.float32)
    energy=np.full_like(flux,np.float32(16384.03))
    driver.fields={name:cp.asarray(flux) for name in names}
    driver.fields.update({"ac"+name:cp.asarray(energy) for name in names})
    expected=energy.copy()
    for _ in range(3):
        expected=(expected+(flux*np.float32(cfg.dt)).astype(np.float32)).astype(np.float32)
        driver.accumulate_radiation_boundary_fluxes(cfg)
    for name in names:
        assert np.array_equal(cp.asnumpy(driver.fields["ac"+name]).view(np.uint32),expected.view(np.uint32))
    for name in names:
        if name.startswith("sw"):driver.fields[name].fill(0)
    driver.accumulate_radiation_boundary_fluxes(cfg)
    for name in names:
        want=expected if name.startswith("sw") else expected+flux*np.float32(cfg.dt)
        assert np.array_equal(cp.asnumpy(driver.fields["ac"+name]).view(np.uint32),want.view(np.uint32))
