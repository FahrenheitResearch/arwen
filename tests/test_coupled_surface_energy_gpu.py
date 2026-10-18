"""Surface energy uses atmospheric DT once per due land-surface call."""
from types import SimpleNamespace
import numpy as np
import pytest
cp=pytest.importorskip("cupy")
from gpuwm.core.physics import PhysicsDriver
from gpuwm.core.physics_inventory import SURFACE_ENERGY_ACCUMULATORS


def test_surface_energy_keeps_the_two_float32_rounds_and_atmospheric_dt():
    driver=object.__new__(PhysicsDriver)
    cfg=SimpleNamespace(dt=15.,bldt=1.,sf_surface_physics=1)
    flux=np.array([[1.003,500.123],[0.,-35.17]],np.float32)
    energy=np.full_like(flux,np.float32(16384.03))
    driver.fields={n:cp.asarray(flux) for n in SURFACE_ENERGY_ACCUMULATORS.values()}
    driver.fields.update({n:cp.asarray(energy) for n in SURFACE_ENERGY_ACCUMULATORS})
    expected=energy.copy()
    for _ in range(3):
        driver.accumulate_surface_energy(cfg)
        expected=(expected+flux*np.float32(cfg.dt)).astype(np.float32)
    for name in SURFACE_ENERGY_ACCUMULATORS:
        np.testing.assert_array_equal(cp.asnumpy(driver.fields[name]).view('u4'),expected.view('u4'))
    # A zero-flux due call keeps the previous accumulated energy.
    for name in SURFACE_ENERGY_ACCUMULATORS.values():driver.fields[name].fill(0.)
    driver.accumulate_surface_energy(cfg)
    for name in SURFACE_ENERGY_ACCUMULATORS:
        np.testing.assert_array_equal(cp.asnumpy(driver.fields[name]).view('u4'),expected.view('u4'))
