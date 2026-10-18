"""Published PBL rates retain the producer and stock end-mass division."""
from types import SimpleNamespace
import numpy as np
import pytest
cp=pytest.importorskip("cupy")
from gpuwm.core.physics import PhysicsDriver
from gpuwm.core.physics_inventory import PBL_HISTORY_RATES


def test_pbl_history_divides_the_coupled_producer_at_completed_mass():
    old=np.array([[90000.03,89001.17]],np.float32)
    end=np.array([[89997.19,89008.41]],np.float32)
    c1=np.array([.71,.23],np.float32);c2=np.array([1000.3,2011.7],np.float32)
    rate=np.array([[[.0031,-.0173]],[[.0103,-.0501]]],np.float32)
    weight=c1[:,None,None]*old[None]+c2[:,None,None]
    coupled=(rate*weight).astype(np.float32)
    driver=object.__new__(PhysicsDriver);driver.radiation_active=False
    driver.fields={n:cp.asarray(coupled) for n in PBL_HISTORY_RATES}
    state=SimpleNamespace(c1h=cp.asarray(c1),c2h=cp.asarray(c2))
    driver.finish_history_mass(state,SimpleNamespace(bl_pbl_physics=1,cu_physics=0),muts=cp.asarray(end))
    expected=coupled/(c1[:,None,None]*end[None]+c2[:,None,None])
    for name in PBL_HISTORY_RATES:
        np.testing.assert_array_equal(cp.asnumpy(driver.fields[name]).view('u4'),expected.view('u4'))
