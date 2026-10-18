"""History must use the retained strict radiation total at completed mass."""
from types import SimpleNamespace

import numpy as np


def test_strict_history_reads_retained_total_without_mutating_it(monkeypatch):
    from gpuwm.core import dycore
    from gpuwm.core.physics import PhysicsDriver
    from gpuwm.io.history_layout import _OUTPUT_FIELDS, physics_history_fields

    monkeypatch.setattr(dycore, "WRF_EXACT", True)
    driver = PhysicsDriver.__new__(PhysicsDriver)
    driver.state = SimpleNamespace(
        c1h=np.array([.75], np.float32), c2h=np.array([1.], np.float32),
        existing_scratch=lambda slot: np.array([[8.]], np.float32),
        total_mu=lambda: np.array([[4.]], np.float32))
    driver.fields = {key: np.zeros((1, 1), np.float32) for key in _OUTPUT_FIELDS.values()}
    driver.rthraten = np.array([[[.5]]], np.float32)
    driver.rthratenlw = np.array([[[.25]]], np.float32)
    driver.rthratensw = np.array([[[.25]]], np.float32)
    driver._wrf_rthraten = np.array([[[6.000003]]], np.float32)
    driver._wrf_rthraten_coupled = True
    driver.surface_energy_diag = driver.sase_active = driver.radiation_active = False
    driver.hmix_k_diag = driver.rainc = None
    driver.mp_physics = 0
    driver._zero_accumulator = lambda: np.zeros((1, 1), np.float32)
    held = driver._wrf_rthraten.copy()
    expected = held / np.float32(7.)
    frame = physics_history_fields(driver)
    np.testing.assert_array_equal(frame["RTHRATEN"].view("u4"), expected.view("u4"))
    assert not np.array_equal(frame["RTHRATEN"].view("u4"), driver.rthraten.view("u4"))
    np.testing.assert_array_equal(driver._wrf_rthraten.view("u4"), held.view("u4"))
    assert driver._wrf_rthraten_coupled
