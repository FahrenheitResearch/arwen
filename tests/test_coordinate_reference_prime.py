"""Thermal reference priming covers stores before any tile step runs."""
from types import SimpleNamespace

import numpy as np

from gpuwm.core.streaming import prime_coordinate_reference, prime_lazy_carriers
from gpuwm.io.restart import _scratch_manifest


class _ThermalState:
    def __init__(self, *, reference=None, elapsed=0.0):
        self.thp = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        self.thb = np.array([298.0, 305.0], dtype=np.float32)
        self.elapsed_seconds = elapsed
        self.physics = None
        self._scratch = {} if reference is None else {"diff1_theta_initial": reference}

    def existing_scratch(self, name):
        return self._scratch.get(name)

    def scratch(self, shape, name):
        if name not in self._scratch:
            self._scratch[name] = np.zeros(shape, dtype=np.float32)
        return self._scratch[name]


def test_priming_captures_source_theta_and_makes_a_checkpoint_carrier():
    state = _ThermalState()
    cfg = SimpleNamespace(diff_opt=1, hmix_k_diag=False, nwp_diagnostics=0,
                          moist=False, mp_physics=0, nz=2, ny=3, nx=4)
    assert "scratch/diff1_theta_initial" in prime_lazy_carriers(state, cfg)
    expected = state.thp + state.thb[:, None, None] - np.float32(300.0)
    np.testing.assert_array_equal(state._scratch["diff1_theta_initial"], expected)
    assert _scratch_manifest(state)["scratch/diff1_theta_initial"] is state._scratch[
        "diff1_theta_initial"]
    # Once the run evolves, repeated inventory priming keeps its original
    # spatially varying reference instead of capturing the current field.
    state.thp += np.float32(5.0)
    state.elapsed_seconds = 10.0
    assert prime_lazy_carriers(state, cfg) == ()
    np.testing.assert_array_equal(state._scratch["diff1_theta_initial"], expected)


def test_priming_keeps_a_restored_original_reference():
    reference = np.full((2, 3, 4), 17.0, dtype=np.float32)
    state = _ThermalState(reference=reference, elapsed=30.0)
    assert prime_coordinate_reference(state, SimpleNamespace(diff_opt=1)) == ()
    assert state._scratch["diff1_theta_initial"] is reference
    assert np.all(reference == 17.0)


def test_metric_diffusion_adds_no_thermal_reference_carrier():
    state = _ThermalState()
    assert prime_coordinate_reference(state, SimpleNamespace(diff_opt=2)) == ()
    assert state._scratch == {}
