"""The preflight says which updraft this ladder and this step start limiting at.

WRF's vertical-velocity limiter (``w_damping = 1``) begins pushing the w
tendency against the motion where the vertical Courant number ``w*dt/dz``
passes 1, which is to say at ``w = dz/dt``. That number depends on nothing
but the ladder and the step, and it is the one line that would have told this
lane what it spent an evening measuring:

* the shipped 49-level ladder at a 15 s step cannot limit below 22 m/s, and
  measured on the card it never fired: no cell reached a Courant number of 1;
* the same step on a ladder refined to 200 m layers cannot limit below
  13 m/s -- an ordinary convective updraft -- and measured on the card it
  fired on 606 cells and halved the storm's peak updraft.

The number is the thinnest layer above the boundary layer divided by the
step. It is a floor on where limiting can begin, not a promise that it will:
what decides is the w the storm actually has at that height.

The run prints it at preflight, before it spends a card on finding out.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "da_cycle_prepared", _ROOT / "tools" / "da_cycle_prepared.py")
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


class _Cfg:
    def __init__(self, eta, dt, w_damping=1, p_top=5000.0, hybrid_opt=2,
                 etac=0.2):
        self.eta_levels = list(eta)
        self.dt = dt
        self.w_damping = w_damping
        self.p_top = p_top
        self.hybrid_opt = hybrid_opt
        self.etac = etac


def _certified():
    from gpuwm.native_wrf_contract import CERTIFIED_ETA_LEVELS
    return np.asarray(CERTIFIED_ETA_LEVELS, dtype=np.float64)


def _refined(dz_max=200.0, nz=110):
    """A ladder whose layers through the convective column are dz_max."""
    spec = importlib.util.spec_from_file_location(
        "ladder", _ROOT / "tools" / "build_stretched_eta_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    depth = mod.analytic_base_terrain_height(5000.0, 290.0)
    thick = mod.stretched_thicknesses(nz, 20.0, dz_max, depth)
    z_full = np.concatenate(([0.0], np.cumsum(thick)))
    return mod.eta_from_heights(z_full, 5000.0, 290.0)


def test_the_shipped_ladder_is_priced_well_above_the_refined_one():
    shipped = driver.limiter_onset(_Cfg(_certified(), 15.0))
    refined = driver.limiter_onset(_Cfg(_refined(), 15.0))
    assert shipped is not None and refined is not None
    assert shipped["w_onset_ms"] > 20.0, (
        f"the shipped ladder cannot limit below about 22 m/s: {shipped}")
    assert shipped["w_onset_ms"] > 1.5 * refined["w_onset_ms"], (
        "the whole point is that the refined ladder starts limiting at a "
        f"much smaller updraft: {shipped} against {refined}")


def test_the_refined_ladder_at_the_same_step_limits_an_ordinary_updraft():
    onset = driver.limiter_onset(_Cfg(_refined(), 15.0))
    assert onset is not None
    assert onset["w_onset_ms"] == pytest.approx(200.0 / 15.0, rel=0.05), (
        f"dz/dt on a 200 m layer at 15 s is 13.3 m/s: {onset}")
    assert onset["thinnest_layer_m"] == pytest.approx(200.0, rel=0.05)
    assert 2000.0 < onset["thinnest_layer_height_m"] < 12000.0, (
        "the layer reported is one an updraft actually meets, not a "
        f"boundary-layer level and not an anvil one: {onset}")


def test_a_shorter_step_raises_the_onset_in_proportion():
    at15 = driver.limiter_onset(_Cfg(_refined(), 15.0))
    at5 = driver.limiter_onset(_Cfg(_refined(), 5.0))
    assert at5["w_onset_ms"] == pytest.approx(3.0 * at15["w_onset_ms"],
                                              rel=0.01)


def test_a_run_with_the_limiter_off_is_not_told_about_it():
    assert driver.limiter_onset(_Cfg(_refined(), 15.0, w_damping=0)) is None


def test_a_thin_layer_in_the_anvil_is_not_mistaken_for_an_updraft_layer():
    """The reference model's 60-level ladder has its thinnest layer above
    2 km at 18.7 km, in the stratosphere, where no updraft meets it."""
    ref60 = np.array([
        1, 0.993814707, 0.985950649, 0.976014256, 0.963557541,
        0.948093116, 0.929123759, 0.90619123, 0.87894237, 0.847207963,
        0.811077714, 0.770949006, 0.727525413, 0.684030771, 0.642961025,
        0.604180932, 0.567562938, 0.532986403, 0.500337601, 0.469508916,
        0.440399021, 0.412912011, 0.386957437, 0.362449884, 0.339308649,
        0.317457527, 0.296824664, 0.277342081, 0.258945674, 0.241574913,
        0.225172549, 0.2096847, 0.195060253, 0.181251153, 0.168211967,
        0.155899644, 0.144273847, 0.133296132, 0.122930467, 0.113142714,
        0.103900604, 0.095173724, 0.0869334266, 0.0791524947, 0.0718053728,
        0.0648678541, 0.0583171472, 0.0521316081, 0.0462909527, 0.0407758839,
        0.0355683193, 0.030651059, 0.026007941, 0.0216237046, 0.0174838807,
        0.0135748768, 0.00988376327, 0.00639845803, 0.00310745789, 0],
        dtype=float)
    onset = driver.limiter_onset(_Cfg(ref60, 15.0))
    assert onset is not None
    assert onset["thinnest_layer_height_m"] < 12000.0, (
        f"a 240 m layer at 18.7 km is not where an updraft is: {onset}")
    assert onset["w_onset_ms"] > 20.0


def test_a_config_that_cannot_be_read_costs_the_run_nothing():
    class _Broken:
        dt = 15.0
        w_damping = 1
    assert driver.limiter_onset(_Broken()) is None
