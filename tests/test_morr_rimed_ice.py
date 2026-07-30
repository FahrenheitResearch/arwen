"""CPU and structural regressions for Morrison's hail/graupel switch.

WRF v4.6.1 makes ``morr_rimed_ice`` a scalar ``&physics`` option with
Registry default 1 (hail).  ``MORR_TWO_MOMENT_INIT`` then selects AG/BG and
RHOG before deriving CG and every BG-dependent gamma/exponent product:
Registry/Registry.EM_COMMON:2663-2666 and
phys/module_mp_morr_two_moment.F:337-411, :483-510.
"""

from __future__ import annotations

import math
from pathlib import Path
import tomllib

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config


REPO = Path(__file__).resolve().parents[1]


def _config(**updates) -> RunConfig:
    values = dict(nx=3, ny=2, nz=4, dx=1000.0, dy=1000.0,
                  ztop=4000.0, dt=6.0, run_seconds=0.0,
                  moist=True, mp_physics=10)
    values.update(updates)
    return RunConfig(**values)


def test_run_config_defaults_to_wrf_hail_and_validates_switch():
    assert _config().morr_rimed_ice == 1
    assert validate_run_config(_config(morr_rimed_ice=0)).morr_rimed_ice == 0
    for bad in (-1, 2):
        with pytest.raises(ValueError, match="morr_rimed_ice"):
            validate_run_config(_config(morr_rimed_ice=bad))


def test_central_rimed_ice_constants_match_wrf_branches():
    from gpuwm.core.morrison_constants import rimed_ice_constants

    graupel = rimed_ice_constants(0)
    hail = rimed_ice_constants(1)
    assert (graupel.ag, graupel.bg, graupel.rhog) == (19.3, 0.37, 400.0)
    assert (hail.ag, hail.bg, hail.rhog) == (114.5, 0.5, 900.0)
    assert graupel.cg == pytest.approx(400.0 * math.pi / 6.0)
    assert hail.cg == pytest.approx(900.0 * math.pi / 6.0)
    with pytest.raises(ValueError, match="morr_rimed_ice"):
        rimed_ice_constants(7)


def test_documented_morrison_constants_default_hail_and_retain_graupel():
    with (REPO / "gpuwm" / "data" / "morrison" /
          "constants.toml").open("rb") as stream:
        constants = tomllib.load(stream)
    assert constants["switches"]["ihail"] == 1
    assert constants["rimed_ice"]["hail"] == {
        "ag": 114.5, "bg": 0.5, "rhog": 900.0}
    assert constants["rimed_ice"]["graupel"] == {
        "ag": 19.3, "bg": 0.37, "rhog": 400.0}


def test_numpy_graupel_fall_speed_uses_selected_wrf_constants():
    from gpuwm.core.morrison_constants import rimed_ice_constants
    from gpuwm.verify.npref import (_MORR_RHOSU,
                                    _np_morrison_fall_speeds)

    q = {name: np.zeros(1) for name in "crisg"}
    n = {name: np.zeros(1) for name in "crisg"}
    q["g"][0] = 1.0e-3
    n["g"][0] = 1.0e4
    rho = np.ones(1)
    temperature = np.full(1, 270.0)

    for option in (0, 1):
        selected = rimed_ice_constants(option)
        vm, vn, _ = _np_morrison_fall_speeds(
            "g", q, n, rho, temperature, morr_rimed_ice=option)
        lam = (selected.rhog * math.pi * n["g"][0] / q["g"][0]) ** (1 / 3)
        an = selected.ag * (_MORR_RHOSU / rho[0]) ** 0.54
        expected_vm = min(
            an * math.gamma(4.0 + selected.bg) / 6.0
            / lam ** selected.bg,
            20.0 * (_MORR_RHOSU / rho[0]) ** 0.54)
        expected_vn = min(
            an * math.gamma(1.0 + selected.bg) / lam ** selected.bg,
            20.0 * (_MORR_RHOSU / rho[0]) ** 0.54)
        np.testing.assert_allclose(vm, [expected_vm], rtol=1.0e-14)
        np.testing.assert_allclose(vn, [expected_vn], rtol=1.0e-14)


def test_radar_init_defaults_hail_and_keeps_explicit_graupel():
    from gpuwm.core.refl import radar_init

    assert radar_init().xam_g == pytest.approx(900.0 * math.pi / 6.0)
    assert radar_init(0).xam_g == pytest.approx(400.0 * math.pi / 6.0)


def test_cuda_sources_receive_selected_rimed_ice_constants_at_runtime():
    morrison = (REPO / "gpuwm" / "core" / "kernels" /
                "morrison.cu").read_text(encoding="utf-8")
    reflectivity = (REPO / "gpuwm" / "core" / "kernels" /
                    "refl.cu").read_text(encoding="utf-8")
    assert "#define MRHOG 400.0f" not in morrison
    assert "19.3f" not in morrison
    assert "0.37f" not in morrison
    assert "morr_ag" in morrison
    assert "morr_bg" in morrison
    assert "morr_rhog" in morrison
    assert "#define RXAM_G (400.0" not in reflectivity
    assert "const double xam_g" in reflectivity
