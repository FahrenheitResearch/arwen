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
    """Both modes' words come from WRF's own init; the kernel picks one."""
    morrison = (REPO / "gpuwm" / "core" / "kernels" /
                "morrison.cu").read_text(encoding="utf-8")
    constants = (REPO / "gpuwm" / "core" / "kernels" /
                 "morrison_constants.cuh").read_text(encoding="utf-8")
    # The dense-ice family and everything derived from it are two-entry
    # tables indexed by the kernels' runtime ``rimed`` argument.
    for name in ("AG", "BG", "RHOG", "CG", "CONS2", "CONS7", "CONS8",
                 "CONS11", "CONS14", "CONS17", "CONS33", "CONS36",
                 "xam_g"):
        assert f"__device__ __constant__ float mo_{name}[2]" in constants, name
        assert f"(mo_{name}[rimed])" in constants, name
    for kernel in ("mo_process", "mo_sediment_impl", "mo_finalize",
                   "mo_refl10cm"):
        body = morrison[morrison.index(kernel):]
        body = body[:body.index("{")]
        assert "int rimed" in body, kernel
    # No literal hail or graupel parameter in the unit itself (900.0f
    # survives only as refl10cm_hm's own ice-density normaliser).
    for literal in ("19.3f", "0.37f", "114.5f", "400.0f"):
        assert literal not in morrison, literal


def test_pgam_reference_density_is_rebuilt_from_the_current_temperature():
    """``module_mp_morr_two_moment.F:3920`` reads T3D, not the entry density.

    All four PGAM sites spell the reference density ``DUM =
    PRES(K)/(287.15*T3D(K))`` (:1558, :2182, :3405, :3920) from the live
    T3D; RHO(K) is built once at :1325.  The transcription spells each site
    from the temperature it holds there; the WRF oracle grades the values
    (tests/test_morrison_wrf461_parity.py).
    """
    kernel = (REPO / "gpuwm" / "core" / "kernels"
              / "morrison.cu").read_text(encoding="utf-8")
    assert kernel.count("FDIV(pres, t3d * 287.15f)") == 3
    assert kernel.count("FDIV(pressure[AT(k)], t3d * 287.15f)") == 1
    finalize = kernel[kernel.index("void mo_finalize("):]
    finalize = finalize[:finalize.index("void mo_exner(")]
    # In the final loop the density reads the temperature after the
    # tendency apply, the evaporation and the phase changes.
    assert (finalize.index("t3d = mo_add(t3d, mo_mul(t3dten, dt));")
            < finalize.index("FDIV(pres, t3d * 287.15f)"))
