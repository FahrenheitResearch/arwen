"""CPU admission coverage for the imported WRF mixing-package helpers."""

from dataclasses import replace

import pytest

from gpuwm.config import (RunConfig, constant_k_mixing_active,
                          wrf_mixing_package_active)


def _config(**changes):
    return replace(RunConfig(nx=8, ny=8, nz=8, dx=100.0, dy=100.0,
                             ztop=1600.0, dt=1.0, run_seconds=0.0), **changes)


@pytest.mark.parametrize("changes,active", [
    ({}, False),
    ({"khdif": 1.0}, True),
    ({"kvdif": 1.0, "bl_pbl_physics": 1}, True),
    ({"isfflx": 0, "tke_heat_flux": 0.2}, True),
    ({"isfflx": 0, "tke_drag_coefficient": 1e-3}, True),
    ({"isfflx": 0}, False),
    ({"isfflx": 1, "sf_sfclay_physics": 1}, True),
    ({"isfflx": 1, "sf_sfclay_physics": 1, "bl_pbl_physics": 1}, False),
    ({"isfflx": 2, "tke_heat_flux": 0.2}, True),
    ({"isfflx": 2, "tke_heat_flux": 0.0}, False),
    ({"diff_opt": 1, "isfflx": 0, "tke_heat_flux": 0.2}, False),
    ({"km_opt": 4, "khdif": 1.0}, False),
])
def test_constant_k_package_keeps_default_zero_and_surface_forcing_admission(changes, active):
    assert constant_k_mixing_active(_config(**changes)) is active


@pytest.mark.parametrize("km_opt,active", [(0, False), (1, False),
                                             (2, True), (3, True), (4, True)])
def test_wrf_package_admits_existing_closures_and_skips_zero_default(km_opt, active):
    assert wrf_mixing_package_active(_config(km_opt=km_opt)) is active


def test_wrf_package_admits_nonzero_constant_k():
    assert wrf_mixing_package_active(_config(khdif=1.0))
