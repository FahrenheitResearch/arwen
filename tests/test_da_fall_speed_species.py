"""Radial velocity's fall speed: on by default, and per species (audit S7).

The cycle driver projected plain ``w`` (``fall_speed="none"``), and where a
fall speed was used it was the Sun and Crook rain relation for every
species, so a 30 dBZ dry-snow cell fell at about 5 m/s.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from gpuwm.da import obsop


def _cell(value):
    return np.full((1, 1, 1), value, np.float64)


def test_a_rain_only_cell_keeps_sun_and_crook_bit_for_bit():
    dbz, p = _cell(40.0), _cell(85000.0)
    rain_only = obsop.reflectivity_fall_speed(
        dbz, p, _cell(1e-3) > 1e-9, surface_pressure=np.full((1, 1), 95000.0))
    blended = obsop.reflectivity_fall_speed(
        dbz, p, _cell(1e-3) > 1e-9, surface_pressure=np.full((1, 1), 95000.0),
        species={"qr": _cell(1e-3), "qs": _cell(0.0)},
        inverse_density=_cell(1.0 / 1.0))
    assert np.array_equal(rain_only, blended)


def test_dry_snow_no_longer_falls_at_the_rain_speed():
    dbz, p = _cell(30.0), _cell(50000.0)
    ps = np.full((1, 1), 95000.0)
    active = _cell(1e-3) > 1e-9
    rain_relation = float(obsop.reflectivity_fall_speed(
        dbz, p, active, surface_pressure=ps)[0, 0, 0])
    snow = float(obsop.reflectivity_fall_speed(
        dbz, p, active, surface_pressure=ps,
        species={"qs": _cell(1e-3)}, inverse_density=_cell(1.0 / 0.7))[0, 0, 0])
    assert rain_relation > 4.0
    assert 0.5 < snow < 2.5


def test_the_frozen_species_order_and_magnitudes():
    rho = _cell(1.0)
    speeds = {name: float(obsop.frozen_mass_weighted_fall_speed(
        name, _cell(1e-3), rho)[0, 0, 0]) for name in ("qs", "qg", "qh")}
    assert speeds["qs"] < speeds["qg"] < speeds["qh"]
    assert 0.8 < speeds["qs"] < 2.0
    assert 1.5 < speeds["qg"] < 4.0
    assert 6.0 < speeds["qh"] < 15.0
    assert float(obsop.frozen_mass_weighted_fall_speed(
        "qs", _cell(0.0), rho)[0, 0, 0]) == 0.0


def test_a_mixed_cell_is_the_mass_weighted_blend():
    rain_vt = _cell(6.0)
    alt = _cell(1.0)
    q = {"qr": _cell(1e-3), "qs": _cell(3e-3)}
    got = float(obsop.species_blended_fall_speed(rain_vt, q, alt)[0, 0, 0])
    snow = float(obsop.frozen_mass_weighted_fall_speed(
        "qs", q["qs"], _cell(1.0))[0, 0, 0])
    assert got == pytest.approx(0.25 * 6.0 + 0.75 * snow, rel=1e-12)


def test_species_and_density_go_together():
    with pytest.raises(ValueError, match="go together"):
        obsop.reflectivity_fall_speed(
            _cell(30.0), _cell(50000.0), None,
            surface_pressure=np.full((1, 1), 95000.0),
            species={"qs": _cell(1e-3)})


def test_the_analysis_member_fall_speed_uses_the_species():
    from gpuwm.da.radar_assimilation import _member_fall_speed

    shape = (2, 1, 1)
    state = {"p": np.array([95000.0, 50000.0]).reshape(shape),
             "alt": np.full(shape, 1.0 / 0.7),
             "qr": np.zeros(shape), "qs": np.full(shape, 1e-3),
             "qg": np.zeros(shape)}
    vt = _member_fall_speed(state, np.full(shape, 30.0), where="test")
    assert np.all(vt > 0.5) and np.all(vt < 2.5)


def test_the_cycle_driver_defaults_the_fall_speed_on(monkeypatch):
    from tools import da_cycle_prepared as driver
    from gpuwm.da import radar_assimilation

    args = types.SimpleNamespace(fall_speed="auto")
    monkeypatch.setattr(radar_assimilation, "reflectivity_route_available",
                        lambda mp: mp == 8)
    assert driver.resolved_fall_speed(args, 8) == "reflectivity"
    assert driver.resolved_fall_speed(args, 50) == "none"
    args.fall_speed = "none"
    assert driver.resolved_fall_speed(args, 8) == "none"


def test_the_route_question_matches_the_registry():
    from gpuwm.da.radar_assimilation import reflectivity_route_available

    assert reflectivity_route_available(8) is True
    assert reflectivity_route_available(None) is False


def test_the_driver_parser_default_is_auto():
    from tools import da_cycle_prepared as driver

    parser_source = open(driver.__file__, encoding="utf-8").read()
    assert '"--fall-speed", default="auto"' in parser_source
