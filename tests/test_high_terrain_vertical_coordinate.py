"""Very high terrain: WRF's hybrid-coordinate limit, named and remedied.

WRF v4.6.1 refuses a domain whose reference dry-pressure column stops
decreasing with height -- ``dyn_em/nest_init_utils.F:1158-1182``, whose own
message says the cause "tends to be caused by very high topography" and
whose only remedy is "reduce etac".  ``doc/README.hybrid_vert_coord:65-74``
names the Himalaya specifically: "Over the Himalayan Plateau with a 10 hPa
model lid, a value of etac = 0.25 causes model failures."

The limit is on ABSOLUTE elevation, through the surface pressure, and not on
steepness or resolution: the reported Peruvian-Andes control (5861 m peaks,
huge relief) is representable and the smooth Tibetan plateau is not.
"""
import numpy as np
import pytest

from gpuwm.core.grid import (
    analytic_base_terrain_height,
    hybrid_column_ordering_refusal,
    hybrid_surface_pressure_floor,
    largest_supported_etac,
    make_vertical_coord,
)
from gpuwm.domain_wizard import _ETA_LEVELS
from gpuwm.ingest.real import _make_real_base_serial

P_TOP = 10000.0
BASE_TEMP = 290.0
LADDER = np.asarray(_ETA_LEVELS, dtype=np.float64)


def _coord(etac=0.2, hybrid_opt=2):
    return make_vertical_coord(49, hybrid_opt=hybrid_opt, etac=etac,
                               eta_levels=LADDER)


def _build(terrain_m, etac=0.2, p_top=P_TOP):
    return _make_real_base_serial(
        _coord(etac), np.full((3, 4), float(terrain_m)), p_top, BASE_TEMP, 2)


# The tester's controls, by their reported maximum terrain height.
@pytest.mark.parametrize("terrain_m", [668.0, 1416.0, 3653.0, 3655.0,
                                       4127.0, 5861.0, 6080.0])
def test_controls_up_to_the_limit_still_initialize(terrain_m):
    base = _build(terrain_m)
    assert np.all(np.diff(base.pb, axis=0) < 0.0)
    # The base geopotential must also increase: an inverted full-level pair
    # makes real.py's hypsometric_opt=2 log produce a negative thickness.
    assert np.all(np.diff(base.phb, axis=0) > 0.0)


@pytest.mark.parametrize("terrain_m", [6087.0, 6200.0, 7000.0, 8848.0])
def test_terrain_above_the_limit_is_refused_with_constraint_and_remedy(
        terrain_m):
    with pytest.raises(ValueError) as caught:
        _build(terrain_m)
    message = str(caught.value)
    assert "hybrid coordinate" in message
    assert "surface pressure stays above" in message
    assert "m of terrain" in message
    assert "etac" in message
    assert "nest_init_utils.F" in message


@pytest.mark.parametrize("terrain_m", [6087.0, 6200.0, 7000.0])
def test_the_printed_etac_remedy_actually_initializes_the_column(terrain_m):
    with pytest.raises(ValueError) as caught:
        _build(terrain_m)
    message = str(caught.value)
    marker = "reduce etac from 0.2 to at most "
    assert marker in message
    remedy = float(message.split(marker, 1)[1].split(",")[0].split(" ")[0])
    base = _build(terrain_m, etac=remedy)
    assert np.all(np.diff(base.pb, axis=0) < 0.0)
    assert np.all(np.diff(base.phb, axis=0) > 0.0)


def test_negative_control_the_unbounded_column_really_does_invert():
    """Without the guard the base state carries a negative-thickness layer."""
    coord = _coord()
    from gpuwm.core.grid import compute_hybrid_coeffs
    import gpuwm.core.constants as c
    hybrid = compute_hybrid_coeffs(LADDER, 2, 0.2, c.P0, P_TOP)
    surface = c.P0 * np.exp(
        -BASE_TEMP / 50.0
        + np.sqrt((BASE_TEMP / 50.0) ** 2
                  - 2.0 * c.G * 6087.0 / (50.0 * c.RD)))
    column = hybrid["c3f"] * (surface - P_TOP) + hybrid["c4f"] + P_TOP
    assert np.any(np.diff(column) >= 0.0), (
        "6087 m must invert the full-level column, else the guard is moot")
    assert coord.hybrid_opt == 2


def test_the_limit_is_elevation_not_steepness():
    """Smooth plateau above the floor fails; jagged relief below it passes."""
    jagged = np.tile(np.asarray([[10.0, 5800.0, 60.0, 5861.0]]), (3, 1))
    smooth = np.full((3, 4), 6300.0)
    assert _make_real_base_serial(
        _coord(), jagged, P_TOP, BASE_TEMP, 2) is not None
    with pytest.raises(ValueError, match="hybrid coordinate"):
        _make_real_base_serial(_coord(), smooth, P_TOP, BASE_TEMP, 2)


def test_identity_hybrid_options_have_no_terrain_ceiling():
    """B = eta exactly, so c4 = 0 and the column cannot invert."""
    for option in (0, 1):
        assert hybrid_surface_pressure_floor(LADDER, option, 0.2, P_TOP) == 0.0


def test_floor_matches_the_closed_form_and_the_reported_boundary():
    floor = hybrid_surface_pressure_floor(LADDER, 2, 0.2, P_TOP)
    height = analytic_base_terrain_height(floor)
    # The reported boundary sits between a passing 6087 m domain and the
    # failing Himalayan/Tibetan ones.
    assert 6000.0 < height < 6120.0
    assert 46000.0 < floor < 46500.0


def test_lower_etac_and_lower_top_both_raise_the_ceiling():
    base = analytic_base_terrain_height(
        hybrid_surface_pressure_floor(LADDER, 2, 0.2, P_TOP))
    lower_etac = analytic_base_terrain_height(
        hybrid_surface_pressure_floor(LADDER, 2, 0.1, P_TOP))
    lower_top = analytic_base_terrain_height(
        hybrid_surface_pressure_floor(LADDER, 2, 0.2, 5000.0))
    assert lower_etac > base + 500.0
    assert lower_top > base + 300.0


def test_no_etac_is_offered_when_none_can_work():
    everest = 31393.0  # Pa, the analytic base state at 8848 m
    assert largest_supported_etac(LADDER, P_TOP, everest) is None
    assert largest_supported_etac(LADDER, P_TOP, 60000.0) is not None


def test_representable_column_produces_no_refusal():
    coord = _coord()
    assert hybrid_column_ordering_refusal(
        coord, P_TOP, np.full((2, 2), 95000.0)) is None
