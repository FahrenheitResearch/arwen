"""The run derives the hybrid coordinate its own ground can order.

WRF refuses a column whose reference dry pressure stops decreasing with
height (v4.6.1 ``dyn_em/nest_init_utils.F:1158-1182``) and names one
remedy: reduce etac.  ``gpuwm.core.grid`` already computed the largest
etac that would order the column and printed it; these tests pin that the
run now APPLIES it -- over every terrain field the run can touch, at the
model top the configuration asked for -- and that the refusal survives
only for terrain no positive etac orders.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.grid import (analytic_base_terrain_height,
                             hybrid_surface_pressure_floor,
                             largest_supported_etac, make_vertical_coord)
from gpuwm.domain_wizard import _ETA_LEVELS
from gpuwm.experiment import VerticalConfig
from gpuwm.ingest.real import _make_real_base_serial
from gpuwm.vertical_adaptation import (ADAPTATION_SCHEMA, TerrainField,
                                       adapt_experiment_vertical,
                                       analytic_base_surface_pressure,
                                       prepared_coordinate_refusal,
                                       survey_vertical_coordinate,
                                       vertical_coordinate_receipt)

P_TOP = 10000.0
BASE_TEMP = 290.0
LADDER = np.asarray(_ETA_LEVELS, dtype=np.float64)

#: The reproduced case: a 12 km parent over a bay whose north edge
#: reaches 6253 m, refused by the shipped 2.7.5 engine at etac = 0.2.
REPRODUCED_PEAK_M = 6252.791200103021


def _field(label, terrain_m, shape=(4, 5), base_temp=BASE_TEMP):
    terrain = np.zeros(shape, dtype=np.float64)
    terrain[-1, -1] = float(terrain_m)
    return TerrainField(label, terrain, base_temp)


def _survey(fields, etac=0.2, hybrid_opt=2, p_top=P_TOP):
    return survey_vertical_coordinate(LADDER, hybrid_opt, etac, p_top, fields)


def test_the_reproduced_column_is_adapted_not_refused():
    adaptation = _survey([_field("d01 static terrain", REPRODUCED_PEAK_M)])
    assert adaptation.adapted
    assert adaptation.etac == 0.184
    assert adaptation.configured_etac == 0.2
    assert adaptation.label == "d01 static terrain"
    assert adaptation.column == (3, 4)
    assert adaptation.terrain_height_m == pytest.approx(REPRODUCED_PEAK_M)


def test_the_derived_coordinate_actually_builds_the_base_state():
    """The point of the derivation: the refused column now initializes."""
    terrain = np.full((3, 4), REPRODUCED_PEAK_M)
    configured = make_vertical_coord(49, hybrid_opt=2, etac=0.2,
                                     eta_levels=LADDER)
    with pytest.raises(ValueError, match="hybrid coordinate"):
        _make_real_base_serial(configured, terrain, P_TOP, BASE_TEMP, 2)
    adaptation = _survey([TerrainField("d01 static terrain", terrain)])
    derived = make_vertical_coord(49, hybrid_opt=2, etac=adaptation.etac,
                                  eta_levels=LADDER)
    base = _make_real_base_serial(derived, terrain, P_TOP, BASE_TEMP, 2)
    assert np.all(np.diff(base.pb, axis=0) < 0.0)
    assert np.all(np.diff(base.phb, axis=0) > 0.0)


def test_the_finest_terrain_governs_not_the_first_one():
    """A nest carries higher peaks than its parent; the nest must win."""
    adaptation = _survey([_field("d01 static terrain", 2000.0),
                          _field("d02 static terrain", 6800.0),
                          _field("d03 static terrain", 3000.0)])
    assert adaptation.label == "d02 static terrain"
    assert adaptation.terrain_height_m == pytest.approx(6800.0)
    assert adaptation.etac < 0.184


def test_a_corridor_a_nest_may_traverse_governs_too():
    """A relocation must never be the first place the coordinate fails."""
    without = _survey([_field("d01 static terrain", 4000.0),
                       _field("d02 static terrain", 500.0)])
    assert not without.adapted
    with_corridor = _survey([_field("d01 static terrain", 4000.0),
                             _field("d02 static terrain", 500.0),
                             _field("d02 statics corridor", 6900.0)])
    assert with_corridor.adapted
    assert with_corridor.label == "d02 statics corridor"


def test_the_model_top_the_configuration_asked_for_is_untouched():
    adaptation = _survey([_field("d01 static terrain", 6800.0)])
    assert adaptation.p_top == P_TOP
    assert "10000 Pa" in adaptation.sentence()


def test_a_representable_run_keeps_its_configured_coordinate():
    adaptation = _survey([_field("d01 static terrain", 3000.0)])
    assert not adaptation.adapted
    assert adaptation.etac == 0.2
    assert adaptation.receipt()["status"] == "AS_CONFIGURED"


def test_terrain_no_positive_etac_orders_stays_the_refusal():
    adaptation = _survey([_field("d01 static terrain", 8848.0)])
    assert adaptation.etac is None
    assert not adaptation.representable
    assert adaptation.receipt()["status"] == "UNREPRESENTABLE"


def test_identity_hybrid_options_have_nothing_to_derive():
    for option in (0, 1):
        assert _survey([_field("d01 static terrain", 8848.0)],
                       hybrid_opt=option) is None


def test_the_survey_refuses_to_answer_from_no_ground_at_all():
    with pytest.raises(ValueError, match="at least one terrain field"):
        _survey([])


def test_an_empty_terrain_field_is_refused_rather_than_counted():
    with pytest.raises(ValueError, match="empty"):
        _survey([TerrainField("d02 static terrain", np.zeros((0, 4)))])


def test_ground_above_the_analytic_profile_loses_every_comparison():
    """NaN pressure is the HIGHEST ground, not the absent one."""
    pressure = analytic_base_surface_pressure(
        np.asarray([[0.0, 30000.0]]), BASE_TEMP)
    assert np.isfinite(pressure[0, 0]) and np.isnan(pressure[0, 1])
    adaptation = _survey([_field("d01 static terrain", 30000.0)])
    assert adaptation.etac is None
    assert adaptation.column == (3, 4)


def test_the_derived_etac_is_the_largest_one_that_works():
    peak = 6800.0
    adaptation = _survey([_field("d01 static terrain", peak)])
    pressure = float(analytic_base_surface_pressure(
        np.asarray([peak]), BASE_TEMP)[0])
    assert adaptation.etac == largest_supported_etac(LADDER, P_TOP, pressure)
    assert pressure > hybrid_surface_pressure_floor(
        LADDER, 2, adaptation.etac, P_TOP)
    # One quantization step higher must NOT order it, or this is not the
    # largest supported value but merely a value that happens to work.
    assert pressure <= hybrid_surface_pressure_floor(
        LADDER, 2, adaptation.etac + 0.001, P_TOP)


def test_the_survey_records_every_field_it_looked_at():
    adaptation = _survey([_field("d01 static terrain", 2000.0, shape=(4, 5)),
                          _field("d02 statics corridor", 6800.0,
                                 shape=(8, 10))])
    surveyed = adaptation.receipt()["surveyed_terrain"]
    assert [entry["field"] for entry in surveyed] == [
        "d01 static terrain", "d02 statics corridor"]
    assert [entry["cells"] for entry in surveyed] == [20, 80]
    assert surveyed[1]["max_terrain_m"] == pytest.approx(6800.0)


def test_a_colder_base_temperature_is_the_stricter_column():
    """ps falls with base_temp, so a domain's own base_temp is used."""
    warm = _survey([TerrainField("d01 static terrain",
                                 np.full((2, 2), 6050.0), 290.0)])
    cold = _survey([TerrainField("d01 static terrain",
                                 np.full((2, 2), 6050.0), 270.0)])
    assert cold.surface_pressure_pa < warm.surface_pressure_pa
    assert not warm.adapted and cold.adapted


def test_the_printed_line_names_the_column_and_the_reason():
    line = _survey([_field("d02 statics corridor", 6800.0)]).sentence()
    assert "d02 statics corridor" in line
    assert "mass point (3, 4)" in line
    assert "6800 m" in line
    assert "etac 0.2 orders only columns above" in line
    assert f"{analytic_base_terrain_height(hybrid_surface_pressure_floor(LADDER, 2, 0.2, P_TOP)):.0f} m" in line


# --- the prepared-artifact side: the model runs the prepared coordinate ---

def test_a_prepared_coordinate_at_or_below_the_configured_one_is_adopted():
    assert prepared_coordinate_refusal(
        label="d01", configured_etac=0.2, prepared_etac=0.184, znw=LADDER,
        p_top=P_TOP, surface_pressure=np.full((2, 2), 45343.0)) is None


def test_a_prepared_coordinate_above_the_configured_one_is_refused():
    message = prepared_coordinate_refusal(
        label="d01", configured_etac=0.2, prepared_etac=0.25, znw=LADDER,
        p_top=P_TOP, surface_pressure=np.full((2, 2), 95000.0))
    assert message is not None
    assert "may only be at or below the configured value" in message


def test_a_prepared_coordinate_that_cannot_order_its_own_column_is_refused():
    message = prepared_coordinate_refusal(
        label="d02", configured_etac=0.2, prepared_etac=0.2, znw=LADDER,
        p_top=P_TOP, surface_pressure=np.full((2, 2), 45343.0))
    assert message is not None
    assert "does not order the prepared column" in message
    assert "describe different atmospheres" in message


# --- the receipt names the route that made the block NOT_APPLICABLE ---
#
# Three routes reach status NOT_APPLICABLE and they are not the same
# fact: the identity hybrid options, a configuration carrying no eta
# ladder, and a cubic coordinate whose surface-pressure floor is already
# at zero.  The receipt is what the run document and a reader of the
# prepared bundle are told, so each route has to say its own reason
# rather than borrow another route's sentence.  The desktop's run view
# is not a reader of it: that view keys on the block's status.


def _configured(hybrid_opt, *, ladder=True, etac=0.2, p_top=P_TOP):
    """A configuration the receipt writer accepts, invariants and all.

    :func:`vertical_coordinate_receipt` reads ``exp.vertical`` and
    nothing else, and says so; the vertical configuration itself is the
    real class with its real invariants, and the object around it is a
    holder.
    """

    return SimpleNamespace(
        vertical=VerticalConfig(
            eta_levels=(tuple(float(value) for value in LADDER)
                        if ladder else ()),
            p_top=float(p_top) if ladder else 0.0,
            hybrid_opt=int(hybrid_opt), etac=float(etac)),
        domains=())


def _why(exp, adaptation):
    return vertical_coordinate_receipt(exp, adaptation)["derivation"]["why"]


def test_an_identity_option_says_b_equals_eta():
    """Route one: the survey itself has nothing to decide."""
    for option in (0, 1):
        adaptation = _survey([_field("d01 static terrain", 8848.0)],
                             hybrid_opt=option)
        assert adaptation is None
        why = _why(_configured(option), adaptation)
        assert f"hybrid_opt {option}" in why
        assert "B(eta) = eta" in why
        assert "eta ladder" not in why


def test_a_configuration_with_no_eta_ladder_says_that_is_why():
    """Route two: preparation returned before it surveyed any ground."""
    for option in (0, 1, 2):
        exp = _configured(option, ladder=False)
        same, adaptation = adapt_experiment_vertical(
            exp, [_field("d01 static terrain", 8848.0)])
        assert same is exp and adaptation is None
        why = _why(exp, adaptation)
        assert "no eta ladder" in why
        assert "B(eta) = eta" not in why


def test_a_cubic_floor_at_zero_pressure_says_that_is_why():
    """Route three: a configured pair that cannot fail to order a column.

    The state the writer is handed is the one
    :func:`survey_vertical_coordinate` returns ``None`` for at
    ``hybrid_opt`` 2: an eta ladder is configured and the floor is
    already at zero pressure.
    """

    exp = _configured(2)
    why = _why(exp, None)
    assert "floor" in why
    assert "no eta ladder" not in why
    assert "B(eta) = eta" not in why
    # The floor is zero exactly where the steepest discrete dB/deta is
    # not above 1, which the ladder and etac set between them; p_top is
    # not in that comparison, so the sentence does not blame it.
    assert "etac" in why
    assert "p_top" not in why


def test_the_three_routes_do_not_share_a_sentence():
    sentences = {_why(_configured(0), None),
                 _why(_configured(0, ladder=False), None),
                 _why(_configured(2), None)}
    assert len(sentences) == 3


def test_the_not_applicable_block_keeps_its_schema_and_its_keys():
    for exp in (_configured(1), _configured(2, ladder=False), _configured(2)):
        block = vertical_coordinate_receipt(exp, None)
        assert set(block) == {"hybrid_opt", "etac", "p_top_pa",
                              "mass_levels", "derivation"}
        derivation = block["derivation"]
        assert set(derivation) == {"schema", "status", "why"}
        assert derivation["schema"] == ADAPTATION_SCHEMA
        assert derivation["status"] == "NOT_APPLICABLE"
        assert isinstance(derivation["why"], str) and derivation["why"]
