"""The run derives the hybrid coordinate its own ground can order.

WRF refuses a column whose reference dry pressure stops decreasing with
height (v4.6.1 ``dyn_em/nest_init_utils.F:1158-1182``) and names one
remedy: reduce etac.  ``gpuwm.core.grid`` already computed the largest
etac that would order the column and printed it; these tests pin that the
run now APPLIES it -- over every terrain field the run can touch, at the
model top the configuration asked for -- and that the refusal survives
only for terrain no positive etac orders.

Order is not the whole requirement.  A generated 1 km forecast under the
highest central Andes was ordered at etac 0.2 with one layer left at 1.2%
of its flat-ground depth, and it stopped at model second 350 with that
layer running away.  The derived etac is the largest one that keeps every
layer of every column at least MIN_LAYER_FRACTION as deep as over flat
ground.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.grid import (SMALLEST_SEARCHED_ETAC,
                             analytic_base_terrain_height,
                             finalize_vertical_coord,
                             hybrid_layer_depth_fractions,
                             hybrid_surface_pressure_floor,
                             largest_supported_etac, make_vertical_coord)
import gpuwm.core.constants as c
from gpuwm.domain_wizard import _ETA_LEVELS
from gpuwm.experiment import VerticalConfig
from gpuwm.ingest.real import _make_real_base_serial
from gpuwm.vertical_adaptation import (ADAPTATION_SCHEMA, MIN_LAYER_FRACTION,
                                       TerrainField,
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
    assert adaptation.etac == 0.048
    assert adaptation.margin_met
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
    assert adaptation.etac < _survey(
        [_field("d01 static terrain", REPRODUCED_PEAK_M)]).etac


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
    peak = 6456.0
    adaptation = _survey([_field("d01 static terrain", peak)])
    pressure = float(analytic_base_surface_pressure(
        np.asarray([peak]), BASE_TEMP)[0])
    assert adaptation.etac == largest_supported_etac(
        LADDER, P_TOP, pressure, min_layer_fraction=MIN_LAYER_FRACTION)
    assert pressure > hybrid_surface_pressure_floor(
        LADDER, 2, adaptation.etac, P_TOP,
        min_layer_fraction=MIN_LAYER_FRACTION)
    # One quantization step higher must NOT keep the margin, or this is
    # not the largest supported value but merely a value that happens to
    # work.
    assert pressure <= hybrid_surface_pressure_floor(
        LADDER, 2, adaptation.etac + 0.001, P_TOP,
        min_layer_fraction=MIN_LAYER_FRACTION)


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
                                 np.full((2, 2), 4800.0), 290.0)])
    cold = _survey([TerrainField("d01 static terrain",
                                 np.full((2, 2), 4800.0), 270.0)])
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


# --- the thinnest layer keeps a measured share of its flat-ground depth ---

#: The model top every generated configuration carries.
GENERATED_P_TOP = 5000.0

#: The highest 1 km cell under the generated central-Andes domain
#: (32.66 S, 70.0 W): ordered by etac 0.2 with layer 20 at 1.2% of its
#: flat depth, and the run stopped at model second 350.
BARELY_ORDERED_PEAK_M = 6456.0


def _generated_survey(peak_m, etac=0.2):
    return _survey([_field("d01 static terrain", peak_m)], etac=etac,
                   p_top=GENERATED_P_TOP)


def test_a_column_the_coordinate_only_just_orders_is_given_a_thicker_layer():
    adaptation = _generated_survey(BARELY_ORDERED_PEAK_M)
    assert adaptation.configured_ordered
    assert adaptation.thinnest_layer == 20
    assert adaptation.configured_layer_fraction == pytest.approx(0.0118,
                                                                 abs=1e-4)
    assert adaptation.adapted
    assert adaptation.etac == 0.076
    assert adaptation.margin_met
    assert adaptation.layer_fraction >= MIN_LAYER_FRACTION
    line = adaptation.sentence()
    assert "etac 0.2 leaves layer 20 over d01 static terrain" in line
    assert "only 1.2% as deep as over flat ground" in line
    assert "so this run uses etac 0.076" in line
    assert f"at least {MIN_LAYER_FRACTION:.0%} as deep" in line


def test_ground_well_below_the_floor_keeps_the_configured_coordinate():
    """The Rockies at 1 km reach 4400 m: every layer keeps a quarter of its
    flat depth at etac 0.2, so the coordinate is left exactly alone."""
    adaptation = _generated_survey(4400.0)
    assert not adaptation.adapted
    assert adaptation.etac == 0.2
    assert adaptation.layer_fraction > 0.25
    assert adaptation.receipt()["status"] == "AS_CONFIGURED"
    assert adaptation.receipt()["thinnest_layer"]["margin_met"]


def test_where_no_etac_reaches_the_margin_the_thickest_layer_is_taken():
    """Everest's neighbourhood at 1 km: the cubic cannot keep that column's
    thinnest layer at the margin for any etac, so the run takes the etac
    that leaves it deepest rather than the one that barely orders it."""
    peak = 8617.0
    pressure = float(analytic_base_surface_pressure(
        np.asarray([peak]), BASE_TEMP)[0])
    assert largest_supported_etac(
        LADDER, GENERATED_P_TOP, pressure,
        min_layer_fraction=MIN_LAYER_FRACTION) is None
    barely = largest_supported_etac(LADDER, GENERATED_P_TOP, pressure)
    adaptation = _generated_survey(peak)
    assert adaptation.etac == SMALLEST_SEARCHED_ETAC < barely
    assert not adaptation.margin_met
    deepest = adaptation.layer_fraction
    assert deepest > hybrid_layer_depth_fractions(
        LADDER, 2, barely, GENERATED_P_TOP, pressure).min() + 0.04
    line = adaptation.sentence()
    assert "the smallest etac searched" in line
    assert f"{deepest:.1%} as deep as over flat ground" in line
    assert f"no etac reaches {MIN_LAYER_FRACTION:.0%}" in line


def test_a_configured_etac_already_below_the_search_stands():
    adaptation = _generated_survey(8617.0, etac=0.0005)
    assert not adaptation.adapted
    assert adaptation.etac == 0.0005


def test_the_layer_fractions_are_the_models_own_layer_mass():
    """Layer k's dry mass is (c1h*mu + c2h)*dnw; its share over flat
    ground is the model's own coefficient arrays, not a separate model."""
    pressure = float(analytic_base_surface_pressure(
        np.asarray([BARELY_ORDERED_PEAK_M]), BASE_TEMP)[0])
    for etac in (0.2, 0.076):
        coord = make_vertical_coord(49, hybrid_opt=2, etac=etac,
                                    eta_levels=LADDER)
        finalize_vertical_coord(coord, GENERATED_P_TOP)
        mu = pressure - GENERATED_P_TOP
        flat = c.P0 - GENERATED_P_TOP
        layer_mass = ((coord.c1h * mu + coord.c2h)
                      / (coord.c1h * flat + coord.c2h))
        fractions = hybrid_layer_depth_fractions(
            LADDER, 2, etac, GENERATED_P_TOP, pressure)
        assert np.all(fractions <= layer_mass + 1e-12)
        assert fractions.min() == pytest.approx(layer_mass.min(), abs=4e-3)
        assert int(np.argmin(fractions)) == int(np.argmin(layer_mass))


def test_the_margin_floor_is_where_the_thinnest_layer_meets_the_margin():
    for etac in (0.2, 0.1):
        floor = hybrid_surface_pressure_floor(
            LADDER, 2, etac, GENERATED_P_TOP,
            min_layer_fraction=MIN_LAYER_FRACTION)
        assert hybrid_layer_depth_fractions(
            LADDER, 2, etac, GENERATED_P_TOP, floor).min() == pytest.approx(
                MIN_LAYER_FRACTION, abs=1e-12)
        ordering = hybrid_surface_pressure_floor(
            LADDER, 2, etac, GENERATED_P_TOP)
        assert hybrid_layer_depth_fractions(
            LADDER, 2, etac, GENERATED_P_TOP, ordering).min() == pytest.approx(
                0.0, abs=1e-12)
    with pytest.raises(ValueError, match="min_layer_fraction"):
        hybrid_surface_pressure_floor(LADDER, 2, 0.2, GENERATED_P_TOP,
                                      min_layer_fraction=1.0)


def test_the_receipt_records_the_thinnest_layer():
    receipt = _generated_survey(BARELY_ORDERED_PEAK_M).receipt()
    assert receipt["schema"] == ADAPTATION_SCHEMA
    layer = receipt["thinnest_layer"]
    assert layer["min_layer_fraction"] == MIN_LAYER_FRACTION
    assert layer["index"] == 20
    assert layer["configured_fraction"] == pytest.approx(0.0118, abs=1e-4)
    assert layer["fraction"] >= MIN_LAYER_FRACTION
    assert layer["margin_met"] is True


# --- the source's own column is colder than the base state ---

#: A 60-level eta ladder with a fine boundary layer, as the 500 m tiles
#: of a world preparation pass carried it.
SIXTY_LEVEL_LADDER = np.asarray([
    1.0, 0.993814707, 0.985950649, 0.976014256, 0.963557541, 0.948093116,
    0.929123759, 0.90619123, 0.87894237, 0.847207963, 0.811077714,
    0.770949006, 0.727525413, 0.684030771, 0.642961025, 0.604180932,
    0.567562938, 0.532986403, 0.500337601, 0.469508916, 0.440399021,
    0.412912011, 0.386957437, 0.362449884, 0.339308649, 0.317457527,
    0.296824664, 0.277342081, 0.258945674, 0.241574913, 0.225172549,
    0.2096847, 0.195060253, 0.181251153, 0.168211967, 0.155899644,
    0.144273847, 0.133296132, 0.122930467, 0.113142714, 0.103900604,
    0.095173724, 0.0869334266, 0.0791524947, 0.0718053728, 0.0648678541,
    0.0583171472, 0.0521316081, 0.0462909527, 0.0407758839, 0.0355683193,
    0.030651059, 0.026007941, 0.0216237046, 0.0174838807, 0.0135748768,
    0.00988376327, 0.00639845803, 0.00310745789, 0.0])

#: Two 1 km columns a world preparation pass refused at the target dry
#: pressure ("target dry pressure is not monotonic ... reduce etac") with
#: the order-only derivation, at the 10000 Pa top those tiles carried:
#: the highest static terrain of each tile, and the lowest dry surface
#: pressure the global source put on that tile (its surface pressure on
#: the model terrain less the vapour column), both as the refusals
#: printed them.  Each sat 250 to 2600 Pa under the 290 K base state's
#: pressure for the tile's highest ground, because the air over it was
#: colder than the base state.
MEASURED_COLD_COLUMNS = (
    (7130.0, 39894.0),
    (5774.0, 45825.0),
)


@pytest.mark.parametrize("peak_m, dry_surface_pa", MEASURED_COLD_COLUMNS)
def test_the_derived_coordinate_orders_a_column_colder_than_its_base_state(
        peak_m, dry_surface_pa):
    """The target dry pressure is built from the SOURCE column, the survey
    from the base state.  The order-only derivation left no room between
    them and refused both columns; the layer margin leaves thousands of
    pascals, and the column initializes.
    """
    ladder, p_top = SIXTY_LEVEL_LADDER, 10000.0
    base = float(analytic_base_surface_pressure(
        np.asarray([peak_m]), BASE_TEMP)[0])
    assert dry_surface_pa < base - 250.0

    order_only = min(0.2, largest_supported_etac(ladder, p_top, base))
    assert hybrid_surface_pressure_floor(
        ladder, 2, order_only, p_top) > dry_surface_pa

    adaptation = survey_vertical_coordinate(
        ladder, 2, 0.2, p_top,
        [TerrainField("d01 static terrain", np.full((2, 2), peak_m))])
    assert hybrid_surface_pressure_floor(
        ladder, 2, adaptation.etac, p_top) < dry_surface_pa - 5000.0
    # The same test real.py applies to the target dry pressure.
    coord = make_vertical_coord(ladder.size - 1, hybrid_opt=2,
                                etac=adaptation.etac, eta_levels=ladder)
    finalize_vertical_coord(coord, p_top)
    mass = dry_surface_pa - p_top
    half = coord.c3h * mass + coord.c4h + p_top
    full = coord.c3f * mass + coord.c4f + p_top
    assert np.all(np.diff(half) < 0.0) and np.all(np.diff(full) < 0.0)
