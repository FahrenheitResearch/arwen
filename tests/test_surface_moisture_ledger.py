"""The ledger is tested against known answers in BOTH directions.

An instrument that only ever reports "clean" on clean input has proven
nothing: it would report clean on broken input too.  Every check below comes
in a pair -- a healthy column that must land inside the FP32 budget, and the
SAME column with one known defect injected that must light up one named
ledger column and leave the others alone.

The four injected defects are the four ways the v1.6.2 nocturnal-dewpoint
story could have been wrong:

* wrong provider  -- the causal story named a writer that is not the writer
* stale carrier   -- a radiative carrier frozen at its seed value
* broken CQS2     -- the moisture exchange coefficient corrupted
* TD conversion   -- the product chain's dewpoint formula wrong

These run on the host: every function under test is FP64 arithmetic over
published FP32 values, so none of them needs a card.
"""
import math

import numpy as np
import pytest

from gpuwm.core import surface_moisture_ledger as sml


# A healthy Gulf-coast marine column, the regime the user report came from.
# QFX is SFCLAY's own flux for these inputs, so both provider identities
# close exactly and the pair is a real cross-check rather than a tautology.
MARINE = {
    "qsfc": 2.1000e-2,
    "qv1": 1.8000e-2,
    "chs": 1.4000e-2,
    "cqs2": 1.7000e-2,
    "psfc": 101000.0,
    "tsk": 299.0,
    "mavail": 1.0,
}


def _sfclay_q2(col):
    return col["qsfc"] + (col["qv1"] - col["qsfc"]) * col["chs"] / col["cqs2"]


def _qfx_from_sfclay(col):
    from gpuwm.core import constants as c
    rho = col["psfc"] / (float(c.RD) * col["tsk"])
    return rho * col["mavail"] * col["chs"] * (col["qsfc"] - col["qv1"])


# ---------------------------------------------------------------- providers


def test_provider_resolution_names_the_last_writer():
    """Noah overwrites SFCLAY; RUC and Noah-MP do not."""
    assert sml.resolve_q2_provider(
        sf_sfclay_physics=1, sf_surface_physics=2) == sml.PROVIDER_NOAH_SFCDIAGS
    assert sml.resolve_q2_provider(
        sf_sfclay_physics=1, sf_surface_physics=3) == sml.PROVIDER_RUC_SFCDIAGS
    assert sml.resolve_q2_provider(
        sf_sfclay_physics=1, sf_surface_physics=4
    ) == sml.PROVIDER_NOAHMP_DIAGNOSTIC
    assert sml.resolve_q2_provider(
        sf_sfclay_physics=1, sf_surface_physics=0) == sml.PROVIDER_SFCLAY
    assert sml.resolve_q2_provider(
        sf_sfclay_physics=0, sf_surface_physics=0) == sml.PROVIDER_NONE


def test_provider_set_matches_the_driver_dispatch():
    """The ledger's SFCDIAGS set cannot drift from the driver's."""
    from gpuwm.core import physics
    assert sml.SFCDIAGS_SCHEMES == physics.LAND_SURFACE_SFCDIAGS_SCHEMES


# ------------------------------------------------- direction 1: clean closes


def test_clean_marine_column_closes_under_both_providers():
    col = dict(MARINE)
    col["qfx"] = _qfx_from_sfclay(col)
    published = _sfclay_q2(col)

    sfclay = sml.expected_q2(sml.PROVIDER_SFCLAY, col)
    noah = sml.expected_q2(sml.PROVIDER_NOAH_SFCDIAGS, col)
    budget = sml.q2_residual_budget(published)

    # Over water MAVAIL == 1, so the two formulas are the same function and
    # both must reproduce the published value inside the FP32 budget.
    assert abs(sfclay - published) <= budget, (sfclay, published, budget)
    assert abs(noah - published) <= budget, (noah, published, budget)


def test_the_two_providers_separate_over_land():
    """MAVAIL < 1 is exactly where the provider column earns its place."""
    col = dict(MARINE, mavail=0.35)
    col["qfx"] = _qfx_from_sfclay(col)
    sfclay = sml.expected_q2(sml.PROVIDER_SFCLAY, col)
    noah = sml.expected_q2(sml.PROVIDER_NOAH_SFCDIAGS, col)
    # Not a rounding difference: the Noah form carries an extra MAVAIL.
    assert abs(sfclay - noah) > 100 * sml.q2_residual_budget(sfclay)


# ------------------------------------ direction 2: each defect lights up one


def test_injected_wrong_provider_lights_up_q2_residual():
    """Attribute a land column to the wrong writer and the residual opens."""
    col = dict(MARINE, mavail=0.35)
    col["qfx"] = _qfx_from_sfclay(col)
    published = _sfclay_q2(col)                    # SFCLAY really wrote it

    honest = published - sml.expected_q2(sml.PROVIDER_SFCLAY, col)
    lied = published - sml.expected_q2(sml.PROVIDER_NOAH_SFCDIAGS, col)

    assert abs(honest) <= sml.q2_residual_budget(published)
    assert abs(lied) > sml.q2_residual_budget(published)


def test_injected_broken_cqs2_lights_up_q2_residual():
    """Corrupt the moisture exchange and the identity stops closing."""
    col = dict(MARINE)
    col["qfx"] = _qfx_from_sfclay(col)
    published = _sfclay_q2(col)

    broken = dict(col, cqs2=col["cqs2"] * 0.5)
    residual = published - sml.expected_q2(sml.PROVIDER_SFCLAY, broken)
    assert abs(residual) > sml.q2_residual_budget(published)


def test_injected_td_conversion_error_lights_up_td2_residual_only():
    """A wrong TD formula moves TD2_RESIDUAL and leaves Q2_RESIDUAL shut."""
    col = dict(MARINE)
    col["qfx"] = _qfx_from_sfclay(col)
    published = _sfclay_q2(col)

    honest_td = sml.dewpoint_k(published, col["psfc"])
    # The classic transcription error: EPS in the numerator instead of the
    # denominator's additive term (e = p*q/EPS rather than p*q/(EPS+q)).
    e_bad = (col["psfc"] / 100.0) * published / 0.622
    lg = math.log(e_bad / 6.112)
    broken_td = 243.5 / (17.67 / lg - 1.0) + 273.15

    def residual_for(product_td):
        ledger = sml.SurfaceMoistureLedger(
            [(0, 0)], td2_product={(0, 0): product_td})
        ledger.capture(fields=_host_fields(col, published),
                       atmosphere=_host_atmosphere(col), model_time=0.0,
                       sf_sfclay_physics=1, sf_surface_physics=0)
        return ledger.rows[0]

    clean = residual_for(honest_td)
    injected = residual_for(broken_td)

    # BOTH directions against the same column.  The clean chain's residual
    # is the instrument's own noise floor -- assert against that measured
    # number rather than a round one, so the separation stated here is the
    # separation the instrument actually delivers.
    assert clean.q2_within_budget and injected.q2_within_budget
    # MEASURED RESOLUTION LIMIT of the TD column: ~3e-7 K, which is the FP32
    # round-trip of Q2 through the surface field and nothing else.  Stated
    # as a number so a future change that coarsens it fails here instead of
    # quietly widening what the instrument can no longer see.
    assert abs(clean.td2_residual) < 1.0e-6, clean.td2_residual
    assert abs(injected.td2_residual) > 0.4, injected.td2_residual
    # A ~0.49 K miss on a 297 K dewpoint is small in absolute terms and
    # six orders of magnitude above the floor.  The collapse under
    # investigation is ~6.5 g/kg, tens of kelvin of dewpoint, so a TD
    # conversion error could not hide inside it unnoticed.
    assert abs(injected.td2_residual) > 1.0e5 * abs(clean.td2_residual)
    # Same 3e-7 K floor: honest_td is computed from the FP64 published
    # value, the ledger's from its FP32 round trip through the field.
    assert abs(clean.td2_from_q2 - honest_td) < 1e-6


def test_injected_stale_carrier_lights_up_carrier_age():
    """A carrier never rewritten reports its seed source and a growing age."""
    ledger = sml.SurfaceMoistureLedger([(0, 0)])
    ledger.note_carrier("glw", source="initial:declared", model_time=0.0,
                        value=300.0)
    ledger.note_carrier("swdown", source="ra_lw_physics=0,ra_sw_physics=1",
                        model_time=3600.0, value=0.0)

    col = dict(MARINE)
    col["qfx"] = _qfx_from_sfclay(col)
    ledger.capture(fields=_host_fields(col, _sfclay_q2(col)),
                   atmosphere=_host_atmosphere(col), model_time=3600.0,
                   sf_sfclay_physics=1, sf_surface_physics=0)
    row = ledger.rows[0]

    # The longwave is an hour stale and says so; the shortwave is current.
    assert row.carriers["glw"]["age_seconds"] == 3600.0
    assert row.carriers["glw"]["source"] == "initial:declared"
    assert row.carriers["swdown"]["age_seconds"] == 0.0


# ------------------------------------------------------- documented branches


def test_the_antarctic_lower_bound_branch_is_reproduced():
    """The ledger knows gpuwm's documented divergence, so it stays quiet."""
    col = {"qsfc": -2.400e-4, "qv1": 7.671e-5, "qfx": -5.548e-7,
           "cqs2": 3.600e-3, "psfc": 80000.0, "tsk": 231.6, "chs": 3.600e-3}
    expected = sml.expected_q2(sml.PROVIDER_NOAH_SFCDIAGS, col)
    # The unbounded inversion is negative here, so gpuwm publishes qv1 and
    # the ledger must expect qv1 -- not the negative number.
    assert expected == col["qv1"]


def test_zero_exchange_is_reported_as_unrecoverable_not_as_zero():
    """isfflx=0 leaves the identity unrecoverable; the ledger says so."""
    col = dict(MARINE, chs=0.0, cqs2=0.0, qfx=0.0)
    assert math.isnan(sml.expected_q2(sml.PROVIDER_SFCLAY, col))


def test_budget_scales_with_magnitude():
    """8 ULP or 2e-6 kg/kg, whichever is larger, at both ends of the range."""
    assert sml.q2_residual_budget(1.0e-5) == sml.Q2_RESIDUAL_BUDGET_KG_KG
    # At 2e-2 kg/kg the ULP term is still under the floor, so the floor is
    # what governs across the whole meteorological range -- recorded here so
    # a future dtype change that breaks that assumption fails loudly.
    assert sml.q2_residual_budget(2.0e-2) == sml.Q2_RESIDUAL_BUDGET_KG_KG


# ------------------------------------------------------------------ helpers


def _host_fields(col, q2):
    def a(v):
        return np.array([[v]], dtype=np.float32)
    return {"tsk": a(col["tsk"]), "psfc": a(col["psfc"]),
            "qsfc": a(col["qsfc"]), "qfx": a(col["qfx"]),
            "cqs2": a(col["cqs2"]), "chs": a(col["chs"]),
            "chs2": a(col["chs"]), "mavail": a(col.get("mavail", 1.0)),
            "ust": a(0.25), "q2": a(q2), "xland": a(2.0),
            "lakemask": a(0.0), "landmask": a(0.0)}


def _host_atmosphere(col):
    return {"temperature": np.array([[[col["tsk"]]]], dtype=np.float32),
            "qv": np.array([[[col["qv1"]]]], dtype=np.float32)}
