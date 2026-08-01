"""THE GATE: the LETKF twin experiment must actually improve the state.

Every other test on this filter checks that a piece computes what it claims.
This one checks that the whole thing is worth running, by comparing against a
truth the experiment knows exactly.  It is the only test here that would fail
if the innovation had the wrong sign, and the only one that distinguishes
"the filter runs" from "the filter works".

The numbers below are thresholds, not the measurements.  The measurements are
printed by ``python -m gpuwm.da.osse`` and recorded in the lane handoff; the
thresholds are set well inside them so this gate reports a regression rather
than sampling noise.  Where a margin is thin -- the unobserved wind
components at R = 20 -- the gate says so in the assertion rather than
quietly tightening the seed until it looks comfortable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpuwm.da.osse import (
    FIELDS,
    OsseSetup,
    build_twin,
    run_cycling_osse,
    run_osse,
    smooth_random_field,
)

#: Seeds the gate runs.  Three, not one: a single-seed gate on a stochastic
#: experiment tests a random draw, and passes or fails for reasons that have
#: nothing to do with the code.
_SEEDS = (20260730, 20261730, 20262730)

#: R = 30.  Within the R <= 40 budget this filter is designed for, and large
#: enough that cross-variable covariance is signal rather than sampling
#: noise.  At R = 20 the unobserved u component improves by as little as
#: 0.08% on some seeds -- still positive, but too thin to gate on.
_MEMBERS = 30


@pytest.fixture(scope="module")
def osse_runs():
    """One analysis per seed, computed once and shared by the gates."""
    return [run_osse(OsseSetup(members=_MEMBERS, seed=s)) for s in _SEEDS]


def test_osse_reduces_error_in_the_observed_field(osse_runs):
    """The headline: assimilating theta makes the theta analysis better.

    Around 21-31% RMSE reduction from ~170 point observations over 8192
    gridpoints, across the gated seeds.  The threshold is 15%.
    """
    for r in osse_runs:
        assert r.n_obs > 0
        imp = r.improvement("theta")
        assert imp > 0.15, (
            f"observed-field RMSE reduction {100 * imp:.2f}% is below the"
            f" 15% gate\n{r.report()}")


def test_osse_reduces_error_in_the_unobserved_fields(osse_runs):
    """The multivariate update, which is what makes this an EnKF.

    Only theta is observed.  qv, u, v and w improve solely through
    ensemble cross-covariance -- there is no balance operator, no
    regression, nothing but the sample covariance between fields.  An
    optimal-interpolation scheme, or an EnKF with the cross-covariance
    blocks broken, would leave these unchanged at best.

    Gated per field on strict improvement, because that held on every seed
    at R = 30; the thin one is u, at +0.89% worst case over a five-seed
    exploration.  The aggregate is gated harder.
    """
    unobserved = [f for f in FIELDS if f != "theta"]
    for r in osse_runs:
        for f in unobserved:
            assert r.improvement(f) > 0.0, (
                f"unobserved field {f} got WORSE: the cross-covariance"
                f" update is not working\n{r.report()}")
        mean_imp = float(np.mean([r.improvement(f) for f in unobserved]))
        assert mean_imp > 0.02, (
            f"mean unobserved improvement {100 * mean_imp:.2f}% below the"
            f" 2% gate\n{r.report()}")


def test_osse_reduces_spread_without_collapsing_it(osse_runs):
    """Spread must fall -- and must not fall to zero.

    An analysis that keeps the prior spread has not used the observations.
    One that collapses it has stopped being an ensemble and will reject
    every observation in the next cycle.  Both failure modes look identical
    in a single-cycle RMSE score, which is why spread is gated separately.
    """
    for r in osse_runs:
        for f in FIELDS:
            sb, sa = r.prior_spread[f], r.analysis_spread[f]
            assert sa < sb, (f, sb, sa, "spread did not contract")
            assert sa > 0.3 * sb, (
                f, sb, sa, "spread collapsed: filter divergence risk")


def test_osse_analysis_is_calibrated(osse_runs):
    """Posterior spread should be the same size as posterior error.

    A well-behaved filter's ensemble spread estimates its own error.  This
    is the diagnostic that catches a localisation radius or an observation
    error that is wrong by a large factor, both of which can still reduce
    RMSE while producing an ensemble that lies about its confidence.  The
    band is deliberately wide -- a factor of two either way -- because
    calibration is not what this lane is tuning.
    """
    for r in osse_runs:
        ratio = r.analysis_spread["theta"] / r.analysis_rmse["theta"]
        assert 0.5 < ratio < 2.0, (
            f"posterior spread/error ratio {ratio:.3f} is far from 1\n"
            f"{r.report()}")


def test_no_da_control_is_bitwise_unchanged():
    """The control arm: no observations, no change at all.

    Not "a small change".  If the no-DA control drifted, every RMSE
    comparison above would be measuring the drift as well as the analysis.
    """
    setup = OsseSetup(members=_MEMBERS, obs_fraction=0.0)
    r = run_osse(setup)
    assert r.n_obs == 0
    for f in FIELDS:
        assert r.analysis_rmse[f] == r.prior_rmse[f]
        assert r.analysis_spread[f] == r.prior_spread[f]
    assert r.diagnostics.active_points == 0


def test_osse_scores_are_reproducible():
    """Same seed, same numbers.  A gate on a moving target is not a gate."""
    a = run_osse(OsseSetup(members=12, seed=99))
    b = run_osse(OsseSetup(members=12, seed=99))
    for f in FIELDS:
        assert a.analysis_rmse[f] == b.analysis_rmse[f]


def test_the_twin_has_cross_variable_covariance():
    """Guard the gate: the multivariate test must not pass vacuously.

    If the synthetic ensemble had no covariance between fields, the
    unobserved-field test above would be measuring sampling noise and would
    pass or fail at random.  It nearly did: the first version of the twin
    drew each field's perturbations independently, and assimilating theta
    made every other field WORSE.  This asserts the property that test
    depends on.
    """
    setup = OsseSetup(members=40, seed=5)
    _, prior, _ = build_twin(setup)
    pert = {f: prior[f] - prior[f].mean(axis=0, keepdims=True)
            for f in FIELDS}
    a = pert["theta"].reshape(setup.members, -1)
    for f in FIELDS:
        if f == "theta":
            continue
        b = pert[f].reshape(setup.members, -1)
        # Correlation between the two fields' member-space perturbation
        # patterns, averaged over gridpoints.
        num = (a * b).sum(axis=0)
        den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
        corr = float(np.mean(num / den))
        assert corr > 0.2, (f, corr, "no usable cross-variable covariance")


def test_smoothed_field_has_the_requested_correlation_length():
    """Guard the gate: the truth must be spatially correlated, not white.

    A white-noise truth would make localisation untestable -- every
    gridpoint would be independent, and an increment that failed to spread
    would score identically to one that spread correctly.
    """
    rng = np.random.default_rng(1)
    a = np.asarray(smooth_random_field(rng, (4, 40, 40), 4.0))
    flat = a.reshape(4, -1)
    assert abs(float(flat.mean())) < 0.1
    assert float(np.sqrt((a ** 2).mean())) == pytest.approx(1.0, abs=1e-6)
    # Lag-1 correlation along x should be high for a 4-gridpoint length.
    lag1 = float(np.mean(a[:, :, :-1] * a[:, :, 1:]))
    lag8 = float(np.mean(a[:, :, :-8] * a[:, :, 8:]))
    assert lag1 > 0.8, lag1
    assert lag8 < lag1


@pytest.mark.parametrize("operator", ["direct", "speed", "both"])
def test_osse_works_for_every_observation_operator(operator):
    """Including a nonlinear H, and two obs types with different radii.

    ``speed`` is sqrt(u^2+v^2): the filter never sees a tangent-linear
    operator, only H applied to each member, so a nonlinear H must work
    without any special handling.
    """
    r = run_osse(OsseSetup(members=_MEMBERS, operator=operator))
    observed = "theta" if operator in ("direct", "both") else "u"
    assert r.improvement(observed) > 0.05, r.report()
    total_prior = sum(r.prior_rmse[f] / r.prior_rmse[f] for f in FIELDS)
    total_anal = sum(r.analysis_rmse[f] / r.prior_rmse[f] for f in FIELDS)
    assert total_anal < total_prior, r.report()


def test_rtps_keeps_spread_up_without_destroying_the_analysis():
    """Inflation trades a little accuracy for the spread cycling needs."""
    plain = run_osse(OsseSetup(members=_MEMBERS, rtps_alpha=0.0))
    relaxed = run_osse(OsseSetup(members=_MEMBERS, rtps_alpha=0.9))
    assert relaxed.analysis_spread["theta"] > plain.analysis_spread["theta"]
    assert relaxed.improvement("theta") > 0.10, relaxed.report()


@pytest.mark.parametrize("alpha", [0.0, 0.8])
def test_cycling_osse_does_not_diverge(alpha):
    """Three cycles with an advecting truth: error must fall, and keep falling.

    A single-cycle gate cannot see filter divergence, which is the failure
    mode that actually ends DA systems: the analysis is good once, the
    spread it consumed is never replaced, and by cycle five the filter
    rejects every observation.

    What this gate does NOT assert is that each cycle improves by some fixed
    percentage.  It did at first, at 10%, and cycle 2 failed at 8.7% -- on a
    filter that was working perfectly.  The fractional improvement shrinks
    precisely BECAUSE the filter is converging: theta RMSE runs
    1.77 -> 1.20 -> 1.02 -> 0.92 -> 0.87 over five cycles, so each cycle has
    less error left to remove than the last.  Gating on a percentage
    penalises success.  The properties that actually distinguish convergence
    from divergence are that every cycle improves at all, and that the
    absolute error never goes back up.
    """
    history = run_cycling_osse(
        OsseSetup(members=_MEMBERS, rtps_alpha=alpha), cycles=3)
    assert len(history) == 3
    for c, r in enumerate(history):
        assert r.improvement("theta") > 0.0, (c, r.report())
        assert r.analysis_spread["theta"] > 0.0, (c, "spread collapsed")
    errs = [h.analysis_rmse["theta"] for h in history]
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < history[0].prior_rmse["theta"] * 0.75, errs


def test_prior_inflation_reaches_the_whole_domain_not_just_the_observed_part():
    """rho = 1.05 in a twin where two thirds of the domain sees no observation.

    Sparser and tighter than the default twin on purpose: 31 observations
    and a 6 km/1.5 km lens leave about a third of the gridpoints active, so
    "did the inflation reach the rest" is a question with a measurable
    answer.  Points with no local observation used to bypass the transform
    and therefore bypass Hunt's rho with it, so prior inflation applied to
    the observed part of the domain and silently did not apply anywhere
    else -- discontinuous at the cutoff, and short of what was asked for by
    exactly the fraction of the domain the radar cannot see.

    The falsification is arithmetic rather than a threshold.  If inflation
    only reached the active points, the whole-domain spread ratio could not
    exceed

        1 + f_active * (sqrt(rho) - 1),

    because sqrt(rho) is the most any single point's spread can grow
    (that is the no-observation limit; an observed point grows less).  The
    measured ratio has to sit above that line and just under sqrt(rho).
    """
    common = dict(members=_MEMBERS, obs_fraction=0.004,
                  horizontal_cutoff_m=6000.0, vertical_cutoff_m=1500.0,
                  rtps_alpha=0.0)
    plain = run_osse(OsseSetup(prior_inflation=1.0, **common))
    inflated = run_osse(OsseSetup(prior_inflation=1.05, **common))
    assert plain.diagnostics.prior_inflation == 1.0
    assert inflated.diagnostics.prior_inflation == 1.05

    d = inflated.diagnostics
    f_active = d.active_points / d.total_points
    # The twin is genuinely observation-sparse, or this test says nothing.
    assert 0.1 < f_active < 0.5, (d.active_points, d.total_points)

    root = math.sqrt(1.05)
    only_active = 1.0 + f_active * (root - 1.0)
    for f in FIELDS:
        ratio = inflated.analysis_spread[f] / plain.analysis_spread[f]
        assert ratio > only_active + 0.005, (f, ratio, only_active)
        assert ratio < root + 1e-9, (f, ratio)
    # And it did not buy that spread by throwing the analysis away: with
    # this few observations the improvement is small, but it is still an
    # improvement and inflation does not reverse it.
    assert inflated.improvement("theta") > 0.01, inflated.report()
    assert inflated.improvement("theta") > plain.improvement("theta")


def test_rtps_holds_spread_across_cycles():
    """The reason to inflate: without it, cycling bleeds the spread away.

    Also the honest caveat.  This OSSE has a PERFECT model -- the truth and
    the ensemble are advected by the same exact operator -- so there is no
    model error for inflation to compensate, and alpha = 0.8 leaves the
    ensemble over-dispersive by cycle three (spread ~1.5 against an error
    of ~0.94).  That is not a bug in RTPS; it is what tuning inflation
    against a perfect-model twin would tell you if you mistook this
    experiment for a tuning exercise.  The gate therefore checks only the
    direction, and the handoff records the caveat.
    """
    plain = run_cycling_osse(OsseSetup(members=_MEMBERS, rtps_alpha=0.0),
                             cycles=3)
    inflated = run_cycling_osse(OsseSetup(members=_MEMBERS, rtps_alpha=0.8),
                                cycles=3)
    assert (inflated[-1].analysis_spread["theta"]
            > plain[-1].analysis_spread["theta"])
    # Both still reduce error: inflation must not buy spread with accuracy.
    for h in (plain, inflated):
        assert h[-1].analysis_rmse["theta"] < h[0].prior_rmse["theta"]
