"""Region-based dealiasing: what it recovers, and what it refuses to guess.

Every case here is one where the truth is known by construction, because a
dealiasing test that cannot state the right answer is only testing that the
code runs.  A synthetic field is folded at a known Nyquist, the unfolder is
asked to recover it, and the assertion is against the pre-fold truth --
exactly, not approximately, because unfolding is integer arithmetic and an
answer that is close is an answer that is wrong.

The adversarial half matters more than the recovery half.  The three cases
:mod:`gpuwm.obs.superob` names as beyond its masks -- a spatially coherent
fold with no gate-to-gate jump in its interior, a couplet straddling the
Nyquist, and a region isolated from any reliable reference -- each get a
test, and in each the demanded outcome is either the true field or an
explicit refusal.  Never a plausible number.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.obs import dealias
from gpuwm.obs.dealias import (ENGINE_VAD_REGION, REASON_NO_NYQUIST,
                               REASON_NONFINITE, REASON_UNRESOLVED,
                               STATE_REJECTED, STATE_UNCHANGED,
                               STATE_UNFOLDED, DealiasParams,
                               DealiasParamsError, WindProfile, dealias_sweep)


def _vad(**kwargs) -> DealiasParams:
    """Parameters for the engine this module tests, named.

    Every case here states a truth this engine's algorithm must recover or
    refuse -- its VAD reference, its anchors, its abstentions -- so it
    names the engine rather than reading whichever one is currently the
    shipped default.  When the default moved to ``region-global`` on
    2026-08-12 these tests kept testing what they were written to test,
    which is the whole reason the engine is named here.
    """

    return DealiasParams(engine=ENGINE_VAD_REGION, **kwargs)

#: A real WSR-88D Doppler-cut Nyquist -- the one the KDMX case reports on
#: every tilt from 0.53 through 6.28 degrees, which is why the 0.8 mask caps
#: those tilts at 20.4 m/s.
NYQUIST = 25.51
INTERVAL = 2.0 * NYQUIST


def fold(truth: np.ndarray, nyquist: float = NYQUIST) -> np.ndarray:
    """What a radar reports for ``truth``: the phase-wrapped velocity."""

    return (truth + nyquist) % (2.0 * nyquist) - nyquist


def _azimuths(rows: int = 360) -> np.ndarray:
    return np.arange(rows, dtype=np.float64) * (360.0 / rows)


def _environment(rows: int = 360, gates: int = 300, *,
                 u: float = 10.0, v: float = 12.0,
                 shear_u: float = 0.0, shear_v: float = 0.0) -> np.ndarray:
    """A smooth VAD-conforming wind field, ``(radial, gate)`` true ``Vr``."""

    azimuth = np.radians(_azimuths(rows))[:, None]
    span = np.arange(gates, dtype=np.float64)[None, :] * 0.25   # km
    return ((u + shear_u * span) * np.sin(azimuth)
            + (v + shear_v * span) * np.cos(azimuth))


# --------------------------------------------------------------------------
# recovery on known folds
# --------------------------------------------------------------------------

def test_smooth_folded_field_is_recovered_exactly():
    """The base case: a wind that outruns Nyquist comes back bit-exact.

    The truth here reaches 32.8 m/s against a 25.51 m/s Nyquist, so 12.7% of
    the sweep is folded and the shipping mask would have discarded every gate
    above 20.4 m/s.  Recovery must be exact -- the fold is an integer
    multiple of 51.02, and adding the right integer reproduces the input
    float for float.
    """

    truth = _environment(shear_u=0.10, shear_v=0.14)
    observed = fold(truth)
    assert np.abs(truth).max() > NYQUIST, "the fixture must actually fold"

    result = dealias_sweep(observed, _azimuths(), NYQUIST, _vad())

    assert result.stats["gates_rejected"] == 0
    assert result.stats["gates_unfolded"] > 0
    np.testing.assert_allclose(result.velocity, truth, atol=1e-9)


def test_states_and_folds_agree_with_the_truth_gate_by_gate():
    """``UNFOLDED`` where a fold was applied, ``UNCHANGED`` where none was.

    The three states are a claim about each gate, not a summary, so they are
    checked against the fold each gate actually needed.
    """

    truth = _environment(shear_u=0.10, shear_v=0.14)
    observed = fold(truth)
    expected_fold = np.rint((truth - observed) / INTERVAL).astype(int)

    result = dealias_sweep(observed, _azimuths(), NYQUIST, _vad())

    np.testing.assert_array_equal(result.fold, expected_fold)
    np.testing.assert_array_equal(result.state == STATE_UNFOLDED,
                                  expected_fold != 0)
    np.testing.assert_array_equal(result.state == STATE_UNCHANGED,
                                  expected_fold == 0)


def test_a_background_wind_profile_can_stand_in_for_the_vad():
    """The model-background route anchors what a VAD would have anchored."""

    truth = _environment(shear_u=0.10, shear_v=0.14)
    observed = fold(truth)
    profile = WindProfile(height_m=np.array([0.0, 10000.0]),
                          u_ms=np.array([10.0, 10.0]),
                          v_ms=np.array([12.0, 12.0]))
    azimuth = np.broadcast_to(_azimuths()[:, None], observed.shape)
    reference = profile.radial_reference(
        azimuth, np.zeros_like(azimuth), np.zeros_like(azimuth))

    result = dealias_sweep(observed, _azimuths(), NYQUIST, _vad(),
                           reference=reference)

    assert result.stats["gates_rejected"] == 0
    np.testing.assert_allclose(result.velocity, truth, atol=1e-9)


# --------------------------------------------------------------------------
# adversarial: the couplet straddling the Nyquist
# --------------------------------------------------------------------------

def _couplet(rows=360, gates=300, *, centre_row=90, centre_gate=150,
             peak=26.0, radius=9.0):
    """A rotational couplet: opposite-signed lobes across one azimuth.

    Centred at azimuth 90 degrees, where the environmental wind projects
    fully onto the beam, so the outbound lobe carries the true field past
    Nyquist while the inbound lobe does not -- precisely the configuration
    that folds one half of a couplet and not the other.  The rotation is
    resolved across several radials rather than jumping in one step, which
    is what a real couplet looks like at this range and what makes it
    resolvable at all; an unresolved jump of about a Nyquist is genuinely
    ambiguous and has its own test below.
    """

    row = np.arange(rows, dtype=np.float64)[:, None]
    gate = np.arange(gates, dtype=np.float64)[None, :]
    across = row - centre_row
    along = (gate - centre_gate) / 2.0
    envelope = np.exp(-(across ** 2 + along ** 2) / (2.0 * radius ** 2))
    return peak * np.tanh(across / 3.0) * envelope


def test_a_couplet_straddling_the_nyquist_survives_intact():
    """The number the whole capability exists for.

    The couplet's outbound lobe reaches past the Nyquist velocity and folds;
    its inbound lobe does not.  The shipping 0.8 mask deletes the folded half
    *and* the top of the unfolded half, which is how a mesocyclone signature
    is erased by quality control.  Both lobes must come back exactly, and the
    couplet's peak-to-peak difference -- the quantity a rotation detector
    reads -- must be the true one.
    """

    truth = _environment() + _couplet()
    observed = fold(truth)
    assert truth.max() > NYQUIST, "the outbound lobe must fold"

    result = dealias_sweep(observed, _azimuths(), NYQUIST, _vad())

    core = np.zeros(truth.shape, dtype=bool)
    core[81:100, 141:160] = True
    assert result.state[core].min() != STATE_REJECTED, (
        "the couplet was rejected; abstention is correct only where the "
        "evidence is ambiguous, and a resolved ramp through the Nyquist is "
        "not ambiguous")
    np.testing.assert_allclose(result.velocity[core], truth[core], atol=1e-9)

    recovered_span = result.velocity[core].max() - result.velocity[core].min()
    true_span = truth[core].max() - truth[core].min()
    assert recovered_span == pytest.approx(true_span, abs=1e-9)

    masked = np.where(np.abs(observed) <= 0.8 * NYQUIST, observed, np.nan)
    masked_span = np.nanmax(masked[core]) - np.nanmin(masked[core])
    assert masked_span < true_span - 5.0, (
        "the fixture must be one the shipping mask actually damages, or "
        "this test proves nothing about the recovery")


# --------------------------------------------------------------------------
# adversarial: the coherent fold with no interior jump
# --------------------------------------------------------------------------

def _moated(field: np.ndarray, rows: slice, gates: slice) -> np.ndarray:
    """Cut a patch free of its surroundings with a ring of missing gates.

    A NaN moat is how an echo island arrives in real data -- clutter
    suppression, low SNR, a range hole -- and it is what makes a region
    unreachable by continuity: there is no adjacent gate pair to vote across.
    """

    out = np.array(field, dtype=np.float64, copy=True)
    out[rows.start - 2:rows.start, gates.start - 2:gates.stop + 2] = np.nan
    out[rows.stop:rows.stop + 2, gates.start - 2:gates.stop + 2] = np.nan
    out[rows.start - 2:rows.stop + 2, gates.start - 2:gates.start] = np.nan
    out[rows.start - 2:rows.stop + 2, gates.stop:gates.stop + 2] = np.nan
    return out


def test_a_coherent_fold_with_no_interior_jump_is_refused_not_guessed():
    """The case :mod:`gpuwm.obs.superob` documents as beyond its masks.

    An isolated patch whose every gate folded together has a plausible
    Nyquist, speeds well inside the 0.8 threshold, no in-cell spread, and no
    gate-to-gate jump anywhere in its interior.  All four shipping masks pass
    it, and it is assimilable and wrong by 51.02 m/s.

    Nothing can resolve it: the moat denies it continuity, and its true
    velocity is far enough from the environmental wind that the reference
    cannot pin it either.  The demanded behaviour is therefore a *refusal*.
    Recovering it is not on the table; quietly keeping it -- which is what
    ships today -- is the failure.
    """

    rows, gates = slice(150, 190), slice(120, 170)
    truth = _environment()
    # A constant 60 m/s across the patch: it folds as a whole, its interior
    # is perfectly smooth, and its folded value (+8.98) is far enough from
    # the environmental wind there (about -9) that the reference cannot
    # mistake it for an unfolded gate.
    truth[rows, gates] = 60.0
    observed = _moated(fold(truth), rows, gates)

    patch = np.zeros(truth.shape, dtype=bool)
    patch[rows, gates] = True
    interior = np.zeros(truth.shape, dtype=bool)
    interior[152:188, 122:168] = True

    # The fixture must genuinely be the adversarial case: folded, smooth,
    # and invisible to every rule that ships.
    assert np.all(np.abs(np.rint(
        (truth[patch] - observed[patch]) / INTERVAL)) == 1)
    assert np.abs(observed[patch]).max() <= 0.8 * NYQUIST, (
        "the patch must pass the 0.8 magnitude mask, or it is not the case "
        "this test is about")
    jumps = np.abs(np.diff(observed[interior.any(axis=1)][:, 122:168], axis=1))
    assert np.nanmax(jumps) < 0.75 * INTERVAL, (
        "the patch interior must be smooth, or the shear scan would have "
        "caught it and this would not be the uncaught case")

    result = dealias_sweep(observed, _azimuths(), NYQUIST, _vad())

    assert np.all(result.state[interior] == STATE_REJECTED)
    assert np.all(result.reason[interior] == REASON_UNRESOLVED)
    assert np.all(np.isnan(result.velocity[interior])), (
        "a refused gate must not ship a number; a consumer reading velocity "
        "without reading state would otherwise assimilate the guess")


def test_a_coherent_fold_the_reference_can_pin_is_recovered():
    """The other half: isolation is not fatal when the wind explains it.

    Here the isolated patch folded because the *environment* there exceeds
    Nyquist, not because of a local perturbation -- which is what actually
    happens on the upper tilts, where a 37 m/s jet folds a whole region
    together.  The fold-aware VAD knows the true wind, so the reference pins
    the patch and it comes back exactly.  The contrast with the test above is
    the point: the unfolder abstains on missing evidence, not on isolation.
    """

    rows, gates = slice(150, 190), slice(120, 170)
    truth = _environment(u=30.0, v=26.0)       # a wind that folds by itself
    observed = _moated(fold(truth), rows, gates)
    interior = np.zeros(truth.shape, dtype=bool)
    interior[152:188, 122:168] = True
    folded = np.rint((truth - fold(truth)) / INTERVAL) != 0
    assert folded[interior].any(), "the patch must contain folded gates"

    result = dealias_sweep(observed, _azimuths(), NYQUIST, _vad())

    resolved = result.state[interior] != STATE_REJECTED
    assert resolved.mean() > 0.99
    np.testing.assert_allclose(result.velocity[interior][resolved],
                               truth[interior][resolved], atol=1e-9)


# --------------------------------------------------------------------------
# adversarial: no reliable reference at all
# --------------------------------------------------------------------------

def test_a_sector_scan_with_no_fittable_reference_refuses_everything():
    """A harmonic fitted inside one sector is an extrapolation.

    With samples in too few azimuth sectors the VAD cannot be believed, so no
    region can anchor, so nothing can be resolved.  The unfolder must then
    reject the whole sweep rather than fall back on "assume no fold", which
    is the assumption that makes a coherent fold assimilable in the first
    place.
    """

    truth = _environment(rows=40)[:40]         # a 40-degree sector
    observed = fold(truth)
    azimuth = np.arange(40, dtype=np.float64)  # 40 x 1 degree = 1.1 sectors

    result = dealias_sweep(observed, azimuth, NYQUIST, _vad(),
                           wraps=False)

    assert result.stats["reference"]["bands_valid"] == 0
    assert result.stats["regions_anchored"] == 0
    finite = np.isfinite(observed)
    assert np.all(result.state[finite] == STATE_REJECTED)
    assert np.all(result.reason[finite] == REASON_UNRESOLVED)


def test_an_ambiguous_jump_of_about_one_nyquist_abstains():
    """Where "fold" and "no fold" are equally good stories, tell neither.

    A patch offset from its surroundings by very nearly one Nyquist velocity
    -- half an interval -- is the genuinely undecidable case: the boundary
    votes land midway between two integers, and rounding either way is a
    coin toss dressed as an answer.
    """

    rows, gates = slice(150, 190), slice(120, 170)
    truth = _environment()
    truth[rows, gates] += NYQUIST              # exactly half an interval
    observed = fold(truth)
    interior = np.zeros(truth.shape, dtype=bool)
    interior[152:188, 122:168] = True

    result = dealias_sweep(observed, _azimuths(), NYQUIST, _vad())

    assert np.all(result.state[interior] == STATE_REJECTED)


# --------------------------------------------------------------------------
# the accounting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [
    lambda: fold(_environment(shear_u=0.10, shear_v=0.14)),
    lambda: fold(_environment() + _couplet()),
    lambda: _moated(fold(_environment(u=30.0, v=26.0)),
                    slice(150, 190), slice(120, 170)),
])
def test_every_gate_lands_in_exactly_one_of_three_states(builder):
    """``unchanged + unfolded + rejected`` is the count of finite gates.

    Asserted rather than assumed: the counters are how the capability is
    judged, and a gate that falls out of the accounting is a gate whose fate
    nobody can audit.
    """

    observed = builder()
    result = dealias_sweep(observed, _azimuths(), NYQUIST, _vad())
    stats = result.stats

    assert (stats["gates_unchanged"] + stats["gates_unfolded"]
            + stats["gates_rejected"]) == stats["gates_finite"]
    assert stats["gates_finite"] == int(np.isfinite(observed).sum())
    assert stats["gates_rejected"] == sum(stats["rejected"].values())

    finite = np.isfinite(observed)
    assert np.all(np.isin(result.state[finite],
                          [STATE_REJECTED, STATE_UNCHANGED, STATE_UNFOLDED]))
    assert np.all(result.state[~finite] == STATE_REJECTED)
    assert np.all(result.reason[~finite] == REASON_NONFINITE)
    # A resolved gate ships a number; a refused one never does.
    assert np.all(np.isfinite(result.velocity[result.state != STATE_REJECTED]))
    assert np.all(np.isnan(result.velocity[result.state == STATE_REJECTED]))


def test_a_sweep_without_a_nyquist_refuses_every_gate():
    """No Nyquist, nothing knowable -- the same refusal superob already makes."""

    observed = fold(_environment())
    result = dealias_sweep(observed, _azimuths(), None, _vad())

    finite = np.isfinite(observed)
    assert np.all(result.state[finite] == STATE_REJECTED)
    assert np.all(result.reason[finite] == REASON_NO_NYQUIST)
    assert result.stats["rejected"]["no_nyquist"] == int(finite.sum())


def test_speeds_beyond_the_physical_bound_are_rejected():
    """An unfolding that lands past any real wind is a failure, not a wind."""

    observed = fold(_environment(shear_u=0.10, shear_v=0.14))
    result = dealias_sweep(observed, _azimuths(), NYQUIST,
                           _vad(max_speed_ms=20.0))

    kept = result.state != STATE_REJECTED
    assert np.all(np.abs(result.velocity[kept]) <= 20.0)
    assert result.stats["rejected"]["speed_out_of_range"] > 0


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"region_join_fraction": 1.5},
    {"region_join_fraction": -0.1},
    {"edge_agreement": 2.0},
    {"max_fold": 0},
    {"max_fold": 2.5},
    {"edge_min_pairs": 0},
    {"max_speed_ms": 0.0},
    {"max_speed_ms": float("nan")},
    {"reference_min_sectors": 13},
    {"anchor_min_gates": -4},
    {"region_join_fraction": "0.5"},
    {"region_join_fraction": True},
    {"keep_beyond_reject_fraction": 1},
])
def test_impossible_parameters_are_refused(kwargs):
    with pytest.raises(DealiasParamsError):
        DealiasParams(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"height_m": [0.0, 1000.0], "u_ms": [1.0], "v_ms": [2.0, 3.0]},
    {"height_m": [1000.0, 0.0], "u_ms": [1.0, 2.0], "v_ms": [2.0, 3.0]},
    {"height_m": [0.0], "u_ms": [1.0], "v_ms": [2.0]},
])
def test_an_unusable_wind_profile_is_refused(kwargs):
    with pytest.raises(DealiasParamsError):
        WindProfile(**{k: np.asarray(v, dtype=float)
                       for k, v in kwargs.items()})


def test_params_round_trip_through_their_payload():
    params = DealiasParams(region_join_fraction=0.45, max_fold=3,
                           keep_beyond_reject_fraction=False)
    payload = params.to_payload()
    assert payload["max_fold"] == 3 and isinstance(payload["max_fold"], int)
    assert payload["keep_beyond_reject_fraction"] is False
    assert DealiasParams(**payload) == params


# --------------------------------------------------------------------------
# the stated limit
# --------------------------------------------------------------------------

def test_a_fold_that_looks_exactly_like_the_environment_is_a_stated_limit():
    """Pinned, because it is the boundary of what one sweep can know.

    An isolated region folded by one interval whose *folded* value happens to
    match the environmental wind is indistinguishable, in this sweep, from an
    unfolded region sitting in that wind.  The moat denies it continuity, and
    the reference agrees with the folded reading -- there is no evidence left
    to appeal to.  The unfolder therefore reports it UNCHANGED, which is also
    what ships today, so the capability is no worse here; it is simply not
    better.

    Breaking this needs evidence from outside the sweep: the previous
    volume's field, or the tilt above and below.  BowEcho's v4 volume solver
    carries exactly those two terms, which is the concrete reason to migrate
    this upstream rather than grow a second engine here.  The test exists so
    that the day the limit moves, it moves deliberately.
    """

    rows, gates = slice(150, 190), slice(120, 170)
    truth = _environment()
    # Folded, this reads -11.02; the environmental wind across the patch runs
    # about -5 to -13.  The reading is unremarkable and it is wrong by 51.02.
    truth[rows, gates] = 40.0
    observed = _moated(fold(truth), rows, gates)
    interior = np.zeros(truth.shape, dtype=bool)
    interior[152:188, 122:168] = True

    result = dealias_sweep(observed, _azimuths(), NYQUIST, _vad())

    assert np.all(result.state[interior] == STATE_UNCHANGED)
    assert np.all(result.fold[interior] == 0)
    assert not np.allclose(result.velocity[interior], truth[interior]), (
        "if this now recovers the truth the limit has moved; say so in the "
        "module docstring and in the dealiasing statement, because both "
        "currently tell a consumer that it has not")


# --------------------------------------------------------------------------
# do no harm
# --------------------------------------------------------------------------

def test_the_masking_only_statement_is_unchanged_to_the_byte():
    """The 718-character promise a shipped file makes has not moved.

    Pinned by digest rather than by eye.  Every observation file written
    without dealiasing carries this string, consumers read it, and a
    capability that rewrote it while claiming to change nothing would have
    changed the contract in the one place a reader looks for the contract.
    """

    import hashlib

    from gpuwm.obs.radar_grid import (_DEALIASING_STATEMENT,
                                      dealiasing_statement)
    from gpuwm.obs.superob import SuperobParams

    assert len(_DEALIASING_STATEMENT) == 718
    assert hashlib.sha256(
        _DEALIASING_STATEMENT.encode("utf-8")).hexdigest() == (
        "5825d41ba7f790237a09576f524289770030672722dbc0a95c2f1601aa5ae42a")
    assert dealiasing_statement(SuperobParams()) is _DEALIASING_STATEMENT


def test_default_params_serialise_to_the_historical_key_set():
    """No ``dealias`` key appears until someone asks for dealiasing.

    ``superob_params`` is a file attribute and ``provenance.superob_params``
    is a JSON blob; adding a key to either changes every byte after it.  The
    identity of the disabled path is that the key set does not move.

    The two ``clear_air_*`` keys are not an exception to that rule; they are
    the clear-air lane's own additions, made and justified before this one,
    and they are spelled out here rather than globbed so that a *third*
    lane adding a field still has to come and say so in this test.
    """

    from gpuwm.obs.superob import SuperobParams

    payload = SuperobParams().to_payload()
    assert set(payload) == {
        "nyquist_reject_fraction", "nyquist_min_ms", "nyquist_max_ms",
        "nyquist_spread_fraction", "shear_fold_fraction",
        "min_reflectivity_dbz", "max_range_km", "max_elevation_deg",
        "z_error_base_dbz", "vr_error_base_ms", "z_error_floor_dbz",
        "vr_error_floor_ms", "refraction_factor", "earth_radius_m",
        "clear_air_min_gates", "clear_air_error_dbz"}
    assert all(isinstance(value, float) for value in payload.values())
    assert "dealias" in SuperobParams(dealias=DealiasParams()).to_payload()


def test_the_unfolder_is_not_reached_when_dealiasing_is_off(monkeypatch):
    """Not "produces the same answer" -- never runs at all.

    A capability that runs and then discards its result is one refactor away
    from not discarding it.  This asserts the disabled path does not enter
    the module, which is the only version of "off" that stays off.
    """

    import gpuwm.obs.superob as superob

    def explode(*args, **kwargs):                # pragma: no cover - the point
        raise AssertionError(
            "dealias_sweep was called with SuperobParams.dealias unset")

    monkeypatch.setattr(superob, "dealias_sweep", explode)

    from test_obs_radar_grid import _grid, _volume

    grid = _grid()
    volume = _volume(grid, reflectivity=[35.0] * 6,
                     velocity=[24.0, -24.0, 12.0, -12.0, 0.0, 30.0],
                     azimuths=(45.0, 90.0, 135.0))
    contribution = superob.superob_volume(volume, grid)
    assert contribution.dealias == {}


def test_the_disabled_path_writes_the_same_bytes_as_a_repeat_build(tmp_path):
    """Two builds, one file: the writer is deterministic and unmoved.

    This is the unit-scale half of the do-no-harm proof; the other half
    rebuilds a real six-cycle case against the pre-change code and compares
    digests, which no unit test can do because it needs the volumes.
    """

    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import (SuperobParams, merge_contributions,
                                   superob_volume)
    from test_obs_radar_grid import _grid, _volume

    grid = _grid()
    volume = _volume(grid, reflectivity=[35.0] * 6,
                     velocity=[24.0, -24.0, 12.0, -12.0, 0.0, 30.0],
                     azimuths=(45.0, 90.0, 135.0))
    digests = []
    for name in ("a.nc", "b.nc"):
        params = SuperobParams()
        contribution = superob_volume(volume, grid, params=params)
        observations = merge_contributions([contribution], grid, params=params)
        receipt = write_radar_grid(tmp_path / name, observations, grid,
                                   valid_time=volume.valid_time,
                                   params=params, provenance={"case": "unit"})
        digests.append(receipt["sha256"])
    assert digests[0] == digests[1]


# --------------------------------------------------------------------------
# the volume-wide profile
# --------------------------------------------------------------------------

def _cut(elevation_deg, *, u, v, rows=360, gates=300, gate_m=250.0):
    """One elevation cut of a volume with a known wind, folded."""

    azimuth = np.radians(_azimuths(rows))[:, None]
    truth = ((u * np.sin(azimuth) + v * np.cos(azimuth))
             * np.cos(np.radians(elevation_deg)))
    truth = np.broadcast_to(truth, (rows, gates))
    ranges = np.arange(gates, dtype=np.float64) * gate_m + 2125.0
    return (elevation_deg, fold(truth), _azimuths(rows), NYQUIST, ranges)


def test_the_volume_profile_recovers_the_wind_the_volume_was_built_from():
    """Pooling every cut's bands gives back ``u(z), v(z)``.

    The cuts here share one wind, so every layer the profile believes must
    reproduce it -- including the layers where the true radial velocity
    exceeds Nyquist and every contributing band was fitted through folds.
    """

    from gpuwm.obs.dealias import volume_wind_profile

    cuts = [_cut(elevation, u=18.0, v=-24.0)
            for elevation in (0.5, 1.5, 2.4, 3.4, 4.3, 6.0, 8.0, 10.0)]
    profile = volume_wind_profile(cuts, _vad())

    assert profile is not None
    assert profile.height_m.size >= 2
    np.testing.assert_allclose(profile.u_ms, 18.0, atol=1.5)
    np.testing.assert_allclose(profile.v_ms, -24.0, atol=1.5)


def test_a_layer_only_one_elevation_reached_is_not_believed():
    """Three bands off one cut are one measurement read three times.

    They share the geometry, the echo, and any error in it, so they cannot
    corroborate each other.  A volume that offers only a single elevation
    must therefore produce no profile at all rather than a confident one.
    """

    from gpuwm.obs.dealias import volume_wind_profile

    assert volume_wind_profile([_cut(0.5, u=18.0, v=-24.0)],
                               _vad()) is None


def test_the_profile_refuses_to_extrapolate_past_what_it_sampled():
    """Outside the sampled heights the profile knows nothing and says so.

    ``np.interp`` holds its end value flat forever, which would assert the
    13 km wind at 20 km and anchor gates to it.  NaN is the honest answer and
    it propagates into "no reference here", which the unfolder already knows
    how to handle.
    """

    profile = WindProfile(height_m=np.array([1000.0, 5000.0]),
                          u_ms=np.array([10.0, 30.0]),
                          v_ms=np.array([0.0, 0.0]))
    azimuth = np.array([90.0, 90.0, 90.0])
    elevation = np.zeros(3)
    reference = profile.radial_reference(
        azimuth, elevation, np.array([500.0, 3000.0, 9000.0]))

    assert np.isnan(reference[0]) and np.isnan(reference[2])
    assert reference[1] == pytest.approx(20.0, abs=1e-9)


# --------------------------------------------------------------------------
# the coarse VAD search: the native kernel may go faster, never elsewhere
# --------------------------------------------------------------------------


def _coarse_band(seed: int, samples: int, *, speed: float, direction: float,
                 noise: float = 1.5) -> tuple:
    """One range band's worth of folded samples, truth known."""

    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, samples)
    truth = (speed * np.cos(direction) * np.cos(azimuth)
             + speed * np.sin(direction) * np.sin(azimuth)
             + rng.normal(0.0, noise, samples))
    return fold(truth), np.sin(azimuth), np.cos(azimuth)


def test_the_native_search_returns_the_seeds_the_exhaustive_one_does():
    """Band for band, tuple for tuple, on bands built to be hard.

    The kernel is allowed to be faster and is not allowed to be different.
    It never ranks candidates: it shortlists them, and this asserts the
    shortlist plus the exact ranking reproduces the exhaustive search --
    including the last band, where every zero-speed candidate ties exactly
    and only an argsort's stability decides which three come back.
    """

    bands = [_coarse_band(index, samples, speed=speed, direction=direction)
             for index, (samples, speed, direction) in enumerate(
                 [(240, 39.7, 1.1), (900, 8.0, 4.0), (11000, 44.0, 2.2),
                  (4000, 21.5, 0.0)])]
    calm = _coarse_band(99, 3000, speed=0.0, direction=0.0, noise=0.0)
    bands.append((np.zeros_like(calm[0]), calm[1], calm[2]))

    exhaustive = [dealias._coarse_seeds(*band, NYQUIST) for band in bands]
    table = dealias._coarse_seed_table(bands, NYQUIST)

    assert table == exhaustive


def test_the_shortlist_guard_widens_rather_than_guessing():
    """A shortlist that cannot be proved sufficient is not used.

    Forcing the tolerance to something enormous makes every guard fail, so
    the shortlist has to grow to the whole grid -- which is the exhaustive
    search.  The seeds must be unchanged, because widening is the safe
    direction and the exact cost still ranks whatever survives.
    """

    band = _coarse_band(7, 2000, speed=33.0, direction=2.5)
    approx = dealias._coarse_cost(*dealias._coarse_subsample(*band), NYQUIST,
                                  dealias._COARSE_A, dealias._COARSE_B)
    values, sin_az, cos_az = dealias._coarse_subsample(*band)

    seeds, priced = dealias._coarse_shortlisted(
        approx, values, sin_az, cos_az, NYQUIST, 3)
    assert priced == dealias._COARSE_SHORTLIST

    original = dealias._COARSE_COST_TOLERANCE
    try:
        dealias._COARSE_COST_TOLERANCE = 1e9
        widened, priced = dealias._coarse_shortlisted(
            approx, values, sin_az, cos_az, NYQUIST, 3)
    finally:
        dealias._COARSE_COST_TOLERANCE = original

    assert priced == dealias._COARSE_A.size
    assert widened == seeds == dealias._coarse_seeds(*band, NYQUIST)


def test_an_absent_native_library_changes_no_decision(monkeypatch):
    """Same sweep, both paths, gate for gate.

    This is the acceptance in miniature: whatever the box has, the two
    routes through the coarse search must produce the same velocities, the
    same states, the same reasons and the same folds.  On a tree with the
    library built one arm is native; on one without it the test still holds
    the refactor to the original arithmetic.
    """

    truth = _environment(rows=360, gates=300, u=31.0, v=-19.0, shear_u=0.35)
    velocity = fold(truth)
    azimuth = _azimuths(360)

    # `_vad()`, not a bare `DealiasParams()`: the coarse seed search this
    # test is about belongs to the vad-region engine, and the shipped
    # default became `region-global` on 2026-08-12. A bare default here
    # dispatches to the other engine, which never runs this code and
    # needs a native library besides.
    native = dealias_sweep(velocity, azimuth, NYQUIST, _vad())
    monkeypatch.setattr(dealias.coarse_cost, "coarse_cost_batch",
                        lambda *args, **kwargs: None)
    plain = dealias_sweep(velocity, azimuth, NYQUIST, _vad())

    np.testing.assert_array_equal(native.velocity, plain.velocity)
    np.testing.assert_array_equal(native.state, plain.state)
    np.testing.assert_array_equal(native.reason, plain.reason)
    np.testing.assert_array_equal(native.fold, plain.fold)
    assert plain.stats["reference"]["coarse_native"] is False
