"""Unit and correctness gates for :mod:`gpuwm.da.letkf`.

The centrepiece is :func:`test_letkf_equals_dense_kalman_gain`, which checks
the batched transform against an independent, deliberately naive
implementation of the same mathematics by a completely different route: an
explicit dense Kalman gain, one gridpoint at a time, with the localised
observation set found by brute-force distance search over the whole domain
rather than by the module's index stencil.  The two share no code beyond
:func:`gaspari_cohn` itself, so agreement is evidence rather than tautology,
and the brute-force neighbour search validates the stencil construction as a
side effect.

That the two must agree is not an approximation.  By the Sherman-Morrison-
Woodbury identity,

    Xb [(R-1)I + Yb^T Rloc^-1 Yb]^-1 Yb^T Rloc^-1
      = Xb Yb^T [(R-1) Rloc + Yb Yb^T]^-1
      = Pb H^T (H Pb H^T + Rloc)^-1 = K,

so the LETKF mean increment IS the Kalman increment, exactly, and the ETKF
analysis covariance IS (I - KH) Pb.  A test that only checked "the RMSE went
down" would pass with a gain that was half the right size.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpuwm.da.letkf import (
    GridGeometry,
    GriddedObs,
    LetkfConfig,
    LetkfDiagnostics,
    LetkfError,
    Localization,
    analyze,
    gaspari_cohn,
)


# ---------------------------------------------------------------------------
# Gaspari-Cohn
# ---------------------------------------------------------------------------

#: Exact rational values of Gaspari and Cohn (1999) eq. (4.10), evaluated by
#: hand from the published polynomial at r = distance / c, c = cutoff / 2.
#: These are the numbers a reader can check against the paper without running
#: anything, which is the point of a table test.
#:
#:   r = 0    -> 1                       (the function is a correlation)
#:   r = 1/2  -> 263/384                 inner branch
#:   r = 1    -> 5/24                    both branches agree here
#:   r = 3/2  -> 19/1152                 outer branch
#:   r = 2    -> 0                       compact support begins
_GC_TABLE = [
    (0.0, 1.0),
    (0.25, 1.0 - 5.0 / 3.0 * 0.0625 + 5.0 / 8.0 * 0.015625
     + 0.5 * 0.00390625 - 0.25 * 0.0009765625),
    (0.5, 263.0 / 384.0),
    (1.0, 5.0 / 24.0),
    (1.5, 19.0 / 1152.0),
    (2.0, 0.0),
]


@pytest.mark.parametrize("r,expected", _GC_TABLE)
def test_gaspari_cohn_published_values(r, expected):
    """Both branches, at points whose exact values are hand-computable."""
    # cutoff = 2 puts c = 1, so r is literally the distance.
    got = float(gaspari_cohn(r, 2.0))
    assert got == pytest.approx(expected, rel=0, abs=1e-15), (r, got, expected)


def test_gaspari_cohn_scales_with_cutoff():
    """The shape is a function of distance/cutoff only."""
    r = np.linspace(0.0, 1.0, 41)
    a = np.asarray(gaspari_cohn(r * 2.0, 2.0))
    b = np.asarray(gaspari_cohn(r * 5000.0, 5000.0))
    assert np.allclose(a, b, rtol=0, atol=1e-14)


def test_gaspari_cohn_is_exactly_zero_at_and_beyond_cutoff():
    """Not "small": zero.  The localisation gate depends on this being exact.

    ``0.0`` rather than ``1e-18`` is what lets the filter drop an
    observation from a gridpoint's batch instead of letting it contribute a
    denormal amount of influence at unbounded range.
    """
    d = np.array([1000.0, 1000.0 + 1e-9, 1500.0, 1e6, np.inf])
    w = np.asarray(gaspari_cohn(d, 1000.0))
    assert np.all(w == 0.0), w


def test_gaspari_cohn_is_a_correlation_on_its_support():
    """Bounded in [0, 1], peaked at zero, and never negative.

    A negative weight would flip the sign of an observation's influence,
    which is worse than dropping it.  The published quintic is nonnegative
    on [0, 2] but evaluates to a small negative near r = 2 in floating point
    without a clamp.
    """
    d = np.linspace(0.0, 2000.0, 20001)
    w = np.asarray(gaspari_cohn(d, 1000.0))
    assert w.min() >= 0.0
    assert w.max() == 1.0
    assert w[0] == 1.0
    # Monotone decreasing on the support -- true of the published function,
    # and the cheapest way to catch a transcription error in one coefficient.
    on = w[d <= 1000.0]
    assert np.all(np.diff(on) <= 1e-15)


def test_gaspari_cohn_branches_join_smoothly():
    """C1 continuity at r = 1, where the two polynomials meet.

    The published function is continuously differentiable there; a
    transcription error in either branch almost always breaks it, and a kink
    in the localisation weight shows up as a visible ring in the analysis
    increment rather than as a test failure anywhere else.
    """
    # cutoff = 2 puts c = 1, so the first argument is r directly and
    # G'(1) = -17/24 in r.  A straddling pair therefore differs by
    # 2*eps*17/24 ~ 1.4e-6, NOT by zero: continuity bounds the gap by the
    # slope, it does not remove it.  An earlier version of this test
    # asserted 1e-11 and failed on a function that is perfectly continuous.
    eps = 1e-6
    slope = 17.0 / 24.0
    left = float(gaspari_cohn(1.0 - eps, 2.0))
    right = float(gaspari_cohn(1.0 + eps, 2.0))
    assert abs(left - right) < 3 * eps * slope

    dl = (float(gaspari_cohn(1.0, 2.0)) - left) / eps
    dr = (right - float(gaspari_cohn(1.0, 2.0))) / eps
    assert dl == pytest.approx(dr, abs=1e-5)
    assert dl == pytest.approx(-slope, abs=1e-5)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_gaspari_cohn_refuses_a_degenerate_cutoff(bad):
    with pytest.raises(LetkfError, match="cutoff"):
        gaspari_cohn(1.0, bad)


def test_gaspari_cohn_handles_integer_input():
    """Integer distances must not silently truncate the weight to 0 or 1."""
    w = np.asarray(gaspari_cohn(np.array([0, 250, 500, 1000, 2000]), 1000.0))
    assert w.dtype.kind == "f"
    assert w[0] == 1.0 and w[-1] == 0.0
    assert 0.0 < w[2] < 1.0


# ---------------------------------------------------------------------------
# A small twin, shared by the correctness tests
# ---------------------------------------------------------------------------

def _tiny_case(members=8, nz=4, ny=6, nx=6, seed=7, obs_error=0.4,
               n_obs=9, fields=("theta", "qv", "u")):
    """A domain small enough for a brute-force reference to be cheap."""
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    grid = GridGeometry(
        dx_m=1000.0, dy_m=1000.0,
        heights_m=np.array([250.0, 800.0, 1600.0, 2800.0][:nz]),
    )
    prior = {
        f: rng.standard_normal((members,) + shape) * (i + 1) + 10.0 * i
        for i, f in enumerate(fields)
    }
    mask = np.zeros(shape, dtype=bool)
    flat = rng.choice(nz * ny * nx, size=n_obs, replace=False)
    mask.reshape(-1)[flat] = True
    # A nonlinear H, so the test does not accidentally only cover identity.
    sim = np.sqrt(prior[fields[0]] ** 2 + 1.0) + 0.3 * prior[fields[-1]]
    values = np.where(
        mask, sim.mean(axis=0) + rng.standard_normal(shape) * obs_error,
        np.nan)
    obs = GriddedObs(
        name="probe", values=values, errors=obs_error, simulated=sim,
        mask=mask,
    )
    return grid, prior, obs, members, shape


def _haversine_m(lat1, lon1, lat2, lon2, radius_m):
    """Great-circle distance, written here and not imported.

    Deliberately a second implementation: the reference is only evidence if
    it shares no code with the thing it checks, and the localisation metric
    is now part of what is being checked.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius_m * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _dense_reference(grid, prior, obs, loc, fields, rho=1.0):
    """Explicit Kalman gain per gridpoint, brute-force neighbour search.

    Independent of the module under test in every way that matters: dense
    matrices instead of the R x R transform, ``np.linalg.solve`` instead of
    ``eigh``, an O(points x observations) distance loop instead of an index
    stencil.  Returns the ensemble-mean increment and the analysis variance,
    per field, on the full grid.
    """
    members = next(iter(prior.values())).shape[0]
    nz, ny, nx = next(iter(prior.values())).shape[1:]
    z = grid.heights_m
    z3 = z if z.ndim == 3 else np.broadcast_to(z[:, None, None],
                                               (nz, ny, nx))
    ko, jo, io = np.nonzero(np.asarray(obs.mask))
    sim = np.asarray(obs.simulated)
    vals = np.asarray(obs.values)
    err = float(obs.errors)

    dmean = {f: np.zeros((nz, ny, nx)) for f in fields}
    var_a = {f: np.zeros((nz, ny, nx)) for f in fields}
    for f in fields:
        # The reference's declared target is the rho-INFLATED prior, and a
        # gridpoint that selects no localised observation keeps exactly that.
        # Initialising to the uninflated prior instead is the mistake that
        # let this oracle ratify a filter which dropped Hunt's inflation at
        # every inactive point: both sides were wrong in the same place, so
        # 1e-9 agreement proved nothing there.  See Hunt et al. (2007)
        # section 3 step 5 -- with no observation rows, Pa~ = rho/(R-1) I and
        # Wa = sqrt(rho) I, so P^a = rho P^b.
        var_a[f] = rho * prior[f].var(axis=0, ddof=1)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if grid.lat_deg is None:
                    dh = np.hypot((io - i) * grid.dx_m,
                                  (jo - j) * grid.dy_m)
                else:
                    dh = _haversine_m(
                        grid.lat_deg[jo, io], grid.lon_deg[jo, io],
                        grid.lat_deg[j, i], grid.lon_deg[j, i],
                        grid.earth_radius_m)
                # The observation's own column, and the analysis point's own
                # column: on terrain these are different altitudes at the
                # same model level.
                dv = np.abs(z3[ko, jo, io] - z3[k, j, i])
                w = (np.asarray(gaspari_cohn(dh, loc.horizontal_m))
                     * np.asarray(gaspari_cohn(dv, loc.vertical_m)))
                sel = w > 0
                if not sel.any():
                    continue
                kk, jj, ii = ko[sel], jo[sel], io[sel]
                yb = sim[:, kk, jj, ii]                      # (R, p)
                ybar = yb.mean(axis=0)
                yb = yb - ybar
                d = vals[kk, jj, ii] - ybar
                rloc = np.diag(err ** 2 / w[sel])
                hph = yb.T @ yb / (members - 1) * rho
                gain_inner = np.linalg.inv(hph + rloc)
                for f in fields:
                    xb = prior[f][:, k, j, i]
                    xb = xb - xb.mean()
                    pht = xb @ yb / (members - 1) * rho
                    kgain = pht @ gain_inner
                    dmean[f][k, j, i] = kgain @ d
                    var_a[f][k, j, i] = (
                        xb @ xb / (members - 1) * rho - kgain @ pht
                    )
    return dmean, var_a


@pytest.mark.parametrize("rho", [1.0, 1.15])
def test_letkf_equals_dense_kalman_gain(rho):
    """The batched transform reproduces the explicit Kalman gain exactly.

    Both the ensemble-mean increment (the gain) and the analysis variance
    (the covariance update).  This is the test that would catch a sign
    error, a factor of R-1, a transposed weight matrix, or a localisation
    applied to the wrong axis -- none of which the OSSE would necessarily
    catch, and several of which still reduce RMSE.
    """
    fields = ("theta", "qv", "u")
    grid, prior, obs, members, shape = _tiny_case(fields=fields)
    loc = Localization(horizontal_m=2500.0, vertical_m=1200.0)
    cfg = LetkfConfig(localization=loc, analysis_fields=fields,
                      prior_inflation=rho, rtps_alpha=0.0)
    inc = analyze(prior, [obs], grid, cfg)

    dmean, var_a = _dense_reference(grid, prior, obs, loc, fields, rho=rho)
    for f in fields:
        got_mean = inc[f].mean(axis=0)
        assert np.allclose(got_mean, dmean[f], rtol=1e-9, atol=1e-11), (
            f, np.abs(got_mean - dmean[f]).max())
        post = prior[f] + inc[f]
        got_var = post.var(axis=0, ddof=1)
        assert np.allclose(got_var, var_a[f], rtol=1e-8, atol=1e-12), (
            f, np.abs(got_var - var_a[f]).max())


def _terrain_case(members=8, nz=4, ny=7, nx=7, seed=19, obs_error=0.4,
                  n_obs=10, fields=("theta", "u"), relief_m=2000.0):
    """A twin on terrain, geolocated, with a genuinely varying map metric.

    The height field follows a ridge of ``relief_m`` relief that decays with
    height, which is what a terrain-following coordinate does; two columns
    at the same model level near the surface are then thousands of metres
    apart vertically.  Latitude/longitude are laid out so the true east-west
    spacing is materially wider than ``dx_m`` claims -- the signature of a
    nominal spacing used without its map factor.
    """
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    ridge = relief_m * np.exp(
        -((np.arange(nx)[None, :] - (nx - 1) / 2.0) ** 2) / 3.0
    ) * np.ones((ny, 1))
    base = np.array([250.0, 800.0, 1600.0, 2800.0][:nz])
    decay = np.linspace(1.0, 0.2, nz)[:, None, None]
    heights = base[:, None, None] + ridge[None] * decay

    # 1 km of latitude is 1/111.2 degree; the longitude step is set to a
    # bigger physical stride than dx_m so the two metrics disagree.
    lat = 40.0 + np.arange(ny)[:, None] * (1000.0 / 111195.0) * np.ones(
        (1, nx))
    lon = -100.0 + np.arange(nx)[None, :] * 0.02 * np.ones((ny, 1))
    grid = GridGeometry(dx_m=1000.0, dy_m=1000.0, heights_m=heights,
                        lat_deg=lat, lon_deg=lon)

    prior = {
        f: rng.standard_normal((members,) + shape) * (i + 1) + 10.0 * i
        for i, f in enumerate(fields)
    }
    mask = np.zeros(shape, dtype=bool)
    flat = rng.choice(nz * ny * nx, size=n_obs, replace=False)
    mask.reshape(-1)[flat] = True
    sim = np.sqrt(prior[fields[0]] ** 2 + 1.0) + 0.3 * prior[fields[-1]]
    values = np.where(
        mask, sim.mean(axis=0) + rng.standard_normal(shape) * obs_error,
        np.nan)
    obs = GriddedObs(name="probe", values=values, errors=obs_error,
                     simulated=sim, mask=mask)
    return grid, prior, obs, members, shape


def test_letkf_equals_dense_kalman_gain_on_terrain_and_a_projection():
    """The same cross-check, on a grid where the two metrics disagree.

    The reference finds its neighbours by brute-force *physical* distance:
    a geodesic between the two mass points' own latitude/longitude, and a
    height difference between the two points' own columns.  Agreeing with
    it is the claim that the filter measures metres and not index counts.
    """
    fields = ("theta", "u")
    grid, prior, obs, members, shape = _terrain_case(fields=fields)
    loc = Localization(horizontal_m=4000.0, vertical_m=1500.0)
    cfg = LetkfConfig(localization=loc, analysis_fields=fields,
                      rtps_alpha=0.0)
    inc = analyze(prior, [obs], grid, cfg)

    dmean, var_a = _dense_reference(grid, prior, obs, loc, fields)
    moved = 0
    for f in fields:
        got_mean = inc[f].mean(axis=0)
        assert np.allclose(got_mean, dmean[f], rtol=1e-9, atol=1e-11), (
            f, np.abs(got_mean - dmean[f]).max())
        post = prior[f] + inc[f]
        assert np.allclose(post.var(axis=0, ddof=1), var_a[f],
                           rtol=1e-8, atol=1e-12)
        moved += int((np.abs(got_mean) > 1e-9).sum())
    assert moved > 0, "the reference and the filter agreed on doing nothing"


def test_vertical_localisation_uses_the_point_s_own_column():
    """Same model level, different altitude, is not zero separation.

    A single representative height column makes every same-level pair
    ``dv = 0`` and hands it a vertical weight of exactly 1, however much
    terrain lies between them.  With the column-dependent field the pair
    2000 m apart under a 4000 m cutoff gets Gaspari-Cohn's 5/24, and the
    difference is a factor of nearly five in that observation's influence.

    Measured through the filter rather than through the weight function:
    one observation, two analysis points at the same level and the same
    horizontal distance from it, differing only in their column's terrain.
    """
    members, nz, ny, nx = 10, 2, 3, 5
    rng = np.random.default_rng(5)
    # Flat except for column i = 4, which is lifted 2000 m.
    heights = np.zeros((nz, ny, nx))
    heights[0] = 500.0
    heights[1] = 1500.0
    heights[:, :, 4] += 2000.0
    grid = GridGeometry(dx_m=1000.0, dy_m=1000.0, heights_m=heights)

    fields = ("theta",)
    prior = {"theta": rng.standard_normal((members, nz, ny, nx)) + 3.0}
    mask = np.zeros((nz, ny, nx), dtype=bool)
    mask[0, 1, 2] = True                      # the observation, on the flat
    sim = prior["theta"] * 1.4
    values = np.where(mask, 20.0, np.nan)
    obs = GriddedObs(name="one", values=values, errors=0.5, simulated=sim,
                     mask=mask)
    loc = Localization(horizontal_m=9000.0, vertical_m=4000.0)
    inc = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0))

    # i = 0 and i = 4 are both two columns from the observation at i = 2, so
    # the horizontal weight is identical and only the vertical differs.
    flat_pt = np.abs(inc["theta"][:, 0, 1, 0]).max()
    lifted_pt = np.abs(inc["theta"][:, 0, 1, 4]).max()
    assert flat_pt > 1e-6
    assert lifted_pt > 0.0
    # 5/24 against 1 in the weight is far more than a rounding difference.
    assert lifted_pt < 0.5 * flat_pt, (flat_pt, lifted_pt)


def test_horizontal_localisation_is_a_geodesic_when_geolocated():
    """A radius in metres must reach the columns that are within it.

    On a projected grid the physical spacing is ``dx / mapfac``, so a
    nominal spacing understates or overstates the reach of a fixed radius
    depending on where the domain sits relative to the standard parallels.
    Here the true east-west spacing is about 1.7 km while ``dx_m`` says
    1 km, so an index-space metric reaches nearly twice as far as it should.
    """
    members, nz, ny, nx = 8, 1, 5, 11
    rng = np.random.default_rng(23)
    lat = 40.0 + np.arange(ny)[:, None] * (1000.0 / 111195.0) * np.ones(
        (1, nx))
    lon = -100.0 + np.arange(nx)[None, :] * 0.02 * np.ones((ny, 1))
    true_dx = float(_haversine_m(lat[0, 0], lon[0, 0], lat[0, 1], lon[0, 1],
                                 6370000.0))
    true_dy = float(_haversine_m(lat[0, 0], lon[0, 0], lat[1, 0], lon[1, 0],
                                 6370000.0))
    assert 1.6e3 < true_dx < 1.8e3, true_dx        # dx_m claims 1000
    assert 0.95e3 < true_dy < 1.05e3, true_dy      # dy_m is honest

    prior = {"theta": rng.standard_normal((members, nz, ny, nx)) + 3.0}
    mask = np.zeros((nz, ny, nx), dtype=bool)
    oj, oi = 2, 5
    mask[0, oj, oi] = True
    sim = prior["theta"] * 1.4
    values = np.where(mask, 20.0, np.nan)
    obs = GriddedObs(name="one", values=values, errors=0.5, simulated=sim,
                     mask=mask)
    loc = Localization(horizontal_m=5000.0, vertical_m=4000.0)
    cfg = LetkfConfig(localization=loc, analysis_fields=("theta",),
                      rtps_alpha=0.0)

    geo = GridGeometry(dx_m=1000.0, dy_m=1000.0,
                       heights_m=np.array([500.0]), lat_deg=lat, lon_deg=lon)
    d_geo = LetkfDiagnostics()
    analyze(prior, [obs], geo, cfg, diagnostics=d_geo)

    nominal = GridGeometry(dx_m=1000.0, dy_m=1000.0,
                           heights_m=np.array([500.0]))
    d_nom = LetkfDiagnostics()
    analyze(prior, [obs], nominal, cfg, diagnostics=d_nom)

    # Brute force, from the coordinates themselves: how many gridpoints are
    # genuinely within 5 km of the observation, and how many an index-space
    # metric believes are.
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    want_geo = int((_haversine_m(lat[oj, oi], lon[oj, oi], lat, lon,
                                 6370000.0) < loc.horizontal_m).sum())
    want_nom = int((np.hypot((ii - oi) * 1000.0, (jj - oj) * 1000.0)
                    < loc.horizontal_m).sum())
    assert d_geo.active_points == want_geo
    assert d_nom.active_points == want_nom
    assert want_geo < want_nom, (want_geo, want_nom)


def test_geometry_refuses_half_a_geolocation():
    with pytest.raises(LetkfError, match="supplied together"):
        GridGeometry(dx_m=1000.0, dy_m=1000.0,
                     heights_m=np.array([100.0, 400.0]),
                     lat_deg=np.zeros((3, 3)))


def test_geometry_refuses_a_non_monotone_column_in_a_3d_height_field():
    z = np.zeros((3, 2, 2))
    z[:, 0, 0] = [100.0, 400.0, 900.0]
    z[:, 0, 1] = [100.0, 400.0, 900.0]
    z[:, 1, 0] = [100.0, 400.0, 900.0]
    z[:, 1, 1] = [100.0, 900.0, 400.0]      # one bad column is enough
    with pytest.raises(LetkfError, match="strictly increasing"):
        GridGeometry(dx_m=1000.0, dy_m=1000.0, heights_m=z)


def test_analysis_perturbations_keep_zero_ensemble_mean():
    """The symmetric square root must not shift the mean it is applied to.

    ``Wa 1 = sqrt(rho) 1`` because the all-ones vector is an eigenvector of
    ``C Yb`` with eigenvalue zero (``Yb 1 = 0``), so the analysis
    perturbations sum to zero over members.  An asymmetric square root --
    for instance one taken as ``U D^-1/2`` without the trailing ``U^T`` --
    breaks this, gives a biased analysis, and still reduces RMSE at first.
    """
    fields = ("theta", "qv")
    grid, prior, obs, members, shape = _tiny_case(fields=fields)
    cfg = LetkfConfig(
        localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
        analysis_fields=fields, rtps_alpha=0.0)
    inc = analyze(prior, [obs], grid, cfg)
    for f in fields:
        post = prior[f] + inc[f]
        pert = post - post.mean(axis=0, keepdims=True)
        assert np.abs(pert.sum(axis=0)).max() < 1e-10


# ---------------------------------------------------------------------------
# Localisation is structural, not numerical
# ---------------------------------------------------------------------------

def test_increment_is_exactly_zero_beyond_the_cutoff():
    """One observation, and a bit-exact zero everywhere outside its lens.

    ``== 0`` deliberately, not ``< tol``.  A filter whose remote increments
    are 1e-16 has O(1) spurious long-range covariance that only looks small
    because the innovation happened to be small; the guarantee worth having
    is that the observation was never in that gridpoint's batch at all.
    """
    members, nz, ny, nx = 10, 4, 15, 15
    rng = np.random.default_rng(11)
    grid = GridGeometry(dx_m=1000.0, dy_m=1000.0,
                        heights_m=np.array([200.0, 700.0, 1400.0, 2400.0]))
    fields = ("theta", "qv")
    prior = {f: rng.standard_normal((members, nz, ny, nx)) + 5.0
             for f in fields}
    mask = np.zeros((nz, ny, nx), dtype=bool)
    ok, oj, oi = 1, 7, 7
    mask[ok, oj, oi] = True
    sim = prior["theta"] * 1.5
    values = np.where(mask, 99.0, np.nan)      # a huge innovation on purpose
    obs = GriddedObs(name="one", values=values, errors=0.5, simulated=sim,
                     mask=mask)
    loc = Localization(horizontal_m=3500.0, vertical_m=900.0)
    diag = LetkfDiagnostics()
    inc = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0),
        diagnostics=diag)

    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    dh = np.hypot((xx - oi) * grid.dx_m, (yy - oj) * grid.dy_m)
    dv = np.abs(grid.heights_m[zz] - grid.heights_m[ok])
    reach = (np.asarray(gaspari_cohn(dh, loc.horizontal_m))
             * np.asarray(gaspari_cohn(dv, loc.vertical_m))) > 0

    for f in fields:
        outside = inc[f][:, ~reach]
        assert np.all(outside == 0.0), (
            f, "nonzero increment outside the cutoff",
            np.abs(outside).max())
        # Guard the guard: if nothing moved anywhere, "zero outside" is
        # vacuous and this test would pass on a filter that did nothing.
        assert np.abs(inc[f][:, reach]).max() > 1e-6

    assert diag.active_points == int(reach.sum())
    assert diag.active_points < diag.total_points


def test_no_observations_is_an_exact_no_op():
    """The no-DA control: no obs, and no obs that pass the mask, both zero."""
    fields = ("theta", "qv", "u")
    grid, prior, obs, members, shape = _tiny_case(fields=fields)
    cfg = LetkfConfig(
        localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
        analysis_fields=fields, rtps_alpha=0.0)

    empty = analyze(prior, [], grid, cfg)
    for f in fields:
        assert np.all(empty[f] == 0.0)

    all_masked = GriddedObs(
        name=obs.name, values=obs.values, errors=obs.errors,
        simulated=obs.simulated, mask=np.zeros(shape, dtype=bool))
    diag = LetkfDiagnostics()
    none = analyze(prior, [all_masked], grid, cfg, diagnostics=diag)
    for f in fields:
        assert np.all(none[f] == 0.0)
    assert diag.active_points == 0


def test_chunking_does_not_change_the_answer():
    """chunk_points is a memory knob; it must not be an accuracy knob."""
    fields = ("theta", "u")
    grid, prior, obs, members, shape = _tiny_case(fields=fields, nx=8, ny=8,
                                                  n_obs=12)
    loc = Localization(horizontal_m=3000.0, vertical_m=1500.0)
    whole = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        chunk_points=10 ** 6))
    diced = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        chunk_points=7))
    for f in fields:
        assert np.array_equal(whole[f], diced[f])


def test_chunk_is_auto_sized_from_the_memory_budget():
    """The default chunk must follow the stencil, not a fixed number.

    A fixed default is a footgun: P is not known until the localisation
    radii meet the grid spacing, and an ordinary 8 km radius on a 2 km grid
    at R = 30 gives P = 405, which turns an 8192-point chunk into hundreds
    of megabytes per array on a card that is very likely shared.  What must
    hold is that the budget moves the chunk and the chunk does not move the
    answer.
    """
    fields = ("theta", "u")
    grid, prior, obs, members, shape = _tiny_case(fields=fields, nx=8, ny=8,
                                                  n_obs=12)
    loc = Localization(horizontal_m=3000.0, vertical_m=1500.0)

    seen = {}
    for budget in (2.0, 8.0, 4096.0):
        d = LetkfDiagnostics()
        inc = analyze(prior, [obs], grid, LetkfConfig(
            localization=loc, analysis_fields=fields, rtps_alpha=0.0,
            memory_budget_mib=budget), diagnostics=d)
        assert d.chunk_points >= 1
        # The budget is a ceiling now, not a hope: the chunk it chose,
        # times the model's per-point price, fits under it.
        assert d.chunk_points * d.solve_bytes_per_point \
            <= int(budget * (1 << 20))
        seen[budget] = (d.chunk_points, inc)

    assert seen[2.0][0] < seen[8.0][0] < seen[4096.0][0]
    # A budget far larger than the domain caps at the domain, not above it.
    assert seen[4096.0][0] <= shape[0] * shape[1] * shape[2]
    # A budget below one gridpoint's price is a refusal with the remedy in
    # it, not a chunk of 1 that ignores the figure it was given.  (That
    # "keep going anyway" behaviour is exactly what the field OOM of
    # 2026-08-05 grew from; tests/test_letkf_chunk_sizing.py pins the rest.)
    with pytest.raises(LetkfError, match="memory_budget_mib"):
        analyze(prior, [obs], grid, LetkfConfig(
            localization=loc, analysis_fields=fields, rtps_alpha=0.0,
            memory_budget_mib=0.01))
    for f in fields:
        base = seen[2.0][1][f]
        for budget in (8.0, 4096.0):
            assert np.array_equal(seen[budget][1][f], base), (f, budget)


def test_two_observation_types_with_different_radii():
    """Per-type localisation, and the multi-stencil concatenation it forces.

    The wider type must reach gridpoints the narrower one cannot, so the
    union of the two influence regions is strictly larger than either.
    """
    members, nz, ny, nx = 8, 3, 13, 13
    rng = np.random.default_rng(3)
    grid = GridGeometry(dx_m=1000.0, dy_m=1000.0,
                        heights_m=np.array([300.0, 900.0, 1800.0]))
    prior = {"theta": rng.standard_normal((members, nz, ny, nx))}
    mask = np.zeros((nz, ny, nx), dtype=bool)
    mask[1, 6, 6] = True
    sim = prior["theta"] * 2.0
    values = np.where(mask, 12.0, np.nan)
    narrow = Localization(horizontal_m=2500.0, vertical_m=1000.0)
    wide = Localization(horizontal_m=5500.0, vertical_m=1000.0)
    cfg = LetkfConfig(localization=narrow, analysis_fields=("theta",),
                      rtps_alpha=0.0)

    def reach(loc_override):
        o = GriddedObs(name="o", values=values, errors=0.5, simulated=sim,
                       mask=mask, localization=loc_override)
        d = LetkfDiagnostics()
        analyze(prior, [o], grid, cfg, diagnostics=d)
        return d.active_points

    n_narrow = reach(None)
    n_wide = reach(wide)
    assert n_wide > n_narrow > 0

    both = [
        GriddedObs(name="n", values=values, errors=0.5, simulated=sim,
                   mask=mask, localization=None),
        GriddedObs(name="w", values=values, errors=0.5, simulated=sim,
                   mask=mask, localization=wide),
    ]
    d = LetkfDiagnostics()
    inc = analyze(prior, both, grid, cfg, diagnostics=d)
    assert d.active_points == n_wide
    assert np.isfinite(inc["theta"]).all()


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------

def test_rtps_alpha_one_restores_the_prior_spread_exactly():
    """alpha = 1 means "relax all the way back", so sigma_a == sigma_b.

    The mean must still move: RTPS relaxes the spread, not the analysis.
    """
    fields = ("theta", "qv")
    grid, prior, obs, members, shape = _tiny_case(fields=fields, n_obs=14)
    loc = Localization(horizontal_m=4000.0, vertical_m=2000.0)
    base = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0))
    full = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=1.0))
    for f in fields:
        sb = prior[f].std(axis=0, ddof=1)
        sa0 = (prior[f] + base[f]).std(axis=0, ddof=1)
        sa1 = (prior[f] + full[f]).std(axis=0, ddof=1)
        assert np.allclose(sa1, sb, rtol=1e-10, atol=1e-12)
        # The analysis actually contracted the spread somewhere, so the
        # restoration is doing work rather than being trivially satisfied.
        assert sa0.min() < sb.max() * 0.999
        # And the mean increment is untouched by the relaxation.
        assert np.allclose(base[f].mean(axis=0), full[f].mean(axis=0),
                           rtol=1e-10, atol=1e-12)


def test_rtps_is_monotone_in_alpha():
    """Every intermediate alpha lands between the two endpoints."""
    fields = ("theta",)
    grid, prior, obs, members, shape = _tiny_case(fields=fields, n_obs=14)
    loc = Localization(horizontal_m=4000.0, vertical_m=2000.0)
    spreads = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        inc = analyze(prior, [obs], grid, LetkfConfig(
            localization=loc, analysis_fields=fields, rtps_alpha=a))
        spreads.append(float((prior["theta"] + inc["theta"]).std(
            axis=0, ddof=1).mean()))
    assert all(b >= a - 1e-12 for a, b in zip(spreads, spreads[1:])), spreads
    assert spreads[-1] > spreads[0]


def test_rtps_alpha_must_be_chosen_and_lands_in_the_diagnostics():
    """No default, because there is no defensible one, and it is recorded.

    ``alpha = 0`` is a bitwise identity: an analysis run with no posterior
    relaxation is indistinguishable from a well-tuned one for a single
    cycle, and only stops working several cycles later, as spread the
    ensemble never got back.  A caller has to say which they meant, and the
    log has to be able to say which they said -- so both inflation settings
    go into the diagnostics beside the spreads they explain.
    """
    with pytest.raises(TypeError, match="rtps_alpha"):
        LetkfConfig(
            localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
            analysis_fields=("theta",))

    fields = ("theta",)
    grid, prior, obs, members, shape = _tiny_case(fields=fields, n_obs=14)
    diag = LetkfDiagnostics()
    analyze(prior, [obs], grid, LetkfConfig(
        localization=Localization(horizontal_m=4000.0, vertical_m=2000.0),
        analysis_fields=fields, rtps_alpha=0.85, prior_inflation=1.1),
        diagnostics=diag)
    assert diag.rtps_alpha == 0.85
    assert diag.prior_inflation == 1.1


def test_prior_inflation_increases_the_increment():
    """rho > 1 trusts the observations more, so the analysis moves further."""
    fields = ("theta",)
    grid, prior, obs, members, shape = _tiny_case(fields=fields, n_obs=14)
    loc = Localization(horizontal_m=4000.0, vertical_m=2000.0)
    small = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, prior_inflation=1.0,
        rtps_alpha=0.0))
    big = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, prior_inflation=1.5,
        rtps_alpha=0.0))
    a = np.abs(small["theta"].mean(axis=0)).sum()
    b = np.abs(big["theta"].mean(axis=0)).sum()
    assert b > a


def _one_obs_case(members=12, nz=4, ny=15, nx=15, seed=11):
    """One observation in the middle, and a lot of grid outside its lens."""
    rng = np.random.default_rng(seed)
    grid = GridGeometry(dx_m=1000.0, dy_m=1000.0,
                        heights_m=np.array([200.0, 700.0, 1400.0, 2400.0]))
    fields = ("theta", "qv")
    prior = {f: rng.standard_normal((members, nz, ny, nx)) + 5.0
             for f in fields}
    mask = np.zeros((nz, ny, nx), dtype=bool)
    ok, oj, oi = 1, 7, 7
    mask[ok, oj, oi] = True
    sim = prior["theta"] * 1.5
    values = np.where(mask, 99.0, np.nan)
    obs = GriddedObs(name="one", values=values, errors=0.5, simulated=sim,
                     mask=mask)
    loc = Localization(horizontal_m=3500.0, vertical_m=900.0)
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    dh = np.hypot((xx - oi) * grid.dx_m, (yy - oj) * grid.dy_m)
    dv = np.abs(grid.heights_m[zz] - grid.heights_m[ok])
    reach = (np.asarray(gaspari_cohn(dh, loc.horizontal_m))
             * np.asarray(gaspari_cohn(dv, loc.vertical_m))) > 0
    return grid, prior, obs, loc, fields, reach, members


@pytest.mark.parametrize("rho", [1.05, 1.44])
def test_prior_inflation_reaches_points_with_no_local_observation(rho):
    """Hunt's rho is a property of the transform, not of the observations.

    With no observation rows, ``Yb`` is empty and the published recipe still
    gives ``A = (R-1)I/rho``, hence ``Pa~ = rho/(R-1) I``, ``Wa =
    sqrt(rho) I`` and ``wa_ = 0``.  So the analysis at an inactive gridpoint
    is ``xbar + sqrt(rho) x'``: the ensemble MEAN is untouched, the
    perturbations are stretched, and the posterior covariance is exactly
    ``rho Pb``.

    Skipping the solve there -- which is the right thing to do for the
    observation increment, since it is what makes remote increments bitwise
    zero -- must not also skip the inflation.  The two are separable and
    this test separates them: outside the lens the mean increment is still
    exactly zero, but the spread has grown by sqrt(rho).

    The failure this catches is spatially discontinuous, which is what makes
    it nasty: as a Gaspari-Cohn weight goes to zero from inside the cutoff
    Wa tends to sqrt(rho) I, and at the cutoff itself the observation is
    dropped and the old code jumped to Wa = I.
    """
    grid, prior, obs, loc, fields, reach, members = _one_obs_case()
    inc = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, prior_inflation=rho,
        rtps_alpha=0.0))

    root = math.sqrt(rho)
    for f in fields:
        xb = prior[f] - prior[f].mean(axis=0, keepdims=True)
        want = (root - 1.0) * xb[:, ~reach]
        got = inc[f][:, ~reach]
        assert np.allclose(got, want, rtol=1e-12, atol=1e-13), (
            f, np.abs(got - want).max())
        # The mean is untouched: this is a perturbation-only transform.
        assert np.abs(got.mean(axis=0)).max() < 1e-12
        # And the posterior covariance is rho times the prior, exactly.
        post = (prior[f] + inc[f])[:, ~reach]
        assert np.allclose(post.var(axis=0, ddof=1),
                           rho * prior[f][:, ~reach].var(axis=0, ddof=1),
                           rtol=1e-12, atol=0.0)
        # Guard the guard: the observed lens moved for a different reason.
        assert np.abs(inc[f][:, reach]).max() > 1e-6


def test_rho_one_still_gives_bitwise_zero_outside_the_lens():
    """The structural guarantee survives the inflation fix.

    ``rho = 1`` makes ``sqrt(rho) = 1`` exactly in binary floating point, so
    the inactive-point transform is the identity and the preallocated zero
    increment is still literally the answer -- no ``1 - 1`` residue, no
    ``eigh`` of a scaled identity, no denormal influence at unbounded range.
    """
    grid, prior, obs, loc, fields, reach, members = _one_obs_case()
    inc = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, prior_inflation=1.0,
        rtps_alpha=0.0))
    for f in fields:
        assert np.all(inc[f][:, ~reach] == 0.0)


def test_rtps_relaxes_the_inflated_inactive_point_too():
    """RTPS is applied to the inactive-point transform, not bypassed by it.

    At an inactive gridpoint the analysis spread is ``sqrt(rho) sigma_b``,
    so Whitaker and Hamill's relaxation gives
    ``(1-alpha) sqrt(rho) sigma_b + alpha sigma_b``.  Writing the whole
    inactive-point scaling as ``s = (1-alpha) sqrt(rho) + alpha`` is what
    keeps ``rho = 1`` an exact identity for every alpha, and keeps
    ``alpha = 1`` a return to the prior spread for every rho -- the same two
    endpoint properties the active branch has.
    """
    grid, prior, obs, loc, fields, reach, members = _one_obs_case()
    rho, alpha = 1.44, 0.25
    inc = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, prior_inflation=rho,
        rtps_alpha=alpha))
    s = (1.0 - alpha) * math.sqrt(rho) + alpha
    for f in fields:
        xb = prior[f] - prior[f].mean(axis=0, keepdims=True)
        got = inc[f][:, ~reach]
        assert np.allclose(got, (s - 1.0) * xb[:, ~reach],
                           rtol=1e-12, atol=1e-13)

    # alpha = 1 restores the prior spread exactly, inflated or not.
    full = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, prior_inflation=rho,
        rtps_alpha=1.0))
    for f in fields:
        assert np.all(full[f][:, ~reach] == 0.0)


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------

def _default_cfg(fields=("theta",)):
    return LetkfConfig(
        localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
        analysis_fields=fields, rtps_alpha=0.0)


def test_zero_spread_ensemble_is_an_error_not_a_nan():
    """Identical members have no covariance.  Say so; do not return NaN.

    Note what this does NOT do: assert the spread is exactly zero.  The mean
    of R identical floats is not bitwise equal to them, so a constant
    ensemble reaches the filter with a spread near 1e-16 and an exact
    ``== 0`` guard misses it entirely -- which is how the first version of
    this module let this case through.
    """
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    prior["theta"] = np.broadcast_to(
        prior["theta"][0], prior["theta"].shape).copy()
    assert prior["theta"].std(axis=0, ddof=1).max() >= 0.0
    with pytest.raises(LetkfError, match="no usable ensemble spread"):
        analyze(prior, [obs], grid, _default_cfg())


def test_identically_zero_field_is_an_error():
    """Zero magnitude and zero spread: the relative guard must still fire."""
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    prior["theta"] = np.zeros_like(prior["theta"])
    with pytest.raises(LetkfError, match="no usable ensemble spread"):
        analyze(prior, [obs], grid, _default_cfg())


def test_single_member_is_refused():
    grid, prior, obs, members, shape = _tiny_case(members=1,
                                                  fields=("theta",))
    with pytest.raises(LetkfError, match="at least 2 ensemble members"):
        analyze(prior, [obs], grid, _default_cfg())


def test_zero_observation_error_is_refused():
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    bad = GriddedObs(name=obs.name, values=obs.values, errors=0.0,
                     simulated=obs.simulated, mask=obs.mask)
    with pytest.raises(LetkfError, match="standard deviation"):
        analyze(prior, [bad], grid, _default_cfg())


def test_nonfinite_observation_where_unmasked_is_refused():
    """A NaN under a True mask is a producer bug; do not average over it."""
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    vals = np.asarray(obs.values).copy()
    k, j, i = np.argwhere(np.asarray(obs.mask))[0]
    vals[k, j, i] = np.nan
    bad = GriddedObs(name=obs.name, values=vals, errors=obs.errors,
                     simulated=obs.simulated, mask=obs.mask)
    with pytest.raises(LetkfError, match="non-finite values where mask"):
        analyze(prior, [bad], grid, _default_cfg())


def test_nonfinite_forward_operator_is_refused():
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    sim = np.asarray(obs.simulated).copy()
    k, j, i = np.argwhere(np.asarray(obs.mask))[0]
    sim[0, k, j, i] = np.inf
    bad = GriddedObs(name=obs.name, values=obs.values, errors=obs.errors,
                     simulated=sim, mask=obs.mask)
    with pytest.raises(LetkfError, match="H\\(x_k\\) is non-finite"):
        analyze(prior, [bad], grid, _default_cfg())


def test_nonfinite_prior_is_refused():
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    prior["theta"][2, 0, 0, 0] = np.nan
    with pytest.raises(LetkfError, match="non-finite"):
        analyze(prior, [obs], grid, _default_cfg())


def test_missing_analysis_field_is_refused():
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    with pytest.raises(LetkfError, match="not present in the prior"):
        analyze(prior, [obs], grid, _default_cfg(("theta", "qsnow")))


def test_level_count_disagreement_is_refused():
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    short = GridGeometry(dx_m=grid.dx_m, dy_m=grid.dy_m,
                         heights_m=grid.heights_m[:2])
    with pytest.raises(LetkfError, match="levels"):
        analyze(prior, [obs], short, _default_cfg())


def test_simulated_shape_disagreement_is_refused():
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    bad = GriddedObs(name=obs.name, values=obs.values, errors=obs.errors,
                     simulated=np.asarray(obs.simulated)[:-1], mask=obs.mask)
    with pytest.raises(LetkfError, match="simulated has shape"):
        analyze(prior, [bad], grid, _default_cfg())


@pytest.mark.parametrize("kwargs,match", [
    ({"analysis_fields": ()}, "empty"),
    ({"analysis_fields": ("a", "a")}, "duplicates"),
    ({"rtps_alpha": 1.5}, "rtps_alpha"),
    ({"rtps_alpha": -0.1}, "rtps_alpha"),
    ({"prior_inflation": 0.0}, "rho"),
    ({"chunk_points": 0}, "chunk_points"),
    ({"solve_dtype": "float16"}, "solve_dtype"),
])
def test_config_refuses_nonsense(kwargs, match):
    base = dict(
        localization=Localization(horizontal_m=1000.0, vertical_m=500.0),
        analysis_fields=("theta",), rtps_alpha=0.0)
    base.update(kwargs)
    with pytest.raises(LetkfError, match=match):
        LetkfConfig(**base)


@pytest.mark.parametrize("kwargs", [
    {"horizontal_m": 0.0, "vertical_m": 500.0},
    {"horizontal_m": -1.0, "vertical_m": 500.0},
    {"horizontal_m": 1000.0, "vertical_m": math.inf},
])
def test_localization_refuses_nonsense(kwargs):
    with pytest.raises(LetkfError):
        Localization(**kwargs)


def test_grid_refuses_non_monotone_heights():
    with pytest.raises(LetkfError, match="strictly increasing"):
        GridGeometry(dx_m=1000.0, dy_m=1000.0,
                     heights_m=np.array([100.0, 400.0, 300.0]))


def test_float32_state_still_squares_the_error_in_the_solve_dtype():
    """The state's precision must not become the solve's precision.

    ``solve_dtype`` is float64 by default and deliberately, but the
    observation error was squared in the *input* dtype before the
    promotion, so a float32 state silently did that one step of the
    transform in float32.  Squaring in ``solve_dtype`` makes the float32
    state's analysis agree with the float64 state's to float32 rounding,
    which is the whole claim of "the gather stays in the input dtype, the
    R x R solve does not".
    """
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    small_err = 2.0e-4          # fine in float64, its square is not float32
    ref_obs = GriddedObs(name=obs.name, values=obs.values, errors=small_err,
                         simulated=obs.simulated, mask=obs.mask)
    ref = analyze(prior, [ref_obs], grid, _default_cfg())

    prior32 = {f: v.astype(np.float32) for f, v in prior.items()}
    obs32 = GriddedObs(
        name=obs.name, values=np.asarray(obs.values, dtype=np.float32),
        errors=np.float32(small_err),
        simulated=np.asarray(obs.simulated, dtype=np.float32),
        mask=obs.mask)
    got = analyze(prior32, [obs32], grid, _default_cfg())
    assert np.all(np.isfinite(got["theta"]))
    assert np.abs(got["theta"]).max() > 1e-6
    assert np.allclose(got["theta"], ref["theta"], rtol=2e-4, atol=2e-5), (
        np.abs(got["theta"] - ref["theta"].astype(np.float32)).max())


def test_underflowing_error_square_fails_closed_with_a_diagnostic():
    """The extreme case must fail loudly, and never through a raw divide.

    A float32 sigma of 1e-25 is perfectly representable; 1e-50 is not.
    Squaring in the input dtype turned ``lambda/sigma^2`` into a division
    by zero, and the transform met an infinity it had no guard for.  It
    should be said plainly that no float64 solve is well conditioned at
    that sigma either -- 1/sigma^2 dwarfs ``(R-1)/rho`` by 1e50 and the
    small eigenvalues are lost in the rounding of the large ones.  The fix
    is not that this case starts working; it is that it fails through a
    named guard instead of an arithmetic accident.

    ``np.errstate(divide="raise")`` is the regression: before, the
    division fired first and this raised ``FloatingPointError``.
    """
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    prior = {f: v.astype(np.float32) for f, v in prior.items()}
    tiny = GriddedObs(
        name=obs.name, values=np.asarray(obs.values, dtype=np.float32),
        errors=np.float32(1e-25),
        simulated=np.asarray(obs.simulated, dtype=np.float32),
        mask=obs.mask)
    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(LetkfError, match="observation error"):
            analyze(prior, [tiny], grid, _default_cfg())


def test_float32_solve_reports_an_underflowed_error_variance_as_letkf_error():
    """Asked for a float32 solve, the square really can underflow.

    The module's contract is that every degenerate input raises
    :class:`LetkfError` with a diagnostic.  A raw ``LinAlgError`` from
    ``eigh``, several hundred lines downstream of the actual cause, is not
    that.
    """
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    tiny = GriddedObs(name=obs.name, values=obs.values,
                      errors=np.float32(1e-25), simulated=obs.simulated,
                      mask=obs.mask)
    cfg = LetkfConfig(
        localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
        analysis_fields=("theta",), rtps_alpha=0.0, solve_dtype="float32")
    with pytest.raises(LetkfError, match="underflow"):
        analyze(prior, [tiny], grid, cfg)


def test_a_backend_eigensolver_failure_is_reported_as_letkf_error(monkeypatch):
    """Fail closed on the one exception path the module cannot pre-empt.

    ``eigh`` can fail to converge for reasons no input check anticipates.
    The caller is promised a ``LetkfError`` naming the analysis that did not
    happen, not a bare LAPACK convergence message.
    """
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))

    def boom(_a):
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    monkeypatch.setattr(np.linalg, "eigh", boom)
    with pytest.raises(LetkfError, match="eigensolver"):
        analyze(prior, [obs], grid, _default_cfg())


def test_localization_narrower_than_a_grid_cell_is_refused():
    """The kilometres-for-metres slip, caught instead of silently obeyed.

    Such a radius yields a stencil holding only the zero offset.  That is
    well defined -- co-located observations and nothing else -- so nothing
    downstream fails; the filter runs, reports increments, and spreads no
    information whatsoever.  Refusing is the only way this is visible.
    """
    grid, prior, obs, members, shape = _tiny_case(fields=("theta",))
    cfg = LetkfConfig(
        localization=Localization(horizontal_m=10.0, vertical_m=10.0),
        analysis_fields=("theta",), rtps_alpha=0.0)
    with pytest.raises(LetkfError, match="does not reach"):
        analyze(prior, [obs], grid, cfg)


# ---------------------------------------------------------------------------
# The GPU path
# ---------------------------------------------------------------------------

def test_batched_cupy_solve_matches_numpy():
    """Same analysis on the device, to float32-of-a-float64-solve tolerance.

    This is the claim the whole design rests on: thousands of R x R
    eigendecompositions issued as one ``cupy.linalg.eigh`` produce the same
    analysis as NumPy's, so the batching is a performance transformation and
    not a numerical one.  ``eigh`` is not bitwise reproducible across
    LAPACK and cuSOLVER -- eigenvectors of near-degenerate eigenvalues are
    only defined up to rotation within their eigenspace -- but the
    SYMMETRIC square root and Pa~ are invariant under that rotation, so the
    increments must agree to ordinary rounding.
    """
    cp = pytest.importorskip("cupy")
    try:
        # cupy's core (elementwise kernels, reductions) loads independently
        # of cuBLAS and cuSOLVER, so "cupy imports" does not imply "cupy can
        # do linear algebra".  A CUDA 12 cupy against a CUDA 13-only toolkit
        # gets all the way to here and then fails on cublas64_12.dll.  Probe
        # the operation actually under test rather than the import.
        cp.linalg.eigh(cp.eye(3, dtype=cp.float64)[None] * 2.0)
    except Exception as exc:                      # pragma: no cover - env
        pytest.skip(f"cupy is present but its linear algebra is not: {exc}")

    fields = ("theta", "qv", "u")
    grid, prior, obs, members, shape = _tiny_case(
        members=16, nz=4, ny=20, nx=20, n_obs=40, fields=fields)
    loc = Localization(horizontal_m=3500.0, vertical_m=1500.0)
    cfg = LetkfConfig(localization=loc, analysis_fields=fields,
                      rtps_alpha=0.6, chunk_points=512)

    host = analyze(prior, [obs], grid, cfg)

    dev_prior = {f: cp.asarray(v) for f, v in prior.items()}
    dev_obs = GriddedObs(
        name=obs.name, values=cp.asarray(obs.values), errors=obs.errors,
        simulated=cp.asarray(obs.simulated), mask=cp.asarray(obs.mask))
    dev = analyze(dev_prior, [dev_obs], grid, cfg)

    for f in fields:
        g = cp.asnumpy(dev[f])
        scale = max(float(np.abs(host[f]).max()), 1e-12)
        assert np.allclose(g, host[f], rtol=1e-8, atol=1e-10 * scale), (
            f, float(np.abs(g - host[f]).max()))
        # The structural zero must survive the device path unchanged.
        assert np.array_equal(g == 0.0, host[f] == 0.0)


# ---------------------------------------------------------------------------
# RTPP, and the claim that both relaxations leave the inactive point alone
# ---------------------------------------------------------------------------


def test_rtpp_alpha_one_restores_the_prior_perturbations_exactly():
    """alpha = 1 under RTPP puts back the prior PERTURBATION itself.

    Stronger than RTPS's alpha = 1, which only restores the prior
    spread: RTPP restores each member's own deviation, so the analysis
    is the prior ensemble rigidly shifted by the mean increment.  That
    is the difference between relaxing an amplitude and relaxing a
    covariance structure, and it is why the two knobs are not
    interchangeable at the same alpha.
    """
    fields = ("theta", "qv")
    grid, prior, obs, members, shape = _tiny_case(fields=fields, n_obs=14)
    loc = Localization(horizontal_m=4000.0, vertical_m=2000.0)
    base = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0))
    full = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=1.0,
        relaxation="rtpp"))
    for f in fields:
        analysis = prior[f] + full[f]
        deviation = analysis - analysis.mean(axis=0, keepdims=True)
        prior_dev = prior[f] - prior[f].mean(axis=0, keepdims=True)
        assert np.allclose(deviation, prior_dev, rtol=1e-10, atol=1e-12)
        # The mean still moved, and by exactly what the unrelaxed
        # analysis moved it by: neither relaxation touches wbar.
        assert np.allclose(base[f].mean(axis=0), full[f].mean(axis=0),
                           rtol=1e-10, atol=1e-12)
        assert np.abs(full[f].mean(axis=0)).max() > 1e-6


def test_rtpp_and_rtps_differ_at_the_same_alpha():
    fields = ("theta",)
    grid, prior, obs, members, shape = _tiny_case(fields=fields, n_obs=14)
    loc = Localization(horizontal_m=4000.0, vertical_m=2000.0)
    common = dict(localization=loc, analysis_fields=fields, rtps_alpha=0.7)
    rtps = analyze(prior, [obs], grid, LetkfConfig(**common))
    rtpp = analyze(prior, [obs], grid,
                   LetkfConfig(relaxation="rtpp", **common))
    assert not np.allclose(rtps["theta"], rtpp["theta"], atol=1e-10)
    # Both are relaxations toward the same prior, so both sit between the
    # unrelaxed analysis spread and the prior spread.
    unrelaxed = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0))
    sb = prior["theta"].std(axis=0, ddof=1)
    s0 = (prior["theta"] + unrelaxed["theta"]).std(axis=0, ddof=1)
    for inc in (rtps["theta"], rtpp["theta"]):
        s = (prior["theta"] + inc).std(axis=0, ddof=1)
        assert np.all(s >= np.minimum(s0, sb) - 1e-12)
        assert np.all(s <= np.maximum(s0, sb) + 1e-12)


def test_rtpp_leaves_the_inactive_point_exactly_where_rtps_does():
    """The docstring's claim, pinned rather than asserted in prose.

    At a gridpoint with no localised observation ``Xa = sqrt(rho) Xb``,
    so RTPS gives ``[(1-a) sqrt(rho) + a] Xb`` and RTPP gives
    ``(1-a) sqrt(rho) Xb + a Xb`` -- the same scalar.  If that ever
    stopped being true, the bitwise-zero-beyond-the-cutoff guarantee
    would hold for one relaxation and not the other, and the
    localisation gate would be testing only whichever one it happened to
    be configured with.
    """
    members, nz, ny, nx = 10, 4, 15, 15
    rng = np.random.default_rng(11)
    grid = GridGeometry(dx_m=1000.0, dy_m=1000.0,
                        heights_m=np.array([200.0, 700.0, 1400.0, 2400.0]))
    fields = ("theta", "qv")
    prior = {f: rng.standard_normal((members, nz, ny, nx)) + 5.0
             for f in fields}
    mask = np.zeros((nz, ny, nx), dtype=bool)
    ok, oj, oi = 1, 7, 7
    mask[ok, oj, oi] = True
    obs = GriddedObs(name="one", values=np.where(mask, 99.0, np.nan),
                     errors=0.5, simulated=prior["theta"] * 1.5, mask=mask)
    loc = Localization(horizontal_m=3500.0, vertical_m=900.0)

    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    dh = np.hypot((xx - oi) * grid.dx_m, (yy - oj) * grid.dy_m)
    dv = np.abs(grid.heights_m[zz] - grid.heights_m[ok])
    reach = (np.asarray(gaspari_cohn(dh, loc.horizontal_m))
             * np.asarray(gaspari_cohn(dv, loc.vertical_m))) > 0

    # rho = 1: both must be BITWISE zero outside the lens.
    for mode in ("rtps", "rtpp"):
        inc = analyze(prior, [obs], grid, LetkfConfig(
            localization=loc, analysis_fields=fields, rtps_alpha=0.9,
            relaxation=mode))
        for f in fields:
            assert np.all(inc[f][:, ~reach] == 0.0), (mode, f)
            assert np.abs(inc[f][:, reach]).max() > 1e-6, (mode, f)

    # rho != 1: not zero any more, but identical between the two modes,
    # and equal to the closed form the module documents.
    rho, alpha = 1.44, 0.9
    scale = (1.0 - alpha) * math.sqrt(rho) + alpha
    out = {}
    for mode in ("rtps", "rtpp"):
        out[mode] = analyze(prior, [obs], grid, LetkfConfig(
            localization=loc, analysis_fields=fields, rtps_alpha=alpha,
            prior_inflation=rho, relaxation=mode))
    for f in fields:
        expected = (prior[f] - prior[f].mean(axis=0, keepdims=True)) * (
            scale - 1.0)
        for mode in ("rtps", "rtpp"):
            assert np.allclose(out[mode][f][:, ~reach],
                               expected[:, ~reach], rtol=1e-12, atol=1e-14)
        assert np.allclose(out["rtps"][f][:, ~reach],
                           out["rtpp"][f][:, ~reach], rtol=0.0, atol=0.0)


def test_relaxation_mode_is_validated_and_recorded():
    fields = ("theta",)
    grid, prior, obs, members, shape = _tiny_case(fields=fields, n_obs=6)
    loc = Localization(horizontal_m=4000.0, vertical_m=2000.0)
    with pytest.raises(LetkfError, match="relaxation must be"):
        LetkfConfig(localization=loc, analysis_fields=fields,
                    rtps_alpha=0.5, relaxation="rtpq")
    diag = LetkfDiagnostics()
    analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.5,
        relaxation="rtpp"), diagnostics=diag)
    assert diag.relaxation == "rtpp"
    assert diag.rtps_alpha == 0.5
    # And the prior spread is reported beside the posterior, which is the
    # pair a cycling run needs to tell the two spread losses apart.
    assert set(diag.prior_spread) == set(diag.posterior_spread) == set(fields)
    for f in fields:
        assert diag.prior_spread[f] > 0.0
