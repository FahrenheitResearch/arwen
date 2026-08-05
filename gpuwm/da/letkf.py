"""LETKF -- local ensemble transform Kalman filter, batched for the GPU.

EXPERIMENTAL.  Nothing in the forecast path calls this module; it is a pure
addition behind an explicit import.  No config default routes through it.

Algorithm authority
-------------------
Hunt, Kostelich and Szunyogh (2007), *Efficient data assimilation for
spatiotemporal chaos: a local ensemble transform Kalman filter*, Physica D
230, 112-126 -- specifically the boxed recipe of their section 2.3, steps
(1)-(9).  The notation below is theirs:

    Xb   (m x R)   background ensemble perturbations about the mean
    Yb   (p x R)   H(x_k) perturbations about the mean of H(x_k)
    C    (R x p)   Yb^T R^-1                     [step 4]
    Pa~  (R x R)   [(R-1)I/rho + C Yb]^-1        [step 5]
    Wa   (R x R)   [(R-1) Pa~]^1/2, SYMMETRIC    [step 6]
    wa_  (R)       Pa~ C (y^o - ybar)            [step 7]
    w_k  (R)       wa_ + Wa[:, k]                [step 8]
    x_k  (m)       xbar + Xb w_k                 [step 9]

Two departures from a literal transcription, both deliberate and both tested:

* the symmetric square root in step 6 is formed from the eigendecomposition
  that already produced Pa~, not from a second factorisation.  One
  ``eigh`` of ``(R-1)I/rho + C Yb = U D U^T`` gives both
  ``Pa~ = U D^-1 U^T`` and ``Wa = U D^-1/2 U^T sqrt(R-1)``.  This is the
  standard implementation and is what makes the batched form cheap.
* gridpoints with no valid localised observation are *excluded from the
  solve entirely* rather than passed through it.  With C = 0 the recipe
  collapses to a closed form -- ``Pa~ = rho/(R-1) I``, ``Wa = sqrt(rho)
  I``, ``wa_ = 0`` -- so the answer is written down instead of computed:
  ``x_k = xbar + sqrt(rho) x'_k``.  Evaluating it through ``eigh`` would
  make the ``rho = 1`` case depend on the eigendecomposition of a scaled
  identity returning exactly ``U U^T = I``, which no LAPACK or cuSOLVER
  contract promises, and the gate on this module asserts increments are
  EXACTLY zero beyond the localisation cutoff at ``rho = 1``.

  :mod:`gpuwm.core.jacobi_eigh` does promise it -- a diagonal matrix clears
  no rotation threshold, so ``U`` comes back the literal identity -- but the
  closed form stays, because the guarantee should not become a property of
  which eigensolver a caller configured.

  What the closed form must NOT do is drop the ``sqrt(rho)`` along with
  the solve.  Prior inflation is a property of the transform, not of the
  observation set: the perturbation stretch applies at every gridpoint,
  and only the *observation* increment is confined to the lens.  An
  earlier version of this module skipped both, which was a spatially
  discontinuous error -- as a Gaspari-Cohn weight approached zero from
  inside the cutoff ``Wa`` tended to ``sqrt(rho) I``, and at the cutoff
  the code jumped to ``Wa = I``.  See :func:`analyze`'s inactive-point
  scaling.

Why batching is the whole GPU win
---------------------------------
The per-gridpoint work is an R x R symmetric eigendecomposition with
R <= ~40.  On its own that is far too small to occupy a GPU -- a single
40x40 ``eigh`` is microseconds of arithmetic wrapped in tens of microseconds
of launch overhead, and a loop over 400k gridpoints spends essentially all
of its wall clock in Python and kernel launches.  Stacking thousands of them
into one ``(G, R, R)`` array and issuing a single batched solve turns the
launch overhead into a rounding error and runs all G problems concurrently.

That batched solve is this project's own kernel --
:mod:`gpuwm.core.jacobi_eigh`, one thread block per gridpoint, cyclic Jacobi
in shared memory -- and not a library call.  It is the same ALGORITHM
cuSOLVER's ``syevj`` uses, so the change is not a numerical opinion; what it
buys is that the DA path no longer depends on a CUDA library that the rest of
this project has never needed, and whose absence presents as "elementwise
CuPy works, the factorisation does not".  ``LetkfConfig.eigensolver`` keeps
``xp.linalg.eigh`` reachable, and is what the A/B in
``tests/test_letkf_eigensolver.py`` switches.  See
``docs/da_jacobi_eigensolver.md``.

Every array in the inner loop
therefore carries a leading gridpoint axis, and the only Python-level
iteration is over chunks (see ``LetkfConfig.chunk_points``), sized to bound
peak memory rather than to bound work.

Localisation
------------
R-localisation (Hunt et al. section 3.2's "observation error inflation"
variant, as used by essentially every operational LETKF): the local
observation error variance is divided by a Gaspari-Cohn weight, so an
observation at the cutoff has infinite error and contributes nothing, and
an observation beyond the cutoff is dropped from the gridpoint's batch.
Horizontal and vertical weights multiply.  The exact piecewise quintic is
:func:`gaspari_cohn`.

Because the observation array is gridded on the *model* grid (contract
``gpuwm-obs.radar-grid.v1``, see :class:`GriddedObs`), the set of
observations local to a gridpoint is a fixed box in index space rather than
a ragged neighbour list.  That is what makes the gather uniform and the
batch rectangular: no padding bookkeeping, no per-point obs counts, one
stencil shared by every gridpoint.

That index box is a *superset*, not the metric.  Distances are physical
metres between the two gridpoints actually being related -- a geodesic
between their mass-point coordinates when the grid is geolocated, and a
height difference taken in each point's own column -- and they are
evaluated per pair inside the batch, because on a projected,
terrain-following grid neither quantity is a function of the index offset
alone.  The stencil is sized from the smallest spacing anywhere on the
grid, so it cannot exclude a pair the metric would have included; slots it
includes that the metric puts at or beyond the cutoff take a weight of
exactly zero and drop out.  See :class:`GridGeometry`.

Inflation
---------
RTPS -- relaxation to prior spread, Whitaker and Hamill (2012) eq. 6:

    Xa <- Xa * [ alpha * (sigma_b - sigma_a) / sigma_a + 1 ]

applied pointwise, per analysis field, after the transform and before the
increment is formed.  ``alpha = 0`` disables it.  The relaxed spread is
``alpha*sigma_b + (1-alpha)*sigma_a``, so ``alpha = 1`` restores the prior
spread exactly and ``alpha`` outside [0, 1] is refused.

**RTPP** (relaxation to prior perturbation, Zhang et al. 2004) is now
available beside it as ``LetkfConfig.relaxation = "rtpp"``:

    Xa <- (1 - alpha) Xa + alpha Xb

It needs no spread diagnostics and it relaxes the analysis *covariance
structure*, not just its amplitude, which is the reason to prefer it when
the localisation is aggressive enough to distort cross-covariances -- as
it is when a rank-``R-1`` ensemble meets a dense radar volume.  The two
are not interchangeable: RTPS restores a target SPREAD pointwise and is
blind to which direction in ensemble space lost it, while RTPP restores a
fraction of the prior PERTURBATION and therefore preserves the prior's
correlations between fields.  Which one a domain wants is an empirical
question, so this module ships both and records in
:class:`LetkfDiagnostics` which one ran, next to the prior and posterior
spread it produced.

One alternative is documented but NOT built here:

* **Additive inflation** (Mitchell and Houtekamer 2000): add scaled random
  draws to the analysis perturbations.  This is the only one of the three
  that can restore *rank*, so it is the right answer when the R <= 40
  subspace has collapsed rather than merely shrunk -- and, unlike either
  relaxation, it is the only one that can counteract spread lost in the
  FORECAST step rather than in the analysis.  It is not in this module
  because it is not a property of the transform: it is a perturbation
  drawn on the model grid and added to a state, which is
  :mod:`gpuwm.da.perturb`'s job and a caller's cycle policy.  A version
  that drew white noise here instead would inject exactly the
  small-scale imbalance the model then has to reject.

Multiplicative prior inflation is available as ``LetkfConfig.prior_inflation``
(the ``rho`` of step 5) since it costs one scalar in the batched solve.

Both relaxations leave the inactive-point closed form unchanged, which is
not a coincidence: at a gridpoint with no localised observation
``Xa = sqrt(rho) Xb``, so RTPS gives ``[(1-alpha) sqrt(rho) + alpha] Xb``
and RTPP gives ``[(1-alpha) sqrt(rho) + alpha] Xb`` -- the same scalar.
The bitwise-zero guarantee beyond the cutoff at ``rho = 1`` therefore
holds for both.

Fail-closed policy
------------------
Every degenerate input raises :class:`LetkfError` with a diagnostic.  None
of them returns NaN, and none silently returns the prior.  Specifically:
ensembles smaller than two members; non-positive or non-finite observation
errors on unmasked observations; non-finite priors or simulated
observations; an analysis field whose prior ensemble has no spread anywhere
(that is not an ensemble); shape disagreements between any two inputs; a
localisation radius that is not finite and positive; and -- as a backstop
that catches anything the specific checks missed -- a non-finite value
anywhere in the computed increments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "LetkfError",
    "RELAXATION_MODES",
    "EIGENSOLVER_MODES",
    "Localization",
    "GridGeometry",
    "GriddedObs",
    "LetkfConfig",
    "LetkfDiagnostics",
    "gaspari_cohn",
    "analyze",
]


class LetkfError(ValueError):
    """Any refusal by this module.  Never raised for a merely hard problem."""


#: Which batched symmetric eigensolver factors ``(R-1)I/rho + C Yb``.
#:
#: ``"auto"`` -- this project's own kernel
#:   (:mod:`gpuwm.core.jacobi_eigh`) whenever the analysis is on the device
#:   and R is inside its supported range, and the array namespace's
#:   ``linalg.eigh`` otherwise.  This is the default.  On numpy it always
#:   means numpy, which is LAPACK in-process and involves no CUDA library
#:   at all.
#:
#: ``"jacobi"`` -- REQUIRE the project kernel and refuse if it cannot take
#:   the problem.  For a caller that has decided cuSOLVER is not allowed to
#:   be a silent fallback.
#:
#: ``"library"`` -- ``xp.linalg.eigh``, i.e. cuSOLVER on the device and
#:   LAPACK on the host.  The escape hatch, and what the A/B in
#:   ``tests/test_jacobi_eigh_gpu.py`` compares against.
#:
#: The choice is recorded in :class:`LetkfDiagnostics` because the two
#: solvers agree to rounding rather than bitwise, so an analysis does not say
#: on its face which one produced it.
EIGENSOLVER_MODES = ("auto", "jacobi", "library")


#: Posterior relaxation schemes, by name.
#:
#: ``"rtps"`` -- relaxation to prior SPREAD, Whitaker and Hamill (2012)
#:   eq. 6.  Rescales each analysis perturbation so the pointwise spread
#:   becomes ``alpha sigma_b + (1 - alpha) sigma_a``.
#:
#: ``"rtpp"`` -- relaxation to prior PERTURBATION, Zhang et al. (2004).
#:   Mixes the perturbations themselves, ``(1 - alpha) Xa + alpha Xb``,
#:   which relaxes the analysis covariance structure rather than only its
#:   amplitude.
#:
#: ``alpha = 0`` is the identity under both, and both reduce to the same
#: scalar at a gridpoint with no localised observation.
RELAXATION_MODES = ("rtps", "rtpp")


def _get_xp(*arrays):
    """numpy, or cupy when any input is a cupy array.

    No cupy import happens unless a cupy array actually shows up, which is
    what keeps ``-m "not gpu"`` honest for callers of this module.
    """
    for a in arrays:
        mod = type(a).__module__
        if mod is not None and mod.split(".")[0] == "cupy":
            import cupy
            return cupy
    return np


# ---------------------------------------------------------------------------
# Gaspari-Cohn
# ---------------------------------------------------------------------------

def gaspari_cohn(distance, cutoff):
    """The Gaspari-Cohn compactly supported correlation function.

    Gaspari and Cohn (1999), *Construction of correlation functions in two
    and three dimensions*, QJRMS 125, 723-757, equation (4.10).  This is the
    fifth-order piecewise rational function that approximates a Gaussian
    while vanishing identically beyond a finite radius -- the property that
    makes it, rather than the Gaussian, the standard localisation kernel.

    Parameters
    ----------
    distance
        Separation, same units as ``cutoff``.  Scalar or array; negative
        values are treated by magnitude.
    cutoff
        The radius at which the function reaches zero AND STAYS THERE.  This
        is the ``2c`` of Gaspari and Cohn's own parameterisation, not their
        ``c``.  The two conventions differ by a factor of two and confusing
        them silently halves or doubles every localisation radius in a
        system, so this module names the unambiguous one: past ``cutoff``
        the weight is exactly ``0.0``, so ``cutoff`` is the distance beyond
        which an observation cannot influence an analysis point.  Gaspari
        and Cohn's ``c`` -- their e-folding-ish length scale, where the
        function takes the value 5/24 -- is ``cutoff / 2``.

    Returns
    -------
    Weights in [0, 1], with ``gaspari_cohn(0, c) == 1.0`` exactly and
    ``gaspari_cohn(d, c) == 0.0`` exactly for ``d >= c``.

    Notes
    -----
    Evaluated in the namespace of ``distance`` (numpy or cupy) and in that
    array's dtype, except that an integer or python-scalar input is promoted
    to float64.  The ``2/(3r)`` term of the outer branch is evaluated with a
    guarded denominator so that the vectorised form does not raise or warn
    on the ``r = 0`` elements it computes and then discards.
    """
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise LetkfError(
            f"Gaspari-Cohn cutoff must be finite and positive, got {cutoff!r}."
            " Use a radius larger than the domain diagonal to mean 'no"
            " localisation'; infinity is refused because it implies an"
            " infinite stencil."
        )
    xp = _get_xp(distance)
    d = xp.abs(xp.asarray(distance))
    if d.dtype.kind != "f":
        d = d.astype(xp.float64)
    one = d.dtype.type(1)

    # r is distance in units of Gaspari-Cohn's c = cutoff/2, so the two
    # polynomial branches are r <= 1 and 1 < r < 2, exactly as published.
    r = d * d.dtype.type(2.0 / cutoff)
    # Both polynomials are evaluated everywhere and selected between
    # afterwards, so an out-of-support distance still flows through r**5.
    # Clamping first keeps an infinite separation -- which is exactly what a
    # masked or out-of-domain slot looks like to a caller -- from producing
    # inf - inf and a RuntimeWarning on a value that is about to be
    # discarded.
    r = xp.minimum(r, d.dtype.type(2))
    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r5 = r4 * r

    inner = (
        -r5 / 4 + r4 / 2 + r3 * (5.0 / 8.0) - r2 * (5.0 / 3.0) + one
    )
    # Guard only the reciprocal: r == 0 elements take the inner branch and
    # this value is discarded, but an unguarded 1/0 still warns and pollutes.
    r_safe = xp.where(r > 0, r, one)
    outer = (
        r5 / 12 - r4 / 2 + r3 * (5.0 / 8.0) + r2 * (5.0 / 3.0)
        - r * 5 + one * 4 - (2.0 / 3.0) / r_safe
    )

    # The outer branch is selected on r < 2, STRICTLY.  Analytically the
    # polynomial is zero at r = 2, but in floating point it evaluates to a
    # few ulp either side of zero, and the whole localisation guarantee --
    # "an increment beyond the cutoff is bitwise zero, not merely small" --
    # rests on the weight at and past the cutoff being the literal 0.0 that
    # drops the observation from the gridpoint's batch.  Selecting the
    # constant instead of the polynomial at r >= 2 is what makes that true
    # rather than nearly true.
    out = xp.where(r <= 1, inner, xp.where(r < 2, outer, xp.zeros_like(d)))
    # The published polynomial is nonnegative on [0, 2] but rounding can
    # produce -1e-17 within an ulp of r == 2; a negative localisation weight
    # would flip the sign of an observation's influence.
    return xp.maximum(out, xp.zeros_like(out))


# ---------------------------------------------------------------------------
# Configuration and contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Localization:
    """Horizontal and vertical Gaspari-Cohn cutoffs, in metres.

    Both are full support radii in the sense of :func:`gaspari_cohn`: an
    observation exactly ``horizontal_m`` away contributes nothing at all.
    The two weights multiply, so the influence region is a lens, not a box.
    """

    horizontal_m: float
    vertical_m: float

    def __post_init__(self) -> None:
        for name in ("horizontal_m", "vertical_m"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise LetkfError(
                    f"Localization.{name} must be finite and positive,"
                    f" got {v!r}."
                )


def _geodesic_m(lat1_rad, lon1_rad, lat2_rad, lon2_rad, radius_m):
    """Great-circle distance on a sphere, in metres, elementwise.

    The haversine form rather than the spherical law of cosines: the
    separations this module measures are localisation radii, kilometres on
    a 6371 km sphere, and ``acos`` of a cosine that close to 1 loses most of
    its significant digits.
    """
    xp = _get_xp(lat1_rad, lat2_rad)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (xp.sin(dlat * 0.5) ** 2
         + xp.cos(lat1_rad) * xp.cos(lat2_rad) * xp.sin(dlon * 0.5) ** 2)
    return (2.0 * radius_m) * xp.arcsin(xp.sqrt(xp.minimum(a, 1.0)))


@dataclass(frozen=True)
class GridGeometry:
    """The metric the localisation distances are measured in.

    Both distances are physical metres between the two gridpoints actually
    being related, not index counts scaled by a nominal spacing.

    heights_m
        Layer-midpoint heights, strictly increasing bottom-up.  Either one
        representative column ``(nz,)`` or -- what a terrain-following model
        really has -- the full field ``(nz, ny, nx)``.  Supply the full
        field: on terrain, two gridpoints at the same model level sit at
        different altitudes, and a single column assigns them a vertical
        separation of zero and a vertical localisation weight of 1 no matter
        how far apart they actually are.  A 2 km same-level height
        difference under a 4 km cutoff earns a weight of 5/24, not 1.
    lat_deg, lon_deg
        Optional mass-point latitude/longitude ``(ny, nx)``.  When supplied,
        horizontal separation is the geodesic between the two points, which
        is the unambiguous physical metric on any projection.  When omitted,
        it is ``hypot(di*dx_m, dj*dy_m)`` -- exact on an unprojected
        Cartesian grid, and an index-space approximation on anything else,
        because a conformal projection's local physical spacing is
        ``dx/m`` for map factor ``m`` and departs from ``dx`` away from the
        standard parallels.  Supply them for a projected domain.
    earth_radius_m
        Sphere the geodesic is measured on.  Defaults to WRF's 6 370 000 m
        (``module_llxy.F``), the same value the projections these grids come
        from are built with; pass the grid's own if it differs.
    """

    dx_m: float
    dy_m: float
    heights_m: np.ndarray
    lat_deg: np.ndarray | None = None
    lon_deg: np.ndarray | None = None
    earth_radius_m: float = 6370000.0

    def __post_init__(self) -> None:
        for name in ("dx_m", "dy_m", "earth_radius_m"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise LetkfError(
                    f"GridGeometry.{name} must be finite and positive,"
                    f" got {v!r}."
                )
        z = np.asarray(self.heights_m, dtype=np.float64)
        if z.ndim not in (1, 3):
            raise LetkfError(
                f"GridGeometry.heights_m must be (nz,) or (nz, ny, nx), got"
                f" shape {z.shape}."
            )
        if z.size == 0:
            raise LetkfError("GridGeometry.heights_m is empty.")
        if not np.all(np.isfinite(z)):
            raise LetkfError("GridGeometry.heights_m has non-finite entries.")
        if z.shape[0] > 1 and not np.all(np.diff(z, axis=0) > 0):
            raise LetkfError(
                "GridGeometry.heights_m must be strictly increasing"
                " (bottom-up) in every column, which it is not."
            )
        object.__setattr__(self, "heights_m", z)

        have = (self.lat_deg is not None, self.lon_deg is not None)
        if any(have) and not all(have):
            raise LetkfError(
                "GridGeometry.lat_deg and lon_deg must be supplied together:"
                " one without the other cannot locate a gridpoint."
            )
        if all(have):
            lat = np.asarray(self.lat_deg, dtype=np.float64)
            lon = np.asarray(self.lon_deg, dtype=np.float64)
            if lat.ndim != 2 or lat.shape != lon.shape:
                raise LetkfError(
                    f"GridGeometry.lat_deg {lat.shape} and lon_deg"
                    f" {lon.shape} must both be (ny, nx) on the mass grid."
                )
            if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
                raise LetkfError(
                    "GridGeometry.lat_deg/lon_deg have non-finite entries.")
            if np.any(np.abs(lat) > 90.0):
                raise LetkfError(
                    "GridGeometry.lat_deg is outside +/-90 degrees; a"
                    " swapped latitude/longitude pair is the usual cause.")
            if z.ndim == 3 and z.shape[1:] != lat.shape:
                raise LetkfError(
                    f"GridGeometry.heights_m {z.shape} does not sit on the"
                    f" {lat.shape} latitude/longitude grid."
                )
            object.__setattr__(self, "lat_deg", lat)
            object.__setattr__(self, "lon_deg", lon)

    @property
    def nz(self) -> int:
        return int(self.heights_m.shape[0])

    @property
    def geodesic(self) -> bool:
        """True when horizontal distance is a geodesic, not an index count."""
        return self.lat_deg is not None

    def height_field(self, ny: int, nx: int) -> np.ndarray:
        """``(nz, ny, nx)`` heights, broadcasting a single column if given."""
        z = self.heights_m
        if z.ndim == 1:
            return np.ascontiguousarray(
                np.broadcast_to(z[:, None, None], (z.size, ny, nx)))
        if z.shape[1:] != (ny, nx):
            raise LetkfError(
                f"GridGeometry.heights_m {z.shape} does not match the"
                f" prior's ({z.shape[0]}, {ny}, {nx}) grid."
            )
        return z

    def min_spacing_m(self) -> tuple[float, float]:
        """Smallest physical spacing between adjacent columns, ``(dx, dy)``.

        This is what sizes the horizontal stencil, and it must be the
        MINIMUM rather than the nominal: on a conformal projection the
        physical spacing is ``dx/m``, so wherever the map factor exceeds 1
        a fixed radius reaches more columns than the nominal spacing
        suggests, and a stencil sized on the nominal value would silently
        truncate the localisation there.
        """
        if self.lat_deg is None:
            return float(self.dx_m), float(self.dy_m)
        lat = np.radians(self.lat_deg)
        lon = np.radians(self.lon_deg)
        r = float(self.earth_radius_m)
        out = []
        for axis in (1, 0):
            if lat.shape[axis] < 2:
                out.append(float(self.dx_m if axis == 1 else self.dy_m))
                continue
            a = [slice(None)] * 2
            b = [slice(None)] * 2
            a[axis] = slice(None, -1)
            b[axis] = slice(1, None)
            d = _geodesic_m(lat[tuple(a)], lon[tuple(a)],
                            lat[tuple(b)], lon[tuple(b)], r)
            out.append(float(d.min()))
        dx_min, dy_min = out
        if not (math.isfinite(dx_min) and dx_min > 0.0
                and math.isfinite(dy_min) and dy_min > 0.0):
            raise LetkfError(
                "GridGeometry.lat_deg/lon_deg give a zero or non-finite"
                " spacing between adjacent columns: duplicated coordinates"
                " cannot define a localisation metric."
            )
        return dx_min, dy_min

    def level_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-level ``(min, max)`` height over all columns, each ``(nz,)``.

        The vertical stencil is an index-space offset list, but the metric
        it stands in for is metres, and on terrain the two do not line up:
        the same ``dk`` spans different distances in different columns.
        These bounds let the stencil be pruned conservatively -- an offset
        is kept if ANY column pair at that offset could fall inside the
        cutoff -- so no pair inside the radius is silently dropped.
        """
        z = self.heights_m
        if z.ndim == 1:
            return z, z
        flat = z.reshape(z.shape[0], -1)
        return flat.min(axis=1), flat.max(axis=1)


@dataclass(frozen=True)
class GriddedObs:
    """One observation type, gridded on the model grid.

    This is this lane's transcription of the ``gpuwm-obs.radar-grid.v1``
    contract.  The radar lane owns the producer; the filter owns only the
    consumer's expectations, which are:

    values
        ``(nz, ny, nx)``.  The observed quantity in its own units, already
        mapped onto model gridpoints.  Entries where ``mask`` is False are
        never read and may be anything, NaN included.
    errors
        ``(nz, ny, nx)`` or a scalar.  Observation error STANDARD DEVIATION,
        same units as ``values``.  Must be finite and strictly positive
        wherever ``mask`` is True.  A standard deviation, not a variance:
        the two differ by a square and a filter tuned against the wrong one
        fails quietly rather than loudly.
    simulated
        ``(R, nz, ny, nx)``.  ``H(x_k)`` for each background member, in the
        same units as ``values``.  Supplied by the caller, which is what
        keeps this module independent of ``gpuwm.da.obsop``: the filter
        never needs to know whether H was a reflectivity forward operator, a
        radial-velocity projection, or the identity.  It also means H is
        applied to the *full* member state before localisation, which is the
        only order that is correct for a nonlinear operator.
    mask
        ``(nz, ny, nx)`` boolean.  True where an observation exists.
    localization
        Optional per-type override.  Reflectivity and radial velocity
        routinely want different radii; when None the filter's default is
        used.
    """

    name: str
    values: object
    errors: object
    simulated: object
    mask: object
    localization: Localization | None = None


@dataclass(frozen=True)
class LetkfConfig:
    """Everything the filter needs that is not data.

    analysis_fields
        Which state fields the analysis updates, in the order increments are
        returned.  Named explicitly rather than inferred from the prior dict
        so that a caller can hand in a full state and update a subset --
        hydrometeors in particular are frequently withheld.
    rtps_alpha
        Relaxation to prior spread, in [0, 1].  REQUIRED, and deliberately:
        there is no defensible default.  ``0.0`` disables the relaxation
        entirely, and a cycling ensemble run without any posterior
        relaxation loses spread until it stops responding to observations
        -- which shows up as a forecast that quietly stops improving, not
        as an error.  Storm-scale ensemble systems typically run somewhere
        in 0.8-0.95, but the right value is settled by cycling statistics
        on the domain in question, so this asks rather than assumes.  It is
        recorded in :class:`LetkfDiagnostics` so the choice appears in the
        log next to its consequences.
    prior_inflation
        The ``rho`` of Hunt et al. step 5: multiplicative inflation of the
        background covariance, applied as ``(R-1)I/rho``.  1.0 disables it.
        Unlike ``rtps_alpha`` this one has a default, because 1.0 is a
        genuine identity -- the transform is exactly what the recipe gives
        with no inflation -- rather than a disabled safety feature.
    chunk_points
        Gridpoints per batched solve, or None (the default) to size it from
        ``memory_budget_mib`` once the stencil is known.

        This bounds peak memory, which is dominated by the gathered
        ``(R, G, P)`` simulated-observation block: R members x G gridpoints
        x P stencil slots x 8 bytes.  A fixed default is a footgun, because
        P is not known until the localisation radii meet the grid spacing:
        a perfectly ordinary 8 km radius on a 2 km grid with 30 members
        gives P = 315 and turns a 8192-point chunk into 620 MiB *per array*,
        several GiB at peak, on a card that is very likely shared.  Setting
        it from a budget instead means the radius and the ensemble size stay
        free parameters and only the number of batches changes.
    memory_budget_mib
        Target peak working set for the batched solve, when
        ``chunk_points`` is None.  Not a hard limit -- it does not count the
        prior, the increments, or the observation arrays, all of which are
        whole-domain and outside this module's control -- but the chunk
        loop's own scratch stays near it.

        512 MiB is a deliberately conservative default for a shared card,
        and it costs throughput.  Measured on an RTX 5090 (sm_120), R = 30,
        a 405-slot stencil, float64 solve, in active gridpoints per second:

            chunk       1 ->     305      (a per-gridpoint loop)
            chunk       8 ->   2,351
            chunk      64 ->  18,623
            chunk     512 ->  71,829
            chunk   2,048 ->  95,452      saturated
            chunk   8,192 ->  94,293
            numpy, same case ->  5,677

        Throughput saturates near 2,048 gridpoints per batch, which at that
        stencil and ensemble size wants roughly 1.2 GiB.  512 MiB gives 877
        and about 55% of peak.  Raise this when the card is yours; the
        answer does not change with it, only the number of batches.
    solve_dtype
        Precision of the R x R eigenproblem.  float64 by default and
        deliberately: the R x R solve is a vanishing fraction of total work
        even at 1/64 rate, the eigenvalues of ``(R-1)I + C Yb`` span the
        ensemble's condition number, and float32 subnormals are flushed in
        every kernel cupy compiles -- it appends ``-ftz=true`` after the
        caller's options -- which float64 sidesteps entirely.  The gather
        and the final ``Xb @ W`` stay in the input dtype.
    eigensolver
        Which batched symmetric eigensolver factors the R x R matrix; see
        :data:`EIGENSOLVER_MODES`.  ``"auto"`` -- the default -- prefers
        this project's own kernel, so nothing on the DA path needs cuSOLVER
        installed at the ensemble sizes that kernel supports.
    """

    localization: Localization
    analysis_fields: tuple[str, ...]
    rtps_alpha: float
    prior_inflation: float = 1.0
    #: Which posterior relaxation ``rtps_alpha`` drives; see
    #: :data:`RELAXATION_MODES`.  The parameter keeps its name under both
    #: because it is the same ``alpha`` of the same published family and
    #: renaming it per mode would silently reset a caller's tuning.
    relaxation: str = "rtps"
    chunk_points: int | None = None
    memory_budget_mib: float = 512.0
    solve_dtype: str = "float64"
    #: See :data:`EIGENSOLVER_MODES`.
    eigensolver: str = "auto"

    def __post_init__(self) -> None:
        if self.eigensolver not in EIGENSOLVER_MODES:
            raise LetkfError(
                f"eigensolver must be one of {EIGENSOLVER_MODES}, got"
                f" {self.eigensolver!r}."
            )
        if self.relaxation not in RELAXATION_MODES:
            raise LetkfError(
                f"relaxation must be one of {RELAXATION_MODES}, got"
                f" {self.relaxation!r}."
            )
        if not self.analysis_fields:
            raise LetkfError(
                "LetkfConfig.analysis_fields is empty: an analysis that"
                " updates nothing is a bug, not a configuration."
            )
        if len(set(self.analysis_fields)) != len(self.analysis_fields):
            raise LetkfError(
                "LetkfConfig.analysis_fields has duplicates:"
                f" {self.analysis_fields!r}."
            )
        rho = float(self.prior_inflation)
        if not math.isfinite(rho) or rho <= 0.0:
            raise LetkfError(
                f"prior_inflation (rho) must be finite and positive, got"
                f" {rho!r}."
            )
        a = float(self.rtps_alpha)
        if not math.isfinite(a) or not (0.0 <= a <= 1.0):
            raise LetkfError(
                f"rtps_alpha must lie in [0, 1], got {a!r}.  Values above 1"
                " inflate past the prior spread and diverge; negative values"
                " deflate an already over-confident analysis."
            )
        if self.chunk_points is not None and int(self.chunk_points) < 1:
            raise LetkfError(
                f"chunk_points must be >= 1 or None, got"
                f" {self.chunk_points!r}."
            )
        budget = float(self.memory_budget_mib)
        if not math.isfinite(budget) or budget <= 0.0:
            raise LetkfError(
                f"memory_budget_mib must be finite and positive, got"
                f" {budget!r}."
            )
        if self.solve_dtype not in ("float32", "float64"):
            raise LetkfError(
                f"solve_dtype must be float32 or float64, got"
                f" {self.solve_dtype!r}."
            )
        object.__setattr__(self, "analysis_fields", tuple(self.analysis_fields))


@dataclass
class LetkfDiagnostics:
    """What the analysis did, for the log and for the gate."""

    members: int = 0
    grid_shape: tuple[int, int, int] = (0, 0, 0)
    #: The two inflation settings the analysis actually ran with.  Recorded
    #: because both are invisible in the increments: an analysis with no
    #: relaxation and no inflation looks exactly like a well-tuned one for
    #: one cycle, and only stops working several cycles later.
    prior_inflation: float = 1.0
    rtps_alpha: float = 0.0
    #: Which relaxation the alpha above drove.  Recorded because RTPS and
    #: RTPP at the same alpha produce different posteriors and the
    #: increments do not say which ran.
    relaxation: str = "rtps"
    #: Which batched eigensolver actually ran -- ``"jacobi"`` or
    #: ``"library"`` -- resolved from ``LetkfConfig.eigensolver`` once, before
    #: the chunk loop.  Recorded for the same reason as ``relaxation``: the
    #: two agree to rounding, not bitwise, so an increment array does not say
    #: on its face which produced it, and a reproducibility receipt needs to.
    eigensolver: str = ""
    #: Sweeps the project kernel needed on its worst matrix, or 0 when the
    #: library solver ran.  A number climbing toward
    #: ``gpuwm.core.jacobi_eigh.SWEEP_CAP`` is the early warning that the
    #: localised matrix is worse conditioned than the recipe implies.
    max_jacobi_sweeps: int = 0
    #: Gridpoints with at least one observation inside the cutoff.  Every
    #: other gridpoint never entered a solve and carries the closed-form
    #: inactive-point transform ``(s - 1) x'`` with
    #: ``s = (1 - rtps_alpha) sqrt(prior_inflation) + rtps_alpha`` -- which
    #: is exactly zero at the default ``prior_inflation = 1`` and is NOT
    #: zero otherwise.  Reading ``active_points`` as "everywhere else the
    #: increment is zero" is only true for that default; the general
    #: statement is the one :func:`analyze` returns.
    active_points: int = 0
    total_points: int = 0
    #: Stencil slots per gridpoint, summed over observation types.  This is
    #: the ``P`` of the batched shapes and the multiplier on peak memory.
    stencil_slots: int = 0
    #: Largest number of valid observations any single gridpoint saw.
    max_local_obs: int = 0
    batches: int = 0
    #: Gridpoints per batched solve actually used, after auto-sizing.
    chunk_points: int = 0
    #: Per field, the RMS of the ensemble-mean increment.
    mean_increment_rms: dict = field(default_factory=dict)
    #: Per field, the domain-mean prior and posterior ensemble spread.
    prior_spread: dict = field(default_factory=dict)
    posterior_spread: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stencil construction
# ---------------------------------------------------------------------------

def _horizontal_stencil(
    loc: Localization, grid: GridGeometry, nx: int, ny: int
):
    """Offsets ``(dj, di)`` that could carry a nonzero weight anywhere.

    The box that circumscribes the cutoff has corners outside it; pruning
    them is not cosmetic.  At a 4:1 radius-to-spacing ratio the box holds 81
    points and the disc 69, and every pruned slot is R floats of gather and
    a column of a matrix product that was going to be multiplied by zero.

    The *weights* are not built here.  They depend on where the two columns
    actually are, not on the offset between their indices, so they are
    evaluated per gridpoint pair in the analysis.  What is built here is a
    superset: the offset's separation is bounded below by the smallest
    physical spacing anywhere on the grid, and an offset whose lower bound
    already exceeds the cutoff cannot be inside it in any column.  Sizing
    on the MINIMUM spacing is what keeps the superset a superset -- a
    conformal projection's spacing is ``dx/m``, so a map factor above 1
    packs more columns inside a fixed radius than the nominal spacing does.

    The half-widths are also clamped to the domain: a radius wider than the
    grid is a legitimate way to say "no horizontal localisation", and
    without the clamp it would allocate a stencil of offsets that can never
    land inside the domain and would be masked off one gather later.
    """
    dx_min, dy_min = grid.min_spacing_m()
    hx = min(nx - 1, int(math.floor(loc.horizontal_m / dx_min)))
    hy = min(ny - 1, int(math.floor(loc.horizontal_m / dy_min)))
    di = np.arange(-hx, hx + 1, dtype=np.int64)
    dj = np.arange(-hy, hy + 1, dtype=np.int64)
    lower = np.hypot(
        di[None, :].astype(np.float64) * dx_min,
        dj[:, None].astype(np.float64) * dy_min,
    )
    keep = np.asarray(gaspari_cohn(lower, loc.horizontal_m),
                      dtype=np.float64) > 0.0
    jj, ii = np.nonzero(keep)
    return dj[jj], di[ii]


def _vertical_stencil(loc: Localization, grid: GridGeometry):
    """Offsets ``dk`` that could carry a nonzero weight anywhere.

    Same contract as the horizontal stencil, and for the same reason: on a
    terrain-following grid the metres spanned by a given ``dk`` depend on
    the column, so the offset list is a superset and the weight is computed
    from the pair's own heights.  An offset is kept when the height
    intervals its two levels occupy over the whole domain come within the
    cutoff of each other; with one representative column those intervals
    are points and this reduces exactly to ``|z[k+dk] - z[k]| < cutoff``.
    """
    zmin, zmax = grid.level_bounds()
    nz = int(zmin.size)
    if nz == 1:
        return np.zeros(1, dtype=np.int64)
    keep = []
    for dk in range(-(nz - 1), nz):
        k = np.arange(max(0, -dk), min(nz, nz - dk))
        if k.size == 0:
            continue
        k2 = k + dk
        gap = np.maximum(np.maximum(zmin[k2] - zmax[k], zmin[k] - zmax[k2]),
                         0.0)
        if bool(np.any(gap < loc.vertical_m)):
            keep.append(dk)
    return np.asarray(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _resolve_eigensolver(xp, members, solve_dtype, config):
    """Pick the eigensolver ONCE, before any chunk runs.

    Deciding per chunk would let a single analysis silently mix two solvers
    -- they agree to rounding, so the seam would be invisible in the
    increments and visible only as an irreproducible receipt.  Deciding here
    also means an unsatisfiable ``eigensolver="jacobi"`` refuses before the
    first gather rather than after the last one.
    """
    mode = str(config.eigensolver)
    if mode == "library":
        return "library"
    if xp is np:
        if mode == "jacobi":
            raise LetkfError(
                "eigensolver='jacobi' needs the analysis on the device: the"
                " kernel is CUDA and this analysis is running in numpy."
                "  Pass cupy arrays, or use eigensolver='auto', which means"
                " numpy's own LAPACK here and needs no CUDA library at all."
            )
        return "library"
    from gpuwm.core.jacobi_eigh import supported
    if supported(members, solve_dtype):
        return "jacobi"
    if mode == "jacobi":
        from gpuwm.core.jacobi_eigh import MAX_K, MIN_K
        raise LetkfError(
            f"eigensolver='jacobi' was required but this project's kernel"
            f" does not take R={members} in {np.dtype(solve_dtype).name}: it"
            f" supports {MIN_K} <= R <= {MAX_K} in float32 or float64."
            "  Use eigensolver='auto' to fall back to the array namespace's"
            " own solver, or reduce the ensemble size."
        )
    return "library"


def _eigendecompose(xp, amat, which):
    """``(evals, evecs, sweeps)`` for a batch of symmetric matrices, ascending.

    Both branches promise the same contract -- eigenvalues ascending,
    eigenvectors in the columns -- so everything downstream is written once.
    ``sweeps`` is 0 for the library branch, which does not report one.
    """
    if which == "jacobi":
        from gpuwm.core.jacobi_eigh import JacobiEighError, batched_eigh
        try:
            return batched_eigh(amat, return_sweeps=True)
        except JacobiEighError as exc:
            raise LetkfError(
                "this project's batched Jacobi eigensolver refused the"
                f" localised LETKF matrix (R-1)I/rho + C Yb: {exc}"
                "  No analysis was produced; do not treat the prior as one."
            ) from exc
    try:
        evals, evecs = xp.linalg.eigh(amat)
        return evals, evecs, 0
    except Exception as exc:                     # LinAlgError and backend kin
        # The one failure mode no input check can pre-empt.  A raw
        # convergence message from LAPACK or cuSOLVER does not name the
        # analysis that did not happen, and this module promises that every
        # failure arrives as LetkfError.
        #
        # On the device this is also where a MISSING cuSOLVER lands, which is
        # a different problem wearing the same exception: elementwise CuPy
        # works, the gather worked, and only the factorisation failed.  Say
        # so, and name the setting that does not need it.
        raise LetkfError(
            "the batched eigensolver failed on the localised LETKF"
            f" matrix (R-1)I/rho + C Yb: {type(exc).__name__}: {exc}."
            "  This is eigensolver='library', which on the device is"
            " cuSOLVER -- a separate CUDA library from the one elementwise"
            " CuPy uses, so the rest of the analysis working says nothing"
            " about whether it is installed.  Run `gpuwm doctor` for a"
            " verdict on it, or use eigensolver='auto' (the default), which"
            " prefers this project's own kernel and needs no CUDA library."
            "  No analysis was produced; do not treat the prior as one."
        ) from exc


def _as_grid(xp, a, shape, what):
    arr = xp.asarray(a)
    if arr.shape != shape:
        raise LetkfError(f"{what} has shape {arr.shape}, expected {shape}.")
    return arr


def _validate_prior(xp, prior, fields, work_dtype):
    """Shapes, finiteness, and the "this is not an ensemble" check."""
    missing = [f for f in fields if f not in prior]
    if missing:
        raise LetkfError(
            f"analysis_fields not present in the prior: {missing!r}."
            f"  Prior has {sorted(prior)!r}."
        )
    shapes = {f: tuple(xp.asarray(prior[f]).shape) for f in fields}
    distinct = set(shapes.values())
    if len(distinct) != 1:
        raise LetkfError(
            f"analysis fields disagree on shape: {shapes!r}.  Every field"
            " must be (members, nz, ny, nx) on the same grid."
        )
    shape = distinct.pop()
    if len(shape) != 4:
        raise LetkfError(
            f"prior fields must be 4-D (members, nz, ny, nx), got {shape}."
        )
    members = shape[0]
    if members < 2:
        raise LetkfError(
            f"LETKF needs at least 2 ensemble members, got {members}."
            "  A one-member 'ensemble' has no covariance to update with."
        )
    out = {}
    for f in fields:
        arr = xp.asarray(prior[f], dtype=work_dtype)
        if not bool(xp.all(xp.isfinite(arr))):
            raise LetkfError(
                f"prior field {f!r} contains non-finite values; the filter"
                " will not launder them into an analysis."
            )
        out[f] = arr
    return out, members, shape[1:]


def _validate_obs(xp, obs, members, shape, work_dtype):
    checked = []
    for k, o in enumerate(obs):
        tag = f"observation batch {k} ({o.name!r})"
        mask = _as_grid(xp, o.mask, shape, f"{tag} mask").astype(bool)
        values = _as_grid(xp, o.values, shape, f"{tag} values").astype(
            work_dtype)
        err = xp.asarray(o.errors, dtype=work_dtype)
        if err.ndim == 0:
            err = xp.broadcast_to(err, shape).copy()
        else:
            err = _as_grid(xp, err, shape, f"{tag} errors").astype(work_dtype)
        sim = xp.asarray(o.simulated, dtype=work_dtype)
        if sim.shape != (members,) + shape:
            raise LetkfError(
                f"{tag} simulated has shape {sim.shape}, expected"
                f" {(members,) + shape} -- H(x_k) for every member on the"
                " model grid."
            )
        if bool(xp.any(mask)):
            if not bool(xp.all(xp.isfinite(values[mask]))):
                raise LetkfError(
                    f"{tag} has non-finite values where mask is True."
                    "  Mask them out instead."
                )
            emask = err[mask]
            if not bool(xp.all(xp.isfinite(emask))) or not bool(
                    xp.all(emask > 0)):
                raise LetkfError(
                    f"{tag} has observation errors that are not finite and"
                    " positive where mask is True.  errors is a standard"
                    " deviation; zero means an observation the filter must"
                    " match exactly, which is not a thing this filter can"
                    " represent."
                )
            if not bool(xp.all(xp.isfinite(sim[:, mask]))):
                raise LetkfError(
                    f"{tag} simulated H(x_k) is non-finite where mask is"
                    " True; the forward operator failed on at least one"
                    " member and the filter will not average over that."
                )
        loc = o.localization
        if loc is not None and not isinstance(loc, Localization):
            raise LetkfError(
                f"{tag} localization must be a Localization or None, got"
                f" {type(loc).__name__}."
            )
        # Masked slots participate in a rectangular gather and are zeroed by
        # weight, but 0 * NaN is NaN.  Substituting a finite placeholder here
        # is cheaper and safer than a where() in the inner loop.
        values = xp.where(mask, values, xp.zeros_like(values))
        err = xp.where(mask, err, xp.ones_like(err))
        sim = xp.where(mask[None], sim, xp.zeros_like(sim))
        checked.append((o.name, values, err, sim, mask, loc))
    return checked


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def _finish(xp, fields, pri, increments, members, diagnostics):
    """The backstop finiteness check and the per-field spread diagnostics.

    Shared by the ordinary exit and the no-observation exit so that both
    report the same quantities computed the same way.  The no-observation
    cycle is no longer trivially zero -- with ``rho != 1`` every gridpoint
    is inactive and every gridpoint is inflated -- so it cannot fill these
    in from constants.
    """
    for f in fields:
        inc = increments[f]
        if not bool(xp.all(xp.isfinite(inc))):
            raise LetkfError(
                f"analysis increment for {f!r} is non-finite.  Every"
                " specific guard passed, so this is a genuine numerical"
                " failure in the transform -- do not apply this analysis."
            )
        diagnostics.mean_increment_rms[f] = float(
            xp.sqrt((inc.mean(axis=0) ** 2).mean()))
        post = pri[f] + inc
        pm = post.mean(axis=0, keepdims=True)
        diagnostics.posterior_spread[f] = float(
            xp.sqrt(((post - pm) ** 2).sum(axis=0) / (members - 1)).mean())


def analyze(
    prior: Mapping[str, object],
    obs: Sequence[GriddedObs],
    grid: GridGeometry,
    config: LetkfConfig,
    diagnostics: LetkfDiagnostics | None = None,
) -> dict:
    """One LETKF analysis.  Returns per-member increments, not the analysis.

    Parameters
    ----------
    prior
        ``{field_name: array (R, nz, ny, nx)}``.  numpy or cupy; the whole
        analysis runs in whichever namespace these arrive in.
    obs
        Observation batches (:class:`GriddedObs`).  An empty sequence, or
        one whose masks are all False, is a legitimate no-observation
        cycle: every gridpoint is inactive, so every increment is the
        inactive-point transform below.  At the default
        ``prior_inflation = 1`` that transform is the exact identity and
        the cycle returns bitwise-zero increments -- the no-DA control.
        With ``prior_inflation != 1`` it is the intended whole-domain
        inflation of an unobserved cycle, not zero.
    grid
        Localisation metric.  ``grid.nz`` must match the prior.
    config
        See :class:`LetkfConfig`.
    diagnostics
        Optional; filled in place if given.

    Returns
    -------
    ``{field_name: array (R, nz, ny, nx)}`` -- the increment to ADD to each
    prior member.  Increments rather than the analysis state because the
    caller owns the state: it knows which fields have positivity constraints,
    which are staggered, and how to journal a partial update.  A filter that
    returns a state has already made those decisions on the caller's behalf.

    At every gridpoint with no observation inside its localisation cutoff
    the returned increment is the closed-form inactive-point transform,
    ``(s - 1) x'`` with ``s = (1 - alpha) sqrt(rho) + alpha``: the ensemble
    mean is untouched and the perturbations carry Hunt's prior inflation
    relaxed by RTPS.  At the default ``rho = 1`` that scaling is the exact
    identity and those increments are bitwise zero -- not nearly zero --
    which is the guarantee the localisation gate rests on.
    """
    if diagnostics is None:
        diagnostics = LetkfDiagnostics()

    fields = config.analysis_fields
    probe = [prior[f] for f in fields if f in prior]
    probe += [o.values for o in obs]
    xp = _get_xp(*probe)

    # dtype objects are namespace-agnostic (cupy reuses numpy's), so these
    # are np.dtype even when every array is on the device.
    work_dtype = np.dtype(np.float64)
    if fields[0] in prior:
        cand = np.dtype(xp.asarray(prior[fields[0]]).dtype)
        if cand.kind == "f":
            work_dtype = cand
    solve_dtype = np.dtype(config.solve_dtype)

    pri, members, shape = _validate_prior(xp, prior, fields, work_dtype)
    nz, ny, nx = shape
    if nz != grid.nz:
        raise LetkfError(
            f"prior has {nz} levels but GridGeometry.heights_m has"
            f" {grid.nz}."
        )
    if grid.geodesic and tuple(grid.lat_deg.shape) != (ny, nx):
        raise LetkfError(
            f"prior is on a ({ny}, {nx}) horizontal grid but"
            f" GridGeometry.lat_deg is {tuple(grid.lat_deg.shape)}."
        )
    checked = _validate_obs(xp, obs, members, shape, work_dtype)

    diagnostics.members = members
    diagnostics.grid_shape = (nz, ny, nx)
    diagnostics.total_points = nz * ny * nx
    diagnostics.prior_inflation = float(config.prior_inflation)
    diagnostics.rtps_alpha = float(config.rtps_alpha)
    diagnostics.relaxation = str(config.relaxation)
    # Resolved before anything is gathered, so an unsatisfiable request
    # refuses at the top and a satisfiable one is recorded even on a cycle
    # that turns out to have no active gridpoint and never reaches a solve.
    eigensolver = _resolve_eigensolver(xp, members, solve_dtype, config)
    diagnostics.eigensolver = eigensolver

    # Prior mean and perturbations, once, for the whole domain.  Xb is what
    # step 9 multiplies; nothing downstream needs the members again.
    xb = {}
    for f in fields:
        m = pri[f].mean(axis=0, keepdims=True)
        xb[f] = pri[f] - m
    sigma_b = {
        f: xp.sqrt((xb[f] ** 2).sum(axis=0) / (members - 1)) for f in fields
    }
    for f in fields:
        # Testing ``spread == 0`` exactly is the obvious check and the wrong
        # one.  Subtracting the mean of R identical floats does not give
        # exactly zero -- ``sum/R`` rounds -- so a genuinely constant
        # ensemble arrives here with a spread around 1e-16 and sails
        # straight through an exact test.  The honest question is whether
        # the spread is negligible against the field's own magnitude.
        scale = float(xp.abs(pri[f]).max())
        widest = float(sigma_b[f].max())
        if widest <= 1e-12 * scale:
            raise LetkfError(
                f"prior field {f!r} has no usable ensemble spread anywhere"
                f" (largest pointwise spread {widest!r} against a field"
                f" magnitude of {scale!r}): the members are identical to"
                " rounding, so there is no background covariance and no"
                " analysis to compute.  This is an ensemble-generation"
                " failure, not something the filter should paper over with"
                " a zero increment.  If the field is deliberately constant,"
                " drop it from analysis_fields."
            )
        diagnostics.prior_spread[f] = float(sigma_b[f].mean())

    # ---- the inactive-point transform, in closed form ------------------
    # A gridpoint with no localised observation has an empty Yb, so Hunt's
    # step 5 gives A = (R-1)I/rho, step 6 gives Wa = sqrt(rho) I and step 7
    # gives wa_ = 0.  RTPS then relaxes that analysis spread, sigma_a =
    # sqrt(rho) sigma_b, back toward the prior:
    #
    #     sigma'_a = (1 - alpha) sqrt(rho) sigma_b + alpha sigma_b,
    #
    # so the entire transform at such a point is the single scalar
    # s = (1 - alpha) sqrt(rho) + alpha applied to the perturbations, and
    # the increment is (s - 1) x'.  The ensemble mean does not move.
    #
    # rho == 1 is special-cased to the literal identity rather than left to
    # (1 - alpha)*1 + alpha: the promise that increments beyond the cutoff
    # are BITWISE zero should not become a property of how that expression
    # happens to round for a particular alpha.
    rho = float(config.prior_inflation)
    inactive_scale = 1.0 if rho == 1.0 else (
        (1.0 - float(config.rtps_alpha)) * math.sqrt(rho)
        + float(config.rtps_alpha))
    if inactive_scale == 1.0:
        increments = {f: xp.zeros_like(pri[f]) for f in fields}
    else:
        step = work_dtype.type(inactive_scale - 1.0)
        increments = {f: xb[f] * step for f in fields}

    if not checked:
        _finish(xp, fields, pri, increments, members, diagnostics)
        return increments

    # ---- the coordinates the localisation metric is measured in --------
    # Flat, float64, one entry per gridpoint (heights) or per column
    # (geolocation).  Distances are always computed in float64 even for a
    # float32 solve: a geodesic differences two nearly equal latitudes, and
    # doing that in float32 would put the rounding error at a good fraction
    # of a grid cell.
    zflat = xp.asarray(grid.height_field(ny, nx), dtype=np.float64).reshape(-1)
    if grid.geodesic:
        latflat = xp.asarray(np.radians(grid.lat_deg).reshape(-1),
                             dtype=np.float64)
        lonflat = xp.asarray(np.radians(grid.lon_deg).reshape(-1),
                             dtype=np.float64)
        xflat = yflat = None
    else:
        latflat = lonflat = None
        cols = np.arange(ny * nx, dtype=np.float64)
        xflat = xp.asarray((cols % nx) * float(grid.dx_m))
        yflat = xp.asarray((cols // nx) * float(grid.dy_m))

    def _horizontal_distance(col_a, col_b):
        """Metres between two mass columns, given their flat indices."""
        if grid.geodesic:
            return _geodesic_m(latflat[col_a], lonflat[col_a],
                               latflat[col_b], lonflat[col_b],
                               float(grid.earth_radius_m))
        return xp.hypot(xflat[col_b] - xflat[col_a],
                        yflat[col_b] - yflat[col_a])

    # ---- stencils, one per distinct localisation spec -----------------
    stencils = []
    total_slots = 0
    for (name, values, err, sim, mask, loc) in checked:
        spec = loc if loc is not None else config.localization
        dj, di = _horizontal_stencil(spec, grid, nx, ny)
        dk = _vertical_stencil(spec, grid)
        if int(dj.size) <= 1:
            # A stencil holding only the zero offset is well defined -- it
            # assimilates co-located observations and nothing else -- but it
            # is never what anyone meant.  It is what a horizontal radius
            # expressed in kilometres against a grid spacing in metres looks
            # like, and that mistake otherwise produces a filter that runs,
            # reports increments, and spreads no information at all.
            raise LetkfError(
                f"observation batch {name!r} has a horizontal localisation"
                f" radius of {spec.horizontal_m!r} m, which does not reach"
                f" even one neighbouring column at dx={grid.dx_m!r} m,"
                f" dy={grid.dy_m!r} m.  No observation could influence any"
                " gridpoint but its own.  Check the units -- radii are"
                " metres here -- or widen the radius."
            )
        # Slot ordering is (vertical outer, horizontal inner) so that the
        # per-pair vertical weight, which carries both axes, broadcasts
        # against the column-only horizontal weight with one reshape and no
        # transpose.
        # The square is taken in solve_dtype, NOT in the input dtype.  The
        # sigma was validated finite and positive while it was still a
        # standard deviation; squaring it in a float32 state's dtype throws
        # that validation away, because a perfectly ordinary float32 sigma
        # of 1e-25 has a square that float32 cannot represent at all.  The
        # solve then divides by zero and meets an infinity it has no guard
        # for.  Promoting first costs one temporary and keeps the default
        # float64 solve immune to the state's precision.
        err2 = err.reshape(-1).astype(solve_dtype) ** 2
        mflat = mask.reshape(-1)
        if bool(xp.any(mflat)) and not bool(xp.all(err2[mflat] > 0)):
            raise LetkfError(
                f"observation batch {name!r} has an observation error whose"
                f" SQUARE underflows to zero in the {config.solve_dtype}"
                " solve, although the error itself is finite and positive."
                "  R^-1 = weight/sigma^2 is then a division by zero and the"
                " transform is meaningless.  Use solve_dtype='float64'"
                " (the default), or express the observation in units that"
                " do not need a standard deviation this small."
            )
        stencils.append({
            "name": name,
            "values": values.reshape(-1),
            "err2": err2,
            "sim": sim.reshape(members, -1),
            "mask": mask.reshape(-1),
            "dk": xp.asarray(dk),
            "dj": xp.asarray(dj),
            "di": xp.asarray(di),
            "hcut": float(spec.horizontal_m),
            "vcut": float(spec.vertical_m),
            "nslots": int(dj.size) * int(dk.size),
        })
        total_slots += int(dj.size) * int(dk.size)
    diagnostics.stencil_slots = total_slots

    npts = nz * ny * nx
    if config.chunk_points is not None:
        chunk = int(config.chunk_points)
    else:
        # Peak scratch per gridpoint, counting the arrays that carry both a
        # stencil and a member axis: the gathered simulated observations,
        # the perturbations taken from them, the transposed-and-weighted C,
        # and the temporaries the matmuls need.  Six is measured slack, not
        # a derivation -- the point is to be right to a factor of two, so
        # that the radius and the ensemble size are free parameters and only
        # the batch count moves.
        per_point = 6 * total_slots * members * solve_dtype.itemsize \
            + 4 * members * members * solve_dtype.itemsize
        budget = int(config.memory_budget_mib * (1 << 20))
        chunk = max(1, min(npts, budget // max(1, per_point)))
    diagnostics.chunk_points = chunk
    ident = xp.eye(members, dtype=solve_dtype)
    scale = solve_dtype.type(members - 1) / solve_dtype.type(
        config.prior_inflation)

    # Flat increment views: the solve works on a flat gridpoint axis and
    # scatters back through these.  reshape(-1) on a freshly allocated
    # C-contiguous zeros array is a view, so the scatter lands in the
    # returned arrays.
    incr_flat = {f: increments[f].reshape(members, -1) for f in fields}
    xb_flat = {f: xb[f].reshape(members, -1) for f in fields}

    max_local = 0
    nactive = 0
    nbatch = 0
    max_sweeps = 0

    for start in range(0, npts, chunk):
        stop = min(start + chunk, npts)
        pts = xp.arange(start, stop)
        kk = pts // (ny * nx)
        rem = pts - kk * (ny * nx)
        jj = rem // nx
        ii = rem - jj * nx

        # ---- phase 1: weights only, no member axis --------------------
        # Cheap enough to throw away: (G, P) versus the (R, G, P) gather it
        # decides whether to do at all.  In a radar-sparse domain most
        # gridpoints have no observation within the cutoff and this phase
        # eliminates them before they cost anything.
        #
        # Both weights are evaluated from the coordinates of the two
        # gridpoints being related, not from the offset between their
        # indices.  Horizontally that is one distance per (analysis column,
        # stencil column) pair; vertically it is one per full slot, because
        # on terrain the height at a given model level is a property of the
        # column.  A precomputed offset -> weight table is cheaper and is
        # what this used to do, but it can only express a metric in which
        # every column is identical, which is exactly the claim a
        # terrain-following projected grid does not support.
        ccol = jj * nx + ii                         # (G,) analysis column
        w_parts = []
        idx_parts = []
        for st in stencils:
            k2 = kk[:, None] + st["dk"][None, :]
            inside_k = (k2 >= 0) & (k2 < nz)
            k2 = xp.clip(k2, 0, nz - 1)
            j2 = jj[:, None] + st["dj"][None, :]
            i2 = ii[:, None] + st["di"][None, :]
            inside_h = (j2 >= 0) & (j2 < ny) & (i2 >= 0) & (i2 < nx)
            j2 = xp.clip(j2, 0, ny - 1)
            i2 = xp.clip(i2, 0, nx - 1)
            col = j2 * nx + i2                      # (G, n_h)
            flat = (k2[:, :, None] * ny + j2[:, None, :]) * nx \
                + i2[:, None, :]
            shape3 = flat.shape
            flat = flat.reshape(shape3[0], -1)
            inside = (inside_k[:, :, None] & inside_h[:, None, :]).reshape(
                flat.shape)
            wh = gaspari_cohn(
                _horizontal_distance(ccol[:, None], col), st["hcut"])
            dz = xp.abs(zflat[flat] - zflat[pts][:, None])
            w = (xp.asarray(gaspari_cohn(dz, st["vcut"])).reshape(shape3)
                 * xp.asarray(wh)[:, None, :]).reshape(flat.shape)
            w = xp.where(inside & st["mask"][flat], w, 0)
            w_parts.append(w.astype(solve_dtype))
            idx_parts.append(flat)

        wloc = xp.concatenate(w_parts, axis=1) if len(w_parts) > 1 \
            else w_parts[0]
        gidx = xp.concatenate(idx_parts, axis=1) if len(idx_parts) > 1 \
            else idx_parts[0]
        nvalid = (wloc > 0).sum(axis=1)
        active = xp.nonzero(nvalid > 0)[0]
        ng = int(active.size)
        if ng == 0:
            continue
        max_local = max(max_local, int(nvalid.max()))
        nactive += ng
        nbatch += 1

        wloc = wloc[active]
        gidx = gidx[active]
        gpts = pts[active]

        # ---- phase 2: the batched transform ---------------------------
        # yb: (G, P, R); d: (G, P); winv: (G, P) = localisation / error^2.
        d_parts = []
        yb_parts = []
        off = 0
        for st in stencils:
            n = st["nslots"]
            sub = gidx[:, off:off + n]
            off += n
            s = st["sim"][:, sub.reshape(-1)].reshape(
                members, ng, n).astype(solve_dtype)
            sbar = s.mean(axis=0)
            yb_parts.append(xp.moveaxis(s - sbar[None], 0, 2))   # (G, n, R)
            d_parts.append(
                st["values"][sub.reshape(-1)].reshape(ng, n).astype(
                    solve_dtype) - sbar)
        yb = xp.concatenate(yb_parts, axis=1) if len(yb_parts) > 1 \
            else yb_parts[0]
        dvec = xp.concatenate(d_parts, axis=1) if len(d_parts) > 1 \
            else d_parts[0]

        err2_parts = []
        off = 0
        for st in stencils:
            n = st["nslots"]
            err2_parts.append(
                st["err2"][gidx[:, off:off + n].reshape(-1)].reshape(
                    ng, n).astype(solve_dtype))
            off += n
        err2 = xp.concatenate(err2_parts, axis=1) if len(err2_parts) > 1 \
            else err2_parts[0]

        good = wloc > 0
        winv = xp.where(good, wloc / err2, 0)
        # err2 > 0 is enforced above, so this cannot be a division by zero;
        # what it can still be is an overflow, when a denormal-but-positive
        # variance meets a float32 solve.  Catch it here, where the cause is
        # still nameable, rather than as a non-positive eigenvalue later.
        if not bool(xp.all(xp.isfinite(winv))):
            raise LetkfError(
                "localised inverse observation error weight/sigma^2"
                f" overflowed the {config.solve_dtype} solve.  The"
                " observation errors are finite and positive but small"
                " enough that their reciprocal variance is not"
                " representable; use solve_dtype='float64' (the default) or"
                " rescale the observation units."
            )
        dvec = xp.where(good, dvec, 0)
        yb = xp.where(good[:, :, None], yb, 0)

        # C = Yb^T R^-1 L, (G, R, P).
        cmat = xp.swapaxes(yb, 1, 2) * winv[:, None, :]
        amat = cmat @ yb                                   # (G, R, R)
        amat = amat + scale * ident[None]
        # Symmetrise: C Yb is symmetric analytically, and the rounding that
        # breaks it is exactly what makes eigh's answer depend on which
        # triangle it reads.
        amat = (amat + xp.swapaxes(amat, 1, 2)) * solve_dtype.type(0.5)

        evals, evecs, sweeps = _eigendecompose(xp, amat, eigensolver)
        max_sweeps = max(max_sweeps, sweeps)
        # (R-1)I/rho is positive definite and C Yb is positive semi-definite,
        # so every eigenvalue is >= (R-1)/rho > 0.  A non-positive one means
        # the arithmetic, not the mathematics, failed.
        if not bool(xp.all(evals > 0)):
            raise LetkfError(
                "the localised LETKF matrix (R-1)I/rho + C Yb came back"
                " with a non-positive eigenvalue, which it cannot have"
                " analytically.  Suspect non-finite simulated observations"
                " or an observation error small enough to overflow 1/err^2;"
                f" smallest eigenvalue {float(evals.min())!r}."
            )
        inv = 1.0 / evals
        # Pa~ = U D^-1 U^T and Wa = U D^-1/2 U^T sqrt(R-1) share U, which is
        # the entire reason step 6 is free.
        pa = (evecs * inv[:, None, :]) @ xp.swapaxes(evecs, 1, 2)
        rt = xp.sqrt(inv * solve_dtype.type(members - 1))
        wa = (evecs * rt[:, None, :]) @ xp.swapaxes(evecs, 1, 2)

        wbar = xp.einsum("grp,gp->gr", cmat, dvec)
        wbar = xp.einsum("grs,gs->gr", pa, wbar)           # (G, R)

        # ---- apply to every analysis field ----------------------------
        alpha = solve_dtype.type(config.rtps_alpha)
        for f in fields:
            xbg = xb_flat[f][:, gpts].astype(solve_dtype)   # (R, G)
            dbar = xp.einsum("mg,gm->g", xbg, wbar)         # mean increment
            xa = xp.einsum("mg,gmk->kg", xbg, wa)           # (R, G)
            if config.rtps_alpha > 0.0:
                if config.relaxation == "rtpp":
                    # Zhang et al. (2004): mix the perturbations, not their
                    # amplitudes.  No spread diagnostic, no 0/0 case -- a
                    # perturbation that is identically zero relaxes to
                    # identically zero, which is right.
                    xa = xa * (1 - alpha) + xbg * alpha
                else:
                    sb = xp.sqrt((xbg ** 2).sum(axis=0) / (members - 1))
                    sa = xp.sqrt((xa ** 2).sum(axis=0) / (members - 1))
                    # sigma_a == 0 implies sigma_b == 0 (analysis
                    # perturbations are linear combinations of prior ones),
                    # so the relaxation factor is 0/0 on a perturbation that
                    # is identically zero.  Any finite factor gives the same
                    # answer; 1 is the one that does not need a special case
                    # downstream.
                    relax = xp.where(sa > 0, alpha * (sb - sa) / xp.where(
                        sa > 0, sa, 1) + 1, 1)
                    xa = xa * relax[None, :]
            incr_flat[f][:, gpts] = (
                dbar[None, :] + xa - xbg).astype(work_dtype)

    diagnostics.active_points = nactive
    diagnostics.max_local_obs = max_local
    diagnostics.batches = nbatch
    diagnostics.max_jacobi_sweeps = max_sweeps

    _finish(xp, fields, pri, increments, members, diagnostics)
    return increments
