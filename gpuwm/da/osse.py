"""OSSE twin experiment: the gate on :mod:`gpuwm.da.letkf`.  EXPERIMENTAL.

An observing-system simulation experiment is the only test that can tell you
a Kalman filter *works*, as opposed to *runs*.  Unit tests on Gaspari-Cohn
values and on CuPy-versus-NumPy agreement check that the pieces do what they
say; they would all pass with a sign error in the innovation, which would
make the filter reliably worse than doing nothing.  The twin experiment
catches that, because it is the only test that knows the truth.

The design is deliberately model-free.  No dycore, no physics, no wrfout: a
truth state is a smoothed Gaussian random field, the prior ensemble is that
truth plus smoothed random error plus a bias, and the observation operator is
supplied here.  That keeps the gate runnable in seconds on any machine, keeps
it from failing for reasons that have nothing to do with the filter, and --
most importantly -- makes the truth *exactly* known, which is what makes the
RMSE numbers mean something.

What the gate asserts (see ``tests/test_letkf_osse.py``):

1. the ensemble-mean RMSE against truth falls, for the observed field;
2. it also falls for fields that are NOT observed, which is the multivariate
   update working through ensemble cross-covariance -- the property that
   distinguishes an EnKF from an optimal interpolation scheme;
3. the ensemble spread falls but stays strictly positive (a filter that
   collapses the spread has stopped being an ensemble);
4. with no observations, the increments are exactly zero -- the no-DA
   control;
5. increments are exactly zero beyond the localisation cutoff.

Two observation operators are provided.  ``"direct"`` observes one state
field pointwise; ``"speed"`` observes ``sqrt(u^2 + v^2)``, which is
genuinely nonlinear and exercises the fact that the filter linearises H
through the ensemble rather than requiring a tangent-linear operator.

Runnable as ``python -m gpuwm.da.osse``.  It is not wired into the gpuwm CLI
and does not read or write any campaign artefact.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from gpuwm.da.letkf import (
    GridGeometry,
    GriddedObs,
    LetkfConfig,
    LetkfDiagnostics,
    Localization,
    analyze,
)

__all__ = [
    "OsseSetup",
    "OsseResult",
    "smooth_random_field",
    "build_twin",
    "run_osse",
    "run_cycling_osse",
    "main",
]


# ---------------------------------------------------------------------------
# Synthetic fields
# ---------------------------------------------------------------------------

def smooth_random_field(rng, shape, length_points, xp=np):
    """A zero-mean, unit-variance field with a spatial correlation length.

    White noise convolved with a Gaussian, via a separable pass in each
    horizontal direction and a shorter one in the vertical.  Spatial
    correlation is not decoration here: an EnKF's whole mechanism is
    covariance between neighbouring points and between variables, and white
    noise has neither.  A twin experiment built on white noise would show a
    filter that only corrects gridpoints it directly observes, and would
    "pass" a localisation bug that zeroed every off-diagonal covariance.

    ``length_points`` is the Gaussian standard deviation in gridpoints,
    horizontally; the vertical uses half of it, since model levels are far
    closer together than columns in any real configuration.
    """
    nz, ny, nx = shape
    a = xp.asarray(rng.standard_normal(shape))

    def kernel(sigma, n):
        half = max(1, int(math.ceil(3 * sigma)))
        half = min(half, max(1, n - 1))
        t = xp.arange(-half, half + 1, dtype=xp.float64)
        k = xp.exp(-0.5 * (t / sigma) ** 2)
        return k / k.sum()

    def blur(arr, axis, sigma, n):
        if sigma <= 0 or n < 3:
            return arr
        k = kernel(sigma, n)
        out = xp.zeros_like(arr)
        half = (k.size - 1) // 2
        for off in range(-half, half + 1):
            # Reflect at the boundary: wrapping would impose a periodicity
            # the localisation stencil does not have, and would put spurious
            # correlation between opposite edges of the domain.
            idx = xp.clip(xp.arange(n) + off, 0, n - 1)
            out = out + k[off + half] * xp.take(arr, idx, axis=axis)
        return out

    a = blur(a, 2, length_points, nx)
    a = blur(a, 1, length_points, ny)
    a = blur(a, 0, max(0.5, length_points / 2.0), nz)
    a = a - a.mean()
    s = float(xp.sqrt((a ** 2).mean()))
    return a / s if s > 0 else a


@dataclass(frozen=True)
class OsseSetup:
    """Everything that defines one twin experiment."""

    nz: int = 8
    ny: int = 32
    nx: int = 32
    members: int = 20
    dx_m: float = 2000.0
    dy_m: float = 2000.0
    dz_m: float = 500.0
    #: Correlation length of truth and of the background error, gridpoints.
    truth_length: float = 4.0
    error_length: float = 3.0
    #: Background error amplitude, as a multiple of each field's own scale.
    error_amplitude: float = 1.0
    #: Fraction of gridpoints carrying an observation.
    obs_fraction: float = 0.02
    obs_error: float = 0.5
    horizontal_cutoff_m: float = 8000.0
    vertical_cutoff_m: float = 2000.0
    #: Relaxation to prior spread.  The filter itself refuses to guess this
    #: (``LetkfConfig.rtps_alpha`` is required); a twin experiment is a
    #: place to sweep it, so it carries a stated storm-scale starting point
    #: rather than the silent 0.0 that means "no relaxation at all".  Set it
    #: to 0.0 explicitly to run the no-relaxation control.
    rtps_alpha: float = 0.9
    prior_inflation: float = 1.0
    seed: int = 20260730
    #: "direct" observes theta; "speed" observes sqrt(u^2+v^2); "both" does
    #: both, with the two types carrying different localisation radii.
    operator: str = "direct"


#: Field scales, so the synthetic state is not dimensionless nonsense.
#: theta [K], qv [kg/kg], u/v/w [m/s].
_FIELD_SCALE = {
    "theta": 2.0,
    "qv": 1.0e-3,
    "u": 3.0,
    "v": 3.0,
    "w": 0.5,
}
_FIELD_BASE = {
    "theta": 300.0,
    "qv": 8.0e-3,
    "u": 5.0,
    "v": -2.0,
    "w": 0.0,
}
FIELDS = ("theta", "qv", "u", "v", "w")


def _h_direct(state):
    """Identity on theta."""
    return state["theta"]


def _h_speed(state):
    """Horizontal wind speed: nonlinear in the state, no tangent linear."""
    return np.sqrt(state["u"] ** 2 + state["v"] ** 2)


_OPERATORS = {"direct": _h_direct, "speed": _h_speed}


def build_twin(setup: OsseSetup):
    """Truth, prior ensemble, and the grid they live on.

    The prior ensemble is truth + a common bias + per-member spread, all
    spatially correlated.  The common bias matters: an ensemble scattered
    symmetrically about the truth already has the right mean, and a filter
    that did nothing at all would score perfectly on ensemble-mean RMSE.
    The bias is what gives the analysis something to correct.

    The five fields share a correlation pattern with independent components
    mixed in, so that cross-variable covariance is real but not degenerate.
    Without shared structure, "did the unobserved fields improve" would be
    testing nothing.
    """
    rng = np.random.default_rng(setup.seed)
    shape = (setup.nz, setup.ny, setup.nx)
    grid = GridGeometry(
        dx_m=setup.dx_m,
        dy_m=setup.dy_m,
        heights_m=(np.arange(setup.nz) + 0.5) * setup.dz_m,
    )

    shared = smooth_random_field(rng, shape, setup.truth_length)
    truth = {}
    for f in FIELDS:
        own = smooth_random_field(rng, shape, setup.truth_length)
        pattern = 0.6 * shared + 0.8 * own
        truth[f] = _FIELD_BASE[f] + _FIELD_SCALE[f] * pattern

    shared_bias = smooth_random_field(rng, shape, setup.error_length)
    bias = {}
    for f in FIELDS:
        own_bias = smooth_random_field(rng, shape, setup.error_length)
        bias[f] = 0.7 * shared_bias + 0.7 * own_bias

    # The member loop is OUTSIDE the field loop, and the shared component is
    # drawn once per member rather than once per member per field.  This is
    # the whole difference between a twin experiment that tests the
    # multivariate update and one that cannot.  With a per-field draw the
    # ensemble has no cross-variable covariance at all, every off-diagonal
    # block of Pb is pure sampling noise, and assimilating theta makes qv, u,
    # v and w measurably WORSE -- which is the correct behaviour of a correct
    # filter fed an ensemble that knows nothing.  Measured on the first
    # version of this file: theta -29.8% RMSE, everything else +2 to +3%.
    perts = {f: [] for f in FIELDS}
    for _ in range(setup.members):
        shared_m = smooth_random_field(rng, shape, setup.error_length)
        for f in FIELDS:
            own_m = smooth_random_field(rng, shape, setup.error_length)
            perts[f].append(0.7 * shared_m + 0.7 * own_m)

    prior = {}
    for f in FIELDS:
        amp = setup.error_amplitude * _FIELD_SCALE[f]
        prior[f] = np.stack(
            [truth[f] + amp * bias[f] + amp * p for p in perts[f]], axis=0
        )
    return truth, prior, grid


def _make_obs(setup, truth, prior, grid, rng, name, operator, loc):
    """One gridded observation batch from the truth, with noise."""
    h = _OPERATORS[operator]
    shape = truth["theta"].shape
    y_true = h(truth)
    mask = rng.random(shape) < setup.obs_fraction
    values = y_true + setup.obs_error * rng.standard_normal(shape)
    sim = np.stack(
        [h({f: prior[f][k] for f in prior}) for k in range(setup.members)],
        axis=0,
    )
    return GriddedObs(
        name=name,
        values=np.where(mask, values, np.nan),   # unmasked garbage on purpose
        errors=setup.obs_error,
        simulated=sim,
        mask=mask,
        localization=loc,
    ), int(mask.sum())


@dataclass
class OsseResult:
    """Scores.  RMSE is against the truth the twin experiment knows."""

    n_obs: int
    members: int
    prior_rmse: dict
    analysis_rmse: dict
    prior_spread: dict
    analysis_spread: dict
    diagnostics: LetkfDiagnostics

    def improvement(self, field_name: str) -> float:
        """Fractional RMSE reduction; positive is better."""
        p = self.prior_rmse[field_name]
        return (p - self.analysis_rmse[field_name]) / p if p > 0 else 0.0

    def report(self) -> str:
        lines = [
            f"OSSE  R={self.members}  obs={self.n_obs}"
            f"  active gridpoints={self.diagnostics.active_points}"
            f"/{self.diagnostics.total_points}"
            f"  stencil slots={self.diagnostics.stencil_slots}"
            f"  max local obs={self.diagnostics.max_local_obs}",
            f"{'field':>8}  {'RMSE prior':>12} {'RMSE anal':>12}"
            f" {'change':>9}  {'spread b':>11} {'spread a':>11}",
        ]
        for f in self.prior_rmse:
            lines.append(
                f"{f:>8}  {self.prior_rmse[f]:12.6g}"
                f" {self.analysis_rmse[f]:12.6g}"
                f" {100 * self.improvement(f):8.2f}%"
                f"  {self.prior_spread[f]:11.5g}"
                f" {self.analysis_spread[f]:11.5g}"
            )
        return "\n".join(lines)


def _rmse(field_ens, truth_field):
    return float(np.sqrt(((field_ens.mean(axis=0) - truth_field) ** 2).mean()))


def _spread(field_ens):
    r = field_ens.shape[0]
    m = field_ens.mean(axis=0, keepdims=True)
    return float(np.sqrt(((field_ens - m) ** 2).sum(axis=0) / (r - 1)).mean())


def run_osse(setup: OsseSetup | None = None, xp=np) -> OsseResult:
    """Build a twin, assimilate once, score against truth."""
    setup = setup or OsseSetup()
    truth, prior, grid = build_twin(setup)
    rng = np.random.default_rng(setup.seed + 1)

    default_loc = Localization(
        horizontal_m=setup.horizontal_cutoff_m,
        vertical_m=setup.vertical_cutoff_m,
    )
    batches = []
    n_obs = 0
    if setup.operator in ("direct", "both"):
        b, n = _make_obs(setup, truth, prior, grid, rng, "theta_point",
                         "direct", None)
        batches.append(b)
        n_obs += n
    if setup.operator in ("speed", "both"):
        # A deliberately different radius, to exercise the per-type override
        # and the multi-stencil concatenation it forces.
        loc = None if setup.operator == "speed" else Localization(
            horizontal_m=setup.horizontal_cutoff_m * 0.75,
            vertical_m=setup.vertical_cutoff_m,
        )
        b, n = _make_obs(setup, truth, prior, grid, rng, "wind_speed",
                         "speed", loc)
        batches.append(b)
        n_obs += n

    config = LetkfConfig(
        localization=default_loc,
        analysis_fields=FIELDS,
        rtps_alpha=setup.rtps_alpha,
        prior_inflation=setup.prior_inflation,
    )
    diag = LetkfDiagnostics()
    if xp is not np:
        prior = {f: xp.asarray(v) for f, v in prior.items()}
        batches = [
            GriddedObs(
                name=b.name,
                values=xp.asarray(b.values),
                errors=b.errors,
                simulated=xp.asarray(b.simulated),
                mask=xp.asarray(b.mask),
                localization=b.localization,
            )
            for b in batches
        ]
    inc = analyze(prior, batches, grid, config, diagnostics=diag)

    post = {f: prior[f] + inc[f] for f in FIELDS}
    if xp is not np:
        prior = {f: xp.asnumpy(v) for f, v in prior.items()}
        post = {f: xp.asnumpy(v) for f, v in post.items()}
    return OsseResult(
        n_obs=n_obs,
        members=setup.members,
        prior_rmse={f: _rmse(prior[f], truth[f]) for f in FIELDS},
        analysis_rmse={f: _rmse(post[f], truth[f]) for f in FIELDS},
        prior_spread={f: _spread(prior[f]) for f in FIELDS},
        analysis_spread={f: _spread(post[f]) for f in FIELDS},
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# Cycling
# ---------------------------------------------------------------------------

def _advect(state_or_ens, shift_j: int, shift_i: int):
    """Uniform advection by whole gridpoints, periodic.

    The crudest forecast model that still couples cycles: it moves error
    around so that cycle N's analysis is not cycle N+1's prior in the same
    place.  A cycling test whose "model" is the identity cannot detect the
    failure mode cycling exists to detect -- an analysis that is fine once
    and diverges when fed back to itself.
    """
    out = np.roll(state_or_ens, shift=shift_j, axis=-2)
    return np.roll(out, shift=shift_i, axis=-1)


def run_cycling_osse(setup: OsseSetup | None = None, cycles: int = 3):
    """Assimilate ``cycles`` times, advecting truth and ensemble between.

    Returns the per-cycle :class:`OsseResult`-like scores.  The property
    worth watching is that RMSE does not creep back up: an
    under-inflated filter reduces error on cycle 1 and then loses the
    ensemble spread it needs to keep doing so.
    """
    setup = setup or OsseSetup()
    truth, ens, grid = build_twin(setup)
    rng = np.random.default_rng(setup.seed + 7)
    default_loc = Localization(
        horizontal_m=setup.horizontal_cutoff_m,
        vertical_m=setup.vertical_cutoff_m,
    )
    config = LetkfConfig(
        localization=default_loc,
        analysis_fields=FIELDS,
        rtps_alpha=setup.rtps_alpha,
        prior_inflation=setup.prior_inflation,
    )
    history = []
    for c in range(cycles):
        b, n = _make_obs(setup, truth, ens, grid, rng,
                         f"cycle{c}", setup.operator
                         if setup.operator != "both" else "direct", None)
        diag = LetkfDiagnostics()
        inc = analyze(ens, [b], grid, config, diagnostics=diag)
        post = {f: ens[f] + inc[f] for f in FIELDS}
        history.append(OsseResult(
            n_obs=n,
            members=setup.members,
            prior_rmse={f: _rmse(ens[f], truth[f]) for f in FIELDS},
            analysis_rmse={f: _rmse(post[f], truth[f]) for f in FIELDS},
            prior_spread={f: _spread(ens[f]) for f in FIELDS},
            analysis_spread={f: _spread(post[f]) for f in FIELDS},
            diagnostics=diag,
        ))
        truth = {f: _advect(truth[f], 1, 1) for f in FIELDS}
        ens = {f: _advect(post[f], 1, 1) for f in FIELDS}
    return history


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m gpuwm.da.osse",
        description="LETKF twin experiment (EXPERIMENTAL; no campaign I/O)",
    )
    p.add_argument("--members", type=int, default=OsseSetup.members)
    p.add_argument("--nx", type=int, default=OsseSetup.nx)
    p.add_argument("--ny", type=int, default=OsseSetup.ny)
    p.add_argument("--nz", type=int, default=OsseSetup.nz)
    p.add_argument("--obs-fraction", type=float,
                   default=OsseSetup.obs_fraction)
    p.add_argument("--rtps-alpha", type=float, default=OsseSetup.rtps_alpha)
    p.add_argument("--operator", choices=("direct", "speed", "both"),
                   default=OsseSetup.operator)
    p.add_argument("--cycles", type=int, default=1,
                   help="more than 1 runs the cycling experiment")
    p.add_argument("--seed", type=int, default=OsseSetup.seed)
    p.add_argument("--gpu", action="store_true",
                   help="run the analysis through cupy")
    a = p.parse_args(argv)

    setup = OsseSetup(
        nz=a.nz, ny=a.ny, nx=a.nx, members=a.members,
        obs_fraction=a.obs_fraction, rtps_alpha=a.rtps_alpha,
        operator=a.operator, seed=a.seed,
    )
    if a.cycles > 1:
        for i, r in enumerate(run_cycling_osse(setup, cycles=a.cycles)):
            print(f"--- cycle {i} ---")
            print(r.report())
        return 0
    xp = np
    if a.gpu:
        import cupy
        xp = cupy
    print(run_osse(setup, xp=xp).report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
