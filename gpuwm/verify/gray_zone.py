"""The published gray-zone partition: how much turbulence a model at a
given grid spacing is supposed to leave to its closure.

Between the mesoscale, where a boundary-layer scheme carries all the
turbulence, and large-eddy resolution, where the flow carries nearly all
of it, lies a range of grid spacings in which neither is true.  Honnert,
Masson and Couvreux (2011) measured what the split ought to be, by taking
large-eddy simulations and coarse-graining them onto successively coarser
meshes: at each mesh the part of the turbulent energy that survives as
resolved motion is what a model at that spacing could in principle
represent, and the rest is what its closure owes.  The result is a
similarity curve -- one function of grid spacing over boundary-layer
depth -- that dry convective and cumulus-topped boundary layers both
follow.

This module is that curve, and nothing else.  It is a REFERENCE, not a
closure: no model path reads it, it participates in no configuration
identity, and it is deliberately kept out of the closure's own authority
module so that scoring a closure against it can never become fitting the
closure to it.

Authority
---------
Honnert, R., V. Masson and F. Couvreux, 2011: "A Diagnostic for
Evaluating the Representation of Turbulence in Atmospheric Models at the
Kilometric Scale."  J. Atmos. Sci. 68, 3112-3131,
doi:10.1175/JAS-D-11-061.1.  Equations (9)-(14), transcribed from the
paper.  Independently corroborated against WRF's own transcription of
the same equations in ``phys/module_bl_mynn.F`` subroutine
``SCALE_AWARE`` (the commented reference block), which agrees term for
term.

Honnert, R., and eleven co-authors, 2020: "The Atmospheric Boundary
Layer and the Gray Zone of Turbulence: A Critical Review."  J. Geophys.
Res. Atmos. 125, e2019JD030317.  Restates the abscissa as
Delta/(z_i + z_c) and the gray-zone bounds used below; revises none of
the 2011 coefficients.

What the curve is
-----------------
The abscissa is ``x = Delta/(h + h_c)`` -- grid spacing over the sum of
the boundary-layer depth and the depth of any cumulus layer above it
(``h_c = 0`` for a dry boundary layer).  The ordinate is the SUBGRID
fraction of the quantity: the paper fits only the subgrid functions and
states that the resolved fraction is one minus them.

The functional form is rational and NOT a hyperbolic tangent.  The
``x**(2/3)`` term is derived rather than fitted: integrating a
Kolmogorov energy spectrum above the filter wavenumber gives a subgrid
energy proportional to ``Delta**(2/3)``, so the curve must approach zero
that way.  That matters for a reader's expectations -- it decays SLOWLY.
At ``x = 0.1`` the subgrid fraction is still 0.22, not zero; five per
cent subgrid is not reached until ``x = 0.013``, roughly 13 m for a
kilometre-deep boundary layer.  The half-and-half point sits at
``x = 0.27``, not at 1: a model whose grid spacing equals its
boundary-layer depth is already 88% subgrid.
"""

from __future__ import annotations

import numpy as np

#: Gray-zone bounds in ``Delta/(h + h_c)`` (Honnert et al. 2020 p. 4;
#: Honnert and Masson 2014).  Below this the flow is essentially
#: resolved, above it essentially parameterized.
GRAY_ZONE = (0.2, 2.0)

#: Honnert et al. (2011) eq. (9): SUBGRID TURBULENT KINETIC ENERGY in
#: the MIXED LAYER, valid 0.05 <= z/h <= 0.85.  Coefficients exactly as
#: published (0.07, 3/21, 3/42).
TKE_MIXED_LAYER = ((0.07,), (3.0 / 21.0, 3.0 / 42.0))
#: Honnert et al. (2011) eq. (10): subgrid TKE in the ENTRAINMENT ZONE,
#: valid 0.85 <= z/h <= 1.1 (4/21, 3/20, 7/21).
TKE_ENTRAINMENT = ((4.0 / 21.0,), (3.0 / 20.0, 7.0 / 21.0))

#: The scatter envelopes published beside each fit, ``a*exp(-(ln x -
#: b)^2/c)``, as ``(a, b, c)``.  These are the first and last
#: vigintiles of the coarse-grained data, i.e. the width of the cloud
#: the central curve was drawn through -- NOT a confidence interval on
#: the fit.  A scoring band built from them is a statement about how
#: much spread the observations themselves carry.
#:
#: NOTE ON THE LOGARITHM BASE, recorded because the paper does not state
#: it: natural log is INFERRED, not verified, from the fact that under
#: ``ln`` the envelope peak lands on the curve's own 50% crossover for
#: every equation that has one, while under log10 it would peak an order
#: of magnitude outside the data range.
TKE_MIXED_LAYER_ENVELOPE = (0.12, -1.9, 5.0)
TKE_ENTRAINMENT_ENVELOPE = (0.15, -1.5, 3.0)


def subgrid_tke_fraction(delta_over_h, entrainment_zone: bool = False):
    """Honnert et al. (2011) eq. (9) (or eq. (10) in the entrainment
    zone): the fraction of turbulent kinetic energy a model at this
    spacing is expected to carry in its closure.

        P(x) = (x^2 + a1*x^(2/3)) / (x^2 + b1*x^(2/3) + b2)

    with ``x = Delta/(h + h_c)``.  Returns values in [0, 1]; the
    complement is the resolved fraction, which the paper defines exactly
    that way rather than fitting it.
    """
    x = np.asarray(delta_over_h, dtype=np.float64)
    (a1,), (b1, b2) = (TKE_ENTRAINMENT if entrainment_zone
                       else TKE_MIXED_LAYER)
    x23 = x ** (2.0 / 3.0)
    return (x * x + a1 * x23) / (x * x + b1 * x23 + b2)


def subgrid_tke_envelope(delta_over_h, entrainment_zone: bool = False):
    """The published scatter envelope as a ``(lo, hi)`` pair.

    ``a*exp(-(ln x - b)^2/c)`` about the central curve, clipped to
    [0, 1].  See the note at the constants: this is the spread of the
    coarse-grained large-eddy data, so a model inside it is
    indistinguishable from the observations the curve was fitted to,
    and a model outside it disagrees with the data and not merely with
    a fit.
    """
    x = np.asarray(delta_over_h, dtype=np.float64)
    a, b, c = (TKE_ENTRAINMENT_ENVELOPE if entrainment_zone
               else TKE_MIXED_LAYER_ENVELOPE)
    centre = subgrid_tke_fraction(x, entrainment_zone)
    half = a * np.exp(-((np.log(x) - b) ** 2) / c)
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half,
                                                     0.0, 1.0)


def crossover(entrainment_zone: bool = False, lo=1e-4, hi=1e4) -> float:
    """The spacing ratio at which the split is half and half.

    Bisected on the curve itself rather than quoted, so it moves with
    the coefficients if they are ever corrected.  Mixed layer: 0.27.
    """
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = ((mid, hi) if subgrid_tke_fraction(mid, entrainment_zone)
                  < 0.5 else (lo, mid))
    return float(np.sqrt(lo * hi))


def partition_from_profiles(e_subgrid, u, v, w) -> dict[str, float]:
    """Measure a model's OWN partition from its fields, Honnert's way.

    Honnert et al. eq. (7) defines the resolved turbulent energy on a
    mesh as the variance of the mesh-resolved velocity about its
    horizontal mean,

        e_res = 0.5*<(u - <u>)^2 + (v - <v>)^2 + (w - <w>)^2>,

    with ``<>`` a horizontal average; the subgrid part is what the
    closure carries.  Applied to a model run rather than to a
    coarse-grained large-eddy field, this is the same quantity: what
    the model resolved, against what its closure held.

    Inputs are ``(nz, ny, nx)`` fields on their mass points.  Returns
    the two energies and the subgrid fraction, per level and column-
    integrated.  NO WINDOW OR DETREND IS APPLIED -- this is a plain
    variance about the plane mean, so it needs no Parseval check and
    cannot be biased by a window that is invalid on organised fields.
    """
    def var(field):
        f = np.asarray(field, dtype=np.float64)
        return np.mean((f - f.mean(axis=(1, 2), keepdims=True)) ** 2,
                       axis=(1, 2))

    e_res = 0.5 * (var(u) + var(v) + var(w))
    e_sgs = np.asarray(e_subgrid, dtype=np.float64).mean(axis=(1, 2))
    total = e_res + e_sgs
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0.0, e_sgs / total, np.nan)
    return {"e_resolved": e_res, "e_subgrid": e_sgs,
            "subgrid_fraction": frac,
            "e_resolved_column": float(np.sum(e_res)),
            "e_subgrid_column": float(np.sum(e_sgs)),
            "subgrid_fraction_column": (
                float(np.sum(e_sgs) / np.sum(total))
                if np.sum(total) > 0.0 else float("nan"))}
