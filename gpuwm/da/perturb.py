"""Initial-condition perturbations for storm-scale ensembles (experimental).

A 30-member ensemble whose members differ only by their random seed is not an
ensemble; it is thirty runs of the same forecast.  Spread has to be *put* into
the initial condition, and it has to be put in at scales the model can carry:
white noise on the grid is annihilated by the first diffusion call, and a
domain-wide offset is a bias, not a perturbation.  This module draws smooth
Gaussian random fields with a prescribed horizontal and vertical correlation
length, tapers them to zero against the lateral boundary so a member's
interior can diverge while its boundary forcing stays the unperturbed one,
and adds them to the prognostic state under hard physical bounds.

Construction
------------
For each field an independent unit-variance white-noise draw ``w`` is filtered
in spectral space by a separable Gaussian kernel

    ``H(k) = exp(-(kx^2 + ky^2) Lh^2 / 4) * exp(-kz^2 Lv^2 / 4)``

so the realized field has the Gaussian correlation ``C(r) = exp(-r^2/2L^2)``
(the autocorrelation of a Gaussian of width ``L/sqrt(2)``).  Its *radially
binned* power spectrum, which carries the ``2 pi k`` annulus Jacobian, is
therefore ``E(k) ~ k exp(-k^2 Lh^2 / 2)`` and peaks at ``k = 1/Lh`` exactly.
That peak is the module's contract with :func:`radial_power_spectrum` and is
what ``tests/test_da_perturb.py`` measures; a spectrum that peaks somewhere
else means the prescribed length scale is not the length scale you got.

Amplitudes are normalized *analytically*, not by dividing out the sample
standard deviation.  With unit-variance white noise and NumPy's DFT
convention the field variance is exactly ``mean_k |H(k)|^2``, and because
``H`` is separable that mean is a product of three cheap 1-D means.  Dividing
by the sample standard deviation would force every member to the same
realized variance, which is wrong: sample variance is supposed to fluctuate
between members.  The realized RMS of each draw is recorded in the provenance
instead.

Balance (2026-10-06: what the member start now keeps)
-----------------------------------------------------
Two measurements drove this.  The DA lanes found the members' spread well
below O-B against every conventional observation type one analysis in
(about 2x in temperature, 1.5-2.5x in wind, 3-5x in dewpoint once the
observation error and the posterior-versus-prior spread are accounted for),
and the members' surface pressure tendency 2-3x the control's over their
first 30 minutes before any analysis had touched them.  The second has its
cause here: the leg-0 perturbation was unbalanced.  Two draws of ``u`` and
``v`` that know nothing of each other put half their kinetic energy into
divergence, which radiates; a theta perturbation under a fixed geopotential
reads as a pressure jump of about ``gamma * dtheta / theta`` (4 hPa per
kelvin), which rings.  The first does NOT (measured 2026-10-06 on a 3 km
crop, 4 members per arm, 60 min): balancing the start cut the insertion
pressure jump 28x and the members' 30-minute surface-pressure noise from
2.1x to 1.3x the control's, but the retained spread moved by only 1-4
points (wind 0.81 to 0.83 at 30 min; theta and vapour unchanged).  The
spread deficit is the configured draw amplitude with nothing in the cycle
growing it, a separate question this module's balance does not answer.

* **Wind: non-divergent by construction** (``wind_mode = "rotational"``,
  the default).  ONE streamfunction is drawn on the corner grid and the
  ``u``/``v`` increments are its C-grid differences, so the model's own
  divergence stencil sees exactly zero at every mass point where the rim
  taper is one.  Each component then takes the scalar rim taper, which
  leaves a divergence of order ``u * grad(taper)`` inside the rim band
  only (the independent draws always carried that term there too); the
  taper is NOT put on the streamfunction, because a streamfunction that
  vanishes across a 5-cell rim at 150 km scale is a 35x tangential wind
  along the boundary (measured, 2026-10-06; see
  :func:`rotational_wind_draw`).  A non-divergent perturbation at scales
  below the Rossby radius is the part geostrophic adjustment keeps; the
  mass field adjusts to it instead of the other way round.  The
  streamfunction's own scale is ``sqrt(3)`` times the configured one,
  which is exactly what puts the WIND's radial spectrum peak at
  ``k = 1/L``, the module's length-scale contract.  ``"independent"``
  keeps the old unrelated draws as a labelled comparison arm.
* **Mass: hydrostatic** (``mass_balance = "hydrostatic"``, the default).
  After the thermodynamic perturbations the column geopotential ``php`` is
  re-integrated at the column's own dry mass by
  :mod:`gpuwm.da.hydrostatic` (the same operator the analysis insertion
  uses), so the perturbed column starts hydrostatic.  ``mu'`` is untouched.
  It needs the run's ``hypsometric_opt`` and a state with a loaded base;
  a state without one must say ``mass_balance = "none"`` rather than get a
  silent zero.

What this still does NOT do (stated plainly)
--------------------------------------------
* **No geostrophic or gradient-wind coupling.**  The streamfunction and
  the theta draw are independent: the wind is balanced with itself, not
  with the temperature.  The thermal-wind tie (theta from the vertical
  shear of the streamfunction) is the next step and is not here.
* **The map factor is not seen.**  Non-divergence is exact on the index
  stencil; on a projected grid the residual divergence is of the order
  of the map factor's variation across the domain.
* **The boundary forcing is perturbed by a separate call.**  The initial
  perturbation is tapered to zero at the rim so it agrees with the
  boundary at the start; :func:`perturbed_lateral_boundaries` then gives
  each member its own boundary tables, ramping from that agreement into the
  member's own draw over the first boundary interval, so spread no longer
  decays toward the rim as the shared inflow floods the domain.  A caller
  that never calls it keeps the shared boundary, and the rim taper is then
  the whole boundary story.  :func:`recycled_difference_perturbations` is a
  documented stub, not code.
* **No perturbation of surface, soil, or physics parameters**, and no
  perturbation of ``w`` or ``mu'``.
* **The initial-condition call has no vertical taper.**  Its perturbations
  reach the model top and surface at full amplitude.  The separate
  :func:`vertical_taper` helper lets a spread-maintenance policy prescribe
  its own surface and model-top attenuation.
* **No flow dependence.**  The correlation length is prescribed and uniform,
  not derived from the analysis error covariance of the day.
* **The horizontal draw is periodic; the physical vertical column is a
  crop of a larger independent-noise draw.**  Each vertical end has a halo
  of at least four correlation lengths, so the nearest periodic image of
  the opposite physical endpoint is more than eight correlation lengths
  away.  The old quarter-column cap did not repair the seam and is retired.
  Long vertical scales may now produce physically coherent columns rather
  than being refused.  The analytic normalization uses the padded spectrum,
  and every draw reports the exact cropped top-to-bottom and half-column
  correlations in ``provenance["fields"][i]["vertical_wrap"]``.  No sample
  normalization or repeated physical column is used.

Multiplicative perturbations, and why the hydrometeors get them
--------------------------------------------------------------
An *additive* Gaussian increment is the wrong instrument for a mixing
ratio.  It is unbounded below, so it manufactures negative mass that has
to be clipped (which adds mass, wetward, everywhere it fires); it is
unbounded above in a variable that spans six orders of magnitude between
cirrus and a hail core, so one amplitude is either meaningless aloft or
absurd in the core; and for a two-moment scheme it breaks the pair --
moving ``qr`` while leaving ``nr`` produces exactly the ``q > 0, N = 0``
cell whose slope closure evaluates to NaN (see :mod:`gpuwm.da.moments`).

So a hydrometeor species is perturbed **multiplicatively**, by a
lognormal factor drawn from the same smooth spectrum as everything else:

    ``f = exp(sigma_ln * clip(g, -k, +k) * taper - c)``,  ``g`` unit-variance,

with ``c = log E[f]`` of the uncorrected factor per column
(:func:`clipped_lognormal_log_mean`), so that ``E[f] = 1`` exactly: the
ensemble MEAN carries the background's mass, and the members carry the
spread.  Without ``c`` the mean took ``exp(sigma_ln^2 / 2)``, +27% at 0.7,
measured on the module's own draw (2026-10-06), which every analysis then
started from.  That ONE factor multiplies every prognostic moment of the species --
mass, number, and NSSL's predicted volume.  Four properties follow, and
each is a property rather than a hope:

* **Positivity is exact, with no clipping.**  ``f > 0`` for any finite
  exponent, so ``q >= 0`` implies ``q f >= 0`` in IEEE arithmetic.  No
  mass is added and none is removed by a repair, because there is no
  repair.
* **Moment consistency is exact.**  ``q`` and ``N`` are scaled by the
  same number, so ``q > 0 <=> q f > 0`` and ``N > 0 <=> N f > 0``: the
  perturbation cannot create a depleted pair.  It is applied only where
  the background pair is *jointly* active (``q`` above the scheme's own
  activity threshold AND ``N > 0``), which closes the one remaining
  route -- a background cell already holding ``q > 0`` with ``N = 0``
  below the threshold, which scaling up would have promoted into an
  offender.
* **The drop size distribution is preserved.**  Morrison's slope is
  ``lam = (six_c N / q)^(1/3)``; the ratio ``q/N`` is invariant under a
  common factor, so ``lam`` -- and with it the mean-mass diameter and
  the scheme's own size-dependent process rates -- is exactly unchanged.
  What moves is the number density, and therefore ``Z``, by
  ``10 log10 f`` dB.  A perturbation that moved ``q`` alone would move
  the particle *size*, which is a much stronger and much less defensible
  claim about what the background got wrong.
* **Clear air stays clear.**  Where the species is absent the factor is
  never applied, so the ensemble does not invent storms the model never
  made.  That is a real limitation as well as a virtue and it is stated
  in the provenance: the filter can move echo that exists and cannot
  create echo that does not.

``sigma_ln`` is a *fractional* amplitude: 0.7 is a factor of two at one
sigma and about 3 dB of reflectivity spread, which is the order of a
storm-scale hydrometeor uncertainty.  ``clip_sigmas`` bounds the tail so
no member gets a factor of a thousand from a three-in-a-million draw.

The same mode is available for ``qv`` through
``FieldPerturbation(mode="lognormal")``, for the second reason above:
one additive amplitude cannot be right for a 15 g/kg boundary layer and
a 0.01 g/kg upper troposphere at once, and a fractional one is.
Signed fields refuse the mode outright -- a lognormal factor on a
quantity that changes sign is not a perturbation, it is a bug.

Nothing here is on a certified forecast path.  Every provenance dict this
module returns carries ``status="experimental"``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gpuwm.core import constants as c
from gpuwm.da import STATUS

__all__ = [
    "FieldPerturbation",
    "SpeciesPerturbation",
    "PerturbationConfig",
    "SUPPORTED_FIELDS",
    "SUPPORTED_SPECIES",
    "PERTURBATION_MODES",
    "PROVENANCE_SCHEMA",
    "apply_perturbations",
    "gaussian_random_field",
    "boundary_taper",
    "vertical_taper",
    "radial_power_spectrum",
    "spectral_peak_wavenumber",
    "fit_gaussian_length_scale",
    "default_array_module",
    "recycled_difference_perturbations",
    "perturbed_lateral_boundaries",
    "boundary_coupling_weights",
    "boundary_perturbation_unavailable",
    "additive_inflation",
    "echo_weight",
    "rotational_wind_draw",
    "WIND_MODES",
    "MASS_BALANCE_MODES",
    "STREAMFUNCTION_SCALE_FACTOR",
    "ECHO_NOISE_THRESHOLD_DBZ",
    "BOUNDARY_PERTURBATION_SCHEMA",
    "ADDITIVE_INFLATION_SCHEMA",
    "DEFAULT_BOUNDARY_TIME_SCALE_HOURS",
]

#: Provenance schema identifier.  Bump on any change to the emitted keys.
PROVENANCE_SCHEMA = "gpuwm.da.perturb/provenance/v2"

#: Domain-separation string mixed into every random stream key, so a seed
#: reused by an unrelated module cannot reproduce these draws.
_STREAM_DOMAIN = "gpuwm.da.perturb/noise/v1"

#: A prescribed length scale below this many grid spacings is not resolved.
_MIN_SCALE_CELLS = 2.0

#: A horizontal length scale above this fraction of the shorter domain span
#: makes the "prescribed length scale" meaningless -- the field is then a
#: domain-wide offset with a few wiggles, and its spectrum has no peak inside
#: the resolved band.
#:
#: The fraction is ``1/(2 pi)``, not the quarter this used to admit.  The
#: documented contract is that the radial spectrum peaks at ``k = 1/L``; on a
#: periodic span ``S`` the lowest nonzero angular wavenumber is ``2 pi / S``,
#: so that peak is resolved only when ``1/L >= 2 pi / S``, i.e.
#: ``L <= S / (2 pi) ~ 0.159 S``.  At the old quarter-span limit probes on
#: 32-, 64- and 128-point domains all measured ``peak * L = 2.356`` instead of
#: 1: the peak had fallen below the fundamental and the estimator was reading
#: the fundamental back.  This is a tightening, and configurations sitting
#: between 0.159 and 0.25 of the span are now refused rather than quietly
#: given a domain-wide offset.
_MAX_HORIZONTAL_SPAN_FRACTION = 1.0 / (2.0 * math.pi)
#: Independent-noise halo on EACH end of the physical column.  The nearest
#: periodic image of the opposite endpoint is then more than 8 Lv away,
#: giving a Gaussian image contribution below exp(-32).  The exact discrete
#: covariance, including remaining image contributions, is in provenance.
_VERTICAL_HALO_SCALES = 4.0


# --------------------------------------------------------------------------
# Field table
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Target:
    """Where one perturbable field lives on the state, and in what units."""

    attribute: str
    stagger: str
    units: str
    #: True when the amplitude is a temperature and has to be divided by the
    #: Exner function before it can be added to the potential-temperature
    #: perturbation the state actually stores.
    exner_from_temperature: bool = False
    #: False for a field that changes sign, which makes a multiplicative
    #: (lognormal) perturbation meaningless on it.
    non_negative: bool = False


#: How one field's draw becomes an increment.
#:
#: ``"additive"`` -- the classical route: ``x <- x + sigma * g``, with
#:   ``sigma`` in the field's own units.  The only defensible mode for a
#:   signed quantity.
#:
#: ``"lognormal"`` -- ``x <- x * exp(sigma_ln * clip(g) - c)``, with
#:   ``sigma_ln`` dimensionless and ``c = log E[exp(sigma_ln * clip(g))]``
#:   (:func:`clipped_lognormal_log_mean`) so the factor's expectation is 1
#:   and the ensemble MEAN keeps the background's value; the factor's
#:   median is then ``exp(-c)``, below 1.  Positivity-preserving in IEEE
#:   arithmetic and scale free, which is what a mixing ratio spanning six
#:   decades needs.  Only available on non-negative fields.
PERTURBATION_MODES = ("additive", "lognormal")

#: How the ``u`` and ``v`` draws are related.
#:
#: ``"rotational"`` (the default) -- ONE streamfunction draw on the corner
#:   grid, ``u = -d(psi)/dy`` and ``v = d(psi)/dx`` by the C-grid
#:   differences, so the wind perturbation is exactly non-divergent on
#:   the model's own divergence stencil wherever the rim taper is 1 (the
#:   components take the scalar rim taper, so the rim band carries a
#:   divergence of order ``u * grad(taper)`` and nothing more).  A
#:   non-divergent perturbation projects onto the slow (balanced) manifold;
#:   the model keeps it.  Both components share the ``u`` spec's length
#:   scales; the ``u`` and ``v`` amplitudes must agree.
#:
#: ``"independent"`` -- the pre-2026-10-06 behaviour: ``u`` and ``v`` are
#:   two unrelated draws.  Half their kinetic energy is divergent and
#:   radiates away as gravity waves within the first hour.  Measured
#:   2026-10-06 together with the mass half (independent draws without
#:   the hydrostatic php against rotational draws with it; the two
#:   half-arms that would split the credit were not run): the members'
#:   surface pressure tendency over the first 30 min fell from 2.1x to
#:   1.3x the control's.  The members' spread deficit against the
#:   conventional observation types is NOT this: the retained spread
#:   barely differs between the two starts.  Kept as a labelled
#:   comparison arm only.
WIND_MODES = ("rotational", "independent")

#: What happens to the column geopotential after a thermodynamic draw.
#:
#: ``"hydrostatic"`` (the default) -- ``php`` is re-integrated at the
#:   column's own dry mass so the perturbed column starts hydrostatic
#:   (:mod:`gpuwm.da.hydrostatic`, the analysis insertion's own operator).
#:   Measured on the 10-01 CONUS members (lane 7, arm basep): the leg-0
#:   surface pressure tendency fell from 29.1 to 21.5 hPa/h with it.
#:   Needs ``hypsometric_opt`` and a state whose base is loaded.
#:
#: ``"none"`` -- the pre-2026-10-06 behaviour; for states without a
#:   vertical coordinate (unit tests, synthetic grids) and for comparison.
MASS_BALANCE_MODES = ("hydrostatic", "none")

#: The streamfunction's scale is this times the configured wind scale.
#: A derivative of a Gaussian field with correlation length ``L_psi`` has
#: the radially binned spectrum ``k^3 exp(-k^2 L_psi^2 / 2)``, which
#: peaks at ``k = sqrt(3) / L_psi``; drawing ``psi`` at ``sqrt(3) L``
#: puts the wind's peak at ``1/L``, the same place a scalar's sits.
STREAMFUNCTION_SCALE_FACTOR = math.sqrt(3.0)


#: The fields this module knows how to perturb.
#:
#: ``"t"`` and ``"theta"`` both land on ``state.thp``; they differ only in
#: what the configured amplitude means.  ``"t"`` is a temperature amplitude in
#: K and is converted with the Exner function (and therefore needs a
#: diagnosed pressure on the state); ``"theta"`` is a potential-temperature
#: amplitude in K and is added straight to ``thp``.  Requesting both is a
#: configuration error rather than a silent double perturbation.
SUPPORTED_FIELDS: Mapping[str, _Target] = {
    "t": _Target("thp", "mass", "K", exner_from_temperature=True),
    "theta": _Target("thp", "mass", "K"),
    "qv": _Target("qv", "mass", "kg/kg", non_negative=True),
    "u": _Target("u", "u_face", "m s-1"),
    "v": _Target("v", "v_face", "m s-1"),
}

#: The hydrometeor mass fields :class:`SpeciesPerturbation` will scale.
#:
#: Spelled as mass-field names because that is the only identity a
#: checkpoint has; the paired number and volume moments are NOT listed
#: here and must never be configured directly -- they are discovered from
#: the state by :func:`gpuwm.da.moments.pairs_present` and scaled by the
#: species' own factor, which is the whole mechanism by which the
#: perturbation stays moment-consistent.  ``qv`` is deliberately absent:
#: it is not a hydrometeor, it has no number moment, and it is perturbed
#: through :class:`FieldPerturbation` like the other scalars.
SUPPORTED_SPECIES = ("qc", "qr", "qi", "qs", "qg", "qh")

#: Default mass activity threshold (kg/kg) below which a species is
#: treated as absent and is not scaled.  This is Morrison's own ``MQSMALL``
#: and the same gate :mod:`gpuwm.da.moments` counts offenders against, so
#: "active" means the same thing to the perturbation and to the guard.
DEFAULT_SPECIES_THRESHOLD = 1.0e-14

#: Fields are applied in this order regardless of the order they appear in
#: the configuration, so two configurations that list the same perturbations
#: differently produce byte-identical states.  Temperature moves before
#: moisture because the supersaturation cap is evaluated against the
#: *perturbed* temperature.
_APPLICATION_ORDER = ("t", "theta", "qv", "u", "v")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldPerturbation:
    """One field's perturbation amplitude and correlation lengths.

    ``amplitude`` is the ensemble 1-sigma of the perturbation in the field's
    own units (see :data:`SUPPORTED_FIELDS`), *before* the rim taper.  The
    realized standard deviation of any single draw fluctuates around it, and
    inside the rim it is deliberately smaller.

    ``vertical_scale_levels`` is measured in model levels, not metres,
    because that is the only vertical coordinate this module can see without
    reaching into the base state; ``0`` decorrelates the levels entirely.
    """

    name: str
    amplitude: float
    length_scale_km: float
    vertical_scale_levels: float = 0.0
    #: ``"additive"`` or ``"lognormal"``; see :data:`PERTURBATION_MODES`.
    #: Under ``"lognormal"`` ``amplitude`` is a dimensionless log-space
    #: sigma, not a value in the field's units.
    mode: str = "additive"
    #: Log-space draws are clipped to this many sigma before the
    #: exponential, so the tail cannot produce an absurd factor.  Ignored
    #: for the additive mode, where the draw is the increment and a large
    #: draw is exactly as large as the amplitude says.
    clip_sigmas: float = 3.0

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_FIELDS:
            raise ValueError(
                f"unknown perturbation field {self.name!r}; supported: "
                + ", ".join(sorted(SUPPORTED_FIELDS)))
        for label, value in (("amplitude", self.amplitude),
                             ("length_scale_km", self.length_scale_km),
                             ("vertical_scale_levels",
                              self.vertical_scale_levels),
                             ("clip_sigmas", self.clip_sigmas)):
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{self.name}: {label} must be finite, got {value!r}")
        if self.amplitude < 0.0:
            raise ValueError(
                f"{self.name}: amplitude must be non-negative, got "
                f"{self.amplitude!r} (a negative 1-sigma is not a sign flip, "
                "it is a typo)")
        if self.length_scale_km <= 0.0:
            raise ValueError(
                f"{self.name}: length_scale_km must be positive, got "
                f"{self.length_scale_km!r}")
        if self.vertical_scale_levels < 0.0:
            raise ValueError(
                f"{self.name}: vertical_scale_levels must be non-negative "
                f"(0 decorrelates the levels), got "
                f"{self.vertical_scale_levels!r}")
        _check_mode(self.name, self.mode, self.clip_sigmas)
        if (self.mode == "lognormal"
                and not SUPPORTED_FIELDS[self.name].non_negative):
            raise ValueError(
                f"{self.name}: mode 'lognormal' multiplies the field by a "
                "strictly positive factor, which is not a perturbation of a "
                "quantity that changes sign -- it would scale a headwind and "
                "a tailwind in opposite physical directions and could never "
                "move a value through zero. Use mode 'additive' for "
                + ", ".join(sorted(n for n, t in SUPPORTED_FIELDS.items()
                                   if not t.non_negative)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FieldPerturbation":
        """Build one spec from a configuration table."""
        known = {"name", "amplitude", "length_scale_km",
                 "vertical_scale_levels", "mode", "clip_sigmas"}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                "unknown perturbation field keys: " + ", ".join(unknown))
        missing = sorted({"name", "amplitude", "length_scale_km"}
                         - set(mapping))
        if missing:
            raise ValueError(
                "perturbation field is missing required keys: "
                + ", ".join(missing))
        return cls(
            name=str(mapping["name"]),
            amplitude=float(mapping["amplitude"]),
            length_scale_km=float(mapping["length_scale_km"]),
            vertical_scale_levels=float(
                mapping.get("vertical_scale_levels", 0.0)),
            mode=str(mapping.get("mode", "additive")),
            clip_sigmas=float(mapping.get("clip_sigmas", 3.0)),
        )


def _check_mode(label: str, mode: str, clip_sigmas: float) -> None:
    """Shared mode/clip validation for both spec kinds."""
    if mode not in PERTURBATION_MODES:
        raise ValueError(
            f"{label}: unknown perturbation mode {mode!r}; supported: "
            + ", ".join(PERTURBATION_MODES))
    if mode == "lognormal" and not (float(clip_sigmas) > 0.0):
        raise ValueError(
            f"{label}: clip_sigmas must be positive under mode 'lognormal', "
            f"got {clip_sigmas!r}; 0 would collapse every factor to exactly "
            "1 and make the perturbation a silent no-op")


@dataclass(frozen=True)
class SpeciesPerturbation:
    """One hydrometeor species' multiplicative, moment-consistent draw.

    ``mass_field`` names the species by its mass mixing ratio, which is
    the only identity a checkpoint carries.  The paired number moment --
    and NSSL's predicted volume moment where the state has one -- are
    discovered from the state itself through
    :func:`gpuwm.da.moments.pairs_present` and scaled by the SAME factor.
    They are never named here, because a caller who could name them
    separately could scale them separately, and that is precisely the
    truncation the pair exists to prevent.

    ``amplitude`` is the log-space sigma: dimensionless, a fractional
    perturbation.  ``0.7`` is a factor of two at one sigma.

    ``threshold_kg_kg`` is the mass below which the species is treated as
    absent and left exactly alone; it defaults to the scheme's own
    activity gate so that "active" means here what it means to the
    moment-consistency guard.
    """

    mass_field: str
    amplitude: float
    length_scale_km: float
    vertical_scale_levels: float = 0.0
    clip_sigmas: float = 3.0
    threshold_kg_kg: float = DEFAULT_SPECIES_THRESHOLD

    def __post_init__(self) -> None:
        if self.mass_field not in SUPPORTED_SPECIES:
            raise ValueError(
                f"unknown perturbation species {self.mass_field!r}; "
                "supported hydrometeor mass fields: "
                + ", ".join(SUPPORTED_SPECIES)
                + ". Number and volume moments are not configurable: they "
                "are scaled by their own species' factor, which is what "
                "keeps the pair consistent")
        for label, value in (("amplitude", self.amplitude),
                             ("length_scale_km", self.length_scale_km),
                             ("vertical_scale_levels",
                              self.vertical_scale_levels),
                             ("clip_sigmas", self.clip_sigmas),
                             ("threshold_kg_kg", self.threshold_kg_kg)):
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{self.mass_field}: {label} must be finite, got "
                    f"{value!r}")
        if self.amplitude < 0.0:
            raise ValueError(
                f"{self.mass_field}: amplitude is a log-space sigma and must "
                f"be non-negative, got {self.amplitude!r}")
        if self.length_scale_km <= 0.0:
            raise ValueError(
                f"{self.mass_field}: length_scale_km must be positive, got "
                f"{self.length_scale_km!r}")
        if self.vertical_scale_levels < 0.0:
            raise ValueError(
                f"{self.mass_field}: vertical_scale_levels must be "
                f"non-negative, got {self.vertical_scale_levels!r}")
        if self.threshold_kg_kg < 0.0:
            raise ValueError(
                f"{self.mass_field}: threshold_kg_kg must be non-negative, "
                f"got {self.threshold_kg_kg!r}")
        _check_mode(self.mass_field, "lognormal", self.clip_sigmas)

    @property
    def name(self) -> str:
        """The spec's identity, for ordering and provenance."""
        return self.mass_field

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]
                     ) -> "SpeciesPerturbation":
        """Build one species spec from a configuration table."""
        known = {"mass_field", "amplitude", "length_scale_km",
                 "vertical_scale_levels", "clip_sigmas", "threshold_kg_kg"}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                "unknown perturbation species keys: " + ", ".join(unknown))
        missing = sorted({"mass_field", "amplitude", "length_scale_km"}
                         - set(mapping))
        if missing:
            raise ValueError(
                "perturbation species is missing required keys: "
                + ", ".join(missing))
        kwargs: dict[str, Any] = {
            "mass_field": str(mapping["mass_field"]),
            "amplitude": float(mapping["amplitude"]),
            "length_scale_km": float(mapping["length_scale_km"]),
            "vertical_scale_levels": float(
                mapping.get("vertical_scale_levels", 0.0)),
            "clip_sigmas": float(mapping.get("clip_sigmas", 3.0)),
        }
        if "threshold_kg_kg" in mapping:
            kwargs["threshold_kg_kg"] = float(mapping["threshold_kg_kg"])
        return cls(**kwargs)


@dataclass(frozen=True)
class PerturbationConfig:
    """Everything :func:`apply_perturbations` needs besides the state+seed.

    ``rim_width`` is in grid cells and must be at least 1: there is no way to
    ask this module to perturb the outermost row, because a member whose
    boundary row disagrees with the shared boundary file is a member with a
    discontinuity, not a member with spread.

    ``rh_cap`` is the relative-humidity ceiling the perturbed moisture field
    is clipped to (``1.0`` = no supersaturation).  ``None`` disables the cap
    entirely, which also removes the module's only need for a diagnosed
    pressure -- but only if no ``"t"`` perturbation is requested, since the
    Exner conversion needs pressure too.

    ``fft_host`` forces the spectral filter onto the host even when the
    state is on the device.  Off it is a performance choice; on it is a
    REPRODUCIBILITY one, and it exists because of a specific measurement.
    The white-noise draw is already host-side Philox so ``noise_sha256``
    identifies the perturbation rather than the machine -- but the filter
    was not, and cuFFT and pocketfft round differently: MEASURED on a
    192x160x49 draw at ``length_scale_km = 6``, a CuPy-filtered ``theta``
    increment and a NumPy-filtered one agree to 2.24e-13 relative and to
    ZERO bits.  That is fine for a resident forecast, where both members
    filter the same way, and it is fatal for a STREAMED one, whose domain
    lives in pinned host RAM and therefore filters on the host by
    construction: the two execution modes produced different members from
    the same seed.  With ``fft_host = True`` a resident member and a
    streamed member of the same seed are byte-identical, which is what
    makes the mode a transport rather than a science change.
    """

    dx_km: float
    dy_km: float
    fields: tuple[FieldPerturbation, ...]
    #: Hydrometeor species scaled multiplicatively with their moments.
    #: Empty is the documented v1 behaviour (hydrometeors untouched) and
    #: is not a defect; it is what leaves the filter with no hydrometeor
    #: covariance to analyse reflectivity through.
    species: tuple[SpeciesPerturbation, ...] = ()
    rim_width: int = 5
    rim_taper: str = "cosine"
    qv_floor: float = 0.0
    rh_cap: float | None = 1.0
    compute_dtype: str = "float64"
    #: Filter on the host even when the state is on the device, so a
    #: resident member and a streamed member of the same seed are
    #: byte-identical.  See the class docstring for the measurement.
    fft_host: bool = False
    #: How ``u`` and ``v`` relate; see :data:`WIND_MODES`.
    wind_mode: str = "rotational"
    #: What happens to ``php`` after a thermodynamic draw; see
    #: :data:`MASS_BALANCE_MODES`.
    mass_balance: str = "hydrostatic"
    #: The run's WRF ``hypsometric_opt`` (1 or 2), which the hydrostatic
    #: re-integration keys its layer operator on.  Required whenever the
    #: mass balance has a column field to act on; a state does not carry
    #: its run configuration, so the caller states it.
    hypsometric_opt: int | None = None

    def __post_init__(self) -> None:
        for label, value in (("dx_km", self.dx_km), ("dy_km", self.dy_km)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"{label} must be a positive finite length, got "
                    f"{value!r}")
        species = tuple(self.species)
        for spec in species:
            if not isinstance(spec, SpeciesPerturbation):
                raise TypeError(
                    "PerturbationConfig.species must hold "
                    "SpeciesPerturbation instances, got "
                    f"{type(spec).__name__}")
        names = [spec.mass_field for spec in species]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                "duplicate perturbation species: " + ", ".join(duplicates))
        object.__setattr__(
            self, "species",
            tuple(sorted(species, key=lambda s: SUPPORTED_SPECIES.index(
                s.mass_field))))
        specs = tuple(self.fields)
        if not specs and not species:
            raise ValueError(
                "PerturbationConfig perturbs nothing: both fields and "
                "species are empty, which is a silent no-op")
        for spec in specs:
            if not isinstance(spec, FieldPerturbation):
                raise TypeError(
                    "PerturbationConfig.fields must hold FieldPerturbation "
                    f"instances, got {type(spec).__name__}")
        names = [spec.name for spec in specs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                "duplicate perturbation fields: " + ", ".join(duplicates))
        if "t" in names and "theta" in names:
            raise ValueError(
                "'t' and 'theta' both perturb state.thp; configure exactly "
                "one of them ('t' is a temperature amplitude converted with "
                "the Exner function, 'theta' is a potential-temperature "
                "amplitude applied directly)")
        object.__setattr__(self, "fields", specs)

        if int(self.rim_width) != self.rim_width or int(self.rim_width) < 1:
            raise ValueError(
                f"rim_width must be an integer >= 1 grid cell, got "
                f"{self.rim_width!r}; there is no 'no taper' setting, "
                "because an untapered member contradicts its own boundary "
                "forcing")
        object.__setattr__(self, "rim_width", int(self.rim_width))
        if self.rim_taper not in ("cosine", "linear"):
            raise ValueError(
                f"rim_taper must be 'cosine' or 'linear', got "
                f"{self.rim_taper!r}")
        if not math.isfinite(float(self.qv_floor)) or self.qv_floor < 0.0:
            raise ValueError(
                f"qv_floor must be finite and non-negative, got "
                f"{self.qv_floor!r}")
        if self.rh_cap is not None:
            if not math.isfinite(float(self.rh_cap)) or self.rh_cap <= 0.0:
                raise ValueError(
                    f"rh_cap must be positive and finite (or None to "
                    f"disable), got {self.rh_cap!r}")
        if self.compute_dtype not in ("float32", "float64"):
            raise ValueError(
                f"compute_dtype must be 'float32' or 'float64', got "
                f"{self.compute_dtype!r}")
        object.__setattr__(self, "fft_host", bool(self.fft_host))

        if self.wind_mode not in WIND_MODES:
            raise ValueError(
                f"unknown wind_mode {self.wind_mode!r}; supported: "
                + ", ".join(WIND_MODES))
        if self.wind_mode == "rotational" and ("u" in names or "v" in names):
            if not ("u" in names and "v" in names):
                raise ValueError(
                    "wind_mode 'rotational' derives u AND v from one "
                    "streamfunction; configuring only "
                    f"{'u' if 'u' in names else 'v'} would leave the other "
                    "component's half of the non-divergent field out and "
                    "the increment divergent. Configure both, or "
                    "wind_mode = 'independent' for a one-component draw")
            u_spec, v_spec = self.spec("u"), self.spec("v")
            for label in ("amplitude", "length_scale_km",
                          "vertical_scale_levels", "mode"):
                if getattr(u_spec, label) != getattr(v_spec, label):
                    raise ValueError(
                        f"wind_mode 'rotational': u and v must agree on "
                        f"{label} (u {getattr(u_spec, label)!r}, v "
                        f"{getattr(v_spec, label)!r}); both components "
                        "are differences of ONE streamfunction and cannot "
                        "carry two amplitudes or two scales")
        if self.mass_balance not in MASS_BALANCE_MODES:
            raise ValueError(
                f"unknown mass_balance {self.mass_balance!r}; supported: "
                + ", ".join(MASS_BALANCE_MODES))
        if self.hypsometric_opt is not None:
            if (int(self.hypsometric_opt) != self.hypsometric_opt
                    or int(self.hypsometric_opt) not in (1, 2)):
                raise ValueError(
                    f"hypsometric_opt must be 1 or 2 (WRF's), got "
                    f"{self.hypsometric_opt!r}")
            object.__setattr__(self, "hypsometric_opt",
                               int(self.hypsometric_opt))

    @property
    def column_field_names(self) -> tuple[str, ...]:
        """The configured perturbations that change the hydrostatic
        column: theta (either spelling), vapour and every species."""
        thermo = tuple(n for n in self.field_names if n in ("t", "theta",
                                                             "qv"))
        return thermo + self.species_names

    @property
    def field_names(self) -> tuple[str, ...]:
        """Configured field names in the canonical application order."""
        configured = {spec.name for spec in self.fields}
        return tuple(n for n in _APPLICATION_ORDER if n in configured)

    @property
    def species_names(self) -> tuple[str, ...]:
        """Configured species mass fields, in canonical order."""
        return tuple(spec.mass_field for spec in self.species)

    def spec(self, name: str) -> FieldPerturbation:
        """Return the configured spec for one field name."""
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PerturbationConfig":
        """Build a configuration from a TOML/JSON table.

        ``fields`` may be a list of tables (each with ``name``) or a table
        keyed by field name.
        """
        known = {"dx_km", "dy_km", "fields", "species", "rim_width",
                 "rim_taper", "qv_floor", "rh_cap", "compute_dtype",
                 "fft_host", "wind_mode", "mass_balance", "hypsometric_opt"}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                "unknown perturbation config keys: " + ", ".join(unknown))
        missing = sorted({"dx_km", "dy_km", "fields"} - set(mapping))
        if missing:
            raise ValueError(
                "perturbation config is missing required keys: "
                + ", ".join(missing))
        raw = mapping["fields"]
        if isinstance(raw, Mapping):
            specs = tuple(
                FieldPerturbation.from_mapping({"name": name, **dict(entry)})
                for name, entry in raw.items())
        else:
            specs = tuple(FieldPerturbation.from_mapping(entry)
                          for entry in raw)
        raw_species = mapping.get("species", ())
        if isinstance(raw_species, Mapping):
            species = tuple(
                SpeciesPerturbation.from_mapping(
                    {"mass_field": name, **dict(entry)})
                for name, entry in raw_species.items())
        else:
            species = tuple(SpeciesPerturbation.from_mapping(entry)
                            for entry in raw_species)
        kwargs: dict[str, Any] = {
            "dx_km": float(mapping["dx_km"]),
            "dy_km": float(mapping["dy_km"]),
            "fields": specs,
            "species": species,
        }
        if "rim_width" in mapping:
            kwargs["rim_width"] = int(mapping["rim_width"])
        if "rim_taper" in mapping:
            kwargs["rim_taper"] = str(mapping["rim_taper"])
        if "qv_floor" in mapping:
            kwargs["qv_floor"] = float(mapping["qv_floor"])
        if "rh_cap" in mapping:
            cap = mapping["rh_cap"]
            kwargs["rh_cap"] = None if cap is None else float(cap)
        if "compute_dtype" in mapping:
            kwargs["compute_dtype"] = str(mapping["compute_dtype"])
        if "fft_host" in mapping:
            kwargs["fft_host"] = bool(mapping["fft_host"])
        if "wind_mode" in mapping:
            kwargs["wind_mode"] = str(mapping["wind_mode"])
        if "mass_balance" in mapping:
            kwargs["mass_balance"] = str(mapping["mass_balance"])
        if "hypsometric_opt" in mapping:
            option = mapping["hypsometric_opt"]
            kwargs["hypsometric_opt"] = (None if option is None
                                         else int(option))
        return cls(**kwargs)


# --------------------------------------------------------------------------
# Array-module plumbing
# --------------------------------------------------------------------------

def default_array_module():
    """CuPy when it is importable and a device is visible, else NumPy."""
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return np
    return cupy


def _array_module(array):
    """The array module that owns ``array``, without importing CuPy first."""
    if isinstance(array, np.ndarray):
        return np
    root = type(array).__module__.split(".")[0]
    if root == "cupy":
        import cupy
        return cupy
    raise TypeError(
        f"unsupported array type {type(array).__module__}."
        f"{type(array).__name__}: expected a NumPy or CuPy array")


def _to_host(array) -> np.ndarray:
    """Copy any supported array to host without assuming which backend."""
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array.get())


#: Tri-state cache for the one-off cuFFT probe: None = not yet asked.
_DEVICE_FFT_AVAILABLE: bool | None = None


def _device_fft_available(xp) -> bool:
    """Whether ``xp`` can actually run an FFT, probed once and remembered.

    A CuPy install can have a perfectly good allocator, kernels, and
    reductions while ``cupy.cuda.cufft`` fails to load -- most commonly when
    the wheel's CUDA minor version and the installed toolkit's cuFFT
    soname disagree.  Asking the library whether it *has* cuFFT is not the
    same as asking whether cuFFT *loads*, so this runs a two-cubed transform
    and believes the answer.
    """
    global _DEVICE_FFT_AVAILABLE
    if xp is np:
        return True
    if _DEVICE_FFT_AVAILABLE is None:
        try:
            xp.fft.rfftn(xp.zeros((2, 2, 2), dtype=np.float64),
                         axes=(0, 1, 2))
        except Exception:
            _DEVICE_FFT_AVAILABLE = False
        else:
            _DEVICE_FFT_AVAILABLE = True
    return _DEVICE_FFT_AVAILABLE


def _device_fft_plan(shape: Sequence[int], dtype, value_type: str):
    """The uncached cuFFT plan one device transform of a draw runs under.

    ``shape`` is the REAL field's ``(nz, ny, nx)`` and ``value_type`` is
    ``"R2C"`` (the forward transform) or ``"C2R"`` (the inverse, whose
    input is the ``nx // 2 + 1`` half spectrum).  The plan comes from
    ``cupy.fft._fft._get_cufft_plan_nd``, the builder ``cupy.fft`` itself
    and ``cupyx.scipy.fft.get_fft_plan`` both call, with the arguments
    ``cupy.fft.rfftn``/``irfftn`` pass it for a C-ordered transform over
    all three axes, so the plan a draw runs under and the plan
    :func:`fft_plan_work_bytes` measures are the same plan.  It is built
    from the shape alone because the admission sizes it before any field
    exists.
    """
    from cupy.cuda import cufft                            # noqa: PLC0415
    from cupy.fft._fft import _get_cufft_plan_nd           # noqa: PLC0415

    nz, ny, nx = (int(extent) for extent in shape)
    double = np.dtype(dtype) == np.float64
    if value_type == "R2C":
        fft_type = cufft.CUFFT_D2Z if double else cufft.CUFFT_R2C
        return _get_cufft_plan_nd((nz, ny, nx), fft_type, axes=(0, 1, 2),
                                  order="C", out_size=nx // 2 + 1,
                                  to_cache=False)
    if value_type == "C2R":
        fft_type = cufft.CUFFT_Z2D if double else cufft.CUFFT_C2R
        return _get_cufft_plan_nd((nz, ny, nx // 2 + 1), fft_type,
                                  axes=(0, 1, 2), order="C", out_size=nx,
                                  to_cache=False)
    raise ValueError(f"value_type must be 'R2C' or 'C2R', got {value_type!r}")


def _draw_shapes(cfg: "PerturbationConfig", mass_shape: Sequence[int]
                 ) -> tuple[tuple[int, int, int], ...]:
    """Every distinct padded FFT shape the configuration draws.

    Under ``wind_mode = "rotational"`` (the default) ``u`` and ``v`` are
    not drawn at all: ONE streamfunction is drawn on the corner grid
    ``(nz, ny + 1, nx + 1)`` and differenced.  That is the shape whose
    plan the card builds and whose work area the admission must price;
    listing the ``u``/``v`` face shapes instead measured two plans the
    draw never runs and priced the one it does at zero (review,
    2026-10-06: the cycle's admission test saw the plan sizes change
    nothing).
    """
    nz, ny, nx = (int(extent) for extent in mass_shape)
    rotational = cfg.wind_mode == "rotational" and "u" in cfg.field_names
    shapes = []
    for name in cfg.field_names:
        levels = cfg.spec(name).vertical_scale_levels
        if rotational and name in ("u", "v"):
            if name == "u":
                shapes.append(_vertical_fft_shape((nz, ny + 1, nx + 1),
                                                  levels)[0])
            continue
        shapes.append(_vertical_fft_shape(_expected_shape(name, nz, ny, nx),
                                          levels)[0])
    shapes.extend(_vertical_fft_shape((nz, ny, nx),
                                      spec.vertical_scale_levels)[0]
                  for spec in cfg.species)
    return tuple(dict.fromkeys(shapes))


def fft_plan_work_bytes(cfg: "PerturbationConfig", mass_shape: Sequence[int],
                        xp=None) -> dict:
    """cuFFT's own work area for each plan a member's perturbation runs.

    ``{shape: (forward_bytes, inverse_bytes)}`` for every distinct padded
    FFT shape the configuration draws, read off the plans themselves: each is
    built with :func:`_device_fft_plan`, exactly as the draw builds it, and
    its work area (allocated from the device pool at the size
    ``cufftMakePlanMany`` reported) is measured and released.  Empty when
    no draw transforms on the device: ``fft_host``, a host namespace, or a
    device whose cuFFT does not load.
    """
    if xp is None:
        xp = default_array_module()
    if cfg.fft_host or xp is np or not _device_fft_available(xp):
        return {}
    dtype = np.float32 if cfg.compute_dtype == "float32" else np.float64
    sizes = {}
    for shape in _draw_shapes(cfg, mass_shape):
        measured = []
        for value_type in ("R2C", "C2R"):
            plan = _device_fft_plan(shape, dtype, value_type)
            area = plan.work_area
            measured.append(0 if area is None else int(area.mem.size))
            del plan, area
        sizes[shape] = tuple(measured)
    # The measured work areas went back to the pool as idle blocks, which
    # cudaMemGetInfo counts as used; hand them to the driver so the free
    # reading an admission takes next sees the card as it was.
    xp.get_default_memory_pool().free_all_blocks()
    return sizes


# --------------------------------------------------------------------------
# Deterministic noise
# --------------------------------------------------------------------------

def _stream_key(seed: int, name: str, shape: Sequence[int]) -> int:
    """A 128-bit Philox key derived from ``(seed, field name, shape)``.

    Deriving the key by hash rather than by ``seed + offset`` means adjacent
    seeds give unrelated streams, two fields of one member never share a
    stream, and the same field on a differently shaped grid is a different
    draw (so a resolution change cannot silently reproduce the old noise on a
    subset of points).
    """
    dims = ",".join(str(int(extent)) for extent in shape)
    payload = f"{_STREAM_DOMAIN}|seed={int(seed)}|field={name}|shape=({dims})"
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:16], "big")


def _white_noise(shape: Sequence[int], *, seed: int, name: str,
                 dtype) -> tuple[np.ndarray, int, str]:
    """Unit-variance host white noise plus its key and SHA-256.

    The draw happens on the host with NumPy's Philox counter-based generator
    even when the filtering will run on the GPU.  That is deliberate: it
    makes ``noise_sha256`` identical on a CUDA box and a CPU box, so the
    provenance stamp identifies the *perturbation* rather than the machine.
    The filtered field itself is not bit-identical across backends -- cuFFT
    and pocketfft round differently -- and this module does not claim it is.
    """
    key = _stream_key(seed, name, shape)
    generator = np.random.Generator(np.random.Philox(key=key))
    noise = generator.standard_normal(tuple(int(s) for s in shape),
                                      dtype=dtype)
    digest = hashlib.sha256(noise.tobytes()).hexdigest()
    return noise, key, digest


# --------------------------------------------------------------------------
# Gaussian random fields
# --------------------------------------------------------------------------

def _axis_filter(n: int, spacing: float, length_scale: float,
                 *, half: bool) -> np.ndarray:
    """Per-axis Gaussian amplitude filter on the (half-)spectrum."""
    freq = (np.fft.rfftfreq(n, d=spacing) if half
            else np.fft.fftfreq(n, d=spacing))
    k = 2.0 * np.pi * freq
    return np.exp(-0.25 * (k * length_scale) ** 2)


def _analytic_variance(n: int, spacing: float,
                       length_scale: float) -> float:
    """``mean_k |H(k)|^2`` on one full axis.

    With unit-variance white noise and NumPy's unnormalized forward DFT, the
    filtered field's variance is the mean of ``|H|^2`` over the *full* k-grid;
    because ``H`` is separable the 3-D mean is the product of these.
    """
    return float(np.mean(
        _axis_filter(n, spacing, length_scale, half=False) ** 2))


def _vertical_fft_shape(shape: Sequence[int], vertical_scale_levels: float
                        ) -> tuple[tuple[int, int, int], int]:
    """Independent-noise FFT domain and physical-column crop offset.

    Padding is fresh noise, never a tiled or reflected physical column.
    The filter is stationary on this larger domain, so cropping preserves
    its analytic point variance while separating the physical endpoints.
    """
    nz, ny, nx = (int(extent) for extent in shape)
    scale = float(vertical_scale_levels)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "vertical_scale_levels must be finite and non-negative; a "
            f"non-finite or negative scale cannot define a covariance, got {scale!r}")
    halo = int(math.ceil(_VERTICAL_HALO_SCALES * scale))
    return (nz + 2 * halo, ny, nx), halo


def vertical_wrap_correlations(nz: int,
                               vertical_scale_levels: float) -> dict[str, Any]:
    """Exact correlations of the physical crop of the actual FFT domain.

    For ``N`` padded levels, the stationary correlation at physical lag
    ``d`` is ``sum_k |H(k)|^2 cos(2 pi k d/N) / sum_k |H(k)|^2``.
    ``top_to_bottom_seam`` retains its legacy key but measures lag ``nz-1``
    in the PADDED domain, not a one-level circular seam.  The periodic FFT
    seam lies outside the physical column.  All values include the exact
    finite-spectrum and periodic-image effects; none are fitted to a draw.
    """
    nz = int(nz)
    if nz < 1:
        raise ValueError(f"vertical correlation needs positive levels, got {nz}")
    scale = float(vertical_scale_levels)
    fft_shape, halo = _vertical_fft_shape((nz, 1, 1), scale)
    padded_nz = fft_shape[0]
    geometry = {
        "vertical_scale_levels": scale,
        "levels": nz,
        "fft_levels": padded_nz,
        "independent_noise_halo_levels": halo,
        "physical_crop_levels": [halo, halo + nz],
        "periodic_seam_in_physical_column": False,
    }
    if nz < 2:
        return {**geometry,
                "note": "a single-level column has no vertical correlation"}
    power = _axis_filter(padded_nz, 1.0, scale, half=False) ** 2
    total = float(power.sum())
    if not (total > 0.0):
        return {**geometry, "note": "the vertical filter has no power on this column"}
    modes = np.fft.fftfreq(padded_nz, d=1.0) * padded_nz

    def rho(lag: int) -> float:
        return float((power * np.cos(2.0 * np.pi * modes * lag / padded_nz)).sum()
                     / total)

    return {
        **geometry,
        "adjacent_interior": rho(1),
        "top_to_bottom_seam": rho(nz - 1),
        "half_column": rho(nz // 2),
        "nearest_periodic_image_lag_levels": padded_nz - (nz - 1),
        "method": ("exact autocorrelation of the padded amplitude filter "
                   "at physical crop lags: sum_k |H(k)|^2 cos(2 pi k d/N) / "
                   "sum_k |H(k)|^2"),
        "caveat": (
            "top_to_bottom_seam is a legacy key for the physical endpoint "
            "correlation; the physical endpoints are nz-1 levels apart. "
            "These are raw-field correlations before any vertical taper."),
    }


def _check_resolvable(shape: Sequence[int], dx_km: float, dy_km: float,
                      spec: FieldPerturbation) -> None:
    """Refuse length scales the grid cannot represent, in either direction."""
    nz, ny, nx = (int(shape[0]), int(shape[1]), int(shape[2]))
    cell = max(float(dx_km), float(dy_km))
    floor = _MIN_SCALE_CELLS * cell
    if spec.length_scale_km < floor:
        raise ValueError(
            f"{spec.name}: length_scale_km={spec.length_scale_km} is below "
            f"{_MIN_SCALE_CELLS:g} grid spacings ({floor:g} km on a "
            f"{dx_km:g} x {dy_km:g} km grid); the grid cannot carry it and "
            "the model's diffusion would remove what is left")
    span = min(nx * float(dx_km), ny * float(dy_km))
    ceiling = _MAX_HORIZONTAL_SPAN_FRACTION * span
    if spec.length_scale_km > ceiling:
        raise ValueError(
            f"{spec.name}: length_scale_km={spec.length_scale_km} exceeds "
            f"span/(2*pi) = {ceiling:g} km on the shorter domain span "
            f"({span:g} km); the documented spectral peak k=1/L is resolved "
            f"only at or below that limit, because the lowest nonzero "
            f"wavenumber a periodic span of {span:g} km carries is "
            f"2*pi/{span:g}. Above it the draw is a domain-wide offset and "
            "'prescribed length scale' stops meaning anything")
    if spec.vertical_scale_levels > 0.0:
        if spec.vertical_scale_levels < _MIN_SCALE_CELLS:
            raise ValueError(
                f"{spec.name}: vertical_scale_levels="
                f"{spec.vertical_scale_levels} is below "
                f"{_MIN_SCALE_CELLS:g} levels; use 0 to decorrelate the "
                "levels outright rather than a sub-level scale that only "
                "looks like a correlation")


def gaussian_random_field(shape: Sequence[int], *, seed: int, name: str,
                          dx_km: float, dy_km: float,
                          length_scale_km: float,
                          vertical_scale_levels: float = 0.0,
                          xp=None, dtype: str = "float64",
                          fft_host: bool = False,
                          ) -> tuple[Any, dict[str, Any]]:
    """A unit-variance Gaussian random field of ``shape`` ``(nz, ny, nx)``.

    Returns ``(field, info)``.  ``info`` carries the stream key, the SHA-256
    of the white-noise draw, the analytic and realized RMS, and the backend
    the FFT ran on -- everything :func:`apply_perturbations` needs to build
    its provenance, and enough for a caller using this helper directly to
    reproduce the draw.

    The field's *expected* variance is exactly 1 by analytic normalization;
    its realized variance fluctuates like any finite sample, which is the
    behaviour an ensemble wants.  For a nonzero vertical scale the FFT
    uses fresh independent noise with a four-scale halo on each end, then
    crops the physical column.  Its variance is normalized from that
    padded spectrum; the crop is neither repeated nor sample-normalized.

    ``fft_host`` runs the transform on the host regardless of ``xp``.  The
    draw already does (host Philox, so the SHA is machine-independent);
    this extends the same guarantee to the FILTER, which is what a domain
    living in pinned host RAM needs in order to produce the same member as
    the resident run.  ``info["fft_backend"]`` records which one ran, so a
    manifest says how a member was filtered and not merely that it was.
    """
    shape = tuple(int(extent) for extent in shape)
    if len(shape) != 3 or any(extent < 1 for extent in shape):
        raise ValueError(
            f"gaussian_random_field needs a positive (nz, ny, nx) shape, "
            f"got {shape}")
    if dtype not in ("float32", "float64"):
        raise ValueError(
            f"dtype must be 'float32' or 'float64', got {dtype!r}")
    for label, value in (("dx_km", dx_km), ("dy_km", dy_km)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{label} must be finite and positive; invalid "
                             f"spacing cannot define a spectrum, got {value!r}")
    if not math.isfinite(float(length_scale_km)) or float(length_scale_km) < 0.0:
        raise ValueError("length_scale_km must be finite and non-negative; "
                         "invalid scale cannot define a covariance, got "
                         f"{length_scale_km!r}")
    if xp is None:
        xp = default_array_module()
    nz, ny, nx = shape
    fft_shape, crop_start = _vertical_fft_shape(shape, vertical_scale_levels)
    padded_nz = fft_shape[0]
    np_dtype = np.float32 if dtype == "float32" else np.float64

    noise, key, digest = _white_noise(fft_shape, seed=seed, name=name,
                                      dtype=np_dtype)

    # Filter wherever the FFT actually works; move the finished field to the
    # requested backend afterwards.  A half-device pipeline would be worse
    # than either whole one.  ``fft_host`` overrides that in the one
    # direction that is always available -- a device that cannot do an FFT
    # is already forced here, and a host that cannot is not a host.
    fft_xp = np if fft_host else (xp if _device_fft_available(xp) else np)
    working = noise if fft_xp is np else fft_xp.asarray(noise)

    if fft_xp is np:
        spectrum = fft_xp.fft.rfftn(working, axes=(0, 1, 2))
    else:
        # Each transform runs under its own uncached plan, dropped as the
        # transform returns, so cuFFT's work area is a transient of this
        # draw (priced by device_working_bytes from fft_plan_work_bytes).
        # cupy's plan cache would keep both plans of every drawn shape,
        # work areas included, on the card for the rest of the process.
        with _device_fft_plan(fft_shape, np_dtype, "R2C"):
            spectrum = fft_xp.fft.rfftn(working, axes=(0, 1, 2))
    hz = _axis_filter(padded_nz, 1.0, float(vertical_scale_levels), half=False)
    hy = _axis_filter(ny, float(dy_km), float(length_scale_km), half=False)
    hx = _axis_filter(nx, float(dx_km), float(length_scale_km), half=True)
    kernel = (hz[:, None, None] * hy[None, :, None]
              * hx[None, None, :]).astype(np_dtype, copy=False)
    if fft_xp is not np:
        kernel = fft_xp.asarray(kernel)
    spectrum = spectrum * kernel
    if fft_xp is np:
        field = fft_xp.fft.irfftn(spectrum, s=fft_shape, axes=(0, 1, 2))
    else:
        with _device_fft_plan(fft_shape, np_dtype, "C2R"):
            field = fft_xp.fft.irfftn(spectrum, s=fft_shape, axes=(0, 1, 2))

    variance = (_analytic_variance(padded_nz, 1.0, float(vertical_scale_levels))
                * _analytic_variance(ny, float(dy_km),
                                     float(length_scale_km))
                * _analytic_variance(nx, float(dx_km),
                                     float(length_scale_km)))
    if not (variance > 0.0) or not math.isfinite(variance):
        raise RuntimeError(
            f"{name}: the Gaussian filter has no power on this grid "
            f"(analytic variance {variance!r}); the requested scales are "
            "degenerate")
    field = field / math.sqrt(variance)
    # Own only the physical crop.  A view would pin the padded allocation
    # through every application and into the next draw's peak memory.
    field = field[crop_start:crop_start + nz].astype(np_dtype, copy=True)

    realized = float(fft_xp.sqrt(fft_xp.mean(field.astype(np.float64) ** 2)))
    if fft_xp is not xp:
        field = xp.asarray(field)
    info = {
        "shape": shape,
        "noise_shape": fft_shape,
        "vertical_crop_levels": [crop_start, crop_start + nz],
        "stream_key_hex": f"{key:032x}",
        "noise_sha256": digest,
        "noise_dtype": dtype,
        #: RMS of the filtered field *before* normalization; the returned
        #: field was divided by exactly this, so its expected RMS is 1.
        "pre_normalization_analytic_rms": math.sqrt(variance),
        "realized_rms": realized,
        "backend": "numpy" if xp is np else "cupy",
        "fft_backend": "numpy" if fft_xp is np else "cupy",
        "length_scale_km": float(length_scale_km),
        "vertical_scale_levels": float(vertical_scale_levels),
        # Exact correlations of the physical crop of the padded spectrum.
        "vertical_wrap": vertical_wrap_correlations(
            nz, float(vertical_scale_levels)),
    }
    return field, info


# --------------------------------------------------------------------------
# Boundary taper
# --------------------------------------------------------------------------

def vertical_taper(nz: int, bottom_width_levels: int = 0,
                   top_width_levels: int = 0, *, kind: str = "cosine",
                   xp=None, dtype=np.float64):
    """Optional ``(nz,)`` attenuation at the physical column's two ends.

    A zero width leaves that end untouched.  A positive width gives exactly
    zero on its end level and reaches one ``width`` levels into the column.
    Each end is independent; overlapping ramps multiply, so a thin column
    remains bounded and needs no artificial untapered-interior requirement.
    This scales amplitude, not a sampled variance or the length scale.
    """
    if int(nz) != nz or nz < 1:
        raise ValueError(f"vertical_taper needs positive integer levels, got {nz!r}")
    widths = (bottom_width_levels, top_width_levels)
    for width in widths:
        if not isinstance(width, (int, np.integer)) or width < 0:
            raise ValueError("vertical taper widths must be non-negative "
                             f"integer levels; invalid widths cannot define a ramp, got {width!r}")
    if kind not in ("cosine", "linear"):
        raise ValueError(f"unknown vertical taper kind {kind!r}")
    if xp is None:
        xp = default_array_module()
    levels = np.arange(int(nz), dtype=np.float64)
    taper = np.ones(int(nz), dtype=np.float64)
    for distance, width in ((levels, bottom_width_levels),
                            (int(nz) - 1 - levels, top_width_levels)):
        if width:
            ratio = np.clip(distance / int(width), 0.0, 1.0)
            ramp = (0.5 * (1.0 - np.cos(np.pi * ratio))
                    if kind == "cosine" else ratio)
            taper *= ramp
    taper = taper.astype(dtype, copy=False)
    return taper if xp is np else xp.asarray(taper)


def boundary_taper(ny: int, nx: int, rim_width: int, *, kind: str = "cosine",
                   xp=None, dtype=np.float64):
    """A ``(ny, nx)`` rim taper: exactly 0 on the edge, exactly 1 inside.

    The taper is a function of the *minimum* index distance to any of the
    four lateral edges, so it is a frame rather than a separable product;
    corners are treated the same as edges instead of being doubly damped.

    Both endpoints are exact in IEEE arithmetic, not merely close: at
    ``d = 0`` the cosine form evaluates ``0.5 * (1 - cos 0) = 0`` and at
    ``d >= rim_width`` it evaluates ``0.5 * (1 - cos pi) = 1``, because
    ``cos(pi)`` is exactly ``-1``.  ``tests/test_da_perturb.py`` asserts
    equality, not tolerance.
    """
    ny, nx, rim_width = int(ny), int(nx), int(rim_width)
    if ny < 1 or nx < 1:
        raise ValueError(f"boundary_taper needs a positive shape, got "
                         f"({ny}, {nx})")
    if rim_width < 1:
        raise ValueError(
            f"rim_width must be >= 1 grid cell, got {rim_width}")
    if kind not in ("cosine", "linear"):
        raise ValueError(f"unknown taper kind {kind!r}")
    if xp is None:
        xp = np
    if 2 * rim_width >= min(ny, nx):
        raise ValueError(
            f"rim_width={rim_width} leaves no untapered interior on a "
            f"({ny}, {nx}) field; the two rims meet")

    dj = np.minimum(np.arange(ny), ny - 1 - np.arange(ny))
    di = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))
    distance = np.minimum(dj[:, None], di[None, :]).astype(np.float64)
    ratio = np.minimum(distance / float(rim_width), 1.0)
    if kind == "cosine":
        taper = 0.5 * (1.0 - np.cos(np.pi * ratio))
    else:
        taper = ratio
    taper = taper.astype(dtype, copy=False)
    return taper if xp is np else xp.asarray(taper)


def _difference_variance(n: int, spacing_m: float,
                         length_scale_m: float) -> float:
    """Variance of the one-cell difference, per unit spacing, of a unit
    Gaussian-filtered axis: ``mean_k |H|^2 |e^{ik dx} - 1|^2 / dx^2``
    over ``mean_k |H|^2``, so it multiplies the normalized field's own
    unit variance.  Units ``1/m^2``."""
    freq = np.fft.fftfreq(int(n), d=float(spacing_m))
    k = 2.0 * np.pi * freq
    h2 = np.exp(-0.5 * (k * float(length_scale_m)) ** 2)
    d2 = (2.0 * np.sin(0.5 * k * float(spacing_m))) ** 2 / float(spacing_m) ** 2
    return float(np.mean(h2 * d2) / np.mean(h2))


def rotational_wind_draw(mass_shape: Sequence[int], *, seed: int,
                         dx_km: float, dy_km: float, length_scale_km: float,
                         vertical_scale_levels: float = 0.0, xp=None,
                         dtype: str = "float64", fft_host: bool = False):
    """Unit-amplitude non-divergent ``(u, v)`` increments from ONE draw.

    A streamfunction ``psi`` is drawn on the corner grid ``(nz, ny+1,
    nx+1)`` at ``sqrt(3)`` times the configured scale
    (:data:`STREAMFUNCTION_SCALE_FACTOR`) and differenced the C-grid way:
    ``u = -(psi[j+1, i] - psi[j, i]) / dy`` on the ``u`` faces and
    ``v = (psi[j, i+1] - psi[j, i]) / dx`` on the ``v`` faces.  The model's
    divergence at every mass point, ``(u[i+1] - u[i]) / dx + (v[j+1] -
    v[j]) / dy``, is then a telescoping sum of the same four corners and
    is exactly zero in exact arithmetic (rounding only, in floating
    point).

    The fields come back UNTAPERED; :func:`apply_perturbations` applies
    the rim taper to each component exactly as it does to a scalar.
    Tapering ``psi`` instead was tried first (2026-10-06) and refused by
    measurement: a non-divergent field that must vanish at the rim has
    ``psi`` constant there, and a ``psi`` of amplitude ``sigma_u *
    L_psi`` dropping to that constant across the rim band is a
    tangential wind of ``sigma_u * L_psi / W``.  With the DA cycle
    tool's own numbers (1.5 m/s at 150 km, a 5-cell rim at 3 km) that is
    35 times the configured amplitude, some 50 m/s of shear along every
    boundary.  Tapering the components keeps the rim band's amplitude
    at or below the spec, exactly as for the scalars, at the price of a
    divergence of order ``u * grad(taper)`` confined to the rim band
    (the same term the independent draws always carried there); the
    interior, where the taper is one, stays exactly non-divergent.

    ``psi`` is scaled ONCE so that the ``u`` increment has unit expected
    variance (the analytic variance of the discrete ``y`` difference,
    :func:`_difference_variance`).  ``v`` gets whatever the ``x``
    difference gives at that scaling -- equal to ``u``'s on a square grid
    of similar extents, and reported as ``v_amplitude_ratio`` otherwise.
    Scaling the two components separately would restore exact unit
    variance on both and destroy exact non-divergence, which is the
    property the whole draw exists for.

    Returns ``(u, v, info)``: the unit increments on the ``(nz, ny,
    nx+1)`` and ``(nz, ny+1, nx)`` face grids and the provenance of the
    draw (the ``psi`` stream key and noise digest, the scales and the
    variances).
    """
    shape = tuple(int(extent) for extent in mass_shape)
    if len(shape) != 3 or any(extent < 1 for extent in shape):
        raise ValueError(
            f"rotational_wind_draw needs a positive (nz, ny, nx) mass "
            f"shape, got {shape}")
    for label, value in (("dx_km", dx_km), ("dy_km", dy_km),
                         ("length_scale_km", length_scale_km)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{label} must be finite and positive, got "
                             f"{value!r}")
    if xp is None:
        xp = default_array_module()
    nz, ny, nx = shape
    psi_scale_km = STREAMFUNCTION_SCALE_FACTOR * float(length_scale_km)
    psi, info = gaussian_random_field(
        (nz, ny + 1, nx + 1), seed=seed, name="psi", dx_km=dx_km,
        dy_km=dy_km, length_scale_km=psi_scale_km,
        vertical_scale_levels=vertical_scale_levels, xp=xp, dtype=dtype,
        fft_host=fft_host)
    dx_m, dy_m = float(dx_km) * 1000.0, float(dy_km) * 1000.0
    psi_scale_m = psi_scale_km * 1000.0
    var_u = _difference_variance(ny + 1, dy_m, psi_scale_m)
    var_v = _difference_variance(nx + 1, dx_m, psi_scale_m)
    if not (var_u > 0.0) or not math.isfinite(var_u):
        raise RuntimeError(
            f"the streamfunction's y difference has no variance "
            f"({var_u!r}) on this grid; the requested scale is degenerate")
    # One scaling for both components: psi in m^2/s per unit u amplitude.
    psi *= psi.dtype.type(1.0 / math.sqrt(var_u))
    u = -(psi[:, 1:, :] - psi[:, :-1, :]) / psi.dtype.type(dy_m)
    v = (psi[:, :, 1:] - psi[:, :, :-1]) / psi.dtype.type(dx_m)
    del psi
    info = dict(info)
    info.update({
        "wind_mode": "rotational",
        "streamfunction_shape": [nz, ny + 1, nx + 1],
        "streamfunction_length_scale_km": psi_scale_km,
        "scale_factor": STREAMFUNCTION_SCALE_FACTOR,
        "u_difference_variance_per_m2": var_u,
        "v_difference_variance_per_m2": var_v,
        #: Expected v amplitude per unit u amplitude at this scaling.
        "v_amplitude_ratio": math.sqrt(var_v / var_u),
        "u_realized_rms": float(xp.sqrt(xp.mean(u.astype(np.float64) ** 2))),
        "v_realized_rms": float(xp.sqrt(xp.mean(v.astype(np.float64) ** 2))),
        "stencil": "u = -d(psi)/dy, v = d(psi)/dx by one-cell C-grid "
                   "differences of a corner-grid psi; the mass-point "
                   "divergence is zero wherever the rim taper is one and "
                   "of order u * grad(taper) inside the rim band",
    })
    return u, v, info


# --------------------------------------------------------------------------
# Spectral diagnostic
# --------------------------------------------------------------------------

def radial_power_spectrum(field, dx_km: float, dy_km: float | None = None,
                          *, bins: int | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Annulus-summed 2-D power spectrum ``E(k)`` of a field.

    Accepts one ``(ny, nx)`` level or a ``(nz, ny, nx)`` stack, in which case
    the per-level spectra are averaged -- with vertically decorrelated levels
    that is an average over ``nz`` independent realizations and it is the
    difference between a peak you can put a tolerance on and one you cannot.

    Returns ``(k, energy)`` with ``k`` in radians per kilometre and

        ``energy[b] = k[b] * mean(|F(k)|^2 over annulus b)``

    The ``k`` factor is the annulus Jacobian -- it is what makes a
    Gaussian-correlated field's spectrum *peak*, at ``k = 1/L``, instead of
    decaying monotonically from ``k = 0``.  It is applied analytically to the
    per-mode *mean* rather than by summing the annulus, because the number of
    discrete modes actually landing in a low-wavenumber annulus is small and
    lumpy, and a storm-scale correlation length puts its peak exactly there.
    Summing instead of averaging moves the measured peak by tens of percent
    from one realization to the next; this form does not.

    The zero mode is dropped: a constant offset is not a length scale.
    Empty annuli come back as zero energy.
    """
    host = _to_host(field)
    if host.ndim == 2:
        host = host[None, :, :]
    if host.ndim != 3:
        raise ValueError(
            f"radial_power_spectrum takes an (ny, nx) level or an "
            f"(nz, ny, nx) stack, got shape {np.shape(field)}")
    nlev, ny, nx = host.shape
    if dy_km is None:
        dy_km = dx_km
    if dx_km <= 0.0 or dy_km <= 0.0:
        raise ValueError("grid spacings must be positive")

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=float(dx_km))
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=float(dy_km))
    kmag = np.hypot(ky[:, None], kx[None, :])

    # Bin only out to the smaller of the two Nyquist wavenumbers, so no
    # annulus is partly outside the sampled rectangle and thus artificially
    # starved of modes.
    kmax = float(min(np.abs(kx).max(), np.abs(ky).max()))
    if bins is None:
        bins = max(8, min(ny, nx) // 2)
    bins = int(bins)
    edges = np.linspace(0.0, kmax, bins + 1)
    flat_k = kmag.ravel()
    keep = (flat_k > 0.0) & (flat_k <= kmax)
    index = np.clip(np.digitize(flat_k[keep], edges) - 1, 0, bins - 1)

    counts = np.bincount(index, minlength=bins).astype(np.float64)
    transform = np.fft.fft2(host.astype(np.float64), axes=(1, 2))
    power = (np.abs(transform) ** 2).reshape(nlev, -1)[:, keep]
    total = np.zeros(bins, dtype=np.float64)
    for level in range(nlev):
        total += np.bincount(index, weights=power[level], minlength=bins)
    occupied = counts > 0.0
    per_mode = np.zeros(bins, dtype=np.float64)
    per_mode[occupied] = total[occupied] / (counts[occupied] * float(nlev))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, centres * per_mode


def spectral_peak_wavenumber(k: np.ndarray, energy: np.ndarray) -> float:
    """Sub-bin peak of a radial spectrum, by log-parabolic interpolation.

    The bin containing ``1/L`` can be several percent wide at the
    wavenumbers a storm-scale perturbation lives at, so taking the argmax
    bin centre reports the binning as much as the field.  Fitting a parabola
    to ``log E`` over the peak bin and its two neighbours -- exact for a
    Gaussian, which this spectrum locally is -- removes that.
    """
    k = np.asarray(k, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    if k.shape != energy.shape or k.ndim != 1 or k.size < 3:
        raise ValueError("k and energy must be matching 1-D arrays of >= 3")
    peak = int(np.argmax(energy))
    if peak == 0 or peak == k.size - 1:
        return float(k[peak])
    left, centre, right = energy[peak - 1:peak + 2]
    if not (left > 0.0 and centre > 0.0 and right > 0.0):
        return float(k[peak])
    ll, lc, lr = math.log(left), math.log(centre), math.log(right)
    denominator = ll - 2.0 * lc + lr
    if denominator == 0.0:
        return float(k[peak])
    offset = 0.5 * (ll - lr) / denominator
    if not (-1.0 <= offset <= 1.0):
        return float(k[peak])
    spacing = float(k[1] - k[0])
    return float(k[peak] + offset * spacing)


def fit_gaussian_length_scale(k: np.ndarray, energy: np.ndarray, *,
                              band: tuple[float, float] = (0.3, 3.0)
                              ) -> float:
    """Recover ``L`` from a radial spectrum, in the same units as ``1/k``.

    The inverse of what :func:`gaussian_random_field` promises.  Removing the
    annulus Jacobian leaves ``P(k) = E(k)/k``, and for the Gaussian
    correlation this module imposes ``log P`` is *exactly* linear in ``k^2``
    with slope ``-L^2/2``.  A straight-line fit over a band around the peak
    therefore pins ``L`` far more tightly than reading the peak off the axis
    does -- and it checks the whole spectral shape, not one point of it.

    ``band`` is the fitted range as a multiple of the measured peak
    wavenumber.
    """
    k = np.asarray(k, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    peak = spectral_peak_wavenumber(k, energy)
    low, high = float(band[0]) * peak, float(band[1]) * peak
    usable = (energy > 0.0) & (k > 0.0) & (k >= low) & (k <= high)
    if int(np.count_nonzero(usable)) < 4:
        raise ValueError(
            "not enough occupied spectral bins around the peak to fit a "
            "length scale; use a larger domain or fewer bins")
    slope = np.polyfit(k[usable] ** 2,
                       np.log(energy[usable] / k[usable]), 1)[0]
    if slope >= 0.0:
        raise ValueError(
            "the spectrum does not fall off with wavenumber (fitted slope "
            f"{slope!r}); this is not a Gaussian-correlated field")
    return float(math.sqrt(-2.0 * slope))


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

def _expected_shape(name: str, nz: int, ny: int, nx: int
                    ) -> tuple[int, int, int]:
    """The ARW-staggered shape one perturbable field must have."""
    stagger = SUPPORTED_FIELDS[name].stagger
    if stagger == "mass":
        return (nz, ny, nx)
    if stagger == "u_face":
        return (nz, ny, nx + 1)
    if stagger == "v_face":
        return (nz, ny + 1, nx)
    raise AssertionError(f"unhandled stagger {stagger!r}")


def _resolve_target(state, name: str):
    """Fetch the state array for one field, failing closed if it is absent."""
    attribute = SUPPORTED_FIELDS[name].attribute
    array = getattr(state, attribute, None)
    if array is None:
        raise ValueError(
            f"cannot perturb {name!r}: state.{attribute} is None "
            "(a dry state has no moisture arrays; perturbing qv on it would "
            "be a silent no-op)")
    if not hasattr(array, "shape"):
        raise TypeError(
            f"state.{attribute} is not an array (got "
            f"{type(array).__name__})")
    return array


def _total_theta(state, xp):
    """Full potential temperature, whether ``thb`` is a column or 3-D."""
    if hasattr(state, "total_theta"):
        return state.total_theta()
    thb = getattr(state, "thb", None)
    if thb is None:
        raise ValueError(
            "state carries neither total_theta() nor thb; the "
            "supersaturation cap and the Exner conversion both need the "
            "full potential temperature")
    base = thb if getattr(thb, "ndim", 0) == 3 else thb[:, None, None]
    return base + state.thp


def _require_pressure(state, xp, reason: str):
    """Return a strictly positive full pressure or refuse to continue."""
    pressure = getattr(state, "p", None)
    if pressure is None:
        raise ValueError(
            f"{reason} needs state.p (full pressure) and the state has none")
    minimum = float(xp.min(pressure))
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError(
            f"{reason} needs a diagnosed positive pressure; state.p has "
            f"minimum {minimum!r}. An un-diagnosed (all-zero) pressure "
            "field would silently produce infinite Exner factors, so this "
            "fails rather than guesses")
    return pressure


def _saturation_mixing_ratio(temperature, pressure, xp):
    """Tetens ``qvs`` over liquid water, in WRF's constants.

    Uses ``SVP1/SVP2/SVP3/SVPT0`` and ``EP2`` from
    :mod:`gpuwm.core.constants` -- the same numbers WRF's own saturation
    calls use -- so the cap this module enforces and the saturation the
    microphysics will see the next step are the same curve.
    """
    denominator = temperature - c.SVP3
    if float(xp.min(denominator)) <= 0.0:
        raise ValueError(
            "the perturbed temperature field reaches the Tetens pole "
            f"(T <= {c.SVP3} K); the temperature perturbation amplitude is "
            "not physical for this state")
    es = (c.SVP1 * 1000.0) * xp.exp(
        c.SVP2 * (temperature - c.SVPT0) / denominator)
    margin = pressure - es
    if float(xp.min(margin)) <= 0.0:
        raise ValueError(
            "saturation vapour pressure meets or exceeds the total "
            "pressure somewhere in the perturbed state; the moisture cap "
            "cannot be evaluated there")
    return c.EP2 * es / margin


def _clip_draw(xp, draw, clip_sigmas: float):
    """The unit-variance draw, bounded to ``+/- clip_sigmas``.

    A unit Gaussian exceeds three sigma about one point in 370, so on a
    domain of a few million points thousands of cells would otherwise take
    a factor of ``exp(3 sigma_ln)`` or worse, and the tail of a lognormal
    is where a "factor of two" perturbation quietly becomes a factor of a
    thousand.  Clipping the exponent rather than the factor keeps the
    bound stated in the units the amplitude is stated in.
    """
    limit = draw.dtype.type(float(clip_sigmas))
    return xp.clip(draw, -limit, limit)


def clipped_lognormal_log_mean(amplitude: float, clip_sigmas: float):
    """``log E[exp(a * clip(g, -k, k))]`` for a unit Gaussian ``g``.

    The ensemble-mean correction of a lognormal factor.  ``exp(a g)`` has
    mean ``exp(a^2 / 2)``, not 1: at the storm-scale amplitude ``a = 0.7``
    it is 1.27, so an ensemble whose members each take an uncorrected
    factor carries 27% more condensate (and number) in its mean than the
    background it was drawn around, before a single observation is used.
    Measured on the 2024-05-21 18Z 3 km crop with the module's own draw:
    factor mean 1.262, mass-weighted 1.266 (analytic 1.268 with the
    2.5-sigma clip).  Subtracting this value from the exponent makes the
    factor's expectation exactly 1 for the clipped draw, so the
    perturbation adds spread and no mass.

    With the clip at ``k`` sigma the exponent's distribution has point
    masses at ``+/- a k``, and the exact mean is

        ``exp(a^2/2) [Phi(k - a) - Phi(-k - a)] + Phi(-k) [exp(-a k) + exp(a k)]``

    (``Phi`` the standard normal CDF).  ``a = 0`` gives exactly 0, so a
    tapered rim whose amplitude is zero keeps its factor at exactly 1.
    Returns a float for a scalar ``amplitude``; a NumPy array in, an
    array out, evaluated per element.
    """
    k = float(clip_sigmas)
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"clip_sigmas must be positive and finite, got {clip_sigmas!r}")

    def _phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _one(a: float) -> float:
        a = float(a)
        if a == 0.0:
            return 0.0
        mean = (math.exp(0.5 * a * a) * (_phi(k - a) - _phi(-k - a))
                + _phi(-k) * (math.exp(-a * k) + math.exp(a * k)))
        return math.log(mean)

    if np.ndim(amplitude) == 0:
        return _one(amplitude)
    values = np.asarray(amplitude, dtype=np.float64)
    unique, inverse = np.unique(values, return_inverse=True)
    table = np.array([_one(a) for a in unique], dtype=np.float64)
    return table[inverse].reshape(values.shape)


def _lognormal_exponent(xp, draw, amplitude: float, clip_sigmas: float,
                        taper):
    """The mean-preserving lognormal exponent on the draw's backend.

    ``amplitude * clip(draw) * taper - log E[exp(amplitude * taper * clip(g))]``,
    the second term per column from :func:`clipped_lognormal_log_mean`
    (the taper scales the amplitude, so the correction follows it; a zero
    taper gives a zero correction and a factor of exactly 1).  Returns the
    exponent and the correction field ``(ny, nx)`` on the host.
    """
    taper_host = np.asarray(taper.get() if hasattr(taper, "get") else taper,
                            dtype=np.float64)
    correction = clipped_lognormal_log_mean(
        float(amplitude) * taper_host, clip_sigmas)
    scaled = (_clip_draw(xp, draw, clip_sigmas) * float(amplitude)
              * taper.astype(draw.dtype, copy=False)[None, :, :])
    exponent = scaled - xp.asarray(correction, dtype=draw.dtype)[None, :, :]
    return exponent, correction


def _state_field_names(state) -> tuple[str, ...]:
    """Every prognostic field the state actually carries, for pair
    detection.  ``None`` attributes are absent, not empty."""
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    return tuple(name for name in STATE_SERIALIZED_ATTRS
                 if getattr(state, name, None) is not None)


def _apply_species_perturbations(state, seed: int, cfg: PerturbationConfig,
                                 xp, mass_grid) -> list[dict[str, Any]]:
    """Scale each configured species and its moments by one common factor.

    The pair structure is *detected from the state* through
    :func:`gpuwm.da.moments.pairs_present` rather than configured, so a
    Morrison state contributes ``(qr, nr)`` and an NSSL state contributes
    ``(qr, qnr)`` -- and a single-moment state contributes neither, in
    which case the mass alone is scaled and the receipt says so.  That is
    the same detection the increment applier's guard uses, deliberately:
    two spellings of "which fields are a pair" is how a guard and the
    thing it guards drift apart.

    Returns one record per species.  Each carries the invariant this
    perturbation rests on as a MEASURED quantity, not an assertion:
    ``depleted_pairs_created`` is the number of cells that came out of
    this call holding mass above the activity threshold with a
    non-positive number moment AND did not arrive that way, and it is
    zero by construction.  Cells the background already carried that way
    -- a cold start with hydrometeor mass and no number concentration is
    the ordinary case -- are outside ``active``, are left exactly as
    found, and are counted in ``depleted_pairs_in_background``.  They are
    reported, never refused: this module neither made them nor can mend
    them, and refusing on them stopped forecasts it had not harmed.
    """
    if not cfg.species:
        return []
    from gpuwm.da.moments import pairs_present

    nz, ny, nx = mass_grid
    available = _state_field_names(state)
    pairs = {pair.mass: pair for pair in pairs_present(available)}
    records: list[dict[str, Any]] = []
    for spec in cfg.species:
        mass = getattr(state, spec.mass_field, None)
        if mass is None:
            raise ValueError(
                f"cannot perturb species {spec.mass_field!r}: the state does "
                f"not carry it. The state has "
                f"{sorted(n for n in available if n.startswith('q'))}; a "
                "species the scheme does not advance cannot be given spread")
        shape = tuple(int(e) for e in mass.shape)
        if shape != (nz, ny, nx):
            raise ValueError(
                f"state.{spec.mass_field} has shape {shape} but the mass "
                f"grid is {(nz, ny, nx)}; a hydrometeor on a staggered grid "
                "is not something this module knows how to interpret")
        if _array_module(mass) is not xp:
            raise TypeError(
                f"state.{spec.mass_field} is on a different array backend "
                "than state.thp; a half-migrated state is not a state")
        pair = pairs.get(spec.mass_field)
        partners: list[str] = []
        if pair is not None:
            partners.append(pair.number)
            if pair.volume is not None:
                partners.append(pair.volume)
        for name in partners:
            partner = getattr(state, name)
            if tuple(int(e) for e in partner.shape) != shape:
                raise ValueError(
                    f"state.{name} is {tuple(partner.shape)} but its own "
                    f"mass field {spec.mass_field} is {shape}; a moment pair "
                    "on two different grids is not a pair")

        _check_resolvable(shape, cfg.dx_km, cfg.dy_km, spec)
        draw, info = gaussian_random_field(
            shape, seed=seed, name=f"species:{spec.mass_field}",
            dx_km=cfg.dx_km, dy_km=cfg.dy_km,
            length_scale_km=spec.length_scale_km,
            vertical_scale_levels=spec.vertical_scale_levels,
            xp=xp, dtype=cfg.compute_dtype, fft_host=cfg.fft_host)
        taper = boundary_taper(ny, nx, cfg.rim_width, kind=cfg.rim_taper,
                               xp=xp, dtype=draw.dtype)
        # Mean-preserving: the exponent carries ``- log E[exp(.)]`` per
        # column (clipped_lognormal_log_mean), so the ensemble-mean mass
        # and number equal the background's.  Without it the mean took
        # exp(sigma^2/2): +27% condensate at sigma 0.7 before any
        # observation (the DA echo surplus's first term, 2026-10-06).
        exponent, mean_correction = _lognormal_exponent(
            xp, draw, spec.amplitude, spec.clip_sigmas, taper)
        factor = xp.exp(exponent)

        # ``active`` is where the background pair is JOINTLY usable.  A
        # cell whose mass is below the scheme's own gate is left exactly
        # alone -- scaling it up could promote a background cell that
        # already held q > 0 with N = 0 (legal below the gate, because the
        # scheme never reads the number there) into a genuine offender.
        threshold = mass.dtype.type(spec.threshold_kg_kg)
        active = mass > threshold
        if pair is not None:
            active = active & (getattr(state, pair.number) > 0)
        active = active & (taper > 0.0)[None, :, :]
        touched = int(xp.count_nonzero(active))

        applied = [spec.mass_field] + partners
        before_mass = float(xp.sum(mass.astype(np.float64)))
        # What the BACKGROUND already carries, measured before a single
        # array is scaled.  A cold start from a source with hydrometeor
        # mass and no number concentration arrives here with pairs the
        # scheme itself will close on its first call, and those cells are
        # not active, so nothing below touches them.  Counting the
        # post-state absolutely made the invariant read as violated by a
        # condition this module neither created nor can repair, and every
        # such forecast was refused after its state had been perturbed.
        pre_negative = mass < 0
        pre_depleted = (None if pair is None else
                        (mass > threshold) & (getattr(state, pair.number) <= 0))
        for name in applied:
            target = getattr(state, name)
            scaled = target * factor.astype(target.dtype, copy=False)
            target[...] = xp.where(active, scaled, target)
        after_mass = float(xp.sum(getattr(state, spec.mass_field)
                                  .astype(np.float64)))

        # The invariant, measured.  Positivity: a strictly positive factor
        # cannot take a non-negative field below zero.  Pair consistency:
        # the same factor on both moments cannot break the pair.  Both are
        # counted rather than claimed, because a claim in a docstring is
        # not a receipt.
        post_mass = getattr(state, spec.mass_field)
        negative = int(xp.count_nonzero((post_mass < 0) & ~pre_negative))
        carried_negative = int(xp.count_nonzero(pre_negative))
        depleted = 0
        carried_depleted = 0
        if pair is not None:
            post_number = getattr(state, pair.number)
            offending = (post_mass > threshold) & (post_number <= 0)
            depleted = int(xp.count_nonzero(offending & ~pre_depleted))
            carried_depleted = int(xp.count_nonzero(pre_depleted))
        record = {
            "species": spec.mass_field,
            "fields_scaled": applied,
            "moment_pair": (None if pair is None else
                            {"mass": pair.mass, "number": pair.number,
                             "volume": pair.volume}),
            "pair_source": ("detected from the state's own field spellings "
                            "via gpuwm.da.moments.pairs_present"),
            "amplitude_log_sigma": float(spec.amplitude),
            "amplitude_units": "log-space sigma (dimensionless)",
            "length_scale_km": float(spec.length_scale_km),
            "vertical_scale_levels": float(spec.vertical_scale_levels),
            "threshold_kg_kg": float(spec.threshold_kg_kg),
            "clip_sigmas": float(spec.clip_sigmas),
            "clipped_points": int(xp.count_nonzero(
                xp.abs(draw) > float(spec.clip_sigmas))),
            "shape": list(shape),
            "stream_key_hex": info["stream_key_hex"],
            "noise_sha256": info["noise_sha256"],
            "noise_dtype": info["noise_dtype"],
            "fft_backend": info["fft_backend"],
            "unit_field_realized_rms": info["realized_rms"],
            "vertical_wrap": info["vertical_wrap"],
            "active_points": touched,
            "total_points": int(nz * ny * nx),
            "factor_min": float(xp.min(factor)),
            "factor_max": float(xp.max(factor)),
            #: The factor's expectation is 1 by construction: the exponent
            #: carries ``- log E[exp(sigma * taper * clip(g))]`` per column.
            "mean_preserving": True,
            "mean_log_correction_max": float(np.max(mean_correction)),
            "mass_before_sum": before_mass,
            "mass_after_sum": after_mass,
            "negative_points": negative,
            "depleted_pairs_created": depleted,
            #: Present in the BACKGROUND and left exactly as found: these
            #: cells are outside ``active``, so no factor reached them.
            #: Reported rather than refused, because the perturbation did
            #: not make them and cannot mend them.
            "negative_points_in_background": carried_negative,
            "depleted_pairs_in_background": carried_depleted,
            "invariants": [
                "positivity: the factor exp(.) is strictly positive, so a "
                "non-negative field stays non-negative with no clipping and "
                "no mass repair (negative_points is the measurement)",
                "moment consistency: mass, number and volume take the SAME "
                "factor, so q/N -- and with it the scheme's slope closure -- "
                "is unchanged and no depleted pair can be CREATED "
                "(depleted_pairs_created is the measurement; pairs the "
                "background already carried are counted separately in "
                "depleted_pairs_in_background and are left untouched)",
                "clear air: the factor is applied only where the background "
                "pair is jointly active, so the ensemble carries spread in "
                "the hydrometeors the model made and invents none where the "
                "model made none",
            ],
        }
        if negative or depleted:
            raise ValueError(
                f"species perturbation of {spec.mass_field!r} CREATED "
                f"{negative} negative cell(s) and {depleted} depleted "
                "moment pair(s), which a strictly positive common factor "
                "cannot do. The background is not what this module assumed "
                "-- a non-finite factor, or a field that changed under it "
                "-- and the state is now perturbed; do not use it. "
                f"({carried_negative} negative cell(s) and "
                f"{carried_depleted} depleted pair(s) the background "
                "already carried are untouched and are not this refusal.)")
        records.append(record)
    return records


def device_working_bytes(cfg: PerturbationConfig,
                         mass_shape: Sequence[int],
                         plan_work_bytes: Mapping | None = None,
                         *, loading_masses: Sequence[str] | None = None
                         ) -> int:
    """Peak device bytes :func:`apply_perturbations` holds beside the state.

    A census of the arrays this module allocates on the state's device,
    for a caller that has to admit the perturbation before the state
    exists.  ``s`` is the compute dtype's width, ``M`` a field's points
    and ``K = N * ny * (nx // 2 + 1)`` its real-FFT spectrum points, with
    ``N`` the padded vertical FFT size.  Draw temporaries are priced on
    that padded shape; application arrays and the returned crop use the
    physical field shape.

    One draw (:func:`gaussian_random_field`) with the device FFT peaks at
    the largest of its stages: the forward transform (the working
    copy ``sM``, the spectrum ``2sK`` and the forward plan's work area),
    the spectrum multiply (the working copy, the old and new spectra
    ``2sK`` each and the kernel ``sK``), the inverse transform (the output
    ``sM`` beside the spectrum, the kernel, cuFFT's copy of its complex
    input and the inverse plan's work area), the normalisation (a second
    ``sM`` beside the first), the owned physical crop, and the realized RMS
    (a float64 copy and its square on the physical crop).  With ``fft_host``
    only the finished field reaches
    the device.

    ``loading_masses`` names the moist masses the state carries
    (:func:`gpuwm.da.hydrostatic.loading_masses_for_scheme`), which the
    hydrostatic balance copies; omitted, every one it can copy is priced.

    ``plan_work_bytes`` is :func:`fft_plan_work_bytes`'s answer, the work
    area each plan reported on the card that will run it; each plan lives
    only for its own transform.  Omitted, the plans are priced at zero,
    which only a caller with no device may do.

    The application loops keep their last iteration's arrays bound while
    the next draw runs: the draw, and the float32 increment (plus the
    exponent, the factor and the scaled field for a lognormal field, and
    the three Boolean masks and the scaled moment for a species).  Each
    field's record adds a float64 RMS of its increment.  Temperature
    fields hold the pressure and the Exner function for the whole call,
    and ``qv`` leaves its increment for the moisture bounds.
    """
    nz, ny, nx = (int(extent) for extent in mass_shape)
    width = 4 if cfg.compute_dtype == "float32" else 8
    mass_points = nz * ny * nx

    plans = {tuple(int(extent) for extent in shape): tuple(sizes)
             for shape, sizes in (plan_work_bytes or {}).items()}

    def draw_peak(shape, vertical_scale_levels) -> int:
        crop_points = math.prod(shape)
        if cfg.fft_host:
            return width * crop_points
        fft_shape, _ = _vertical_fft_shape(shape, vertical_scale_levels)
        points = math.prod(fft_shape)
        spectrum = fft_shape[0] * fft_shape[1] * (fft_shape[2] // 2 + 1)
        forward, inverse = plans.get(fft_shape, (0, 0))
        return max(width * points + 2 * width * spectrum + int(forward),
                   width * points + 5 * width * spectrum,
                   2 * width * points + 5 * width * spectrum + int(inverse),
                   3 * width * points + 3 * width * spectrum,
                   2 * width * points + width * crop_points + 3 * width * spectrum,
                   width * points + (width + 16) * crop_points + 3 * width * spectrum)

    held = 0
    if any(SUPPORTED_FIELDS[name].exner_from_temperature
           for name in cfg.field_names):
        held += 2 * 4 * mass_points
    peak = 0
    kept = 0

    # The hydrostatic mass balance (default since 2026-10-06): float64
    # copies of thp, p and every loading mass the state carries are taken
    # BEFORE the draws and held for the whole call, the same capture is
    # taken again after the bounds, and the column integration works in
    # float64 temporaries of the mass shape (two loading sums, two
    # recurrence outputs and their difference, the analysed pressure,
    # two inverse densities with their factors, the layer operator, the
    # thickness change, the summed dphp and the masked write).  The
    # census cannot see the state, so a caller that knows the scheme names
    # the masses it carries (``loading_masses``) and one that does not
    # gets all of them priced: an admission that under-prices this is an
    # out-of-memory on the first member, which is the breakage this line
    # prevents.  Pricing hail for a scheme that allocates none put a
    # 601 x 601 Thompson member over a 32 GiB card's 28 GiB budget
    # (tests/test_da_recent_case.py).
    column_capture = 0
    if cfg.mass_balance == "hydrostatic" and cfg.column_field_names:
        from gpuwm.da.hydrostatic import LOADING_MASSES
        carried = (LOADING_MASSES if loading_masses is None
                   else tuple(name for name in LOADING_MASSES
                              if name in set(loading_masses)))
        column_capture = 8 * (2 + len(carried)) * mass_points
        held += column_capture

    rotational = cfg.wind_mode == "rotational" and "u" in cfg.field_names
    if rotational:
        # ONE streamfunction draw on the corner grid; then psi beside the
        # two differences (each a temporary of its component's size
        # before the division writes the result); then both unit
        # components held until each is applied in the loop below.
        psi_shape = (nz, ny + 1, nx + 1)
        psi_points = math.prod(psi_shape)
        u_points = nz * ny * (nx + 1)
        v_points = nz * (ny + 1) * nx
        peak = max(peak,
                   held + draw_peak(psi_shape,
                                    cfg.spec("u").vertical_scale_levels),
                   held + width * (psi_points + 2 * u_points),
                   held + width * (psi_points + u_points + 2 * v_points))
        held += width * (u_points + v_points)

    for name in cfg.field_names:
        shape = _expected_shape(name, nz, ny, nx)
        points = math.prod(shape)
        if rotational and name in ("u", "v"):
            # The unit component already held becomes this iteration's
            # draw; nothing new is drawn.
            held -= width * points
        else:
            peak = max(peak, held + kept + draw_peak(
                shape, cfg.spec(name).vertical_scale_levels))
        kept = width * points + 4 * points
        if cfg.spec(name).mode == "lognormal":
            kept += 2 * width * points + 4 * points
        peak = max(peak, held + kept + 16 * points)
        if name == "qv":
            held += 4 * mass_points
    for _species in cfg.species:
        peak = max(peak, held + kept + draw_peak(
            (nz, ny, nx), _species.vertical_scale_levels))
        kept = 3 * width * mass_points + 3 * mass_points + 4 * mass_points
        peak = max(peak, held + kept + (width + 1) * mass_points)
    if column_capture:
        peak = max(peak, held + kept + column_capture + 14 * 8 * mass_points)
    return int(peak)


def apply_perturbations(state, seed: int, cfg: PerturbationConfig
                        ) -> dict[str, Any]:
    """Perturb ``state`` in place and return the provenance for that member.

    This exact signature is the contract the ensemble engine codes against.
    ``state`` is mutated; the return value is a JSON-serializable dict that
    fully identifies the perturbation (seed, per-field amplitudes and scales,
    the SHA-256 of each white-noise draw, the realized statistics, the taper,
    and every bound that fired).

    The state is expected to look like ``gpuwm.core.state.DomainState``:
    ``thp``/``qv`` on mass points ``(nz, ny, nx)``, ``u`` on ``(nz, ny,
    nx+1)``, ``v`` on ``(nz, ny+1, nx)``, backed by either NumPy or CuPy.
    Shapes are checked against each other; a field whose staggering does not
    match the mass grid is an error, not a broadcast.

    Read the module docstring for what balance is **not** imposed.  Nothing
    here re-balances mass or wind, and the members share one boundary file.

    **Caller post-condition.**  ``state.p``/``al``/``alt`` are diagnostics of
    ``(thp, php, mup, qv)``; perturbing ``thp`` and ``qv`` leaves them stale.
    The caller must run ``gpuwm.core.diagnostics.update_diagnostics(state,
    ...)`` before the first step.  This module deliberately does not call it:
    the diagnostic is a CUDA-only path and importing it here would make a
    NumPy-backed perturbation impossible.  The returned provenance records
    the requirement under ``"post_conditions"``.  For the same reason the
    supersaturation cap is evaluated against the pressure *as it stands on
    entry* -- a few-kelvin theta perturbation moves the pressure by well
    under a percent, so the cap is accurate to that, and it is a cap rather
    than an equality anyway.

    **Rim invariant.**  Wherever the taper is exactly zero this call is the
    identity, byte for byte -- including the moisture bounds, which are
    confined to the taper-active region.  A clamp that repaired the rim
    would break exactly the boundary consistency the taper exists to keep.

    Perturbing these fields does not disturb ``setup_fingerprint``
    (``gpuwm.state_serialization_contract``): ``u``/``v``/``thp``/``qv`` are
    serialized arrays, not setup arrays, so a perturbed member still restarts
    against the same base state and boundary tables.
    """
    if not isinstance(cfg, PerturbationConfig):
        raise TypeError(
            "cfg must be a PerturbationConfig (build one with "
            "PerturbationConfig.from_mapping for table-driven callers), got "
            f"{type(cfg).__name__}")
    if int(seed) != seed:
        raise TypeError(f"seed must be an integer, got {seed!r}")
    seed = int(seed)

    mass = _resolve_target(state, "theta")  # state.thp, always present
    if getattr(mass, "ndim", 0) != 3:
        raise ValueError(
            f"state.thp must be 3-D (nz, ny, nx), got shape "
            f"{getattr(mass, 'shape', None)}")
    xp = _array_module(mass)
    nz, ny, nx = (int(extent) for extent in mass.shape)

    names = cfg.field_names
    targets: dict[str, Any] = {}
    for name in names:
        array = _resolve_target(state, name)
        expected = _expected_shape(name, nz, ny, nx)
        if tuple(int(e) for e in array.shape) != expected:
            raise ValueError(
                f"state.{SUPPORTED_FIELDS[name].attribute} has shape "
                f"{tuple(array.shape)} but the ARW staggering for {name!r} "
                f"on this ({nz}, {ny}, {nx}) mass grid requires {expected}")
        if _array_module(array) is not xp:
            raise TypeError(
                f"state.{SUPPORTED_FIELDS[name].attribute} is on a "
                "different array backend than state.thp; a half-migrated "
                "state is not a state")
        targets[name] = array

    needs_pressure_for = [n for n in names
                          if SUPPORTED_FIELDS[n].exner_from_temperature]
    exner = None
    if needs_pressure_for:
        pressure = _require_pressure(
            state, xp, f"the temperature perturbation {needs_pressure_for}")
        exner = (pressure / c.P0) ** c.RCP

    column_before = _capture_column_for_balance(state, cfg, xp)

    rotational = None
    if cfg.wind_mode == "rotational" and "u" in names:
        wind = cfg.spec("u")
        _check_resolvable((nz, ny, nx + 1), cfg.dx_km, cfg.dy_km, wind)
        u_unit, v_unit, wind_info = rotational_wind_draw(
            (nz, ny, nx), seed=seed, dx_km=cfg.dx_km, dy_km=cfg.dy_km,
            length_scale_km=wind.length_scale_km,
            vertical_scale_levels=wind.vertical_scale_levels, xp=xp,
            dtype=cfg.compute_dtype, fft_host=cfg.fft_host)
        rotational = {"u": u_unit, "v": v_unit}
        del u_unit, v_unit

    field_records: list[dict[str, Any]] = []
    qv_increment = None
    for name in names:
        spec = cfg.spec(name)
        target = targets[name]
        shape = tuple(int(e) for e in target.shape)
        if rotational is not None and name in rotational:
            # One component of the streamfunction draw; it takes the
            # same rim taper a scalar does (see rotational_wind_draw for
            # why the taper is not on psi).
            draw = rotational.pop(name)
            info = wind_info
        else:
            _check_resolvable(shape, cfg.dx_km, cfg.dy_km, spec)
            draw, info = gaussian_random_field(
                shape, seed=seed, name=name, dx_km=cfg.dx_km,
                dy_km=cfg.dy_km, length_scale_km=spec.length_scale_km,
                vertical_scale_levels=spec.vertical_scale_levels,
                xp=xp, dtype=cfg.compute_dtype, fft_host=cfg.fft_host)
        taper = boundary_taper(shape[1], shape[2], cfg.rim_width,
                               kind=cfg.rim_taper, xp=xp,
                               dtype=draw.dtype)
        # Write ONLY where the taper is active.  ``target += increment``
        # over the whole array looks like the identity wherever the
        # increment is exactly zero, and is -- except for signed zero:
        # IEEE ``-0.0 + 0.0`` is ``+0.0``, so a rim holding -0.0 came
        # back numerically equal and byte-different, and the state sha
        # sees bytes.  Selecting with ``where`` keeps the original words.
        active = (taper > 0.0)[None, :, :]
        factor_record: dict[str, Any] | None = None
        if spec.mode == "lognormal":
            # The taper multiplies the EXPONENT, so a zero taper gives
            # exactly exp(0) = 1 and the rim is untouched by construction
            # as well as by the ``where``.
            # Mean-preserving, as the species factor is: the exponent
            # carries ``- log E[exp(.)]`` per column, so the ensemble
            # mean of the field equals the background's.
            exponent, mean_correction = _lognormal_exponent(
                xp, draw, spec.amplitude, spec.clip_sigmas, taper)
            factor = xp.exp(exponent)
            scaled = (target * factor.astype(target.dtype, copy=False))
            increment = scaled - target
            target[...] = xp.where(active, scaled, target)
            factor_record = {
                "factor_min": float(xp.min(factor)),
                "factor_max": float(xp.max(factor)),
                "mean_preserving": True,
                "mean_log_correction_max": float(np.max(mean_correction)),
                "clip_sigmas": float(spec.clip_sigmas),
                "clipped_points": int(xp.count_nonzero(
                    xp.abs(draw) > float(spec.clip_sigmas))),
            }
        else:
            increment = draw * float(spec.amplitude) * taper[None, :, :]
            if SUPPORTED_FIELDS[name].exner_from_temperature:
                increment = increment / exner
            increment = increment.astype(target.dtype, copy=False)
            target[...] = xp.where(active, target + increment, target)
        if name == "qv":
            qv_increment = xp.where(active, increment,
                                    xp.zeros_like(increment))

        record = {
            "name": name,
            "attribute": SUPPORTED_FIELDS[name].attribute,
            "stagger": SUPPORTED_FIELDS[name].stagger,
            "units": SUPPORTED_FIELDS[name].units,
            "amplitude": float(spec.amplitude),
            "length_scale_km": float(spec.length_scale_km),
            "vertical_scale_levels": float(spec.vertical_scale_levels),
            "shape": list(shape),
            "stream_key_hex": info["stream_key_hex"],
            "noise_sha256": info["noise_sha256"],
            "noise_dtype": info["noise_dtype"],
            "fft_backend": info["fft_backend"],
            "unit_field_realized_rms": info["realized_rms"],
            "increment_rms": float(
                xp.sqrt(xp.mean(increment.astype(np.float64) ** 2))),
            "increment_min": float(xp.min(increment)),
            "increment_max": float(xp.max(increment)),
            "vertical_wrap": info["vertical_wrap"],
            "mode": spec.mode,
        }
        if name in ("u", "v"):
            record["wind_mode"] = (
                "rotational" if info.get("wind_mode") == "rotational"
                else "independent")
        if info.get("wind_mode") == "rotational":
            record["unit_field_realized_rms"] = info[f"{name}_realized_rms"]
            record["streamfunction"] = {
                "shape": info["streamfunction_shape"],
                "length_scale_km": info["streamfunction_length_scale_km"],
                "scale_factor": info["scale_factor"],
                "noise_sha256": info["noise_sha256"],
                "psi_realized_rms": info["realized_rms"],
                "u_difference_variance_per_m2":
                    info["u_difference_variance_per_m2"],
                "v_difference_variance_per_m2":
                    info["v_difference_variance_per_m2"],
                "v_amplitude_ratio": info["v_amplitude_ratio"],
                "stencil": info["stencil"],
            }
            if name == "v":
                record["analytic_amplitude"] = (
                    float(spec.amplitude) * info["v_amplitude_ratio"])
        if factor_record is not None:
            record["lognormal"] = factor_record
            record["amplitude_units"] = "log-space sigma (dimensionless)"
        if SUPPORTED_FIELDS[name].exner_from_temperature:
            record["exner_converted"] = True
            record["applied_to"] = "potential temperature (theta')"
        field_records.append(record)

    species_records = _apply_species_perturbations(state, seed, cfg, xp,
                                                   (nz, ny, nx))

    bounds = _enforce_bounds(state, cfg, xp, perturbed=set(names),
                             qv_increment=qv_increment)

    mass_balance = _apply_mass_balance(state, cfg, column_before)
    del column_before

    combined = hashlib.sha256()
    for record in field_records:
        combined.update(record["noise_sha256"].encode("ascii"))
    for record in species_records:
        combined.update(record["noise_sha256"].encode("ascii"))
    return {
        "schema": PROVENANCE_SCHEMA,
        "module": "gpuwm.da.perturb",
        "status": STATUS,
        "seed": seed,
        "backend": "numpy" if xp is np else "cupy",
        #: "numpy" alongside a "cupy" backend means the device FFT was
        #: unavailable and the filtering ran on the host -- correct, slower,
        #: and never silent.
        "fft_backend": ("numpy" if (cfg.fft_host
                                    or not _device_fft_available(xp))
                        else ("numpy" if xp is np else "cupy")),
        #: True means the filter was pinned to the host on purpose, so this
        #: member is byte-reproducible against one perturbed on a domain
        #: living in pinned host RAM (the streamed execution mode).  False
        #: means the member is reproducible only against another member
        #: filtered on the same backend.
        "fft_host": bool(cfg.fft_host),
        "compute_dtype": cfg.compute_dtype,
        "mass_grid": [nz, ny, nx],
        "grid_spacing_km": {"dx": float(cfg.dx_km), "dy": float(cfg.dy_km)},
        "application_order": list(names),
        "fields": field_records,
        "species": species_records,
        "species_order": list(cfg.species_names),
        "noise_sha256": combined.hexdigest(),
        "taper": {
            "kind": cfg.rim_taper,
            "rim_width_cells": cfg.rim_width,
            "boundary_value": 0.0,
            "interior_value": 1.0,
            "axes": "lateral only (no vertical taper)",
        },
        "bounds": bounds,
        "wind_mode": cfg.wind_mode,
        "mass_balance": mass_balance,
        "post_conditions": [
            "wherever the rim taper is zero this call was the identity, "
            "byte for byte, bounds and mass balance included",
            "state.p / state.al / state.alt are now stale: run "
            "gpuwm.core.diagnostics.update_diagnostics(state, ...) before "
            "the first step (it folds the re-integrated php into p)",
            "the supersaturation cap was evaluated against the pressure as "
            "it stood on entry, not against the re-diagnosed pressure",
        ],
        "balance_imposed": [
            ("wind: u and v are C-grid differences of one streamfunction, "
             "so the mass-point divergence of the increment is zero "
             "wherever the rim taper is one; inside the rim band the "
             "component taper leaves a divergence of order u * grad(taper) "
             "(rotational)")
            if cfg.wind_mode == "rotational" else
            ("wind: none; u and v are independent draws and half their "
             "kinetic energy is divergent (comparison arm)"),
            ("mass: php re-integrated hydrostatically at the column's own "
             "dry mass after the thermodynamic draw")
            if mass_balance.get("applied") else
            "mass: none (" + str(mass_balance.get("reason")) + ")",
        ],
        "balance_not_imposed": [
            "mass: mu' is untouched (no surface-pressure perturbation)",
            "wind: the streamfunction and the theta draw are independent; "
            "no geostrophic, gradient-wind or thermal-wind coupling, and "
            "the map factor is not seen by the divergence stencil",
            "boundary: this call leaves the boundary tables alone; only "
            "the rim taper keeps the member consistent with them, until a "
            "caller attaches perturbed_lateral_boundaries for the member",
            "hydrometeors: a species factor scales the moments together, "
            "which preserves the drop size distribution exactly and the "
            "column's condensate loading not at all -- the perturbed "
            "member is not re-balanced for the buoyancy its new "
            "condensate mass implies",
            "vertical: the physical column is cropped from an enlarged "
            "independent-noise FFT draw; exact endpoint correlation is in "
            "each field's vertical_wrap.top_to_bottom_seam and its crop "
            "geometry is in vertical_wrap. This "
            "initial-condition call does not apply vertical attenuation",
        ],
    }


def _mass_balance_fields(state, cfg: PerturbationConfig) -> tuple[str, ...]:
    """The configured column fields the state actually carries."""
    return tuple(name for name in cfg.column_field_names
                 if name in ("t", "theta")
                 or getattr(state, name, None) is not None)


def _capture_column_for_balance(state, cfg: PerturbationConfig, xp):
    """Snapshot the column (``thp``, the loading masses, the diagnosed
    ``p``) BEFORE the thermodynamic draws, or ``None`` when the mass
    balance has nothing to do.  Every refusal is raised here, before a
    single field is written, so a refused call leaves the state alone."""
    if cfg.mass_balance != "hydrostatic":
        return None
    if not _mass_balance_fields(state, cfg):
        return None
    if cfg.hypsometric_opt is None:
        raise ValueError(
            "mass_balance 'hydrostatic' needs hypsometric_opt (the run's "
            "WRF option, 1 or 2) to integrate the column with the same "
            "layer operator the dycore diagnoses pressure with, and a "
            "state does not carry its run configuration. State it in the "
            "PerturbationConfig, or set mass_balance = 'none' for a state "
            "without a vertical coordinate")
    from gpuwm.da import hydrostatic as hydro

    missing = [attr for attr in hydro.COLUMN_SETUP_ATTRS
               if getattr(state, attr, None) is None]
    if missing:
        raise ValueError(
            "mass_balance 'hydrostatic' cannot integrate a column on this "
            "state: it carries no " + ", ".join(missing) + ". A state "
            "without a loaded base needs mass_balance = 'none' rather "
            "than a silent zero")
    if int(cfg.hypsometric_opt) == 2 and getattr(state, "p_top", None) is None:
        raise ValueError(
            "mass_balance 'hydrostatic' with hypsometric_opt=2 needs "
            "state.p_top (load_base); a state without a loaded base "
            "needs mass_balance = 'none'")
    if int(cfg.hypsometric_opt) == 1:
        dnw = getattr(state, "dnw", None)
        if dnw is None or not bool(_array_module(dnw).any(dnw)):
            raise ValueError(
                "mass_balance 'hydrostatic' with hypsometric_opt=1 needs a "
                "vertical coordinate (state.dnw is absent or all zero), "
                "else every layer operator is zero and the balance is a "
                "silent no-op; a state without a loaded base needs "
                "mass_balance = 'none'")
    _require_pressure(state, xp, "the hydrostatic mass balance")
    return hydro.capture_column(state)


def _apply_mass_balance(state, cfg: PerturbationConfig, column_before
                        ) -> dict[str, Any]:
    """Re-integrate ``php`` so the perturbed columns start hydrostatic."""
    if cfg.mass_balance != "hydrostatic":
        return {"applied": False, "mode": cfg.mass_balance,
                "reason": "mass_balance = 'none' (configured)"}
    fields = _mass_balance_fields(state, cfg)
    if column_before is None:
        return {"applied": False, "mode": cfg.mass_balance,
                "reason": "no thermodynamic or species perturbation was "
                          "configured, so the column did not change"}
    from gpuwm.da import hydrostatic as hydro

    receipt = hydro.rebalance_columns(
        state, column_before, hypsometric_opt=int(cfg.hypsometric_opt),
        names=fields, how="perturbation, after the draws and bounds")
    receipt["mode"] = cfg.mass_balance
    receipt["perturbed_column_fields"] = list(fields)
    return receipt


def _enforce_bounds(state, cfg: PerturbationConfig, xp,
                    perturbed: Iterable[str], qv_increment
                    ) -> dict[str, Any]:
    """Clamp the perturbed moisture field, and report what the clamp did.

    Runs whenever moisture was perturbed.  Leaving a mixing ratio negative is
    not a rounding detail: the microphysics will happily advect it, and the
    first positive-definite renormalization will manufacture mass to hide it.

    **The clamp is confined to the taper-active region.**  Where the taper is
    exactly zero this function is the identity, byte for byte, even if the
    incoming state violates the bound there.  Two reasons.  The rim has to
    match the shared boundary file, and a clamp that "helpfully" repaired it
    would break exactly the consistency the taper exists to preserve; and a
    perturbation module that silently repairs a state defect it did not
    cause is a module that hides the defect.  Pre-existing violations inside
    the perturbed region are counted and reported rather than hidden.
    """
    perturbed = set(perturbed)
    report: dict[str, Any] = {
        "qv_floor": float(cfg.qv_floor),
        "qv_floor_clipped_points": 0,
        "rh_cap": None if cfg.rh_cap is None else float(cfg.rh_cap),
        "rh_cap_clipped_points": 0,
        "evaluated": False,
        "scope": "taper-active points only; the untapered rim is returned "
                 "byte-identical even where it violates a bound",
    }
    if "qv" not in perturbed:
        report["skipped_reason"] = "no moisture perturbation was configured"
        return report

    qv = _resolve_target(state, "qv")
    report["evaluated"] = True
    ny, nx = int(qv.shape[1]), int(qv.shape[2])
    active = boundary_taper(ny, nx, cfg.rim_width, kind=cfg.rim_taper,
                            xp=xp, dtype=np.float64) > 0.0
    active = active[None, :, :]

    # The incoming state, reconstructed BEFORE either clamp runs.  Doing
    # it afterwards -- ``qv - qv_increment`` on the already-clipped field
    # -- reconstructed the ceiling minus the increment at every clipped
    # point, which is not the state that arrived, and undercounted
    # pre-existing supersaturation wherever the increment was positive.
    incoming = None
    if cfg.rh_cap is not None and qv_increment is not None:
        incoming = qv - qv_increment

    floor = qv.dtype.type(cfg.qv_floor)
    breaches = active & (qv < floor)
    below = int(xp.count_nonzero(breaches))
    if below:
        qv[...] = xp.where(breaches, floor, qv)
    report["qv_floor_clipped_points"] = below

    if cfg.rh_cap is not None:
        pressure = _require_pressure(state, xp, "the supersaturation cap")
        theta = _total_theta(state, xp)
        temperature = theta * (pressure / c.P0) ** c.RCP
        qvs = _saturation_mixing_ratio(temperature, pressure, xp)
        ceiling = (qvs * cfg.rh_cap).astype(qv.dtype, copy=False)
        breaches = active & (qv > ceiling)
        above = int(xp.count_nonzero(breaches))
        if above:
            qv[...] = xp.where(breaches, ceiling, qv)
        report["rh_cap_clipped_points"] = above
        if incoming is not None:
            # What the incoming state was already doing, so a repair this
            # module performs is visible in the manifest rather than silent.
            report["pre_existing_supersaturated_points"] = int(
                xp.count_nonzero(active & (incoming > ceiling)))
            report["pre_existing_basis"] = (
                "qv as it stood on entry (post-perturbation minus the "
                "increment, snapshotted before either clamp), against the "
                "same ceiling the cap applied")
        report["saturation_formula"] = (
            "Tetens over liquid water, gpuwm.core.constants "
            "SVP1/SVP2/SVP3/SVPT0 with EP2")
    else:
        report["skipped_reason"] = "rh_cap is None (cap disabled by config)"
    return report


# --------------------------------------------------------------------------
# Per-member lateral boundaries and additive inflation
# --------------------------------------------------------------------------

#: Provenance schema of :func:`perturbed_lateral_boundaries`.
BOUNDARY_PERTURBATION_SCHEMA = "gpuwm.da.perturb/lateral-boundaries/v1"

#: Provenance schema of :func:`additive_inflation`.
ADDITIVE_INFLATION_SCHEMA = "gpuwm.da.perturb/additive-inflation/v1"

#: Default e-folding time of a member's boundary perturbation between
#: boundary frames.  The perturbation stands in for the driving model's own
#: error, which decorrelates on synoptic time scales; six hours keeps
#: hourly frames strongly correlated (0.85 frame to frame) so the forcing
#: does not jump, and lets a three-hourly series wander (0.61).
DEFAULT_BOUNDARY_TIME_SCALE_HOURS = 6.0

#: Which boundary table each perturbable field lands on.  Boundary tables
#: are in WRF's coupled units (gpuwm/ingest/lateral_bc.py
#: ``_coupled_device_fields``): ``theta`` is the coupled perturbation
#: theta, so both a temperature and a potential-temperature amplitude land
#: there.
_BOUNDARY_TABLE = {"u": "u", "v": "v", "theta": "theta", "t": "theta",
                   "qv": "qv"}


def boundary_coupling_weights(state) -> dict[str, np.ndarray]:
    """Host ``float64`` weights that turn an uncoupled increment into the
    coupled units the boundary tables hold, per table, on each field's own
    stagger.

    The SAME formula as ``gpuwm.ingest.lateral_bc._coupled_device_fields``
    (u and v: half-level mass at the face, the boundary faces taking their
    adjacent cell's mass, divided by the map factor; scalars: half-level
    mass), evaluated on ``state`` as it stands.  A boundary perturbation is
    a fraction of a field's spread, so weighting every frame by the
    starting state's column mass is a sub-percent approximation of the
    amplitude, and it is the one this module states.  ``"exner"`` is the
    mass-point Exner function, for a temperature amplitude.
    """
    mu = _to_host(state.total_mu()).astype(np.float64)
    mux = 0.5 * (mu + np.roll(mu, 1, axis=1))
    mux = np.concatenate([mux, mux[:, :1]], axis=1)
    muy = 0.5 * (mu + np.roll(mu, 1, axis=0))
    muy = np.concatenate([muy, muy[:1, :]], axis=0)
    mux[:, 0] = mu[:, 0]
    mux[:, -1] = mu[:, -1]
    muy[0, :] = mu[0, :]
    muy[-1, :] = mu[-1, :]
    c1h = _to_host(state.c1h).astype(np.float64)[:, None, None]
    c2h = _to_host(state.c2h).astype(np.float64)[:, None, None]
    weights = {
        "u": c1h * mux[None] + c2h,
        "v": c1h * muy[None] + c2h,
        "theta": c1h * mu[None] + c2h,
    }
    weights["qv"] = weights["theta"]
    if getattr(state, "has_msf", False):
        weights["u"] = weights["u"] / _to_host(state.msfu).astype(
            np.float64)[None]
        weights["v"] = weights["v"] / _to_host(state.msfv).astype(
            np.float64)[None]
    pressure = getattr(state, "p", None)
    if pressure is not None:
        weights["exner"] = (_to_host(pressure).astype(np.float64)
                            / c.P0) ** c.RCP
    return weights


def _boundary_frames(intervals) -> list[float]:
    """Frame times (s): every interval's start, then the last one's end."""
    times = [float(interval.start_seconds) for interval in intervals]
    times.append(float(intervals[-1].end_seconds))
    return times


def boundary_perturbation_unavailable(boundaries) -> str | None:
    """Why a member's boundaries cannot be perturbed here, or ``None``.

    A state with no boundary tables (a periodic or idealized domain) has
    nothing to perturb, and a streamed series (its intervals declare
    ``bounds`` and load as the run reaches them) cannot have a member's
    perturbation built ahead of intervals that may not be prepared yet.
    The caller records the reason and runs the shared tables.
    """
    if boundaries is None:
        return "the state carries no lateral boundary tables"
    intervals = getattr(boundaries, "intervals", None)
    if intervals is None:
        return (f"the boundary object {type(boundaries).__name__} holds no "
                "interval series (a rolling nest boundary is the parent's)")
    if getattr(intervals, "bounds", None) is not None:
        return ("a streamed boundary series loads its intervals as the run "
                "reaches them, so a member's perturbation cannot be built "
                "ahead of them; this member runs the shared tables")
    return None


def perturbed_lateral_boundaries(boundaries, cfg: PerturbationConfig, *,
                                 seed: int, coupling: Mapping[str, Any],
                                 scale: float = 1.0,
                                 time_scale_hours: float =
                                 DEFAULT_BOUNDARY_TIME_SCALE_HOURS,
                                 ) -> tuple[Any, dict[str, Any]]:
    """One member's own lateral boundary forcing: ``(boundaries, record)``.

    Why.  Members that share one boundary file lose their spread toward the
    rim as the forecast runs, because the unperturbed inflow floods the
    domain; on a small storm-scale domain that matters more than the
    initial perturbation does.  So each member's boundary tables get a
    perturbation of their own.

    What.  For every field of ``cfg.fields`` the boundary tables carry
    (``u``, ``v``, ``theta`` from a ``"theta"`` or ``"t"`` spec, ``qv``), a
    unit Gaussian random field with the spec's own length scales is drawn
    at every boundary frame and sliced to the four sides exactly as the
    boundary builder slices a snapshot
    (:func:`gpuwm.ingest.lateral_bc.extract_lateral_side`):

    * frame 0 (the run's start) gets NO perturbation: the member's initial
      state is tapered to the shared boundary at the rim
      (:func:`apply_perturbations`), and the boundary must agree with it;
    * frame 1 is the member's OWN initial draw -- the same seed, field name
      and length scales :func:`apply_perturbations` used, untapered -- so
      the forcing ramps over the first interval into the pattern the
      member's interior already carries;
    * frame ``j >= 2`` is ``rho * Z(j-1) + sqrt(1 - rho**2) * G(j)``, an
      AR(1) in time with ``rho = exp(-dt / time_scale)`` and ``G(j)`` a
      fresh draw on its own stream, so each frame keeps unit variance and
      the forcing never jumps.

    Additive fields add ``scale * amplitude * Z`` times the coupling weight
    (:func:`boundary_coupling_weights`), divided by the Exner function for a
    ``"t"`` spec.  A ``"lognormal"`` ``qv`` multiplies the coupled table by
    ``exp(scale * amplitude * clip(Z))``: positive by construction, as the
    initial perturbation is.  Hydrometeor and number tables, ``phi`` and
    ``mu`` are not perturbed (stated in the record).

    Between frames the perturbation follows the interval's own time law:
    the value gains the start frame's perturbation and the tendency is
    chosen so the interval reaches the end frame's perturbation exactly at
    its end (with a rational law ``value + t (tendency + t q) / (1 + t d)``
    the added tendency is ``(P_end - P_start) (1 + T d) / T``).

    Deterministic in ``(seed, field, grid, frame)``: the same member gets
    the same boundaries on every leg and in every process, which is what
    keeps its restart sets' setup fingerprint (which hashes the attached
    tables) stable across legs.
    """
    from gpuwm.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         LateralBoundaries, SideBoundary,
                                         evaluate_boundary_side,
                                         extract_lateral_side)

    if not isinstance(cfg, PerturbationConfig):
        raise TypeError("cfg must be a PerturbationConfig")
    if int(seed) != seed:
        raise TypeError(f"seed must be an integer, got {seed!r}")
    scale = float(scale)
    time_scale_s = float(time_scale_hours) * 3600.0
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(f"scale must be finite and >= 0, got {scale!r}")
    if not math.isfinite(time_scale_s) or time_scale_s <= 0.0:
        raise ValueError(
            f"time_scale_hours must be positive, got {time_scale_hours!r}")
    intervals = tuple(boundaries.intervals)
    if getattr(boundaries.intervals, "bounds", None) is not None:
        raise ValueError(
            "a streamed boundary series holds intervals that may not be "
            "prepared yet, so a member's perturbation cannot be built ahead "
            "of them; perturb an eagerly attached series")
    width = int(boundaries.spec_bdy_width)
    frames = _boundary_frames(intervals)
    available = set(intervals[0].fields)
    record: dict[str, Any] = {
        "schema": BOUNDARY_PERTURBATION_SCHEMA, "status": STATUS,
        "seed": int(seed), "scale": scale,
        "time_scale_hours": float(time_scale_hours),
        "frames_seconds": frames, "fields": [],
        "not_perturbed": sorted(
            name for name in available
            if name not in {_BOUNDARY_TABLE[spec.name]
                            for spec in cfg.fields}),
    }
    if scale == 0.0:
        record["note"] = "scale 0: the shared boundaries, unchanged"
        return boundaries, record

    # Side perturbations per table, per frame: {table: [frame -> sides]}.
    added: dict[str, list] = {}
    plans = []
    for spec in cfg.fields:
        table = _BOUNDARY_TABLE[spec.name]
        if table not in available:
            record["fields"].append({"name": spec.name, "table": table,
                                     "perturbed": False,
                                     "reason": "no such boundary table"})
            continue
        weight = coupling.get(table)
        if spec.mode == "additive" and weight is None:
            raise ValueError(
                f"{spec.name}: no coupling weight for the {table!r} table; "
                "build them with boundary_coupling_weights(state)")
        shape = (tuple(int(n) for n in weight.shape) if weight is not None
                 else None)
        if shape is None:
            west = intervals[0].fields[table].west.value
            south = intervals[0].fields[table].south.value
            shape = (int(west.shape[0]), int(west.shape[1]),
                     int(south.shape[2]))
        exner = None
        if spec.name == "t":
            exner = coupling.get("exner")
            if exner is None:
                raise ValueError(
                    "a temperature amplitude needs the Exner function "
                    "(boundary_coupling_weights(state) on a state with p)")
        record["fields"].append({
            "name": spec.name, "table": table, "perturbed": True,
            "mode": spec.mode, "amplitude": float(spec.amplitude),
            "length_scale_km": float(spec.length_scale_km),
            "vertical_scale_levels": float(spec.vertical_scale_levels),
            "frame_streams": ([spec.name] + [
                f"{spec.name}/lateral-boundary/frame{i}"
                for i in range(2, len(frames))]),
            "uncoupled_rms_per_frame": None,
        })
        plans.append((spec, table, weight, shape, exner, record["fields"][-1]))

    def field_frames(plan):
        """One field's side perturbation at every frame: ``per_frame, rms``.

        Every frame's draw is independent of the others (its own stream), so
        they are drawn on threads -- NumPy's Philox fill, its FFT and
        hashlib all run without the GIL -- and combined in frame order, the
        serial result byte for byte; the fields run side by side the same
        way.  Drawn one after another they were 40 s of one host core per
        member at the start of every 9 km CONUS run (box L probe,
        2026-10-06), with the member's card idle meanwhile.
        """
        spec, table, weight, shape, exner, _entry = plan

        def frame_draw(index):
            name = (spec.name if index == 1
                    else f"{spec.name}/lateral-boundary/frame{index}")
            draw, _info = gaussian_random_field(
                shape, seed=int(seed), name=name, dx_km=cfg.dx_km,
                dy_km=cfg.dy_km, length_scale_km=spec.length_scale_km,
                vertical_scale_levels=spec.vertical_scale_levels,
                xp=np, dtype=cfg.compute_dtype, fft_host=True)
            return np.asarray(draw, dtype=np.float64)

        draws = _boundary_draws(frame_draw, range(1, len(frames)))
        per_frame = [None]
        z = None
        rms = []
        for index in range(1, len(frames)):
            draw = draws.pop(index)
            if z is None:
                z = draw
            else:
                rho = math.exp(-(frames[index] - frames[index - 1])
                               / time_scale_s)
                z = rho * z + math.sqrt(1.0 - rho * rho) * draw
            if spec.mode == "lognormal":
                exponent = (_clip_draw(np, z, spec.clip_sigmas)
                            * float(spec.amplitude) * scale)
                per_frame.append(("factor", extract_lateral_side_all(
                    extract_lateral_side, {table: np.exp(exponent)},
                    width)[table]))
            else:
                increment = z * float(spec.amplitude) * scale
                if exner is not None:
                    increment = increment / exner
                coupled = increment * weight
                rms.append(float(np.sqrt(np.mean(increment ** 2))))
                per_frame.append(("add", extract_lateral_side_all(
                    extract_lateral_side, {table: coupled}, width)[table]))
        return per_frame, rms

    for plan, (per_frame, rms) in zip(
            plans, _boundary_draws(lambda k: field_frames(plans[k]),
                                   range(len(plans))).values()):
        added[plan[1]] = per_frame
        plan[5]["uncoupled_rms_per_frame"] = rms or None

    def side_perturbation(table, frame, side_name, base_value):
        entry = added[table][frame]
        if entry is None:
            return np.zeros_like(base_value)
        kind, sides = entry
        if kind == "factor":
            return base_value * (sides[side_name] - 1.0)
        return sides[side_name]

    new_intervals = []
    for index, interval in enumerate(intervals):
        duration = float(interval.end_seconds - interval.start_seconds)
        fields = {}
        for table, boundary in interval.fields.items():
            if table not in added:
                fields[table] = boundary
                continue
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                side = getattr(boundary, side_name)
                start = np.asarray(side.value, dtype=np.float64)
                end = evaluate_boundary_side(side, duration)[0]
                p_start = side_perturbation(table, index, side_name, start)
                p_end = side_perturbation(table, index + 1, side_name, end)
                rate = (p_end - p_start) / duration
                if side.time_law is not None:
                    rate = rate * (1.0 + duration * np.asarray(
                        side.time_law.denominator_rate, dtype=np.float64))
                sides[side_name] = SideBoundary(
                    start + p_start,
                    np.asarray(side.tendency, dtype=np.float64) + rate,
                    side.time_law)
            fields[table] = FieldBoundary(**sides)
        new_intervals.append(BoundaryInterval(
            interval.start_seconds, interval.end_seconds, fields))
    perturbed = LateralBoundaries(
        tuple(new_intervals), boundaries.spec_bdy_width,
        boundaries.spec_zone, boundaries.relax_zone,
        seam_sides=tuple(boundaries.seam_sides))
    digest = hashlib.sha256()
    for interval in new_intervals:
        for table in sorted(interval.fields):
            for side_name in ("west", "east", "south", "north"):
                side = getattr(interval.fields[table], side_name)
                digest.update(np.ascontiguousarray(side.value).tobytes())
                digest.update(np.ascontiguousarray(side.tendency).tobytes())
    record["tables_sha256"] = digest.hexdigest()
    return perturbed, record


#: Threads one boundary field's frames are drawn on (the fields of a member
#: run side by side too, so a member uses up to four times this many).
BOUNDARY_DRAW_THREADS = 5


def _boundary_draws(draw, indices) -> dict:
    """``{index: draw(index)}``, drawn on up to :data:`BOUNDARY_DRAW_THREADS`
    threads.  Each draw is a pure function of its index, so the result is
    the serial one whatever order the threads finish in."""
    indices = list(indices)
    if len(indices) < 2:
        return {index: draw(index) for index in indices}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(
            max_workers=min(BOUNDARY_DRAW_THREADS, len(indices)),
            thread_name_prefix="boundary-draw") as pool:
        return dict(zip(indices, pool.map(draw, indices)))


def extract_lateral_side_all(extract, snapshot, width):
    """``{table: {side: array}}`` for every side, by the builder's slicer."""
    out: dict[str, dict[str, np.ndarray]] = {name: {} for name in snapshot}
    for side_name in ("west", "east", "south", "north"):
        for name, array in extract(snapshot, side_name, width).items():
            out[name][side_name] = np.asarray(array, dtype=np.float64)
    return out


def additive_inflation(priors: Mapping[int, Mapping[str, Any]],
                       cfg: PerturbationConfig, *, seed: int, leg: int,
                       scale: float, weight=None,
                       stream: str = "additive-inflation",
                       ) -> tuple[dict[int, dict], dict]:
    """Additive inflation after an analysis: ``(noise_by_member, record)``.

    Why.  RTPS and RTPP only rescale the spread the ensemble still has;
    neither can restore spread lost in the forecast step or rank lost by a
    collapsed subspace, and members started from one draw and relaxed every
    analysis lose both (Mitchell and Houtekamer 2000; Dowell and Wicker
    2009 add noise after each storm-scale analysis for the same reason).

    What.  For every member ``m`` (the keys of ``priors``) and every field
    of ``cfg.fields``, a fresh smooth draw from the module's own generator:
    seed ``seed + m``, stream ``"<field>/additive-inflation/leg<leg>"``, the
    spec's own length scales, host Philox and host FFT (so the draw is the
    same on every machine), tapered at the rim like the initial
    perturbation.  Additive fields add ``scale * amplitude * draw``
    (divided by the Exner function of the prior's ``p`` for a ``"t"``
    spec); a ``"lognormal"`` ``qv`` adds ``qv * (exp(scale * amplitude *
    clip(draw)) - 1)``.  Then the ENSEMBLE MEAN of the added noise is
    removed at every cell, so the analysis mean is exactly the filter's
    and only the spread grows.

    ``scale`` is the fraction of the initial perturbation's amplitude added
    per analysis; ``0`` returns ``({}, record)``.  Hydrometeor species are
    not inflated: a multiplicative factor there must move each species'
    moments together and is left to the model, which rebuilds condensate
    spread from the inflated wind, temperature and vapour within a leg.

    ``weight`` (``(ny, nx)`` on mass points, values in [0, 1]) scales the
    noise column by column, carried to the u and v faces by averaging the
    two adjacent mass columns; ``stream`` names the draw stream, so a
    weighted call (:func:`echo_weight`, audit S5) never repeats the
    domain-wide one.

    Returns host arrays keyed by state attribute (``u``, ``v``, ``thp``,
    ``qv``) in each prior's dtype.  Members with no key in ``priors`` get
    nothing; one member gets nothing (an ensemble of one has no mean to
    keep, and the noise would be all bias).
    """
    if not isinstance(cfg, PerturbationConfig):
        raise TypeError("cfg must be a PerturbationConfig")
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(f"scale must be finite and >= 0, got {scale!r}")
    members = sorted(int(index) for index in priors)
    record: dict[str, Any] = {
        "schema": ADDITIVE_INFLATION_SCHEMA, "status": STATUS,
        "scale": scale, "leg": int(leg), "seed": int(seed),
        "members": len(members), "fields": [],
        "mean_removed": True, "rim_width_cells": cfg.rim_width,
        "species_inflated": False,
    }
    if scale == 0.0 or len(members) < 2:
        record["note"] = ("scale 0: nothing added" if scale == 0.0 else
                          "one member: no ensemble mean to keep")
        return {}, record
    noise: dict[int, dict] = {index: {} for index in members}
    # Members are drawn side by side: each draw is its own seeded stream
    # and host FFT, so threads change when it is computed, never what.  One
    # after another this was about 0.5 s a draw, 32 members x four fields x
    # (inflation + echo noise), two minutes of every 9 km CONUS analysis.
    from concurrent.futures import ThreadPoolExecutor

    from gpuwm.da import ensemble_stats

    def member_added(name, spec, attribute, index):
        prior = priors[index]
        if attribute not in prior:
            raise ValueError(
                f"member {index}: the prior carries no {attribute!r}, "
                f"which the {name!r} inflation draws on")
        target = np.asarray(prior[attribute])
        shape = tuple(int(n) for n in target.shape)
        stream_name = f"{name}/{stream}/leg{int(leg)}"
        draw, _ = gaussian_random_field(
            shape, seed=int(seed) + index, name=stream_name,
            dx_km=cfg.dx_km, dy_km=cfg.dy_km,
            length_scale_km=spec.length_scale_km,
            vertical_scale_levels=spec.vertical_scale_levels,
            xp=np, dtype=cfg.compute_dtype, fft_host=True)
        draw = np.asarray(draw, dtype=np.float64)
        taper = boundary_taper(shape[1], shape[2], cfg.rim_width,
                               kind=cfg.rim_taper, xp=np,
                               dtype=np.float64)[None]
        if weight is not None:
            taper = taper * _weight_on(weight, shape[1:])[None]
        if spec.mode == "lognormal":
            exponent = (_clip_draw(np, draw, spec.clip_sigmas)
                        * float(spec.amplitude) * scale * taper)
            added = target.astype(np.float64) * np.expm1(exponent)
        else:
            added = draw * float(spec.amplitude) * scale * taper
            if SUPPORTED_FIELDS[name].exner_from_temperature:
                if "p" not in prior:
                    raise ValueError(
                        f"member {index}: a temperature amplitude "
                        "needs the prior's pressure 'p'")
                added = added / (np.asarray(prior["p"], np.float64)
                                 / c.P0) ** c.RCP
        return added

    workers = max(1, min(32, len(members)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in cfg.field_names:
            spec = cfg.spec(name)
            attribute = SUPPORTED_FIELDS[name].attribute
            stack = np.stack(list(pool.map(
                lambda index: member_added(name, spec, attribute, index),
                members)))
            # stack -= stack.mean(axis=0, keepdims=True), span by span: the
            # member mean of a point is the same sequential sum whichever
            # thread forms it.
            flat = stack.reshape(len(members), -1)

            def demean(span, _flat=flat):
                lo, hi = span
                block = _flat[:, lo:hi]
                block -= block.mean(axis=0, keepdims=True)

            list(pool.map(demean, ensemble_stats._spans(
                flat.shape[1], 4 * workers)))
            for slot, index in enumerate(members):
                noise[index][attribute] = stack[slot].astype(
                    np.asarray(priors[index][attribute]).dtype)
            squares = np.empty_like(flat)

            def square(span, _flat=flat, _out=squares):
                lo, hi = span
                np.multiply(_flat[:, lo:hi], _flat[:, lo:hi],
                            out=_out[:, lo:hi])

            list(pool.map(square, ensemble_stats._spans(
                flat.shape[1], 4 * workers)))
            record["fields"].append({
                "name": name, "attribute": attribute, "mode": spec.mode,
                "amplitude_fraction": scale,
                "amplitude": float(spec.amplitude) * scale,
                "length_scale_km": float(spec.length_scale_km),
                "added_rms": float(np.sqrt(
                    ensemble_stats.exact_sum(squares.reshape(-1))
                    / squares.size)),
                "added_max_abs": max(pool.map(
                    lambda span, _flat=flat: float(
                        np.abs(_flat[:, span[0]:span[1]]).max()),
                    ensemble_stats._spans(flat.shape[1], 4 * workers))),
            })
            del stack, flat, squares
    return noise, record


def _weight_on(weight, shape) -> np.ndarray:
    """A mass-point ``(ny, nx)`` weight on ``shape``: itself, or averaged
    onto the u (``nx + 1``) or v (``ny + 1``) faces."""
    w = np.asarray(weight, dtype=np.float64)
    ny, nx = int(shape[0]), int(shape[1])
    if w.shape == (ny, nx):
        return w
    if w.shape == (ny, nx - 1):
        padded = np.pad(w, ((0, 0), (1, 1)), mode="edge")
        return 0.5 * (padded[:, 1:] + padded[:, :-1])
    if w.shape == (ny - 1, nx):
        padded = np.pad(w, ((1, 1), (0, 0)), mode="edge")
        return 0.5 * (padded[1:, :] + padded[:-1, :])
    raise ValueError(f"weight {w.shape} does not sit on a field {shape}")


#: Default observed-echo threshold (dBZ) of :func:`echo_weight`.  Dowell
#: and Wicker (2009, J. Atmos. Oceanic Technol. 26, 911-927) add their
#: noise where observed reflectivity exceeds 25 dBZ.
ECHO_NOISE_THRESHOLD_DBZ = 25.0


def echo_weight(z_obs, z_mask, *, threshold_dbz: float =
                ECHO_NOISE_THRESHOLD_DBZ, dx_km: float, dy_km: float,
                length_scale_km: float) -> np.ndarray:
    """``(ny, nx)`` weight in [0, 1]: 1 in columns where the radar observed
    echo at or above ``threshold_dbz`` at any level, spread smoothly over
    about one ``length_scale_km`` around them (a Gaussian of that scale,
    renormalised so a large echo region sits at 1).

    Why (audit S5).  An ensemble filter can only build echo some member
    has: where no member has the storm the background covariance between
    reflectivity and the state is zero and the observation moves nothing.
    Dowell and Wicker (2009) answer this by adding smooth noise to wind,
    temperature and moisture where radar observes echo, so the members
    differ there and the next analyses have spread to work with.  This is
    that weight; :func:`additive_inflation` draws the noise.
    """
    z = np.asarray(z_obs, dtype=np.float64)
    mask = np.asarray(z_mask, dtype=bool)
    echo = np.where(mask, z, -np.inf).max(axis=0) >= float(threshold_dbz)
    if not echo.any():
        return np.zeros(echo.shape, dtype=np.float64)
    ny, nx = echo.shape
    sigma = float(length_scale_km)
    # Zero-padded to twice the grid so the smoothing does not wrap around.
    pad_y, pad_x = ny, nx
    field = np.zeros((ny + pad_y, nx + pad_x))
    field[:ny, :nx] = echo
    ky = np.fft.fftfreq(ny + pad_y, d=float(dy_km))
    kx = np.fft.fftfreq(nx + pad_x, d=float(dx_km))
    kernel = np.exp(-2.0 * (np.pi * sigma) ** 2
                    * (ky[:, None] ** 2 + kx[None, :] ** 2))
    smooth = np.fft.ifft2(np.fft.fft2(field) * kernel).real[:ny, :nx]
    smooth = np.clip(smooth / 0.5, 0.0, 1.0)
    smooth[echo] = 1.0
    return smooth


# --------------------------------------------------------------------------
# Documented stubs -- routes that exist on paper only
# --------------------------------------------------------------------------

def recycled_difference_perturbations(*args, **kwargs):
    """Not built in v1.  Perturbations recycled from forecast differences.

    The idea: take the difference between two forecasts valid at the same
    time (or one forecast and its own state some hours earlier), rescale it
    to the desired analysis-error magnitude, and use *that* as the
    perturbation.  Because the difference is a difference of two model
    trajectories it is already in the model's own balance -- no gravity-wave
    shock on the first step -- and it carries flow-dependent structure that a
    prescribed isotropic Gaussian cannot.

    What it needs that this module does not have: two archived states on the
    same grid, a rescaling target (a spread climatology or an observed
    innovation variance), and a policy for the boundary rows, since a
    recycled difference is nonzero everywhere including the rim.  Until all
    three exist, calling this raises rather than silently falling back to the
    Gaussian route.
    """
    raise NotImplementedError(
        "recycled-difference perturbations are a documented v1 non-goal; "
        "use apply_perturbations with a PerturbationConfig, or implement "
        "this route with its own provenance schema")
