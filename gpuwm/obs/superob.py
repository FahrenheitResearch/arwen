"""Gate-to-cell superobbing: many range gates in, one observation per cell out.

Two moments, two very different reductions, for physical reasons rather
than stylistic ones.

**Reflectivity** is averaged in *linear* Z and reported back in dBZ, because
dBZ is a logarithm and the mean of logarithms is not the logarithm of the
mean — averaging dBZ directly biases a cell containing one strong core and
several weak gates low by many dB.  The in-cell maximum is carried beside
the mean because convective assimilation usually wants the core, and a
consumer that has to reconstruct one from the other cannot.

**Radial velocity** is averaged per contributing radar and never across
radars: two radars looking at the same cell measure different projections of
the same wind, and their mean is a number with no observation operator.  The
beam unit vector is averaged with the velocities and renormalized, so every
retained velocity ships the look direction that turns it into an
observation.

**Clear air is an observation, and it is built from measurements only.**
A cell is reported clear when enough gates were *measured* inside it and
all of them came back below the significant-echo floor, and no radar saw
echo there.  The gates that support that claim are finite decoded values;
a missing gate never contributes.

Which gates can support that claim depends on what the pack it came from
was able to tell us, and there are two regimes.

**The measurement-only regime** (:data:`CLEAR_AIR_SOURCE`, the default).
A ``gpuwm-obs.radar-sweeps.v1`` pack carries one plane per moment, in which
every unusable gate is the same NaN.  Three quite different things produce
that NaN: raw gate code 0 (*below threshold* -- the radar looked and
detected nothing, which is precisely the clear-air observation), raw code 1
(*range folded* -- an ambiguous second-trip return that may be a storm),
and a radial that never carried the moment at all.  Downstream they are
indistinguishable, so NaN cannot be evidence of clear air without
fabricating observations wholesale: on a real KDMX volume that is about
9.7 million ambiguous gates against 1.1e4 unambiguous ones.  This regime
therefore uses only the unambiguous remainder -- gates that decoded to a
real number below ``min_reflectivity_dbz`` -- and accepts a thin product.

**The censored regime** (:data:`CLEAR_AIR_SOURCE_CENSOR`), available when
the pack is a ``gpuwm-obs.radar-sweeps.v2`` written by
``rw_nexrad decode --censor-flags``.  There the decoder's own reason for
each NaN rides beside it as a :class:`~gpuwm.obs.sweeps.Censor` code, so
"below threshold" is separable from "range folded" and from "never
collected", and the first of those becomes what it always was in the
signal: a measurement of nothing.  Pass ``clear_air_from_censor=True`` to
:func:`superob_volume` to use it.

**Range-folded gates are never clear air, in either regime.**  Raw code 1
means the radar cannot say which trip the return came from, and the answer
may be a storm.  It is admitted by no configuration of this module: the
censored regime tests for equality with one code
(:data:`~gpuwm.obs.sweeps.Censor.BELOW_THRESHOLD`) rather than for
"non-echo", and the measurement-only regime never sees a non-finite gate at
all.  ``clear_air_source`` in the written file records which regime
produced the zeroes, so a consumer always knows which coverage and which
error model it is holding.

**Aliasing is masked by default, and corrected on request.**  With
``SuperobParams.dealias`` left at its default ``None`` nothing in this
module has changed: the four fail-closed masks below are the whole of the
alias defense, no velocity is ever modified, and the observation files this
stage produces are byte-for-byte the ones it has always produced.  Setting
that field to a :class:`gpuwm.obs.dealias.DealiasParams` turns on
region-based unfolding -- see :mod:`gpuwm.obs.dealias` for the algorithm and
its abstention rule -- which runs per sweep *before* everything below, so
the masks then see velocities whose fold state is known rather than
suspected.  Two things change when it is on and nothing else does:

* a gate the unfolder could not resolve is dropped and counted, whatever its
  magnitude, because "I cannot tell" is not an observation;
* a gate the unfolder *did* resolve is no longer bounded by
  ``nyquist_reject_fraction`` of Nyquist but by an absolute physical speed,
  because that fraction exists to drop gates that might be folded and this
  one's fold state is known.  This is the entire recovery: at a Nyquist of
  25.51 m/s the 0.8 rule caps the assimilable wind at 20.4 m/s, and a
  mesocyclone's couplet lives above that.

**Aliasing is masked, not corrected** (the default path).  Four structural
defenses run, and it matters exactly what each of them can and cannot see:

1. a gate whose speed exceeds a configurable fraction of the sweep's Nyquist
   velocity is dropped and counted;
2. a sweep reporting no Nyquist velocity, or one outside the plausible band,
   has *every* velocity gate dropped and counted;
3. a cell whose retained gates disagree by more than a configurable fraction
   of Nyquist — a fold caught inside one cell — is dropped whole and counted;
4. a **gate-to-gate radial shear scan** along each radial, on the raw gates
   before any magnitude mask, flags range-adjacent pairs whose difference
   exceeds a fraction of the full Nyquist *interval* ``2 * Vn``.  A fold
   between neighbours produces a jump of very nearly ``2 * Vn``; the two
   gates flanking such a jump are dropped and counted, and per-sweep
   boundary counts go into provenance whether or not anything was dropped.

**What rule 4 can catch:** the *edge* of a folded region, where an unfolded
gate sits beside a folded one.  Both gates in such a pair are necessarily
near opposite Nyquist limits, which is the least trustworthy data in the
sweep, so the loss is narrow and lands where it should.

**What no rule here can catch:** a spatially coherent fold covering a whole
region.  At Nyquist 32 m/s a true +69 m/s folds to +5 m/s; a patch of gates
that all fold together has a present and plausible Nyquist, passes the
0.8 magnitude test, has zero in-cell spread, and has no gate-to-gate jump
anywhere in its interior.  It is a smooth, plausible, wrong wind field, and
it is assimilable.  Nothing short of true dealiasing — unwrapping the region
against a global reference rather than testing neighbours — excludes it.
The counters exist so that a region whose *boundary* was flagged is visible
in provenance even when its interior survived; they are evidence, not a
guarantee, and this module does not claim otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from numbers import Real

import numpy as np

from gpuwm.obs.dealias import (STATE_REJECTED, DealiasParams, DealiasParamsError,
                               dealias_sweep, volume_wind_profile)
from gpuwm.obs.geometry import (REFRACTION_FACTOR, gate_locations)
from gpuwm.obs.sweeps import Censor, RadarVolume
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.projection import EARTH_RADIUS_M

#: Moment tokens this stage understands.
REFLECTIVITY = "REF"
VELOCITY = "VEL"

#: How the clear-air observations in a product were established.  Written
#: into the observation file so a consumer can tell what it is trusting; a
#: consumer that does not recognise the value must refuse the file rather
#: than assume either one.
#:
#: The two regimes differ in *coverage*, not merely in count.  The
#: measurement-only zeroes sit where a gate decoded to a real number below
#: the floor, which on a quiet volume is a sparse scatter near the radar.
#: The censored zeroes additionally cover everywhere the radar reported
#: below-threshold, which is most of the clear sky it looked at.  Reading
#: one as the other misstates how much of the domain was observed, which is
#: exactly the error that turns a thin honest product into a confident
#: wrong one.
CLEAR_AIR_SOURCE = "finite_below_floor"
CLEAR_AIR_SOURCE_CENSOR = "below_threshold_and_finite_below_floor"

#: Every value :data:`CLEAR_AIR_SOURCE` may take.
CLEAR_AIR_SOURCES = (CLEAR_AIR_SOURCE, CLEAR_AIR_SOURCE_CENSOR)

#: Radials processed per geometry pass.  Peak memory, not correctness.
_RADIAL_BLOCK = 120


class SuperobParamsError(ValueError):
    """A superob parameter that cannot mean what the pass needs it to mean.

    Separate from :class:`ValueError` at the call site so a caller can tell
    "you configured this stage impossibly" from "this volume is malformed".
    Never a warning: every one of these values multiplies or bounds a
    physical threshold, and a wrong one produces observations that are
    finite, plausible and wrong.
    """


#: Fractions of a Nyquist velocity or Nyquist interval.  Outside ``[0, 1]``
#: they stop being fractions: a reject fraction above 1 admits speeds the
#: sweep cannot unambiguously measure, which is precisely the aliased data
#: the gate exists to drop, and a negative one is not an interval at all.
_FRACTION_FIELDS = (
    "nyquist_reject_fraction",
    "nyquist_spread_fraction",
    "shear_fold_fraction",
)

#: Values that bound or scale a physical quantity and are meaningless at or
#: below zero.  The four error fields are standard deviations: a zero sigma
#: is an infinitely confident observation, which in a filter is not a small
#: error but a hard constraint the analysis cannot argue with.
_POSITIVE_FIELDS = (
    "nyquist_min_ms",
    "nyquist_max_ms",
    "max_range_km",
    "z_error_base_dbz",
    "vr_error_base_ms",
    "z_error_floor_dbz",
    "vr_error_floor_ms",
    "refraction_factor",
    "earth_radius_m",
    "clear_air_min_gates",
    "clear_air_error_dbz",
)

#: Values that may take any sign but must be a number.  A NaN reflectivity
#: floor compares False against every gate, so it silently drops the entire
#: volume; an infinite one drops it loudly.  Neither is a floor.
_FINITE_FIELDS = ("min_reflectivity_dbz",)


def _fold_boundaries(values: np.ndarray, nyquist: float,
                     params) -> tuple[np.ndarray, int, int]:
    """Flag gates flanking a range-adjacent jump that only a fold explains.

    ``values`` is ``(radials, gates)`` of **raw** velocities, range-ordered
    along the second axis, before any magnitude mask.  Raw is the only place
    this test works: once the 0.8 gate has run, a folded gate and its
    unfolded neighbour are both inside the retained band and their
    difference is unremarkable.

    A NaN breaks the chain rather than bridging it — two gates either side
    of a data gap are not neighbours, and comparing them would invent a
    boundary out of a range hole.

    Returns ``(flags, boundaries, pairs_tested)``: the per-gate flags, how
    many adjacent pairs were flagged, and how many were comparable at all.
    The last is the denominator without which the middle number means
    nothing.
    """

    flags = np.zeros(values.shape, dtype=bool)
    if values.shape[1] < 2:
        return flags, 0, 0
    delta = np.abs(np.diff(values, axis=1))
    finite = np.isfinite(delta)
    boundary = finite & (delta > params.shear_fold_fraction * 2.0 * nyquist)
    flags[:, :-1] |= boundary
    flags[:, 1:] |= boundary
    return flags, int(boundary.sum()), int(finite.sum())


@dataclass
class _DealiasTotals:
    """Volume-wide three-state account, summed over sweeps.

    ``unchanged + unfolded + rejected`` equals the number of finite velocity
    gates the unfolder was offered, exactly, for every volume.  That identity
    is the whole contract: a gate cannot quietly fall out of the accounting,
    and a test asserts it rather than trusting it.
    """

    sweeps_dealiased: int = 0
    gates_offered: int = 0
    gates_unchanged: int = 0
    gates_unfolded: int = 0
    gates_rejected: int = 0
    #: The subset of rejections that reached the gridding stage -- a refused
    #: gate outside the grid or beyond range costs nothing, and reporting it
    #: as a loss would overstate the price of abstention.
    gates_refused_at_grid: int = 0
    regions: int = 0
    regions_anchored: int = 0
    regions_linked: int = 0
    regions_unresolved: int = 0
    regions_conflict: int = 0
    edges: int = 0
    edges_confident: int = 0
    edges_violated: int = 0
    reference_bands: int = 0
    reference_bands_valid: int = 0
    rejected: dict = field(default_factory=dict)
    fold_histogram: dict = field(default_factory=dict)

    def add(self, stats: dict) -> None:
        self.sweeps_dealiased += 1
        self.gates_offered += int(stats["gates_finite"])
        self.gates_unchanged += int(stats["gates_unchanged"])
        self.gates_unfolded += int(stats["gates_unfolded"])
        self.gates_rejected += int(stats["gates_rejected"])
        for name in ("regions", "regions_anchored", "regions_linked",
                     "regions_unresolved", "regions_conflict", "edges",
                     "edges_confident", "edges_violated"):
            setattr(self, name, getattr(self, name) + int(stats[name]))
        reference = stats.get("reference") or {}
        self.reference_bands += int(reference.get("bands", 0))
        self.reference_bands_valid += int(reference.get("bands_valid", 0))
        for reason, count in stats["rejected"].items():
            self.rejected[reason] = self.rejected.get(reason, 0) + int(count)
        for fold, count in stats["fold_histogram"].items():
            key = str(int(fold))
            self.fold_histogram[key] = self.fold_histogram.get(key, 0) + int(count)

    def to_payload(self) -> dict:
        payload = {key: value for key, value in asdict(self).items()
                   if not isinstance(value, dict)}
        payload["rejected"] = dict(sorted(self.rejected.items()))
        payload["fold_histogram"] = {
            key: self.fold_histogram[key]
            for key in sorted(self.fold_histogram, key=int)}
        payload["accounting_balances"] = bool(
            self.gates_unchanged + self.gates_unfolded + self.gates_rejected
            == self.gates_offered)
        return payload


def _dealias_velocity_sweep(sweep, nyquist, dealias_params, site,
                            velocity_reference, params) -> dict:
    """Unfold one sweep's velocity and package what the caller needs.

    The returned ``velocity`` plane carries the unfolded value where the
    unfolder resolved the gate and the **raw** value where it did not, so
    downstream finiteness bookkeeping is untouched; ``resolved`` is the mask
    that actually decides what may be assimilated.
    """

    moment = sweep.moments[VELOCITY]
    raw = np.asarray(moment.data, dtype=np.float64)
    reference = None
    if velocity_reference is not None:
        ranges = moment.slant_range_m()
        azimuth = np.broadcast_to(sweep.azimuth_deg[:, None], raw.shape)
        elevation = np.broadcast_to(sweep.elevation_deg[:, None], raw.shape)
        _lat, _lon, height, *_ = gate_locations(
            site.lat_deg, site.lon_deg, site.alt_m, azimuth,
            np.broadcast_to(ranges[None, :], raw.shape), elevation,
            earth_radius_m=params.earth_radius_m,
            refraction_factor=params.refraction_factor)
        reference = velocity_reference.radial_reference(
            azimuth, elevation, height - site.alt_m)

    result = dealias_sweep(raw, sweep.azimuth_deg, nyquist, dealias_params,
                           reference=reference)
    resolved = result.state != STATE_REJECTED
    velocity = np.where(resolved, result.velocity, raw)
    # ``band_fits`` is the raw harmonic coefficients the volume-profile pass
    # consumes -- hundreds of entries per sweep, some carrying a NaN residual
    # where a band was taken from the profile rather than fitted.  It is
    # working state, not provenance: it would bloat the attribute and the
    # JSON writer refuses NaN outright, which is how it announced itself.
    reference = {key: value
                 for key, value in (result.stats.get("reference") or {}).items()
                 if key != "band_fits"}
    record = {
        "sweep_index": int(sweep.sweep_index),
        "elevation_angle_deg": float(sweep.elevation_angle_deg),
        **{key: value for key, value in result.stats.items()
           if key not in ("fold_histogram", "reference")},
        "reference": reference,
        "fold_histogram": {str(int(k)): int(v)
                           for k, v in sorted(result.stats["fold_histogram"].items())},
    }
    return {"result": result, "velocity": velocity, "resolved": resolved,
            "record": record}


def _believable_nyquist(reported, params) -> float | None:
    """The sweep's Nyquist velocity, or None when it cannot be believed."""

    if reported is None:
        return None
    value = float(reported)
    if not np.isfinite(value):
        return None
    if not (params.nyquist_min_ms <= value <= params.nyquist_max_ms):
        return None
    return value


@dataclass(frozen=True)
class SuperobParams:
    """Every tunable, in one hashable place — it goes into provenance."""

    #: Drop a velocity gate whose speed exceeds this fraction of Nyquist.
    nyquist_reject_fraction: float = 0.8
    #: Physically plausible Nyquist velocities for an S-band weather radar,
    #: m/s.  A reported value outside this band is metadata this stage does
    #: not believe -- a mis-parsed field, a non-WSR-88D convention -- and
    #: an unbelieved Nyquist masks every velocity in its sweep rather than
    #: licensing a threshold nothing supports.  The WSR-88D range runs from
    #: about 8 m/s on the slowest surveillance PRF to about 35 m/s on the
    #: fastest Doppler cut; the band is wide enough to admit other radars
    #: and narrow enough to reject a reinterpreted calibration constant.
    nyquist_min_ms: float = 4.0
    nyquist_max_ms: float = 100.0
    #: Drop a cell whose retained velocities span more than this fraction
    #: of Nyquist (a fold inside one cell).
    nyquist_spread_fraction: float = 0.5
    #: Gate-to-gate radial shear: a range-adjacent pair whose velocities
    #: differ by more than this fraction of the **full Nyquist interval**
    #: ``2 * Vn`` is a fold boundary.  A single fold produces a jump of
    #: almost exactly ``2 * Vn``, so 0.75 sits well below that and well
    #: above what a pair can reach once both gates have survived the 0.8
    #: magnitude gate (at most ``1.6 * Vn``, and only for a pair straddling
    #: the Nyquist limits in opposite directions -- which is the fold
    #: signature again).  Raising it toward 1.0 flags only near-perfect
    #: wraps; lowering it starts costing real tornadic gate-to-gate shear,
    #: which at close range is the signal this whole lane exists to carry.
    shear_fold_fraction: float = 0.75
    #: Reflectivity floor, dBZ: gates below are "no echo", not observations.
    min_reflectivity_dbz: float = -15.0
    #: Ignore gates beyond this slant range, km.
    max_range_km: float = 250.0
    #: Ignore sweeps above this antenna elevation, degrees.
    max_elevation_deg: float = 20.0
    #: Base observation-error standard deviations.
    z_error_base_dbz: float = 5.0
    vr_error_base_ms: float = 2.0
    #: Error floors: a cell with many gates must not claim implausible skill.
    z_error_floor_dbz: float = 2.0
    vr_error_floor_ms: float = 1.0
    #: Effective-earth multiplier for beam propagation.
    refraction_factor: float = REFRACTION_FACTOR
    earth_radius_m: float = EARTH_RADIUS_M
    #: How many *finite* below-floor gates must land in a cell before it is
    #: reported as observed clear air.  Stored as a float because every
    #: field of this dataclass is (``__post_init__`` normalizes the lot);
    #: it is used as a ``>=`` threshold against an integer gate count.
    #:
    #: This is not a smoothing knob.  One below-floor gate in a cell that
    #: the beam otherwise clipped is a geometry accident; requiring several
    #: independent gates to agree is what makes "the radar looked here and
    #: measured no significant return" a statement about the cell rather
    #: than about one range bin at its corner.
    clear_air_min_gates: float = 4.0
    #: Observation-error standard deviation for a clear-air zero, dBZ.
    #:
    #: Deliberately NOT ``z_error_base_dbz``.  A zero is a different
    #: measurement with a different error budget: it carries no in-cell
    #: variance to estimate from, its representativeness error is dominated
    #: by partial beam filling (a cell the beam only clipped can be clear
    #: where the beam looked and stormy where it did not), and the
    #: consequence of believing it too hard is erasing real convection.
    #: WoFS-family systems assign clear-air reflectivity a markedly larger
    #: sigma_o than echo for exactly this reason, and this default follows
    #: that practice rather than inheriting the echo error by omission.
    clear_air_error_dbz: float = 7.5
    #: Region-based velocity dealiasing, or ``None`` for the masking-only
    #: behaviour this stage has always had.
    #:
    #: ``None`` is not merely the default, it is the *identity*: every other
    #: field here is a float, ``to_payload`` emits only floats while this is
    #: None, and the observation file's ``superob_params`` attribute and
    #: ``dealiasing`` statement are therefore unchanged to the byte.  A
    #: consumer reading a file written without dealiasing cannot tell that
    #: this field was ever added, which is the point: turning the capability
    #: on is a decision someone makes, and off is not a decision at all.
    dealias: DealiasParams | None = None

    def __post_init__(self) -> None:
        self.validate()
        # Normalize the runtime type once, at the only point where this
        # object is being built rather than merely used: every field is a
        # Python ``float`` from here on, so ``params.max_range_km * 1000.0``
        # and ``to_payload`` mean the same thing whether the caller passed
        # ``250``, ``np.float32(250)`` or ``250.0``.  ``validate`` above has
        # already refused anything that is not a real number, so this
        # normalizes what is sound and repairs nothing that is not.
        for field_ in fields(self):
            if field_.name == "dealias":
                continue
            object.__setattr__(self, field_.name,
                               float(getattr(self, field_.name)))

    def _check_runtime_types(self) -> None:
        """Refuse a field whose runtime type is not a real number.

        ``validate`` used to read every field as ``float(getattr(...))``
        and keep the original object, so
        ``SuperobParams(max_range_km="250", nyquist_reject_fraction=True)``
        constructed successfully: ``float("250")`` is 250.0 and
        ``float(True)`` is 1.0, both of which pass every range check, and
        the string then raised ``TypeError`` at the first arithmetic use
        (``params.max_range_km * 1000.0``).  Failing loudly at first use is
        better than producing observations, but the parameter validator
        that the pipeline calls at every entry point was not validating its
        own runtime schema -- and the Rust side refuses these outright, so
        the Python surface was accepting what its counterpart would not.

        ``bool`` is excluded explicitly because ``True`` is an ``int`` in
        Python: ``nyquist_reject_fraction=True`` is 1.0, an in-range
        fraction, and it means the caller passed a flag where a threshold
        belongs.
        """

        for field_ in fields(self):
            if field_.name == "dealias":
                continue
            value = getattr(self, field_.name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise SuperobParamsError(
                    f"{field_.name} is {value!r} ({type(value).__name__}); "
                    "every superob parameter is a real number. A numeric "
                    "string passes float() and then raises TypeError at the "
                    "first arithmetic use, and a bool passes every range "
                    "check as 0.0 or 1.0 -- neither is a value this stage "
                    "will convert on the caller's behalf, because the "
                    "conversion is the caller stating what they meant")
            try:
                float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise SuperobParamsError(
                    f"{field_.name} is {value!r} ({type(value).__name__}) "
                    "but cannot be represented as a Python float; every "
                    "superob parameter is stored and consumed as a float, "
                    "so this is not a usable real-number value") from exc

    def validate(self) -> "SuperobParams":
        """Refuse any value that cannot do the job the field is named for.

        Called from ``__post_init__`` *and* from every entry point that
        consumes a parameter set, because construction is not the only way
        one arrives.  ``dataclasses.replace`` re-runs ``__init__`` and is
        covered by the former; ``object.__setattr__`` past the ``frozen=True``
        guard, an instance rebuilt from a JSON payload by a future reader, or
        a subclass that overrides a default are not.  The values are read
        back off ``self`` here rather than trusted from construction time, so
        the check is against what the pass is about to use.

        Returns ``self`` so a caller can write ``params.validate()`` inline.
        """

        # Types before ranges: every range check below reads the field
        # through float(), which is exactly how a numeric string and a bool
        # got past this function.
        self._check_runtime_types()
        for name in _FRACTION_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise SuperobParamsError(
                    f"{name} is {value!r}; it is a fraction of a Nyquist "
                    "velocity or Nyquist interval and must lie in [0, 1]. "
                    "Above 1 the gate admits speeds the sweep cannot "
                    "unambiguously measure -- the aliased data it exists to "
                    "drop; below 0 it is not an interval")
        for name in _POSITIVE_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise SuperobParamsError(
                    f"{name} is {value!r}; it must be finite and strictly "
                    "positive. Zero or negative makes the quantity it bounds "
                    "or scales meaningless, and for the error fields a zero "
                    "standard deviation is an infinitely confident "
                    "observation rather than a small uncertainty")
        for name in _FINITE_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise SuperobParamsError(
                    f"{name} is {value!r}; it must be a finite number. A NaN "
                    "threshold compares False against every gate and drops "
                    "the volume in silence")
        if not self.nyquist_min_ms < self.nyquist_max_ms:
            raise SuperobParamsError(
                f"nyquist_min_ms {self.nyquist_min_ms!r} is not below "
                f"nyquist_max_ms {self.nyquist_max_ms!r}; the pair is the "
                "plausible band a reported Nyquist velocity is believed "
                "inside, and a band that is empty or a single point believes "
                "nothing, so every velocity in every sweep would be masked "
                "for an implausible Nyquist")
        min_gates = float(self.clear_air_min_gates)
        if min_gates < 1.0:
            raise SuperobParamsError(
                f"clear_air_min_gates is {min_gates!r}; it counts gates and "
                "must be at least 1. Below 1 every cell the beam never "
                "reached would satisfy the threshold with its zero count, "
                "turning 'no data' into 'observed clear' across the whole "
                "grid -- which is the one failure this product must never "
                "have")
        elevation = float(self.max_elevation_deg)
        if not np.isfinite(elevation) or not (0.0 < elevation <= 90.0):
            raise SuperobParamsError(
                f"max_elevation_deg is {elevation!r}; an antenna elevation "
                "ceiling must lie in (0, 90]. At or below 0 no sweep is ever "
                "used; above 90 the ceiling is not an elevation and cannot "
                "exclude anything")
        if self.dealias is not None:
            if not isinstance(self.dealias, DealiasParams):
                raise SuperobParamsError(
                    f"dealias is {self.dealias!r} "
                    f"({type(self.dealias).__name__}); it is either None -- "
                    "masking only, this stage's original behaviour -- or a "
                    "DealiasParams. A dict or a bool here would be a caller "
                    "asking for dealiasing without saying how it should "
                    "abstain, and the abstention rules are the safety")
            try:
                self.dealias.validate()
            except DealiasParamsError as error:
                raise SuperobParamsError(
                    f"dealias parameters are unusable: {error}") from error
        return self

    def to_payload(self) -> dict:
        payload = {key: float(value) for key, value in asdict(self).items()
                   if key != "dealias"}
        # Present only when dealiasing ran.  A file written without it
        # carries the same key set it always has, which is what makes the
        # disabled path byte-identical rather than merely equivalent.
        if self.dealias is not None:
            payload["dealias"] = self.dealias.to_payload()
        return payload


@dataclass
class CensorCounts:
    """The gate census a ``v2`` pack's censor planes make possible.

    Only reflectivity is broken out, because reflectivity is the only
    moment this stage draws clear air from.  Velocity's censor plane is read
    and checked but never consulted for an observation, so counting it here
    would suggest an influence it does not have.
    """

    #: Reflectivity gates the decoder called measured, below threshold,
    #: range folded, and never collected, before any geometry filter.
    reflectivity_measured: int = 0
    reflectivity_below_threshold: int = 0
    reflectivity_range_folded: int = 0
    reflectivity_not_collected: int = 0
    #: Below-threshold gates that survived every geometry filter and were
    #: counted toward a cell's clear-air support.
    clear_air_gates_admitted: int = 0
    #: Range-folded gates seen anywhere in the pass.  Always equal to
    #: ``reflectivity_range_folded`` plus the velocity plane's, and always
    #: entirely refused: this number exists so the refusal is visible in
    #: provenance rather than merely asserted in a docstring.
    range_folded_gates_refused: int = 0


@dataclass
class SuperobCounts:
    """What the pass did and, more importantly, what it refused."""

    gates_considered: int = 0
    gates_out_of_grid: int = 0
    gates_out_of_column: int = 0
    gates_below_floor: int = 0
    gates_nonfinite: int = 0
    gates_beyond_range: int = 0
    sweeps_used: int = 0
    sweeps_skipped_elevation: int = 0
    velocity_gates_rejected_nyquist: int = 0
    velocity_gates_rejected_no_nyquist: int = 0
    velocity_cells_rejected_spread: int = 0
    sweeps_without_nyquist: int = 0
    sweeps_with_implausible_nyquist: int = 0
    sweeps_with_nyquist_disagreement: int = 0
    #: Gate-to-gate shear scan.  ``pairs_tested`` is the denominator these
    #: only mean anything against: five boundaries in a million pairs and
    #: five in fifty are very different volumes.
    velocity_gate_pairs_tested: int = 0
    velocity_fold_boundaries: int = 0
    velocity_radials_fold_suspect: int = 0
    velocity_sweeps_fold_suspect: int = 0
    velocity_gates_rejected_shear: int = 0
    #: The censor census, or ``None`` when the pack carried no censor
    #: planes.  ``None`` rather than an all-zero record because the two are
    #: different statements, and because it is what keeps
    #: :meth:`to_payload` -- and therefore every observation file's
    #: ``provenance`` attribute, and therefore every committed obs digest --
    #: byte-identical to what it was before this field existed.
    censor: CensorCounts | None = None

    def to_payload(self) -> dict:
        payload = {key: int(value) for key, value in asdict(self).items()
                   if key != "censor"}
        if self.censor is not None:
            payload["censor"] = {key: int(value) for key, value
                                 in asdict(self.censor).items()}
        return payload


@dataclass
class RadarContribution:
    """One radar's gridded contribution, before the multi-radar merge."""

    site_id: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    valid_time: str
    z_linear_sum: np.ndarray
    z_count: np.ndarray
    #: Per cell, the number of gates that were *measured* and came back
    #: below ``min_reflectivity_dbz``.  See :func:`superob_volume` for the
    #: four conditions a gate must already have satisfied to be counted.
    z0_count: np.ndarray
    z_max_dbz: np.ndarray
    z_sumsq_dbz: np.ndarray
    z_sum_dbz: np.ndarray
    vr_sum: np.ndarray
    vr_sumsq: np.ndarray
    vr_count: np.ndarray
    vr_min: np.ndarray
    vr_max: np.ndarray
    beam_east: np.ndarray
    beam_north: np.ndarray
    beam_up: np.ndarray
    nyquist_min: np.ndarray
    vr_rejected: np.ndarray
    counts: SuperobCounts = field(default_factory=SuperobCounts)
    provenance: dict = field(default_factory=dict)
    #: Which of :data:`CLEAR_AIR_SOURCES` produced ``z0_count``.  Carried
    #: per radar because the merge has to refuse a mixture: two radars whose
    #: zeroes mean different things cannot be summed into one count.
    clear_air_source: str = CLEAR_AIR_SOURCE
    #: One record per velocity-carrying sweep, whether or not anything was
    #: flagged: an absence of fold boundaries is evidence too, and only
    #: means something beside the number of pairs that were tested.
    fold_suspicion: list = field(default_factory=list)
    #: The dealiasing account -- three states, every rejection with a reason
    #: -- or empty when dealiasing did not run.  Empty is how a consumer
    #: tells "no folds were found" from "nobody looked", which the counters
    #: alone cannot say.
    dealias: dict = field(default_factory=dict)


def superob_volume(volume: RadarVolume, grid: TargetGrid, *,
                   params: SuperobParams | None = None,
                   clear_air_from_censor: bool = False,
                   velocity_reference=None) -> RadarContribution:
    """Grid one radar volume onto ``grid``.

    Accumulators only — the dBZ/velocity/error reduction happens once, in
    :func:`merge_contributions`, so a multi-radar product and a single-radar
    product go through exactly the same arithmetic.

    ``clear_air_from_censor`` selects the censored regime described in the
    module docstring.  It is off by default and the default path is
    unchanged, arithmetic included.  Asking for it against a pack that has
    no censor planes is a hard error rather than a silent downgrade: the
    caller asked for a coverage this volume cannot supply, and quietly
    returning the thin product under the wrong ``clear_air_source`` is the
    one outcome that would mislead the DA side.

    It is a keyword rather than a :class:`SuperobParams` field on purpose.
    ``SuperobParams.to_payload`` is serialized verbatim into every
    observation file's ``superob_params`` attribute, so a new field there
    would change the bytes of files that are otherwise identical -- and the
    regime is already recorded, exactly once and where a consumer looks for
    it, as ``clear_air_source``.

    ``velocity_reference`` is an optional
    :class:`gpuwm.obs.dealias.WindProfile` -- the model background wind --
    used only when ``params.dealias`` is set, and used only to *supplement*
    the volume's own VAD: it seeds the harmonic fit and fills the range
    bands the fit could not qualify.  It is never allowed to override a
    band the volume itself resolved, because the volume measured the wind
    and the background guessed it.
    """

    params = (params or SuperobParams()).validate()
    censor_counts: CensorCounts | None = None
    if clear_air_from_censor:
        missing = [
            f"sweep {sweep.sweep_index} {name}"
            for sweep in volume.sweeps
            for name, moment in sweep.moments.items()
            if name == REFLECTIVITY and moment.censor is None
        ]
        if missing:
            raise ValueError(
                "clear_air_from_censor needs a pack whose reflectivity "
                "carries censor planes, and "
                f"{volume.pack_path.name} (schema {volume.pack_schema}) does "
                f"not: {missing[0]}"
                + (f" and {len(missing) - 1} more" if len(missing) > 1 else "")
                + ". Re-decode the volume with `rw_nexrad decode "
                "--censor-flags`. Falling back to the measurement-only "
                "regime here would publish a thin product under the "
                "censored regime's clear_air_source, which claims a "
                "coverage it does not have")
        censor_counts = CensorCounts()

    dealias_params = params.dealias
    if dealias_params is not None and velocity_reference is None:
        # Derive the anchor from the volume itself before unfolding any of
        # it.  A per-sweep fit is one range band's view of one height; the
        # volume crossed most heights several times, from different
        # elevations at different ranges, and pooling those is the only
        # cross-check available without an external model field.  Measured
        # on the real case this is what stops a sparse far-range band from
        # anchoring thousands of gates to a wind no other sweep saw.
        velocity_reference = volume_wind_profile(
            ((sweep.elevation_angle_deg,
              sweep.moments[VELOCITY].data,
              sweep.azimuth_deg,
              _believable_nyquist(sweep.nyquist_velocity_ms, params),
              sweep.moments[VELOCITY].slant_range_m())
             for sweep in volume.sweeps
             if VELOCITY in sweep.moments
             and sweep.elevation_angle_deg <= params.max_elevation_deg),
            dealias_params)
    shape = (grid.nz, grid.ny, grid.nx)
    zeros = lambda: np.zeros(shape, dtype=np.float64)     # noqa: E731
    counts = SuperobCounts()

    z_linear_sum = zeros()
    z_sum_dbz = zeros()
    z_sumsq_dbz = zeros()
    z_count = np.zeros(shape, dtype=np.int64)
    z0_count = np.zeros(shape, dtype=np.int64)
    z_max_dbz = np.full(shape, -np.inf, dtype=np.float64)
    vr_sum = zeros()
    vr_sumsq = zeros()
    vr_count = np.zeros(shape, dtype=np.int64)
    vr_min = np.full(shape, np.inf, dtype=np.float64)
    vr_max = np.full(shape, -np.inf, dtype=np.float64)
    beam_east = zeros()
    beam_north = zeros()
    beam_up = zeros()
    nyquist_min = np.full(shape, np.inf, dtype=np.float64)
    vr_rejected = np.zeros(shape, dtype=np.int64)

    fold_suspicion: list[dict] = []
    counts.censor = censor_counts
    dealias_records: list[dict] = []
    dealias_totals = _DealiasTotals()

    site = volume.site
    max_range_m = params.max_range_km * 1000.0
    clear_air_source = (CLEAR_AIR_SOURCE_CENSOR if clear_air_from_censor
                        else CLEAR_AIR_SOURCE)

    for sweep in volume.sweeps:
        if sweep.elevation_angle_deg > params.max_elevation_deg:
            counts.sweeps_skipped_elevation += 1
            continue
        nyquist = _believable_nyquist(sweep.nyquist_velocity_ms, params)

        # Dealiasing runs once per sweep, on the whole cut, before any
        # blocking or range masking.  Regions span radial blocks and range
        # limits; unfolding a slice at a time would cut every region at the
        # block seam and turn continuity -- the evidence the method rests on
        # -- into an artifact of a memory-management constant.
        dealiased = None
        if dealias_params is not None and VELOCITY in sweep.moments:
            dealiased = _dealias_velocity_sweep(
                sweep, nyquist, dealias_params, site,
                velocity_reference, params)
            dealias_records.append(dealiased["record"])
            dealias_totals.add(dealiased["result"].stats)
        if sweep.nyquist_velocity_ms is None:
            counts.sweeps_without_nyquist += 1
        elif nyquist is None:
            counts.sweeps_with_implausible_nyquist += 1
        if sweep.nyquist_radials_disagree:
            counts.sweeps_with_nyquist_disagreement += 1
        counts.sweeps_used += 1
        sweep_pairs = 0
        sweep_boundaries = 0
        sweep_suspect_radials = 0
        sweep_shear_rejected = 0

        for product, moment in sweep.moments.items():
            if product not in (REFLECTIVITY, VELOCITY):
                continue
            ranges = moment.slant_range_m()                     # (gates,)
            in_range = ranges <= max_range_m
            counts.gates_beyond_range += int(
                (~in_range).sum()) * sweep.radial_count
            if not np.any(in_range):
                continue
            ranges = ranges[in_range]

            # Radial blocks bound peak memory: a super-res sweep is 720 x
            # 1832 gates and the geometry pass holds a dozen float64
            # temporaries per gate, which is gigabytes if done in one go.
            for start in range(0, sweep.radial_count, _RADIAL_BLOCK):
                stop = min(start + _RADIAL_BLOCK, sweep.radial_count)
                values = np.asarray(moment.data[start:stop][:, in_range],
                                    dtype=np.float64)
                # The decoder's reason for each NaN, when the pack carried
                # one.  Read for both moments so the range-folded refusal
                # can be counted honestly, consulted for clear air only on
                # reflectivity.
                codes = (None if moment.censor is None or censor_counts is None
                         else moment.censor[start:stop][:, in_range])
                clear_flag = np.zeros(values.shape, dtype=bool)
                if codes is not None:
                    folded = codes == Censor.RANGE_FOLDED
                    censor_counts.range_folded_gates_refused += int(
                        folded.sum())
                    if product == REFLECTIVITY:
                        # Equality with ONE code, never "not an echo".  This
                        # is the line that keeps range-folded gates out of
                        # the clear-air path: code 2 is not code 1, and no
                        # setting of any parameter makes it so.
                        clear_flag = codes == Censor.BELOW_THRESHOLD
                        censor_counts.reflectivity_measured += int(
                            (codes == Censor.MEASURED).sum())
                        censor_counts.reflectivity_below_threshold += int(
                            clear_flag.sum())
                        censor_counts.reflectivity_range_folded += int(
                            folded.sum())
                        censor_counts.reflectivity_not_collected += int(
                            (codes == Censor.NOT_COLLECTED).sum())
                # Where dealiasing ran, the velocities from here down are the
                # unfolded ones -- including the shear scan, which now sees a
                # field whose folds have been removed and so measures what
                # the unfolder missed rather than what it was asked to fix.
                # A gate the unfolder rejected keeps its RAW value and is
                # excluded by `resolved` instead, so `finite` and
                # gates_nonfinite still count exactly what they always
                # counted: missing data, not refused data.
                resolved = None
                if dealiased is not None and product == VELOCITY:
                    values = dealiased["velocity"][start:stop][:, in_range]
                    resolved = dealiased["resolved"][start:stop][:, in_range]
                # The shear scan runs here, on raw range-ordered gates,
                # because this is the last point at which a fold and its
                # unfolded neighbour still differ by the Nyquist interval.
                if product == VELOCITY and nyquist is not None:
                    fold_flags, boundaries, pairs = _fold_boundaries(
                        values, nyquist, params)
                    sweep_pairs += pairs
                    sweep_boundaries += boundaries
                    sweep_suspect_radials += int(
                        fold_flags.any(axis=1).sum())
                else:
                    fold_flags = np.zeros(values.shape, dtype=bool)
                azimuth = sweep.azimuth_deg[start:stop, None]
                elevation = sweep.elevation_deg[start:stop, None]
                lat, lon, height, east, north, up = gate_locations(
                    site.lat_deg, site.lon_deg, site.alt_m,
                    np.broadcast_to(azimuth, values.shape),
                    np.broadcast_to(ranges[None, :], values.shape),
                    np.broadcast_to(elevation, values.shape),
                    earth_radius_m=params.earth_radius_m,
                    refraction_factor=params.refraction_factor)

                finite = np.isfinite(values).ravel()
                counts.gates_considered += int(values.size)
                counts.gates_nonfinite += int((~finite).sum())

                # A gate is worth placing if it is a measurement OR if the
                # decoder said the radar looked here and found nothing.  In
                # the measurement-only regime ``clear_flag`` is all false
                # and this is exactly ``finite``, which is why every count
                # below is unchanged there.
                usable = finite | clear_flag.ravel()

                i_frac, j_frac = grid.mass_index(lat.ravel(), lon.ravel())
                i_index = np.rint(i_frac).astype(np.intp)
                j_index = np.rint(j_frac).astype(np.intp)
                on_grid = grid.inside(i_index, j_index) & usable
                counts.gates_out_of_grid += int(usable.sum() - on_grid.sum())
                if not np.any(on_grid):
                    continue

                clear_on_grid = clear_flag.ravel()[on_grid]
                i_index = i_index[on_grid]
                j_index = j_index[on_grid]
                level = grid.level_index(i_index, j_index,
                                         height.ravel()[on_grid])
                in_column = level >= 0
                counts.gates_out_of_column += int((~in_column).sum())
                if not np.any(in_column):
                    continue

                flat = np.ravel_multi_index(
                    (level[in_column], j_index[in_column],
                     i_index[in_column]), shape)
                value_sel = values.ravel()[on_grid][in_column]
                clear_sel = clear_on_grid[in_column]

                if product == REFLECTIVITY:
                    # A below-threshold gate is NaN, and NaN >= x is False,
                    # so it lands in ``~echo`` without a special case --
                    # which is the point: it is below the floor by the
                    # radar's own report rather than by our arithmetic.
                    echo = value_sel >= params.min_reflectivity_dbz
                    counts.gates_below_floor += int((~echo & ~clear_sel).sum())
                    if censor_counts is not None:
                        censor_counts.clear_air_gates_admitted += int(
                            clear_sel.sum())
                    # --- clear air: the radar looked here and measured
                    # nothing significant ---
                    #
                    # A gate reaches this line only after four independent
                    # conditions, and every one of them is load-bearing for
                    # the claim "observed clear" rather than "no data":
                    #
                    # 1. ACCOUNTED FOR.  ``on_grid`` above ANDs in
                    #    ``usable``, which is ``finite`` plus -- only in the
                    #    censored regime -- gates the decoder explicitly
                    #    marked below threshold.  A gate that is NaN for any
                    #    OTHER reason never arrives: not a range-folded
                    #    gate (raw 1, which may be a storm), not a gate on a
                    #    radial that never carried the moment.  In the
                    #    measurement-only regime the set is just ``finite``,
                    #    because there NaN is irreducibly ambiguous and is
                    #    never evidence of anything.
                    # 2. WITHIN RANGE.  ``in_range`` trimmed the far end, so
                    #    a cell past ``max_range_km`` accumulates nothing.
                    # 3. ON GRID and IN COLUMN.  The gate was placed in a
                    #    real model cell by the same geometry the echo
                    #    observations use; a cell no beam traverses -- below
                    #    the lowest tilt, behind terrain, outside the scan --
                    #    is never named here at all, and so ends the pass
                    #    with a zero count rather than a clear-air claim.
                    # 4. BELOW THE FLOOR.  The measured value is a real
                    #    number that is smaller than the significant-echo
                    #    threshold.
                    #
                    # Note what is *not* asserted: this counts gates, not
                    # cells.  Whether the cell as a whole is clear is decided
                    # in ``merge_contributions``, where the count meets the
                    # echo count and the minimum-gate threshold, because a
                    # cell containing one clear gate and one echo gate is a
                    # cell with echo in it.
                    flat_clear = flat[~echo]
                    if flat_clear.size:
                        np.add.at(z0_count.reshape(-1), flat_clear, 1)
                    flat_z = flat[echo]
                    dbz = value_sel[echo]
                    if flat_z.size:
                        np.add.at(z_linear_sum.reshape(-1), flat_z,
                                  np.power(10.0, dbz / 10.0))
                        np.add.at(z_sum_dbz.reshape(-1), flat_z, dbz)
                        np.add.at(z_sumsq_dbz.reshape(-1), flat_z, dbz * dbz)
                        np.add.at(z_count.reshape(-1), flat_z, 1)
                        np.maximum.at(z_max_dbz.reshape(-1), flat_z, dbz)
                    continue

                # --- radial velocity: fail closed on aliasing signatures ---
                if nyquist is None:
                    counts.velocity_gates_rejected_no_nyquist += int(flat.size)
                    np.add.at(vr_rejected.reshape(-1), flat, 1)
                    continue
                # A gate whose fold state the unfolder could not establish is
                # dropped here whatever its magnitude.  It is dropped BEFORE
                # the magnitude test rather than after so the two losses can
                # never be confused in the counters: one is "too fast to
                # trust", the other is "I could not tell", and conflating
                # them would hide exactly the number this capability has to
                # be judged on.
                # `vr_rejected` is incremented once, below, off `~keep` --
                # which already excludes these, since `within` is masked by
                # `keep_resolved`.  Adding them here too would count a
                # refused gate twice in the array a consumer reads as "how
                # many gates were dropped over this cell".
                if resolved is not None:
                    keep_resolved = resolved.ravel()[on_grid][in_column]
                    dealias_totals.gates_refused_at_grid += int(
                        (~keep_resolved).sum())
                else:
                    keep_resolved = np.ones(flat.size, dtype=bool)
                if (resolved is not None
                        and dealias_params.keep_beyond_reject_fraction):
                    # The 0.8 rule drops gates that MIGHT be folded.  These
                    # were resolved, so what remains to bound them is
                    # physics, not the Nyquist interval -- and this is where
                    # the couplet that the 0.8 rule removes comes back.
                    within = np.abs(value_sel) <= dealias_params.max_speed_ms
                else:
                    within = (np.abs(value_sel)
                              <= params.nyquist_reject_fraction * nyquist)
                within = within & keep_resolved
                counts.velocity_gates_rejected_nyquist += int(
                    (keep_resolved & ~within).sum())
                # A gate flanking a fold boundary is dropped even when its
                # own magnitude is unremarkable: that is the whole point of
                # the scan, since a folded gate's magnitude is by definition
                # small.  Counted separately so the two losses never blur.
                flanking = fold_flags.ravel()[on_grid][in_column]
                shear_only = within & flanking
                counts.velocity_gates_rejected_shear += int(shear_only.sum())
                sweep_shear_rejected += int(shear_only.sum())
                keep = within & ~flanking
                if np.any(~keep):
                    np.add.at(vr_rejected.reshape(-1), flat[~keep], 1)
                if not np.any(keep):
                    continue
                flat_v = flat[keep]
                speed = value_sel[keep]
                east_sel = east.ravel()[on_grid][in_column][keep]
                north_sel = north.ravel()[on_grid][in_column][keep]
                up_sel = up.ravel()[on_grid][in_column][keep]
                np.add.at(vr_sum.reshape(-1), flat_v, speed)
                np.add.at(vr_sumsq.reshape(-1), flat_v, speed * speed)
                np.add.at(vr_count.reshape(-1), flat_v, 1)
                np.minimum.at(vr_min.reshape(-1), flat_v, speed)
                np.maximum.at(vr_max.reshape(-1), flat_v, speed)
                np.add.at(beam_east.reshape(-1), flat_v, east_sel)
                np.add.at(beam_north.reshape(-1), flat_v, north_sel)
                np.add.at(beam_up.reshape(-1), flat_v, up_sel)
                np.minimum.at(nyquist_min.reshape(-1), flat_v, float(nyquist))

        if VELOCITY in sweep.moments:
            counts.velocity_gate_pairs_tested += sweep_pairs
            counts.velocity_fold_boundaries += sweep_boundaries
            counts.velocity_radials_fold_suspect += sweep_suspect_radials
            if sweep_boundaries:
                counts.velocity_sweeps_fold_suspect += 1
            fold_suspicion.append({
                "sweep_index": int(sweep.sweep_index),
                "elevation_angle_deg": float(sweep.elevation_angle_deg),
                "nyquist_ms": None if nyquist is None else float(nyquist),
                "nyquist_radials_disagree": bool(
                    sweep.nyquist_radials_disagree),
                "radial_count": int(sweep.radial_count),
                "gate_pairs_tested": sweep_pairs,
                "fold_boundaries": sweep_boundaries,
                "radials_fold_suspect": sweep_suspect_radials,
                "gates_rejected_shear": sweep_shear_rejected,
            })

    # A cell whose retained velocities span too much of the Nyquist interval
    # is a fold caught inside one cell: drop it whole rather than average
    # across the wrap.
    spread = np.where(vr_count > 0, vr_max - vr_min, 0.0)
    folded = ((vr_count > 1)
              & np.isfinite(nyquist_min)
              & (spread > params.nyquist_spread_fraction * nyquist_min))
    counts.velocity_cells_rejected_spread = int(folded.sum())
    if np.any(folded):
        vr_rejected[folded] += vr_count[folded]
        for array in (vr_sum, vr_sumsq, beam_east, beam_north, beam_up):
            array[folded] = 0.0
        vr_count[folded] = 0

    return RadarContribution(
        site_id=site.id, lat_deg=site.lat_deg, lon_deg=site.lon_deg,
        alt_m=site.alt_m, valid_time=volume.valid_time,
        z_linear_sum=z_linear_sum, z_count=z_count, z0_count=z0_count,
        z_max_dbz=z_max_dbz,
        z_sumsq_dbz=z_sumsq_dbz, z_sum_dbz=z_sum_dbz,
        vr_sum=vr_sum, vr_sumsq=vr_sumsq, vr_count=vr_count,
        vr_min=vr_min, vr_max=vr_max,
        beam_east=beam_east, beam_north=beam_north, beam_up=beam_up,
        nyquist_min=nyquist_min, vr_rejected=vr_rejected,
        counts=counts, provenance=volume.provenance(),
        clear_air_source=clear_air_source,
        fold_suspicion=fold_suspicion,
        dealias=({} if dealias_params is None else {
            "params": dealias_params.to_payload(),
            "totals": dealias_totals.to_payload(),
            "sweeps": dealias_records,
        }))


@dataclass
class GriddedObservations:
    """The reduced fields, one step short of NetCDF."""

    z_obs: np.ndarray
    z_mask: np.ndarray
    z_err: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    z_count: np.ndarray
    vr_obs: np.ndarray
    vr_mask: np.ndarray
    vr_err: np.ndarray
    vr_count: np.ndarray
    vr_rejected: np.ndarray
    vr_beam_east: np.ndarray
    vr_beam_north: np.ndarray
    vr_beam_up: np.ndarray
    radars: list[dict]
    counts: list[dict]
    provenance: list[dict]
    #: Per-radar, per-sweep gate-to-gate shear scan records.  Defaulted so
    #: a caller assembling this structure by hand -- a test, a downstream
    #: lane -- is not forced to invent fold statistics it never measured.
    fold_suspicion: list = field(default_factory=list)
    #: Clear-air ("zero") observations, or None for a product that carries
    #: none.  ``z0_mask`` is 1 where at least one radar measured this cell
    #: and every radar that measured it found no significant echo;
    #: ``z0_count`` is the supporting measured-gate count and ``z0_err``
    #: the standard deviation to assimilate it with.
    #:
    #: ``None`` rather than an all-false mask is the default because the
    #: two are different statements.  An all-false mask asserts "this
    #: volume was examined for clear air and none was established"; None
    #: says "clear air was never assessed here".  A caller assembling this
    #: structure by hand has done the latter, and the writer omits the
    #: variables entirely rather than shipping a mask that claims an
    #: assessment nobody made.
    #:
    #: There is deliberately **no** ``z0_obs``: the dBZ value a zero
    #: differences against is the *forward operator's* clear-air floor
    #: (-35 dBZ for mp_physics 1/6/8/10, 0 dBZ for NSSL mp18), which is a
    #: property of the model the DA lane is running, not of the radar.
    #: Writing a value here would bake one scheme's floor into an
    #: observation file that outlives the run that consumed it, and a
    #: floor mismatch manufactures a 35 dB innovation out of two agreeing
    #: clear skies.  The DA adapter supplies the value.
    z0_mask: np.ndarray | None = None
    z0_count: np.ndarray | None = None
    z0_err: np.ndarray | None = None
    #: Which of :data:`CLEAR_AIR_SOURCES` established the zeroes above.
    #: Meaningless when ``z0_mask`` is None, and defaulted to the
    #: measurement-only regime so a hand-assembled product reads as the
    #: conservative one.
    clear_air_source: str = CLEAR_AIR_SOURCE
    #: Per-radar dealiasing account -- params, three-state totals, per-sweep
    #: records -- empty when dealiasing did not run.  Empty is meaningful:
    #: it is how a consumer tells "no folds were found" from "nobody
    #: looked", which no counter can say on its own.
    dealias: list = field(default_factory=list)


def merge_contributions(contributions, grid: TargetGrid, *,
                        params: SuperobParams | None = None,
                        z_reduce: str = "max") -> GriddedObservations:
    """Reduce one or more radars' accumulators into the output fields.

    Reflectivity merges across radars (the maximum of the maxima, the
    count-weighted mean of the linear sums); radial velocity does not, and
    keeps a leading ``radar`` axis.
    """

    params = (params or SuperobParams()).validate()
    contributions = list(contributions)
    if not contributions:
        raise ValueError("no radar contributions to merge")
    if z_reduce not in ("max", "mean"):
        raise ValueError(
            f"z_reduce must be 'max' or 'mean', got {z_reduce!r}")
    shape = (grid.nz, grid.ny, grid.nx)
    for contribution in contributions:
        if contribution.z_count.shape != shape:
            raise ValueError(
                f"contribution from {contribution.site_id} has shape "
                f"{contribution.z_count.shape}, grid is {shape}")

    # Zeroes from the two regimes cover different fractions of the domain,
    # so summing them would produce a count whose coverage nobody can state
    # and a single ``clear_air_source`` that would be a lie about half its
    # inputs.  Refuse rather than pick one.
    sources = {c.clear_air_source for c in contributions}
    if len(sources) > 1:
        raise ValueError(
            "the contributions establish clear air by different means "
            f"({sorted(sources)}), and their z0_counts cannot be summed: the "
            "two regimes cover different fractions of the domain, so the "
            "merged count would describe no coverage in particular. Rebuild "
            "every volume in the set the same way")
    clear_air_source = sources.pop()

    z_linear = sum(c.z_linear_sum for c in contributions)
    z_count = sum(c.z_count for c in contributions)
    z0_count = sum(c.z0_count for c in contributions)
    z_sum_dbz = sum(c.z_sum_dbz for c in contributions)
    z_sumsq_dbz = sum(c.z_sumsq_dbz for c in contributions)
    z_max = np.maximum.reduce([c.z_max_dbz for c in contributions])

    has_z = z_count > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z_mean = np.where(has_z,
                          10.0 * np.log10(np.maximum(z_linear, 1e-30)
                                          / np.maximum(z_count, 1)),
                          0.0)
        mean_dbz = np.where(has_z, z_sum_dbz / np.maximum(z_count, 1), 0.0)
        variance = np.where(
            z_count > 1,
            np.maximum(z_sumsq_dbz / np.maximum(z_count, 1)
                       - mean_dbz * mean_dbz, 0.0),
            0.0)
    z_max = np.where(has_z, z_max, 0.0)
    z_obs = np.where(has_z, z_max if z_reduce == "max" else z_mean, 0.0)
    z_err = np.where(
        has_z,
        np.maximum(
            params.z_error_floor_dbz,
            np.sqrt(params.z_error_base_dbz ** 2 / np.maximum(z_count, 1)
                    + variance)),
        0.0)
    z_mask = has_z.astype(np.int8)

    # --- clear-air zeroes -------------------------------------------------
    #
    # Two conditions, and the second is the one that keeps this honest.
    #
    # ``z0_count >= clear_air_min_gates`` says enough gates independently
    # measured this cell and found nothing.  ``~has_z`` says *no* radar saw
    # echo here.  The second is a veto, not a tiebreak: reflectivity is
    # summed across radars above, so ``has_z`` is true if ANY contributing
    # radar found echo in the cell, and a cell one radar calls clear while
    # another sees a storm in it is not clear.  The nearer radar is usually
    # the one seeing the storm -- the other is looking through it, over it,
    # or at a range where its beam has broadened past the cell -- so
    # deferring to the echo is also the physically right call, not merely
    # the conservative one.
    #
    # What this cannot see, and what therefore belongs in ``z0_err`` rather
    # than in this mask:
    #
    # * ATTENUATION.  A cell behind a heavy core can measure genuinely
    #   below-floor because the signal never got back, not because the sky
    #   is empty.  No attenuation correction exists in this lane
    #   (``gpuwm.da.obsop`` states the same for the forward operator), so
    #   this is real, unmodelled, and one-sided towards false clear air.
    # * PARTIAL BEAM FILLING.  At range the sampling volume is much larger
    #   than a model cell; "clear where the beam looked" and "clear
    #   throughout the cell" diverge with distance.
    # * BEAM BLOCKAGE.  A blocked ray returns clutter (counted as echo, so
    #   harmless here) or nothing (NaN, so never counted at all) -- but a
    #   *partially* blocked ray returns a weakened real echo that can fall
    #   below the floor.
    #
    # None of the three can be detected from the gridded product alone.
    # They are the reason ``clear_air_error_dbz`` is a separate, larger
    # sigma_o rather than an inherited one.
    has_z0 = (z0_count >= params.clear_air_min_gates) & ~has_z
    z0_mask = has_z0.astype(np.int8)
    z0_err = np.where(has_z0, params.clear_air_error_dbz, 0.0)

    n_radar = len(contributions)
    vr_obs = np.zeros((n_radar,) + shape, dtype=np.float64)
    vr_err = np.zeros_like(vr_obs)
    vr_mask = np.zeros((n_radar,) + shape, dtype=np.int8)
    vr_count = np.zeros((n_radar,) + shape, dtype=np.int32)
    vr_rejected = np.zeros((n_radar,) + shape, dtype=np.int32)
    beam = [np.zeros((n_radar,) + shape, dtype=np.float64) for _ in range(3)]

    for index, contribution in enumerate(contributions):
        count = contribution.vr_count
        vectors = (contribution.beam_east, contribution.beam_north,
                   contribution.beam_up)
        norm = np.sqrt(sum(component ** 2 for component in vectors))
        # A cell whose contributing beams summed to nothing has no look
        # direction, and a radial velocity without one is not an
        # observation: Vr = u*east + v*north + w*up is undefined for a zero
        # vector.  Such a cell is masked and counted here rather than
        # shipped with a zero beam under a true mask, which would silently
        # zero every innovation it touched.
        has_vr = (count > 0) & (norm > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(has_vr, contribution.vr_sum
                            / np.maximum(count, 1), 0.0)
            variance = np.where(
                count > 1,
                np.maximum(contribution.vr_sumsq / np.maximum(count, 1)
                           - mean * mean, 0.0),
                0.0)
        beamless = (count > 0) & ~has_vr
        vr_obs[index] = np.where(has_vr, mean, 0.0)
        vr_count[index] = np.where(has_vr, count, 0)
        vr_rejected[index] = (contribution.vr_rejected
                              + np.where(beamless, count, 0))
        vr_mask[index] = has_vr.astype(np.int8)
        vr_err[index] = np.where(
            has_vr,
            np.maximum(
                params.vr_error_floor_ms,
                np.sqrt(params.vr_error_base_ms ** 2 / np.maximum(count, 1)
                        + variance)),
            0.0)
        for axis, component in enumerate(vectors):
            beam[axis][index] = np.where(has_vr, component
                                         / np.where(has_vr, norm, 1.0), 0.0)

    radars = [{
        "id": contribution.site_id,
        "lat_deg": float(contribution.lat_deg),
        "lon_deg": float(contribution.lon_deg),
        "alt_m": float(contribution.alt_m),
        "valid_time": contribution.valid_time,
    } for contribution in contributions]

    return GriddedObservations(
        z_obs=z_obs, z_mask=z_mask, z_err=z_err, z_max=z_max, z_mean=z_mean,
        z_count=z_count.astype(np.int32),
        z0_mask=z0_mask, z0_count=z0_count.astype(np.int32), z0_err=z0_err,
        clear_air_source=clear_air_source,
        vr_obs=vr_obs, vr_mask=vr_mask, vr_err=vr_err, vr_count=vr_count,
        vr_rejected=vr_rejected,
        vr_beam_east=beam[0], vr_beam_north=beam[1], vr_beam_up=beam[2],
        radars=radars,
        counts=[c.counts.to_payload() for c in contributions],
        provenance=[c.provenance for c in contributions],
        fold_suspicion=[list(c.fold_suspicion) for c in contributions],
        dealias=[dict(c.dealias) for c in contributions
                 if getattr(c, "dealias", None)])
