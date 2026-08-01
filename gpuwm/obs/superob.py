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

**Aliasing is masked, not corrected.**  There is no dealiasing in v1.  Four
structural defenses run, and it matters exactly what each of them can and
cannot see:

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

from gpuwm.obs.geometry import (REFRACTION_FACTOR, gate_locations)
from gpuwm.obs.sweeps import RadarVolume
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.projection import EARTH_RADIUS_M

#: Moment tokens this stage understands.
REFLECTIVITY = "REF"
VELOCITY = "VEL"

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
        elevation = float(self.max_elevation_deg)
        if not np.isfinite(elevation) or not (0.0 < elevation <= 90.0):
            raise SuperobParamsError(
                f"max_elevation_deg is {elevation!r}; an antenna elevation "
                "ceiling must lie in (0, 90]. At or below 0 no sweep is ever "
                "used; above 90 the ceiling is not an elevation and cannot "
                "exclude anything")
        return self

    def to_payload(self) -> dict:
        return {key: float(value) for key, value in asdict(self).items()}


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

    def to_payload(self) -> dict:
        return {key: int(value) for key, value in asdict(self).items()}


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
    #: One record per velocity-carrying sweep, whether or not anything was
    #: flagged: an absence of fold boundaries is evidence too, and only
    #: means something beside the number of pairs that were tested.
    fold_suspicion: list = field(default_factory=list)


def superob_volume(volume: RadarVolume, grid: TargetGrid, *,
                   params: SuperobParams | None = None) -> RadarContribution:
    """Grid one radar volume onto ``grid``.

    Accumulators only — the dBZ/velocity/error reduction happens once, in
    :func:`merge_contributions`, so a multi-radar product and a single-radar
    product go through exactly the same arithmetic.
    """

    params = (params or SuperobParams()).validate()
    shape = (grid.nz, grid.ny, grid.nx)
    zeros = lambda: np.zeros(shape, dtype=np.float64)     # noqa: E731
    counts = SuperobCounts()

    z_linear_sum = zeros()
    z_sum_dbz = zeros()
    z_sumsq_dbz = zeros()
    z_count = np.zeros(shape, dtype=np.int64)
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

    site = volume.site
    max_range_m = params.max_range_km * 1000.0

    for sweep in volume.sweeps:
        if sweep.elevation_angle_deg > params.max_elevation_deg:
            counts.sweeps_skipped_elevation += 1
            continue
        nyquist = _believable_nyquist(sweep.nyquist_velocity_ms, params)
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

                i_frac, j_frac = grid.mass_index(lat.ravel(), lon.ravel())
                i_index = np.rint(i_frac).astype(np.intp)
                j_index = np.rint(j_frac).astype(np.intp)
                on_grid = grid.inside(i_index, j_index) & finite
                counts.gates_out_of_grid += int(finite.sum() - on_grid.sum())
                if not np.any(on_grid):
                    continue

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

                if product == REFLECTIVITY:
                    echo = value_sel >= params.min_reflectivity_dbz
                    counts.gates_below_floor += int((~echo).sum())
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
                within = (np.abs(value_sel)
                          <= params.nyquist_reject_fraction * nyquist)
                counts.velocity_gates_rejected_nyquist += int((~within).sum())
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
        z_linear_sum=z_linear_sum, z_count=z_count, z_max_dbz=z_max_dbz,
        z_sumsq_dbz=z_sumsq_dbz, z_sum_dbz=z_sum_dbz,
        vr_sum=vr_sum, vr_sumsq=vr_sumsq, vr_count=vr_count,
        vr_min=vr_min, vr_max=vr_max,
        beam_east=beam_east, beam_north=beam_north, beam_up=beam_up,
        nyquist_min=nyquist_min, vr_rejected=vr_rejected,
        counts=counts, provenance=volume.provenance(),
        fold_suspicion=fold_suspicion)


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

    z_linear = sum(c.z_linear_sum for c in contributions)
    z_count = sum(c.z_count for c in contributions)
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
        vr_obs=vr_obs, vr_mask=vr_mask, vr_err=vr_err, vr_count=vr_count,
        vr_rejected=vr_rejected,
        vr_beam_east=beam[0], vr_beam_north=beam[1], vr_beam_up=beam[2],
        radars=radars,
        counts=[c.counts.to_payload() for c in contributions],
        provenance=[c.provenance for c in contributions],
        fold_suspicion=[list(c.fold_suspicion) for c in contributions])
