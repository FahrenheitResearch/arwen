"""The acoustic substep count each domain's own terrain needs.

THE DEFECT THIS EXISTS FOR.  A generated 500 m forecast over the central
Andes (33 S, 70.1 W) ran the four acoustic substeps per 2.5 s step every
generated configuration carries, and its surface w passed 200 m/s over
3657 m terrain at model second 40.  The same prepared forecast ran its full
hour with six substeps and with eight.  Nothing in the configuration was
wrong for the ground it was written for; four substeps are simply not
stable over ground that steep.

THE PHYSICS.  The split-explicit acoustic step advances the horizontal
pressure gradient explicitly and the vertical one implicitly, off-centred
by ``epssm``.  On a sloping terrain-following surface the explicit
horizontal gradient carries the slope times the vertical pressure
gradient, so steep ground couples the explicit half of the step to the
fast vertical acoustic modes the implicit half exists to hold.  WRF's own
guidance is the same statement from the other side: the off-centering
should exceed the slope for stability (Dudhia, from the MM5 slope
analysis).  Every shipped profile carries ``epssm = 0.5``.

MEASURED, NOT ASSUMED.  :data:`STABLE_SLOPE_BY_OFFCENTERING` is the
engine's own stability map: a bell ridge in a uniform cross-ridge wind,
integrated through the production ``step()`` with the generated dynamics
(the 49-level hybrid ladder, ``smdiv``/``emdiv``, ``w_damping``, the
slope-tapered sixth-order filter, the Rayleigh lid) at 250 m, 500 m and
1 km on the generated 5 s per km clock, ridges 750 m to 5 km tall, cross
winds 10 to 25 m/s, ``epssm`` 0.1 to 1.0, with isolated mountains and a
ridge at 45 degrees to the grid beside them.  Over that map:

* the slope at which four substeps first fail rises with ``epssm`` (0.45
  at 0.1, 0.75 at 0.5) and does not move with the wind; the tallest,
  widest ridges fail first, so they set every bound;
* six substeps hold 0.1 to 0.2 more slope, and eight hold no more than
  six;
* a shorter long step does not move the four-substep boundary between 2
  and 5 s per km, so a larger count, not a smaller ``dt``, is the remedy;
* the steepest slope in any direction governs: a ridge at 45 degrees to
  the grid failed four substeps at a slope of 0.75 though no face saw
  more than 0.53, a little below the ridge along the grid, and an
  isolated mountain is no more stable than a ridge;
* the bounds below held three hours on the widest ridge across the grid
  at 250 m and 500 m: at ``epssm`` 0.5, four substeps at a slope of 0.70
  and six at 0.85 (six at 0.90 held an hour and a half at 250 m, then
  failed).

The real forecasts agree: the Andes at 500 m (steepest slope 0.89) fails
with four and runs with six; Hawaii at 1 km (0.33), the Rockies at 500 m
(0.36) and the Iowa plains (0.02) run with four.

WHERE THE MAP ENDS.  The wind in the map is 10 to 25 m/s.  Under a jet at
crest height the four-substep bound falls over the tallest ground: at
1 km, crests of 5.5 and 6.5 km stopped on four substeps at slopes of 0.6
to 0.65 under 50 to 60 m/s and held three hours on six, while a 5 km
crest held 0.65 on four at 70 m/s and 500 m held 0.65 on four at 60 m/s.
At 3 km, 4.5 and 5 km crests of slope 0.4 under 60 to 70 m/s stopped
within two minutes on four and on six alike, so there the long step,
not the substep count, is the limit.  This rule reads slope and
off-centering only; :mod:`gpuwm.terrain_clock` reads the crest-level
wind beside the same slope and takes six substeps or a shorter long step
where its own map, measured under winds to 100 m/s, says either is needed.

WHAT THIS DOES.  Each domain's steepest slope in any direction
(:func:`steepest_slope`, map factor applied) is read off the terrain that
domain integrates.  Where it reaches the bound below which four substeps
were measured stable at the domain's ``epssm``, ``time_step_sound`` is
raised to six.  A count is never lowered, so a configuration that asked
for more keeps it, and a domain on gentler ground runs exactly as
configured.  Past the slope six substeps hold, the run still takes six
(the most stable count measured) and says plainly that its ground is
steeper than any stable measurement.

UNDER THE ADAPTIVE CLOCK the configured count is not the one that runs:
the clock derives the count from its live step every root step, and that
gives four at every short step.  There the rule reads the count the clock
takes at its shortest steps (:func:`gpuwm.core.adaptive_clock
.least_sound_steps`) and writes six as ``min_time_step_sound``, the floor
the clock keeps its derived count at or above, so the six reach the
dynamics and a domain on gentler ground keeps WRF's derived count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping

import numpy as np

#: Receipt schema for the derivation this module performs.
ACOUSTIC_ADAPTATION_SCHEMA = "gpuwm-acoustic-substep-adaptation-v1"

#: The count every derivation here raises to.  Eight was measured no more
#: stable than six at any slope, so a larger count would cost acoustic
#: work and buy nothing.
STEEP_TERRAIN_SOUND_STEPS = 6

#: The engine's measured stability map, per vertical off-centering.
#:
#: Rows are ``(epssm, four_below, six_below)``: at an ``epssm`` of at
#: least the row's, four substeps were stable on every ridge whose
#: steepest slope stayed below ``four_below``, and six below
#: ``six_below``.  Each bound sits under the lowest slope at which that
#: count failed on any measured ridge, along the grid or across it, and a
#: ridge at each bound held three hours.  An ``epssm`` between rows takes
#: the row below it, and one above the last row the last row.
STABLE_SLOPE_BY_OFFCENTERING: tuple[tuple[float, float, float], ...] = (
    (0.1, 0.40, 0.50),
    (0.2, 0.55, 0.65),
    (0.3, 0.60, 0.70),
    (0.4, 0.65, 0.80),
    (0.5, 0.70, 0.85),
)


@dataclass(frozen=True)
class SlopeReading:
    """The steepest slope of one domain's terrain, and the face it is on.

    ``face`` is ``(axis, j, i)`` with ``axis`` ``"x"`` for the face between
    mass points ``(j, i - 1)`` and ``(j, i)`` and ``"y"`` for the face
    between ``(j - 1, i)`` and ``(j, i)``.
    """

    label: str
    slope: float
    face: tuple[str, int, int]

    @property
    def degrees(self) -> float:
        return math.degrees(math.atan(self.slope))


def steepest_slope(terrain, dx: float, dy: float, *, msfu=None,
                   msfv=None, label: str = "") -> SlopeReading:
    """The steepest ground slope of a terrain field, in any direction.

    Read at every face, where the acoustic step's horizontal pressure
    gradient lives: the height step across the face over the ground
    distance between its two mass points, combined with the slope along
    the face (the centred differences of the two columns it separates,
    averaged onto it).  The direction matters: a ridge at 45 degrees to
    the grid shows each face only 0.71 of its slope, and four substeps
    failed on it at its whole slope, a little below where they failed on
    a ridge along the grid.

    The distance between neighbouring mass points on the map plane is
    ``dx`` (``dy``); on the ground it is that over the face's map factor.
    ``msfu`` is ``(ny, nx + 1)`` and ``msfv`` ``(ny + 1, nx)``, the
    staggered factors the dynamics use; omitted, the factor is one.
    """

    h = np.asarray(terrain, dtype=np.float64)
    if h.ndim != 2 or h.size == 0:
        raise ValueError(
            f"terrain for {label or 'a domain'} must be a non-empty (ny, nx) "
            f"array, got shape {h.shape}")
    if not np.isfinite(h).all():
        raise ValueError(
            f"terrain for {label or 'a domain'} carries non-finite heights; "
            "its steepest slope cannot be read")
    ny, nx = h.shape
    across_x = (np.gradient(h, float(dx), axis=1) if nx > 1
                else np.zeros_like(h))
    across_y = (np.gradient(h, float(dy), axis=0) if ny > 1
                else np.zeros_like(h))
    candidates = []
    if nx > 1:
        slope = np.hypot(np.diff(h, axis=1) / float(dx),
                         0.5 * (across_y[:, 1:] + across_y[:, :-1]))
        if msfu is not None:
            slope = slope * np.asarray(msfu, dtype=np.float64)[:, 1:-1]
        j, i = np.unravel_index(int(np.argmax(slope)), slope.shape)
        candidates.append((float(slope[j, i]), ("x", int(j), int(i) + 1)))
    if ny > 1:
        slope = np.hypot(np.diff(h, axis=0) / float(dy),
                         0.5 * (across_x[1:, :] + across_x[:-1, :]))
        if msfv is not None:
            slope = slope * np.asarray(msfv, dtype=np.float64)[1:-1, :]
        j, i = np.unravel_index(int(np.argmax(slope)), slope.shape)
        candidates.append((float(slope[j, i]), ("y", int(j) + 1, int(i))))
    if not candidates:
        return SlopeReading(label, 0.0, ("x", 0, 0))
    slope, face = max(candidates, key=lambda item: item[0])
    return SlopeReading(label, slope, face)


def stable_slopes(epssm: float) -> tuple[float, float, float]:
    """The measured row that applies at this off-centering.

    The row with the largest ``epssm`` not above the configured one: a
    value between measured rows is held to the less stable of the two.
    Below the first row the first row applies, which is the least stable
    measured and so the conservative reading.
    """

    epssm = float(epssm)
    chosen = STABLE_SLOPE_BY_OFFCENTERING[0]
    for row in STABLE_SLOPE_BY_OFFCENTERING:
        if row[0] <= epssm + 1e-12:
            chosen = row
    return chosen


@dataclass(frozen=True)
class AcousticAdaptation:
    """One domain's reading and the substep count it runs.

    ``configured`` is the count the domain would take at its shortest
    steps without the rule: its ``time_step_sound`` on a fixed clock, the
    adaptive clock's derived floor on an adaptive one.  ``adaptive`` says
    which, and on the adaptive clock ``time_step_sound`` is the floor the
    rule writes as ``min_time_step_sound``.
    """

    grid_id: int
    reading: SlopeReading
    epssm: float
    configured: int
    time_step_sound: int
    four_below: float
    six_below: float
    adaptive: bool = False

    @property
    def adapted(self) -> bool:
        return self.time_step_sound != self.configured

    @property
    def beyond_measured(self) -> bool:
        return self.reading.slope >= self.six_below

    @property
    def status(self) -> str:
        if self.beyond_measured:
            return "BEYOND_MEASURED"
        return "ADAPTED" if self.adapted else "AS_CONFIGURED"

    def _runs(self) -> str:
        """What the domain runs, in the words its clock makes true."""

        if self.adaptive:
            return (f"at least {self.time_step_sound} substeps per step on "
                    f"its adaptive clock")
        return f"{self.time_step_sound} substeps per step"

    def _instead(self) -> str:
        if self.adaptive:
            return (f" instead of the {self.configured} that clock takes at "
                    "short steps")
        return f" instead of {self.configured}"

    def sentence(self) -> str:
        """The plain line a run prints when this domain's count changed."""

        reading = self.reading
        return (
            f"acoustic substeps: {reading.label}'s steepest terrain slope is "
            f"{reading.slope:.2f} ({reading.degrees:.0f} degrees), and four "
            f"acoustic substeps per step were measured stable only below "
            f"{self.four_below:.2f} at epssm {self.epssm:g}, so "
            f"{reading.label} runs {self._runs()}{self._instead()}")

    def beyond_sentence(self) -> str:
        """The plain line a run prints over ground past every stable count."""

        reading = self.reading
        change = self._instead() if self.adapted else ""
        return (
            f"acoustic substeps: {reading.label}'s steepest terrain slope is "
            f"{reading.slope:.2f} ({reading.degrees:.0f} degrees), steeper "
            f"than {self.six_below:.2f}, the steepest ground any acoustic "
            f"substep count was measured stable on at epssm {self.epssm:g}; "
            f"{reading.label} runs {self._runs()}"
            f"{change}, the most stable count measured, and may still stop "
            "over that ground.  The same area at a coarser grid spacing has "
            "gentler slopes")

    def receipt(self) -> dict:
        reading = self.reading
        axis, j, i = reading.face
        row = {
            "grid_id": int(self.grid_id),
            "status": self.status,
            "configured_time_step_sound": int(self.configured),
            "time_step_sound": int(self.time_step_sound),
            "epssm": float(self.epssm),
            "steepest_slope": float(reading.slope),
            "steepest_slope_degrees": float(reading.degrees),
            "steepest_slope_face": {"axis": axis, "j": int(j), "i": int(i)},
            "four_substeps_stable_below": float(self.four_below),
            "six_substeps_stable_below": float(self.six_below),
        }
        if self.adaptive:
            # Only on the adaptive clock, so every fixed-clock record
            # reads as it did; the floor only where this rule set it.
            row["adaptive"] = True
            if self.adapted:
                row["min_time_step_sound"] = int(self.time_step_sound)
        return row


def derive_acoustics(grid_id: int, run, reading: SlopeReading
                     ) -> AcousticAdaptation:
    """The count one domain runs: the count it takes without the rule, or
    six when its ground reaches the measured four-substep boundary."""

    from gpuwm.core.adaptive_clock import least_sound_steps

    epssm = float(run.epssm)
    _, four_below, six_below = stable_slopes(epssm)
    configured = least_sound_steps(run)
    count = configured
    if reading.slope >= four_below:
        count = max(configured, STEEP_TERRAIN_SOUND_STEPS)
    return AcousticAdaptation(
        grid_id=int(grid_id), reading=reading, epssm=epssm,
        configured=configured, time_step_sound=count,
        four_below=four_below, six_below=six_below,
        adaptive=bool(getattr(run, "use_adaptive_time_step", False)))


def adapted_run(run, count: int):
    """``run`` with its acoustic substep count raised to ``count``.

    ``time_step_sound`` is never lowered.  On the adaptive clock, which
    derives its count from the live step, ``count`` is also written as
    ``min_time_step_sound``, the floor that count keeps, or the clock
    would put it back to four at the next short step.  ``run`` itself when
    nothing moves.
    """

    from gpuwm.core.adaptive_clock import sound_steps_floor

    fields = {}
    if int(count) > int(run.time_step_sound):
        fields["time_step_sound"] = int(count)
    if (bool(getattr(run, "use_adaptive_time_step", False))
            and int(count) > max(4, sound_steps_floor(run))):
        fields["min_time_step_sound"] = int(count)
    return replace(run, **fields) if fields else run


def adapt_experiment_acoustics(
        exp, readings: Mapping[int, SlopeReading], *,
        announce: Callable[[str], None] | None = None,
        caution: Callable[[str], None] | None = None):
    """Return ``(experiment, adaptations)`` with each read domain's count.

    ``readings`` maps ``grid_id`` to the steepest slope of the terrain that
    domain integrates (a moving nest's reading covers its corridor).  A
    domain with no reading keeps its configured count.  ``caution``
    receives the line for every domain past the six-substep boundary and
    ``announce`` the line for every other domain whose count changed;
    ``None`` prints nothing.
    """

    adaptations = []
    domains = []
    for dc in exp.domains:
        reading = readings.get(int(dc.grid_id))
        if reading is None:
            domains.append(dc)
            continue
        adaptation = derive_acoustics(dc.grid_id, dc.run, reading)
        adaptations.append(adaptation)
        if adaptation.adapted:
            dc = replace(dc, run=adapted_run(
                dc.run, adaptation.time_step_sound))
        # One line per domain: the caution already names the count it runs.
        if adaptation.beyond_measured:
            if caution is not None:
                caution(adaptation.beyond_sentence())
        elif adaptation.adapted and announce is not None:
            announce(adaptation.sentence())
        domains.append(dc)
    if not any(adaptation.adapted for adaptation in adaptations):
        return exp, tuple(adaptations)
    return replace(exp, domains=tuple(domains)), tuple(adaptations)


#: The mechanism behind both lines, printed under ``--explain``.
_WHY = (
    "The acoustic step advances the horizontal pressure gradient "
    "explicitly and the vertical one implicitly; on a sloping "
    "terrain-following surface the explicit gradient carries the slope "
    "times the vertical one, so steep ground feeds the fast vertical sound "
    "waves into the explicit half of the step.  The engine's measured "
    "stability map (gpuwm/acoustic_adaptation.py) gives, for each epssm, "
    "the slope at which four substeps per step fail and the slope six "
    "hold; more than six holds no more.")


def adapt_experiment_to_terrain(exp, readings: Mapping[int, SlopeReading]):
    """:func:`adapt_experiment_acoustics` in the run's own voice.

    Each changed domain prints one warning line, and each domain past the
    six-substep boundary one more; ``--explain`` adds the mechanism.
    """

    from gpuwm.explain import warn

    return adapt_experiment_acoustics(
        exp, readings,
        announce=lambda sentence: warn(sentence, _WHY),
        caution=lambda sentence: warn(sentence, _WHY))


def acoustic_receipt(adaptations) -> dict:
    """The derivation for the run document: every domain read, changed or
    not, because a record that appears only when something changed cannot
    show that nothing did."""

    return {
        "schema": ACOUSTIC_ADAPTATION_SCHEMA,
        "domains": [adaptation.receipt() for adaptation in adaptations],
        "measured_map": [
            {"epssm": row[0], "four_substeps_stable_below": row[1],
             "six_substeps_stable_below": row[2]}
            for row in STABLE_SLOPE_BY_OFFCENTERING],
    }


def readings_from_static(exp, static_by_grid_id, *, grids_by_grid_id=None
                         ) -> dict[int, SlopeReading]:
    """One reading per domain from the static fields each domain runs on.

    ``static_by_grid_id`` maps ``grid_id`` to a static-field mapping holding
    ``HGT_M`` and, where the preparation wrote them, ``MAPFAC_U`` and
    ``MAPFAC_V``.  A grid object with ``mapfac_u()``/``mapfac_v()`` supplies
    the factors when the static fields do not.
    """

    readings = {}
    for dc in exp.domains:
        gid = int(dc.grid_id)
        static = static_by_grid_id.get(gid)
        if static is None or "HGT_M" not in static:
            continue
        msfu = static.get("MAPFAC_U") if hasattr(static, "get") else None
        msfv = static.get("MAPFAC_V") if hasattr(static, "get") else None
        grid = (grids_by_grid_id or {}).get(gid)
        if msfu is None and grid is not None and hasattr(grid, "mapfac_u"):
            msfu = grid.mapfac_u()
        if msfv is None and grid is not None and hasattr(grid, "mapfac_v"):
            msfv = grid.mapfac_v()
        readings[gid] = steepest_slope(
            np.asarray(static["HGT_M"]), float(dc.run.dx), float(dc.run.dy),
            msfu=msfu, msfv=msfv, label=f"d{gid:02d}")
    return readings


def fold_corridor_reading(readings: dict, grid_id: int, run,
                          corridor_terrain) -> None:
    """Hold a relocating nest to the steepest ground it can move over.

    A following nest is read at its start like any domain, but it can be
    moved mid-run onto steeper ground than it starts on, and the count it
    took at its start would not hold there.  ``corridor_terrain`` is its
    statics corridor at its own resolution -- the ground it can reach --
    and the steeper of that reading and the start one governs, in place.
    """

    gid = int(grid_id)
    reach = steepest_slope(
        np.asarray(corridor_terrain), float(run.dx), float(run.dy),
        label=f"d{gid:02d}")
    held = readings.get(gid)
    if held is None or reach.slope > held.slope:
        readings[gid] = reach
