"""The long time step each domain's ground and crest-level wind allow.

THE DEFECT THIS EXISTS FOR.  The acoustic substep rule
(:mod:`gpuwm.acoustic_adaptation`) was measured under cross-ridge winds of
10 to 25 m/s.  Under a jet at crest height the same ground stops runs that
rule leaves alone: at 1 km, 5.5 and 6.5 km crests of slope 0.6 to 0.65
stopped on four substeps under 50 to 60 m/s, and at 3 km, 4.5 and 5 km
crests of slope 0.4 under 60 to 70 m/s stopped within two minutes on four
AND six substeps.  There no substep count is the remedy; the long step
is.

MEASURED, NOT ASSUMED.  :data:`MAP_PATH` is the engine's own map: a bell
ridge across the grid in a uniform cross-ridge wind, integrated for half
an hour through the production ``step()`` with the generated dynamics
(the 49-level ladder, the etac the vertical survey derives for the crest,
``epssm`` 0.5, ``smdiv``/``emdiv``, ``w_damping``, the slope-tapered
sixth-order filter, the Rayleigh lid), at 500 m to 12 km, crests of 1.5
to 8.85 km and winds of 20 to 100 m/s.  Each row is one spacing, crest
height, grid-read steepest slope and substep count; for each crest-level
wind it holds the longest step, in seconds per kilometre of grid spacing,
that ran without stopping and without its vertical velocity passing four
times the wind times the slope plus 20 m/s.  ``None`` means no measured
step held.  The probe domain is periodic and 96 to 256 columns wide, so
from 50 m/s up the flow can come back round to the ridge inside the half
hour; every step held there was run again on a domain the flow cannot
wrap, and lowered to what held on it (at 1 km, a 4.5 km crest of slope
0.4 held 5 s/km at 90 and 100 m/s on 96 columns and stopped within
fifteen minutes on 1024).  Then every pair the rule picks at 1 and 3 km
from the generated step under 40 to 80 m/s ran three hours on a ridge
whose flow cannot wrap the domain in that time: 135 of 142 held, and
where one stopped the row now holds the step that held three hours
there, or nothing.  Over the map:

* at 3 km the generated 15 s step holds every wind to 90 m/s on slopes
  to about 0.25 under crests to 4.5 km.  Steeper ground needs a shorter
  step from 40 to 80 m/s: under a 4.5 km crest a slope of 0.41 holds
  4.0 s/km at 50 m/s on four substeps and 4.5 on six, and 3.0 and 3.5
  at 100;
* at 1 km six substeps hold the generated 5 s step to 90 m/s on slopes
  to about 0.55 under crests to 4.5 km, to 70 m/s on slopes to about
  0.65 under a 5.5 km crest and to 60 m/s on slopes to about 0.75 under
  a 6.5 km crest.  Steeper ground needs a shorter step from 50 to 80 m/s,
  and slopes near 1 from 30;
* the taller the crest, the thinner the layer over it and the lower the
  wind past which no measured step holds at all: at 3 km, 80 m/s under a
  5.5 km crest, 70 under 6.5 km, 60 under 8 km and 40 under 8.85 km.
  There the layer, not the step, is the limit.

WHAT THIS DOES.  Each domain is read for its steepest slope (the reading
the substep rule takes), its crest height (its highest ground) and its
crest-level wind: the strongest wind its start state and boundary data
carry over the forecast window, from the ground up to the first model
level at or above the crest height.  The speed is taken whole, not its
component across the ground, because the wind turns over a forecast and
real ground faces every way; that is the conservative reading.  The map
is read on its conservative side (the next steeper slope, taller crest,
stronger wind and both neighbouring spacings) and the domain runs the
cheapest pair it holds: the smallest whole division of its configured step,
then the fewest substeps that hold at it.  A domain the map holds as
configured runs exactly as configured.  The step is only ever divided by a
whole number, so every cadence that was a whole number of steps stays
one; a nest's step ratio grows by the division its parent does not
already give it.  Under the adaptive clock the held step becomes a
ceiling on ``max_time_step``.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from gpuwm.core import constants as _constants

#: Receipt schema for the derivation this module performs.
TERRAIN_CLOCK_SCHEMA = "gpuwm-terrain-clock-adaptation-v1"

#: The measured map this rule reads.
MAP_PATH = Path(__file__).with_name("terrain_clock_map.json")

#: The two substep counts the map measured.  A configured count of six or
#: more reads the six-substep rows; any smaller count the four-substep rows.
MAPPED_SOUND_STEPS = (4, 6)

#: How far a domain's crest may sit above a mapped crest and still read
#: that row, as a share of the crest.  The mapped heights are the ridges'
#: nominal heights, their own highest cells sit 1 to 13 percent under them,
#: and the height acts through the layer over the crest, which the 15
#: percent margin of :mod:`gpuwm.vertical_adaptation` sets alike on both:
#: a 6456.3 m crest is the 6456 m row's ground, not the 8000 m row's.
CREST_MATCH = 0.01

#: Gravitational acceleration that turns geopotential into height: the
#: dynamics' own.
_G = float(_constants.G)


@dataclass(frozen=True)
class MapRow:
    dx_m: float
    crest_m: float
    slope: float
    sound_steps: int
    #: Longest held step in s/km per mapped wind, ``None`` where none held.
    stable: tuple[float | None, ...]


@dataclass(frozen=True)
class StableStepMap:
    winds: tuple[float, ...]
    ladder: tuple[float, ...]
    rows: tuple[MapRow, ...]
    seconds: float

    @property
    def top(self) -> float:
        """The longest step measured, in s/km: a row holding it held every
        step the map tried."""
        return float(self.ladder[0])


@functools.lru_cache(maxsize=1)
def measured_map() -> StableStepMap:
    document = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    winds = tuple(float(w) for w in document["winds_m_s"])
    rows = []
    for row in document["rows"]:
        stable = tuple(None if value is None else float(value)
                       for value in row["stable_s_per_km"])
        if len(stable) != len(winds):
            raise ValueError(
                f"{MAP_PATH.name}: a row carries {len(stable)} winds, the "
                f"map declares {len(winds)}")
        rows.append(MapRow(float(row["dx_m"]), float(row["crest_m"]),
                           float(row["slope"]), int(row["sound_steps"]),
                           stable))
    return StableStepMap(winds, tuple(float(v) for v in
                                      document["ladder_s_per_km"]),
                         tuple(rows), float(document["seconds"]))


@dataclass(frozen=True)
class MapReading:
    """What the map says for one domain at one substep count."""

    #: Longest held step in s/km, or ``None`` when the map measured no
    #: step holding there.
    per_km: float | None
    #: The mapped value the reading was taken at, each on its
    #: conservative side.
    dx_rows: tuple[float, ...]
    crest_row: float
    slope_row: float | None
    wind_row: float
    #: Which of the domain's readings lie past the map's edge.
    beyond: tuple[str, ...] = ()
    #: ``per_km`` where the map holds a step; past the strongest wind it
    #: holds one at, the step it held at the strongest wind it did, s/km:
    #: the most stable pair measured, which such a domain runs.
    most_stable_per_km: float | None = None


def _sound_row(sound_steps: int) -> int:
    return 6 if int(sound_steps) >= 6 else 4


def read_map(dx: float, crest_m: float, slope: float, wind: float,
             sound_steps: int, table: StableStepMap | None = None
             ) -> MapReading:
    """The longest held step for a domain, read on the map's safe side.

    Spacing: both mapped spacings around ``dx`` (the nearest one past
    either end).  Crest: the lowest mapped crest at or above the domain's
    highest ground, the ridge as tall as its mountains.  A lower ridge of
    the same slope is a narrower one, a few cells wide, which is a
    different ground and not a safer reading of this one.  Slope: at that
    crest, the gentlest mapped slope at or above the domain's, and every
    gentler row, so a noisy row can only lower the answer.  Wind: the
    lowest mapped wind at or above the domain's, and every weaker one.
    The answer is the least of all of them.
    """

    table = measured_map() if table is None else table
    count = _sound_row(sound_steps)
    rows = [row for row in table.rows if row.sound_steps == count]
    beyond = []
    spacings = sorted({row.dx_m for row in rows})
    dx = float(dx)
    if dx <= spacings[0]:
        dx_rows = (spacings[0],)
        if dx < spacings[0] * (1.0 - 1e-9):
            beyond.append("spacing")
    elif dx >= spacings[-1]:
        dx_rows = (spacings[-1],)
        if dx > spacings[-1] * (1.0 + 1e-9):
            beyond.append("spacing")
    else:
        exact = [s for s in spacings if abs(s - dx) <= 1e-6 * s]
        if exact:
            dx_rows = (exact[0],)
        else:
            lower = max(s for s in spacings if s < dx)
            upper = min(s for s in spacings if s > dx)
            dx_rows = (lower, upper)
    winds = table.winds
    above_w = [i for i, w in enumerate(winds) if w >= float(wind) - 1e-9]
    if above_w:
        wind_index = above_w[0]
    else:
        wind_index = len(winds) - 1
        beyond.append("wind")

    selected = []
    slope_rows = []
    crest_rows = []
    slope_beyond = crest_beyond = False
    for spacing in dx_rows:
        crests = sorted({row.crest_m for row in rows
                         if row.dx_m == spacing})
        above = [h for h in crests
                 if h >= float(crest_m) * (1.0 - CREST_MATCH) - 1e-6]
        if above:
            crest_row = above[0]
        else:
            crest_row = crests[-1]
            crest_beyond = True
        crest_rows.append(crest_row)
        governing = [row for row in rows if row.dx_m == spacing
                     and row.crest_m == crest_row]
        steeper = sorted(row.slope for row in governing
                         if row.slope >= float(slope) - 1e-9)
        if steeper:
            slope_cap = steeper[0]
        else:
            slope_cap = max(row.slope for row in governing)
            slope_beyond = True
        slope_rows.append(slope_cap)
        selected.extend(row for row in governing
                        if row.slope <= slope_cap + 1e-12)
    if crest_beyond:
        beyond.append("crest")
    if slope_beyond:
        beyond.append("slope")

    def held_through(index):
        values = []
        for row in selected:
            window = row.stable[:index + 1]
            if any(v is None for v in window):
                return None
            values.append(min(window))
        return min(values) if values else None

    per_km = held_through(wind_index)
    most_stable = per_km
    if per_km is None:
        # Past the strongest wind every governing row held: what they held
        # at the strongest wind they all did, the most stable pair measured.
        for index in range(wind_index - 1, -1, -1):
            most_stable = held_through(index)
            if most_stable is not None:
                break
    return MapReading(
        per_km=per_km, dx_rows=tuple(dx_rows), crest_row=max(crest_rows),
        slope_row=max(slope_rows) if slope_rows else None,
        wind_row=winds[wind_index], beyond=tuple(beyond),
        most_stable_per_km=most_stable)


# ---------------------------------------------------------------------------
# Crest-level wind, read from the inputs a run integrates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrestWind:
    """The strongest wind a domain's inputs carry at or below its crest.

    ``when`` names the input it came from ("start" or "boundary at +3 h"),
    ``source`` which domain's inputs, and ``height_m`` the height of the
    model level it sits on.
    """

    label: str
    crest_height_m: float
    wind_m_s: float
    when: str
    source: str
    height_m: float

    def receipt(self) -> dict:
        return {"crest_height_m": float(self.crest_height_m),
                "crest_level_wind_m_s": float(self.wind_m_s),
                "when": self.when, "source": self.source,
                "height_m": float(self.height_m)}


def crest_band(heights: np.ndarray, crest_height: float) -> np.ndarray:
    """Levels from the ground up to the first at or above the crest.

    ``heights`` is ``(nlevel, ...)`` ordered upward.  A level belongs when
    it is the lowest one or the level below it is under the crest height,
    so the column over the crest itself contributes its lowest level.
    """

    heights = np.asarray(heights, dtype=np.float64)
    under = np.asarray(heights[:-1] < float(crest_height))
    first = np.ones((1,) + heights.shape[1:], dtype=bool)
    return np.concatenate([first, under], axis=0)


def _strongest(speed, heights, crest_height):
    """``(speed, height)`` of the strongest wind in the crest band."""

    speed = np.asarray(speed, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    band = crest_band(heights, crest_height)
    masked = np.where(band & np.isfinite(speed), speed, -np.inf)
    index = int(np.argmax(masked))
    value = float(masked.flat[index])
    if not math.isfinite(value):
        return None
    return value, float(heights.flat[index])


def mass_speed(u, v) -> np.ndarray:
    """Horizontal wind speed at mass points from C-grid ``u``/``v``."""

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ua = 0.5 * (u[..., :-1] + u[..., 1:])
    va = 0.5 * (v[..., :-1, :] + v[..., 1:, :])
    return np.hypot(ua, va)


def mass_heights(geopotential_full) -> np.ndarray:
    """Heights of the mass levels from full-level geopotential."""

    z = np.asarray(geopotential_full, dtype=np.float64) / _G
    return 0.5 * (z[:-1] + z[1:])


def _band_heights(php, phb, crest_height):
    """Each band level's mass-point heights, NaN where a column is past it.

    Walks up one level at a time, so no 3-D height array is held, and
    stops at the first level every column has already climbed past.
    """

    levels = []
    previous = None
    for k in range(int(phb.shape[0]) - 1):
        z = 0.5 * ((np.asarray(phb[k], dtype=np.float64)
                    + np.asarray(php[k], dtype=np.float64))
                   + (np.asarray(phb[k + 1], dtype=np.float64)
                      + np.asarray(php[k + 1], dtype=np.float64))) / _G
        if previous is None:
            inside = np.ones(z.shape, dtype=bool)
        else:
            inside = previous < float(crest_height)
            if not inside.any():
                break
        levels.append(np.where(inside, z, np.nan))
        previous = z
    return levels


def _strongest_over_levels(u, v, levels):
    """``(speed, height)`` of the strongest wind on the band levels."""

    best = None
    for k, heights in enumerate(levels):
        uk = np.asarray(u[k], dtype=np.float64)
        vk = np.asarray(v[k], dtype=np.float64)
        speed = np.hypot(0.5 * (uk[:, :-1] + uk[:, 1:]),
                         0.5 * (vk[:-1, :] + vk[1:, :]))
        masked = np.where(np.isfinite(heights) & np.isfinite(speed), speed,
                          -np.inf)
        index = int(np.argmax(masked))
        value = float(masked.flat[index])
        if math.isfinite(value) and (best is None or value > best[0]):
            best = (value, float(heights.flat[index]))
    return best


_START_KEYS = {"u": "state/u", "v": "state/v", "php": "state/php",
               "phb": "base/phb"}


@dataclass(frozen=True)
class StartWinds:
    """One domain's start state, read level by level when asked.

    ``fields`` returns ``u``, ``v``, ``php`` or ``phb`` by name; the arrays
    are loaded two at a time and dropped, so a large domain costs two of
    its 3-D fields at once, not a copy of the state.
    """

    source: str
    fields: Callable[[str], np.ndarray]

    @classmethod
    def from_fields(cls, source, u, v, php, phb):
        arrays = {"u": u, "v": v, "php": php, "phb": phb}
        return cls(source, arrays.__getitem__)

    def strongest(self, crest_height):
        levels = _band_heights(self.fields("php"), self.fields("phb"),
                               crest_height)
        found = _strongest_over_levels(self.fields("u"), self.fields("v"),
                                       levels)
        return None if found is None else (found[0], found[1], "start")


def _face_mass(mu, low):
    """Column mass on the faces of an x-side slab.

    ``mu`` is ``(ny, w)`` over the slab's mass columns in grid order.  On
    the low side the slab's faces are 0..w-1 and face 0 is the domain's
    outer face; on the high side they are the last w faces and the last is
    the outer one.  An outer face takes its boundary cell, as the coupled
    boundary values were built.
    """

    inner = 0.5 * (mu[:, :-1] + mu[:, 1:])
    if low:
        return np.concatenate([mu[:, :1], inner], axis=1)
    return np.concatenate([inner, mu[:, -1:]], axis=1)


def _row_mass(mu):
    """Column mass on the staggered rows across an x-side slab."""

    inner = 0.5 * (mu[:-1] + mu[1:])
    return np.concatenate([mu[:1], inner, mu[-1:]], axis=0)


@dataclass(frozen=True)
class BoundaryGeometry:
    """What uncoupling a root's boundary winds needs."""

    mub: np.ndarray
    phb: np.ndarray
    c1h: np.ndarray
    c2h: np.ndarray
    c1f: np.ndarray
    c2f: np.ndarray
    msfu: np.ndarray | None
    msfv: np.ndarray | None


@dataclass(frozen=True)
class BoundaryWinds:
    """A root's lateral boundary data over the forecast window.

    Read side by side on the boundary slabs themselves, so the cost is the
    frame and not the domain.
    """

    source: str
    boundaries: object
    geometry: BoundaryGeometry
    run_seconds: float

    def instants(self):
        seen = []
        end = float(self.run_seconds)
        for interval in self.boundaries.intervals:
            start = float(interval.start_seconds)
            stop = float(interval.end_seconds)
            for t in (start, min(stop, end)):
                if start <= t <= stop and t <= end + 1e-6:
                    seen.append((interval, t - start, t))
            if stop >= end:
                break
        return seen

    def strongest(self, crest_height):
        best = None
        for interval, offset, at in self.instants():
            fields = interval.fields
            if not {"u", "v", "mu"} <= set(fields):
                continue
            self._refuse_another_grid(fields)
            for side in ("west", "east", "south", "north"):
                found = self._side(fields, side, offset, crest_height)
                if found is not None and (best is None or found[0] > best[0]):
                    best = (found[0], found[1], _boundary_when(at))
        return best

    def _refuse_another_grid(self, fields):
        """Refuse boundary tables or map factors off the base state's grid.

        Breakage it prevents: the reading pairs every boundary column with
        the base state's column mass, ground and map factors by position,
        so tables built on another grid stopped a prepared tree's preflight
        with a bare NumPy broadcast error that named neither the domain nor
        the mismatch.  Such boundaries cannot drive this domain either: the
        restore refuses them against its state
        (``gpuwm.ingest.lateral_bc._validate_lateral_interval``), so the
        reading refuses them by name instead of guessing.
        """
        from gpuwm.ingest.lateral_bc import _boundary_field_shape

        geo = self.geometry
        ny, nx = np.shape(geo.mub)
        nz = int(np.size(geo.c1h))
        cells = f"{nz} levels over {ny} x {nx} columns"

        def dims(shape):
            return " x ".join(str(int(n)) for n in shape)

        wanted = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
                  "mu": (1, ny, nx), "phi": (nz + 1, ny, nx)}
        for name, expected in wanted.items():
            if name not in fields:
                continue
            found = _boundary_field_shape(fields[name])[:3]
            if tuple(found) != expected:
                raise ValueError(
                    f"{self.source}'s lateral boundary data is for another "
                    f"grid than its base state: its {name} tables belong to "
                    f"a {dims(found)} field, where the base state's {cells} "
                    f"make it {dims(expected)}.  The time-step reading "
                    "pairs each boundary column with the base state's "
                    "column mass and ground, and the forecast's restore "
                    "refuses boundary tables of another shape, so neither "
                    "can use them; prepare the boundaries and the start "
                    "state together, in one preparation of this domain")
        for name, value, expected in (
                ("MAPFAC_U", geo.msfu, (ny, nx + 1)),
                ("MAPFAC_V", geo.msfv, (ny + 1, nx))):
            if value is not None and tuple(np.shape(value)) != expected:
                raise ValueError(
                    f"{self.source}'s {name} is {dims(np.shape(value))}, "
                    f"where its base state's {ny} x {nx} columns make it "
                    f"{dims(expected)}.  The time-step reading scales each "
                    "boundary wind by the map factor at its column, so it "
                    "cannot read map factors from another grid; prepare "
                    "the domain's static fields and its start state on one "
                    "grid")

    def _side(self, fields, side, offset, crest_height):
        from gpuwm.ingest.lateral_bc import evaluate_boundary_side

        geo = self.geometry

        def value(name):
            part = getattr(fields[name], side)
            array = np.asarray(evaluate_boundary_side(part, offset)[0],
                               dtype=np.float64).reshape(part.value.shape)
            if side == "east":
                array = array[..., ::-1]
            elif side == "north":
                array = array[..., ::-1, :]
            return array

        across_x = side in ("west", "east")
        low = side in ("west", "south")
        # A south or north slab is an x-side slab with the two horizontal
        # axes exchanged, v and u trading places, and msfv and msfu.
        normal = value("u" if across_x else "v")
        along = value("v" if across_x else "u")
        mu = value("mu")[0]
        phi = value("phi") if "phi" in fields else None
        mub = np.asarray(geo.mub, dtype=np.float64)
        phb = np.asarray(geo.phb, dtype=np.float64)
        msf_normal = geo.msfu if across_x else geo.msfv
        msf_along = geo.msfv if across_x else geo.msfu
        if not across_x:
            normal = np.swapaxes(normal, -1, -2)
            along = np.swapaxes(along, -1, -2)
            mu = mu.T
            phi = None if phi is None else np.swapaxes(phi, -1, -2)
            mub = mub.T
            phb = np.swapaxes(phb, -1, -2)
            msf_normal = None if msf_normal is None else np.asarray(
                msf_normal).T
            msf_along = None if msf_along is None else np.asarray(
                msf_along).T
        w = mu.shape[-1]
        columns = slice(0, w) if low else slice(mub.shape[-1] - w, None)
        faces = (slice(0, w) if low
                 else slice(mub.shape[-1] + 1 - w, None))
        total = mub[:, columns] + mu
        c1h = np.asarray(geo.c1h, dtype=np.float64)[:, None, None]
        c2h = np.asarray(geo.c2h, dtype=np.float64)[:, None, None]
        normal = normal / (c1h * _face_mass(total, low)[None] + c2h)
        along = along / (c1h * _row_mass(total)[None] + c2h)
        if msf_normal is not None:
            normal = normal * np.asarray(msf_normal,
                                         dtype=np.float64)[:, faces][None]
        if msf_along is not None:
            along = along * np.asarray(msf_along,
                                       dtype=np.float64)[:, columns][None]
        if phi is not None:
            c1f = np.asarray(geo.c1f, dtype=np.float64)[:, None, None]
            c2f = np.asarray(geo.c2f, dtype=np.float64)[:, None, None]
            weight = c1f * total[None] + c2f
            # The top full level carries no column mass (c1f = c2f = 0), so
            # its coupled perturbation is zero and says nothing: the base
            # geopotential stands there.
            php = np.divide(phi, weight, out=np.zeros_like(phi),
                            where=weight != 0.0)
        else:
            php = 0.0
        heights = mass_heights(phb[:, :, columns] + php)
        # Mass cells with both of their faces in the slab: all but the one
        # next to the interior.
        cells = slice(0, w - 1) if low else slice(1, w)
        speed = np.hypot(0.5 * (normal[..., :-1] + normal[..., 1:]),
                         0.5 * (along[..., :-1, cells]
                                + along[..., 1:, cells]))
        return _strongest(speed, heights[..., cells], crest_height)


def _boundary_when(seconds: float) -> str:
    if float(seconds) <= 0.0:
        return "boundary at the start"
    hours = float(seconds) / 3600.0
    text = f"{hours:.2f}".rstrip("0").rstrip(".")
    return f"boundary at +{text} h"


def strongest_crest_wind(label: str, crest_height: float,
                         sources: Iterable) -> CrestWind | None:
    """The strongest crest-band wind across a domain's input sources."""

    best = None
    for source in sources:
        found = source.strongest(crest_height)
        if found is None:
            continue
        if best is None or found[0] > best[0]:
            best = (found[0], found[1], found[2], source.source)
    if best is None:
        return None
    return CrestWind(label=label, crest_height_m=float(crest_height),
                     wind_m_s=best[0], when=best[2], source=best[3],
                     height_m=best[1])


def _cache_array(reader, key):
    if key not in reader.arrays:
        return None
    return np.asarray(reader.read_array(key))


def start_winds_from_cache(reader, source: str) -> StartWinds | None:
    """Start winds from a prepared cache (every prepared door writes one)."""

    if not all(key in reader.arrays for key in _START_KEYS.values()):
        return None
    return StartWinds(source,
                      lambda name: reader.read_array(_START_KEYS[name]))


def boundary_geometry_from_cache(reader, static) -> BoundaryGeometry | None:
    arrays = {name: _cache_array(reader, key) for name, key in (
        ("mub", "base/mub"), ("phb", "base/phb"), ("c1h", "coord/c1h"),
        ("c2h", "coord/c2h"), ("c1f", "coord/c1f"), ("c2f", "coord/c2f"))}
    if any(value is None for value in arrays.values()):
        return None
    return BoundaryGeometry(
        msfu=_static(static, "MAPFAC_U"), msfv=_static(static, "MAPFAC_V"),
        **arrays)


def _static(static, name):
    if static is None:
        return None
    getter = getattr(static, "get", None)
    value = getter(name) if getter is not None else None
    return None if value is None else np.asarray(value, dtype=np.float64)


#: The boundary fields the wind reading uncouples, of the ones a prepared
#: cache carries.
_BOUNDARY_FIELDS = ("u", "v", "mu", "phi")


def cache_boundaries(reader):
    """The root's boundary winds from its prepared cache, or ``None``.

    Only the fields the reading needs are read.  A cache whose inventory
    holds no complete u, v and mu tables for every interval (a nested
    child's, which carries none) gives ``None``: the reading then rests on
    the start state, and the restore that follows is what judges the
    cache.
    """

    from gpuwm.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries)
    from gpuwm.ingest.prepared_cache import _reader_boundary_side

    lbc = reader.header.get("metadata", {}).get("lbc")
    if not isinstance(lbc, dict) or not isinstance(lbc.get("intervals"),
                                                   list):
        return None
    sides = ("west", "east", "south", "north")
    intervals = []
    for index, row in enumerate(lbc["intervals"]):
        names = [name for name in _BOUNDARY_FIELDS
                 if name in row.get("fields", ())]
        present = [name for name in names if all(
            f"lbc/{index}/{name}/{side}/{part}" in reader.arrays
            for side in sides for part in ("value", "tendency"))]
        if not {"u", "v", "mu"} <= set(present):
            return None
        fields = {name: FieldBoundary(**{
            side: _reader_boundary_side(reader, f"lbc/{index}/{name}/{side}")
            for side in sides}) for name in present}
        intervals.append(BoundaryInterval(
            float(row["start_seconds"]), float(row["end_seconds"]), fields))
    if not intervals:
        return None
    return LateralBoundaries(
        tuple(intervals), int(lbc["spec_bdy_width"]),
        int(lbc.get("spec_zone", 1)), int(lbc.get("relax_zone", 4)))


def start_winds_from_wrfinput(restored, source: str) -> StartWinds | None:
    """Start winds from a wrfinput file's own arrays (U, V, PH, PHB)."""

    raw = getattr(restored, "raw", None)
    names = {"u": "U", "v": "V", "php": "PH", "phb": "PHB"}
    if raw is None or not set(names.values()) <= set(raw):
        return None
    return StartWinds(source,
                      lambda name: np.squeeze(np.asarray(raw[names[name]])))


def boundary_geometry_from_wrfinput(restored, static
                                    ) -> BoundaryGeometry | None:
    raw = getattr(restored, "raw", None)
    names = ("MUB", "PHB", "C1H", "C2H", "C1F", "C2F")
    if raw is None or not set(names) <= set(raw):
        return None

    def field(name):
        return np.squeeze(np.asarray(raw[name], dtype=np.float64))
    return BoundaryGeometry(
        mub=field("MUB"), phb=field("PHB"), c1h=field("C1H"),
        c2h=field("C2H"), c1f=field("C1F"), c2f=field("C2F"),
        msfu=_static(static, "MAPFAC_U"), msfv=_static(static, "MAPFAC_V"))


# ---------------------------------------------------------------------------
# The decision per domain, and the experiment it runs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockAdaptation:
    """One domain's reading and the step and substeps it runs."""

    grid_id: int
    label: str
    slope: float
    crest: CrestWind | None
    dx: float
    configured_dt: Fraction
    configured_sound: int
    division: int
    time_step_sound: int
    held_per_km: float | None
    reading: MapReading | None
    adaptive: bool = False
    ceiling: Fraction | None = None

    @property
    def dt(self) -> Fraction:
        return self.configured_dt / self.division

    @property
    def adapted(self) -> bool:
        return (self.division != 1
                or self.time_step_sound != self.configured_sound
                or self.ceiling is not None)

    @property
    def beyond_measured(self) -> bool:
        return self.reading is not None and (
            self.reading.per_km is None or bool(self.reading.beyond))

    @property
    def unheld(self) -> bool:
        """The map holds no step at this wind over this ground."""
        return self.reading is not None and self.reading.per_km is None

    @property
    def status(self) -> str:
        if self.crest is None:
            return "NO_WIND_READING"
        if self.unheld or (self.adapted and self.beyond_measured):
            return "BEYOND_MEASURED"
        return "ADAPTED" if self.adapted else "AS_CONFIGURED"

    def _what(self) -> str:
        crest = self.crest
        return (f"{self.label}'s steepest terrain slope is {self.slope:.2f} "
                f"and the strongest wind its start and boundary data carry "
                f"up to its {crest.crest_height_m:.0f} m crest is "
                f"{crest.wind_m_s:.0f} m/s ({crest.when})")

    def _held(self) -> str:
        if self.held_per_km is None:
            return "no measured step"
        return "steps up to " + _seconds_float(
            self.held_per_km * self.dx / 1000.0)

    def _runs(self) -> str:
        parts = []
        if self.division != 1:
            parts.append(f"{_seconds(self.dt)} steps instead of "
                         f"{_seconds(self.configured_dt)}")
        if self.time_step_sound != self.configured_sound:
            # On the adaptive clock the count is a floor under the one
            # the clock derives from its live step.
            least = "at least " if self.adaptive else ""
            parts.append(f"{least}{self.time_step_sound} acoustic substeps "
                         f"per step instead of {self.configured_sound}")
        if self.ceiling is not None:
            parts.append(f"an adaptive step capped at "
                         f"{_seconds(self.ceiling)}")
        return f"{self.label} runs " + " and ".join(parts)

    def sentence(self) -> str:
        """The plain line a run prints when this domain's clock changed."""

        return (f"time step: {self._what()}; the measured map holds "
                f"{self._held()} there with {self.time_step_sound} "
                f"substeps, so {self._runs()}")

    def beyond_sentence(self) -> str:
        """The line for a domain past the map's measured edge."""

        reading = self.reading
        if not self.adapted:
            return (f"time step: {self._what()}; the measured map holds no "
                    f"step there at any substep count, so {self.label} runs "
                    "as configured and may still stop")
        if reading.per_km is None:
            # Past the strongest wind the map holds a step at: the most
            # stable pair it measured, which is not a held step here.
            return (f"time step: {self._what()}; the measured map holds no "
                    f"step at this wind, and the most stable pair it "
                    f"measured holds {self._held()} with "
                    f"{self.time_step_sound} substeps at a weaker wind, so "
                    f"{self._runs()}, and may still stop")
        return (f"time step: {self._what()}; the measured map holds "
                f"{self._held()} there with {self.time_step_sound} "
                f"substeps but was not measured this far out in "
                f"{', '.join(reading.beyond)}, so {self._runs()}, and may "
                "still stop")

    def receipt(self) -> dict:
        row = {
            "grid_id": int(self.grid_id),
            "status": self.status,
            "steepest_slope": float(self.slope),
            "configured_dt_s": _rational(self.configured_dt),
            "dt_s": _rational(self.dt),
            "step_division": int(self.division),
            "configured_time_step_sound": int(self.configured_sound),
            "time_step_sound": int(self.time_step_sound),
            "held_s_per_km": self.held_per_km,
            "adaptive": bool(self.adaptive),
        }
        if self.ceiling is not None:
            row["max_time_step_s"] = _rational(self.ceiling)
        if self.adaptive and self.time_step_sound != self.configured_sound:
            row["min_time_step_sound"] = int(self.time_step_sound)
        if self.crest is not None:
            row.update(self.crest.receipt())
        if self.reading is not None:
            row["map_rows"] = {
                "dx_m": list(self.reading.dx_rows),
                "crest_m": self.reading.crest_row,
                "slope": self.reading.slope_row,
                "wind_m_s": self.reading.wind_row,
                "beyond": list(self.reading.beyond)}
        return row


def _seconds(value: Fraction) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return f"{value.numerator} s"
    return _seconds_float(float(value))


def _seconds_float(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return f"{text} s"


def _rational(value: Fraction) -> dict:
    value = Fraction(value)
    return {"seconds": float(value), "numerator": value.numerator,
            "denominator": value.denominator}


def _holds(per_km, configured_per_km, top) -> bool:
    if per_km is None:
        return False
    return configured_per_km <= per_km * (1.0 + 1e-9) or per_km >= top


def derive_clock(grid_id: int, run, dt: Fraction, slope: float,
                 crest: CrestWind | None, *, label: str | None = None,
                 table: StableStepMap | None = None) -> ClockAdaptation:
    """The step and substep count one domain runs.

    ``dt`` is the domain's configured step as an exact rational.  Without a
    wind reading the domain runs as configured.  Otherwise the configured
    pair stands when the map holds it; else the smallest whole division of
    the step the map holds, at the fewest substeps (the configured count or
    six) that hold it.  Under the adaptive clock the held step is a ceiling
    and the configured step, the clock's first, is divided only when it is
    above that ceiling; the count read is the one that clock takes at its
    shortest steps (:func:`gpuwm.core.adaptive_clock.least_sound_steps`),
    since it derives its count from the live step and gives four at every
    short one, and a larger count it needs becomes the floor under it.
    """

    from gpuwm.core.adaptive_clock import least_sound_steps

    table = measured_map() if table is None else table
    label = label or f"d{int(grid_id):02d}"
    dt = Fraction(dt)
    dx = float(run.dx)
    configured_sound = least_sound_steps(run)
    adaptive = bool(getattr(run, "use_adaptive_time_step", False))
    base = dict(grid_id=int(grid_id), label=label, slope=float(slope),
                crest=crest, dx=dx, configured_dt=dt,
                configured_sound=configured_sound, adaptive=adaptive)
    if crest is None:
        return ClockAdaptation(division=1, time_step_sound=configured_sound,
                               held_per_km=None, reading=None, **base)
    km = dx / 1000.0
    configured_per_km = float(dt) / km
    if adaptive:
        from gpuwm.core.adaptive_clock import wrf_default_clamps
        upper = _adaptive_upper(run)
        if upper is None:
            upper = Fraction(wrf_default_clamps(run.dx, run.dy)[1])
        configured_per_km = max(configured_per_km, float(upper) / km)
    counts = [configured_sound]
    if configured_sound < 6:
        counts.append(6)
    readings = {count: read_map(dx, crest.crest_height_m, slope,
                                crest.wind_m_s, count, table)
                for count in counts}
    own = readings[configured_sound]
    if _holds(own.per_km, configured_per_km, table.top) and not own.beyond:
        return ClockAdaptation(division=1, time_step_sound=configured_sound,
                               held_per_km=own.per_km, reading=own, **base)
    choices = []
    for count, reading in readings.items():
        if reading.per_km is None:
            continue
        if reading.per_km >= table.top:
            choices.append((1, count, reading, reading.per_km))
            continue
        need = float(dt) / km / reading.per_km
        division = max(1, math.ceil(need - 1e-9))
        choices.append((division, count, reading, reading.per_km))
    if choices:
        division, count, reading, held = min(
            choices, key=lambda item: (item[0], item[1]))
    else:
        # Past every stable measurement: the most stable pair measured.
        count = max(counts)
        reading = readings[count]
        held = reading.most_stable_per_km
        division = (1 if held is None else
                    max(1, math.ceil(float(dt) / km / held - 1e-9)))
    ceiling = None
    if adaptive and held is not None and held < table.top:
        # On the adaptive clock's hundredth-of-a-second lattice, rounded
        # down so the cap never sits above what the map held.
        ceiling = Fraction(math.floor(held * dx / 1000.0 * 100.0 + 1e-9),
                           100)
        upper = _adaptive_upper(run)
        if upper is not None and upper <= ceiling:
            ceiling = None
    return ClockAdaptation(division=int(division), time_step_sound=int(count),
                           held_per_km=held, reading=reading,
                           ceiling=ceiling, **base)


def _adaptive_upper(run):
    from gpuwm.config import _adaptive_interval

    return _adaptive_interval(run.max_time_step, run.max_time_step_den,
                              "max_time_step")


def _adaptive_lower(run):
    from gpuwm.config import _adaptive_interval

    return _adaptive_interval(run.min_time_step, run.min_time_step_den,
                              "min_time_step")


def _divisions(exp, wanted: Mapping[int, int]) -> dict[int, tuple[int, int]]:
    """Per domain ``(division, parent_time_step_ratio)``.

    A root's step is divided by what it needs.  A nest's division is the
    smallest one at least its own need whose ratio to its parent's
    division keeps its step ratio whole, so a parent cut deeply enough
    can leave its nest's step where it was.
    """

    result: dict[int, tuple[int, int]] = {}
    for dc in exp.domains:
        gid = int(dc.grid_id)
        need = max(1, int(wanted.get(gid, 1)))
        if int(dc.parent_id) == 0:
            result[gid] = (need, int(dc.parent_time_step_ratio))
            continue
        parent_division = result[int(dc.parent_id)][0]
        ratio = int(dc.parent_time_step_ratio)
        division = need
        while (ratio * division) % parent_division:
            division += 1
        result[gid] = (division, ratio * division // parent_division)
    return result


def retime_experiment(exp, divisions: Mapping[int, int],
                      sound: Mapping[int, int],
                      ceilings: Mapping[int, Fraction] | None = None):
    """The experiment with each domain's step divided and count raised.

    Returns ``(experiment, {grid_id: (division, ratio)})``.  The root's
    step is written as WRF writes it (whole seconds plus a fraction) and
    every ``run.dt`` is rebuilt as the chained single-precision value the
    clock validates.  On the adaptive clock a raised count is also written
    as ``min_time_step_sound``, or the clock, which derives its count from
    the live step, would put it back to four.  Domains with no change keep
    every field.
    """

    from gpuwm.acoustic_adaptation import adapted_run

    plan = _divisions(exp, divisions)
    ceilings = dict(ceilings or {})
    by_id = {}
    old_by_id = {int(dc.grid_id): np.float32(dc.run.dt)
                 for dc in exp.domains}
    domains = []
    for dc in exp.domains:
        gid = int(dc.grid_id)
        division, ratio = plan[gid]
        run = dc.run
        if int(dc.parent_id) == 0:
            dt = exp.dt_exact(gid) / division
            whole = dt.numerator // dt.denominator
            fraction = dt - whole
            fields = dict(time_step=int(whole),
                          time_step_fract_num=int(fraction.numerator),
                          time_step_fract_den=int(fraction.denominator))
            if division == 1:
                fields = {}
                dt32 = np.float32(run.dt)
            else:
                dt32 = (np.float32(whole) + np.float32(fraction.numerator)
                        / np.float32(fraction.denominator))
        else:
            fields = {}
            parent_id = int(dc.parent_id)
            if ratio != int(dc.parent_time_step_ratio):
                fields["parent_time_step_ratio"] = int(ratio)
            if (not fields and by_id[parent_id] == old_by_id[parent_id]):
                dt32 = np.float32(run.dt)
            else:
                dt32 = by_id[parent_id] / np.float32(ratio)
        run_fields = {}
        if float(dt32) != float(run.dt):
            run_fields["dt"] = float(dt32)
        counted = adapted_run(run, int(sound.get(gid, 0)))
        for name in ("time_step_sound", "min_time_step_sound"):
            if getattr(counted, name, None) != getattr(run, name, None):
                run_fields[name] = getattr(counted, name)
        ceiling = ceilings.get(gid)
        if ceiling is not None:
            ceiling = Fraction(ceiling)
            run_fields.update(_interval_fields("max_time_step", ceiling))
            lower = _adaptive_lower(run)
            if lower is None:
                from gpuwm.core.adaptive_clock import wrf_default_clamps
                lower = Fraction(wrf_default_clamps(run.dx, run.dy)[2])
            if lower > ceiling:
                run_fields.update(_interval_fields("min_time_step", ceiling))
            start = _interval_start(run)
            if start is not None and start > ceiling:
                run_fields.update(_interval_fields("starting_time_step",
                                                   ceiling))
        if run_fields:
            fields["run"] = replace(run, **run_fields)
        by_id[gid] = dt32
        domains.append(replace(dc, **fields) if fields else dc)
    if all(new is old for new, old in zip(domains, exp.domains)):
        return exp, plan
    return replace(exp, domains=tuple(domains)), plan


def _interval_start(run):
    from gpuwm.config import _adaptive_interval

    return _adaptive_interval(run.starting_time_step,
                              run.starting_time_step_den,
                              "starting_time_step")


def _interval_fields(name: str, value: Fraction) -> dict:
    value = Fraction(value)
    if value.denominator == 1:
        return {name: int(value.numerator), f"{name}_den": 0}
    return {name: int(value.numerator), f"{name}_den": int(value.denominator)}


def adapt_experiment_clock(
        exp, slopes: Mapping[int, float], winds: Mapping[int, CrestWind],
        *, announce: Callable[[str], None] | None = None,
        caution: Callable[[str], None] | None = None,
        table: StableStepMap | None = None):
    """Return ``(experiment, adaptations)`` with each read domain's clock.

    ``slopes`` maps ``grid_id`` to the steepest slope the substep rule read
    and ``winds`` to the domain's crest-level wind.  A domain missing
    either runs as configured.  ``announce`` receives one line per changed
    domain, ``caution`` the line for a changed domain past the map's edge.
    """

    adaptations = []
    for dc in exp.domains:
        gid = int(dc.grid_id)
        if gid not in slopes:
            continue
        wind = winds.get(gid)
        # Without a wind reading nothing changes, and the step is only
        # recorded, so an experiment without the rational clock still reads.
        dt = (exp.dt_exact(gid) if wind is not None else
              Fraction(float(getattr(dc.run, "dt", 0.0))))
        adaptations.append(derive_clock(
            gid, dc.run, dt, float(slopes[gid]), wind, table=table))
    changed = [a for a in adaptations if a.adapted]
    adapted, final = exp, list(adaptations)
    if changed:
        adapted, plan = retime_experiment(
            exp, {a.grid_id: a.division for a in changed},
            {a.grid_id: a.time_step_sound for a in changed},
            {a.grid_id: a.ceiling for a in changed if a.ceiling is not None})
        final = []
        for adaptation in adaptations:
            division, _ratio = plan[adaptation.grid_id]
            if division != adaptation.division:
                adaptation = replace(adaptation, division=int(division))
            final.append(adaptation)
    # One line per changed domain.  A domain the map holds can still take a
    # finer step from its parent's division, and says so in its own words;
    # one over ground and wind the map holds no step for says so even when
    # its clock stays as configured.
    changed_ids = {a.grid_id for a in changed}
    for adaptation in final:
        if adaptation.grid_id in changed_ids or adaptation.unheld:
            if adaptation.beyond_measured:
                if caution is not None:
                    caution(adaptation.beyond_sentence())
            elif announce is not None:
                announce(adaptation.sentence())
        elif adaptation.division != 1 and announce is not None:
            announce(
                f"time step: {adaptation.label} runs "
                f"{_seconds(adaptation.dt)} steps instead of "
                f"{_seconds(adaptation.configured_dt)} because its parent's "
                "step was divided and a nest's step divides its parent's")
    return adapted, tuple(final)


#: The mechanism, printed under ``--explain``.
_WHY = (
    "Under a strong wind at crest height, flow forced over steep ground "
    "moves farther per long step than the split-explicit step can hold, "
    "and more acoustic substeps do not help once the long step itself is "
    "too long.  The engine's measured map (gpuwm/terrain_clock.py and "
    "gpuwm/terrain_clock_map.json) gives, for each grid spacing, crest "
    "height, slope, crest-level wind and substep count, the longest step "
    "that held; the run takes the smallest whole division of its step "
    "that the map holds.")


def adapt_experiment_clock_to_terrain(exp, slopes, winds):
    """:func:`adapt_experiment_clock` in the run's own voice."""

    from gpuwm.explain import warn

    return adapt_experiment_clock(
        exp, slopes, winds,
        announce=lambda sentence: warn(sentence, _WHY),
        caution=lambda sentence: warn(sentence, _WHY))


def clock_receipt(adaptations) -> dict:
    """The derivation for the run document: every domain read, changed or
    not."""

    table = measured_map()
    return {
        "schema": TERRAIN_CLOCK_SCHEMA,
        "domains": [adaptation.receipt() for adaptation in adaptations],
        "map": {"path": MAP_PATH.name, "winds_m_s": list(table.winds),
                "ladder_s_per_km": list(table.ladder),
                "seconds": table.seconds, "rows": len(table.rows)},
    }


def crest_heights(terrain_by_grid_id: Mapping[int, object]
                  ) -> dict[int, float]:
    """Each domain's crest height: the highest ground it can touch."""

    return {int(gid): float(np.nanmax(np.asarray(terrain, dtype=np.float64)))
            for gid, terrain in terrain_by_grid_id.items()}


def tree_crest_winds(exp, crests: Mapping[int, float],
                     sources_by_grid_id: Mapping[int, Sequence],
                     root_boundary_sources: Sequence = ()
                     ) -> dict[int, CrestWind]:
    """Each domain's crest-level wind across its own and its ancestors'
    inputs.

    A nest's boundary data is its parent's forecast, which no input holds
    before the run, so a nest is read over its own start state, every
    ancestor's start state and the root's boundary data, each at the
    nest's own crest height.
    """

    by_id = {int(dc.grid_id): dc for dc in exp.domains}
    winds = {}
    for dc in exp.domains:
        gid = int(dc.grid_id)
        if gid not in crests:
            continue
        sources = list(sources_by_grid_id.get(gid, ()))
        parent = int(getattr(dc, "parent_id", 0))
        while parent != 0:
            sources.extend(sources_by_grid_id.get(parent, ()))
            parent = int(getattr(by_id[parent], "parent_id", 0))
        sources.extend(root_boundary_sources)
        wind = strongest_crest_wind(f"d{gid:02d}", crests[gid], sources)
        if wind is not None:
            winds[gid] = wind
    return winds


# ---------------------------------------------------------------------------
# The doors.
# ---------------------------------------------------------------------------


def slopes_from_acoustics(acoustic) -> dict[int, float]:
    """The steepest slope each domain's substep derivation read."""

    return {int(a.grid_id): float(a.reading.slope) for a in acoustic}


def clock_for_domains(exp, acoustic, *, statics: Mapping[int, object],
                      starts: Mapping[int, object],
                      boundary: BoundaryWinds | None = None,
                      corridors: Mapping[int, object] | None = None,
                      announce: bool = True):
    """Every prepared door's derivation: ``(experiment, adaptations)``.

    ``statics`` maps ``grid_id`` to the static fields each domain runs on
    (``HGT_M`` gives its crest), ``starts`` to its :class:`StartWinds`,
    ``boundary`` is the root's :class:`BoundaryWinds` and ``corridors``
    maps a following nest to the terrain of the corridor it can reach.
    """

    slopes = slopes_from_acoustics(acoustic)
    terrain = {}
    for gid, static in statics.items():
        field = _static(static, "HGT_M")
        if field is not None:
            terrain[int(gid)] = field
    crests = crest_heights(terrain)
    for gid, reach in (corridors or {}).items():
        if reach is None:
            continue
        top = float(np.nanmax(np.asarray(reach, dtype=np.float64)))
        crests[int(gid)] = max(crests.get(int(gid), top), top)
    sources = {int(gid): [start] for gid, start in starts.items()
               if start is not None}
    winds = tree_crest_winds(exp, crests, sources,
                             [] if boundary is None else [boundary])
    if announce:
        return adapt_experiment_clock_to_terrain(exp, slopes, winds)
    return adapt_experiment_clock(exp, slopes, winds)


def clock_for_prepared_cache(exp, acoustic, *, readers, statics,
                             boundaries=None, corridors=None):
    """The derivation on doors whose domains carry prepared caches.

    ``readers`` maps ``grid_id`` to each domain's
    :class:`gpuwm.ingest.prepared_cache.PreparedCacheReader`; the root's
    boundary data is ``boundaries`` when the door holds it apart (met_em),
    else the one its own cache carries.
    """

    root = int(exp.domains[0].grid_id)
    starts = {int(gid): start_winds_from_cache(reader, f"d{int(gid):02d}")
              for gid, reader in readers.items()}
    reader = readers.get(root)
    if boundaries is None and reader is not None:
        boundaries = cache_boundaries(reader)
    boundary = None
    if boundaries is not None and reader is not None:
        geometry = boundary_geometry_from_cache(reader, statics.get(root))
        if geometry is not None:
            boundary = BoundaryWinds(f"d{root:02d}", boundaries, geometry,
                                     float(exp.run_seconds))
    return clock_for_domains(exp, acoustic, statics=statics, starts=starts,
                             boundary=boundary, corridors=corridors)


def clock_for_wrfinput(exp, acoustic, *, restored, statics, boundaries,
                       corridors=None):
    """The derivation on the wrfinput door, from the files' own arrays."""

    root = int(exp.domains[0].grid_id)
    starts = {int(gid): start_winds_from_wrfinput(item, f"d{int(gid):02d}")
              for gid, item in restored.items()}
    boundary = None
    if boundaries is not None and root in restored:
        geometry = boundary_geometry_from_wrfinput(restored[root],
                                                   statics.get(root))
        if geometry is not None:
            boundary = BoundaryWinds(f"d{root:02d}", boundaries, geometry,
                                     float(exp.run_seconds))
    return clock_for_domains(exp, acoustic, statics=statics, starts=starts,
                             boundary=boundary, corridors=corridors)


@dataclass(frozen=True)
class SnapshotWinds:
    """Decoded forcing on its own grid, over one domain's footprint.

    The ``gpuwm run`` door reads the forcing before it prepares anything,
    because the prepared state already carries the step its physics was
    built for.  The footprint is every source column inside the domain's
    extent, one source spacing wider.
    """

    source: str
    snapshots: tuple
    latitude: np.ndarray
    longitude: np.ndarray
    start_time: object

    def strongest(self, crest_height):
        from gpuwm.ingest.horiz import source_coordinate_transform

        best = None
        for snapshot in self.snapshots:
            fields = snapshot.fields
            if not {"UU", "VV", "GHT"} <= set(fields):
                continue
            transform, projected = source_coordinate_transform(snapshot)
            y, x = transform(np.asarray(self.latitude, dtype=np.float64),
                             np.asarray(self.longitude, dtype=np.float64))
            y = np.asarray(y, dtype=np.float64)
            x = np.asarray(x, dtype=np.float64)
            axis_y = np.asarray(snapshot.latitude, dtype=np.float64)
            axis_x = np.asarray(snapshot.longitude, dtype=np.float64)
            rows = _axis_window(axis_y, y)
            cols = (_axis_window(axis_x, x) if projected
                    else _longitude_window(axis_x, x))
            if rows.size == 0 or cols.size == 0:
                continue
            order = np.argsort(-np.asarray(snapshot.levels_hpa,
                                           dtype=np.float64))

            def cut(name):
                field = np.asarray(fields[name], dtype=np.float64)
                return field[order][:, rows][:, :, cols]
            speed = np.hypot(cut("UU"), cut("VV"))
            heights = cut("GHT")
            ground = fields.get("SOILHGT")
            if ground is not None:
                ground = np.asarray(ground, dtype=np.float64)[rows][:, cols]
                speed = np.where(heights >= ground[None], speed, np.nan)
            found = _strongest(speed, heights, crest_height)
            if found is None:
                continue
            seconds = (snapshot.valid_time - self.start_time).total_seconds()
            when = "start" if seconds <= 0 else _boundary_when(seconds)
            if best is None or found[0] > best[0]:
                best = (found[0], found[1], when)
        return best


def _axis_window(axis, values):
    """Indices of a 1-D source axis inside the values' extent, widened by
    one source spacing on each side."""

    axis = np.asarray(axis, dtype=np.float64)
    if axis.size < 2:
        return np.arange(axis.size)
    step = float(np.nanmax(np.abs(np.diff(axis))))
    low = float(np.nanmin(values)) - step
    high = float(np.nanmax(values)) + step
    return np.flatnonzero((axis >= low) & (axis <= high))


def _longitude_window(axis, values):
    """Indices of a longitude axis inside the values' extent, widened by one
    source spacing on each side, across the seam wherever it falls.

    Longitude wraps, so the extent is the shortest arc holding every value:
    the circle minus its widest gap between neighbouring values.  A window
    from the smallest value to the largest instead reads every column of a
    domain that straddles the source's seam (0 degrees on a 0 to 360
    source, the dateline on a -180 to 180 one), and picks up winds half a
    world away.  A domain clear of the seam gets the window it always had.
    """

    axis = np.asarray(axis, dtype=np.float64)
    if axis.size < 2:
        return np.arange(axis.size)
    values = np.asarray(values, dtype=np.float64)
    values = np.unique(np.mod(values[np.isfinite(values)], 360.0))
    if values.size == 0:
        return np.arange(0)
    gaps = np.diff(np.append(values, values[0] + 360.0))
    widest = int(np.argmax(gaps))
    start = float(values[(widest + 1) % values.size])
    span = 360.0 - float(gaps[widest])
    spacing = np.abs(np.diff(axis))
    step = float(np.nanmax(np.minimum(spacing, 360.0 - spacing)))
    offset = np.mod(np.mod(axis, 360.0) - (start - step), 360.0)
    # A column exactly on the window's low edge can come back from the two
    # wraps a rounding short of 360 rather than at 0; it is on the edge.
    offset = np.where(offset > 360.0 - _WRAP_ROUNDING, offset - 360.0, offset)
    return np.flatnonzero(offset <= span + 2.0 * step + _WRAP_ROUNDING)


#: Degrees of rounding the wraps above may leave on a window edge.
_WRAP_ROUNDING = 1.0e-9


def forcing_window(snapshots: Mapping, start_time, run_seconds: float):
    """The snapshots a run's window reads: every one from the start to the
    end, and the nearest outside either end when that end falls between
    two."""

    from datetime import timedelta

    end = start_time + timedelta(seconds=float(run_seconds))
    times = sorted(snapshots)
    chosen = [t for t in times if start_time <= t <= end]
    later = [t for t in times if t > end]
    if later and (not chosen or chosen[-1] < end):
        chosen.append(later[0])
    earlier = [t for t in times if t < start_time]
    if earlier and (not chosen or chosen[0] > start_time):
        chosen.insert(0, earlier[-1])
    return tuple(snapshots[t] for t in chosen)
