"""Conventional observations for the regional LETKF: aircraft, sondes,
motion vectors and mesonet, as table rows.

What arrives
------------
Neutral observation tables (``gpuwm-obs.table.v2``, or the ten-column v1)
written by the Rust front doors in ``tools/rustwx/crates/rw-obs``:
``rw_madis`` (MADIS aircraft, sondes and mesonet), ``rw_amv`` (GOES derived
motion winds), ``rw_igra2`` (sondes), ``rw_prepbufr`` (NCEP prepbufr).  The
doors own every decode, unit conversion, platform QC flag and observation
error; nothing here parses a source format.

What decides what
-----------------
An observation TYPE is a row of a type table
(``gpuwm/data/da/conventional-obs-types.v1.json`` by default, or any file
of the same schema): a source pattern, a variable and a measurement select
the rows it governs, and the type names its operator, its batch, its error
inflation and floor, its background-check threshold and its localization.
The operator is chosen by what the number IS, not by who measured it:

``pressure_level``
    A point at its own pressure (``level_pa``): temperature, wind
    components or dewpoint, interpolated bilinearly in the horizontal
    and linearly in log pressure inside EACH member's own column (full
    pressure ``p``, temperature ``(thb + thp)(p/P0)^(Rd/cp)``, winds
    destaggered and rotated to earth-relative with the grid's own
    SINALPHA/COSALPHA, dewpoint from the member's vapour mixing ratio at
    the observation's pressure).  No vertical extrapolation: a report
    above the top or below the lowest mass level is counted and dropped.
``surface``
    A station report against each member's leg-end 2 m / 10 m
    diagnostics (``t2``, ``q2``/``psfc`` for dewpoint, ``u10``/``v10``
    rotated to earth-relative), bilinear at the station, refused when the
    station elevation and the model terrain differ by more than the
    table's ``surface_elevation_max_diff_m``.

A new platform is therefore a new table row, never a new code path.

Placement and superobbing
-------------------------
The filter's batches are gridded (one value per gridpoint per batch, the
:class:`gpuwm.da.letkf.GriddedObs` contract), so each report is placed on
the nearest mass point horizontally and on the model level nearest its
pressure in log pressure (ensemble-mean column), level 0 for surface
reports.  Reports of one batch landing on one gridpoint are averaged into
one superobservation (value, H(x) per member, error: the mean of the
reports'), which is how a hub airport's few hundred climbs and descents
in an hour enter as a profile rather than as one report.

Time
----
A report is assimilated by the analysis at ``t`` when its valid time is
in ``(t - window_before, t + window_after]``; with hourly analyses and the
default thirty minutes either side every report has exactly one home.
The operator is 3-D: the members' state at ``t`` stands for the window.

Every screen is counted per type in the provenance; nothing is dropped in
silence.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gpuwm.da.letkf import Localization, PointSet, point_batch

SCHEMA = "gpuwm-da.conventional-observations.v1"
TYPE_TABLE_SCHEMA = "gpuwm-da.conventional-obs-types.v1"
DEFAULT_TYPE_TABLE = (Path(__file__).resolve().parents[1] / "data" / "da"
                      / "conventional-obs-types.v1.json")
TABLE_SCHEMAS = ("gpuwm-obs.table.v2", "gpuwm-obs.table.v1")
TABLE_HEADER_V1 = ("source", "station_id", "latitude_deg", "longitude_deg",
                   "elevation_m", "level_pa", "valid_time", "variable",
                   "value", "error")
OPERATORS = ("pressure_level", "surface")
#: The variables each operator can simulate.
OPERATOR_VARIABLES = {
    "pressure_level": ("temperature_k", "wind_u_m_s", "wind_v_m_s",
                       "dewpoint_k"),
    "surface": ("temperature_k", "wind_u_m_s", "wind_v_m_s", "dewpoint_k"),
}
#: The member surface diagnostics each surface variable reads.
SURFACE_DIAGNOSTICS = {
    "temperature_k": ("t2",),
    "dewpoint_k": ("q2", "psfc"),
    "wind_u_m_s": ("u10", "v10"),
    "wind_v_m_s": ("u10", "v10"),
}

_RD, _CP, _P0 = 287.0, 7.0 * 287.0 / 2.0, 1.0e5
_EPS = 0.622


class ConventionalObsError(ValueError):
    """A refusal by this module; the message names what to change."""


# ---------------------------------------------------------------------------
# the type table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObsType:
    name: str
    source: str
    variable: str
    measurement: str
    operator: str
    batch: str
    error_inflation: float
    error_floor: float
    gross_sigma: float
    horizontal_m: float
    vertical_m: float
    use: bool = True
    #: Reports at pressures below this (higher up) are not used; None
    #: keeps every level.  Humidity aloft is where sensors fail.
    level_min_pa: float | None = None

    def matches(self, source: str, variable: str, measurement: str) -> bool:
        return (variable == self.variable
                and (self.measurement == "*" or measurement == self.measurement)
                and fnmatch.fnmatchcase(source, self.source))

    def localization(self) -> Localization:
        return Localization(horizontal_m=float(self.horizontal_m),
                            vertical_m=float(self.vertical_m))


@dataclass(frozen=True)
class TypeTable:
    types: tuple[ObsType, ...]
    window_before_seconds: float
    window_after_seconds: float
    surface_elevation_max_diff_m: float
    path: str
    sha256: str

    def type_of(self, source: str, variable: str, measurement: str):
        for row in self.types:
            if row.matches(source, variable, measurement):
                return row
        return None

    def batch_localizations(self) -> tuple[Localization, ...]:
        seen = {}
        for row in self.types:
            if row.use:
                seen.setdefault(row.batch, row.localization())
        return tuple(seen.values())

    def needs_surface(self) -> bool:
        return any(row.use and row.operator == "surface" for row in self.types)


def read_type_table(path: str | Path | None = None, *,
                    overrides: Mapping[str, Mapping[str, object]] | None = None
                    ) -> TypeTable:
    """Read a type table; ``overrides`` maps a batch name (or ``"*"``) to
    fields replaced on every row of that batch (a tuning harness's knob)."""
    import hashlib

    path = Path(path) if path is not None else DEFAULT_TYPE_TABLE
    raw = path.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("schema") != TYPE_TABLE_SCHEMA:
        raise ConventionalObsError(
            f"{path} is not a {TYPE_TABLE_SCHEMA} table "
            f"(schema {doc.get('schema')!r})")
    defaults = doc.get("defaults") or {}
    fields = ObsType.__dataclass_fields__
    rows = []
    for entry in doc.get("types") or ():
        entry = dict(entry)
        for key in ("*", entry.get("batch")):
            if overrides and key in overrides:
                entry.update(overrides[key])
        unknown = sorted(set(entry) - set(fields))
        if unknown:
            raise ConventionalObsError(
                f"{path}: type {entry.get('name')!r} has unknown fields "
                f"{unknown}")
        row = ObsType(**entry)
        if row.operator not in OPERATORS:
            raise ConventionalObsError(
                f"{path}: type {row.name!r} names operator {row.operator!r}; "
                f"this route has {OPERATORS}")
        if row.variable not in OPERATOR_VARIABLES[row.operator]:
            raise ConventionalObsError(
                f"{path}: type {row.name!r}: the {row.operator} operator "
                f"simulates {OPERATOR_VARIABLES[row.operator]}, not "
                f"{row.variable!r}")
        for name in ("error_inflation", "error_floor", "gross_sigma",
                     "horizontal_m", "vertical_m"):
            value = float(getattr(row, name))
            if not math.isfinite(value) or value <= 0:
                raise ConventionalObsError(
                    f"{path}: type {row.name!r} has {name} = {value!r}; it "
                    "must be finite and positive")
        rows.append(row)
    by_batch = {}
    for row in rows:
        if not row.use:
            continue
        first = by_batch.setdefault(row.batch, row)
        if (first.horizontal_m, first.vertical_m) != (row.horizontal_m,
                                                      row.vertical_m):
            raise ConventionalObsError(
                f"{path}: batch {row.batch!r} carries two localizations "
                f"({first.name}: {first.horizontal_m}/{first.vertical_m} m, "
                f"{row.name}: {row.horizontal_m}/{row.vertical_m} m); a "
                "batch is solved with one")
        if first.variable != row.variable:
            raise ConventionalObsError(
                f"{path}: batch {row.batch!r} mixes {first.variable} and "
                f"{row.variable}; one batch holds one quantity")
        if first.operator != row.operator:
            raise ConventionalObsError(
                f"{path}: batch {row.batch!r} mixes the {first.operator} and "
                f"{row.operator} operators; their levels differ")
    return TypeTable(
        types=tuple(rows),
        window_before_seconds=float(defaults.get("window_before_seconds", 1800)),
        window_after_seconds=float(defaults.get("window_after_seconds", 1800)),
        surface_elevation_max_diff_m=float(
            defaults.get("surface_elevation_max_diff_m", 200)),
        path=str(path), sha256=hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------------------------
# the neutral table rows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rows:
    """Columnar rows of one or more neutral tables."""

    source: np.ndarray
    station_id: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    elevation_m: np.ndarray
    level_pa: np.ndarray        # NaN for a surface row
    valid_seconds: np.ndarray   # seconds since 1970-01-01T00:00Z
    variable: np.ndarray
    value: np.ndarray
    error: np.ndarray
    measurement: np.ndarray
    receipts: tuple

    def __len__(self) -> int:
        return int(self.value.size)


def _epoch_seconds(text: str) -> float:
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    when = datetime.fromisoformat(text)
    if when.tzinfo is None:
        raise ConventionalObsError(
            f"table time {text!r} has no UTC offset; the doors write Z")
    return when.astimezone(timezone.utc).timestamp()


def read_tables(paths: Sequence[str | Path]) -> Rows:
    """Read neutral tables into columns.  The doors wrote and verified the
    bytes; this reads the columns the analysis needs and counts the rest."""
    import hashlib

    cols = defaultdict(list)
    receipts = []
    for path in paths:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if header is None or tuple(header[:10]) != TABLE_HEADER_V1:
            raise ConventionalObsError(
                f"{path} does not start with the neutral table header "
                f"{','.join(TABLE_HEADER_V1)}")
        position = {name: i for i, name in enumerate(header)}
        counts = Counter()
        time_cache: dict[str, float] = {}
        for record in reader:
            if not record:
                continue
            counts["rows"] += 1
            try:
                level = record[5].strip()
                stamp = record[6]
                seconds = time_cache.get(stamp)
                if seconds is None:
                    seconds = time_cache[stamp] = _epoch_seconds(stamp)
                cols["source"].append(record[0])
                cols["station_id"].append(record[1])
                cols["lat"].append(float(record[2]))
                cols["lon"].append(float(record[3]))
                cols["elevation_m"].append(
                    float(record[4]) if record[4].strip() else math.nan)
                cols["level_pa"].append(float(level) if level else math.nan)
                cols["valid_seconds"].append(seconds)
                cols["variable"].append(record[7])
                cols["value"].append(float(record[8]))
                cols["error"].append(float(record[9]))
                cols["measurement"].append(
                    record[position["measurement"]]
                    if "measurement" in position
                    and len(record) > position["measurement"] else "")
            except (ValueError, IndexError):
                counts["rows_unreadable"] += 1
                continue
        receipts.append({"path": str(path),
                         "sha256": hashlib.sha256(raw).hexdigest(),
                         "bytes": len(raw), "counts": dict(counts)})
    strings = ("source", "station_id", "variable", "measurement")
    return Rows(
        **{name: np.asarray(cols[name], dtype=object if name in strings
                            else np.float64)
           for name in ("source", "station_id", "lat", "lon", "elevation_m",
                        "level_pa", "valid_seconds", "variable", "value",
                        "error", "measurement")},
        receipts=tuple(receipts))


# ---------------------------------------------------------------------------
# member columns
# ---------------------------------------------------------------------------

def _dewpoint_k(vapour_pressure_pa):
    """Dewpoint over liquid water (Bolton 1980 inverse), K."""
    e_hpa = np.maximum(np.asarray(vapour_pressure_pa, np.float64) / 100.0,
                       1e-10)
    ln = np.log(e_hpa / 6.112)
    return 243.5 * ln / (17.67 - ln) + 273.15


def _vapour_pressure_pa(mixing_ratio, pressure_pa):
    w = np.maximum(np.asarray(mixing_ratio, np.float64), 0.0)
    return w * np.asarray(pressure_pa, np.float64) / (_EPS + w)


def _corner_weights(fi, fj, nx, ny):
    """Bilinear corners (j, i) and weights for fractional mass indices."""
    i0 = np.clip(np.floor(fi).astype(np.int64), 0, nx - 2)
    j0 = np.clip(np.floor(fj).astype(np.int64), 0, ny - 2)
    ai = np.clip(fi - i0, 0.0, 1.0)
    aj = np.clip(fj - j0, 0.0, 1.0)
    corners = ((j0, i0, (1 - aj) * (1 - ai)), (j0, i0 + 1, (1 - aj) * ai),
               (j0 + 1, i0, aj * (1 - ai)), (j0 + 1, i0 + 1, aj * ai))
    return corners


def _member_columns(state, thb, js, is_, rotation, need):
    """Columns ``(nz, n)`` at mass points ``(js, is_)`` of one member."""
    out = {}
    p = np.asarray(state["p"])[:, js, is_].astype(np.float64)
    out["p"] = p
    if "t" in need:
        if thb is not None:
            thp = np.asarray(state["thp"])[:, js, is_].astype(np.float64)
            base = np.asarray(thb, np.float64)
            base = base[:, None] if base.ndim == 1 else base[:, js, is_]
            out["t"] = (base + thp) * (p / _P0) ** (_RD / _CP)
        else:
            # Without the setup base state: the equation of state the
            # model diagnoses p and alt with, inverted (the form the
            # increment applier's saturation cap uses; it agrees with
            # (thb + thp) Pi to about 1e-4 K).
            alt = np.asarray(state["alt"])[:, js, is_].astype(np.float64)
            qv = np.asarray(state["qv"])[:, js, is_].astype(np.float64)
            out["t"] = p * alt / (_RD * (1.0 + (461.6 / _RD) * qv))
    if "td" in need:
        qv = np.asarray(state["qv"])[:, js, is_].astype(np.float64)
        out["td"] = _dewpoint_k(_vapour_pressure_pa(qv, p))
    if "u" in need or "v" in need:
        u = np.asarray(state["u"])
        v = np.asarray(state["v"])
        um = 0.5 * (u[:, js, is_].astype(np.float64)
                    + u[:, js, is_ + 1].astype(np.float64))
        vm = 0.5 * (v[:, js, is_].astype(np.float64)
                    + v[:, js + 1, is_].astype(np.float64))
        if rotation is not None:
            sina = np.asarray(rotation[0])[js, is_]
            cosa = np.asarray(rotation[1])[js, is_]
            um, vm = um * cosa - vm * sina, um * sina + vm * cosa
        out["u"], out["v"] = um, vm
    return out


def _interp_logp(p_col, x_col, target_pa):
    """Linear in log p, per column (``(nz, n)``, p falling with k); NaN
    outside the column's mass levels (no extrapolation)."""
    logp = np.log(p_col)
    lt = np.log(target_pa)
    # first level whose pressure is at or below the target
    below = logp <= lt[None, :]
    has = below.any(axis=0)
    k1 = np.where(has, np.argmax(below, axis=0), 0)
    k0 = np.maximum(k1 - 1, 0)
    n = np.arange(p_col.shape[1])
    l0, l1 = logp[k0, n], logp[k1, n]
    x0, x1 = x_col[k0, n], x_col[k1, n]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(l1 != l0, (lt - l0) / (l1 - l0), 0.0)
    out = x0 + w * (x1 - x0)
    inside = has & (k1 > 0) & (lt <= logp[0, n])
    exact_bottom = has & (k1 == 0) & np.isclose(lt, logp[0, n])
    return np.where(inside | exact_bottom, out, np.nan)


# ---------------------------------------------------------------------------
# the batches
# ---------------------------------------------------------------------------

def conventional_batches(rows: Rows, table: TypeTable, *, states, thb,
                         grid, analysis_time: datetime, surface=None,
                         rotation=None, shape=None):
    """``(batches, provenance)`` for one analysis.

    ``states`` is the member sequence in checkpoint order, each a mapping
    with ``p``, ``thp``, ``qv``, ``u``, ``v`` (a
    :class:`gpuwm.da.radar_assimilation.CheckpointStateView` or the
    harness's saved arrays).  ``thb`` is the base potential temperature,
    or None to take temperature from the equation of state (``p``,
    ``alt``, ``qv``).
    ``grid`` is the analysis :class:`gpuwm.obs.target_grid.TargetGrid`.
    ``surface`` maps ``t2``/``q2``/``psfc``/``u10``/``v10`` to
    ``(members, ny, nx)`` leg-end diagnostics (needed only when a surface
    type is in use).  ``rotation`` is ``(sinalpha, cosalpha)`` at mass
    points.
    """
    if analysis_time.tzinfo is None:
        # The experiment clock is UTC without an offset (WRF's calendar);
        # the surface adapter reads it the same way.
        analysis_time = analysis_time.replace(tzinfo=timezone.utc)
    members = len(states)
    if members < 2:
        raise ConventionalObsError("an ensemble analysis needs two members")
    nz, ny, nx = (int(grid.nz), int(grid.ny), int(grid.nx)) if shape is None \
        else tuple(int(v) for v in shape)
    t = analysis_time.astimezone(timezone.utc).timestamp()
    lo, hi = t - table.window_before_seconds, t + table.window_after_seconds
    counts: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()

    # -- type assignment ------------------------------------------------------
    n_rows = len(rows)
    type_index = np.full(n_rows, -1, np.int64)
    cache = {}
    for r in range(n_rows):
        key = (rows.source[r], rows.variable[r], rows.measurement[r])
        if key not in cache:
            row_type = table.type_of(*key)
            cache[key] = -1 if row_type is None else table.types.index(row_type)
        type_index[r] = cache[key]
    totals["rows_read"] = n_rows
    totals["rows_without_type"] = int(np.count_nonzero(type_index < 0))
    unmatched = Counter(k for k, v in cache.items() if v < 0)
    in_window = (rows.valid_seconds > lo) & (rows.valid_seconds <= hi)
    totals["rows_outside_window"] = int(np.count_nonzero(
        (type_index >= 0) & ~in_window))

    used_types = [i for i, row in enumerate(table.types) if row.use]
    for index in range(len(table.types)):
        if not table.types[index].use:
            totals["rows_of_unused_types"] += int(np.count_nonzero(
                (type_index == index) & in_window))

    # -- per-type QC on the row itself ----------------------------------------
    selected = []
    fi_all, fj_all = grid.mass_index(rows.lat, rows.lon) if n_rows else (
        np.zeros(0), np.zeros(0))
    fi_all = np.asarray(fi_all, np.float64)
    fj_all = np.asarray(fj_all, np.float64)
    for index in used_types:
        row_type = table.types[index]
        c = counts[row_type.name]
        take = np.nonzero((type_index == index) & in_window)[0]
        c["in_window"] = int(take.size)
        if not take.size:
            continue
        good = np.isfinite(rows.value[take]) & np.isfinite(rows.error[take]) \
            & (rows.error[take] > 0)
        c["invalid_value_or_error"] = int(np.count_nonzero(~good))
        take = take[good]
        fi, fj = fi_all[take], fj_all[take]
        inside = (np.isfinite(fi) & np.isfinite(fj) & (fi >= 0) & (fj >= 0)
                  & (fi <= nx - 1) & (fj <= ny - 1))
        c["outside_domain"] = int(np.count_nonzero(~inside))
        take = take[inside]
        if row_type.operator == "pressure_level":
            p = rows.level_pa[take]
            ok = np.isfinite(p) & (p > 0)
            c["no_pressure"] = int(np.count_nonzero(~ok))
            take = take[ok]
            if row_type.level_min_pa is not None:
                ok = rows.level_pa[take] >= float(row_type.level_min_pa)
                c["above_level_min"] = int(np.count_nonzero(~ok))
                take = take[ok]
        else:
            ok = np.isfinite(rows.elevation_m[take])
            c["no_station_elevation"] = int(np.count_nonzero(~ok))
            take = take[ok]
            fi, fj = fi_all[take], fj_all[take]
            terrain = np.zeros(take.size)
            for jj, ii, w in _corner_weights(fi, fj, nx, ny):
                terrain += w * np.asarray(grid.terrain_m)[jj, ii]
            ok = np.abs(rows.elevation_m[take] - terrain) \
                <= table.surface_elevation_max_diff_m
            c["surface_elevation_mismatch"] = int(np.count_nonzero(~ok))
            take = take[ok]
        if take.size:
            selected.append((index, take))

    # -- H(x) -----------------------------------------------------------------
    need_columns = any(table.types[i].operator == "pressure_level"
                       for i, _ in selected)
    pl = [(i, take) for i, take in selected
          if table.types[i].operator == "pressure_level"]
    sf = [(i, take) for i, take in selected
          if table.types[i].operator == "surface"]
    hx = {}
    if need_columns:
        all_take = np.concatenate([take for _, take in pl])
        fi, fj = fi_all[all_take], fj_all[all_take]
        corners = _corner_weights(fi, fj, nx, ny)
        # unique corner columns, read once per member
        cells = np.unique(np.concatenate(
            [jj * nx + ii for jj, ii, _ in corners]))
        cj, ci = cells // nx, cells % nx
        slot = {int(c): s for s, c in enumerate(cells)}
        corner_slots = [np.fromiter((slot[int(v)] for v in jj * nx + ii),
                                    np.int64, count=all_take.size)
                        for jj, ii, _ in corners]
        var_of = np.empty(all_take.size, dtype=object)
        offset = 0
        for i, take in pl:
            var_of[offset:offset + take.size] = table.types[i].variable
            offset += take.size
        key_of = {"temperature_k": "t", "wind_u_m_s": "u", "wind_v_m_s": "v",
                  "dewpoint_k": "td"}
        need = {key_of[v] for v in set(var_of)}
        target = rows.level_pa[all_take]
        sims = np.full((members, all_take.size), np.nan)
        logp_mean = np.zeros((nz, cells.size))
        for m, state in enumerate(states):
            col = _member_columns(state, thb, cj, ci, rotation, need)
            logp_mean += np.log(col["p"]) / members
            for key in need:
                pick = np.nonzero(var_of == {v: k for k, v in key_of.items()}[key])[0]
                if not pick.size:
                    continue
                value = np.zeros(pick.size)
                for (_, _, w), cs in zip(corners, corner_slots):
                    p_c = col["p"][:, cs[pick]]
                    x_c = col[key][:, cs[pick]]
                    value += w[pick] * _interp_logp(p_c, x_c, target[pick])
                sims[m, pick] = value
        # vertical placement: nearest level in log p, ensemble-mean column
        # at the nearest mass point
        ni = np.clip(np.rint(fi).astype(np.int64), 0, nx - 1)
        nj = np.clip(np.rint(fj).astype(np.int64), 0, ny - 1)
        near = np.fromiter((slot.get(int(v), -1) for v in nj * nx + ni),
                           np.int64, count=all_take.size)
        missing = near < 0
        if missing.any():
            # the nearest point is always one of the four corners
            raise ConventionalObsError("internal: nearest mass point is not "
                                       "a bilinear corner")
        kk = np.argmin(np.abs(logp_mean[:, near] - np.log(target)[None, :]),
                       axis=0)
        offset = 0
        for i, take in pl:
            sl = slice(offset, offset + take.size)
            hx[i] = (sims[:, sl], kk[sl], nj[sl], ni[sl])
            offset += take.size
    if sf:
        if surface is None:
            raise ConventionalObsError(
                "surface conventional types are in use and no member surface "
                "diagnostics were supplied; enable them (the cycle captures "
                "them when a surface type is in the table) or set the "
                "surface types' use to false")
        for i, take in sf:
            row_type = table.types[i]
            fi, fj = fi_all[take], fj_all[take]
            corners = _corner_weights(fi, fj, nx, ny)

            def at(name):
                field = np.asarray(surface[name], np.float64)
                if field.shape != (members, ny, nx):
                    raise ConventionalObsError(
                        f"surface diagnostic {name} is {field.shape}, "
                        f"expected {(members, ny, nx)}")
                out = np.zeros((members, take.size))
                for jj, ii, w in corners:
                    out += w[None, :] * field[:, jj, ii]
                return out
            if row_type.variable == "temperature_k":
                sim = at("t2")
            elif row_type.variable == "dewpoint_k":
                sim = _dewpoint_k(_vapour_pressure_pa(at("q2"), at("psfc")))
            else:
                u10, v10 = at("u10"), at("v10")
                if rotation is not None:
                    sina = np.zeros(take.size)
                    cosa = np.zeros(take.size)
                    for jj, ii, w in corners:
                        sina += w * np.asarray(rotation[0])[jj, ii]
                        cosa += w * np.asarray(rotation[1])[jj, ii]
                    u10, v10 = (u10 * cosa - v10 * sina,
                                u10 * sina + v10 * cosa)
                sim = u10 if row_type.variable == "wind_u_m_s" else v10
            ni = np.clip(np.rint(fi).astype(np.int64), 0, nx - 1)
            nj = np.clip(np.rint(fj).astype(np.int64), 0, ny - 1)
            hx[i] = (sim, np.zeros(take.size, np.int64), nj, ni)

    # -- background check, superob, batches -----------------------------------
    groups: dict[str, dict] = {}
    for i, take in selected:
        row_type = table.types[i]
        c = counts[row_type.name]
        sim, kk, nj, ni = hx[i]
        finite = np.all(np.isfinite(sim), axis=0)
        c["outside_vertical_column"] = int(np.count_nonzero(~finite))
        y = rows.value[take]
        err = np.maximum(rows.error[take] * row_type.error_inflation,
                         row_type.error_floor)
        mean = sim.mean(axis=0)
        var = sim.var(axis=0, ddof=1)
        with np.errstate(invalid="ignore"):
            passed = finite & (np.abs(y - mean) <= row_type.gross_sigma
                               * np.sqrt(err ** 2 + var))
        c["background_check"] = int(np.count_nonzero(finite & ~passed))
        keep = np.nonzero(passed)[0]
        c["accepted_reports"] = int(keep.size)
        if not keep.size:
            continue
        flat = (kk[keep] * ny + nj[keep]) * nx + ni[keep]
        g = groups.setdefault(row_type.batch, {
            "type": row_type, "flat": [], "y": [], "err": [], "sim": [],
            "names": Counter()})
        g["flat"].append(flat)
        g["y"].append(y[keep])
        g["err"].append(err[keep])
        g["sim"].append(sim[:, keep])
        g["names"][row_type.name] += int(keep.size)

    batches, batch_receipts = [], []
    for batch_name in sorted(groups):
        g = groups[batch_name]
        flat = np.concatenate(g["flat"])
        y = np.concatenate(g["y"])
        err = np.concatenate(g["err"])
        sim = np.concatenate(g["sim"], axis=1)
        cells, inverse, per = np.unique(flat, return_inverse=True,
                                        return_counts=True)
        ys = np.bincount(inverse, weights=y) / per
        es = np.bincount(inverse, weights=err) / per
        ss = np.stack([np.bincount(inverse, weights=sim[m]) / per
                       for m in range(members)])
        name = f"conventional:{batch_name}"
        row_type = g["type"]
        batches.append(point_batch(
            name, (nz, ny, nx),
            PointSet(flat_index=cells, values=ys, errors=es, simulated=ss),
            localization=row_type.localization()))
        d = ys - ss.mean(axis=0)
        batch_receipts.append({
            "name": name, "kind": "conventional", "batch": batch_name,
            "variable": row_type.variable, "operator": row_type.operator,
            "reports": int(y.size), "superobservations": int(cells.size),
            "reports_by_type": dict(g["names"]),
            "localization_horizontal_m": float(row_type.horizontal_m),
            "localization_vertical_m": float(row_type.vertical_m),
            "innovation_mean": float(d.mean()),
            "innovation_rms": float(np.sqrt(np.mean(d ** 2))),
            "error_mean": float(es.mean()),
            "spread_mean": float(ss.std(axis=0, ddof=1).mean()),
        })
    provenance = {
        "schema": SCHEMA,
        "analysis_time": analysis_time.astimezone(timezone.utc).isoformat(),
        "window_seconds": [-table.window_before_seconds,
                           table.window_after_seconds],
        "type_table": {"path": table.path, "sha256": table.sha256},
        "tables": list(rows.receipts),
        "totals": dict(totals),
        "unmatched_row_kinds": {"|".join(k): n for k, n in
                                sorted(unmatched.items())},
        "types": {name: dict(c) for name, c in sorted(counts.items())},
        "batches": batch_receipts,
        "operator": ("3-D: member columns at the analysis time; log-pressure "
                     "linear, bilinear horizontal, no extrapolation; surface "
                     "types read leg-end 2 m / 10 m diagnostics"),
        "superob": "mean per gridpoint per batch (value, H(x), error)",
    }
    return batches, provenance


def window_times(analysis_time: datetime, table: TypeTable):
    """The ``(start, end]`` valid-time window of one analysis."""
    return (analysis_time - timedelta(seconds=table.window_before_seconds),
            analysis_time + timedelta(seconds=table.window_after_seconds))
