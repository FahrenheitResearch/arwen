"""``gpuwm downscale --follow``: a step-down that moves.

Two things can move in a step-down, and this door handles both.

THE PARENT MOVES.  A storm run's following nest writes every history frame
where the nest was at that instant, so its frames do not share one grid and
``gpuwm downscale`` (which places its child once on one parent grid) refuses
them.  Every frame of such a nest still lies on ONE lattice: a nest moves by
whole cells of its own parent, so its cells always land on the same ground
points.  This door reads each frame's place on that lattice off the frame
itself (its first mass point through the parent's own projection,
:func:`frame_footprints`), so any producer's moving frames are read on their
own footprint.  For each stretch of the run it cuts, from every frame of the
stretch, the same window of ground (:func:`crop_frame`): a parent archive
that no longer moves, which the ordinary offline child reads unchanged.

THE CHILD MOVES.  A following child is a sequence of ordinary offline
children, one per ``--follow-every`` seconds, each placed where the storm
will be over its stretch (:func:`plan_segments`): ``storm`` follows the
parent run's storm track (``storm-track.dNN.csv``, written by the storm
tracker), or the moving parent's own centre when the run wrote none;
``parent`` keeps the child's place inside the moving parent; ``fixed``
holds it on the ground.  A segment is joined to the one before it the way a
moving nest's move is: it starts from the parent at its new placement and
then carries the earlier segment's last checkpoint, bitwise, onto every cell
both placements cover (:mod:`gpuwm.offline_child_warm`), and the freshly
covered strip takes the parent's accumulated rain since the child began
(:func:`strip_accumulations`).  So the forecast continues across a move
instead of spinning up again; only the strip starts from the parent.

THE RECORD.  The run folder holds one frame per output time (the segments'
own frames, linked in time order), the last segment's final checkpoint (so
the run is itself a parent for a further step, moving or not), the maps, and
``moving-child.json``: every segment's window, placement, move and receipt,
and every frame's footprint, so a further step can be checked inside it.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FOLLOW_MODES = ("storm", "parent", "fixed", "target")
DEFAULT_FOLLOW_EVERY_SECONDS = 3600.0

#: Parent cells kept around the child inside each segment's window: SINT's
#: +-2 donor stencil plus room for the child's own relaxation reach.
CROP_MARGIN_PARENT_CELLS = 6

MOVING_CHILD_RECORD = "moving-child.json"
MOVING_CHILD_SCHEMA = "gpuwm.moving-child.v1"
SEGMENTS_DIR = "segments"

#: Fields that identify a frame's grid (offline_child._STATIC_GEOMETRY_FIELDS
#: minus the vertical coordinate, which a move does not touch).  Every window
#: of one segment carries the first frame's values, after checking the others
#: agree, so the window archive has one geometry.
_GEOMETRY_OVERRIDES = ("XLAT", "XLONG", "MAPFAC_M", "HGT")
#: How far the frames of one segment may disagree on those fields before the
#: window is refused (degrees for XLAT/XLONG, metres for HGT).
_GEOMETRY_TOLERANCE = {"XLAT": 1e-3, "XLONG": 1e-3, "MAPFAC_M": 1e-4,
                       "HGT": 25.0}

_FRAME_NAME = re.compile(
    r"wrfout_d(?P<dom>\d{2})_(?P<t>\d{4}-\d{2}-\d{2}[_:]\d{2}[_:]\d{2}[_:]\d{2})$")


class MovingDownscaleRefusal(ValueError):
    """A moving step-down that cannot be planned or run, and why."""


@dataclass(frozen=True)
class Footprint:
    """One parent frame's place on the moving parent's lattice."""

    path: Path
    valid_time: datetime
    origin_i: int     # 0-based lattice index of the frame's first mass column
    origin_j: int
    nx: int
    ny: int

    def to_json(self) -> dict:
        return {"path": str(self.path),
                "valid_time": self.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "origin": [self.origin_i, self.origin_j],
                "shape": [self.ny, self.nx]}


@dataclass(frozen=True)
class Segment:
    """One stretch of the following child, placed once."""

    index: int
    start: datetime
    end: datetime
    frames: tuple[Footprint, ...]
    child_i: int      # 0-based lattice index of the child's first parent cell
    child_j: int
    span_i: int       # child extent in parent cells
    span_j: int
    window_i: int     # 0-based lattice index of the window's first column
    window_j: int
    window_nx: int
    window_ny: int
    clamped: bool
    target: tuple[float, float]   # where the track wanted the child (lattice)

    def to_json(self) -> dict:
        return {"index": self.index,
                "start": self.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": self.end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "child_lattice_start": [self.child_i, self.child_j],
                "child_span_parent_cells": [self.span_i, self.span_j],
                "window_lattice_start": [self.window_i, self.window_j],
                "window_shape": [self.window_ny, self.window_nx],
                "clamped_to_parent": self.clamped,
                "track_target_lattice": [round(self.target[0], 3),
                                         round(self.target[1], 3)],
                "frames": [f.path.name for f in self.frames]}


def _frame_time(path: Path) -> datetime:
    match = _FRAME_NAME.match(path.name)
    if match is None:
        raise MovingDownscaleRefusal(f"{path} is not a wrfout_dNN_<time> frame")
    text = re.sub(r"[_:]", "-", match.group("t"))
    return datetime.strptime(text, "%Y-%m-%d-%H-%M-%S")


def _frame_domain(path: Path) -> int:
    return int(_FRAME_NAME.match(path.name).group("dom"))


def _corner(path: Path) -> tuple[float, float, int, int, float]:
    """First mass point (lat, lon), the mass shape and DX of one frame."""
    from gpuwm.offline_child import open_parent_history

    with open_parent_history(path) as dataset:
        lat = dataset.variables["XLAT"].read_window(i0=0, ni=1, j0=0, nj=1)
        lon = dataset.variables["XLONG"].read_window(i0=0, ni=1, j0=0, nj=1)
        return (float(np.asarray(lat).ravel()[0]),
                float(np.asarray(lon).ravel()[0]),
                len(dataset.dimensions["south_north"]),
                len(dataset.dimensions["west_east"]),
                float(dataset.getncattr("DX")))


def _workers(count: int) -> int:
    return max(1, min(int(count), os.cpu_count() or 1, 32))


def frame_footprints(frames) -> tuple[list[Footprint], object]:
    """Every frame's lattice origin, read off the frame through the projection.

    Returns the footprints and the first frame's projected grid, whose mass
    indices (1-based) define the lattice.  A frame whose first mass point is
    not on that lattice -- not a whole number of cells from the first
    frame's -- is refused: it was not written by a nest moving on one parent.
    """
    from gpuwm.offline_child_geography import parent_projected_grid

    frames = [Path(f) for f in frames]
    grid, _evidence = parent_projected_grid(frames[0])
    with ThreadPoolExecutor(max_workers=_workers(len(frames))) as pool:
        corners = list(pool.map(_corner, frames))
    dx0 = corners[0][4]
    footprints = []
    for path, (lat, lon, ny, nx, dx) in zip(frames, corners):
        if not math.isclose(dx, dx0, rel_tol=1e-6):
            raise MovingDownscaleRefusal(
                f"{path.name} has DX={dx:g} m and {frames[0].name} {dx0:g} m; "
                "one moving parent keeps one spacing")
        x, y = grid.latlon_to_ij(np.float64(lat), np.float64(lon))
        oi, oj = float(x) - 1.0, float(y) - 1.0
        ri, rj = round(oi), round(oj)
        if abs(oi - ri) > 0.05 or abs(oj - rj) > 0.05:
            raise MovingDownscaleRefusal(
                f"{path.name}'s first mass point sits {oi - ri:+.3f}, "
                f"{oj - rj:+.3f} cells off the lattice of {frames[0].name}; "
                "its frames are not one nest moving by whole parent cells")
        footprints.append(Footprint(path=path, valid_time=_frame_time(path),
                                    origin_i=int(ri), origin_j=int(rj),
                                    nx=int(nx), ny=int(ny)))
    return footprints, grid


def parent_moves(footprints) -> bool:
    return len({(f.origin_i, f.origin_j) for f in footprints}) > 1


# ---------------------------------------------------------------------------
# Where the child goes
# ---------------------------------------------------------------------------

def find_storm_track(frames, domain: int) -> Path | None:
    """The parent run's storm-track file for ``domain``, if it wrote one."""
    name = f"storm-track.d{int(domain):02d}.csv"
    folder = Path(frames[0]).resolve().parent
    for candidate in (folder, *list(folder.parents)[:3]):
        path = candidate / name
        if path.is_file():
            return path
    return None


def read_storm_track(path: Path) -> list[tuple[datetime, float, float]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                when = datetime.strptime(row["valid_time"], "%Y-%m-%d_%H:%M:%S")
                lat, lon = float(row["lat_deg"]), float(row["lon_deg"])
            except (KeyError, ValueError):
                continue
            if math.isfinite(lat) and math.isfinite(lon):
                rows.append((when, lat, lon))
    rows.sort()
    return rows


def _interpolate_track(track, when: datetime) -> tuple[float, float] | None:
    if not track:
        return None
    if when <= track[0][0]:
        return track[0][1], track[0][2]
    if when >= track[-1][0]:
        return track[-1][1], track[-1][2]
    for (t0, a0, b0), (t1, a1, b1) in zip(track, track[1:]):
        if t0 <= when <= t1:
            span = (t1 - t0).total_seconds()
            w = 0.0 if span <= 0 else (when - t0).total_seconds() / span
            return a0 + w * (a1 - a0), b0 + w * (b1 - b0)
    return None


def child_targets(footprints, grid, *, mode: str, point, track=None):
    """Where the child's centre should be at each frame, on the lattice.

    The child keeps the offset the user gave at the start: from the storm
    (``storm``), from the moving parent's frame (``parent``), or from the
    ground (``fixed``).  Returns ``(targets, basis)``.
    """
    first = footprints[0]
    if point is None:
        # No point given: the child starts on the storm (the track's start)
        # or, without a track, at the centre of the moving parent's frame.
        start = (_interpolate_track(track, first.valid_time)
                 if mode == "storm" and track else None)
        if start is None:
            px = first.origin_i + (first.nx - 1) / 2.0
            py = first.origin_j + (first.ny - 1) / 2.0
        else:
            px, py = (float(v) - 1.0 for v in grid.latlon_to_ij(
                np.float64(start[0]), np.float64(start[1])))
    else:
        lat, lon = point
        px, py = grid.latlon_to_ij(np.float64(lat), np.float64(lon))
        px, py = float(px) - 1.0, float(py) - 1.0
    if mode == "fixed":
        return [(px, py) for _ in footprints], "fixed on the ground"
    if mode == "storm" and track:
        start = _interpolate_track(track, first.valid_time)
        sx0, sy0 = (float(v) - 1.0 for v in grid.latlon_to_ij(
            np.float64(start[0]), np.float64(start[1])))
        targets = []
        for f in footprints:
            here = _interpolate_track(track, f.valid_time)
            sx, sy = (float(v) - 1.0 for v in grid.latlon_to_ij(
                np.float64(here[0]), np.float64(here[1])))
            targets.append((px + sx - sx0, py + sy - sy0))
        return targets, "the parent run's storm track"
    basis = ("the moving parent's own centre (the run wrote no storm track)"
             if mode == "storm" else "its place in the moving parent")
    return [(px + f.origin_i - first.origin_i, py + f.origin_j - first.origin_j)
            for f in footprints], basis


def _start_offset(span: int) -> int:
    """Parent cells between a centred child's first cell and its centre.

    The door's own rule (downscale._centered_placement): a child of ``span``
    parent cells centred on parent cell c starts at c - offset.
    """
    return -int(math.floor(0.5 - (span - 1) / 2.0))


def plan_segments(footprints, targets, *, span_i: int, span_j: int,
                  start: datetime, seconds: float, every: float,
                  edge: int, margin: int = CROP_MARGIN_PARENT_CELLS):
    """Cut the run into stretches and place the child once in each."""
    by_time = {f.valid_time: (f, t) for f, t in zip(footprints, targets)}
    end = start + timedelta(seconds=float(seconds))
    bounds = []
    t = start
    while t < end:
        bounds.append((t, min(end, t + timedelta(seconds=float(every)))))
        t = bounds[-1][1]
    segments = []
    for index, (ta, tb) in enumerate(bounds):
        rows = [by_time[when] for when in sorted(by_time) if ta <= when <= tb]
        if not rows or rows[0][0].valid_time != ta or rows[-1][0].valid_time != tb:
            raise MovingDownscaleRefusal(
                f"the parent has no frame at {ta:%Y-%m-%d %H:%M} or "
                f"{tb:%Y-%m-%d %H:%M}; --follow-every must be a whole number "
                "of the parent's frame interval and the run must end on a frame")
        frames = tuple(row[0] for row in rows)
        tx = float(np.mean([row[1][0] for row in rows]))
        ty = float(np.mean([row[1][1] for row in rows]))
        ci = int(math.floor(round(tx) - (span_i - 1) / 2.0 + 0.5))
        cj = int(math.floor(round(ty) - (span_j - 1) / 2.0 + 0.5))
        lo_i = max(f.origin_i + edge + margin for f in frames)
        hi_i = min(f.origin_i + f.nx - edge - margin - span_i for f in frames)
        lo_j = max(f.origin_j + edge + margin for f in frames)
        hi_j = min(f.origin_j + f.ny - edge - margin - span_j for f in frames)
        if lo_i > hi_i or lo_j > hi_j:
            raise MovingDownscaleRefusal(
                f"from {ta:%H:%M} to {tb:%H:%M} the moving parent never leaves "
                f"room for a {span_i}x{span_j}-parent-cell child: it moves "
                f"{max(f.origin_i for f in frames) - min(f.origin_i for f in frames)}"
                f" x {max(f.origin_j for f in frames) - min(f.origin_j for f in frames)}"
                f" cells in that time and keeps {edge} boundary cells plus a "
                f"{margin}-cell margin on each side.  Ask for a smaller child or "
                "a shorter --follow-every")
        clamped_i = min(max(ci, lo_i), hi_i)
        clamped_j = min(max(cj, lo_j), hi_j)
        segments.append(Segment(
            index=index, start=ta, end=tb, frames=frames,
            child_i=clamped_i, child_j=clamped_j, span_i=span_i, span_j=span_j,
            window_i=clamped_i - margin, window_j=clamped_j - margin,
            window_nx=span_i + 2 * margin, window_ny=span_j + 2 * margin,
            clamped=(clamped_i, clamped_j) != (ci, cj), target=(tx, ty)))
    return segments


# ---------------------------------------------------------------------------
# One window of ground from every frame of a segment
# ---------------------------------------------------------------------------

_SPATIAL = {
    ("south_north", "west_east"): (0, 0),
    ("south_north", "west_east_stag"): (0, 1),
    ("south_north_stag", "west_east"): (1, 0),
    ("south_north_stag", "west_east_stag"): (1, 1),
}


def _container(format_name: str) -> str:
    text = str(format_name).lower()
    if "64bit_data" in text or "cdf5" in text or "cdf-5" in text:
        return "cdf5"
    if text in ("classic", "netcdf3_classic", "cdf1", "cdf-1"):
        return "cdf1"
    return "cdf2"


def _typed_attrs(path: Path):
    """Global and per-variable attributes with their stored types."""
    from gpuwm.io.netcdf_serialization import netCDF4

    with netCDF4.Dataset(path) as dataset:
        glob = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        per_var = {name: {a: var.getncattr(a) for a in var.ncattrs()}
                   for name, var in dataset.variables.items()}
    return glob, per_var


def read_window_fields(path: Path, names, *, i0: int, j0: int, ni: int,
                       nj: int) -> dict[str, np.ndarray]:
    """Mass-grid windows of ``names`` (raw stored values, Rust decode)."""
    from gpuwm.offline_child import open_parent_history

    with open_parent_history(path) as dataset:
        present = [n for n in names if n in dataset.variables]
        if not present:
            return {}
        decoded = dataset._decode(present, raw=True, scale=False,
                                  window=(int(i0), int(ni), int(j0), int(nj)))
        return {name: np.asarray(decoded[name][0], dtype=np.float64)
                for name in present}


def crop_frame(source: Path, target: Path, *, origin_i: int, origin_j: int,
               window_i: int, window_j: int, nx: int, ny: int,
               overrides=None, centre=None, attributes=None) -> dict:
    """Write the window of ``source`` covering lattice cells
    [window_i, window_i+nx) x [window_j, window_j+ny) to ``target``.

    Read with the Rust NetCDF reader (windows cut in Rust) and written with
    the Rust classic writer, byte for byte in every stored value; the fields
    in ``overrides`` (the segment's reference geometry) replace the frame's
    own, which :func:`crop_segment` checked agree within tolerance.
    """
    from gpuwm.io.nc_writer_bridge import ClassicSchema
    from gpuwm.offline_child import open_parent_history

    i0 = int(window_i - origin_i)
    j0 = int(window_j - origin_j)
    # The typed attributes are read through netCDF4, which is not safe to
    # call from several threads at once: a caller cutting frames in a pool
    # reads them first, on its own thread, and hands them in.
    glob, attrs = attributes if attributes is not None else _typed_attrs(source)
    with open_parent_history(source) as dataset:
        dims = dataset.dimensions
        full_nx = len(dims["west_east"])
        full_ny = len(dims["south_north"])
        if i0 < 0 or j0 < 0 or i0 + nx > full_nx or j0 + ny > full_ny:
            raise MovingDownscaleRefusal(
                f"the window [{i0}, {i0 + nx}) x [{j0}, {j0 + ny}) leaves "
                f"{source.name} ({full_nx} x {full_ny})")
        groups: dict[tuple, list[str]] = {}
        for name, variable in dataset.variables.items():
            key = tuple(variable.dimensions[-2:]) if variable.ndim >= 2 else ()
            groups.setdefault(_SPATIAL.get(key, None) if key in _SPATIAL
                              else None, []).append(name)
        values: dict[str, np.ndarray] = {}
        for stagger, names in groups.items():
            if stagger is None:
                decoded = dataset._decode(names, raw=True, scale=False)
            else:
                decoded = dataset._decode(
                    names, raw=True, scale=False,
                    window=(i0, nx + stagger[1], j0, ny + stagger[0]))
            for name in names:
                values[name] = decoded[name][0]
        sizes = {"west_east": nx, "west_east_stag": nx + 1,
                 "south_north": ny, "south_north_stag": ny + 1}
        schema = ClassicSchema(_container(dataset.format))
        dimids = {}
        unlimited = None
        for name, dimension in dims.items():
            if dimension.isunlimited():
                unlimited = name
                dimids[name] = schema.def_dim(name, 0, unlimited=True)
            else:
                dimids[name] = schema.def_dim(
                    name, sizes.get(name, len(dimension)))
        updates = {
            "WEST-EAST_GRID_DIMENSION": np.int32(nx + 1),
            "SOUTH-NORTH_GRID_DIMENSION": np.int32(ny + 1),
            "WEST-EAST_PATCH_END_UNSTAG": np.int32(nx),
            "WEST-EAST_PATCH_END_STAG": np.int32(nx + 1),
            "SOUTH-NORTH_PATCH_END_UNSTAG": np.int32(ny),
            "SOUTH-NORTH_PATCH_END_STAG": np.int32(ny + 1),
            "GPUWM_MOVING_WINDOW": np.asarray([window_i, window_j, nx, ny],
                                              dtype=np.int32),
        }
        if centre is not None:
            updates["CEN_LAT"] = np.float32(centre[0])
            updates["CEN_LON"] = np.float32(centre[1])
        for name, value in glob.items():
            if name in updates:
                continue
            schema.put_global_attr(name, value)
        for name, value in updates.items():
            if name in glob or name.startswith("GPUWM_MOVING") or name in (
                    "CEN_LAT", "CEN_LON"):
                schema.put_global_attr(name, value)
        varids = {}
        for name, variable in dataset.variables.items():
            varids[name] = schema.def_var(
                name, variable.dtype, [dimids[d] for d in variable.dimensions])
            for attr, value in attrs.get(name, {}).items():
                schema.put_var_attr(varids[name], attr, value)
        records = {name for name, variable in dataset.variables.items()
                   if unlimited is not None and variable.dimensions
                   and variable.dimensions[0] == unlimited}
        nrec = len(dims[unlimited]) if unlimited is not None else 0
        with schema.create(target) as writer:
            for name, variable in dataset.variables.items():
                data = (overrides or {}).get(name)
                if data is None:
                    array = np.asarray(values[name])
                else:
                    array = np.asarray(data).reshape(np.shape(values[name]))
                dtype = variable.dtype
                if dtype.kind != "S":
                    array = array.astype(dtype)
                if name in records:
                    for record in range(nrec):
                        writer.write_record(record, varids[name], array[record])
                else:
                    writer.write_var(varids[name], array)
    return {"source": str(source), "target": str(target),
            "window_in_frame": [i0, j0, nx, ny]}


def _segment_geometry(segment: Segment) -> tuple[dict, dict]:
    """The segment's reference geometry, checked across its frames."""
    reference = None
    disagreement = {name: 0.0 for name in _GEOMETRY_OVERRIDES}
    readings = []
    with ThreadPoolExecutor(max_workers=_workers(len(segment.frames))) as pool:
        readings = list(pool.map(lambda f: read_window_fields(
            f.path, _GEOMETRY_OVERRIDES,
            i0=segment.window_i - f.origin_i, j0=segment.window_j - f.origin_j,
            ni=segment.window_nx, nj=segment.window_ny), segment.frames))
    reference = readings[0]
    for frame, reading in zip(segment.frames[1:], readings[1:]):
        for name, value in reading.items():
            if name not in reference:
                continue
            diff = float(np.max(np.abs(value - reference[name])))
            if name == "XLONG":
                diff = float(np.max(np.abs(
                    (value - reference[name] + 180.0) % 360.0 - 180.0)))
            disagreement[name] = max(disagreement[name], diff)
            if diff > _GEOMETRY_TOLERANCE[name]:
                raise MovingDownscaleRefusal(
                    f"{frame.path.name} and {segment.frames[0].path.name} "
                    f"disagree on {name} by {diff:g} over the same ground "
                    f"(tolerance {_GEOMETRY_TOLERANCE[name]:g}); the moving "
                    "parent's frames do not describe one surface there")
    return reference, disagreement


def crop_segment(segment: Segment, folder: Path, grid) -> dict:
    """Every frame of ``segment`` cut to its window, in ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    reference, disagreement = _segment_geometry(segment)
    # The first frame's geometry, written into every window (crop_frame
    # gives it the variable's own shape and type).
    overrides = dict(reference)
    cx = segment.window_i + (segment.window_nx + 1) / 2.0
    cy = segment.window_j + (segment.window_ny + 1) / 2.0
    centre = tuple(float(v) for v in grid.ij_to_latlon(
        np.float64(cx), np.float64(cy)))

    started = time.perf_counter()
    attributes = {frame.path: _typed_attrs(frame.path) for frame in segment.frames}

    def one(frame: Footprint):
        return crop_frame(
            frame.path, folder / frame.path.name, origin_i=frame.origin_i,
            origin_j=frame.origin_j, window_i=segment.window_i,
            window_j=segment.window_j, nx=segment.window_nx,
            ny=segment.window_ny, overrides=overrides, centre=centre,
            attributes=attributes[frame.path])

    with ThreadPoolExecutor(max_workers=_workers(len(segment.frames))) as pool:
        receipts = list(pool.map(one, segment.frames))
    return {"frames": receipts, "geometry_disagreement": disagreement,
            "centre_latlon": list(centre), "reference": reference,
            "seconds": time.perf_counter() - started,
            "workers": _workers(len(segment.frames))}


def window_physics_evidence(restart: Path, target: Path, *, nx: int,
                            ny: int) -> Path:
    """The parent checkpoint's header, re-shaped to a segment's window.

    ``gpuwm downscale`` binds a child's physics from its parent's checkpoint
    header and checks the header's grid against the frames'; a window's
    frames are smaller than the moving parent's, so the header handed on
    says the window's shape.  Nothing else in it changes, and no array is
    copied (the door reads the header only).
    """
    from gpuwm.io.restart import _HEADER_KEY, read_restart_header

    header = copy.deepcopy(read_restart_header(restart))
    header.setdefault("config", {})
    header["config"]["nx"] = int(nx)
    header["config"]["ny"] = int(ny)
    header["moving_window_of"] = str(restart)
    payload = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        np.savez(handle, **{_HEADER_KEY: payload})
    return target


def _bilinear_to_child(field: np.ndarray, *, ratio: int, span_i: int,
                       span_j: int) -> np.ndarray:
    """Parent cells (with a one-cell ring round the child) onto child cells.

    ``field`` covers parent cells [-1, span + 1) relative to the child's
    first parent cell; child cell c sits at parent coordinate
    (c + 0.5) / ratio - 0.5 (WRF's nest geometry, odd or even ratio).
    """
    def axis(n_child: int):
        x = (np.arange(n_child) + 0.5) / ratio - 0.5 + 1.0   # +1: the ring
        i0 = np.floor(x).astype(int)
        return i0, x - i0

    jy, wy = axis(span_j * ratio)
    ix, wx = axis(span_i * ratio)
    a = field[np.ix_(jy, ix)]
    b = field[np.ix_(jy, ix + 1)]
    c = field[np.ix_(jy + 1, ix)]
    d = field[np.ix_(jy + 1, ix + 1)]
    wx = wx[np.newaxis, :]
    wy = wy[:, np.newaxis]
    return ((1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * c + wx * d))


def strip_accumulations(segment: Segment, first: Footprint, *, ratio: int,
                        target: Path) -> Path | None:
    """The parent's rain since the child began, on this segment's child grid.

    The value at segment start minus the value at the child's start, at the
    same ground; ground the parent did not cover at the child's start counts
    from the parent's own start there.  Interpolated bilinearly onto the
    child's cells (WRF interpolates the exposed cells' rain from the parent).
    The warm start writes it only where the child takes the parent's rain:
    the freshly covered strip and the earlier child's specified ring.
    """
    from gpuwm.offline_child_warm import STRIP_ACCUMULATOR_HISTORY

    names = tuple(STRIP_ACCUMULATOR_HISTORY.values())
    now = segment.frames[0]
    # The child's parent cells with a one-cell ring (inside the window's
    # margin, so inside the frame).
    gi, gj = segment.child_i - 1, segment.child_j - 1
    ni, nj = segment.span_i + 2, segment.span_j + 2
    later = read_window_fields(now.path, names, i0=gi - now.origin_i,
                               j0=gj - now.origin_j, ni=ni, nj=nj)
    if not later:
        return None
    lo_i = max(gi, first.origin_i)
    hi_i = min(gi + ni, first.origin_i + first.nx)
    lo_j = max(gj, first.origin_j)
    hi_j = min(gj + nj, first.origin_j + first.ny)
    earlier = {}
    if lo_i < hi_i and lo_j < hi_j:
        earlier = read_window_fields(
            first.path, names, i0=lo_i - first.origin_i,
            j0=lo_j - first.origin_j, ni=hi_i - lo_i, nj=hi_j - lo_j)
    out = {}
    for slot, name in STRIP_ACCUMULATOR_HISTORY.items():
        if name not in later:
            continue
        value = np.array(later[name].reshape(nj, ni))
        if name in earlier:
            before = earlier[name].reshape(hi_j - lo_j, hi_i - lo_i)
            value[lo_j - gj:hi_j - gj, lo_i - gi:hi_i - gi] -= before
        value = np.maximum(value, 0.0)
        out[slot] = _bilinear_to_child(
            value, ratio=ratio, span_i=segment.span_i,
            span_j=segment.span_j).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **out)
    return target


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

def _child_extent(args, ratio: int) -> tuple[int, int]:
    if args.child_size is None:
        raise MovingDownscaleRefusal(
            "--follow needs --child-size NX[,NY]: a following child keeps one "
            "extent through every move")
    parts = [int(p) for p in str(args.child_size).split(",")]
    nx = parts[0]
    ny = parts[1] if len(parts) > 1 else parts[0]
    nx, ny = nx - nx % ratio, ny - ny % ratio
    if min(nx, ny) < 2 * ratio:
        raise MovingDownscaleRefusal(
            f"--child-size {args.child_size} is smaller than two parent cells "
            f"at ratio {ratio}")
    return nx, ny


def _segment_args(args, **changes) -> argparse.Namespace:
    namespace = argparse.Namespace(**vars(args))
    namespace.follow = None
    namespace.follow_every = None
    namespace.dry_run = False
    namespace.child_config = None
    namespace.i_parent_start = None
    namespace.j_parent_start = None
    for key, value in changes.items():
        setattr(namespace, key, value)
    return namespace


def _link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(os.path.relpath(source, target.parent))
    except OSError:
        shutil.copy2(source, target)


class _Collector:
    """Links each segment's frames and maps into the run folder as they land.

    One frame per output time: a segment after the first starts on the time
    the one before it ended, and that earlier frame (the forecast itself) is
    the one kept.  Maps the same way, by their path under ``png/``.  Swept
    every few seconds while a segment runs, so a reader of the run folder
    sees the first map as soon as the first segment draws it.
    """

    def __init__(self, out: Path):
        self.out = out
        self.frames: set[str] = set()
        self.maps: set[str] = set()
        self._lock = threading.Lock()

    def sweep(self, folder: Path) -> None:
        with self._lock:
            for frame in sorted(folder.glob("wrfout_d*")):
                if _FRAME_NAME.match(frame.name) is None or frame.name in self.frames:
                    continue
                self.frames.add(frame.name)
                _link(frame, self.out / frame.name)
            png = folder / "png"
            if not png.is_dir():
                return
            for picture in sorted(png.rglob("*.png")):
                relative = str(picture.relative_to(png))
                if relative in self.maps:
                    continue
                self.maps.add(relative)
                target = self.out / "png" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                _link(picture, target)

    def watch(self, folder: Path, every: float = 5.0):
        stop = threading.Event()

        def loop():
            while not stop.wait(every):
                try:
                    self.sweep(folder)
                except OSError:
                    pass

        thread = threading.Thread(target=loop, name="moving-collector",
                                  daemon=True)
        thread.start()
        return stop, thread


def _target_track(args, footprints, point):
    """``--follow target``: the target's track over the parent's frames, and its record."""
    from gpuwm.setup_targets import SetupTargetRefusal, TargetConfig, follow_track, read_path

    target = getattr(args, "follow_target", None)
    if not target:
        raise MovingDownscaleRefusal(
            "--follow target needs --follow-target (low, cape-shear, boundary, object or path)")
    segment = getattr(args, "follow_segment", None)
    if segment is not None:
        try:
            values = [float(v) for v in str(segment).split(",")]
        except ValueError:
            values = []
        if len(values) != 4:
            raise MovingDownscaleRefusal("--follow-segment is LAT,LON,LAT,LON")
        segment = [values[:2], values[2:]]
    try:
        config = TargetConfig.for_target(target, threshold=getattr(args, "follow_threshold", None),
                                         boundary_field=getattr(args, "follow_boundary_field", None),
                                         object_field=getattr(args, "follow_field", None), segment=segment)
        if config.target == "path":
            if getattr(args, "follow_path", None) is None:
                raise MovingDownscaleRefusal("--follow-target path needs --follow-path FILE (timed points)")
            config.path = read_path(args.follow_path)
        start_point = point
        if start_point is None and segment is not None:
            start_point = ((segment[0][0] + segment[1][0]) / 2.0, (segment[0][1] + segment[1][1]) / 2.0)
        track, fixes = follow_track([f.path for f in footprints], [f.valid_time for f in footprints],
                                    config, start_point)
    except SetupTargetRefusal as error:
        raise MovingDownscaleRefusal(str(error)) from None
    basis = ("the user's path (not a tracker)" if config.target == "path" else
             f"the {config.target} target, fixed on each parent frame from model state")
    hits = sum(1 for f in fixes if f.get("ok"))
    return track, {"target": config.target, "basis": basis, "fixes": fixes, "hits": hits,
                   "frames": len(fixes), "config": {k: v for k, v in vars(config).items() if k != "path"}}


def _write_track_window(source: Path, target: Path, *, start: datetime,
                        end: datetime) -> None:
    with open(source, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    kept = [rows[0]]
    for row in rows[1:]:
        try:
            when = datetime.strptime(row[0], "%Y-%m-%d_%H:%M:%S")
        except (ValueError, IndexError):
            continue
        if start <= when <= end:
            kept.append(row)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(kept)


def _child_frame_footprints(frames) -> list[dict]:
    rows = []
    for frame in frames:
        lat, lon, ny, nx, _dx = _corner(frame)
        rows.append({"frame": frame.name, "first_mass_point": [lat, lon],
                     "shape": [ny, nx]})
    return rows


def moving_downscale_main(args) -> int:
    """``gpuwm downscale --follow MODE``."""
    from gpuwm.downscale import (
        _discover_parent_series, _frame_spacing_m, _parse_point,
        _resolve_parent_restart, downscale_main)
    from gpuwm.io.restart import read_restart_header

    started = time.perf_counter()
    mode = str(args.follow)
    if mode not in FOLLOW_MODES:
        raise MovingDownscaleRefusal(f"--follow must be one of {FOLLOW_MODES}")
    if args.child_config is not None:
        raise MovingDownscaleRefusal(
            "--follow derives each segment's child from --point; pass "
            "--point and --child-size, not --child-config")
    if args.parent_restart is None:
        raise MovingDownscaleRefusal(
            "--follow binds the child's physics from the parent's gpuwm "
            "checkpoint: pass --parent-restart PATH|latest")
    if getattr(args, "warm_from", None) is not None:
        raise MovingDownscaleRefusal(
            "--follow joins its own segments; --warm-from is the seam it uses, "
            "not an input to it")
    ratio = int(args.ratio if args.ratio is not None else 3)
    child_nx, child_ny = _child_extent(args, ratio)
    span_i, span_j = child_nx // ratio, child_ny // ratio
    every = float(args.follow_every or DEFAULT_FOLLOW_EVERY_SECONDS)

    frames = _discover_parent_series([Path(p) for p in args.parent],
                                     args.parent_domain)
    domain = _frame_domain(frames[0])
    restart = _resolve_parent_restart(args.parent_restart, frames, domain)
    header = read_restart_header(restart)
    parent_config = dict(header.get("config") or {})
    edge = int(parent_config.get("spec_bdy_width", 5)) + 1

    footprints, grid = frame_footprints(frames)
    footprints.sort(key=lambda f: f.valid_time)
    start = footprints[0].valid_time
    window = (footprints[-1].valid_time - start).total_seconds()
    seconds = float(args.hours) * 3600.0 if args.hours is not None else window
    if seconds > window + 1e-6:
        raise MovingDownscaleRefusal(
            f"the child asks for {seconds / 3600:g} h and the parent frames "
            f"cover {window / 3600:g} h from {start:%Y-%m-%d %H:%M}")
    interval = (footprints[1].valid_time - start).total_seconds()
    every = min(every, seconds)
    if every % interval or seconds % interval:
        raise MovingDownscaleRefusal(
            f"--follow-every {every:g} s and the run length {seconds:g} s must "
            f"be whole numbers of the parent's {interval:g} s frame interval")
    output_interval = float(args.output_interval_seconds or interval)
    if every % output_interval:
        raise MovingDownscaleRefusal(
            f"--follow-every {every:g} s must be a whole number of the child's "
            f"{output_interval:g} s output interval, so every segment ends on "
            "a frame")

    point = None if args.point is None else _parse_point(args.point)
    if point is None and mode == "fixed":
        raise MovingDownscaleRefusal(
            "--follow fixed needs --point LAT,LON: where on the ground the "
            "child stays")
    target_record = None
    if mode == "target":
        # A severe-weather setup, fixed on the parent's own frames (gpuwm.setup_targets): a surface low, the
        # CAPE x shear area, a dryline/front, a thresholded feature, or the user's own path.
        target_track, target_record = _target_track(args, footprints, point)
        targets, basis = child_targets(footprints, grid, mode="storm",
                                       point=None, track=target_track)
        basis = target_record["basis"]
        track_path = None
    else:
        track_path = find_storm_track(frames, domain) if mode == "storm" else None
        track = read_storm_track(track_path) if track_path is not None else None
        targets, basis = child_targets(footprints, grid, mode=mode,
                                       point=point, track=track)
    segments = plan_segments(footprints, targets, span_i=span_i, span_j=span_j,
                             start=start, seconds=seconds, every=every,
                             edge=edge)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise MovingDownscaleRefusal(f"--out {out} already holds files")
    out.mkdir(parents=True, exist_ok=True)
    if target_record is not None:
        from gpuwm.setup_targets import write_track_csv
        track_path = out / "target-track.csv"
        write_track_csv(track_path, target_track)
        (out / "target-fixes.json").write_text(
            json.dumps(target_record, indent=2, default=str) + "\n", encoding="utf-8")
    record = {
        "schema": MOVING_CHILD_SCHEMA,
        "parent": {"frames_dir": str(Path(frames[0]).resolve().parent),
                   "domain": domain, "restart": str(restart),
                   "moves": parent_moves(footprints),
                   "spacing_m": _frame_spacing_m(frames[0]),
                   "frame_interval_seconds": interval,
                   "footprints": [f.to_json() for f in footprints]},
        "follow": {"mode": mode, "basis": basis,
                   "target": None if target_record is None else target_record["target"],
                   "track": None if track_path is None else str(track_path),
                   "every_seconds": every,
                   "point": None if point is None else list(point)},
        "child": {"ratio": ratio, "nx": child_nx, "ny": child_ny,
                  "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "hours": seconds / 3600.0,
                  "output_interval_seconds": output_interval},
        "segments": [],
    }
    print(f"gpuwm downscale --follow {mode}: {len(segments)} segment(s) of "
          f"{every / 60:g} min, child {child_nx}x{child_ny} at ratio {ratio}, "
          f"following {basis}", flush=True)
    runs = []
    previous = None
    collector = _Collector(out)
    for segment in segments:
        seg_started = time.perf_counter()
        name = f"s{segment.index:02d}"
        crop_dir = out / SEGMENTS_DIR / f"{name}-parent"
        cropped = crop_segment(segment, crop_dir, grid)
        evidence = window_physics_evidence(
            restart, crop_dir / "parent-physics.npz",
            nx=segment.window_nx, ny=segment.window_ny)
        # The child's centre cell in the window, so the door's own centring
        # puts its first cell exactly where the plan did.
        local_i = segment.child_i - segment.window_i + _start_offset(span_i)
        local_j = segment.child_j - segment.window_j + _start_offset(span_j)
        point_lat = float(cropped["reference"]["XLAT"].reshape(
            segment.window_ny, segment.window_nx)[local_j, local_i])
        point_lon = float(cropped["reference"]["XLONG"].reshape(
            segment.window_ny, segment.window_nx)[local_j, local_i])
        warm = {}
        if previous is not None:
            shift = ((segment.child_i - previous["segment"].child_i) * ratio,
                     (segment.child_j - previous["segment"].child_j) * ratio)
            strip = strip_accumulations(
                segment, footprints[0], ratio=ratio,
                target=crop_dir / "strip-accumulations.npz")
            warm = {"warm_from": Path(previous["final_restart"]),
                    "warm_shift": f"{shift[0]},{shift[1]}",
                    "warm_strip": strip}
        seg_out = out / SEGMENTS_DIR / name
        seg_args = _segment_args(
            args, parent=[str(crop_dir)], parent_domain=domain,
            parent_restart=str(evidence),
            point=f"{point_lat!r},{point_lon!r}",
            child_size=f"{child_nx},{child_ny}", ratio=ratio,
            hours=(segment.end - segment.start).total_seconds() / 3600.0,
            out=seg_out, simulation_start=start, **warm)
        print(f"gpuwm downscale --follow: segment {segment.index + 1} of "
              f"{len(segments)}, {segment.start:%H:%M}-{segment.end:%H:%M}, "
              f"child at lattice ({segment.child_i}, {segment.child_j})"
              + ("" if previous is None else
                 f", moved {warm['warm_shift']} child cells"), flush=True)
        stop, watcher = collector.watch(seg_out)
        try:
            rc = downscale_main(seg_args)
        finally:
            stop.set()
            watcher.join(timeout=10)
            if seg_out.is_dir():
                collector.sweep(seg_out)
        if previous is None:
            # The run folder's own child config: the first segment's, which
            # every segment shares but for its placement.
            for config in sorted(seg_out.glob("*.toml")):
                shutil.copy2(config, out / config.name)
        report = json.loads((seg_out / "report.json").read_text("utf-8"))
        placed = report.get("placement") or {}
        expected = (segment.child_i - segment.window_i + 1,
                    segment.child_j - segment.window_j + 1)
        if (placed.get("i_parent_start"), placed.get("j_parent_start")) != expected:
            raise MovingDownscaleRefusal(
                f"segment {segment.index} was placed at "
                f"({placed.get('i_parent_start')}, {placed.get('j_parent_start')}) "
                f"in its window, not the planned {expected}")
        run = {"segment": segment, "folder": str(seg_out), "rc": rc,
               "final_restart": report.get("final_restart"),
               "result": report.get("result")}
        runs.append(run)
        record["segments"].append({
            **segment.to_json(),
            "folder": str(seg_out), "result": report.get("result"),
            "window_frames": cropped["frames"],
            "window_seconds": cropped["seconds"],
            "window_workers": cropped["workers"],
            "geometry_disagreement": cropped["geometry_disagreement"],
            "window_centre_latlon": cropped["centre_latlon"],
            "child_centre_latlon": [point_lat, point_lon],
            "warm_start": report.get("warm_start"),
            "wall_seconds": time.perf_counter() - seg_started,
            "child_wall_seconds": report.get("wall_seconds"),
        })
        (out / MOVING_CHILD_RECORD).write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8")
        if rc != 0 or report.get("result") != "PASS":
            break
        previous = run
    last = runs[-1].get("final_restart") if runs else None
    if last:
        _link(Path(last), out / Path(last).name)
    if track_path is not None:
        # The storm this child followed, over its own window, under the
        # child's own domain number: a further step inside this moving child
        # follows the same storm (find_storm_track looks beside the frames).
        _write_track_window(track_path, out / f"storm-track.d{domain + 1:02d}.csv",
                            start=start, end=start + timedelta(seconds=seconds))
    assembled = {"frames": len(collector.frames), "maps": len(collector.maps)}
    passed = (len(runs) == len(segments)
              and all(r["result"] == "PASS" for r in runs))
    record["frames"] = _child_frame_footprints(
        sorted(p for p in out.glob("wrfout_d*")
               if _FRAME_NAME.match(p.name)))
    record["assembled"] = assembled
    record["result"] = "PASS" if passed else "FAIL"
    record["wall_seconds"] = time.perf_counter() - started
    (out / MOVING_CHILD_RECORD).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    (out / "report.json").write_text(json.dumps({
        "result": record["result"], "pipeline": "gpuwm-moving-downscale",
        "segments": len(runs), "planned_segments": len(segments),
        "frames": assembled["frames"], "maps": assembled["maps"],
        "final_restart": runs[-1].get("final_restart") if runs else None,
        "wall_seconds": record["wall_seconds"],
        "record": MOVING_CHILD_RECORD}, indent=2) + "\n", encoding="utf-8")
    print(f"gpuwm downscale --follow: {record['result']}, {len(runs)} of "
          f"{len(segments)} segments, {assembled['frames']} frames, "
          f"{assembled['maps']} maps, {record['wall_seconds']:.0f} s", flush=True)
    return 0 if passed else 1


__all__ = ["CROP_MARGIN_PARENT_CELLS", "DEFAULT_FOLLOW_EVERY_SECONDS",
           "FOLLOW_MODES", "MOVING_CHILD_RECORD", "MOVING_CHILD_SCHEMA",
           "MovingDownscaleRefusal", "child_targets", "crop_frame",
           "frame_footprints", "moving_downscale_main", "plan_segments",
           "strip_accumulations", "window_physics_evidence"]
