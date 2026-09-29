"""Each boundary a cycle completes, written as a frame and drawn as it lands.

THE DEFECT THIS CLOSES: ``gpuwm cycle`` drew no picture at any point of a
run.  Its parent engines publish anchors (``anchors/``), which are the
state a cycle resumes from and not a history frame, and nothing under
``gpuwm/cycle/`` wrote a wrfout or raised the landing hook the other
routes draw from.  The 2.7.5 to 2.7.7 notes carried it as a known limit
with no reason beyond that.  Every other route that writes history draws
each frame as it lands (:mod:`gpuwm.live_products`); this is the same
drawing for the cycle.

What a boundary becomes
-----------------------
A frame is written only where the parent can be placed on the Earth: the
parent's planes must sit on a latitude/longitude grid, taken, first match
wins, from

1. the parent's own ``XLAT``/``XLONG`` planes (a recorded series that
   carries its coordinates beside its fields),
2. ``--parent-geo-file``,
3. the first ``--placement-obs-file``, whose radar grid IS the target
   model grid by its own contract.

Every 2-D plane the boundary carries on that grid goes into
``wrfout/wrfout_d01_<valid time>`` through
:func:`gpuwm.io.surface_wrfout.write_surface_wrfout`, the writer the other
two lanes whose forecasts are 2-D snapshots already use; the parent's
``composite_reflectivity`` becomes the frame's reflectivity, so the
renderer draws the composite the parent carried.  Nothing the parent does
not carry is drawn: the writer adds only what the renderer needs to open
the file (a zero one-level ``T`` and the unrotated-grid pair, neither
drawn as a picture), never a zero terrain or column mass that would be
drawn as a 0 m Terrain Height map.  The frame is moved onto
its name only once it is complete, then handed to the same
:class:`gpuwm.live_products.LiveProducts` every other route uses, which
draws it into ``png/`` while the next leg runs.  When the cycle ends the
shared finalize stage (:func:`gpuwm.go_cli._render_stage`) draws only
the frames that drawing could not prove it published.

What is not drawn, and why
--------------------------
A parent on an MPAS mesh (a boundary stamped ``mpas-cuda`` or
``mpas-cuda-frames``, which is what every model parent the door runs
through the port's bridge comes back as, or a replayed MPAS series)
publishes cell arrays (``rho`` is ``(nCells, nLevels)``), and its anchor
carries no cell coordinates, so no plane of it can be placed on a map;
such a run says so in one line and draws nothing.  A replayed parent
whose planes are on a grid but that names no coordinates for it says
which flag supplies them.

Telemetry never fails a cycle: every failure here is a line on stderr
and a ``warning`` event, and the anchors, receipts and ledger are
written exactly as before.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

#: The folder under the cycle root the boundary frames are written into.
FRAMES_DIRNAME = "wrfout"

#: The folder under the cycle root the pictures are drawn into.
PICTURES_DIRNAME = "png"

#: The cycle's own event stream, in the run-plan event schema every other
#: route writes, so one reader serves them all.
EVENTS_FILENAME = "events.jsonl"

#: The parent plane that becomes the frame's reflectivity.  It is the
#: name the cycle's tracker already reads the parent's composite under
#: (:data:`gpuwm.cycle.cli.DEFAULT_TRACKER_FIELD`).
COMPOSITE_PLANE = "composite_reflectivity"

#: The parent is one grid.
DOMAIN = 1

#: The coordinate planes, in the spellings a parent series may carry.
_LATITUDE = ("XLAT", "LAT")
_LONGITUDE = ("XLONG", "LON")

_EARTH_RADIUS_M = 6_370_000.0


class ParentGrid:
    """Where the parent's planes are: 2-D latitude, longitude and spacing."""

    def __init__(self, *, lat, lon, dx_m: float, source: str,
                 dx_source: str):
        self.lat = np.asarray(lat, dtype=np.float64)
        self.lon = np.asarray(lon, dtype=np.float64)
        self.dx_m = float(dx_m)
        self.source = str(source)
        self.dx_source = str(dx_source)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.lat.shape)


def _first_plane(arrays: Mapping[str, Any], names: Sequence[str]):
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name])
    return None


def _spacing_m(lat: np.ndarray, lon: np.ndarray) -> float:
    """The grid spacing at the grid's centre, measured off its coordinates.

    The mean of the great-circle distances to the east and north
    neighbours of the centre cell.  Used only when the run names no
    ``--parent-dx-m``, and the frame says which it was.
    """

    ny, nx = lat.shape
    j, i = max(0, ny // 2 - 1), max(0, nx // 2 - 1)
    distances = []
    for dj, di in ((0, 1), (1, 0)):
        if j + dj >= ny or i + di >= nx:
            continue
        phi1, phi2 = np.radians(lat[j, i]), np.radians(lat[j + dj, i + di])
        dphi = phi2 - phi1
        dlam = np.radians(lon[j + dj, i + di] - lon[j, i])
        a = (math.sin(dphi / 2.0) ** 2
             + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2)
        distances.append(2.0 * _EARTH_RADIUS_M
                         * math.asin(math.sqrt(min(1.0, a))))
    usable = [value for value in distances if value > 0.0]
    return float(sum(usable) / len(usable)) if usable else 0.0


def _latlon_file(path: Path):
    from gpuwm.cycle.cli import _latlon_from

    return _latlon_from(Path(path))


#: The parent kinds whose state is MPAS model output.  Their boundaries
#: are cell arrays on a mesh, never a grid.  Read off the boundary's own
#: record as well as the kind asked for, because the kind is earned
#: (:func:`gpuwm.cycle.mpas_bridge.stamp_for_segment`): every model
#: parent the door runs goes through the port's bridge and comes back
#: stamped one of these.
MESH_PARENT_KINDS = ("mpas-cuda", "mpas-cuda-frames")

#: Why a mesh parent draws nothing, in one sentence.
MESH_SENTENCE = (
    "this parent runs on an MPAS mesh: its boundaries are cell arrays and "
    "its anchor carries no cell coordinates, so no plane of it can be "
    "placed on a map")


def resolve_grid(record: Mapping[str, Any], *, geo_file=None,
                 obs_files: Sequence[Any] = (), dx_m: float | None = None,
                 parent_kind: str | None = None,
                 read_file: Callable = _latlon_file
                 ) -> tuple[ParentGrid | None, str | None]:
    """The parent's grid, or ``None`` and the sentence saying why not."""

    if (parent_kind in MESH_PARENT_KINDS
            or record.get("kind") in MESH_PARENT_KINDS):
        return None, MESH_SENTENCE
    arrays = {**dict(record.get("prognostic_arrays") or {}),
              **dict(record.get("derived_arrays") or {})}
    lat = _first_plane(arrays, _LATITUDE)
    lon = _first_plane(arrays, _LONGITUDE)
    source = "the parent's own XLAT/XLONG"
    if lat is None or lon is None:
        lat = lon = None
        named = geo_file if geo_file else (obs_files[0] if obs_files
                                            else None)
        if named is not None:
            lat, lon = read_file(Path(named))
            source = str(named)
    planes = {name: np.shape(value) for name, value in arrays.items()
              if np.ndim(value) >= 1}
    if lat is None or lon is None:
        if any(len(shape) == 2 for shape in planes.values()):
            return None, (
                "the parent's planes are not placed on the Earth: it carries "
                "no XLAT/XLONG of its own and this run named no coordinates "
                "for it. Pass --parent-geo-file PATH (XLAT/XLONG on the "
                "parent's mass grid) to draw every boundary")
        return None, ("the parent carries no 2-D plane, so there is nothing "
                      "to draw")
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.ndim != 2 or lat.shape != lon.shape:
        return None, (
            f"the parent's coordinates from {source} are latitude "
            f"{tuple(lat.shape)} and longitude {tuple(lon.shape)}; a map "
            "needs both on one 2-D grid, which a mesh of cells is not")
    if dx_m is not None and float(dx_m) > 0.0:
        spacing, dx_source = float(dx_m), "--parent-dx-m"
    else:
        spacing = _spacing_m(lat, lon)
        dx_source = "measured off the coordinates"
        if not spacing > 0.0:
            return None, (
                f"the parent's coordinates from {source} do not change "
                "between neighbouring cells, so its grid spacing cannot be "
                "measured; pass --parent-dx-m METRES")
    grid = ParentGrid(lat=lat, lon=lon, dx_m=spacing, source=source,
                      dx_source=dx_source)
    if not any(shape == grid.shape for name, shape in planes.items()
               if name not in (*_LATITUDE, *_LONGITUDE)):
        described = ", ".join(f"{name} {shape}" for name, shape
                              in sorted(planes.items())[:4])
        return None, (
            f"no plane of the parent is on the {grid.shape} grid from "
            f"{source} ({described}), so there is nothing to place on a "
            "map")
    return grid, None


def boundary_snapshot(record: Mapping[str, Any], grid: ParentGrid
                      ) -> tuple[dict[str, np.ndarray], list[str]]:
    """The planes of one boundary on the parent's grid, and the rest.

    Returns the snapshot :func:`write_surface_wrfout` takes and the names
    of the planes left out because they are not on the grid.
    """

    snapshot: dict[str, np.ndarray] = {"XLAT": grid.lat, "XLONG": grid.lon}
    left_out: list[str] = []
    for block in ("prognostic_arrays", "derived_arrays"):
        for name, value in dict(record.get(block) or {}).items():
            if name in (*_LATITUDE, *_LONGITUDE):
                continue
            array = np.asarray(value)
            if array.ndim == 0:
                continue
            if array.shape == grid.shape and array.dtype.kind in "biuf":
                snapshot[str(name)] = array
            else:
                left_out.append(str(name))
    return snapshot, sorted(set(left_out))


class BoundaryPictures:
    """One cycle's boundaries, each written as a frame and drawn as it lands.

    Hand :meth:`boundary_completed` to the supervisor as its
    ``on_cycle_completed``.  :meth:`finish` is a cycle that completed,
    :meth:`close` one that did not.  ``live_products`` and
    ``first_products`` are what the shared finalize stage reads off its
    observer.
    """

    #: The cycle has no first-frame render: every boundary goes through
    #: the one every-frame render.
    first_products = None

    def __init__(self, *, root, clock, render_products=None,
                 geo_file=None, obs_files: Sequence[Any] = (),
                 dx_m: float | None = None, parent_kind: str | None = None,
                 runner=None, out=None, err=None):
        from gpuwm import runplan
        from gpuwm.live_products import LiveProducts

        self.root = Path(root)
        self.clock = clock
        self._geo_file = geo_file
        self._obs_files = list(obs_files or ())
        self._dx_m = dx_m
        self._parent_kind = parent_kind
        self._out = sys.stdout if out is None else out
        self._err = sys.stderr if err is None else err
        self.plan = {"run": self.root,
                     "wrfout_dir": self.root / FRAMES_DIRNAME,
                     "render": self.root / PICTURES_DIRNAME,
                     "render_products": (None if render_products is None
                                         else str(render_products))}
        self.events = runplan.EventStream(self.root / EVENTS_FILENAME,
                                          mirror=None)
        self.live_products = LiveProducts(
            self.plan, report=self._drawn, warn=self.warn, runner=runner)
        self.grid: ParentGrid | None = None
        self._refused: str | None = None
        self._told_left_out = False
        self.frames: list[Path] = []
        #: The frames on a whole hour, the only ones a window closes on.
        self._whole_hours: list[Path] = []

    # -- telling -----------------------------------------------------------

    def _say(self, text: str, *, err: bool = False) -> None:
        stream = self._err if err else self._out
        try:
            print(text, file=stream, flush=True)
        except Exception:  # noqa: BLE001 - a closed stream never fails a cycle
            pass

    def warn(self, code: str, message: str, **fields) -> None:
        self._say(f"cycle: pictures: {message}", err=True)
        try:
            self.events.emit("warning", code=code, message=message,
                             **{key: _plain(value)
                                for key, value in fields.items()})
        except Exception:  # noqa: BLE001 - telemetry never fails a cycle
            pass

    def _drawn(self, entry: Mapping[str, Any]) -> None:
        try:
            self.events.emit(
                "live_products_ready", domain=int(entry["domain"]),
                valid_time=_plain(entry["valid_time"]),
                frame=str(entry["frame"]), pictures=entry["pictures"],
                render_seconds=entry["render_seconds"],
                queued=entry["queued"])
        except Exception:  # noqa: BLE001 - telemetry never fails a cycle
            pass
        self._say(f"cycle: drew {Path(str(entry['frame'])).name}: "
                  f"{entry['pictures']} picture(s) in "
                  f"{float(entry['render_seconds']):.1f} s")

    # -- the hook ----------------------------------------------------------

    def boundary_completed(self, cycle_index: int,
                           parent_record: Mapping[str, Any],
                           receipt: Mapping[str, Any] | None = None
                           ) -> Path | None:
        """Write this boundary's frame and queue it; never raises."""

        try:
            return self._publish(int(cycle_index), parent_record)
        except Exception as error:  # noqa: BLE001 - never fail a cycle
            self.warn("cycle_frame_failed",
                      f"boundary {int(cycle_index)} was not written as a "
                      f"frame ({type(error).__name__}: {error}); its anchor "
                      "and receipt are unaffected",
                      cycle=int(cycle_index))
            return None

    def _publish(self, cycle_index: int, record: Mapping[str, Any]
                 ) -> Path | None:
        if self._refused is not None:
            return None
        if self.grid is None:
            grid, why = resolve_grid(record, geo_file=self._geo_file,
                                     obs_files=self._obs_files,
                                     dx_m=self._dx_m,
                                     parent_kind=self._parent_kind)
            if grid is None:
                self._refused = why
                self._say(f"cycle: pictures: none: {why}")
                try:
                    self.events.emit("warning", code="cycle_pictures_none",
                                     message=why, cycle=int(cycle_index))
                except Exception:  # noqa: BLE001 - telemetry never fails
                    pass
                return None
            self.grid = grid
            self._say(f"cycle: pictures: each boundary is drawn into "
                      f"{self.plan['render']} as it lands; grid "
                      f"{grid.shape[1]}x{grid.shape[0]} placed by "
                      f"{grid.source}, spacing {grid.dx_m:.0f} m "
                      f"({grid.dx_source})")
        snapshot, left_out = boundary_snapshot(record, self.grid)
        if left_out and not self._told_left_out:
            self._told_left_out = True
            self._say("cycle: pictures: not on the parent's grid, so not "
                      "drawn: " + ", ".join(left_out))
        path = self._write(cycle_index, snapshot)
        valid = self.clock.valid_time(cycle_index)
        valid_text = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.frames.append(path)
        if (valid.minute, valid.second, valid.microsecond) == (0, 0, 0):
            self._whole_hours.append(path)
        try:
            self.events.emit("output_committed", domain=DOMAIN,
                             valid_time=valid_text, path=str(path),
                             cycle=int(cycle_index))
        except Exception:  # noqa: BLE001 - telemetry never fails a cycle
            pass
        self.live_products.frame_committed(domain=DOMAIN,
                                           valid_time=valid_text, path=path)
        return path

    def _write(self, cycle_index: int, snapshot: dict) -> Path:
        from gpuwm.io.surface_wrfout import write_surface_wrfout
        from gpuwm.io.wrfout import wrfout_filename

        valid = self.clock.valid_time(cycle_index)
        directory = self.root / FRAMES_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)
        naive = _naive_utc(valid)
        final = directory / wrfout_filename(naive, DOMAIN)
        staging = directory / f".{final.name}.tmp.{os.getpid()}"
        staging.unlink(missing_ok=True)
        try:
            write_surface_wrfout(
                staging, snapshot,
                time_str=naive.strftime("%Y-%m-%d_%H:%M:%S"),
                dx=self.grid.dx_m, grid_id=DOMAIN,
                start_time=_naive_utc(self.clock.epoch_anchor),
                global_attrs={"GPUWM_CYCLE_INDEX": int(cycle_index),
                              "GPUWM_CYCLE_GRID_SOURCE": self.grid.source,
                              "GPUWM_CYCLE_DX_SOURCE": self.grid.dx_source},
                title=f"gpuwm cycle boundary {cycle_index}",
                composite_key=COMPOSITE_PLANE)
            os.replace(staging, final)
        finally:
            staging.unlink(missing_ok=True)
        return final

    # -- the end of the run ------------------------------------------------

    def finish(self) -> bool:
        """A cycle that completed: draw what drawing as it landed did not.

        The shared finalize stage stops the every-frame render, reads its
        record off disk and draws only the frames it cannot prove were
        drawn, exactly as it does for a forecast.
        """

        try:
            if self.grid is None:
                self.live_products.stop()
                return False
            from gpuwm.go_cli import (GoStageFailed, _render_stage,
                                      printable, render_command)

            try:
                return bool(_render_stage(self.plan, explain=False,
                                          observer=self, door="cycle",
                                          windows=self._windows()))
            except GoStageFailed as failure:
                from gpuwm.cycle.contracts import CycleRefusal

                raise CycleRefusal(
                    "the cycle completed, but its pictures were not all "
                    f"drawn: the render stage exited {failure.code}. Its "
                    f"frames are in {self.plan['wrfout_dir']}; draw them "
                    "with: " + printable(render_command(self.plan)),
                    render_exit=failure.code) from failure
        finally:
            self.close_events()

    def _windows(self) -> bool:
        """Whether the finalize stage draws windows, asked of the engine.

        WHAT BREAKAGE THIS PREVENTS (gate law): a boundary frame holds the
        parent's planes and none of the history fields a window folds
        (accumulated rainfall, 10 m wind, 2 m fields, updraft helicity
        maxima), so the finalize stage's windowed pass over an hourly
        cycle drew nothing and the renderer exited 1 ("batch render
        incomplete: rendered=0 skipped=49").  A real 3-cycle hourly run
        drew all 33 of its pictures as they landed and still ended on
        "FAILED render (exit 1)" and "cycle: stopped at render".

        The engine is asked over the boundaries on whole hours, which are
        the only frames a window closes on, and only when the request
        asks for a window and two such frames exist (with fewer the stage
        draws no window anyway).  A question the engine could not answer
        leaves the stage as it was.
        """

        from gpuwm.live_products import engine_draws_windows, windowed_request

        if (not windowed_request(self.plan.get("render_products"))
                or len(self._whole_hours) < 2):
            return True
        if engine_draws_windows(self._whole_hours,
                                beside=self.plan["render"]) is not False:
            return True
        self._say("cycle: pictures: no windowed picture is drawn (rainfall "
                  "totals, 1 h and run maxima): the boundaries hold none of "
                  "the fields those fold")
        return False

    def close(self, *, stopped: bool) -> None:
        """A cycle that did not complete.

        ``stopped`` is a user's stop: nothing more is drawn and the render
        in flight is ended.  A cycle that halted on its own finishes
        drawing the boundaries it completed.
        """

        live = self.live_products
        try:
            live.halt() if stopped else live.stop()
        except Exception:  # noqa: BLE001 - a stop never raises past here
            pass
        self.close_events()

    def close_events(self) -> None:
        try:
            self.events.close()
        except Exception:  # noqa: BLE001 - a closed stream never fails a cycle
            pass


def _naive_utc(when: datetime) -> datetime:
    if when.tzinfo is None:
        return when
    from datetime import timezone

    return when.astimezone(timezone.utc).replace(tzinfo=None)


def _plain(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def arm(args, clock, *, root, out=None, err=None) -> BoundaryPictures | None:
    """The pictures of one ``gpuwm cycle``, or ``None`` when none are drawn.

    ``--render-products none`` draws nothing; a renderer this install
    does not have is said once, with its remedy, and draws nothing.
    """

    from gpuwm.live_products import live_render_requested

    products = getattr(args, "render_products", None)
    if not live_render_requested(products):
        return None
    from gpuwm.go_cli import render_extra_missing

    missing = render_extra_missing()
    if missing is not None:
        stream = sys.stderr if err is None else err
        print(f"cycle: pictures: none: {missing}", file=stream, flush=True)
        if "remedy" not in str(missing):
            print("  remedy: gpuwm setup", file=stream, flush=True)
        return None
    return BoundaryPictures(
        root=root, clock=clock, render_products=products,
        geo_file=getattr(args, "parent_geo_file", None),
        obs_files=list(getattr(args, "placement_obs_file", None) or ()),
        dx_m=getattr(args, "parent_dx_m", None),
        parent_kind=getattr(args, "parent_kind", None), out=out, err=err)


__all__ = ["BoundaryPictures", "COMPOSITE_PLANE", "DOMAIN",
           "EVENTS_FILENAME", "FRAMES_DIRNAME", "MESH_PARENT_KINDS",
           "MESH_SENTENCE", "PICTURES_DIRNAME", "ParentGrid", "arm",
           "boundary_snapshot", "resolve_grid"]
