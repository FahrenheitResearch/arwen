"""Write a real wrfout frame from a 2-D surface snapshot.

Two lanes in this tree publish their forecasts as ``.npz`` and nothing
else, so the production renderer -- which reads wrfouts -- could not draw
them at all and both grew a matplotlib renderer of their own:

* ``tilestream/run_bigdomain.py`` writes ``bigdom_<n>_<tag>.npz``
  (``bigdomain_render.py``: *"cannot be pointed at a pinned host store, so
  it is not the tool for a domain that never becomes a wrfout"*);
* ``tools/da_cycle_prepared.py`` writes ``cycle/composites/legNN_*.npz``
  carrying ``refl_colmax``.

The fix is a writer, not a new reader.  ``.npz`` is a container with no
geolocation contract and no schema; teaching the Rust renderer to read it
would be exactly the per-format adapter the arbitrary-acceptance test
forbids, and the producing lanes already hold everything a wrfout needs.
``tilestream/output.py`` and ``tilestream/run_case_hrrr.py`` write real
wrfouts from the pinned store already; this is the same move for the two
lanes whose snapshot is 2-D.

## What the vertical axis means here

A surface snapshot has no profile, so these files are written with
``bottom_top`` of length 1 and the composite reflectivity stored as the
single ``REFL_10CM`` level.  The renderer's composite is the column
maximum, and the column maximum of a one-level column is that level, so
the product it draws is exactly the composite the lane computed -- not an
approximation of it.

That is a real limitation and it is stated in the file: ``GPUWM_VERTICAL``
is set to ``surface-snapshot`` and ``TITLE`` says so, so a reader who
opens one of these looking for a sounding finds the answer in the file
rather than in a wrong plot.  Anything that needs a profile must be
written from the model state, which is what
:class:`tilestream.output.StoreHistoryWriter` is for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

#: Global attribute declaring what this file's vertical axis is.
VERTICAL_ATTR = "GPUWM_VERTICAL"

#: Its value for a surface snapshot.
SURFACE_SNAPSHOT = "surface-snapshot"

#: Snapshot key -> wrfout variable, for the keys whose wrfout name is NOT
#: their own.
#:
#: THE SHAPE IS THE CONTRACT HERE, not this table.  Every other entry a
#: snapshot carries reaches the file under its own key, by shape: see
#: :func:`write_surface_wrfout`.  So a producing lane that grows a field
#: publishes it without a row here, and this map holds only the renames,
#: because a rename is the one thing a shape cannot state.
#:
#: What used to stand here was a row per plain surface field, on the
#: premise that a name the renderer has no selector for is a name nothing
#: reads.  That premise was wrong in both directions and the table drifted
#: from both lanes: ``WrfoutWriter`` types an unknown ``(ny, nx)`` field
#: as ``f4`` on its own (``_dims_for``'s ``(ny, nx)`` row) and the render
#: door draws any 2-D plane a wrfout carries (``--products var:<name>``,
#: ``--list-products``), so an unlisted name was not unread, it was
#: DROPPED -- which is what happened to ``tilestream.bigdomain``'s
#: ``COSZEN`` on every frame -- while the table carried a ``Q2`` row
#: neither lane produces.
RENAME: dict[str, str] = {
    # Column extremes of vertical velocity, under WRF's OWN names for
    # them.  A surface snapshot has no profile, so these are the only way
    # the updraft reaches a panel at all -- and they are what
    # ``tilestream/bigdomain_render.py`` drew as its ``wmax`` product, the
    # last weather field that lane had no production renderer for.
    #
    # WRF's spelling rather than an invented one: `W_UP_MAX`/`W_DN_MAX`
    # are the Registry's names for the column maximum updraft and
    # downdraft, so a reader who knows wrfout knows these without being
    # told.  What differs from WRF is the AVERAGING WINDOW -- WRF's are
    # running maxima between history writes and these are the snapshot's
    # own instant -- and that divergence is stated in the file by
    # ``GPUWM_W_EXTREME_SEMANTICS`` rather than left for a reader to
    # assume.
    "WMAX": "W_UP_MAX",
    "WMIN": "W_DN_MAX",
}

#: Global attribute stating what ``W_UP_MAX``/``W_DN_MAX`` mean here.
W_EXTREME_ATTR = "GPUWM_W_EXTREME_SEMANTICS"

#: Its value.
W_EXTREME_INSTANTANEOUS = (
    "INSTANTANEOUS column extremes of vertical velocity at this frame's "
    "valid time, NOT WRF's running maximum between history writes")

#: The global attributes ``rw_wrfbatch`` reads a run origin from, most
#: authoritative first.
#:
#: It needs one of them.  Without it the renderer refuses the whole file
#: -- *"has no sound WRF run origin: START_DATE unavailable;
#: SIMULATION_START_DATE unavailable; XTIME fallback failed"* -- because
#: every product's lead time is measured from the origin, and a frame with
#: no origin has no lead.  A surface snapshot carries none of the three on
#: its own, so this writer requires the producing lane to state it.
ORIGIN_ATTRS = ("SIMULATION_START_DATE", "START_DATE")

#: Snapshot keys that carry the grid rather than a forecast field.
GEOLOCATION_FIELDS: dict[str, str] = {
    "LAT": "XLAT",
    "LON": "XLONG",
    "XLAT": "XLAT",
    "XLONG": "XLONG",
    "HT": "HGT",
    "HGT": "HGT",
}


#: What a netCDF variable name may be, so a snapshot key that cannot be one
#: is reported rather than handed to the library as an error with no name.
#:
#: A carrier label is free-form on the producing side: ``bigdomain.snapshot``
#: builds ``LAT``/``LON`` out of ``radiation/latitude_deg``, and a future
#: carrier could arrive with the slash still on it.
_NETCDF_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


#: The concrete ``Path`` class on this platform, which is what
#: :class:`SurfaceWrfoutWrite` extends.  ``pathlib.Path`` itself is the
#: abstract front door and carries no flavour, so a subclass of it cannot
#: be instantiated on every supported interpreter; its concrete form can.
_PATH = type(Path())


class SurfaceWrfoutWrite(_PATH):
    """The file a write produced, and what did and did not reach it.

    It IS the path.  ``os.fspath``, ``str()``, ``.name``, ``.stat()`` and
    every other ``Path`` operation work on it unchanged, so a caller that
    wants only the file keeps working without being edited -- which is the
    difference between a writer that reports more and a signature change
    that forces every call site in the tree to move at once.

    ``passed_through`` is the snapshot keys published under their own name
    by the shape rule; ``skipped`` maps a key that did NOT reach the file
    to the reason, so a lane can say what it lost instead of losing it
    quietly.  The ``*_units`` siblings and the bookkeeping scalars are in
    neither: they are not fields, and naming them every frame would bury
    the one entry that matters.

    Both default to empty at CLASS level, so a derived path -- ``.parent``,
    ``.with_suffix(...)``, anything ``pathlib`` builds by copying the type
    -- answers them rather than raising for an attribute the copy never
    got.  ``skipped`` defaults to a read-only mapping so the empty default
    cannot be mutated into shared state.
    """

    passed_through: tuple[str, ...] = ()
    skipped: Mapping[str, str] = MappingProxyType({})

    @classmethod
    def _record(cls, path, passed_through, skipped) -> "SurfaceWrfoutWrite":
        """The written path, carrying what did and did not reach it."""

        written = cls(path)
        written.passed_through = tuple(passed_through)
        written.skipped = MappingProxyType(dict(skipped))
        return written

    @property
    def path(self) -> Path:
        """This same file as a plain :class:`~pathlib.Path`.

        For a caller that wants to hand the file on without the report
        riding along with it.
        """

        return Path(str(self))

    def skipped_report(self) -> str:
        """One line naming what did not reach the file, or ``""``.

        Every lane prints this rather than formatting its own, so two
        doors cannot come to describe one write differently.
        """

        return ", ".join(f"{name} ({why})"
                         for name, why in sorted(self.skipped.items()))


class SurfaceSnapshotRefusal(ValueError):
    """A snapshot that cannot become a wrfout, with the reason named."""


def _plane(values, ny: int, nx: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (ny, nx):
        raise SurfaceSnapshotRefusal(
            f"{name} is {array.shape}; the grid is ({ny}, {nx}).  A frame "
            "with a field on a different grid than its own coordinates "
            "would place that field somewhere it was not computed")
    return array


def write_surface_wrfout(path, snapshot: dict, *, time_str: str,
                         dx: float, dy: float | None = None,
                         global_attrs: dict | None = None,
                         grid_id: int | None = None,
                         start_time=None,
                         title: str = "gpuwm surface snapshot",
                         composite_key: str = "REFL_COMPOSITE"
                         ) -> SurfaceWrfoutWrite:
    """One wrfout frame from a 2-D snapshot; what reached it and what did not.

    ``snapshot`` is the lane's own dict.  ``XLAT``/``XLONG`` (or ``LAT``/
    ``LON``) are required -- without them the file has no geolocation and
    the renderer would have nothing to project -- and everything else is
    written when present.

    ## What reaches the file

    THE SHAPE DECIDES, not a table of names.  After the geolocation and
    the composite are placed, every remaining entry that is a 2-D array of
    this frame's own ``(ny, nx)`` grid is published under its own key
    (:data:`RENAME` supplies the name for the two keys whose wrfout
    spelling differs).  ``WrfoutWriter`` types an unknown ``(ny, nx)``
    field as ``f4`` and the render door draws any 2-D plane a wrfout
    carries, so there is nothing a per-field row would add -- and a lane
    that grows a carrier gets it in the wrfout without editing this file.

    Everything else is REPORTED, not dropped in silence:
    :class:`SurfaceWrfoutWrite` carries ``passed_through`` and a
    ``skipped`` map of key to reason, and the lanes print the second.
    The ``*_units`` siblings and the bookkeeping scalars (``elapsed_s``,
    ``nx``, ``ny``, ``nz``, ``dx``, ``dt``, and any other 0-d entry) are
    in neither: they are not fields, and naming them on every frame would
    bury the entry that matters.

    ``T`` is placed BEFORE the shape rule, because it is the file's
    structure rather than a stand-in for a measurement: a snapshot entry of
    the same name is reported as already written instead of replacing the
    mass coordinate with a surface plane.  The rotation pair
    (``SINALPHA``/``COSALPHA``) goes in after it, so a lane that knows its
    grid rotation publishes it.

    ## What the file does not invent

    A field the snapshot does not carry is not written as zeros.  THE
    BREAKAGE: this writer used to add ``HGT`` and ``MU`` as zero planes,
    and the renderer draws every plane a wrfout carries, so every frame
    drawn with the default product set got a Terrain Height map reading
    0 m across mountains and flat-zero ``wrf_hgt``, ``wrf_terrain`` and
    ``wrf_mu`` pictures: on a ``gpuwm cycle`` boundary, four of the eleven
    pictures were of fields the frame did not hold.  MEASURED on ``rw_wrfbatch`` (2.8 line): a frame without the two
    reads and draws exactly its own planes (exit 0), and one without ``T``
    is refused, so ``T`` is the one placeholder the file needs.  A lane
    that has its terrain or column mass passes it (``HGT``/``HT``, ``MU``)
    and it is written as the measurement it is.

    A 2-D array on a grid that is not this frame's is the one refusal, and
    it is :func:`_plane`'s, unchanged: it names a field placed where it
    was not computed.

    The RUN ORIGIN is required too, and by the same argument.  Either
    ``global_attrs`` already carries one of :data:`ORIGIN_ATTRS` (which is
    what :func:`gpuwm.io.wrfout.wrf_global_attrs` puts there) or
    ``start_time`` states it as a ``datetime``.  Neither is a refusal
    rather than a file, because a frame with no origin is refused by
    ``rw_wrfbatch`` after it is written and the lane that wrote it has
    already reported success.

    The composite reflectivity, if the snapshot has one, becomes the
    single ``REFL_10CM`` level; see the module docstring for why that is
    exact rather than approximate.

    ## What comes back

    The written path, and nothing a caller has to unwrap:
    :class:`SurfaceWrfoutWrite` extends ``Path``, so a caller that wants
    only the file uses the return value as the file, unchanged, and one
    that wants the report reads ``passed_through``/``skipped`` off the
    same object.
    """

    from gpuwm.io.wrfout import WrfoutWriter

    path = Path(path)
    resolved: dict[str, np.ndarray] = {}
    # Snapshot keys already published under some name, so the shape rule
    # below does not publish them a second time.
    consumed: set[str] = set()
    latitude_key, latitude = _first_present(snapshot, ("XLAT", "LAT"))
    longitude_key, longitude = _first_present(snapshot, ("XLONG", "LON"))
    if latitude is None or longitude is None:
        raise SurfaceSnapshotRefusal(
            f"{path.name}: the snapshot carries no latitude/longitude "
            f"(looked for XLAT/LAT and XLONG/LON; it has "
            f"{', '.join(sorted(k for k in snapshot if not k.endswith('_units')))}).  "
            "A wrfout without coordinates renders nothing, and guessing a "
            "grid from the array shape would put the forecast on the wrong "
            "part of the Earth")
    latitude = np.asarray(latitude, dtype=np.float32)
    longitude = np.asarray(longitude, dtype=np.float32)
    if latitude.ndim != 2 or latitude.shape != longitude.shape:
        raise SurfaceSnapshotRefusal(
            f"{path.name}: latitude is {latitude.shape} and longitude is "
            f"{longitude.shape}; both must be the same 2-D mass grid")
    ny, nx = latitude.shape
    resolved["XLAT"] = latitude
    resolved["XLONG"] = longitude
    consumed.update({latitude_key, longitude_key})

    for key, name in RENAME.items():
        if key in snapshot:
            resolved[name] = _plane(snapshot[key], ny, nx, key)
            consumed.add(key)
    for key, name in GEOLOCATION_FIELDS.items():
        if name in resolved or key not in snapshot:
            continue
        resolved[name] = _plane(snapshot[key], ny, nx, key)
        consumed.add(key)

    composite = snapshot.get(composite_key)
    if composite is not None:
        resolved["REFL_10CM"] = _plane(
            composite, ny, nx, composite_key)[None, :, :]
        consumed.add(composite_key)

    # The renderer opens a wrfout by its mass coordinate and refuses one
    # without ``T``; a surface snapshot has no profile, so this is the
    # stated zero of a file that says it is surface-only.  ``T`` is 3-D,
    # (1, ny, nx), so it is never drawn as a plane.
    #
    # It goes in BEFORE the shape rule: it is STRUCTURAL, and a snapshot
    # that happened to carry a 2-D ``T`` would otherwise replace the mass
    # coordinate with a surface plane the renderer would read as a profile
    # it is not.  Placed first, such an entry is REPORTED in ``skipped``
    # as already written instead, so the lane is told.
    #
    # No ``MU`` and no ``HGT`` stand-in (see "What the file does not
    # invent" above): the renderer needs neither to read the file and
    # draws both, as a flat column mass and a 0 m terrain map.
    resolved.setdefault("T", np.zeros((1, ny, nx), np.float32))

    passed_through, skipped = _pass_through(
        snapshot, resolved, consumed, ny, nx)

    # These two go in LAST, so a snapshot that carries one of them itself
    # wins over the placeholder rather than being overwritten by it.
    # The wrfout import reads these two to rotate grid-relative winds into
    # earth-relative ones.  A snapshot that does not carry the rotation is
    # declaring an unrotated grid, which is what the identity pair says.
    resolved.setdefault("SINALPHA", np.zeros((ny, nx), np.float32))
    resolved.setdefault("COSALPHA", np.ones((ny, nx), np.float32))

    attrs = dict(global_attrs or {})
    attrs[VERTICAL_ATTR] = SURFACE_SNAPSHOT
    if "W_UP_MAX" in resolved or "W_DN_MAX" in resolved:
        attrs[W_EXTREME_ATTR] = W_EXTREME_INSTANTANEOUS
    if grid_id is not None:
        attrs.setdefault("GRID_ID", int(grid_id))
    if not any(attrs.get(name) for name in ORIGIN_ATTRS):
        if start_time is None:
            raise SurfaceSnapshotRefusal(
                f"{path.name}: the snapshot declares no run origin.  "
                f"rw_wrfbatch measures every product's lead time from "
                f"{'/'.join(ORIGIN_ATTRS)} and falls back to XTIME, and a "
                f"surface snapshot carries none of the three -- so this "
                f"file would be written, reported as a success, and then "
                f"refused by the renderer it exists to feed ('has no sound "
                f"WRF run origin').  Pass start_time=<datetime>, or "
                f"global_attrs=wrf_global_attrs(grid, start_time)")
        stamp = start_time.strftime("%Y-%m-%d_%H:%M:%S")
        for name in ORIGIN_ATTRS:
            attrs[name] = stamp

    with WrfoutWriter(path, nx=nx, ny=ny, nz=1, dx=float(dx),
                      dy=float(dy if dy is not None else dx),
                      title=title, global_attrs=attrs) as writer:
        writer.write_frame(time_str, resolved)
    return SurfaceWrfoutWrite._record(path, passed_through, skipped)


def _pass_through(snapshot: dict, resolved: dict, consumed: set, ny: int,
                  nx: int) -> tuple[list[str], dict[str, str]]:
    """Publish every remaining 2-D grid-shaped entry under its own key.

    Returns the keys published and a map of key to why-not for the rest.
    A 0-d entry is a bookkeeping scalar (``elapsed_s``, ``nx``, ``dx``;
    and they come back 0-d, not Python floats, because
    ``tilestream/bigdomain_render.py`` reloads a snapshot with
    ``dict(np.load(...))``) and a ``*_units`` sibling is a colour-bar
    label, so neither is reported: they were never candidates.

    A 2-D entry is placed by :func:`_plane`, which means a 2-D array on a
    grid that is not this frame's is REFUSED by name rather than reported.
    That is the same check the named fields have always had and the same
    breakage it has always named: a field placed where it was not
    computed.  It also keeps an odd shape away from ``WrfoutWriter``'s
    dimension table, which indexes by shape and would raise a bare
    ``KeyError`` with no field name attached.
    """

    passed_through: list[str] = []
    skipped: dict[str, str] = {}
    for key, value in snapshot.items():
        if key in consumed or key.endswith("_units"):
            continue
        array = np.asarray(value)
        if array.ndim == 0:
            continue
        target = RENAME.get(key, GEOLOCATION_FIELDS.get(key, key))
        if target in resolved:
            skipped[key] = f"already written as {target}"
        elif _NETCDF_NAME.fullmatch(key) is None:
            skipped[key] = "not a netCDF variable name"
        elif array.dtype.kind not in "biuf":
            skipped[key] = f"dtype {array.dtype} is not a numeric field"
        elif array.ndim != 2:
            skipped[key] = (f"is {array.ndim}-D {array.shape}, not a 2-D "
                            f"field on this frame's ({ny}, {nx}) grid; "
                            "this file declares itself surface-only")
        else:
            resolved[key] = _plane(array, ny, nx, key)
            passed_through.append(key)
    return passed_through, skipped


def _first_present(snapshot: dict, keys: tuple[str, ...]):
    """The first of ``keys`` the snapshot has, as ``(key, value)``.

    The key comes back too because the caller has to record which spelling
    it consumed: ``LAT`` and ``XLAT`` are the same field, and the one that
    lost must not then be published a second time by the shape rule.
    """

    for key in keys:
        if key in snapshot:
            return key, snapshot[key]
    return None, None


def snapshot_wrfout_path(npz_path) -> Path:
    """The wrfout that goes beside an ``.npz`` snapshot.

    Beside it rather than instead of it: the ``.npz`` is what the lane's
    own analysis code already reads, and this change is additive.  The
    name keeps the ``wrfout_`` prefix and a ``dNN`` token because that is
    what ``rw_wrfbatch``'s domain-token reader looks for when the file
    declares no ``GRID_ID``.
    """

    npz_path = Path(npz_path)
    return npz_path.with_name(f"wrfout_{npz_path.stem}.nc")
