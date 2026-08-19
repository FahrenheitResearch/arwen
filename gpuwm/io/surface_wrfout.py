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

from pathlib import Path

import numpy as np

#: Global attribute declaring what this file's vertical axis is.
VERTICAL_ATTR = "GPUWM_VERTICAL"

#: Its value for a surface snapshot.
SURFACE_SNAPSHOT = "surface-snapshot"

#: Snapshot key -> wrfout variable, for the plain 2-D surface fields.
#:
#: Only fields the renderer's import lane actually consumes are mapped: a
#: variable it has no selector for lands in the store as a generic
#: ``var:`` plane, which is useful, but a mapping table full of names
#: nothing reads is a table nobody can audit.
SURFACE_FIELDS: dict[str, str] = {
    "T2": "T2",
    "U10": "U10",
    "V10": "V10",
    "PSFC": "PSFC",
    "RAINC": "RAINC",
    "RAINNC": "RAINNC",
    "SNOWNC": "SNOWNC",
    "PBLH": "PBLH",
    "HFX": "HFX",
    "SWDOWN": "SWDOWN",
    "TSK": "TSK",
    "Q2": "Q2",
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
                         composite_key: str = "REFL_COMPOSITE") -> Path:
    """One wrfout frame from a 2-D snapshot; returns the path written.

    ``snapshot`` is the lane's own dict.  ``XLAT``/``XLONG`` (or ``LAT``/
    ``LON``) are required -- without them the file has no geolocation and
    the renderer would have nothing to project -- and everything else is
    written when present.

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
    """

    from gpuwm.io.wrfout import WrfoutWriter

    path = Path(path)
    resolved: dict[str, np.ndarray] = {}
    latitude = _first_present(snapshot, ("XLAT", "LAT"))
    longitude = _first_present(snapshot, ("XLONG", "LON"))
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

    for key, name in SURFACE_FIELDS.items():
        if key in snapshot:
            resolved[name] = _plane(snapshot[key], ny, nx, key)
    for key, name in GEOLOCATION_FIELDS.items():
        if name in resolved or key not in snapshot:
            continue
        resolved[name] = _plane(snapshot[key], ny, nx, key)
    resolved.setdefault("HGT", np.zeros((ny, nx), np.float32))
    # The wrfout import reads these two to rotate grid-relative winds into
    # earth-relative ones.  A snapshot that does not carry the rotation is
    # declaring an unrotated grid, which is what the identity pair says.
    resolved.setdefault("SINALPHA", np.zeros((ny, nx), np.float32))
    resolved.setdefault("COSALPHA", np.ones((ny, nx), np.float32))

    composite = snapshot.get(composite_key)
    if composite is not None:
        resolved["REFL_10CM"] = _plane(
            composite, ny, nx, composite_key)[None, :, :]
    # The importer's preflight wants the mass-coordinate pair; a surface
    # snapshot has no profile, so they are the honest zeros of a file that
    # declares itself surface-only rather than invented values.
    resolved.setdefault("T", np.zeros((1, ny, nx), np.float32))
    resolved.setdefault("MU", np.zeros((ny, nx), np.float32))

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
    return path


def _first_present(snapshot: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in snapshot:
            return snapshot[key]
    return None


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
