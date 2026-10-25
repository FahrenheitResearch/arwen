# SPDX-License-Identifier: Apache-2.0
"""A member's history frames in memory: the exact words the wrfout writer writes.

A resident member session keeps every ensemble member in one process across
analysis cycles.  The consumer of a member's state at a cycle instant (an
analysis that reads WRF-form fields) needs the frame the member's history
file would hold, without the file.  :class:`FrameSink` sits on the member's
history writer (:meth:`gpuwm.io.wrfout.PerDomainWrfoutWriters.attach_frame_sink`)
and is handed, on the writer thread, the composed host fields the file is
written from; it keeps a copy of the selected fields on WRF's own axes
(:func:`gpuwm.io.wrfout.wrf_axis_array`), so a :class:`MemberFrame`'s words are
the file's words by construction.  With ``write_files=False`` on the writer
the frame lands in the sink only.

:func:`frame_from_wrfout` reads the same record from a history file, for the
instants a session did not produce (the first cycle's background comes from
the spin-up legs' files) and for the equality check against a files run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Condition
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

SCHEMA = "gpuwm.member-frame/1"
WRF_TIME = "%Y-%m-%d_%H:%M:%S"

#: The thin set: the WRF-form fields a state-reading analysis takes from a
#: member (winds, theta, pressure and geopotential with their bases, dry mass,
#: moisture and the hydrometeor masses and numbers, simulated reflectivity,
#: the surface and land fields its observation operators read, and the
#: column's coordinate scalars).  Fields a run does not write are simply
#: absent from its frames.
THIN_FIELDS = (
    "U", "V", "W", "T", "P", "PB", "PH", "PHB", "MU", "MUB", "P_TOP", "PSFC",
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL",
    "QNICE", "QNRAIN", "QNCLOUD", "QNSNOW", "QNGRAUPEL", "QNHAIL", "QVGRAUPEL", "QVHAIL",
    "QNWFA", "QNIFA", "REFL_10CM",
    "T2", "Q2", "U10", "V10", "TSK", "SNOWC", "SNOWH", "SEAICE", "VEGFRA", "LAI", "ZNT", "TMN", "SNOW",
    "CLDFRA", "RE_CLOUD", "RE_ICE", "RE_SNOW", "SMOIS", "TSLB", "SH2O", "HGT", "LANDMASK", "XLAND",
    "IVGTYP", "ISLTYP", "LU_INDEX", "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "SINALPHA",
    "COSALPHA", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
    "ZNU", "ZNW", "C1H", "C2H", "C1F", "C2F", "C3H", "C4H", "C3F", "C4F", "RDNW", "RDN", "DNW", "DN",
    "Times",
)


@dataclass(frozen=True)
class MemberFrame:
    """One member's history record at one instant, as host arrays.

    ``arrays`` maps WRF names to the exact words the history writer writes for
    that instant (without the record axis); ``attrs`` are the file's global
    attributes.  ``member_id`` is the ensemble's global, 1-based member ID.
    """

    member_id: int
    grid_id: int
    valid_time: str
    arrays: Mapping[str, np.ndarray]
    attrs: Mapping[str, object] = field(default_factory=dict)
    source: str = "history-writer"
    #: Each field's spatial dimension names as the history file declares them (no ``Time``).
    dims: Mapping[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.member_id, bool) or int(self.member_id) < 1:
            raise ValueError("a member frame names a global, 1-based member ID")
        datetime.strptime(self.valid_time, WRF_TIME)
        object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))
        object.__setattr__(self, "dims", MappingProxyType({k: tuple(v) for k, v in dict(self.dims).items()}))

    @property
    def nbytes(self) -> int:
        return int(sum(int(a.nbytes) for a in self.arrays.values()))

    def receipt(self) -> dict:
        return {"schema": SCHEMA, "member_id": int(self.member_id), "grid_id": int(self.grid_id),
                "valid_time": self.valid_time, "source": self.source, "fields": sorted(self.arrays),
                "bytes": self.nbytes}


def frame_dims(name, shape, *, nx: int, ny: int, nz: int) -> tuple:
    """A field's spatial dimension names as the history writer declares them
    (:meth:`gpuwm.io.wrfout.WrfoutWriter._dims_for`'s table, the soil and the snow stacks)."""
    from gpuwm.io.wrfout import _SNOW_LAYER_FIELDS, _SNSO_LAYER_FIELDS, _SOIL_LAYER_FIELDS

    shape = tuple(int(x) for x in shape)
    if name in _SOIL_LAYER_FIELDS:
        return ("soil_layers_stag", "south_north", "west_east")
    if name in _SNOW_LAYER_FIELDS:
        return ("snow_layers_stag", "south_north", "west_east")
    if name in _SNSO_LAYER_FIELDS:
        return ("snso_layers_stag", "south_north", "west_east")
    table = {
        (): (), (nz,): ("bottom_top",), (nz + 1,): ("bottom_top_stag",),
        (nz, ny, nx): ("bottom_top", "south_north", "west_east"),
        (nz, ny, nx + 1): ("bottom_top", "south_north", "west_east_stag"),
        (nz, ny + 1, nx): ("bottom_top", "south_north_stag", "west_east"),
        (nz + 1, ny, nx): ("bottom_top_stag", "south_north", "west_east"),
        (ny, nx): ("south_north", "west_east"), (ny, nx + 1): ("south_north", "west_east_stag"),
        (ny + 1, nx): ("south_north_stag", "west_east"),
    }
    if shape not in table:
        raise ValueError(f"{name}: shape {shape} has no history dimension on a {nz}x{ny}x{nx} grid")
    return table[shape]


def _wrf_time(valid_time) -> str:
    if isinstance(valid_time, str):
        datetime.strptime(valid_time, WRF_TIME)
        return valid_time
    return valid_time.strftime(WRF_TIME)


class FrameSink:
    """Keeps one member's frames from its history writer.

    ``fields=None`` keeps every field the writer writes; otherwise the named
    ones the run writes.  ``keep`` frames are held, oldest dropped first (the
    session asks for each instant once, after the writer has drained).
    Called on writer threads; :meth:`frame` waits for an instant to land.
    """

    def __init__(self, member_id: int, *, fields: Iterable[str] | None = THIN_FIELDS, keep: int = 2):
        if int(keep) < 1:
            raise ValueError("a frame sink keeps at least one frame")
        self.member_id = int(member_id)
        self.fields = None if fields is None else tuple(fields)
        self.keep = int(keep)
        self._frames: dict[tuple[int, str], MemberFrame] = {}
        self._order: list[tuple[int, str]] = []
        self._condition = Condition()
        self.copied_bytes = 0
        self.frames_seen = 0

    def __call__(self, *, grid_id, valid_time, time_str=None, fields, nx, ny, nz, global_attrs=None):
        from gpuwm.io.wrfout import wrf_axis_array

        wanted = fields if self.fields is None else {name: fields[name] for name in self.fields if name in fields}
        arrays, dims = {}, {}
        for name, value in wanted.items():
            host = np.array(value, copy=True, order="C", subok=False)
            arrays[name] = wrf_axis_array(name, host, nx=int(nx), ny=int(ny), nz=int(nz))
            try:
                dims[name] = frame_dims(name, arrays[name].shape, nx=int(nx), ny=int(ny), nz=int(nz))
            except ValueError:
                pass        # a field outside the table (the fire stacks): its words are kept, its dims unnamed
        frame = MemberFrame(self.member_id, int(grid_id if grid_id is not None else 1),
                            time_str if time_str is not None else _wrf_time(valid_time), arrays,
                            dict(global_attrs or {}), dims=dims)
        key = (frame.grid_id, frame.valid_time)
        with self._condition:
            self._frames[key] = frame
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            while len(self._order) > self.keep:
                self._frames.pop(self._order.pop(0), None)
            self.copied_bytes += frame.nbytes
            self.frames_seen += 1
            self._condition.notify_all()

    def frame(self, valid_time, *, grid_id: int = 1, timeout: float | None = None) -> MemberFrame:
        key = (int(grid_id), _wrf_time(valid_time))
        with self._condition:
            if not self._condition.wait_for(lambda: key in self._frames, timeout=timeout):
                raise TimeoutError(f"member {self.member_id}: no frame at {key[1]} for d{key[0]:02d}")
            return self._frames[key]

    def has(self, valid_time, *, grid_id: int = 1) -> bool:
        with self._condition:
            return (int(grid_id), _wrf_time(valid_time)) in self._frames


def frame_from_wrfout(path, member_id: int, *, fields: Iterable[str] | None = THIN_FIELDS,
                      record: int = 0) -> MemberFrame:
    """A history file's record as a :class:`MemberFrame` (the words on disk)."""
    from gpuwm.netcdf_bridge import open_dataset

    path = Path(path)
    with open_dataset(path) as dataset:
        if fields == "all":
            fields = None
        names = list(dataset.variables) if fields is None else [n for n in fields if n in dataset.variables]
        words = dataset.stored_words([*names, "Times"])
        arrays, dims = {}, {}
        for name in names:
            variable = dataset.variables[name]
            if variable.dimensions and variable.dimensions[0] == "Time":
                arrays[name] = words[name][record]
                dims[name] = tuple(variable.dimensions[1:])
            else:
                arrays[name] = words[name]
                dims[name] = tuple(variable.dimensions)
        times = words["Times"][record]
        valid = b"".join(np.asarray(times).tolist()).decode("ascii").rstrip("\x00")
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        grid_id = int(attrs.get("GRID_ID", 1))
    return MemberFrame(int(member_id), grid_id, valid, arrays, attrs, source=str(path), dims=dims)


def compare_frames(frame: MemberFrame, reference: MemberFrame) -> dict:
    """Word-for-word comparison of two frames' common fields (raw bytes, -0.0 and NaN payloads counted)."""
    rows = {}
    for name in sorted(set(frame.arrays) & set(reference.arrays)):
        a, b = np.asarray(frame.arrays[name]), np.asarray(reference.arrays[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            rows[name] = {"n": int(a.size), "differing_words": None,
                          "shape": [list(a.shape), list(b.shape)], "dtype": [str(a.dtype), str(b.dtype)]}
            continue
        if a.dtype.kind in "fiu" and a.size:
            wa = np.ascontiguousarray(a).view(np.uint8).reshape(a.size, -1)
            wb = np.ascontiguousarray(b).view(np.uint8).reshape(b.size, -1)
            differing = int(np.any(wa != wb, axis=1).sum())
        else:
            differing = int(np.sum(a != b))
        row = {"n": int(a.size), "differing_words": differing}
        if differing and a.dtype.kind == "f":
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            row["max_abs"] = float(np.nanmax(delta))
        rows[name] = row
    only = {"frame_only": sorted(set(frame.arrays) - set(reference.arrays)),
            "reference_only": sorted(set(reference.arrays) - set(frame.arrays))}
    differing = sum(int(r["differing_words"] or 0) for r in rows.values()) + sum(
        1 for r in rows.values() if r["differing_words"] is None)
    same_fields = not any(only.values())
    return {"valid_time": frame.valid_time, "member_id": int(frame.member_id), "fields": rows,
            "compared_words": int(sum(r["n"] for r in rows.values())), "differing_words": int(differing),
            **only, "verdict": "bitwise" if differing == 0 and same_fields else "DIFFER"}


__all__ = ["FrameSink", "MemberFrame", "SCHEMA", "THIN_FIELDS", "compare_frames", "frame_dims",
           "frame_from_wrfout"]
