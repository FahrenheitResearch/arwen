"""Binary I/O shared by the RRTM LW column oracle's generator and comparer.

Input (``columns.bin``), little-endian, read by ``run_rrtm_lw.F90``::

    int32 magic 0x52524C57, int32 version 1, int32 nsets
    per set:
      int32 ncol, nz, yr, icloud, warm_rain, f_qv, f_qc, f_qr, f_qi, f_qs, f_qg
      float32 p_top (Pa), julian
      level fields p8w, t8w            (nz+1) x ncol, bottom level first
      mass fields  p3d, pi3d, dz8w, t3d, rho3d, qv, qc, qr, qi, qs, qg,
                   cldfra              nz x ncol, bottom level first
      tsk, emiss                       ncol

i.e. every field is C-order ``(nlev, ncol)`` -- one model level of every
column at a time, the order the Fortran reads it into ``a(:, k, 1)``.

Output (``wrf-out.bin``) is a sequence of tagged records written by the
driver: a 32-byte name, int32 kind (0 float32, 1 int32, 2 float64), int32
ndim, int32 dims (C order), then the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 0x52524C57
LEVEL_FIELDS = ("p8w", "t8w")
MASS_FIELDS = ("p3d", "pi3d", "dz8w", "t3d", "rho3d", "qv", "qc", "qr",
               "qi", "qs", "qg", "cldfra")
SURFACE_FIELDS = ("tsk", "emiss")
FLAG_NAMES = ("f_qv", "f_qc", "f_qr", "f_qi", "f_qs", "f_qg")


@dataclass
class ColumnSet:
    """One RRTMLWRAD call: ``ncol`` columns sharing ``nz`` and ``p_top``.

    Every array is BOTTOM-UP along its level axis and shaped
    ``(ncol, nlev)``, the WRF 3-D array's ``(i, k)`` with ``j`` dropped.
    """

    name: str
    p_top: float
    yr: int
    icloud: int
    warm_rain: bool
    flags: tuple[bool, ...]
    fields: dict[str, np.ndarray]
    labels: list[str] = field(default_factory=list)
    julian: float = 180.0

    @property
    def ncol(self) -> int:
        return int(self.fields["t3d"].shape[0])

    @property
    def nz(self) -> int:
        return int(self.fields["t3d"].shape[1])


def write_inputs(path: str | Path, sets: list[ColumnSet]) -> None:
    with open(path, "wb") as out:
        out.write(np.asarray([MAGIC, 1, len(sets)], dtype="<i4").tobytes())
        for cs in sets:
            ncol, nz = cs.ncol, cs.nz
            header = [ncol, nz, cs.yr, cs.icloud, int(cs.warm_rain),
                      *[int(f) for f in cs.flags]]
            out.write(np.asarray(header, dtype="<i4").tobytes())
            out.write(np.asarray([cs.p_top, cs.julian], dtype="<f4").tobytes())
            for name in LEVEL_FIELDS:
                arr = np.asarray(cs.fields[name], dtype=np.float32)
                if arr.shape != (ncol, nz + 1):
                    raise ValueError(f"{name}: {arr.shape} != {(ncol, nz + 1)}")
                out.write(np.ascontiguousarray(arr.T).astype("<f4").tobytes())
            for name in MASS_FIELDS:
                arr = np.asarray(cs.fields[name], dtype=np.float32)
                if arr.shape != (ncol, nz):
                    raise ValueError(f"{name}: {arr.shape} != {(ncol, nz)}")
                out.write(np.ascontiguousarray(arr.T).astype("<f4").tobytes())
            for name in SURFACE_FIELDS:
                arr = np.asarray(cs.fields[name], dtype=np.float32)
                if arr.shape != (ncol,):
                    raise ValueError(f"{name}: {arr.shape} != {(ncol,)}")
                out.write(arr.astype("<f4").tobytes())


def read_inputs(path: str | Path) -> list[ColumnSet]:
    data = Path(path).read_bytes()
    pos = 0

    def take(dtype, count):
        nonlocal pos
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
        pos += arr.nbytes
        return arr

    magic, version, nsets = take("<i4", 3)
    if magic != MAGIC or version != 1:
        raise ValueError("not an RRTM LW oracle input file")
    sets = []
    for index in range(int(nsets)):
        h = take("<i4", 11)
        ncol, nz, yr, icloud, warm = (int(v) for v in h[:5])
        flags = tuple(bool(v) for v in h[5:])
        p_top, julian = (float(v) for v in take("<f4", 2))
        fields = {}
        for name in LEVEL_FIELDS:
            fields[name] = take("<f4", ncol * (nz + 1)).reshape(
                nz + 1, ncol).T.astype(np.float32)
        for name in MASS_FIELDS:
            fields[name] = take("<f4", ncol * nz).reshape(
                nz, ncol).T.astype(np.float32)
        for name in SURFACE_FIELDS:
            fields[name] = take("<f4", ncol).astype(np.float32)
        sets.append(ColumnSet(name=f"set{index + 1}", p_top=p_top, yr=yr,
                              icloud=icloud, warm_rain=bool(warm),
                              flags=flags, fields=fields, julian=julian))
    if pos != len(data):
        raise ValueError("trailing bytes in oracle input")
    return sets


_KINDS = {0: "<f4", 1: "<i4", 2: "<f8"}


def read_outputs(path: str | Path) -> tuple[dict, list[dict]]:
    """Return ``(tables, sets)``; each set is ``{name: array}``."""
    data = Path(path).read_bytes()
    pos = 0

    def take(dtype, count):
        nonlocal pos
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
        pos += arr.nbytes
        return arr

    magic, version, nsets = take("<i4", 3)
    if magic != MAGIC or version != 1:
        raise ValueError("not an RRTM LW oracle output file")
    tables: dict[str, np.ndarray] = {}
    sets: list[dict] = []
    current: dict | None = None
    target = None
    while pos < len(data):
        name = data[pos:pos + 32].decode("ascii").strip()
        pos += 32
        kind, ndim = (int(v) for v in take("<i4", 2))
        dims = tuple(int(v) for v in take("<i4", ndim))
        values = take(_KINDS[kind], int(np.prod(dims))).reshape(dims).copy()
        if name == "TABLES":
            target = tables
            continue
        if name == "SET":
            current = {"_header": values}
            target = current
            continue
        if name == "ENDSET":
            sets.append(current)
            current = None
            target = None
            continue
        if name == "END":
            break
        if target is None:
            raise ValueError(f"record {name} outside a block")
        target[name] = values
    if len(sets) != int(nsets):
        raise ValueError(f"expected {nsets} sets, read {len(sets)}")
    return tables, sets
