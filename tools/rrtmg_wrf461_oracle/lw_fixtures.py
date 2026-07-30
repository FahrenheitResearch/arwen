"""Reader for RRTMG LW oracle fixture files written by lw_binio.F90.

Record layout (little-endian int32 header, Fortran-order payload):
  name_len, name bytes, dtype (0=float32, 1=int32), rank, dims(rank), payload

Arrays come back as numpy arrays with Fortran-order axes preserved
(shape as declared in the Fortran, C-ordered via .reshape(dims, order='F')).
"""

from __future__ import annotations

import struct

import numpy as np

_DTYPES = {0: np.dtype("<f4"), 1: np.dtype("<i4")}


def read_fixture(path):
    """Return dict name -> scalar or ndarray for one fixture file."""
    out = {}
    with open(path, "rb") as fh:
        data = fh.read()
    off = 0
    n = len(data)
    while off < n:
        if off + 4 > n:
            raise ValueError(f"{path}: truncated header at {off}")
        (nlen,) = struct.unpack_from("<i", data, off)
        off += 4
        name = data[off:off + nlen].decode("ascii")
        off += nlen
        dtype_code, rank = struct.unpack_from("<ii", data, off)
        off += 8
        dims = struct.unpack_from(f"<{rank}i", data, off) if rank else ()
        off += 4 * rank
        dt = _DTYPES[dtype_code]
        count = 1
        for d in dims:
            count *= d
        nbytes = count * dt.itemsize
        if off + nbytes > n:
            raise ValueError(f"{path}: truncated payload for {name}")
        arr = np.frombuffer(data, dtype=dt, count=count, offset=off)
        off += nbytes
        if rank == 0:
            out[name] = arr[0]
        else:
            out[name] = arr.reshape(dims, order="F")
    return out


def read_fixture_prefix(path, prefix):
    """Read only records whose name starts with prefix (cheap scan)."""
    full = read_fixture(path)
    return {k: v for k, v in full.items() if k.startswith(prefix)}


if __name__ == "__main__":
    import sys

    fx = read_fixture(sys.argv[1])
    for k, v in fx.items():
        if np.ndim(v) == 0:
            print(f"{k} = {v}")
        else:
            print(f"{k}: shape={v.shape} min={v.min():.6g} max={v.max():.6g}")
