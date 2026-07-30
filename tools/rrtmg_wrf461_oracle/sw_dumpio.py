"""Reader for the SWD1 stream-binary dumps written by sw_dump_tables.F90 and
sw_fixture_driver.F90.

Format (BIG-endian: the oracle build uses -fconvert=big-endian for
RRTMG_SW_DATA, and gfortran applies it to all unformatted I/O):
    magic   : 4 bytes b"SWD1"
    records : name_len int32 | name bytes | dtype int32 (0=f32, 1=i32)
              | rank int32 | dims int32[rank] (Fortran order) | payload

Arrays are stored in Fortran (column-major) element order; they are returned
as numpy arrays with the declared Fortran shape (order="F"), so indexing in
Python matches the Fortran source 1:1 up to the 0/1 base offset.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

_MAGIC = b"SWD1"


def read_swd(path):
    """Read an SWD1 file -> dict[str, np.ndarray | np.float32 | np.int32]."""
    out = {}
    data = Path(path).read_bytes()
    if data[:4] != _MAGIC:
        raise ValueError(f"{path}: not an SWD1 file")
    off = 4
    n = len(data)
    while off < n:
        (name_len,) = struct.unpack_from(">i", data, off)
        off += 4
        name = data[off:off + name_len].decode("ascii")
        off += name_len
        dtype_code, rank = struct.unpack_from(">ii", data, off)
        off += 8
        dims = struct.unpack_from(f">{rank}i", data, off) if rank else ()
        off += 4 * rank
        count = 1
        for d in dims:
            count *= d
        dt = np.dtype(">f4") if dtype_code == 0 else np.dtype(">i4")
        arr = np.frombuffer(data, dtype=dt, count=count, offset=off)
        arr = arr.astype(arr.dtype.newbyteorder("="))
        off += 4 * count
        if rank == 0:
            out[name] = arr[0]
        else:
            out[name] = arr.reshape(dims, order="F")
    return out


def to_npz(swd_path, npz_path):
    """Convert an SWD1 file to a compressed .npz archive."""
    d = read_swd(swd_path)
    np.savez_compressed(npz_path, **d)
    return len(d)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        n_records = to_npz(sys.argv[1], sys.argv[2])
        print(f"wrote {sys.argv[2]}: {n_records} records")
    elif len(sys.argv) == 2:
        d = read_swd(sys.argv[1])
        for k, v in d.items():
            shape = getattr(v, "shape", ())
            print(k, shape if shape else v)
    else:
        print("usage: sw_dumpio.py in.swd [out.npz]")
        sys.exit(2)
