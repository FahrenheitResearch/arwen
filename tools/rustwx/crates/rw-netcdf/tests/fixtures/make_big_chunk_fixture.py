#!/usr/bin/env python3
"""Build big-chunk.nc4: zlib chunks larger than the reader's chunk cache.

NCAR's v4.4 CONUS 2.5 km restart stores each 3D field as 1 x 49 x 600 x 750
zlib+shuffle chunks of 88 MB, larger than the 64 MiB chunk cache, so a
reader that walks such a variable in row slabs inflates every chunk once
per slab.  This fixture has the same shape of problem at a size the repo
can carry: U(Time=1, bottom_top=17, south_north=2000, west_east=1000)
float32 in two 1 x 17 x 1000 x 1000 chunks (68 MB each), deflate plus
shuffle as WRF writes (level 9 rather than WRF's 2, to keep the file
small).  Values are exact in float32:
U[0, k, j, i] = k + (j % 64) / 64, the same along a row.

The netCDF C library (through netCDF4-python) writes it.  Run with TMPDIR
set to an owned scratch directory:
    python make_big_chunk_fixture.py --output-dir /path/to/fixtures
"""

import argparse
from pathlib import Path

import netCDF4
import numpy as np

SHAPE = (1, 17, 2000, 1000)
CHUNK = (1, 17, 1000, 1000)


def values() -> np.ndarray:
    k = np.arange(SHAPE[1], dtype=np.float32)[:, None, None]
    j = (np.arange(SHAPE[2]) % 64).astype(np.float32)[None, :, None] / 64
    plane = np.broadcast_to(k + j, SHAPE[1:3] + (1,))
    return np.broadcast_to(plane, SHAPE[1:]).astype(np.float32)[None]


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "big-chunk.nc4"
    with netCDF4.Dataset(target, "w", format="NETCDF4") as ds:
        for name, size in zip(("Time", "bottom_top", "south_north", "west_east"), SHAPE):
            ds.createDimension(name, None if name == "Time" else size)
        u = ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east"),
                              zlib=True, complevel=9, shuffle=True, chunksizes=CHUNK)
        u.units = "m s-1"
        u[:] = values()
    print(f"{target}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    generate(parser.parse_args().output_dir)
