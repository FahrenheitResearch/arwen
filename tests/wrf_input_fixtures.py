"""Small generated WRF inputs shared by public suites and private campaigns.

These fixtures must ship independently of the excluded campaign harness: the
public scalar-boundary and supplied-physics suites both construct them.
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np


def _variable(dataset, name, dimensions, values, *, compressed=False,
              dtype="f4"):
    options = {"zlib": True, "complevel": 1} if compressed else {}
    variable = dataset.createVariable(name, dtype, dimensions, **options)
    variable[...] = np.asarray(values, dtype=dtype)
    return variable


def _small_wrfinput(path: Path, *, nz: int = 2, ny: int = 4, nx: int = 5,
                    moisture_names=("QVAPOR",)) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("bottom_top", nz),
                ("bottom_top_stag", nz + 1), ("south_north", ny),
                ("south_north_stag", ny + 1), ("west_east", nx),
                ("west_east_stag", nx + 1)):
            dataset.createDimension(name, size)
        mass = ("Time", "bottom_top", "south_north", "west_east")
        _variable(dataset, "U", ("Time", "bottom_top", "south_north",
                                  "west_east_stag"), 1.0)
        _variable(dataset, "V", ("Time", "bottom_top", "south_north_stag",
                                  "west_east"), 2.0)
        _variable(dataset, "W", ("Time", "bottom_top_stag", "south_north",
                                  "west_east"), 3.0)
        t = np.arange(nz * ny * nx, dtype=np.float32).reshape(1, nz, ny, nx)
        _variable(dataset, "T", mass, t)
        _variable(dataset, "T_INIT", mass, np.float32(300.0))
        _variable(dataset, "PH", ("Time", "bottom_top_stag", "south_north",
                                   "west_east"), 4.0)
        _variable(dataset, "PHB", ("Time", "bottom_top_stag", "south_north",
                                    "west_east"), 5.0)
        _variable(dataset, "MU", ("Time", "south_north", "west_east"), 6.0)
        _variable(dataset, "MUB", ("Time", "south_north", "west_east"), 7.0)
        for index, name in enumerate(moisture_names):
            _variable(dataset, name, mass, 0.001 * (index + 1))
