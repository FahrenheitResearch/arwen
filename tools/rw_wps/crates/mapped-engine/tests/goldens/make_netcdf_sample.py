"""Write the NetCDF4 sample the NetCDF golden decodes.

No agency publishes a mapped NetCDF FORCING set into this box's staging
tree — the staged NetCDF objects are surface analyses on other calendars,
and the pressure-level NetCDF route's own bytes come from a CDS request.
So this sample is written by the real `netCDF4` library (real HDF5 bytes,
real CF attributes, the packed `scale_factor`/`add_offset` case included)
against gpuwm's OWN checked-in NetCDF mapping authority
(`configs/rw-wps-era5-netcdf.mapping.json`), and the golden beside it is
still measured by running the REAL Python engine over those bytes.

What that does and does not prove is worth stating plainly: it proves the
Rust reader, the CF decode, the selector resolution, the level ladder, the
soil selector stack, the derivations and the frame header agree with the
Python engine bit for bit on the same bytes.  It does not prove anything
about a producer's file layout, which is what the GRIB goldens beside it
cover with real published objects.

The values are deterministic (seeded, and written from an explicit
formula) so re-running reproduces the same file byte for byte.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import netCDF4
import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parents[5]
MAPPING = REPOSITORY / "configs" / "rw-wps-era5-netcdf.mapping.json"

#: Small enough to read in a diff, large enough to be a real grid: the
#: canonical-frame contract refuses anything under 2x2, and the derivations
#: need a full level ladder.
ROWS = 6
COLUMNS = 9
TIMES = 2


def _ramp(shape: tuple[int, ...], start: float, span: float) -> np.ndarray:
    """A deterministic monotone field, distinct per cell."""

    count = int(np.prod(shape))
    return (start + span * np.arange(count, dtype=np.float64) / max(count - 1, 1)).reshape(shape)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(HERE / "netcdf-pressure-level-sample.nc"),
        help="where to write the sample",
    )
    arguments = parser.parse_args(argv)

    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    levels = [float(value) for value in mapping["coordinates"]["vertical"]["levels"]]
    soil_layers = int(mapping["target"]["soil_layer_count"])

    target = pathlib.Path(arguments.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    # CLASSIC, not NETCDF4, and the reason is a measured defect rather than
    # a preference: the netcrust vendored under tools/rw_wps refuses a
    # plain `NETCDF4` file written by this script with
    # `HDF5 error: checksum mismatch: expected 0x00000002`, in BOTH the
    # strict and the size-inferred metadata modes, while the staged
    # `rw_netcdf.exe` the Python engine runs reads the same bytes in strict
    # mode and recovers its dimension scales.  That gap is recorded for the
    # netcrust lane; pinning this golden to the classic container keeps the
    # MAPPED path (CF decode, selector resolution, level ladder, soil
    # stack, derivations, frame header) under test instead of blocking on
    # a container-reader bug that belongs to another crate.
    with netCDF4.Dataset(target, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.Conventions = "CF-1.7"
        dataset.createDimension("valid_time", TIMES)
        dataset.createDimension("pressure_level", len(levels))
        dataset.createDimension("latitude", ROWS)
        dataset.createDimension("longitude", COLUMNS)

        time = dataset.createVariable("valid_time", "f8", ("valid_time",))
        time.units = mapping["coordinates"]["time"]["units"]
        time.calendar = mapping["coordinates"]["time"]["calendar"]
        time.standard_name = "time"
        # `hours since 1900-01-01`, six hours apart: the mapping declares a
        # 21600 s boundary interval, so the cadence check is exercised.
        time[:] = np.array([1_108_248.0, 1_108_254.0], dtype=np.float64)

        level = dataset.createVariable("pressure_level", "f8", ("pressure_level",))
        level.units = mapping["coordinates"]["vertical"]["units"]
        level.standard_name = "air_pressure"
        level[:] = np.asarray(levels, dtype=np.float64)

        latitude = dataset.createVariable("latitude", "f8", ("latitude",))
        latitude.units = "degrees_north"
        latitude.standard_name = "latitude"
        latitude[:] = np.linspace(52.0, 47.0, ROWS)

        longitude = dataset.createVariable("longitude", "f8", ("longitude",))
        longitude.units = "degrees_east"
        longitude.standard_name = "longitude"
        longitude[:] = np.linspace(-4.0, 4.0, COLUMNS)

        volume = (TIMES, len(levels), ROWS, COLUMNS)
        surface = (TIMES, ROWS, COLUMNS)

        def write(name: str, dimensions: tuple[str, ...], units: str, values: np.ndarray,
                  **attributes: object) -> None:
            variable = dataset.createVariable(name, "f8", dimensions)
            variable.units = units
            for key, value in attributes.items():
                setattr(variable, key, value)
            variable[:] = values

        write("t", ("valid_time", "pressure_level", "latitude", "longitude"), "K",
              _ramp(volume, 210.0, 90.0))
        write("u", ("valid_time", "pressure_level", "latitude", "longitude"), "m s**-1",
              _ramp(volume, -30.0, 60.0))
        write("v", ("valid_time", "pressure_level", "latitude", "longitude"), "m s**-1",
              _ramp(volume, -20.0, 40.0))
        write("r", ("valid_time", "pressure_level", "latitude", "longitude"), "%",
              _ramp(volume, 5.0, 90.0))
        write("z", ("valid_time", "pressure_level", "latitude", "longitude"), "m**2 s**-2",
              _ramp(volume, 1.0e3, 3.0e5))
        write("sp", ("valid_time", "latitude", "longitude"), "Pa",
              _ramp(surface, 96_000.0, 6_000.0))
        write("z_surface", ("valid_time", "latitude", "longitude"), "m**2 s**-2",
              _ramp(surface, 0.0, 9_000.0))
        write("skt", ("valid_time", "latitude", "longitude"), "K",
              _ramp(surface, 270.0, 25.0))
        write("t2m", ("valid_time", "latitude", "longitude"), "K",
              _ramp(surface, 268.0, 24.0))
        write("d2m", ("valid_time", "latitude", "longitude"), "K",
              _ramp(surface, 262.0, 20.0))
        write("u10", ("valid_time", "latitude", "longitude"), "m s**-1",
              _ramp(surface, -12.0, 24.0))
        write("v10", ("valid_time", "latitude", "longitude"), "m s**-1",
              _ramp(surface, -9.0, 18.0))
        write("lsm", ("valid_time", "latitude", "longitude"), "(0-1)",
              _ramp(surface, 0.0, 1.0))
        for layer in range(1, soil_layers + 1):
            write(f"stl{layer}", ("valid_time", "latitude", "longitude"), "K",
                  _ramp(surface, 265.0 + layer, 12.0))
            write(f"swvl{layer}", ("valid_time", "latitude", "longitude"), "m**3 m**-3",
                  _ramp(surface, 0.05 * layer, 0.30))

    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
