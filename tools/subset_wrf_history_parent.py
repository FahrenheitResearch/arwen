"""Create a compact, lossless horizontal cutout of WRF parent history.

The output retains the exact offline-child trajectory/static variables and
their native dtypes while cropping mass and staggered horizontal dimensions.
It is useful for moving a storm-centered parent forcing window to a rental
without transferring an entire large-domain history stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import netCDF4


_OFFLINE_MP10_VARIABLES = frozenset({
    "Times", "T", "U", "V", "W", "PH", "PHB", "MU", "MUB", "HGT",
    "P", "PB", "PSFC", "P_TOP", "ZNU", "ZNW", "MAPFAC_M",
    "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA",
    "XLAT", "XLONG", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW",
    "QGRAUP", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subset(source: Path, destination: Path, *, i0: int, j0: int,
           nx: int, ny: int) -> dict[str, object]:
    started = time.perf_counter()
    source = source.resolve()
    destination = destination.resolve()
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(source) as incoming:
        source_nx = len(incoming.dimensions["west_east"])
        source_ny = len(incoming.dimensions["south_north"])
        if min(i0, j0) < 0 or min(nx, ny) < 1:
            raise ValueError("crop origin must be nonnegative and size positive")
        if i0 + nx > source_nx or j0 + ny > source_ny:
            raise ValueError("crop exceeds the source mass grid")
        missing = sorted(_OFFLINE_MP10_VARIABLES - set(incoming.variables))
        if missing:
            raise ValueError(f"source lacks offline MP10 variables {missing}")
        with netCDF4.Dataset(temporary, "w", format="NETCDF4") as outgoing:
            outgoing.setncatts({name: incoming.getncattr(name)
                                for name in incoming.ncattrs()})
            outgoing.setncattr("GPUWM_PARENT_CROP_I0", int(i0))
            outgoing.setncattr("GPUWM_PARENT_CROP_J0", int(j0))
            outgoing.setncattr("GPUWM_PARENT_ORIGINAL_NX", int(source_nx))
            outgoing.setncattr("GPUWM_PARENT_ORIGINAL_NY", int(source_ny))
            for name, dimension in incoming.dimensions.items():
                if name == "west_east":
                    size = nx
                elif name == "west_east_stag":
                    size = nx + 1
                elif name == "south_north":
                    size = ny
                elif name == "south_north_stag":
                    size = ny + 1
                else:
                    size = len(dimension)
                outgoing.createDimension(name, size)
            for name in sorted(_OFFLINE_MP10_VARIABLES):
                variable = incoming.variables[name]
                fill = (variable.getncattr("_FillValue")
                        if "_FillValue" in variable.ncattrs() else None)
                options = {"fill_value": fill}
                if variable.dtype.kind not in {"S", "U", "O"} and variable.ndim:
                    options.update(zlib=True, complevel=1, shuffle=True)
                target = outgoing.createVariable(
                    name, variable.datatype, variable.dimensions, **options)
                target.setncatts({
                    attr: variable.getncattr(attr)
                    for attr in variable.ncattrs() if attr != "_FillValue"
                })
                selection = []
                for dimension in variable.dimensions:
                    if dimension == "west_east":
                        selection.append(slice(i0, i0 + nx))
                    elif dimension == "west_east_stag":
                        selection.append(slice(i0, i0 + nx + 1))
                    elif dimension == "south_north":
                        selection.append(slice(j0, j0 + ny))
                    elif dimension == "south_north_stag":
                        selection.append(slice(j0, j0 + ny + 1))
                    else:
                        selection.append(slice(None))
                target[:] = variable[tuple(selection)]
    os.replace(temporary, destination)
    return {
        "source": str(source),
        "source_sha256": _sha256(source),
        "destination": str(destination),
        "destination_sha256": _sha256(destination),
        "crop_i0": int(i0), "crop_j0": int(j0),
        "crop_nx": int(nx), "crop_ny": int(ny),
        "original_nx": int(source_nx), "original_ny": int(source_ny),
        "bytes": destination.stat().st_size,
        "seconds": time.perf_counter() - started,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--i0", type=int, required=True)
    parser.add_argument("--j0", type=int, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    receipt = subset(
        args.source, args.destination,
        i0=args.i0, j0=args.j0, nx=args.nx, ny=args.ny)
    encoded = json.dumps(receipt, indent=2, sort_keys=True)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_name(args.receipt.name + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, args.receipt)
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
