#!/usr/bin/env python3
"""Build a dense-attribute, compressed NetCDF-4 fixture and native receipt.

The netCDF C library writes the input and reads every output variable.
NCO rewrites it with deflate and shuffle, with history disabled so the
fixture contains no build-directory paths. More than eight attributes
force HDF5 fractal heaps and v2 B-tree attribute indexes.

Requires netCDF4-python, NumPy, and the NCO ncks executable. Run with
TMPDIR set to an owned scratch directory, outside the source tree:
    python make_dense_attrs_fixture.py --output-dir /path/to/fixtures
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import netCDF4
import numpy as np


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "dense-attrs-nco.nc4"
    with tempfile.TemporaryDirectory(prefix="dense-attrs-") as temporary:
        source = Path(temporary) / "input.nc4"
        with netCDF4.Dataset(source, "w", format="NETCDF4") as ds:
            ds.createDimension("Time", None)
            ds.createDimension("nCells", 2)
            ds.createDimension("StrLen", 64)
            ds.title = "Dense attribute checksum fixture"
            for index in range(16):
                ds.setncattr(f"metadata_{index:02}", f"global value {index:02}")
            time = ds.createVariable("Time", "i4", ("Time",))
            time.units = "hours since 2024-01-01 00:00:00"
            time[:] = [0, 1]
            field = ds.createVariable("t2m", "f4", ("Time", "nCells"))
            field.units = "K"
            for index in range(16):
                field.setncattr(f"metadata_{index:02}", f"variable value {index:02}")
            field[:] = [[289.906005859375, 290.25], [291.5, 288.0]]
            text = ds.createVariable("xtime", "S1", ("Time", "StrLen"))
            rows = [b"2024-01-01_00:00:00", b"2024-01-01_01:00:00"]
            text[:] = np.frombuffer(
                b"".join(row.ljust(64, b"\0") for row in rows), dtype="S1"
            ).reshape(2, 64)
        subprocess.run(
            ["ncks", "-h", "-O", "-4", "-L", "1", str(source), str(target)],
            check=True,
        )
        print(f"deleted input.nc4 ({source.stat().st_size} bytes)")

    variables = {}
    with netCDF4.Dataset(target) as ds:
        ds.set_auto_maskandscale(False)
        dimensions = [
            {"name": name, "len": len(dim), "unlimited": dim.isunlimited()}
            for name, dim in ds.dimensions.items()
        ]
        for name, variable in ds.variables.items():
            values = np.asarray(variable[:])
            record = {
                "dimensions": list(variable.dimensions),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "filters": variable.filters(),
            }
            if values.dtype.kind == "S":
                record["bytes_hex"] = values.tobytes().hex()
            else:
                floats = values.astype("<f8")
                record["values"] = floats.ravel().tolist()
                record["f64_le_hex"] = floats.tobytes().hex()
            variables[name] = record

    raw = target.read_bytes()
    heaps = [index for index in range(len(raw)) if raw.startswith(b"FRHP", index)]
    if len(heaps) < 2:
        raise RuntimeError("fixture must contain root and variable fractal heaps")
    # This fixture uses eight-byte HDF5 offsets and lengths. The managed
    # free-space amount is part of the 142-byte header preceding its checksum.
    heap = heaps[0]
    checksum_offset = heap + 142
    version = subprocess.run(
        ["ncks", "--version"], check=True, capture_output=True, text=True
    )
    receipt = {
        "source": "netCDF C library and NCO ncks -h -4 -L 1",
        "netcdf_library_version": netCDF4.__netcdf4libversion__,
        "hdf5_library_version": netCDF4.__hdf5libversion__,
        "nco_version": (version.stdout + version.stderr).split()[4],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "fractal_heap_offsets": heaps,
        "corruption_checksum_offset": checksum_offset,
        "corruption_checksum_le_hex": raw[checksum_offset:checksum_offset + 4].hex(),
        "dimensions": dimensions,
        "variables": variables,
    }
    receipt_path = output_dir / "dense-attrs-nco.native.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target.name} ({len(raw)} bytes, sha256 {receipt['sha256']})")
    print(f"wrote {receipt_path.name} ({receipt_path.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    generate(parser.parse_args().output_dir)
