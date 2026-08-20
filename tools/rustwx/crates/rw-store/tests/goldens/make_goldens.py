"""Emit tiny golden files in each classic format for byte-level writer checks."""
import sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

CASES = {
    "cdf1": "NETCDF3_CLASSIC",
    "cdf2": "NETCDF3_64BIT_OFFSET",
    "cdf5": "NETCDF3_64BIT_DATA",
}

for tag, fmt in CASES.items():
    p = out / f"golden_{tag}.nc"
    if p.exists():
        p.unlink()
    with Dataset(p, "w", format=fmt) as d:
        d.setncattr("TITLE", " OUTPUT FROM GOLDEN")
        d.setncattr("DX", np.float32(22000.0))
        d.setncattr("MAP_PROJ", np.int32(1))
        d.createDimension("Time", None)
        d.createDimension("DateStrLen", 19)
        d.createDimension("west_east", 3)
        d.createDimension("south_north", 2)
        t = d.createVariable("Times", "S1", ("Time", "DateStrLen"))
        it = d.createVariable("ITIMESTEP", "i4", ("Time",))
        xl = d.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
        xl.setncattr("units", "degree_north")
        xl.setncattr("FieldType", np.int32(104))
        hgt = d.createVariable("HGT_M", "f4", ("south_north", "west_east"))
        # two records
        for r, stamp in enumerate(("2026-08-10_13:00:00", "2026-08-10_14:00:00")):
            t[r] = np.array(list(stamp), dtype="S1")
            it[r] = 100 + r
            xl[r] = np.arange(6, dtype="f4").reshape(2, 3) + r * 10.0
        hgt[:] = (np.arange(6, dtype="f4") * 0.5).reshape(2, 3)
    print(tag, p, p.stat().st_size)

# dump the CDF-5 header bytes for layout derivation
for tag in CASES:
    b = (out / f"golden_{tag}.nc").read_bytes()
    print(f"--- {tag} first 256 bytes")
    for i in range(0, min(256, len(b)), 16):
        chunk = b[i:i+16]
        hexs = " ".join(f"{c:02x}" for c in chunk)
        asc = "".join(chr(c) if 32 <= c < 127 else "." for c in chunk)
        print(f"{i:04x}  {hexs:<47}  {asc}")
