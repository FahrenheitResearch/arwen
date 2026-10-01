"""Pack geogrid_fixture.sh's runs into the end-to-end WPS geogrid fixture.

    python make_geogrid_fixture.py WORK_DIR OUT.npz

Keys: ``raw_extended`` (float32, WPS's own unsmoothed HGT_M on the domain
widened by the 3-cell halo, the smoother's input), ``hgt_<setting>``
(float32, geogrid.exe's HGT_M for each setting), ``settings`` (rows of
setting name, smooth_option, smooth_passes as geogrid read them) and
``provenance``.  Refuses unless the widened run's interior equals the
no-smoothing run bit for bit (the lattices coincide) and passes=0 equals
no smoothing (WPS's ``do ipass=1,npass``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np

#: setting name -> (smooth_option, smooth_passes) as GEOGRID.TBL gave it.
SETTINGS = {
    "none": ("none", 0),
    "passes0": ("smth-desmth_special", 0),
    "special1": ("smth-desmth_special", 1),
    "special2": ("smth-desmth_special", 2),
    "special3": ("smth-desmth_special", 3),
    "smdes1": ("smth-desmth", 1),
    "smdes2": ("smth-desmth", 2),
    "smdes3": ("smth-desmth", 3),
    "121x1": ("1-2-1", 1),
    "121x2": ("1-2-1", 2),
    "121x3": ("1-2-1", 3),
    "121x5": ("1-2-1", 5),
}


def hgt(run: Path) -> np.ndarray:
    with netCDF4.Dataset(run / "geo_em.d01.nc") as ds:
        ds.set_auto_mask(False)
        field = ds.variables["HGT_M"]
        if field.dtype != np.float32:
            raise SystemExit(f"{run}: HGT_M is {field.dtype}, not float32")
        return np.array(field[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    raw = hgt(args.work / "wide-none")
    fields = {name: hgt(args.work / name) for name in SETTINGS}
    if not np.array_equal(raw[3:-3, 3:-3].view(np.uint32),
                          fields["none"].view(np.uint32)):
        raise SystemExit("the widened run's interior is not the domain's "
                         "unsmoothed terrain; the lattices do not coincide")
    if not np.array_equal(fields["passes0"].view(np.uint32),
                          fields["none"].view(np.uint32)):
        raise SystemExit("smooth_passes=0 did not leave HGT_M unsmoothed")
    payload = {
        "raw_extended": raw,
        "settings": np.array([[name, option, str(passes)]
                              for name, (option, passes) in SETTINGS.items()]),
        "provenance": np.array((args.work / "provenance.txt").read_text()),
    }
    for name, field in fields.items():
        payload[f"hgt_{name}"] = field
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes), domain "
          f"{fields['none'].shape}, terrain {float(raw.min()):.0f} to "
          f"{float(raw.max()):.0f} m")


if __name__ == "__main__":
    main()
