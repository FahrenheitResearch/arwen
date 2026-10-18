"""Write tests/data/thompson_fork_column_oracle.npz from column-oracle batch
dumps (gpu_column_oracle.py ``batch-*.npz``) of one column set.

usage:
    python make_column_fixture.py OUT.npz BATCH.npz [BATCH.npz ...]

The first batch's inputs are stored once (``in_*``: the adapter's own
inputs, theta, pressure, geopotential, vertical velocity, the eleven
species and the two surface emissions); every batch must use those columns
or a leading subset of them.  Per batch ``b<i>`` the file keeps its column
count, dt, ITIMESTEP and every output the unmodified fork's Fortran
returned (``b<i>_fork_<field>``, radii in the fork's metres).
tests/test_thompson_fork_column_gpu.py runs WOOF's adapter on those inputs
and requires the fork's words.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

INPUTS = ("p", "th", "geop", "w", "qv", "qc", "qr", "qi", "qs", "qg", "ni",
          "nr", "nc", "nwfa", "nifa", "nwfa2d", "nifa2d", "pii", "dz")
OUTPUTS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa",
           "nifa", "th", "refl", "re_cloud_m", "re_ice_m", "re_snow_m",
           "rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
           "graupelncv", "sr")


def main(argv):
    out = Path(argv[1])
    batches = [Path(p) for p in argv[2:]]
    data = {}
    meta = {"batches": [],
            "boundary": "WRF physics Exner (prepare_microphysics_exner)",
            "log10f": "glibc 2.39 log10f",
            "source": "tools/thompson_fork_oracle/"
            "gpu_column_oracle.py against NOAA-EMC/HRRR v4.1.21 "
            "module_mp_thompson.F (sha256 4d600111...)"}
    first = np.load(batches[0])
    labels = first["labels"]
    for name in INPUTS:
        data[f"in_{name}"] = np.asarray(first[f"in_{name}"], np.float32)
    data["labels"] = labels
    for i, path in enumerate(batches):
        z = np.load(path)
        ncol = z["in_p"].shape[0]
        for name in INPUTS:
            if not np.array_equal(z[f"in_{name}"].view(np.uint32),
                                  data[f"in_{name}"][:ncol].view(np.uint32)):
                raise SystemExit(f"{path}: inputs are not a leading subset")
        m = re.search(r"dt([0-9.]+)-step(\d+)", path.name)
        dt, step = float(m.group(1)), int(m.group(2))
        for name in OUTPUTS:
            data[f"b{i}_fork_{name}"] = np.asarray(z[f"fork_{name}"],
                                                   np.float32)
        meta["batches"].append({"index": i, "ncol": int(ncol), "dt": dt,
                                "itimestep": step,
                                "dump_sha256": hashlib.sha256(
                                    path.read_bytes()).hexdigest()})
    data["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(out, **data)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(sys.argv)
