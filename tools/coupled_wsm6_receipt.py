"""Hash complete stock and WOOF WSM6 call boundaries without moving arrays."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from coupled_dycore_receipt import read_wrf
from coupled_trace_receipt import digest

FIELDS = dict(th="theta", rho="rho", pi_phy="pii", p="pressure", dz8w="dz",
              qv_curr="qv", qc_curr="qc", qr_curr="qr", qi_curr="qi",
              qs_curr="qs", qg_curr="qg")


def collect(directory, arm, shape, spec_zone=0):
    if spec_zone < 0 or 2 * spec_zone >= min(shape[-2:]):
        raise ValueError("specified ring must leave active model columns")
    rows, missing = {}, []
    for tag in ("mp_wsm6_in", "mp_wsm6_out"):
        for wn, on in FIELDS.items():
            key = f"{tag}/{wn}"
            try:
                if arm == "wrf":
                    value = read_wrf(directory, 1, 0, tag, wn, shape)
                else:
                    paths = list(directory.glob(f"*_s001_*_{tag}.npz"))
                    if len(paths) != 1:
                        raise ValueError(f"{tag}: {len(paths)} matching dumps")
                    with np.load(paths[0], allow_pickle=False) as z:
                        value = z["x_" + on]
                    if value.shape != shape:
                        raise ValueError("WSM6 input shape differs from the required domain")
                rows[key] = digest(value)
                active = value if not spec_zone else value[:, spec_zone:-spec_zone, spec_zone:-spec_zone]
                rows[key].update(spec_zone=spec_zone, active=digest(active))
            except (ValueError, OSError, KeyError) as error:
                missing.append(dict(key=key, error=str(error)))
    return dict(arm=arm, rows=rows, missing=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--arm", choices=("wrf", "woof"), required=True)
    parser.add_argument("--shape", type=int, nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spec-zone", type=int, default=0)
    args = parser.parse_args()
    result = collect(args.directory, args.arm, tuple(args.shape), args.spec_zone)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(rows=len(result["rows"]), missing=result["missing"])))
    return int(bool(result["missing"]))


if __name__ == "__main__":
    raise SystemExit(main())
