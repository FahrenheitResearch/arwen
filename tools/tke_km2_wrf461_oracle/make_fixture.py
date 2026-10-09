"""Pack the cases and the WRF -O0 words into the committed test fixture.

The fields that do not depend on the namelist arm (BN2 and the six
deformation tensors) are stored once per case after checking that every arm
produced the same words; the four exchange coefficients and the TKE
tendency are stored per arm.

Usage: python make_fixture.py RUNS_DIR BUILD_RECEIPT OUT_DIR
       (RUNS_DIR holds cases/ and wrf-noopt/ as run_all.sh writes them)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arms import ARMS  # noqa: E402

PER_ARM = ("kmh", "kmv", "khh", "khv", "rtke")
SHARED = ("bn2", "d11", "d22", "d12", "d33", "d13", "d23")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path)
    ap.add_argument("receipt", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.runs / "cases" / "cases.json").read_text())
    files, cases = {}, []
    for case in index["cases"]:
        name = case["name"]
        with np.load(args.runs / "cases" / case["file"]) as data:
            arrays = {k: data[k] for k in data.files}
        path = args.out / f"case-{name}.npz"
        np.savez_compressed(path, **arrays)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        words = {}
        per = {arm[0]: dict(np.load(args.runs / "wrf-noopt" / f"wrf-{name}-{arm[0]}.npz"))
               for arm in ARMS}
        first = per[ARMS[0][0]]
        for field in SHARED:
            for arm in ARMS[1:]:
                if not np.array_equal(per[arm[0]][field].view(np.uint32),
                                      first[field].view(np.uint32)):
                    raise ValueError(f"{name}: {field} depends on the arm")
            words[field] = first[field]
        for arm in ARMS:
            for field in PER_ARM:
                words[f"{arm[0]}/{field}"] = per[arm[0]][field]
        path = args.out / f"wrf461-O0-{name}.npz"
        np.savez_compressed(path, **words)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases.append({"name": name, "columns": case["columns"]})
        print(name)
    manifest = {
        "schema": "tke-km2-wrf461-column-oracle-v1",
        "wrf_build": json.loads(args.receipt.read_text()),
        "arms": [list(a) for a in ARMS],
        "per_arm_fields": list(PER_ARM), "shared_fields": list(SHARED),
        "cases": cases, "files": files,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                            encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
