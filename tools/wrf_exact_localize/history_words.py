"""Hash WRF multi-frame history through the Rust reader, without retaining data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpuwm import netcdf_bridge
from gpuwm.verify_exact.replay import _STORED_DTYPE


def hashes(path):
    ds = netcdf_bridge.open_dataset(path)
    ds.set_auto_maskandscale(False)
    fields = {}
    for name, var in ds.variables.items():
        if var.is_character or var.dimensions[:1] != ("Time",):
            continue
        values = np.asarray(var[:])
        dtype = _STORED_DTYPE.get(var.stored_dtype)
        if dtype is not None:
            values = values.astype(dtype)
        fields[name] = [hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                        for a in values]
    ds.close()
    return fields


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("history", type=Path)
    p.add_argument("--reference-history", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    current = hashes(a.history)
    r = dict(schema="gpuwm.history-words/v1", history=str(a.history), hashes=current)
    if a.reference_history:
        reference = hashes(a.reference_history)
        r.update(reference=str(a.reference_history),
                 absent=sorted(set(reference) - set(current)),
                 extra=sorted(set(current) - set(reference)),
                 differing={n: [i for i, (x, y) in enumerate(zip(reference[n], current[n])) if x != y]
                            for n in reference.keys() & current.keys()
                            if reference[n] != current[n]})
        r["all_identical"] = not (r["absent"] or r["extra"] or r["differing"])
    a.out.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "hashes"}, indent=2))
    return 0 if r.get("all_identical", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
