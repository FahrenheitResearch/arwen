"""Seal a capture into the compact committed fixture for the GPU gate.

The full capture (about 49 MB) stays on the box that made it.  The repository
keeps what the bitwise gate needs: the synthetic input states (the real-state
inputs are rebuilt from tests/data/wrf471_diffusion) and the SHA-256 of every
WRF output array.  Bitwise equality of an engine array with the WRF array is
equality of their SHA-256 over the float32 words, so the gate compares
digests and needs no WRF words in the tree.  ``compare.py`` on the full
capture is the instrument that reports ULP distances.

    python tools/wrf461_km3_oracle/seal.py CAPTURE tests/data/wrf461_km3
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def words_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def inputs_sha256(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.capture / "manifest.json").read_text())
    sealed = {"schema": "wrf461-km3-column-oracle-digests-v1",
              "build": {k: manifest["build"][k] for k in
                        ("wrf_release", "wrf_commit", "compiler", "libc", "flags",
                         "links_libmvec", "source_sha256", "routines", "extra_sources",
                         "wrappers", "generated_sha256")},
              "columns": manifest["columns"], "cases": {}}
    for case in manifest["cases"]:
        with np.load(a.capture / case["file"]) as data:
            arrays = {k.removeprefix("input__"): data[k] for k in data.files if k.startswith("input__")}
            wrf = {k: data[k] for k in data.files if k.startswith("wrf__")}
            meta = json.loads(str(data["meta_json"]))
        row = {"meta": meta, "inputs_sha256": inputs_sha256(arrays),
               "wrf_sha256": {k: words_sha256(v) for k, v in sorted(wrf.items())},
               "stock_w_differs": {}}
        for iso in (0, 1):
            for flux in (0, 1, 2):
                s = wrf[f"wrf__iso{iso}__v{flux}_stock_w"].view(np.uint32)
                c = wrf[f"wrf__iso{iso}__v{flux}_wcontrol_w"].view(np.uint32)
                row["stock_w_differs"][f"iso{iso}_v{flux}"] = int((s != c).sum())
        if meta["family"] != "retained-real-state":
            target = a.output / f"inputs-{case['name']}.npz"
            np.savez_compressed(target, **arrays)
            row["inputs_file"] = target.name
            row["inputs_file_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        sealed["cases"][case["name"]] = row
    (a.output / "digests.json").write_text(json.dumps(sealed, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")
    print(a.output / "digests.json", len(sealed["cases"]), "cases", sealed["columns"], "columns")


if __name__ == "__main__":
    main()
