"""Cut the in-tree WRF v4.6.1 swint_opt = 1 fixture from an oracle build.

Usage: python cut_fixture.py BUILD_DIRECTORY OUTPUT.npz

Keeps every input and every WRF output word for a column subset of each
scenario (every regime label kept, all steps), so tests/test_swint_
interpolation.py can replay WOOF over them without the Fortran.  The WRF
words stored are the ``O0cr`` run: unmodified v4.6.1 routines at -O0 with
a correctly rounded asinf (see build.sh).  The receipt records, per
scenario, how many words the box's own C library run (``O0``) differs
from it, which is the C library's asinf and nothing else.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROWS = ("coszen_loc", "coszen", "bb", "bx", "gg", "gx", "coszen_ref",
        "swdown_ref", "swddir_ref", "swdown", "swddir", "swddni", "swddif",
        "gsw")
PER_LABEL = {"conus_day": 4, "polar": 3, "subsolar": 4, "edge": 2,
             "stress": 64, "clamp": 24}
INPUTS = ("xlat", "xlon", "julian", "xtime", "gmt", "radflag", "elapsed",
          "swdown", "swddir", "albedo", "label", "start")


def main() -> None:
    build = Path(sys.argv[1])
    out = Path(sys.argv[2])
    cols = np.load(build / "columns.npz")
    rng = np.random.default_rng(7)
    pack = {"radt_minutes": cols["radt_minutes"]}
    receipt = {"wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
               "wrf_source_sha256": {
                   "phys/module_radiation_driver.F":
                       "779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa"},
               "flags": "-O0 -ffp-contract=off",
               "reference": "O0cr (correctly rounded asinf preloaded)",
               "rows": list(ROWS), "scenarios": {}}
    receipt["build_receipt"] = (build / "receipt.txt").read_text()
    names = [str(s) for s in cols["scenarios"]]
    for name in names:
        ncol = cols[f"{name}/xlat"].size
        nstep = cols[f"{name}/julian"].size
        cr = np.fromfile(build / f"{name}.O0cr.out.bin", np.float32).reshape(
            nstep, len(ROWS), ncol)
        o0 = np.fromfile(build / f"{name}.O0.out.bin", np.float32).reshape(
            nstep, len(ROWS), ncol)
        labels = cols[f"{name}/label"].astype(str)
        keep = []
        for lab in sorted(set(labels.tolist())):
            idx = np.flatnonzero(labels == lab)
            k = min(PER_LABEL[name], idx.size)
            keep += sorted(rng.choice(idx, k, replace=False).tolist())
        keep = np.array(sorted(keep))
        for key in INPUTS:
            v = cols[f"{name}/{key}"]
            if key in ("xlat", "xlon", "label"):
                v = v[keep]
            elif key in ("swdown", "swddir", "albedo"):
                v = v[:, keep]
            pack[f"{name}/{key}"] = v
        pack[f"{name}/wrf"] = np.ascontiguousarray(cr[:, :, keep])
        differ = cr.view(np.uint32) != o0.view(np.uint32)
        receipt["scenarios"][name] = {
            "columns_in_build": int(ncol), "columns_kept": int(keep.size),
            "steps": int(nstep),
            "c_library_asinf_words_differing_all_columns": int(differ.sum()),
            "c_library_asinf_steps_differing": int(differ.any((1, 2)).sum()),
        }
    pack["scenarios"] = np.array(names)
    pack["receipt"] = np.array(json.dumps(receipt, indent=1))
    np.savez_compressed(out, **pack)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / "SHA256").write_text(f"{digest}  {out.name}\n",
                                       encoding="ascii", newline="\n")
    print(json.dumps(receipt["scenarios"], indent=1))
    print(out, out.stat().st_size, digest)


if __name__ == "__main__":
    main()
