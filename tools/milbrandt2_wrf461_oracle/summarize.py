"""Summarise every WOOF-vs-WRF comparison of the mp=9 oracle into one JSON.

    python summarize.py WORKDIR OUT.json

For every input set ``in_<set>.bin`` with a Fortran output
``out_<set>_O2.bin`` it compares each ``woof_<set>_<mode>[_<layout>].bin``
present and records, per field, the count of differing float32 words, the
maximum total-order ULP distance and the maximum absolute difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from compare import compare, read_output
    from gen_columns import read_stream

    work = Path(sys.argv[1])
    result = {}
    for inp in sorted(work.glob("in_*.bin")):
        name = inp.stem[3:]
        ref_path = work / f"out_{name}_O2.bin"
        if not ref_path.exists():
            continue
        ncol, nz, nsteps, dt, _, _ = read_stream(inp)
        ref = read_output(ref_path, ncol, nz, nsteps)
        entry = {"columns": ncol, "levels": nz, "steps": nsteps, "dt": dt,
                 "runs": {}}
        for run in sorted(work.glob(f"woof_{name}_*.bin")):
            got = read_output(run, ncol, nz, nsteps)
            rows = compare(ref, got)
            fields = {k: {"differ": n, "words": size, "max_ulp": mu,
                          "max_abs": md}
                      for k, n, size, mu, md, _ in rows}
            entry["runs"][run.stem[len(f"woof_{name}_"):]] = {
                "differing_words": int(sum(r[1] for r in rows)),
                "words": int(sum(r[2] for r in rows)),
                "max_ulp": int(max(r[3] for r in rows)),
                "max_abs": float(max(r[4] for r in rows)),
                "fields": fields}
        result[name] = entry
    Path(sys.argv[2]).write_text(json.dumps(result, indent=1) + "\n")
    for name, entry in result.items():
        for run, r in entry["runs"].items():
            print(f"{name:10s} {run:22s} differing {r['differing_words']:8d} "
                  f"of {r['words']:9d}  max ULP {r['max_ulp']:11d}  "
                  f"max abs {r['max_abs']:.3e}")


if __name__ == "__main__":
    main()
