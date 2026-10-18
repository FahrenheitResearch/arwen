"""Bulk random columns for the cos/acos sensitivity search (CPU oracle first).

    python make_refute_bulk.py OUTDIR NFILES NCOL

Same profile generator as make_refute_cases.py (column()), seed 2026100743,
grids cycled from FUZZ_GRIDS, 2 steps per file.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("mrc", Path(__file__).with_name("make_refute_cases.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
nfiles, ncol = int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(2026100743)
index = {}
for i in range(nfiles):
    nk, ztop, dt = m.FUZZ_GRIDS[i % len(m.FUZZ_GRIDS)]
    cols = [m.column(rng, nk, ztop) for _ in range(ncol)]
    name = f"bulk{i:03d}-n{nk}"
    m.write(out / f"cases-{name}.bin", cols, nk, 2, dt)
    index[name] = {"nk": nk, "ztop": ztop, "dt": dt, "ncol": ncol, "nsteps": 2}
(out / "bulk-index.json").write_text(json.dumps({"seed": 2026100743, "sets": index}, indent=1) + "\n")
print(len(index), "files", nfiles * ncol, "columns")
