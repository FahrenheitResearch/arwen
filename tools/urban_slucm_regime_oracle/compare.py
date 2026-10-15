"""Compare the engine's SLUCM GPU kernel with a regime oracle, word for word.

    python tools/urban_slucm_regime_oracle/compare.py FIXTURE_DIR OUT.json

FIXTURE_DIR is what ``build.sh`` writes.  Every variant in it is run twice
through ``gpuwm.core.urban_ucm.run_columns``:

* ``replay``: every row from WRF's own state at entry (one call each);
* ``carried``: each column's first step from WRF's state, then the kernel's
  OWN state carried through the remaining steps, as a forecast carries it.

Every output (19) and every state word after the call (44) is compared as
float32 bits.  The arithmetic is whatever the process was started with:
``GPUWM_WRF_EXACT=1`` for the strict build, unset for the default one.  The
JSON records which.  Nothing here decides a tolerance.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.urban_ucm_oracle import (
    carried_replay, load_variant, replay)

REGIMES = {
    1: "convective hot afternoon", 2: "convective tropical humid",
    3: "convective shower 25 mm/h", 4: "stable night clear calm",
    5: "stable night cold inversion", 6: "snow season day",
    7: "freezing rain", 8: "arctic night", 9: "coastal marine",
    10: "high terrain day", 11: "high terrain night", 12: "desert",
    13: "windstorm", 14: "dew night", 15: "canopy-wind edge",
    16: "irrigation night 300 s", 17: "near calm sunny",
    18: "first level 3 m above roof",
}


def _measure(port, ref, scenario):
    fields, bad_rows = {}, np.zeros(len(scenario), bool)
    worst_ulp, worst_abs = 0, 0.0
    for key, want in ref.items():
        got = np.asarray(port[key], np.float32)
        want = np.asarray(want, np.float32)
        same = got.view(np.uint32) == want.view(np.uint32)
        both_nan = np.isnan(got) & np.isnan(want)
        diff = ~(same | both_nan)
        ulp = fp32_ulp_distance(got, want)
        mx = int(np.max(ulp)) if ulp.size else 0
        ad = float(np.nanmax(np.abs(got.astype(np.float64) - want.astype(np.float64)))) \
            if want.size else 0.0
        fields[key] = {"rows_differ": int(diff.sum()), "max_ulp": mx, "max_abs": ad}
        bad_rows |= diff
        worst_ulp = max(worst_ulp, mx)
        worst_abs = max(worst_abs, ad)
    by_regime = {}
    for sc in sorted(set(scenario.tolist())):
        sel = scenario == sc
        by_regime[REGIMES.get(int(sc), str(sc))] = {
            "rows": int(sel.sum()), "rows_differ": int((bad_rows & sel).sum())}
    return {"rows": int(len(scenario)), "words": int(len(scenario) * len(ref)),
            "rows_differ": int(bad_rows.sum()), "max_ulp": worst_ulp,
            "max_abs": worst_abs,
            "fields_differ": {k: v for k, v in fields.items() if v["rows_differ"]},
            "by_regime": by_regime}


def main(argv):
    root, out = Path(argv[1]), Path(argv[2])
    names = sorted(p.name[:-len(".csv.gz")] for p in root.glob("*.csv.gz"))
    result = {"fixture_dir": str(root),
              "gpuwm_wrf_exact": os.environ.get("GPUWM_WRF_EXACT", "unset"),
              "compiler": (root / "compiler.txt").read_text().strip().splitlines(),
              "variants": {}}
    for name in names:
        rows, _, _ = load_variant(name, root)
        port, ref, codes = replay(name, root=root)
        rep = _measure(port, ref, rows["scenario"])
        rep["kernel_codes_nonzero"] = int(np.count_nonzero(codes))
        steps = rows["step"]
        first = np.nonzero(steps == steps.min())[0]
        order = np.concatenate([first + k for k in range(int(steps.max() - steps.min()) + 1)])
        cport, cref = carried_replay(name, root=root)
        car = _measure(cport, cref, rows["scenario"][order])
        result["variants"][name] = {"replay": rep, "carried": car}
        print(f"{name:24s} replay rows {rep['rows']:5d} differ {rep['rows_differ']:4d} "
              f"maxulp {rep['max_ulp']:6d} | carried differ {car['rows_differ']:4d} "
              f"maxulp {car['max_ulp']:6d}  codes {rep['kernel_codes_nonzero']}")
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv)
