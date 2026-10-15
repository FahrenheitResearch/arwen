"""Compare the engine's BEP+BEM GPU column with the regime oracle, word for word.

    python tools/urban_bem_regime_oracle/compare.py FIXTURE_DIR OUT.json

FIXTURE_DIR holds ``bep_bem_<variant>.npz`` (``to_npz.py``).  Each variant is
run through ``gpuwm.core.urban_bem.launch_bep_bem_columns`` from WRF's own
initial state, the engine carrying its OWN state through every call, and every
output of every call plus every state array after the last call is compared as
float32 bits.  The arithmetic is whatever the process was started with:
``GPUWM_WRF_EXACT=1`` for the strict build, unset for the default one; the
JSON records which, and the sha256 of the source string and options NVRTC
compiled.  Nothing here decides a tolerance.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))

import _urban_bem_regime as R  # noqa: E402


def main(argv):
    root, out = Path(argv[1]), Path(argv[2])
    import cupy as cp

    from gpuwm.core import urban_bem as ub

    source = ub.module_source()
    result = {
        "fixture_dir": str(root),
        "gpuwm_wrf_exact": os.environ.get("GPUWM_WRF_EXACT", "unset"),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
        "module_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "module_options": list(ub.MODULE_OPTIONS),
        "variants": {},
    }
    total_words = total_differ = 0
    worst_ulp, worst_abs = 0, 0.0
    names = sorted(p.name[len("bep_bem_"):-len(".npz")]
                   for p in root.glob("bep_bem_*.npz"))
    for name in names:
        fx = R.load_fixture(name, root)
        regimes = R.read_regimes(HERE / "REGIMES.csv",
                                 bool(int(fx["season_winter"][0])))
        rep = R.measure(fx)
        labels = R.column_labels(fx, regimes)
        frc = R.to_gpuwm(fx["state0/frc_urb2d"])[0]
        bad = rep["column_bad"]
        mx = max((r["max_ulp"] for r in rep["fields"].values()), default=0)
        ab = max((r["max_abs"] for r in rep["fields"].values()), default=0.0)
        worst_ulp, worst_abs = max(worst_ulp, mx), max(worst_abs, ab)
        total_words += rep["words"]
        total_differ += rep["words_differ"]
        by_regime: dict[str, dict] = {}
        for i, label in enumerate(labels):
            key = label.split(" / class")[0]
            row = by_regime.setdefault(key, {"columns": 0, "columns_differ": 0})
            row["columns"] += 1
            row["columns_differ"] += int(bad[i])
        result["variants"][name] = {
            "ncol": int(fx["ncol"][0]), "urban_columns": int((frc > 0).sum()),
            "nsteps": int(fx["nsteps"][0]), "dt": float(fx["step1/dt"][0]),
            "words": rep["words"], "words_differ": rep["words_differ"],
            "max_ulp": mx, "max_abs": ab,
            "wrf_stop_call": rep["wrf_stop_call"],
            "engine_stop": rep["engine_stop"],
            "columns_differ": [labels[i] for i in np.flatnonzero(bad)],
            "fields_differ": {k: v for k, v in rep["fields"].items()
                              if v["words_differ"]},
            "by_regime": by_regime,
        }
        print(f"{name:14s} cols {int(fx['ncol'][0]):4d} urban {int((frc > 0).sum()):4d} "
              f"steps {int(fx['nsteps'][0]):3d} words {rep['words']:10d} "
              f"differ {rep['words_differ']:8d} max_ulp {mx:8d} max_abs {ab:.3e}"
              + (f"  WRF stopped in call {rep['wrf_stop_call']}; engine: "
                 f"{rep['engine_stop']}" if rep["wrf_stop_call"] else ""))
    result["total"] = {"words": total_words, "words_differ": total_differ,
                       "max_ulp": worst_ulp, "max_abs": worst_abs}
    print(f"TOTAL words {total_words} differ {total_differ} max_ulp {worst_ulp} "
          f"max_abs {worst_abs:.3e}  GPUWM_WRF_EXACT={result['gpuwm_wrf_exact']}")
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv)
