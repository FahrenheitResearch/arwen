#!/usr/bin/env python3
"""Cut an mp=8 process-rate fixture from one oracle set's inputs.

When ``oracle.py`` finds a word that differs, the first question is which
process rate differs first.  This writes the set's step-1 inputs (all
columns, or the ``--columns`` index list) through
``tools/thompson_real_column_parity/make_fixture.py``'s ``_answers`` with
the rate-instrumented WRF build (``build_wrf.sh``), in the layout of
``tests/data/thompson_real_columns_wrf461_mp8.npz``.  Then

    python tools/thompson_real_column_parity/mp8_wrf_ulp.py OUT.npz \
        --backend host --rates

grades every one of the sixty-four rates and every output word of that call.

usage: rates_fixture.py RATES_BUILD_DIR WRF_SET.npz OUT.npz
                        [--columns I J ...] [--state-step N]

``--state-step N`` takes the inputs from the oracle build's state after
step N instead of the set's initial inputs (theta, the eight species), so a
difference that first appears at step N+1 can be graded as a single call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARITY = HERE.parent / "thompson_real_column_parity"
for _p in (str(PARITY), str(HERE.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_fixture as MF  # noqa: E402

f32 = np.float32
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("build", type=Path)
    ap.add_argument("set", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--columns", type=int, nargs="*")
    ap.add_argument("--state-step", type=int, default=0)
    a = ap.parse_args(argv)
    z = np.load(a.set)
    cols = (np.arange(z["in_p"].shape[0]) if not a.columns
            else np.asarray(a.columns))
    raw = {"p": z["in_p"][cols], "th": z["in_th"][cols],
           "geop": z["in_geop"][cols], "w": z["in_w"][cols]}
    for s in SPECIES:
        raw[s] = z[f"in_{s}"][cols]
    if a.state_step:
        k = a.state_step - 1
        raw["th"] = z["oracle_th"][k][cols]
        for s in SPECIES:
            raw[s] = z[f"oracle_{s}"][k][cols]
    n = len(cols)
    raw["nwfa2d"] = np.zeros(n, f32)
    raw["nifa2d"] = np.zeros(n, f32)
    for s in ("nc", "nwfa", "nifa"):
        raw[s] = np.zeros_like(raw["qv"])
    raw = {k: np.ascontiguousarray(v, dtype=f32) for k, v in raw.items()}
    arrays = MF._answers(a.build, raw, float(z["dt"]), "oracle-rates", 8)
    arrays["labels"] = z["labels"][cols]
    arrays["origin"] = np.array([a.set.name] * n)
    arrays["columns_from"] = np.array(a.set.name)
    np.savez_compressed(a.out, **arrays)
    print(a.out, n, "columns, mp_physics 8, dt", float(z["dt"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
