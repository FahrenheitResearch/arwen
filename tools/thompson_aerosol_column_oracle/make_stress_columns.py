#!/usr/bin/env python3
"""Stress column sets for the mp=28 column oracle, beyond the 153.

Four sets at different level counts, each written as its own
``columns-nzN.npz`` in the ``make_columns.py`` layout:

* ``nz`` 2, 17, 49 and 73: every synthetic regime of ``make_columns.py``,
  two seeded variants each (seed 739031 + nz), so the shallow (<= 64 level)
  and generic (> 64) kernel paths and a two-level column are all exercised.
* at 49 levels, also: seven isothermal columns at the phase thresholds
  (190, 233.15, 253.15, 273.14999, 273.15, 273.15003 and 310 K) at relative
  humidities 0.001, 1.0 and 1.08, each carrying 1e-4 kg/kg of every
  condensate; a high-terrain column at 60 % of its pressure with the ground
  raised 6 km; and three convective columns whose five condensates are all
  the smallest positive float32, all 1e-20 and all 0.02 kg/kg.

``--bounded`` writes one 49-level set (``bounded/columns-nz49.npz``) in
which the warm isothermal columns stop at 700 hPa instead of 50 hPa, so no
level is supersaturated past its total-pressure limit.

These are one-step stress inputs, not physically matched states; their
use is to show the port and WRF take the same branch everywhere, including
far outside the regimes a forecast visits.  The columns were first written
for an independent check of this oracle and are reproduced here so the
sets can be rebuilt from the tree.

usage: make_stress_columns.py OUT_DIR [--bounded]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_columns as m  # noqa: E402

ISOTHERMAL_K = (190.0, 233.15, 253.15, 273.14999, 273.15, 273.15003, 310.0)
ISOTHERMAL_RH = (0.001, 1.0, 1.08)


def build(nz: int, bounded: bool) -> list:
    eta = m.eta_levels
    m.NZ = nz
    m.eta_levels = lambda n=nz: eta(n)
    rng = np.random.default_rng(739031 + nz)
    entries = []
    for name, regime in m.REGIMES.items():
        for variant in range(2):
            c, t, p, nums = regime(rng)
            c = m.finish(c, t, p, nums, rng)
            entries.append((f"new seed {name} {variant} nz{nz}", name, c))
    if nz == 49:
        for temp in ISOTHERMAL_K:
            for rh in ISOTHERMAL_RH:
                m.P_TOP = 70000.0 if bounded and temp >= 273.0 else 5000.0
                c, t, p, _ = m.column(
                    101325.0, 0.0, lambda pp, v=temp: np.full_like(pp, v),
                    lambda pp, v=rh: np.full_like(pp, v),
                    lambda zz: np.zeros_like(zz))
                for s in ("qc", "qr", "qi", "qs", "qg"):
                    c[s] = np.full(nz, 1e-4)
                c = m.finish(c, t, p, dict(nc_cm3=100, ni_l=100,
                                           nwfa_cc=800, nifa_l=1000), rng)
                entries.append((f"isothermal {temp}K RH{rh}",
                                "isothermal phase thresholds", c))
        m.P_TOP = 5000.0
        c, t, p, nums = m.regime_high_terrain(rng)
        c = m.finish(c, t, p, nums, rng)
        c["p"] *= 0.6
        c["geop"] += np.float64(6000 * 9.81)
        entries.append(("thin air raised terrain", "thin air", c))
        for mass in (np.nextafter(np.float32(0), np.float32(1)), 1e-20, 0.02):
            c, t, p, nums = m.regime_convective(rng)
            c = m.finish(c, t, p, nums, rng)
            for s in ("qc", "qr", "qi", "qs", "qg"):
                c[s][:] = mass
            entries.append((f"all hydrometeors {mass}",
                            "tiny or heavy hydrometeors", c))
    m.eta_levels = eta
    return entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--bounded", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipt = []
    for nz in ((49,) if a.bounded else (2, 17, 49, 73)):
        entries = build(nz, a.bounded)
        cols = m.stack(entries)
        assert all(np.isfinite(v).all() for v in cols.values()
                   if v.dtype == np.float32)
        np.savez(out / f"columns-nz{nz}.npz", **cols)
        receipt.append({"nz": nz, "columns": len(entries),
                        "labels": [str(x) for x in cols["labels"]]})
    (out / "columns.json").write_text(json.dumps(receipt, indent=1) + "\n")
    for r in receipt:
        print(f"nz {r['nz']}: {r['columns']} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
