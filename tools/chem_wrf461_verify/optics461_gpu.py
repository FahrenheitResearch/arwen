"""Measure WOOF's GOCART optics kernel against the WRF 4.6.1 optics oracle.

    python tools/chem_wrf461_verify/optics461_gpu.py FIX461 OUT.json

FIX461 is build_optics461.sh's fixtures folder.  Each case is run through
``gpuwm.core.chem_optics.gocart_optics`` exactly as
tests/test_chem_optics_wrf471_parity.py runs it against the published 4.7.1
oracle, on the same inputs (the inputs are checked word for word against the
published case first), and every column output WRF writes is compared:
tauaer, extaer, waer, gaer, bscoef (4 SW bands), tauaerlw, extaerlw (16 LW
bands), EXTCOF55 and AOD5502D, plus the RRTMG band conversion of the
oracle's own tauaer/waer/gaer.  The per-bin intermediates (radius, number,
refractive index) are not compared: 4.6.1 has 8 sections, WOOF and 4.7.1 have
9, so they have no common shape.

The report also gives the published-4.7.1-vs-4.6.1 distance of each output, so
it shows the WOOF miss is exactly the WRF version difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from gpuwm.verify.chem_oracle import load_case, ulp_table

ROOT = Path(__file__).resolve().parents[2]
FIX471 = ROOT / "tests/data/oracles/chem/gocart/optics"
INPUTS = ("chem", "alt", "relhum", "dz8w")
SW_LW = ("tauaer", "extaer", "waer", "gaer", "bscoef", "tauaerlw", "extaerlw")


def _bands(v):
    if v.ndim == 2:
        v = v[:, :, None]
    return np.ascontiguousarray(v[:, :, None, :])


def _field(v):
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(v.transpose(1, 2, 0)))


def main(fix461: str, out: str) -> dict:
    import cupy as cp
    from gpuwm.core.chem_optics import data_dir, gocart_optics, rrtmg_sw_bands
    rows = json.loads((data_dir() / "gocart_simple_optics_rows.json").read_text())
    fix461 = Path(fix461)
    cases = sorted(p.name for p in fix461.iterdir() if (p / "chem.bin").is_file())
    report = {"cases": {}, "woof_vs_461": {}, "wrf471_vs_461": {}, "woof_vs_471": {}}

    def fold(bucket, name, table):
        agg = report[bucket].setdefault(name, {"max_ulp": 0, "n_nonzero": 0, "n": 0})
        agg["max_ulp"] = max(agg["max_ulp"], table["max_ulp"])
        agg["n_nonzero"] += table["n_nonzero"]
        agg["n"] += table["n"]

    for case in cases:
        f = load_case(fix461 / case)
        g = load_case(FIX471 / case)
        for name in INPUTS:
            np.testing.assert_array_equal(f[name].view(np.uint32), g[name].view(np.uint32),
                                          err_msg=f"{case}/{name}: inputs differ")
        fields = [_field(f["chem"][:, :, :, i + 1]) for i in range(19)]
        o = gocart_optics(rows, fields, _field(f["alt"]), _field(f["relhum"]), _field(f["dz8w"]))
        pairs = {name: (cp.asnumpy(o[name]), _bands(f[name]), _bands(g[name])) for name in SW_LW}
        pairs["EXTCOF55"] = (cp.asnumpy(o["EXTCOF55"]), f["EXTCOF55"].transpose(1, 2, 0).copy(),
                             g["EXTCOF55"].transpose(1, 2, 0).copy())
        pairs["AOD5502D"] = (cp.asnumpy(o["AOD5502D"]), f["AOD5502D"].reshape(1, -1),
                             g["AOD5502D"].reshape(1, -1))
        b = rrtmg_sw_bands(*[cp.asarray(_bands(f[n])) for n in ("tauaer", "waer", "gaer")])
        for name in ("tau", "ssa", "asm"):
            ref = f["band_" + name].transpose(1, 2, 0)[:, :, None, :].copy()
            pairs["band_" + name + " (from 4.6.1 tauaer)"] = (cp.asnumpy(b[name]), ref, ref)
        per_case = {}
        for name, (woof, r461, r471) in pairs.items():
            t = ulp_table(woof, r461)
            per_case[name] = t
            fold("woof_vs_461", name, t)
            fold("wrf471_vs_461", name, ulp_table(r471, r461))
            fold("woof_vs_471", name, ulp_table(woof, r471))
        report["cases"][case] = {
            "max_ulp_vs_461": max(t["max_ulp"] for t in per_case.values()),
            "values_differing_vs_461": sum(t["n_nonzero"] for t in per_case.values()),
            "values": sum(t["n"] for t in per_case.values())}
    report["n_cases"] = len(cases)
    Path(out).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2])
    for bucket in ("woof_vs_461", "wrf471_vs_461", "woof_vs_471"):
        print(bucket)
        for name, t in r[bucket].items():
            print(f"  {name:34s} max_ulp {t['max_ulp']:>11d}  differing {t['n_nonzero']:>6d} / {t['n']}")
