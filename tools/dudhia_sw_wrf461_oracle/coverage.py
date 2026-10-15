#!/usr/bin/env python3
"""Which SWPARA branches the oracle columns reach.

    python coverage.py BUILD_DIR [--json OUT]

Replays SWPARA's control flow in float32 on the host for every column and
case and counts each branch: night return, the three XMU intervals, the
ALW cap at 3.999, the XABS/XALB/XABSC clamps to zero, the TRANS0 < 1
rescale and the SDOWN floor at 1e-9.  Values come from NumPy's float32
power/log10, not the C library's, so a branch decided within an ulp of
its threshold may count differently from WRF's run; the counts describe
the oracle's reach and grade nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.verify import dudhia_sw_oracle as do  # noqa: E402

f = np.float32


ALBTAB = np.asarray([0, 0, 0, 0, 69, 58, 40, 15, 90, 80, 70, 60,
                     94, 90, 82, 78, 96, 92, 85, 80], np.float32)
ABSTAB = np.asarray([0, 0, 0, 0, 0, 2.5, 4, 5, 0, 2.6, 7, 10,
                     0, 3.3, 10, 14, 0, 3.7, 10, 15], np.float32)
XMUVAL = np.asarray([0.0, 0.2, 0.5, 1.0], np.float32)


def _interp(tab, iil, iu, jjl, ju, xi, yj):
    a, b = f(xi - f(iil)), f(f(iu) - xi)
    c, d = f(yj - f(jjl)), f(f(ju) - yj)
    s = f(f(tab[(ju - 1) * 4 + iu - 1] * a) * c)
    s = f(s + f(f(tab[(ju - 1) * 4 + iil - 1] * b) * c))
    s = f(s + f(f(tab[(jjl - 1) * 4 + iu - 1] * a) * d))
    return f(s + f(f(tab[(jjl - 1) * 4 + iil - 1] * b) * d))


def walk(col, inputs, case, counts):
    nz = inputs["t"].shape[0]
    mu = inputs["coszen"][col]
    solcon = inputs["solcon"][case]
    cssca = f(f(inputs["swrad_scat"][case]) * f(1.0e-5))
    icloud = int(inputs["icloud"][case])
    if mu <= f(1.0e-9) or solcon <= f(1.0e-9):
        counts["night_return"] += 1
        return
    counts["day_columns"] += 1
    iil = 3 if mu > f(0.5) else 2 if mu > f(0.2) else 1
    counts[f"xmu_interval_iil_{iil}"] += 1
    iu = iil + 1
    xi = f(f(f(mu - XMUVAL[iil - 1]) / f(XMUVAL[iu - 1] - XMUVAL[iil - 1]))
           + f(iil))
    sd1 = f(solcon * mu)
    sd = sd1
    ww = uv = dsca = dabs = dscld = totabs = oldalb = oldabc = f(0.0)
    for kk in range(nz):
        k = nz - 1 - kk
        g = {n: inputs[n][k, col] for n in do.INPUT_FIELDS}
        counts["layers"] += 1
        ro = f(g["p"] / f(f(287.0) * g["t"]))
        dz = g["dz8w"]
        uv = f(uv + f(f(f(ro * max(f(0), g["qv"])) * dz) * f(1000)))
        if icloud:
            q = f(max(f(0), g["qc"]) + f(f(0.1) * max(f(0), g["qi"])))
            q = f(q + f(f(0.05) * max(f(0), g["qr"])))
            q = f(q + f(f(0.02) * max(f(0), g["qs"])))
            q = f(q + f(f(0.05) * max(f(0), g["qg"])))
            ww = f(ww + f(f(f(ro * f(1000)) * dz) * q))
        wgm = f(ww / mu)
        ugcm = f(f(uv * f(1.0e-4)) / mu)
        old = totabs
        totabs = f(f(f(2.9) * ugcm)
                   / f(np.power(f(f(1) + f(f(141.5) * ugcm)), f(0.635))
                       + f(f(5.925) * ugcm)))
        xsca = f(f(cssca * f(ro * dz)) / mu)
        xabs = f(f(f(totabs - old) * f(f(sd1 - dscld) - dsca)) / sd)
        if xabs < 0:
            counts["xabs_clamped"] += 1
            xabs = f(0)
        alw = f(np.log10(f(wgm + f(1))))
        if alw > f(3.999):
            counts["alw_capped_at_3.999"] += 1
            alw = f(3.999)
        jjl = int(alw) + 1
        counts[f"water_path_interval_jjl_{jjl}"] += 1
        yj = f(alw + f(1))
        alba = _interp(ALBTAB, iil, iu, jjl, jjl + 1, xi, yj)
        absc = _interp(ABSTAB, iil, iu, jjl, jjl + 1, xi, yj)
        avail = f(f(sd1 - dsca) - dabs)
        xalb = f(f(f(alba - oldalb) * avail) / sd)
        xabsc = f(f(f(absc - oldabc) * avail) / sd)
        if xalb < 0:
            counts["xalb_clamped"] += 1
            xalb = f(0)
        if xabsc < 0:
            counts["xabsc_clamped"] += 1
            xabsc = f(0)
        dscld = f(dscld + f(f(f(xalb + xabsc) * sd) * f(0.01)))
        dsca = f(dsca + f(xsca * sd))
        dabs = f(dabs + f(xabs * sd))
        oldalb, oldabc = alba, absc
        trans = f(f(f(f(f(100) - xalb) - xabsc) - f(xabs * f(100)))
                  - f(xsca * f(100)))
        if trans < f(1):
            counts["trans0_below_1_rescaled"] += 1
            trans = f(1)
        new_sd = f(f(sd * trans) * f(0.01))
        if new_sd < f(1.0e-9):
            counts["sdown_floor_1e-9"] += 1
        sd = max(f(1.0e-9), new_sd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    inputs = do.read_inputs_bin(args.build_dir / "dudhia-inputs.bin")
    from collections import Counter
    counts = Counter()
    for case in range(inputs["solcon"].size):
        for col in range(inputs["t"].shape[1]):
            walk(col, inputs, case, counts)
    text = json.dumps(dict(sorted(counts.items())), indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
