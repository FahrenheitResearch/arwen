#!/usr/bin/env python3
"""Bitwise comparison of the WOOF GPU run against the WRF Fortran driver.

Every compared word is float32; the distance is the ULP distance between the
two words' positions on the ordered float32 line (``0`` means the same bits,
with ``+0`` and ``-0`` counted as distinct).  Writes ``comparison-<arm>.json``
and prints a one-line verdict per arm.

SFCEVP: WRF 4.6.1 LSMRUC adds ``qfx*dt`` to SFCEVP twice on every land step
(module_sf_ruclsm.F:1095 and :1116).  That is a WRF defect and WOOF does not
copy it, so WOOF's SFCEVP is graded against the driver's single-count word
(``sfcevp_once_acc`` free running, ``sfcevp_once_step`` replayed; the same
float32 expression applied once).  The raw WRF
word is still reported, as ``sfcevp_wrf_double_count``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layout  # noqa: E402
from columns import PROFILES  # noqa: E402


def ordered(a):
    w = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.int64)
    return np.where(w & 0x80000000, 0xffffffff - w, w | 0x80000000)


def ulp(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = np.abs(ordered(a) - ordered(b))
    return d


def grade_field(woof, wrf):
    d = ulp(woof, wrf)
    invalid = ~np.isfinite(woof) | ~np.isfinite(wrf)
    d = np.where(invalid, np.maximum(d, 1), d)
    with np.errstate(invalid="ignore"):
        diff = np.abs(woof.astype(np.float64) - wrf.astype(np.float64))
    diff = np.where(invalid, np.inf, diff)
    return d, diff


def compare(case_dir: Path, arm: str, woof_name: str = "woof.npz",
            sfcevp: str = "once") -> dict:
    meta = json.loads((case_dir / "case.json").read_text())
    ncol, nzs, nsteps = meta["ncol"], meta["nzs"], meta["nsteps"]
    regime = np.array(meta["regime"])
    woof = np.load(case_dir / woof_name)
    wrf = layout.read_outputs(case_dir / f"outputs-{arm}.bin", ncol, nzs, nsteps)
    report = {"arm": arm, "woof": woof_name, "ncol": ncol, "nzs": nzs,
              "nsteps": nsteps, "wrf_exact": meta.get("wrf_exact"),
              "device": meta.get("device"), "init": {}, "steps": [],
              "fields": {}}
    total_words = 0
    total_bad = 0
    worst = 0
    # cold start
    for name in layout.INIT_2D + layout.INIT_PROFILES:
        d, diff = grade_field(woof[f"init__{name}"], wrf["init"][name])
        report["init"][name] = {"max_ulp": int(d.max()),
                                "words": int(d.size),
                                "differing": int(np.count_nonzero(d))}
        total_words += d.size
        total_bad += int(np.count_nonzero(d))
        worst = max(worst, int(d.max()))
    bad_regimes: dict[str, set] = {}
    first_bad_step = None
    for k in range(1, nsteps + 1):
        out = wrf["steps"][k - 1]
        step_report = {}
        for name in layout.OUT_2D + PROFILES:
            ours = woof[f"step{k}__{name}"]
            ref = out[name]
            if name == "sfcevp":
                raw_d, _ = grade_field(ours, ref)
                step_report["sfcevp_wrf_double_count"] = {
                    "max_ulp": int(raw_d.max()),
                    "differing": int(np.count_nonzero(raw_d))}
                if sfcevp == "once":
                    ref = out[layout.SFCEVP_ONCE_STEP if "replay" in woof_name
                              else layout.SFCEVP_ONCE_ACC]
            d, diff = grade_field(ours, ref)
            nbad = int(np.count_nonzero(d))
            entry = {"max_ulp": int(d.max()), "differing": nbad,
                     "max_abs": float(diff.max())}
            if nbad:
                cols = np.nonzero(d.reshape(-1, ncol).max(axis=0))[0]
                entry["columns"] = [int(c) for c in cols[:40]]
                for c in cols:
                    bad_regimes.setdefault(str(regime[c]), set()).add(name)
                if first_bad_step is None:
                    first_bad_step = k
            step_report[name] = entry
            agg = report["fields"].setdefault(
                name, {"max_ulp": 0, "differing": 0, "max_abs": 0.0})
            agg["max_ulp"] = max(agg["max_ulp"], entry["max_ulp"])
            agg["differing"] += nbad
            agg["max_abs"] = max(agg["max_abs"], entry["max_abs"])
            total_words += d.size
            total_bad += nbad
            worst = max(worst, int(d.max()))
        report["steps"].append(step_report)
    report["total_words"] = int(total_words)
    report["differing_words"] = int(total_bad)
    report["max_ulp"] = int(worst)
    report["first_differing_step"] = first_bad_step
    report["regimes_with_differences"] = {k: sorted(v) for k, v in
                                          sorted(bad_regimes.items())}
    report["regimes"] = sorted(set(meta["regime"]))
    (case_dir / f"comparison-{arm}{'' if woof_name == 'woof.npz' else '-' + woof_name.split('.')[0]}.json").write_text(
        json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("case_dir", type=Path)
    p.add_argument("--arms", default="defined,stock,o0")
    p.add_argument("--woof", default="woof.npz")
    p.add_argument("--sfcevp", choices=("once", "wrf"), default="once")
    args = p.parse_args(argv)
    rc = 0
    for arm in args.arms.split(","):
        if not (args.case_dir / f"outputs-{arm}.bin").exists():
            continue
        r = compare(args.case_dir, arm, args.woof, args.sfcevp)
        bad = {n: v for n, v in r["fields"].items() if v["differing"]}
        print(f"{args.case_dir.name} arm={arm} woof={args.woof} "
              f"exact={r['wrf_exact']} words={r['total_words']} "
              f"differing={r['differing_words']} max_ulp={r['max_ulp']} "
              f"first_step={r['first_differing_step']} "
              f"init={ {n: v['max_ulp'] for n, v in r['init'].items()} } "
              f"bad={ {n: (v['max_ulp'], v['differing']) for n, v in bad.items()} }")
        if r["differing_words"]:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
