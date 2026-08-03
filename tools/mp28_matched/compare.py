"""Evaluate the mp=28 matched-trajectory comparison against its declared rule.

Reads the six run directories written by ``run_arwen.py`` and
``extract_wrfout.py`` and computes exactly the metrics M1-M8 and the verdict
conditions V1-V4 declared in
``docs/public/validation/mp28-matched-trajectory.md`` BEFORE any run existed.
Nothing here selects a statistic; the statistics were fixed in that commit.

Usage:
    python compare.py --runs DIR --out DIR [--wrf-build vec]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: Analysis window (s).  Declared: 0-5400 primary, 5400-7200 secondary.
PRIMARY_END = 5400.0

#: The verdict rule's checkpoints for V2.
V2_TIMES = (1800.0, 3600.0, 5400.0)

#: M1-M6 scalar metric keys as they appear in a series row.
SCALAR_METRICS = (
    "rainnc_sum", "rainnc_max", "w_max",
    "qc_mean", "qc_max", "qr_mean", "qr_max", "qi_mean", "qi_max",
    "qs_mean", "qs_max", "qg_mean", "qg_max",
    "nc_mean", "nc_max", "nwfa_mean", "nifa_mean",
)

#: The three headline aerosol metrics V2 is stated on.
HEADLINE = ("nc_mean", "nwfa_mean", "rainnc_sum")

#: Fields entering M8, the normalised RMS trajectory-divergence curve.
M8_FIELDS_MP8 = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                 "QNRAIN", "QNICE", "W", "T")
M8_FIELDS_MP28 = M8_FIELDS_MP8 + ("QNCLOUD", "QNWFA", "QNIFA")


def load_run(d: Path) -> dict:
    series = json.loads((d / "series.json").read_text())
    frames = {}
    for p in sorted(d.glob("frame_t*.npz")):
        t = float(p.stem.split("frame_t")[1])
        frames[t] = p
    return {"dir": d, "series": {round(r["time_s"], 3): r for r in series},
            "frames": frames}


def at(run: dict, t: float, key: str):
    row = run["series"].get(round(t, 3))
    return None if row is None else row.get(key)


def rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b||2 / ||b||2, with an all-zero reference reported as 0/None."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    den = float(np.sqrt((b * b).mean()))
    num = float(np.sqrt(((a - b) ** 2).mean()))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def frame_divergence(arwen: dict, wrf: dict, fields) -> dict:
    out = {}
    for t in sorted(set(arwen["frames"]) & set(wrf["frames"])):
        A = np.load(arwen["frames"][t])
        W = np.load(wrf["frames"][t])
        row = {}
        for f in fields:
            if f in A.files and f in W.files:
                row[f] = rms_diff(A[f], W[f])
        out[t] = row
    return out


def byte_identical(a: Path, b: Path) -> dict:
    """Dual-run byte comparison -- the 5090 has no ECC."""
    result = {"frames_compared": 0, "identical": True, "differing": []}
    for p in sorted(a.glob("frame_t*.npz")):
        q = b / p.name
        if not q.exists():
            result["identical"] = False
            result["differing"].append(f"{p.name}: missing in second run")
            continue
        A, B = np.load(p), np.load(q)
        result["frames_compared"] += 1
        for f in A.files:
            if not np.array_equal(A[f], B[f], equal_nan=True):
                result["identical"] = False
                result["differing"].append(f"{p.name}:{f}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--wrf-build", default="vec")
    ap.add_argument("--as-model", default="arwen",
                    help="which pair stands in the tested-model slot: "
                         "'arwen' (the comparison) or a WRF build tag such "
                         "as 'novec' (WRF against itself -- the control)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R = args.runs
    B = args.wrf_build

    if args.as_model == "arwen":
        tested = (R / "arwen-mp08-a", R / "arwen-mp28-a")
        tested_dual = (R / "arwen-mp08-b", R / "arwen-mp28-b")
    else:
        tested = (R / f"wrf-{args.as_model}-mp08-x",
                  R / f"wrf-{args.as_model}-mp28-x")
        tested_dual = None
    runs = {
        "arwen_mp08": load_run(tested[0]),
        "arwen_mp28": load_run(tested[1]),
        "wrf_mp08": load_run(R / f"wrf-{B}-mp08-x"),
        "wrf_mp28": load_run(R / f"wrf-{B}-mp28-x"),
    }
    alt = {}
    for name, tag in (("wrf_alt_mp08", "novec-mp08"),
                      ("wrf_alt_mp28", "novec-mp28")):
        p = R / f"wrf-{tag}-x"
        if (p / "series.json").exists():
            alt[name] = load_run(p)

    times = sorted(set(runs["wrf_mp08"]["series"])
                   & set(runs["arwen_mp08"]["series"])
                   & set(runs["wrf_mp28"]["series"])
                   & set(runs["arwen_mp28"]["series"]))
    primary = [t for t in times if t <= PRIMARY_END]

    report: dict = {"times": times, "primary_end_s": PRIMARY_END,
                    "wrf_build": B, "tested_model": args.as_model,
                    "tested_dirs": [str(p) for p in tested]}

    # --- dual-run corruption screen -------------------------------------
    report["dual_run"] = (
        {"mp08": byte_identical(tested[0], tested_dual[0]),
         "mp28": byte_identical(tested[1], tested_dual[1])}
        if tested_dual is not None else
        {"note": "dual-run screen applies to the GPU model only"})

    # --- t = 0 state agreement (the initialisation floor) ----------------
    report["t0_field_rms"] = {
        "mp08": frame_divergence(runs["arwen_mp08"], runs["wrf_mp08"],
                                 M8_FIELDS_MP8).get(0.0, {}),
        "mp28": frame_divergence(runs["arwen_mp28"], runs["wrf_mp28"],
                                 M8_FIELDS_MP28).get(0.0, {}),
    }

    # --- M1-M6 raw series ------------------------------------------------
    raw = {}
    for name, run in list(runs.items()) + list(alt.items()):
        raw[name] = {m: [at(run, t, m) for t in times] for m in SCALAR_METRICS}
    report["raw"] = raw

    # --- signature, floor, WRF flag ambiguity ---------------------------
    sig, floor, ambig = {}, {}, {}
    for m in SCALAR_METRICS:
        d_w, d_a, f, v = [], [], [], []
        for t in times:
            w8, w28 = at(runs["wrf_mp08"], t, m), at(runs["wrf_mp28"], t, m)
            a8, a28 = at(runs["arwen_mp08"], t, m), at(runs["arwen_mp28"], t, m)
            d_w.append(None if w8 is None or w28 is None else w28 - w8)
            d_a.append(None if a8 is None or a28 is None else a28 - a8)
            f.append(None if w8 is None or a8 is None else abs(a8 - w8))
            if "wrf_alt_mp28" in alt:
                x28 = at(alt["wrf_alt_mp28"], t, m)
                x8 = at(alt["wrf_alt_mp08"], t, m)
                v.append(None if None in (w28, x28) else
                         {"mp28": abs(w28 - x28),
                          "mp08": None if None in (w8, x8) else abs(w8 - x8)})
        sig[m] = {"wrf": d_w, "arwen": d_a}
        floor[m] = f
        if v:
            ambig[m] = v
    report["signature"] = sig
    report["floor_mp08_model_pair"] = floor
    report["wrf_flag_ambiguity"] = ambig

    # --- M7 vertical profiles at the V2 checkpoints ----------------------
    profiles = {}
    for t in V2_TIMES:
        if t not in runs["wrf_mp28"]["frames"]:
            continue
        entry = {}
        for name, key in (("wrf_mp08", "wrf_mp08"), ("wrf_mp28", "wrf_mp28"),
                          ("arwen_mp08", "arwen_mp08"),
                          ("arwen_mp28", "arwen_mp28")):
            z = np.load(runs[key]["frames"][t])
            entry[name] = {
                "qc": z["QCLOUD"].astype(np.float64).mean(axis=(1, 2)).tolist(),
                "qi": z["QICE"].astype(np.float64).mean(axis=(1, 2)).tolist(),
            }
        profiles[t] = entry
    report["profiles"] = profiles

    # --- M8 trajectory divergence ---------------------------------------
    report["m8_divergence"] = {
        "mp08": {str(k): v for k, v in frame_divergence(
            runs["arwen_mp08"], runs["wrf_mp08"], M8_FIELDS_MP8).items()},
        "mp28": {str(k): v for k, v in frame_divergence(
            runs["arwen_mp28"], runs["wrf_mp28"], M8_FIELDS_MP28).items()},
    }

    # --- V1: sign agreement where WRF's signature clears the floor -------
    v1_pairs, v1_agree, v1_detail = 0, 0, []
    for m in SCALAR_METRICS:
        for i, t in enumerate(times):
            if t > PRIMARY_END:
                continue
            dw, da, f = sig[m]["wrf"][i], sig[m]["arwen"][i], floor[m][i]
            if dw is None or da is None or f is None:
                continue
            if abs(dw) <= f:
                continue
            v1_pairs += 1
            ok = (dw > 0) == (da > 0)
            v1_agree += int(ok)
            if not ok:
                v1_detail.append({"metric": m, "time_s": t,
                                  "delta_wrf": dw, "delta_arwen": da,
                                  "floor": f})
    report["V1"] = {"resolvable_pairs": v1_pairs, "sign_agree": v1_agree,
                    "fraction": (v1_agree / v1_pairs) if v1_pairs else None,
                    "threshold": 0.90,
                    "pass": bool(v1_pairs and v1_agree / v1_pairs >= 0.90),
                    "disagreements": v1_detail}

    # --- V2: floor-calibrated magnitude on the headline metrics ----------
    v2_rows, v2_pass = [], True
    for m in HEADLINE:
        for t in V2_TIMES:
            if t not in times:
                continue
            i = times.index(t)
            dw, da, f = sig[m]["wrf"][i], sig[m]["arwen"][i], floor[m][i]
            if dw is None or da is None or f is None:
                continue
            bound = max(0.5 * abs(dw), f)
            ok = abs(da - dw) <= bound
            v2_pass &= ok
            v2_rows.append({"metric": m, "time_s": t, "delta_wrf": dw,
                            "delta_arwen": da, "abs_diff": abs(da - dw),
                            "bound": bound, "floor": f, "pass": bool(ok)})
    report["V2"] = {"rows": v2_rows, "pass": bool(v2_pass)}

    # --- V3: mp=28 disagreement not > 3x the mp=8 disagreement -----------
    v3_rows, v3_pass = [], True
    for m in SCALAR_METRICS[:13]:          # M1-M4
        for i, t in enumerate(times):
            if t > PRIMARY_END or t == 0.0:
                continue
            w8, w28 = at(runs["wrf_mp08"], t, m), at(runs["wrf_mp28"], t, m)
            a8, a28 = at(runs["arwen_mp08"], t, m), at(runs["arwen_mp28"], t, m)
            if None in (w8, w28, a8, a28):
                continue
            d8, d28 = abs(a8 - w8), abs(a28 - w28)
            scale = max(abs(w8), abs(w28))
            if scale == 0.0 or d8 / max(scale, 1e-300) < 1e-6:
                continue                    # both agree at the noise level
            ratio = d28 / d8 if d8 > 0 else float("inf")
            ok = ratio <= 3.0
            v3_pass &= ok
            v3_rows.append({"metric": m, "time_s": t, "d_mp08": d8,
                            "d_mp28": d28, "ratio": ratio, "pass": bool(ok)})
    for t_str, row8 in report["m8_divergence"]["mp08"].items():
        t = float(t_str)
        if t > PRIMARY_END or t == 0.0:
            continue
        row28 = report["m8_divergence"]["mp28"].get(t_str, {})
        for f, d8 in row8.items():
            d28 = row28.get(f)
            if d28 is None or not np.isfinite(d8) or d8 <= 0.0:
                continue
            ratio = d28 / d8
            ok = ratio <= 3.0
            v3_pass &= ok
            v3_rows.append({"metric": f"M8:{f}", "time_s": t, "d_mp08": d8,
                            "d_mp28": d28, "ratio": ratio, "pass": bool(ok)})
    report["V3"] = {"rows": v3_rows, "pass": bool(v3_pass),
                    "worst": max((r["ratio"] for r in v3_rows), default=None)}

    # --- V4: finite, bounded, no depletion trend -------------------------
    NWFA_FLOOR, NWFA_CEIL = 11.1e6, 9999.0e6
    NIFA_FLOOR, NIFA_CEIL = 5.0e3, 9999.0e6
    nonfinite, bound_viol = [], []
    for key in ("arwen_mp08", "arwen_mp28"):
        for t, p in sorted(runs[key]["frames"].items()):
            z = np.load(p)
            for f in z.files:
                if not np.isfinite(z[f]).all():
                    nonfinite.append({"run": key, "time_s": t, "field": f})
    for t, p in sorted(runs["arwen_mp28"]["frames"].items()):
        z = np.load(p)
        for f, lo, hi in (("QNWFA", NWFA_FLOOR, NWFA_CEIL),
                          ("QNIFA", NIFA_FLOOR, NIFA_CEIL)):
            lo_v = float(z[f].min())
            hi_v = float(z[f].max())
            if lo_v < lo or hi_v > hi:
                bound_viol.append({"time_s": t, "field": f,
                                   "min": lo_v, "max": hi_v})
    nwfa = [at(runs["arwen_mp28"], t, "nwfa_mean") for t in times]
    nifa = [at(runs["arwen_mp28"], t, "nifa_mean") for t in times]
    def monotone_drop(v):
        v = [x for x in v if x is not None]
        return bool(v) and all(b <= a for a, b in zip(v, v[1:])) \
            and v[-1] < 0.99 * v[0]
    report["V4"] = {
        "nonfinite": nonfinite, "bound_violations": bound_viol,
        "nwfa_mean": nwfa, "nifa_mean": nifa,
        "nwfa_monotone_depletion": monotone_drop(nwfa),
        "nifa_monotone_depletion": monotone_drop(nifa),
        "pass": bool(not nonfinite and not bound_viol
                     and not monotone_drop(nwfa)
                     and not monotone_drop(nifa)),
    }

    report["verdict"] = {
        "V1": report["V1"]["pass"], "V2": report["V2"]["pass"],
        "V3": report["V3"]["pass"], "V4": report["V4"]["pass"],
        "ship": bool(report["V1"]["pass"] and report["V2"]["pass"]
                     and report["V3"]["pass"] and report["V4"]["pass"]),
    }
    (args.out / "comparison.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["verdict"], indent=1))
    print("V1", report["V1"]["sign_agree"], "/", report["V1"]["resolvable_pairs"])
    print("V3 worst ratio", report["V3"]["worst"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
