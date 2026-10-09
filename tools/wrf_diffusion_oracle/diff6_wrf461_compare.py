"""Grade WOOF's product sixth-order path against compiled WRF 4.6.1, word for word.

For every oracle case this runs exactly what ``dycore.prepare_fixed_tendencies``
runs for one diff6 row under ``diff_6th_form = "wrf_461"``:

* ``RunConfig`` built from the case (dt formed as WOOF forms it from the
  namelist rational, experiment.py: float32 time_step + num/den);
* the row's factor and dt from ``_diff6_factor`` / ``_diff6_dt`` (dry rows
  take dt, moist and scalar rows dt/3, as WRF's rk_scalar_tend does);
* ``dycore.add_diff6_row``, the production row itself (through
  ``gpuwm.verify.diff6_wrf461_oracle.launch_diff6_wrf461_case``), accumulating in
  place onto the carrying slot with WRF's loop bounds and seam face.

The carrying slot starts as the case's incoming tendency (zero, or the
case's nonzero ``tendency`` input); WRF forms ``(T0 + tendency_x) +
tendency_y`` on it, and so must the product.
The pre-fix composition (increment on a zeroed temporary, then added)
is also measured on every nonzero-incoming case, as a discrimination check.

Run once with GPUWM_WRF_EXACT=1 (strict) and once without (default).
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import numpy as np

MAPS = ("MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY")


def ulp_stats(got, want):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    got = np.ascontiguousarray(got, dtype=np.float32); want = np.ascontiguousarray(want, dtype=np.float32)
    changed = got.view(np.uint32) != want.view(np.uint32)
    d = fp32_ulp_distance(got, want)
    return changed, {"words": int(got.size), "different_words": int(changed.sum()),
                     "max_ulp": int(d.max(initial=0)),
                     "max_abs": float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)), initial=0)),
                     "nonfinite": int((~np.isfinite(got)).sum() + (~np.isfinite(want)).sum())}


def config(case):
    from gpuwm.verify.diff6_wrf461_oracle import diff6_wrf461_config
    return diff6_wrf461_config(case)


def product_row(cfg, case, inputs, **kw):
    """The production row; cfg is rebuilt from the case by the shared helper."""
    from gpuwm.verify.diff6_wrf461_oracle import launch_diff6_wrf461_case
    return launch_diff6_wrf461_case(case, inputs, **kw)


def regimes(state, group):
    """Mass-column regime masks (diagnostic labels only; every column is graded)."""
    land = state["XLAND"] < 1.5
    out = {"ocean": ~land, "high_terrain": state["HGT"] > 1500.0}
    if "W" in state:
        conv = np.abs(state["W"]).max(0) > 1.0
        if "QGRAUP" in state:
            conv |= state["QGRAUP"].max(0) > 1e-4
        out["convective"] = conv
    if "QSNOW" in state:
        out["snow_ice"] = (state["QSNOW"] + state["QICE"]).max(0) > 1e-5
    if group.startswith("conus"):
        th = state["T"]
        out["stable_night_land"] = land & ((th[4] - th[0]) > 2.0)
    return out


def column_of(changed, stagger, ny, nx):
    """Collapse a mismatch map to mass columns (staggered faces -> donor cell)."""
    c = changed.any(axis=0)
    return c[:ny, :nx]


def compare(oracle_dir, result, groups=None):
    import cupy as cp
    from gpuwm import wrf_exact
    from gpuwm.core.dycore import launch_diff6
    meta = json.loads((oracle_dir / "oracle.json").read_text())
    rows, regime_rows = [], {}
    states = {}
    for case in meta["cases"]:
        g = case["group"]
        if groups and g not in groups:
            continue
        gdir = oracle_dir / g
        if g not in states:
            states = {g: dict(np.load(gdir / "state.npz"))}
        state = states[g]
        ref = np.load(gdir / f"ref-{case['case']:04d}.npy")
        ipath = gdir / f"inputs-{case['case']:04d}.npz"
        if ipath.exists():
            inputs = dict(np.load(ipath))
        else:
            var = case["var"]
            full = var == "W"
            inputs = {"field": state[var], "tendency": np.zeros_like(state[var]), "mut": state["MUT"],
                      "c1": state["C1F" if full else "C1H"], "c2": state["C2F" if full else "C2H"],
                      "phb": state["PHB"], **{m: state[m] for m in MAPS}}
        cfg = config(case)
        got = product_row(cfg, case, inputs)
        changed, stats = ulp_stats(got, ref)
        row = {k: case[k] for k in ("case", "group", "var", "stagger", "opt", "slopeopt", "mode_name", "time_step",
                                    "num", "den", "dx", "dy", "factor", "accumulate", "scalar_row")}
        row["product"] = stats
        if stats["different_words"]:
            idx = np.argwhere(changed)[:4].tolist()
            row["first_differences"] = [{"index": i, "woof": hex(int(got.view(np.uint32)[tuple(i)])),
                                         "wrf": hex(int(ref.view(np.uint32)[tuple(i)]))} for i in idx]
            rows_hit = sorted({i[1] for i in np.argwhere(changed).tolist()})
            row["rows_with_differences"] = rows_hit[:12]
        if case["accumulate"]:
            _, row["legacy_composition"] = ulp_stats(product_row(cfg, case, inputs, legacy_composition=True), ref)
        if case["scalar_row"] and case["case"] % 7 == 0:
            mut = product_row(cfg, case, inputs, scalar_dt_mutation=True)
            row["mutation_scalar_dt_not_divided_by_3_changed_words"] = int((mut.view(np.uint32) != ref.view(np.uint32)).sum())
        rows.append(row)
        if not ipath.exists():          # a full-domain real state: label its columns
            nz, ny, nx = case["dims"]
            cols = column_of(changed, case["stagger"], ny, nx)
            for name, mask in regimes(state, g).items():
                r = regime_rows.setdefault(f"{g}:{name}", {"columns": int(mask.sum()), "cases": 0,
                                                           "column_cases_with_difference": 0})
                r["cases"] += 1
                r["column_cases_with_difference"] += int((cols & mask).sum())
        print(f"case {case['case']:4d} {g:22s} {case['var']:6s} opt={case['opt']} slope={case['slopeopt']} "
              f"{case['mode_name']:9s} diff={stats['different_words']} max_ulp={stats['max_ulp']}", flush=True)
    summary = {
        "strict": wrf_exact.ENABLED,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__, "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "launcher_sha256": hashlib.sha256(inspect.getsource(launch_diff6).encode()).hexdigest(),
        "oracle_json_sha256": hashlib.sha256((oracle_dir / "oracle.json").read_bytes()).hexdigest(),
        "cases": len(rows),
        "words": sum(r["product"]["words"] for r in rows),
        "different_words": sum(r["product"]["different_words"] for r in rows),
        "max_ulp": max(r["product"]["max_ulp"] for r in rows),
        "max_abs": max(r["product"]["max_abs"] for r in rows),
        "regimes": regime_rows,
        "compile_receipt": wrf_exact.compile_receipt() if wrf_exact.ENABLED else None,
        "rows": rows,
    }
    for label, pred in (("zero_incoming_tendency", lambda r: not r["accumulate"]),
                        ("nonzero_incoming_tendency", lambda r: r["accumulate"])):
        sel = [r for r in rows if pred(r)]
        if sel:
            summary[label] = {"cases": len(sel), "words": sum(r["product"]["words"] for r in sel),
                              "different_words": sum(r["product"]["different_words"] for r in sel),
                              "max_ulp": max(r["product"]["max_ulp"] for r in sel)}
    ip = [r["legacy_composition"] for r in rows if "legacy_composition" in r]
    if ip:
        summary["legacy_composition_nonzero_incoming"] = {
            "cases": len(ip), "words": sum(s["words"] for s in ip),
            "different_words": sum(s["different_words"] for s in ip),
            "max_ulp": max(s["max_ulp"] for s in ip), "max_abs": max(s["max_abs"] for s in ip)}
    by_mode = {}
    for r in rows:
        m = by_mode.setdefault(f"{r['mode_name']}:{r['var'] if r['var'] in ('U', 'V', 'W') else 'mass'}",
                               {"cases": 0, "different_words": 0, "max_ulp": 0})
        m["cases"] += 1; m["different_words"] += r["product"]["different_words"]
        m["max_ulp"] = max(m["max_ulp"], r["product"]["max_ulp"])
    summary["by_mode_and_stagger"] = by_mode
    result.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "compile_receipt")}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("oracle", type=Path); p.add_argument("result", type=Path)
    p.add_argument("--groups", nargs="*")
    a = p.parse_args()
    compare(a.oracle, a.result, a.groups)
