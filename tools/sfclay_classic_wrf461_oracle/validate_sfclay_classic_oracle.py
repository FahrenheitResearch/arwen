"""Grade WOOF's classic MM5 surface layer against the WRF 4.6.1 oracle.

    python validate_sfclay_classic_oracle.py COLUMNS WRF_OUTPUT [--json OUT]

Runs the real kernel (``gpuwm.core.sfclay.sfclay``, option 91) on the same
input words the WRF driver read, step by step, and compares every one of the
34 output fields bit for bit.  Two modes:

* free-running: step k+1 starts from WOOF's own step-k inout fields, the
  way a forecast runs;
* replay: step k+1 starts from WRF's step-k inout fields, so a difference
  is attributed to the step that made it.

It also compares SFCLAYINIT's two 1001-entry tables word for word.  Prints
per-field maximum ULP distance and the number of differing words; asserts
nothing (tests/test_sfclay_classic_wrf461_parity.py is the gate).  Whether
the strict build is on is read from GPUWM_WRF_EXACT, as everywhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

INPUT_FIELDS = ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "znt",
                "ust", "ustm", "pblh", "mavail", "xland", "lakemask",
                "mol", "hfx", "qfx", "qsfc", "zol")
INOUT_FIELDS = ("znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol")
OUTPUT_FIELDS = (
    "znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol", "regime",
    "psim", "psih", "fm", "fh", "lh", "u10", "v10", "th2", "t2",
    "q2", "chs", "chs2", "cqs2", "flhc", "flqc", "qgh", "rmol",
    "wspd", "br", "gz1oz0", "cpm", "ck", "cka", "cd", "cda")


def _f32(words):
    return np.asarray(words, dtype=np.uint32).view(np.float32)


def load_columns(path):
    """``[(name, isfflx, isftcflx, iz0tlnd, dx, {field: float32[ncol]})]``."""
    lines = Path(path).read_text(encoding="ascii").split("\n")
    ngroup, nstep = map(int, lines[0].split())
    pos, groups = 1, []
    for _ in range(ngroup):
        name, ncol, isfflx, isftcflx, iz0tlnd, dxw = lines[pos].split()
        ncol = int(ncol)
        rows = np.array([[int(w) for w in lines[pos + 1 + k].split()]
                         for k in range(ncol)], dtype=np.uint64).astype(np.uint32)
        pos += 1 + ncol
        fields = {f: _f32(rows[:, j]) for j, f in enumerate(INPUT_FIELDS)}
        groups.append((name, int(isfflx), int(isftcflx), int(iz0tlnd),
                       float(_f32([int(dxw)])[0]), fields))
    return groups, nstep


def load_wrf(path):
    """``(table {'psim','psih'}: uint32[1001], rows {(g,s): uint32[ncol,34]})``."""
    table = {"psim": np.zeros(1001, np.uint32), "psih": np.zeros(1001, np.uint32)}
    rows: dict[tuple[int, int], list] = {}
    for line in Path(path).read_text(encoding="ascii").split("\n"):
        if not line:
            continue
        parts = line.split()
        if parts[0] == "TABLE":
            n = int(parts[1])
            table["psim"][n] = int(parts[2]) & 0xFFFFFFFF
            table["psih"][n] = int(parts[3]) & 0xFFFFFFFF
        else:
            g, s, _ = (int(x) for x in parts[1:4])
            rows.setdefault((g, s), []).append(
                [int(x) & 0xFFFFFFFF for x in parts[4:]])
    return table, {k: np.array(v, dtype=np.uint32) for k, v in rows.items()}


def ulp_distance(a_words, b_words):
    """|a - b| in float32 ULPs on the ordered integer line; NaN pairs equal."""
    a = np.asarray(a_words, np.uint32).astype(np.int64)
    b = np.asarray(b_words, np.uint32).astype(np.int64)
    oa = np.where(a >= 2**31, 2**31 - a, a)
    ob = np.where(b >= 2**31, 2**31 - b, b)
    d = np.abs(oa - ob)
    both_nan = np.isnan(_f32(a.astype(np.uint32))) & np.isnan(_f32(b.astype(np.uint32)))
    return np.where(both_nan, 0, d)


def woof_table():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    psim = cp.zeros(1001, cp.float32)
    psih = cp.zeros(1001, cp.float32)
    get_kernel("sfclay", "sfclay_classic_table")((8,), (128,), (psim, psih, np.int32(1001)))
    return {"psim": psim.get().view(np.uint32), "psih": psih.get().view(np.uint32)}


def woof_step(fields, state, *, isfflx, isftcflx, iz0tlnd, dx):
    import cupy as cp
    from gpuwm.core.sfclay import sfclay
    dev = lambda a: cp.asarray(np.asarray(a, np.float32).reshape(1, -1))  # noqa: E731
    got = sfclay(dev(fields["u"]), dev(fields["v"]), dev(fields["t"]),
                 dev(fields["qv"]), dev(fields["p"]), dev(fields["dz8w"]),
                 dev(fields["psfc"]), dev(fields["tsk"]), dev(state["znt"]),
                 dev(fields["pblh"]), dev(fields["mavail"]), dev(fields["xland"]),
                 option=91, qsfc=dev(state["qsfc"]), zol=dev(state["zol"]),
                 ust=dev(state["ust"]), ustm=dev(state["ustm"]),
                 mol=dev(state["mol"]), hfx=dev(state["hfx"]),
                 qfx=dev(state["qfx"]), lakemask=dev(fields["lakemask"]),
                 dx=dx, isfflx=bool(isfflx), isftcflx=isftcflx, iz0tlnd=iz0tlnd)
    return np.stack([getattr(got, f).get().reshape(-1).view(np.uint32)
                     for f in OUTPUT_FIELDS], axis=1)


def grade(columns_path, wrf_path):
    """Grade every raw output word, including corrected CK."""
    groups, nstep = load_columns(columns_path)
    wrf_table, wrf_rows = load_wrf(wrf_path)
    out = {"exact": os.environ.get("GPUWM_WRF_EXACT", "0") == "1",
           "columns": sum(len(g[5]["u"]) for g in groups), "steps": nstep,
           "groups": [g[0] for g in groups]}
    try:
        wt = woof_table()
    except Exception as exc:          # a tree older than the table entry point
        out["table"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    else:
        out["table"] = {k: {"max_ulp": int(ulp_distance(wt[k], wrf_table[k]).max()),
                            "differing": int((wt[k] != wrf_table[k]).sum())}
                        for k in ("psim", "psih")}
    for mode in ("free", "replay"):
        mx = {f: 0 for f in OUTPUT_FIELDS}
        nd = {f: 0 for f in OUTPUT_FIELDS}
        worst = {}
        for gi, (name, isfflx, isftcflx, iz0tlnd, dx, fields) in enumerate(groups, 1):
            state = {f: fields[f].copy() for f in INOUT_FIELDS}
            for step in range(1, nstep + 1):
                got = woof_step(fields, state, isfflx=isfflx, isftcflx=isftcflx,
                                iz0tlnd=iz0tlnd, dx=dx)
                ref = wrf_rows[(gi, step)]
                d = ulp_distance(got, ref)
                for j, f in enumerate(OUTPUT_FIELDS):
                    for c in range(d.shape[0]):
                        if got[c, j] == ref[c, j]:
                            continue
                        nd[f] += 1
                        if d[c, j] > mx[f]:
                            mx[f] = int(d[c, j])
                            worst[f] = {"group": name, "column": c, "step": step,
                                        "woof": float(_f32([got[c, j]])[0]),
                                        "wrf": float(_f32([ref[c, j]])[0])}
                src = got if mode == "free" else ref
                state = {f: _f32(src[:, OUTPUT_FIELDS.index(f)]).copy()
                         for f in INOUT_FIELDS}
        out[mode] = {"max_ulp": mx, "differing_words": nd, "worst": worst,
                     "overall_max_ulp": max(mx.values()),
                     "words_compared": out["columns"] * nstep * len(OUTPUT_FIELDS)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("columns")
    ap.add_argument("wrf")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    res = grade(args.columns, args.wrf)
    print(f"strict build (GPUWM_WRF_EXACT=1): {res['exact']}")
    print(f"{res['columns']} columns x {res['steps']} steps, groups: {', '.join(res['groups'])}")
    for k, v in res["table"].items():
        if k == "unavailable":
            print(f"table: {v}")
        else:
            print(f"table {k}tb: max {v['max_ulp']} ULP, {v['differing']} of 1001 entries differ")
    for mode in ("free", "replay"):
        r = res[mode]
        print(f"\n[{mode}] overall max {r['overall_max_ulp']} ULP over {r['words_compared']} words")
        for f in OUTPUT_FIELDS:
            if r["max_ulp"][f]:
                print(f"  {f:7s} max {r['max_ulp'][f]:>10d} ULP  {r['differing_words'][f]:>4d} words  worst {r['worst'][f]}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
