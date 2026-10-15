"""Grade WOOF's RRTM longwave against the WRF v4.6.1 column oracle.

    python compare.py INPUT.bin WRF_OUT.bin [--labels labels.json]
                      [--device gpu|cpu] [--json OUT.json] [--stages]
    python compare.py --wrf-vs-wrf A.bin B.bin [--json OUT.json]

The first form runs WOOF's scheme boundary
:func:`gpuwm.core.rrtm_lw.rrtmlwrad_columns` on the exact float32 inputs the
Fortran read and compares every output word for word: RTHRATEN, RTHRATENC,
GLW, OLR.  ``--stages`` also compares the intermediates RRTM's own
subroutines leave behind (MM5ATM, SETCOEF, GASABS, RTRN) and the combined
coefficient tables rrtm_lookuptable builds, to locate a first divergence.

The second form compares two WRF outputs (the -O0 reference against WRF's
own -O2 -ftree-vectorize build) the same way.

A match is 0 ULP AND identical bits (signed zeros count).  ULP distance is
measured on the ordered-integer image of float32, so it is defined across
zero; NaN equals NaN.  Nothing here asserts; the gate is
tests/test_rrtm_lw_wrf461_parity.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_io import read_inputs, read_outputs  # noqa: E402

FINAL_FIELDS = ("rthraten", "rthratenc", "glw", "olr")


def _ordered(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def diff(want, got, mask=None) -> dict:
    """Bitwise and ULP comparison of two equal-shape arrays."""
    want = np.asarray(want)
    got = np.asarray(got)
    if want.shape != got.shape:
        return {"shape_mismatch": [list(want.shape), list(got.shape)]}
    if mask is not None:
        want = want[mask]
        got = got[mask]
    if want.dtype.kind == "f":
        want32 = want.astype(np.float32)
        got32 = got.astype(np.float32)
        both_nan = np.isnan(want32) & np.isnan(got32)
        bits_differ = (want32.view(np.int32) != got32.view(np.int32)) & ~both_nan
        ulp = np.abs(_ordered(want32) - _ordered(got32))
        ulp = np.where(both_nan, 0, ulp)
        absd = np.abs(want32.astype(np.float64) - got32.astype(np.float64))
        absd = np.where(both_nan, 0.0, absd)
        worst = int(np.argmax(ulp)) if ulp.size else 0
        return {
            "n": int(want.size), "bits_differ": int(bits_differ.sum()),
            "max_ulp": int(ulp.max()) if ulp.size else 0,
            "max_abs": float(np.nanmax(absd)) if absd.size else 0.0,
            "worst_flat_index": worst,
        }
    differ = want != got
    return {"n": int(want.size), "bits_differ": int(differ.sum()),
            "max_ulp": int(np.abs(want.astype(np.int64)
                                  - got.astype(np.int64)).max())
            if want.size else 0, "max_abs": None}


def _locate(shape, flat, labels):
    index = np.unravel_index(flat, shape) if len(shape) else ()
    col = int(index[0]) if len(index) else None
    return {"index": [int(v) for v in index],
            "column": labels[col] if (labels and col is not None
                                      and col < len(labels)) else col}


def run_woof(cs, device: str, stages: bool):
    from gpuwm.core.rrtm import rrtm_default_trace_gases
    from gpuwm.core.rrtm_lw import rrtm_layer_count, rrtmlwrad_columns

    if device == "gpu":
        import cupy as xp
    else:
        xp = np
    f = {k: xp.asarray(v) for k, v in cs.fields.items()}
    gases = rrtm_default_trace_gases(cs.yr)
    nlayers = rrtm_layer_count(cs.nz, cs.p_top)
    out = rrtmlwrad_columns(
        t3d=f["t3d"], p3d=f["p3d"], pi3d=f["pi3d"], dz8w=f["dz8w"],
        qv=f["qv"], qc=f["qc"], qr=f["qr"], qi=f["qi"], qs=f["qs"],
        qg=f["qg"], cldfra=f["cldfra"], p8w=f["p8w"], t8w=f["t8w"],
        tsk=f["tsk"], emiss=f["emiss"], nlayers=nlayers,
        co2vmr=gases["co2"], n2ovmr=gases["n2o"], ch4vmr=gases["ch4"],
        icloud=cs.icloud, stages=stages)

    def host(value):
        if isinstance(value, dict):
            return {k: host(v) for k, v in value.items()}
        return value.get() if hasattr(value, "get") else np.asarray(value)

    return host(out), nlayers, gases


def compare_set(cs, wrf, woof, nlayers, labels, stages: bool) -> dict:
    report = {"ncol": cs.ncol, "nz": cs.nz, "nlayers": nlayers,
              "wrf_nlayers": int(wrf["_header"][2]), "fields": {}}
    for name in FINAL_FIELDS:
        d = diff(wrf[name], woof[name])
        if "worst_flat_index" in d:
            d["worst"] = _locate(wrf[name].shape, d["worst_flat_index"],
                                 labels)
        report["fields"][name] = d
    if not stages:
        return report
    st = woof["stages"]
    laytrop = wrf["laytrop"]
    lay = np.arange(1, nlayers + 1)[None, :]
    trop = lay <= laytrop[:, None]
    pairs = {
        "o3prof": (wrf["o3prof"], st["o3prof"], None),
        "pavel": (wrf["pavel"], st["pavel"], None),
        "tavel": (wrf["tavel"], st["tavel"], None),
        "pz": (wrf["pz"], st["pz"], None),
        "tz": (wrf["tz"], st["tz"], None),
        "cldfrac": (wrf["cldfrac"], st["cldfrac"], None),
        "taucloud": (wrf["taucloud"], st["taucloud"], None),
        "coldry": (wrf["coldry"], st["coldry"], None),
        "tbound": (wrf["tbound"], st["tbound"], None),
    }
    for m in range(7):
        pairs[f"wkl{m + 1}"] = (wrf["wkl"][:, :, m], st["wkl"][:, :, m], None)
    for name in ("colh2o", "colco2", "colo3", "coln2o", "colch4", "colo2",
                 "co2mult", "fac00", "fac01", "fac10", "fac11", "forfac",
                 "jp", "jt", "jt1", "laytrop", "layswtch", "laylow"):
        pairs[name] = (wrf[name], st[name], None)
    for name in ("selffac", "selffrac", "indself"):
        pairs[name] = (wrf[name], st[name], trop)
    for name in ("taug", "pfrac", "itr"):
        pairs[name] = (wrf[name], st[name], None)
    pairs["totdflux"] = (wrf["totdflux"], st["totdflux"], None)
    pairs["totuflux"] = (wrf["totuflux"], st["totuflux"], None)
    pairs["htr"] = (wrf["htr"], st["htr"], None)
    pairs["htrc"] = (wrf["htrc"], st["htrc"], None)
    pairs["tten"] = (wrf["tten_topdown"][:, ::-1], st["tten"], None)
    pairs["ttenc"] = (wrf["ttenc_topdown"][:, ::-1], st["ttenc"], None)
    report["stages"] = {}
    for name, (w, g, mask) in pairs.items():
        d = diff(w, g, mask)
        if mask is None and "worst_flat_index" in d:
            d["worst"] = _locate(np.asarray(w).shape, d["worst_flat_index"],
                                 labels)
        report["stages"][name] = d
    # Planck table range: WRF reads TOTPLNK(T-159) unguarded; a level
    # outside [160, 340) K would be an out-of-bounds read, not a reference.
    tz = wrf["tz"]
    tav = wrf["tavel"]
    report["planck_in_table"] = bool(
        (tz >= 160).all() and (tz < 340).all() and (tav >= 160).all()
        and (tav < 340).all())
    return report


def compare_tables(wrf_tables: dict) -> dict:
    from gpuwm.core.rrtm_lw import RRTM_BPADE, RRTM_FLUXFAC
    from gpuwm.core.rrtm_tables import load_rrtm_lw_tables

    t = load_rrtm_lw_tables()
    out = {}
    out["FLUXFAC"] = diff(wrf_tables["FLUXFAC"], np.asarray([RRTM_FLUXFAC]))
    out["BPADE"] = diff(wrf_tables["BPADE"], np.asarray([RRTM_BPADE]))
    out["HEATFAC"] = diff(wrf_tables["HEATFAC"],
                          np.asarray(t.statics["HEATFAC"][:1], np.float32))
    for name, mine in (("TAU", t.tau), ("TF", t.tf), ("TRANS", t.trans),
                       ("CORR1", t.corr1), ("CORR2", t.corr2),
                       ("RWGT", t.rwgt)):
        out[name] = diff(wrf_tables[name], mine)
    for band, arr in t.absa.items():
        out[f"ABSA{band}"] = diff(wrf_tables[f"ABSA{band}"], arr.T)
    for band, arr in t.absb.items():
        out[f"ABSB{band}"] = diff(wrf_tables[f"ABSB{band}"], arr.T)
    for band, arr in t.selfrefc.items():
        out[f"SELFREFC{band}"] = diff(wrf_tables[f"SELFREFC{band}"], arr.T)
    for name, arr in t.combined.items():
        if name in wrf_tables:
            out[name] = diff(wrf_tables[name].reshape(-1), arr.reshape(-1))
    return out


def _summarise(report: dict) -> dict:
    worst = {}
    for set_name, rep in report["sets"].items():
        for name, d in rep["fields"].items():
            cur = worst.setdefault(name, {"bits_differ": 0, "max_ulp": 0,
                                          "max_abs": 0.0, "n": 0})
            cur["bits_differ"] += d.get("bits_differ", 0)
            cur["n"] += d.get("n", 0)
            cur["max_ulp"] = max(cur["max_ulp"], d.get("max_ulp", 0))
            cur["max_abs"] = max(cur["max_abs"], d.get("max_abs") or 0.0)
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("wrf_out")
    ap.add_argument("--labels")
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--json")
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--wrf-vs-wrf", action="store_true",
                    help="compare two WRF outputs: INPUT is the reference")
    args = ap.parse_args(argv)

    if args.wrf_vs_wrf:
        _, ref_sets = read_outputs(args.input)
        _, other_sets = read_outputs(args.wrf_out)
        report = {"mode": "wrf-vs-wrf", "sets": {}}
        for index, (a, b) in enumerate(zip(ref_sets, other_sets)):
            fields = {name: diff(a[name], b[name]) for name in a
                      if not name.startswith("_")}
            report["sets"][f"set{index + 1}"] = {"fields": fields}
    else:
        sets = read_inputs(args.input)
        tables, wrf_sets = read_outputs(args.wrf_out)
        labels = (json.loads(Path(args.labels).read_text())
                  if args.labels else {})
        label_lists = list(labels.values()) if labels else []
        from gpuwm import wrf_exact
        report = {"mode": "woof-vs-wrf", "device": args.device,
                  "strict": bool(wrf_exact.ENABLED), "sets": {}}
        if args.stages:
            report["tables"] = compare_tables(tables)
        for index, (cs, wrf) in enumerate(zip(sets, wrf_sets)):
            names = (label_lists[index]["columns"]
                     if index < len(label_lists) else None)
            set_name = (list(labels)[index] if labels else f"set{index + 1}")
            woof, nlayers, gases = run_woof(cs, args.device, args.stages)
            rep = compare_set(cs, wrf, woof, nlayers, names, args.stages)
            rep["gases"] = diff(wrf["gases_f64"].astype(np.float32),
                                np.asarray([gases["co2"], gases["n2o"],
                                            gases["ch4"]], np.float32))
            report["sets"][set_name] = rep
        report["summary"] = _summarise(report)

    text = json.dumps(report, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    # human summary
    if report["mode"] == "woof-vs-wrf":
        print(f"strict={report['strict']} device={report['device']}")
        for name, d in report["summary"].items():
            print(f"  {name:10s} n={d['n']:6d} bits_differ={d['bits_differ']:6d}"
                  f" max_ulp={d['max_ulp']:>10d} max_abs={d['max_abs']:.3e}")
        if args.stages:
            first = {}
            for set_name, rep in report["sets"].items():
                for name, d in rep.get("stages", {}).items():
                    if d.get("bits_differ"):
                        cur = first.setdefault(name, [0, 0])
                        cur[0] += d["bits_differ"]
                        cur[1] = max(cur[1], d["max_ulp"])
            print("  stages with differences:", first or "none")
            bad = {k: (v["bits_differ"], v["max_ulp"])
                   for k, v in report.get("tables", {}).items()
                   if v.get("bits_differ")}
            print("  tables with differences:", bad or "none")
            print("  planck in table:", {k: r.get("planck_in_table")
                                         for k, r in report["sets"].items()})
    else:
        for set_name, rep in report["sets"].items():
            bad = {k: (v["bits_differ"], v["max_ulp"])
                   for k, v in rep["fields"].items() if v.get("bits_differ")}
            print(set_name, bad or "identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
