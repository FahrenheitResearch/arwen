#!/usr/bin/env python3
"""Attribute the production build's differences from WRF to the two row-37
corrections, column by column.

Three builds of WOOF's Morrison run over the same inputs:

  off   MORR_ROW37_OFF: WRF's arithmetic alone (graded bitwise elsewhere)
  prod  the production build: both corrections
  nocap production with only the deposition-nucleation vapour cap removed
        (the wide freezing-rate blocks stay)

A column that differs from WRF in ``prod`` but not in ``nocap`` moved
because of the nucleation cap; a column that still differs in ``nocap``
moved because of the wide freezing rates.  For every column the script also
records whether WRF's own output there is non-finite (the float exponential
overflow WRF suffers in that branch) and the column's coldest temperature.

usage:
  row37_attribution.py regimes REGIME_DIR [--json F]
  row37_attribution.py stress IN.bin WRF.bin [IN.bin WRF.bin ...] [--json F]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (REPO_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import regime_columns as rc  # noqa: E402

CAP_OPEN = "                    if (mnuccd > avail) {"


def _install(variant):
    """Make kernels.load_module compile morrison as ``variant``."""
    import cupy as cp
    from gpuwm.core import kernels
    if not hasattr(_install, "original"):
        _install.original = kernels.load_module
    original = _install.original

    def load(name):
        if name != "morrison" or variant == "prod":
            return original(name)
        source = kernels.module_source("morrison")
        if variant == "off":
            source = "#define MORR_ROW37_OFF 1" + chr(10) + source
        elif variant == "nocap":
            if source.count(CAP_OPEN) != 1:
                raise RuntimeError("nucleation cap block not found once")
            source = source.replace(
                CAP_OPEN, "                    if (false && mnuccd > avail) {")
        module = cp.RawModule(code=source,
                              options=kernels.module_options("morrison"))
        module.compile()
        return module

    kernels.load_module = load
    kernels.get_kernel.cache_clear()


def _column_bad(got, want, ncol):
    bad = np.zeros(ncol, bool)
    fields = {}
    for name in rc.LEVEL_OUT + rc.SURF:
        g = np.ascontiguousarray(got[name], np.float32)
        w = np.ascontiguousarray(want[name], np.float32)
        b = (g.view(np.uint32) != w.view(np.uint32)) & ~(
            np.isnan(g) & np.isnan(w))
        colb = b.any(axis=0) if b.ndim == 2 else b
        fields[name] = int(b.sum())
        bad |= colb
    return bad, fields


def attribute(cases, labels=None):
    """cases: list of (name, ncol, nz, mode, dt, fields, surf, steps, want)."""
    out = {}
    for variant in ("off", "prod", "nocap"):
        _install(variant)
        for name, ncol, nz, mode, dt, fields, surf, steps, want in cases:
            got, _ = rc.run_gpu(ncol, nz, mode, dt, fields, surf, steps)
            bad, per_field = _column_bad(got, want, ncol)
            out.setdefault(name, {})[variant] = (bad, per_field)
    report = {}
    for name, ncol, nz, mode, dt, fields, surf, steps, want in cases:
        off_bad, off_f = out[name]["off"]
        prod_bad, prod_f = out[name]["prod"]
        nocap_bad, nocap_f = out[name]["nocap"]
        nonfinite = np.zeros(ncol, bool)
        for key in rc.LEVEL_OUT:
            nonfinite |= ~np.isfinite(want[key]).all(axis=0)
        tmin = (fields["th"] * fields["pii"]).min(axis=0)
        cap_cols = prod_bad & ~nocap_bad
        freeze_cols = nocap_bad
        entry = {
            "columns": ncol, "mode": mode, "dt": float(dt), "steps": steps,
            "off_differing_columns": int(off_bad.sum()),
            "prod_differing_columns": int(prod_bad.sum()),
            "prod_field_mismatches": {k: v for k, v in prod_f.items() if v},
            "cap_columns": int(cap_cols.sum()),
            "freeze_columns": int(freeze_cols.sum()),
            "freeze_columns_with_nonfinite_wrf": int(
                (freeze_cols & nonfinite).sum()),
            "wrf_nonfinite_columns": int(nonfinite.sum()),
            "cap_column_tmin_range": (
                [float(tmin[cap_cols].min()), float(tmin[cap_cols].max())]
                if cap_cols.any() else None),
            "freeze_column_tmin_range": (
                [float(tmin[freeze_cols].min()),
                 float(tmin[freeze_cols].max())]
                if freeze_cols.any() else None),
        }
        if labels is not None:
            lab = np.asarray(labels)
            entry["cap_regimes"] = sorted(set(lab[cap_cols].tolist()))
            entry["freeze_regimes"] = sorted(set(lab[freeze_cols].tolist()))
        report[name] = entry
        print(name, json.dumps(entry), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    rg = sub.add_parser("regimes")
    rg.add_argument("outdir", type=Path)
    rg.add_argument("--json", type=Path)
    st = sub.add_parser("stress")
    st.add_argument("files", type=Path, nargs="+")
    st.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    cases = []
    labels = None
    if args.cmd == "regimes":
        meta = json.loads((args.outdir / "sets.json").read_text())
        labels = meta["labels"]
        for name, info in meta["sets"].items():
            ncol, nz, mode, dt, fields, surf = rc.read_input(
                args.outdir / f"{name}.in.bin")
            want = rc.read_output(args.outdir / f"{name}.wrf.bin", ncol, nz)
            cases.append((name, ncol, nz, mode, dt, fields, surf,
                          info["steps"], want))
    else:
        files = list(args.files)
        if len(files) % 2:
            parser.error("stress takes IN.bin WRF.bin pairs")
        for inp, wrf in zip(files[0::2], files[1::2]):
            ncol, nz, mode, dt, fields, surf = rc.read_input(inp)
            want = rc.read_output(wrf, ncol, nz)
            cases.append((inp.stem, ncol, nz, mode, dt, fields, surf, 1,
                          want))
    report = attribute(cases, labels)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
