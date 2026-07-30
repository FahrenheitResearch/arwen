"""Gate cmbgb band reductions against the post-init Fortran module dump.

Usage:
  python lw_gate_cmbgb.py [--frag fragment.py] [--bands 2-16] [--fixdir D]

Each _cmbgbN(mod, rwgt, out) must write every reduced array of its band
into `out` under 'kgNN/<name>'.  Required names are derived from the raw
variable list (kao->ka, kbo_mco2->kb_mco2, selfrefo->selfref, ...), and
each is compared bitwise against the oracle's post-init dump.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from lw_gate import (DEFAULT_FIXDIR, RAW_VARS,  # noqa: E402
                     load_coeffs_fixture, raw_from_coeffs_fixture,
                     ulp_report)


def reduced_name(raw_name):
    if raw_name.startswith("kao"):
        return "ka" + raw_name[3:]
    if raw_name.startswith("kbo"):
        return "kb" + raw_name[3:]
    assert raw_name.endswith("o"), raw_name
    return raw_name[:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frag", default=None)
    ap.add_argument("--bands", default="1-16")
    ap.add_argument("--fixdir", default=DEFAULT_FIXDIR)
    args = ap.parse_args()

    lohi = args.bands.split("-")
    bands = range(int(lohi[0]), int(lohi[-1]) + 1)

    from gpuwm.core import rrtmg_lw as port

    impls = dict(port._CMBGB_IMPLS)
    if args.frag:
        spec = importlib.util.spec_from_file_location("lw_frag", args.frag)
        frag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frag)
        for band in bands:
            fn = getattr(frag, f"_cmbgb{band}", None)
            if fn is not None:
                impls[band] = fn

    cfx = load_coeffs_fixture(args.fixdir)
    raw = raw_from_coeffs_fixture(cfx)
    rwgt = port._compute_rwgt()
    m = ulp_report(rwgt, cfx["rrlw_wvn/rwgt"], "rwgt")
    if m:
        print("FAIL:", m)
        sys.exit(1)

    bad = []
    for band in bands:
        if band not in impls:
            bad.append(f"band {band}: no _cmbgb{band} implementation")
            continue
        mod = raw[f"rrlw_kg{band:02d}"]
        out = {}
        impls[band](mod, rwgt, out)
        for raw_name in RAW_VARS[band]:
            red = reduced_name(raw_name)
            key = f"kg{band:02d}/{red}"
            if key not in out:
                bad.append(f"band {band}: missing {key}")
                continue
            want = cfx[f"rrlw_kg{band:02d}/{red}"]
            msg = ulp_report(out[key], want, key)
            if msg:
                bad.append(f"band {band}: {msg}")
    if bad:
        print(f"cmbgb: FAIL ({len(bad)})")
        for line in bad[:25]:
            print(" ", line)
        sys.exit(1)
    print(f"cmbgb: OK, bands {lohi[0]}..{lohi[-1]} bitwise vs post-init dump")


if __name__ == "__main__":
    main()
