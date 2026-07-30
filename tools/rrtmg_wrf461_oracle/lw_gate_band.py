"""Gate one taugb band implementation against all case fixtures, bitwise.

Usage:
  python lw_gate_band.py --band N [--frag path/to/fragment.py] [--fixdir D]

Without --frag, gates the band implementation registered in
gpuwm.core.rrtmg_lw.TAUGB_IMPLS.  With --frag, loads `_taugbN` from the
fragment module instead (the fragment may import helpers from
gpuwm.core.rrtmg_lw).  Exit code 0 only if every case matches bitwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root

from lw_fixtures import read_fixture  # noqa: E402
from lw_gate import (DEFAULT_FIXDIR, band_slice, case_paths,  # noqa: E402
                     load_coeffs_fixture, port_coeffs_from_fixture,
                     state_from_fixture, ulp_report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--frag", default=None)
    ap.add_argument("--cuda", action="store_true",
                    help="gate the rlw_taugbN CUDA kernel (dual-run)")
    ap.add_argument("--fixdir", default=DEFAULT_FIXDIR)
    args = ap.parse_args()

    from gpuwm.core.rrtmg_lw import NGPTLW, TAUGB_IMPLS

    if args.cuda:
        import cupy as cp
        from gpuwm.core.rrtmg_lw import gpu_taugb, gpu_preflight
        gpu_preflight()
        impl = None
    elif args.frag:
        spec = importlib.util.spec_from_file_location("lw_frag", args.frag)
        frag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frag)
        impl = getattr(frag, f"_taugb{args.band}")
    else:
        impl = TAUGB_IMPLS[args.band]

    C = port_coeffs_from_fixture(load_coeffs_fixture(args.fixdir))
    s = band_slice(args.band)
    nruns = 2 if args.cuda else 1   # dual-run rule for GPU-measured claims
    for run in range(1, nruns + 1):
        bad = []
        npass = 0
        for path in case_paths(args.fixdir):
            fx = read_fixture(path)
            nl = int(fx["meta/nlayers"])
            st = state_from_fixture(fx)
            if args.cuda:
                taug_d, fracs_d = gpu_taugb(args.band, st, C)
                taug = cp.asnumpy(taug_d)[0]
                fracs = cp.asnumpy(fracs_d)[0]
            else:
                taug = np.zeros((nl, NGPTLW), dtype=np.float32)
                fracs = np.zeros((nl, NGPTLW), dtype=np.float32)
                with np.errstate(all="ignore"):
                    impl(st, C, taug, fracs)
            m1 = ulp_report(taug[:, s], fx["taumol/taug"][:, s], "taug")
            m2 = ulp_report(fracs[:, s], fx["taumol/fracs"][:, s], "fracs")
            if m1 or m2:
                bad.append(f"{os.path.basename(path)}: {m1 or ''} {m2 or ''}")
            else:
                npass += 1
        total = npass + len(bad)
        tag = f" [cuda run {run}/{nruns}]" if args.cuda else ""
        if bad:
            print(f"band {args.band}{tag}: FAIL {len(bad)}/{total}")
            for line in bad[:20]:
                print(" ", line)
            sys.exit(1)
        print(f"band {args.band}{tag}: OK, {total}/{total} cases at max_ulp 0")


if __name__ == "__main__":
    main()
