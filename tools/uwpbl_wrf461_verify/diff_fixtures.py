"""Word-by-word difference between two oracle builds' fixtures.

    python diff_fixtures.py DIR_A DIR_B [--json OUT.json]

For every cases-*.bin stem present in both directories (for example the
-O0 reference and the -O2 build of build.sh), compares step 1's outputs,
where both builds received the same input words, and reports mismatched
words, max ULP and max absolute difference per stem.  Later steps are fed
by each build's own tendencies, so their inputs differ and they are not
compared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import FIELDS, _merge, grade_field  # noqa: E402
from gpuwm.verify.uwpbl_oracle import load, step_arrays  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    out = {}
    for man in sorted(Path(args.a).glob("cases-*.manifest")):
        stem = man.stem
        if stem.endswith("-spy") or not (Path(args.b) / man.name).exists():
            continue
        fa, fb = load(Path(args.a) / stem), load(Path(args.b) / stem)
        sa, sb = step_arrays(fa, 1), step_arrays(fb, 1)
        for name in ("t", "u", "v", "qv", "kvm3d_in"):
            assert np.array_equal(sa[name].view(np.uint32),
                                  sb[name].view(np.uint32)), (stem, name)
        acc = {"words": 0, "mismatched": 0, "max_abs": 0.0}
        cols = np.zeros(sa["u"].shape[0], bool)
        for f in FIELDS:
            rec, bad = grade_field(sb[f], sa[f])
            _merge(acc, rec)
            cols |= bad.reshape(len(cols), -1).any(axis=1)
        acc["columns_differing"] = int(cols.sum())
        acc["columns"] = int(len(cols))
        out[stem] = acc
        print(f"{stem}: step 1, {acc['mismatched']} of {acc['words']} words "
              f"differ in {acc['columns_differing']} of {acc['columns']} "
              f"columns, max ULP {acc.get('max_ulp', 0)}, "
              f"max abs {acc['max_abs']:.3e}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
