"""Grade WOOF's Eta similarity surface layer against the WRF v4.6.1 oracle.

    python tools/myjsfc_wrf461_oracle/compare_myjsfc.py [--fixture F.npz]
        [--json OUT.json] [--tables-only | --cpu-authority]

Arithmetic is the process's: run once with GPUWM_WRF_EXACT=1 (strict
build) and once without (default).  Exit status 0 only if every table word
and every field of every call is bit-identical.  ``--cpu-authority``
grades the float32 CPU twin (gpuwm.verify.myj_ref.np_myjsfc_column) in
place of the CUDA kernel; it needs no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from gpuwm.verify import myjsfc_oracle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(myjsfc_oracle.FIXTURE))
    ap.add_argument("--json")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--cpu-authority", action="store_true")
    args = ap.parse_args()
    fixture = myjsfc_oracle.load(args.fixture)
    tables = myjsfc_oracle.compare_tables(fixture)
    if args.tables_only:
        report = None
    elif args.cpu_authority:
        report = myjsfc_oracle.measure_cpu_authority(fixture)
    else:
        report = myjsfc_oracle.measure(fixture)
    mode = ("CPU authority (myj_ref.np_myjsfc_column)" if args.cpu_authority
            else "strict (GPUWM_WRF_EXACT=1)"
            if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default")
    sets = myjsfc_oracle.set_names(fixture)
    calls = sum(int(fixture[f"{s}/meta"][0]) * int(fixture[f"{s}/meta"][2])
                for s in sets)
    header = (f"arithmetic: {mode}; sets {sets}; {calls} column-calls; "
              f"fixture {args.fixture}")
    sys.stdout.write(myjsfc_oracle.format_report(tables, report, header))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump({"header": header, "tables": tables, "report": report},
                      out, indent=1)
    bad = any(s["mismatch"] for s in tables.values())
    if report is not None:
        bad = bad or any(s["mismatch"] for mode_ in ("replay", "free_run")
                         for s in report[mode_].values())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
