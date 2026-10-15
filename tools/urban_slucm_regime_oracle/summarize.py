"""Summarise compare.py JSON results as plain-text tables.

    python tools/urban_slucm_regime_oracle/summarize.py RESULT.json [...]

Per result: words compared, variants at 0 ULP, and for every variant that is
not, the rows that differ, the worst ULP and absolute difference, and the
fields and regimes involved (replay and carried).
"""
from __future__ import annotations

import json
import sys


def main(paths):
    for path in paths:
        d = json.load(open(path))
        vs = d["variants"]
        words = sum(v["replay"]["words"] + v["carried"]["words"] for v in vs.values())
        rows = sum(v["replay"]["rows"] for v in vs.values())
        exact = [n for n, v in vs.items()
                 if v["replay"]["rows_differ"] == 0 and v["carried"]["rows_differ"] == 0]
        print(f"## {path}  GPUWM_WRF_EXACT={d['gpuwm_wrf_exact']}  {' / '.join(d['compiler'])}")
        print(f"   {len(vs)} variants, {rows} column-steps, {words} float32 words "
              f"(replay + carried); {len(exact)} variants at 0 ULP")
        for name, v in sorted(vs.items()):
            if name in exact:
                continue
            for mode in ("replay", "carried"):
                m = v[mode]
                fields = sorted(m["fields_differ"].items(), key=lambda t: -t[1]["rows_differ"])
                regimes = [k for k, x in m["by_regime"].items() if x["rows_differ"]]
                print(f"   {name:22s} {mode:7s} rows differ {m['rows_differ']:4d}/{m['rows']:4d} "
                      f"max ULP {m['max_ulp']} max abs {m['max_abs']:.3g}")
                print(f"      fields: " + ", ".join(
                    f"{k}({x['rows_differ']},{x['max_ulp']}ulp)" for k, x in fields[:8])
                    + (" ..." if len(fields) > 8 else ""))
                print(f"      regimes: " + ", ".join(regimes))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
