"""Re-measure tests/data/oracles/chem/gocart/sulfur/ULP_BASELINE.json on a GPU.

    python tools/chem_wrf461_verify/regen_sulfur_baseline.py [--write]

Runs every sulfur oracle case through the CUDA kernel exactly as
tests/test_chem_sulfur_wrf471_parity.py does (native and oracle-solar modes,
plus the absent-role "lite" form for zero-role cases) and prints the largest
ULP per mode.  With --write it replaces ULP_BASELINE.json, in the file's own
layout (indent 2, trailing newline).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from chem_gocart_checks import FIXTURES, sulfur_case  # noqa: E402
from gpuwm.verify.chem_oracle import load  # noqa: E402


def main(write: bool) -> int:
    baseline = {}
    worst: dict[str, int] = {}
    for name, case in load(FIXTURES / "sulfur").items():
        entry = {}
        for mode, oracle_solar in (("native", False), ("oracle_solar", True)):
            entry[mode] = sulfur_case(case, oracle_solar=oracle_solar)
            if case["zero_roles"]:
                entry[mode + "_lite"] = sulfur_case(case, oracle_solar=oracle_solar, lite=True)
        baseline[name] = dict(sorted(entry.items()))
        for mode, tables in entry.items():
            worst[mode] = max(worst.get(mode, 0), *(t["max_ulp"] for t in tables.values()))
    print(json.dumps(worst, sort_keys=True))
    if write:
        (FIXTURES / "sulfur/ULP_BASELINE.json").write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--write" in sys.argv[1:]))
