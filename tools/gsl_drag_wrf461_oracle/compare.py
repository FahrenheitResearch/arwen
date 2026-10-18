"""Run the port on the GSL drag oracle and write a JSON receipt.

    python compare.py [FIXTURE] OUT.json [--no-spp-pattern]

FIXTURE is the packed .npz (default: tests/data/oracles/gsl_drag) or a raw
build.sh fixture directory (fixtures, fixtures-wrfflags, fixtures-cr-atan2).
The receipt records whether GPUWM_WRF_EXACT was set, the device and NVRTC,
and per case and field: lanes, differing lanes, max ULP, max abs difference
and the differing columns.  --no-spp-pattern launches the spp_pbl = 1 cases
without the pattern (the port's behaviour before it took one).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main(argv) -> int:
    pass_spp = "--no-spp-pattern" not in argv
    args = [a for a in argv if not a.startswith("--")]
    from gpuwm.verify import gsl_drag_oracle as oracle
    fixture = Path(args[0]) if len(args) == 2 else oracle.FIXTURE
    out = Path(args[-1])
    import cupy as cp
    cases = oracle.load(fixture)
    result = oracle.measure(cases, pass_spp=pass_spp)
    dev = cp.cuda.Device()
    receipt = {
        "fixture": str(fixture),
        "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", "unset"),
        "device": cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode(),
        "compute_capability": dev.compute_capability,
        "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
        "pass_spp_pattern": pass_spp,
        "summary": oracle.summary(result),
        "cases": result,
    }
    try:
        from gpuwm.wrf_exact import compile_receipt
        receipt["compile_receipt"] = compile_receipt()
    except Exception as exc:   # the receipt is evidence, not a gate
        receipt["compile_receipt"] = repr(exc)
    out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt["summary"]))
    for case, fields in result.items():
        bad = {f: (s["differ"], s["max_ulp"], s["columns"][:12])
               for f, s in fields.items() if s["differ"]}
        if bad:
            print(case, bad)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
