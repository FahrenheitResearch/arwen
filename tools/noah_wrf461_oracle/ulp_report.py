#!/usr/bin/env python3
"""Every nonzero ULP distance from the Noah CUDA column to WRF's ``lsm``.

    python tools/noah_wrf461_oracle/ulp_report.py OUT.json [FIXTURE_DIR]

Runs ``launch_noah`` on each of the four switch fixtures (the packaged ones,
or the CSVs in FIXTURE_DIR), then ``gpuwm.core.noah_sfcdiags`` on its outputs,
and writes, per fixture, every (case, field, layer) whose float32 word differs
from the one WRF's ``lsm`` (or SFCDIAGS after it) wrote, with both words.
Run it once with ``GPUWM_WRF_EXACT=1`` (the strict build: no FMA contraction,
no flush to zero, IEEE division and square root) and once without it (default
arithmetic); the arithmetic mode is recorded in the output.

It asserts nothing.  ``tests/test_noah_wrf461_parity.py`` is the gate; this
is the instrument used to find out what a nonzero row in that gate is made of.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.core.fp32_ulp import fp32_ulp_distance  # noqa: E402
from gpuwm.verify.noah_oracle import (  # noqa: E402
    NOAH_ORACLE_FILES,
    OUTPUT_COLUMNS,
    SFCDIAGS_COLUMNS,
    SOIL_OUTPUT_COLUMNS,
    load_noah_oracle,
    noah_port_outputs,
    noah_sfcdiags_outputs,
)


def _same_word(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Bitwise equality, except that any NaN equals any NaN."""
    same = got.view(np.uint32) == want.view(np.uint32)
    return same | (np.isnan(got) & np.isnan(want))


def report(directory: Path | None) -> dict:
    import cupy

    from gpuwm.certify.compile_platform import nvrtc_build

    out = {
        "strict": os.environ.get("GPUWM_WRF_EXACT", "0") == "1",
        "device": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "capability": str(cupy.cuda.Device().compute_capability),
        "nvrtc": nvrtc_build(),
        "fixtures": {},
    }
    for name in NOAH_ORACLE_FILES:
        fixture = load_noah_oracle(name, directory)
        port = noah_port_outputs(fixture)
        port.update(noah_sfcdiags_outputs(fixture, port))
        diffs = []
        field_max: dict[str, int] = {}
        single = (*OUTPUT_COLUMNS, *SFCDIAGS_COLUMNS)
        for field in (*single, *SOIL_OUTPUT_COLUMNS):
            if field in OUTPUT_COLUMNS:
                got = np.ascontiguousarray(port[field], np.float32)[None]
                want = np.ascontiguousarray(fixture.reference[field])[None]
            elif field in SFCDIAGS_COLUMNS:
                got = np.ascontiguousarray(port[field], np.float32)[None]
                want = np.ascontiguousarray(
                    fixture.sfcdiags_reference[field])[None]
            else:
                got = np.ascontiguousarray(port[field], np.float32)
                want = np.ascontiguousarray(fixture.soil_reference[field])
            same = _same_word(got, want)
            ulp = fp32_ulp_distance(got, want)
            ulp = np.where(same, 0, ulp)
            field_max[field] = int(ulp.max())
            for layer, _, col in zip(*np.nonzero(~same)):
                diffs.append({
                    "case": int(fixture.cases[col]),
                    "glacial": bool(fixture.glacial[col]),
                    "field": field,
                    "layer": (None if field in single
                              else int(layer) + 1),
                    "ulp": int(ulp[layer, 0, col]),
                    "got": float(got[layer, 0, col]),
                    "want": float(want[layer, 0, col]),
                })
        out["fixtures"][name] = {
            "switches": fixture.switches,
            "ncase": fixture.ncase,
            "field_max_ulp": field_max,
            "differing_words": len(diffs),
            "diffs": diffs,
        }
    out["max_ulp"] = max(max(f["field_max_ulp"].values())
                         for f in out["fixtures"].values())
    out["differing_words"] = sum(f["differing_words"]
                                 for f in out["fixtures"].values())
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    directory = Path(argv[2]) if len(argv) > 2 else None
    result = report(directory)
    Path(argv[1]).write_text(json.dumps(result, indent=1) + "\n",
                             encoding="utf-8")
    print(f"strict={result['strict']} {result['device']} "
          f"sm_{result['capability']} NVRTC {result['nvrtc']}: "
          f"max ULP {result['max_ulp']}, "
          f"{result['differing_words']} differing words")
    for name, fx in result["fixtures"].items():
        worst = {k: v for k, v in fx["field_max_ulp"].items() if v}
        print(f"  {name}: {fx['ncase']} columns, {fx['differing_words']}"
              f" differing words; nonzero fields {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
