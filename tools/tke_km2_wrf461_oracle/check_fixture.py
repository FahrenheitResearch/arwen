"""Run WOOF's km_opt=2 launcher on the committed fixture and compare words.

Prints one JSON object: per field, the number of words compared and the
number whose float32 bit pattern differs from the WRF -O0 word.  The
arithmetic build is the process environment's (GPUWM_WRF_EXACT=1 and
GPUWM_WRF_EXACT_DIFFUSION=1 for the strict build); it must be set before
this process imports gpuwm, which is why tests run it as a subprocess.

Usage: python check_fixture.py FIXTURE_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare import word_stats  # noqa: E402
from woof_side import woof_chain  # noqa: E402


def check(fixture: Path) -> dict:
    manifest = json.loads((fixture / "manifest.json").read_text())
    fields = manifest["per_arm_fields"] + manifest["shared_fields"]
    total = {f: {"words": 0, "different": 0, "max_ulp": 0} for f in fields}
    for case in manifest["cases"]:
        name = case["name"]
        with np.load(fixture / f"case-{name}.npz") as data:
            meta = json.loads(str(data["meta_json"]))
            arrays = {k: data[k] for k in data.files if k != "meta_json"}
        with np.load(fixture / f"wrf461-O0-{name}.npz") as wrf:
            ref = {k: wrf[k] for k in wrf.files}
        for arm in manifest["arms"]:
            got = woof_chain(arrays, meta, tuple(arm))
            for f in fields:
                key = f"{arm[0]}/{f}" if f in manifest["per_arm_fields"] else f
                s = word_stats(got[f], ref[key])
                t = total[f]
                t["words"] += s["words"]
                t["different"] += s["different"]
                t["max_ulp"] = max(t["max_ulp"], s["max_ulp"])
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", type=Path)
    args = ap.parse_args()
    print(json.dumps(check(args.fixture)))


if __name__ == "__main__":
    main()
