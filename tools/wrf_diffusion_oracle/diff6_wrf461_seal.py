"""Seal a compact repository fixture from the full WRF 4.6.1 sixth-order oracle.

The full oracle (diff6_wrf461_cases.py) holds 542 cases and 430 million
words, too large for the repository.  ``select`` keeps a few MB: every
boundary mode and staggering with a nonzero incoming tendency on real
convective, snow/ice, steep-ridge and map-extreme states, the hydrometeor
rows, fractional time steps and the subnormal and tiny-tail probes, so the
committed test still grades each mode, staggering, limiter and slope
option, the scalar dt/3 row and the in-place association.  Inputs and WRF
references are copied word for word.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

PREFERRED = ("iowa_storm", "snowband", "steep_ridge", "map_extremes")
EDGE = ("subnormal_checkerboard", "tiny_tails")


def select(cases):
    """One nonzero-incoming case per (boundary mode, staggering) from the
    real convective, snow/ice, steep and map-extreme states, every
    accumulated hydrometeor row, the fractional-dt rows, and the 'u' and 'v'
    rows of the two edge probes."""
    chosen, seen = [], set()
    for group in PREFERRED:
        for c in cases:
            if c["group"] != group or not c["accumulate"]:
                continue
            key = (c["mode_name"], c["stagger"])
            hydro = c["var"] in ("QSNOW", "QICE", "QGRAUP")
            if key not in seen or hydro or (c["num"] and group == "iowa_storm"):
                seen.add(key)
                chosen.append(c)
    for group in ("steep_ridge", "map_extremes"):     # slope clamp, map extremes
        extra = [c for c in cases if c["group"] == group and c["accumulate"] and c not in chosen]
        chosen += extra[:2]
    chosen += [c for c in cases if c["group"] in EDGE and c["var"] in ("U", "V")
               and c["mode_name"] in ("periodic", "specified", "open_x")]
    return sorted(chosen, key=lambda c: c["case"])


def seal(oracle, output):
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((oracle / "oracle.json").read_text())
    arrays, cases = {}, []
    for case in select(meta["cases"]):
        g = case["group"]
        prefix = f"c{case['case']:04d}__"
        with np.load(oracle / g / f"inputs-{case['case']:04d}.npz") as src:
            for key in src.files:
                arrays[prefix + key] = src[key]
        arrays[prefix + "reference"] = np.load(oracle / g / f"ref-{case['case']:04d}.npy")
        cases.append(case)
    np.savez_compressed(output / "diff6-wrf461-columns.npz", **arrays)
    keep = {k: meta[k] for k in ("wrf_version", "wrf_commit", "source_sha256", "constants_sha256",
                                 "routine_slice_sha256", "routine_slice_equals_wrf471", "routine_source_lines",
                                 "compiler", "builds", "real_kind_bytes", "slice_is_byte_unmodified",
                                 "driver_sha256", "sources", "strict_equals_stock", "tools_sha256")}
    keep.update(schema="wrf461-diff6-columns-v1", full_oracle_sha256=hashlib.sha256((oracle / "oracle.json").read_bytes()).hexdigest(),
                full_oracle_cases=len(meta["cases"]), full_oracle_words=sum(c["words"] for c in meta["cases"]),
                cases=cases)
    (output / "diff6-wrf461-columns.json").write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8", newline="\n")
    sums = "".join(hashlib.sha256((output / n).read_bytes()).hexdigest() + "  " + n + "\n"
                   for n in ("diff6-wrf461-columns.npz", "diff6-wrf461-columns.json"))
    (output / "diff6-wrf461-sha256sums.txt").write_text(sums, encoding="utf-8", newline="\n")
    print(f"sealed cases={len(cases)} words={sum(c['words'] for c in cases)} "
          f"bytes={(output / 'diff6-wrf461-columns.npz').stat().st_size}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("oracle", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args()
    seal(a.oracle, a.output)
