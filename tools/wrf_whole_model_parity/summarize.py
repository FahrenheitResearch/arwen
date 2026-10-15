"""Score WOOF's whole-model runs against stock WRF and WRF's own build spread.

    python summarize.py OUT_DIR SUMMARY.json

OUT_DIR holds compare.py reports named ``<arm>.<cand>-vs-wrfscalar.json``,
where ``cand`` is ``wrfvec`` (WRF 4.6.1 built with vectorisation and glibc
libmvec, the yardstick) or ``woof-<tree>-<mode>``.  The referee is WRF 4.6.1
built scalar (-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize).

THE RULE, fixed before any WOOF run was scored (2026-10-12):

* ``strict``: the first output frame at which any field common to both
  sides differs in any word, and how many fields differ there.  Strict mode
  is graded word for word; a non-zero count is reported as such.
* the spread ratio, per key field and frame: interior RMS of
  (WOOF - WRF scalar) over interior RMS of (WRF vectorised - WRF scalar), the
  5-cell specified-boundary rim excluded.  A WRF build differs from another
  build of itself by its arithmetic alone, so a port bug shows as a ratio far
  above 1 on the fields the scheme drives.
* verdict per arm and mode at the last frame: ``inside-spread`` when every
  key field's ratio is <= 3, ``near-spread`` when every ratio is <= 10,
  otherwise ``outside-spread`` with the fields that are outside named.  A
  field whose yardstick RMS is 0 (both WRF builds agree on every interior
  word) is reported with WOOF's RMS and counted outside the spread unless
  WOOF's RMS is also 0.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

KEY_FIELDS = ("U", "V", "W", "T", "PH", "MU", "P", "QVAPOR", "QCLOUD",
              "QRAIN", "QICE", "QSNOW", "QGRAUP", "RAINC", "RAINNC", "T2",
              "Q2", "U10", "V10", "PSFC", "HFX", "LH", "TSK", "SWDOWN", "GLW",
              "OLR", "PBLH", "UST")
INSIDE, NEAR = 3.0, 10.0


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _rms(entry):
    if not entry or "n" not in entry:
        return None
    if not entry.get("differ"):
        return 0.0
    return entry.get("rms_interior")


def score(yard, cand):
    frames_y = {f["time"]: f["fields"] for f in yard["frames"]}
    rows = []
    first_diff = None
    for frame in cand["frames"]:
        t = frame["time"]
        differing = sorted(k for k, v in frame["fields"].items()
                           if v.get("differ") or "shape_cand" in v)
        if differing and first_diff is None:
            first_diff = {"time": t, "fields": len(differing),
                          "lead": differing[:8]}
        ratios = {}
        for name in KEY_FIELDS:
            c = _rms(frame["fields"].get(name))
            y = _rms(frames_y.get(t, {}).get(name))
            if c is None or y is None:
                continue
            if y == 0.0:
                ratios[name] = {"woof_rms": c, "wrf_spread_rms": 0.0,
                                "ratio": 0.0 if c == 0.0 else math.inf}
            else:
                ratios[name] = {"woof_rms": c, "wrf_spread_rms": y,
                                "ratio": c / y}
        rows.append({"time": t, "fields_differing": len(differing),
                     "fields_compared": len(frame["fields"]),
                     "ratios": ratios})
    last = rows[-1] if rows else {"ratios": {}}
    worst = {k: v["ratio"] for k, v in last["ratios"].items()}
    outside = sorted(k for k, r in worst.items() if r > INSIDE)
    far = sorted(k for k, r in worst.items() if r > NEAR)
    verdict = ("inside-spread" if not outside else
               "near-spread" if not far else "outside-spread")
    return {"first_differing_frame": first_diff, "frames": rows,
            "last_frame": last.get("time"), "verdict": verdict,
            "outside_3x": outside, "outside_10x": far,
            "max_ratio_last": max(worst.values()) if worst else None}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out_dir, summary_path = argv[0], argv[1]
    result = {"rule": __doc__.split("THE RULE")[1].strip(), "arms": {}}
    for yard_path in sorted(glob.glob(os.path.join(
            out_dir, "*.wrfvec-vs-wrfscalar.json"))):
        arm = os.path.basename(yard_path).split(".")[0]
        yard = _load(yard_path)
        entry = {"wrf_vectorised_vs_scalar": score(yard, yard)}
        for cand_path in sorted(glob.glob(os.path.join(
                out_dir, f"{arm}.woof-*-vs-wrfscalar.json"))):
            label = os.path.basename(cand_path)[len(arm) + 1:].split(
                "-vs-")[0]
            entry[label] = score(yard, _load(cand_path))
        result["arms"][arm] = entry
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=float)
    for arm, entry in result["arms"].items():
        for label, s in entry.items():
            fd = s["first_differing_frame"]
            print(f"{arm:14s} {label:24s} first-diff "
                  f"{(fd or {}).get('time', 'none'):20s} "
                  f"verdict {s['verdict']:15s} max-ratio "
                  f"{s['max_ratio_last'] if s['max_ratio_last'] is None else round(s['max_ratio_last'], 2)}"
                  f" outside3x {','.join(s['outside_3x'])}")


if __name__ == "__main__":
    main()
