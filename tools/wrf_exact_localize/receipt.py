"""Receipt of a localize-pair comparison: WRF 4.6.1 with dump hooks against strict WOOF.

usage: python receipt.py --wrf-dump D --woof-dump D --wrf-rec R --woof-rec R
                         --steps N --case NAME --out receipt.json

What it records, every word compared (cmp4.py, cmpq.py, cmp_hashes.py beside it):

* for each step and each matched point of solve_em/rk_tendency/the acoustic loop/the
  moist scalar path, the number of differing words;
* for each history frame, the fields whose SHA-256 differs.

A row is EXCLUDED, with its reason recorded, only when its two sides are not the
same quantity, so its count says nothing about either model:

* ``k*`` per-term rows that follow different sums: WOOF adds the theta advection
  first and WRF the u advection, so a WOOF intermediate holds a different subset of
  terms than the WRF intermediate dumped at the same label; the aligned per-term rows
  stay in the receipt;
* ``coef``: WRF's a/alpha/gamma are solve_em's tile-local arrays, which the
  memory-dimension crop does not read (w and phi after each substep carry them);
* acoustic ``p``: WRF stores the divergence-damped p'' (calc_p_rho), WOOF the undamped
  one with its damping applied where it is read;
* acoustic ``ww`` and the moist path's ``ww_m``/``advect_tend`` on the specified ring:
  WRF computes values there it never applies (rk_update_scalar starts inside the ring)
  and the in-loop ww is another quantity than WOOF's ww''.

A receipt with ``all_compared_identical`` true is the claim the tests pin.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROW = re.compile(r"^\s+(?P<label>.+?)\s+(?:IDENTICAL \((?P<n>\d+)\)|differ (?P<d>\d+)/(?P<t>\d+)"
                 r"|IDENTICAL$|\(missing\)$|shape .*)")
EXCLUDE = (
    (re.compile(r"^k[0-9A]_\w+ \w+_tend$"), "intermediate of a different term sum"),
    (re.compile(r"^coef "), "WRF tile-local array, not readable by the memory crop"),
    (re.compile(r"^ss(prep|\d+) p$"), "WRF stores divergence-damped p''; WOOF the undamped"),
    (re.compile(r"^ss(prep|\d+) ww$"), "in-loop ww is another quantity; ring values unused"),
    (re.compile(r"^ww_m$"), "specified-ring values WRF never applies"),
    (re.compile(r"^advect_tend vs flux_div$"), "specified-ring values WRF never applies"),
)


def _parse(text: str, section: str):
    rows = []
    stage = None
    for line in text.splitlines():
        if line.startswith("==="):
            stage = line.strip("= ").strip()
            continue
        m = ROW.match(line)
        if not m:
            continue
        label = m["label"].strip()
        if m["d"] is not None:
            differ, total = int(m["d"]), int(m["t"])
        elif m["n"] is not None:
            differ, total = 0, int(m["n"])
        elif "IDENTICAL" in line:
            differ, total = 0, None
        else:
            continue
        reason = next((r for pat, r in EXCLUDE if pat.search(label)), None)
        if reason == "intermediate of a different term sum" and differ == 0:
            reason = None
        rows.append({"section": section, "stage": stage, "point": label, "differ": differ,
                     "words": total, "excluded": reason})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("wrf-dump", "woof-dump", "wrf-rec", "woof-rec", "case", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--steps", type=int, required=True)
    a = ap.parse_args()
    steps = []
    for step in range(1, a.steps + 1):
        out = subprocess.run([sys.executable, str(HERE / "cmp4.py"), a.wrf_dump, a.woof_dump, str(step)],
                             capture_output=True, text=True, check=True).stdout
        rows = _parse(out, "dycore")
        out = subprocess.run([sys.executable, str(HERE / "cmpq.py"), a.wrf_dump, a.woof_dump, str(step)],
                             capture_output=True, text=True, check=True).stdout
        rows += [r for r in _parse(out, "moist") if r["point"] != "WOOF update keys:"]
        compared = [r for r in rows if r["excluded"] is None]
        if not compared:
            raise SystemExit(f"step {step}: no point compared -- wrong dump directory? "
                             "An empty comparison proves nothing.")
        steps.append({"step": step, "points_compared": len(compared),
                      "points_differing": [r for r in compared if r["differ"]],
                      "points_excluded": len(rows) - len(compared), "rows": rows})
    hist = subprocess.run([sys.executable, str(HERE / "cmp_hashes.py"),
                           str(Path(a.wrf_rec) / "hashes.json"), str(Path(a.woof_rec) / "hashes.json")],
                          capture_output=True, text=True, check=True).stdout
    frames = {}
    for line in hist.splitlines():
        m = re.match(r"frame (\d+): (\d+) differ(?:: (.*))?", line)
        if m:
            frames[int(m[1])] = (m[3] or "").split()
    header = hist.splitlines()[0]
    # One label list for the whole run; each step stores its counts against it.
    labels, index = [], {}
    for s in steps:
        for row in s["rows"]:
            key = (row["section"], row["stage"], row["point"])
            if key not in index:
                index[key] = len(labels)
                labels.append({k: row[k] for k in ("section", "stage", "point", "words", "excluded")})
    compact = []
    for s in steps:
        counts = {index[(r["section"], r["stage"], r["point"])]: r["differ"] for r in s["rows"]}
        compact.append({"step": s["step"], "points_compared": s["points_compared"],
                        "points_excluded": s["points_excluded"],
                        "points_differing": [[d["section"], d["stage"], d["point"], d["differ"]]
                                             for d in s["points_differing"]],
                        "differ_by_label": [counts.get(i) for i in range(len(labels))]})
    receipt = {
        "schema": "gpuwm.wrf-exact-localize/v2", "case": a.case, "steps": a.steps,
        "history_header": header,
        "history_frames_differing_fields": frames,
        "all_compared_identical": all(not s["points_differing"] for s in steps),
        "per_step": compact, "labels": labels,
    }
    Path(a.out).write_text(json.dumps(receipt, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("case", "steps", "all_compared_identical")}),
          "compared per step:", sorted({s["points_compared"] for s in steps}),
          "history fields differing:", sorted({f for v in frames.values() for f in v}))


if __name__ == "__main__":
    main()
