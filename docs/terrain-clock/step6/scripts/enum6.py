"""enum6.py OUT.json : step 6. For each of the six cases' local-face domains (the tree runner's own preflight, or the
saved NCAR door fixtures, exactly as wind5.py captures them), every map cell the faces read at the clock's substep
counts with the step-6 floor (1.5 x input), flagged where the cell is on a ridge a candidate arm ran but at a wind
column that arm never ran (top null): there the engine reads the shipped cell, measured under the retired old bound.
CPU only."""
import json, sys, os, math
from pathlib import Path
import numpy as np
sys.path.insert(0, "/work/tclock/s4d")
import wind5
from gpuwm import terrain_clock as tc, terrain_clock_local as tl
from gpuwm.core.adaptive_clock import least_sound_steps

R = Path("/opt/dlami/nvme/tclock/runs")
out = {"input_wind_floor": tl.INPUT_WIND_FLOOR, "cases": {}}
cases = [("ncar12", "fixture", ["/work/tclock/results/step2-tree7/door-12-local_face/faces-d01.npz",
                                 "/work/tclock/s3/results/ncar12-local_face/run-receipt.json"]),
         ("ncar25", "fixture", ["/work/tclock/results/step2-tree7/door-25-local_face/faces-d01.npz",
                                 "/work/tclock/s3/results/ncar25-local_face/run-receipt.json"])]
for c in ["la-santaana-250107-48h", "sf-diablo-210118-48h", "tor18z-48h", "bos0400-48h"]:
    P = sorted(R.glob(f"{c}-prep/run-*/chain/prep"))[0]
    cases.append((c, "hrrr", [str(P), f"/work/tclock/s3/cfg/{c}.cand.toml"]))
only = sys.argv[2:] if len(sys.argv) > 2 else None
cand = tl.candidate_document()
cand_rows = {}
for r in cand["rows"]:
    cand_rows.setdefault((r["settings"], float(r["dx_m"]), float(r["crest_m"]), int(r["sound_steps"])), []).append(r)
cw = [float(w) for w in cand["winds_m_s"]]
for label, kind, args in cases:
    if only and label not in only:
        continue
    if kind == "hrrr":
        rec, clock = wind5._capture_hrrr(Path(args[0]), Path(args[1]))
    else:
        rec, clock = wind5._capture_fixture(Path(args[0]), Path(args[1]))
    res = {}
    for gid, r in sorted(rec.items()):
        f = r["faces"]; a = r["adaptation"]; run = r["run"]
        table = f.table; winds = list(table.winds)
        axis, j, i, slope, crest, wind_in = tl._faces(f.fields)
        local = np.where(np.isfinite(wind_in), wind_in, f.domain_wind)
        read = tl.face_wind_read(local, f.domain_wind, f.margin)
        counts = [least_sound_steps(run)] + ([6] if least_sound_steps(run) < 6 else [])
        flagged = {}
        cols_hist = {}
        for count in counts:
            keys = tl._keys(table, f.dx, slope, crest, read, count)
            u, first, inv = np.unique(keys, return_index=True, return_inverse=True)
            nfaces = np.bincount(inv)
            srow = tc._sound_row(count)
            rows = [x for x in table.rows if x.sound_steps == srow]
            for g, k in enumerate(first):
                rd = tc.read_map(f.dx, float(crest[k]), float(slope[k]), float(read[k]), count, table)
                wi = [n for n, x in enumerate(winds) if x >= read[k] - 1e-9]
                wi = wi[0] if wi else len(winds) - 1
                cols_hist[f"x{count}-{winds[wi]:g}"] = cols_hist.get(f"x{count}-{winds[wi]:g}", 0) + int(nfaces[g])
                for sp in rd.dx_rows:
                    crests = sorted({x.crest_m for x in rows if x.dx_m == sp})
                    crow = [h for h in crests if h >= float(crest[k]) * (1 - tc.CREST_MATCH) - 1e-6]
                    crow = crow[0] if crow else crests[-1]
                    at = [x for x in rows if x.dx_m == sp and x.crest_m == crow]
                    st = sorted(x.slope for x in at if x.slope >= float(slope[k]) - 1e-9)
                    cap = st[0] if st else max(x.slope for x in at)
                    for x in at:
                        if x.slope > cap + 1e-12:
                            continue
                        for w in range(wi + 1):
                            for arm in f.arms:
                                cr = [c_ for c_ in cand_rows.get((arm, sp, crow, srow), [])
                                      if abs(float(c_["slope"]) - x.slope) <= 6e-5]
                                if not cr:
                                    continue
                                c_ = cr[0]
                                ci = cw.index(winds[w]) if winds[w] in cw else len(cw)
                                if ci < len(cw) and c_["top_s_per_km"][ci] is not None:
                                    continue
                                key = (arm, sp, crow, c_["ridge_slope"], srow, winds[w])
                                e = flagged.setdefault(key, {"faces": 0, "groups": 0, "slope": x.slope,
                                                             "engine_entry": x.stable[w],
                                                             "engine_top": None if x.tried is None else x.tried[w],
                                                             "cand_prev_entry": c_["stable_s_per_km"][ci - 1] if ci else None,
                                                             "cand_prev_top": c_["top_s_per_km"][ci - 1] if ci else None})
                                e["faces"] += int(nfaces[g]); e["groups"] += 1
        res[str(gid)] = {"dx": f.dx, "arms": list(f.arms), "counts": counts, "domain_wind": f.domain_wind,
                         "max_read": float(read.max()), "decision": wind5._decision(a),
                         "columns": cols_hist,
                         "flagged": [{"arm": k[0], "dx": k[1], "crest": k[2], "ridge_slope": k[3], "sound_steps": k[4],
                                      "wind": k[5], **v} for k, v in sorted(flagged.items())]}
        print(label, gid, res[str(gid)]["decision"], "max read", round(float(read.max()), 2), cols_hist,
              "flagged", len(flagged), flush=True)
    out["cases"][label] = res
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
