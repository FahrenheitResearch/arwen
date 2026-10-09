"""sum6.py BEFORE_DIR AFTER_DIR : the six cases' clock decisions from the wind checks (wind5.py) before (step 5 tree)
and after (step 6 tree): per domain, the decision (step, substeps, status, ceiling), the least limit at the read
winds, the headroom (least limit over the longest step run), and the hourly check (hours whose decision re-derived at
the actual winds equals the one read, faces that would refuse a step that ran)."""
import json
import math
import sys
from pathlib import Path

CASES = [("ncar12", "NCAR 12 km"), ("ncar25", "NCAR 2.5 km"), ("la-santaana-250107-48h", "LA Santa Ana"),
         ("sf-diablo-210118-48h", "SF Diablo"), ("tor18z-48h", "Toronto"), ("bos0400-48h", "Boston")]


def read(path):
    doc = json.loads(Path(path).read_text())
    out = {}
    for gid, dom in doc["domains"].items():
        h = dom["header"]
        a = h["as_run"]
        times = dom["times"]
        same = sum(1 for t in times if t["hindsight_same"])
        refuse = sum(t["all"].get("would_refuse_step_run", 0) for t in times)
        env = dom.get("envelope") or {}
        least = h["least_limit_at_read_s"]
        least = None if least is None or not math.isfinite(least) else least
        out[gid] = {
            "dx_km": h["dx_m"] / 1000.0, "status": h["status"], "dt_s": a["dt_s"], "substeps": a["time_step_sound"],
            "division": a["division"], "ceiling_s": a["ceiling_s"], "limit_per_km": a["limit_per_km"],
            "cap_per_km": a["cap_per_km"], "cap_source": a["cap_source"], "run_step_s": h["run_step_s"],
            "least_limit_s": least,
            "headroom": None if least is None else round(least / h["run_step_s"], 4),
            "reproduces": h["reproduces_as_run"], "hours": len(times), "hours_same": same,
            "refusals": refuse, "envelope_same": env.get("hindsight_same"),
            "envelope_refuse": (env.get("all") or {}).get("would_refuse_step_run"),
            "envelope_lowest_limit_at_actual_s": (env.get("all") or {}).get("lowest_limit_at_actual_s"),
            "governing": h.get("governing")}
    return out


before, after = Path(sys.argv[1]), Path(sys.argv[2])
table = {}
for case, name in CASES:
    b = read(before / f"{case}.json")
    a = read(after / f"{case}.json")
    table[case] = {"name": name, "before": b, "after": a}
    for gid in sorted(set(b) | set(a)):
        for tag, x in (("before", b.get(gid)), ("after", a.get(gid))):
            if x is None:
                continue
            print(f"{name:13s} d0{gid} {tag:6s} {x['status']:15s} dt {x['dt_s']:.4g} s x{x['substeps']} "
                  f"div {x['division']} ceiling {x['ceiling_s']} limit {x['limit_per_km']} cap {x['cap_per_km']} "
                  f"({x['cap_source']}) run {x['run_step_s']:.4g} s least {x['least_limit_s']} "
                  f"headroom {x['headroom']} hours {x['hours_same']}/{x['hours']} refuse {x['refusals']} "
                  f"env {x['envelope_same']} env_refuse {x['envelope_refuse']} repro {x['reproduces']}")
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(table, indent=1))
