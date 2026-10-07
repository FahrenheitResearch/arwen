"""mkargv.py PLAN ARM OUT UUID...: the 10-01 CONUS campaign DA argv (run-plan.jsonl, arm da) with lane 7's arms.

ARM: base (one-shot, the default), hyd (one-shot + hydrostatic rebalance), 3d (in-RK IAU over 30 min),
3dh (3d + rebalance), split (the obs lane's post-step IAU over 30 min, for reference), 4d / 4dh (centred
60 min window, rewound legs). No --save-ensemble; history every 600 s; MASPT over each leg's first 60 min."""
import json
import shlex
import sys

plan, arm, out, *uuids = sys.argv[1:]
a = next(r for r in map(json.loads, open(plan)) if r.get("arm") == "da")["argv"]


def span(flag):
    i = a.index(flag)
    j = i + 1
    while j < len(a) and not a[j].startswith("--"):
        j += 1
    return i, j


def setv(flag, vals):
    if flag in a:
        i, j = span(flag)
        a[i + 1:j] = vals
    else:
        a.extend([flag, *vals])


def drop(flag):
    if flag in a:
        i, j = span(flag)
        del a[i:j]


setv("--forecast-device-uuids", uuids)
drop("--save-ensemble")
# history stays at the plan's 120 s: the S1 sheet reads the analysis + 2 min frame
setv("--out", [f"{out}/da"])
setv("--maspt-minutes", ["60"])
arms = {
    "base": [],
    "hyd": ["--hydrostatic-rebalance"],
    "3d": ["--iau-mode", "3d", "--iau-window-seconds", "1800"],
    "3dh": ["--iau-mode", "3d", "--iau-window-seconds", "1800", "--hydrostatic-rebalance"],
    "split": ["--iau-mode", "split", "--iau-window-seconds", "1800"],
    "4d": ["--iau-mode", "4d", "--iau-window-seconds", "3600"],
    "4dh": ["--iau-mode", "4d", "--iau-window-seconds", "3600", "--hydrostatic-rebalance"],
}
smoke = arm.startswith("smoke-")
if smoke:
    # One card, four members in turn, two observed hours then one free hour: exercises every
    # insertion path in the real model before the 8-card arms.
    arm = arm[len("smoke-"):]
    obs = [a[i + 1] for i, x in enumerate(a) if x == "--obs"][:2]
    grids = [a[i + 1] for i, x in enumerate(a) if x == "--grid-wrfout"][:2]
    for flag in ("--obs", "--grid-wrfout"):
        while flag in a:
            drop(flag)
    for o, g in zip(obs, grids):
        a.extend(["--obs", o, "--grid-wrfout", g])
    setv("--members", ["4"])
    setv("--forecast-members-per-card", ["serial"])
    setv("--leg-durations-seconds", ["3600.0", "3600.0", "3600.0"])
    setv("--free-legs", ["1"])
    setv("--rain-forecast-start-seconds", ["7200.0"])
if arm.startswith("m8-"):
    # The 4D comparison suite: 4D runs on the serial route (one card), so every arm of the
    # suite does, at 8 members, the full recipe otherwise (3 analyses, 6 h free forecast).
    arm = arm[len("m8-"):]
    setv("--members", ["8"])
    setv("--forecast-members-per-card", ["serial"])
if arm.startswith("4d"):
    setv("--forecast-members-per-card", ["serial"])
for flag in arms[arm]:
    a.append(flag)
# the lane venv: editable-bound to the lane tree, so the provenance gate sees one version
a[0] = "/work/da-iau-7/venv/bin/python"
a = [a[0], "-m", "tools.da_letkf_route_shadow", "--record", f"{out}/shadow", "--production", "cuda-obs-sparse",
     "--no-shadow", "--"] + a[3:]
print(shlex.join(a))
