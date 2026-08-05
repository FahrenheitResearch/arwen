#!/usr/bin/env python3
"""Emit the WRF-side node bundle for ONE battery experiment config.

The battery's third and fourth arms are stock WRF v4.6.1 on the nodes and
its one-ULP twin (spec sections 6.1, 6.2, 6.4).  Two rules govern how their
inputs come to exist, and both are about not writing the same physics
twice:

* **The mirrored namelist is GENERATED from the battery config, never
  hand-written beside it** (spec section 6.3).  The generator is
  :func:`gpuwm.hrrr_route_inputs.render_namelist_input`, the same renderer
  the HRRR route uses for its own native/stock pair -- so a physics switch
  that differs between the two arms is impossible by construction rather
  than by review, and a config the route refuses is refused HERE too,
  before a node is booked.
* **The twin is the existing instrument.**  ``tools/n5s/perturb_ulp.py``
  applies exactly one documented floating-point ULP to a ``wrfinput``
  field and writes its own ``perturbation.json``; this tool emits the
  command rather than a second perturbation implementation.

Everything this writes is text: namelists, a plan document, and a launch
script.  It runs no forecast, contacts no node, and requires no GPU -- the
point is that the whole node campaign can be composed, reviewed and
committed before either venue is free.

Wall-clock projections come from the committed speed anchors
(``docs/public/receipts/obsbattery/B4-SPEED-ANCHORS.json``), which carries
the status of each anchor's evidence; this tool never restates a speed
number of its own.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

PLAN_SCHEMA = "gpuwm-battery-wrf-node-plan-v1"
SPEED_ANCHORS = (
    Path("docs") / "public" / "receipts" / "obsbattery"
    / "B4-SPEED-ANCHORS.json")

#: The battery's twin rung 1, exactly as the spec registers it: one
#: documented FP ULP on the wrfinput theta field.  ``T`` is WRF's
#: perturbation potential temperature and is ``perturb_ulp.py``'s own
#: default field.  Rung 2 (seeded FP32 noise) is a registered escalation
#: that only fires if the shakedown finds the rung-1 envelope degenerate,
#: and it is re-registered as one motion -- never chosen per case, and
#: never chosen here.
TWIN_FIELD = "T"
TWIN_MEMBER_ORDINAL = 0

#: Standing operational law on the nodes: anything that will outlive a
#: 60 minute supervision window is launched detached.  A battery-shape
#: 24 h WRF integration is projected at ~3.5 h, so every arm this tool
#: plans is a detached launch, not a foreground one.
DETACH_THRESHOLD_SECONDS = 55 * 60


def _speed_anchors(repository_root: Path) -> dict:
    path = repository_root / SPEED_ANCHORS
    if not path.is_file():
        raise FileNotFoundError(
            f"{SPEED_ANCHORS} is missing; the node plan quotes its speed "
            "numbers from that receipt rather than carrying its own")
    return json.loads(path.read_text(encoding="utf-8"))


def _wrf_projection(exp, anchors: dict, *, ranks: int) -> dict[str, object]:
    entry = anchors["anchors"]["wrf_v461_nodes_24rank"]
    cells = sum(d.run.nx * d.run.ny * d.run.nz for d in exp.domains)
    scale = cells / float(entry["cells"])
    rate = float(entry["seconds_per_sim_minute"]) * scale
    if ranks != int(entry["ranks"]):
        rate = rate * int(entry["ranks"]) / ranks
    wall = rate * float(exp.run_seconds) / 60.0
    return {
        "anchor": "wrf_v461_nodes_24rank",
        "anchor_status": entry["status"],
        "anchor_seconds_per_sim_minute": entry["seconds_per_sim_minute"],
        "anchor_cells": entry["cells"],
        "anchor_ranks": entry["ranks"],
        "requested_ranks": ranks,
        "scaling": ("linear in cells; and inversely in ranks when ranks "
                    "differ from the anchor's, which is perfect-scaling "
                    "and therefore optimistic"),
        "cells": cells,
        "projected_seconds_per_sim_minute": round(rate, 4),
        "projected_wall_seconds": round(wall, 1),
        "projected_wall_hours": round(wall / 3600.0, 3),
        "projected_wall_hours_control_plus_twin": round(
            2.0 * wall / 3600.0, 3),
        "detached_launch_required": wall > DETACH_THRESHOLD_SECONDS,
    }


def _wps_namelist(exp) -> str:
    from gpuwm.domain_wizard import render_wps_namelist

    projection = {
        "map_proj": exp.projection.map_proj,
        "ref_lat": exp.projection.ref_lat,
        "ref_lon": exp.projection.ref_lon,
        "truelat1": exp.projection.truelat1,
        "truelat2": exp.projection.truelat2,
        "stand_lon": exp.projection.stand_lon,
    }
    dims = [(domain.run.nx, domain.run.ny) for domain in exp.domains]
    ratios = tuple(domain.parent_grid_ratio for domain in exp.domains)
    return render_wps_namelist(
        projection, dims, ratios,
        root_dx_m=float(exp.dx_exact(exp.domains[0].grid_id)),
        source="hrrr")


def _launch_script(*, plan: dict) -> str:
    """The detached launch, written once per arm, tmux per standing law."""

    ranks = plan["ranks"]
    hours = plan["projection"]["projected_wall_hours"]
    detached = plan["projection"]["detached_launch_required"]
    why = (f"# The projection is {hours} h per arm, past the ~55 minute "
           "supervision\n# window, so detaching is mandatory rather than "
           "tidy."
           if detached else
           f"# The projection is {hours} h per arm, inside the ~55 minute\n"
           "# supervision window; this still detaches, because an arm that\n"
           "# overruns its projection must not die with the session that\n"
           "# started it.")
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by tools/battery_wrf_node_plan.py -- do not hand-edit.",
        "# One battery WRF arm.",
        why,
        "set -euo pipefail",
        "",
        'ARM="${1:?arm name: control or twin}"',
        'RUNROOT="${2:?absolute path to the arm run directory}"',
        'WRF_BUILD="${3:?absolute path to the pinned campaign WRF build}"',
        '# The bundle this script was written into, resolved from the',
        '# script itself so a caller cannot pair it with another config.',
        'BUNDLE="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"',
        "",
        "# The pinned campaign build is never rebuilt.  Standing law: the",
        "# reference bundle is not campaign source state, so an arm that",
        "# cannot find this exact build must stop rather than compile one.",
        'test -x "$WRF_BUILD/main/wrf.exe"',
        'test -f "$BUNDLE/namelist.input"',
        "",
        'mkdir -p "$RUNROOT"',
        'cp "$BUNDLE/namelist.input" "$RUNROOT/namelist.input"',
        'cd "$RUNROOT"',
        'SESSION="battery-wrf-$ARM-$(date -u +%Y%m%dT%H%M%SZ)"',
        "",
        "# What the session actually runs, written as a file with its own",
        "# interpreter rather than squeezed into the tmux command string.",
        "# The environment it needs is built INSIDE it, because a session",
        "# created under a tmux server that was already running inherits",
        "# nothing from this shell.  Quoted heredoc: every expansion below",
        "# belongs to the arm's shell, not this one.",
        "cat > \"$RUNROOT/run_arm.sh\" <<'RUN_ARM'",
        "#!/usr/bin/env bash",
        "# Generated by tools/battery_wrf_node_plan.py -- do not hand-edit.",
        "set -o pipefail",
        "",
        "fail() {",
        '  echo "$1" > launch.error',
        "  echo 127 > exit.code",
        "  exit 127",
        "}",
        "",
        "# Published to the tmux server by the launcher below; an empty",
        "# value here means that publication did not take, which is the",
        "# failure that killed all 24 ranks on /main/wrf.exe.",
        '[ -n "${WRF_BUILD:-}" ] || fail "WRF_BUILD is empty inside the '
        'tmux session: the launcher\'s tmux setenv -g did not take"',
        '[ -x "$WRF_BUILD/main/wrf.exe" ] || fail "no executable at '
        '$WRF_BUILD/main/wrf.exe"',
        "",
        "# mpirun is on neither the default nor the login PATH of the",
        "# campaign nodes; oneAPI's setvars.sh is what puts it there.",
        "if [ -f /opt/intel/oneapi/setvars.sh ]; then",
        "  # shellcheck source=/dev/null",
        "  . /opt/intel/oneapi/setvars.sh > setvars.log 2>&1",
        "fi",
        'command -v mpirun > /dev/null 2>&1 || fail "mpirun is not on PATH '
        "and /opt/intel/oneapi/setvars.sh did not put it there (absent, or "
        'sourced without providing it); this arm cannot launch"',
        "",
        f'mpirun -np {ranks} "$WRF_BUILD"/main/wrf.exe'
        " > wrf.stdout 2> wrf.stderr",
        "echo $? > exit.code",
        "RUN_ARM",
        'chmod +x "$RUNROOT/run_arm.sh"',
        "",
        "# Both campaign nodes hold long-lived unrelated tmux sessions, so",
        "# a server is already up and a new session under it does NOT",
        "# inherit this shell's environment -- `export WRF_BUILD` reached",
        "# nothing.  The value goes to the SERVER, which is what a new",
        "# session inherits from; start-server is a no-op when one is",
        "# already running, and is what makes setenv work when one is not.",
        "tmux start-server",
        'tmux setenv -g WRF_BUILD "$WRF_BUILD"',
        'tmux new-session -d -s "$SESSION" -c "$RUNROOT" '
        '"$RUNROOT/run_arm.sh"',
        'echo "launched $SESSION in $RUNROOT"',
        'echo "attach with: tmux attach -t $SESSION"',
        'echo "re-run in the foreground with: '
        '(cd $RUNROOT && WRF_BUILD=$WRF_BUILD ./run_arm.sh)"',
        "",
    ]
    return "\n".join(lines)


def build(config_path: Path, outdir: Path, *, ranks: int,
          repository_root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    """Write the node bundle and return its plan document."""

    from gpuwm.experiment import load_experiment
    from gpuwm.hrrr_route_inputs import (
        FORCING_INTERVAL_SECONDS, render_namelist_input, render_target_domain)

    config_path = Path(config_path)
    outdir = Path(outdir)
    exp = load_experiment(config_path)
    anchors = _speed_anchors(repository_root)

    # Refuses HERE if the suite is one the route cannot express -- which is
    # the whole point of generating rather than hand-writing: the WRF arm
    # cannot silently run a physics the ArWen arm is refused for.
    namelist_input = render_namelist_input(exp, stock=True)
    native_namelist_input = render_namelist_input(exp)
    wps_namelist = _wps_namelist(exp)
    target_domain = render_target_domain(exp)

    outdir.mkdir(parents=True, exist_ok=True)
    written = {
        "namelist.input": namelist_input,
        "namelist.native.input": native_namelist_input,
        "namelist.wps": wps_namelist,
        "d01-target.json": target_domain,
    }
    for name, text in written.items():
        (outdir / name).write_text(text, encoding="ascii")

    projection = _wrf_projection(exp, anchors, ranks=ranks)
    plan = {
        "schema": PLAN_SCHEMA,
        "experiment_config": str(config_path).replace("\\", "/"),
        "experiment_name": exp.name,
        "start_time": exp.start_time.isoformat(),
        "end_time": (
            exp.start_time + timedelta(seconds=float(exp.run_seconds))
        ).isoformat(),
        "run_seconds": float(exp.run_seconds),
        "forcing_interval_seconds": FORCING_INTERVAL_SECONDS,
        "ranks": ranks,
        "files": sorted(written),
        "arms": [
            {
                "arm": "W",
                "role": "stock WRF v4.6.1 control",
                "wrfinput": "wrfinput_d01 as exported, unmodified",
                "runs": 1,
            },
            {
                "arm": "W-twin",
                "role": ("perturbed WRF control: prices what chaos alone "
                         "moves each score at each lead"),
                "wrfinput": (
                    "one documented FP ULP on the wrfinput theta field, "
                    "rung 1 of the registered escalation ladder"),
                "runs": 1,
                "command": (
                    "python tools/n5s/perturb_ulp.py "
                    "<control>/wrfinput_d01 <twin>/wrfinput_d01 "
                    f"--field {TWIN_FIELD} "
                    f"--member-ordinal {TWIN_MEMBER_ORDINAL} "
                    "--record <twin>/perturbation.json"),
                "degeneracy_check": (
                    "if the shakedown case yields |S_twin - S_ctl| = 0 on "
                    "the primary score, the ladder escalates to rung 2 "
                    "(seeded FP32 noise) as ONE re-registered motion "
                    "before any battery case is scored -- never per case"),
            },
        ],
        "projection": projection,
        "detached_launch": {
            "required": projection["detached_launch_required"],
            "reason": ("projected wall clock exceeds the 55 minute "
                       "supervision window"),
            "script": "launch_arm.sh",
        },
        "speed_anchor_receipt": str(SPEED_ANCHORS).replace("\\", "/"),
    }
    (outdir / "launch_arm.sh").write_text(
        _launch_script(plan=plan), encoding="ascii")
    plan["files"] = sorted(plan["files"] + ["launch_arm.sh", "plan.json"])
    (outdir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--ranks", type=int, default=24,
        help="MPI ranks per arm (default 24, the speed anchor's own)")
    return parser


def main(argv=None) -> int:
    from gpuwm.hrrr_route_inputs import HrrrRouteInputError

    args = _parser().parse_args(argv)
    if args.ranks < 1:
        raise ValueError("ranks must be positive")
    try:
        plan = build(args.experiment_config, args.outdir, ranks=args.ranks)
    except HrrrRouteInputError as refusal:
        # Not a defect in this tool: the mirrored namelist comes from the
        # route's own renderer, so a suite the ArWen arm cannot run is a
        # suite the WRF arm must not be given either.  Report it as the
        # route's refusal, with the tool that explains it.
        print(f"the run route refuses this suite: {refusal}", file=sys.stderr)
        print("run tools/battery_route_preflight.py --experiment-config "
              f"{args.experiment_config} for every gate's answer",
              file=sys.stderr)
        return 1
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
