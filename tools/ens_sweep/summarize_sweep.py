"""Collapse the sweep's per-run receipts into the one table it exists to fill.

N vs cost vs skill vs memory, plus the cost-scaling fit that says whether
"cost is linear in N" survives contact with a measurement.

Nothing is estimated here.  Every column is read from a receipt a run
wrote: wall clock from the driver's own per-trajectory timings and the
wrapper's clock, LETKF solve seconds from the cycle report, peak VRAM
from an nvidia-smi sampler scoped to that run alone, peak container RAM
from cgroup memory.current, and skill from the scorer that reproduces
the published baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def peak_from(path: Path, scale: float = 1.0):
    if not path.is_file():
        return None
    values = []
    for line in path.read_text(errors="replace").splitlines():
        token = line.split(",")[0].split()[0].strip() if line.strip() else ""
        try:
            values.append(float(token))
        except ValueError:
            continue
    return round(max(values) * scale, 1) if values else None


def load_run(root: Path, members: int) -> dict | None:
    out = root / f"cycle-n{members}"
    report_path = out / "cycle-report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    legs = report["legs"]

    solves, spreads, innovations, control_innov = [], [], [], []
    leg_walls = []
    member_leg_walls = []
    for leg in legs:
        walls = [t["wall_seconds"] for t in leg["trajectories"].values()
                 if "wall_seconds" in t]
        leg_walls.append(round(sum(walls), 1))
        member_leg_walls.extend(walls)
        analysis = leg.get("analysis")
        if analysis:
            solves.append(analysis["solve_seconds"])
            innovation = (analysis.get("innovations") or [{}])[0]
            spreads.append(round(innovation.get("ensemble_spread_mean",
                                                float("nan")), 4))
            innovations.append(round(innovation.get("innovation_rms",
                                                    float("nan")), 4))
            control_innov.append(round(
                analysis["control_vr"]["innovation_rms_ms"], 4))

    record: dict = {
        "members": members,
        "total_wall_seconds": report.get("total_wall_seconds"),
        "integration_wall_seconds": round(sum(leg_walls), 1),
        "wall_per_leg_seconds": leg_walls,
        "mean_member_leg_seconds": round(float(np.mean(member_leg_walls)), 3),
        "letkf_solve_seconds": solves,
        "letkf_solve_total": round(sum(solves), 1),
        "obs_space_spread": spreads,
        "innovation_rms_ensemble": innovations,
        "innovation_rms_control": control_innov,
        "peak_vram_mib": peak_from(out / "vram-samples.csv"),
        "peak_container_ram_gib": peak_from(out / "hostmem-samples.txt",
                                            1.0 / 1024 ** 3),
    }
    wall_txt = out / "wall.txt"
    if wall_txt.is_file():
        record["wrapper_wall_seconds"] = wall_txt.read_text().strip()

    score_path = out / "score.json"
    if score_path.is_file():
        score = json.loads(score_path.read_text(encoding="utf-8"))
        record["fss"] = [f["ensemble_mean"]["fss30_27km"]
                         for f in score["frames"]]
        record["cols35"] = [f["ensemble_mean"]["cols_gt35_in_echo"]
                            for f in score["frames"]]
        record["fss_control"] = [f.get("control", {}).get("fss30_27km")
                                 for f in score["frames"]]
        record["cols35_obs"] = [f["obs_cols_gt35"] for f in score["frames"]]
        record["fss_member_mean"] = [f["member_fss"]["mean"]
                                     for f in score["frames"]]
        record["fss_mean_over_leads"] = round(
            float(np.mean(record["fss"])), 4)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--members", type=int, action="append", required=True)
    parser.add_argument("--ceiling-probe", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = [r for r in (load_run(args.root, n) for n in args.members)
            if r is not None]

    payload: dict = {
        "schema": "gpuwm-da.ensemble-sweep-summary.v1",
        "runs": runs,
    }

    # -- does cost actually scale linearly in N? ------------------------
    # The claim under test is "near-linear, because members advance
    # serially".  The honest form of that claim has an intercept: the
    # control trajectory and the LETKF solve are paid whatever N is.
    complete = [r for r in runs if r.get("total_wall_seconds")]
    if len(complete) >= 2:
        n = np.array([r["members"] for r in complete], float)
        t = np.array([r["total_wall_seconds"] for r in complete], float)
        slope, intercept = np.polyfit(n, t, 1)
        predicted = slope * n + intercept
        payload["cost_model"] = {
            "form": "total_wall_seconds = slope * N + intercept",
            "slope_seconds_per_member": round(float(slope), 2),
            "intercept_seconds": round(float(intercept), 1),
            "max_residual_seconds": round(
                float(np.max(np.abs(t - predicted))), 1),
            "max_residual_percent": round(
                float(np.max(np.abs(t - predicted) / t) * 100), 2),
            "note": "twelve legs; the intercept is the control trajectory "
                    "plus per-leg fixed work, which N does not pay for",
        }
        seconds_per_member_leg = [
            (r["members"], r["mean_member_leg_seconds"]) for r in complete]
        payload["cost_model"]["mean_member_leg_seconds_by_n"] = \
            seconds_per_member_leg

    if args.ceiling_probe and args.ceiling_probe.is_file():
        payload["ceiling_probe"] = [
            json.loads(line) for line in
            args.ceiling_probe.read_text().splitlines() if line.strip()]

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- the table ------------------------------------------------------
    print(f"{'N':>5} {'total s':>9} {'s/member-leg':>13} {'solve s':>9} "
          f"{'VRAM MiB':>9} {'RAM GiB':>8} {'FSS +15':>8} {'FSS +90':>8} "
          f"{'FSS mean':>9}")
    for r in runs:
        fss = r.get("fss") or []
        print(f"{r['members']:>5} "
              f"{r.get('total_wall_seconds', 0):>9.1f} "
              f"{r['mean_member_leg_seconds']:>13.2f} "
              f"{r['letkf_solve_total']:>9.1f} "
              f"{str(r.get('peak_vram_mib')):>9} "
              f"{str(r.get('peak_container_ram_gib')):>8} "
              f"{(f'{fss[0]:.4f}' if fss else '-'):>8} "
              f"{(f'{fss[-1]:.4f}' if fss else '-'):>8} "
              f"{str(r.get('fss_mean_over_leads', '-')):>9}")
    if "cost_model" in payload:
        model = payload["cost_model"]
        print(f"\ncost: {model['slope_seconds_per_member']} s per member "
              f"+ {model['intercept_seconds']} s fixed; worst residual "
              f"{model['max_residual_percent']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
