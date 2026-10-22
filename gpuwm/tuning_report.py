"""Analysis figures and Rust weather maps from a completed tuning receipt."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from gpuwm.tuning import read_json, write_json, digest, verify_packet


def charts(packet, receipt_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    round_ = verify_packet(packet)
    receipt = read_json(receipt_path)
    scores = receipt["scores"]
    if not scores["decision"].get("heldout"):
        raise ValueError("figures require real training and held-out scores")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rank = {row["member"]: row for row in scores["ranking"]}
    saved = []
    for parameter in round_["parameter_space"]:
        name = parameter["name"]
        members = [member for member in round_["members"] if name not in member["inactive_parameters"]]
        fig, ax = plt.subplots(figsize=(6.4, 4), constrained_layout=True)
        for key, caption, color in (("training_score", "Tuning days", "#245c9c"), ("heldout_score", "Held-out days", "#c15b24")):
            ax.scatter([member["parameters"][name] for member in members],
                       [rank[member["id"]][key] for member in members], label=caption, color=color, s=32)
        ax.set_xlabel(f"{name} ({parameter.get('units', 'option')})")
        ax.set_ylabel("Mean daily temperature RMSE (K)")
        ax.set_title("Station temperature score")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
        if parameter["kind"] == "categorical": ax.set_xticks(parameter["values"])
        path = output / f"score-{name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append({"path": str(path), "sha256": digest(path), "active_settings": len(members)})
    write_json(output / "chart-receipt.json", {"input_sha256": digest(receipt_path), "figures": saved,
        "interpretation": "Marginal score scatter from a joint parameter search, not isolated causal effects"})
    return saved


def maps(packet, receipt_path, runs, output, timeout):
    from gpuwm.tuning_collect import FRAME
    round_ = verify_packet(packet)
    decision = read_json(receipt_path)["scores"]["decision"]
    if decision.get("heldout") is None:
        raise ValueError("maps require a frozen winner and completed held-out receipt")
    output, runs = Path(output), Path(runs)
    results = []
    for day in round_["train_days"] + round_["heldout_days"]:
        valid = day + "T22:00:00"
        for member in dict.fromkeys((round_["baseline_member"], decision["winner"])):
            # Render the same valid time and every original domain. Python
            # enumerates files; the renderer reads and processes all weather fields.
            for domain in round_["domain_inventory"]:
                found = []
                for path in (runs / day / member / "run").rglob("wrfout_*"):
                    match = FRAME.fullmatch(path.name)
                    if match and match.groups() == (domain["id"], day, "22", "00", "00"):
                        found.append(path)
                if len(found) != 1:
                    raise ValueError(f"one retained 22 UTC map frame required for {day}/{member}/{domain['id']}")
                source = found[0]
                folder = output / round_["round_id"] / member
                folder.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-B", "-m", "gpuwm.cli", "render", str(source),
                           "--engine", "rust", "--products", "t2", "--out", str(folder),
                           "--layout", "nested", "--run-stamp", "off"]
                log = folder / f"render-{domain['id']}-{day}.log"
                with log.open("w", encoding="utf-8") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=True,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1"))
                results.append({"member": member, "day": day, "domain": domain["id"],
                    "input_sha256": digest(source), "command": command, "log_sha256": digest(log)})
    write_json(output / "map-receipt.json", {"receipt_sha256": digest(receipt_path), "maps": results})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("charts", "maps"))
    parser.add_argument("--packet", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    try:
        if args.mode == "charts": charts(args.packet, args.receipt, args.out)
        else:
            if not args.runs: parser.error("maps requires --runs")
            maps(args.packet, args.receipt, args.runs, args.out, args.timeout_seconds)
        return 0
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"tuning report: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
