"""Run the carried verification lanes' real ancestry, CPU and GPU gates.

Run on a sprint box. No successful ancestry or skipped GPU test substitutes
for a column comparison. Every subprocess has its own log and exit status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "tools" / "verify_lanes_stack.json"


def run(command, env, log):
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {log.name}", flush=True)
    return {"command": command, "exit_code": result.returncode,
            "log": str(log)}


def ancestry(lanes, revision="HEAD"):
    results = []
    for lane in lanes:
        if not lane["take"]:
            continue
        command = ["git", "merge-base", "--is-ancestor", lane["branch"], revision]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        print(f"{'PASS' if completed.returncode == 0 else 'FAIL'} ancestry "
              f"{lane['branch']} {lane['tip']}", flush=True)
        results.append({"lane": lane["branch"], "command": command,
                        "exit_code": completed.returncode,
                        "error": completed.stderr.strip()})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("ancestry", "cpu", "gpu", "all"),
                        default="all")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lane", action="append", default=[])
    parser.add_argument("--gpu-cache", type=Path,
                        help="Reuse executed nodes shared by lanes at this exact revision")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    known = {lane["branch"] for lane in plan["lanes"] if lane["take"]}
    if set(args.lane) - known:
        parser.error("Requested lane is unknown or intentionally skipped")
    lanes = [lane for lane in plan["lanes"]
             if not args.lane or lane["branch"] in args.lane]
    results = []
    revision = subprocess.check_output(["git", "rev-parse", args.revision],
                                       cwd=ROOT, text=True).strip()
    source_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD",
                                           "--", "gpuwm", "tests", "tools"], cwd=ROOT)
    source_diff_sha256 = hashlib.sha256(source_diff).hexdigest()
    cache = {"revision": revision, "source_diff_sha256": source_diff_sha256,
             "nodes": {"strict": {}, "default": {}}}
    if args.gpu_cache and args.gpu_cache.exists():
        cache = json.loads(args.gpu_cache.read_text())
        if (cache["revision"] != revision or
                cache.get("source_diff_sha256") != source_diff_sha256):
            parser.error("GPU cache is for different source bytes; use a fresh receipt path")
    if args.phase in ("ancestry", "all"):
        results.extend(ancestry(lanes, args.revision))
    if args.phase in ("cpu", "all"):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
        env.pop("GPUWM_WRF_EXACT", None)
        tests = sorted({path for lane in lanes if lane["take"]
                        for path in lane.get("tests", [])})
        collection = args.out / "collection.json"
        collect = run([sys.executable, "-m", "pytest", "-q", "--collect-only",
                       "-p", "tools.verify_lanes_stack_pytest",
                       "--stack-collection", str(collection), *tests], env,
                      args.out / "collect.log")
        results.append(collect)
        nodes = []
        if collection.exists() and collect["exit_code"] == 0:
            nodes = [row["nodeid"] for row in json.loads(collection.read_text())
                     if not row["gpu"]]
        if nodes:
            suite = run([sys.executable, "-m", "pytest", "-q", "-n", "16",
                         "--junitxml", str(args.out / "cpu.xml"), *nodes], env,
                        args.out / "cpu.log")
            results.append(suite)
        else:
            suite = {"exit_code": 1, "error": "No CPU nodes collected"}
            results.append(suite)
        for lane in lanes:
            if not lane["take"]:
                continue
            tests = lane.get("tests", [])
            if not tests:
                results.append({"lane": lane["branch"], "exit_code": 1,
                                "error": "No committed CPU/unit gate listed"})
                continue
            covered = [node for node in nodes if node.split("::", 1)[0] in tests]
            results.append({"lane": lane["branch"], "tests": tests,
                            "cpu_nodes": covered, "exit_code": suite["exit_code"],
                            "log": str(args.out / "cpu.log"),
                            "execution": "shared nodes run once for all owning lanes"})
        for name, command in plan["shared_gates"].items():
            results.append(run([sys.executable, *command], env,
                               args.out / (name + ".log")))
    if args.phase in ("gpu", "all"):
        if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error("GPU gates require the sprint card queue and its assigned CUDA_VISIBLE_DEVICES")
        files = sorted({path for lane in lanes if lane["take"]
                        for path in lane.get("gpu_tests", [])})
        collection = args.out / "gpu-collection.json"
        collect = run([sys.executable, "-m", "pytest", "-q", "--collect-only",
                       "-p", "tools.verify_lanes_stack_pytest",
                       "--stack-collection", str(collection), *files], dict(os.environ),
                      args.out / "gpu-collect.log")
        results.append(collect)
        gpu_nodes = []
        if collection.exists() and collect["exit_code"] == 0:
            gpu_nodes = [row["nodeid"] for row in json.loads(collection.read_text())
                         if row["gpu"]]
        for lane in lanes:
            if not lane["take"] or not lane.get("gpu_tests"):
                continue
            nodes = [node for node in gpu_nodes
                     if node.split("::", 1)[0] in lane["gpu_tests"]]
            if not nodes:
                results.append({"lane": lane["branch"], "exit_code": 1,
                                "error": "No GPU nodes collected for the listed gate"})
                continue
            for mode in ("strict", "default"):
                env = dict(os.environ)
                env.pop("GPUWM_NO_LOCAL_GPU", None)
                env.pop("GPUWM_WRF_EXACT", None)
                if mode == "strict":
                    env["GPUWM_WRF_EXACT"] = "1"
                name = lane["branch"].split("/")[-1] + "-" + mode
                comparison_path = args.out / (name + "-words.json")
                pending = [node for node in nodes if node not in cache["nodes"][mode]]
                if pending:
                    result = run([sys.executable, "-m", "pytest", "-q", "-s", "-m", "gpu",
                              "-p", "tools.verify_lanes_stack_pytest",
                              "--stack-receipt", str(comparison_path),
                              *pending], env, args.out / (name + ".log"))
                    if comparison_path.exists():
                        words = json.loads(comparison_path.read_text())
                        for node in pending:
                            reports = [row for row in words.get("reports", [])
                                       if row["nodeid"] == node]
                            events = [event for event in words["events"] if event["test"] == node]
                            counted = [row for event in events for row in event["rows"]]
                            failed = not reports or any(r["outcome"] == "failed" for r in reports)
                            cache["nodes"][mode][node] = {
                                "exit_code": 1 if failed else 0,
                                "compared_words": sum(r["words"] for r in counted),
                                "differing_words": sum(r["differing"] for r in counted),
                                "reports": reports, "log": result["log"]}
                        if args.gpu_cache:
                            args.gpu_cache.write_text(json.dumps(cache, indent=2) + "\n")
                else:
                    result = {"exit_code": 0, "log": "shared executed nodes in GPU cache"}
                result.update(lane=lane["branch"], mode=mode)
                if args.gpu_cache:
                    coverage = [cache["nodes"][mode].get(node, {"exit_code": 1}) for node in nodes]
                    result["exit_code"] = max([result["exit_code"], *(r["exit_code"] for r in coverage)])
                    result["compared_words"] = sum(r.get("compared_words", 0) for r in coverage)
                    result["differing_words"] = sum(r.get("differing_words", 0) for r in coverage)
                    result["nodes"] = {node: cache["nodes"][mode].get(node) for node in nodes}
                    result["execution"] = "Each mode/node runs once and covers every owning lane"
                    if not any(report["when"] == "call" and report["outcome"] == "passed"
                               for row in coverage for report in row.get("reports", [])):
                        result.update(exit_code=1, error="All GPU nodes were skipped or failed")
                    print(f"{lane['branch']} {mode}: compared={result['compared_words']} "
                          f"differing={result['differing_words']}", flush=True)
                elif comparison_path.exists():
                    words = json.loads(comparison_path.read_text())
                    result["compared_words"] = words["compared_words"]
                    result["differing_words"] = words["differing_words"]
                    print(f"{lane['branch']} {mode}: compared={words['compared_words']} "
                          f"differing={words['differing_words']}", flush=True)
                results.append(result)
    receipt = {"schema": "gpuwm.verify-lanes-stack/v1", "phase": args.phase,
               "revision": revision,
               "source_diff_sha256": source_diff_sha256,
               "results": results, "passed": all(r["exit_code"] == 0 for r in results)}
    (args.out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
