#!/usr/bin/env python3
"""Orchestrate actual archived-source replay; all decoding/comparison is Rust."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fixture_path = args.fixture.resolve()
    fixture = json.loads(fixture_path.read_text())
    base = fixture_path.parent
    output = args.output.resolve()
    output.mkdir()
    (output / "tmp").mkdir()
    env = dict(os.environ, RAYON_NUM_THREADS="24", TMPDIR=str(output / "tmp"))
    commands = []

    def run(tool, config_args):
        command = [str((base / tool).resolve()), *map(str, config_args)]
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        entry = dict(command=command, exit_code=result.returncode,
                     cli_wall_ms=(time.perf_counter() - start) * 1000,
                     stderr=result.stderr)
        if result.returncode == 0:
            entry["result"] = json.loads(result.stdout)
        else:
            entry["stdout"] = result.stdout
        commands.append(entry)
        (output / "receipt.json").write_text(json.dumps(commands, indent=2) + "\n")
        result.check_returncode()

    jobs = []
    for source in fixture["archives"]:
        job = dict(source)
        job["archive"] = str((base / job["archive"]).resolve())
        job["output"] = str(output / "extracted")
        jobs.append(job)
    archive_config = output / "extract.json"
    archive_config.write_text(json.dumps(jobs, indent=2) + "\n")
    run(fixture["archive_tool"], [archive_config])

    config = fixture["network_config"]
    for key in ("edges", "attributes"):
        config[key]["path"] = str(output / "extracted" / config[key]["path"])
    config["output"] = str(output / "network")
    network_config = output / "prepare.json"
    network_config.write_text(json.dumps(config, indent=2) + "\n")
    run(fixture["network_tool"], ["subset-source-network", network_config])
    run(fixture["network_tool"], ["replay-source-network", "--strict",
        output / "network" / "network.json", base / fixture["native_capture"]])


if __name__ == "__main__":
    main()
