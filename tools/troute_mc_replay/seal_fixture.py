#!/usr/bin/env python3
"""Seal the actual native calls after whole-case observation equivalence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from replay import INPUTS, OUTPUTS, PIN, sha, validate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("clean_output", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tool", type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads((args.capture / "execution.json").read_text())
    if execution["source_pin"] != PIN or execution["returncode"]:
        raise ValueError("successful pinned native case execution required")
    if execution["observer"]["method"] != "external-ptrace-entry-return-x86_64":
        raise ValueError("use external observation: wrapper frames perturb native secant scratch")
    tool = str(args.tool.resolve())
    equivalent = subprocess.run([tool, "compare", str(args.clean_output),
                                str(args.capture / "flowveldepth.pkl"), "1"],
                               check=True, capture_output=True, text=True)
    nonperturbation = json.loads(equivalent.stdout)
    split = json.loads(subprocess.check_output(
        [tool, "split", str(args.capture / "calls.raw"), str(args.output)], text=True))
    for name in ("calls.raw", "execution.json", "run.yaml", "native.log"):
        shutil.copyfile(args.capture / name, args.output / name)
    source = Path(execution["cwd"]).parents[1]
    for name in ("LICENSE", "TERMS.md"):
        shutil.copyfile(source / name, args.output / name)
    manifest = dict(schema="troute-mc-calls-v1", source_pin=PIN,
                    license_quote="Apache License; Version 2.0, January 2004",
                    license_sha256=sha(args.output / "LICENSE"),
                    calls=split["calls"], inputs=INPUTS, outputs=OUTPUTS,
                    native=execution, nonperturbation=nonperturbation,
                    split=split, capture_sha256=sha(args.output / "calls.raw"),
                    clean_result_sha256=sha(args.clean_output),
                    observed_result_sha256=sha(args.capture / "flowveldepth.pkl"))
    for name, path, width in (("input_array", "inputs.f32", 15), ("expected_array", "expected.f32", 6)):
        manifest[name] = dict(path=path, dtype="<f4", shape=[split["calls"], width], sha256=sha(args.output / path))
    manifest["caller_seeds"] = dict(path="caller-seeds.f32", sha256=sha(args.output / "caller-seeds.f32"),
                                    meaning="all six output words initialized to +0 by reach.pyx")
    validate(manifest)
    (args.output / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
