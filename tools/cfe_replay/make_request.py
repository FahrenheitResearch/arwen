#!/usr/bin/env python3
"""Bind a real GPU run receipt to the shared comparator's request template.

Metadata only. Rust remains the sole payload reader, hasher and comparator.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("actual", type=Path, help="downloaded run directory, including run.json")
    parser.add_argument("output", type=Path, help="new comparison request")
    parser.add_argument("--initial-manifest", type=Path,
                        help="local copy of the production initial manifest sealed by the run")
    args = parser.parse_args()
    request = json.loads(args.template.read_text())
    run = json.loads((args.actual / "run.json").read_text())
    if request["candidate_revision"] != run["candidate_revision"]:
        raise ValueError("GPU receipt candidate differs from the prepared fixture")
    request["numerics"] = run["numerics"]["mode"]
    # Deliberately require the actual process command, never a guessed template command.
    unset_mode = ["-u", "GPUWM_NUMERICS"] if run["environment"].get("GPUWM_NUMERICS") is None else []
    request["command"] = ["env", *unset_mode, *[f"{key}={value}" for key, value in run["environment"].items()
                                   if value is not None], *run["command"]]
    if bool(run.get("initial_manifest")) != bool(args.initial_manifest):
        raise ValueError("production initialization proof requires its local sealed initial manifest")
    if args.initial_manifest:
        request["provenance"]["evidence"].append({
            "path":str(args.initial_manifest.resolve(strict=True)),
            "sha256":run["initial_manifest_sha256"]})
    for field in request["fields"]:
        field["actual"] = str((args.actual / Path(field["actual"]).name).resolve(strict=True))
    with args.output.open("x") as stream:
        json.dump(request, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
