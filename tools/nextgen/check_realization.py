#!/usr/bin/env python3
"""Check table metadata against exact pinned native configuration files.

This is a configuration comparison, never numerical BMI/GPU replay evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

from gpuwm.hydro.config import materialize_reference, resolve_configuration


def leaves(value):
    if isinstance(value, dict):
        return sum(leaves(item) for item in value.values())
    if isinstance(value, list):
        return sum(leaves(item) for item in value)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("realization")
    parser.add_argument("--source", action="append", required=True,
                        help="source repository/identity=local root, repeatable")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--bindings", type=Path,
                        help="JSON object of portable reference path bindings")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = dict(value.split("=", 1) for value in args.source)
    started = time.perf_counter()
    selected = resolve_configuration(args.realization, table=args.table)
    row = selected.as_dict()
    reference = (row["reference"] if args.bindings is None else
                 materialize_reference(args.realization, json.loads(args.bindings.read_text()),
                                       table=args.table))
    checks = []
    for source in row["sources"]:
        raw = (Path(roots[source["repository"]]) / source["path"]).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if source["path"].endswith(".json"):
            parsed = json.loads(raw)
            key = "realization"
        else:
            import yaml  # Reference-tool configuration metadata only.
            parsed = yaml.safe_load(raw)
            key = "routing"
        actual = reference[source.get("reference_key", key)]
        checks.append({"source": source, "actual_sha256": sha,
                       "hash_equal": sha == source["sha256"],
                       "metadata_equal": json.dumps(actual, sort_keys=True, allow_nan=False) ==
                       json.dumps(parsed, sort_keys=True, allow_nan=False),
                       "leaf_values": leaves(parsed)})
    passed = all(c["hash_equal"] and c["metadata_equal"] for c in checks)
    result = {"kind": "configuration-metadata-only", "realization": args.realization,
              "row_sha256": selected.sha256,
              "checks": checks, "passed": passed,
              "wall_seconds": time.perf_counter() - started,
              "parallelism": "one setup process, no numerical field work",
              "module_replay": "not performed", "gpu_step_timing": "not measured"}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
