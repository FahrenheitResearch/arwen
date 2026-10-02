"""Package already-cropped Rust FP32 binary arrays without numeric changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fields = {}
    for row in (args.input_dir / "shapes.tsv").read_text().splitlines():
        name, shape = row.split("\t")
        fields[name] = np.frombuffer((args.input_dir / f"{name}.f32").read_bytes(), dtype="<f4").reshape(
            tuple(map(int, shape.split(",")))
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    metadata = {
        "schema": "bigstep-wrf471-state-v1",
        "file": args.output.name,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "fields": {name: {"shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                   for name, value in sorted(fields.items())},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
