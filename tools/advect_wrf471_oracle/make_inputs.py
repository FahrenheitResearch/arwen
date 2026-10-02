"""Package native Rust fixture extraction as the oracle's input NPZ files.

NetCDF decoding, cropping, mass coupling and case transformations are performed
by extract_inputs.rs. Python only launches that executable and serializes its
already constructed float32 arrays. The source is copied before this tool runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--rustc", default="rustc")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    source_hash = sha256(args.source)
    extractor_source = Path(__file__).with_name("extract_inputs.rs")
    if not args.no_build:
        args.extractor.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [args.rustc, "--edition=2024", "-O", str(extractor_source),
             "-o", str(args.extractor)], check=True,
        )
    subprocess.run(
        [str(args.extractor.resolve()), str(args.source),
         str(args.raw_directory), source_hash], check=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for description in sorted(args.raw_directory.glob("*/arrays.json")):
        case = json.loads(description.read_text())
        arrays = {}
        array_hashes = {}
        for key, definition in case["arrays"].items():
            path = description.parent / definition["file"]
            shape = tuple(definition["shape"])
            arrays[key] = np.fromfile(path, dtype="<f4").reshape(shape)
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"nonfinite source input {key}")
            array_hashes[key] = sha256(path)
        name = case["metadata"]["case_id"]
        target = args.output / f"{name}.npz"
        np.savez_compressed(target, **arrays)
        cases.append({
            "name": name, "file": target.name, "metadata": case["metadata"],
            "input_sha256": sha256(target), "array_sha256": array_hashes,
        })
    manifest = {
        "schema_version": 1,
        "source_file": args.source.name,
        "source_sha256": source_hash,
        "source_size_bytes": args.source.stat().st_size,
        "extractor_source_sha256": sha256(extractor_source),
        "array_order": "float32 little-endian, C-order (z,y,x); natural staggering",
        "cases": cases,
    }
    (args.output / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packaged {len(cases)} input cases from {args.source.name}, SHA256 {source_hash}")


if __name__ == "__main__":
    main()
