"""Replay active km_opt=4 exact arithmetic against retained WRF words."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


def _oracle_helper(directory: Path, name: str):
    """One compiled-WRF diffusion helper, loaded by path from ``directory``.

    The helpers live in tools/wrf_diffusion_oracle, which the wheel does not
    carry, so this file names them by the path given as --oracle-tools
    rather than by a bare import the wheel's import scan reads as a
    distribution.  Registered in sys.modules under its own name, so a
    sibling helper's bare import of it is this module.
    """
    path = directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None or not path.is_file():
        raise FileNotFoundError(f"no oracle helper {name}.py under --oracle-tools")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--oracle-tools", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get("GPUWM_WRF_EXACT") != "1" or os.environ.get("GPUWM_WRF_EXACT_DIFFUSION") != "1":
        raise ValueError("select both exact mode and its diffusion substage before import")
    # The helpers import their own siblings (cases, deformation_reference)
    # by bare name, so their directory goes on the path first.
    sys.path.insert(0, str(args.oracle_tools))
    port_outputs = _oracle_helper(args.oracle_tools, "deformation_compare").port_outputs
    compare_case = _oracle_helper(args.oracle_tools, "horizontal_compare").compare_case
    from gpuwm.verify.diffusion_oracle import word_comparison
    from gpuwm.core.kernels import module_source
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=256 * 1024 * 1024)
    args.out.mkdir(parents=True, exist_ok=True)
    deformation_manifest = json.loads((args.fixtures / "deformation-manifest.json").read_text())
    if len(deformation_manifest["cases"]) != 14:
        raise ValueError("complete native diffusion corpus requires fourteen deformation fixtures")
    for case in deformation_manifest["cases"]:
        digest = hashlib.sha256((args.fixtures / case["file"]).read_bytes()).hexdigest()
        if digest != case["sha256"]:
            raise ValueError(f"retained native fixture hash changed: {case['file']}")
    deformation = {}
    for path in sorted(args.fixtures.glob("deformation-*.npz")):
        with np.load(path) as data:
            arrays = {k.removeprefix("input__"): data[k] for k in data.files if k.startswith("input__")}
            meta = json.loads(str(data["meta_json"]))
            actual = port_outputs(arrays, meta, km_opt=4, isotropic=0)
            prefix = "ref__km4_iso0__"
            row = {}
            for key in data.files:
                if key.startswith(prefix):
                    name = key.removeprefix(prefix)
                    row[name] = {**word_comparison(actual[name], data[key]),
                                 "gpu_sha256": hashlib.sha256(actual[name].tobytes()).hexdigest()}
            deformation[path.stem] = row
    manifest = json.loads((args.fixtures / "horizontal/manifest.json").read_text())
    if len(manifest["cases"]) != 14:
        raise ValueError("complete native diffusion corpus requires fourteen horizontal fixtures")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((args.fixtures / "horizontal" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"retained native horizontal fixture hash changed: {name}")
    horizontal = {case["name"]: compare_case(args.fixtures / "horizontal" / case["file"], case)
                  for case in manifest["cases"]}
    cp.cuda.Device().synchronize()
    def summarize(rows):
        fields = {}
        for case in rows.values():
            for name, metric in case.items():
                row = fields.setdefault(name, {"words": 0, "different_words": 0,
                                                "max_ulp": 0, "max_absolute": 0.0})
                row["words"] += metric["words"]
                row["different_words"] += metric["different_words"]
                row["max_ulp"] = max(row["max_ulp"], metric["max_ulp"])
                row["max_absolute"] = max(row["max_absolute"], metric["max_absolute"])
        return fields
    receipt = {"schema": "wrf-exact-active-diffusion-oracle-v1",
               "gpu": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
               "source_sha256": hashlib.sha256(module_source("smag2d").encode()).hexdigest(),
               "deformation": deformation, "horizontal": horizontal,
               "summary": {"deformation": summarize(deformation), "horizontal": summarize(horizontal)}}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["summary"], indent=2))
    compared = [v for fields in receipt["summary"].values() for v in fields.values()]
    if sum(row["words"] for row in compared) != 2_722_902:
        raise ValueError("complete native diffusion corpus word count changed")
    if any(row["different_words"] for row in compared):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
