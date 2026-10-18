"""Reproduce frozen diffusion words after compiler-cache orchestration changes.

Run under GPU ownership with GPUWM_NUMERICS=strict. Original numerical
receipts remain immutable; this tool emits only the measured source extension.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root, scratch = args.root.resolve(), args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    if os.environ.get("GPUWM_NUMERICS") != "strict":
        raise SystemExit("receipt requires GPUWM_NUMERICS=strict")
    sys.path[:0] = [str(root), str(root / "tests"),
                    str(root / "tools/wrf_diffusion_oracle")]
    import cupy as cp
    from assembled_legacy_proofs import _load, _word_rows
    from gpuwm.core.kernels import module_options, module_source
    from tools.assembled_legacy_replay import coordinate_native_proof

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    assert props["major"] >= 10, "word certification requires Blackwell or newer"
    folder = root / "tests/data/assembled_legacy_proofs"
    driver = _load(root, "driver-source.json")
    coordinate = _load(root, "coordinate-native.json")
    rows = {}
    for family, module in (("horizontal-driver", "horizontal_driver"),
                           ("vertical-driver", "vertical_driver"),
                           ("km-mutations", "deformation_mutations")):
        path = folder / (family + ".json")
        assert sha(path) == driver["artifacts"][path.name]
        frozen = json.loads(path.read_text())["cases"]
        data = root / "tests/data/wrf471_diffusion"
        manifest = json.loads((data / family / "manifest.json").read_text())
        compare = importlib.import_module(module).compare_case
        actual = {}
        for case in manifest["cases"]:
            key = case["file"] if family == "km-mutations" else case["name"]
            args_ = (data / case["file"], data / family / case["file"]) if family == "km-mutations" else (data / family / case["file"],)
            actual[key] = compare(*args_)
            assert actual[key] == frozen[key], (family, key, "frozen words changed")
        assert actual.keys() == frozen.keys()
        rows[family] = {key: {"words": row["words"], "gpu_sha256": row["gpu_sha256"]}
                        for key, row in _word_rows(actual).items()}
        print(family, len(rows[family]), sum(row["words"] for row in rows[family].values()), "unchanged words", flush=True)

    prefix = scratch / "coordinate-current"
    subprocess.run([sys.executable, str(root / "tools/wrf_diffopt1_oracle/capture.py"),
                    str(root / "tests/data/wrf471_diff_opt1"), str(prefix),
                    "--mode", "all"], check=True)
    measured = coordinate_native_proof(root / "tests/data/wrf471_diff_opt1", prefix)
    assert measured["native_exact"] and measured["native_different_words"] == 0
    assert measured["fields"] == coordinate["fields"], "coordinate words changed"
    rows["coordinate"] = {key: {"words": row["current_to_native"]["words"],
                                "gpu_sha256": row["current_sha256"]}
                          for key, row in measured["fields"].items()}
    cache = "gpuwm/nvrtc_ptx_cache.py"
    result = {
        "schema": "diffusion-cache-source-extension-v1",
        "generator_sha256": sha(__file__), "numerics": "strict",
        "device": str(props["name"]), "nvrtc_version": list(cp.cuda.nvrtc.getVersion()),
        "cupy_version": cp.__version__,
        "module_source_sha256": {name: hashlib.sha256(module_source(name).encode()).hexdigest()
                                  for name in ("smag2d", "diff_opt1")},
        "module_options": {name: list(module_options(name)) for name in ("smag2d", "diff_opt1")},
        "source": cache, "before": coordinate["source_inputs"][cache],
        "after": {"bytes": (root / cache).stat().st_size, "sha256": sha(root / cache)},
        "fields": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("coordinate", len(rows["coordinate"]), measured["words"], "unchanged words", flush=True)
    print("receipt", args.output, sha(args.output), flush=True)


if __name__ == "__main__":
    main()
