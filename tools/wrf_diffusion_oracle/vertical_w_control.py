"""Independent WRF driver control for the declared vertical-w xkmv choice.

The compiled WRF source and its driver ABI remain unchanged. Only the
driver's xkmh input is replaced with the prepared xkmv words. The stock
fixtures stay immutable; the other driver outputs must remain stock-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def coefficient_matched_prepared(prepared):
    """Change one caller input, preserving all original coefficient words."""
    controlled = dict(prepared)
    controlled["kmh"] = np.array(prepared["kmv"], dtype=np.float32, order="F", copy=True)
    if controlled["kmh"].shape != prepared["kmh"].shape:
        raise ValueError("vertical-w coefficient layouts differ")
    for name, value in prepared.items():
        if name != "kmh" and controlled[name] is not value:
            raise ValueError(f"the coefficient control changed another input: {name}")
    return controlled


def match_gpu_receipt(proof, gpu):
    """Join complete native array hashes to an already measured GPU capture."""
    rows = {}
    for name, cases in gpu["cases"].items():
        for arm, fields in cases.items():
            for field, value in fields.items():
                key = name + "/" + arm + "/" + field
                native = proof["fields"][key]
                if value["words"] != native["words"]:
                    raise ValueError(f"GPU/native word count differs: {key}")
                rows[key] = {"words": value["words"], "current_gpu_sha256": value["gpu_sha256"],
                             "controlled_native_sha256": native["controlled_native_sha256"],
                             "stock_native_sha256": native["stock_native_sha256"],
                             "current_matches_declared_native": value["gpu_sha256"] == native["controlled_native_sha256"],
                             "current_stock_native_different_words": value["different_words"]}
    if set(rows) != set(proof["fields"]):
        raise ValueError("GPU/native field inventories differ")
    return {"fields": rows, "arrays": len(rows), "words": sum(row["words"] for row in rows.values()),
            "current_stock_native_different_words": sum(row["current_stock_native_different_words"] for row in rows.values()),
            "mismatching_control_arrays": sum(not row["current_matches_declared_native"] for row in rows.values()),
            "all_current_match_declared_native": all(row["current_matches_declared_native"] for row in rows.values())}


def native_control(data, preparation, library):
    # Existing oracle modules use their local sibling imports when run as CLIs.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cases import core, pad2
    from deformation_reference import reference_for_case
    from vertical_driver import FIELDS, reference
    from gpuwm.verify.smallstep_oracle import word_metrics

    data, preparation, library = Path(data), Path(preparation), Path(library)
    manifest = json.loads((data / "manifest.json").read_text())
    rows, inputs = {}, {}
    for entry in manifest["cases"]:
        path = data / entry["file"]
        if sha(path) != manifest["files"][entry["file"]]:
            raise RuntimeError(f"stock fixture identity changed: {path.name}")
        inputs[entry["file"]] = {"sha256": sha(path), "bytes": path.stat().st_size}
        with np.load(path, allow_pickle=False) as fixture:
            arrays = {key.removeprefix("input_"): fixture[key] for key in fixture.files if key.startswith("input_")}
            meta = json.loads(str(fixture["meta_json"]))
            for km in (2, 4):
                prepared = reference_for_case(preparation, arrays, meta, km_opt=km)
                for key in ("ust", "hfx", "qfx"):
                    prepared[key] = pad2(arrays[key], meta["nx"], meta["ny"], meta["bx"], meta["by"])
                for key in ("kmh", "kmv", "khv"):
                    coefficient = core(prepared[key], meta["nx"], meta["ny"], meta["nz"])
                    if not np.array_equal(coefficient.view(np.uint32), fixture[f"km{km}_" + key].view(np.uint32)):
                        raise RuntimeError(f"prepared coefficient differs from immutable fixture: {entry['name']}/{km}/{key}")
                controlled = coefficient_matched_prepared(prepared)
                changed_inputs = [key for key in prepared if array_sha(prepared[key]) != array_sha(controlled[key])]
                if any(key != "kmh" for key in changed_inputs):
                    raise RuntimeError("another WRF driver input changed beside the vertical-w coefficient")
                prepared_hashes = {key: array_sha(value) for key,value in prepared.items()}
                controlled_hashes = {key: array_sha(value) for key,value in controlled.items()}
                for flux in (0, 1, 2):
                    stock = reference(library, prepared, meta, km_opt=km, isfflx=flux)
                    matched = reference(library, controlled, meta, km_opt=km, isfflx=flux)
                    if ({key: array_sha(value) for key,value in prepared.items()} != prepared_hashes
                            or {key: array_sha(value) for key,value in controlled.items()} != controlled_hashes):
                        raise RuntimeError("native call changed a caller input")
                    arm = f"km{km}_flux{flux}"
                    for field in FIELDS:
                        frozen = fixture["wrf_" + arm + "_" + field]
                        stock_metric = word_metrics(stock[field], frozen)
                        control_metric = word_metrics(matched[field], frozen)
                        if stock_metric["different_words"]:
                            raise RuntimeError(f"compiled stock driver does not reproduce immutable fixture: {entry['name']}/{arm}/{field}")
                        if field != "w" and control_metric["different_words"]:
                            raise RuntimeError(f"coefficient control moved a non-W output: {entry['name']}/{arm}/{field}")
                        key = entry["name"] + "/" + arm + "/" + field
                        rows[key] = {"words": int(frozen.size),
                                     "stock_native_sha256": array_sha(frozen),
                                     "controlled_native_sha256": array_sha(matched[field]),
                                     "stock_reproduction": stock_metric, "control_to_stock": control_metric,
                                     "changed_driver_inputs": changed_inputs,
                                     "stock_w_coefficient": "xkmh", "declared_w_coefficient": "xkmv",
                                     "stock_w_coefficient_sha256": array_sha(prepared["kmh"]),
                                     "declared_w_coefficient_sha256": array_sha(prepared["kmv"])}
    root = Path(__file__).resolve().parents[2]
    tool_names = ("tools/wrf_diffusion_oracle/vertical_w_control.py", "tools/wrf_diffusion_oracle/vertical_driver.py",
                  "tools/wrf_diffusion_oracle/vertical_driver_complete.F90", "tools/wrf_diffusion_oracle/deformation_reference.py",
                  "tools/wrf_diffusion_oracle/cases.py", "gpuwm/verify/smallstep_oracle.py", "gpuwm/core/fp32_ulp.py")
    return {"schema": "vertical-declared-xkmv-native-control-v1",
            "engine_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "control": "unmodified compiled WRF vertical driver; only prepared kmh input replaced by copied kmv",
            "case_count": len(manifest["cases"]), "arrays": len(rows),
            "words": sum(row["words"] for row in rows.values()),
            "stock_w_control_different_words": sum(row["control_to_stock"]["different_words"] for row in rows.values()),
            "non_w_control_different_words": sum(row["control_to_stock"]["different_words"] for key,row in rows.items() if not key.endswith("/w")),
            "library_sha256": sha(library), "preparation_library_sha256": sha(preparation),
            "library_build_receipt": json.loads((library.parent / "build-receipt.json").read_text()),
            "preparation_build_receipt": json.loads((preparation.parent / "build-receipt.json").read_text()),
            "stock_manifest_sha256": sha(data / "manifest.json"), "fixtures": inputs,
            "tool_inputs": {name: {"sha256": sha(root / name), "bytes": (root / name).stat().st_size} for name in tool_names},
            "fields": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("preparation", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu-receipt", type=Path)
    args = parser.parse_args()
    proof = native_control(args.data, args.preparation, args.library)
    if args.gpu_receipt:
        proof["gpu_receipt_sha256"] = sha(args.gpu_receipt)
        proof["gpu_comparison"] = match_gpu_receipt(proof, json.loads(args.gpu_receipt.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({key: proof[key] for key in ("case_count", "arrays", "words", "stock_w_control_different_words", "non_w_control_different_words")}))
    return int(bool(args.gpu_receipt) and not proof["gpu_comparison"]["all_current_match_declared_native"])


if __name__ == "__main__":
    raise SystemExit(main())
