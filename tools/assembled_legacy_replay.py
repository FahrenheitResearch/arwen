"""Run maintained captures; accept only unchanged frozen output words."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def word_hashes(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        if "gpu_sha256" in value:
            result[prefix] = {key: value[key] for key in (
                "gpu_sha256", "words", "different_words", "max_ulp",
                "max_absolute", "nonfinite_words") if key in value}
        for key, child in value.items():
            result.update(word_hashes(child, prefix + "/" + key))
    return result


def coordinate_native_proof(data, prefix):
    """Compare the complete capture with immutable WRF and prior GPU arrays."""
    import numpy as np
    from gpuwm.verify.smallstep_oracle import word_metrics

    data, prefix = Path(data), Path(prefix)
    original = json.loads((data / "merged-gpu.json").read_text())
    captured = json.loads(prefix.with_suffix(".json").read_text())
    native_sha = sha(data / "wrf471.npz")
    if original["native_archive_sha256"] != native_sha or captured["native_archive_sha256"] != native_sha:
        raise RuntimeError("native WRF archive identity changed")
    if original["archive_sha256"] != sha(data / "merged-gpu.npz"):
        raise RuntimeError("prior GPU archive identity changed")
    if captured["archive_sha256"] != sha(prefix.with_suffix(".npz")):
        raise RuntimeError("current GPU archive identity changed")
    cases = json.loads((data / "wrf471.json").read_text())["cases"]
    fields_by_family = {"horizontal": ("tendency",), "km4": ("km", "kh"),
                       "deform": ("d11", "d22", "d12"),
                       "km2": ("km", "kh", "kmv", "khv", "tke", "bn2")}
    expected_cases = {(case["name"], case["family"]) for case in cases}
    if (len(expected_cases) != len(cases) or len(captured["cases"]) != len(cases)
            or {(row["name"], row["family"]) for row in captured["cases"]} != expected_cases):
        raise RuntimeError("capture case inventory differs from native schema")
    rows = {}
    with np.load(data / "wrf471.npz", allow_pickle=False) as native, \
            np.load(data / "merged-gpu.npz", allow_pickle=False) as old, \
            np.load(prefix.with_suffix(".npz"), allow_pickle=False) as current:
        for case in cases:
            for field in fields_by_family[case["family"]]:
                name = case["name"] + "_" + field
                native_name = case["name"] + ("_expected" if case["family"] == "horizontal" else "_expected_" + field)
                actual, reference, prior = current[name], native[native_name], old[name]
                if any(value.dtype != np.float32 or value.shape != reference.shape
                       for value in (actual, reference, prior)):
                    raise RuntimeError(f"coordinate array layout differs: {name}")
                rows[name] = {"family": case["family"], "field": field,
                              "current_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                              "native_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                              "prior_gpu_sha256": hashlib.sha256(prior.tobytes()).hexdigest(),
                              "current_to_native": word_metrics(actual, reference),
                              "prior_gpu_to_native": word_metrics(prior, reference),
                              "current_to_prior_gpu": word_metrics(actual, prior)}
                previously_equal = prior.view(np.uint32) == reference.view(np.uint32)
                rows[name]["previously_native_equal_words"] = int(np.count_nonzero(previously_equal))
                rows[name]["changed_previously_native_equal_words"] = int(np.count_nonzero(
                    previously_equal & (actual.view(np.uint32) != prior.view(np.uint32))))
                if case["family"] == "km2" and field in ("km", "kh", "tke"):
                    rows[name]["prior_5_ulp_budget_report_only"] = rows[name]["current_to_native"]["max_ulp"] <= 5
        if set(rows) != set(current.files) or set(rows) != set(old.files):
            raise RuntimeError("coordinate output inventory differs from the original archive")
    return {"schema": "assembled-coordinate-native-word-proof-v1",
            "native_archive_sha256": native_sha,
            "native_case_schema_sha256": sha(data / "wrf471.json"),
            "prior_gpu_archive_sha256": sha(data / "merged-gpu.npz"),
            "current_gpu_archive_sha256": sha(prefix.with_suffix(".npz")),
            "case_count": len(cases), "field_count": len(rows),
            "words": sum(row["current_to_native"]["words"] for row in rows.values()),
            "native_different_words": sum(row["current_to_native"]["different_words"] for row in rows.values()),
            "prior_gpu_different_words": sum(row["current_to_prior_gpu"]["different_words"] for row in rows.values()),
            "previously_native_equal_words": sum(row["previously_native_equal_words"] for row in rows.values()),
            "changed_previously_native_equal_words": sum(row["changed_previously_native_equal_words"] for row in rows.values()),
            "native_exact": all(row["current_to_native"]["different_words"] == 0 for row in rows.values()),
            "fields": rows}


def acoustic_native_rows(current, frozen, case):
    """Compare actual acoustic outputs with WRF, retaining supplied c2a."""
    import numpy as np
    from gpuwm.verify.smallstep_oracle import word_metrics

    rows = {}
    for field, value in current.items():
        wrf_key, prior_key = case + "__" + field + "_wrf", case + "__" + field + "_native"
        if wrf_key in frozen:
            reference, kind = frozen[wrf_key], "compiled_WRF_output"
        elif field == "c2a" and prior_key in frozen:
            reference, kind = frozen[prior_key], "retained_supplied_c2a_input"
        else:
            raise RuntimeError(f"independent acoustic reference absent: {case}/{field}")
        if value.dtype != np.float32 or reference.dtype != np.float32 or value.shape != reference.shape:
            raise RuntimeError(f"acoustic reference layout differs: {case}/{field}")
        row = {"reference_kind": kind, "current_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
               "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
               "current_to_reference": word_metrics(value, reference)}
        if prior_key in frozen:
            prior = frozen[prior_key]
            if prior.dtype != np.float32 or prior.shape != reference.shape:
                raise RuntimeError(f"prior acoustic layout differs: {case}/{field}")
            equal = prior.view(np.uint32) == reference.view(np.uint32)
            row.update(prior_gpu_sha256=hashlib.sha256(prior.tobytes()).hexdigest(),
                       prior_gpu_to_reference=word_metrics(prior, reference),
                       current_to_prior_gpu=word_metrics(value, prior),
                       previously_reference_equal_words=int(np.count_nonzero(equal)),
                       changed_previously_reference_equal_words=int(np.count_nonzero(
                           equal & (value.view(np.uint32) != prior.view(np.uint32)))))
        rows[case + "/" + field] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("tip")
    parser.add_argument("--family", choices=("drivers", "coordinate", "acoustic"), required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tip = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if tip != args.tip:
        raise SystemExit(f"source tip mismatch: {tip} versus {args.tip}")
    sys.path.insert(0, str(root))
    import cupy as cp
    import numpy as np
    from gpuwm import wrf_exact
    from gpuwm.core.kernels import diffusion_kernel, function_options, module_options, module_source
    if wrf_exact.ENABLED:
        raise SystemExit("default capture requires GPUWM_WRF_EXACT=0")
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    if int(device["major"]) < 10:
        raise SystemExit("legacy word certification requires a Blackwell or newer card")
    audit = {
        "engine_commit": tip, "family": args.family,
        "device": str(device["name"]), "compute_capability": [device["major"], device["minor"]],
        "nvrtc_version": list(cp.cuda.nvrtc.getVersion()), "cupy_version": cp.__version__,
        "module_source_sha256": {name: hashlib.sha256(module_source(name).encode()).hexdigest()
                                 for name in ("smag2d", "diff_opt1", "acoustic")},
        "module_options": {name: list(module_options(name)) for name in ("smag2d", "diff_opt1", "acoustic")},
        "acoustic_function_options": {name: list(function_options("acoustic", name, module_options("acoustic")))
                                      for name in ("calc_coefs", "advance_w_phi", "advance_w_phi_msf")},
        "default_acoustic_diffusion_route": diffusion_kernel("acoustic", "advance_w_phi"),
        "strict": wrf_exact.ENABLED, "default_diffusion": wrf_exact.DIFFUSION_ENABLED,
        "status": "running",
    }
    manifest = root.parent / "MANIFEST.txt"
    with manifest.open("a") as inventory:
        inventory.write(str(out) + "\n")
        inventory.write(str(out / "source-audit.json") + "\n")
    write(out / "source-audit.json", audit)
    try:
        if args.family == "drivers":
            data = root / "tests/data/wrf471_diffusion"
            tools = root / "tools/wrf_diffusion_oracle"
            calls = (
                ("horizontal-driver", [str(tools / "horizontal_driver.py"), "compare", str(data / "horizontal-driver")]),
                ("vertical-driver", [str(tools / "vertical_driver.py"), "compare", str(data / "vertical-driver")]),
                ("km-mutations", [str(tools / "deformation_mutations.py"), "compare", str(data), str(data / "km-mutations")]),
            )
            rows = {}
            for name, command in calls:
                target = out / (name + ".json")
                with (out / (name + ".log")).open("w") as log:
                    process = subprocess.run([sys.executable, *command, str(target)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    rows[name] = {"exit_code": process.returncode, "pass": False}
                    continue
                original_path = data / name / "gpu-receipt.json"
                original = word_hashes(json.loads(original_path.read_text())["cases"])
                current = word_hashes(json.loads(target.read_text())["cases"])
                changed = {key: {"prior": original.get(key), "current": current.get(key)}
                           for key in sorted(set(original) | set(current))
                           if key not in original or key not in current
                           or original[key]["gpu_sha256"] != current[key]["gpu_sha256"]
                           or original[key].get("words") != current[key].get("words")}
                rows[name] = {"original_receipt_sha256": sha(original_path),
                              "capture_sha256": sha(target), "arrays": len(current),
                              "changed_arrays": len(changed), "changes": changed,
                              "pass": bool(current) and not changed}
            audit["comparisons"] = rows
            audit["pass"] = all(row["pass"] for row in rows.values())
        elif args.family == "coordinate":
            data = root / "tests/data/wrf471_diff_opt1"
            prefix = out / "coordinate-current"
            with (out / "coordinate-capture.log").open("w") as log:
                process = subprocess.run([sys.executable, str(root / "tools/wrf_diffopt1_oracle/capture.py"),
                                          str(data), str(prefix), "--mode", "all"], cwd=root,
                                         stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"coordinate capture failed with exit {process.returncode}")
            native_proof = coordinate_native_proof(data, prefix)
            proof_inputs = (
                "gpuwm/core/kernels/smag2d.cu", "gpuwm/core/kernels/diff_opt1.cu",
                "gpuwm/core/kernels/common.cuh", "gpuwm/core/kernels/__init__.py",
                "gpuwm/core/constants.py", "gpuwm/core/dycore.py",
                "gpuwm/nvrtc_ptx_cache.py", "gpuwm/wrf_exact.py",
                "gpuwm/verify/smallstep_oracle.py", "gpuwm/core/fp32_ulp.py",
                "tools/assembled_legacy_replay.py", "tools/wrf_diffopt1_oracle/capture.py",
                "tools/wrf_diffopt1_oracle/model_case.py", "tools/wrf_diffopt1_oracle/source_transition.py",
                "tests/data/wrf471_diff_opt1/wrf471.npz", "tests/data/wrf471_diff_opt1/wrf471.json",
                "tests/data/wrf471_diff_opt1/merged-gpu.npz", "tests/data/wrf471_diff_opt1/merged-gpu.json")
            native_proof["source_inputs"] = {name: {"sha256": sha(root / name), "bytes": (root / name).stat().st_size}
                                             for name in proof_inputs}
            native_proof["capture_identity"] = {key: audit[key] for key in (
                "engine_commit", "device", "compute_capability", "nvrtc_version",
                "cupy_version", "module_source_sha256", "module_options", "strict", "default_diffusion")}
            write(out / "direct-native-proof.json", native_proof)
            audit["native_proof"] = {key: native_proof[key] for key in (
                "case_count", "field_count", "words", "native_different_words",
                "prior_gpu_different_words", "previously_native_equal_words",
                "changed_previously_native_equal_words", "native_exact")}
            spec = importlib.util.spec_from_file_location("owned_source_transition", root / "tools/wrf_diffopt1_oracle/source_transition.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Call the maintained refusal logic without changing any source fixture.
            transition = module.transition(data, prefix, root)
            write(out / "measured-source-transition.json", transition)
            audit["comparisons"] = {key: transition[key] for key in (
                "case_count", "field_count", "words", "different_words", "forecast_runs")}
            audit["pass"] = transition["different_words"] == 0
        else:
            from gpuwm.verify.smallstep_vertical_oracle import vertical_cases, vertical_port_outputs
            data = root / "tests/data/wrf471_smallstep/vertical-reference.npz"
            rows, native_rows = {}, {}
            with np.load(data, allow_pickle=False) as frozen:
                for name, raw, metadata in vertical_cases():
                    got = vertical_port_outputs(raw, metadata, reference_inputs={
                        key: frozen[name + "__" + key + "_input"] for key in ("history_p", "history_old")})
                    native_rows.update(acoustic_native_rows(got, frozen, name))
                    for key, current in got.items():
                        reference_key = name + "__" + key + "_native"
                        if key == "ph_diag":
                            # The maintained native comparison excludes ph_diag:
                            # the archive records only its WRF reference, not native output.
                            if reference_key in frozen:
                                raise RuntimeError("native ph_diag inventory changed")
                            continue
                        reference = frozen[reference_key]
                        if current.shape != reference.shape or current.dtype != reference.dtype:
                            raise RuntimeError(f"native array layout changed: {name}/{key}")
                        different = int(np.count_nonzero(current.view(np.uint32) != reference.view(np.uint32)))
                        rows[name + "/" + key] = {"words": int(current.size), "different_words": different,
                            "actual_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
                            "recorded_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                            "all_uint32_words_equal_frozen": different == 0}
                    print(name, flush=True)
            prior = json.loads((root / "tools/smallstep_wrf471_oracle/receipts/rtx5090-bw-acoustic-word-reproduction.json").read_text())
            if set(rows) != set(prior["arrays"]):
                raise RuntimeError("acoustic frozen output inventory differs from the prior reproduction")
            audit.update(recorded_fixture_sha256=sha(data), array_comparisons=len(rows), arrays=rows,
                         output_words=sum(row["words"] for row in rows.values()),
                         different_words=sum(row["different_words"] for row in rows.values()))
            native_proof = {
                "schema": "assembled-acoustic-native-word-proof-v1", "recorded_fixture_sha256": sha(data),
                "array_comparisons": len(native_rows), "prior_gpu_array_comparisons": len(rows),
                "words": sum(row["current_to_reference"]["words"] for row in native_rows.values()),
                "wrf_different_words": sum(row["current_to_reference"]["different_words"] for row in native_rows.values()
                                           if row["reference_kind"] == "compiled_WRF_output"),
                "supplied_c2a_different_words": sum(row["current_to_reference"]["different_words"] for row in native_rows.values()
                                                   if row["reference_kind"] == "retained_supplied_c2a_input"),
                "previously_reference_equal_words": sum(row.get("previously_reference_equal_words", 0) for row in native_rows.values()),
                "changed_previously_reference_equal_words": sum(row.get("changed_previously_reference_equal_words", 0) for row in native_rows.values()),
                "native_exact": all(row["current_to_reference"]["different_words"] == 0 for row in native_rows.values()),
                "capture_identity": {key: audit[key] for key in (
                    "engine_commit", "device", "compute_capability", "nvrtc_version", "cupy_version",
                    "module_source_sha256", "module_options", "strict", "default_diffusion")},
                "source_inputs": {name: {"sha256": sha(root / name), "bytes": (root / name).stat().st_size} for name in (
                    "gpuwm/core/kernels/acoustic.cu", "gpuwm/core/kernels/common.cuh", "gpuwm/core/kernels/__init__.py",
                    "gpuwm/core/constants.py", "gpuwm/core/acoustic.py", "gpuwm/verify/smallstep_vertical_oracle.py",
                    "gpuwm/verify/smallstep_oracle.py", "gpuwm/verify/default_kernel_source.py", "gpuwm/core/fp32_ulp.py",
                    "gpuwm/wrf_exact.py", "gpuwm/nvrtc_ptx_cache.py", "tools/assembled_legacy_replay.py",
                    "tools/smallstep_wrf471_oracle/vertical_workspace.py", "tests/data/wrf471_smallstep/vertical-reference.npz",
                    "tests/data/wrf471_smallstep/cases.json", "tests/data/wrf471_smallstep/real-state.npz")},
                "fields": native_rows}
            write(out / "acoustic-direct-native-proof.json", native_proof)
            audit["native_proof"] = {key: native_proof[key] for key in (
                "array_comparisons", "prior_gpu_array_comparisons", "words", "wrf_different_words",
                "supplied_c2a_different_words", "previously_reference_equal_words",
                "changed_previously_reference_equal_words", "native_exact")}
            published_proof = {
                "schema": "assembled-published-acoustic-word-proof-v1",
                "recorded_fixture_sha256": sha(data), "array_comparisons": len(rows),
                "words": audit["output_words"], "changed_published_words": audit["different_words"],
                "published_exact": audit["different_words"] == 0,
                "default_acoustic_diffusion_route": audit["default_acoustic_diffusion_route"],
                "acoustic_function_options": audit["acoustic_function_options"],
                "capture_identity": native_proof["capture_identity"],
                "source_inputs": native_proof["source_inputs"], "fields": rows,
                "scope": "published default GPU identity; WRF differences are reported separately"}
            write(out / "acoustic-published-default-proof.json", published_proof)
            audit["pass"] = audit["different_words"] == 0
        audit["status"] = "complete"
    except BaseException as error:
        audit.update(status="refused", error=f"{type(error).__name__}: {error}", **{"pass": False})
    audit["source_fingerprint_unchanged"] = audit["module_source_sha256"] == {
        name: hashlib.sha256(module_source(name).encode()).hexdigest()
        for name in audit["module_source_sha256"]}
    if not audit["source_fingerprint_unchanged"]:
        audit.update(status="refused", error="kernel source changed during capture", **{"pass": False})
    raw_capture = out / "coordinate-current.npz"
    if raw_capture.exists():
        audit["raw_capture_deleted_after_receipt"] = {
            "path": str(raw_capture), "bytes": raw_capture.stat().st_size,
            "sha256": sha(raw_capture)}
    with manifest.open("a") as inventory:
        for path in sorted(out.rglob("*")):
            inventory.write(str(path) + "\n")
    write(out / "source-audit.json", audit)
    if raw_capture.exists():
        raw_capture.unlink()
    print(json.dumps({key: audit[key] for key in ("family", "engine_commit", "status", "pass")}), flush=True)
    return int(not audit["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
