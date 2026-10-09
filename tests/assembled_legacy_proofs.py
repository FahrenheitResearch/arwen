"""CPU validation of measured current-source numerical proof receipts."""
import hashlib
import json
from pathlib import Path

PROOF_SHAS = {
    "acoustic-published.json": "eaa4aa22945ca9318d6c78fc602e67078217ceb8ec9a4ecddff3879af21159e5",
    "coordinate-native.json": "cc2b60d79350baef396495621d77909b98d1df1cf9b21eebae930b7d4b9af2e0",
    "coordinate-negative.json": "b34dc1c6bc836e39451f01a4b77031e0d75ac467ca8af842eb6b396e5d5bc653",
    "driver-source.json": "2afbd55a644e549a9b8a0a2a77cdd184423b523065d3156133dbe17b5b48a546",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(root, name):
    path = Path(root) / "tests/data/assembled_legacy_proofs" / name
    assert _sha(path) == PROOF_SHAS[name], name
    return json.loads(path.read_text())


def _identity(proof, modules):
    from gpuwm.core.kernels import module_options, module_source
    capture = proof["capture_identity"]
    assert capture["device"] == "b'NVIDIA RTX PRO 6000 Blackwell Server Edition'"
    assert capture["compute_capability"] == [12, 0]
    assert capture["nvrtc_version"] == [13, 2]
    assert capture["cupy_version"] == "14.2.0"
    assert capture["strict"] is False
    assert capture["default_diffusion"] is True
    for module in modules:
        assert capture["module_source_sha256"][module] == hashlib.sha256(module_source(module).encode()).hexdigest()
        assert capture["module_options"][module] == list(module_options(module))


def _sources(root, proof):
    assert proof["source_inputs"]
    for name, row in proof["source_inputs"].items():
        path = Path(root) / name
        assert path.stat().st_size == row["bytes"], name
        assert _sha(path) == row["sha256"], name


def validate_published_acoustic(root):
    import numpy as np
    from gpuwm.core.kernels import diffusion_kernel, function_options, module_options
    proof = _load(root, "acoustic-published.json")
    assert proof["schema"] == "assembled-published-acoustic-word-proof-v1"
    assert (proof["array_comparisons"], proof["words"], proof["changed_published_words"]) == (132, 305632, 0)
    assert proof["published_exact"] is True
    assert proof["default_acoustic_diffusion_route"] is False
    assert diffusion_kernel("acoustic", "advance_w_phi") is False
    for entry in ("calc_coefs", "advance_w_phi", "advance_w_phi_msf"):
        assert proof["acoustic_function_options"][entry] == list(function_options("acoustic", entry, module_options("acoustic"))) == ["-std=c++17"]
    _identity(proof, ("acoustic",))
    _sources(root, proof)
    assert {"gpuwm/core/kernels/acoustic.cu", "gpuwm/verify/smallstep_vertical_oracle.py",
            "tools/smallstep_wrf471_oracle/vertical_workspace.py"} <= set(proof["source_inputs"])
    path = Path(root) / "tests/data/wrf471_smallstep/vertical-reference.npz"
    assert _sha(path) == proof["recorded_fixture_sha256"]
    with np.load(path, allow_pickle=False) as frozen:
        expected = {key[:-7].replace("__", "/"): key for key in frozen.files if key.endswith("_native")}
        assert set(proof["fields"]) == set(expected)
        for name, key in expected.items():
            row, array = proof["fields"][name], frozen[key]
            assert row["different_words"] == 0 and row["all_uint32_words_equal_frozen"] is True
            assert row["words"] == array.size and array.dtype == np.float32
            assert row["actual_sha256"] == row["recorded_sha256"] == hashlib.sha256(array.tobytes()).hexdigest()
    return proof


def validate_coordinate(root):
    import numpy as np
    proof = _load(root, "coordinate-native.json")
    assert proof["schema"] == "assembled-coordinate-native-word-proof-v1"
    assert (proof["case_count"], proof["field_count"], proof["words"], proof["native_different_words"]) == (88, 272, 178128, 0)
    assert proof["native_exact"] is True
    assert (proof["prior_gpu_different_words"], proof["previously_native_equal_words"],
            proof["changed_previously_native_equal_words"]) == (25435, 152693, 0)
    _identity(proof, ("smag2d", "diff_opt1"))
    _sources(root, proof)
    data = Path(root) / "tests/data/wrf471_diff_opt1"
    assert _sha(data / "wrf471.npz") == proof["native_archive_sha256"]
    assert _sha(data / "wrf471.json") == proof["native_case_schema_sha256"]
    assert _sha(data / "merged-gpu.npz") == proof["prior_gpu_archive_sha256"]
    cases = json.loads((data / "wrf471.json").read_text())["cases"]
    families = {"horizontal": ("tendency",), "km4": ("km", "kh"), "deform": ("d11", "d22", "d12"),
                "km2": ("km", "kh", "kmv", "khv", "tke", "bn2")}
    expected = {case["name"] + "_" + field: case["name"] + ("_expected" if case["family"] == "horizontal" else "_expected_" + field)
                for case in cases for field in families[case["family"]]}
    assert set(proof["fields"]) == set(expected)
    with np.load(data / "wrf471.npz", allow_pickle=False) as native, np.load(data / "merged-gpu.npz", allow_pickle=False) as prior:
        for name, native_name in expected.items():
            row, reference, old = proof["fields"][name], native[native_name], prior[name]
            assert reference.dtype == old.dtype == np.float32 and reference.shape == old.shape
            assert row["current_to_native"]["words"] == reference.size
            assert row["current_to_native"]["different_words"] == 0
            assert row["current_sha256"] == row["native_sha256"] == hashlib.sha256(reference.tobytes()).hexdigest()
            assert row["prior_gpu_sha256"] == hashlib.sha256(old.tobytes()).hexdigest()
            equal = int(np.count_nonzero(old.view(np.uint32) == reference.view(np.uint32)))
            assert row["previously_native_equal_words"] == equal
            assert row["changed_previously_native_equal_words"] == 0
    return proof


def _word_rows(value, prefix=""):
    if isinstance(value, dict) and "gpu_sha256" in value:
        return {prefix: value}
    rows = {}
    for name, child in value.items():
        rows.update(_word_rows(child, prefix + "/" + name if prefix else name))
    return rows


def validate_drivers(root):
    root = Path(root)
    proof = _load(root, "driver-source.json")
    assert proof["schema"] == "assembled-driver-native-word-proof-v1"
    _identity(proof, ("smag2d",))
    _sources(root, proof)
    folder = root / "tests/data/assembled_legacy_proofs"
    artifacts = {}
    for name, digest in proof["artifacts"].items():
        assert _sha(folder / name) == digest, name
        artifacts[name] = json.loads((folder / name).read_text())
    native = artifacts["vertical-native-control.json"]
    assert native["schema"] == "vertical-declared-xkmv-native-control-v1"
    assert (native["case_count"], native["arrays"], native["words"],
            native["stock_w_control_different_words"], native["non_w_control_different_words"]) == (14, 1764, 21500976, 143424, 0)
    _sources(root, {"source_inputs": native["tool_inputs"]})
    for name, row in native["fixtures"].items():
        path = root / "tests/data/wrf471_diffusion/vertical-driver" / name
        assert path.stat().st_size == row["bytes"] and _sha(path) == row["sha256"]
    manifest = root / "tests/data/wrf471_diffusion/vertical-driver/manifest.json"
    assert _sha(manifest) == native["stock_manifest_sha256"]
    for receipt in (native["library_build_receipt"], native["preparation_build_receipt"]):
        assert receipt["wrf_commit"] == "f52c197ed39d12e087d02c50f412d90d418f6186"
        assert "-ffp-contract=off" in receipt["commands"][0] and "-fcheck=bounds" in receipt["commands"][0]
    totals = {"horizontal-driver": (364, 5575696), "km-mutations": (490, 4609920),
              "vertical-driver": (1764, 21500976)}
    answer = {}
    for family, (arrays, words) in totals.items():
        pin = artifacts[family + ".json"]
        assert pin["kernel_source_sha256"] == proof["capture_identity"]["module_source_sha256"]["smag2d"]
        assert pin["device"] == proof["capture_identity"]["device"]
        rows = _word_rows(pin["cases"])
        assert len(rows) == arrays and sum(row["words"] for row in rows.values()) == words
        if family != "vertical-driver":
            assert all(row["different_words"] == row["max_ulp"] == 0 for row in rows.values())
        else:
            assert set(rows) == set(native["fields"])
            stock_differences = 0
            for key, row in rows.items():
                reference = native["fields"][key]
                assert row["words"] == reference["words"]
                assert row["gpu_sha256"] == reference["controlled_native_sha256"], key
                assert reference["stock_reproduction"]["different_words"] == 0
                # Isotropic packets already have identical kmh/kmv words.
                # Require the exact byte-change inventory, including that
                # zero-change control, rather than claiming a change there.
                changed = ([] if reference["stock_w_coefficient_sha256"] ==
                           reference["declared_w_coefficient_sha256"] else ["kmh"])
                assert reference["changed_driver_inputs"] == changed
                assert (reference["stock_w_coefficient"], reference["declared_w_coefficient"]) == ("xkmh", "xkmv")
                assert row["different_words"] == reference["control_to_stock"]["different_words"]
                stock_differences += row["different_words"]
                if key.rsplit("/", 1)[1] != "w":
                    assert row["different_words"] == 0
                    assert reference["stock_native_sha256"] == reference["controlled_native_sha256"]
            assert stock_differences == 143424
        answer[family] = pin
    return answer


def validate_coordinate_negative(root):
    import ast
    from gpuwm.core.kernels import module_source
    positive = validate_coordinate(root)
    proof = _load(root, "coordinate-negative.json")
    assert proof["schema"] == "current-coordinate-donor-negative-control-v1"
    assert proof["positive_proof_sha256"] == PROOF_SHAS["coordinate-native.json"]
    assert (proof["case_count"], proof["field_count"], proof["words"], proof["changed_native_words"]) == (40, 216, 139968, 492)
    assert proof["native_archive_sha256"] == positive["native_archive_sha256"]
    assert proof["capture_identity"] == positive["capture_identity"]
    assert proof["control_device"] == ast.literal_eval(positive["capture_identity"]["device"]).decode()
    assert proof["source_inputs"] == positive["source_inputs"]
    capture = Path(root) / "tools/wrf_diffopt1_oracle/capture.py"
    assert _sha(capture) == proof["capture_tool_sha256"]
    assert _sha(Path(root) / "tools/coordinate_donor_negative_control.py") == proof["measurement_tool_sha256"]
    tree = ast.parse(capture.read_text())
    donor = next(ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "donor" for target in node.targets))
    production = module_source("smag2d")
    assert production.count(donor) == 1
    assert proof["controlled_module_source_sha256"]["smag2d"] == hashlib.sha256(production.replace(donor, "").encode()).hexdigest()
    assert proof["controlled_module_source_sha256"]["diff_opt1"] == positive["capture_identity"]["module_source_sha256"]["diff_opt1"]
    expected = {name for name, row in positive["fields"].items() if row["family"] in ("km2", "deform")}
    assert set(proof["fields"]) == expected
    assert proof["changed_fields"] == {"deform_1_d12": 54, "deform_2_d12": 72, "deform_3_d12": 120,
                                      "deform_5_d12": 54, "deform_6_d12": 72, "deform_7_d12": 120}
    for name, row in proof["fields"].items():
        assert row["native_sha256"] == positive["fields"][name]["native_sha256"]
        assert row["words"] == positive["fields"][name]["current_to_native"]["words"]
        assert row["different_words"] == proof["changed_fields"].get(name, 0)
        if name not in proof["changed_fields"]:
            assert row["control_sha256"] == row["native_sha256"]
    return proof
