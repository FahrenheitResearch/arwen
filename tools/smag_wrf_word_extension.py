"""Record current Smag answers only for the merged compiled-WRF TKE fixes.

Use compiled cuda_cross_target_replay artifacts under GPU ownership. Every
input and every unaffected kernel must still match the original fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.cuda_cross_target_replay import load_modules, replay_scenario

CORRECTED = {"wrf_tke_rhs", "wrf_calc_n2", "wrf_tke_km"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import cupy as cp
    assert int(cp.cuda.Device().compute_capability) >= 100
    modules, compile_receipt = load_modules(args.artifacts)
    original = json.loads(args.original.read_text())
    cases = []
    corrected = set()
    for case in original["scenarios"]:
        rows = replay_scenario(modules, tuple(case["scenario"]))
        assert not any(any(row["different_words"]) for row in rows), "compiler targets differ"
        actual = {(row["variant"], row["kernel"]): row for row in rows
                  if row["comparison"] == "compiler_target" and row["compute_target"] == 120}
        expected = {(row["variant"], row["kernel"]): row for row in case["kernels"]}
        assert actual.keys() == expected.keys()
        changed = []
        for key, reference in expected.items():
            row = actual[key]
            assert row["input_sha256"] == reference["input_sha256"], key
            assert list(row["output_arg_indices"]) == reference["output_arg_indices"], key
            if row["output_sha256"] != reference["output_sha256"]:
                assert row["kernel"] in CORRECTED, (key, "unexpected arithmetic change")
                changed.append({name: row[name] for name in (
                    "kernel", "variant", "input_sha256", "output_arg_indices", "output_sha256")})
        corrected.update(row["kernel"] for row in changed)
        cases.append({"scenario": case["scenario"], "kernels": changed})
        print(case["scenario"], [row["kernel"] for row in changed], flush=True)
    assert corrected == CORRECTED
    result = {
        "schema": "smag-compiled-wrf-word-extension-v1",
        "original_fixture_sha256": sha(args.original), "generator_sha256": sha(__file__),
        "reason": "449e3d697af merged compiled-WRF N2, TKE coefficient and tendency fixes after the original randomized fixture",
        "module_source_sha256": compile_receipt["source_sha256"],
        "options": compile_receipt["targets"]["120"]["production"]["options"],
        "device": str(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]),
        "scenarios": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output, sha(args.output), flush=True)


if __name__ == "__main__":
    main()
