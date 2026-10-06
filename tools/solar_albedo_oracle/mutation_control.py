"""Prove the compiled-fork fixture rejects a changed normalization formula.

Usage: python mutation_control.py MODULE.py FIXTURE.npz OUTPUT_DIRECTORY

A private copy of the actual module changes the normalization numerator
from 1.0 to 0.9. The oracle comparison must fail, then the copy is restored
byte for byte and replayed successfully. The production source is read only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replay(path, fixture):
    namespace = {"__name__": "solar_albedo_mutation_target"}
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    fields = {key[3:]: fixture[key].copy() for key in fixture.files
              if key.startswith("in_")}
    outputs = []
    for call, cosine in enumerate(fixture["coszen"]):
        if call:
            fields["albedo"][...] = np.float32(0.123)
        namespace["update_solar_albedo_host"](
            fields, cosine, initialize=call == 0)
        outputs.append({key: fields[key].copy()
                        for key in ("albsol", "albbcksol")})
    return outputs


def compare(outputs, fixture):
    changes = 0
    for call, output in enumerate(outputs):
        for name, actual in output.items():
            expected = fixture[f"out_{name}"][call]
            changes += int(np.count_nonzero(actual.view(np.uint32)
                                           != expected.view(np.uint32)))
    return changes


def main():
    source, fixture_path, output_dir = map(Path, sys.argv[1:4])
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "mutation_target.py"
    original = source.read_bytes()
    before = b"numerator = F(1.0) + twice_d"
    after = b"numerator = F(0.9) + twice_d"
    assert original.count(before) == 1
    target.write_bytes(original)
    with np.load(fixture_path, allow_pickle=False) as fixture:
        assert compare(replay(target, fixture), fixture) == 0
        mutated = original.replace(before, after)
        try:
            target.write_bytes(mutated)
            outputs = replay(target, fixture)
            changed_words = compare(outputs, fixture)
            assert changed_words > 0
            detected = False
            try:
                np.testing.assert_array_equal(
                    outputs[0]["albsol"].view(np.uint32),
                    fixture["out_albsol"][0].view(np.uint32))
            except AssertionError:
                detected = True
            assert detected, "normalization mutation survived the oracle"
        finally:
            target.write_bytes(original)
        assert target.read_bytes() == original
        restored_changes = compare(replay(target, fixture), fixture)
        assert restored_changes == 0
    assert source.read_bytes() == original
    receipt = {"source_sha256": digest(original),
               "mutation_sha256": digest(mutated),
               "restored_sha256": digest(target.read_bytes()),
               "fixture_sha256": digest(fixture_path.read_bytes()),
               "mutation": "normalization numerator 1.0 changed to 0.9",
               "mutation_detected": detected, "changed_words": changed_words,
               "restored_changed_words": restored_changes,
               "production_source_unchanged": True}
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
