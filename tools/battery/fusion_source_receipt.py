"""Extend two orchestration pins after measured, unchanged legacy word replays.

This does not replace native fixtures or accept a tolerance. The maintained
capture tool may report its historical GPU-archive differences; current
coordinate outputs must still match native WRF, acoustic outputs must match
the published words, and every base/tip output hash must remain equal.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = "31da186c803f4776247e36f629109785caf97fda"
SOURCES = ("gpuwm/core/acoustic.py", "gpuwm/core/dycore.py")
GENERATOR = "tools/assembled_legacy_replay.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def source_pin(revision, name):
    raw = git("show", revision + ":" + name)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def measure(replay, tip):
    from tools.assembled_legacy_replay import word_hashes

    measurements, artifacts, identities, inputs = {}, {}, {}, {}
    for family, count, words in (("coordinate", 272, 178128),
                                  ("drivers", 2618, 31686592),
                                  ("acoustic", 132, 305632)):
        arms = []
        for arm, revision in (("base", BASE), ("tip", tip)):
            folder = replay / f"{arm}-{family}"
            audit = read(folder / "source-audit.json")
            assert audit["engine_commit"] == revision
            assert audit["source_fingerprint_unchanged"]
            assert audit["compute_capability"][0] >= 10
            assert audit["strict"] is False and audit["default_diffusion"] is True
            keys = ("device", "compute_capability", "nvrtc_version", "cupy_version",
                    "module_source_sha256", "module_options", "strict", "default_diffusion",
                    "acoustic_function_options", "default_acoustic_diffusion_route")
            identities[f"{arm}-{family}"] = {key: audit[key] for key in keys}
            files = [folder / "source-audit.json"]
            if family == "coordinate":
                path = folder / "direct-native-proof.json"
                proof = read(path)
                assert proof["native_exact"] and proof["native_different_words"] == 0
                assert proof["changed_previously_native_equal_words"] == 0
                actual = {key: {"sha256": row["current_sha256"],
                                "words": row["current_to_native"]["words"]}
                          for key, row in proof["fields"].items()}
                files.append(path)
            elif family == "drivers":
                assert audit["status"] == "complete"
                actual = {}
                for name in ("horizontal-driver", "vertical-driver", "km-mutations"):
                    path = folder / f"{name}.json"
                    files.append(path)
                    actual.update({name + key: {"sha256": row["gpu_sha256"], "words": row["words"]}
                                   for key, row in word_hashes(read(path)["cases"]).items()})
                proof = None
            else:
                assert audit["status"] == "complete" and audit["pass"]
                path = folder / "acoustic-published-default-proof.json"
                proof = read(path)
                assert proof["published_exact"] and proof["changed_published_words"] == 0
                actual = {key: {"sha256": row["actual_sha256"], "words": row["words"]}
                          for key, row in proof["fields"].items()}
                files.append(path)
            if proof is not None:
                for name, recorded in proof["source_inputs"].items():
                    assert source_pin(revision, name) == recorded, (arm, name)
                inputs[f"{arm}-{family}"] = proof["source_inputs"]
            assert (len(actual), sum(row["words"] for row in actual.values())) == (count, words)
            for path in files:
                artifacts[str(path.relative_to(replay))] = sha(path)
            arms.append(actual)
        assert identities[f"base-{family}"] == identities[f"tip-{family}"], family
        assert arms[0] == arms[1], f"STOP: {family} output words changed"
        measurements[family] = dict(arrays=count, words=words, changed_arrays=0, different_words=0, changes={})
    return measurements, artifacts, identities, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    tip = git("rev-parse", "HEAD").decode().strip()
    measurements, artifacts, identities, inputs = measure(args.replay.resolve(), tip)
    folder = ROOT / "tests/data/assembled_legacy_proofs"
    extension = read(folder / "source-extension.json")
    sources = {}
    for name, transition in extension["sources"].items():
        after = source_pin(tip, name)
        if name in SOURCES:
            before = source_pin(BASE, name)
            assert transition["after"] == before, name
            assert before != after, name
            sources[name] = dict(before=before, after=after)
        else:
            assert transition["after"] == after, name
    assert set(sources) == set(SOURCES)
    result = dict(schema="measured-fusion-source-replay-v1", base=BASE, tip=tip,
                  sources=sources, measurements=measurements, artifacts=artifacts,
                  capture_identity=identities["tip-coordinate"], source_inputs=inputs,
                  generator=GENERATOR, generator_sha256=sha(ROOT / GENERATOR),
                  recorder="tools/battery/fusion_source_receipt.py", recorder_sha256=sha(__file__),
                  numerics="strict", scope="unchanged base/tip words; coordinate native exact and acoustic published exact")
    payload = json.dumps(result, indent=2) + "\n"
    args.out.write_bytes(payload.encode())
    if args.write:
        receipt = folder / "fusion-source-replay.json"
        receipt.write_bytes(payload.encode())
        for name, transition in sources.items():
            extension["sources"][name]["after"] = transition["after"]
        extension["fusion_source_replay"] = dict(receipt=receipt.name, sha256=sha(receipt))
        extension["measured_tip"] = tip
        extension_path = folder / "source-extension.json"
        extension_path.write_bytes((json.dumps(extension, indent=2) + "\n").encode())
        path = ROOT / "tests/assembled_legacy_proofs.py"
        text = path.read_text()
        node = next(node for node in ast.parse(text).body if isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "SOURCE_EXTENSION_SHA" for t in node.targets))
        lines = text.splitlines(keepends=True)
        lines[node.lineno - 1:node.end_lineno] = [f"SOURCE_EXTENSION_SHA = {sha(extension_path)!r}\n"]
        path.write_bytes("".join(lines).encode())
    print(json.dumps(dict(measurements=measurements, written=args.write), indent=2))


if __name__ == "__main__":
    main()
