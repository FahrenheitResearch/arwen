"""The offline profile-authoring tool is generic; the models are DATA.

Drew's ruling (2026-08-17): model names live in data, never in code or in
tool filenames, and that reaches the OFFLINE generators too.  Two
one-shot scripts used to carry a model in their filename and the model's
whole authority content in Python -- so a new model meant a new script,
which is exactly the per-model adapter the arbitrary acceptance test
forbids.

The breakage this file prevents: a packaged profile whose derivation is
unreproducible.  Every authority these specs name is regenerated here
and compared BYTE FOR BYTE against the document the wheel ships, so a
spec that drifts from its committed authority fails instead of rotting
quietly (which is exactly what had happened to the retired AIGEFS
script -- it no longer reproduced two of the four documents it claimed
to author).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "author_source_profile.py"
SPEC_DIR = ROOT / "gpuwm" / "authorities" / "specs"

#: The two scripts this tool replaces.  Named so the retirement is a
#: measured fact rather than an intention, and so re-adding either as a
#: shipped tool fails a test.
RETIRED = (
    ROOT / "tools" / "authoring_gefs_profile_gen.py",
    ROOT / "work" / "author_aigefs_hybrid_authorities.py",
)

#: Model/source tokens that must not appear in the GENERIC tool's source.
#: This is the ruling itself, enforced rather than only recorded.
MODEL_TOKENS = (
    "gefs", "aigefs", "aigfs", "gdas", "hrrr", "rrfs", "icon",
    "ecmwf", "ifs", "aifs", "rap", "gem", "gdps", "20crv3",
)


def _specs() -> list[Path]:
    return sorted(SPEC_DIR.glob("*.profile-spec.json"))


def test_the_generic_tool_exists() -> None:
    assert TOOL.is_file(), f"{TOOL} is the converged authoring tool"


@pytest.mark.parametrize("retired", RETIRED, ids=lambda p: p.name)
def test_the_model_named_generators_no_longer_ship(retired: Path) -> None:
    assert not retired.exists(), (
        f"{retired.relative_to(ROOT)} names a model in a tool filename and "
        "holds that model's authority content in Python; it is retired to "
        "the model-gauntlet staging tree as a provenance receipt")


def test_the_tool_names_no_model() -> None:
    """A per-model branch in the generic tool would fail the same ruling."""
    lowered = TOOL.read_text(encoding="utf-8").lower()
    named = sorted({token for token in MODEL_TOKENS if token in lowered})
    assert not named, (
        f"the generic authoring tool names {named}; a model belongs in a "
        "spec file, not in the tool that reads specs")


def test_there_is_at_least_one_spec() -> None:
    assert _specs(), f"no *.profile-spec.json under {SPEC_DIR}"


@pytest.mark.parametrize("spec", _specs(), ids=lambda p: p.name)
def test_a_spec_regenerates_its_authorities_byte_identically(
        spec: Path, tmp_path: Path) -> None:
    """The committed authority is what this spec produces, byte for byte."""
    completed = subprocess.run(
        [sys.executable, str(TOOL), str(spec), "--out-root", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    declared = json.loads(spec.read_text(encoding="utf-8"))
    outputs = [document["output"] for document in declared["documents"]]
    assert outputs, f"{spec.name} declares no documents"

    drifted = []
    for relative in outputs:
        regenerated = (tmp_path / relative).read_bytes()
        committed = (ROOT / relative).read_bytes()
        if regenerated != committed:
            drifted.append(relative)
    assert not drifted, (
        f"{spec.name} no longer reproduces {drifted}; the shipped authority "
        "and its declared derivation have diverged")


def test_an_unknown_operation_refuses_by_name(tmp_path: Path) -> None:
    """A silent no-op on a misspelled op would author a wrong authority."""
    spec = tmp_path / "broken.profile-spec.json"
    spec.write_text(json.dumps({
        "schema": "gpuwm-source-profile-authoring-spec-v1",
        "documents": [{
            "output": "out.json",
            "base": {"kind": "empty"},
            "steps": [{"op": "st", "path": ["a"], "value": 1}],
        }],
    }), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(TOOL), str(spec), "--out-root", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "st" in completed.stderr, completed.stderr


def test_an_unknown_schema_refuses_by_name(tmp_path: Path) -> None:
    spec = tmp_path / "wrong-schema.profile-spec.json"
    spec.write_text(json.dumps({
        "schema": "something-else-v9",
        "documents": [],
    }), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(TOOL), str(spec), "--out-root", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "something-else-v9" in completed.stderr, completed.stderr
