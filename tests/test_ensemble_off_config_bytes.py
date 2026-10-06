"""Disabled ensemble orchestration preserves staging's emitted config bytes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm.toml_document import emit_experiment_toml
from gpuwm.ensemble.door import request_for_payload

BASELINE = json.loads((Path(__file__).parent / "data/ensemble_off_emit_golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(BASELINE["cases"]))
def test_disabled_ensemble_emits_exact_current_staging_config_bytes(name):
    case = BASELINE["cases"][name]
    original = case["input_utf8"].encode("utf-8")
    document = tomllib.loads(original.decode("utf-8"))
    before = deepcopy(document)
    assert request_for_payload(original) is None
    emitted = emit_experiment_toml(document).encode("utf-8")
    expected = bytes.fromhex(case["emitted_utf8_hex"])
    assert emitted == expected
    assert hashlib.sha256(emitted).hexdigest() == case["emitted_sha256"]
    assert document == before
    assert b"[ensemble]" not in emitted


def test_emission_baseline_names_the_exact_staging_source():
    assert BASELINE["schema"] == "ensemble-off-emitted-config-baseline.v1"
    assert len(BASELINE["staging_revision"]) == 40
    assert BASELINE["emitter_path"] == "gpuwm/toml_document.py"
    assert len(BASELINE["emitter_source_sha256"]) == 64
    for case in BASELINE["cases"].values():
        expected = bytes.fromhex(case["emitted_utf8_hex"])
        assert hashlib.sha256(expected).hexdigest() == case["emitted_sha256"]
