"""Prepared complete generic bytes against independently captured old source."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from test_mix_selection_byte_controls import build_experiment, import_documents


PARENT = "1077c6e8abe502aab59de103d7e6d70dbb8588a1"
MATRIX_SHA256 = "47f581f453640588e8cf72a132780570761ca60b8ce8cf627fc730c4c66a2583"
CASES = ("explicit_false", "short_false", "mixed_false_true", "omitted_theta")
FIXTURES = Path(__file__).parent / "fixtures/mix_additional_parent_controls"


@pytest.mark.parametrize("parsed", (False, True))
@pytest.mark.parametrize("case", CASES)
def test_generic_false_and_omitted_theta_bytes_match_original_parent(case, parsed, tmp_path):
    matrix_bytes = (FIXTURES / "matrix.json").read_bytes()
    assert hashlib.sha256(matrix_bytes).hexdigest() == MATRIX_SHA256
    matrix = json.loads(matrix_bytes)
    assert matrix["schema"] == "mix-parent-additional-inputs-v1"
    assert matrix["audit_parent"] == PARENT
    assert set(matrix["cases"]) == set(CASES)
    pins = json.loads((FIXTURES / "historical-pins.json").read_bytes())
    assert pins["mode"] == "baseline" and pins["audit_parent"] == PARENT
    assert pins["matrix_sha256"] == MATRIX_SHA256
    pin = pins["cases"][case]
    relative = Path(pin["file"])
    assert len(relative.parts) == 1 and relative.suffix == ".toml"
    expected = (FIXTURES / relative).read_bytes()
    assert len(expected) == pin["bytes"]
    assert hashlib.sha256(expected).hexdigest() == pin["sha256"]
    emitted, _ = import_documents(
        tmp_path, deepcopy(matrix["cases"][case]), parsed=parsed)
    assert emitted.encode("utf-8") == expected
    exp = build_experiment(tomllib.loads(emitted), source="original generic bytes")
    assert [domain.run.mix_full_fields for domain in exp.domains] == [True, True]
