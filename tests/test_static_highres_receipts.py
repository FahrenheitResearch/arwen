"""Receipt links remain valid when a high-resolution cache is reused."""
from copy import deepcopy
import json

import pytest

from gpuwm.static.highres_production import HighresStaticConfig, _write_receipt


def _receipt(config):
    return {
        "grid": {"domain_id": 1, "dx": 3000.0, "dy": 3000.0},
        "config": config.echo(),
        "case_date": "2026-01-01",
        "created_utc": "2026-01-02T00:00:00+00:00",
        "status": "APPLIED",
        "coverage": {"cells_replaced": 20},
    }


@pytest.mark.parametrize("change", [
    {"case_date": "2026-01-02"},
    {"created_utc": "2026-01-02T01:00:00+00:00"},
    {"status": "REFUSED"},
    {"coverage": {"cells_replaced": 19}},
], ids=["case-date", "creation-time", "status", "coverage"])
def test_changed_receipt_preserves_previous_link(tmp_path, change):
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path)
    first = _receipt(config)
    second = deepcopy(first)
    second.update(change)
    first_path = _write_receipt(config, first)
    original = first_path.read_bytes()

    second_path = _write_receipt(config, second)

    assert first_path != second_path
    assert first_path.read_bytes() == original
    assert json.loads(first_path.read_text())["case_date"] == first["case_date"]
    saved = json.loads(second_path.read_text())
    for key, value in change.items():
        assert saved[key] == value
    assert first["receipt_path"] == str(first_path.resolve())
    assert second["receipt_path"] == str(second_path.resolve())


def test_same_receipt_rewrite_preserves_path_and_bytes(tmp_path):
    config = HighresStaticConfig(enabled=True, cache_root=tmp_path)
    receipt = _receipt(config)
    payload = deepcopy(receipt)
    path = _write_receipt(config, receipt)
    original = path.read_bytes()

    # The writer adds its own path to the caller's dictionary.
    assert _write_receipt(config, receipt) == path
    assert path.read_bytes() == original
    assert json.loads(original) == payload
    # Mapping insertion order is not part of receipt identity.
    assert _write_receipt(config, dict(reversed(list(payload.items())))) == path
    assert path.read_bytes() == original
