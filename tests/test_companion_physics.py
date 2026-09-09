import copy
import hashlib
import json

import pytest

from gpuwm import companion_domains as editor
from gpuwm.companion_physics import repairs
from test_companion_domains import configured_case


def draft(source, grid_id=0):
    return {"schema": editor.REQUEST_SCHEMA, "config_path": str(source),
            "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "action": {"kind": "set_physics", "grid_id": grid_id,
                       "settings": {"mp_physics": 9, "moist": True, "moist_cq": True,
                                    "ra_lw_physics": 4, "ra_sw_physics": 4,
                                    "ra_rrtmg_variant": "rte-rrtmgp"}}}


def test_repairs_keep_unverified_runnable_options_and_publish_nothing(tmp_path):
    source, raw = configured_case(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    request = draft(source)
    result = repairs(request)
    assert result["valid"] is False
    assert "cloud particles" in result["summary"]
    assert result["created"] is result["forecast_started"] is False
    closest = result["options"][0]
    assert closest["action"]["settings"] == dict(request["action"]["settings"], ra_rrtmg_variant="rrtmg_legacy")
    assert [change["field"] for change in closest["changes"]] == ["ra_rrtmg_variant"]
    assert any(row["maturity"] == "implemented-unverified" for option in result["options"] for row in option["evidence"])
    # Every advertised result really passes the parser with all domain overrides.
    for option in result["options"]:
        candidate = copy.deepcopy(raw)
        editor._apply(candidate, option["action"], source)
        editor._build(candidate, source)
        assert option["validation"]["forecast_run"] == "not_run"
    assert any("Longwave radiation will be disabled." in option["effects"] for option in result["options"])
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_single_domain_repair_does_not_change_siblings(tmp_path):
    source, raw = configured_case(tmp_path)
    raw["shared"].update(moist=True, moist_cq=True, mp_physics=9, ra_rrtmg_variant="rrtmg_legacy")
    from gpuwm.toml_document import emit_experiment_toml
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    request = draft(source, 2)
    # These defaults are shared; keep the request within per-domain scope.
    request["action"]["settings"].pop("moist")
    request["action"]["settings"].pop("moist_cq")
    result = repairs(request)
    closest = result["options"][0]
    candidate = copy.deepcopy(raw)
    editor._apply(candidate, closest["action"], source)
    assert candidate["domain"][0] == raw["domain"][0]
    assert candidate["domain"][2] == raw["domain"][2]
    assert candidate["shared"] == raw["shared"]
    assert candidate["domain"][1]["mp_physics"] == 9
    assert candidate["domain"][1]["ra_rrtmg_variant"] == "rrtmg_legacy"


def test_valid_draft_and_stale_source(tmp_path):
    source, _ = configured_case(tmp_path)
    request = draft(source)
    request["action"]["settings"]["ra_rrtmg_variant"] = "rrtmg_legacy"
    result = repairs(request)
    assert result["valid"] is True
    assert result["options"] == []
    source.write_bytes(source.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="configuration changed"):
        repairs(request)
