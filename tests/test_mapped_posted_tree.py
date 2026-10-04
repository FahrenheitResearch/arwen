"""Posted mapped tree authority completion, without model numerics."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from gpuwm import mapped_direct
from gpuwm.ingest import boundary_stream
from test_posted_mapped_preparation import _manifest, _markers


def _seal_inputs(tmp_path, monkeypatch):
    manifest = _manifest()
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = boundary_stream.input_plan(manifest, lead_role_prefix="",
        route_table_sha256="7" * 64, fixed_rows=["donor/gdas.f000"])
    placeholder = boundary_stream.as_posted_placeholder(
        boundary_stream.input_plan_sha256(plan))
    source = SimpleNamespace(
        finish=lambda: (path, digest, "complete native composition"),
        markers={int(key): value for key, value in _markers().items()},
        posted=SimpleNamespace(route_table_sha256=lambda: "7" * 64,
                               waits=[{"lead": 3}]))
    source_identity = {"adapter": "mapped", "target_contract": {"domains": 2},
        "input_manifest_sha256": placeholder,
        "composition_receipt_sha256": placeholder}
    seen = {}
    monkeypatch.setattr(mapped_direct, "mapped_composition_receipt",
        lambda bundle: {"complete_bundle": bundle})
    monkeypatch.setattr(mapped_direct, "decoded_vertical_ladder",
        lambda *_args: None)
    monkeypatch.setattr(mapped_direct, "_source_top_pressure_pa",
        lambda snapshots, **_contract: 5000.)
    monkeypatch.setattr(mapped_direct, "_require_source_top",
        lambda *args: seen.setdefault("top_checked", True))
    class Writer:
        root = tmp_path / "output"
        def write_posted_leads(self, markers, **kwargs):
            seen["leads"] = (markers, kwargs)
        def seal_cache(self, **kwargs):
            seen["seal"] = kwargs
            return {"content_sha256": "c" * 64}
    Writer.root.mkdir()
    (Writer.root / "source-evidence").mkdir()
    arguments = dict(writer=Writer(), plan={"plan": plan,
        "fixed_rows": ["donor/gdas.f000"]}, mapping_contract={}, snapshots=(),
        exp=SimpleNamespace(root=SimpleNamespace(grid_id=1)), cfg=object(),
        source_identity=source_identity, static_cache_sha256="a" * 64,
        namelist_sha256="b" * 64, forcing_identity={"forcing_hours": (0, 3)})
    return source, arguments, seen, digest


def test_tree_seal_uses_completed_source_for_ordinary_root_identity(tmp_path, monkeypatch):
    source, arguments, seen, digest = _seal_inputs(tmp_path, monkeypatch)
    def builder(identity, manifest_sha256):
        assert seen["top_checked"]
        assert identity["input_manifest_sha256"] == manifest_sha256 == digest
        assert len(identity["composition_receipt_sha256"]) == 64
        assert identity["target_contract"] == {"domains": 2}
        seen["builder"] = (identity, manifest_sha256)
        return {"ordinary_root_binding": identity, "grid_id": 1}
    result = mapped_direct._seal_posted_mapped(source,
        identity_builder=builder, **arguments)
    assert result["identity"] == seen["seal"]["identity"]
    assert result["identity"]["grid_id"] == 1
    assert result["manifest_sha256"] == seen["seal"]["manifest_sha256"] == digest
    assert result["source_identity"] == seen["builder"][0]
    assert set(seen["leads"][0]) == {0, 3}
    evidence = arguments["writer"].root / "source-evidence/input-manifest.json"
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == digest


def test_tree_seal_refuses_an_endpoint_not_bound_by_its_posted_marker(tmp_path, monkeypatch):
    source, arguments, seen, _digest = _seal_inputs(tmp_path, monkeypatch)
    source.markers[3]["objects"][0]["sha256"] = "0" * 64
    with pytest.raises(boundary_stream.BoundaryStreamError, match="not an object"):
        mapped_direct._seal_posted_mapped(source,
            identity_builder=lambda *_args: pytest.fail("unbound endpoint reached initializer identity"),
            **arguments)
    assert "seal" not in seen and "leads" not in seen


def test_tree_seal_cannot_change_its_mapping_plan(tmp_path, monkeypatch):
    source, arguments, seen, _digest = _seal_inputs(tmp_path, monkeypatch)
    source.posted.route_table_sha256 = lambda: "8" * 64
    with pytest.raises(ValueError, match="not the input plan"):
        mapped_direct._seal_posted_mapped(source,
            identity_builder=lambda *_args: pytest.fail("changed plan reached root identity"),
            **arguments)
    assert "seal" not in seen
