"""A live head prices prepared urban workspaces before choosing to chain."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpuwm.core import preflight as pf
from gpuwm.ingest import boundary_stream as stream
from test_urban_workspace_price import _city_tree, _land_use


def _prepared_head(exp, urban=500):
    statics = {int(dc.grid_id): {"LU_INDEX": _land_use(
        (dc.run.ny, dc.run.nx), urban)} for dc in exp.domains}
    children = tuple(SimpleNamespace(domain=dc,
                                     static_fields=statics[int(dc.grid_id)])
                     for dc in exp.domains[1:])
    return statics[int(exp.root.grid_id)], children


@pytest.mark.parametrize("single", [False, True])
def test_prepared_workspace_keeps_a_fitting_cuda_head_chained(tmp_path, single):
    """The configuration bound cannot delay a prepared head that fits."""
    exp = _city_tree(tmp_path)
    if single:
        exp = replace(exp, domains=(exp.root,))
    static, children = _prepared_head(exp)
    counts = stream.prepared_head_urban_columns(
        exp, static, child_results=children)
    assert counts == {int(dc.grid_id): 500 for dc in exp.domains}
    prepared = pf.admission_estimate(exp, source="gfs", urban_columns=counts)
    bound = pf.admission_estimate(exp, source="gfs")
    budget = (prepared.peak_envelope_bytes + bound.peak_envelope_bytes) // 2
    assert prepared.peak_envelope_bytes < budget < bound.peak_envelope_bytes
    producer = pf.GIB // 4
    card = (budget + producer + pf.EXTERNAL_MARGIN_BYTES, 0)
    unknown = stream.chained_admission(
        experiment=exp, backend="cuda", device_bytes=producer,
        card=card, source="gfs")
    assert not unknown["admitted"]
    staging = tmp_path / "staging"
    staging.mkdir()
    writer = stream.PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "prepared", identity={},
        chained=True)
    decision = writer.admit(
        experiment=exp, backend="cuda", device_bytes=producer,
        card=card, source="gfs", urban_columns=counts)
    assert writer.chained and decision["admitted"]
    assert decision["forecast_bytes"] == prepared.peak_envelope_bytes
    assert decision["forecast_bytes"] + producer <= decision["budget_bytes"]
    larger_static, larger_children = _prepared_head(exp, urban=5000)
    larger = stream.prepared_head_urban_columns(
        exp, larger_static, child_results=larger_children)
    refused = stream.chained_admission(
        experiment=exp, backend="cuda", device_bytes=producer,
        card=card, source="gfs", urban_columns=larger)
    assert not refused["admitted"]
    assert refused["forecast_bytes"] + producer > refused["budget_bytes"]


@pytest.mark.parametrize("moving", ["follow", "spawn"])
def test_chained_counts_keep_changing_ground_at_its_bound(tmp_path, moving):
    exp = _city_tree(tmp_path)
    child = replace(exp.domains[1], **{moving: object()})
    grandchild = replace(child, grid_id=3, parent_id=2, follow=None, spawn=None)
    exp = replace(exp, domains=(exp.root, child, grandchild))
    static, children = _prepared_head(exp)
    assert stream.prepared_head_urban_columns(
        exp, static, child_results=children) == {1: 500}


def test_chained_counts_leave_unprepared_children_unknown(tmp_path):
    exp = _city_tree(tmp_path)
    static, _ = _prepared_head(exp)
    assert stream.prepared_head_urban_columns(exp, static) == {1: 500}


def test_preparation_only_install_does_not_read_a_forecast_count(monkeypatch):
    monkeypatch.setattr(stream, "forecast_installed", lambda: False)
    assert stream.prepared_head_urban_columns(object(), {}) is None
