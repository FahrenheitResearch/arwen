"""A prepared urban count governs automatic resident admission too."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpuwm.core import preflight as pf, streaming as st
from test_urban_workspace_price import _city_tree, _land_use
from tilestream.autoplan import Machine


def _auto_tree(tmp_path, *, single=False):
    exp = _city_tree(tmp_path)
    if single:
        exp = replace(exp, domains=(exp.root,))
    options = st.StreamingOptions(
        mode="auto", resident_context=st.ResidentAdmissionContext(exp))
    return replace(exp, tiles=options)


def _between_prices(exp, counts):
    prepared = pf.admission_estimate(exp, urban_columns=counts)
    bound = pf.admission_estimate(exp)
    budget = (prepared.peak_envelope_bytes + bound.peak_envelope_bytes) // 2
    assert prepared.peak_envelope_bytes < budget < bound.peak_envelope_bytes
    return prepared, Machine(budget + pf.EXTERNAL_MARGIN_BYTES, 256 * pf.GIB)


def test_auto_single_admits_the_prepared_workspace_before_road_selection(tmp_path):
    """A fitting prepared domain stays resident at a budget below its bound."""
    exp = _auto_tree(tmp_path, single=True)
    counts = {1: 500}
    expected, machine = _between_prices(exp, counts)
    priced = st.cold_single_domain_admission(
        exp, machine=machine, options=exp.tiles, urban_columns=counts)
    assert priced.peak_envelope_bytes == expected.peak_envelope_bytes
    decision = st.cold_single_domain_decision(
        exp, machine=machine, urban_columns=counts)
    assert not decision.stream
    assert decision.resident_bytes == expected.peak_envelope_bytes
    assert (decision.detail["resident_admission"]["envelope_bytes"]
            == expected.peak_envelope_bytes)


def test_auto_tree_admits_each_prepared_workspace_before_road_selection(tmp_path):
    """The tree's automatic decision uses its prepared counts for every rung."""
    exp = _auto_tree(tmp_path)
    counts = {1: 500, 2: 500}
    expected, machine = _between_prices(exp, counts)
    rows = {}
    outcome = st.cold_tree_streaming_decision(
        exp, st._config_tree_nodes(exp.domains), machine=machine,
        decisions=rows, urban_columns=counts)
    assert outcome is not None
    assert len(rows) == 2 and all(not row.stream for row in rows.values())
    assert outcome.resident_subset_envelope_bytes == expected.peak_envelope_bytes
    for row in rows.values():
        assert (row.detail["resident_admission"]["envelope_bytes"]
                == expected.peak_envelope_bytes)


@pytest.mark.parametrize("urban", [500, 5000])
def test_live_head_prices_urban_columns_in_the_tree_admission(
        tmp_path, monkeypatch, urban):
    """The prepared land cover decides whether an urban root streams.

    Priced at every column urban, the root streams; priced from its 500
    prepared urban columns the whole tree fits resident, and at 5000 the
    root still streams.  Since A186 a streamed [tiles] root streams from the
    chained head too, so the old seal-wait answer this test also read is
    retired with ROOT_STORE_NEEDS_SEAL; what stays is the admission itself.
    """
    from gpuwm import prepared_domain_tree_forecast as tree

    exp = _auto_tree(tmp_path)
    exp = replace(exp, domains=(exp.root, replace(exp.domains[1], tiles=st.OFF)))
    _, machine = _between_prices(exp, {1: 500, 2: 500})
    reader = SimpleNamespace(header={})
    inputs = SimpleNamespace(
        experiment=exp, source=None, stream_head={"head_sha256": "a" * 64},
        domains=tuple(SimpleNamespace(
            grid_id=dc.grid_id, cache_reader=reader,
            static_fields={"LU_INDEX": _land_use(
                (dc.run.ny, dc.run.nx), urban)}) for dc in exp.domains))
    counts = tree.tree_urban_columns(inputs)
    assert counts == {1: urban, 2: urban}
    nodes = tree._prepared_planning_nodes(inputs)
    bound_decisions, prepared_decisions = {}, {}
    st.cold_tree_streaming_decision(exp, nodes, machine=machine,
                                    decisions=bound_decisions)
    st.cold_tree_streaming_decision(exp, nodes, machine=machine,
                                    decisions=prepared_decisions,
                                    urban_columns=counts)
    assert bound_decisions[1].stream
    assert not bound_decisions[2].stream
    if urban == 500:
        assert not any(row.stream for row in prepared_decisions.values())
    else:
        assert prepared_decisions[1].stream


@pytest.mark.parametrize("options", [st.OFF, st.StreamingOptions(
    mode="on", tile_nx=64, tile_ny=64, nbuffers=2)])
def test_explicit_single_road_still_skips_urban_admission_price(
        tmp_path, monkeypatch, options):
    """A count does not make an explicit road consult automatic admission."""
    exp = _auto_tree(tmp_path, single=True)

    def forbidden(*args, **kwargs):
        pytest.fail("an explicit road priced automatic admission")

    monkeypatch.setattr(pf, "admission_estimate", forbidden)
    assert st.cold_single_domain_admission(
        exp, options=options, urban_columns={1: 500}) is None
    decision = st.cold_single_domain_decision(
        exp, options=options, urban_columns={1: 500})
    assert decision.stream is (options.mode == "on")
