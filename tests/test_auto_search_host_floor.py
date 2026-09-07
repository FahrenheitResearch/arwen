"""Impossible host stores must not multiply the automatic subset search."""
from dataclasses import replace
import tomllib

import pytest

from gpuwm.branch import emit_experiment_toml
from gpuwm.experiment import load_experiment
from gpuwm.core import preflight as pf, streaming as st
from tilestream import autoplan as ap
from test_check_nested_mixed_road import _nested_auto_tiles

GIB = 1024 ** 3


def _experiment(tmp_path, domains=4, *, host_gib=2, write_mode="ring",
                root_n=300, child_n=600, root_off=False):
    raw = tomllib.loads(_nested_auto_tiles(tmp_path).read_text(encoding="utf-8"))
    raw["domain"][0].update(nx=root_n, ny=root_n)
    raw["domain"][1].update(nx=child_n, ny=child_n)
    raw["domain"] = [raw["domain"][0]] + [
        dict(raw["domain"][1], grid_id=gid) for gid in range(2, domains + 1)]
    if host_gib is not None:
        raw["tiles"]["host_budget_bytes"] = int(host_gib * GIB)
    raw["tiles"]["write_mode"] = write_mode
    if root_off:
        raw["domain"][0]["tiles"] = {"mode": "off"}
    path = tmp_path / "validated.toml"
    path.write_text(emit_experiment_toml(raw), encoding="utf-8")
    return load_experiment(path)


def _walk(exp, free_gib=24):
    estimate = pf.estimate_experiment(exp)
    nodes = st._config_tree_nodes(exp.domains)
    machine = ap.Machine(int(free_gib * GIB), 256 * GIB)
    rows = {}
    result = st.decide_tree(nodes, exp.tiles, machine=machine, decisions=rows,
                           resident_estimate=estimate)
    return result, rows


@pytest.mark.parametrize("domains", [4, 8, 12, 16])
def test_host_impossible_choices_terminate_after_one_real_walk(tmp_path, monkeypatch, domains):
    exp = _experiment(tmp_path, domains)
    estimate = pf.estimate_experiment(exp)
    nodes = st._config_tree_nodes(exp.domains)
    budget = int(23.5 * GIB)
    # The existing global/process floor pruning cannot decide this case.
    assert st._resident_subset_envelope(estimate, nodes, set()) < budget
    assert st._tree_process_overhead_bytes(nodes) + st._tree_radiation_transient_bytes(nodes) < budget
    child_floor = st.radiation_footprint(exp.domains[1].run, exp.tiles).store_bytes(600 * 600 * 49)
    assert child_floor > exp.tiles.host_budget_bytes
    original = st._decide_tree
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(st, "_decide_tree", counted)
    with pytest.raises(st.StreamingRefused, match="must remain resident") as error:
        _walk(exp)
    assert "host allowance" in str(error.value)
    assert "required resident envelope" in str(error.value)
    assert len(calls) == 1  # Twelve domains previously repeated 4,097 real walks.


def test_host_impossible_streaming_can_keep_that_domain_resident(tmp_path):
    exp = _experiment(tmp_path, 2, root_n=704, child_n=63)
    root_floor = st.radiation_footprint(exp.root.run, exp.tiles).store_bytes(704 * 704 * 49)
    assert root_floor > exp.tiles.host_budget_bytes
    result, rows = _walk(exp, free_gib=22)
    assert not rows[1].stream and rows[2].stream
    assert result.configured_mixed_envelope_bytes <= int(21.5 * GIB)
    assert result.host_spent_bytes <= exp.tiles.host_budget_bytes


def test_two_shadow_stores_are_required_while_ring_alternative_remains(tmp_path):
    ring = _experiment(tmp_path, 3, host_gib=6, root_off=True)
    result, rows = _walk(ring)
    assert not rows[1].stream and any(row.stream for row in rows.values())
    assert result.host_spent_bytes <= 6 * GIB
    shadow = replace(ring, tiles=replace(ring.tiles, write_mode="shadow"))
    with pytest.raises(st.StreamingRefused, match="must remain resident"):
        _walk(shadow)


def test_ample_host_preserves_the_existing_fitting_alternative(tmp_path):
    exp = _experiment(tmp_path, 8, host_gib=None)
    result, rows = _walk(exp)
    # Same roads as the committed-before control on this public configuration.
    assert [row.stream for row in rows.values()] == [False, False, True, True, True, True, True, True]
    assert result.configured_mixed_envelope_bytes <= int(23.5 * GIB)
    assert result.host_spent_bytes <= result.host_budget_bytes


def test_all_resident_fit_needs_no_streamed_host_allowance(tmp_path):
    exp = _experiment(tmp_path, 2, host_gib=0.001)
    result, rows = _walk(exp, free_gib=64)
    assert not any(row.stream for row in rows.values())
    assert result.host_spent_bytes == 0