"""Reach a proven automatic road quickly without losing mixed alternatives."""
import tomllib

import pytest

from gpuwm.branch import emit_experiment_toml
from gpuwm.experiment import load_experiment
from gpuwm.core import streaming as st
from test_auto_search_host_floor import _experiment, _walk, GIB


@pytest.mark.parametrize("domains", [8, 12, 16])
def test_many_automatic_domains_reach_a_fitting_road_in_linear_candidates(
        tmp_path, monkeypatch, domains):
    exp = _experiment(tmp_path, domains, host_gib=None)
    original = st._decide_tree
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs["forced_stream"])
        return original(*args, **kwargs)
    monkeypatch.setattr(st, "_decide_tree", counted)
    result, rows = _walk(exp)
    # A bound on this regression fixture, not a production search cutoff.
    # The 8-domain control previously needed221 real walks;12/16 each
    # exceeded a512-walk review cap even though a fitting plan existed.
    assert len(calls) <= 3 * domains
    assert any(row.stream for row in rows.values())
    assert result.configured_mixed_envelope_bytes <= int(23.5 * GIB)
    assert (result.process_overhead_bytes + result.vram_spent_bytes
            + result.radiation_transient_bytes) <= int(23.5 * GIB)
    assert result.host_spent_bytes <= result.host_budget_bytes
    # Selected decisions keep the actual caller's source of admission.
    assert all(row.detail["resident_admission"]["budget_bytes"] == int(23.5 * GIB)
               for row in rows.values())


def test_exhaustive_fallback_preserves_a_nonprefix_mixed_road(tmp_path, monkeypatch):
    _experiment(tmp_path, 4, host_gib=16, root_n=704, child_n=63)
    path = tmp_path / "validated.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    raw["domain"][1].update(nx=600, ny=600)
    path.write_text(emit_experiment_toml(raw), encoding="utf-8")
    exp = load_experiment(path)
    original = st._decide_tree
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs["forced_stream"])
        return original(*args, **kwargs)
    monkeypatch.setattr(st, "_decide_tree", counted)
    result, rows = _walk(exp, free_gib=18)
    # The root and its large child stream while both small siblings remain
    # resident. Neither one changed choice nor a reverse-order cumulative
    # prefix finds this actual configured/host/coupling fit.
    assert [row.stream for row in rows.values()] == [True, True, False, False]
    assert frozenset({1, 2, 3, 4}) in calls  # fast all-auto alternative tried
    assert calls[-1] == frozenset({1, 2})
    assert result.configured_mixed_envelope_bytes <= int(17.5 * GIB)
    assert result.host_spent_bytes <= 16 * GIB