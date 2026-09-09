"""``gpuwm go`` refuses on the tile PLANNER, never on the planning REPORT.

The planning report (``streaming.tree_road_plan``) promises never to raise:
it runs inside pricing surfaces that answer before the user spends anything.
Until this file existed the promise was kept by a bare ``except Exception``
that wrote the exception's text into the SAME attribute a genuine planner
refusal used, and ``gpuwm go`` read that attribute as a refusal -- so a
``TypeError`` anywhere in the pricing walk blocked a tree that runs, and
did so even on the branch whose comment says "never refuse on a card we
cannot see" (ENG-014).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpuwm import go_cli
from gpuwm.core import streaming
from gpuwm.experiment import load_experiment
from test_starter_template import tile_starter


def _road(**fields):
    base = {"refusal": None, "refusal_resource": None, "report_error": None}
    base.update(fields)
    return SimpleNamespace(**base)


def test_a_report_failure_never_refuses_and_is_named_as_the_reports():
    for card_seen in (True, False):
        refuse, note = go_cli.planner_gate(
            _road(report_error="TypeError: unsupported operand"),
            card_seen=card_seen)
        assert refuse is False
        assert "tile-planning report failed" in note
        assert "TypeError: unsupported operand" in note
        assert "resident price" in note


def test_a_genuine_memory_refusal_refuses_only_on_a_measured_card():
    road = _road(refusal="no auto road fits the 5.50 GiB admission budget",
                 refusal_resource="memory")
    assert go_cli.planner_gate(road, card_seen=True) == (
        True, "native tile planner refused this configuration: "
        "no auto road fits the 5.50 GiB admission budget")
    refuse, note = go_cli.planner_gate(road, card_seen=False)
    assert refuse is False, "a memory verdict against a card nobody measured"
    assert "not refusing on a card we cannot see" in note
    assert "no auto road fits" in note


@pytest.mark.parametrize("resource", ["geometry", None])
def test_a_configuration_refusal_refuses_with_or_without_a_card(resource):
    road = _road(refusal="tile halo wider than the domain", refusal_resource=resource)
    for card_seen in (True, False):
        refuse, note = go_cli.planner_gate(road, card_seen=card_seen)
        assert refuse is True
        assert "tile halo wider than the domain" in note


def test_no_report_at_all_is_not_a_refusal():
    assert go_cli.planner_gate(None, card_seen=False) == (False, None)
    assert go_cli.planner_gate(_road(), card_seen=True) == (False, None)


def test_the_report_keeps_a_walk_exception_apart_from_a_refusal(tmp_path, monkeypatch):
    """``tree_road_plan`` files a TypeError under ``report_error``, not ``refusal``."""
    from gpuwm import starter_template as st
    path, raw = tile_starter(tmp_path, nested=True)
    raw["tiles"] = {"mode": "auto"}
    path.write_text(st.render_tables(raw), encoding="utf-8")
    exp = load_experiment(path)

    def broken(*args, **kwargs):
        raise TypeError("unsupported operand type(s) inside the pricing walk")
    monkeypatch.setattr(streaming, "decide_tree", broken)
    road = streaming.tree_road_plan(exp)
    assert road.refusal is None and road.refusal_resource is None
    assert not road.priced
    assert road.report_error == (
        "TypeError: unsupported operand type(s) inside the pricing walk")

    def refused(*args, **kwargs):
        raise streaming.StreamingRefused("no auto road fits", resource="memory")
    monkeypatch.setattr(streaming, "decide_tree", refused)
    road = streaming.tree_road_plan(exp)
    assert road.report_error is None
    assert road.refusal == "no auto road fits" and road.refusal_resource == "memory"
