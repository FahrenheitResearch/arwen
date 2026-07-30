"""CPU contract tests for the Phase-5 Task-14(s) model skeleton."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields
from types import NoneType, SimpleNamespace
from typing import Literal, cast, get_type_hints

import pytest

from gpuwm.config import RunConfig
from gpuwm.core.clock import DomainClock, DomainTicks, Schedule
from gpuwm.core.model import (PERIOD_BEGIN, DomainNode, ExperimentState,
                              FeedbackScratch, NestCoupler, RestartPhase)
from gpuwm.core.state import DomainState
from gpuwm.experiment import DomainConfig
from gpuwm.static.lambert import LambertGrid


class DummyCoupler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def force(self, node: DomainNode) -> None:
        self.calls.append(("force", node.cfg.grid_id))

    def feedback_prepare(self, node: DomainNode,
                         out: FeedbackScratch) -> None:
        out.payload = ("prepared", node.cfg.grid_id)
        self.calls.append(("feedback_prepare", node.cfg.grid_id))

    def feedback_commit(self, node: DomainNode) -> None:
        self.calls.append(("feedback_commit", node.cfg.grid_id))

    def feedback_finalize(self, node: DomainNode) -> None:
        self.calls.append(("feedback_finalize", node.cfg.grid_id))


def _cfg(grid_id: int, parent_id: int, *,
         run_grid_id: int | None = None) -> DomainConfig:
    is_root = parent_id == 0
    run = RunConfig(
        nx=8, ny=8, nz=4, dx=12000.0, dy=12000.0, ztop=12000.0,
        dt=60.0, run_seconds=60.0, specified=is_root, nested=not is_root,
        grid_id=grid_id if run_grid_id is None else run_grid_id)
    return DomainConfig(
        grid_id=grid_id, parent_id=parent_id,
        i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1 if is_root else 3,
        parent_time_step_ratio=1 if is_root else 3,
        history_interval_s=60.0, run=run,
        time_step=60 if is_root else None)


def _clock(grid_id: int) -> DomainClock:
    spec = cast(DomainTicks, SimpleNamespace(grid_id=grid_id))
    return DomainClock(spec, tick_den=1, run_ticks=60)


def _grid() -> LambertGrid:
    return LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=12000.0, dy=12000.0, e_we=9, e_sn=9)


def _node(grid_id: int, *, parent: DomainNode | None = None,
          parent_id: int | None = None,
          coupler: NestCoupler | None = None,
          run_grid_id: int | None = None,
          clock_grid_id: int | None = None) -> DomainNode:
    if parent_id is None:
        parent_id = 0 if parent is None else parent.cfg.grid_id
    return DomainNode(
        cfg=_cfg(grid_id, parent_id, run_grid_id=run_grid_id),
        grid=_grid(), state=cast(DomainState, object()),
        clock=_clock(grid_id if clock_grid_id is None else clock_grid_id),
        parent=parent, children=[], coupler=coupler)


def _experiment_state(root: DomainNode,
                      *nodes: DomainNode) -> ExperimentState:
    all_nodes = (root, *nodes)
    return ExperimentState(
        root=root,
        nodes_by_grid_id={node.cfg.grid_id: node for node in all_nodes},
        schedule=cast(Schedule, object()), memory_ledger=None,
        experiment_fingerprint="fixture-fingerprint")


def test_domain_node_field_contract() -> None:
    assert [field.name for field in fields(DomainNode)] == [
        "cfg", "grid", "state", "clock", "parent", "children", "coupler"]
    assert get_type_hints(DomainNode) == {
        "cfg": DomainConfig,
        "grid": LambertGrid,
        "state": DomainState,
        "clock": DomainClock,
        "parent": DomainNode | None,
        "children": list[DomainNode],
        "coupler": NestCoupler | None,
    }


def test_experiment_state_field_contract() -> None:
    assert [field.name for field in fields(ExperimentState)] == [
        "root", "nodes_by_grid_id", "schedule", "memory_ledger",
        "experiment_fingerprint"]
    assert get_type_hints(ExperimentState) == {
        "root": DomainNode,
        "nodes_by_grid_id": Mapping[int, DomainNode],
        "schedule": Schedule,
        "memory_ledger": object | None,
        "experiment_fingerprint": str,
    }
    assert get_type_hints(ExperimentState.node)["return"] is DomainNode
    assert (get_type_hints(ExperimentState.walk_parent_first)["return"]
            == Iterator[DomainNode])
    assert PERIOD_BEGIN == "PERIOD_BEGIN"
    assert RestartPhase == Literal["PERIOD_BEGIN"]


def test_nest_coupler_is_runtime_structural_protocol() -> None:
    coupler = DummyCoupler()
    root = _node(1)
    assert isinstance(coupler, NestCoupler)
    assert get_type_hints(NestCoupler.force) == {
        "node": DomainNode, "return": NoneType}
    assert get_type_hints(NestCoupler.feedback_prepare) == {
        "node": DomainNode, "out": FeedbackScratch, "return": NoneType}
    assert get_type_hints(NestCoupler.feedback_commit) == {
        "node": DomainNode, "return": NoneType}
    assert get_type_hints(NestCoupler.feedback_finalize) == {
        "node": DomainNode, "return": NoneType}

    scratch = FeedbackScratch()
    coupler.force(root)
    coupler.feedback_prepare(root, scratch)
    coupler.feedback_commit(root)
    coupler.feedback_finalize(root)
    assert scratch.payload == ("prepared", 1)
    assert coupler.calls == [
        ("force", 1),
        ("feedback_prepare", 1),
        ("feedback_commit", 1),
        ("feedback_finalize", 1),
    ]


def test_parent_first_walk_on_four_node_chain() -> None:
    root = _node(1)
    d02 = _node(2, parent=root)
    d03 = _node(3, parent=d02)
    d04 = _node(4, parent=d03)
    root.children.append(d02)
    d02.children.append(d03)
    d03.children.append(d04)
    state = _experiment_state(root, d02, d03, d04)

    assert [node.cfg.grid_id for node in state.walk_parent_first()] == [
        1, 2, 3, 4]
    assert state.node(3) is d03
    with pytest.raises(KeyError):
        state.node(99)


def test_parent_first_walk_on_branching_tree() -> None:
    root = _node(1)
    d02 = _node(2, parent=root)
    d03 = _node(3, parent=root)
    d04 = _node(4, parent=d02)
    root.children.extend((d02, d03))
    d02.children.append(d04)

    state = _experiment_state(root, d02, d03, d04)
    assert [node.cfg.grid_id for node in state.walk_parent_first()] == [
        1, 2, 4, 3]


@pytest.mark.parametrize(
    ("make_bad_node", "message"),
    [
        (lambda: _node(1, run_grid_id=9), "grid_id mismatch"),
        (lambda: _node(1, clock_grid_id=9), "grid_id mismatch"),
        (lambda: _node(2, parent=_node(1), parent_id=0),
         "root DomainNode.*parent=None"),
        (lambda: _node(1, coupler=DummyCoupler()),
         "root DomainNode.*coupler=None"),
        (lambda: _node(2, parent_id=1), "parent-before-child"),
        (lambda: _node(2, parent=_node(1), parent_id=9),
         "parent_id mismatch"),
    ],
)
def test_domain_node_rejects_inconsistent_links(make_bad_node,
                                                message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_bad_node()
