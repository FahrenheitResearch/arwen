"""Rank host-thread placement and the shared-lock notice (tilestream.ranks).

THE BREAKAGE THESE GUARD: on a two-socket box the rank threads floated
across sockets while their cards sat on one node, so every launch and
readback crossed the socket link.  A rank thread must be bound to its card's
local CPUs on a multi-node box, never bound on a one-node box or when the
firmware reports no locality (-1 is unknown, not node 0), and never bound to
CPUs this process may not use.  And a multi-card run on an interpreter whose
lock is on must say so once, by name, instead of silently taking turns.
"""
import pytest

from tilestream import node_probe
from tilestream import ranks


def _probe(monkeypatch, rows, allowed):
    monkeypatch.setattr(ranks.sys, "platform", "linux")
    monkeypatch.setattr(ranks.os, "sched_setaffinity", lambda *a: None, raising=False)
    monkeypatch.setattr(ranks.os, "sched_getaffinity", lambda pid: set(allowed), raising=False)
    monkeypatch.setattr(node_probe, "gpu_affinity", lambda: rows)


def _row(device, node, local, required):
    return dict(device=device, pci=f"0000:{device:02x}:00.0", sysfs_present=True,
                numa_node=node, local_cpus=local, binding_required=required,
                usable_cpus=local)


def test_two_socket_box_binds_each_rank_to_its_card_node(monkeypatch):
    node0, node2 = list(range(0, 4)), list(range(64, 68))
    _probe(monkeypatch, [_row(0, 0, node0, True), _row(3, 2, node2, True)],
           allowed=node0 + node2 + [100])
    rows = ranks.rank_placements([0, 3])
    assert [r["cpus"] for r in rows] == [node0, node2]
    assert [r["numa_node"] for r in rows] == [0, 2]
    assert all(r["reason"] is None and r["applied"] is False for r in rows)


def test_binding_never_leaves_the_allowed_cpu_set(monkeypatch):
    _probe(monkeypatch, [_row(1, 1, [32, 33, 34], True)], allowed=[33, 200])
    assert ranks.rank_placements([1])[0]["cpus"] == [33]
    _probe(monkeypatch, [_row(1, 1, [32, 33, 34], True)], allowed=[200])
    row = ranks.rank_placements([1])[0]
    assert row["cpus"] is None and "allowed" in row["reason"]


@pytest.mark.parametrize("node, required", [(0, False), (-1, False), (None, False)])
def test_one_node_or_unknown_locality_binds_nothing(monkeypatch, node, required):
    _probe(monkeypatch, [_row(0, node, [0, 1], required)], allowed=[0, 1, 2])
    row = ranks.rank_placements([0])[0]
    assert row["cpus"] is None and "unbound" in row["reason"]


def test_repeated_cards_share_one_locality(monkeypatch):
    _probe(monkeypatch, [_row(0, 0, [0, 1], True), _row(1, 2, [8, 9], True)],
           allowed=range(16))
    assert [r["cpus"] for r in ranks.rank_placements([0, 1, 0, 1])] == [
        [0, 1], [8, 9], [0, 1], [8, 9]]


def test_unreadable_probe_binds_nothing(monkeypatch):
    _probe(monkeypatch, [], allowed=[0])

    def broken():
        raise OSError("sysfs unreadable")

    monkeypatch.setattr(node_probe, "gpu_affinity", broken)
    rows = ranks.rank_placements([0, 1])
    assert all(r["cpus"] is None and "could not be read" in r["reason"] for r in rows)


def test_no_affinity_platform_says_so(monkeypatch):
    monkeypatch.setattr(ranks.sys, "platform", "win32")
    rows = ranks.rank_placements([0])
    assert rows[0]["cpus"] is None and "unavailable" in rows[0]["reason"]


def test_shared_lock_notice_is_given_once_and_names_the_remedy(monkeypatch, capsys):
    monkeypatch.setattr(ranks, "_GIL_NOTICE_GIVEN", False)
    report = dict(python="3.12.3", free_threaded_build=False, gil_enabled=True,
                  python_gil_env=None)
    ranks._gil_notice(4, report)
    ranks._gil_notice(4, report)
    err = capsys.readouterr().err
    assert err.count("share one interpreter lock") == 1
    assert "python3.14t" in err


def test_no_notice_for_one_rank_or_a_free_lock(monkeypatch, capsys):
    monkeypatch.setattr(ranks, "_GIL_NOTICE_GIVEN", False)
    ranks._gil_notice(1, dict(python="3.12.3", free_threaded_build=False,
                              gil_enabled=True, python_gil_env=None))
    ranks._gil_notice(4, dict(python="3.14.7", free_threaded_build=True,
                              gil_enabled=False, python_gil_env="0"))
    assert capsys.readouterr().err == ""


def test_free_threaded_build_that_relocked_names_python_gil(monkeypatch, capsys):
    monkeypatch.setattr(ranks, "_GIL_NOTICE_GIVEN", False)
    ranks._gil_notice(2, dict(python="3.14.7", free_threaded_build=True,
                              gil_enabled=True, python_gil_env=None))
    assert "PYTHON_GIL=0" in capsys.readouterr().err


@pytest.mark.parametrize("nodes, expected", [
    ({0: 0, 1: 0, 2: 0, 3: 2}, "staged"),
    ({0: 0, 1: 0, 2: 0, 3: 0}, "peer"),
    ({0: -1, 1: -1, 2: -1, 3: -1}, "peer"),
    (None, "peer"),
    ({0: 0, 1: None, 2: 0, 3: -1}, "peer")])
def test_auto_stages_every_seam_when_cards_span_numa_nodes(nodes, expected):
    """Concurrent peer copies that mix in-node and cross-socket pairs collapsed
    (245 ms against 14.8 ms staged for the HRRR 2x2 exchange); unknown
    locality never counts as a second node."""
    devices = [0, 1, 2, 3]
    peers = {(a, b): True for a in devices for b in devices if a != b}
    paths = ranks.choose_transports(devices, "auto", peers, nodes=nodes)
    assert {paths[(a, b)] for a in devices for b in devices if a != b} == {expected}
    assert all(paths[(a, a)] == "local" for a in devices)
    assert ranks.choose_transports(devices, "peer", peers, nodes=nodes)[(0, 3)] == "peer"
    assert ranks.spans_numa_nodes(nodes) is (expected == "staged")


def test_a_replaced_exchange_keeps_the_whole_exchange_after_the_step():
    """The rank gate's no-exchange and stale controls replace exchange_events
    on the run.  If the threaded sweep moved seams through its own schedule
    anyway, a control that must change the answer would silently pass."""
    import inspect
    source = inspect.getsource(ranks.RankedRun.sweep)
    assert '"exchange_events" in vars(self)' in source
    assert "schedule = None if timing or replaced else self.exchange_schedule()" in source
