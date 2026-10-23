"""The whole-replay routing kernel is byte-identical to the per-step launches.

Breakage it guards (fix/hydro-attach-replay, 2026-10-22): the routed spin-up
replay at hydro attach launched 25,920 routing steps one kernel at a time from
one host thread (17.5 s on a 5070 Ti host, 29.5 s on a 4090 host, before
forecast step 1). gpuwm.hydro.routed_init now runs the spin-up in one launch;
this pins its result to the per-step sequence it replaced (the lateral kernel
per hour, then router.launch per routing step) on a synthetic dendritic
network with a level-pool reservoir hook, on a card that holds every node's
worker at once and on one that does not.
"""
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.hydro import routed_init, troute_mc
from gpuwm.hydro.reservoirs import CONTROLS, PARAMETERS, SOURCE_PIN, Reservoirs

HOURS = 5
DT = 300.0
HOOK_LEVEL = 2


def _network(seed):
    """Levels of MC nodes, each draining into the next level; one reservoir
    node after level HOOK_LEVEL feeding a level HOOK_LEVEL + 1 node."""
    rng = np.random.default_rng(seed)
    sizes = [40, 20, 10, 5, 3, 2, 1, 1, 1]
    levels, start = [], 0
    for size in sizes:
        levels.append(list(range(start, start + size)))
        start += size
    reservoir = start                     # state row for the lake
    n = start + 1
    upstream = {i: [] for i in range(n)}
    for k in range(1, len(levels)):
        for j, node in enumerate(levels[k - 1]):
            upstream[levels[k][j % len(levels[k])]].append(node)
    # The lake takes two level-HOOK_LEVEL reaches away from their receiver
    # and releases into a level HOOK_LEVEL + 1 reach.
    taken = levels[HOOK_LEVEL][:2]
    for node in taken:
        for receiver in levels[HOOK_LEVEL + 1]:
            if node in upstream[receiver]:
                upstream[receiver].remove(node)
    upstream[reservoir] = list(taken)
    upstream[levels[HOOK_LEVEL + 1][0]].append(reservoir)
    offsets = np.zeros(n + 1, np.int32)
    for i in range(n):
        offsets[i + 1] = offsets[i] + len(upstream[i])
    indices = np.asarray([u for i in range(n) for u in upstream[i]], np.int32)
    ordered = np.asarray([node for level in levels for node in level], np.int32)
    edges = [0]
    for level in levels:
        edges.append(edges[-1] + len(level))
    parameters = np.empty((n, 8), np.float32)
    parameters[:, 0] = rng.uniform(1500, 4000, n)      # dx
    parameters[:, 1] = rng.uniform(4, 30, n)           # bw
    parameters[:, 2] = parameters[:, 1] * 1.5          # tw
    parameters[:, 3] = parameters[:, 1] * 4            # twcc
    parameters[:, 4] = rng.uniform(0.03, 0.06, n)      # n
    parameters[:, 5] = parameters[:, 4] * 2            # ncc
    parameters[:, 6] = rng.uniform(0.5, 2.0, n)        # cs
    parameters[:, 7] = rng.uniform(1e-4, 5e-3, n)      # s0
    state = np.zeros((n, 3), np.float32)
    state[:, 0] = rng.uniform(0.5, 20, n)
    state[:, 1] = rng.uniform(0.2, 1.0, n)
    state[:, 2] = rng.uniform(0.2, 1.5, n)
    depth = rng.uniform(0, 4e-3, (n, HOURS))           # one catchment per node, m/h
    depth[rng.random((n, HOURS)) < 0.3] = 0.0
    area = rng.uniform(2, 40, n)                       # km2
    return dict(n=n, reservoir=reservoir, offsets=offsets, indices=indices, ordered=ordered,
                edges=edges, parameters=parameters, state=state, depth=depth, area=area)


def _reservoirs():
    arrays = {key: cp.zeros(1, cp.float32) for key in (
        "previous_inflow", "inflow", "lateral", "initial_height",
        "levelpool_release", "levelpool_height")}
    arrays["height"] = cp.full(1, 9.6, cp.float32)
    arrays["parameters"] = cp.asarray(
        [[25], [10], [12], [0.4], [8], [60], [2], [0.1], [1.5]], dtype=cp.float32)
    arrays.update({key: cp.zeros(1, cp.int32) for key in (
        "indices", "dynamic_type", "active_source", "fault")})
    arrays["types"] = cp.ones(1, cp.int32)
    arrays["offsets"] = cp.zeros(2, cp.int64)
    arrays.update({key: cp.zeros(1, cp.float64) for key in ("clock", "release", "assimilated")})
    arrays["carriers"] = cp.zeros((3, 1), cp.float64)
    arrays["controls"] = cp.zeros((8, 1), cp.float64)
    arrays["series"] = cp.empty(0, cp.float64)
    arrays["times"] = cp.empty(0, cp.float64)
    metadata = dict(parameter_columns=PARAMETERS, control_columns=CONTROLS,
                    source_pin=SOURCE_PIN, validated_csr=True,
                    network_sha256="synthetic", parameters_sha256="synthetic",
                    series_sha256="synthetic", time_origin="synthetic",
                    reservoir_order_sha256="synthetic")
    return Reservoirs(arrays, metadata)


def _build(net, root):
    """Router + lake hook + runoff + network packet, freshly from ``net``."""
    n = net["n"]
    lateral = cp.zeros(n, cp.float32)
    router = troute_mc.MuskingumCunge(
        dt_seconds=DT, parameters=cp.asarray(net["parameters"]), state=cp.asarray(net["state"]),
        lateral=lateral, upstream_offsets=cp.asarray(net["offsets"]),
        upstream_indices=cp.asarray(net["indices"]), ordered_nodes=cp.asarray(net["ordered"]),
        level_offsets=net["edges"], boundary_previous=cp.zeros(n, cp.float32),
        boundary_current=cp.zeros(n, cp.float32),
        identity=dict(network_sha256="synthetic", parameters_sha256="synthetic",
                      ordering_sha256={}, time_origin="synthetic"),
        reach_ids=cp.arange(n, dtype=cp.int64))
    reservoirs = _reservoirs()
    hook = troute_mc.ReservoirLevel(
        router, reservoirs, rows=cp.zeros(1, cp.int32),
        node_indices=cp.asarray([net["reservoir"]], cp.int32),
        selection_sha256="synthetic", mapping_sha256="synthetic", validated_mapping=True)
    router.bind_level_hooks({HOOK_LEVEL: [hook]})
    runoff = SimpleNamespace(
        arrays=dict(area=cp.asarray(net["area"]), offsets=cp.arange(n + 1, dtype=cp.int64),
                    indices=cp.arange(n, dtype=cp.int32), routing_lateral=lateral,
                    status=cp.zeros(n, cp.int32), routing_discharge_m3_s=cp.zeros(n, cp.float32),
                    routing_volume_m3=cp.zeros(n, cp.float32)),
        restart_identity={"area_units": "km2"})
    payload = np.ascontiguousarray(net["depth"]).tobytes()
    (root / "runoff.f64").write_bytes(payload)
    declaration = dict(hours=HOURS, payload=dict(
        path="runoff.f64", dtype="<f8", shape=[n, HOURS], bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest()))
    network = SimpleNamespace(manifest={"routing": {"initial_replay": declaration}},
                              catchment_ids=tuple(range(n)), node_ids=tuple(range(n)))
    return router, reservoirs, runoff, network


def _stepwise(router, reservoirs, runoff, network):
    """The per-step launch sequence the replay used to enqueue from the host."""
    from gpuwm.core.ieee_module import compile_module
    from gpuwm.numerics import is_strict
    flags = ["--std=c++17", "--ftz=false"]
    if is_strict():
        flags += ["--fmad=false", "--prec-div=true", "--prec-sqrt=true"]
    source = (Path(routed_init.__file__).with_name("kernels") / "runoff_replay.cu").read_text()
    kernel = compile_module(source, flags, "test.runoff.replay").get_function("replay_lateral")
    a, n = runoff.arrays, len(network.node_ids)
    reservoirs.arrays["clock"].fill(-HOURS * 3600.)
    depth = cp.asarray(np.ascontiguousarray(_stepwise.depth))
    for hour in range(HOURS):
        kernel(((n + 127) // 128,), (128,), (depth, a["area"], a["offsets"], a["indices"],
               a["routing_lateral"], a["status"], np.int32(n), np.int32(HOURS),
               np.int32(hour), np.float64(1e6)))
        for _ in range(int(3600 / DT)):
            router.launch(DT)
    cp.cuda.Device().synchronize()


def _carriers(router, reservoirs, runoff):
    out = {"router/" + k: v for k, v in router.arrays.items() if isinstance(v, cp.ndarray)}
    out.update({"reservoirs/" + k: v for k, v in reservoirs.arrays.items()})
    out["runoff/status"] = runoff.arrays["status"]
    return {k: v.get().tobytes() for k, v in out.items()}


@pytest.mark.parametrize("resident", [None, 3])
def test_whole_replay_is_byte_identical_to_the_per_step_launches(monkeypatch, resident):
    net = _network(7)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        router, reservoirs, runoff, network = _build(net, root)
        _stepwise.depth = net["depth"]
        _stepwise(router, reservoirs, runoff, network)
        expected = _carriers(router, reservoirs, runoff)
        assert np.isfinite(router.arrays["state"].get()).all()
        assert int(reservoirs.arrays["clock"].get()[0]) == 0

        router, reservoirs, runoff, network = _build(net, root)
        if resident is not None:
            # A card that holds 3 blocks (12 workers) for 85 nodes: each
            # worker owns several nodes, in step order.
            monkeypatch.setattr(routed_init, "_resident_blocks", lambda kernel: resident)
        record = routed_init.replay(router=router, network=network, runoff=runoff,
                                    reservoirs=reservoirs, manifest=str(root / "manifest.json"))
        actual = _carriers(router, reservoirs, runoff)
    assert record["routing_steps"] == HOURS * int(3600 / DT)
    workers = net["n"] if resident is None else resident * routed_init.BLOCK // 32
    assert record["parallelism"].startswith(
        f"GPU: one launch for all {record['routing_steps']} routing steps, "
        f"{workers} warp workers for {net['n']} nodes")
    # The replay zeroes the lateral producer sums after its last step.
    for key in ("router/lateral",):
        expected.pop(key)
        actual.pop(key)
    assert set(actual) == set(expected)
    different = sorted(k for k in expected if actual[k] != expected[k])
    assert different == []
