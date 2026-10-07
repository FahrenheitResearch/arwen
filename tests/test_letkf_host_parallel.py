"""The host-parallel LETKF solve gives the single-process bytes.

Breakage prevented: a parallel host analysis that differs from the
single-process one by rounding would make the host path useless as the
oracle the device solves are judged against, and would make a cycle's
analysis depend on how many cores the box had.
"""
from dataclasses import replace

import numpy as np
import pytest

from gpuwm.da import letkf
from gpuwm.da import letkf_host_parallel as hp
from test_radar_assimilation import grid, world  # noqa: F401  (fixtures)
from gpuwm.da.letkf import (GridGeometry, GriddedObs, LetkfConfig,
                           LetkfDiagnostics, LetkfError, Localization)

needs_fork = pytest.mark.skipif(not hp.fork_available(),
                                reason="host-parallel solve needs fork")


def geodesic_problem(*, fields=4, batches=5, members=9, ny=10, nx=11, nz=30,
                     seed=4410):
    """Terrain-following, geodesic, windowed and unwindowed batches."""
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    jj, ii = np.mgrid[0:ny, 0:nx]
    lat = 31.0 + 0.027 * jj + 0.001 * ii
    lon = -89.0 + 0.0315 * ii - 0.0005 * jj
    terrain = (jj * 13.0 + ii * 7.0) % 90.0
    heights = np.arange(nz)[:, None, None] * 300.0 + terrain + 50.0
    grid = GridGeometry(dx_m=3000.0, dy_m=3000.0, heights_m=heights,
                        lat_deg=lat, lon_deg=lon)
    prior = {f"field_{n}": rng.normal(size=(members, *shape)) + n
             for n in range(fields)}
    obs = []
    for n in range(batches):
        mask = rng.random(shape) < 0.03
        mask[0, 0, 0] = mask[-1, -1, -1] = True
        sim = 0.8 * prior["field_0"] + 0.02 * prior[f"field_{fields - 1}"] ** 2
        values = sim.mean(axis=0) + rng.normal(size=shape) * 0.3
        errors = np.full(shape, 0.7 + 0.05 * n)
        sim = np.where(mask[None], sim, np.nan)
        values = np.where(mask, values, np.nan)
        errors = np.where(mask, errors, np.nan)
        window = (1, ny - 2, 2, nx - 1) if n % 2 else None  # inclusive ends
        if window:
            values, errors, sim, mask = (
                a[..., 1:ny - 1, 2:nx] for a in (values, errors, sim, mask))
        loc = Localization(9000.0 + 1500.0 * (n % 2), 2000.0) if n % 3 == 2 else None
        obs.append(GriddedObs(f"observation_{n}", values, errors, sim, mask,
                              localization=loc, window=window))
    cfg = LetkfConfig(Localization(12000.0, 3000.0), tuple(prior), 0.6,
                      chunk_points=257, memory_budget_mib=48,
                      eigensolver="library", host_workers=1)
    return prior, obs, grid, cfg


def _solve(prior, obs, grid, cfg, monkeypatch=None):
    """One analysis; with ``monkeypatch``, the fully serial reference.

    ``GPUWM_DA_HOST_WORKERS=1`` keeps the chunk loop in one process AND the
    whole-field passes on one thread: the plain single-core analysis.
    """
    diag = LetkfDiagnostics()
    if monkeypatch is None:
        out = letkf.analyze(prior, obs, grid, cfg, diag, solve_namespace=np)
        return out, diag
    with monkeypatch.context() as m:
        m.setenv(hp.HOST_WORKERS_ENV, "1")
        assert hp.field_threads(14) == 1
        out = letkf.analyze(prior, obs, grid, cfg, diag, solve_namespace=np)
    return out, diag


@needs_fork
@pytest.mark.parametrize("search", ["forward", "index"])
@pytest.mark.parametrize("workers", [2, 3, 7])
def test_every_worker_count_gives_the_single_process_bytes(workers, search,
                                                           monkeypatch):
    monkeypatch.delenv(hp.HOST_WORKERS_ENV, raising=False)
    prior, obs, grid, cfg = geodesic_problem()
    cfg = replace(cfg, neighbor_search=search)
    reference, ref_diag = _solve(prior, obs, grid, cfg, monkeypatch)
    assert ref_diag.host_workers == 1
    assert ref_diag.neighbor_search == search
    assert ref_diag.host_row_blocks == 0 and ref_diag.host_transform_pieces == 0
    result, diag = _solve(prior, obs, grid, replace(cfg, host_workers=workers))
    assert diag.host_workers == workers
    # The budget is shared, so the workers really did split packed solves
    # into pieces, and on the forward walk chunks into row blocks; the
    # roster (built once, before the fork) has no per-chunk walk to split.
    if search == "forward":
        assert diag.host_row_blocks > 0
    else:
        assert diag.host_row_blocks == 0
        # The roster itself was built level by level on the workers, and
        # the reference built it in one process: the bytes below compare
        # the two builds as well as the two chunk loops.
        assert ref_diag.neighbor_index["workers"] == 1
        assert diag.neighbor_index["workers"] == workers
        assert diag.neighbor_index["neighbours"] == \
            ref_diag.neighbor_index["neighbours"]
    assert diag.host_transform_pieces > ref_diag.device_chunks
    for name in prior:
        assert result[name].dtype == reference[name].dtype
        assert result[name].tobytes() == reference[name].tobytes(), name
    for name in ("active_points", "max_local_obs", "batches",
                 "device_chunks", "staging_bytes", "reach_chunks_skipped",
                 "reach_span_splits", "reach_batches_evaluated",
                 "reach_batches_skipped"):
        assert getattr(diag, name) == getattr(ref_diag, name), name
    assert diag.mean_increment_rms == ref_diag.mean_increment_rms
    assert diag.posterior_spread == ref_diag.posterior_spread


@needs_fork
def test_inflated_prior_and_float32_state_stay_identical(monkeypatch):
    monkeypatch.delenv(hp.HOST_WORKERS_ENV, raising=False)
    # rho != 1 starts every increment from the inactive-point transform,
    # which the shared output must carry in, not overwrite with zeros.
    prior, obs, grid, cfg = geodesic_problem(fields=3, batches=3, seed=77)
    prior = {k: v.astype(np.float32) for k, v in prior.items()}
    obs = [replace(o, simulated=np.asarray(o.simulated, np.float32)) for o in obs]
    cfg = replace(cfg, prior_inflation=1.15, relaxation="rtpp")
    reference, _ = _solve(prior, obs, grid, cfg, monkeypatch)
    result, diag = _solve(prior, obs, grid, replace(cfg, host_workers=4))
    assert diag.host_workers == 4
    for name in prior:
        assert result[name].dtype == np.float32
        assert result[name].tobytes() == reference[name].tobytes(), name


@needs_fork
def test_a_worker_failure_reaches_the_caller_as_the_same_error():
    prior, obs, grid, cfg = geodesic_problem(fields=2, batches=2, seed=5)
    bad = obs[0]
    sim = np.array(bad.simulated, copy=True)
    hit = np.argwhere(np.asarray(bad.mask))[-1]
    sim[(slice(None), *hit)] = np.inf
    obs = [replace(bad, simulated=sim)] + obs[1:]
    with pytest.raises(LetkfError):
        _solve(prior, obs, grid, cfg)
    with pytest.raises(LetkfError):
        _solve(prior, obs, grid, replace(cfg, host_workers=3))


def test_field_passes_on_threads_keep_order_and_the_first_refusal(monkeypatch):
    monkeypatch.delenv(hp.HOST_WORKERS_ENV, raising=False)
    monkeypatch.setattr(hp, "available_cpus", lambda: 256)
    assert hp.field_threads(14) == 14
    assert hp.field_threads(40) == hp.FIELD_THREADS_CAP
    assert hp.map_ordered(lambda x: x * x, range(9), 4) == [x * x for x in range(9)]

    def boom(x):
        if x in (3, 6):
            raise KeyError(x)
        return x
    with pytest.raises(KeyError) as caught:
        hp.map_ordered(boom, range(9), 4)
    assert caught.value.args == (3,)
    # A non-finite prior field is refused by name, the first in field order.
    prior, obs, grid, cfg = geodesic_problem(fields=4, batches=1)
    prior["field_2"][0, 0, 0, 0] = np.nan
    prior["field_3"][0, 0, 0, 0] = np.inf
    with pytest.raises(LetkfError, match="field_2"):
        _solve(prior, obs, grid, cfg)


def test_resolver_sizes_from_the_request_the_environment_and_the_work(monkeypatch):
    monkeypatch.delenv(hp.HOST_WORKERS_ENV, raising=False)
    monkeypatch.setattr(hp, "fork_available", lambda: True)
    monkeypatch.setattr(hp, "available_cpus", lambda: 256)
    big = hp.HOST_PARALLEL_MIN_PAIRS
    assert hp.resolve_host_workers(None, tasks=500, pairs=big)[0] == hp.HOST_WORKERS_AUTO_CAP
    assert hp.resolve_host_workers(None, tasks=10, pairs=big)[0] == 10
    workers, reason = hp.resolve_host_workers(None, tasks=500, pairs=big - 1)
    assert workers == 1 and "below" in reason
    # An explicit request is honoured even for a small analysis.
    assert hp.resolve_host_workers(8, tasks=500, pairs=1)[0] == 8
    assert hp.resolve_host_workers(1, tasks=500, pairs=big)[0] == 1
    monkeypatch.setenv(hp.HOST_WORKERS_ENV, "32")
    workers, reason = hp.resolve_host_workers(8, tasks=500, pairs=1)
    assert workers == 32 and hp.HOST_WORKERS_ENV in reason
    monkeypatch.setenv(hp.HOST_WORKERS_ENV, "0")
    with pytest.raises(ValueError):
        hp.resolve_host_workers(None, tasks=500, pairs=big)
    monkeypatch.delenv(hp.HOST_WORKERS_ENV)
    monkeypatch.setattr(hp, "fork_available", lambda: False)
    workers, reason = hp.resolve_host_workers(8, tasks=500, pairs=big)
    assert workers == 1 and "fork" in reason


def test_config_refuses_a_worker_count_below_one():
    prior, obs, grid, cfg = geodesic_problem(fields=1, batches=1)
    with pytest.raises(LetkfError):
        replace(cfg, host_workers=0)


def test_piece_and_block_sizes_default_small_and_never_exceed_the_share(monkeypatch):
    monkeypatch.delenv(hp.PIECE_MIB_ENV, raising=False)
    monkeypatch.delenv(hp.BLOCK_MIB_ENV, raising=False)
    share = 224 << 20
    assert hp.piece_bytes(share) == hp.PIECE_BYTES_DEFAULT
    assert hp.block_bytes(share) == hp.BLOCK_BYTES_DEFAULT
    assert hp.piece_bytes(1 << 20) == 1 << 20
    monkeypatch.setenv(hp.PIECE_MIB_ENV, "2")
    monkeypatch.setenv(hp.BLOCK_MIB_ENV, "1000")
    assert hp.piece_bytes(share) == 2 << 20
    assert hp.block_bytes(share) == share


@needs_fork
def test_the_whole_radar_analysis_is_byte_identical_and_says_how_it_ran(
        world, grid, monkeypatch):
    # End to end through assimilate_radar_grid: positivity, restagger and the
    # receipts see the same increments whichever way the solve ran.
    from gpuwm.da.radar_assimilation import make_assimilate
    from test_radar_assimilation import _config

    cfg = _config(chunk_points=97)
    monkeypatch.setenv(hp.HOST_WORKERS_ENV, "1")
    reference, ref_prov = make_assimilate(world.obs_path, grid, cfg)(
        0, world.member_states)
    monkeypatch.setenv(hp.HOST_WORKERS_ENV, "3")
    result, prov = make_assimilate(world.obs_path, grid, cfg)(
        0, world.member_states)
    assert ref_prov["filter"]["host_workers"] == 1
    assert prov["filter"]["host_workers"] == 3
    assert hp.HOST_WORKERS_ENV in prov["filter"]["host_workers_reason"]
    assert set(result) == set(reference)
    for index in reference:
        for name in reference[index]:
            assert (result[index][name].tobytes()
                    == reference[index][name].tobytes()), (index, name)


def _dying_call(index):
    """Stand-in for hp._call whose worker dies on its second task."""
    import os
    if index == hp._DIE_AT:
        os._exit(9)
    return hp._TASK(index)


@needs_fork
def test_a_lost_worker_raises_instead_of_hanging(monkeypatch):
    """Breakage: a SIGKILLed Pool worker left imap_unordered waiting
    forever, so the cycle hung holding every card lock."""
    monkeypatch.setattr(hp, "_DIE_AT", 3, raising=False)
    monkeypatch.setattr(hp, "_call", _dying_call)
    got = []
    with pytest.raises(hp.HostWorkerLost):
        for value in hp.run(lambda i: i * i, range(8), 2):
            got.append(value)
    assert hp._TASK is None


@needs_fork
@pytest.mark.parametrize("search", ["forward", "index"])
def test_a_lost_worker_falls_back_to_the_single_process_bytes(search,
                                                              monkeypatch):
    monkeypatch.delenv(hp.HOST_WORKERS_ENV, raising=False)
    prior, obs, grid, cfg = geodesic_problem()
    cfg = replace(cfg, neighbor_search=search)
    reference, ref_diag = _solve(prior, obs, grid, cfg, monkeypatch)
    monkeypatch.setattr(hp, "_DIE_AT", 1, raising=False)
    monkeypatch.setattr(hp, "_call", _dying_call)
    result, diag = _solve(prior, obs, grid, replace(cfg, host_workers=3))
    assert diag.host_workers == 1
    assert "HostWorkerLost" in diag.host_workers_reason
    for name in prior:
        assert result[name].tobytes() == reference[name].tobytes(), name
    for name in ("active_points", "max_local_obs", "batches",
                 "device_chunks", "reach_chunks_skipped",
                 "reach_batches_evaluated"):
        assert getattr(diag, name) == getattr(ref_diag, name), name
