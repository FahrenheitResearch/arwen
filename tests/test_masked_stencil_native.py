"""The native HRRR route's soil stencil in Rust against its NumPy oracle.

``gpuwm_masked_bilinear_stencil_f64`` and ``gpuwm_masked_stencil_apply_f32``
replaced the single-core NumPy builder (and its SciPy k-d tree) and the
NumPy apply that mapped HRRR soil temperature and soil moisture onto every
native HRRR target grid.  Both are kept verbatim in
:mod:`gpuwm.verify.hrrr_stencil_oracle`, and every assertion here compares
BYTES: indices, float32 weights, the applied soil, the report key by key
in its key order, and every refusal's words and facts.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest import hrrr
from gpuwm.ingest.cpu_backend import (
    MASKED_STENCIL_ENTRY,
    CpuPreprocessBackend,
    MaskedChainUnavailable,
)
from gpuwm.verify import hrrr_stencil_oracle as oracle

pytestmark = pytest.mark.requires_capability("masked_stencil_bridge")

ROOT = Path(__file__).resolve().parents[1]
EDGES = hrrr.WINDOW_EDGES


def _native() -> CpuPreprocessBackend:
    return CpuPreprocessBackend()


def _outcome(build, *args, **kwargs):
    try:
        return build(*args, **kwargs)
    except (ValueError, AssertionError) as error:
        return error


def _same_outcome(got, want):
    if isinstance(want, BaseException):
        assert type(got) is type(want), (got, want)
        assert str(got) == str(want)
        if isinstance(want, hrrr.SurfaceDonorSearchError):
            assert got.fallback_radius_cells == want.fallback_radius_cells
            assert got.required_radius_cells == want.required_radius_cells
            assert got.unresolved_targets == want.unresolved_targets
        return
    assert not isinstance(got, BaseException), got
    for mine, theirs in zip(got[:3], want[:3]):
        assert mine.dtype == theirs.dtype and mine.shape == theirs.shape
        assert mine.tobytes() == theirs.tobytes()
    assert list(got[3].items()) == list(want[3].items())
    assert json.dumps(got[3]) == json.dumps(want[3])


def _same_float32(got, want):
    """Every number bit for bit, and NaN exactly where the oracle has one.

    Which NaN survives where two meet is NumPy's loop artifact: its vector
    body keeps the first operand's, its tail the second's (so an element's
    position and the CPU's vector width decide the sign), and no other
    implementation can follow that.  The route admits no NaN at all.
    """
    assert got.dtype == want.dtype and got.shape == want.shape
    nan = np.isnan(want)
    assert np.array_equal(np.isnan(got), nan)
    assert got[~nan].tobytes() == want[~nan].tobytes()


def _pair(x, y, valid, apply, radius, closed, workers=None):
    want = _outcome(oracle._build_masked_bilinear_stencil, x, y, valid,
                    apply, fallback_radius=radius, closed_edges=closed)
    got = _outcome(hrrr._build_masked_bilinear_stencil, x, y, valid, apply,
                   fallback_radius=radius, closed_edges=closed,
                   native=_native(), workers=workers)
    _same_outcome(got, want)
    return got


def _coordinates(rng, ny, nx, shape):
    """Fractional, integer and half-integer coordinates: ties on purpose."""
    x = rng.uniform(0.0, nx - 1.0 - 1.0e-9, shape)
    y = rng.uniform(0.0, ny - 1.0 - 1.0e-9, shape)
    pick = rng.random(shape)
    x = np.where(pick < 0.2, np.floor(x), x)
    y = np.where(pick < 0.2, np.floor(y), y)
    x = np.where((pick >= 0.2) & (pick < 0.4),
                 np.minimum(np.floor(x) + 0.5, nx - 1.5), x)
    y = np.where((pick >= 0.2) & (pick < 0.4),
                 np.minimum(np.floor(y) + 0.5, ny - 1.5), y)
    return x, y


@pytest.mark.parametrize("seed", range(40))
def test_random_windows_match_the_oracle_byte_for_byte(seed):
    rng = np.random.default_rng(seed)
    ny, nx = int(rng.integers(4, 40)), int(rng.integers(4, 40))
    density = [0.0, 0.003, 0.02, 0.1, 0.5, 1.0][seed % 6]
    valid = rng.random((ny, nx)) < density
    shape = (int(rng.integers(1, 12)), int(rng.integers(1, 12)))
    x, y = _coordinates(rng, ny, nx, shape)
    apply = rng.random(shape) < 0.7
    radius = int(rng.choice([0, 1, 2, 3, 8]))
    closed = tuple(edge for edge in EDGES if rng.random() < 0.4)
    for workers in (1, 3, 64):
        _pair(x, y, valid, apply, radius, closed, workers)


def test_distance_ties_go_to_the_lowest_row_then_column():
    valid = np.zeros((20, 20), dtype=bool)
    for row, col in ((6, 10), (10, 6), (10, 14), (14, 10)):
        valid[row, col] = True
    x = np.array([[10.0, 10.0]])
    y = np.array([[10.0, 10.0]])
    apply = np.ones(x.shape, dtype=bool)
    for radius, closed in ((8, ()), (2, EDGES), (0, EDGES)):
        got = _pair(x, y, valid, apply, radius, closed)
        assert got[0][0, 0, 0] == 6 and got[1][0, 0, 0] == 10


def test_every_refusal_matches_in_words_and_facts():
    rng = np.random.default_rng(7)
    valid = rng.random((12, 12)) < 0.05
    valid[:, :] = False
    valid[1, 1] = True
    x, y = _coordinates(rng, 12, 12, (3, 4))
    apply = np.ones(x.shape, dtype=bool)
    # Unresolved: land far from the targets, open edges.
    _pair(x + 4.0 * (x < 6), y + 4.0 * (y < 6), valid, apply, 1, ())
    # No valid cell at all.
    _pair(x, y, np.zeros((12, 12), dtype=bool), apply, 2, ())
    # Non-finite, negative radius, unknown edge, off the window.
    bad = x.copy()
    bad[0, 0] = np.nan
    _pair(bad, y, valid, apply, 2, ())
    _pair(x, y, valid, apply, -1, ())
    _pair(x, y, valid, apply, 2, ("west", "up"))
    _pair(x + 20.0, y, valid, apply, 2, ())
    _pair(x - 5.0, y, valid, apply, 2, ())
    with pytest.raises(ValueError, match="equal shapes"):
        hrrr._build_masked_bilinear_stencil(x, y[:, :2], valid, apply)
    with pytest.raises(ValueError, match="must be 2-D"):
        hrrr._build_masked_bilinear_stencil(x, y, valid[0], apply)


@pytest.mark.parametrize("seed", range(6))
def test_targets_on_the_last_column_and_row_take_their_own_cell(seed):
    """A target that IS the source grid has its last column on ``nx - 1``
    and its last row on ``ny - 1``.  Both builders refused it as leaving
    the window (floor + 1 is past the edge), so the identity route's soil
    stencil could not be built on the whole native grid.  Such a point now
    takes its own cell with weight 1, spelled as the cell before with a
    unit fraction; one step past the last cell is still refused.  Rust and
    the oracle agree byte for byte, including where the own cell is not
    valid and the donor search runs."""
    rng = np.random.default_rng(100 + seed)
    ny, nx = int(rng.integers(2, 30)), int(rng.integers(2, 30))
    valid = rng.random((ny, nx)) < [1.0, 0.7, 0.3, 0.05, 1.0, 0.5][seed]
    rows, cols = np.indices((ny, nx), dtype=np.float64)
    apply = rng.random((ny, nx)) < 0.8
    closed = EDGES if seed % 2 else ()
    for workers in (1, 3):
        got = _pair(cols, rows, valid, apply, 8, closed, workers)
    if isinstance(got, BaseException):
        return
    iy, ix, weights, _report = got
    assert iy.min() >= 0 and iy.max() < ny and ix.min() >= 0 and ix.max() < nx
    own = valid & apply
    # Every valid applied target on a whole cell copies that cell: the
    # weight on it is exactly 1, on the other corners exactly 0.
    on_own = (iy == rows.astype(np.int32)) & (ix == cols.astype(np.int32))
    total = np.where(on_own, weights, np.float32(0.0)).sum(axis=0)
    assert np.all(total[own] == np.float32(1.0))
    assert np.all(np.where(on_own, np.float32(0.0), weights)[:, own] == 0.0)
    # The last column and row are the cell before with a unit fraction.
    last = own[:, -1]
    assert np.all(ix[0, last, -1] == nx - 2)
    assert np.all(ix[1, last, -1] == nx - 1)
    assert np.all(weights[1, :-1, -1][own[:-1, -1]] == np.float32(1.0))
    top = own[-1, :]
    assert np.all(iy[0, -1, top] == ny - 2)
    assert np.all(iy[2, -1, top] == ny - 1)
    # One step past the last cell is refused alike by both.
    _pair(cols + 1.0e-9, rows, valid, apply, 8, closed)
    _pair(cols, rows + 1.0e-9, valid, apply, 8, closed)


def test_a_large_sparse_window_matches_with_distant_donors_listed():
    """Many fallback targets, donors past the radius and past the listing
    cap, a whole-grid window (every edge closed) and chunks of targets
    spread over many workers."""
    rng = np.random.default_rng(11)
    ny, nx = 160, 170
    valid = rng.random((ny, nx)) < 0.0015
    x, y = _coordinates(rng, ny, nx, (90, 70))
    apply = rng.random(x.shape) < 0.9
    got = _pair(x, y, valid, apply, 8, EDGES, 64)
    report = got[3]
    assert report["distant_donor_count"] > len(report["distant_donors"])
    assert report["distant_donors_not_listed"] > 0
    assert report["distant_donors"][0]["window_reach_cells"] is None
    for workers in (1, 7):
        _pair(x, y, valid, apply, 8, EDGES, workers)


def _stencil_arrays(rng, ny, nx, shape):
    valid = rng.random((ny, nx)) < 0.3
    x, y = _coordinates(rng, ny, nx, shape)
    apply = rng.random(shape) < 0.8
    return oracle._build_masked_bilinear_stencil(
        x, y, valid, apply, fallback_radius=8, closed_edges=EDGES)


@pytest.mark.parametrize("lead", [(), (3,), (2, 2)])
def test_apply_matches_the_oracle_on_every_layer(lead):
    rng = np.random.default_rng(len(lead))
    ny, nx = 23, 29
    iy, ix, weights, report = _stencil_arrays(rng, ny, nx, (17, 13))
    field = rng.normal(280.0, 20.0, (*lead, ny, nx)).astype(np.float32)
    flat = field.reshape(-1)
    specials = [np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0e38, -1.0e-40]
    for index, pick in enumerate(rng.integers(0, flat.size, 30)):
        flat[pick] = specials[index % len(specials)]
    stencil = hrrr._CpuMaskedBilinearStencil(
        (ny, nx), iy, ix, weights, report, native=_native(), workers=5)
    want = oracle.apply_masked_bilinear_stencil(stencil, field)
    got = stencil.apply(field)
    _same_float32(got, want)
    finite = np.where(np.isfinite(field), field, np.float32(281.5))
    assert (stencil.apply(finite).tobytes()
            == oracle.apply_masked_bilinear_stencil(stencil, finite).tobytes())
    # The soil route's composition: water targets take the fill.
    select = rng.random((17, 13)) < 0.6
    skin = rng.normal(290.0, 5.0, (17, 13)).astype(np.float32)
    for fill in (skin, 1.0):
        fill_host = (np.broadcast_to(skin, want.shape) if fill is skin
                     else np.full(want.shape, 1.0, dtype=np.float32))
        composed = np.where(select, want, fill_host)
        _same_float32(stencil.apply_selected(field, select, fill), composed)


def test_the_route_maps_soil_as_the_oracle_composes_it():
    """The whole native HRRR soil mapping of a real window (HRRR's own land
    mask around a two-cell island whose donor lies past the radius)."""
    from test_hrrr_island_donor import (_HostBackend, _ISLAND, _nest_grid,
                                        _window_snapshot)

    snapshot = _window_snapshot()
    grid = _nest_grid()
    plan = hrrr._ProjectedCpuPlan(snapshot, *grid.latlon_mass(),
                                  _HostBackend())
    # HRRR's own coast on the nest (direct and renormalised cells, and
    # coastal cells the two masks disagree on), plus the island.
    landmask = (np.asarray(plan.apply(snapshot.fields["LANDSEA"],
                                      method="nearest")) >= 0.5
                ).astype(np.float64)
    for row, col in _ISLAND:
        landmask[row, col] = 1.0
    report: dict = {}
    mapped = hrrr.interpolate_hrrr_to_lambert(
        snapshot, grid, target_landmask=landmask,
        soil_mapping_report=report, surface_fallback_radius=8,
        backend=_HostBackend(), target_name="domain 2")
    source_land = np.asarray(snapshot.fields["LANDSEA"]) >= 0.5
    target_land = landmask.astype(bool)
    iy, ix, weights, want_report = oracle._build_masked_bilinear_stencil(
        plan.x_host, plan.y_host, source_land, target_land,
        fallback_radius=8, closed_edges=hrrr._native_edges_of(snapshot))
    stencil = hrrr._CpuMaskedBilinearStencil(
        plan.source_shape, iy, ix, weights, want_report)
    skin = np.asarray(mapped.fields["SKINTEMP"], dtype=np.float32)
    for name, fill in (("SOILT", np.broadcast_to(skin, (9, *skin.shape))),
                       ("SOILW", None)):
        land = oracle.apply_masked_bilinear_stencil(
            stencil, snapshot.fields[name])
        if fill is None:
            fill = np.full(land.shape, 1.0, dtype=np.float32)
        want = np.where(target_land[None], land, fill)
        got = np.asarray(mapped.fields[name])
        assert got.dtype == want.dtype and got.tobytes() == want.tobytes()
    stencil_report = report["land_stencil"]
    for key, value in want_report.items():
        if key == "distant_donors":
            for mine, theirs in zip(stencil_report[key], value):
                assert {k: mine[k] for k in theirs} == theirs
        else:
            assert stencil_report[key] == value, key


def test_a_library_without_the_stencil_is_refused_by_name_with_the_remedy():
    stale = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    stale.path = Path("/opt/old/libgpuwm_preprocess_cpu.so")
    stale.masked_stencil_entry = False
    with pytest.raises(MaskedChainUnavailable) as caught:
        stale.require_masked_stencil()
    message = str(caught.value)
    assert MASKED_STENCIL_ENTRY in message
    assert "native HRRR route" in message
    from gpuwm.bridges import cpu_bridge_remedy
    assert cpu_bridge_remedy(stale.path.name) in message
    with pytest.raises(MaskedChainUnavailable):
        hrrr._build_masked_bilinear_stencil(
            np.zeros((1, 1)), np.zeros((1, 1)), np.ones((3, 3), dtype=bool),
            np.ones((1, 1), dtype=bool), native=stale, workers=1)


def test_an_empty_target_set_is_refused_in_plain_words():
    with pytest.raises(
            ValueError,
            match="^masked-bilinear stencil has no target points to map$"):
        hrrr._build_masked_bilinear_stencil(
            np.zeros((0,)), np.zeros((0,)), np.ones((3, 3), dtype=bool),
            np.zeros((0,), dtype=bool), workers=1)


def test_a_receipt_names_no_entry_its_library_lacks():
    from gpuwm.ingest import preprocess_backend

    stale = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    stale.path = Path("/opt/old/libgpuwm_preprocess_cpu.so")
    stale.wps_masked_chain_entry = True
    stale.masked_stencil_entry = False
    receipt = preprocess_backend._masked_chain_receipt(stale)
    assert receipt["bridge"] is None
    assert MASKED_STENCIL_ENTRY in receipt["unavailable"]
    live = preprocess_backend.resolve_preprocess_backend(
        "cpu", workers=3).receipt()["masked_surface_chain"]
    assert live["bridge"] is not None
    assert live["stencil_entry"] == MASKED_STENCIL_ENTRY
    assert live["workers"] == 3


def test_the_cuda_backend_takes_host_workers_for_its_host_steps():
    from gpuwm.ingest import preprocess_backend

    backend = preprocess_backend.CudaPreprocessBackend(host_workers=5)
    native, workers = backend.wps_masked_chain_engine()
    assert workers == 5 and native.masked_stencil_entry
    with pytest.raises(ValueError, match="workers must be positive"):
        preprocess_backend.CudaPreprocessBackend(host_workers=0)
    with pytest.raises(ValueError, match="cpu_bridge applies only to the CPU"):
        preprocess_backend.resolve_preprocess_backend(
            "cuda", cpu_bridge="/tmp/libgpuwm_preprocess_cpu.so")


def test_the_sealed_distribution_runs_the_stencil_at_both_worker_counts():
    from gpuwm import native_wrf_distribution

    receipt = native_wrf_distribution._cpu_masked_stencil_self_test(
        _native())
    assert receipt["status"] == "PASS"
    assert receipt["worker_counts"] == [1, 3]
    assert len(receipt["output_sha256"]) == 64
    contract = native_wrf_distribution.distribution_contract("linux-x86_64")
    from gpuwm.ingest.cpu_backend import MASKED_STENCIL_IMPLEMENTATION
    assert (contract["preprocess_backends"]["masked_bilinear_stencil"]
            == MASKED_STENCIL_IMPLEMENTATION)


def test_nothing_at_run_time_imports_the_stencil_oracle_or_a_tree():
    """The NumPy builder is a test oracle, never a silent runtime fallback."""
    offenders = []
    for base in ("gpuwm", "tilestream", "tools"):
        for path in sorted((ROOT / base).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith("gpuwm/verify/"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "hrrr_stencil_oracle" not in text:
                continue
            for node in ast.walk(ast.parse(text)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [
                        alias.name for alias in node.names]
                if any("hrrr_stencil_oracle" in name for name in names):
                    offenders.append(relative)
    assert offenders == []
    source = (ROOT / "gpuwm" / "ingest" / "hrrr.py").read_text(
        encoding="utf-8")
    assert "cKDTree" not in source
