"""The lake skin search and the water-temperature blends in Rust against NumPy.

``gpuwm_lake_water_nearest_f64``, ``gpuwm_masked_bilinear_blend_f64``,
``gpuwm_component_fill_f64`` and ``gpuwm_overlay_bilinear_sample_f64``
replaced single-core NumPy data steps: the per-lake nearest source water
search of ``interpolate_lake_skin_temperature``, the renormalised donor
blend and the in-component hole fill of the water-temperature assembly,
and the corner blend of the water-temperature overlay sample.  The NumPy
code is kept verbatim in :mod:`gpuwm.verify.water_blend_oracle`, and
every assertion here compares BYTES at 1, 3 and 64 workers.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest import water_overlay, water_temperature
from gpuwm.ingest.cpu_backend import (
    LAKE_WATER_NEAREST_ENTRY,
    WATER_BLEND_ENTRY,
    CpuPreprocessBackend,
    MaskedChainUnavailable,
)
from gpuwm.verify import water_blend_oracle as oracle

pytestmark = pytest.mark.requires_capability("water_blend_bridge")

ROOT = Path(__file__).resolve().parents[1]
WORKERS = (1, 3, 64)


def _native() -> CpuPreprocessBackend:
    return CpuPreprocessBackend()


def _same(got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert got.tobytes() == want.tobytes()


def _oracle_lakes(skin, water, y, x):
    y = np.asarray(y, dtype=np.float64).reshape(1, -1)
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    lakes = np.ones(y.shape, dtype=bool)
    return oracle.lake_skin_search(skin, water, y, x, lakes).ravel()


# ---- lake skin search -------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_random_lake_searches_match_the_oracle(seed):
    rng = np.random.default_rng(seed)
    ny, nx = (int(v) for v in rng.integers(5, 90, size=2))
    density = [0.0005, 0.003, 0.02, 0.2][seed % 4]
    water = rng.random((ny, nx)) < density
    water[rng.integers(ny), rng.integers(nx)] = True
    skin = 250.0 + 40.0 * rng.random((ny, nx))
    n = 300
    # Integers, halves and quarters force distance ties; random values
    # exercise the pow bound.
    y = np.where(rng.random(n) < 0.4,
                 np.round(rng.uniform(0, ny - 1, n) * 4.0) / 4.0,
                 rng.uniform(0, ny - 1, n))
    x = np.where(rng.random(n) < 0.4,
                 np.round(rng.uniform(0, nx - 1, n) * 4.0) / 4.0,
                 rng.uniform(0, nx - 1, n))
    want = _oracle_lakes(skin, water, y, x)
    for workers in WORKERS:
        _same(_native().lake_water_nearest(skin, water, y, x,
                                           workers=workers), want)


def test_lake_search_ties_take_the_first_cell_in_row_major_order():
    skin = np.arange(900, dtype=np.float64).reshape(30, 30)
    water = np.zeros((30, 30), dtype=bool)
    # Four cells at distance 10 from (15, 15), one in each direction.
    for j, i in ((5, 15), (15, 5), (15, 25), (25, 15)):
        water[j, i] = True
    y = np.array([15.0])
    x = np.array([15.0])
    want = _oracle_lakes(skin, water, y, x)
    assert want[0] == skin[5, 15]
    for workers in WORKERS:
        _same(_native().lake_water_nearest(skin, water, y, x,
                                           workers=workers), want)


def test_lake_search_over_a_single_water_cell_and_a_full_window():
    skin = np.full((3, 200), 280.0)
    skin[1, 199] = 301.5
    water = np.zeros((3, 200), dtype=bool)
    water[1, 199] = True
    y = np.array([0.0, 2.0, 1.0])
    x = np.array([0.0, 0.25, 198.9])
    want = _oracle_lakes(skin, water, y, x)
    for workers in WORKERS:
        _same(_native().lake_water_nearest(skin, water, y, x,
                                           workers=workers), want)


def test_a_source_without_water_is_refused_as_the_oracle_refuses():
    skin = np.ones((6, 6))
    water = np.zeros((6, 6), dtype=bool)
    with pytest.raises(RuntimeError) as want:
        _oracle_lakes(skin, water, np.array([2.0]), np.array([2.0]))
    with pytest.raises(RuntimeError) as got:
        _native().lake_water_nearest(skin, water, np.array([2.0]),
                                     np.array([2.0]))
    assert str(got.value) == str(want.value)


def test_an_unrolled_ring_search_matches_the_oracle():
    from gpuwm.ingest.horiz import unrolled_source_ring

    rng = np.random.default_rng(11)
    longitude = np.arange(0.0, 360.0, 2.0)
    skin = 260.0 + 30.0 * rng.random((40, longitude.size))
    water = rng.random((40, longitude.size)) < 0.01
    water[20, 1] = True
    y = rng.uniform(0, 39, 200)
    x = rng.uniform(0, longitude.size - 1, 200)
    (uskin, uwater), ux, period = unrolled_source_ring(
        (skin, water), longitude, x)
    assert period == longitude.size
    want = _oracle_lakes(uskin, uwater, y, ux)
    for workers in WORKERS:
        _same(_native().lake_water_nearest(uskin, uwater, y, ux,
                                           workers=workers), want)


# ---- masked bilinear blend ------------------------------------------

def _random_blend_case(seed):
    rng = np.random.default_rng(100 + seed)
    sny, snx = (int(v) for v in rng.integers(2, 40, size=2))
    lat = np.linspace(20.0, 20.0 + 0.5 * (sny - 1), sny)
    lon = np.linspace(-100.0, -100.0 + 0.5 * (snx - 1), snx)
    shape = tuple(int(v) for v in rng.integers(1, 50, size=2))
    target_lat = rng.uniform(lat[0] - 1, lat[-1] + 1, shape)
    target_lon = rng.uniform(lon[0] - 1, lon[-1] + 1, shape)
    corners = water_temperature._bilinear_corners(
        lat, lon, target_lat, target_lon)
    field = 270.0 + 30.0 * rng.random((sny, snx))
    field[rng.random((sny, snx)) < 0.05] = np.nan
    field[rng.random((sny, snx)) < 0.02] = -0.0
    donors = (rng.random((sny, snx)) < [0.0, 0.3, 0.7, 1.0][seed % 4]) \
        & np.isfinite(field)
    return field, donors, corners, shape


@pytest.mark.parametrize("seed", range(8))
def test_random_blends_match_the_oracle(seed):
    field, donors, corners, shape = _random_blend_case(seed)
    for floor in (1e-6, 0.0, 0.5):
        want = oracle.normalized_masked_bilinear(
            field, donors, corners, shape, denominator_floor=floor)
        for workers in WORKERS:
            _same(_native().masked_bilinear_blend(
                field, donors, corners, shape, denominator_floor=floor,
                workers=workers), want)


def test_one_corner_set_reused_across_water_bodies_matches_the_oracle():
    """The assembly blends every body over one corner set; reuse is exact."""
    field, donors, corners, shape = _random_blend_case(2)
    native = _native()
    rng = np.random.default_rng(7)
    for _ in range(4):
        body = donors & (rng.random(donors.shape) < 0.5)
        _same(native.masked_bilinear_blend(field, body, corners, shape),
              oracle.normalized_masked_bilinear(field, body, corners, shape))
    other_field, other_donors, other_corners, other_shape =         _random_blend_case(5)
    _same(native.masked_bilinear_blend(
        other_field, other_donors, other_corners, other_shape),
        oracle.normalized_masked_bilinear(
            other_field, other_donors, other_corners, other_shape))


def test_the_route_blend_is_the_native_one():
    field, donors, corners, shape = _random_blend_case(3)
    _same(water_temperature.normalized_masked_bilinear(
        field, donors, corners, shape),
        oracle.normalized_masked_bilinear(field, donors, corners, shape))


# ---- component fill ------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_random_component_fills_match_the_oracle(seed):
    rng = np.random.default_rng(200 + seed)
    ny, nx = (int(v) for v in rng.integers(1, 60, size=2))
    values = 270.0 + 30.0 * rng.random((ny, nx))
    values[rng.random((ny, nx)) < [0.2, 0.6, 0.95, 1.0][seed % 4]] = np.nan
    values[rng.random((ny, nx)) < 0.02] = np.inf
    values[rng.random((ny, nx)) < 0.02] = -0.0
    component = rng.random((ny, nx)) < 0.8
    for sweeps in (1000, 3, 0):
        want = oracle._fill_within_component(values, component, sweeps)
        for workers in WORKERS:
            _same(_native().component_fill(
                values, component, max_sweeps=sweeps, workers=workers),
                want)


def test_the_route_fill_is_the_native_one():
    values = np.full((5, 7), np.nan)
    values[2, 3] = 281.25
    component = np.ones((5, 7), dtype=bool)
    component[:, 0] = False
    _same(water_temperature._fill_within_component(values, component),
          oracle._fill_within_component(values, component))


# ---- overlay sample ------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_random_overlay_samples_match_the_oracle(seed):
    rng = np.random.default_rng(300 + seed)
    ny, nx = (int(v) for v in rng.integers(2, 30, size=2))
    latitude = np.linspace(30.0, 30.0 + 0.25 * (ny - 1), ny)
    longitude = np.linspace(-90.0, -90.0 + 0.25 * (nx - 1), nx)
    temperature = 270.0 + 30.0 * rng.random((ny, nx))
    valid = rng.random((ny, nx)) < [0.0, 0.5, 0.9, 1.0][seed % 4]
    overlay = SimpleNamespace(latitude=latitude, longitude=longitude,
                              temperature_k=temperature, valid=valid)
    shape = tuple(int(v) for v in rng.integers(1, 40, size=2))
    target_lat = rng.uniform(latitude[0] - 1, latitude[-1] + 1, shape)
    target_lon = rng.uniform(longitude[0] - 1, longitude[-1] + 1, shape)
    target_lat.flat[0] = latitude[-1]
    target_lon.flat[0] = longitude[-1]
    want = oracle.masked_bilinear_sample(overlay, target_lat, target_lon)
    got = water_overlay.masked_bilinear_sample(overlay, target_lat, target_lon)
    _same(got[0], want[0])
    _same(got[1], want[1])
    # The route's own coordinates, then the corner blend at every worker
    # count against the oracle.
    lon_mid = 0.5 * (longitude[0] + longitude[-1])
    unwrapped = target_lon + 360.0 * np.round((lon_mid - target_lon) / 360.0)
    inside = ((target_lat >= latitude[0]) & (target_lat <= latitude[-1])
              & (unwrapped >= longitude[0]) & (unwrapped <= longitude[-1]))
    y = np.interp(target_lat, latitude, np.arange(latitude.size))
    x = np.interp(unwrapped, longitude, np.arange(longitude.size))
    y0 = np.clip(np.floor(y).astype(np.intp), 0, latitude.size - 2)
    x0 = np.clip(np.floor(x).astype(np.intp), 0, longitude.size - 2)
    fy = np.clip(y - y0, 0.0, 1.0)
    fx = np.clip(x - x0, 0.0, 1.0)
    for workers in WORKERS:
        mine = _native().overlay_bilinear_sample(
            temperature, valid, y0, x0, fy, fx, inside, workers=workers)
        _same(mine[0], want[0])
        _same(mine[1], want[1])


# ---- component labelling ------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_random_masks_label_as_the_oracle_labels(seed):
    rng = np.random.default_rng(400 + seed)
    ny, nx = (int(v) for v in rng.integers(1, 80, size=2))
    mask = rng.random((ny, nx)) < [0.0, 0.1, 0.4, 0.6, 0.9][seed % 5]
    want_labels, want_count = oracle._label_components(mask)
    got_labels, got_count = _native().label_components(mask)
    assert got_count == want_count
    _same(got_labels, want_labels)


def test_spirals_and_pinches_label_as_the_oracle_labels():
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, :] = mask[:, 8] = mask[8, :] = mask[2:, 0] = True
    mask[2, 0:7] = mask[2:7, 6] = mask[6, 2:7] = mask[4:7, 2] = True
    mask[4, 4] = True
    mask[1, 1] = True
    for candidate in (mask, ~mask, np.zeros((3, 4), bool),
                      np.ones((1, 7), bool), np.eye(6, dtype=bool)[::-1]):
        want = oracle._label_components(candidate)
        got = _native().label_components(candidate)
        assert got[1] == want[1]
        _same(got[0], want[0])


# ---- refusals and boundaries ---------------------------------------

def test_a_library_without_the_water_entries_is_refused_by_name():
    stale = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    stale.path = Path("/opt/old/libgpuwm_preprocess_cpu.so")
    stale.water_blend_entry = False
    with pytest.raises(MaskedChainUnavailable) as caught:
        stale.require_water_blends()
    message = str(caught.value)
    assert WATER_BLEND_ENTRY in message
    assert LAKE_WATER_NEAREST_ENTRY in message
    assert "water temperature" in message
    assert "rebuild or re-fetch" in message


def test_a_corner_off_the_source_is_refused():
    field = np.ones((3, 3))
    donors = np.ones((3, 3), dtype=bool)
    corners = ((np.array([[3]]), np.array([[0]]), np.array([[1.0]])),)
    with pytest.raises(ValueError, match="outside the source grid"):
        _native().masked_bilinear_blend(field, donors, corners, (1, 1))


def test_the_sealed_distribution_runs_the_water_blends():
    from gpuwm import native_wrf_distribution

    result = native_wrf_distribution._cpu_water_blend_self_test(_native())
    assert result["status"] == "PASS"
    assert result["worker_counts"] == [1, 3]


def test_nothing_at_run_time_imports_the_water_blend_oracle():
    """The NumPy code is a test oracle, never a silent runtime fallback."""
    offenders = []
    for base in ("gpuwm", "tilestream", "tools"):
        root = ROOT / base
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith("gpuwm/verify/"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "water_blend_oracle" not in text:
                continue
            for node in ast.walk(ast.parse(text)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [
                        alias.name for alias in node.names]
                if any("water_blend_oracle" in name for name in names):
                    offenders.append(relative)
    assert offenders == []
    horiz = (ROOT / "gpuwm" / "ingest" / "horiz.py").read_text(
        encoding="utf-8")
    assert "def _nearest_finite_source_water" not in horiz
