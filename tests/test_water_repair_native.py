"""The water repairs and the per-body water assembly in Rust against NumPy.

``gpuwm_water_repair_f64`` and ``gpuwm_water_bodies_f64`` replaced the
last single-core NumPy steps of the water-temperature assembly: the box
repairs of water cells whose provider left no admissible temperature
(``_fill_missing_water_temperature`` and its helpers) and the per-body
loop of ``assemble_water_temperature``, which built whole-domain masks
once per water body.  The NumPy code is kept verbatim in
:mod:`gpuwm.verify.water_blend_oracle`, the old assembly whole, and every
assertion here compares BYTES at 1, 3 and 64 workers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest import cpu_backend, water_temperature
from gpuwm.ingest.cpu_backend import (
    WATER_BODIES_ENTRY,
    WATER_REPAIR_ENTRY,
    CpuPreprocessBackend,
    MaskedChainUnavailable,
)
from gpuwm.ingest.water_temperature import (
    MAX_WATER_TEMPERATURE_K,
    MIN_WATER_TEMPERATURE_K,
    SOURCE_NEAREST_WATER,
    SOURCE_SURROUNDING_SKIN,
    label_surface_components,
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


def _blobs(rng, shape, density, grow):
    mask = rng.random(shape) < density
    for _ in range(grow):
        padded = np.pad(mask, 1)
        mask = mask | (
            (padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2]
             | padded[1:-1, 2:]) & (rng.random(shape) < 0.5))
    return mask


def _surface(rng, ny, nx):
    land = _blobs(rng, (ny, nx), 0.08, 3)
    lake = _blobs(rng, (ny, nx), 0.02, 1) & ~land
    return land, lake


def _repair_case(seed):
    rng = np.random.default_rng(seed)
    ny, nx = (int(v) for v in rng.integers(4, 70, size=2))
    land, lake = _surface(rng, ny, nx)
    labels, _ = label_surface_components(land, lake)
    water = ~land
    values = 250.0 + 40.0 * rng.random((ny, nx))
    bad_water = [0.0, 0.03, 0.4, 0.9, 1.0][seed % 5]
    bad_land = [0.0, 0.2, 0.7, 1.0][seed % 4]
    pick = rng.random((ny, nx))
    poison = rng.choice(np.array([np.nan, 0.0, 500.0, -np.inf, np.inf]),
                        size=(ny, nx))
    values = np.where(water & (pick < bad_water), poison, values)
    values = np.where(land & (pick < bad_land), poison, values)
    source = rng.integers(0, 5, size=(ny, nx)).astype(np.int8)
    return values, source, water, np.asarray(labels)


def _oracle_repair(values, source, water, labels):
    return oracle._fill_missing_water_temperature(
        values.copy(), source.copy(), water, labels)


def _native_repair(values, source, water, labels, workers):
    return _native().water_repair(
        values, source, water, labels, minimum=MIN_WATER_TEMPERATURE_K,
        maximum=MAX_WATER_TEMPERATURE_K,
        nearest_water_code=SOURCE_NEAREST_WATER,
        surrounding_skin_code=SOURCE_SURROUNDING_SKIN, workers=workers)


def _check_repair(values, source, water, labels):
    want = _oracle_repair(values, source, water, labels)
    for workers in WORKERS:
        got = _native_repair(values, source, water, labels, workers)
        _same(got[0], np.asarray(want[0], dtype=np.float64))
        _same(got[1], np.asarray(want[1], dtype=np.int8))
        assert got[2] == want[2]
        _same(got[3], want[3])
    return want


# ---- the box repairs -------------------------------------------------

@pytest.mark.parametrize("seed", range(40))
def test_random_repairs_match_the_oracle(seed):
    _check_repair(*_repair_case(seed))


def test_every_repair_path_is_exercised_and_matches():
    """Own body, nearest water, surrounding skin and the stranded shore."""
    ny, nx = 24, 30
    labels = np.zeros((ny, nx), dtype=np.int32)
    labels[2:6, 2:9] = 1          # a body with a bad hole
    labels[2:5, 20:24] = 2        # wholly bad: nearest water
    labels[15:19, 3:7] = 3        # wholly bad, used by the skin case below
    water = labels > 0
    values = np.full((ny, nx), 290.0) + np.arange(ny * nx).reshape(ny, nx) / 97.0
    values[3:5, 4:6] = np.nan
    values[2:5, 20:24] = 0.0
    values[15:19, 3:7] = np.nan
    source = np.ones((ny, nx), dtype=np.int8)
    want = _check_repair(values, source, water, labels)
    assert want[2]["own_body"] == 4
    assert want[2]["nearest_water"] == 12 + 16

    # No admissible water anywhere: the skin around each body, and a
    # shore with no admissible skin takes the nearest admissible cell.
    values[2:6, 2:9] = np.nan
    values[14:20, 2:8] = np.nan   # the shore of body 3 is inadmissible
    want = _check_repair(values, source, water, labels)
    assert want[2]["surrounding_skin"] > 0
    assert want[3].sum() == water.sum()


def test_a_domain_with_nothing_admissible_leaves_the_cells_for_the_refusal():
    labels = np.ones((5, 6), dtype=np.int32)
    labels[:, :2] = 0
    water = labels > 0
    values = np.full((5, 6), np.nan)
    source = np.zeros((5, 6), dtype=np.int8)
    want = _check_repair(values, source, water, labels)
    assert not want[3].any()


def test_a_single_cell_and_a_whole_grid_body_match():
    for shape in ((1, 1), (1, 7), (7, 1), (9, 9)):
        labels = np.ones(shape, dtype=np.int32)
        water = np.ones(shape, dtype=bool)
        values = np.full(shape, 280.0)
        values.flat[len(values.flat) // 2] = np.nan
        _check_repair(values, np.zeros(shape, dtype=np.int8), water, labels)


def test_nearest_water_ties_take_the_first_donor_in_row_major_order():
    ny, nx = 21, 41
    labels = np.zeros((ny, nx), dtype=np.int32)
    labels[10, 20] = 1
    labels[10, 3] = 2
    labels[10, 37] = 3
    water = labels > 0
    values = np.where(water, 280.0, 300.0)
    values[10, 37] = 281.0
    values[10, 20] = np.nan
    want = _check_repair(values, np.zeros((ny, nx), dtype=np.int8), water,
                         labels)
    # (10, 3) and (10, 37) are both 17 cells away: the first in row-major
    # order wins.
    assert want[0][10, 20] == 280.0


def test_a_negative_label_is_refused():
    labels = np.array([[1, -1], [1, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="water-body label"):
        _native_repair(np.full((2, 2), np.nan), np.zeros((2, 2), np.int8),
                       labels != 0, labels, 1)


# ---- the per-body assembly -------------------------------------------

def _assembly_case(seed):
    rng = np.random.default_rng(1000 + seed)
    ny, nx = (int(v) for v in rng.integers(6, 60, size=2))
    land, lake = _surface(rng, ny, nx)
    tlat = np.linspace(30.0, 30.0 + 0.05 * ny, ny)[:, None] + np.zeros(nx)
    tlon = np.linspace(-90.0, -90.0 + 0.05 * nx, nx)[None, :] + np.zeros((ny, 1))
    tlat = tlat + 0.01 * rng.standard_normal((ny, nx))
    tlon = tlon + 0.01 * rng.standard_normal((ny, nx))
    step = [0.25, 0.1, 0.5][seed % 3]
    lat = np.arange(29.0, 31.0 + 0.05 * ny, step)
    lon = np.arange(-91.0, -89.0 + 0.05 * nx, step)
    sst = 270.0 + 30.0 * rng.random((lat.size, lon.size))
    sst[rng.random(sst.shape) < [0.0, 0.2, 0.6, 0.97][seed % 4]] = np.nan
    sst[rng.random(sst.shape) < 0.03] = 100.0
    skin = 260.0 + 40.0 * rng.random((ny, nx))
    skin[rng.random((ny, nx)) < [0.0, 0.05, 0.5][seed % 3]] = np.nan
    lake_water = None
    if seed % 2:
        lake_water = 275.0 + 10.0 * rng.random((ny, nx))
        lake_water[rng.random((ny, nx)) < [0.1, 0.6, 1.0][seed % 3]] = np.nan
    have_source = seed % 5 != 4
    kwargs = dict(
        mapped_sst=None, mapped_skin=skin, target_land=land, target_lake=lake,
        mapped_lake_water=lake_water)
    if have_source:
        kwargs.update(source_sst=sst, source_lat=lat, source_lon=lon,
                      target_lat=tlat, target_lon=tlon)
    return kwargs


def _run(function, kwargs):
    try:
        return function(**kwargs)
    except ValueError as error:
        return ("refused", str(error))


def _check_assembly(kwargs, monkeypatch):
    want = _run(oracle.assemble_water_temperature, kwargs)
    for workers in WORKERS:
        monkeypatch.setattr(cpu_backend, "available_cpu_count",
                            lambda workers=workers: workers)
        got = _run(water_temperature.assemble_water_temperature, kwargs)
        if isinstance(want[0], str):
            assert got == want
            continue
        assert not isinstance(got[0], str), got
        _same(got[0], want[0])
        _same(got[1], want[1])
        assert json.dumps(got[2], sort_keys=True) == json.dumps(
            want[2], sort_keys=True)
    return want


@pytest.mark.parametrize("seed", range(40))
def test_random_assemblies_match_the_oracle(seed, monkeypatch):
    _check_assembly(_assembly_case(seed), monkeypatch)


def test_many_small_bodies_and_the_listed_fallback_cells_match(monkeypatch):
    """Hundreds of one-cell lakes, all declined by the lake model."""
    ny, nx = 60, 80
    land = np.ones((ny, nx), dtype=bool)
    land[::3, ::3] = False
    lake = ~land
    skin = np.full((ny, nx), 285.0)
    lake_water = np.full((ny, nx), np.nan)
    want = _check_assembly(dict(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, mapped_lake_water=lake_water), monkeypatch)
    assert want[2]["components"] == int(lake.sum())
    assert want[2]["lake_fallback_cells"] == int(lake.sum())


def test_a_body_with_a_hole_its_blend_cannot_reach_is_filled(monkeypatch):
    """Coverage passes, and the four-neighbour fill closes the rest."""
    kwargs = _assembly_case(0)
    kwargs["source_sst"] = np.where(
        np.arange(kwargs["source_sst"].size).reshape(
            kwargs["source_sst"].shape) % 5 == 0,
        np.nan, kwargs["source_sst"])
    _check_assembly(kwargs, monkeypatch)


def test_a_label_above_the_declared_bodies_is_refused():
    labels = np.array([[1, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="water-body label"):
        _native().water_bodies(
            labels=labels, lake_class=np.array([False, False]),
            skin=np.ones((1, 2)), values=np.ones((1, 2)),
            source=np.zeros((1, 2), dtype=np.int8), codes=(1, 2, 4),
            min_coverage=0.5, minimum=170.0, maximum=400.0, max_listed=4)


# ---- refusals, the seal and the boundary -----------------------------

def test_a_library_without_the_repair_entries_is_refused_by_name():
    stale = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    stale.path = Path("/opt/old/libgpuwm_preprocess_cpu.so")
    stale.water_blend_entry = False
    with pytest.raises(MaskedChainUnavailable) as caught:
        stale.require_water_blends()
    message = str(caught.value)
    assert WATER_BODIES_ENTRY in message
    assert WATER_REPAIR_ENTRY in message
    assert "rebuild or re-fetch" in message


def test_the_sealed_self_test_covers_the_repairs():
    from gpuwm import native_wrf_distribution

    result = native_wrf_distribution._cpu_water_blend_self_test(_native())
    assert result["status"] == "PASS"
    assert "water_repair_own_body_and_nearest" in result["cases"]
    assert "water_bodies_analysis_and_skin" in result["cases"]


def test_the_numpy_repairs_left_the_runtime_module():
    text = (ROOT / "gpuwm" / "ingest" / "water_temperature.py").read_text(
        encoding="utf-8")
    for name in ("def _propagate_into", "def _nearest_donor",
                 "def _label_boxes", "def _donor_index",
                 "selection = labels == label"):
        assert name not in text
