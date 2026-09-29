"""The source owner of each water body and the CPU surface-nearest search.

``gpuwm_component_owner_f64`` replaced ``_component_owner_of_source`` (a
NumPy ``np.unique`` and a Python loop over every source cell and body
pair) and ``gpuwm_masked_nearest_f32`` replaced ``_masked_nearest_cpu``
(a NumPy window scan) of the CPU preprocessing backend.  The NumPy code is
kept in :mod:`gpuwm.verify.water_blend_oracle`, and every assertion here
compares BYTES at 1, 3 and 64 workers.  The same file holds the review's
worker test: an explicit ``--preprocess-workers`` count reaches every
water entry of the Rust library.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest import cpu_backend, water_temperature
from gpuwm.ingest.cpu_backend import (
    COMPONENT_OWNER_ENTRY,
    MASKED_NEAREST_ENTRY,
    WATER_ENTRIES,
    CpuPreprocessBackend,
    MaskedChainUnavailable,
    automatic_workers,
    host_step_workers,
)
from gpuwm.ingest.preprocess_backend import (
    ParallelCpuPreprocessBackend,
    resolve_preprocess_backend,
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


# ---- the source owner ------------------------------------------------

def _owner_case(seed):
    rng = np.random.default_rng(seed)
    sny, snx = (int(v) for v in rng.integers(2, 14, size=2))
    lat0 = float(rng.uniform(-60.0, 60.0))
    dlat = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 1.0))
    lon0 = float(rng.choice([rng.uniform(-180.0, 180.0),
                             rng.uniform(0.0, 360.0), 359.75, -180.0]))
    dlon = float(rng.uniform(0.1, 1.0))
    source_lat = lat0 + dlat * np.arange(sny)
    source_lon = lon0 + dlon * np.arange(snx)
    ty, tx = (int(v) for v in rng.integers(1, 40, size=2))
    # Targets spread a little past the source, on half cells (np.rint
    # ties), on whole turns of longitude, and a few not finite.
    fy = rng.uniform(-2.0, sny + 1.0, size=(ty, tx))
    fx = rng.uniform(-2.0, snx + 1.0, size=(ty, tx))
    half = rng.random((ty, tx)) < 0.2
    fy[half] = np.floor(fy[half]) + 0.5
    fx[half] = np.floor(fx[half]) + 0.5
    target_lat = lat0 + dlat * fy
    target_lon = (lon0 + dlon * fx
                  + 360.0 * rng.integers(-2, 3, size=(ty, tx)))
    poison = rng.random((ty, tx))
    target_lat[poison < 0.02] = np.nan
    target_lon[(poison > 0.02) & (poison < 0.03)] = np.inf
    labels = rng.integers(0, int(rng.integers(1, 6)) + 1,
                          size=(ty, tx)).astype(np.int32)
    return labels, source_lat, source_lon, target_lat, target_lon, (sny, snx)


@pytest.mark.parametrize("seed", range(60))
def test_random_owners_match_the_oracle(seed):
    case = _owner_case(seed)
    with np.errstate(invalid="ignore"):
        want = oracle._component_owner_of_source(*case)
    for workers in WORKERS:
        _same(_native().component_owner(*case, workers=workers), want)


def test_a_tie_goes_to_the_higher_label_as_the_stable_oracle_rules():
    labels = np.array([[1, 2, 3, 3, 2, 1]], dtype=np.int32)
    lat = np.zeros((1, 6))
    lon = np.zeros((1, 6))
    case = (labels, np.array([0.0, 1.0]), np.array([0.0, 1.0]), lat, lon,
            (2, 2))
    want = oracle._component_owner_of_source(*case)
    assert want[0, 0] == 3
    for workers in WORKERS:
        _same(_native().component_owner(*case, workers=workers), want)


def test_no_target_inside_owns_nothing():
    labels = np.ones((2, 2), dtype=np.int32)
    far = np.full((2, 2), 500.0)
    case = (labels, np.array([0.0, 1.0]), np.array([0.0, 1.0]), far, far,
            (3, 3))
    _same(_native().component_owner(*case),
          oracle._component_owner_of_source(*case))


def test_the_route_owner_is_the_native_one(monkeypatch):
    calls = []
    real = CpuPreprocessBackend.component_owner

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("workers"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(CpuPreprocessBackend, "component_owner", spy)
    case = _owner_case(3)
    _same(water_temperature._component_owner_of_source(*case, workers=2),
          oracle._component_owner_of_source(*case))
    assert calls == [2]


# ---- the surface-nearest search -------------------------------------

def _nearest_case(seed):
    rng = np.random.default_rng(1000 + seed)
    sny, snx = (int(v) for v in rng.integers(2, 30, size=2))
    lat0 = float(rng.uniform(-50.0, 50.0))
    dlat = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.5))
    lon0 = float(rng.uniform(-170.0, 170.0))
    dlon = float(rng.uniform(0.05, 0.5))
    latitude = lat0 + dlat * np.arange(sny)
    longitude = lon0 + dlon * np.arange(snx)
    field = (250.0 + 50.0 * rng.random((sny, snx))).astype(np.float32)
    field[rng.random((sny, snx)) < 0.1] = np.nan
    field[rng.random((sny, snx)) < 0.02] = np.inf
    source_land = rng.random((sny, snx)) < float(rng.uniform(0.0, 1.0))
    ty, tx = (int(v) for v in rng.integers(1, 30, size=2))
    fy = rng.uniform(0.0, sny - 1.0, size=(ty, tx))
    fx = rng.uniform(0.0, snx - 1.0, size=(ty, tx))
    half = rng.random((ty, tx)) < 0.3
    fy[half] = np.minimum(np.floor(fy[half]) + 0.5, sny - 1.0)
    fx[half] = np.minimum(np.floor(fx[half]) + 0.5, snx - 1.0)
    target_lat = lat0 + dlat * fy
    target_lon = lon0 + dlon * fx
    target_land = rng.random((ty, tx)) < 0.5
    options = {
        "surface": ("match", "land", "water")[seed % 3],
        "fill_value": float(rng.choice([0.0, -1.0, 273.15, 1e-40])),
        "search_radius": int((0, 1, 2, 3, 8)[seed % 5]),
    }
    return ((field, latitude, longitude, target_lat, target_lon,
             source_land, target_land), options)


def _run_nearest(function, args, options):
    try:
        return function(*args, **options)
    except ValueError as error:
        return ("refused", str(error))


@pytest.mark.parametrize("seed", range(60))
def test_random_surface_searches_match_the_oracle(seed):
    args, options = _nearest_case(seed)
    for strict in (False, True):
        want = _run_nearest(oracle._masked_nearest_cpu, args,
                            {**options, "strict": strict})
        for workers in WORKERS:
            backend = ParallelCpuPreprocessBackend(workers=workers)
            got = _run_nearest(backend.masked_nearest, args,
                               {**options, "strict": strict})
            if isinstance(want[0], str):
                assert got == want
            else:
                assert not isinstance(got[0], str), got
                _same(got, want)


def test_the_first_of_equally_near_cells_wins():
    latitude = np.array([0.0, 1.0, 2.0])
    longitude = np.array([0.0, 1.0, 2.0])
    field = np.arange(9, dtype=np.float32).reshape(3, 3)
    source_land = np.zeros((3, 3), dtype=bool)
    source_land[1, 1] = True
    args = (field, latitude, longitude, np.array([[1.0]]),
            np.array([[1.0]]), source_land, np.array([[False]]))
    want = oracle._masked_nearest_cpu(*args, surface="water")
    assert float(want[0, 0]) == 1.0
    for workers in WORKERS:
        _same(ParallelCpuPreprocessBackend(workers=workers).masked_nearest(
            *args, surface="water"), want)


def test_a_library_without_the_search_is_refused_by_name():
    stale = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    stale.path = Path("/opt/old/libgpuwm_preprocess_cpu.so")
    stale.masked_nearest_entry = False
    with pytest.raises(MaskedChainUnavailable) as caught:
        stale.require_masked_nearest()
    message = str(caught.value)
    assert MASKED_NEAREST_ENTRY in message
    assert "rebuild or re-fetch" in message


def test_a_refusal_names_only_the_entries_missing():
    stale = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    stale.path = Path("/opt/old/libgpuwm_preprocess_cpu.so")
    stale.water_blend_entry = False
    stale.water_blend_missing = (COMPONENT_OWNER_ENTRY,)
    with pytest.raises(MaskedChainUnavailable) as caught:
        stale.require_water_blends()
    message = str(caught.value)
    assert COMPONENT_OWNER_ENTRY in message
    for name in WATER_ENTRIES:
        if name != COMPONENT_OWNER_ENTRY:
            assert name not in message


# ---- workers: --preprocess-workers reaches every water entry ----------

class _Recorder:
    """The library, with the worker argument of every water call kept."""

    def __init__(self, library, calls):
        self._library = library
        self._calls = calls

    def __getattr__(self, name):
        function = getattr(self._library, name)
        if name not in WATER_ENTRIES + (MASKED_NEAREST_ENTRY,) \
                or name == cpu_backend.LABEL_COMPONENTS_ENTRY:
            return function

        def call(*args):
            self._calls.append((name, int(args[-1])))
            return function(*args)

        return call


def _water_case():
    rng = np.random.default_rng(5)
    ny, nx = 24, 30
    land = np.zeros((ny, nx), dtype=bool)
    land[:, 12:18] = True
    land[3:6, 3:6] = True
    lake = np.zeros((ny, nx), dtype=bool)
    lake[18:22, 22:27] = True
    skin = 280.0 + rng.random((ny, nx))
    skin[20, 24] = np.nan
    target_lat, target_lon = np.meshgrid(
        np.linspace(40.0, 42.0, ny), np.linspace(-90.0, -87.0, nx),
        indexing="ij")
    source_lat = np.linspace(39.5, 42.5, 9)
    source_lon = np.linspace(-90.5, -86.5, 11)
    source_sst = 285.0 + rng.random((9, 11))
    source_sst[4, 5] = np.nan
    statics = water_temperature.WaterTemperatureStatics(
        route="test", land=land, lake=lake, lake_category=21,
        policy=water_temperature.DEFAULT_WATER_TEMPERATURE_POLICY)
    return statics, dict(
        mapped_sst=None, mapped_skin=skin, source_sst=source_sst,
        source_lat=source_lat, source_lon=source_lon,
        target_lat=target_lat, target_lon=target_lon)


def test_preprocess_workers_two_reach_the_water_entries(monkeypatch):
    engine = resolve_preprocess_backend("cpu", workers=2)
    workers = host_step_workers(engine)
    assert workers == 2
    native = cpu_backend.water_blend_backend()
    calls: list = []
    monkeypatch.setattr(native, "_library",
                        _Recorder(native._library, calls))
    monkeypatch.setattr(engine._native, "_library",
                        _Recorder(engine._native._library, calls))
    statics, kwargs = _water_case()
    water_temperature.assemble_for_route(statics, **kwargs, workers=workers)
    field = np.arange(16, dtype=np.float64).reshape(4, 4)
    corners = ((np.zeros((2, 2), dtype=np.intp),
                np.zeros((2, 2), dtype=np.intp), np.ones((2, 2))),)
    water_temperature.normalized_masked_bilinear(
        field, field > 3, corners, (2, 2), workers=workers)
    water_temperature._fill_within_component(
        np.where(field > 5, field, np.nan), field > 1, workers=workers)
    native.lake_water_nearest(field, field > 10, np.array([0.0]),
                              np.array([0.0]), workers=workers)
    engine.masked_nearest(
        field, np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([[1.0]]), np.array([[1.0]]), field > 7, np.array([[False]]),
        strict=False)
    called = dict(calls)
    assert set(called) == {
        "gpuwm_component_owner_f64", "gpuwm_water_bodies_f64",
        "gpuwm_water_repair_f64", "gpuwm_masked_bilinear_blend_f64",
        "gpuwm_component_fill_f64", "gpuwm_lake_water_nearest_f64",
        "gpuwm_masked_nearest_f32"}
    assert all(count == 2 for _, count in calls), calls


def test_no_worker_count_takes_the_automatic_one():
    assert CpuPreprocessBackend._water_workers(None) == automatic_workers()
    assert host_step_workers(None) == automatic_workers()
    assert host_step_workers("cpu") == automatic_workers()
    assert ParallelCpuPreprocessBackend().host_step_workers == \
        automatic_workers()
    assert ParallelCpuPreprocessBackend(workers=5).host_step_workers == 5


_WATER_CALLS = {
    "assemble_for_route", "assemble_horizontal_water_temperature",
    "interpolate_lake_skin_temperature", "overlay_snapshot_sequence",
    "assembled_water_temperature",
}


def test_every_runtime_water_call_passes_the_preparation_workers():
    """A new route that drops the workers would run the water steps on the
    automatic count whatever ``--preprocess-workers`` said."""
    missing = []
    for path in [*(ROOT / "gpuwm").rglob("*.py"),
                 *(ROOT / "tilestream").rglob("*.py"),
                 ROOT / "tools" / "hrrr_single_domain_benchmark.py"]:
        if "verify" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name not in _WATER_CALLS:
                continue
            if not any(keyword.arg == "workers" for keyword in node.keywords):
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
    # The one wrapper forwards its own argument on.
    missing = [item for item in missing
               if not item.startswith(
                   str(Path("gpuwm/ingest/water_temperature.py")))]
    assert missing == []


def test_the_shared_backend_keeps_no_corner_cache():
    native = cpu_backend.water_blend_backend()
    field = np.arange(16, dtype=np.float64).reshape(4, 4)
    corners = ((np.zeros((2, 2), dtype=np.intp),
                np.zeros((2, 2), dtype=np.intp), np.ones((2, 2))),)
    native.masked_bilinear_blend(field, field > 3, corners, (2, 2))
    assert not hasattr(native, "_blend_corners")


# ---- the seal and the boundary --------------------------------------

def test_the_sealed_self_test_covers_labelling_owner_and_search():
    from gpuwm import native_wrf_distribution

    result = native_wrf_distribution._cpu_water_blend_self_test(_native())
    assert result["status"] == "PASS"
    for case in ("labelling_eight_connected", "owner_tie_higher_label",
                 "surface_nearest_first_of_tie"):
        assert case in result["cases"]


def test_the_numpy_owner_and_search_left_the_runtime_modules():
    owner = (ROOT / "gpuwm" / "ingest" / "water_temperature.py").read_text(
        encoding="utf-8")
    assert "np.unique(key, return_counts=True)" not in owner
    backend = (ROOT / "gpuwm" / "ingest" / "preprocess_backend.py").read_text(
        encoding="utf-8")
    assert "for dj in range(-radius, radius + 1)" not in backend
