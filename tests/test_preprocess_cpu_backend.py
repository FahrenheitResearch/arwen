from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ingest.backend_contract import (
    ArrayParityRule,
    build_backend_receipt,
    build_workload_contract,
    compare_backend_outputs,
)
from gpuwm.ingest.cpu_backend import CpuPreprocessBackend, resolve_cpu_bridge
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend
from gpuwm.native_wrf_distribution import distribution_contract
from gpuwm.static.lambert import LambertGrid
from gpuwm.verify.npref import interpolate_regular_np, np_wrf_real_vert_interp


def _backend() -> CpuPreprocessBackend:
    try:
        return CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU bridge is not built: {exc}")


def test_regular_rust_workers_are_byte_stable_and_match_authority():
    backend = _backend()
    latitude = np.linspace(28.0, 38.0, 11, dtype=np.float64)
    longitude = np.linspace(-105.0, -94.0, 12, dtype=np.float64)
    target_lat = np.linspace(29.25, 36.75, 7)[:, None] \
        + np.zeros((7, 9))
    target_lon = np.zeros((7, 9)) \
        + np.linspace(-103.75, -95.25, 9)[None, :]
    rng = np.random.default_rng(91)
    source = rng.normal(size=(3, latitude.size, longitude.size)).astype(
        np.float32)
    source[:, 3, 4] = 0.0

    for method in ("nearest", "bilinear", "parabolic"):
        serial = backend.interpolate_regular(
            source, latitude, longitude, target_lat, target_lon,
            method=method, workers=1)
        parallel = backend.interpolate_regular(
            source, latitude, longitude, target_lat, target_lon,
            method=method, workers=17)
        np.testing.assert_array_equal(parallel, serial)
        authority = interpolate_regular_np(
            source, latitude, longitude, target_lat, target_lon,
            method=method)
        np.testing.assert_allclose(parallel, authority, rtol=3.0e-6,
                                   atol=3.0e-6)


def test_reusable_regular_plan_is_worker_stable():
    backend = _backend()
    latitude = np.linspace(28.0, 38.0, 11, dtype=np.float64)
    longitude = np.linspace(-105.0, -94.0, 12, dtype=np.float64)
    target_lat = np.linspace(29.25, 36.75, 7)[:, None] \
        + np.zeros((7, 9))
    target_lon = np.zeros((7, 9)) \
        + np.linspace(-103.75, -95.25, 9)[None, :]
    source = np.arange(2 * 11 * 12, dtype=np.float32).reshape(2, 11, 12)
    plan = backend.regular_plan(
        latitude, longitude, target_lat, target_lon)
    serial = plan.apply(source, method="parabolic", workers=1)
    parallel = plan.apply(source, method="parabolic", workers=8)
    np.testing.assert_array_equal(parallel, serial)


def test_rust_vertical_is_dynamic_worker_stable_and_matches_wrf_authority():
    backend = _backend()
    rng = np.random.default_rng(92)
    nsource, ntarget, ny, nx = 73, 91, 4, 6
    base = np.geomspace(100000.0, 5000.0, nsource)
    source = base[:, None, None] * (
        1.0 + rng.normal(0.0, 2.0e-4, (nsource, ny, nx)))
    source = np.sort(source, axis=0)[::-1]
    values = (280.0 + 0.01 * np.sqrt(source)
              + rng.normal(0.0, 0.1, source.shape))
    surface_pressure = rng.uniform(100100.0, 102000.0, (ny, nx))
    surface_values = 291.0 + rng.normal(0.0, 0.5, (ny, nx))
    eta = np.linspace(0.999, 0.06, ntarget)[:, None, None]
    target = (5000.0 + eta * (surface_pressure[None] - 5000.0))

    serial = backend.wrf_vertical_interpolate(
        values, surface_values, source, surface_pressure, target,
        extrap="temperature", workers=1)
    parallel = backend.wrf_vertical_interpolate(
        values, surface_values, source, surface_pressure, target,
        extrap="temperature", workers=13)
    common_backend = resolve_preprocess_backend("cpu", workers=13)
    common_plan = common_backend.prepare_wrf_vertical(
        source, surface_pressure, target)
    common = common_plan.apply(
        values, surface_values, extrap="temperature",
        values_are_finite=True)
    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(common, parallel)
    authority = np_wrf_real_vert_interp(
        values, surface_values, source, surface_pressure, target,
        extrap="temperature")
    np.testing.assert_allclose(parallel, authority, rtol=3.0e-5,
                               atol=5.0e-3)
    assert parallel.shape == (ntarget, ny, nx)
    assert nsource > 64


def _horizontal_fixture():
    grid = LambertGrid(
        ref_lat=40.0, ref_lon=-85.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-85.0, dx=100_000.0, dy=100_000.0,
        e_we=6, e_sn=5)
    latitude = np.linspace(34.0, 46.0, 9, dtype=np.float64)
    longitude = np.linspace(267.0, 283.0, 10, dtype=np.float64)
    lon2, lat2 = np.meshgrid(longitude, latitude)
    levels = np.array([500.0, 1000.0], dtype=np.float64)
    base = lat2 + 0.1 * lon2
    landsea = (lon2 >= 275.0).astype(np.float64)
    snapshot = Era5Snapshot(
        valid_time=datetime(1974, 4, 3, 12), levels_hpa=levels,
        latitude=latitude, longitude=longitude,
        fields={
            "Z": np.stack((9.81 * base, 9.81 * (base + 1.0))),
            "T": np.stack((base + 190.0, base + 200.0)),
            "U": np.stack((base - 60.0, base - 58.0)),
            "V": np.stack((0.5 * base - 20.0, 0.5 * base - 19.0)),
            "RH": np.stack((base, base + 2.0)),
            "PSFC": 90_000.0 + base,
            "T2": 250.0 + base,
            "D2": 245.0 + base,
            "U10": base - 60.0,
            "V10": 0.5 * base - 20.0,
            "LANDSEA": landsea,
            "SKINTEMP": 250.0 + base,
            "SST": np.where(landsea < 0.5, 270.0 + base, np.nan),
            "SEAICE": np.where(
                landsea < 0.5,
                np.clip((275.0 - lon2) / 8.0, 0.0, 1.0), np.nan),
            "SOILGEO": 9.81 * (100.0 + base),
            "ST000007": 260.0 + base,
            "SM000007": 0.2 + 0.001 * base,
            "SNOW_EC": 0.01 * base,
        })
    catalog = SimpleNamespace(
        snapshots=(snapshot,), units={"SOILGEO": "m2 s-2"})
    return snapshot, grid, catalog


def test_full_horizontal_cpu_path_is_parallel_byte_stable():
    _backend()
    snapshot, grid, catalog = _horizontal_fixture()
    serial = interpolate_era5_to_lambert(
        snapshot, grid, source_orography_catalog=catalog,
        backend="cpu", workers=1)
    parallel = interpolate_era5_to_lambert(
        snapshot, grid, source_orography_catalog=catalog,
        backend="cpu", workers=8)
    assert set(serial.fields) == set(parallel.fields)
    for name in serial.fields:
        assert isinstance(parallel.fields[name], np.ndarray)
        np.testing.assert_array_equal(parallel.fields[name], serial.fields[name])


def test_backend_selector_is_explicit_and_rejects_cpu_options_on_cuda():
    native = _backend()
    cpu = resolve_preprocess_backend("cpu", workers=3)
    assert cpu.name == "cpu"
    receipt = cpu.receipt()
    assert receipt["workers"] == 3
    assert receipt["bridge"]["name"] == native.path.name
    assert receipt["bridge"]["sha256"] == hashlib.sha256(
        native.path.read_bytes()).hexdigest()
    assert resolve_preprocess_backend(None).name == "cuda"
    with pytest.raises(ValueError, match="apply only to the CPU"):
        resolve_preprocess_backend("cuda", workers=3)
    with pytest.raises(ValueError, match="cuda.*cpu.*auto"):
        resolve_preprocess_backend("silent-substitution")


@pytest.mark.parametrize(
    ("cupy_version", "runtime_version", "device_count", "expected"),
    [
        ("13.1.0", 12_900, 1, "cuda"),
        ("12.3.0", 12_900, 1, "cpu"),
        ("13.1.0", 13_000, 1, "cpu"),
        ("13.1.0", 12_900, 0, "cpu"),
    ],
)
def test_auto_backend_uses_only_the_certified_cuda_runtime_family(
    monkeypatch, cupy_version, runtime_version, device_count, expected,
):
    class Runtime:
        @staticmethod
        def runtimeGetVersion():
            return runtime_version

        @staticmethod
        def getDeviceCount():
            return device_count

    cuda = SimpleNamespace(
        name="cuda", array_module=SimpleNamespace(
            __version__=cupy_version,
            cuda=SimpleNamespace(runtime=Runtime())
    ))
    cpu = SimpleNamespace(name="cpu")
    monkeypatch.setattr(
        "gpuwm.ingest.preprocess_backend.CudaPreprocessBackend",
        lambda: cuda,
    )
    monkeypatch.setattr(
        "gpuwm.ingest.preprocess_backend.ParallelCpuPreprocessBackend",
        lambda **_kwargs: cpu,
    )
    assert resolve_preprocess_backend("auto", workers=3).name == expected


def test_sealed_cpu_distribution_forces_auto_to_cpu_and_blocks_cuda(
        tmp_path, monkeypatch):
    from conftest import complete_runtime_manifest

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(complete_runtime_manifest(platform_name="windows-x86_64")),
        encoding="utf-8")
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    cpu = SimpleNamespace(name="cpu")
    monkeypatch.setattr(
        "gpuwm.ingest.preprocess_backend.ParallelCpuPreprocessBackend",
        lambda **_kwargs: cpu,
    )
    monkeypatch.setattr(
        "gpuwm.ingest.preprocess_backend.CudaPreprocessBackend",
        lambda: pytest.fail("sealed CPU-only auto attempted CUDA"),
    )

    assert resolve_preprocess_backend("auto", workers=8).name == "cpu"
    with pytest.raises(ValueError, match="absent from the sealed"):
        resolve_preprocess_backend("cuda")


@requires_gpu
@pytest.mark.gpu
def test_rust_cpu_and_cuda_transforms_satisfy_declared_numeric_parity():
    import cupy as cp

    from gpuwm.ingest.horiz import interpolate_regular_gpu
    from gpuwm.ingest.vert import wrf_vert_interp_gpu

    backend = _backend()
    rng = np.random.default_rng(93)

    latitude = np.linspace(25.0, 45.0, 31, dtype=np.float64)
    longitude = np.linspace(-115.0, -80.0, 37, dtype=np.float64)
    target_lat = np.linspace(27.1, 42.8, 13)[:, None] + np.zeros((13, 17))
    target_lon = np.zeros((13, 17)) + np.linspace(-112.2, -82.3, 17)[None]
    horizontal_source = rng.normal(
        size=(5, latitude.size, longitude.size)).astype(np.float32)
    horizontal_source[:, 4, 8] = 0.0
    cpu_horizontal = backend.interpolate_regular(
        horizontal_source, latitude, longitude, target_lat, target_lon,
        method="parabolic", workers=8)
    cuda_horizontal = cp.asnumpy(interpolate_regular_gpu(
        horizontal_source, latitude, longitude, target_lat, target_lon,
        method="parabolic"))

    # WPS uses 1e-20 as a legitimate-zero sentinel, then tests b*c in its
    # overlapping parabolic branch.  CUDA flushes the 1e-40 product to zero;
    # the native CPU bridge must make that policy explicit rather than inherit
    # host subnormal behavior.  This reduced real GFS RH stencil previously
    # produced 0.0 on CUDA and 3.28849 on CPU.
    zero_sentinel_source = np.asarray([[
        [8.4, 9.3, 10.2, 11.4],
        [1.1, 2.0, 3.4, 5.1],
        [0.0, 0.0, 0.0, 0.2],
        [2.2, 1.7, 1.2, 0.8],
    ]], dtype=np.float32)
    sentinel_latitude = np.arange(4, dtype=np.float64)
    sentinel_longitude = np.arange(4, dtype=np.float64)
    sentinel_target_latitude = np.asarray([[1.002899169921875]])
    sentinel_target_longitude = np.asarray([[1.9373931884765625]])
    cpu_sentinel = backend.interpolate_regular(
        zero_sentinel_source, sentinel_latitude, sentinel_longitude,
        sentinel_target_latitude, sentinel_target_longitude,
        method="parabolic", workers=8)
    cuda_sentinel = cp.asnumpy(interpolate_regular_gpu(
        zero_sentinel_source, sentinel_latitude, sentinel_longitude,
        sentinel_target_latitude, sentinel_target_longitude,
        method="parabolic"))
    np.testing.assert_array_equal(cpu_sentinel, cuda_sentinel)
    np.testing.assert_array_equal(cpu_sentinel, np.zeros_like(cpu_sentinel))

    nsource, ntarget, ny, nx = 37, 83, 5, 7
    base = np.geomspace(100000.0, 7000.0, nsource)
    source_pressure = base[:, None, None] * (
        1.0 + rng.normal(0.0, 1.0e-4, (nsource, ny, nx)))
    source_pressure = np.sort(source_pressure, axis=0)[::-1]
    values = (250.0 + 0.02 * np.sqrt(source_pressure)
              + rng.normal(0.0, 0.2, source_pressure.shape))
    surface_pressure = rng.uniform(100100.0, 102500.0, (ny, nx))
    surface_values = rng.uniform(287.0, 294.0, (ny, nx))
    eta = np.linspace(0.998, 0.08, ntarget)[:, None, None]
    target_pressure = 7000.0 + eta * (surface_pressure[None] - 7000.0)
    cpu_vertical = backend.wrf_vertical_interpolate(
        values, surface_values, source_pressure, surface_pressure,
        target_pressure, extrap="temperature", workers=8)
    cuda_vertical = cp.asnumpy(wrf_vert_interp_gpu(
        cp.asarray(values, dtype=cp.float32),
        cp.asarray(surface_values, dtype=cp.float32),
        cp.asarray(source_pressure, dtype=cp.float32),
        cp.asarray(surface_pressure, dtype=cp.float32),
        cp.asarray(target_pressure, dtype=cp.float32),
        extrap="temperature"))
    cuda_common = resolve_preprocess_backend("cuda").prepare_wrf_vertical(
        source_pressure, surface_pressure, target_pressure).apply(
            values, surface_values, extrap="temperature")
    np.testing.assert_array_equal(cp.asnumpy(cuda_common), cuda_vertical)

    parity = compare_backend_outputs(
        {"horizontal": cuda_horizontal, "vertical": cuda_vertical},
        {"horizontal": cpu_horizontal, "vertical": cpu_vertical},
        rules={
            "horizontal": ArrayParityRule(rtol=3.0e-6, atol=3.0e-6),
            "vertical": ArrayParityRule(rtol=3.0e-5, atol=5.0e-3),
        },
    )
    assert parity["status"] == "PASS", parity


@requires_gpu
@pytest.mark.gpu
def test_full_horizontal_cpu_and_cuda_paths_satisfy_numeric_parity():
    import cupy as cp

    _backend()
    snapshot, grid, catalog = _horizontal_fixture()
    cpu = interpolate_era5_to_lambert(
        snapshot, grid, source_orography_catalog=catalog,
        backend="cpu", workers=8)
    cuda = interpolate_era5_to_lambert(
        snapshot, grid, source_orography_catalog=catalog, backend="cuda")
    rules = {
        name: ArrayParityRule(
            mode="byte_exact", rtol=0.0, atol=0.0)
        if name in {"PSFC", "LANDSEA", "SKINTEMP", "SST", "XICE", "ST000007",
                    "SM000007", "SNOW_EC", "SOURCE_OROGRAPHY"}
        else ArrayParityRule(rtol=3.0e-6, atol=2.0e-4)
        for name in cpu.fields
    }
    parity = compare_backend_outputs(
        {name: cp.asnumpy(value) if hasattr(value, "get") else value
         for name, value in cuda.fields.items()},
        cpu.fields, rules=rules)
    assert parity["status"] == "PASS", parity


def test_backend_workload_identity_excludes_backend_and_recomputes_parity(
        tmp_path):
    manifest = tmp_path / "source.json"
    manifest.write_text('{"schema":"fixture","cycle":"2026-07-20T00:00:00Z"}',
                        encoding="utf-8")
    workload = build_workload_contract(
        manifest,
        target={"domains": [{"id": 1, "nx": 7, "ny": 5}]},
        vertical={"eta_levels": [1.0, 0.73, 0.2, 0.0], "p_top": 5000.0},
        state_inventory=("theta", "qv", "u", "v"),
        forcing_times=("2026-07-20T00:00:00Z", "2026-07-20T03:00:00Z"),
    )
    value = np.arange(35, dtype=np.float32).reshape(5, 7)
    implementation = hashlib.sha256(b"cpu-fixture").hexdigest()
    receipt = build_backend_receipt(
        workload, backend="cpu-rust-threads",
        implementation_sha256=implementation, workers=8,
        outputs={"theta": value})
    assert receipt["workload_sha256"] == workload["workload_sha256"]
    assert receipt["outputs"]["theta"]["sha256"] == hashlib.sha256(
        memoryview(value).cast("B")).hexdigest()

    numeric = value.copy()
    numeric[2, 3] = np.nextafter(numeric[2, 3], np.float32(np.inf))
    result = compare_backend_outputs(
        {"theta": value}, {"theta": numeric},
        rules={"theta": ArrayParityRule(rtol=1.0e-6, atol=1.0e-6)})
    assert result["status"] == "PASS"
    strict = compare_backend_outputs(
        {"theta": value}, {"theta": numeric},
        rules={"theta": ArrayParityRule(
            mode="byte_exact", rtol=0.0, atol=0.0)})
    assert strict["status"] == "FAIL"


def test_explicit_missing_cpu_bridge_reports_path(tmp_path):
    missing = tmp_path / "missing-bridge"
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        resolve_cpu_bridge(missing)


def test_cpu_env_override_fails_loud_and_wins_when_present(monkeypatch,
                                                           tmp_path):
    """GPUWM_CPU_PREPROCESS_BRIDGE naming a missing file is a hard error
    citing the variable and path -- never a silent fall-through to a
    different library (the bridges.py override contract, now shared)."""
    from gpuwm.ingest import cpu_backend

    missing = tmp_path / "missing-cpu-library"
    monkeypatch.setenv(cpu_backend.CPU_BRIDGE_ENV, str(missing))
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_cpu_bridge()
    assert cpu_backend.CPU_BRIDGE_ENV in str(excinfo.value)
    assert str(missing) in str(excinfo.value)

    present = tmp_path / "gpuwm_preprocess_cpu.dll"
    present.write_bytes(b"not a real library")
    monkeypatch.setenv(cpu_backend.CPU_BRIDGE_ENV, str(present))
    assert resolve_cpu_bridge() == present.resolve()


def test_cpu_library_candidates_share_the_bridge_resolution_order(
        monkeypatch):
    """One resolver, not two: the CPU library walks exactly the bridge
    executables' directories in the same order (env override, checkout
    release/debug, libexec, user dir)."""
    from gpuwm import bridges
    from gpuwm.ingest import cpu_backend

    monkeypatch.delenv(cpu_backend.CPU_BRIDGE_ENV, raising=False)
    monkeypatch.delenv(bridges.BRIDGE_ENV["grib1_bridge"], raising=False)
    library_dirs = [
        candidate.parent
        for candidate in cpu_backend.cpu_bridge_candidates()]
    bridge_dirs = [
        candidate.parent
        for candidate in bridges.bridge_candidates("grib1_bridge")]
    assert library_dirs == bridge_dirs
