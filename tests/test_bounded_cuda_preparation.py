"""Admission and bitwise gates for host-retained CUDA preparation."""
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ingest import preprocess_backend as backend
from gpuwm.ingest.bounded_cuda import BoundedCudaPreprocessBackend
from gpuwm.ingest.preparation_price import price_preparation, SourceInventory


def test_own_grid_admission_uses_chunks_on_32_gib(monkeypatch):
    monkeypatch.setattr("gpuwm.ingest.preparation_workers.host_available_bytes", lambda: 128 * 1024**3)
    monkeypatch.setattr("gpuwm.ingest.boundary_stream.process_memory_bytes", lambda: (3 * 1024**3, 5 * 1024**3))
    from gpuwm.config import RunConfig
    cfg = RunConfig(nx=1797, ny=1057, nz=50, dx=3000, dy=3000,
                    dt=10, run_seconds=60, ztop=20000, moist=True,
                    terrain_opt=1, mp_physics=8)
    inventory = SourceInventory(levels=39, level_fields=11, surface_planes=29,
                                source_points=1799 * 1059, device_real_columns=True)
    price = price_preparation("mapped", [cfg], inventory, platform="linux")
    assert price.need_bytes > 32 * 1024**3
    original = backend.CudaPreprocessBackend(host_workers=1)
    original.selection = {"requested": "auto"}
    chosen = backend.admit_preparation(original, price,
        probe={"free_bytes": 31 * 1024**3, "total_bytes": 32 * 1024**3})
    assert isinstance(chosen, BoundedCudaPreprocessBackend)
    assert chosen.array_module is np
    assert chosen.selection["device_fit"]["fits"]
    assert chosen.selection["device_fit"]["need_bytes"] < 2 * 1024**3
    assert chosen.selection["device_fit"]["unchunked_need_bytes"] == price.need_bytes
    assert chosen.selection["host_fit"]["fits"]
    assert chosen.selection["host_fit"]["producer_rss_bytes"] == 3 * 1024**3
    assert chosen.selection["host_fit"]["producer_peak_bytes"] == (
        3 * 1024**3 + chosen.selection["host_fit"]["need_bytes"])
    from types import SimpleNamespace
    from gpuwm.ingest.preparation_price import price_preparation_floor
    floor = price_preparation_floor("mapped", SimpleNamespace(domains=[SimpleNamespace(run=cfg)]))
    before_decode = backend.admit_preparation(original, floor,
        probe={"free_bytes": 31 * 1024**3, "total_bytes": 32 * 1024**3})
    monkeypatch.setattr("gpuwm.ingest.boundary_stream.process_memory_bytes", lambda: (7 * 1024**3, 9 * 1024**3))
    binding = backend.admit_preparation(before_decode, price,
        probe={"free_bytes": 31 * 1024**3, "total_bytes": 32 * 1024**3})
    assert binding.selection["host_fit"]["producer_rss_bytes"] == 7 * 1024**3
    monkeypatch.setattr("gpuwm.ingest.boundary_stream.process_memory_bytes", lambda: (13 * 1024**3, 15 * 1024**3))
    binding = backend.admit_preparation(binding, price,
        probe={"free_bytes": 31 * 1024**3, "total_bytes": 32 * 1024**3})
    assert binding.selection["host_fit"]["producer_rss_bytes"] == 7 * 1024**3
    monkeypatch.setattr("gpuwm.ingest.preparation_workers.host_available_bytes", lambda: 8 * 1024**3)
    from gpuwm.ingest.memory_refusal import InitializationMemoryRefused
    with pytest.raises(InitializationMemoryRefused, match="exhaust host memory"):
        backend.admit_preparation(original, price,
            probe={"free_bytes": 31 * 1024**3, "total_bytes": 32 * 1024**3})


def test_a_build_that_fits_the_card_stays_on_the_card_when_the_host_cannot_retain_it(
        monkeypatch, capsys):
    """THE BREAKAGE: 2.8.5 staging refused single-card runs 2.8.4 prepared.

    Every mapped CUDA preparation above the 1 GiB pool took the
    host-retained bounded path, which needs the whole array envelope in
    host RAM and refused when the host had less, although the whole build
    fit the card and the full-device preparation retains nothing on the
    host.  The case is the one the 2.8.5 review reproduced: 700x600x50,
    priced near 14.3 GiB, 22 GiB free on the card, 6 GiB of host RAM.
    """
    from types import SimpleNamespace
    from gpuwm.config import RunConfig
    from gpuwm.ingest.memory_refusal import InitializationMemoryRefused
    from gpuwm.ingest.preparation_price import price_preparation_floor

    gib = 1024**3
    monkeypatch.setattr("gpuwm.ingest.preparation_workers.host_available_bytes", lambda: 6 * gib)
    monkeypatch.setattr("gpuwm.ingest.boundary_stream.process_memory_bytes", lambda: (3 * gib, 5 * gib))
    cfg = RunConfig(nx=700, ny=600, nz=50, dx=3000, dy=3000,
                    dt=10, run_seconds=60, ztop=20000, moist=True,
                    terrain_opt=1, mp_physics=8)
    inventory = SourceInventory(levels=39, level_fields=11, surface_planes=29,
                                source_points=702 * 602, device_real_columns=True)
    price = price_preparation("mapped", [cfg], inventory, platform="linux")
    card = {"free_bytes": 22 * gib, "total_bytes": 24 * gib}
    # The whole build fits the card; the host cannot retain its envelope.
    assert 6 * gib < price.need_bytes - price.terms["cuda_context"]
    assert price.need_bytes < card["free_bytes"]

    def named(requested):
        original = backend.CudaPreprocessBackend(host_workers=1)
        original.selection = {"requested": requested}
        return original

    for requested in ("cuda", "auto"):
        chosen = backend.admit_preparation(named(requested), price, probe=card)
        # The full-device backend 2.8.4 admitted, with its whole price.
        assert type(chosen) is backend.CudaPreprocessBackend
        assert chosen.selection["requested"] == requested
        assert chosen.selection["device_fit"]["fits"] is True
        assert chosen.selection["device_fit"]["need_bytes"] == price.need_bytes
        # No bounded contract is published for a chained forecast to read.
        assert "chunking" not in chosen.selection
        host_fit = chosen.selection["host_fit"]
        assert host_fit["fits"] is False
        assert host_fit["available_bytes"] == 6 * gib
        assert host_fit["route"].startswith("full-device preparation")
    assert capsys.readouterr().err.count("preparing on the card") == 2

    # The mapped door's own order: the floor is admitted before the decode
    # (bounded, while the host still holds that envelope), and the decoded
    # price after it.  Nothing was built on the floor's contract, so the
    # decoded build still takes the card.
    floor = price_preparation_floor(
        "mapped", SimpleNamespace(domains=[SimpleNamespace(run=cfg)]))
    monkeypatch.setattr("gpuwm.ingest.preparation_workers.host_available_bytes", lambda: 128 * gib)
    before_decode = backend.admit_preparation(named("cuda"), floor, probe=card)
    assert isinstance(before_decode, BoundedCudaPreprocessBackend)
    assert before_decode.selection["host_fit"]["price_is_floor"] is True
    monkeypatch.setattr("gpuwm.ingest.preparation_workers.host_available_bytes", lambda: 6 * gib)
    after_decode = backend.admit_preparation(before_decode, price, probe=card)
    assert type(after_decode) is backend.CudaPreprocessBackend
    assert after_decode.host_workers == 1
    assert "chunking" not in after_decode.selection
    assert after_decode.selection["device_fit"]["need_bytes"] == price.need_bytes
    # A second admission of the same decision says its line once.
    capsys.readouterr()
    again = backend.admit_preparation(after_decode, price, probe=card)
    assert type(again) is backend.CudaPreprocessBackend
    assert "preparing on the card" not in capsys.readouterr().err

    # The two refusals that stand, each naming its breakage.  Neither the
    # card nor the host holds the build:
    with pytest.raises(InitializationMemoryRefused, match="exhaust host memory"):
        backend.admit_preparation(
            named("cuda"), price,
            probe={"free_bytes": 12 * gib, "total_bytes": 24 * gib})
    # and a bounded backend already admitted on a decoded price, whose one
    # batch a chained forecast may have reserved the card against:
    monkeypatch.setattr("gpuwm.ingest.preparation_workers.host_available_bytes", lambda: 128 * gib)
    bounded = backend.admit_preparation(named("cuda"), price, probe=card)
    assert isinstance(bounded, BoundedCudaPreprocessBackend)
    assert bounded.selection["host_fit"]["price_is_floor"] is False
    monkeypatch.setattr("gpuwm.ingest.preparation_workers.host_available_bytes", lambda: 6 * gib)
    with pytest.raises(InitializationMemoryRefused, match="exhaust host memory"):
        backend.admit_preparation(bounded, price, probe=card)


def _equal(left, right):
    left = left.get() if hasattr(left, "get") else left
    right = right.get() if hasattr(right, "get") else right
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert left.tobytes() == right.tobytes()


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("cells", [23, 192, 4096])
def test_bounded_real_state_and_boundaries_are_bit_identical(cells):
    from test_real_init import _analyzed_hrrr_real_init
    from gpuwm.ingest.lateral_bc import StateBoundaryFrames
    ordinary = backend.CudaPreprocessBackend(host_workers=1)
    bounded = BoundedCudaPreprocessBackend(device_budget_bytes=1024**3, host_workers=1)
    bounded.chunk_cells = cells
    frames = []
    reference_results = None
    for engine in (ordinary, bounded):
        result_states = []
        boundary = StateBoundaryFrames(spec_bdy_width=3, spec_zone=1, relax_zone=2)
        for index in range(2):
            result, cfg = _analyzed_hrrr_real_init(
                8, shape=(11, 13), terrain_m=100.0 + index,
                preprocess_backend=engine, state_backend="preprocess")
            if reference_results is not None:
                expected = reference_results[index]
                for name, value in vars(expected.state).items():
                    if hasattr(value, "shape") and hasattr(value, "dtype"):
                        _equal(value, getattr(result.state, name))
                for name in ("surface_pressure", "surface_qv", "dry_mass", "dry_pressure",
                             "total_pressure", "total_geopotential", "total_specific_volume",
                             "integrated_moisture_pressure"):
                    _equal(getattr(expected, name), getattr(result, name))
                assert expected.hydrometeor_initialization == result.hydrometeor_initialization
                assert expected.surface_moisture_floor == result.surface_moisture_floor
            result_states.append(result)
            boundary.add_state(result.state, index=index)
        if reference_results is None:
            reference_results = result_states
        frames.append(boundary._frames)
    for index in frames[0]:
        for side in frames[0][index]:
            for field in frames[0][index][side]:
                _equal(frames[0][index][side][field], frames[1][index][side][field])


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("method", ["nearest", "bilinear", "parabolic"])
def test_bounded_horizontal_level_and_row_batches(method):
    rng = np.random.default_rng(284)
    lat, lon = np.arange(17, dtype=np.float64), np.arange(19, dtype=np.float64)
    ty, tx = np.meshgrid(np.linspace(0, 16, 23), np.linspace(0, 18, 21), indexing="ij")
    fields = rng.normal(size=(7, 17, 19)).astype(np.float32)
    fields.flat[:4] = [0.0, -0.0, np.nextafter(np.float32(0), np.float32(1)),
                       np.nextafter(np.float32(0), np.float32(-1))]
    full = backend.CudaPreprocessBackend().regular_plan(lat, lon, ty, tx)
    bounded = BoundedCudaPreprocessBackend(device_budget_bytes=1024**3)
    bounded.chunk_cells = 53
    small = bounded.regular_plan(lat, lon, ty, tx)
    _equal(full.apply(fields, method=method, source_support=True),
           small.apply(fields, method=method, source_support=True))


@requires_gpu
@pytest.mark.gpu
def test_bounded_stagger_retains_neighbor_rows_and_special_words():
    from gpuwm.ingest import real_device
    rng = np.random.default_rng(285)
    words = rng.integers(0, 2**64 - 1, (3, 7, 11), dtype=np.uint64)
    words.flat[:8] = [0, 0x8000000000000000, 1, 0x8000000000000001,
                     0x7ff8000000001234, 0xfff8000000005678,
                     0x7ff0000000000000, 0xfff0000000000000]
    pressure = words.view(np.float64)
    bounded = BoundedCudaPreprocessBackend(device_budget_bytes=1024**3)
    bounded.chunk_cells = 40
    for name in ("_pressure_at_u", "_pressure_at_v"):
        _equal(getattr(real_device, name)(pressure), getattr(bounded.real_ops, name)(pressure))


@requires_gpu
@pytest.mark.gpu
def test_bounded_rh_initialization_retains_ordinary_cuda_bits():
    from test_real_init import _synthetic_horizontal_snapshot
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.ingest.real import initialize_real
    cfg = RunConfig(nx=13, ny=12, nz=7, dx=12000, dy=12000, ztop=16000,
                    dt=30, run_seconds=60, moist=True, terrain_opt=1,
                    hybrid_opt=2, etac=0.2, base_temp=290.0, mp_physics=8)
    source = _synthetic_horizontal_snapshot(np, cfg.ny, cfg.nx)
    terrain = np.zeros((cfg.ny, cfg.nx))
    ordinary = backend.CudaPreprocessBackend(host_workers=1)
    bounded = BoundedCudaPreprocessBackend(device_budget_bytes=1024**3, host_workers=1)
    bounded.chunk_cells = 23
    results = [initialize_real(source, cfg, make_vertical_coord(cfg.nz, hybrid_opt=2,
                    etac=0.2, eta_levels=np.linspace(1.0, 0.0, cfg.nz + 1)), terrain,
                    source_orography=terrain, p_top=10000, preprocess_backend=engine,
                    state_backend="preprocess") for engine in (ordinary, bounded)]
    for name, value in vars(results[0].state).items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            _equal(value, getattr(results[1].state, name))


@requires_gpu
@pytest.mark.gpu
def test_bounded_wind_rotation_broadcasts_in_batches():
    rng = np.random.default_rng(286)
    u, v = (rng.normal(size=(7, 5, 11)).astype(np.float32) for _ in range(2))
    angle = rng.uniform(-1.0, 1.0, size=(5, 11))
    sine, cosine = np.sin(angle), np.cos(angle)
    ordinary = backend.CudaPreprocessBackend()
    bounded = BoundedCudaPreprocessBackend(device_budget_bytes=1024**3)
    bounded.chunk_cells = 23
    for expected, actual in zip(ordinary.rotate_earth_to_grid(u, v, sine, cosine),
                                bounded.rotate_earth_to_grid(u, v, sine, cosine)):
        _equal(expected, actual)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("layered", [False, True])
def test_host_retained_aerosol_closure_never_uploads_a_whole_field(monkeypatch, layered):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.ingest.closure_device import thompson_cold_start_moment_closure
    shape = (3, 11, 13)
    rng = np.random.default_rng(287)
    fields = {name: rng.uniform(0, 0.001, shape).astype(np.float32)
              for name in ("qc", "qr", "qi")}
    fields.update({name: np.zeros(shape, np.float32) for name in ("nc", "nr", "ni")})
    source = SimpleNamespace(**{name: cp.asarray(value) for name, value in fields.items()})
    target = SimpleNamespace(**{name: value.copy() for name, value in fields.items()})
    cfg = SimpleNamespace(mp_physics=28)
    alt = rng.uniform(0.5, 2.0, shape).astype(np.float32)
    aerosol = rng.uniform(0, 1e8, shape if layered else shape[-2:]).astype(np.float32)
    aerosol.flat[::3] = 0
    land = rng.integers(0, 2, shape[-2:]).astype(np.float32)
    temperature = rng.uniform(230, 300, shape).astype(np.float32)
    expected = thompson_cold_start_moment_closure(source, cp, cfg, alt,
        aerosol_number=aerosol, landmask=land, temperature=temperature)
    upload = cp.asarray
    def bounded_upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray):
            assert value.shape != shape, "uploaded a whole host field"
        return upload(value, *args, **kwargs)
    monkeypatch.setattr(cp, "asarray", bounded_upload)
    actual = thompson_cold_start_moment_closure(target, np, cfg, alt,
        aerosol_number=aerosol, landmask=land, temperature=temperature, chunk_cells=23)
    assert expected == actual
    for name in ("nc", "nr", "ni"):
        _equal(getattr(source, name), getattr(target, name))


@requires_gpu
@pytest.mark.gpu
def test_temperature_gather_accepts_strided_host_arrays_without_flattening():
    import cupy as cp
    from gpuwm.ingest.closure_device import make_temperature_provider
    from gpuwm.ingest.real import _temperature_from_potential_temperature
    class GatherOnly(np.ndarray):
        def ravel(self, *args, **kwargs):
            raise AssertionError("flattened a whole strided host field")
    rng = np.random.default_rng(288)
    theta = rng.uniform(250, 350, (3, 5, 7)).transpose(1, 2, 0)
    pressure = np.broadcast_to(np.array([90000.0, 50000.0, 10000.0]), theta.shape)
    expected = _temperature_from_potential_temperature(theta, pressure).astype(np.float32)
    indices = np.array([0, 1, 6, 29, 55, theta.size - 1])
    provider = make_temperature_provider(theta.view(GatherOnly), pressure.view(GatherOnly))
    _equal(expected.flat[indices], provider(cp.asarray(indices)))
