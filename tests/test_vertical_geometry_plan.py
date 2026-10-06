"""Built immutable Rust geometry preserves the original WRF field bytes."""
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pytest

from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend


@pytest.fixture
def backend():
    try:
        backend = resolve_preprocess_backend("cpu", workers=4)
    except (FileNotFoundError, OSError) as error:
        pytest.skip(str(error))
    if not hasattr(backend._native._library, "gpuwm_wrf_vertical_plan_new"):
        pytest.skip("built CPU bridge lacks immutable vertical geometry")
    return backend


def _columns(nsource=73, ntarget=17, *, ascending=False):
    rng = np.random.default_rng(394)
    shape = (5, 7)
    source = np.geomspace(105000., 4000., nsource).astype(np.float32)[:, None, None]
    source = source * rng.uniform(.999, 1.001, (1, *shape)).astype(np.float32)
    surface = rng.uniform(94000., 107000., shape).astype(np.float32)
    target = np.geomspace(110000., 6000., ntarget).astype(np.float32)[:, None, None]
    target = target * np.ones((1, *shape), np.float32)
    values = rng.uniform(-10., 320., source.shape).astype(np.float32)
    values[::3, 0, 0] = -0.0
    values[1::4, 0, 1] = np.nextafter(np.float32(0), np.float32(1))
    sv = rng.uniform(280., 300., shape).astype(np.float32)
    if ascending:
        source, values = source[::-1].copy(), values[::-1].copy()
    return source, surface, target, values, sv


def _same_call(backend, plan, source, surface, target, values, sv, **options):
    try:
        original = backend._native.wrf_vertical_interpolate(
            values, sv, source, surface, target, workers=4, **options)
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error), match="^" + __import__("re").escape(str(error)) + "$"):
            plan.apply(values, sv, **options)
        return
    planned = plan.apply(values, sv, **options)
    assert planned.shape == original.shape
    assert planned.dtype == original.dtype
    assert planned.tobytes() == original.tobytes()


@pytest.mark.parametrize("nsource,ascending", product((2, 8, 73, 129), (False, True)))
def test_original_native_bytes_match_across_every_operator_mode(backend, nsource, ascending):
    source, surface, target, values, sv = _columns(nsource, ascending=ascending)
    plan = backend.prepare_wrf_vertical(source, surface, target)
    for logp, force, zap, vboundb, extrap in product(
            (False, True), (0, 1, target.shape[0]),
            (0., 499.99997, 500., 500.00003), (0, 1, 4, target.shape[0] + 1),
            ("constant", "temperature")):
        _same_call(backend, plan, source, surface, target, values, sv,
            interp_in_logp=logp, force_sfc_in_vinterp=force,
            zap_close_levels=zap, vboundb=vboundb, extrap=extrap)


def test_production_source_level_and_species_mask_replays_keep_every_bit(backend):
    source, surface, target, values, sv = _columns()
    plan = backend.prepare_wrf_vertical(source, surface, target)
    zeros = np.zeros(surface.shape, np.float32)
    options = {"interp_in_logp": True, "extrap": "constant",
               "vboundb": target.shape[0] + 1}
    for level in range(source.shape[0]):
        mask = np.zeros(source.shape, np.float32)
        mask[level] = 1.
        _same_call(backend, plan, source, surface, target, mask, zeros, **options)
    for mask in (values > 0, values > 100, values > 200, values < 0):
        _same_call(backend, plan, source, surface, target,
                   mask.astype(np.float32), zeros, **options)
    assert next(iter(plan._geometry.values())) is not None


@pytest.mark.parametrize("ascending", (False, True))
def test_native_orientation_uses_untouched_input_buffers(backend, monkeypatch, ascending):
    from gpuwm.ingest import cpu_backend
    source, surface, target, values, sv = _columns(ascending=ascending)
    reference = backend._native.wrf_vertical_interpolate(
        values, sv, source, surface, target, workers=4)
    before = [array.tobytes() for array in (source, surface, target, values, sv)]
    for array in (source, surface, target, values, sv):
        array.flags.writeable = False
    def forbidden(*args, **kwargs):
        raise AssertionError("the native default must not compare pressure arrays in Python")
    monkeypatch.setattr(cpu_backend.np, "all", forbidden)
    plan = backend.prepare_wrf_vertical(source, surface, target)
    result = plan.apply(values, sv)
    assert next(iter(plan._geometry.values())) is not None
    assert result.tobytes() == reference.tobytes()
    assert [array.tobytes() for array in (source, surface, target, values, sv)] == before


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf, np.finfo(np.float32).max))
def test_nonfinite_and_overflow_field_behavior_matches_original(backend, bad):
    source, surface, target, values, sv = _columns()
    plan = backend.prepare_wrf_vertical(source, surface, target)
    values[3, 1, 2] = bad
    _same_call(backend, plan, source, surface, target, values, sv)
    sv[0, 1] = bad
    _same_call(backend, plan, source, surface, target, values, sv)


@pytest.mark.parametrize("which", ("source", "surface", "target"))
def test_live_geometry_mutation_never_serves_stale_plan_bytes(backend, which):
    source, surface, target, values, sv = _columns()
    plan = backend.prepare_wrf_vertical(source, surface, target)
    _same_call(backend, plan, source, surface, target, values, sv)
    changed = {"source": source, "surface": surface, "target": target}[which]
    changed.flat[0] += np.float32(.5)
    _same_call(backend, plan, source, surface, target, values, sv)
    assert next(iter(plan._geometry.values())) is None


def test_plan_budget_decline_keeps_original_checked_road(backend, monkeypatch):
    from gpuwm.ingest import preparation_workers
    monkeypatch.setattr(preparation_workers, "host_available_bytes", lambda: 0)
    source, surface, target, values, sv = _columns()
    plan = backend.prepare_wrf_vertical(source, surface, target)
    _same_call(backend, plan, source, surface, target, values, sv)
    assert next(iter(plan._geometry.values())) is None


def test_all_live_plans_share_memory_budget_and_release_once(backend, monkeypatch):
    import gc
    from gpuwm.ingest import cpu_backend, preparation_workers
    gc.collect()
    initial = cpu_backend._vertical_geometry_bytes
    source, surface, target, values, sv = _columns()
    first = backend._native.prepare_vertical_geometry(source, surface, target, workers=4)
    assert first is not None
    retained = first.nbytes
    assert cpu_backend._vertical_geometry_bytes == initial + retained
    monkeypatch.setattr(preparation_workers, "host_available_bytes", lambda: retained * 8)
    second = backend._native.prepare_vertical_geometry(source, surface, target, workers=4)
    assert second is None
    first.close()
    first.close()
    assert cpu_backend._vertical_geometry_bytes == initial
    third = backend._native.prepare_vertical_geometry(source, surface, target, workers=4)
    assert third is not None
    third.close()
    assert cpu_backend._vertical_geometry_bytes == initial


def test_older_bridge_without_additive_plan_keeps_original_bytes(backend, monkeypatch):
    monkeypatch.setattr(backend._native, "prepare_vertical_geometry", lambda *a, **k: None)
    source, surface, target, values, sv = _columns()
    _same_call(backend, backend.prepare_wrf_vertical(source, surface, target),
               source, surface, target, values, sv)


def test_unsupported_mode_type_retains_original_validation_order(backend):
    source, surface, target, values, sv = _columns()
    plan = backend.prepare_wrf_vertical(source, surface, target)
    _same_call(backend, plan, source, surface, target, values[:3], sv,
               zap_close_levels=500 + 0j)
    assert not plan._geometry


def test_concurrent_fields_keep_plan_owner_alive(backend):
    source, surface, target, values, sv = _columns()
    plan = backend.prepare_wrf_vertical(source, surface, target)
    def apply(offset):
        field = values + np.float32(offset)
        expected = backend._native.wrf_vertical_interpolate(
            field, sv, source, surface, target, workers=4)
        assert plan.apply(field, sv).tobytes() == expected.tobytes()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(apply, range(8)))


def test_native_close_waits_for_the_same_owners_active_apply(backend, monkeypatch):
    from threading import Event
    source, surface, target, values, sv = _columns()
    native = backend._native.prepare_vertical_geometry(source, surface, target, workers=4)
    assert native is not None
    entered, proceed, close_started = Event(), Event(), Event()
    original = native._apply
    def blocking_apply(*args, **kwargs):
        entered.set()
        assert proceed.wait(10)
        # A close racing this point would invalidate the actual native ABI.
        assert native._release.alive
        return original(*args, **kwargs)
    monkeypatch.setattr(native, "_apply", blocking_apply)
    def close():
        close_started.set()
        native.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(native.apply, values, sv, source, surface, target,
                             extrap="constant", vboundb=4, workers=4)
        assert entered.wait(10)
        closing = pool.submit(close)
        assert close_started.wait(10)
        assert not closing.done()
        proceed.set()
        result = active.result(10)
        closing.result(10)
    assert not native._release.alive
    assert result.tobytes() == backend._native.wrf_vertical_interpolate(
        values, sv, source, surface, target, workers=4).tobytes()
