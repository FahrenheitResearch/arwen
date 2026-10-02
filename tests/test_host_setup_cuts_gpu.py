"""Gather/scatter and pinned receipt identity on the active card."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize('scheme', [8, 28])
@pytest.mark.parametrize('subnormal', [False, True])
def test_device_gather_matches_frozen_closure(scheme, subnormal):
    import cupy as cp
    from gpuwm.ingest import real
    from gpuwm.boundary_fields import COLD_START_SEEDED_NUMBERS
    from host_setup_reference import namespace
    rng = np.random.default_rng(281)
    shape = (3, 9, 11)
    host, device = SimpleNamespace(), SimpleNamespace()
    for mass, number in COLD_START_SEEDED_NUMBERS[scheme]:
        q = rng.uniform(0., 0.003, shape).astype(np.float32)
        q.ravel()[::3] = 0
        q.ravel()[1::17] = (np.nextafter(np.float32(0.), np.float32(1.))
                            if subnormal else np.float32(5e-13))
        n = rng.uniform(1., 1e6, shape).astype(np.float32)
        n.ravel()[::2] = 0
        setattr(host, mass, q.copy())
        setattr(host, number, n.copy())
        setattr(device, mass, cp.asarray(q))
        setattr(device, number, cp.asarray(n))
    cfg = SimpleNamespace(mp_physics=scheme)
    alt = rng.uniform(0.2, 3., shape).astype(np.float32)
    kw = dict(temperature=rng.uniform(220., 310., shape).astype(np.float32),
              aerosol_number=rng.uniform(0., 1e8, shape).astype(np.float32),
              landmask=rng.integers(0, 2, shape[-2:]))
    def run(function, state, xp):
        with np.errstate(all='ignore'):
            try:
                return function(state, xp, cfg, alt, **kw)
            except ValueError as error:
                return str(error)
    expected = run(namespace()['_thompson_cold_start_moment_closure'], host, np)
    actual = run(real._thompson_cold_start_moment_closure, device, cp)
    assert json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True)
    for _, number in COLD_START_SEEDED_NUMBERS[scheme]:
        assert getattr(host, number).tobytes() == getattr(device, number).get().tobytes()


@requires_gpu
def test_pinned_hash_matches_original_bytes():
    import cupy as cp
    from gpuwm.ingest import real
    rng = np.random.default_rng(281)
    field = rng.normal(size=(7, 21, 23)).astype(np.float32)
    field.ravel()[::5] = np.float32(-0.)
    on_card = cp.asarray(field)
    copied = real._receipt_host_float32(on_card)
    assert copied.tobytes() == field.tobytes()
    with real._column_pool(2) as pool:
        hasher = real._ReceiptHasher(pool)
        assert hasher.submit(on_card).result() == real.array_correspondence_fingerprint(field)
    for field in (np.zeros((2, 17), np.float32),
                  np.full((2, 17), np.float32(-0.)),
                  np.array([np.nextafter(np.float32(0.), np.float32(1.)),
                            np.nextafter(np.float32(0.), np.float32(-1.)), 0., -0.]),
                  np.array([1., 3., -5., 2.], np.float32)):
        field = field.astype(np.float32)
        copied = real._receipt_host_float32(cp.asarray(field))
        assert real._receipt_array_fingerprint(copied) == real.array_correspondence_fingerprint(field)


@requires_gpu
def test_device_perimeter_matches_full_readback_for_all_bit_classes(monkeypatch):
    import cupy as cp
    from gpuwm.ingest import lateral_bc
    rng = np.random.default_rng(281)
    bits = rng.integers(0, 2**32, (3, 10, 12), dtype=np.uint32)
    bits.ravel()[:6] = [0, 0x80000000, 1, 0x80000001, 0x7fc00001, 0xffc00001]
    fields = {'qv': cp.asarray(bits.view(np.float32))}
    monkeypatch.setattr(lateral_bc, '_coupled_device_fields', lambda state: fields)
    old = lateral_bc.StateBoundaryFrames(spec_bdy_width=3, spec_zone=1, relax_zone=2)
    new = lateral_bc.StateBoundaryFrames(spec_bdy_width=3, spec_zone=1, relax_zone=2)
    with np.errstate(all='ignore'):
        old.add_snapshot(lateral_bc.domain_boundary_snapshot(None))
        new.add_state(None)
    for side in old._frames[0]:
        assert old._frames[0][side]['qv'].tobytes() == new._frames[0][side]['qv'].tobytes()
