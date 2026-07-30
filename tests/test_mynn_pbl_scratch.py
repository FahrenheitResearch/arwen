"""Can a physics scheme allocate its working set outside the arena again?

This file is the gate, and it exists because MYNN got all the way to a
production-width timing measurement while allocating **54,595 bytes per
column in 484 pool allocations per step** that no part of the preflight knew
about.  At the 360,000 columns of a d04 nest that is 18,743.8 MiB per step
against an estimate that did not move at all.  The failure mode is not
slowness.  It is that ``run_alloc_preflight`` returns a comfortable headroom
number for a configuration that will run the card out of memory, on hardware
with no ECC, where the observed symptom of running out is corrupted output
rather than an exception.

Five properties are gated, and each one is shown failing before it is shown
passing, because a gate nobody has watched fail is not evidence:

1. **No raw device allocation in a physics GPU module, and no new one
   anywhere.**  That one now lives in
   ``tests/test_physics_allocation_inventory.py``, and it had to move: it
   reads source with :mod:`ast` and opens no device, but ``_carried_hash``
   below imports cupy, and ``conftest`` marks a whole module ``gpu`` when a
   non-test function in it does.  So the ratchet ran only on the card, was
   red there for days, and no ``GPUWM_NO_LOCAL_GPU=1`` run could report it --
   four Noah-MP modules and one RUC function were missing from the inventory
   the entire time.  Do not move it back.
2. **Every slot is write-before-read.**
   :func:`test_poisoning_a_constant_zero_slot_is_visible` shows that
   corrupting a slot that is *not* write-before-read is caught, so the
   poison lever can see a lie, before
   :func:`test_poisoning_the_workspace_leaves_the_forecast_alone` NaN-fills
   every slot that is classified write-before-read and requires the same
   hash.
3. **The chunk boundary is not a seam.**  A carried forecast at several
   chunk widths must produce one identical hash.
4. **The declared workspace does not grow with the domain.**
5. **The registry prices exactly what the solver asks for**, in both
   directions.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu


def test_the_declared_workspace_does_not_grow_with_the_domain():
    """A 600x600 nest must declare no more workspace than a 501x501 one.

    This is the property that lets a launch gate refuse a configuration.  A
    workspace proportional to ``ny * nx`` would price honestly and still not
    fit: at 360,000 columns the old working set was 18.3 GiB of churn on a
    card whose correctness rail is 29,500 MiB total.

    ``MYNN_PBL_COLUMN_CHUNK`` is a ceiling, not a fixed width, so the claim
    is a bound: identical for every domain at or above the chunk, and never
    larger below it.  A test that demanded equality with a 64x64 grid would
    be demanding that a 4,096-column domain allocate a 16,384-column
    workspace, which is not the property anyone wants.
    """
    from gpuwm.config import RunConfig
    from gpuwm.core import preflight as pf
    from gpuwm.core.mynn_pbl_scratch import MYNN_PBL_COLUMN_CHUNK

    common = dict(nz=49, dx=750.0, dy=750.0, ztop=16000.0, dt=3.0,
                  run_seconds=60.0, moist=True, mp_physics=6,
                  sf_sfclay_physics=5, sf_surface_physics=2,
                  bl_pbl_physics=5)
    d04 = RunConfig(nx=600, ny=600, **common)
    d03 = RunConfig(nx=501, ny=501, **common)
    tiny = RunConfig(nx=64, ny=64, **common)

    def chunk_slots(cfg):
        return {name: shape
                for name, shape in pf.mynn_pbl_scratch_slots(cfg).items()
                if not name.startswith("mynn_pbl_out_")}

    assert pf.mynn_pbl_column_chunk(d04) == MYNN_PBL_COLUMN_CHUNK
    assert pf.mynn_pbl_column_chunk(d03) == MYNN_PBL_COLUMN_CHUNK
    assert chunk_slots(d04) == chunk_slots(d03)
    # Below the ceiling the workspace shrinks; it never grows.
    assert pf.mynn_pbl_column_chunk(tiny) == 64 * 64
    for name, shape in chunk_slots(tiny).items():
        assert shape <= chunk_slots(d04)[name], name
    # And the whole thing is a bounded, reportable number rather than a
    # fraction of the card.  The d04 total includes the six full-width
    # returned tendency fields, which are the only term that scales.
    assert pf.mynn_pbl_scratch_bytes_for(d04) < 2 * 1024 ** 3


def test_the_predicate_reductions_match_the_spellings_they_replaced():
    """``-ftz=true`` is appended by CuPy unconditionally.

    The validators used to be ``bool(cp.isfinite(a).all())``,
    ``cp.any(a <= 0.0)`` and ``cp.any(a != 0.0)``; they are now
    ``ReductionKernel``s with the same comparisons.  Both go through the
    same NVRTC invocation with the same flag, so a positive subnormal
    flushes to +0 and reads as non-positive in both -- but that is a claim
    about the toolchain, so it is measured here rather than asserted, on
    both signed zeros and on the smallest normal, subnormal and NaN.
    """
    pytest.importorskip("cupy")
    import cupy as cp

    from gpuwm.core.mynn_pbl_gpu import (
        _nonfinite, _nonpositive, _nonzero)

    tiny = np.finfo(np.float32).tiny
    probes = np.array([
        0.0, -0.0, np.nextafter(np.float32(0), np.float32(1)),
        -np.nextafter(np.float32(0), np.float32(1)),
        tiny, -tiny, tiny / 2.0, 1.0, -1.0,
        np.inf, -np.inf, np.nan,
    ], dtype=np.float32)
    for value in probes:
        column = cp.full(8, value, dtype=cp.float32)
        assert (int(_nonpositive()(column)) != 0) \
            == bool(cp.any(column <= 0.0)), value
        assert (int(_nonzero()(column)) != 0) \
            == bool(cp.any(column != 0.0)), value
        assert (int(_nonfinite()(column)) != 0) \
            == (not bool(cp.isfinite(column).all())), value


def test_the_flag_mask_is_a_mask_and_not_an_accumulator():
    """A CuPy reduction writes ``out``; it does not fold what is there.

    ``_flag_mask`` depends on that, and the refusal message in
    ``mynn_tendencies_nomf_cuda`` depends on ``_flag_mask``: it names the
    arrays that are actually nonzero.  An implementation that ORed several
    arrays into one word would report the wrong names, so the per-array
    result is pinned here, including for a group longer than the flag block.
    """
    pytest.importorskip("cupy")
    import cupy as cp

    from gpuwm.core.mynn_pbl_gpu import _flag_mask, _nonzero

    flags = cp.zeros(4, dtype=cp.int32)
    arrays = [cp.zeros(3, dtype=cp.float32) for _ in range(10)]
    for index in (0, 3, 4, 9):
        arrays[index][1] = np.float32(1.0)
    expected = [index in (0, 3, 4, 9) for index in range(10)]
    assert _flag_mask(_nonzero(), arrays, flags) == expected


# ---------------------------------------------------------------------------
# The forecast-neutrality gates.  These need a device.
# ---------------------------------------------------------------------------


def _carried_hash(state, driver, cfg):
    """SHA-256 over every carried array a MYNN restart would have to hold."""
    import cupy as cp

    from gpuwm.core.mynn_pbl_runtime import (
        MYNN_PBL_DIAGNOSTICS_2D, MYNN_PBL_DIAGNOSTICS_INT_2D,
        MYNN_PBL_STATE_3D,
    )

    digest = hashlib.sha256()
    for name in ("u", "v", "w", "thp", "mup", "phi"):
        array = getattr(state, name, None)
        if array is not None:
            digest.update(name.encode())
            digest.update(cp.asnumpy(array).tobytes())
    if getattr(state, "moist", None) is not None:
        for name in sorted(state.moist):
            digest.update(name.encode())
            digest.update(cp.asnumpy(state.moist[name]).tobytes())
    names = (*MYNN_PBL_STATE_3D, *MYNN_PBL_DIAGNOSTICS_2D,
             *MYNN_PBL_DIAGNOSTICS_INT_2D,
             "exch_h", "exch_m", "pblh", "rmol", "kpbl")
    for name in names:
        digest.update(name.encode())
        digest.update(cp.asnumpy(driver.fields[name]).tobytes())
    return digest.hexdigest()


def _forecast(steps=6, chunk=None, poison=None, nx=24, ny=18):
    """Run a real MYNN 5/5 forecast and return the carried-state hash.

    ``poison`` is called with the live workspace before every step, which is
    how the write-before-read classification is put under load: a slot the
    kernel does not fully overwrite shows up as a NaN in the forecast.
    """
    from gpuwm.core.dycore import step
    from gpuwm.core.mynn_pbl_scratch import MynnPblScratch

    from test_mynn_pbl_runtime import _build

    state, cfg, driver = _build(nx=nx, ny=ny)
    if chunk is not None or poison is not None:
        import gpuwm.core.mynn_pbl_runtime as runtime

        original = runtime.mynn_pbl_step

        def patched(atmosphere, fields, **kwargs):
            if chunk is not None:
                kwargs["column_chunk"] = chunk
            if poison is not None:
                nz = atmosphere["theta"].shape[0]
                ncol = (atmosphere["theta"].shape[1]
                        * atmosphere["theta"].shape[2])
                width = min(chunk or 16384, ncol)
                poison(MynnPblScratch.from_state(kwargs["state"], width, nz))
            return original(atmosphere, fields, **kwargs)

        runtime.mynn_pbl_step = patched
        import gpuwm.core.physics as physics_module
        physics_module.mynn_pbl_step = patched
        try:
            for _ in range(steps):
                step(state, cfg)
        finally:
            runtime.mynn_pbl_step = original
            physics_module.mynn_pbl_step = original
    else:
        for _ in range(steps):
            step(state, cfg)
    return _carried_hash(state, driver, cfg)


@requires_gpu
def test_the_column_chunk_is_not_a_seam():
    """Every MYNN kernel owns one column, so the split must be bitwise.

    Measured separately on a synthetic 360,000-column d04 batch, every chunk
    from 2,048 to unchunked produced one identical carried-state hash; this
    is the same property through the real model, where the chunk walk also
    owns the field transposes and the write-back.
    """
    reference = _forecast(chunk=None)
    for chunk in (10_000_000, 512, 137, 1):
        assert _forecast(chunk=chunk) == reference, chunk


@requires_gpu
def test_poisoning_a_constant_zero_slot_is_visible():
    """The failing form, first: the poison lever must be able to see a lie.

    ``mynn_pbl_zero_layer`` and its five siblings are read-before-write by
    design -- WRF passes them to systems this identity switches off and every
    reader requires zero.  If they were classified write-before-read and
    handed to the shared arena, a neighbouring domain's writes would land in
    a tendency solve that is supposed to have no mass flux.  NaN-filling them
    is what that misclassification would look like, and it must not pass
    silently.

    It does not: the driver's own finiteness validator refuses the call.
    That refusal is the visibility, and it is asserted rather than assumed,
    because the alternative -- a NaN that reaches the forecast -- would be
    the same evidence with a worse blast radius.
    """
    from gpuwm.core.mynn_pbl_scratch import MYNN_PBL_CONSTANT_ZERO_SLOTS
    from gpuwm.core.state import DTYPE

    reference = _forecast()

    def poison_the_constants(work):
        for slot in MYNN_PBL_CONSTANT_ZERO_SLOTS:
            work._buffers[slot].fill(DTYPE(np.nan))

    with pytest.raises((ValueError, FloatingPointError)) as excinfo:
        _forecast(poison=poison_the_constants)
    assert "finite" in str(excinfo.value)
    # And the unpoisoned run that produced ``reference`` is reproducible, so
    # the refusal above is attributable to the poison and nothing else.
    assert _forecast() == reference


@requires_gpu
def test_poisoning_the_workspace_leaves_the_forecast_alone():
    """Every slot classified write-before-read really is.

    This is the property that lets the shared arena alias MYNN's slots
    between domains at all.  ``MynnPblScratch.poison`` NaN-fills every slot
    outside the constant-zero set before each step; a slot the owning kernel
    does not fully overwrite would carry the NaN into the forecast, and the
    preceding test shows that such a NaN is not silently absorbed.
    """
    assert _forecast(poison=lambda work: work.poison()) == _forecast()


@requires_gpu
def test_the_registry_prices_exactly_the_slots_the_solver_asks_for():
    """Both directions: no unpriced slot, no stale registry row.

    This is what closes the variable-slot allowlist entry in
    tests/test_preflight.py.  The solver asks for its workspace through two
    loops over shape functions, so the slot names are variables; the check
    that matters is that the set it actually requests at runtime equals the
    set the registry prices, name for name and shape for shape.
    """
    from gpuwm.core import preflight as pf
    from gpuwm.core.dycore import step
    from gpuwm.core.state import DomainState

    from test_mynn_pbl_runtime import _build

    requested: dict[str, tuple[int, ...]] = {}
    original = DomainState.scratch

    def recording(self, shape, slot, dtype=None):
        buffer = original(self, shape, slot, dtype=dtype)
        if slot.startswith("mynn_pbl_"):
            requested[slot] = tuple(buffer.shape)
        return buffer

    DomainState.scratch = recording
    try:
        state, cfg, driver = _build(nx=24, ny=18)
        for _ in range(2):
            step(state, cfg)
    finally:
        DomainState.scratch = original

    priced = {name: tuple(shape)
              for name, shape in pf.mynn_pbl_scratch_slots(cfg).items()}
    assert requested, "no MYNN slot was drawn from DomainState.scratch"
    assert set(requested) == set(priced), (
        f"unpriced: {sorted(set(requested) - set(priced))}; "
        f"stale registry rows: {sorted(set(priced) - set(requested))}")
    assert requested == priced

    # And every one of them is classified in the lifetime audit, which is
    # what decides whether the shared arena may back it.
    for slot in priced:
        assert pf.scratch_slot_lifetime(slot) is not None, slot


@requires_gpu
def test_the_workspace_refuses_a_batch_wider_than_it_declared():
    """A chunk wider than the workspace must be refused, not silently wrong.

    Without this the capacity check would be the difference between a clear
    error and a view that reads past its slot into the next one.  Shown on a
    real leaf rather than on the guard in isolation, so the refusal is proved
    to be reachable from a call anyone could make.
    """
    import cupy as cp

    from gpuwm.core.mynn_pbl_gpu import mynn_pblh_scale_columns_cuda
    from gpuwm.core.mynn_pbl_scratch import MynnPblScratch

    ncol, nz = 8, 30
    thetav = cp.full((ncol, nz), np.float32(300.0), dtype=cp.float32)
    qke = cp.full((ncol, nz), np.float32(0.5), dtype=cp.float32)
    zw = cp.asarray(np.tile(np.arange(nz + 1, dtype=np.float32) * 100.0,
                            (ncol, 1)))
    dz = cp.full((ncol, nz), np.float32(100.0), dtype=cp.float32)
    xland = cp.ones(ncol, dtype=cp.float32)
    dx = cp.full(ncol, np.float32(750.0), dtype=cp.float32)

    wide = MynnPblScratch.standalone(ncol, nz)
    ok = mynn_pblh_scale_columns_cuda(thetav, qke, zw, dz, xland, dx,
                                      scratch=wide)
    assert int(cp.isfinite(ok.zi).all())

    narrow = MynnPblScratch.standalone(4, nz)
    with pytest.raises(ValueError, match="holds 4 columns"):
        mynn_pblh_scale_columns_cuda(thetav, qke, zw, dz, xland, dx,
                                     scratch=narrow)
