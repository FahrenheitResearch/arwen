"""Tiny CUDA parity probes; no forecast or tiled physics execution."""
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.core import streaming, storm_tracking as st
from gpuwm.core.attribute_tracking import ATTRIBUTE_UNITS, attribute_plane
from gpuwm.core.streamed_state import CanonicalStoreState
from test_attribute_tracking import config, footprint


@requires_gpu
@pytest.mark.parametrize("attribute", list(ATTRIBUTE_UNITS))
@pytest.mark.parametrize("reduction", ["column_max", "column_min", "column_mean", "model_level"])
@pytest.mark.parametrize("extremum", ["max", "min"])
def test_actual_cupy_resident_device_host_and_canonical_parity(attribute, reduction, extremum):
    import cupy as cp
    # requires_gpu checks the CUDA prerequisite. These bounded arrays and
    # host/device parity assertions have no dependency on a GPU model name.
    j, i = np.mgrid[:40, :50]
    bump = np.exp(-((j-19.)**2+(i-25.)**2)/8.)
    sign = 1. if extremum == "max" else -1.
    base = .01 if attribute in ("qv", "qc", "qr") else 0.
    amplitude = .004 if base else 4.
    nz = 4 if attribute == "w" else 3
    volume = np.stack([base + sign * amplitude * bump * (1.+k/10.) for k in range(nz)]).astype(np.float32)
    carrier = "thp" if attribute == "theta" else attribute
    setup = np.full((3, 40, 50), 300., np.float32)
    threshold = (300. if attribute == "theta" else base) + sign * amplitude / 2.
    cfg = config(attribute=attribute, extremum=extremum, threshold=threshold,
        reduction=reduction, model_level=1 if reduction == "model_level" else None,
        refine_grid_id=3)
    host_reference = SimpleNamespace(**{carrier: volume.copy()}, thb=setup.copy(), nz=3, ny=40, nx=50)
    expected = attribute_plane(host_reference, cfg)
    reference_source = st.RefinementSource(grid_id=3, state=host_reference,
        origin_i=0., origin_j=0., scale_i=1., scale_j=1., edge_margin_cells=2, dx_m=1000.)
    expected_fix = st.StormTracker(cfg).locate(host_reference, footprint(), 0.,
        refinement=reference_source)
    originals = {"carrier": volume.copy(), "base": setup.copy()}
    # Resident CuPy is the direct scientific operator on device arrays.
    resident = SimpleNamespace(**{carrier: cp.asarray(volume)}, thb=cp.asarray(setup), nz=3, ny=40, nx=50)
    cp.cuda.get_current_stream().synchronize()
    modes = {"resident": resident}
    drains = []
    for mode in ("device", "host"):
        stream = cp.cuda.Stream(non_blocking=True)
        if mode == "device":
            live = cp.empty_like(getattr(resident, carrier))
        else:
            memory = cp.cuda.alloc_pinned_memory(volume.nbytes)
            live = np.frombuffer(memory, dtype=volume.dtype, count=volume.size).reshape(volume.shape)
        # Actual asynchronous CUDA work must be landed through the reader's
        # drain before it sees the store. The endpoint is the production class;
        # this probe supplies only its transport, without executing any physics.
        with stream:
            if mode == "device":
                live[...] = getattr(resident, carrier)
            else:
                getattr(resident, carrier).get(out=live, stream=stream, blocking=False)
        def drain(stream=stream, mode=mode):
            drains.append(mode)
            stream.synchronize()
        store = {"state/" + carrier: live}
        mirror = SimpleNamespace(**{carrier: cp.full(volume.shape, 9999., dtype=cp.float32)},
            thb=cp.full(setup.shape, -9999., dtype=cp.float32), nz=3, ny=40, nx=50)
        endpoint = streaming.StreamedDomain(SimpleNamespace(store=store, drain=drain),
            decision=None, state=mirror,
            geography={"setup/thb": setup if mode == "host" else resident.thb})
        streaming.publish_store(mirror, endpoint)
        modes[mode] = mirror
    # Production canonical facade maps full host arrays from a poisoned slab.
    slab = np.full((3, 2, 50), 9999., np.float32)
    if attribute == "w":
        slab = np.full((4, 2, 50), 9999., np.float32)
    template = SimpleNamespace(**{carrier: slab}, thb=np.full((3, 2, 50), -9999., np.float32),
        total_theta=lambda: pytest.fail("canonical state used a resident method"))
    modes["canonical"] = CanonicalStoreState(template, SimpleNamespace(nx=50, ny=40, nz=3),
        store={"state/"+carrier: volume.copy()}, geography={"setup/thb": setup.copy()}, scalars={},
        inventory={"state/"+carrier: slab}, geography_inventory={"setup/thb": template.thb})
    for mode, live_state in modes.items():
        np.testing.assert_array_equal(attribute_plane(live_state, cfg), expected, err_msg=mode)
        tracker = st.StormTracker(cfg)
        source = st.RefinementSource(grid_id=3, state=live_state, origin_i=0., origin_j=0.,
            scale_i=1., scale_j=1., edge_margin_cells=2, dx_m=1000.)
        fix = tracker.locate(live_state, footprint(), 0., refinement=source)
        assert fix.center_parent_ij == expected_fix.center_parent_ij
        assert fix.extremum == expected_fix.extremum
        assert fix.evidence["refinement"]["refine_extremum"] == expected_fix.extremum
        assert fix.evidence["extremum_units"] == ATTRIBUTE_UNITS[attribute]
        assert fix.evidence["refinement"]["applied"]
        window = attribute_plane(live_state, cfg, window=(10, 30, 15, 35))
        np.testing.assert_array_equal(window[10:30, 15:35], expected[10:30, 15:35])
    assert set(drains) == {"device", "host"}
    np.testing.assert_array_equal(cp.asnumpy(getattr(resident, carrier)), originals["carrier"])
    np.testing.assert_array_equal(cp.asnumpy(resident.thb), originals["base"])
    for mode in ("device", "host"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(modes[mode], carrier)),
            np.full(volume.shape, 9999., dtype=np.float32))
        stored = streaming.domain_store(modes[mode])["state/"+carrier]
        np.testing.assert_array_equal(cp.asnumpy(stored), originals["carrier"])
    assert cp.get_default_memory_pool().total_bytes() < 1024**3


@requires_gpu
def test_native_float32_promotion_preserves_subnormals_and_nonfinite_classification():
    import cupy as cp
    from gpuwm.core.attribute_tracking import _float64_layer
    rng = np.random.default_rng(20260906)
    bits = rng.integers(0, 2**32, size=4096, dtype=np.uint32)
    bits[:12] = [0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x807fffff,
                 0x800000, 0x80800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000]
    values = bits.view(np.float32)
    with np.errstate(invalid="ignore"):
        expected = values.astype(np.float64)
    actual = _float64_layer(cp.asarray(values), cp).get()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@requires_gpu
def test_cuda_probe_memory_receipt():
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    print({"probe": "native-attribute-follow-parity", "device": props["name"].decode(),
        "device_total_bytes": props["totalGlobalMem"],
        "pool_held_bytes": cp.get_default_memory_pool().total_bytes(),
        "pool_live_bytes": cp.get_default_memory_pool().used_bytes(),
        "forecast_steps": 0})
