"""Live host stores supply bounded nest operands without resident shadows."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.nest_interp import (
    bdy_interp1, copy_fcn, feedback_child_window, feedback_parent_bounds,
    register_nest, window_registration)
from gpuwm.core.nest_operands import NestWindowSource, boundary_windows


def _host_facade(host):
    prognostics = {"mup", "u", "v", "w", "thp", "php", "qv", "qc", "qr",
                   "qi", "qs", "qg", "nr", "ni", "ns", "ng", "p", "al", "alt"}
    store, geography, metadata = {}, {}, {}
    for name, value in vars(host).items():
        if name in prognostics:
            store[f"state/{name}"] = value.copy()
        elif isinstance(value, np.ndarray) and value.ndim >= 2:
            geography[f"setup/{name}"] = value.copy()
        else:
            metadata[name] = value
    owner = SimpleNamespace(store=store, _geography=geography,
                            template_state=SimpleNamespace(**metadata),
                            decision=SimpleNamespace(tile_ny=7, tile_nx=8))
    facade = SimpleNamespace(_streamed_domain=owner, **metadata)
    for key, value in store.items():
        setattr(facade, key.split("/", 1)[1], value)
    return facade


def test_canonical_source_rejects_horizontal_template_fallback():
    template = SimpleNamespace(mup=np.zeros((2, 3)), c1h=np.ones(4))
    owner = SimpleNamespace(store={}, _geography={}, template_state=template)
    source = NestWindowSource(SimpleNamespace(_streamed_domain=owner))
    assert source.array("c1h") is template.c1h
    with pytest.raises(RuntimeError, match="slab template"):
        source.array("mup")


def test_canonical_source_keeps_numpy_setup_scalars_on_host():
    top = np.float32(5717.7817)
    source = NestWindowSource(SimpleNamespace(p_top=top, has_msf=np.bool_(True)))
    assert source.device("p_top") is top
    assert source.device("has_msf") == np.bool_(True)


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stagger", ["", "x", "y"])
def test_boundary_chunks_cover_each_global_table_once(ratio, stagger):
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=7,
                        j_parent_start=8, child_nx=30, child_ny=33,
                        parent_nx=64, parent_ny=64, stagger=stagger, wrapper="bdy")
    seen = {side: np.zeros((reg.nyc, 5) if side in ("west", "east")
                           else (5, reg.nxc), np.int8)
            for side in ("west", "east", "south", "north")}
    for side, win, dest in boundary_windows(reg, 5, (7, 8)):
        seen[side][dest[1:]] += 1
        cropped, donor = window_registration(reg, win)
        assert (cropped.nyc, cropped.nxc) == (win[0].stop-win[0].start,
                                              win[1].stop-win[1].start)
    for count in seen.values():
        assert np.all(count == 1)


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stagger", ["", "x", "y"])
def test_feedback_donors_fit_one_ratio_footprint_per_parent_cell(ratio, stagger):
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=7,
                        j_parent_start=8, child_nx=60, child_ny=60,
                        parent_nx=80, parent_ny=80, stagger=stagger)
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg)
    for j, i in ((jlo, ilo), (jhi, ihi)):
        donor = feedback_child_window(reg, (slice(j, j+1), slice(i, i+1)))
        assert donor[0].stop-donor[0].start == (1 if stagger == "y" else ratio)
        assert donor[1].stop-donor[1].start == (1 if stagger == "x" else ratio)


@pytest.mark.gpu
def test_gpu_canonical_coupling_windows_match_full_fields_bitwise():
    import cupy as cp
    from test_nest_coupler import _State, _DeviceState, _run
    from gpuwm.ingest.lateral_bc import couple_nest_field

    host = _State(_run(31, 29, nested=True, grid_id=2))
    rng = np.random.default_rng(212)
    for name, value in vars(host).items():
        if isinstance(value, np.ndarray):
            value += rng.uniform(-.05, .05, value.shape).astype(np.float32)
    host.thb = np.broadcast_to(host.thb[:, None, None], (2, 29, 31)).copy()
    host.thb += rng.normal(size=host.thb.shape).astype(np.float32)
    resident = _DeviceState(host, cp)
    source = NestWindowSource(_host_facade(host))
    for kind in ("mu", "u", "v", "w", "t", "ph", "qv", "qc", "qr", "nr"):
        attr = {"mu": "mup", "t": "thp", "ph": "php"}.get(kind, kind)
        field = resident.mup[None] if kind == "mu" else getattr(resident, attr)
        expected = cp.empty(field.shape, dtype=cp.float32)
        couple_nest_field(resident, kind, out=expected)
        ny, nx = field.shape[-2:]
        for win in ((slice(0, 5), slice(0, 7)),
                    (slice(4, 11), slice(8, 13)),
                    (slice(ny-5, ny), slice(nx-7, nx))):
            got = source.coupled(kind, win)
            np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),
                cp.asnumpy(expected[(...,)+win]).view(np.uint32))
    assert source.host_to_device_bytes > 0
    assert source.max_operand_bytes < resident.w.nbytes


@pytest.mark.gpu
@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stagger", ["", "x", "y"])
def test_gpu_window_forcing_and_restriction_match_full_bitwise(ratio, stagger):
    import cupy as cp
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=7,
                        j_parent_start=8, child_nx=30, child_ny=33,
                        parent_nx=64, parent_ny=64, stagger=stagger, wrapper="bdy")
    rng = np.random.default_rng(901)
    parent = cp.asarray(rng.normal(size=(3, reg.nyp, reg.nxp)).astype(np.float32))
    child = cp.asarray(rng.normal(size=(3, reg.nyc, reg.nxc)).astype(np.float32))
    expected = bdy_interp1(parent, child, reg, parent_dt_fp32=np.float32(17.25))
    actual = {s: tuple(cp.empty_like(v) for v in pair) for s, pair in expected.items()}
    for side, win, dest in boundary_windows(reg, 5, (7, 8)):
        cropped, donor = window_registration(reg, win)
        result = bdy_interp1(cp.ascontiguousarray(parent[(...,)+donor]),
            cp.ascontiguousarray(child[(...,)+win]), cropped,
            parent_dt_fp32=np.float32(17.25), sides=(side,))
        for got, value in zip(actual[side], result[side]):
            got[dest] = value
    for side in actual:
        for got, want in zip(actual[side], expected[side]):
            np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),
                                          cp.asnumpy(want).view(np.uint32))
    want = parent.copy()
    copy_fcn(want, child, reg)
    got = parent.copy()
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg)
    for j in range(jlo, jhi+1, 3):
        for i in range(ilo, ihi+1, 2):
            win = (slice(j, min(j+3, jhi+1)), slice(i, min(i+2, ihi+1)))
            donor = feedback_child_window(reg, win)
            target = cp.empty((3, win[0].stop-j, win[1].stop-i), dtype=cp.float32)
            copy_fcn(target, cp.ascontiguousarray(child[(...,)+donor]), reg,
                     parent_window=win, child_window=donor)
            got[(...,)+win] = target
    np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),
                                  cp.asnumpy(want).view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("host_parent", [False, True])
def test_gpu_force_entrypoint_uses_host_child_without_full_field_scratch(host_parent):
    import cupy as cp
    from test_nest_coupler import _DeviceState, _nodes
    from gpuwm.core.nest import NestCoupler

    parent, child = _nodes()
    parent.state = _DeviceState(parent.state, cp)
    child.state = _DeviceState(child.state, cp)
    control = NestCoupler(child)
    control.force(child)
    expected = {kind: {side: tuple(cp.asnumpy(v) for v in pair)
                       for side, pair in sides.items()}
                for kind, sides in control._last_tables.items()}

    parent, child = _nodes()
    parent.state = _host_facade(parent.state) if host_parent else _DeviceState(parent.state, cp)
    child.state = _host_facade(child.state)
    slots = {}
    def scratch(shape, slot, dtype=None):
        assert slot not in ("nest_parent_field", "nest_child_field")
        if slot not in slots:
            slots[slot] = cp.empty(shape, dtype=dtype or np.float32)
        return slots[slot]
    child.state.scratch = scratch
    bounded = NestCoupler(child)
    bounded.force(child)
    for kind, sides in bounded._last_tables.items():
        for side, pair in sides.items():
            for got, want in zip(pair, expected[kind][side]):
                np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),
                                              want.view(np.uint32))
    assert bounded.force_sync_bytes > 0
    assert bounded.force_count == 1 and bounded.valid
    assert child.state._lateral_boundary_device.rolling_generation == 1


@pytest.mark.gpu
@pytest.mark.parametrize("smooth_option", [0, 1, 2])
@pytest.mark.parametrize("host_parent", [False, True])
def test_gpu_feedback_entrypoint_reads_host_child_and_matches_resident(smooth_option, host_parent):
    import cupy as cp
    from test_nest_coupler import _DeviceState, _nodes
    from gpuwm.core.model import FeedbackScratch
    from gpuwm.core.nest import NestCoupler

    results = []
    for bounded in (False, True):
        parent, child = _nodes()
        rng = np.random.default_rng(703)
        for state in (parent.state, child.state):
            for name, value in vars(state).items():
                if isinstance(value, np.ndarray):
                    value += rng.normal(size=value.shape).astype(np.float32)
            ny, nx = state.mup.shape
            state.thb = np.broadcast_to(state.thb[:, None, None], (2, ny, nx)).copy()
            state.thb += rng.normal(size=state.thb.shape).astype(np.float32)
        parent.state = (_host_facade(parent.state) if bounded and host_parent
                        else _DeviceState(parent.state, cp))
        if bounded:
            child.state = _host_facade(child.state)
            slots = {}
            def scratch(shape, slot, dtype=None):
                assert slot != "nest_child_field"
                if host_parent:
                    assert slot != "nest_parent_field"
                if slot not in slots:
                    slots[slot] = cp.empty(shape, dtype=dtype or np.float32)
                return slots[slot]
            child.state.scratch = scratch
        else:
            child.state = _DeviceState(child.state, cp)
        child.clock.ticks = parent.clock.ticks
        coupler = NestCoupler(child, feedback=1, smooth_option=smooth_option)
        coupler.feedback_prepare(child, FeedbackScratch())
        coupler.feedback_commit(child)
        if bounded and host_parent and smooth_option:
            assert coupler.feedback_host_scratch_bytes > 0
        results.append({name: cp.asnumpy(getattr(parent.state, name))
                        for name in ("mup", "u", "v", "w", "thp", "php")})
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name].view(np.uint32),
                                      results[1][name].view(np.uint32), err_msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("host_parent", [False, True])
def test_gpu_cam_ozone_transfer_writes_only_canonical_child_chunks(host_parent):
    import cupy as cp
    from gpuwm.core.cam_ozone import CARRIER_KEY, transfer_parent_ozone
    from gpuwm.core.nest_interp import sint
    from test_nest_coupler import _State, _run

    reg = register_nest(nri=3, nrj=3, i_parent_start=7, j_parent_start=8,
                        child_nx=30, child_ny=33, parent_nx=64, parent_ny=64)
    parent = _State(_run(64, 64, nested=False, grid_id=1))
    child = _host_facade(_State(_run(30, 33, nested=True, grid_id=2)))
    ozone = np.random.default_rng(317).random((2, 64, 64), dtype=np.float32)
    expected = cp.asnumpy(sint(cp.asarray(ozone), reg))
    if host_parent:
        parent = _host_facade(parent)
        parent._streamed_domain.store[CARRIER_KEY] = ozone
    parent.physics = SimpleNamespace(o3rad=cp.full((2, 2, 2), -99., np.float32)
                                    if host_parent else cp.asarray(ozone),
                                    call_counts={"cam_ozone": 4})
    canonical = np.full((2, 33, 30), -77., np.float32)
    child._streamed_domain.store[CARRIER_KEY] = canonical
    child._streamed_domain.scalars = {"call_counts": {"cam_ozone": 0}}
    child.physics = SimpleNamespace(
        cam_ozone=SimpleNamespace(mode="parent-interpolated"),
        o3rad=cp.full((2, 2, 2), -88., np.float32), call_counts={"cam_ozone": 0})
    node = SimpleNamespace(state=child, parent=SimpleNamespace(state=parent))
    moved = transfer_parent_ozone(node, reg)
    np.testing.assert_array_equal(canonical.view(np.uint32), expected.view(np.uint32))
    assert child._streamed_domain.scalars["call_counts"]["cam_ozone"] == 1
    assert moved >= canonical.nbytes
    assert bool(cp.all(child.physics.o3rad == -88.))


@pytest.mark.gpu
@pytest.mark.parametrize("base3d", [False, True])
@pytest.mark.parametrize("hypsometric_opt", [1, 2])
@pytest.mark.parametrize("smooth_option", [0, 2])
def test_gpu_feedback_finalize_uses_canonical_host_columns_exactly(
        base3d, hypsometric_opt, smooth_option):
    import cupy as cp
    from dataclasses import replace
    from gpuwm.core.nest import NestCoupler
    from test_nest_coupler import _DeviceState, _nodes

    results = []
    for bounded in (False, True):
        parent, child = _nodes()
        parent.cfg.run = replace(parent.cfg.run, hypsometric_opt=hypsometric_opt)
        state = parent.state
        state.mub2d.fill(85000.)
        state.phb = np.array([0., 5000., 15000.], np.float32)
        state.dphb_resid = np.zeros(2, np.float32)
        state.alb = np.ones(2, np.float32)
        state.rdnw = np.full(2, -2., np.float32)
        state.c3h = np.array([.75, .25], np.float32)
        state.c3f = np.array([1., .5, 0.], np.float32)
        state.c4h = state.dc4f = np.zeros(2, np.float32)
        state.c4f = np.zeros(3, np.float32)
        state.dc3f = np.full(2, .5, np.float32)
        state.p_top = 10000.
        for name in ("p", "al", "alt"):
            setattr(state, name, np.full(state.thp.shape, -99., np.float32))
        rng = np.random.default_rng(413)
        if base3d:
            for name in ("thb", "phb", "dphb_resid", "alb"):
                profile = getattr(state, name)
                field = np.broadcast_to(profile[:, None, None],
                                        (len(profile), *state.mup.shape)).copy()
                field += rng.uniform(-.01, .01, field.shape).astype(np.float32)
                setattr(state, name, field)
        if bounded:
            parent.state = _host_facade(state)
            child.state = _host_facade(child.state)
        else:
            parent.state = _DeviceState(state, cp)
            child.state = _DeviceState(child.state, cp)
        coupler = NestCoupler(child, feedback=1, smooth_option=smooth_option)
        coupler._prepared_feedback = {}
        coupler.feedback_finalize(child)
        results.append({name: cp.asnumpy(getattr(parent.state, name))
                        for name in ("p", "al", "alt")})
        assert coupler._prepared_feedback is None
    for name in results[0]:
        assert np.isfinite(results[0][name]).all()
        np.testing.assert_array_equal(results[0][name].view(np.uint32),
                                      results[1][name].view(np.uint32), err_msg=name)
