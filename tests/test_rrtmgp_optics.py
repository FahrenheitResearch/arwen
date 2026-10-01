"""Bitwise gates for coalesced fills and skipped workspace output."""
import numpy as np
import pytest


def test_clear_upper_fields_matches_separate_copy_fill():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.rrtmgp import _clear_upper_fields, _append_clear_upper_layers
    rng = np.random.default_rng(143)
    for shape in ((1, 4), (971, 59), (7, 49)):
        values = tuple(cp.asarray(rng.standard_normal(shape).astype(np.float32))
                       for _ in range(5))
        values[0][0, 0] = np.float32(-0.0)
        for upper in (0, 1, 17):
            want = tuple(_append_clear_upper_layers(v, upper, xp=cp)
                         for v in values)
            names = tuple(f"test.optics.{i}" for i in range(5))
            for reused in (False, True):
                got = _clear_upper_fields(values, upper, xp=cp,
                                          names=names if reused else None)
                for a, b in zip(want, got):
                    assert bool(cp.all(a.view(cp.uint32) == b.view(cp.uint32)))
                if reused and upper:
                    for v in got:
                        v.fill(np.float32(-7.5))
                    again = _clear_upper_fields(values, upper, xp=cp, names=names)
                    for a, b in zip(want, again):
                        assert bool(cp.all(a.view(cp.uint32) == b.view(cp.uint32)))


@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_skip_col_dry_preserves_gas_outputs_and_reserved_slot(kind):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.rrtmgp import _gas_optics, gas_optics, load_gas_tables
    from tests.test_rrtmgp import _rfmip_columns
    tables = load_gas_tables(kind)
    inputs = tuple(cp.asarray(v, dtype=cp.float32) for v in
                   _rfmip_columns(tables, sites=(4, 31, 88)))
    want = gas_optics(tables, *inputs)
    reserved = cp.full(inputs[0].shape, np.float32(-7.5))
    got = _gas_optics(tables, *inputs, metadata=None, validate=True,
                      zero_g_sentinel=False, col_dry_out=reserved,
                      compute_col_dry=False)
    assert got.col_dry is None
    assert want.col_dry is not None
    assert bool(cp.all(reserved == np.float32(-7.5)))
    for field in (("tau",) if kind == "lw" else ("tau", "ssa", "g")):
        assert bool(cp.all(getattr(want, field).view(cp.uint32) ==
                           getattr(got, field).view(cp.uint32)))


@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_gas_optimized_matches_frozen_base_bits(kind, monkeypatch):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import kernels
    from gpuwm.core.rrtmgp import gas_optics, load_gas_tables
    from tests.test_rrtmgp import _rfmip_columns
    reference_kernel = kernels.get_kernel("rrtmgp_gas", "rrtmgp_gas_optics_reference")
    original = kernels.get_kernel
    tables = load_gas_tables(kind)
    inputs = tuple(cp.asarray(v, dtype=cp.float32) for v in
                   _rfmip_columns(tables, sites=tuple(range(100))))
    want = gas_optics(tables, *inputs)
    def reference_launch(grid, block, args, **kw):
        reference_kernel((int(args[-14]) * int(args[-13]),), (64,), args,
                         shared_mem=4 * max(int(args[-2]), int(args[-1])))
    monkeypatch.setattr(kernels, "get_kernel", lambda module, entry:
                        reference_launch if entry == "rrtmgp_gas_optics" else
                        original(module, entry))
    got = gas_optics(tables, *inputs)
    for field in (("tau",) if kind == "lw" else ("tau", "ssa", "g")):
        assert bool(cp.all(getattr(want, field).view(cp.uint32) ==
                           getattr(got, field).view(cp.uint32)))


def test_reserved_col_dry_slot_keeps_workspace_layout_and_audit():
    import inspect
    from gpuwm.core import preflight
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace
    from gpuwm.core.rrtmgp import RRTMGP_WORKSPACE_LIFETIME_AUDIT, RRTMGPRadiation
    layouts = preflight.rrtmgp_workspace_phases(3, 2)
    workspace = SharedRRTMGPChunkWorkspace(
        nz=3, column_chunk=2, _array_module=np,
        _phase_layouts_input=layouts)
    for phase, items in layouts.items():
        views = workspace.phase(phase, 2)
        assert set(views) == set(items) == set(RRTMGP_WORKSPACE_LIFETIME_AUDIT[phase])
        assert all(np.shares_memory(v, workspace.storage) for v in views.values())
    for kind in ("lw", "sw"):
        assert "col_dry" in layouts[f"{kind}_optics"]
        assert "col_dry" not in layouts[f"{kind}_rte"]
    source = inspect.getsource(RRTMGPRadiation.__call__)
    assert source.count("compute_col_dry=False") == 2
    assert 'work["col_dry"]' not in source


def test_gas_reference_kernel_retains_base_source():
    import hashlib
    from pathlib import Path
    source = (Path(__file__).parents[1] / "gpuwm/core/kernels/rrtmgp_gas.cu").read_text()
    begin = source.index('extern "C" __global__ void rrtmgp_gas_optics_reference(')
    end = source.index('__device__ __forceinline__ float planck_interp', begin)
    frozen = source[begin:end].replace("rrtmgp_gas_optics_reference(",
                                      "rrtmgp_gas_optics(")
    assert hashlib.sha256(frozen.encode()).hexdigest() == "2971c4dc38ac19592b469489094f14c7d8f5a9f86059fd7489b50e2272f915d1"


def test_profile_concatenation_matches_retained_expressions():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.rrtmgp import _append_profile_fields
    rng = np.random.default_rng(379)
    for ncol, nlay, upper_nlay in ((1, 4, 1), (971, 59, 17), (7, 49, 4)):
        model = tuple(cp.asarray(rng.normal(size=(ncol, nlay + int(i in (1,3))))
                                 .astype(np.float32)) for i in range(5))
        model[0][0, 0] = np.float32(-0.0)
        for dtype in (cp.float32, cp.float64):
            upper = tuple(cp.broadcast_to(cp.asarray(
                rng.normal(size=(1, upper_nlay)),
                dtype=dtype if i in (2,3) else cp.float32), (ncol, upper_nlay))
                for i in range(5))
            got = _append_profile_fields(model, upper, xp=cp)
            for a, b, out in zip(model, upper, got):
                want = cp.ascontiguousarray(cp.concatenate((a,b), axis=1))
                assert want.dtype == out.dtype
                word = cp.uint64 if out.dtype == cp.float64 else cp.uint32
                assert bool(cp.all(want.view(word) == out.view(word)))


@pytest.mark.parametrize("mu_dtype", [np.float32, np.float64])
def test_sw_broadcasts_match_retained_expressions(mu_dtype):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.rrtmgp import _prepare_sw_inputs
    for ncol, ngpt, nlay in ((1, 9, 4), (971, 224, 60), (7, 3, 49)):
        albedo = cp.arange(ncol, dtype=cp.float32)
        solar = cp.arange(ngpt, dtype=cp.float32)
        mu = cp.arange(ncol, dtype=mu_dtype)
        if mu_dtype == np.float64:
            mu = mu + np.float64(2.0 ** -25)
        mu_reference = cp.asarray(mu, dtype=cp.float32)
        albedo[0] = np.float32(-0.0)
        want = tuple(cp.ascontiguousarray(v) for v in (
            cp.broadcast_to(albedo[:,None], (ncol,ngpt)),
            cp.broadcast_to(solar[None,:], (ncol,ngpt)),
            cp.broadcast_to(mu_reference[:,None], (ncol,nlay))))
        dirty = tuple(cp.full(v.shape, np.float32(-7.5)) for v in want)
        for out in (None, dirty):
            got = _prepare_sw_inputs(albedo, solar, mu, nlay, out=out, xp=cp)
            for a,b in zip(want,got):
                assert bool(cp.all(a.view(cp.uint32) == b.view(cp.uint32)))


def test_flux_pair_copy_and_daylight_merge_match_retained_expressions():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.rrtmgp import _store_model_flux_pair, _model_flux_interfaces
    rng = np.random.default_rng(724)
    for ncol, model_nlay, upper in ((1, 4, 1), (971, 59, 17), (7, 49, 0)):
        sources = tuple(cp.asarray(rng.normal(size=(ncol, model_nlay+upper+1))
                                   .astype(np.float32)) for _ in range(2))
        sources[0][0, 0] = np.float32(-0.0)
        sources[1][0, 0] = np.float32(np.nan)
        for daylight in (None, cp.arange(ncol) % 2 == 0, cp.zeros(ncol, cp.bool_)):
            outputs = tuple(cp.full((ncol,model_nlay+1), np.float32(-7.5))
                            for _ in range(2))
            _store_model_flux_pair(*sources, model_nlay, *outputs,
                                   daylight=daylight, xp=cp)
            for source, out in zip(sources, outputs):
                want = _model_flux_interfaces(source, model_nlay, xp=cp)
                if daylight is not None:
                    want = cp.where(daylight[:,None], want, np.float32(0.0))
                assert bool(cp.all(want.view(cp.uint32) == out.view(cp.uint32)))


def test_seeded_radiation_gas_arrays_match_frozen_base(monkeypatch):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import kernels
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace
    from tilestream import harness, physics_inventory
    from tilestream.rrtmgp_bench import PHYSICS_FULL
    from tilestream.rrtmgp_lazy import attach_lazy
    cfg = harness.make_config(64, 64, 59, **dict(PHYSICS_FULL,
        ra_sw_physics=4, ra_lw_physics=4, ra_rrtmg_variant="rte-rrtmgp"))
    state, driver = physics_inventory.default_builder(cfg)
    workspace = SharedRRTMGPChunkWorkspace(
        nz=59, column_chunk=3125, p_top=float(state.p_top))
    attach_lazy(state, workspace)
    original = kernels.get_kernel
    reference = original("rrtmgp_gas", "rrtmgp_gas_optics_reference")
    checked = []
    def lookup(module, entry):
        fun = original(module, entry)
        if entry != "rrtmgp_gas_optics": return fun
        def launch(grid, block, args, **kw):
            fun(grid, block, args, **kw)
            ref_args = list(args)
            tau = cp.empty_like(args[-16])
            ssa = cp.empty_like(args[-15]) if int(args[-3]) else tau
            ref_args[-16], ref_args[-15] = tau, ssa
            reference((int(args[-14]) * int(args[-13]),), (64,), tuple(ref_args),
                      shared_mem=4 * max(int(args[-2]), int(args[-1])))
            assert bool(cp.all(tau.view(cp.uint32) == args[-16].view(cp.uint32)))
            if int(args[-3]):
                assert bool(cp.all(ssa.view(cp.uint32) == args[-15].view(cp.uint32)))
            checked.append((int(args[-3]), int(args[-14])))
        return launch
    monkeypatch.setattr(kernels, "get_kernel", lookup)
    harness.run_steps(state, cfg, 2)
    assert {band for band, _ in checked} == {0, 1}
    assert {ncol for _, ncol in checked} == {3125, 971}


def test_lw_cap_temperature_fusion_matches_float64_cupy_expressions():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.rrtmgp import _append_profile_fields
    rng = np.random.default_rng(1231)
    for ncol, nlay, upper_nlay in ((1, 4, 1), (971, 59, 17), (7, 49, 4)):
        model = tuple(cp.asarray(rng.uniform(150, 310,
                    size=(ncol, nlay + int(i in (1,3)))).astype(np.float32))
                      for i in range(5))
        for uniform in (False, True):
            nrow = 1 if uniform else ncol
            climo_top = cp.asarray(rng.uniform(190, 280, (nrow,1)), dtype=cp.float64)
            climo_upper = cp.asarray(rng.uniform(170, 270, (nrow,upper_nlay)),
                                     dtype=cp.float64)
            up_play = cp.broadcast_to(cp.arange(upper_nlay, dtype=cp.float32)[None,:],
                                      (ncol,upper_nlay))
            up_plev = up_play
            up_qv = cp.broadcast_to(model[4][:,-1:], (ncol,upper_nlay))
            top = model[3][:,-1:]
            upper_tlev = climo_upper + (top - climo_top)
            all_upper_tlev = cp.concatenate((top, upper_tlev), axis=1)
            upper_tlay = np.float32(0.5) * (all_upper_tlev[:,:-1] + all_upper_tlev[:,1:])
            want = tuple(cp.ascontiguousarray(cp.concatenate((a,b),axis=1))
                for a,b in zip(model, (up_play,up_plev,upper_tlay,upper_tlev,up_qv)))
            got = _append_profile_fields(model, (up_play,up_plev,
                cp.broadcast_to(climo_top,(ncol,1)),
                cp.broadcast_to(climo_upper,(ncol,upper_nlay)),up_qv), xp=cp,
                derive_lw_temperature=True)
            for a,b in zip(want,got):
                assert a.dtype == b.dtype
                word = cp.uint64 if a.dtype == cp.float64 else cp.uint32
                assert bool(cp.all(a.view(word) == b.view(word)))


def test_shared_solver_temperature_casts_match_cupy_and_keep_profile():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.rrtmgp import _solver_temperature_profile, _RadiationColumnProfile
    rng = np.random.default_rng(1597)
    for ncol, nlay in ((1, 4), (971, 76), (7, 49)):
        tlay = cp.asarray(rng.uniform(150,310,(ncol,nlay)), dtype=cp.float64)
        tlev = cp.asarray(rng.uniform(150,310,(ncol,nlay+1)), dtype=cp.float64)
        tlay[0,0] = np.float64(-0.0)
        original = _RadiationColumnProfile(None,None,tlay,tlev,None,nlay,0)
        got = _solver_temperature_profile(original,xp=cp)
        assert original.tlay is tlay and original.tlev is tlev
        for a,b in ((tlay,got.tlay),(tlev,got.tlev)):
            want = cp.asarray(a,dtype=cp.float32)
            assert bool(cp.all(want.view(cp.uint32) == b.view(cp.uint32)))
