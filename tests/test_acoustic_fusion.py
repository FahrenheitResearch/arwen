"""The joined acoustic substep launches (gpuwm/core/acoustic_fusion.py).

The breakage these prevent: a joined launch that writes one word the
separate kernels would not -- a value forwarded across a former kernel
boundary into a new multiply-add, a frame w copy read before its source
column is solved, a sumflux element added twice or not at all -- would move
every forecast while each kernel's own tests still pass.  The CPU half pins
the generated text to the original bodies; the GPU half runs both paths on
the same inputs and compares every word.
"""
import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu

OPTIONS = ("-std=c++17", "-arch=compute_89")


def _source():
    from gpuwm.core.kernels import module_source
    return module_source("acoustic")


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("frame", [False, True])
@pytest.mark.parametrize("sumflux", [False, True])
def test_joined_unit_keeps_original_bodies(mapped, frame, sumflux):
    from gpuwm.core import acoustic_fusion as fusion

    source = _source()
    unit, uv_names, column_names, receipt = fusion.joined_unit(
        source, mapped, frame, sumflux, OPTIONS)
    # Every separate entry is compiled out; only the two joined ones remain.
    assert unit.count("#if 0  /* separate kernel") == source.count('extern "C" __global__') - 1
    assert receipt["column"]["entries"][:2] == (
        ["advance_mu_th_msf", "advance_w_phi_msf"] if mapped
        else ["advance_mu_th", "advance_w_phi"])
    assert ("frame_spec_zone" in column_names) == frame
    assert ("sumflux_first" in column_names) == sumflux
    assert "emdiv_spec_zone" in uv_names and "spec_zone" in uv_names
    # Each phase is separated by the compiler memory barrier.
    phases = 2 + int(frame) * 2 + int(sumflux)
    column = unit[unit.index("void " + fusion.COLUMN_ENTRY):]
    assert column.count('asm volatile("" ::: "memory");') == phases - 1
    # The frame copy left the frame body and moved to its source column.
    if frame:
        assert "source_c]" not in column.split("specified-frame zero-gradient w")[0].split(
            "phase: advance_specified_phi_w")[1]


def test_joined_unit_refuses_a_changed_frame_copy():
    from gpuwm.core import acoustic_fusion as fusion

    source = _source().replace("w_pp[tid] = w_pp[(size_t)k * st + source_c];",
                               "w_pp[tid] = w_pp[(size_t)k * st + source_c] * 1.0f;")
    with pytest.raises(fusion.FusionUnsupported):
        fusion.joined_unit(source, True, True, True, OPTIONS)


def test_bind_refuses_two_values_for_one_name():
    from gpuwm.core import acoustic_fusion as fusion

    with pytest.raises(fusion.FusionUnsupported):
        fusion.bind(("nz",), (("nz",), (np.int32(3),), {}), (("nz",), (np.int32(4),), {}))
    assert fusion.bind(("nz",), (("nz",), (np.int32(3),), {}),
                       (("nz",), (np.int32(3),), {})) == (np.int32(3),)


def _state(seed, *, msf, hill, specified, moist):
    from gpuwm.verify.npref import random_acoustic_state
    state, cfg = random_acoustic_state(seed=seed, nz=10, ny=9, nx=13,
                                       msf_amp=0.1 if msf else 0.0,
                                       hill_height=300.0 if hill else 0.0,
                                       moist=moist, mp_physics=8 if moist else 0)
    if moist:
        rng = np.random.default_rng(seed + 1)
        import cupy as cp
        for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
            array = getattr(state, name, None)
            if array is not None:
                array[...] = cp.asarray(1e-3 * rng.random(array.shape), dtype=cp.float32)
    if specified:
        cfg = dataclasses.replace(cfg, specified=True, spec_zone=1,
                                  relax_zone=4, spec_bdy_width=5)
    return state, cfg


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("msf", [False, True])
@pytest.mark.parametrize("specified", [False, True])
@pytest.mark.parametrize("emdiv", [False, True])
@pytest.mark.parametrize("moist", [False, True])
@pytest.mark.parametrize("join", ["default", "column", "column_frame_separate",
                                  "column_sumflux_separate"])
def test_joined_substeps_write_the_separate_words(msf, specified, emdiv, moist, join):
    import cupy as cp

    from gpuwm.core import acoustic, acoustic_fusion, dycore
    from gpuwm.core.kernels import get_kernel

    if acoustic.WRF_EXACT:
        pytest.skip("the strict build keeps the separate kernels")
    column = {"JOIN_COLUMN": True, "JOIN_COLUMN_FRAME": True, "JOIN_COLUMN_SUMFLUX": True}
    knobs = {"default": {}, "column": column,
             "column_frame_separate": {**column, "JOIN_COLUMN_FRAME": False},
             "column_sumflux_separate": {**column, "JOIN_COLUMN_SUMFLUX": False}}[join]
    results = []
    for joined in (False, True):
        state, cfg = _state(11, msf=msf, hill=True, specified=specified, moist=moist)
        cfg = dataclasses.replace(cfg, emdiv=0.01 if emdiv else 0.0)
        nz, ny, nx = state.p.shape
        rng = np.random.default_rng(5)
        mudf = None
        if emdiv:
            mudf = state.scratch(state.mup.shape, "acoustic_mudf")
            mudf[...] = cp.asarray(rng.standard_normal(mudf.shape), dtype=cp.float32)
        sums = tuple(state.scratch(shape, name) for shape, name in (
            ((nz, ny, nx + 1), "rk_ru_m"), ((nz, ny + 1, nx), "rk_rv_m"),
            ((nz + 1, ny, nx), "rk_ww_m")))
        for array in sums:
            array[...] = cp.asarray(rng.standard_normal(array.shape), dtype=cp.float32)
        dtau = 0.75
        coefficients = acoustic.prepare_acoustic_coefficients(state, cfg, dtau)
        emdiv_launch = dycore._prepare_emdiv_filter_launch(state, cfg, mudf) if emdiv else None
        saved = {name: getattr(acoustic_fusion, name) for name in knobs}
        acoustic_fusion.ENABLED = joined
        for name, value in knobs.items():
            setattr(acoustic_fusion, name, value)
        try:
            launch = acoustic.prepare_acoustic_substep_launch(
                state, cfg, dtau, coefficients, mudf=mudf,
                emdiv_args=emdiv_launch.args if emdiv else None, sumflux=sums)
            column_joined = acoustic_fusion.JOIN_COLUMN
            sumflux_joined = column_joined and acoustic_fusion.JOIN_COLUMN_SUMFLUX
        finally:
            acoustic_fusion.ENABLED = True
            for name, value in saved.items():
                setattr(acoustic_fusion, name, value)
        absorbed = launch.absorbs
        if joined:
            assert launch.joined[1] == column_joined
            assert (("sumflux" in absorbed) == sumflux_joined
                    and (("emdiv" in absorbed) == emdiv))
        zero = dycore._prepare_sumflux_launch("zero_sumflux", sums)
        accumulate = dycore._prepare_sumflux_launch(
            "accumulate_sumflux", sums, (state.u_pp, state.v_pp, state.ww_pp))
        for index in range(3):
            first = index == 0
            if emdiv and "emdiv" not in absorbed:
                emdiv_launch()
            if first and "sumflux" not in absorbed:
                zero()
            launch(first=first)
            if "sumflux" not in absorbed:
                accumulate()
        cp.cuda.runtime.deviceSynchronize()
        words = {}
        for name, value in vars(state).items():
            if isinstance(value, cp.ndarray) and value.dtype == cp.float32:
                words[name] = cp.asnumpy(value).view(np.uint32)
        for name, array in zip(("ru_m", "rv_m", "ww_m"), sums):
            words[name] = cp.asnumpy(array).view(np.uint32)
        if mudf is not None:
            words["mudf"] = cp.asnumpy(mudf).view(np.uint32)
        results.append(words)
    separate, joined_words = results
    assert separate.keys() == joined_words.keys()
    for name in separate:
        differ = int(np.count_nonzero(separate[name] != joined_words[name]))
        assert differ == 0, f"{name}: {differ} words differ"


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("count", [1, 2, 3])
def test_joined_pd_fold_writes_the_separate_words(count):
    """``q0_eff = q0; q0_eff += dt*S/chm0`` per source, joined and separate."""
    import cupy as cp

    from gpuwm.core.moist import _pd_fold_kernel

    rng = np.random.default_rng(count)
    shape = (7, 9, 11)

    def field(scale):
        values = (scale * rng.standard_normal(shape)).astype(np.float32)
        flat = values.reshape(-1)
        # Signed zeros, subnormals, infinities and NaN reach the same ops.
        flat[:6] = np.array([0.0, -0.0, 1e-41, -1e-41, np.inf, np.nan], dtype=np.float32)
        return cp.asarray(values)

    q0 = field(1e-3)
    sources = [field(1e-5) for _ in range(count)]
    chm0 = cp.abs(field(1e3)) + cp.float32(1.0)
    dt = 7.5
    separate = q0.copy()
    for source in sources:
        separate += dt * source / chm0
    joined = cp.empty_like(q0)
    _pd_fold_kernel(count)(q0, *sources, chm0, np.float32(dt), joined)
    assert np.array_equal(cp.asnumpy(separate).view(np.uint32),
                          cp.asnumpy(joined).view(np.uint32))


def test_a_swapped_kernel_accessor_keeps_the_separate_kernels(monkeypatch):
    """The small-step WRF oracles swap acoustic.get_kernel for probe kernels;
    a joined launch would bypass the probe and grade the wrong kernel."""
    from gpuwm.core import acoustic

    if acoustic.WRF_EXACT:
        pytest.skip("the strict build keeps the separate kernels anyway")
    assert acoustic._fusion_enabled(49)
    monkeypatch.setattr(acoustic, "get_kernel", lambda *args: None)
    assert not acoustic._fusion_enabled(49)
    monkeypatch.undo()
    monkeypatch.setattr(acoustic, "_w_phi_kernel", lambda *args: None)
    assert not acoustic._fusion_enabled(49)
