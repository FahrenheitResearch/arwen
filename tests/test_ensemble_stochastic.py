"""Stochastic mechanism contracts and GPU member/restart identity checks.

GPU checks compare independent counter vectors and spectral invariants.
They are not observation-based calibration or full WRF oracle results.
"""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import pytest

from gpuwm.ensemble.stochastic import (
    RNG_VERSION, StochasticConfig, StochasticTimestepHook, WrfSkebs, WrfStochasticPattern,
    apply_nonmicrophysics_sppt, philox4x32_10, spp_pattern_generators,
)


def test_philox_published_known_answers():
    # Random123/tests/kat_vectors: Philox4x32, 10 rounds.
    assert philox4x32_10((0, 0, 0, 0), (0, 0)) == (
        0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    assert philox4x32_10((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2) == (
        0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)
    assert RNG_VERSION.endswith(".v1")


def test_reference_settings_and_named_unsupported_vertical_structure():
    assert StochasticConfig.wrf_reference("sppt").stddev == 0.5
    assert StochasticConfig.wrf_reference("skebs_psi").backscatter == 1e-5
    assert StochasticConfig.wrf_reference("spp_pbl").lengthscale_m == 700000
    assert StochasticConfig.wrf_reference("spp_lsm").timescale_s == 86400
    with pytest.raises(ValueError, match="vertical phase rotation"):
        StochasticConfig(vertical_structure=1)
    with pytest.raises(ValueError, match="reverse physical tendencies"):
        StochasticConfig(stddev=0.7)
    with pytest.raises(ValueError, match="finite and positive"):
        StochasticConfig(timescale_s=math.nan)
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        philox4x32_10((True, 0, 0, 0), (0, 0))


def test_disabled_hook_identity_and_missing_spp_consumers_without_cuda():
    hook = StochasticTimestepHook((1, 1), dx=1, dy=1, dt=1, member_seed=0,
                                  enabled=False, sppt=StochasticConfig())
    original = object()
    hook.before_timestep(0)
    assert hook.after_nonmicrophysics(original, tendency_scope="nonmicrophysics") is original
    with pytest.raises(ValueError, match="spp_levels"):
        StochasticTimestepHook((32, 36), dx=12000, dy=9000, dt=60,
                              member_seed=0, spp=True)
    with pytest.raises(ValueError, match="at least 12 points"):
        WrfStochasticPattern(StochasticConfig(), (11, 12),
                             dx=12000, dy=9000, dt=60, member_seed=0)


def test_spp_config_keys_and_kinds_validate_before_device_allocation():
    kwargs = dict(shape_yx=(32,36),dx=12000,dy=9000,dt=60,member_seed=0,
                  spp=True,spp_levels={"conv":4,"pbl":9})
    with pytest.raises(ValueError,match="exactly match spp_levels"):
        StochasticTimestepHook(**kwargs,spp_configs={"pbl":StochasticConfig.wrf_reference("spp_pbl")})
    with pytest.raises(ValueError,match="kind spp_conv"):
        StochasticTimestepHook(**kwargs,spp_configs={
            "conv":StochasticConfig.wrf_reference("spp_pbl"),
            "pbl":StochasticConfig.wrf_reference("spp_pbl")})
    hook = StochasticTimestepHook(**kwargs,enabled=False,spp_configs={"ignored":object()})
    assert hook.spp == {} and hook.spp_configs == {}


def _cupy():
    import os
    if os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"):
        pytest.skip("GPUWM_NO_LOCAL_GPU: stochastic runtime requires a permitted node")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device")
    return cp


def _pattern(seed=83, config=None, shape=(32, 36)):
    return WrfStochasticPattern(config or StochasticConfig(), shape,
                                dx=12000.0, dy=9000.0, dt=60.0,
                                member_seed=seed)


@pytest.mark.gpu
def test_gpu_philox_words_match_scalar_and_known_vectors():
    cp = _cupy()
    from gpuwm.ensemble.stochastic import _kernel
    counters = [(0, 0, 0, 0), (0xFFFFFFFF,) * 4,
                (42, 0, 1, 0), (42, 59, 5, 4)]
    seed = 0xBAD5EED112233445
    values = cp.asarray(counters, dtype=cp.uint32)
    result = cp.empty_like(values)
    _kernel("stoch_rng_words")((1,), (32,), (result, values, np.uint64(seed), np.int32(4)))
    expected = [philox4x32_10(c, (seed & 0xFFFFFFFF, seed >> 32)) for c in counters]
    np.testing.assert_array_equal(cp.asnumpy(result), np.asarray(expected, dtype=np.uint32))


@pytest.mark.gpu
def test_member_order_launch_geometry_and_restart_identity():
    cp = _cupy()
    first, second = _pattern(51), _pattern(72)
    reordered_second, reordered_first = _pattern(72), _pattern(51)
    for step in range(3):
        a, b = first.advance(step, block_size=64), second.advance(step, block_size=256)
        b2 = reordered_second.advance(step, block_size=96)
        a2 = reordered_first.advance(step, block_size=192)
        assert bool(cp.array_equal(a, a2))
        assert bool(cp.array_equal(b, b2))
    assert not bool(cp.array_equal(a, b))
    restart = first.snapshot()
    restored = _pattern(51)
    restored.restore(restart)
    assert bool(cp.array_equal(first.advance(3), restored.advance(3)))
    with pytest.raises(ValueError, match="metadata"):
        second.restore(restart)
    with pytest.raises(ValueError, match="consecutive"):
        first.advance(5)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["sppt", "skebs_psi", "skebs_theta", "spp_pbl"])
def test_spectral_setup_matches_independent_equations(kind):
    cp = _cupy()
    cfg = StochasticConfig.wrf_reference(kind)
    process = _pattern(config=cfg)
    ny, nx = process.shape
    rx, ry = nx * process.dx, ny * process.dy
    high = min(nx // 2, ny // 2) - 5
    chi = np.zeros((ny, nx), dtype=np.float64)
    gamma = 0.0
    for y in range(ny):
        for x in range(nx):
            rho2 = (x / rx) ** 2 + (y / ry) ** 2
            rho = math.sqrt(rho2)
            if rho2 == 0 or not (((cfg.min_wavenumber - .5) / rx <= rho < (high + .5) / rx)
                                 or ((cfg.min_wavenumber - .5) / ry <= rho < (high + .5) / ry)):
                continue
            if kind.startswith("skebs"):
                chi[y, x] = rho2 ** (cfg.spectral_exponent * .5)
                gamma += rho2 ** (cfg.spectral_exponent + (kind == "skebs_psi"))
            else:
                chi[y, x] = math.exp(-2 * math.pi ** 2 * cfg.lengthscale_m ** 2 * rho2)
                gamma += chi[y, x] ** 2
    gamma *= 4
    if kind.startswith("skebs"):
        alpha = process.dt / cfg.timescale_s
        sigma2 = 1 / (12 * alpha)
        energy = alpha * cfg.backscatter / (process.dt * sigma2 * gamma)
        f0 = math.sqrt(energy) / (2 * math.pi) if kind == "skebs_psi" else math.sqrt(300 * energy / 1004.5)
    else:
        f0 = cfg.stddev * math.sqrt(-math.expm1(-2 * process.dt / cfg.timescale_s) / (2 * gamma))
    expected = np.empty((ny, nx))
    for y in range(ny):
        for x in range(nx):
            expected[y, x] = chi[min(y, ny-y), min(x, nx-x)] * f0
    np.testing.assert_allclose(cp.asnumpy(process.amplitude), expected, rtol=2e-6, atol=1e-25)


@pytest.mark.gpu
def test_small_domain_long_correlation_has_finite_nonzero_forcing():
    cp = _cupy()
    process = _pattern(config=replace(StochasticConfig.wrf_reference("spp_pbl"),
                                     lengthscale_m=2e6), shape=(16, 18))
    field = process.advance(0)
    assert bool(cp.all(cp.isfinite(field)))
    assert float(cp.max(cp.abs(field))) > 0


@pytest.mark.gpu
def test_hermitian_symmetry_and_rotational_skebs():
    cp = _cupy()
    process = WrfSkebs((32, 36), dx=12000, dy=9000, dt=60, member_seed=51)
    output = process.advance(0)
    spectrum = cp.asnumpy(process.psi.spectrum)
    ny, nx = spectrum.shape
    mirror = np.conj(spectrum[np.ix_((-np.arange(ny)) % ny, (-np.arange(nx)) % nx)])
    np.testing.assert_array_equal(spectrum, mirror)
    assert np.max(np.abs(np.fft.ifft2(spectrum.astype(np.complex128)).imag)) < 1e-14
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=12000)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=9000)
    u = cp.asnumpy(output["u"])
    v = cp.asnumpy(output["v"])
    divergence = np.fft.ifft2(1j * kx[None, :] * np.fft.fft2(u)
                             + 1j * ky[:, None] * np.fft.fft2(v)).real
    assert np.max(np.abs(divergence)) < 2e-11
    assert float(cp.max(cp.abs(output["theta"]))) > 0


@pytest.mark.gpu
def test_sppt_common_factor_scopes_and_spp_distinct_streams():
    cp = _cupy()
    field = cp.full((16, 18), 0.25, dtype=cp.float32)
    tendencies = {k: cp.ones((4, 15, 17), dtype=cp.float32)
                  for k in ("u", "v", "theta", "qv")}
    result = apply_nonmicrophysics_sppt(field, tendencies, tendency_scope="nonmicrophysics")
    for name in tendencies:
        assert bool(cp.all(result[name] == 1.25))
        assert bool(cp.all(tendencies[name] == 1.0))
    with pytest.raises(ValueError, match="microphysics increment"):
        apply_nonmicrophysics_sppt(field, tendencies, tendency_scope="all")
    generators = spp_pattern_generators((32, 36), dx=12000, dy=9000,
                                        dt=60, member_seed=51)
    assert len({p.stream for p in generators.values()}) == 3
    assert all(bool(cp.all(cp.isfinite(p.advance(0)))) for p in generators.values())


@pytest.mark.gpu
def test_hook_wrf_order_staggered_fields_held_inputs_and_restart():
    cp = _cupy()
    kwargs = dict(dx=12000, dy=9000, dt=60, member_seed=22,
                  sppt=StochasticConfig.wrf_reference("sppt"),
                  skebs_psi=StochasticConfig.wrf_reference("skebs_psi"),
                  skebs_theta=StochasticConfig.wrf_reference("skebs_theta"))
    hook = StochasticTimestepHook((32, 36), **kwargs)
    shapes = {"u": (3, 31, 36), "v": (3, 32, 35),
              "theta": (3, 31, 35), "qv": (3, 31, 35)}
    held = {k: cp.ones(shape, dtype=cp.float32) for k, shape in shapes.items()}
    mass = {k: cp.full(shapes[k], 500.0, dtype=cp.float32) for k in ("u", "v", "theta")}
    hook.before_timestep(0)
    expected = {}
    for name, values in held.items():
        ny, nx = values.shape[-2:]
        added = values
        if name != "qv":
            added = values + hook._forcing[name][:ny, :nx] * mass[name]
        expected[name] = added * (1 + hook._pattern[:ny, :nx])
    with pytest.raises(ValueError, match="not been applied"):
        hook.before_timestep(1)
    with pytest.raises(ValueError, match="after applying"):
        hook.snapshot()
    result = hook.after_nonmicrophysics(held, tendency_scope="nonmicrophysics", mass_factors=mass)
    for name in shapes:
        assert bool(cp.array_equal(result[name], expected[name]))
        assert bool(cp.all(held[name] == 1))
    resumed = StochasticTimestepHook((32, 36), **kwargs)
    resumed.restore(hook.snapshot())
    hook.before_timestep(1)
    resumed.before_timestep(1)
    first = hook.after_nonmicrophysics(held, tendency_scope="nonmicrophysics", mass_factors=mass)
    second = resumed.after_nonmicrophysics(held, tendency_scope="nonmicrophysics", mass_factors=mass)
    assert all(bool(cp.array_equal(first[k], second[k])) for k in first)


@pytest.mark.gpu
def test_native_wrf_setup_oracle():
    """Measured setup comparison; requires externally compiled exact WRF source.

    The coefficient tolerance covers the declared binary64 normalization vs
    native REAL arithmetic, including WRF's cancellation in 1-exp(-dt/tau).
    This oracle does not run either FFT and therefore does not validate FFT
    or RNG agreement. Native nonfinite rows are recorded as defects, not
    scored as numerical agreement.
    """
    cp = _cupy()
    directory = os.environ.get("GPUWM_STOCH_WRF_ORACLE")
    if not directory:
        pytest.skip("set GPUWM_STOCH_WRF_ORACLE to the compiled WRF setup oracle outputs")
    root = Path(directory)
    rows = []
    for shape, folder in [((32, 36), root), ((12, 12), root / "edge12")]:
        log = (folder / "oracle.log").read_text()
        for letter, kind in [("P", "sppt"), ("W", "skebs_psi"),
                             ("T", "skebs_theta"), ("Q", "spp_pbl")]:
            native = np.fromfile(folder / f"amplitude-{letter}.bin", dtype=np.float32).reshape(shape)
            process = _pattern(config=StochasticConfig.wrf_reference(kind), shape=shape)
            actual = cp.asnumpy(process.amplitude)
            alpha = float(re.search(rf"ORACLE{letter}alpha\s+(\S+)", log).group(1))
            # dt/tau in native REAL and exp rounded in native REAL.
            assert abs(process.alpha - alpha) < 3e-8
            row = {"shape": shape, "scheme": kind, "native_finite": bool(np.isfinite(native).all()),
                   "actual_finite": bool(np.isfinite(actual).all()),
                   "alpha_max_abs": abs(process.alpha - alpha)}
            assert row["actual_finite"]
            if row["native_finite"]:
                error = np.abs(native.astype(np.float64) - actual)
                active = np.abs(native) > max(float(np.max(np.abs(native))) * 1e-6, 1e-30)
                row.update(max_abs=float(error.max()),
                           max_relative_active=float(np.max(error[active] / np.abs(native[active]))))
                assert row["max_relative_active"] < 3e-5, row
                np.testing.assert_allclose(actual, native, rtol=3e-5,
                                           atol=float(np.max(np.abs(native))) * 1e-10)
            else:
                assert kind == "spp_pbl", row
                assert np.count_nonzero(actual) > 0
            rows.append(row)
    edge_log = (root / "edge11" / "oracle.log").read_text()
    assert "NaN" in edge_log
    print("NATIVE_WRF_STOCHASTIC_SETUP " + json.dumps(rows, sort_keys=True))


@pytest.mark.gpu
def test_native_wrf_update_oracle_with_prescribed_innovations():
    """Exact native UPDATE_STOCH given the same samples, separate from RNG."""
    cp = _cupy()
    directory = os.environ.get("GPUWM_STOCH_WRF_ORACLE")
    if not directory:
        pytest.skip("set GPUWM_STOCH_WRF_ORACLE to the compiled WRF update oracle outputs")
    from gpuwm.ensemble.stochastic import _kernel
    root = Path(directory)
    rows = []
    for shape, folder in [((32, 36), root), ((12, 12), root / "edge12")]:
        ny, nx = shape
        index = cp.arange(ny * nx).reshape(shape)
        amplitude = ((index % 13 + 1) * 0.03125).astype(cp.float32)
        noise = cp.empty(shape, dtype=cp.complex64)
        noise.real = ((index % 7 - 3) * 0.25).astype(cp.float32)
        noise.imag = ((index % 5 - 2) * 0.5).astype(cp.float32)
        spectrum = cp.full(shape, complex(0.03125, -0.0625), dtype=cp.complex64)
        for step in (1, 2):
            _kernel("stoch_update_given_noise")(((nx*ny+127)//128,), (128,), (
                spectrum, amplitude, noise, np.int32(nx), np.int32(ny), np.float32(0.125)))
            native = np.fromfile(folder / f"update-{step}.bin", dtype=np.float32).reshape(2, ny, nx)
            actual = cp.asnumpy(spectrum)
            np.testing.assert_array_equal(actual.real.view(np.uint32), native[0].view(np.uint32))
            np.testing.assert_array_equal(actual.imag.view(np.uint32), native[1].view(np.uint32))
            rows.append({"shape": shape, "step": step, "words": 2*nx*ny,
                         "different_words": 0})
    print("NATIVE_WRF_STOCHASTIC_UPDATE " + json.dumps(rows, sort_keys=True))


@pytest.mark.gpu
def test_spp_default_vertical_broadcast_streams_and_restart():
    cp = _cupy()
    kwargs = dict(dx=12000, dy=9000, dt=60, member_seed=73,
                  spp=True, spp_levels={"conv": 4, "pbl": 9, "lsm": 6})
    first = StochasticTimestepHook((32, 36), **kwargs)
    first.before_timestep(0)
    held = {name: cp.ones((9,31,35), dtype=cp.float32)
            for name in ("u", "v", "theta", "qv")}
    for name, depth in kwargs["spp_levels"].items():
        values = first.parameter_patterns[name]
        assert values.shape == (depth,31,35)
        assert values[0].flags.c_contiguous
        assert all(bool(cp.array_equal(values[0], values[k])) for k in range(depth))
    assert not bool(cp.array_equal(first.parameter_patterns["conv"][0], first.parameter_patterns["pbl"][0]))
    applied = first.after_nonmicrophysics(held, tendency_scope="nonmicrophysics")
    assert all(applied[name] is held[name] for name in held)
    second = StochasticTimestepHook((32,36), **kwargs)
    second.restore(first.snapshot())
    for step in (1,2):
        first.before_timestep(step)
        second.before_timestep(step)
        assert all(bool(cp.array_equal(first.parameter_patterns[name], second.parameter_patterns[name]))
                   for name in kwargs["spp_levels"])
        first.after_nonmicrophysics(held, tendency_scope="nonmicrophysics")
        second.after_nonmicrophysics(held, tendency_scope="nonmicrophysics")


@pytest.mark.gpu
def test_accepted_adaptive_steps_and_restart_preserve_every_pattern_word():
    cp = _cupy()
    kwargs = dict(dx=12000, dy=9000, dt=60, member_seed=73,
                  sppt=StochasticConfig.wrf_reference("sppt"),
                  skebs_psi=StochasticConfig.wrf_reference("skebs_psi"),
                  skebs_theta=StochasticConfig.wrf_reference("skebs_theta"),
                  spp=True, spp_levels={"pbl": 9, "lsm": 6})
    first, independent = (StochasticTimestepHook((32, 36), **kwargs) for _ in range(2))
    held = {name: cp.ones((9, 31, 35), dtype=cp.float32)
            for name in ("u", "v", "theta", "qv")}
    for step, dt in enumerate((60.0, 45.0, 30.0, 42.5)):
        first.set_time_step(dt)
        independent.set_time_step(dt)
        first.before_timestep(step)
        independent.before_timestep(step)
        left = first.after_nonmicrophysics(held, tendency_scope="nonmicrophysics")
        right = independent.after_nonmicrophysics(held, tendency_scope="nonmicrophysics")
        for name in held:
            np.testing.assert_array_equal(cp.asnumpy(left[name]).view(np.uint32),
                                          cp.asnumpy(right[name]).view(np.uint32))
        if step == 1:
            resumed = StochasticTimestepHook((32, 36), **kwargs)
            resumed.restore(first.snapshot())
            assert resumed.sppt.dt == 45.0
            independent = resumed


@pytest.mark.gpu
def test_spp_amplitude_configs_preserve_seeds_and_bind_restart():
    cp = _cupy()
    levels = {"conv":4,"pbl":9,"lsm":6}
    configs = {name:StochasticConfig.wrf_reference("spp_"+name) for name in levels}
    half = {name:replace(config,stddev=config.stddev*.5) for name,config in configs.items()}
    kwargs = dict(shape_yx=(32,36),dx=12000,dy=9000,dt=60,member_seed=73,
                  spp=True,spp_levels=levels)
    original = StochasticTimestepHook(**kwargs)
    explicit = StochasticTimestepHook(**kwargs,spp_configs=configs)
    scaled = StochasticTimestepHook(**kwargs,spp_configs=half)
    held = {name:cp.zeros((9,31,35),cp.float32) for name in ("u","v","theta","qv")}
    for tick in range(3):
        for hook in (original,explicit,scaled):
            hook.before_timestep(tick)
        for name in levels:
            cp.testing.assert_array_equal(original.parameter_patterns[name],explicit.parameter_patterns[name])
            cp.testing.assert_array_equal(scaled.parameter_patterns[name],original.parameter_patterns[name]*cp.float32(.5))
        for hook in (original,explicit,scaled):
            hook.after_nonmicrophysics(held,tendency_scope="nonmicrophysics")
    resumed = StochasticTimestepHook(**kwargs,spp_configs=half)
    resumed.restore(scaled.snapshot())
    with pytest.raises(ValueError,match="metadata"):
        original.restore(scaled.snapshot())
    resumed.before_timestep(3)
    scaled.before_timestep(3)
    for name in levels:
        cp.testing.assert_array_equal(resumed.parameter_patterns[name],scaled.parameter_patterns[name])
