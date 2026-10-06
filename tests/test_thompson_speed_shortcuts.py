"""Thompson's exact shortcuts and level-parallel fallout against the full sweeps.

The reference is the same translation unit compiled with
THOMPSON_NO_EXACT_SHORTCUTS defined, which removes every exact shortcut and
nothing else; a level-parallel kernel is compared with the column kernel it
replaces.  Every output is compared as raw words, signed zeros included.
"""
from functools import lru_cache

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_NO_SHORTCUTS = (("THOMPSON_NO_EXACT_SHORTCUTS", 1),)


def _require_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    return cp


@lru_cache(None)
def _reference(unit):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import module_source, module_source_int_defines
    assert "#ifndef THOMPSON_NO_EXACT_SHORTCUTS" in module_source(unit)
    return cp.RawModule(code=module_source_int_defines(unit, _NO_SHORTCUTS),
                        options=("-std=c++17",))


@pytest.mark.parametrize("nz", [2, 59, 65])
@pytest.mark.parametrize("species", ["rain", "ice", "snow", "graupel", "cloud", "aa_cloud"])
@pytest.mark.parametrize("held", [False, True])
def test_empty_and_active_columns_match_base_bits(monkeypatch, nz, species, held):
    cp = _require_device()
    from gpuwm.core import thompson as classic
    from gpuwm.core import thompson_aerosol_sed as aerosol
    module = aerosol if species == "aa_cloud" else classic
    unit = "thompson_aerosol_sed" if species == "aa_cloud" else "thompson"
    rng = np.random.default_rng(329)
    shape = (nz, 2, 8)
    q = np.zeros(shape, np.float32)
    q[:, :, 0] = np.float32(-0.0)
    q[:, :, 1] = np.float32(1.0e-12)
    q[:, :, 2] = np.float32(5.0e-13)
    q[:, :, 3] = np.float32(-1.0e-13)
    q[:, :, 4] = rng.uniform(1.0e-6, 1.0e-3, (nz, 2))
    q[nz // 2, :, 5] = 1.0e-4
    q[-1, :, 6] = 1.0e-4
    number = rng.uniform(1, 10000, shape).astype(np.float32)
    number[:, :, 0] = np.float32(-0.0)
    env = [np.full(shape, v, np.float32) for v in (270, 80000, .002, 150)]
    env[3][:, :, 7] = .5  # guard fallback
    density = np.full(shape, 1.05, np.float32)
    surface = np.full((2, 8), np.float32(-0.0))

    def run():
        mass = cp.asarray(q)
        num = cp.asarray(number)
        temp, pres, vapor, depth = map(cp.asarray, env)
        rho = cp.asarray(density)
        accum = [cp.asarray(surface) for _ in range(4)]
        kw = {"reference_density": rho} if held else {}
        dt = 10.0
        if species == "rain":
            if held:
                rho[:, :, :4] = 0.0
                rho[:, :, 4] = 0.0  # Positive mass still receives the MVD bound
                kw["density_carries_rain_presence"] = True
            classic.launch_rain_sedimentation(mass, num, temp, pres, vapor,
                depth, *accum[:2], dt, accumulate_surface=held, **kw)
        elif species == "ice":
            classic.launch_ice_sedimentation(mass, num, temp, pres, vapor,
                depth, *accum, dt, **kw)
        elif species == "snow":
            if held:
                kw.update(snow_melt_marker=cp.ones(shape, cp.float32),
                    melt_rain_qr=cp.full(shape, 1.0e-4, cp.float32),
                    melt_rain_nr=cp.full(shape, 1000, cp.float32),
                    reference_temperature=temp.copy(), velocity_boost=cp.ones(shape, cp.float32),
                    melt_rain_density=rho.copy(), melt_rain_density_carries_presence=True)
            classic.launch_snow_sedimentation(mass, temp, pres, vapor,
                depth, *accum, dt, accumulate_surface=held, **kw)
        elif species == "graupel":
            if held:
                kw.update(active_columns=cp.ones((2, 8), cp.float32),
                          graupel_number_shadow=num)
            classic.launch_graupel_sedimentation(mass, temp, pres, vapor,
                depth, *accum, dt, accumulate_surface=held, **kw)
        elif species == "cloud":
            if held:
                rain_mask = cp.ones((2, 8), cp.float32)
                cloud_mask = cp.ones((2, 8), cp.float32)
                cloud_mask[:, 6] = 0.0
                kw.update(rain_active_columns=rain_mask, cloud_active_columns=cloud_mask)
            classic.launch_cloud_sedimentation(mass, temp, pres, vapor,
                cp.zeros(shape, cp.float32), depth, dt, **kw)
        else:
            tendency = cp.zeros(shape, cp.float32)
            tendency[:, :, 0] = cp.float32(-0.0)
            mask_kw = {}
            if held:
                rain_mask = cp.ones((2, 8), cp.float32)
                cloud_mask = cp.ones((2, 8), cp.float32)
                cloud_mask[:, 6] = 0.0
                mask_kw.update(rain_active_columns=rain_mask, cloud_active_columns=cloud_mask)
            aerosol.launch_aa_cloud_sedimentation(mass, num, tendency,
                temp, pres, vapor, cp.zeros(shape, cp.float32), depth, dt,
                reference_density=rho, **mask_kw)
            accum.append(tendency)
        return [cp.asnumpy(a).view(np.uint32) for a in (mass, num, *accum)]

    actual = run()
    def reference_kernel(name, symbol):
        assert name == unit, (name, unit)
        if "_levels_" in symbol:
            # The column kernel the level-parallel one replaces, one column
            # per thread.
            function = _reference(unit).get_function(symbol.replace("_levels", ""))
            def launch(grid, block, arguments):
                ncol = int(arguments[-2]) * int(arguments[-1])
                function(((ncol + 31) // 32,), (32,), arguments)
            return launch
        return _reference(unit).get_function(symbol)
    # The aerosol-aware units launch through aerosol_kernel (the active
    # generation's build, b0556bd76); classic Thompson through get_kernel.
    monkeypatch.setattr(module, "aerosol_kernel" if module is aerosol else "get_kernel",
                        reference_kernel)
    expected = run()
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_empty_cell_effective_radius_matches_the_full_path():
    cp = _require_device()
    from gpuwm.core.thompson import launch_effective_radius

    rng = np.random.default_rng(531)
    n = 1031
    fields = [rng.uniform(220, 305, n), rng.uniform(8000, 100000, n),
              rng.uniform(1e-5, 0.02, n), rng.uniform(0, 1e-3, n),
              rng.uniform(0, 1e-3, n), rng.uniform(0, 1e6, n),
              rng.uniform(0, 1e-3, n)]
    fields = [a.astype(np.float32) for a in fields]
    for i in (3, 4, 6):
        fields[i][::2] = 0.0
        fields[i][1::4] = -0.0
    # Empty cells also meet a non-finite density and signed-zero inputs.
    special = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], np.float32)
    for i in (0, 1, 2, 5):
        fields[i][:5] = special
    for i in (3, 4, 6):
        fields[i][:5] = 0.0
    # Non-empty exceptional and threshold cells keep the complete path.
    for i in (3, 4, 6):
        fields[i][5:10] = [np.nan, np.inf, -np.inf, 1e-12, -1e-12]
    device = [cp.asarray(a) for a in fields]
    expected = [cp.empty(n, cp.float32) for _ in range(3)]
    actual = [cp.empty(n, cp.float32) for _ in range(3)]
    _reference("thompson").get_function("thompson_effective_radius")(
        ((n + 255) // 256,), (256,), (*device, *expected, np.int32(n)))
    launch_effective_radius(*device, *actual)
    for ref, got in zip(expected, actual):
        np.testing.assert_array_equal(cp.asnumpy(ref).view(np.uint32),
                                      cp.asnumpy(got).view(np.uint32))
