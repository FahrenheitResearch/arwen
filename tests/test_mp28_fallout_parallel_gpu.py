"""Default exact fallout against the retained serial kernels, raw words."""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("nz", [2, 17, 49, 64, 73])
@pytest.mark.parametrize("kind", ["snow", "graupel"])
@pytest.mark.parametrize("accumulate", [True, False])
def test_parallel_fallout_keeps_every_serial_word(monkeypatch, nz, kind, accumulate):
    cp = pytest.importorskip("cupy")
    from gpuwm.core import thompson_aerosol_sed as sed
    rng = np.random.default_rng(472)
    shape = (nz, 2, 7)
    def dev(value):
        return cp.full(shape, value, cp.float32)
    mass = np.zeros(shape, np.float32)
    mass[:, :, 0] = -0.0
    mass[:, :, 1] = 1.e-12
    mass[nz // 2:, :, 2:] = rng.uniform(1e-6, 1e-3, (nz-nz//2, 2, 5))
    tendency = np.zeros(shape, np.float32)
    tendency[:, :, 1] = -1.e-14
    tendency[:, :, 3] = 1.e-7
    temp, pres, qv, dz, rho = (dev(x) for x in (267, 80000, 0.002, 150, 1.05))
    dz[:, :, 6] = .5  # multiple substeps
    rain = dev(0)
    rain[nz//3:, :, 4:] = 1.0
    melt = dev(0); melt[:nz//2, :, 5:] = 1
    active = cp.ones(shape[1:], cp.float32); active[:, :2] = 0
    def run():
        q = cp.asarray(mass); n = dev(1000)
        qt = cp.asarray(tendency); nt = dev(0)
        ppt = cp.empty(shape[1:], cp.float32)
        if kind == "snow":
            sed.launch_aa_snow_sedimentation(q, temp, pres, qv, dz, ppt, 20.,
                reference_density=rho, reference_temperature=temp,
                snow_melt_marker=melt, velocity_boost=dev(1), qr1d=dev(1e-4),
                nr1d=dev(1000), qrten=dev(1e-8), nrten=dev(0), rain_density=rain,
                qsten=qt if accumulate else None)
        else:
            sed.launch_aa_graupel_sedimentation(q, n, temp, pres, qv, dz, ppt, 20.,
                reference_density=rho, active_columns=active, rain_density=rain,
                qgten=qt if accumulate else None, ngten=nt if accumulate else None)
        return [cp.asnumpy(x).view(np.uint32) for x in (q, n, qt, nt, ppt)]
    got = run()
    kernel = sed.aerosol_kernel
    def serial(module, name):
        if "_levels_" not in name:
            return kernel(module, name)
        func = kernel(module, name.replace("_levels", ""))
        def launch(grid, block, args):
            columns = int(args[-2]) * int(args[-1])
            func(((columns + 31)//32,), (32,), args)
        return launch
    monkeypatch.setattr(sed, "aerosol_kernel", serial)
    expected = run()
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)
