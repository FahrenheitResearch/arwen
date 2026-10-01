"""Bit checks for classic Thompson cold-cell launch geometry."""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_cold_blocks_match_base_bits(monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    import gpuwm.core.thompson as t
    from gpuwm.core.thompson_runtime import load_classic_device_tables
    from gpuwm.physics_compat import thompson_table_root

    owner = load_classic_device_tables(thompson_table_root())
    rng = np.random.default_rng(817)
    n = 1093
    fields = {name: rng.uniform(1e-6, 2e-3, n).astype(np.float32)
              for name in ("qi", "qs", "qg", "qr", "qc")}
    fields.update(ni=rng.uniform(1, 2e5, n).astype(np.float32),
                  nr=rng.uniform(1, 2e5, n).astype(np.float32),
                  temperature=rng.uniform(230, 290, n).astype(np.float32),
                  pressure=rng.uniform(15000, 100000, n).astype(np.float32),
                  qv=rng.uniform(1e-5, 0.018, n).astype(np.float32),
                  shadow=np.zeros(n, np.float32), boost=np.zeros(n, np.float32))
    for name in ("qi", "qs", "qg", "qr", "qc"):
        fields[name][::3] = 0.0
        fields[name][1::7] = 1e-12
    reference = {name: cp.asarray(a) for name, a in fields.items()}
    actual = {name: cp.asarray(a) for name, a in fields.items()}

    def launch(f):
        t.launch_frozen_vapor_network_from_owner(
            *(f[name] for name in ("qi", "ni", "qs", "qg", "qr", "nr",
                                  "temperature", "pressure", "qv")),
            owner, 6.0, qc=f["qc"], graupel_number_shadow=f["shadow"],
            snow_velocity_boost=f["boost"])

    original_get_kernel = t.get_kernel

    def base_get(unit, symbol):
        # The same compiled kernel at the former 256-thread launch.
        assert unit == "thompson"
        kernel = original_get_kernel(unit, symbol)

        def base_call(grid, block, args):
            return kernel(((int(args[-1]) + 255) // 256,), (256,), args)

        return base_call

    with monkeypatch.context() as context:
        context.setattr(t, "get_kernel", base_get)
        launch(reference)
    launch(actual)
    for name in fields:
        np.testing.assert_array_equal(cp.asnumpy(reference[name]).view(np.uint32),
                                      cp.asnumpy(actual[name]).view(np.uint32),
                                      err_msg=name)
