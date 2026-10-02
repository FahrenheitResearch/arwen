"""Bit checks for classic Thompson network geometry and register bounds."""
from functools import lru_cache

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@lru_cache(None)
def _unbounded_reference():
    import cupy as cp
    from gpuwm.core.kernels import module_source
    source = module_source("thompson")
    bounds = "__launch_bounds__(256, 2)"
    assert source.count(bounds) == 1
    return cp.RawModule(code=source.replace(bounds, ""),
                        options=("-std=c++17",))


@pytest.mark.parametrize("network", ["cold", "warm"])
@pytest.mark.parametrize("n", [1093, 131071])
def test_network_blocks_and_register_bound_match_base_bits(monkeypatch, network, n):
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
    fields = {name: rng.uniform(1e-6, 2e-3, n).astype(np.float32)
              for name in ("qi", "qs", "qg", "qr", "qc")}
    fields.update(ni=rng.uniform(1, 2e5, n).astype(np.float32),
                  nr=rng.uniform(1, 2e5, n).astype(np.float32),
                  temperature=rng.uniform(230, 290, n).astype(np.float32),
                  pressure=rng.uniform(15000, 100000, n).astype(np.float32),
                  qv=rng.uniform(1e-5, 0.018, n).astype(np.float32),
                  shadow=rng.uniform(1, 2e5, n).astype(np.float32),
                  boost=np.zeros(n, np.float32),
                  graupel_marker=np.zeros(n, np.float32),
                  snow_marker=np.zeros(n, np.float32))
    if network == "warm":
        fields["temperature"][:] = rng.uniform(272, 305, n).astype(np.float32)
    fields["graupel_marker"][:] = fields["temperature"] >= np.float32(273.15)
    for name in ("qi", "qs", "qg", "qr", "qc"):
        fields[name][::3] = 0.0
        fields[name][1::7] = 1e-12
        fields[name][2::19] = np.float32(-0.0)
    reference = {name: cp.asarray(a) for name, a in fields.items()}
    actual = {name: cp.asarray(a) for name, a in fields.items()}

    def launch(f):
        if network == "cold":
            t.launch_frozen_vapor_network_from_owner(
                *(f[name] for name in ("qi", "ni", "qs", "qg", "qr", "nr",
                                      "temperature", "pressure", "qv")),
                owner, 6.0, qc=f["qc"], graupel_number_shadow=f["shadow"],
                snow_velocity_boost=f["boost"])
        else:
            t.launch_warm_frozen_source_network_from_owner(
                *(f[name] for name in ("qc", "qr", "nr", "qs", "qg", "shadow",
                                      "graupel_marker", "snow_marker",
                                      "temperature", "pressure", "qv")),
                owner, 6.0)

    def base_get(unit, symbol):
        # The former unconstrained kernel and launch geometry.
        assert unit == "thompson"
        kernel = _unbounded_reference().get_function(symbol)

        def base_call(grid, block, args):
            threads = 128 if network == "cold" else 256
            return kernel(((int(args[-1]) + threads - 1) // threads,),
                          (threads,), args)

        return base_call

    with monkeypatch.context() as context:
        context.setattr(t, "get_kernel", base_get)
        launch(reference)
    launched_blocks = []
    production_get = t.get_kernel

    def record_get(unit, symbol):
        kernel = production_get(unit, symbol)

        def record_call(grid, block, args):
            launched_blocks.append(block[0])
            return kernel(grid, block, args)

        return record_call

    with monkeypatch.context() as context:
        context.setattr(t, "get_kernel", record_get)
        launch(actual)
    major = int(cp.cuda.runtime.getDeviceProperties(actual["qc"].device.id)["major"])
    expected_threads = 256 if network == "warm" and major in (9, 10) else 64
    assert launched_blocks == [expected_threads]
    for name in fields:
        np.testing.assert_array_equal(cp.asnumpy(reference[name]).view(np.uint32),
                                      cp.asnumpy(actual[name]).view(np.uint32),
                                      err_msg=name)
