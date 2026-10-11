"""Full warm and cold GPU replay against the shared host authority."""
import numpy as np
import pytest
cp = pytest.importorskip("cupy")
from tools.fast_sbm_host import host as H, replay as RP
from tools.fast_sbm_host.device import Device
from tools.fast_sbm_wrf480_oracle import fixtures as fx

@pytest.mark.parametrize("build", ("o32", "o64"))
@pytest.mark.parametrize("family", ("kc1", "kc2", "kc2g", "kc2edge"))
@pytest.mark.parametrize("parallel", (False, True))
def test_device_full_calls_match_host(family, build, parallel):
    hail = 0 if family == "kc2g" else 1
    flags = H.flags_corrected(hail) if parallel else H.flags_wrf(hail)
    host = RP.host_for(build, flags)
    gpu = Device(host)
    case = RP.Case(family, build)
    count = 0
    for step, pre, post in fx.iter_replay(f"{family}_replay_{build}"):
        state = case.state(pre.fields)
        want = {name: x.copy(order="F") for name, x in state.items()}
        hs, _ = host.call(step, 12.0, 1000.0, 1000.0, case.tile.dims(), want)
        gs, _ = gpu.call(step, 12.0, 1000.0, 1000.0, case.tile.dims(), state, parallel=parallel)
        assert hs == 0
        assert np.all(gs == 0), (family, build, step, gs)
        for name in H.STATE_FIELDS:
            a, b = state[name], want[name]
            word = np.uint32 if a.dtype == np.float32 else np.uint64
            assert np.array_equal(a.ravel(order="F").view(word), b.ravel(order="F").view(word)), (family, build, parallel, step, name, np.max(np.abs(a-b)))
        count += 1
    assert count == {"kc1":16, "kc2":7, "kc2g":3, "kc2edge":3}[family]


def test_parallel_refuses_cross_column_carries():
    gpu = Device(RP.host_for("o32", H.flags_wrf()))
    with pytest.raises(ValueError, match="cross-column carries"):
        gpu.call(1, 12., 1000., 1000., [0]*18, {}, parallel=True)
