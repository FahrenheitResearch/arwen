"""Channel-only publication binding; CUDA instrumentation, no physics claim."""
import json
import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def prepared(monkeypatch):
    import cupy as cp
    from gpuwm.hydro.assembly import prepare

    module = ModuleType("gpuwm._channel_transport")

    class Router:
        def __init__(self, **kwargs):
            self.arrays = {
                "ids": cp.asarray([17, 900, 29], dtype=cp.int64),
                "published": cp.zeros((2, 3), dtype=cp.float64),
                "channel_ids": cp.asarray([29, 17], dtype=cp.int64),
                "channel_published": cp.zeros((2, 2), dtype=cp.float64),
            }
            self.arrays["channel_published"][1] = cp.asarray([-0.0, -0.0], dtype=cp.float64)
            self.restart_identity = {"instrumentation": "different channel and full-network axes"}
            self.kernel = cp.RawKernel('''extern "C" __global__ void publish(double* q, double* c) {
                int i=threadIdx.x;
                if(i<3) {q[i]+=1.;q[3+i]+=100.;}
                if(i<2) {c[i]+=2.;c[2+i]+=.25;}
            }''', "publish")
            self.kernel.compile()
            self.calls = 0

        def advance(self, dt_seconds, *, stream=None):
            self.kernel((1,), (3,), (self.arrays["published"], self.arrays["channel_published"]),
                        stream=stream)

        def reach_output(self):
            return {"reach_id": self.arrays["ids"], "discharge_m3_s": self.arrays["published"][0],
                    "depth_m": self.arrays["published"][1]}

        def channel_output(self):
            self.calls += 1
            return {"reach_id": self.arrays["channel_ids"],
                    "channel_depth_m": self.arrays["channel_published"][1]}

    class Resource:
        arrays = {}
        restart_identity = {"instrumentation": "prepare-only output must not be selected"}

        def channel_output(self):
            raise AssertionError("prepare-only resource is not the routing publisher")

    module.router = Router
    module.resource = lambda **kwargs: Resource()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    rows = [{"name": "resource", "factory": module.__name__ + ":resource", "phase": "prepare"},
            {"name": "route", "factory": module.__name__ + ":router",
             "parallelism": "one CUDA block x3 threads, transport only"}]

    def build():
        return prepare(settings={"stages": rows, "interval_seconds": 60}, state=None,
                       physics=None, device_id=cp.cuda.runtime.getDevice(), mode="standalone")
    return build


def test_channel_axis_stream_checkpoint_and_profile(prepared):
    import cupy as cp
    from gpuwm.nextgen_execution import bind_driver, measure

    driver = prepared()
    sample = driver.channel_output()
    assert sample is driver.channel_output()
    assert sample["reach_id"].get().tolist() == [29, 17]
    assert driver.reach_output()["reach_id"].get().tolist() == [17, 900, 29]
    assert driver.components["route"].calls == 1
    assert sample["channel_depth_m"].get().tobytes() == b"\0\0\0\0\0\0\0\x80" * 2
    pointers = {key: value.data.ptr for key, value in driver.arrays.items()}
    with cp.cuda.Stream(non_blocking=True) as stream:
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)
        borrowed = cp.empty_like(sample["channel_depth_m"])
        cp.copyto(borrowed, sample["channel_depth_m"])
        stream.synchronize()
    assert borrowed.get().tolist() == [.25, .25]
    assert driver.reach_output()["depth_m"].get().tolist() == [100., 100., 100.]
    metadata = driver.checkpoint_metadata()
    saved = {key: value.get() for key, value in driver.arrays.items()}
    fresh = prepared()
    fresh_pointers = {key: value.data.ptr for key, value in fresh.arrays.items()}
    fresh.validate_checkpoint(metadata, saved)
    for key, value in saved.items():
        fresh.arrays[key].set(value)
    fresh.restore_metadata(metadata)
    for key, value in saved.items():
        assert fresh.arrays[key].get().tobytes() == value.tobytes(), key
    assert fresh.channel_output()["channel_depth_m"].get().tobytes() == borrowed.get().tobytes()
    assert fresh.steps == 1
    assert fresh_pointers == {key: value.data.ptr for key, value in fresh.arrays.items()}
    assert pointers == {key: value.data.ptr for key, value in driver.arrays.items()}
    assert driver.components["route"].calls == 1
    def sequence():
        candidate = prepared()
        return bind_driver(candidate, parallelism={s.name: s.parallelism for s in candidate.prepared_stages},
                           metadata={"scope": "channel-output transport only"})
    receipt = measure(sequence, steps=16, warmup=1)
    print(json.dumps({"scope": "channel-output transport only", "bytes_restored": sum(v.nbytes for v in saved.values()),
                      "different_bytes": 0, "channel_ids": [29, 17], "network_ids": [17, 900, 29],
                      "timing": receipt}, sort_keys=True))


@pytest.mark.parametrize("defect", ["wrong_units", "unregistered", "strided", "wrong_dtype"])
def test_invalid_channel_binding_rejected(prepared, defect):
    import cupy as cp
    from gpuwm.hydro import ResidentHydroDriver

    driver = prepared()
    sample = dict(driver.channel_output())
    if defect == "wrong_units":
        sample["depth_m"] = sample.pop("channel_depth_m")
    elif defect == "unregistered":
        sample["channel_depth_m"] = cp.zeros(2, cp.float64)
    elif defect == "strided":
        sample["channel_depth_m"] = cp.zeros(4, cp.float64)[::2]
    else:
        sample["channel_depth_m"] = cp.zeros(2, cp.int32)
    with pytest.raises((KeyError, ValueError)):
        ResidentHydroDriver(device_id=driver.device_id, interval_seconds=60, identity={},
                            after=driver.after_stages, channel_output=sample)
