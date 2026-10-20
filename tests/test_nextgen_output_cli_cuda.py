"""Actual root CLI and Rust checkpoint transport; no hydrologic replay claim."""
import json
import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.gpu


def test_root_hydro_rust_checkpoint_continuation(tmp_path, monkeypatch, capsys):
    import cupy as cp
    from gpuwm.cli import main
    from gpuwm.hydro import output

    prepared = []
    factory = ModuleType("gpuwm._nextgen_transport")

    class Resource:
        def __init__(self):
            self.arrays = {"counter": cp.zeros(2, cp.int32)}
            self.restart_identity = {"instrumentation": "ordered hook resource"}

        def advance(self, dt_seconds, *, stream=None):
            raise AssertionError("resource must advance only through its owner hook")

    class Transport:
        def __init__(self, identity, resource):
            self.arrays = {
                "reach_id": cp.asarray([17, 29], dtype=cp.int64),
                "flow": cp.zeros(2, cp.float64),
                "signed_zero": cp.full(2, -0.0, cp.float64),
            }
            self.resource = resource
            self.restart_identity = {"instrumentation": "checkpoint transport", "identity": identity}
            self.kernel = cp.RawKernel('''extern "C" __global__ void update(double* q, int* n) {
                const int i=threadIdx.x;
                n[i]+=1;
                q[i]+=(double)(i+1)*0.125;
            }''', "update")
            self.kernel.compile()
            self.pointers = {key: value.data.ptr for key, value in self.arrays.items()}
            prepared.append(self)

        def advance(self, dt_seconds, *, stream=None):
            self.kernel((1,), (2,), (self.arrays["flow"], self.resource.arrays["counter"]), stream=stream)

        def reach_output(self):
            return {"reach_id": self.arrays["reach_id"], "discharge_m3_s": self.arrays["flow"]}

    factory.resource = lambda **kwargs: Resource()
    factory.prepare = lambda **kwargs: Transport(kwargs["options"]["identity"], kwargs["components"]["resource"])
    monkeypatch.setitem(sys.modules, factory.__name__, factory)

    config = tmp_path / "hydro.toml"
    text = '''[hydro]
enabled = true
interval_seconds = 60
[[hydro.stages]]
name = "resource"
factory = "gpuwm._nextgen_transport:resource"
phase = "prepare"
[[hydro.stages]]
name = "transport"
factory = "gpuwm._nextgen_transport:prepare"
parallelism = "one CUDA block, two threads"
[hydro.stages.options]
identity = "continuation-v1"
[[hydro.stages]]
name = "history"
factory = "gpuwm.hydro.output:prepare_history"
parallelism = "one block x256 reach threads plus one cursor thread"
[hydro.stages.options]
source = "transport"
capacity_steps = 24
cadence_steps = 1
origin = "2023-04-02 00:00:00"
types = ["wb", "wb"]
formats = ["analysis_csv"]
'''
    config.write_text(text)
    histories = []
    original = output.prepare

    def writer(**kwargs):
        histories.append(kwargs["driver"])
        return original(**kwargs)

    monkeypatch.setattr(output, "prepare", writer)

    def run(directory, seconds, restart=None, expected=0):
        arguments = ["hydro", "run", str(config), "--seconds", str(seconds),
                     "--outdir", str(directory)]
        if restart:
            arguments += ["--restart", str(restart)]
        assert main(arguments) == expected

    full, first, resumed = (tmp_path / name for name in ("full", "first", "resumed"))
    run(full, 1440)
    run(first, 720)
    run(resumed, 720, first / "restart.nc")
    a, b = histories[0], histories[2]
    assert a.steps == b.steps == 24
    assert a.channel_output() is None and b.channel_output() is None
    assert a.elapsed_seconds == b.elapsed_seconds == 1440
    assert [stage.name for stage in b.prepared_stages] == ["transport", "history"]
    assert "resource/counter" in b.arrays
    assert b.components["resource"].arrays["counter"].get().tolist() == [24, 24]
    assert set(a.arrays) == set(b.arrays)
    size = 0
    for key in a.arrays:
        assert a.arrays[key].get().tobytes() == b.arrays[key].get().tobytes(), key
        size += a.arrays[key].nbytes
    for component in prepared:
        assert component.pointers == {key: value.data.ptr for key, value in component.arrays.items()}
    products = sorted((full / "history").rglob("*.csv"))
    assert products
    for product in products:
        assert product.read_bytes() == (resumed / product.relative_to(full)).read_bytes()
    receipt = json.loads((resumed / "receipt.json").read_text())
    assert receipt["steps"] == 12
    assert receipt["boundary_synchronizations"] == 2
    assert receipt["driver_per_step_host_data_work"] is False
    assert receipt["stages"][1]["name"] == "history"

    # A real serialized checkpoint with a different identity must not mutate
    # a freshly prepared device state before the first continuation launch.
    config.write_text(text.replace('identity = "continuation-v1"', 'identity = "wrong"'))
    run(tmp_path / "wrong", 720, first / "restart.nc", expected=2)
    assert "composition, hydrofabric or parameters differ" in capsys.readouterr().err
    rejected = prepared[-1]
    assert not bool(cp.any(rejected.arrays["flow"]).item())
    assert not bool(cp.any(rejected.resource.arrays["counter"]).item())
    assert rejected.pointers == {key: value.data.ptr for key, value in rejected.arrays.items()}
    config.write_text(text)
    from gpuwm.hydro.runtime import prepare, read_settings
    from gpuwm.nextgen_execution import bind_driver, measure

    def sequence():
        driver = prepare(settings=read_settings(config), device_id=0)
        return bind_driver(driver, parallelism={stage.name: stage.parallelism
                                               for stage in driver.prepared_stages},
                           metadata={"scope": "driver/output transport instrumentation"})

    timing = measure(sequence, steps=16, warmup=1)
    print(json.dumps({"scope": "root CLI Rust checkpoint transport instrumentation",
                      "array_count": len(a.arrays), "compared_bytes": size,
                      "differing_bytes": 0, "continuation_steps": b.steps,
                      "full": json.loads((full / "receipt.json").read_text()),
                      "resumed": receipt, "timing": timing}, sort_keys=True))
