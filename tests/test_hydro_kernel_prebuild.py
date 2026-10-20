"""The hydro door's cold kernel compile (hydro-world VERIFY2 calibration, 2026-10-18).

A cold door run spent about 56 s at one core compiling the Noah-OWP physics
module while the card idled. The door module now omits the raw ABI kernel
nothing in the door launches, and the compile starts on its own thread before
any stage prepares. Measured on an RTX 5070 Ti: longest one-core,
card-idle stretch 22.8 s cold (was ~56 s), 0 kernels compiled on the second
run, history/streamflow.csv byte-identical to the run before the change.
"""
from pathlib import Path

from gpuwm.hydro import assembly
from gpuwm.hydro.noahowp import factory, runtime

PHYSICS = Path(runtime.__file__).with_name("physics.cu")


def test_door_module_omits_only_the_raw_abi_kernel():
    for numerics in ("fast", "strict"):
        source = runtime.physics_source(numerics)
        assert 'void noahowp_step(' not in source
        for kernel in ("noahowp_step_monitored", "noahowp_export_bmi", "noahowp_copy_forcing",
                       "noahowp_step_cell"):
            assert f"void {kernel}(" in source
    # tools/noahowp_replay compiles the whole file and still finds the raw kernel there.
    assert 'extern "C" __global__ void noahowp_step(' in PHYSICS.read_text()


def test_noah_row_starts_its_compile_before_any_stage_prepares(monkeypatch):
    started = []
    monkeypatch.setattr(runtime, "prebuild", lambda numerics, device_id: started.append((numerics, device_id)))
    rows = [{"name": "network", "factory": "gpuwm.hydro.troute_mc:prepare"},
            {"name": "noah", "factory": "gpuwm.hydro.noahowp.factory:prepare_slab", "options": {}}]
    assembly.prebuild_kernels(rows, settings={"numerics": "fast"}, device_id=0)
    assert started == [("fast", 0)]
    assert callable(factory.prepare_slab_prebuild)
