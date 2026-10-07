"""A captured native one-ULP fuel recovery cannot consume negative fuel."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from tools.sfire_wrf471_oracle.fixture import words

ROOT = Path(__file__).resolve().parents[1] / "tools/sfire_coupled_ideal/fixtures/driver-burn"


def _load():
    receipt = json.loads((ROOT / "driver-burn-receipt.json").read_text())
    path = ROOT / "driver-burn-capture.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["fixture_sha256"]
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    for name, value in arrays.items():
        spec = receipt["arrays"][name]
        assert list(value.shape) == spec["shape"]
        assert str(value.dtype) == spec["dtype"]
        assert hashlib.sha256(value.tobytes()).hexdigest() == spec["sha256"]
    return arrays


def test_compiled_native_capture_authority():
    a = _load()
    selected = np.s_[1:-1, 1:-1]
    assert np.count_nonzero(a["raw_burnt"][selected] < 0) == 1
    assert a["raw_burnt"][237, 283] == np.float32(-1.4901161193847656e-8)
    assert a["fixed_fuel"][237, 283] == a["old"][237, 283]
    assert a["fixed_burnt"][237, 283] == 0


@requires_gpu
def test_corrected_driver_matches_compiled_captured_control():
    import cupy as cp
    from gpuwm.core import sfire_core as core
    a = _load()
    old, future = cp.asarray(a["old"]), cp.asarray(a["next"])
    burnt = cp.zeros_like(old)
    bounds = (1, 480, 1, 480)
    core._launch("sfire_driver_burn", core._size(bounds),
                 (old, future, burnt, *core._ints(482, *bounds)))
    for name, actual in (("fixed_fuel", old), ("fixed_burnt", burnt)):
        result = words(cp.asnumpy(actual)[1:-1, 1:-1], a[name][1:-1, 1:-1])
        print(json.dumps({"corrected_capture/" + name: result}))
        assert result["different_words"] == 0
        assert result["max_ulp"] == 0
        assert result["nonfinite_differences"] == 0
    assert bool(cp.all(burnt >= 0))


@requires_gpu
def test_driver_keeps_native_invalid_values_and_signed_zero():
    import cupy as cp
    from gpuwm.core import sfire_core as core
    a = _load()
    old = cp.zeros((3, 10), dtype=cp.float32)
    future = cp.zeros_like(old)
    old[1, 1:9] = cp.asarray(a["edge_old"])
    future[1, 1:9] = cp.asarray(a["edge_next"])
    burnt = cp.zeros_like(old)
    bounds = (1, 8, 1, 1)
    core._launch("sfire_driver_burn", core._size(bounds),
                 (old, future, burnt, *core._ints(10, *bounds)))
    for name, actual in (("edge_fixed_fuel", old), ("edge_fixed_burnt", burnt)):
        result = cp.asnumpy(actual)[1, 1:9]
        reference = a[name]
        assert np.array_equal(np.isnan(result), np.isnan(reference))
        defined = ~np.isnan(reference)
        assert result[defined].tobytes() == reference[defined].tobytes()
    # The source guard still sees invalid input. It is never made finite.
    assert bool(cp.isnan(old[1, 1]))
    assert bool(cp.isposinf(old[1, 2]))
    assert bool(cp.isneginf(old[1, 3]))


@requires_gpu
def test_correction_removes_negative_heat_and_smoke_source():
    import cupy as cp
    from gpuwm.core.sfire_phys import heat_fluxes
    from gpuwm.core.sfire_atm import add_fire_tracer_emissions
    a = _load()
    old = cp.asarray(a["raw_burnt"])
    fixed = cp.asarray(a["fixed_burnt"])
    fuel = cp.ones_like(fixed)
    moisture = cp.full_like(fixed, np.float32(.08))
    raw_h, raw_q = heat_fluxes(2.5, fuel, old, moisture)
    h, q = heat_fluxes(2.5, fuel, fixed, moisture)
    assert raw_h[237, 283].item() < 0 and raw_q[237, 283].item() < 0
    assert h[237, 283].item() == 0 and q[237, 283].item() == 0
    assert bool(cp.all(h >= 0)) and bool(cp.all(q >= 0))
    tracer = cp.zeros((2, 96, 96), dtype=cp.float32)
    rho = cp.ones_like(tracer)
    dz = cp.full_like(tracer, np.float32(10))
    add_fire_tracer_emissions(tracer, fixed[1:-1, 1:-1], fuel[1:-1, 1:-1],
                             rho=rho, dz8w=dz, sr_x=5, sr_y=5,
                             smoke_yield=.02, scheme=0, domain=(1, 94, 1, 94))
    assert bool(cp.isfinite(tracer).all()) and bool(cp.all(tracer >= 0))
    assert float(tracer.sum().item()) > 0
