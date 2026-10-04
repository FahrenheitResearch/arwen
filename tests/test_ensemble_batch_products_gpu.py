"""GPU product arithmetic, roster isolation, input identity and output gates."""

import math

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ensemble.batch_products import (FieldProducts, PreparedCompoundField,
    RainWindowHistory, ThresholdCondition, prepare_product_frame, write_product_frame)

pytestmark = [pytest.mark.gpu, requires_gpu]


def _statistics(values):
    count, ny, nx = values.shape
    mean = np.empty((ny, nx), np.float32)
    spread = np.empty_like(mean)
    for y in range(ny):
        for x in range(nx):
            words = [float(values[m, y, x]) for m in range(count)]
            average = sum(words) / count
            variance = sum((word - average) * (word - average) for word in words)
            mean[y, x] = average
            spread[y, x] = math.sqrt(variance / (count - 1)) if count > 1 else 0
    return mean, spread


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40, 65])
@pytest.mark.parametrize("relation", ["ge", "gt", "le", "lt"])
def test_products_match_independent_cpu_controls_and_preserve_inputs(members, relation):
    import cupy as cp
    rng = np.random.default_rng(1259 + members)
    values = rng.uniform(-8, 20, (members, 7, 9)).astype(np.float32)
    values[:, 0, 0] = 5.0
    fields = {"diagnostic": cp.asarray(values)}
    request = FieldProducts("diagnostic", "K", (0, 5, 10), comparison=relation,
                            paintball=True, spaghetti=True)
    prepared = prepare_product_frame(fields, (request,), available_bytes=1 << 28)
    before = fields["diagnostic"].get().tobytes()
    pool = cp.get_default_memory_pool()
    allocated = pool.used_bytes()
    out = prepared()
    cp.cuda.get_current_stream().synchronize()
    assert pool.used_bytes() == allocated
    assert fields["diagnostic"].get().tobytes() == before
    mean, spread = _statistics(values)
    for key, expected in (("mean", mean), ("spread", spread),
                          ("min", values.min(axis=0)), ("max", values.max(axis=0))):
        assert out[f"diagnostic:{key}"].get().tobytes() == expected.tobytes(), key
    np.testing.assert_array_equal(out["diagnostic:finite_count"].get(), members)
    compare = {"ge": np.greater_equal, "gt": np.greater,
               "le": np.less_equal, "lt": np.less}[relation]
    masks = np.asarray([compare(values, threshold) for threshold in request.thresholds])
    expected_prob = (masks.sum(axis=1, dtype=np.uint32).astype(np.float64) / members).astype(np.float32)
    assert out["diagnostic:probability"].get().tobytes() == expected_prob.tobytes()
    bits = np.zeros((3, (members + 63) // 64, 7, 9), np.uint64)
    for member in range(members):
        bits[:, member // 64] |= masks[:, member].astype(np.uint64) << np.uint64(member % 64)
    assert out["diagnostic:paintball"].get().tobytes() == bits.tobytes()
    crossings = (masks[:, :, :-1, :-1].astype(np.uint8)
                 | (masks[:, :, :-1, 1:].astype(np.uint8) << 1)
                 | (masks[:, :, 1:, 1:].astype(np.uint8) << 2)
                 | (masks[:, :, 1:, :-1].astype(np.uint8) << 3))
    assert out["diagnostic:spaghetti"].get().tobytes() == crossings.tobytes()
    assert prepared.receipt()["launches_per_frame"] == 2


def test_nonfinite_member_masks_aggregate_without_changing_roster():
    import cupy as cp
    values = np.ones((4, 3, 5), np.float32)
    values[1, 1, 1] = np.nan
    values[2, 1, 2] = np.inf
    values[3, 1, 3] = -np.inf
    frame = prepare_product_frame({"field": cp.asarray(values)},
                                  (FieldProducts("field", "1", (0.5,), paintball=True, spaghetti=True),),
                                  available_bytes=1 << 20)
    out = frame()
    for name in ("mean", "spread", "min", "max"):
        result = out[f"field:{name}"].get()
        assert np.isnan(result[1, 1:4]).all()
        assert np.isfinite(result[:, 0]).all()
    np.testing.assert_array_equal(out["field:finite_count"].get()[1, 1:4], 3)
    assert np.isnan(out["field:probability"].get()[0, 1, 1:4]).all()
    assert (out["field:spaghetti"].get() == 255).any()


def test_products_without_thresholds_use_only_statistics_allocations():
    import cupy as cp
    frame = prepare_product_frame({"field": cp.ones((1, 4, 3), cp.float32)},
                                  (FieldProducts("field", "K"),), available_bytes=1 << 20)
    out = frame()
    assert len(out) == 5
    np.testing.assert_array_equal(out["field:spread"].get(), 0)


def test_joint_fire_weather_conditions_reduce_complete_roster():
    import cupy as cp
    wind = np.arange(60, dtype=np.float32).reshape(4, 3, 5)
    humidity = np.full_like(wind, 15)
    humidity[2] = 25
    humidity[1, 1, 2] = np.nan
    inputs = {"wind": cp.asarray(wind), "humidity": cp.asarray(humidity)}
    before = {name: a.get().tobytes() for name, a in inputs.items()}
    event = PreparedCompoundField(inputs, (ThresholdCondition("wind", "m s-1", 20),
                                           ThresholdCondition("humidity", "%", 20, "le")),
                                  available_bytes=1 << 20)
    output = event()
    expected = ((wind >= 20) & (humidity <= 20)).astype(np.float32)
    expected[1, 1, 2] = np.nan
    np.testing.assert_array_equal(output.get(), expected)
    assert {name: a.get().tobytes() for name, a in inputs.items()} == before
    frame = prepare_product_frame({"fire_weather": output},
                                  (FieldProducts("fire_weather", "1", (0.5,), paintball=True),),
                                  available_bytes=1 << 20)
    products = frame()
    np.testing.assert_array_equal(products["fire_weather:probability"].get()[0, :, 0],
                                  expected[:, :, 0].mean(axis=0))
    replacement = cp.full_like(inputs["wind"], 25)
    event.rebind_fields({"wind": replacement, "humidity": inputs["humidity"]})
    rebound = event().get()
    assert (rebound[0] == 1).all()
    assert (rebound[2] == 0).all()
    assert np.isnan(rebound[1, 1, 2])
    assert {name: a.get().tobytes() for name, a in inputs.items()} == before


def test_rain_windows_match_retained_counters_through_ring_wrap():
    import cupy as cp
    history = RainWindowHistory(shape=(3, 5), members=10, output_interval_ticks=60,
                                window_ticks=(120, 180), available_bytes=1 << 20)
    rain = cp.zeros((10, 3, 5), cp.float32)
    for step in range(12):
        values = (np.arange(150).reshape(10, 3, 5) * np.float32(0.125) + np.float32(step * 0.5)).astype(np.float32)
        rain.set(values)
        before = rain.get().tobytes()
        history.capture(rain, step * 60)
        assert rain.get().tobytes() == before
        for window in (120, 180):
            if step * 60 < window:
                with pytest.raises(ValueError, match="complete retained history"):
                    history.window(window)
            else:
                expected = np.full((10, 3, 5), (window // 60) * 0.5, np.float32)
                assert history.window(window).get().tobytes() == expected.tobytes()
    with pytest.raises(ValueError, match="missing frame"):
        history.capture(rain, 900)
    rain.fill(0)
    with pytest.raises(ValueError, match="reset"):
        history.capture(rain, 720)


def test_rust_writer_roundtrip_contains_products_without_member_wrfouts(tmp_path):
    import cupy as cp
    from gpuwm import netcdf_bridge
    values = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    frame = prepare_product_frame({"field": cp.asarray(values)},
                                  (FieldProducts("field", "K", (20,), paintball=True, spaghetti=True),),
                                  available_bytes=1 << 20)
    frame()
    output = tmp_path / "ensemble-products.nc"
    coords = np.zeros((3, 5), np.float32)
    write_product_frame(output, frame, valid_time="2024-05-25_18:00:00", latitude=coords, longitude=coords)
    with netcdf_bridge.Dataset(output) as reader:
        variable = reader.variables["field_mean"]
        # The Rust reader's public numeric transport is float64, while
        # inventory reports the file's actual external type. Require NC_FLOAT
        # first, then prove the transport widens every stored float exactly.
        assert variable.dtype == np.dtype("float32")
        variable.set_auto_maskandscale(False)
        decoded = np.asarray(variable[:])
        assert decoded.dtype == np.dtype("float64")
        stored_words = decoded.astype(np.float32)
        assert stored_words.astype(np.float64).tobytes() == decoded.tobytes()
        assert stored_words.tobytes() == frame.outputs["field:mean"].get().tobytes()
        assert int(reader.ensemble_members) == 4
    assert list(tmp_path.iterdir()) == [output]
