"""Allocation and product-schema controls that do not initialize CUDA."""

import numpy as np
import pytest

from gpuwm.ensemble.batch_products import (FieldProducts, ThresholdCondition,
    compound_memory_plan, product_memory_plan, rain_memory_plan)


def test_product_ledger_prices_each_output_and_paintball_word():
    requests = (FieldProducts("refl", "dBZ", (20, 40), paintball=True, spaghetti=True),)
    plan = product_memory_plan(requests, {"refl": (3, 5)}, members=65, reserved_bytes=1024)
    rows = {r["name"]: r for r in plan.inventory(65)}
    assert rows["refl:mean"]["shape"] == (3, 5)
    assert rows["refl:probability"]["shape"] == (2, 3, 5)
    assert rows["refl:paintball"]["shape"] == (2, 2, 3, 5)
    assert rows["refl:spaghetti"]["shape"] == (2, 65, 2, 4)
    assert sum(r["payload_bytes"] for r in rows.values()) == 300 + 8 + 120 + 480 + 1040
    assert plan.required_bytes(65) == 1024 + sum(r["allocated_bytes"] for r in rows.values())
    with pytest.raises(MemoryError, match="exhaust device memory"):
        plan.admit(65, available_bytes=plan.required_bytes(65) - 1)


@pytest.mark.parametrize("kwargs,match", [
    ({"field": "bad-name", "units": "K"}, "identifier"),
    ({"field": "t2", "units": ""}, "units"),
    ({"field": "t2", "units": "K", "comparison": "gte"}, "comparison"),
    ({"field": "t2", "units": "K", "thresholds": (np.inf,)}, "finite"),
    ({"field": "t2", "units": "K", "thresholds": (1, 1 + 1e-10)}, "duplicate"),
    ({"field": "t2", "units": "K", "paintball": True}, "thresholds"),
])
def test_threshold_identity_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FieldProducts(**kwargs)


def test_contour_grid_and_roster_validation():
    request = FieldProducts("t2", "K", (300,), spaghetti=True)
    with pytest.raises(ValueError, match="two-dimensional"):
        product_memory_plan((request,), {"t2": (4, 3, 5)}, members=10)
    with pytest.raises(ValueError, match="duplicate"):
        product_memory_plan((request, request), {"t2": (3, 5)}, members=10)
    with pytest.raises(ValueError, match="input shape"):
        product_memory_plan((request,), {}, members=10)
    with pytest.raises(TypeError, match="integer"):
        product_memory_plan((request,), {"t2": (3, 5)}, members=True)


def test_rain_ledger_is_bounded_by_longest_window():
    plan = rain_memory_plan((3, 5), members=10, output_interval_ticks=60,
                            window_ticks=(3600, 10800, 21600))
    rows = {r["name"]: r for r in plan.inventory(10)}
    assert rows["rain:history"]["shape"] == (10, 361, 3, 5)
    assert rows["rain:history"]["payload_bytes"] == 10 * 361 * 3 * 5 * 4
    assert rows["rain:window:3600"]["shape"] == (10, 3, 5)
    assert rows["rain:invalid"]["shape"] == (1,)
    with pytest.raises(ValueError, match="interpolation"):
        rain_memory_plan((3, 5), members=10, output_interval_ticks=61, window_ticks=(3600,))


def test_product_source_has_no_literal_float_division():
    from gpuwm.ensemble.batch_products import _SOURCE, _RAIN_SOURCE, _COMPOUND_SOURCE
    import re
    assert not re.findall(r"/\s*(?:\d+\.\d*|\d+[eE][+-]?\d+)[fFlL]?", _SOURCE + _RAIN_SOURCE + _COMPOUND_SOURCE)


def test_compound_event_ledger_includes_argument_tables():
    conditions = (ThresholdCondition("wind", "m s-1", 15),
                  ThresholdCondition("humidity", "%", 20, "le"))
    plan = compound_memory_plan((3, 5), conditions)
    rows = {r["name"]: r for r in plan.inventory(10)}
    assert rows["compound:event"]["shape"] == (10, 3, 5)
    assert rows["compound:pointers"]["dtype"] == np.dtype("uint64").str
    assert rows["compound:thresholds"]["payload_bytes"] == 8
