"""CPU pins for the pre-registered N2 peer-state comparator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gpuwm.verify.state_equiv import fp32_state_equiv_report


def _advance(value: np.float32, ulps: int) -> np.float32:
    """Construct an adjacent positive FP32 pattern by integer bits."""
    bits = np.asarray([value], dtype=np.float32).view(np.uint32)
    bits[0] = np.uint32(int(bits[0]) + ulps)
    return bits.view(np.float32)[0]


def _report(offsets):
    reference = np.ones(len(offsets), dtype=np.float32)
    candidate = np.asarray([_advance(np.float32(1.0), n) for n in offsets],
                           dtype=np.float32)
    return fp32_state_equiv_report(
        SimpleNamespace(x=candidate), SimpleNamespace(x=reference), ("x",))


def test_fp32_state_equiv_pins_signed_key_and_nearest_rank_p99():
    report = _report([0] * 99 + [2])
    assert report["pass"]
    assert report["fields"]["x"] == {
        "count": 100,
        "max_ulp": 2,
        "p99_ulp": 0,
        "mean_signed_ulp": 0.02,
        "pass": True,
        "reason": None,
    }
    # Signed zero encodings are adjacent monotone bit-pattern keys.
    zeros = fp32_state_equiv_report(
        {"x": np.array([0.0], np.float32)},
        {"x": np.array([-0.0], np.float32)}, ("x",))
    assert zeros["aggregate"]["mean_signed_ulp"] == 1.0
    negative = fp32_state_equiv_report(
        {"x": np.array([0xBF7FFFFF], np.uint32).view(np.float32)},
        {"x": np.array([0xBF800000], np.uint32).view(np.float32)}, ("x",))
    assert negative["aggregate"]["mean_signed_ulp"] == 1.0


def test_fp32_state_equiv_pins_each_guard_independently():
    max_guard = _report([0] * 100 + [9])
    assert max_guard["aggregate"]["max_ulp"] == 9
    assert max_guard["aggregate"]["p99_ulp"] == 0
    assert not max_guard["pass"]

    p99_guard = _report([0] * 98 + [3, 3])
    assert p99_guard["aggregate"]["max_ulp"] == 3
    assert p99_guard["aggregate"]["p99_ulp"] == 3
    assert abs(p99_guard["aggregate"]["mean_signed_ulp"]) <= 0.25
    assert not p99_guard["pass"]

    mean_guard = _report([1] * 26 + [-1] * 74)
    assert mean_guard["aggregate"]["max_ulp"] == 1
    assert mean_guard["aggregate"]["p99_ulp"] == 1
    assert mean_guard["aggregate"]["mean_signed_ulp"] == -0.48
    assert not mean_guard["pass"]


def test_fp32_state_equiv_empty_and_nonfinite_are_blocking_failures():
    empty = fp32_state_equiv_report(
        {"x": np.empty(0, np.float32)}, {"x": np.empty(0, np.float32)},
        ("x",))
    assert not empty["pass"]
    assert empty["fields"]["x"]["reason"] == "empty comparison"

    for bad in (np.nan, np.inf, -np.inf):
        report = fp32_state_equiv_report(
            {"x": np.array([bad], np.float32)},
            {"x": np.array([0.0], np.float32)}, ("x",))
        assert not report["pass"]
        assert report["fields"]["x"]["reason"] == "non-finite value"


def test_fp32_state_equiv_aggregate_spans_every_named_field():
    candidate = {"a": np.ones(3, np.float32),
                 "b": np.ones((2, 2), np.float32)}
    reference = {name: value.copy() for name, value in candidate.items()}
    report = fp32_state_equiv_report(candidate, reference, ("a", "b"))
    assert report["pass"]
    assert report["aggregate"]["count"] == 7
    assert set(report["fields"]) == {"a", "b"}


def test_fp32_state_equiv_verdict_uses_registered_all_element_reduction():
    candidate = {
        "small": np.asarray(
            [_advance(np.float32(1.0), 3)] * 2, dtype=np.float32),
        "large": np.ones(198, dtype=np.float32),
    }
    reference = {name: np.ones_like(value)
                 for name, value in candidate.items()}
    report = fp32_state_equiv_report(
        candidate, reference, ("small", "large"))
    assert not report["fields"]["small"]["pass"]
    assert report["aggregate"]["p99_ulp"] == 0
    assert report["aggregate"]["mean_signed_ulp"] == 0.03
    assert report["pass"]
