"""Built native host scans retain reports, bounds and auxiliary rounding."""
from dataclasses import asdict

import numpy as np
import pytest

from gpuwm.core import health


@pytest.fixture
def native():
    entry = health._native_health_entry()
    if entry is None:
        pytest.skip("built CPU bridge lacks native host health scan")
    return entry


def same_report(before, after):
    left, right = asdict(before), asdict(after)
    for item in (left, right):
        value = item["first_bad_value"]
        if value is not None:
            item["first_bad_value"] = np.asarray(value, dtype=np.float64).view(np.uint64).item()
    assert left == right


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, -0.0, 0.0,
                                    500.0, 500.00002, -500.00002))
def test_native_dense_reports_match_original_at_every_bound(native, dtype, value):
    fields = {"u": np.asarray([0., value, 1.], dtype=dtype),
              "p": np.asarray([100000., 0.], dtype=dtype)}
    same_report(health._validate_fields_numpy(fields, phase="control"),
                health.validate_fields_cpu(fields, phase="control"))


@pytest.mark.parametrize("left,right", ((np.float32, np.float32),
                                        (np.float32, np.float64),
                                        (np.float64, np.float32),
                                        (np.float64, np.float64)))
@pytest.mark.parametrize("mode", ("direct", "level"))
def test_auxiliary_addition_rounds_in_original_dtype(native, left, right, mode):
    values = np.asarray([[[16777216., -0.0]], [[-300., np.inf]]], dtype=left)
    auxiliary = (np.asarray([1., 300.], dtype=right) if mode == "level"
                 else np.asarray([[[1., 0.]], [[300., -np.inf]]], dtype=right))
    fields = [health.HealthField("thp", values, health.FieldRule("theta", 0., 16777216.),
                                  auxiliary, mode, 999)]
    with np.errstate(all="ignore"):
        same_report(health._validate_fields_numpy(fields), health.validate_fields_cpu(fields))


def test_first_native_or_legacy_field_and_all_classes_are_preserved(native):
    fields = [health.HealthField("first", np.asarray([1, 0], dtype=np.int32),
                                  health.FieldRule("surface", 0., strict_lower=True)),
              health.HealthField("second", np.asarray([np.inf], dtype=np.float64),
                                  health.FieldRule("wind")),
              health.HealthField("third", np.asarray([-1.], dtype=np.float32),
                                  health.FieldRule("moisture", 0.))]
    same_report(health._validate_fields_numpy(fields), health.validate_fields_cpu(fields))
    same_report(health._validate_fields_numpy(fields[::-1]), health.validate_fields_cpu(fields[::-1]))


@pytest.mark.parametrize("values", (np.asarray(-0.0, np.float32),
    np.empty((0, 2), np.float32), np.arange(12., dtype=np.float32).reshape(3, 4).T,
    np.arange(6., dtype=np.float64)[::-1], np.asarray([np.nan], dtype=np.complex64),
    np.asarray([1], dtype=">f4"), np.asarray([0, 1], dtype=np.uint64)))
def test_scalar_empty_and_uncommon_layouts_keep_original_interface(native, values):
    fields = [health.HealthField("a", values, health.FieldRule("generic", 0., strict_lower=True))]
    try:
        before = health._validate_fields_numpy(fields)
    except (TypeError, ValueError) as error:
        with pytest.raises(type(error)):
            health.validate_fields_cpu(fields)
    else:
        same_report(before, health.validate_fields_cpu(fields))


def test_unknown_or_invalid_auxiliary_keeps_original_error(native):
    values = np.ones((3, 2), np.float32)
    for mode, auxiliary in (("unknown", values), ("direct", values[:1]),
                             ("level", np.ones(2, np.float32))):
        fields = [health.HealthField("a", values, health.FieldRule("generic"), auxiliary, mode)]
        with pytest.raises(ValueError) as before:
            health._validate_fields_numpy(fields)
        with pytest.raises(ValueError, match="^" + __import__("re").escape(str(before.value)) + "$"):
            health.validate_fields_cpu(fields)


def test_missing_additive_abi_uses_original_checked_scan(monkeypatch):
    monkeypatch.setattr(health, "_native_health_entry", lambda: None)
    fields = {"u": np.asarray([501.], np.float32)}
    same_report(health._validate_fields_numpy(fields), health.validate_fields_cpu(fields))


def test_arrays_and_auxiliaries_are_not_mutated(native):
    values = np.asarray([[[1., -0.0]], [[300., 301.]]], np.float32)
    auxiliary = np.asarray([300., 300.], np.float32)
    before = values.tobytes(), auxiliary.tobytes()
    field = health.HealthField("thp", values, health.rule_for_field("thp"), auxiliary, "level")
    health.validate_fields_cpu([field])
    assert before == (values.tobytes(), auxiliary.tobytes())


def test_first_bad_index_after_chunk_boundary_is_exact(native):
    values = np.ones((2, (1 << 20) + 7), np.float32)
    values.flat[(1 << 20) + 1] = np.nan
    values.flat[-1] = np.inf
    fields = {"p": values, "u": np.asarray([np.nan], np.float32)}
    same_report(health._validate_fields_numpy(fields), health.validate_fields_cpu(fields))


def test_strict_numpy_error_policy_preserves_auxiliary_overflow_exception(native):
    values = np.asarray([np.finfo(np.float32).max], np.float32)
    field = health.HealthField("a", values, health.FieldRule("generic"), values, "direct")
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(FloatingPointError, match="overflow encountered in add"):
            health.validate_fields_cpu([field])


def test_default_warning_policy_preserves_bad_auxiliary_warning(native):
    values = np.asarray([np.finfo(np.float32).max], np.float32)
    field = health.HealthField("a", values, health.FieldRule("generic"), values, "direct")
    with np.errstate(over="warn", invalid="warn"):
        with pytest.warns(RuntimeWarning, match="overflow encountered in add"):
            report = health.validate_fields_cpu([field])
    assert not report.ok
    assert report.first_bad_field == "a"


def test_numpy_callback_policy_preserves_bad_auxiliary_callback(native):
    values = np.asarray([np.finfo(np.float32).max], np.float32)
    field = health.HealthField("a", values, health.FieldRule("generic"), values, "direct")
    observed = []
    previous = np.geterrcall()
    try:
        np.seterrcall(lambda error, flag: observed.append((error, flag)))
        with np.errstate(over="call"):
            report = health.validate_fields_cpu([field])
    finally:
        np.seterrcall(previous)
    assert not report.ok
    assert observed and observed[0][0] == "overflow"


def test_finite_float32_bound_cast_keeps_numpy_overflow_warning(native):
    field = health.HealthField("a", np.asarray([1.], np.float32), health.FieldRule("generic", upper=1e100))
    with np.errstate(over="warn"):
        with pytest.warns(RuntimeWarning, match="overflow encountered in cast"):
            assert health.validate_fields_cpu([field]).ok


@pytest.mark.parametrize("mode", ("call", "log", "print"))
def test_custom_numpy_error_policy_keeps_requested_legacy_route(native, monkeypatch, mode):
    marker = health.ValidationReport(True, 0, phase="legacy_policy")
    observed = []
    monkeypatch.setattr(health, "_validate_fields_numpy", lambda *a, **k: observed.append(k) or marker)
    with np.errstate(over=mode):
        assert health.validate_fields_cpu({}, phase="policy_control") is marker
    assert observed == [{"phase": "policy_control"}]
