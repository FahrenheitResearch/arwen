"""Byte controls against the original all-month NumPy ozone chain."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from gpuwm.core import portable_math
from gpuwm.ingest import wrf_ozone as ozone


@pytest.fixture
def native():
    entries = ozone._native_ozone_entries()
    if entries is None:
        pytest.skip("rebuilt CPU bridge with ozone additive ABI required")
    return entries


@pytest.fixture(scope="module")
def climo():
    return ozone.load_ozone_climatology()


def reference_latitude(monkeypatch, values, climo):
    with monkeypatch.context() as patch:
        patch.setattr(ozone, "_native_ozone_entries", lambda: None)
        return ozone.interp_ozone_to_latitudes(values, climo)


def reference_chain(monkeypatch, values, climo, julian):
    return ozone.ozn_time_int(None, julian, reference_latitude(monkeypatch, values, climo))


def same_bytes(actual, expected):
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def latitudes(climo):
    rng = np.random.default_rng(74189)
    return np.concatenate((climo.lat, np.nextafter(climo.lat, np.float32(-np.inf)),
                           np.nextafter(climo.lat, np.float32(np.inf)),
                           np.array([-120, -90, 0, 90, 120], dtype=np.float32),
                           rng.uniform(-110, 110, 73).astype(np.float32)))


@pytest.mark.parametrize("workers", [1, 2, 4, 17])
def test_full_latitude_native_matches_original_at_every_bracket(native, monkeypatch, climo, workers):
    values = latitudes(climo)
    expected = reference_latitude(monkeypatch, values, climo)
    with portable_math.worker_limit(workers):
        actual = ozone.interp_ozone_to_latitudes(values, climo)
    same_bytes(actual, expected)
    assert actual.flags.c_contiguous and actual.flags.writeable


DATES = [np.float32(v) for v in (-366, -1, 0, 1, 364, 365, 366, 730)]
for date in ozone.DATE_OZ:
    boundary = np.float32(date - 1)
    DATES.extend((np.nextafter(boundary, np.float32(-np.inf)), boundary,
                  np.nextafter(boundary, np.float32(np.inf))))


@pytest.mark.parametrize("julian", DATES)
def test_fused_chain_matches_original_at_every_month_boundary(native, monkeypatch, climo, julian):
    values = latitudes(climo)
    expected = reference_chain(monkeypatch, values, climo, julian)
    same_bytes(ozone.ozn_latitude_time_int(object(), julian, values, climo), expected)


@pytest.mark.parametrize("values", [np.float32(-9.76709), np.array([], np.float32),
                                    np.empty((2, 0), np.float32),
                                    np.array([[15, -15], [35, -35]], np.float32).T,
                                    np.linspace(-95, 95, 31, dtype=np.float32)[::-2]])
def test_scalar_empty_and_noncontiguous_shapes(native, monkeypatch, climo, values):
    expected_full = reference_latitude(monkeypatch, values, climo)
    same_bytes(ozone.interp_ozone_to_latitudes(values, climo), expected_full)
    expected = ozone.ozn_time_int(None, np.float32(274.875), expected_full)
    actual = ozone.ozn_latitude_time_int(None, np.float32(274.875), values, climo)
    same_bytes(actual, expected)
    assert actual.shape == np.shape(values) + (59,)


@pytest.mark.parametrize("workers", [1, 4, 17])
def test_signed_zero_subnormal_and_random_climatology_bytes(native, monkeypatch, climo, workers):
    rng = np.random.default_rng(15331)
    data = rng.uniform(-1e-5, 1e-5, climo.ozmix.shape).astype(np.float32)
    special = np.array([0, -0.0, 1e-44, -1e-44, 1e-38, -1e-38, 3e-38, -3e-38], np.float32)
    data[::2, :, :] = np.resize(special, data[::2].shape)
    unusual = climo._replace(ozmix=data)
    values = latitudes(climo)
    with np.errstate(all="ignore"):
        full = reference_latitude(monkeypatch, values, unusual)
        expected = ozone.ozn_time_int(None, np.float32(2.375), full)
        with portable_math.worker_limit(workers):
            same_bytes(ozone.interp_ozone_to_latitudes(values, unusual), full)
            same_bytes(ozone.ozn_latitude_time_int(None, np.float32(2.375), values, unusual), expected)


def test_native_fused_call_does_not_materialize_all_months(native, monkeypatch, climo):
    def forbidden(*args, **kwargs):
        raise AssertionError("all-month latitude intermediate was requested")
    monkeypatch.setattr(ozone, "interp_ozone_to_latitudes", forbidden)
    actual = ozone.ozn_latitude_time_int(None, np.float32(274.875), climo.lat, climo)
    assert actual.shape == (64, 59)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_xlat_error_precedes_bad_calendar_and_missing_climatology(native, climo, value):
    values = np.array([value], np.float32)
    with pytest.raises(ValueError, match="XLAT contains non-finite values"):
        ozone.ozn_latitude_time_int(None, object(), values, climo)
    with pytest.raises(ValueError, match="XLAT contains non-finite values"):
        ozone.interp_ozone_to_latitudes(values, SimpleNamespace())


@pytest.mark.parametrize("mutation", ["latitude_dtype", "ozone_dtype", "unsorted_latitude", "nonfinite_ozone"])
def test_unusual_external_climatologies_keep_original_bytes(native, monkeypatch, climo, mutation):
    latitude, data = climo.lat.copy(), climo.ozmix.copy()
    if mutation == "latitude_dtype":
        latitude = latitude.astype(np.float64)
    elif mutation == "ozone_dtype":
        data = data.astype(np.float64)
    elif mutation == "unsorted_latitude":
        latitude[20], latitude[21] = latitude[21], latitude[20]
    else:
        data[1, 31, 5] = np.float32(np.nan)
    unusual = climo._replace(lat=latitude, ozmix=data)
    values = np.array([-110, -33, 0, 110], np.float32)
    expected = reference_chain(monkeypatch, values, unusual, np.float32(149.5))
    same_bytes(ozone.ozn_latitude_time_int(None, np.float32(149.5), values, unusual), expected)


def test_invalid_climatology_shape_precedes_bad_calendar(native, monkeypatch, climo):
    invalid = climo._replace(ozmix=climo.ozmix[:-1])
    with pytest.raises(ValueError) as old:
        reference_chain(monkeypatch, np.array([0], np.float32), invalid, object())
    with pytest.raises(type(old.value), match="cannot reshape"):
        ozone.ozn_latitude_time_int(None, object(), np.array([0], np.float32), invalid)


def test_older_bridge_keeps_original_chain(monkeypatch, climo):
    values = np.array([-95, 0, 95], np.float32)
    expected = reference_chain(monkeypatch, values, climo, np.float32(14.75))
    monkeypatch.setattr(portable_math, "_load", lambda: SimpleNamespace())
    assert ozone._native_ozone_entries() is None
    same_bytes(ozone.ozn_latitude_time_int(None, np.float32(14.75), values, climo), expected)


def test_native_unexpected_failure_is_not_silently_reinterpreted(monkeypatch, climo):
    monkeypatch.setattr(ozone, "_native_ozone_entries", lambda: (lambda *args: 0,
                                                              lambda *args: 127,
                                                              lambda *args: 127))
    with pytest.raises(RuntimeError, match="native ozone interpolation failed"):
        ozone.ozn_latitude_time_int(None, 0, np.array([0], np.float32), climo)
    with pytest.raises(RuntimeError, match="native ozone interpolation failed"):
        ozone.interp_ozone_to_latitudes(np.array([0], np.float32), climo)


def test_fallback_time_input_validation_precedes_calendar(monkeypatch):
    monkeypatch.setattr(ozone, "_native_ozone_entries", lambda: None)
    with pytest.raises(ValueError, match="float32 with a trailing 12-month axis"):
        ozone.ozn_time_int(None, object(), np.zeros((1, 59, 12), np.float64))


def overflowing_climo(climo, *, month=None):
    data = climo.ozmix.copy()
    selected = slice(None) if month is None else month
    data[:, 0, selected] = -np.finfo(np.float32).max
    data[:, 1, selected] = np.finfo(np.float32).max
    return climo._replace(ozmix=data)


@pytest.mark.parametrize("month", [None, 0])
def test_default_warnings_and_unused_month_overflow_match_original(native, monkeypatch, climo, month):
    unusual = overflowing_climo(climo, month=month)
    values = climo.lat[:1]
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        expected = reference_chain(monkeypatch, values, unusual, np.float32(274.875))
    with warnings.catch_warnings(record=True) as current:
        warnings.simplefilter("always")
        actual = ozone.ozn_latitude_time_int(None, np.float32(274.875), values, unusual)
    same_bytes(actual, expected)
    assert old
    assert [(w.category, str(w.message)) for w in current] == [(w.category, str(w.message)) for w in old]


@pytest.mark.parametrize("category", ["over", "invalid", "divide", "under"])
def test_strict_error_policies_decline_native_before_library_load(native, monkeypatch, climo, category):
    def forbidden():
        raise AssertionError("strict error policy attempted native work")
    monkeypatch.setattr(portable_math, "_load", forbidden)
    unusual = overflowing_climo(climo)
    with np.errstate(all="ignore", **{category: "raise"}):
        assert ozone._native_ozone_entries() is None
        if category == "over":
            with pytest.raises(FloatingPointError, match="overflow"):
                ozone.ozn_latitude_time_int(None, np.float32(274.875), climo.lat[:1], unusual)


def test_underflow_raise_uses_original_exception(native, monkeypatch, climo):
    data = climo.ozmix.copy()
    data[:, 0, :] = np.float32(1e-44)
    data[:, 1, :] = np.float32(2e-44)
    unusual = climo._replace(ozmix=data)
    values = np.array([(climo.lat[0] + climo.lat[1]) * np.float32(.5)], np.float32)
    with np.errstate(all="ignore", under="raise"):
        with pytest.raises(FloatingPointError) as old:
            reference_chain(monkeypatch, values, unusual, np.float32(274.875))
        with pytest.raises(type(old.value)) as current:
            ozone.ozn_latitude_time_int(None, np.float32(274.875), values, unusual)
    assert str(current.value) == str(old.value)


@pytest.mark.parametrize("mode", ["call", "log"])
def test_callback_and_log_policies_retain_original_events(native, monkeypatch, climo, mode):
    unusual = overflowing_climo(climo)
    values = climo.lat[:1]
    previous = np.geterrcall()
    events = []
    class Logger:
        def write(self, message):
            events.append(message)
    callback = (lambda error, flags: events.append((error, flags))) if mode == "call" else Logger()
    try:
        np.seterrcall(callback)
        with np.errstate(all="ignore", over=mode, invalid=mode):
            expected = reference_chain(monkeypatch, values, unusual, np.float32(274.875))
            old = list(events)
            events.clear()
            actual = ozone.ozn_latitude_time_int(None, np.float32(274.875), values, unusual)
            assert ozone._native_ozone_entries() is None
        same_bytes(actual, expected)
        assert old and events == old
    finally:
        np.seterrcall(previous)


def test_print_policy_retains_original_messages(native, monkeypatch, climo, capfd):
    unusual = overflowing_climo(climo)
    with np.errstate(all="ignore", over="print", invalid="print"):
        expected = reference_chain(monkeypatch, climo.lat[:1], unusual, np.float32(274.875))
        old = capfd.readouterr()
        actual = ozone.ozn_latitude_time_int(None, np.float32(274.875), climo.lat[:1], unusual)
        current = capfd.readouterr()
    same_bytes(actual, expected)
    assert old.err and current == old
