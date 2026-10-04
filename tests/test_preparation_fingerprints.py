"""Receipt bytes, masks and unusual extrema retain the existing contract."""
import hashlib
import struct

import numpy as np
import pytest

from gpuwm.ingest import preparation_fingerprints as native


def reference(value):
    array = np.ascontiguousarray(np.asarray(value))
    if not array.size:
        raise ValueError("cannot fingerprint an empty correspondence array")
    packed = np.packbits(np.ravel(array != 0.0), bitorder="little")
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "nonzero_mask_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
            "nonzero_count": int(np.count_nonzero(array)),
            "minimum": float(np.min(array)), "maximum": float(np.max(array))}


def assert_receipt(actual, expected):
    assert actual is not None
    for key in expected:
        if key in ("minimum", "maximum"):
            assert struct.pack("=d", actual[key]) == struct.pack("=d", expected[key]), key
        else:
            assert actual[key] == expected[key], key


@pytest.fixture
def entry():
    result = native._native_entry()
    if result is None:
        pytest.skip("CPU bridge predates native preparation fingerprints")
    return result


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.uint8, np.int16, np.uint16,
                                   np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("workers", [1, 8, 24])
def test_batch_matches_original_receipts(dtype, workers, entry):
    rng = np.random.default_rng(5828402)
    values = rng.integers(0, 100, (3, 271, 263)).astype(dtype)
    values.ravel()[::7] = 0
    if np.issubdtype(dtype, np.signedinteger):
        values.ravel()[::13] *= -1
        values.flat[:2] = (np.iinfo(dtype).min, np.iinfo(dtype).max)
    elif np.issubdtype(dtype, np.unsignedinteger):
        values.flat[1] = np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        values *= dtype(0.01)
        values.flat[:7] = [0, -0., -1., np.inf, -np.inf,
                           np.nextafter(dtype(0), dtype(1)), np.nextafter(dtype(0), dtype(-1))]
    other = np.ascontiguousarray(values.ravel()[::-1])
    actual = native.array_fingerprints((values, other), workers=workers)
    assert len(actual) == 2
    for got, wanted in zip(actual, (reference(values), reference(other))):
        assert_receipt(got, wanted)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 65535, 65536, 65537])
def test_mask_padding_and_block_boundaries(length, entry):
    values = np.zeros(length, np.float32)
    values[[0, length // 2, length - 1]] = [-1., np.nan, np.inf]
    with np.errstate(all="ignore"):
        assert_receipt(native.array_fingerprint(values), reference(values))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ["signed_zero", "payload_nan", "all_infinite"])
def test_unusual_extrema_keep_numpy_reduction_bits(dtype, case, entry):
    values = np.full(65539, dtype(1))
    if case == "signed_zero":
        values[:] = 0
        values[::3] = -0.
    elif case == "payload_nan":
        unsigned = np.uint32 if dtype is np.float32 else np.uint64
        bits = [0x7fc01234, 0xffc05678] if dtype is np.float32 else [0x7ff8000000001234, 0xfff8000000005678]
        values[17] = np.array(bits[0], dtype=unsigned).view(dtype)
        values[-8] = np.array(bits[1], dtype=unsigned).view(dtype)
    else:
        values[:] = np.inf
    with np.errstate(all="ignore"):
        assert_receipt(native.array_fingerprint(values), reference(values))


def test_readonly_unaligned_native_buffer_does_not_copy(entry, monkeypatch):
    storage = bytearray(4 * 65539 + 1)
    array = np.ndarray((65539,), np.float32, buffer=storage, offset=1)
    array[:] = np.arange(array.size, dtype=np.float32)
    array.flags.writeable = False
    expected = reference(array)
    addresses = []
    def record(jobs, count, workers, results):
        addresses.append(jobs[0].data)
        return entry(jobs, count, workers, results)
    monkeypatch.setattr(native, "_native_entry", lambda: record)
    assert_receipt(native.array_fingerprint(array), expected)
    assert addresses == [array.ctypes.data]


def test_noncanonical_boolean_storage_matches_numpy(entry):
    values = np.array([0, 1, 2, 255, 0, 128, 0, 5, 0], np.uint8).view(np.bool_)
    assert_receipt(native.array_fingerprint(values), reference(values))


@pytest.mark.parametrize("layout", ["strided", "float16", "big_endian", "scalar"])
def test_public_fingerprint_preserves_normalization_and_fallback(layout, entry):
    from gpuwm.ingest.real import array_correspondence_fingerprint
    values = np.arange(100, dtype=np.float64).reshape(10, 10)
    if layout == "strided":
        values = values[::-2, ::3]
    elif layout == "float16":
        values = values.astype(np.float16)
    elif layout == "big_endian":
        values = values.astype(">f8")
    else:
        values = np.array(-0.0)
    assert_receipt(array_correspondence_fingerprint(values), reference(values))


def test_existing_gpu_reductions_need_only_native_byte_digest(entry, monkeypatch):
    from gpuwm.ingest.real import _receipt_array_fingerprint, _ReceiptHostArray
    values = np.arange(19, dtype=np.float32).view(_ReceiptHostArray)
    original = reference(values)
    values.receipt_reductions = {key: value for key, value in original.items()
                                if key not in ("shape", "dtype", "sha256")}
    values.receipt_reductions.update(extrema_on_host=False, zero_extrema_on_host=False)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Python bytes/hash fallback was reached")
    monkeypatch.setattr(hashlib, "sha256", forbidden)
    assert_receipt(_receipt_array_fingerprint(values), original)


def test_empty_and_old_bridge_keep_original_behavior(monkeypatch):
    from gpuwm.ingest.real import array_correspondence_fingerprint
    monkeypatch.setattr(native, "_native_entry", lambda: None)
    values = np.array([0., -0., 1., np.nan], np.float64)
    assert_receipt(array_correspondence_fingerprint(values), reference(values))
    with pytest.raises(ValueError, match="empty correspondence array"):
        array_correspondence_fingerprint(np.empty(0, np.float32))
    assert native.array_fingerprints(()) == []


def test_cpu_real_receipts_batch_two_fields_and_match_original_hashes(entry, monkeypatch):
    from test_real_init import _analyzed_hrrr_real_init
    original = native.array_fingerprints
    calls = []
    def observed(arrays, **kwargs):
        values = tuple(arrays)
        calls.append((len(values), kwargs.get("workers")))
        return original(values, **kwargs)
    monkeypatch.setattr(native, "array_fingerprints", observed)
    actual, _cfg = _analyzed_hrrr_real_init(6, preprocess_backend="cpu",
        init_kwargs={"preprocess_workers": 8})
    assert any(count == 2 and workers == 2 for count, workers in calls)
    assert max(count for count, _workers in calls) <= 2
    monkeypatch.setattr(native, "_native_entry", lambda: None)
    expected, _cfg = _analyzed_hrrr_real_init(6, preprocess_backend="cpu",
        init_kwargs={"preprocess_workers": 8})
    assert actual.hydrometeor_initialization == expected.hydrometeor_initialization
    for name, value in vars(expected.state).items():
        if isinstance(value, np.ndarray):
            assert getattr(actual.state, name).tobytes() == value.tobytes(), name


@pytest.mark.parametrize("error", [FileNotFoundError, OSError])
def test_missing_optional_library_and_broken_loader_are_distinct(error, monkeypatch):
    from gpuwm.core import portable_math as pm
    def failed():
        raise error("test native loader failure")
    monkeypatch.setattr(pm, "_load", failed)
    if error is FileNotFoundError:
        assert native._native_entry() is None
    else:
        with pytest.raises(OSError, match="test native loader failure"):
            native._native_entry()
