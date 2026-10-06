"""The native merge accepts field buffers without a second full field set."""
import ctypes
import json

import numpy as np
import pytest

from gpuwm.static import highres, rust_bridge


def test_registration_preserves_strided_and_integer_fields_without_concatenation(monkeypatch):
    library = rust_bridge.load()
    assert hasattr(library, "gpuwm_static_highres_fieldset_new_ptrs")
    fields = {
        "strided": np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2],
        "categories": np.arange(24, dtype=np.int16).reshape(2, 3, 4),
    }

    def refuse_concatenation(*args, **kwargs):
        raise AssertionError("registration allocated a concatenated field set")

    monkeypatch.setattr(np, "concatenate", refuse_concatenation)
    handle = highres._fieldset_new(rust_bridge, fields)
    try:
        actual = rust_bridge.fieldset_to_dict(handle)
        for name, field in fields.items():
            assert actual[name].tobytes() == np.asarray(field, dtype=np.float64).tobytes()
    finally:
        rust_bridge.fieldset_free(handle)


@pytest.mark.parametrize("fields,count,message", [
    ([{"name": "x", "planes": 1, "ny": 2, "nx": 2}], 0, "pointer count"),
    ([{"name": "x", "planes": 1, "ny": 2, "nx": 2}], 1, "data pointer"),
    ([{"name": "x", "planes": 2**63, "ny": 2, "nx": 2}], 1, "overflow"),
])
def test_pointer_registration_rejects_invalid_buffers(fields, count, message):
    library = rust_bridge.load()
    # The production registration binds this additive entry point.
    handle = highres._fieldset_new(rust_bridge, {})
    rust_bridge.fieldset_free(handle)
    spec = json.dumps({"fields": fields}).encode()
    buf = (ctypes.c_uint8 * len(spec)).from_buffer_copy(spec)
    pointers = (ctypes.POINTER(ctypes.c_double) * count)()
    out = ctypes.c_uint64(0)
    code = library.gpuwm_static_highres_fieldset_new_ptrs(
        buf, len(spec), pointers, count, ctypes.byref(out))
    assert code != 0
    assert message in rust_bridge.last_error(library)
    assert out.value == 0


def test_failed_override_registration_releases_baseline_handle(monkeypatch):
    handles = []
    calls = 0

    def register(bridge, fields):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("invalid override")
        return 123

    monkeypatch.setattr(highres, "_fieldset_new", register)
    monkeypatch.setattr(rust_bridge, "fieldset_free", handles.append)
    with pytest.raises(ValueError, match="invalid override"):
        highres._merge_via_rust(rust_bridge, {}, {}, mode="all")
    assert handles == [123]
