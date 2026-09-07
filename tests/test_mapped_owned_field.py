"""Stream allocation transfer preserves canonical checks and public ownership."""
import gc
import weakref

import numpy as np
import pytest

from gpuwm.mapped_source import CanonicalField, _array_sha256
from gpuwm.mapped_engine_bridge import FrameSet


def metadata(values):
    return dict(name="custom_atmospheric_field", units="1", axes=("vertical", "y", "x"),
                location="mass", staggering="none", values=values,
                missing_count=int(np.isnan(values).sum()), source_references=("custom:input",))


def reader(tmp_path, values):
    stream = tmp_path / "frames.f64"
    stream.write_bytes(values.tobytes())
    frames = object.__new__(FrameSet)
    frames._stream = stream
    frames._declared_bytes = values.nbytes
    document = dict(name="custom_atmospheric_field", offset=0, length=values.nbytes,
                    shape=list(values.shape), dtype="<f8", sha256=_array_sha256(values),
                    units="1", axes=["vertical", "y", "x"], location="mass",
                    staggering="none", missing_count=int(np.isnan(values).sum()),
                    source_references=["custom:input"])
    return frames, document


def test_public_constructor_still_copies_borrowed_input():
    values = np.arange(48, dtype=np.float64).reshape(2,4,6)
    field = CanonicalField(**metadata(values))
    before = field.values.tobytes()
    values.fill(9)
    assert field.values.tobytes() == before
    assert not np.shares_memory(values, field.values)
    assert not field.values.flags.writeable


def test_stream_transfers_exact_allocation_and_retains_no_alias(tmp_path, monkeypatch):
    values = np.arange(48, dtype=np.float64).reshape(2,4,6)
    values[0,0,0] = -0.0
    values[0,0,1] = np.nan
    frames, document = reader(tmp_path, values)
    allocations = []
    original = np.empty
    def observe(*args, **kw):
        array = original(*args, **kw)
        allocations.append(weakref.ref(array))
        return array
    monkeypatch.setattr(np, "empty", observe)
    field = frames._read_field(0, document)
    assert len(allocations) == 1
    assert allocations[0]() is field.values
    assert field.values.tobytes() == values.tobytes()
    assert _array_sha256(field.values) == document["sha256"]
    assert field.axes == tuple(document["axes"])
    assert field.values.flags.owndata and not field.values.flags.writeable
    with pytest.raises(ValueError):
        field.values[0,0,0] = 123
    del frames
    gc.collect()
    assert field.values.tobytes() == values.tobytes()
    del field
    gc.collect()
    assert allocations[0]() is None


@pytest.mark.parametrize("mutation, message", [
    ("hash", "hashes to"), ("length", "declares.*bytes"),
    ("missing", "missing count"), ("axes", "rank"),
    ("infinity", "infinity"), ("short", "ends at byte")])
def test_reader_preserves_full_validation(tmp_path, mutation, message):
    values = np.arange(48, dtype=np.float64).reshape(2,4,6)
    if mutation == "infinity":
        values[0,0,0] = np.inf
    frames, document = reader(tmp_path, values)
    if mutation == "hash":
        document["sha256"] = "0" * 64
    elif mutation == "length":
        document["length"] -= 8
    elif mutation == "missing":
        document["missing_count"] = 1
    elif mutation == "axes":
        document["axes"] = ["y", "x"]
    elif mutation == "short":
        frames._stream.write_bytes(b"short")
    with pytest.raises(ValueError, match=message):
        frames._read_field(0, document)


def test_private_transfer_rejects_nonowning_or_wrong_dtype_buffers():
    values = np.ones((2,4,6), dtype=np.float64)
    for bad in (values.view(), values[...,::2], values.astype(np.float32)):
        with pytest.raises(ValueError, match="owning contiguous float64"):
            CanonicalField._take_owned_stream_array(**metadata(bad))
