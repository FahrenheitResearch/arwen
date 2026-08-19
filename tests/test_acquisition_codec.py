"""The acquisition codec set: bz2 objects decode once, by magic, generically."""

import bz2

import pytest

from gpuwm.ingest.codec import object_codec, staged_decoded_object


def test_plain_object_passes_through_untouched(tmp_path):
    plain = tmp_path / "plain.grib2"
    plain.write_bytes(b"GRIB1234")
    staging = tmp_path / "staging"
    staging.mkdir()
    assert object_codec(plain) is None
    assert staged_decoded_object(plain, staging) == plain
    assert list(staging.iterdir()) == []


def test_bz2_object_is_identified_by_magic_not_name(tmp_path):
    payload = b"GRIB-payload-bytes" * 64
    wrapped = tmp_path / "no-extension-at-all"
    wrapped.write_bytes(bz2.compress(payload))
    staging = tmp_path / "staging"
    staging.mkdir()
    assert object_codec(wrapped) == "bz2"
    staged = staged_decoded_object(wrapped, staging)
    assert staged != wrapped
    assert staged.parent == staging
    assert staged.read_bytes() == payload


def test_truncated_bz2_names_the_breakage(tmp_path):
    corrupt = tmp_path / "field.grib2.bz2"
    corrupt.write_bytes(bz2.compress(b"payload" * 200)[:20])
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(ValueError, match="bz2 magic but does not decompress"):
        staged_decoded_object(corrupt, staging)


def test_two_objects_stage_without_collision(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    outputs = []
    for index in range(2):
        wrapped = tmp_path / f"field-{index}.grib2.bz2"
        wrapped.write_bytes(bz2.compress(b"payload-%d" % index))
        outputs.append(staged_decoded_object(wrapped, staging))
    assert len({str(path) for path in outputs}) == 2
    assert outputs[0].read_bytes() == b"payload-0"
    assert outputs[1].read_bytes() == b"payload-1"
