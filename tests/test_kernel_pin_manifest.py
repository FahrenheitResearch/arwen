"""Additive scheme source pins are data; changes to pinned bytes still fail."""
import hashlib
import json

import pytest

import test_kernel_source_freeze_per_module as freeze


def _manifest(tmp_path, **groups):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"schema": "gpuwm-kernel-source-pins-v1",
                               "modules": {}, "headers": {},
                               "composed_headers": {}, **groups}))
    return path


def test_arbitrary_new_module_and_header_use_manifest_pins(tmp_path, monkeypatch):
    raw = b"// new scheme source\n"
    (tmp_path / "future.cu").write_bytes(raw)
    (tmp_path / "future.cuh").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    manifest = _manifest(tmp_path, modules={"future": sha}, headers={"future.cuh": sha})
    pins = freeze._extension_pins(manifest)
    monkeypatch.setattr(freeze, "KERNELS", tmp_path)
    monkeypatch.setattr(freeze, "PINNED", pins["modules"])
    monkeypatch.setattr(freeze, "PINNED_HEADERS", pins["headers"])
    freeze.test_the_kernel_source_is_byte_identical_to_its_pin("future")
    freeze.test_prepended_header_is_byte_identical_to_its_pin("future.cuh")
    for name, check, argument in (
        ("future.cu", freeze.test_the_kernel_source_is_byte_identical_to_its_pin, "future"),
        ("future.cuh", freeze.test_prepended_header_is_byte_identical_to_its_pin, "future.cuh"),
    ):
        (tmp_path / name).write_bytes(raw + b"// changed identity\n")
        with pytest.raises(AssertionError, match="changed"):
            check(argument)


@pytest.mark.parametrize("group,name,digest", [
    ("modules", "../escape", "0" * 64),
    ("headers", "future.cuh", "not-a-digest"),
    ("headers", "future.py", "0" * 64),
])
def test_invalid_extension_pins_refuse(tmp_path, group, name, digest):
    path = _manifest(tmp_path, **{group: {name: digest}})
    with pytest.raises(ValueError):
        freeze._extension_pins(path)


def test_extension_cannot_silently_override_an_existing_pin(tmp_path):
    path = _manifest(tmp_path, modules={"existing": "0" * 64})
    with pytest.raises(ValueError, match="existing pin"):
        freeze._merge_extension_pins({"existing": "1" * 64},
                                     freeze._extension_pins(path)["modules"])
