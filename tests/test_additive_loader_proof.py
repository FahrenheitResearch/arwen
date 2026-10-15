"""Additive registration preserves old evidence; semantic mutations do not."""
import hashlib
from pathlib import Path
import pytest
from additive_loader_proof import require_additive_loader

BASE = b'_EXTRA_HEADERS: dict[str, tuple[str, ...]] = {"old": ("old.cuh",)}\ndef compile_unit(x):\n    return x + 1\n'
SHA = hashlib.sha256(BASE).hexdigest()

def check(source, baseline=BASE):
    return require_additive_loader(source, baseline, SHA, len(BASE))

def test_new_literal_registration_is_the_only_permitted_extension():
    new = BASE.replace(b'{"old":', b'{"new": ("new.cuh",), "old":')
    assert check(new) == ['new']
    assert check(BASE) == []

@pytest.mark.parametrize('source', [
    BASE.replace(b'x + 1', b'x + 2'),
    BASE.replace(b'old.cuh', b'other.cuh'),
    BASE.replace(b'{"old": ("old.cuh",)}', b'{}'),
    BASE.replace(b'{"old":', b'{"old": ("other.cuh",), "old":'),
    BASE.replace(b'{"old": ("old.cuh",)}', b'build_headers()'),
    BASE + b'compile_unit = lambda x: x + 2\n',
])
def test_existing_semantics_and_headers_stay_sealed(source):
    with pytest.raises(ValueError): check(source)

def test_changed_preimage_cannot_attest_current_source():
    with pytest.raises(ValueError, match='preimage'):
        check(BASE, BASE.replace(b'x + 1', b'x + 2'))
