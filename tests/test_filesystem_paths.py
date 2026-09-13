"""I/O expansion must not create a second identity for the same location."""
import os
import errno
from pathlib import Path

import pytest

from gpuwm.filesystem_paths import canonical_path, io_path


def test_io_and_identity_retain_one_actual_file(tmp_path):
    path = tmp_path / 'data'
    io_path(path).write_bytes(b'one file')
    assert canonical_path(path) == canonical_path(io_path(path)) == path.resolve()
    assert io_path(path).read_bytes() == b'one file'
    if os.name == 'nt':
        assert str(io_path(path)).startswith('\\\\?\\')
    else:
        assert io_path(Path('relative')) == Path('relative')


@pytest.mark.skipif(os.name != 'nt', reason='Windows extended drive/UNC spellings')
def test_unc_io_and_resolved_identity_share_one_spelling(monkeypatch):
    raw = Path(r'\\server\share\data')
    extended = Path(r'\\?\UNC\server\share\data')
    assert io_path(raw) == extended
    assert io_path(extended) == extended
    # No network server is involved: this controls normalization of the value
    # Windows resolution returns. Actual local-file resolution is tested above.
    monkeypatch.setattr(Path, 'resolve', lambda self: extended)
    assert canonical_path(raw) == canonical_path(extended) == raw


def test_resolution_failure_is_not_turned_into_a_different_location(monkeypatch):
    def fail(self):
        raise OSError('unresolved link')
    monkeypatch.setattr(Path, 'resolve', fail)
    with pytest.raises(OSError, match='unresolved link'):
        canonical_path('broken')


@pytest.mark.parametrize('code', [errno.ENOENT, errno.ENOSPC])
def test_replace_does_not_retry_or_hide_other_io_failures(tmp_path, monkeypatch, code):
    from gpuwm import filesystem_paths
    source, target = tmp_path / 'prepared', tmp_path / 'published'
    source.write_bytes(b'complete new bytes')
    target.write_bytes(b'prior bytes')
    failure = OSError(code, 'actual operation failed')
    def replace(*args):
        raise failure
    monkeypatch.setattr(filesystem_paths.os, 'replace', replace)
    monkeypatch.setattr(filesystem_paths.time, 'sleep', lambda _: pytest.fail('non-permission error was retried'))
    with pytest.raises(OSError) as caught:
        filesystem_paths.replace_file_with_retry(source, target)
    assert caught.value is failure
    assert source.read_bytes() == b'complete new bytes'
    assert target.read_bytes() == b'prior bytes'
