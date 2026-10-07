"""Source-install preparation can keep its cache in an owned workspace."""
from pathlib import Path

from gpuwm import runtime_manifest as manifest


def test_source_cache_respects_absolute_xdg_and_roundtrips(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    path = manifest._content_cache_path(root)
    assert path.parent == cache / "gpuwm" / "source-content"
    entries = {"gpuwm/core/sfire.py": {"sha256": "0123", "size": 4}}
    manifest._store_content_cache(path, root, entries)
    assert manifest._load_content_cache(path, root) == entries


def test_relative_xdg_does_not_move_cache_into_working_directory(monkeypatch):
    root = Path("source").resolve()
    monkeypatch.setenv("XDG_CACHE_HOME", "relative-cache")
    path = manifest._content_cache_path(root)
    assert path.parent == Path.home() / ".gpuwm" / "cache" / "source-content"
