"""Source-only archive pins cannot borrow an unrelated parent Git revision."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import regional_rain_score as scorer


def test_valid_archive_marker_does_not_call_ambient_git(tmp_path, monkeypatch):
    revision = "a"*40
    (tmp_path/".engine-export-sha").write_text(revision+"\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("archive scorer must not consult ambient Git")
    monkeypatch.setattr(scorer.subprocess,"run",forbidden)
    assert scorer.source_revision(tmp_path)==revision
    (tmp_path/".engine-export-sha").unlink()
    assert scorer.source_revision(tmp_path) is None


def test_malformed_archive_marker_is_refused(tmp_path, monkeypatch):
    (tmp_path/".engine-export-sha").write_text("not-a-source-revision\n")
    monkeypatch.setattr(scorer.subprocess,"run",lambda *a,**k:pytest.fail("malformed marker queried Git"))
    with pytest.raises(ValueError,match="40-hex"):
        scorer.source_revision(tmp_path)


def test_checkout_marker_must_match_its_own_git_head(tmp_path,monkeypatch):
    (tmp_path/".engine-export-sha").write_text("a"*40)
    (tmp_path/".git").mkdir()
    def git(command,**kwargs):
        return SimpleNamespace(stdout=str(tmp_path) if "--show-toplevel" in command else "b"*40)
    monkeypatch.setattr(scorer.subprocess,"run",git)
    with pytest.raises(ValueError,match="differs.*Git HEAD"):
        scorer.source_revision(tmp_path)
