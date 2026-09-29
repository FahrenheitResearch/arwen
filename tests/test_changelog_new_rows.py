"""tools/dev/changelog_new_rows.py lists the rows a condensed release section is missing.

The 2.8.0 section was condensed from integrate/2.8 at one commit; rows merged after it are folded in
at the cut from this tool's list.  A row it missed would ship absent from the release note, so the
cases here are the ways a row reaches the line: added under a group or an area, reworded, carried
on a continuation line, and written with the page's old name.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("changelog_new_rows", ROOT / "tools" / "dev" / "changelog_new_rows.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _tool()

BASE_LOG = """# Changelog

## 9.9.0 (unreleased)

New:

**The page**

- A page opens.
- A row that goes on
  over a second line.

Fixed:

- A fix that stays.
- A fix someone rewords.

## 9.8.0 (2026-01-01)

Fixed:

- An old fix.
"""

REF_LOG = BASE_LOG.replace(
    "- A fix someone rewords.\n",
    "- A fix someone reworded.\n- `gpuwm fetch` no longer fails; the storm wiki shows it (`/api/wiki/event/ID`).\n",
).replace("- A page opens.\n", "- A page opens.\n- A new page row.\n")


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=True)
    return done.stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@localhost")
    _git(root, "config", "user.name", "test")
    (root / "CHANGELOG.md").write_bytes(BASE_LOG.encode("utf-8"))
    _git(root, "add", "CHANGELOG.md")
    _git(root, "commit", "-q", "-m", "base")
    base = _git(root, "rev-parse", "HEAD")
    (root / "CHANGELOG.md").write_bytes(REF_LOG.encode("utf-8"))
    _git(root, "commit", "-q", "-am", "rows")
    return root, base


def test_rows_keep_their_group_their_area_and_their_continuation_lines():
    found = TOOL.rows(TOOL.section(BASE_LOG, "9.9.0"))
    assert [(r.group, r.area, r.text) for r in found] == [
        ("New", "The page", "- A page opens."),
        ("New", "The page", "- A row that goes on\n  over a second line."),
        ("Fixed", "", "- A fix that stays."),
        ("Fixed", "", "- A fix someone rewords."),
    ]
    assert TOOL.section(BASE_LOG, "9.7.0") is None


def test_new_and_reworded_rows_are_listed_under_their_headings_and_worded_as_the_weather_library(repo):
    root, base = repo
    found = TOOL.compare(root, base, "HEAD", "9.9.0")
    added = [(r["group"], r["area"], r["text"]) for r in found["added"]]
    assert added == [
        ("New", "The page", "- A new page row."),
        ("Fixed", "", "- A fix someone reworded."),
        ("Fixed", "", "- `gpuwm fetch` no longer fails; the Weather Library shows it (`/api/library/event/ID`)."),
    ]
    assert [r["text"] for r in found["removed"]] == ["- A fix someone rewords."]
    assert found["reworded_as_weather_library"]
    assert found["tree_section_words"] == len(TOOL.section(REF_LOG, "9.9.0").split())
    assert found["tree_section_words_with_added"] == found["tree_section_words"] + sum(
        r["words"] for r in found["added"])
    text = TOOL.report(found)
    assert "[New / The page]\n- A new page row." in text and "[Fixed]" in text


def test_raw_prints_a_row_as_written(repo):
    root, base = repo
    found = TOOL.compare(root, base, "HEAD", "9.9.0", raw=True)
    assert any("the storm wiki shows it (`/api/wiki/event/ID`)" in r["text"] for r in found["added"])


def test_the_same_commit_lists_nothing_and_a_missing_ref_or_section_exits_2(repo, capsys):
    root, _ = repo
    assert TOOL.main(["HEAD", "--ref", "HEAD", "--root", str(root), "--version", "9.9.0"]) == 0
    assert "No row was added, reworded or removed" in capsys.readouterr().out
    assert TOOL.main(["no-such-commit", "--root", str(root), "--version", "9.9.0"]) == 2
    assert TOOL.main(["HEAD", "--ref", "HEAD", "--root", str(root), "--version", "9.7.0"]) == 2
    assert "has no ## 9.7.0 section" in capsys.readouterr().err


def test_the_word_room_is_the_release_s_own_allowance_or_the_common_limit(tmp_path):
    # The report's "(limit N)" is what the release-notes test holds that release to; a report quoting
    # the common limit for a release given more room would send rows out of the note that fit in it.
    assert TOOL.word_limit(ROOT, "2.8.0") == 7000
    assert TOOL.word_limit(ROOT, "2.8.1") == 3500
    test = tmp_path / "tests" / "test_release_notes_are_public_facing.py"
    test.parent.mkdir()
    test.write_text('MAX_WORDS = 3500  # a note\n\nRELEASE_WORD_ALLOWANCE: dict[str, int] = {"9.9.0": 8000}\n',
                    encoding="utf-8")
    assert TOOL.word_limit(tmp_path, "9.9.0") == 8000
    assert TOOL.word_limit(tmp_path, "9.8.0") == 3500
    test.write_text("MAX_WORDS = 3500\n", encoding="utf-8")
    assert TOOL.word_limit(tmp_path, "9.9.0") == 3500
    assert TOOL.word_limit(tmp_path / "elsewhere", "9.9.0") is None
