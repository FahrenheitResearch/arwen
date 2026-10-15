"""The FTZ receipt tools stay inside the tree they live in (D1, 2026-10-05).

Breakage: ``python tools/ftz_receipt/claim_census.py`` run by file name
resolved ``tools.ftz_receipt`` through an editable install of a different
checkout, took that checkout as its default root and overwrote an untracked
census file there.  A decoy ``tools`` package on PYTHONPATH stands in for the
other checkout here; the real one is never touched.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools.ftz_receipt import route_inventory as ri

REPO = Path(__file__).resolve().parents[1]


def _decoy(tmp_path: Path) -> Path:
    pkg = tmp_path / "decoy" / "tools" / "ftz_receipt"
    pkg.mkdir(parents=True)
    sentinel = tmp_path / "decoy-imported"
    (tmp_path / "decoy" / "tools" / "__init__.py").write_text(
        f"open({str(sentinel)!r}, 'w').close()\n", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "route_inventory.py").write_text(textwrap.dedent(f"""
        from pathlib import Path
        def repo_root(start=None):
            return Path({str(tmp_path / 'decoy')!r})
        def require_same_tree(entry_file):
            return None
    """), encoding="utf-8")
    (pkg / "sass.py").write_text("", encoding="utf-8")
    return sentinel


@pytest.mark.parametrize("script", ["claim_census.py"])
def test_run_by_file_name_imports_its_own_tree(tmp_path, script):
    sentinel = _decoy(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "decoy"))
    out = subprocess.run(
        [sys.executable, "-X", "utf8",
         str(REPO / "tools" / "ftz_receipt" / script), "--help"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
        timeout=120)
    assert out.returncode == 0, out.stderr
    assert not sentinel.exists(), "imported the decoy tree's tools package"
    assert not (tmp_path / "decoy" / "gpuwm").exists()


def test_cross_tree_helper_is_refused(tmp_path):
    other = tmp_path / "other" / "tools" / "ftz_receipt" / "claim_census.py"
    with pytest.raises(SystemExit, match="refusing"):
        ri.require_same_tree(other)


def test_same_tree_helper_is_accepted():
    ri.require_same_tree(REPO / "tools" / "ftz_receipt" / "claim_census.py")
