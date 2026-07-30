"""The executable bit on the README's first command is load-bearing.

v1.0.0 shipped `install.sh` as mode 100644.  `./install.sh` -- README
line 1 of the install section -- answered `Permission denied` on every
fresh Linux and macOS clone, which two independent first-time-user
pilots hit within their first minute (ARWEN-NODE1 B1, ARWEN-NODE2 PP-6).
NTFS has no POSIX mode bit and this repository is developed on Windows,
so nothing about the working tree can catch a regression: only the git
index records the mode, and only a check on the index sees it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Every path whose executable bit a fresh clone depends on.
REQUIRED_EXECUTABLE = ("install.sh",)


def _index_modes(*paths: str) -> dict[str, str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-s", *paths],
        check=True, capture_output=True, text=True).stdout
    modes = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        modes[path] = meta.split()[0]
    return modes


requires_git = pytest.mark.skipif(
    not (REPO / ".git").exists(),
    reason="not a git checkout (installed package or exported tree)",
)

#: RELEASE-EXCLUDE keeps ``work/`` out of the public snapshot, so the
#: builder itself is absent from a published tree and the test that
#: imports it has nothing to assert.  A published clone is still a git
#: checkout, so `requires_git` does not cover this.
requires_builder = pytest.mark.skipif(
    not (REPO / "work" / "build_release_snapshot.py").is_file(),
    reason="work/build_release_snapshot.py is not in this tree "
           "(published snapshot: the builder is publisher scaffolding)",
)


@requires_git
def test_install_sh_is_executable_in_the_git_index():
    modes = _index_modes(*REQUIRED_EXECUTABLE)
    assert set(modes) == set(REQUIRED_EXECUTABLE), (
        f"expected {REQUIRED_EXECUTABLE} tracked; got {sorted(modes)}")
    for path, mode in modes.items():
        assert mode == "100755", (
            f"{path} is mode {mode} in the git index; a fresh clone will "
            f"refuse ./{path} with 'Permission denied'.  Fix with: "
            f"git update-index --chmod=+x {path}")


@requires_git
@requires_builder
def test_the_snapshot_builder_refuses_to_drop_a_required_executable():
    """The publish path is where v1.0.0 actually lost the bit.

    `git archive` carries mode 100755, but extracting the tar on NTFS
    and re-committing records 100644 for everything.  The snapshot
    builder therefore enumerates the executables itself and FAILs if a
    required one is missing, and prints the update-index line to run.
    """
    sys.path.insert(0, str(REPO / "work"))
    try:
        import build_release_snapshot as snap
    finally:
        sys.path.pop(0)

    assert snap.REQUIRED_EXECUTABLE == REQUIRED_EXECUTABLE
    executables = snap.executable_paths(str(REPO))
    for path in REQUIRED_EXECUTABLE:
        assert path in executables
    # An exclusion rule that swallowed install.sh would be caught too:
    # kept_executables filters by `matches`, and main() FAILs on any
    # REQUIRED_EXECUTABLE that does not survive.
    rules = snap.read_exclusions()
    for path in REQUIRED_EXECUTABLE:
        assert snap.matches(path, rules) is None


@requires_git
def test_the_readme_leads_with_the_mode_independent_form():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "bash install.sh" in readme
    # And it must say what to do if the mode bit is missing anyway.
    assert "Permission denied" in readme
