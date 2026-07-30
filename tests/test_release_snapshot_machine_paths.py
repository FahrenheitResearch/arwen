"""No shipped file may carry one machine's absolute paths.

RELEASE-EXCLUDE has always *said* "no developer-specific absolute
paths".  Nothing enforced it, and the v1.0.1 exit audit found 51 such
lines across 25 tracked files that every exclusion rule had let
through -- a WSL home in an executable default, a Windows profile in a
committed process capture, a developer's tree in an oracle receipt.

An exclusion manifest cannot close that class: it cannot name the file
that grows a machine path next week.  So the snapshot builder scans
every staged text file and refuses to build.  This module holds the
tests for that scan, plus the regression that matters most -- the
current tree, minus its exclusions, is clean.

Every offending string below is assembled from fragments rather than
written out, because this file is one of the files the scan reads.  A
literal here would flag the test that tests the scan, which is both
confusing and, at the moment it happened, indistinguishable from a real
finding.  ``tools/build_native_wrf_distribution.py`` does the same for
the same reason.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "work" / "build_release_snapshot.py"

#: The snapshot builder lives under ``work/``, which RELEASE-EXCLUDE
#: keeps out of the public release -- so in a published tree there is
#: nothing here to test.  A development checkout always has it.
requires_builder = pytest.mark.skipif(
    not BUILDER.is_file(),
    reason="work/build_release_snapshot.py is not in this tree "
           "(published snapshot: the builder is publisher scaffolding)",
)

requires_git = pytest.mark.skipif(
    not (REPO / ".git").exists(),
    reason="not a git checkout (installed package or exported tree)",
)


def _snap():
    sys.path.insert(0, str(REPO / "work"))
    try:
        import build_release_snapshot as module
    finally:
        sys.path.pop(0)
    return module


# The four shapes a per-user absolute path takes, none of them naming a
# real person -- built at runtime so this file stays clean.
_BS = chr(92)
WINDOWS_PROFILE = "C:" + _BS + "Users" + _BS + "somebody" + _BS + "notes.txt"
WSL_PROFILE = "/mnt/" + "c/Users/" + "somebody/Downloads/bundle"
POSIX_HOME = "/ho" + "me/somebody/wrf-build"
MAC_HOME = "/Us" + "ers/somebody/Library/data"
ROOT_HOME = "/ro" + "ot/artifacts"


@requires_builder
@pytest.mark.parametrize(
    ("payload", "kind"),
    (
        (WINDOWS_PROFILE, "Windows user profile"),
        ("C:/" + "Users/somebody/notes.txt", "Windows user profile"),
        (WSL_PROFILE, "WSL-mounted user profile"),
        (POSIX_HOME, "POSIX home directory"),
        (MAC_HOME, "macOS home directory"),
        (ROOT_HOME, "root home directory"),
    ),
)
def test_every_machine_path_shape_is_caught(payload, kind):
    snap = _snap()
    found = snap.machine_path_violations(f"prefix\nSRC={payload}\nsuffix\n")
    assert [(number, label) for number, label, _ in found] == [(2, kind)]


@requires_builder
def test_ordinary_absolute_paths_are_not_flagged():
    """The bar is a *per-user* path, not any absolute path."""

    snap = _snap()
    benign = "\n".join([
        "prefix = /usr/local/share/wps-geog",
        "OUT=/tmp/noahmp-oracle",
        "cache = /var/cache/arwen",
        'root = "C:' + _BS + 'Program Files' + _BS + 'NVIDIA"',
        "home = $HOME/WRF_BUILD/LIBRARIES",
    ])
    assert snap.machine_path_violations(benign) == []


@requires_builder
def test_the_scan_walks_a_staged_tree_and_skips_vendored_and_binary(tmp_path):
    snap = _snap()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(
        f"run it from {POSIX_HOME}\n", encoding="utf-8")
    # Third-party source: upstream authors' paths are theirs, the trees
    # are checksum-locked, and editing them breaks the build.
    (tmp_path / "crate" / "vendor" / "foo").mkdir(parents=True)
    (tmp_path / "crate" / "vendor" / "foo" / "lib.rs").write_text(
        f"// built at {MAC_HOME}\n", encoding="utf-8")
    # Not a text suffix the scan reads.
    (tmp_path / "capture.bin").write_text(
        f"{WINDOWS_PROFILE}\n", encoding="utf-8")

    hits = snap.machine_path_hits(str(tmp_path))
    assert [(rel, number) for rel, number, _, _ in hits] == [
        ("docs/guide.md", 1)]


@requires_builder
def test_the_scan_is_wired_into_the_snapshot_verdict():
    """A finding must FAIL the build, not merely print."""

    source = BUILDER.read_text(encoding="utf-8")
    assert "machine_paths = machine_path_hits(SNAP)" in source
    verdict = source.split("ok = ", 1)[1].split("\n\n", 1)[0]
    assert "not machine_paths" in verdict


@requires_builder
@requires_git
def test_the_staged_release_tree_carries_no_machine_paths():
    """The regression the audit finding actually asks for.

    Applies RELEASE-EXCLUDE to the tracked tree exactly as the builder
    does, then reads every in-scope file.  A new machine path anywhere
    in ArWen fails here, days before anyone runs the release build.
    """

    snap = _snap()
    rules = snap.read_exclusions()
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        check=True, capture_output=True, text=True).stdout.splitlines()

    def kept(rel: str) -> bool:
        parts = rel.split("/")
        return not any(
            snap.matches("/".join(parts[:index]), rules)
            for index in range(1, len(parts) + 1))

    offenders = []
    scanned = 0
    for rel in tracked:
        if not rel or not kept(rel) or not snap.scanned(rel):
            continue
        path = REPO / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        for number, kind, line in snap.machine_path_violations(text):
            offenders.append(f"{rel}:{number} ({kind}) {line}")

    assert scanned > 500, "the scan found almost nothing to read"
    assert offenders == [], (
        f"{len(offenders)} developer-absolute path(s) would ship:\n  "
        + "\n  ".join(offenders)
        + "\nParameterize/relativize the path if the file is useful to "
          "the public, else add it to RELEASE-EXCLUDE.txt.")


@requires_builder
@requires_git
def test_the_campaign_harness_exclusions_still_match_something():
    """A rule that matches nothing is manifest drift, not hygiene."""

    snap = _snap()
    rules = snap.read_exclusions()
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    for rule in ("tools/n5s/**", "tests/test_n5s_toolchain.py",
                 "tools/rrtmg_wrf461_oracle/sw_fixtures/"
                 "sw-oracle-sha256sums.txt"):
        assert rule in rules, f"{rule} is no longer in RELEASE-EXCLUDE.txt"
        root = rule[:-3] if rule.endswith("/**") else rule
        assert any(rel == root or rel.startswith(root + "/")
                   for rel in tracked), f"{rule} matches no tracked path"


@requires_builder
@requires_git
def test_nothing_shipped_imports_the_excluded_campaign_harness():
    """Excluding a package must not leave a dangling import behind."""

    snap = _snap()
    rules = snap.read_exclusions()
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    needle = "tools." + "n5s"
    importers = []
    for rel in tracked:
        parts = rel.split("/")
        if any(snap.matches("/".join(parts[:i]), rules)
               for i in range(1, len(parts) + 1)):
            continue
        if needle in (REPO / rel).read_text(encoding="utf-8",
                                            errors="replace"):
            importers.append(rel)
    assert importers == [], (
        f"these ship but import the excluded harness: {importers}")
