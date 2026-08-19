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


def _release_tree_files() -> list[str]:
    """Every path the release would carry, git checkout or not.

    The tree scan used to be gated on ``git ls-files``, and the 2.5.0
    Linux shakeout measured what that costs: run against the EXPORTED
    tree -- exactly the tree that gets packaged -- this module reported
    "9 passed, 3 skipped", and the three skipped were the only ones that
    read the tree at all.  A gate that reports green in the one context
    it exists for is not a gate, so off-checkout the walk stands in for
    the index.  In a checkout the index still wins: it is the only
    answer that excludes a developer's untracked scratch, which is not
    shipped and whose paths would be noise.
    """

    if (REPO / ".git").exists():
        return subprocess.run(
            ["git", "-C", str(REPO), "ls-files"],
            check=True, capture_output=True, text=True).stdout.splitlines()
    found = []
    for path in REPO.rglob("*"):
        if path.is_file():
            found.append(path.relative_to(REPO).as_posix())
    return found


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
#: The same Windows path as it appears inside JSON -- one separator per
#: level becomes two, because JSON escapes the backslash.  Every receipt
#: and proof document in the tree writes it this way.
WINDOWS_PROFILE_JSON = WINDOWS_PROFILE.replace(_BS, _BS + _BS)


@requires_builder
@pytest.mark.parametrize(
    ("payload", "kind"),
    (
        (WINDOWS_PROFILE, "Windows user profile"),
        ("C:/" + "Users/somebody/notes.txt", "Windows user profile"),
        # Hole B, measured on the 2.5.0 tip: the marker demanded exactly
        # one separator after the drive colon, so every JSON-escaped
        # spelling walked past it.  Five such lines shipped in both
        # wheels and both sdists while the gate reported 12 passed.
        (WINDOWS_PROFILE_JSON, "Windows user profile"),
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
        # A TEMPLATE is the remedy, not the defect: the user segment is
        # substituted at run time and names nobody.  Widening the
        # Windows marker to see through JSON escaping must not start
        # flagging the shapes that exist to avoid the problem --
        # tests/test_report_bundle.py builds exactly the first one to
        # prove the report bundler redacts a home directory.
        'out = f"C:' + _BS * 2 + 'Users' + _BS * 2 + '{username}"',
        "out = %USERPROFILE%" + _BS + "gpuwm",
        "out = $env:USERPROFILE/gpuwm",
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
    # Real binary: a NUL in the first block is what "not text" means.
    (tmp_path / "capture.bin").write_bytes(
        b"\x00\x01\x02" + WINDOWS_PROFILE.encode("utf-8") + b"\n")

    hits = snap.machine_path_hits(str(tmp_path))
    assert [(rel, number) for rel, number, _, _ in hits] == [
        ("docs/guide.md", 1)]


@requires_builder
@pytest.mark.parametrize(
    "name",
    (
        # Hole A, measured on the 2.5.0 tip: SCANNED_SUFFIXES listed 22
        # suffixes and these were not among them, so eleven developer
        # paths -- including three shipped WPS namelists pointing at a
        # geography root that exists on one box -- were never read at
        # all.  A suffix ALLOWLIST is the defect, not the entries it is
        # missing: the file that grows a machine path next has a suffix
        # nobody thought of, which is how a .cu comment, a .wps config
        # and an extensionless script all walked past a green gate.
        "kernels/thompson.cu",
        "kernels/thompson.cuh",
        "configs/probe.namelist.wps",
        "tools/run-the-case",
        "notes.rst",
        "Makefile",
    ),
)
def test_the_scan_reads_text_by_content_not_by_suffix(tmp_path, name):
    snap = _snap()
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"path = {POSIX_HOME}\n", encoding="utf-8")

    hits = snap.machine_path_hits(str(tmp_path))
    assert [(rel, number) for rel, number, _, _ in hits] == [
        (name, 1)], f"{name} was not read, so its machine path ships unseen"


@requires_builder
def test_the_scan_is_wired_into_the_snapshot_verdict():
    """A finding must FAIL the build, not merely print."""

    source = BUILDER.read_text(encoding="utf-8")
    assert "machine_paths = machine_path_hits(SNAP)" in source
    verdict = source.split("ok = ", 1)[1].split("\n\n", 1)[0]
    assert "not machine_paths" in verdict


@requires_builder
def test_the_staged_release_tree_carries_no_machine_paths():
    """The regression the audit finding actually asks for.

    Applies RELEASE-EXCLUDE to the release file set exactly as the
    builder does, then reads every in-scope file.  A new machine path
    anywhere in ArWen fails here, days before anyone runs the release
    build -- and now in an exported tree too, where this used to skip.
    """

    snap = _snap()
    rules = snap.read_exclusions()

    def kept(rel: str) -> bool:
        parts = rel.split("/")
        return not any(
            snap.matches("/".join(parts[:index]), rules)
            for index in range(1, len(parts) + 1))

    offenders = []
    scanned = 0
    for rel in _release_tree_files():
        if not rel or not kept(rel) or not snap.in_scan_scope(rel):
            continue
        path = REPO / rel
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if not snap.reads_as_text(raw):
            continue
        scanned += 1
        text = raw.decode("utf-8", "replace")
        for number, kind, line in snap.machine_path_violations(text):
            offenders.append(f"{rel}:{number} ({kind}) {line}")

    assert scanned > 500, "the scan found almost nothing to read"
    assert offenders == [], (
        f"{len(offenders)} developer-absolute path(s) would ship:\n  "
        + "\n  ".join(offenders)
        + "\nParameterize/relativize the path if the file is useful to "
          "the public, else add it to RELEASE-EXCLUDE.txt.")


@requires_builder
def test_every_pinned_record_the_scan_forgives_is_still_the_pinned_bytes():
    """The one allowance, and the proof it is still what it claims.

    ``ALLOWED_MACHINE_PATH_FILES`` forgives files whose exact bytes are
    pinned by a committed manifest: a WPS namelist recorded as the
    provenance of the WRF reference wrfouts is a RECORD of one build on
    one box, and editing the developer path out of it would break the
    ``namelist_sha256`` the manifest beside it pins -- falsifying the
    record rather than cleaning it.

    An allowance nobody re-checks is just a hole with a comment, so
    this reads the pin: if the file's bytes ever stop matching the
    manifest that justifies forgiving them, it is no longer a pinned
    record and the allowance dies here.
    """

    import hashlib
    import json

    snap = _snap()
    assert snap.ALLOWED_MACHINE_PATH_FILES, (
        "the allowance list is empty; delete it rather than leaving an "
        "unused escape hatch in a gate")
    for rel, pin in snap.ALLOWED_MACHINE_PATH_FILES.items():
        path = REPO / rel
        if not path.is_file():
            pytest.skip(f"{rel} is not in this tree")
        manifest = json.loads(
            (REPO / pin.manifest).read_text(encoding="utf-8"))
        expected = manifest
        for key in pin.keys:
            expected = expected[key]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, (
            f"{rel} is forgiven by the machine-path scan because "
            f"{pin.manifest} pins its bytes, but it now hashes "
            f"{digest} against the pinned {expected}. It is no longer "
            "the record the allowance describes: scrub the machine "
            "path and drop the allowance, or restore the pinned bytes.")


@requires_builder
@requires_git
def test_the_campaign_harness_exclusions_still_match_something():
    """A rule that matches nothing is manifest drift, not hygiene.

    Checkout-only on purpose, unlike the scan above: an EXPORTED tree
    has already had the exclusions applied, so every rule matching
    nothing there is the correct answer, not drift.
    """

    snap = _snap()
    rules = snap.read_exclusions()
    tracked = _release_tree_files()
    for rule in ("tools/n5s/**", "tests/test_n5s_toolchain.py",
                 "tools/rrtmg_wrf461_oracle/sw_fixtures/"
                 "sw-oracle-sha256sums.txt"):
        assert rule in rules, f"{rule} is no longer in RELEASE-EXCLUDE.txt"
        root = rule[:-3] if rule.endswith("/**") else rule
        assert any(rel == root or rel.startswith(root + "/")
                   for rel in tracked), f"{rule} matches no tracked path"


@requires_builder
def test_nothing_shipped_imports_the_excluded_campaign_harness():
    """Excluding a package must not leave a dangling import behind."""

    snap = _snap()
    rules = snap.read_exclusions()
    tracked = [rel for rel in _release_tree_files() if rel.endswith(".py")]
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
