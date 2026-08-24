"""The one manifest validator, and the one install-identity resolver.

Both exist because a field user ran the published wheel and hit, in
order: a decoder resolved against a source tree that a wheel does not
have, ``git rev-parse`` exiting 128 with the working directory in
site-packages, and a hand-authored manifest that satisfied the first
consumer and died at the third.  Every test here is one of those.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import complete_runtime_manifest
from gpuwm.runtime_manifest import (
    IdentityError,
    MANIFEST_ENV,
    ManifestError,
    RUNTIME_SCHEMA,
    git_checkout_root,
    load_manifest,
    manifest_defects,
    manifest_from_environment,
    provenance,
    validate_manifest,
    wheel_record_identity,
)


def test_a_complete_manifest_has_no_defects():
    assert manifest_defects(complete_runtime_manifest()) == []
    document = complete_runtime_manifest()
    assert validate_manifest(document) is document


def test_the_field_users_two_key_manifest_names_every_missing_field():
    """The document that passed one gate and died at a later one.

    ``{"schema": ..., "status": "READY"}`` satisfied the identity
    helper, so a preparation started; several minutes later the
    preprocessing selector refused it for ``contract.platform.backends``
    -- a key nothing had checked at the door.  All four are named now,
    together, in the first read.
    """

    defects = manifest_defects(
        {"schema": RUNTIME_SCHEMA, "status": "READY"})
    joined = "\n".join(defects)
    assert "artifact" in joined
    assert "source" in joined
    assert "contract" in joined
    assert "payload" in joined
    assert len(defects) == 4


@pytest.mark.parametrize("mutate,expected", (
    (lambda d: d.__setitem__("schema", "gpuwm-something-else"), "schema"),
    (lambda d: d.__setitem__("status", "DRAFT"), "status"),
    (lambda d: d["artifact"].__setitem__("gpuwm_version", "0.0.0"),
     "artifact.gpuwm_version"),
    (lambda d: d["source"].__setitem__("worktree_clean", False),
     "source.worktree_clean"),
    (lambda d: d["source"].pop("tree"), "source.tree"),
    (lambda d: d["contract"]["platform"].__setitem__("backends", []),
     "contract.platform.backends"),
    (lambda d: d["contract"]["platform"].__setitem__(
        "backends", ["cpu", "cpu"]), "contract.platform.backends"),
    (lambda d: d["contract"]["platform"].__setitem__(
        "backends", ["cpu", "opencl"]), "contract.platform.backends"),
    (lambda d: d.__setitem__("payload", {}), "payload"),
))
def test_every_required_field_is_actually_required(mutate, expected):
    document = complete_runtime_manifest()
    mutate(document)
    defects = manifest_defects(document)
    assert any(defect.startswith(expected) for defect in defects), defects


def test_the_refusal_names_the_document_and_the_way_out(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": RUNTIME_SCHEMA}), encoding="utf-8")
    with pytest.raises(ManifestError) as error:
        load_manifest(path)
    message = str(error.value)
    assert str(path) in message
    assert f"unset {MANIFEST_ENV}" in message


def test_an_unreadable_manifest_is_a_manifest_error(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("not json at all", encoding="utf-8")
    with pytest.raises(ManifestError, match="readable JSON"):
        load_manifest(path)


def test_environment_lookup_validates_before_it_returns(tmp_path,
                                                        monkeypatch):
    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    assert manifest_from_environment() is None

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(complete_runtime_manifest()), encoding="utf-8")
    monkeypatch.setenv(MANIFEST_ENV, str(path))
    resolved, payload = manifest_from_environment()
    assert resolved == path.resolve()
    assert payload["contract"]["platform"]["backends"]

    path.write_text(json.dumps({"schema": RUNTIME_SCHEMA}), encoding="utf-8")
    with pytest.raises(ManifestError):
        manifest_from_environment()


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_git_checkout_root_says_no_for_a_directory_in_no_repository(
        tmp_path):
    assert git_checkout_root(tmp_path) is None


def test_git_checkout_root_says_no_for_a_directory_inside_someone_elses_repo(
        tmp_path):
    """A venv nested in an unrelated repository is not provenance.

    The negative control for the failure that is worse than exit 128:
    resolving *successfully* against a repository that has nothing to do
    with this code, and binding its commit into a receipt.
    """

    try:
        subprocess.run(["git", "init", "--quiet", str(tmp_path)],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available here")
    nested = tmp_path / "lib" / "site-packages"
    nested.mkdir(parents=True)
    assert git_checkout_root(tmp_path) == tmp_path.resolve()
    assert git_checkout_root(nested) is None


class _RecordEntry(str):
    """One ``RECORD`` row, with pip's hash object attached."""

    def __new__(cls, name, digest_value):
        item = super().__new__(cls, name)
        item.hash = type("Hash", (), {"mode": "sha256",
                                      "value": digest_value})()
        return item


class _StubDistribution:
    """An installed distribution with a known RECORD, and nothing else.

    A stub rather than this tree, because how THIS checkout happens to
    be installed (editable, not at all, or as a wheel) must not decide
    whether the wheel identity path is exercised.  The end-to-end proof
    that a real wheel answers here is a wheel in a fresh venv, which is
    a different kind of test than this one.
    """

    version = "9.9.9"
    metadata = {"Name": "gpuwm"}

    def __init__(self, rows):
        self.files = [_RecordEntry(name, value) for name, value in rows]

    def locate_file(self, name):
        return f"/nowhere/{name}"


def test_provenance_from_a_non_repository_directory_binds_the_wheel(
        tmp_path, monkeypatch):
    """The bug, exactly: identity resolved from site-packages.

    ``provenance`` is handed a directory that is not a checkout -- which
    is what every pip install hands it -- and must come back with the
    installed distribution's identity instead of raising
    ``CalledProcessError: returned non-zero exit status 128``.
    """

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    stub = _StubDistribution((("gpuwm/__init__.py", "AAAA"),
                              ("gpuwm/doctor.py", "BBBB")))
    monkeypatch.setattr("gpuwm.runtime_manifest.installed_distribution",
                        lambda *_a, **_k: stub)
    identity = provenance(tmp_path)
    assert identity["identity_source"] == "installed-wheel-record"
    assert identity["git_commit"] is None
    wheel = identity["installed_wheel"]
    assert wheel["distribution_name"] == "gpuwm"
    assert wheel["distribution_version"] == "9.9.9"
    assert wheel["record_file_count"] == 2
    assert len(wheel["record_aggregate_sha256"]) == 64


def test_the_wheel_identity_is_stable_and_changes_with_the_artifacts(
        monkeypatch):
    rows = (("gpuwm/__init__.py", "AAAA"), ("gpuwm/doctor.py", "BBBB"))
    monkeypatch.setattr("gpuwm.runtime_manifest.installed_distribution",
                        lambda *_a, **_k: _StubDistribution(rows))
    first = wheel_record_identity()
    assert first == wheel_record_identity()

    monkeypatch.setattr(
        "gpuwm.runtime_manifest.installed_distribution",
        lambda *_a, **_k: _StubDistribution(
            (("gpuwm/__init__.py", "AAAA"), ("gpuwm/doctor.py", "CCCC"))))
    assert wheel_record_identity()["record_aggregate_sha256"] \
        != first["record_aggregate_sha256"]


def test_a_distribution_with_no_hashed_record_is_named_as_such(monkeypatch):
    monkeypatch.setattr("gpuwm.runtime_manifest.installed_distribution",
                        lambda *_a, **_k: _StubDistribution(()))
    with pytest.raises(IdentityError, match="no hashed RECORD"):
        wheel_record_identity()


def test_this_installs_own_identity_resolves_one_of_the_three_ways(
        monkeypatch):
    """Whatever this tree is, it can name itself.  No exceptions."""

    from pathlib import Path

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    root = Path(__file__).resolve().parent.parent
    assert provenance(root)["identity_source"] in (
        "git", "installed-wheel-record",
        "gpuwm-native-distribution-manifest")


def test_provenance_prefers_a_bound_manifest_over_everything(
        tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    document = complete_runtime_manifest()
    document["source"]["commit"] = "a" * 40
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv(MANIFEST_ENV, str(path))
    identity = provenance(tmp_path)
    assert identity["identity_source"] == "gpuwm-native-distribution-manifest"
    assert identity["git_commit"] == "a" * 40
    assert len(identity["distribution_manifest_sha256"]) == 64


def test_provenance_reports_a_checkout_as_a_checkout(monkeypatch):
    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if git_checkout_root(root) is None:
        pytest.skip("this tree is not a git checkout")
    identity = provenance(root)
    assert identity["identity_source"] == "git"
    assert len(identity["git_commit"]) == 40
    assert len(identity["git_tree"]) == 40


def test_identity_refuses_when_nothing_can_answer(tmp_path, monkeypatch):
    """The negative control: no manifest, no checkout, no distribution,
    and no source tree either.

    Watched firing -- with any ONE of the patches below removed the call
    succeeds, which is precisely why every path has to be taken away.
    ``git_identity`` is among them because a checkout serving an
    editable install answers for itself, and on a developer's box this
    test runs inside exactly such a tree.
    """

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    monkeypatch.setattr(
        "gpuwm.runtime_manifest.installed_distribution", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "gpuwm.provenance.providing_distribution", lambda *_a, **_k: None)
    monkeypatch.setattr("gpuwm.provenance.git_identity", lambda *_a, **_k: None)
    with pytest.raises(IdentityError, match="no provenance to bind"):
        provenance(tmp_path)


def test_the_hrrr_route_binds_an_identity_from_a_non_git_directory(
        tmp_path, monkeypatch):
    """The consumer, not just the helper: the benchmark's own receipt.

    ``tools/hrrr_single_domain_benchmark.py::_source_identity`` is what
    exited 128 for the field user, before any of their data was read.
    """

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    probe = subprocess.run(
        [sys.executable, "-c",
         "import json\n"
         "from tools.hrrr_single_domain_benchmark import _source_identity\n"
         "print(json.dumps(_source_identity()['identity_source']))\n"],
        cwd=tmp_path, capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": str(
                 __import__("pathlib").Path(__file__).resolve().parent.parent)})
    assert probe.returncode == 0, probe.stderr
    assert json.loads(probe.stdout.strip()) in (
        "git", "installed-wheel-record", "installed-editable-source")


# ---------------------------------------------------------------------------
# The editable install: a RECORD without hashes is not a broken install
# ---------------------------------------------------------------------------

class _UnhashedRecordEntry(str):
    """One RECORD row pip wrote with no digest at all."""

    def __new__(cls, name):
        item = super().__new__(cls, name)
        item.hash = None
        return item


class _EditableDistribution:
    """An install pip wired to a source tree, with an unhashed RECORD.

    The shape a ``pip install -e .`` actually leaves behind: RECORD rows
    with an empty digest field, because what pip wrote is a redirect,
    not the code.  ``importlib.metadata`` reports those rows with
    ``hash is None``, which is exactly what made the wheel identity
    declare this install unbindable.
    """

    version = "9.9.9"
    metadata = {"Name": "gpuwm"}

    def __init__(self, source_root, *, editable_marker=True):
        self.files = [_UnhashedRecordEntry("gpuwm/__init__.py"),
                      _UnhashedRecordEntry("gpuwm/doctor.py")]
        self._source_root = Path(source_root)
        self._editable_marker = editable_marker

    def read_text(self, name):
        if name != "direct_url.json" or not self._editable_marker:
            return None
        return json.dumps({"dir_info": {"editable": True},
                           "url": self._source_root.resolve().as_uri()})

    def locate_file(self, name):
        return f"/nowhere/{name}"


def _committed_checkout(tmp_path):
    """A real git checkout with one commit, or skip."""

    root = tmp_path / "source"
    root.mkdir()
    try:
        subprocess.run(["git", "init", "--quiet", str(root)],
                       check=True, capture_output=True)
        for key, value in (("user.email", "t@example.invalid"),
                           ("user.name", "t")):
            subprocess.run(["git", "-C", str(root), "config", key, value],
                           check=True, capture_output=True)
        (root / "tracked.txt").write_text("one\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "tracked.txt"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(root), "commit", "--quiet",
                        "-m", "one"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available here")
    return root


def test_an_editable_install_binds_its_source_tree_instead_of_refusing(
        tmp_path, monkeypatch):
    """The field refusal, exactly: prepare died on an editable install.

    ``IdentityError: installed distribution gpuwm has no hashed RECORD
    entries`` fired before a byte of the user's data was read.  An
    editable install's RECORD legitimately carries no digests; the
    identity it CAN prove is the source tree pip pointed at, which is
    what the run-plan receipt has bound all along.
    """

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    source = _committed_checkout(tmp_path)
    monkeypatch.setattr("gpuwm.runtime_manifest.installed_distribution",
                        lambda *_a, **_k: _EditableDistribution(source))
    identity = provenance(tmp_path / "not-a-repo-here")

    assert identity["identity_source"] == "installed-editable-source"
    assert identity["installed_wheel"] is None
    editable = identity["installed_editable"]
    assert editable["install_kind"] == "editable"
    assert Path(editable["source_root"]) == source.resolve()
    assert editable["distribution_name"] == "gpuwm"
    assert editable["distribution_version"] == "9.9.9"
    git = editable["git"]
    assert len(git["commit_full"]) == 40
    assert git["branch"]
    assert git["dirty"] is False
    assert git["dirty_files"] == 0
    # The shape every existing receipt consumer reads, answered for real.
    assert identity["git_commit"] == git["commit_full"]
    assert len(identity["git_tree"]) == 40
    assert identity["git_status_short"] == []


def test_a_dirty_editable_tree_is_allowed_and_says_so(tmp_path, monkeypatch):
    """A run's receipts must never claim a clean identity from a dirty tree."""

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    source = _committed_checkout(tmp_path)
    (source / "tracked.txt").write_text("two\n", encoding="utf-8")
    monkeypatch.setattr("gpuwm.runtime_manifest.installed_distribution",
                        lambda *_a, **_k: _EditableDistribution(source))
    identity = provenance(tmp_path / "elsewhere")
    git = identity["installed_editable"]["git"]
    assert git["dirty"] is True
    assert git["dirty_files"] == 1
    assert identity["git_status_short"]


def test_a_pep660_editable_that_locate_file_cannot_answer_still_binds(
        tmp_path, monkeypatch):
    """The other editable shape: no package in site-packages at all.

    ``installed_distribution()`` returns None for every PEP 660
    editable, so that install hit the FIRST refusal -- "no installed
    distribution provides this gpuwm package" -- while the source tree
    was named in ``direct_url.json`` the whole time.
    """

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    source = _committed_checkout(tmp_path)
    monkeypatch.setattr("gpuwm.runtime_manifest.installed_distribution",
                        lambda *_a, **_k: None)
    monkeypatch.setattr("gpuwm.provenance.providing_distribution",
                        lambda *_a, **_k: _EditableDistribution(source))
    identity = provenance(tmp_path / "elsewhere")
    assert identity["identity_source"] == "installed-editable-source"
    assert Path(identity["installed_editable"]["source_root"]) \
        == source.resolve()


def test_an_editable_install_with_no_readable_git_is_still_refused(
        tmp_path, monkeypatch):
    """The genuinely unbindable case keeps its refusal, plus the remedy.

    No hashed RECORD and no source identity either: there is nothing to
    bind, and inventing one would be worse than stopping.
    """

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    bare = tmp_path / "copied-source"
    bare.mkdir()
    monkeypatch.setattr("gpuwm.runtime_manifest.installed_distribution",
                        lambda *_a, **_k: _EditableDistribution(bare))
    monkeypatch.setattr("gpuwm.provenance.git_identity", lambda *_a, **_k: None)
    with pytest.raises(IdentityError, match="no hashed RECORD") as raised:
        provenance(tmp_path / "elsewhere")
    assert "editable" in str(raised.value)


def test_a_wheel_still_binds_its_record_and_pays_for_no_git_subprocess(
        tmp_path, monkeypatch):
    """The wheel path is unchanged, and does not grow a git probe."""

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    stub = _StubDistribution((("gpuwm/__init__.py", "AAAA"),))
    monkeypatch.setattr("gpuwm.runtime_manifest.installed_distribution",
                        lambda *_a, **_k: stub)

    def _refuse(*_a, **_k):
        raise AssertionError("a wheel identity must not probe git")

    monkeypatch.setattr("gpuwm.provenance.git_identity", _refuse)
    identity = provenance(tmp_path)
    assert identity["identity_source"] == "installed-wheel-record"
    assert identity["installed_editable"] is None
