"""The provenance resolver: which tree runs, and does it agree with itself.

Every install shape here is built out of REAL files -- a real package
directory, a real ``.dist-info`` read by a real ``PathDistribution``, a
real ``git init`` with a real dirty file -- rather than mocks.  The
shapes that have actually broken this project were ones no mock would
have invented: a distribution whose ``locate_file`` names a path that
does not exist, and a ``__version__`` that is correct-looking and
belongs to a different tree.

The last section is the important one.  A resolver that returns a
constant "everything is fine" would pass any test that only checks the
shape of its output, and that is precisely the failure mode this module
exists to prevent -- so every scenario below is replayed against two
constant resolvers, one that always reports a healthy install and one
that always reports a broken one, and each scenario must FAIL against
both.  A test that cannot fail is worse than no test.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from importlib.metadata import PathDistribution
from pathlib import Path
from typing import Callable

import pytest

from gpuwm import provenance
from gpuwm.provenance import Provenance, describe_provenance

WORKTREE = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# real files on disk
# ---------------------------------------------------------------------------

def _package(root: Path) -> Path:
    package = root / "gpuwm"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    return package


def _dist_info(site: Path, *, name="gpuwm", version="1.8.7",
               editable_at: Path | None = None) -> PathDistribution:
    """A real .dist-info on disk, read by a real PathDistribution."""

    site.mkdir(parents=True, exist_ok=True)
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8")
    if editable_at is not None:
        (info / "direct_url.json").write_text(json.dumps({
            "url": editable_at.resolve().as_uri(),
            "dir_info": {"editable": True},
        }), encoding="utf-8")
    return PathDistribution(info)


def _pyproject(root: Path, *, name="gpuwm", version="1.8.7") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pyproject.toml"
    path.write_text(
        f'[project]\nname = "{name}"\nversion = "{version}"\n',
        encoding="utf-8")
    return path


def _git_init(root: Path) -> None:
    """A real repository with one real commit, or skip."""

    init = subprocess.run(["git", "init", "-q", str(root)],
                          capture_output=True, text=True)
    if init.returncode != 0:
        pytest.skip("no usable git on this machine")
    (root / "tracked.txt").write_text("original\n", encoding="utf-8")
    for arguments in (["add", "-A"],
                      ["-c", "user.email=t@t", "-c", "user.name=t",
                       "commit", "-q", "-m", "fixture"]):
        subprocess.run(["git", "-C", str(root), *arguments],
                       capture_output=True, check=False)


# ---------------------------------------------------------------------------
# the scenarios, each an install shape with the claims that define it
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    build: Callable[[Path], Provenance]
    check: Callable[[Provenance], None]


# -- a wheel -----------------------------------------------------------

def _build_wheel(tmp: Path) -> Provenance:
    site = tmp / "site-packages"
    package = _package(site)
    return describe_provenance(
        package, _dist_info(site, version="1.8.0"),
        reported_version="1.8.0")


def _check_wheel(p: Provenance) -> None:
    assert p.install_kind == "wheel"
    assert p.distribution_name == "gpuwm"
    assert p.metadata_version == "1.8.0"
    assert p.package_path.endswith(os.sep.join(("site-packages", "gpuwm")))
    # A wheel ships no pyproject.toml; pip wrote its code and metadata
    # together, so the metadata IS the code's declaration -- and the
    # resolver has to SAY that rather than leave it a silent tautology.
    assert p.code_version == "1.8.0"
    assert p.code_version_source == "wheel-metadata"
    assert p.versions_agree is True
    assert p.git is None
    assert p.is_consistent


# -- an editable install of a clean checkout ---------------------------

def _build_editable(tmp: Path) -> Provenance:
    source = tmp / "home" / "user" / "gpuwm"
    package = _package(source)
    _pyproject(source, version="1.8.7")
    _git_init(source)
    return describe_provenance(
        package, _dist_info(tmp / "site-packages", version="1.8.7",
                            editable_at=source),
        reported_version="1.8.7")


def _check_editable(p: Provenance) -> None:
    assert p.install_kind == "editable"
    assert p.metadata_version == "1.8.7"
    assert p.code_version == "1.8.7"
    assert p.code_version_source.endswith("pyproject.toml")
    assert p.versions_agree is True
    assert p.source_root.endswith("gpuwm")
    assert p.git and p.git["branch"]
    assert len(p.git["commit"]) == 12
    assert p.git["dirty"] is False
    assert p.is_consistent


# -- a plain source tree, nothing installed ----------------------------

def _build_source_tree(tmp: Path) -> Provenance:
    _pyproject(tmp, version="1.8.7")
    return describe_provenance(
        _package(tmp), None,
        reported_version=provenance.UNKNOWN_VERSION)


def _check_source_tree(p: Provenance) -> None:
    assert p.install_kind == "source-tree"
    assert p.distribution_name is None
    assert p.metadata_version is None
    assert p.code_version == "1.8.7"
    # Nothing to compare is not a defect, and must not be reported as
    # one: a gate that refused this would refuse every fresh clone.
    assert p.versions_agree is None
    assert p.metadata_is_borrowed is False
    assert p.disagreement is None
    assert p.is_consistent
    assert any("no metadata version to compare" in note for note in p.notes)


# -- a dirty working tree ----------------------------------------------

def _build_dirty(tmp: Path) -> Provenance:
    source = tmp / "checkout"
    package = _package(source)
    _pyproject(source, version="1.8.7")
    _git_init(source)
    (source / "tracked.txt").write_text("EDITED\n", encoding="utf-8")
    (source / "scratch.log").write_text("untracked\n", encoding="utf-8")
    return describe_provenance(package, None, reported_version=None)


def _check_dirty(p: Provenance) -> None:
    assert p.git is not None
    assert p.git["dirty"] is True
    assert p.git["dirty_files"] == 1
    # The untracked file must NOT be what makes it dirty.  Folding
    # untracked scratch into the flag would light it permanently on
    # every real checkout and train readers to ignore it.
    assert p.git["untracked_files"] == 1
    assert "(dirty)" in p.banner()


# -- THE disagreement: a stale install over newer code ------------------

def _build_stale_editable(tmp: Path) -> Provenance:
    """The reported field case: plots labelled 1.6.2 on a 1.8.7 tree."""

    source = tmp / "home" / "user" / "gpuwm"
    package = _package(source)
    _pyproject(source, version="1.8.7")
    return describe_provenance(
        package, _dist_info(tmp / "site-packages", version="1.6.2",
                            editable_at=source),
        reported_version="1.6.2")


def _check_stale_editable(p: Provenance) -> None:
    assert p.metadata_version == "1.6.2"
    assert p.code_version == "1.8.7"
    assert p.versions_agree is False
    assert not p.is_consistent
    assert "VERSION DISAGREEMENT" in p.disagreement
    assert "1.6.2" in p.disagreement and "1.8.7" in p.disagreement
    # The banner is where a user actually meets this.
    assert "VERSION DISAGREEMENT" in p.banner()


# -- THE other disagreement: a borrowed version -------------------------

def _build_borrowed(tmp: Path) -> Provenance:
    """Measured live on this box.

    No distribution provides the running code, yet ``gpuwm.__version__``
    still returns a number, because it asks metadata BY NAME and some
    other ``.dist-info`` answered.  The number describes a different
    tree.  It is at its most dangerous when it happens to match, which
    is why the resolver judges provenance and not just digits.
    """

    _pyproject(tmp, version="1.8.7")
    return describe_provenance(
        _package(tmp), None, reported_version="1.8.7")


def _check_borrowed(p: Provenance) -> None:
    assert p.distribution_name is None
    assert p.reported_version == "1.8.7"
    assert p.code_version == "1.8.7"
    # Numerically identical, and still a disagreement: the reported
    # number is not backed by the code that is running.
    assert p.metadata_is_borrowed is True
    assert p.versions_agree is False
    assert not p.is_consistent
    assert "BORROWED" in p.disagreement


SCENARIOS = [
    Scenario("wheel", _build_wheel, _check_wheel),
    Scenario("editable", _build_editable, _check_editable),
    Scenario("source-tree", _build_source_tree, _check_source_tree),
    Scenario("dirty-tree", _build_dirty, _check_dirty),
    Scenario("disagreement-stale-install", _build_stale_editable,
             _check_stale_editable),
    Scenario("disagreement-borrowed-version", _build_borrowed,
             _check_borrowed),
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_the_resolver_names_each_install_shape(scenario, tmp_path):
    scenario.check(scenario.build(tmp_path))


# ---------------------------------------------------------------------------
# the tests must be able to fail
# ---------------------------------------------------------------------------

#: The most dangerous constant a broken resolver could return: a
#: confident, self-consistent, healthy install.
_ALWAYS_FINE = Provenance(
    package_path="/constant/gpuwm", source_root="/constant",
    install_kind="wheel", distribution_name="gpuwm",
    metadata_version="1.8.7", reported_version="1.8.7",
    code_version="1.8.7", code_version_source="wheel-metadata",
    versions_agree=True)

#: The opposite constant, so that the scenarios asserting a HEALTHY
#: install are proven to be testing agreement too, not merely accepting
#: whatever they are handed.
_ALWAYS_BROKEN = Provenance(
    package_path="/constant/gpuwm", source_root="/constant",
    install_kind="source-tree", reported_version="9.9.9",
    metadata_is_borrowed=True, versions_agree=False,
    disagreement="VERSION IS BORROWED: constant")


@pytest.mark.parametrize("constant", [_ALWAYS_FINE, _ALWAYS_BROKEN],
                         ids=["always-fine", "always-broken"])
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_every_scenario_fails_against_a_constant_resolver(
        scenario, constant, tmp_path):
    """Proof that each scenario above can fail.

    Built first, so that a scenario which skips for want of git skips
    here too rather than passing vacuously.
    """

    scenario.build(tmp_path)
    with pytest.raises(AssertionError):
        scenario.check(constant)


def test_agreement_responds_to_the_version_alone(tmp_path):
    """Discrimination with the paths held fixed.

    The scenario checks above would also fail a constant on its path,
    which is a real defect but not the one that matters.  This holds
    EVERYTHING constant except the installed metadata version and
    requires the verdict to flip -- so `versions_agree` is proven to be
    a measurement of the versions, not a decoration.
    """

    source = tmp_path / "gpuwm-checkout"
    package = _package(source)
    _pyproject(source, version="1.8.7")

    def verdict(installed: str) -> Provenance:
        return describe_provenance(
            package, _dist_info(tmp_path / f"site-{installed}",
                                version=installed, editable_at=source),
            reported_version=installed)

    matched, stale = verdict("1.8.7"), verdict("1.6.2")
    assert matched.package_path == stale.package_path
    assert matched.install_kind == stale.install_kind == "editable"
    assert matched.versions_agree is True and matched.disagreement is None
    assert stale.versions_agree is False and stale.disagreement


def test_dirty_responds_to_the_working_tree_alone(tmp_path):
    """Same discrimination for the dirty flag: one edit flips it."""

    source = tmp_path / "checkout"
    package = _package(source)
    _git_init(source)
    clean = describe_provenance(package, None)
    (source / "tracked.txt").write_text("EDITED\n", encoding="utf-8")
    dirty = describe_provenance(package, None)
    assert clean.git["commit"] == dirty.git["commit"]
    assert clean.git["dirty"] is False and dirty.git["dirty"] is True


# ---------------------------------------------------------------------------
# the serialisable form and the human form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_the_serialised_form_is_json_and_stable(scenario, tmp_path):
    """A receipt embeds this; a Path object would break json.dumps."""

    payload = scenario.build(tmp_path).as_dict()
    text = json.dumps(payload, sort_keys=True)
    assert json.loads(text) == payload
    assert payload["schema"] == provenance.PROVENANCE_SCHEMA
    assert set(payload) == {
        "schema", "package_path", "source_root", "install_kind",
        "distribution_name", "metadata_version", "reported_version",
        "metadata_is_borrowed", "code_version", "code_version_source",
        "versions_agree", "disagreement", "git", "notes"}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_the_banner_is_one_line_and_names_the_tree(scenario, tmp_path):
    resolved = scenario.build(tmp_path)
    line = resolved.banner()
    assert "\n" not in line
    assert line.startswith("gpuwm")
    assert resolved.source_root in line or resolved.package_path in line
    # An inconsistent install never gets a banner that looks clean.
    assert (resolved.disagreement is None) == resolved.is_consistent


# ---------------------------------------------------------------------------
# it must never raise, and never be expensive
# ---------------------------------------------------------------------------

def test_a_directory_that_is_not_a_repository_reports_no_git(tmp_path):
    assert provenance.git_identity(tmp_path) is None


def test_an_absent_git_binary_is_not_an_error(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git: not found")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert provenance.git_identity(tmp_path) is None


def test_a_repository_with_no_commits_reports_no_identity(tmp_path):
    """A checkout, but nothing to bind: an absence, not an invention."""

    if subprocess.run(["git", "init", "-q", str(tmp_path)],
                      capture_output=True).returncode != 0:
        pytest.skip("no usable git on this machine")
    assert provenance.git_identity(tmp_path) is None


def test_a_foreign_pyproject_is_not_this_codes_version(tmp_path):
    """A gpuwm package inside somebody else's repository.

    Binding a stranger's version is the same class of error as binding a
    stranger's commit, which `git_checkout_root` already refuses.
    """

    _pyproject(tmp_path, name="somebody-elses-project", version="0.4.2")
    version, note = provenance.pyproject_version(tmp_path)
    assert version is None
    assert "does not publish this package" in note
    assert describe_provenance(_package(tmp_path), None).code_version is None


def test_an_unreadable_pyproject_is_a_note_not_a_crash(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project\nname =",
                                             encoding="utf-8")
    version, note = provenance.pyproject_version(tmp_path)
    assert version is None and "unreadable" in note


def test_a_dynamic_version_is_a_legitimate_absence(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "gpuwm"\ndynamic = ["version"]\n', encoding="utf-8")
    version, note = provenance.pyproject_version(tmp_path)
    assert version is None and "no static" in note


def test_the_resolver_never_raises_even_when_everything_is_broken(
        monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("the estate is on fire")

    monkeypatch.setattr(provenance, "providing_distribution", explode)
    resolved = provenance.resolve(refresh=True)
    assert isinstance(resolved, Provenance)
    assert "\n" not in resolved.banner()
    assert json.dumps(resolved.as_dict())
    assert any("could not be resolved" in note for note in resolved.notes)


def test_the_answer_is_cached_so_it_can_be_called_at_every_start(monkeypatch):
    provenance.resolve(refresh=True)
    calls = []
    monkeypatch.setattr(provenance, "providing_distribution",
                        lambda path: calls.append(path))
    provenance.resolve()
    provenance.resolve()
    assert calls == [], "a cached resolve re-measured the machine"
    provenance.resolve(refresh=True)
    assert len(calls) == 1, "refresh=True did not re-measure"


def test_the_git_identity_is_the_executing_trees_not_the_installs(tmp_path):
    """Worktree code must stamp the WORKTREE's commit, not the install's.

    An editable install's PEP 610 direct_url names the MAIN checkout.
    When a parallel worktree's code is what actually imports (its cwd
    precedes site-packages on sys.path), the receipt banner used to
    stamp the MAIN checkout's git identity into receipts written by
    WORKTREE code -- found independently by two gauntlet lanes.  The
    identity belongs to the tree that owns the executing package.
    """

    install = tmp_path / "main-checkout"
    _package(install)
    _pyproject(install, version="1.8.7")
    _git_init(install)
    worktree = tmp_path / "worktree"
    package = _package(worktree)
    _pyproject(worktree, version="1.8.7")
    _git_init(worktree)
    (worktree / "tracked.txt").write_text("worktree line\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "-A"],
                   capture_output=True, check=False)
    subprocess.run(["git", "-C", str(worktree), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-q", "-m", "worktree"],
                   capture_output=True, check=False)

    resolved = describe_provenance(
        package, _dist_info(tmp_path / "site-packages", version="1.8.7",
                            editable_at=install),
        reported_version="1.8.7")

    def _head(root: Path) -> str:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip()

    assert resolved.git is not None
    assert resolved.git["commit_full"] == _head(worktree)
    assert resolved.git["commit_full"] != _head(install)


def test_a_src_layout_still_falls_back_to_the_editable_root(tmp_path):
    """The executing tree wins only when it IS a checkout top.

    In a src/ layout the package's parent is not the repository root, so
    probing it yields nothing; the distribution's editable root is then
    the only honest identity left and must still be reported.
    """

    project = tmp_path / "project"
    package = _package(project / "src")
    _pyproject(project, version="1.8.7")
    _git_init(project)

    resolved = describe_provenance(
        package, _dist_info(tmp_path / "site-packages", version="1.8.7",
                            editable_at=project),
        reported_version="1.8.7")

    assert resolved.git is not None
    head = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False).stdout.strip()
    assert resolved.git["commit_full"] == head


# ---------------------------------------------------------------------------
# against the artifact: this very process, and a clean child process
# ---------------------------------------------------------------------------

def test_it_resolves_this_running_worktree():
    """The resolver must name the tree the suite is actually running from."""

    resolved = provenance.resolve(refresh=True)
    assert Path(resolved.package_path) == WORKTREE / "gpuwm"
    assert Path(provenance.__file__).resolve().parent \
        == Path(resolved.package_path)
    if resolved.git:
        assert len(resolved.git["commit"]) == 12
        assert resolved.git["branch"]


def test_it_imports_without_numpy_or_cupy_in_a_pinned_child():
    """A startup banner runs before anything heavy, so prove it can.

    Run in a child with PYTHONSAFEPATH=1 and PYTHONPATH pinned to this
    worktree, and the child asserts its own gpuwm path -- an editable
    install elsewhere on this machine maps `gpuwm` to a different tree,
    and a subprocess that silently imported THAT would make this
    measurement void.
    """

    import sys

    program = (
        "import sys, json;"
        "import gpuwm.provenance as p;"
        "print(json.dumps({"
        "'file': p.__file__,"
        "'heavy': sorted(m for m in sys.modules"
        " if m.split('.')[0] in {'numpy','cupy','netCDF4','xarray','scipy'}),"
        "'banner': p.banner()}))"
    )
    environment = dict(os.environ)
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONPATH"] = str(WORKTREE)
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        cwd=str(WORKTREE.parent), env=environment, timeout=120)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert Path(payload["file"]).resolve() == WORKTREE / "gpuwm" \
        / "provenance.py", "the child imported a different tree; void"
    assert payload["heavy"] == []
    assert payload["banner"].startswith("gpuwm")
