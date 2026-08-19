"""One answer to "which tree is executing", for every entry point.

A version number cannot answer that question, and this project has now
paid for the difference more than once.

  * ``gpuwm.__version__`` is read from distribution METADATA.  Metadata
    is a claim made by a ``.dist-info`` directory, not by the code that
    is running.  Measured on the reference box: an editable
    ``gpuwm-1.8.7`` dist-info points at one source tree while a worktree
    elsewhere serves the imports -- and ``gpuwm.__version__`` cheerfully
    reports ``1.8.7`` to the worktree's process.  The number is not
    wrong so much as it is *borrowed*: it describes a different tree.
  * A bare ``python -c "import gpuwm"`` run from a source directory
    imports THAT directory and reports whatever metadata it can find.
    Two false version readings came from exactly this.
  * A user reports plots labelled 1.6.2 while believing they installed
    1.8, and until this module existed there was no way to tell them
    whether the label was honest.

So :func:`resolve` answers the five things a version number cannot, for
the process that is asking:

  1. WHERE the imported ``gpuwm`` package actually lives on disk;
  2. WHAT KIND of install that is -- a wheel, an editable install, or a
     plain source tree;
  3. its git identity when it is a checkout -- branch, short sha, and
     whether the working tree is dirty;
  4. the version distribution metadata reports;
  5. the version the CODE itself declares, and whether the two AGREE.

Point 5 is the one that matters.  The other four describe an install;
only the fifth catches an install that is lying about itself, which is
the failure that has actually cost this project runs.

Reuse, not a fifth mechanism.  This project already resolves pieces of
this in several places and they stay authoritative:
:func:`gpuwm.runtime_manifest.provenance` is still the receipt-binding
identity ladder (sealed manifest, then checkout, then wheel RECORD),
:func:`gpuwm.runtime_manifest.git_checkout_root` is still the one
"is this a checkout of THIS tree" predicate -- reused below, including
its refusal of a venv nested in somebody else's repository -- and
:func:`gpuwm.runtime_manifest.wheel_record_identity` is still how a
wheel's bytes are bound.  What was missing was a single cheap resolver
that names the running tree in one call, and one place where the
metadata-versus-code question is asked at all.

Cost and safety.  Two short subprocesses at most (git is asked exactly
one question, ``status --porcelain=v2 --branch``, which carries the
branch, the commit and the dirt together), one small TOML read, and the
answer is cached for the life of the process.  Nothing here imports
numpy, cupy, or any ingest module, so a startup banner can call it
before anything heavy loads.  Nothing here raises: a machine with no
git, a wheel with no repository, an unreadable ``pyproject.toml`` and a
half-removed install are all normal, and each degrades to a stated
absence rather than an exception on somebody's first run.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

#: The serialised shape's schema name.  Receipts embed :meth:`
#: Provenance.as_dict`, so the shape is versioned like every other
#: document this product writes.
PROVENANCE_SCHEMA = "gpuwm-provenance-v1"

#: What ``gpuwm/__init__.py`` reports when no distribution metadata
#: exists at all.  Distinguishing it from a real number is what makes a
#: BORROWED version detectable: a bare source tree honestly says it does
#: not know, while a borrowed one confidently states someone else's.
UNKNOWN_VERSION = "0+unknown"

#: Install kinds :func:`describe_provenance` can report.
INSTALL_KINDS = ("wheel", "editable", "source-tree")

#: Distribution names that may legitimately publish the ``gpuwm``
#: package.  Mirrors ``runtime_manifest._CANDIDATE_DISTRIBUTIONS``;
#: ``rw-wps`` is the preprocessing-only wheel that ships the same
#: package directory under a second name.
CANDIDATE_DISTRIBUTIONS = ("gpuwm", "rw-wps")

#: git is asked one short question and is allowed to be missing, hung,
#: or pointed at something that is not a repository.
GIT_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    """What the running process is, and whether it is self-consistent.

    Frozen because a receipt embeds it and a banner prints it: the two
    must be the same statement, and a mutable one invites a caller to
    "correct" a field after the fact.
    """

    package_path: str
    source_root: str
    install_kind: str
    distribution_name: str | None = None
    metadata_version: str | None = None
    reported_version: str | None = None
    metadata_is_borrowed: bool = False
    code_version: str | None = None
    code_version_source: str | None = None
    versions_agree: bool | None = None
    disagreement: str | None = None
    git: dict | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- the serialisable form ------------------------------------------
    def as_dict(self) -> dict:
        """A JSON-safe mapping suitable for embedding in a receipt.

        Every value is a string, bool, int, None, list or dict -- no
        ``Path`` objects, which is the mistake that makes a receipt
        writer fail at ``json.dumps`` months after the field was added,
        and no platform-dependent spelling beyond the paths themselves.
        """

        return {
            "schema": PROVENANCE_SCHEMA,
            "package_path": self.package_path,
            "source_root": self.source_root,
            "install_kind": self.install_kind,
            "distribution_name": self.distribution_name,
            "metadata_version": self.metadata_version,
            "reported_version": self.reported_version,
            "metadata_is_borrowed": self.metadata_is_borrowed,
            "code_version": self.code_version,
            "code_version_source": self.code_version_source,
            "versions_agree": self.versions_agree,
            "disagreement": self.disagreement,
            "git": dict(self.git) if self.git else None,
            "notes": list(self.notes),
        }

    # -- the human form --------------------------------------------------
    def banner(self) -> str:
        """One line naming the running tree, for a startup banner.

        Always one line, always leads with the version a reader is about
        to quote back, and never hides a disagreement at the end of a
        paragraph -- if the install is inconsistent, that clause is in
        this line or the line has failed at its job.
        """

        version = self.reported_version or self.metadata_version
        lead = f"gpuwm {version}" if version else "gpuwm"
        where = {
            "wheel": f"installed wheel at {self.package_path}",
            "editable": f"editable source at {self.source_root}",
        }.get(self.install_kind, f"source tree at {self.source_root}")
        parts = [where]
        git = self.git or {}
        if git.get("commit"):
            branch = git.get("branch")
            state = "dirty" if git.get("dirty") else "clean"
            parts.append(
                f"git {git['commit']} on "
                f"{branch if branch else 'a detached HEAD'} ({state})")
        line = f"{lead} -- {', '.join(parts)}"
        if self.disagreement:
            line += f"  !! {self.disagreement}"
        return line

    @property
    def is_consistent(self) -> bool:
        """True when nothing about this install contradicts itself.

        ``versions_agree is None`` -- nothing to compare -- counts as
        consistent: an unanswerable question is not a defect, and a gate
        that refused it would refuse every clean wheel on a box with no
        git.
        """

        return self.versions_agree is not False


# ---------------------------------------------------------------------------
# The distribution that actually provides the running code
# ---------------------------------------------------------------------------

def direct_url(distribution) -> dict:
    """PEP 610 ``direct_url.json`` for a distribution, or ``{}``.

    ``dir_info.editable`` is the modern, authoritative editable marker,
    and ``url`` names the source directory the editable install points
    at -- which is the only reliable way to match an editable install to
    the code it serves (see :func:`providing_distribution`).
    """

    try:
        text = distribution.read_text("direct_url.json")
    except Exception:                                   # noqa: BLE001
        return {}
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _editable_root(distribution) -> Path | None:
    """The source directory an editable install points at, or None."""

    info = direct_url(distribution)
    if not info.get("dir_info", {}).get("editable", False):
        return None
    url = info.get("url")
    if not isinstance(url, str) or not url.startswith("file:"):
        return None
    try:
        from urllib.parse import unquote, urlparse

        path = unquote(urlparse(url).path)
        # ``file:///C:/x`` parses to ``/C:/x`` on Windows.
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return Path(path).resolve()
    except Exception:                                   # noqa: BLE001
        return None


def providing_distribution(package_path: Path):
    """The distribution that provides the package at ``package_path``.

    A generalisation of
    :func:`gpuwm.runtime_manifest.installed_distribution`, which is
    tried FIRST and left authoritative for the case it was written for:
    a wheel, where ``locate_file`` genuinely names the installed file.

    It has to be generalised because ``locate_file`` cannot answer for
    an editable install, and answers *confidently wrong*.  A PEP 660
    editable puts no package in ``site-packages`` -- a ``.pth`` redirects
    the import elsewhere -- so ``locate_file("gpuwm/__init__.py")``
    returns a ``site-packages`` path that does not exist.  Measured on
    the reference box: an editable ``gpuwm-1.8.7`` whose
    ``direct_url.json`` names a source tree reports a ``locate_file`` of
    ``...site-packages/gpuwm/__init__.py``, a file that is not there.
    ``installed_distribution()`` therefore returns ``None`` for EVERY
    editable install, including one that is serving the running import,
    and the caller concludes "no installed distribution provides this
    code" about a distribution that provides exactly this code.

    So the editable case is matched the way an editable install is
    actually wired: by asking whether the running package lives inside
    the source directory ``direct_url.json`` names.
    """

    package_path = Path(package_path).resolve()
    try:
        from gpuwm.runtime_manifest import installed_distribution

        anchor = Path(__file__).resolve().parent
        if package_path == anchor:
            # Only meaningful for the package this module belongs to;
            # installed_distribution() anchors on its own __file__.
            found = installed_distribution()
            if found is not None:
                return found
    except Exception:                                   # noqa: BLE001
        pass
    for name in CANDIDATE_DISTRIBUTIONS:
        try:
            dist = metadata.distribution(name)
        except Exception:                               # noqa: BLE001
            continue
        try:
            located = Path(dist.locate_file("gpuwm/__init__.py")).resolve()
            if located == package_path / "__init__.py":
                return dist
        except Exception:                               # noqa: BLE001
            pass
        root = _editable_root(dist)
        if root is not None:
            try:
                if package_path == root or package_path.is_relative_to(root):
                    return dist
            except Exception:                           # noqa: BLE001
                continue
    return None


# ---------------------------------------------------------------------------
# git, in exactly one subprocess
# ---------------------------------------------------------------------------

def git_identity(root: Path) -> dict | None:
    """``{branch, commit, commit_full, dirty, ...}`` for a checkout, else None.

    One subprocess, not four.  ``status --porcelain=v2 --branch`` is the
    only git command that reports the commit, the branch and the working
    tree's cleanliness together, which is what makes this affordable at
    every process start.  Its output is a stable machine format -- v2
    exists precisely so tools stop parsing human output.

    ``dirty`` means TRACKED changes: staged, unstaged, or unmerged.
    Untracked files are counted separately rather than folded in,
    because an untracked scratch file does not change which committed
    code is executing, and reporting a checkout as dirty because someone
    left a ``.log`` beside it would train readers to ignore the flag.

    Returns ``None`` -- never raises -- when git is absent, hung,
    broken, or when ``root`` is not the top level of its own repository.
    That last refusal is :func:`gpuwm.runtime_manifest.git_checkout_root`
    and it is reused deliberately: a venv created inside somebody else's
    repository must report no git identity rather than bind a stranger's
    commit to this run.
    """

    root = Path(root)
    try:
        from gpuwm.runtime_manifest import git_checkout_root

        if git_checkout_root(root) is None:
            return None
    except Exception:                                   # noqa: BLE001
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v2", "--branch"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
            check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit_full: str | None = None
    branch: str | None = None
    tracked = 0
    untracked = 0
    for line in completed.stdout.splitlines():
        if line.startswith("# branch.oid "):
            value = line[len("# branch.oid "):].strip()
            commit_full = None if value == "(initial)" else value
        elif line.startswith("# branch.head "):
            value = line[len("# branch.head "):].strip()
            branch = None if value == "(detached)" else value
        elif line[:2] in ("1 ", "2 ", "u "):
            tracked += 1
        elif line.startswith("? "):
            untracked += 1
    if commit_full is None:
        # A repository with no commits yet.  It is a checkout, but there
        # is no identity to report, and inventing one would be worse.
        return None
    return {
        "commit": commit_full[:12],
        "commit_full": commit_full,
        "branch": branch,
        "dirty": tracked > 0,
        "dirty_files": tracked,
        "untracked_files": untracked,
    }


# ---------------------------------------------------------------------------
# The version the CODE declares
# ---------------------------------------------------------------------------

def pyproject_version(source_root: Path) -> tuple[str | None, str | None]:
    """``(version, note)`` declared by ``source_root/pyproject.toml``.

    This is the only place in a source tree where a version is written
    down by hand, which is exactly what makes it the CODE's claim as
    opposed to the installed metadata's claim.  ``gpuwm/__init__.py``
    deliberately does not carry a constant -- it reads metadata -- so
    without this read there is no second opinion to compare against and
    the disagreement that matters is undetectable.

    The declared ``[project].name`` is checked against the names that
    may publish this package.  A ``gpuwm`` package directory can sit
    inside an unrelated repository that has its own ``pyproject.toml``,
    and binding a stranger's version number would be the same class of
    error as binding a stranger's git commit.
    """

    path = Path(source_root) / "pyproject.toml"
    try:
        if not path.is_file():
            return None, None
        import tomllib

        project = tomllib.loads(path.read_text(encoding="utf-8")).get(
            "project", {})
    except Exception as error:                          # noqa: BLE001
        return None, f"pyproject.toml is unreadable ({error})"
    name = project.get("name")
    if name not in CANDIDATE_DISTRIBUTIONS:
        return None, (f"{path} declares project {name!r}, which does not "
                      "publish this package, so its version is not this "
                      "code's version")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        # A dynamic version (setuptools-scm and friends) is a legitimate
        # absence, not a defect.
        return None, f"{path} declares no static [project].version"
    return version, None


# ---------------------------------------------------------------------------
# The pure resolver
# ---------------------------------------------------------------------------

def describe_provenance(package_path, distribution, *,
                        reported_version: str | None = None,
                        probe_git=git_identity) -> Provenance:
    """Resolve provenance from an explicit package path and distribution.

    The pure half, separated from the machine so a test can build BOTH
    arguments out of real files -- a real package directory, a real
    ``.dist-info`` read by a real ``PathDistribution``, a real ``git
    init`` -- instead of asserting against whatever happens to be
    installed on the box running the suite.  That separation is copied
    from :func:`gpuwm.version_cli.describe_install`, which adopted it
    after the reference box turned out to carry an install shape no
    fixture would have invented.

    ``reported_version`` is what ``gpuwm.__version__`` says in the
    process being described.  It is injected rather than read so that
    the BORROWED case -- a source tree whose ``__version__`` comes from
    a foreign ``.dist-info`` -- is reproducible in a test without
    installing anything.
    """

    package_path = Path(package_path).resolve()
    source_root = package_path.parent
    notes: list[str] = []

    # -- kind ------------------------------------------------------------
    editable_root = None
    if distribution is None:
        install_kind = "source-tree"
        distribution_name = None
        metadata_version = None
    else:
        distribution_name = str(distribution.metadata["Name"])
        metadata_version = str(distribution.metadata["Version"])
        editable_root = _editable_root(distribution)
        outside = False
        try:
            site_dir = Path(distribution.locate_file("")).resolve()
            outside = not package_path.is_relative_to(site_dir)
        except Exception:                               # noqa: BLE001
            outside = False
        # Two independent signals, because one is not enough: PEP 610
        # covers a modern `pip install -e`, and "the package is not
        # inside the directory its own metadata lives in" covers
        # setup.py develop, a hand-written .pth, and anything else that
        # resolves an import outside site-packages.
        install_kind = "editable" if (editable_root or outside) else "wheel"

    # -- git -------------------------------------------------------------
    # The EXECUTING tree first: the checkout that owns the imported
    # package is the code actually running, and that is the identity a
    # receipt must carry.  An editable install's PEP 610 root names
    # where pip installed FROM, which is a different tree exactly when a
    # parallel worktree's code shadows the install on sys.path -- the
    # case where stamping the install's commit misattributes every
    # receipt the worktree writes.  The editable root remains the
    # fallback for layouts where the package's parent is not a checkout
    # top (src/ layouts).
    try:
        git = probe_git(source_root)
    except Exception:                                   # noqa: BLE001
        git = None
    if git is None and editable_root is not None \
            and editable_root != source_root:
        try:
            git = probe_git(editable_root)
        except Exception:                               # noqa: BLE001
            git = None

    # -- the code's own claim ---------------------------------------------
    code_version: str | None = None
    code_version_source: str | None = None
    if install_kind == "wheel":
        # A wheel ships no pyproject.toml, so there is no independent
        # declaration to read.  pip wrote the code and the metadata
        # together and RECORD binds every installed file to it, so the
        # metadata IS this code's declaration -- stated explicitly
        # rather than left as a silent tautology.  A caller who wants
        # the bytes checked as well as the claim can pay for
        # runtime_manifest.wheel_record_identity(verify=True).
        code_version = metadata_version
        code_version_source = "wheel-metadata"
        notes.append("a wheel carries no independent version declaration; "
                     "pip wrote its code and metadata together")
    else:
        candidates: list[Path] = []
        for candidate in (source_root, editable_root):
            # An editable install of a flat-layout tree makes these two
            # the same directory; reading it twice would report the same
            # defect twice in `notes`.
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)
        for candidate in candidates:
            found, note = pyproject_version(candidate)
            if note:
                notes.append(note)
            if found:
                code_version = found
                code_version_source = f"{Path(candidate) / 'pyproject.toml'}"
                break

    # -- do they agree? ----------------------------------------------------
    versions_agree: bool | None = None
    disagreement: str | None = None
    borrowed = False
    if distribution is None:
        if reported_version and reported_version != UNKNOWN_VERSION:
            # Nothing pip knows about provides this code, yet
            # gpuwm.__version__ still produced a number: it came from
            # some OTHER distribution's metadata.  This is the live
            # failure -- the number describes a different tree, and it
            # is most dangerous when it happens to match.
            borrowed = True
            versions_agree = False
            disagreement = (
                f"VERSION IS BORROWED: this process reports "
                f"{reported_version!r} from distribution metadata that does "
                f"NOT provide the code at {package_path}; the running tree "
                f"declares "
                + (f"{code_version!r}" if code_version else "no version")
                + " -- the reported number describes a different install")
        else:
            notes.append("no installed distribution provides this code, so "
                         "there is no metadata version to compare")
    else:
        if code_version is not None and metadata_version is not None:
            versions_agree = code_version == metadata_version
            if not versions_agree:
                disagreement = (
                    f"VERSION DISAGREEMENT: distribution metadata says "
                    f"{metadata_version!r} but the code at {source_root} "
                    f"declares {code_version!r}; the install is stale "
                    "against its own source tree")
        if (reported_version and metadata_version
                and reported_version != metadata_version
                and versions_agree is not False):
            versions_agree = False
            disagreement = (
                f"VERSION DISAGREEMENT: this process reports "
                f"{reported_version!r} but the distribution providing its "
                f"code ({distribution_name}) is {metadata_version!r}")

    return Provenance(
        package_path=str(package_path),
        source_root=str(editable_root if editable_root is not None
                        else source_root),
        install_kind=install_kind,
        distribution_name=distribution_name,
        metadata_version=metadata_version,
        reported_version=reported_version,
        metadata_is_borrowed=borrowed,
        code_version=code_version,
        code_version_source=code_version_source,
        versions_agree=versions_agree,
        disagreement=disagreement,
        git=git,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# The live resolver
# ---------------------------------------------------------------------------

_CACHE: Provenance | None = None


def resolve(*, refresh: bool = False) -> Provenance:
    """Provenance of the ``gpuwm`` this process actually imported.

    Cached for the life of the process: the answer cannot change without
    the code changing underneath a running interpreter, and an entry
    point that calls this in a loop should not pay a subprocess each
    time.  ``refresh=True`` re-measures for the one caller that wants to
    see a working tree go dirty mid-run.

    Never raises.  A machine with no git, a half-removed install, a
    package whose ``__file__`` is ``None`` -- all seen in the wild --
    degrade to a Provenance that states what it could not determine.
    """

    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    try:
        import gpuwm

        location = getattr(gpuwm, "__file__", None)
        if location is None:
            # A namespace package or a half-removed install: there is no
            # single file to point at.  Seen on the reference box.
            location = Path(gpuwm.__path__[0]) / "__init__.py"
        package_path = Path(location).resolve().parent
        resolved = describe_provenance(
            package_path, providing_distribution(package_path),
            reported_version=getattr(gpuwm, "__version__", None))
    except Exception as error:                          # noqa: BLE001
        # A banner must never be the reason a run dies.
        resolved = Provenance(
            package_path="<unresolved>", source_root="<unresolved>",
            install_kind="source-tree",
            notes=(f"provenance could not be resolved: {error!r}",))
    _CACHE = resolved
    return resolved


def banner() -> str:
    """The one-line startup form for the running process."""

    return resolve().banner()


__all__ = [
    "CANDIDATE_DISTRIBUTIONS", "GIT_TIMEOUT_S", "INSTALL_KINDS",
    "PROVENANCE_SCHEMA", "Provenance", "UNKNOWN_VERSION", "banner",
    "describe_provenance", "direct_url", "git_identity",
    "providing_distribution", "pyproject_version", "resolve",
]
