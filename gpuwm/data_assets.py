"""One resolver for every packaged reference-data path under ``gpuwm/data``.

Two directories of that tree do not ship inside the ``gpuwm`` wheel any
more.  They ship in the ``gpuwm-data`` distribution, which ``gpuwm``
declares as a hard dependency at its own exact version, and this module is
the only place that knows which is which.

Why
---
PyPI rejects any file over 100 MiB.  The 2.5.0 ``gpuwm`` wheel measured
103.62 MiB (108,649,669 bytes) with the Rust bridge artifacts staged.  The
RRTMGP directory and the Thompson table directory were 64.21 MiB of that
compressed -- ``qr_acr_qsV2.dat`` at 29.79 MiB and
``rrtmgp-gas-lw-g256.nc`` at 19.69 MiB alone are half the wheel -- and
they are pure data: no code, no platform coupling, nothing a wheel tag
describes.  Moving them puts ``gpuwm`` at 39.41 MiB and the companion at
64.31 MiB, both single ``py3-none-any``-eligible files well under the cap,
with room for the native artifacts still landing on the 2.5.0 line.

Nothing else about them changed.  The companion mirrors the old layout
exactly (``gpuwm_data/data/rrtmgp/...`` for ``gpuwm/data/rrtmgp/...``), so
the same bytes reach the same call sites at the same relative path.
``tests/test_companion_distribution.py`` hashes a moved member through
this resolver against a recorded SHA-256 to keep that literal.

The shape of the rule
---------------------
:data:`COMPANION_TREES` names DIRECTORIES, never filenames.  An
enumeration of files here would drift behind the tree the way
``package-data`` once drifted 52 files behind it; a directory rule cannot,
because a table added beside its siblings is already covered.  Moving the
next directory out is one entry in that tuple plus the ``git mv`` -- no
new call site, no new code path.

The refusals
------------
Both name a concrete breakage and both end in the command that fixes it:

* companion missing -- every radiation scheme and every Thompson run
  cannot read its tables, which on a bare ``pip install gpuwm`` should be
  impossible (it is a hard dependency), so the case this catches is an
  install someone edited: ``pip uninstall gpuwm-data``, a partially
  restored venv, a vendored tree copied without it.
* version skew -- the two distributions are cut from one commit and
  pinned ``==``, so a mismatch means a hand-installed sibling.  The tables
  are versioned data: reading 2.4.1's RRTMGP k-distribution under 2.5.0's
  radiation code is a silently different numerical setup, not a missing
  file, and that is exactly the class of error a certification capsule
  cannot see.
"""

from __future__ import annotations

from pathlib import Path

#: PyPI name of the companion distribution, spelled once.
COMPANION_DISTRIBUTION = "gpuwm-data"

#: Import name of the package inside it.
COMPANION_PACKAGE = "gpuwm_data"

#: ``gpuwm/data``-relative directories the companion owns, as POSIX-style
#: relative paths.  Everything else under ``gpuwm/data`` still ships
#: inside the ``gpuwm`` wheel and is resolved by :func:`package_data_root`.
#:
#: Both entries are bulk reference tables with a single loader each --
#: ``gpuwm.core.rrtmgp.DATA_DIR`` and
#: ``gpuwm.physics_compat.packaged_thompson_table_root`` -- which is why
#: they were chosen over the same number of megabytes spread across the
#: oracle directories: the split had to be measurable in the wheel and
#: invisible everywhere else.
COMPANION_TREES: tuple[str, ...] = (
    "rrtmgp",
    "thompson/tables",
)

#: In-package data root: ``<site-packages>/gpuwm/data``.
_PACKAGE_DATA_ROOT = Path(__file__).resolve().parent / "data"

#: What ``gpuwm.__version__`` reports when no distribution provides the
#: code that is running -- a source tree nobody installed.
_UNKNOWN_VERSION = "0+unknown"


def package_data_root() -> Path:
    """The ``gpuwm/data`` directory that still ships inside this wheel."""

    return _PACKAGE_DATA_ROOT


def _required_companion_version() -> str:
    """The version the installed ``gpuwm`` was built against.

    Read from distribution metadata rather than restated, because the two
    are cut from one commit at one version and ``gpuwm`` pins
    ``gpuwm-data==`` that exact string.
    """

    from gpuwm import __version__

    return __version__


def companion_install_command() -> str:
    """The one pip line that fixes a missing or skewed companion.

    Spelled ONCE, and consumed by both refusals in this module and by
    :data:`gpuwm.capabilities.COMPANION_DATA` -- so the sentence a
    reader meets at a front door and the sentence a table load raises
    cannot drift into naming different commands for the same gap.  Same
    discipline as :mod:`gpuwm.static.geog_stack`, which owns the
    geography stack's remedy for the same reason.

    Version-less when ``gpuwm`` itself is an uninstalled source tree:
    ``pip install gpuwm-data==0+unknown`` is a line that cannot be
    typed, and a remedy nobody can run is not a remedy.
    """

    required = _required_companion_version()
    if required == _UNKNOWN_VERSION:
        return f"pip install {COMPANION_DISTRIBUTION}"
    return f"pip install {COMPANION_DISTRIBUTION}=={required}"


def _refuse_missing(detail: str) -> "ModuleNotFoundError":
    """The refusal, carrying the MODULE that is missing in ``.name``.

    ``name=`` is not decoration.  :func:`gpuwm.cli.main` answers a
    ``ModuleNotFoundError`` by looking ``ModuleNotFoundError.name`` up in
    :data:`gpuwm.capabilities.REQUIREMENTS` and printing a refusal for
    what it finds; a raise without ``name`` resolves to nothing, falls
    through the branch and re-raises.  Measured, on an install-shaped
    tree with the companion uninstalled: `gpuwm check` ended in 15
    traceback frames at exit 1 and `gpuwm domain` in 17, both with this
    carefully-worded message as the last line of the stack.  The words
    were already right; nothing was reading them as a refusal.
    """

    if _required_companion_version() == _UNKNOWN_VERSION:
        # gpuwm is a source tree nobody installed, so the companion's
        # absence is a missing sibling directory rather than an edited
        # install, and the remedy says so.
        return ModuleNotFoundError(
            f"gpuwm REFUSES to resolve packaged reference data: the "
            f"{COMPANION_DISTRIBUTION} distribution is not importable "
            f"({detail}), and this gpuwm is an uninstalled source tree "
            f"with no sibling gpuwm-data/ directory beside it.  A "
            f"checkout carries both; this one is incomplete.  Either "
            f"restore gpuwm-data/ from the repository, or install the "
            f"companion:\n    {companion_install_command()}",
            name=COMPANION_PACKAGE)
    return ModuleNotFoundError(
        f"gpuwm REFUSES to resolve packaged reference data: the "
        f"{COMPANION_DISTRIBUTION} distribution is not importable "
        f"({detail}).  It carries the RRTMGP k-distribution and "
        f"cloud-optics tables and the Thompson microphysics lookup "
        f"tables -- so without it every radiation scheme and every "
        f"mp_physics=8/28 run fails at table load, and `gpuwm check` "
        f"cannot complete a preflight.  It is a HARD dependency of "
        f"gpuwm and a plain `pip install gpuwm` installs it; this state "
        f"means the install was edited afterwards.  Fix it:\n"
        f"    {companion_install_command()}",
        name=COMPANION_PACKAGE)


#: Where the companion's sources sit inside a source checkout, relative to
#: the repository root.  One repository, two distributions.
_CHECKOUT_RELATIVE = ("gpuwm-data", COMPANION_PACKAGE, "data")


#: The marker that says "the directory above ``gpuwm/`` is THIS project's
#: repository root".  It is the root ``pyproject.toml`` with
#: ``[project] name = "gpuwm"`` in it, and the property that makes it work
#: is that no wheel of either distribution can place such a file there:
#: setuptools ships package data, the file sits at the project root, and
#: the project root is not a package of either distribution.  Measured on
#: both built wheels -- neither has a top-level ``pyproject.toml`` member --
#: and gated in tests/test_package_data_coverage.py, which pins that no
#: declared package is rooted at the repository root.
_CHECKOUT_MARKER = "pyproject.toml"


def _is_checkout_root(candidate: Path) -> bool:
    """Whether ``candidate`` is this project's repository root.

    Deliberately not "does a pyproject.toml exist": ANY project's root
    has one of those, and an install unpacked inside somebody else's
    source tree would answer yes.  The name is what is checked.
    """

    import tomllib

    marker = candidate / _CHECKOUT_MARKER
    try:
        with marker.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, ValueError):
        # Absent, unreadable, or not TOML.  Not a checkout of this
        # project, and a malformed file is not a reason to start
        # trusting a directory.
        return False
    project = document.get("project")
    return isinstance(project, dict) and project.get("name") == "gpuwm"


def _checkout_root() -> Path | None:
    """The companion's sources beside a checkout's own ``gpuwm/``, or None.

    The second rung, and the same shape as
    :func:`gpuwm.bridges.artifact_candidates`: a checkout's own copy
    outranks nothing and answers only when the installed package cannot,
    so a working tree needs no ``pip install -e`` of the sibling before
    ``python -m gpuwm.cli`` can read a table.

    Guarded on the directory above ``gpuwm/`` being THIS PROJECT'S
    REPOSITORY, which is a different question from the one this guard
    used to ask.  It tested ``(repo_root / "gpuwm" / "__init__.py")``,
    and that file is exactly what an install and a checkout have in
    common: inside site-packages the test is TRUE, so the fallback was
    live on every installed wheel.  Reproduced end to end -- an
    install-shaped tree with a `gpuwm-data/gpuwm_data/data/rrtmgp`
    directory beside it, `gpuwm domain` opened the decoy's bytes as the
    k-distribution -- and the version check cannot catch it, because
    this rung returns before :func:`_check_version` is ever reached.  So
    unlabelled bytes become the numerical setup of the run and the
    receipt records a clean pass, which is the exact failure this module
    exists to make impossible.

    The marker is the root ``pyproject.toml`` naming ``gpuwm``.  It is in
    every checkout by construction -- it is what builds this project --
    and it is in no wheel of either distribution, because the project
    root is not a package and only package data ships.
    """

    repo_root = Path(__file__).resolve().parent.parent
    if not _is_checkout_root(repo_root):
        return None                    # installed: no checkout to fall to
    candidate = repo_root.joinpath(*_CHECKOUT_RELATIVE)
    return candidate if candidate.is_dir() else None


def companion_root() -> Path:
    """Directory the companion lays its ``gpuwm/data`` mirror out under.

    Resolved with :mod:`importlib.resources` against the installed
    package -- not by walking up from ``gpuwm/``, which would find
    nothing in a wheel install, and not by an environment variable, which
    would make "which tables did this run read" unanswerable from the
    capsule.  A source checkout falls to its own sibling directory; see
    :func:`_checkout_root` for why that cannot fire on an install.
    """

    import importlib.resources as resources

    detail: str
    try:
        anchor = resources.files(COMPANION_PACKAGE)
    except ModuleNotFoundError as error:
        detail = str(error)
    else:
        root = Path(str(anchor)) / "data"
        if root.is_dir():
            _check_version()
            return root
        detail = (f"{COMPANION_PACKAGE} imported from {anchor} but "
                  f"carries no data/ directory")
    checkout = _checkout_root()
    if checkout is not None:
        return checkout
    raise _refuse_missing(detail)


#: Set once the installed pair has been checked.  Every table load asks
#: for a directory, and an ``importlib.metadata`` lookup walks
#: site-packages; the versions of two installed distributions cannot
#: change inside one process, so asking twice buys nothing.
_VERSION_CHECKED = False


def _check_version() -> None:
    """Refuse a companion whose version is not this ``gpuwm``'s."""

    from importlib.metadata import PackageNotFoundError, version

    global _VERSION_CHECKED
    if _VERSION_CHECKED:
        return
    required = _required_companion_version()
    if required == _UNKNOWN_VERSION:
        # gpuwm itself is being read out of a source tree that was never
        # installed.  There is no version to match against, so there is
        # no skew to detect and a refusal here could name no breakage.
        return
    try:
        found = version(COMPANION_DISTRIBUTION)
    except PackageNotFoundError:
        # Importable but not installed: a source checkout on sys.path.
        # Same reasoning as above -- nothing to compare.
        return
    if found == required:
        _VERSION_CHECKED = True
        return
    raise ImportError(
        f"gpuwm REFUSES to read packaged reference data from a "
        f"mismatched companion: gpuwm {required} requires "
        f"{COMPANION_DISTRIBUTION} {required}, found {found}.  The two "
        f"are cut from one commit and pinned `==`, so this install was "
        f"edited.  The tables are versioned data, not interchangeable "
        f"files: running {required}'s radiation and microphysics against "
        f"{found}'s k-distribution and lookup tables is a different "
        f"numerical setup that produces numbers instead of an error, and "
        f"no certification capsule can see it.  Fix it:\n"
        f"    {companion_install_command()}")


class CompanionDataMissing(FileNotFoundError):
    """A companion that imports, version-checks, and lost a member.

    The third companion state, beside "missing" and "skewed", and it is
    the one the other two refusals cannot see: :func:`companion_root`'s
    check is directory-level, so an importable package carrying a
    ``data/`` directory answers even when a member a run needs is not in
    it.  Real, not hypothetical: the first public-tree CI build shipped
    a companion wheel with no ``rrtmgp/*.nc`` in it (a repository-wide
    ``*.nc`` gitignore rule swallowed them when the release snapshot was
    staged), and every front door died in a bare ``FileNotFoundError``
    out of a NetCDF open mid-preflight.

    ``FileNotFoundError`` so that ``errno``-minded callers keep working;
    its own class so :func:`gpuwm.cli.main` can print it as the refusal
    it is instead of relaying fifteen frames.
    """


def require_companion_member(relative: str | Path) -> Path:
    """Resolve a companion-owned path and refuse BY NAME if it is absent.

    The named refusal for the state :class:`CompanionDataMissing`
    documents.  Loaders that open companion members directly call this
    instead of joining onto a directory, so an incomplete companion is a
    sentence with the member's name and the pip line -- at the front
    door, before any traceback -- rather than a NetCDF open error.
    """

    posix = str(relative).replace("\\", "/").strip("/")
    path = data_path(posix)
    if path.is_file():
        return path
    if _required_companion_version() == _UNKNOWN_VERSION:
        # An uninstalled source tree: the sibling checkout directory is
        # answering, and the file is gone from it.
        raise CompanionDataMissing(
            f"gpuwm REFUSES to read packaged reference data: {posix} is "
            f"absent from the {COMPANION_DISTRIBUTION} data tree at "
            f"{path.parent}.  This gpuwm is an uninstalled source tree "
            f"answering out of its sibling gpuwm-data/ directory, and a "
            f"checkout always carries this member -- this one lost it.  "
            f"Restore the file from the repository, or install the "
            f"companion:\n    {companion_install_command()}")
    remedy = companion_install_command().replace(
        "pip install", "pip install --force-reinstall", 1)
    raise CompanionDataMissing(
        f"gpuwm REFUSES to read packaged reference data: {posix} is "
        f"absent from the {COMPANION_DISTRIBUTION} data tree at "
        f"{path.parent}.  The companion is importable and its version "
        f"matches, but this member is not in it, so the scheme that "
        f"reads it fails at table load -- an intact "
        f"{COMPANION_DISTRIBUTION} always carries it, which means this "
        f"install was edited or its wheel was built from an incomplete "
        f"tree.  Fix it:\n    {remedy}")


def _is_companion(relative: str) -> bool:
    posix = relative.replace("\\", "/").strip("/")
    return any(posix == tree or posix.startswith(tree + "/")
               for tree in COMPANION_TREES)


def data_path(relative: str | Path) -> Path:
    """Resolve a ``gpuwm/data``-relative path to where its bytes live.

    ``data_path("rrtmgp/rrtmgp-gas-lw-g256.nc")`` answers out of the
    companion; ``data_path("noah_tables/VEGPARM.TBL")`` answers out of
    this wheel.  Callers state the path they always stated and never
    which distribution carries it -- that is the whole point of routing
    both roots through one function.
    """

    posix = str(relative).replace("\\", "/").strip("/")
    root = companion_root() if _is_companion(posix) else _PACKAGE_DATA_ROOT
    return root.joinpath(*posix.split("/")) if posix else root


#: The two directories, resolved.  Named because they are what the
#: loaders ask for, and a loader asking for a directory should not have
#: to know the spelling of the tree that holds it.
def rrtmgp_data_dir() -> Path:
    """``gpuwm/data/rrtmgp`` as it now resolves (companion)."""

    return data_path("rrtmgp")


def thompson_table_dir() -> Path:
    """``gpuwm/data/thompson/tables`` as it now resolves (companion)."""

    return data_path("thompson/tables")


__all__ = ["COMPANION_DISTRIBUTION", "COMPANION_PACKAGE", "COMPANION_TREES",
           "CompanionDataMissing", "companion_install_command",
           "companion_root", "data_path", "package_data_root",
           "require_companion_member", "rrtmgp_data_dir",
           "thompson_table_dir"]
