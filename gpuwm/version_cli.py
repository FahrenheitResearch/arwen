"""``gpuwm version``: which code is actually running, and will pip move it.

A field user upgraded repeatedly and nothing changed.  He had run
``pip install -e ~/gpuwm`` months earlier, at 1.6.2; the editable
install's path entry shadows every wheel pip puts in site-packages, so
each ``pip install --upgrade gpuwm`` succeeded, reported a new version
in its own output, and left the 1.6.2 source tree serving every import.
Nothing in the product told him which code he was running, and the one
number he could see -- ``gpuwm.__version__`` -- is read from
distribution METADATA and therefore reported the number pip had just
written, not the number of the code doing the reporting.

So this command answers four questions the version number alone cannot:

  * WHICH version -- taken from the distribution that actually provides
    the imported package (``installed_distribution`` locates the package
    file that was imported), never from whichever ``.dist-info`` happens
    to carry the name;
  * WHERE the running code lives -- a site-packages wheel, or a source
    tree, with the path;
  * WHETHER pip can move it -- an editable install cannot be upgraded by
    ``pip install --upgrade``, and saying so is the whole point;
  * WHAT the index has -- one PyPI lookup, so "am I behind" does not
    require a second tool.

The network is strictly optional and strictly last.  Every local line is
printed BEFORE the lookup is attempted, the lookup has a short timeout,
and any failure at all -- offline, proxy, DNS, a 404, a slow index --
prints nothing rather than an error.  A version command that cannot
answer on a plane is a broken version command.

Like :mod:`gpuwm.update_cli`, this prints and never executes: replacing
a package's files under the process importing them is how a
half-upgraded install happens.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

#: The index lookup is a courtesy, not a dependency.  Two seconds is
#: long enough for a warm connection and short enough that a reader on a
#: dead network sees the local answer without noticing the wait.
PYPI_TIMEOUT_S = 2.0

# git is no longer run from this module at all; :mod:`gpuwm.provenance`
# owns every git question this product asks about its own tree, and its
# GIT_TIMEOUT_S is the one that applies.


def _direct_url(distribution) -> dict:
    """``direct_url.json`` for a distribution, or ``{}``.

    Delegates to :mod:`gpuwm.provenance`, which is the one resolver for
    "which tree is executing".  Kept under this name because it is what
    this module's readers and tests already call.
    """

    from gpuwm.provenance import direct_url

    return direct_url(distribution)


def _git_identity(root: Path) -> dict:
    """``{"commit": ..., "branch": ...}`` for a checkout, or ``{}``.

    A thin adapter over :func:`gpuwm.provenance.git_identity`, which
    asks git ONE question (``status --porcelain=v2 --branch``) instead
    of the two ``rev-parse`` calls this used to make, and which is the
    single implementation of every guard that matters: no git on PATH, a
    git that hangs, a directory that is not a repository, a repository
    with no commits, a detached HEAD, and -- the one that would
    otherwise bind a stranger's history -- a venv created inside
    somebody ELSE's repository.

    The shape stays ``{}``-for-absent and omits ``branch`` on a detached
    HEAD, because :func:`headline` distinguishes those and this module's
    output is pinned by tests.
    """

    from gpuwm.provenance import git_identity

    found = git_identity(root)
    if not found:
        return {}
    identity = {"commit": found["commit"]}
    if found.get("branch"):
        identity["branch"] = found["branch"]
    return identity


def describe_install(package_root: Path, distribution) -> dict:
    """What an install IS, given where its code lives and what claims it.

    The pure half, separated from the machine, so a test can build BOTH
    arguments out of real files -- a real package directory and a real
    ``importlib.metadata`` distribution reading a real ``.dist-info`` --
    instead of asserting against whatever happens to be installed on the
    box running the suite.  That mattered here: the reference box has a
    stale gpuwm whose ``__file__`` is ``None``, which is a shape no
    fixture would have thought to invent.

    ``editable`` is decided by two independent signals because one is
    not enough: PEP 610's ``dir_info.editable`` covers a modern
    ``pip install -e``, and "the package is not inside the directory its
    own metadata lives in" covers legacy ``setup.py develop``, a
    hand-written ``.pth``, and anything else that makes an import
    resolve outside site-packages while a distribution still claims it.
    Either one means pip cannot replace this code by upgrading.
    """

    package_root = Path(package_root).resolve()
    source_root = package_root.parent
    shape: dict = {
        "package_root": package_root,
        "source_root": source_root,
        "distribution": None,
        "version": None,
        "editable": False,
        "site_dir": None,
        "git": _git_identity(source_root),
    }
    if distribution is None:
        # Nothing pip knows about provides the code that is running:
        # a plain source tree on PYTHONPATH, or a wheel that a checkout
        # is shadowing.  Reporting gpuwm.__version__ here is exactly the
        # trap this command exists to close -- that number would come
        # from some OTHER distribution's metadata.
        return shape
    shape["distribution"] = str(distribution.metadata["Name"])
    shape["version"] = str(distribution.metadata["Version"])
    try:
        site_dir = Path(distribution.locate_file("")).resolve()
        shape["site_dir"] = site_dir
        outside = not package_root.is_relative_to(site_dir)
    except (OSError, ValueError):
        outside = False
    marked = bool(_direct_url(distribution).get("dir_info", {})
                  .get("editable", False))
    shape["editable"] = bool(marked or outside)
    return shape


def install_shape() -> dict:
    """:func:`describe_install` for the gpuwm that is actually imported.

    The distribution is located by
    :func:`gpuwm.provenance.providing_distribution` rather than by
    :func:`gpuwm.runtime_manifest.installed_distribution`, which cannot
    answer for the very case this command exists to report.  A PEP 660
    editable install puts no package in ``site-packages`` -- a ``.pth``
    redirects the import -- so ``locate_file`` names a file that is not
    there and the match fails.  Measured on the reference box: an
    editable ``gpuwm 1.8.7`` pointing at its own source tree was
    reported as "no installed distribution provides it", which dropped
    the version from the headline of the one command whose job is to
    print it.
    """

    import gpuwm
    from gpuwm.provenance import providing_distribution

    location = getattr(gpuwm, "__file__", None)
    if location is None:
        # A namespace package, or a half-removed install: there is no
        # single file to point at, so there is no location to report.
        # Seen in the wild on the reference box, from a stale wheel.
        location = Path(gpuwm.__path__[0]) / "__init__.py"
    package_root = Path(location).resolve().parent
    try:
        distribution = providing_distribution(package_root)
    except Exception:                                   # noqa: BLE001
        distribution = None
    return describe_install(package_root, distribution)


def headline(shape: dict | None = None) -> str:
    """One line: the version, where it lives, and its git identity.

    Also used as ``gpuwm doctor``'s header, so that the report which
    diagnoses an estate says up front WHICH copy of gpuwm produced it.
    """

    shape = install_shape() if shape is None else shape
    if shape["distribution"] is None:
        lead = "gpuwm"
        where = [f"source tree at {shape['source_root']}"]
    else:
        lead = f"gpuwm {shape['version']}"
        where = [f"editable source at {shape['source_root']}"
                 if shape["editable"]
                 else f"installed wheel at {shape['package_root']}"]
    git = shape["git"]
    if git.get("commit"):
        branch = git.get("branch")
        where[0] += (f", git {git['commit']}"
                     + (f" on {branch}" if branch and branch != "HEAD"
                        else " on a detached HEAD"))
    if shape["distribution"] is None:
        where.append("no installed distribution provides it")
    return f"{lead} ({'; '.join(where)})"


def pypi_latest(distribution: str, *,
                timeout: float = PYPI_TIMEOUT_S) -> str | None:
    """The index's latest version, or ``None`` for ANY reason.

    Offline, proxied, rate-limited, renamed, slow, or serving something
    that is not the JSON we expect: all of them are ``None``.  This is
    a courtesy line on an informational command, and there is no failure
    here worth showing a reader instead of their own version.
    """

    url = f"https://pypi.org/pypi/{distribution}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8-sig"))
        latest = payload["info"]["version"]
    except Exception:                                   # noqa: BLE001
        return None
    return str(latest) if latest else None


def _is_behind(installed: str | None, latest: str | None) -> bool | None:
    """``True``/``False``, or ``None`` when the two cannot be compared."""

    if not installed or not latest:
        return None
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(installed) < Version(latest)
        except InvalidVersion:
            return None
    except ImportError:
        # packaging is not a declared dependency.  Without it the only
        # honest comparison is equality: "different" is not "behind".
        return None if installed == latest else True


def local_report(shape: dict | None = None) -> list[str]:
    """Every line that needs no network, built so a test can read them."""

    shape = install_shape() if shape is None else shape
    lines = [headline(shape)]
    if shape["distribution"] is None:
        lines.append(
            "  Nothing pip knows about provides this code, so "
            "`pip install --upgrade` would install a SECOND copy beside "
            "it rather than replace it.")
        lines.append(f"  Update the tree at {shape['source_root']} the way "
                     "you obtained it.")
        return lines
    if shape["editable"]:
        lines.append(
            "  This is an EDITABLE install: the source tree above is what "
            "every `import gpuwm` resolves to, and it shadows any wheel "
            "pip puts in site-packages.")
        lines.append(
            "  `pip install --upgrade "
            f"{shape['distribution']}` will report success and change "
            "NOTHING about the code that runs.")
        if shape["git"].get("commit"):
            lines.append(
                f"  Update it with: git -C {shape['source_root']} pull")
        else:
            lines.append(f"  Update the tree at {shape['source_root']} the "
                         "way you obtained it.")
    else:
        from gpuwm.update_cli import upgrade_command

        lines.append(f"  Installed as a wheel; upgrade with: "
                     f"{upgrade_command(shape['distribution'])}")
    return lines


def pypi_report(shape: dict, *, timeout: float = PYPI_TIMEOUT_S) -> list[str]:
    """The staleness line, or no lines at all.  Never raises."""

    if shape["distribution"] is None:
        return []
    latest = pypi_latest(shape["distribution"], timeout=timeout)
    if latest is None:
        return []
    behind = _is_behind(shape["version"], latest)
    line = f"  PyPI latest is {latest}"
    if behind is False:
        return [f"{line} -- this install is current."]
    if behind is None:
        return [f"{line} (installed: {shape['version']})."]
    if shape["editable"]:
        return [f"{line}, and this install will NOT get there via pip; "
                "see the update line above."]
    return [f"{line} -- this install is behind it."]


def version_main(args=None) -> int:
    """Print local lines first, then try the index.  Never fails."""

    shape = install_shape()
    for line in local_report(shape):
        print(line, flush=True)
    if getattr(args, "offline", False):
        return 0
    for line in pypi_report(shape, timeout=getattr(
            args, "pypi_timeout", PYPI_TIMEOUT_S)):
        print(line)
    return 0


def register_cli(subparsers):
    parser = subparsers.add_parser(
        "version",
        help="which gpuwm is actually running: version, import location, "
             "whether it is an editable install pip cannot upgrade, its "
             "git identity, and how that compares to PyPI")
    parser.add_argument(
        "--offline", action="store_true",
        help="skip the PyPI lookup entirely (it is already skipped "
             "silently whenever the network does not answer)")
    parser.add_argument(
        "--pypi-timeout", type=float, default=PYPI_TIMEOUT_S,
        metavar="SECONDS", dest="pypi_timeout",
        help=f"seconds to wait for the index (default {PYPI_TIMEOUT_S})")
    parser.set_defaults(func=version_main)
    return parser
