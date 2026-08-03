"""``gpuwm update``: the upgrade command for THIS environment, printed.

A reader who has a version and wants the next one reaches for ``gpuwm
update`` before they reach for a README, and until now that gesture
answered ``invalid choice: 'update'`` -- an error message that says the
tool does not know the word, not that the upgrade is ``pip``.  A field
run on the shipped 1.5.0 wheel filed exactly that.

This command PRINTS and never executes.  Upgrading the interpreter's own
packages from inside a process that interpreter is running is how a
half-replaced install happens: pip would be rewriting files under the
running process's import path, and the failure mode is a traceback in a
module that no longer matches the one already imported.  So the whole
value here is discovery -- naming the distribution this install actually
came from, and the exact command for this interpreter -- and the reader
runs it from their shell, where it is safe.

The distribution is resolved, not assumed.  Two distributions publish
the ``gpuwm`` package (the full wheel and the preprocessing-only one),
and :func:`gpuwm.runtime_manifest.installed_distribution` already answers
"which one installed the code that is running" by locating the package
file that was actually imported.  Printing ``pip install --upgrade
gpuwm`` to somebody whose install came from the other distribution would
be an instruction that installs a second copy rather than upgrading the
one they have.

Downloaded assets are not part of the wheel and do not move when it is
replaced: the bridges and the physics tables are staged under the user
data directory precisely so an upgrade cannot delete them.  That is
stated here because the alternative -- a reader who assumes a 315 MiB
download has to be repeated -- costs the download.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path


def _preserved_asset_dirs() -> list[Path]:
    """User-level directories a wheel replacement does not touch.

    Resolved through the modules that own them rather than re-spelled
    here, so a staging location that moves moves in one place.  A home
    directory that cannot be resolved at all is not an error worth a
    nonzero exit on an informational command; the line is simply not
    printed.
    """

    directories: list[Path] = []
    try:
        from gpuwm.bridges import default_bridge_dir

        directories.append(default_bridge_dir())
    except (ImportError, RuntimeError, OSError):  # pragma: no cover
        pass
    try:
        from gpuwm.physics_compat import user_thompson_table_root

        directories.append(user_thompson_table_root())
    except (ImportError, RuntimeError, OSError):  # pragma: no cover
        pass
    return directories


def upgrade_command(distribution: str) -> str:
    """The upgrade line for THIS interpreter, pasteable as printed.

    ``sys.executable -m pip`` rather than a bare ``pip``: a machine with
    several interpreters has several ``pip`` entry points, and the one
    first on PATH is regularly not the one that owns the environment the
    reader just ran ``gpuwm`` from.  Naming the interpreter removes the
    guess.
    """

    return (f"{shlex.quote(sys.executable)} -m pip install --upgrade "
            f"{distribution}")


def update_report() -> list[str]:
    """The printed lines, as a list, so a test can read them.

    Built rather than printed so the composition is inspectable without
    capturing stdout, which is the same shape ``doctor`` uses for its
    own report.
    """

    from gpuwm import __version__
    from gpuwm.runtime_manifest import installed_distribution

    distribution = installed_distribution()
    lines: list[str] = []
    if distribution is None:
        # A source tree that was never installed.  There is no
        # distribution to upgrade and no version to compare against, so
        # saying "run pip install --upgrade gpuwm" would install a
        # SECOND copy beside the checkout that is running -- the exact
        # confusion this command exists to prevent.
        package_root = Path(__file__).resolve().parent
        lines.append(
            f"gpuwm update: this gpuwm is running from {package_root}, "
            "which no installed distribution provides.")
        lines.append(
            "  There is nothing for pip to upgrade here: update the "
            "source tree the way you obtained it.")
        lines.append(
            "  # `pip install --upgrade` would install a SECOND copy "
            "beside this one, not replace it.")
        return lines

    name = distribution.metadata["Name"]
    lines.append(f"gpuwm update: {name} {__version__} is installed; "
                 "this command prints the upgrade and runs nothing.")
    lines.append(f"  upgrade: {upgrade_command(name)}")
    lines.append("  # the short form, when the right pip is on PATH:")
    lines.append(f"  #   pip install --upgrade {name}")
    lines.append(
        "  Run it from your shell, not from here: replacing a package's "
        "files under the process importing them is how a half-upgraded "
        "install happens.")
    preserved = _preserved_asset_dirs()
    if preserved:
        lines.append(
            "  Downloaded assets are NOT part of the wheel and survive "
            "the upgrade:")
        for directory in preserved:
            lines.append(f"    {directory}")
        lines.append(
            "  # `gpuwm doctor` re-checks the whole estate afterwards; "
            "`gpuwm setup` stages anything it names.")
    return lines


def update_main(args=None) -> int:
    """Print the report.  No network, no subprocess, no filesystem write."""

    for line in update_report():
        print(line)
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "update",
        help="print the exact upgrade command for this install (the "
             "distribution that provides it and the interpreter running "
             "it); prints only -- nothing is downloaded or replaced")
    parser.set_defaults(func=update_main)
    return parser


__all__ = ["register_cli", "update_main", "update_report",
           "upgrade_command"]
