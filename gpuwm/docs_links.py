"""Links into the public repository's tree for the installed release.

THE BREAKAGE THIS PREVENTS: the refusal and help texts that point a wheel
user at a document (a wheel carries no docs tree) linked the repository's
``main`` branch. ``main`` moves with every release, so a site that pins
2.9.0 for a season is sent to the instructions of whatever release came
after it, and a section a refusal names can be renamed or gone. The link
names the tag of the installed release instead, the same tree the README's
links name (``tools/release/bump_version.py --check`` holds those). Only an
uninstalled source tree or a development build, which has no tag, falls
back to ``main``.
"""

from __future__ import annotations

import re

REPOSITORY_URL = "https://github.com/FahrenheitResearch/arwen"

_RELEASE = re.compile(r"\d+\.\d+\.\d+")


def release_ref(version: str | None = None) -> str:
    """The git ref of the installed release: ``vX.Y.Z``, or ``main`` without one."""
    if version is None:
        from gpuwm import __version__ as version
    return f"v{version}" if _RELEASE.fullmatch(str(version)) else "main"


def tree_url(kind: str = "blob", version: str | None = None) -> str:
    """``<repository>/<kind>/<ref>/``, ready for a repository-relative path."""
    return f"{REPOSITORY_URL}/{kind}/{release_ref(version)}/"
