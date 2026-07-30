"""The gpuwm package.

``__version__`` is read from the installed distribution's metadata, not
declared here.  A hand-maintained constant is a promise to update two
files at every cut, and that promise was broken: the wheel shipped
``0.1.1`` for four releases, and 1.1.1's prepared-cache provenance
refusal used it to tell a 1.1.1 user "this is gpuwm 0.1.1" -- a sentence
whose whole job is to name the release that is speaking.

The fallback covers the one case metadata cannot answer: a source tree
that was never installed, where there is no distribution to ask.  It
says so rather than inventing a number.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

#: The distribution this package is published as.
DISTRIBUTION_NAME = "gpuwm"

try:
    __version__ = _distribution_version(DISTRIBUTION_NAME)
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0+unknown"

__all__ = ["DISTRIBUTION_NAME", "__version__"]
