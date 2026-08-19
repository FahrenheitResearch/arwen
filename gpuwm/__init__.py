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

import time as _time

#: The earliest instant any gpuwm code runs in this process, on both
#: clocks: :data:`LAUNCH_MONOTONIC` for durations and
#: :data:`LAUNCH_UNIX_MS` for the wall a receipt is stamped with.
#:
#: Captured HERE, on the first line of the package that any entry point
#: must import, and before this module's own metadata lookup, because
#: everything a chain wants to measure is measured FROM launch.  The
#: 2026-08-16 pre-sim audit had to answer "how long from typing the
#: command to step 1" with an external wrapper's timestamps, and the
#: reason nothing inside the product could answer it was that nothing
#: inside the product knew when it started.
#:
#: This is not the process's start: the interpreter has already booted
#: and imported this package's parents by now.  It is the earliest
#: instant gpuwm can observe, it is stated as such wherever it is
#: reported, and the difference is milliseconds.
LAUNCH_MONOTONIC = _time.monotonic()
LAUNCH_UNIX_MS = int(_time.time() * 1000)

from importlib.metadata import PackageNotFoundError  # noqa: E402
from importlib.metadata import version as _distribution_version  # noqa: E402

#: The distribution this package is published as.
DISTRIBUTION_NAME = "gpuwm"

try:
    __version__ = _distribution_version(DISTRIBUTION_NAME)
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0+unknown"

__all__ = ["DISTRIBUTION_NAME", "LAUNCH_MONOTONIC", "LAUNCH_UNIX_MS",
           "__version__"]
