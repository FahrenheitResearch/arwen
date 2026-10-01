"""The gpuwm package.

``__version__`` is read from the installed distribution's metadata, not
declared here.  A hand-maintained constant is a promise to update two
files at every cut, and that promise was broken: the wheel shipped
``0.1.1`` for four releases, and 1.1.1's prepared-cache provenance
refusal used it to tell a 1.1.1 user "this is gpuwm 0.1.1" -- a sentence
whose whole job is to name the release that is speaking.

The metadata read is the one belonging to the distribution that installed
THIS package, found by its own files (:func:`owning_distribution`), not
the first distribution of a matching name on the path.  Asked by name, a
staged ``rw-wps`` package, or any gpuwm run beside a second gpuwm
install (a venv and a ``--user`` install), read the other install's
number, and the rw-wps runtime check then refused it ("rw-wps version
mismatch").

The fallbacks cover what ownership cannot answer.  A tree no installed
distribution provides reads the first matching name on the path, as
before, which is what lets :mod:`gpuwm.provenance` call that number
BORROWED and :mod:`gpuwm.provenance_gate` refuse it when it disagrees
with the tree's own declaration.  With no distribution to ask at all it
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

from importlib import metadata as _metadata  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

#: The distribution this package is published as.
DISTRIBUTION_NAME = "gpuwm"

#: Every distribution that ships this package directory, in the order the
#: version is read: gpuwm, then the standalone RW-WPS preparation package
#: built from the same source.  Reading gpuwm alone, a clean rw-wps
#: install reported 0+unknown and its own runtime check refused it
#: ("rw-wps version mismatch: metadata=2.8.1, module=0+unknown").
DISTRIBUTION_NAMES = (DISTRIBUTION_NAME, "rw-wps")

#: The RECORD row a distribution that installed this package carries for
#: the package's own ``__init__.py``.  Both names ship the same package
#: directory, so this one row is what tells their installs apart.
PACKAGE_INIT = "gpuwm/__init__.py"


def candidate_distributions() -> list:
    """Every distribution on the path that may publish this package.

    In the order the version is read: each of :data:`DISTRIBUTION_NAMES`
    in turn and, under one name, every distribution the path carries,
    first on the path first.  ``importlib.metadata.version(name)`` stops
    at the first of a name, so a second install of it further down the
    path -- a venv's beside a ``--user`` one -- was never seen, even when
    it was the one serving the import.
    """

    found: list = []
    for name in DISTRIBUTION_NAMES:
        try:
            found.extend(_metadata.distributions(name=name))
        except Exception:                               # noqa: BLE001
            # An unreadable path entry must not fail ``import gpuwm``.
            continue
    return found


def _resolved(path) -> _Path | None:
    try:
        return _Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def installs_package(distribution, init_file, *,
                     entry: str = PACKAGE_INIT) -> bool:
    """True when ``distribution``'s own files put ``init_file`` where it is.

    ``init_file`` is a resolved path.  A wheel, and ``pip install
    --target``, lists every file it wrote in ``RECORD``: the row for
    ``entry``, located the way the distribution locates its files, must
    BE ``init_file``.  A ``.dist-info`` whose RECORD has no such row
    installed no such file however close by it sits -- an editable
    install's redirect, or metadata left beside another distribution's
    package -- and does not own it.  Metadata with no RECORD at all (an
    ``.egg-info``) sits beside the package it describes, so the file it
    locates for ``entry`` is its whole claim.
    """

    try:
        located = _resolved(distribution.locate_file(entry))
    except Exception:                                   # noqa: BLE001
        return False
    if located is None or located != init_file:
        return False
    try:
        record = distribution.read_text("RECORD")
    except Exception:                                   # noqa: BLE001
        record = None
    if record is None:
        return True
    if entry not in record.replace("\\", "/"):
        return False
    import csv
    import io

    return any(row and row[0].replace("\\", "/") == entry
               for row in csv.reader(io.StringIO(record)))


def editable_source_root(distribution) -> _Path | None:
    """The source directory a PEP 610 editable install points at, or None.

    ``direct_url.json``'s ``dir_info.editable`` marks the install, and its
    ``file:`` url names the tree.  An editable install's RECORD describes
    only its redirect, so this is the one claim it makes on the code it
    serves.
    """

    try:
        text = distribution.read_text("direct_url.json")
    except Exception:                                   # noqa: BLE001
        return None
    if not text:
        return None
    import json

    try:
        info = json.loads(text)
    except ValueError:
        return None
    if not isinstance(info, dict):
        return None
    dir_info = info.get("dir_info")
    if not isinstance(dir_info, dict) or not dir_info.get("editable", False):
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
        return _Path(path).resolve()
    except Exception:                                   # noqa: BLE001
        return None


def owning_distribution(package_dir=None, *, candidates=None):
    """The installed distribution that provides the package, or None.

    ``package_dir`` is a ``gpuwm`` package directory, this running one by
    default.  Every candidate is asked for exact evidence before any is
    asked for loose evidence:

    1. its own files install ``package_dir/__init__.py``
       (:func:`installs_package`), which is a wheel, a ``--target``
       install or an ``.egg-info`` beside its source;
    2. it is an editable install whose source directory holds
       ``package_dir`` (:func:`editable_source_root`).

    In that order because a venv created inside an editable install's
    checkout puts a wheel's package inside the editable root: the wheel's
    RECORD is the proof, the editable root only a neighbourhood.
    """

    init_file = _resolved(__file__ if package_dir is None
                          else _Path(package_dir) / "__init__.py")
    package = _resolved(_Path(__file__).parent if package_dir is None
                        else package_dir)
    if init_file is None or package is None:
        return None
    if candidates is None:
        candidates = candidate_distributions()
    for distribution in candidates:
        if installs_package(distribution, init_file):
            return distribution
    for distribution in candidates:
        root = editable_source_root(distribution)
        if root is not None and (package == root
                                 or package.is_relative_to(root)):
            return distribution
    return None


def version_distribution():
    """The distribution ``__version__`` is read from, or None.

    The one that provides this package (:func:`owning_distribution`).
    Only when none does, the first of :data:`DISTRIBUTION_NAMES` on the
    path, the order the version was always read in: the number then
    describes another install, and :mod:`gpuwm.provenance` reports it as
    borrowed rather than this function hiding it.
    """

    candidates = candidate_distributions()
    owner = owning_distribution(candidates=candidates)
    if owner is not None:
        return owner
    return candidates[0] if candidates else None


def _installed_version() -> str:
    distribution = version_distribution()
    if distribution is not None:
        try:
            found = distribution.version
        except Exception:                               # noqa: BLE001
            found = None
        if found:
            return str(found)
    return "0+unknown"  # an uninstalled source tree


__version__ = _installed_version()

# Before any gpuwm module imports CuPy: on Windows with the CUDA 12 toolkit
# wheels, CuPy 14.2 warns on import that it cannot find CUDA when it can,
# and that line headed every failed command's text in the desktop.  See
# gpuwm/cupy_windows_warning.py.
from gpuwm import cupy_windows_warning as _cupy_windows_warning  # noqa: E402

_cupy_windows_warning.quiet_false_cuda_path_warning()

__all__ = ["DISTRIBUTION_NAME", "DISTRIBUTION_NAMES", "LAUNCH_MONOTONIC",
           "LAUNCH_UNIX_MS", "PACKAGE_INIT", "__version__",
           "candidate_distributions", "editable_source_root",
           "installs_package", "owning_distribution",
           "version_distribution"]
