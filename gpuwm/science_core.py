"""The one place the mandated science core's version window is written.

``wrf`` (pip distribution ``wrf-rust``) is the mandated science core for
gpuwm's derived quantities.  Until now four modules each carried their own
copy of the string ``0.2.35`` and compared against it with ``==``, and the
pyproject extra pinned the same version exactly.  That combination had one
consequence nobody wanted: a user who already ran a *newer* wrf-rust got it
**downgraded** by ``pip install gpuwm[render]``, and the runtime checks
refused the newer core outright if they kept it.

So the window is a range, and it lives here once:

* ``SCIENCE_CORE_FLOOR`` -- the oldest release the products are certified
  against.  Below it, refuse: the diagnostics genuinely differ.
* ``SCIENCE_CORE_CEILING`` -- the first release we do not vouch for.  0.3
  is where upstream is free to change the diagnostic surface, so the
  refusal is at the minor boundary rather than at every patch bump.
* ``SCIENCE_CORE_INSTALL_FLOOR`` -- the oldest release pip may RESOLVE.
  A different axis from the floor and it moves for a different reason:
  the floor is about what a core already on the box computes, this is
  about what the index can put there.  See its own note.

``__version__`` is deliberately NOT the authority anywhere in the tree.
wrf-rust 0.2.35 shipped with ``__version__ == "0.2.34"`` -- an attribute the
author forgot to bump -- so the installed *distribution* metadata answers
and the attribute is recorded beside it as a note.  0.2.38 and 0.2.39 do
agree with their own metadata (MEASURED), which does not make the attribute
an authority: two releases getting it right is not a guarantee, and the
check costs nothing.
"""

from __future__ import annotations

#: The pip distribution name; the import name is ``wrf``.
SCIENCE_CORE_DISTRIBUTION = "wrf-rust"

#: Oldest certified release (inclusive) -- the RUNTIME floor, judging a
#: core that is already installed.  What pip may resolve is a different
#: number: see ``SCIENCE_CORE_INSTALL_FLOOR``.
SCIENCE_CORE_FLOOR = "0.2.35"

#: First release outside the window (exclusive).
SCIENCE_CORE_CEILING = "0.3"

#: The oldest release pip may RESOLVE, and the reason the [render] extra
#: no longer carries an environment marker.
#:
#: MEASURED against the index, 2026-08-17 -- ``pypi.org/pypi/wrf-rust``,
#: 26 artifacts on 0.2.39::
#:
#:     wrf_rust-0.2.39-cp3{10,11,12,13,14}-...-macosx_10_12_x86_64.whl
#:                                          macosx_11_0_arm64.whl
#:                                          manylinux_2_17_aarch64.whl
#:                                          manylinux_2_17_x86_64.whl
#:                                          win_amd64.whl
#:     wrf_rust-0.2.39.tar.gz
#:
#: 0.2.39 is the first release with a wheel for every interpreter in the
#: supported range.  0.2.35..0.2.38 stop at cp313, so on a 3.14 box pip
#: falls back to their sdist, that sdist does not build (its pyo3 caps at
#: 3.13), and pip fails a WHOLE resolution when one requirement fails --
#: `pip install 'gpuwm[gpu-cu13,render]'`, the line install.sh itself
#: runs, installed NO gpuwm at all on weather-node-1's python3.14.4.
#: That is the concrete breakage this floor prevents, and it is an
#: install-time fact: a 0.2.38 already sitting on a 3.13 box renders
#: every product it ever did, which is why SCIENCE_CORE_FLOOR does not
#: follow this number up.
SCIENCE_CORE_INSTALL_FLOOR = "0.2.39"

#: The release the 2.5.0 suites are actually exercised against.
#:
#: This is a RECORD, not a gate.  The floor stays at 0.2.35 because no
#: breakage has been named for 0.2.35..0.2.38, and a refusal that cannot
#: name what it prevents is not a refusal.  What this constant fixes is a
#: different defect: until 2.5.0 the tree said "gpuwm depends on wrf-rust
#: 0.2.35" while the box served an *editable* checkout whose own
#: ``__version__`` read 0.2.34 -- a dependency naming a state that no
#: `pip install` could reproduce.  The environment is a plain
#: `pip install wrf-rust==0.2.39` from the index (0.2.39 agrees with its
#: own ``__version__``, MEASURED), and this is where that release is
#: written down so the next reader knows which core the recorded test
#: counts belong to.  Re-recorded from 0.2.38 when the install floor
#: moved: certifying against a release the extra no longer installs is
#: exactly the mismatch this constant exists to catch.
SCIENCE_CORE_CERTIFIED = "0.2.39"

#: The requirement string, identical to the one in pyproject's [render]
#: extra -- INSTALL floor, because this string is what pip is handed.
#: Quoted in refusals and install hints so a user is told exactly what to
#: type, and what they type has to be a line that resolves on their
#: interpreter.
SCIENCE_CORE_REQUIREMENT = (
    f"{SCIENCE_CORE_DISTRIBUTION}>={SCIENCE_CORE_INSTALL_FLOOR},"
    f"<{SCIENCE_CORE_CEILING}")

#: The newest CPython the science core can be INSTALLED on, as a pair.
#:
#: A third axis, and it is not a version preference: it is an
#: availability fact about the index, read off the same wheel matrix that
#: sets ``SCIENCE_CORE_INSTALL_FLOOR``.  0.2.39 publishes cp310 through
#: cp314 and nothing above, so 3.14 is the ceiling and 3.15 -- when it
#: arrives -- is over it.
#:
#: 2026-08-17 moved this from (3, 13) to (3, 14).  Upstream published the
#: cp314 wheels, which is the only thing that could move it: raising it
#: is not this project's call, wrf-rust is a separate repository, and the
#: number here follows what the index actually serves.
#:
#: The gap this names is the one that took weather-node-1's install down
#: on python3.14.4: an interpreter over the ceiling gets no wheel, pip
#: falls back to an sdist whose pyo3 caps below it, and pip fails the
#: WHOLE resolution when one requirement fails.  Nothing about that
#: mechanism was specific to 3.14, so the ceiling stays, the sentence
#: that names it is derived from the ceiling rather than written down
#: (:func:`python_gap_sentence`), and doctor keeps the path that reports
#: an excluded core by name instead of by a pip line that installs
#: nothing.
SCIENCE_CORE_PYTHON_CEILING = (3, 14)

__all__ = [
    "SCIENCE_CORE_CEILING",
    "SCIENCE_CORE_CERTIFIED",
    "SCIENCE_CORE_DISTRIBUTION",
    "SCIENCE_CORE_FLOOR",
    "SCIENCE_CORE_INSTALL_FLOOR",
    "SCIENCE_CORE_PYTHON_CEILING",
    "SCIENCE_CORE_REQUIREMENT",
    "installed_science_core_version",
    "python_gap_sentence",
    "python_supports_science_core",
    "science_core_refusal",
    "version_supported",
    "version_tuple",
]


def version_tuple(version: object) -> tuple[int, ...] | None:
    """``"0.2.38"`` -> ``(0, 2, 38)``; anything unparseable -> ``None``.

    Deliberately does not import ``packaging``: that distribution is not a
    declared gpuwm dependency, and a version check that itself raises
    ``ImportError`` on a lean install would be worse than the exact-match
    comparison it replaces.  Only the numeric release segment is read --
    a trailing suffix (``"0.2.38.post1"``, ``"0.2.39rc1"``) stops the walk
    and the leading numbers decide, which is the behaviour a floor/ceiling
    window needs.  A string with no leading number at all is ``None``, so
    an unnameable reader is refused rather than silently accepted.
    """
    if version is None:
        return None
    parts: list[int] = []
    for chunk in str(version).strip().split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
        if len(digits) != len(chunk):
            break
    return tuple(parts) or None


def version_supported(version: object) -> bool:
    """Is ``version`` inside ``[floor, ceiling)``?"""
    found = version_tuple(version)
    if found is None:
        return False
    floor = version_tuple(SCIENCE_CORE_FLOOR)
    ceiling = version_tuple(SCIENCE_CORE_CEILING)
    assert floor is not None and ceiling is not None
    return floor <= found and found < ceiling


def installed_science_core_version() -> str | None:
    """The installed distribution version, or ``None`` if not installed."""
    from importlib import metadata

    try:
        return metadata.version(SCIENCE_CORE_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def python_supports_science_core(python: tuple[int, ...] | None = None
                                 ) -> bool:
    """Can the science core be INSTALLED on this interpreter?

    Separate from :func:`version_supported`, which judges a core that is
    already here.  This one judges the index: below the ceiling pip finds
    a wheel, above it pip finds an sdist that cannot build.  A caller
    that confuses the two reports "wrf-rust is missing, run pip" at a
    reader for whom pip has nothing to install.
    """
    import sys

    found = tuple(python or sys.version_info[:2])[:2]
    return found <= SCIENCE_CORE_PYTHON_CEILING


def python_gap_sentence(python: tuple[int, ...] | None = None) -> str:
    """The one sentence a door prints when pip has no wheel to offer.

    DERIVED from :data:`SCIENCE_CORE_PYTHON_CEILING`, never written down
    beside it.  Its predecessor was a constant reading "render extra
    needs Python <= 3.13 until wrf-rust publishes 3.14 wheels", which
    upstream falsified the day it published cp314 wheels -- and a
    constant cannot notice.  Doctor, and anything else that reports the
    gap, calls this; the sentence then tracks the ceiling with no second
    edit to forget.
    """
    import sys

    ceiling = ".".join(str(part) for part in SCIENCE_CORE_PYTHON_CEILING)
    found = tuple(python or sys.version_info[:2])[:2]
    return (f"{SCIENCE_CORE_DISTRIBUTION} publishes no wheel for Python "
            f"{'.'.join(str(part) for part in found)}: the [render] extra "
            f"needs Python <= {ceiling}")


def science_core_refusal(version: object) -> str:
    """The sentence a caller raises when ``version`` is outside the window."""
    return (f"{SCIENCE_CORE_DISTRIBUTION} version outside the certified "
            f"window: gpuwm requires {SCIENCE_CORE_REQUIREMENT}, "
            f"found {version}")
