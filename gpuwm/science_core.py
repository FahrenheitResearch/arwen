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

``__version__`` is deliberately NOT the authority anywhere in the tree.
wrf-rust 0.2.35 shipped with ``__version__ == "0.2.34"`` -- an attribute the
author forgot to bump -- so the installed *distribution* metadata answers
and the attribute is recorded beside it as a note.
"""

from __future__ import annotations

#: The pip distribution name; the import name is ``wrf``.
SCIENCE_CORE_DISTRIBUTION = "wrf-rust"

#: Oldest certified release (inclusive).
SCIENCE_CORE_FLOOR = "0.2.35"

#: First release outside the window (exclusive).
SCIENCE_CORE_CEILING = "0.3"

#: The requirement string, identical to the one in pyproject's [render]
#: extra.  Quoted in refusals and install hints so a user is told exactly
#: what to type.
SCIENCE_CORE_REQUIREMENT = (
    f"{SCIENCE_CORE_DISTRIBUTION}>={SCIENCE_CORE_FLOOR},"
    f"<{SCIENCE_CORE_CEILING}")

__all__ = [
    "SCIENCE_CORE_CEILING",
    "SCIENCE_CORE_DISTRIBUTION",
    "SCIENCE_CORE_FLOOR",
    "SCIENCE_CORE_REQUIREMENT",
    "installed_science_core_version",
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


def science_core_refusal(version: object) -> str:
    """The sentence a caller raises when ``version`` is outside the window."""
    return (f"{SCIENCE_CORE_DISTRIBUTION} version outside the certified "
            f"window: gpuwm requires {SCIENCE_CORE_REQUIREMENT}, "
            f"found {version}")
