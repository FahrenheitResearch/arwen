"""Basenames read out of recorded receipt paths, the same on every platform.

A prepared tree records the paths its preparation actually opened.  Those
strings are provenance, and they are spelled in the preparing platform's
dialect: a tree prepared on Windows records
``C:\\Users\\<user>\\prep\\case\\experiment.toml``, and the same tree
prepared on Linux records ``/home/<user>/prep/case/experiment.toml``.

:class:`pathlib.Path` reads a recorded string with the RUNNING platform's
separator rules, not the recording platform's.  A Windows spelling read on
POSIX therefore contains no separators at all, and ``.name`` returns the
whole string.  Every receipt check that compared a basename that way
refused a prepared tree for having merely crossed platforms -- prepare on
the desktop, run on the node -- while its bytes and its SHA-256 matched
exactly.  The concrete breakage: a Windows-prepared tree could not be run
on Linux, and the refusal blamed the file's contents.

:func:`receipt_basename` reads both spellings on both platforms.  Use it
wherever a recorded path meets a name, and never :class:`pathlib.Path` --
``Path`` is for paths this process resolves on this machine, and a recorded
receipt path is not one of those.
"""

from __future__ import annotations

_SEPARATORS = ("\\", "/")


def receipt_basename(value: object) -> str:
    """Return the final component of a recorded path, in either dialect.

    Both separators are honoured regardless of the running platform, a
    leading ``X:`` drive designator is dropped the way Windows itself drops
    it, and trailing separators are ignored so a recorded directory keeps
    its own name.  A root has no name, which is what :class:`pathlib.Path`
    reports too.
    """

    text = str(value)
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        text = text[2:]
    text = text.rstrip("".join(_SEPARATORS))
    if not text:
        return ""
    for separator in _SEPARATORS:
        text = text.rsplit(separator, 1)[-1]
    return text
