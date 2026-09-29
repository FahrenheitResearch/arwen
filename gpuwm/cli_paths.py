"""A path the reader typed that is not there, said as a sentence.

Every door takes files and folders on its command line, and most of
them open the path where the work needs it rather than checking it at
the parser.  A missing or mistyped path then left as a
FileNotFoundError traceback from deep inside the handler: `gpuwm
spectral cross-box` on a receipt that was not there, `gpuwm domain
--forcing` on a file that had not been downloaded yet, and the same on
most of the three hundred path options the CLI carries.

The refusal boundary asks :func:`supplied_path_refusal` about every
FileNotFoundError, NotADirectoryError, IsADirectoryError and
PermissionError that reaches it.  When the path the error names is one
the reader supplied, the answer is one sentence naming the option and
the path, exit 2.  When it is a path the program chose for itself, the
answer is ``None`` and the traceback stands: a file this program expected
to have written and did not is a defect, and dressing it as a usage
error would hide it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

_ERRORS = (FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError)


def _key(value) -> str | None:
    try:
        text = os.fsdecode(value)
    except TypeError:
        return None
    if not text or text == "-":
        return None
    # A Windows extended-length spelling (\\?\C:\... or \\?\UNC\host\...),
    # which deep_io_path hands the operating system for a long path, names
    # the same file as the spelling the reader typed.
    if text[:8].lower() == "\\\\?\\unc\\":
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    return os.path.normcase(os.path.abspath(os.path.expanduser(text)))


def _values(value) -> Iterable:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _values(item)
    elif isinstance(value, (str, os.PathLike)):
        yield value


def _all_actions(parser: argparse.ArgumentParser, seen: set[int] | None = None):
    """Every action of ``parser`` and of every subcommand under it."""

    seen = set() if seen is None else seen
    if id(parser) in seen:
        return
    seen.add(id(parser))
    for action in parser._actions:
        yield action
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from _all_actions(sub, seen)


def _invoked_actions(parser: argparse.ArgumentParser, args: argparse.Namespace):
    """The actions of the parsers this invocation passed through, root first.

    Two subcommands may give one ``dest`` to different arguments
    (``receipt`` is a positional of `gpuwm spectral check` and of `gpuwm
    spectral-op check`), so the option is looked for on the chosen path
    before anywhere else.
    """

    current, seen = parser, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chosen = None
        for action in current._actions:
            yield action
            if (isinstance(action, argparse._SubParsersAction)
                    and action.dest not in (None, argparse.SUPPRESS)):
                name = getattr(args, action.dest, None)
                if isinstance(name, str) and name in action.choices:
                    chosen = action.choices[name]
        current = chosen


def _spelling(parser: argparse.ArgumentParser | None, dest: str,
              args: argparse.Namespace, tokens: Sequence[str]) -> str | None:
    """The option the reader typed for ``dest``, or the argument's name."""

    if parser is None:
        return None
    typed = {token.split("=", 1)[0] for token in tokens}
    for actions in (_invoked_actions(parser, args), _all_actions(parser)):
        named = None
        for action in actions:
            if action.dest != dest:
                continue
            if action.option_strings:
                for option in action.option_strings:
                    if option in typed:
                        return option
                # Not typed: the path is this option's default.
                named = named or max(action.option_strings, key=len)
            elif named is None:
                named = (str(action.metavar) if action.metavar
                         else action.dest.upper())
        if named is not None:
            return named
    return None


def supplied_path_refusal(error: BaseException, args: argparse.Namespace, *,
                          parser: argparse.ArgumentParser | None = None,
                          tokens: Sequence[str] = ()) -> str | None:
    """One sentence for a supplied path the operating system refused.

    ``None`` unless ``error`` is a file-system error whose path is a
    value in ``args``: only a path the reader gave (or a default the
    reader left in place) is the reader's to correct.
    """

    if not isinstance(error, _ERRORS):
        return None
    named = [name for name in (getattr(error, "filename", None),
                               getattr(error, "filename2", None)) if name is not None]
    wanted = {key for key in map(_key, named) if key is not None}
    if not wanted:
        return None
    for dest, value in sorted(vars(args).items()):
        if dest.startswith("_"):
            continue
        for item in _values(value):
            if _key(item) not in wanted:
                continue
            typed = os.fsdecode(item)
            where = Path(typed).expanduser()
            spelling = _spelling(parser, dest, args, tokens)
            if spelling is None:
                label = typed
            elif spelling.startswith("-"):
                label = f"{spelling} {typed}"
            else:
                label = f"{typed} (the {spelling} argument)"
            shown = "" if where.is_absolute() else f" (looked for {where.resolve()})"
            if isinstance(error, NotADirectoryError):
                return (f"{label}: part of this path is a file, not a "
                        f"folder{shown}")
            if where.is_dir() and isinstance(error, (IsADirectoryError, PermissionError)):
                return f"{label} is a folder; a file is needed here{shown}"
            if isinstance(error, PermissionError):
                return f"{label}: permission denied{shown}"
            if isinstance(error, IsADirectoryError):
                return f"{label} is a folder; a file is needed here{shown}"
            folder = where.resolve().parent
            if not where.exists() and not folder.is_dir():
                # A path to be written whose folder is missing is the
                # folder's fault, and saying the file is missing sends the
                # reader looking for something they meant to create.
                return f"{label}: the folder {folder} does not exist"
            return f"{label}: no such file or folder{shown}"
    return None


__all__ = ["supplied_path_refusal"]
