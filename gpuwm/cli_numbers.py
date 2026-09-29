"""Numeric command-line values whose allowed range is stated at the parser.

``type=float`` accepts ``nan``, ``inf`` and ``-inf``, and ``type=int``
accepts any sign.  An option that means a size, a count, a duration or a
port then carried a value outside its meaning into its handler, where it
ended as whatever the first arithmetic on it raised: an OverflowError
traceback from ``int(inf)`` or ``Fraction(inf)``, a sentence such as
"cannot convert NaN to integer ratio" that names no option, or no error
at all for a negative size.

These types refuse at the parser instead.  argparse prints the option
the value was given to, so every refusal reads
``argument --dt-s: must be a finite number of zero or more, not '-1'``
and exits 2 before any file is read or any work starts.

Use them for an option whose handler does not already check the same
range by name; a handler check with its own wording stays where it is.
"""

from __future__ import annotations

import argparse
import math
from typing import Callable


def _float(text: str, need: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}") from None
    return value


def _int(text: str, need: str) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}") from None


def finite_float(text: str) -> float:
    """Any finite number."""
    need = "a finite number"
    value = _float(text, need)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}")
    return value


def positive_float(text: str) -> float:
    """A finite number above zero: a size, a spacing, a duration."""
    need = "a finite number above zero"
    value = _float(text, need)
    if not (math.isfinite(value) and value > 0.0):
        raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}")
    return value


def nonnegative_float(text: str) -> float:
    """A finite number of zero or more."""
    need = "a finite number of zero or more"
    value = _float(text, need)
    if not (math.isfinite(value) and value >= 0.0):
        raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}")
    return value


def positive_int(text: str) -> int:
    """A whole number of one or more: a count, a size in cells."""
    need = "a whole number of 1 or more"
    value = _int(text, need)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}")
    return value


def nonnegative_int(text: str) -> int:
    """A whole number of zero or more: an offset, a lead hour."""
    need = "a whole number of 0 or more"
    value = _int(text, need)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}")
    return value


def int_at_least(low: int) -> Callable[[str], int]:
    """A whole number of ``low`` or more."""
    need = f"a whole number of {low} or more"

    def parse(text: str) -> int:
        value = _int(text, need)
        if value < low:
            raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}")
        return value

    parse.__name__ = f"int_at_least_{low}"
    return parse


def int_between(low: int, high: int) -> Callable[[str], int]:
    """A whole number from ``low`` to ``high`` inclusive."""
    need = f"a whole number from {low} to {high}"

    def parse(text: str) -> int:
        value = _int(text, need)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}")
        return value

    parse.__name__ = f"int_{low}_to_{high}"
    return parse


def float_between(low: float, high: float) -> Callable[[str], float]:
    """A finite number from ``low`` to ``high`` inclusive."""
    need = f"a number from {low:g} to {high:g}"

    def parse(text: str) -> float:
        value = _float(text, need)
        if not (math.isfinite(value) and low <= value <= high):
            raise argparse.ArgumentTypeError(f"must be {need}, not {text!r}")
        return value

    parse.__name__ = f"float_{low:g}_to_{high:g}"
    return parse


__all__ = ["finite_float", "float_between", "int_at_least", "int_between",
           "nonnegative_float", "nonnegative_int", "positive_float", "positive_int"]
