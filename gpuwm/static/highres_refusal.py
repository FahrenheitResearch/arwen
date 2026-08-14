"""The named refusal type for the high-resolution static path.

A leaf module on purpose.  :mod:`gpuwm.static.highres_production` owns the
refusals themselves and re-exports this class, so every existing
``from gpuwm.static.highres_production import HighresRefusal`` keeps
working -- but the CLI's refusal boundary needs the *name* at except-clause
time on every single invocation, and importing the production module to
get it would drag numpy and the static builders into `gpuwm version`.

This module imports nothing.  That is its entire reason to exist.
"""
from __future__ import annotations


class HighresRefusal(RuntimeError):
    """A named, deliberate refusal of the high-resolution path."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"[static.highres] refused ({reason}): {detail}")
        self.reason = reason
        self.detail = detail
