"""Scoped owners for concurrent experiments' original CFL reductions."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import os


CFL_BANK_NAMES = ("_WRF_CFL_STAT", "_WRF_CFL_LABEL", "_WRF_CFL_LAST",
                  "_WRF_CFL_CALLS", "_WRF_CFL_DOMAIN_STEP", "_WRF_CFL_EVENTS")


def _probe_enabled():
    value = os.environ.get("GPUWM_WRF_CFL_PROBE")
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


@dataclass
class MemberCflState:
    """Independent clocks consume independent reductions under the same IDs."""
    enabled: bool = field(default_factory=_probe_enabled)
    banks: dict = field(default_factory=lambda: {name: {} for name in CFL_BANK_NAMES})


_MEMBER = ContextVar("gpuwm_member_cfl_owner", default=None)


def current_cfl_member():
    return _MEMBER.get()


@contextmanager
def member_cfl_scope():
    """Isolate the original grid-ID banks without changing their reduction.

    The ensemble worker enters this once for its complete ordinary model.
    Copied contexts share this owner within that member. Scope exit must
    follow completion of the member's CUDA queues.
    """
    owner = MemberCflState()
    token = _MEMBER.set(owner)
    try:
        yield owner
    finally:
        for bank in owner.banks.values():
            bank.clear()
        _MEMBER.reset(token)


__all__ = ["CFL_BANK_NAMES", "MemberCflState", "current_cfl_member", "member_cfl_scope"]
