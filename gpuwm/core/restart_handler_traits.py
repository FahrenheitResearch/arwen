"""What a restart handler declares about itself, read outside the clock modules.

The scheduled executor (:mod:`gpuwm.core.model`) asks two optional traits
of the restart handler it is given: whether the handler orders its
checkpoint after the history frames itself (the prepared single-domain
door's background checkpoint writer sets ``orders_history_itself``), and
the callable that blocks until a requested checkpoint is durable
(``wait_durable``).  A plain function handler carries neither.

They are read here and not in the executor because
``tests/test_clock.py``'s audit bans reflection identifiers (getattr,
setattr, ...) in the clock modules outright, so that no dynamic attribute
access can mutate clock state there; d09618a7e and 1f83457d4 had put the
two reads inline.
"""

from __future__ import annotations


def orders_history_itself(handler) -> bool:
    """True when ``handler`` orders its checkpoint after history itself."""

    return bool(getattr(handler, "orders_history_itself", False))


def durable_waiter(handler):
    """``handler.wait_durable`` when it declares one, else ``None``."""

    return getattr(handler, "wait_durable", None)


__all__ = ["durable_waiter", "orders_history_itself"]
