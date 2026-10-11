"""C1 rental time ledger. No GPU or field-processing dependencies."""

__all__ = ["Ledger", "emit", "span"]


def __getattr__(name):
    # Keep `python -m gpuwm.timeline.ledger` free of runpy double-import warnings.
    if name in __all__:
        from . import ledger
        return getattr(ledger, name)
    raise AttributeError(name)
