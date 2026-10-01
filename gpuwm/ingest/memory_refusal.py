"""A typed pre-allocation refusal, distinct from an allocator failure."""


class InitializationMemoryRefused(MemoryError):
    """A measured initialization budget cannot admit the requested case."""


class ResidentMemoryRefused(InitializationMemoryRefused):
    """A domain held whole on the card does not fit the card's free memory.

    Raised before the constructor, never from inside it: the priced bytes,
    the free bytes and the named terms travel with it so a caller quoting
    the refusal states this arithmetic rather than a parallel one.
    """

    def __init__(self, message, *, need_bytes=None, free_bytes=None,
                 terms=None, what=None):
        super().__init__(message)
        self.need_bytes = None if need_bytes is None else int(need_bytes)
        self.free_bytes = None if free_bytes is None else int(free_bytes)
        self.terms = dict(terms or {})
        self.what = what
