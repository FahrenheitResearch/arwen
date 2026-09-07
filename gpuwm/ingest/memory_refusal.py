"""A typed pre-allocation refusal, distinct from an allocator failure."""


class InitializationMemoryRefused(MemoryError):
    """A measured initialization budget cannot admit the requested case."""
