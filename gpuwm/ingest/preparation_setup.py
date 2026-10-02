"""Setup reused only while a boundary-frame preparation owns it."""
from contextvars import ContextVar
from threading import RLock
import weakref

import numpy as np


_ACTIVE = ContextVar('forcing_setup_owners', default=())


class PreparationSetup:
    def __init__(self):
        self.lock = RLock()
        self.base = {}
        self.horizontal = {}
        self.backends = {}
        self.closed = False

    def activate(self):
        # The context never owns the caches. Abandoned preparations release
        # them with their frame builder, including failed preparations.
        references = tuple(reference for reference in _ACTIVE.get()
                           if reference() is not None and not reference().closed)
        _ACTIVE.set(references + (weakref.ref(self),))

    def close(self):
        with self.lock:
            self.base.clear()
            self.horizontal.clear()
            self.backends.clear()
            self.closed = True


def current_setup():
    references = tuple(reference for reference in _ACTIVE.get()
                       if reference() is not None and not reference().closed)
    _ACTIVE.set(references)
    return references[-1]() if references else None


def array_key(value):
    """Exact shape, dtype and bytes, with no digest collision assumption."""
    value = np.asarray(value)
    return value.shape, value.dtype.str, value.tobytes(order='C')
