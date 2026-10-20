"""Scoped Python output capture for concurrent in-process preparations."""
from contextlib import contextmanager
import sys
import threading

_lock = threading.RLock()
_local = threading.local()
_users = 0
_proxies = None


class _Stream:
    def __init__(self, channel, fallback):
        self.channel, self.fallback = channel, fallback

    def current(self):
        stack = getattr(_local, 'stack', ())
        return stack[-1][self.channel] if stack else self.fallback

    def write(self, text):
        return self.current().write(text)

    def flush(self):
        return self.current().flush()

    def __getattr__(self, name):
        return getattr(self.current(), name)


def current_stream(stream):
    return stream.current() if isinstance(stream, _Stream) else stream


@contextmanager
def redirect_output(stdout, stderr):
    """Route each caller to its own log without serializing preparation work.

    Python's redirect_stdout mutates process globals. Overlapping scopes can
    restore another member's already closed log and turn success into exit 74.
    Only proxy installation is locked; each thread keeps its own destination.
    """
    global _users, _proxies
    with _lock:
        if not _users:
            _proxies = (_Stream(0, sys.stdout), _Stream(1, sys.stderr))
            sys.stdout, sys.stderr = _proxies
        _users += 1
        stack = getattr(_local, 'stack', None)
        if stack is None:
            stack = _local.stack = []
        stack.append((stdout, stderr))
    try:
        yield
    finally:
        with _lock:
            stack.pop()
            _users -= 1
            if not _users:
                sys.stdout, sys.stderr = (stream.fallback for stream in _proxies)
                _proxies = None
