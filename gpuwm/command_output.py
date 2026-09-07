"""Bounded live subprocess output for CLI hosts, without shell execution."""

from __future__ import annotations

import codecs
import io
import os
import subprocess
import threading


OUTPUT_CHUNK_SIZE = 16384


def text_chunks(text):
    """Bound temporary encoding/tail buffers even for in-process large writes."""
    for offset in range(0, len(text), OUTPUT_CHUNK_SIZE):
        yield text[offset:offset + OUTPUT_CHUNK_SIZE]


class AdapterOutputError(RuntimeError):
    """The child ran, but its output could not be retained or delivered."""


class DiagnosticLog:
    """Raise the first log failure once; keep the error-report path usable.

    Both adapter streams share this log. After a write/flush failure, further
    log writes are disabled so reporting that failure cannot recurse through
    the same broken destination. The owning CLI still reports failure.
    """
    def __init__(self, stream):
        self.stream = stream
        self.failure = None

    @property
    def closed(self):
        return self.stream.closed

    def _failed(self, error):
        if self.failure is None:
            self.failure = error
            raise AdapterOutputError(str(error)) from error

    def write(self, text):
        if self.failure is None:
            try:
                return self.stream.write(text)
            except (OSError, ValueError) as error:
                self._failed(error)
        return len(text)

    def flush(self):
        if self.failure is None and not self.closed:
            try:
                self.stream.flush()
            except (OSError, ValueError) as error:
                self._failed(error)

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            self.stream.close()
        except (OSError, ValueError) as closing_error:
            # A cleanup failure must not replace an earlier launch/IO error.
            if error_type is None:
                self._failed(closing_error)
        return False


def run_streamed(command, stdout, stderr):
    """Drain both pipes concurrently; deliver flushed chunks before child exit.

    Receipts can be large or contain no newlines. Fixed-size reads and an
    incremental decoder keep memory bounded and preserve split UTF-8 characters.
    A failing destination is remembered while both pipes continue draining.
    """
    failures = []
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    with subprocess.Popen(command, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=environment) as process:
        def drain(pipe, destination):
            decoder = io.IncrementalNewlineDecoder(
                codecs.getincrementaldecoder("utf-8")("replace"), translate=True)
            failed = False
            try:
                while True:
                    block = pipe.read1(OUTPUT_CHUNK_SIZE)
                    text = decoder.decode(block, final=not block)
                    if text and not failed:
                        try:
                            destination.write(text)
                            destination.flush()
                        except Exception as error:
                            failures.append(error)
                            failed = True
                    if not block:
                        break
            except Exception as error:
                failures.append(error)
            finally:
                pipe.close()

        workers = [threading.Thread(target=drain, args=pair, daemon=True)
                   for pair in ((process.stdout, stdout), (process.stderr, stderr))]
        for worker in workers:
            worker.start()
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            for worker in workers:
                worker.join()
    if failures:
        raise AdapterOutputError(str(failures[0])) from failures[0]
    return subprocess.CompletedProcess(command, code)
