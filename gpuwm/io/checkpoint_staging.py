"""Device-to-host copies of a deferred checkpoint through one pinned slab.

THE COST THIS REMOVES.  A deferred checkpoint (``write_tree_restart(...,
defer_publish=True)``) takes its host copy of every device array on the
stepping thread, through ``ndarray.get()``: a pageable, synchronous copy per
array.  Measured on a 300 x 300 x 49 member (172 arrays, 850 MiB, RTX 4090):
251 to 273 ms per checkpoint at 3.3 GB/s, with the host thread blocked for
all of it.  A pinned destination copies at about 26 GB/s and the copy is
asynchronous to the host.

HOW THE ORDER IS KEPT.  Each copy is enqueued on the stream the model runs
on (the current stream at the snapshot), so it starts after every kernel
already enqueued there -- the step that produced the state -- and every
kernel enqueued after it (the next step) starts after the copy.  That is the
order the synchronous ``get()`` gave, without the host waiting for it.  The
writer thread waits on the event recorded after the last copy before it
reads a byte (:meth:`StagedSnapshot.wait`).

WHAT IS STAGED.  Only a C-contiguous CuPy array on the current device with
at least one element.  Anything else (a host array, a Fortran-ordered or
strided view, an array on another card) takes the original
``_owned_host`` route, unchanged, so the archive holds exactly the arrays,
shapes, dtypes and memory order the synchronous writer gives.

THE SLAB.  One pinned allocation per checkpoint member, carved into
256-byte-aligned views.  It comes from CuPy's pinned memory pool, so the
next checkpoint of the same domain, which needs the same size, reuses it
instead of page-locking 850 MiB again.  At most one generation is in flight
(the checkpoint publisher waits for the previous one first), so at most one
slab per domain is held.

``GPUWM_CHECKPOINT_PINNED=0`` takes the synchronous route for every array.
"""

from __future__ import annotations

import os

import numpy as np

#: Environment switch.  Default on.
ENV = "GPUWM_CHECKPOINT_PINNED"

_ALIGN = 256


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def _cupy():
    try:
        import cupy as cp
    except Exception:  # noqa: BLE001 - no CuPy, nothing to stage
        return None
    return cp


class StagedSnapshot:
    """The pinned host copies of one checkpoint member's device arrays."""

    def __init__(self):
        self._done = None
        self._holds: list = []
        self.staged_bytes = 0
        self.staged_arrays = 0
        self.fallback_arrays = 0

    def stage(self, manifest, owned_host):
        """``{key: host array}`` for ``manifest``, in its sorted key order.

        Device arrays that qualify are copied into the slab asynchronously
        on the current stream; every other value goes through
        ``owned_host`` exactly as the synchronous writer does.
        """
        cp = _cupy()
        keys = sorted(manifest)
        eligible = {}
        if cp is not None and enabled():
            candidates = {key: manifest[key] for key in keys
                          if isinstance(manifest[key], cp.ndarray)
                          and manifest[key].size > 0
                          and manifest[key].flags.c_contiguous}
            if candidates:
                # A device array exists, so a device answers.
                device_id = int(cp.cuda.Device().id)
                eligible = {key: value for key, value in candidates.items()
                            if int(value.device.id) == device_id}
        offsets = {}
        total = 0
        for key in keys:
            if key in eligible:
                offsets[key] = total
                total += -(-int(eligible[key].nbytes) // _ALIGN) * _ALIGN
        hosts = {}
        if total:
            stream = cp.cuda.get_current_stream()
            slab = cp.cuda.alloc_pinned_memory(total)
            raw = np.frombuffer(slab, dtype=np.uint8, count=total)
            for key, value in eligible.items():
                start = offsets[key]
                view = raw[start:start + int(value.nbytes)].view(
                    value.dtype).reshape(value.shape)
                value.get(out=view, stream=stream, blocking=False)
                hosts[key] = view
            done = cp.cuda.Event()
            done.record(stream)
            self._done = done
            # The slab and the device arrays stay referenced until the copy
            # has completed and the archive has been written.
            self._holds = [slab, tuple(eligible.values())]
            self.staged_bytes = int(sum(int(v.nbytes) for v in eligible.values()))
            self.staged_arrays = len(eligible)
        out = {}
        try:
            for key in keys:
                if key in hosts:
                    out[key] = hosts[key]
                else:
                    out[key] = owned_host(manifest[key])
                    self.fallback_arrays += 1
        except BaseException:
            # The slab must not return to the pool while copies still land.
            self.release()
            raise
        return out

    def wait(self) -> None:
        """Block until every staged copy has landed in the slab."""
        done, self._done = self._done, None
        if done is not None:
            done.synchronize()
        # The device arrays may go now; the slab stays with the host views.
        if self._holds:
            self._holds = [self._holds[0]]

    def release(self) -> None:
        """Drop the slab (after the archive is written, or when abandoned)."""
        done, self._done = self._done, None
        if done is not None:
            done.synchronize()
        self._holds = []


__all__ = ["ENV", "StagedSnapshot", "enabled"]
