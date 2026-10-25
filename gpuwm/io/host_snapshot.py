"""Owned, stream-ordered host snapshots without one barrier per device field."""
from __future__ import annotations

import warnings

import numpy as np


def snapshot_arrays(manifest, *, owned=False):
    """Copy a named array inventory, completing all transfers before return.

    Contiguous CuPy fields on the current device share one pinned allocation
    and one stream barrier. Other arrays retain their ordinary ``get`` path.
    The host views retain the allocation, so deferred writers own their data.
    Host arrays are copied when ``owned`` requests a deferred snapshot.
    """
    ordered = sorted(manifest)
    device_keys = [key for key in ordered
                   if hasattr(manifest[key], '__cuda_array_interface__')]
    cp = None
    eligible = []
    if device_keys:
        import cupy as cp
        device = cp.cuda.runtime.getDevice()
        eligible = [key for key in device_keys
                    if isinstance(manifest[key], cp.ndarray)
                    and manifest[key].device.id == device
                    and manifest[key].flags.c_contiguous
                    and manifest[key].nbytes]

    offsets = {}
    size = 0
    for key in eligible:
        size = (size + 63) // 64 * 64
        offsets[key] = size
        size += manifest[key].nbytes
    slab = None
    if size:
        try:
            slab = cp.cuda.alloc_pinned_memory(size)
        except (MemoryError, cp.cuda.runtime.CUDARuntimeError) as exc:
            warnings.warn(
                f'pinned checkpoint snapshot allocation of {size} bytes failed; '
                f'using complete pageable copies for this snapshot: {exc}',
                RuntimeWarning, stacklevel=2)
            offsets.clear()

    arrays = {}
    stream = cp.cuda.get_current_stream() if offsets else None
    try:
        for key in ordered:
            value = manifest[key]
            if key in offsets:
                host = np.ndarray(value.shape, dtype=value.dtype,
                                  buffer=slab, offset=offsets[key])
                # Retain the destination before enqueueing, including when a
                # later transfer fails. Never release an in-flight host view.
                arrays[key] = host
                cp.cuda.runtime.memcpyAsync(
                    host.ctypes.data, value.data.ptr, value.nbytes,
                    cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
            elif hasattr(value, 'get'):
                arrays[key] = np.asarray(value.get())
            else:
                arrays[key] = (np.array(value, copy=True) if owned
                               else np.asarray(value))
    finally:
        if stream is not None:
            # Also on exceptions: no partial snapshot escapes and no buffer
            # is released while the device can still be writing into it.
            stream.synchronize()
    return arrays
