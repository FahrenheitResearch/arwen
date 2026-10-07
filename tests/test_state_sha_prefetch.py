"""Whole-state hashes copy the next array while hashing the current one.

CPU only.  THE BREAKAGE THIS PREVENTS: an increment application hashed a
9 km CONUS member's whole state twice, copying each array off the card and
then hashing it, one after the other.  The prefetching hash must give the
serial digest exactly.
"""
from __future__ import annotations

import hashlib

import numpy as np

from gpuwm.ensemble import state_sha


def _serial(arrays):
    digest = hashlib.sha256(state_sha.STATE_SHA_CONTRACT.encode("ascii") + b"\0")
    for name, value in arrays:
        host = np.ascontiguousarray(np.asarray(value))
        for part in (name.encode("utf-8"), str(host.dtype).encode("ascii"),
                     repr(host.shape).encode("ascii"), host.tobytes()):
            digest.update(part)
            digest.update(b"\0")
    return digest.hexdigest()


def test_prefetching_hash_is_the_serial_digest():
    rng = np.random.default_rng(0)
    arrays = [(f"f{i}", rng.standard_normal((3, 4, 5 + i)).astype(
        np.float32 if i % 2 else np.float64)) for i in range(7)]
    for count in (0, 1, 2, 7):
        assert state_sha.hash_state_arrays(arrays[:count]) == _serial(arrays[:count])
