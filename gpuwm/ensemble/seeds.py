"""Deterministic member seeds (EXPERIMENTAL).

A member's seed is a pure function of the ensemble's base seed and the
member index, so the same ensemble config reproduces the same member
streams on any machine, in any order, after any resume.  The derivation
string is versioned: changing how seeds are derived must change
:data:`SEED_DERIVATION` so that old manifests cannot be mistaken for new
ones.
"""

from __future__ import annotations

import hashlib

#: Versioned derivation label.  Recorded in every ensemble manifest.
SEED_DERIVATION = "gpuwm-ensemble-seed.v1"

#: Seeds are 64-bit unsigned: wide enough for any RNG this project uses,
#: narrow enough to survive JSON round-trips without precision games
#: (json.dumps writes it exactly; json.loads reads it back as an int).
SEED_BITS = 64


def member_seed(base_seed: int, index: int) -> int:
    """The seed for member ``index`` of an ensemble based on ``base_seed``.

    Deterministic, order-independent, and collision-resistant across both
    arguments: seeds are the leading 64 bits of
    ``sha256("gpuwm-ensemble-seed.v1:<base>:<index>")``.  Sequential base
    seeds therefore do not produce correlated member seeds, which a naive
    ``base_seed + index`` would.
    """
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise TypeError(f"base_seed must be an int, got {base_seed!r}")
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"member index must be an int, got {index!r}")
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    if index < 0:
        raise ValueError(f"member index must be non-negative, got {index}")
    payload = f"{SEED_DERIVATION}:{base_seed}:{index}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[: SEED_BITS // 8], "big")
