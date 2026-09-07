"""Host physics caches bound to the exact geography that produced them.

Owners declare ``geography_cache_dependencies`` as cache attribute -> tuples
of (live input attribute, retained snapshot attribute). A cache reader calls
``geography_cache`` before use; gathering or relocation may overwrite a live
input in place. Snapshots are independent arrays and keep the input dtype.
"""
from __future__ import annotations

import numpy as np


def geography_cache(owner, name: str, compute):
    """Reuse a cache only while every declared input is exactly unchanged."""
    dependencies = owner.geography_cache_dependencies[name]
    if not dependencies:
        raise ValueError(f"geography cache {name!r} has no declared inputs")
    current = [(snapshot, np.asarray(getattr(owner, live)))
               for live, snapshot in dependencies]
    cached = getattr(owner, name, None)
    if cached is not None and all(
            isinstance(getattr(owner, snapshot, None), np.ndarray)
            and getattr(owner, snapshot).dtype == value.dtype
            and np.array_equal(getattr(owner, snapshot), value)
            for snapshot, value in current):
        return cached
    result = compute()
    snapshots = [(snapshot, value.copy()) for snapshot, value in current]
    setattr(owner, name, result)
    for snapshot, value in snapshots:
        setattr(owner, snapshot, value)
    return result


def geography_cache_snapshots(owner) -> dict[str, np.ndarray]:
    """Actual retained input arrays for the geography memory inventory."""
    return {snapshot: value
            for dependencies in getattr(
                owner, "geography_cache_dependencies", {}).values()
            for _live, snapshot in dependencies
            if isinstance(value := getattr(owner, snapshot, None), np.ndarray)}
