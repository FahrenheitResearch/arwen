"""Host metadata inventory of resident NextGen CUDA allocations.

This inspects initialized storage at setup/output boundaries, never values or
hydrologic steps. It is not a pre-factory memory estimate or an OOM guarantee.
"""
from collections.abc import Mapping
import threading
import weakref


def _merged_bytes(intervals):
    total, end = 0, 0
    for lo, hi in sorted(intervals):
        if hi > end:
            total += hi - max(lo, end)
            end = hi
    return total


class ResidentInventory:
    """Count backing storage once and reject sharing between live members.

    All driver.arrays carriers are restart-writable in the resident ABI, so
    sharing even disjoint views of a mutable allocation between live members
    is refused. Weak references permit allocator reuse after a member dies;
    this inventory must not keep a completed member's GPU memory alive.
    """
    def __init__(self):
        self._live, self._rows = {}, {}
        self._lock = threading.RLock()

    def inspect(self, driver, *, member_id, device_id):
        bound_member = getattr(driver, "member_id", None)
        if bound_member is not None and bound_member != member_id:
            raise ValueError("NextGen driver member identity differs from the capture member")
        arrays = driver.arrays
        if not isinstance(arrays, Mapping) or not arrays:
            raise ValueError("NextGen driver.arrays must expose all resident carriers and workspace")
        if int(driver.device_id) != int(device_id):
            raise ValueError("NextGen member driver belongs to another physical card")
        intervals, carriers, live_storage = [], [], []
        groups = {}
        for name, array in arrays.items():
            if not isinstance(name, str) or not name:
                raise ValueError("NextGen resident carriers need nonempty names")
            if int(array.device.id) != int(device_id):
                raise ValueError(f"NextGen carrier {name} belongs to another physical card")
            # CuPy MemoryPointer.mem owns the complete allocation, even when
            # array is a sliced, transposed, broadcast or negatively strided view.
            memory = array.data.mem
            lo, size = int(memory.ptr), int(memory.size)
            hi = lo + size
            if size < 0 or (size and lo <= 0):
                raise ValueError(f"NextGen carrier {name} has invalid backing storage")
            shape, strides = tuple(map(int, array.shape)), tuple(map(int, array.strides))
            if len(shape) != len(strides) or any(n < 0 for n in shape):
                raise ValueError(f"NextGen carrier {name} has invalid shape/strides")
            if all(shape):
                offsets = tuple((n - 1) * stride for n, stride in zip(shape, strides))
                first = int(array.data.ptr) + sum(min(0, x) for x in offsets)
                last = int(array.data.ptr) + sum(max(0, x) for x in offsets) + int(array.dtype.itemsize)
                if first < lo or last > hi:
                    raise ValueError(f"NextGen carrier {name} extends outside its allocation")
            group = groups.setdefault((lo, hi), len(groups))
            carriers.append({"name": name, "shape": list(shape), "strides": list(strides),
                "dtype": str(array.dtype), "logical_bytes": int(array.nbytes), "allocation": group})
            if size:
                intervals.append((lo, hi))
                live_storage.append((weakref.ref(array), lo, hi))
        with self._lock:
            for other, (card, live) in self._live.items():
                if other == member_id or card != int(device_id):
                    continue
                for ref, lo, hi in live:
                    if ref() is not None and any(a < hi and lo < b for a, b in intervals):
                        raise ValueError(f"NextGen members {other} and {member_id} share live resident storage")
            row = {"member_id": member_id, "device_id": int(device_id),
                "logical_bytes": sum(value["logical_bytes"] for value in carriers),
                "allocation_bytes": _merged_bytes(intervals),
                "allocations": [{"index": group, "bytes": hi-lo}
                    for (lo, hi), group in groups.items()], "carriers": carriers}
            self._live[member_id] = (int(device_id), live_storage)
            self._rows[member_id] = row
            return row

    def receipt(self):
        with self._lock:
            return {"basis": "initialized CUDA allocation metadata; shared views deduplicated",
                "pre_factory_admission": False,
                "scope": "driver.arrays only; allocator caches, context and output pools excluded",
                "members": [self._rows[member] for member in sorted(self._rows)]}
