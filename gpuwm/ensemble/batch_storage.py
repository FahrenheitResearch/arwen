"""Explicit member-slab storage and allocation accounting for CUDA batches.

The allocation plan is the allocator's input, not a fitted member multiplier.
It covers named array allocations only. Context, compiled-kernel local memory,
output queues and allocator retention must be supplied as separate reservations
before the plan can be used for forecast admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from operator import index
from typing import Literal

import numpy as np

Ownership = Literal["member", "shared"]


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    value = index(value)
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class BatchArraySpec:
    """One non-aliased allocation, with no implicit shared-field inference."""

    name: str
    shape: tuple[int, ...]
    ownership: Ownership
    dtype: str = "float32"

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("batch array needs a nonempty name")
        shape = tuple(_positive(n, f"{self.name} extent") for n in self.shape)
        if not shape:
            raise ValueError(f"{self.name} needs at least one array dimension")
        if self.ownership not in ("member", "shared"):
            raise ValueError(f"{self.name} has unknown ownership {self.ownership!r}")
        dtype = np.dtype(self.dtype)
        if (dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None
                or dtype.kind not in "biufc" or dtype.itemsize < 1):
            raise TypeError(f"{self.name} needs a nonempty numeric scalar device dtype")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype.str)

    @property
    def slab_bytes(self):
        return prod(self.shape) * np.dtype(self.dtype).itemsize

    def allocation_shape(self, members):
        members = _positive(members, "members")
        return ((members,) + self.shape
                if self.ownership == "member" else self.shape)


@dataclass(frozen=True)
class BatchMemoryPlan:
    """Allocation-exact payload plus explicit, separately reserved overhead.

    ``allocation_quantum`` prices each backing allocation at the allocator's
    block granularity. ``reserved_bytes`` belongs to other named allocations
    or a measured context/pool reserve, and is never called array payload.
    """

    arrays: tuple[BatchArraySpec, ...]
    reserved_bytes: int
    allocation_quantum: int = 512

    def __post_init__(self):
        arrays = tuple(self.arrays)
        if not arrays or any(not isinstance(a, BatchArraySpec) for a in arrays):
            raise ValueError("batch memory plan needs array specifications")
        if len({a.name for a in arrays}) != len(arrays):
            raise ValueError("duplicate batch allocation names would hide device memory")
        if isinstance(self.reserved_bytes, (bool, np.bool_)):
            raise TypeError("reserved_bytes must be an integer")
        reserve = index(self.reserved_bytes)
        if reserve < 0:
            raise ValueError("reserved_bytes must not be negative")
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "reserved_bytes", reserve)
        object.__setattr__(self, "allocation_quantum",
                           _positive(self.allocation_quantum, "allocation_quantum"))

    def inventory(self, members):
        members = _positive(members, "members")
        quantum = self.allocation_quantum
        rows = []
        for spec in self.arrays:
            count = members if spec.ownership == "member" else 1
            payload = spec.slab_bytes * count
            allocated = ((payload + quantum - 1) // quantum) * quantum
            rows.append({"name": spec.name, "ownership": spec.ownership,
                         "shape": spec.allocation_shape(members), "dtype": spec.dtype,
                         "payload_bytes": payload, "allocated_bytes": allocated})
        return tuple(rows)

    def required_bytes(self, members):
        return self.reserved_bytes + sum(row["allocated_bytes"]
                                         for row in self.inventory(members))

    def largest_that_fits(self, available_bytes, *, max_members):
        max_members = _positive(max_members, "max_members")
        if isinstance(available_bytes, (bool, np.bool_)):
            raise TypeError("available_bytes must be an integer")
        available = index(available_bytes)
        if available < 0:
            raise ValueError("available_bytes must not be negative")
        low, high = 0, max_members
        while low < high:
            mid = (low + high + 1) // 2
            if self.required_bytes(mid) <= available:
                low = mid
            else:
                high = mid - 1
        return low

    def admit(self, members, *, available_bytes):
        members = _positive(members, "members")
        if isinstance(available_bytes, (bool, np.bool_)):
            raise TypeError("available_bytes must be an integer")
        available = index(available_bytes)
        capacity = self.largest_that_fits(available, max_members=members)
        if capacity != members:
            raise MemoryError(
                f"{members} resident ensemble members need "
                f"{self.required_bytes(members)} bytes; {available} bytes are "
                f"available and at most {capacity} members fit. Allocation "
                "would exhaust device memory before the batch can advance.")


class BatchStorage:
    """One contiguous (member, ...) backing per mutable field.

    NumPy is accepted for small allocation-contract tests. Forecast callers
    supply CuPy explicitly. Shared buffers are made read-only to NumPy callers;
    CUDA callers must keep shared pointers const in every kernel specification.
    """

    def __init__(self, plan, members, *, array_module, available_bytes):
        self.members = _positive(members, "members")
        plan.admit(self.members, available_bytes=available_bytes)
        self.plan = plan
        self.specs = {spec.name: spec for spec in plan.arrays}
        self.arrays = {spec.name: array_module.zeros(
            spec.allocation_shape(self.members), dtype=spec.dtype)
            for spec in plan.arrays}

    @property
    def payload_bytes(self):
        return sum(array.nbytes for array in self.arrays.values())

    def member_view(self, name, member):
        if isinstance(member, (bool, np.bool_)):
            raise TypeError("member index must be an integer")
        member = index(member)
        if not 0 <= member < self.members:
            raise IndexError(f"member {member} outside [0, {self.members})")
        spec, array = self.specs[name], self.arrays[name]
        if spec.ownership == "member":
            return array[member]
        if isinstance(array, np.ndarray):
            view = array.view()
            view.flags.writeable = False
            return view
        return array

    def pointer_stride_bytes(self, name):
        spec = self.specs[name]
        return spec.slab_bytes if spec.ownership == "member" else 0

    def verify_shared_inputs(self, name, inputs):
        """Verify setup bytes before binding a shared reference field.

        This is an admission check over already prepared host buffers. It does
        not decode, regrid, or move forecast data off the GPU.
        """
        spec = self.specs[name]
        if spec.ownership != "shared":
            raise ValueError(f"{name} is member-owned")
        inputs = tuple(inputs)
        if len(inputs) != self.members:
            raise ValueError(f"{name} needs one prepared input per member")
        first_bytes = None
        for member, value in enumerate(inputs):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} shared admission needs prepared host arrays")
            if value.shape != spec.shape or value.dtype != np.dtype(spec.dtype):
                raise ValueError(f"{name} member {member} shape/dtype differs from allocation")
            data = value.tobytes(order="C")
            if first_bytes is None:
                first_bytes = data
            elif data != first_bytes:
                raise ValueError(
                    f"{name} differs in member {member}; sharing this input "
                    "would change that member's single-run arithmetic")
