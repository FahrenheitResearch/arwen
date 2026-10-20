"""Metadata-only allocation contracts for table-selected hydro factories.

Factories publish ``<entry>_memory(settings, options, context, components)``.
The hook reads prepared metadata, never model values or initialized CUDA arrays.
Missing owner inventories remain named gaps, never zero-byte shipping evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import nullcontext
import importlib
import math
from types import MappingProxyType

_WIDTHS = {"float32": 4, "float64": 8, "int32": 4, "uint32": 4,
           "int64": 8, "uint64": 8, "uint8": 1, "bool": 1}


@dataclass(frozen=True)
class MemoryContext:
    mode: str
    numerics: str
    interval_seconds: float
    parent_shape: tuple[int, ...] = ()
    duration_seconds: float | None = None

    def __post_init__(self):
        if self.mode not in ("coupled", "standalone") or self.numerics not in ("fast", "strict"):
            raise ValueError("hydro memory context needs mode and numerics")
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("hydro memory context needs a positive interval")
        if self.duration_seconds is not None and (
                not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0):
            raise ValueError("hydro memory duration must be finite and positive")
        object.__setattr__(self, "parent_shape", tuple(self.parent_shape))
        if any(type(n) is not int or n < 0 for n in self.parent_shape):
            raise ValueError("hydro parent shape needs nonnegative integer extents")


@dataclass(frozen=True)
class ArrayMemory:
    """One backing allocation or a view of an earlier named backing.

    Alias targets are full ``stage/array`` names. Offsets address the backing,
    not another view. The owner includes padding in backing_bytes and names
    setup/step/output temporaries as well as checkpoint carriers. Lifetimes
    are diagnostic: the pool may retain freed blocks, so unrelated transient
    buffers are conservatively summed, never assumed reusable.
    """
    name: str
    shape: tuple[int, ...]
    dtype: str
    lifetime: str = "resident"
    alias_of: str | None = None
    offset_bytes: int = 0
    backing_bytes: int | None = None
    checkpointed: bool = False
    status: bool = False
    # Diagnostic key within the stage; checkpoint names remain unchanged.
    status_name: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.name.startswith("/"):
            raise ValueError("hydro memory arrays need relative names")
        object.__setattr__(self, "shape", tuple(self.shape))
        if any(type(n) is not int or n < 0 for n in self.shape) or self.dtype not in _WIDTHS:
            raise ValueError(f"invalid hydro memory shape/dtype for {self.name}")
        if self.lifetime not in ("resident", "prepare", "step", "output"):
            raise ValueError(f"unknown hydro allocation lifetime for {self.name}")
        if type(self.offset_bytes) is not int or self.offset_bytes < 0:
            raise ValueError("hydro memory offset must be a nonnegative integer")
        if self.alias_of is not None and (not isinstance(self.alias_of, str) or not self.alias_of):
            raise ValueError("hydro alias must name an earlier allocation")
        if self.alias_of is None and self.offset_bytes:
            raise ValueError("a hydro backing cannot have a view offset")
        if self.backing_bytes is not None and (self.alias_of is not None
                or type(self.backing_bytes) is not int or self.backing_bytes < self.nbytes):
            raise ValueError("hydro backing bytes must cover the array and cannot override an alias")
        if self.checkpointed and self.lifetime != "resident":
            raise ValueError("checkpointed hydro arrays must be resident")
        if self.status and (not self.checkpointed or self.dtype not in ("int32", "uint32")
                            or len(self.shape) != 1 or not self.shape[0]):
            raise ValueError("hydro status must be a checkpointed nonempty int32/uint32 vector")

        if self.status_name is not None and (
                not self.status or not isinstance(self.status_name, str)
                or not self.status_name or "/" in self.status_name):
            raise ValueError("hydro status name must bind a status carrier to a flat diagnostic key")
        if self.status and "/" in (self.status_name or self.name):
            raise ValueError("hydro status name must be a flat diagnostic key")

    @property
    def nbytes(self):
        return math.prod(self.shape) * _WIDTHS[self.dtype]


@dataclass(frozen=True)
class StageMemory:
    arrays: tuple[ArrayMemory, ...]
    evidence: str
    # Owner's incremental kernel/local-memory device bound, outside the pool.
    # None is an explicit missing inventory, never an inferred zero.
    non_pool_bytes: int | None = None
    gaps: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "arrays", tuple(self.arrays))
        if not self.evidence or not all(isinstance(a, ArrayMemory) for a in self.arrays):
            raise ValueError("hydro stage memory needs owner evidence and typed arrays")
        if len({a.name for a in self.arrays}) != len(self.arrays):
            raise ValueError("duplicate hydro memory array names")
        statuses = [a.status_name or a.name for a in self.arrays if a.status]
        if len(set(statuses)) != len(statuses):
            raise ValueError("duplicate hydro status names")
        if self.non_pool_bytes is not None and (type(self.non_pool_bytes) is not int
                                               or self.non_pool_bytes < 0):
            raise ValueError("hydro non-pool device bound must be a nonnegative integer")
        object.__setattr__(self, "gaps", tuple(self.gaps))
        if any(not isinstance(gap, str) or not gap for gap in self.gaps):
            raise ValueError("hydro memory gaps must be named")


@dataclass(frozen=True)
class HydroMemoryPlan:
    context: MemoryContext
    stages: tuple[tuple[str, StageMemory], ...]
    gaps: tuple[str, ...]
    settings_sha256: str

    def inventory(self, *, members=1):
        if type(members) is not int or members < 1:
            raise ValueError("hydro memory members must be a positive integer")
        backings, arrays, names = {}, [], set()
        for stage, plan in self.stages:
            for array in plan.arrays:
                key = f"{stage}/{array.name}"
                if key in names:
                    raise ValueError(f"duplicate hydro allocation {key}")
                names.add(key)
                if array.alias_of is None:
                    size = array.nbytes if array.backing_bytes is None else array.backing_bytes
                    # CuPy rounds each independent allocation, before member
                    # replication. This is not one member-batched backing.
                    backings[key] = {"bytes": size, "allocated_bytes": (size + 511) // 512 * 512,
                                     "lifetime": array.lifetime}
                else:
                    target = backings.get(array.alias_of)
                    if target is None or array.offset_bytes + array.nbytes > target["bytes"]:
                        raise ValueError(f"hydro alias {key} is outside its earlier backing {array.alias_of}")
                    if target["lifetime"] != "resident" and target["lifetime"] != array.lifetime:
                        raise ValueError(f"hydro alias {key} outlives its backing")
                arrays.append(dict(asdict(array), name=key, logical_bytes=array.nbytes,
                    status_name=f"{stage}/{array.status_name or array.name}" if array.status else None))
        resident = sum(b["allocated_bytes"] for b in backings.values() if b["lifetime"] == "resident")
        transient = sum(b["allocated_bytes"] for b in backings.values() if b["lifetime"] != "resident")
        non_pool = sum(stage.non_pool_bytes or 0 for _, stage in self.stages)
        row = {"schema": "arwen.hydro.memory-plan/v1", "complete": not self.gaps,
                "pre_factory_inventory": True, "gaps": list(self.gaps),
                "settings_sha256": self.settings_sha256, "context": asdict(self.context),
                "members": members, "independent_members": True,
                "resident_bytes": resident * members, "transient_bytes": transient * members,
                "non_pool_bytes": non_pool * members,
                "required_bytes": (resident + transient + non_pool) * members,
                "backings": backings, "arrays": arrays,
                "evidence": {name: stage.evidence for name, stage in self.stages},
                "scope": "owner pool allocations and incremental non-pool bounds; no cadence reuse"}
        from .config import _digest
        row["inventory_sha256"] = _digest(row)
        return row


def _workspace_memory(components):
    """Bound the default writer and driver's boundary reductions.

    The writer may pack every checkpoint carrier, including overlapping
    views, simultaneously. Price each logical carrier independently even if
    it is contiguous today. Status reductions need one scalar per vector,
    a stack, and reduction scratch bounded by each nonempty input vector.
    These buffers are not checkpointed model state.
    """
    arrays = []
    statuses = 0
    for stage, value in components.items():
        if value is None:
            continue
        for index, array in enumerate(value.arrays):
            if array.checkpointed:
                arrays.append(ArrayMemory(f"pack/{stage}/{index}", array.shape,
                                          array.dtype, lifetime="output"))
                arrays.append(ArrayMemory(f"restore/{stage}/{index}", array.shape,
                                          array.dtype, lifetime="output"))
            if array.status:
                statuses += 1
                arrays.append(ArrayMemory(f"reduce/{stage}/{index}", array.shape,
                                          array.dtype, lifetime="output"))
                arrays.append(ArrayMemory(f"scalar/{stage}/{index}", (), "int32",
                                          lifetime="output"))
    if statuses:
        arrays.append(ArrayMemory("status_stack", (statuses,), "int32", lifetime="output"))
    # The framework's generated reduction/copy kernels still need a device
    # resource bound; an allocation inventory alone cannot assert that bound.
    return StageMemory(tuple(arrays), "driver.py boundary_report; output.py _snapshot/restore_checkpoint",
                       None if arrays else 0)


def plan(settings, *, context):
    """Resolve all metadata hooks before any prebuild or stage factory call."""
    from .config import _copy, _digest
    components, gaps = {}, []
    for row in settings.get("stages", ()):
        name = row["name"]
        if not isinstance(name, str) or not name or "/" in name or name in components:
            raise ValueError("hydro memory stages need unique flat names")
        module, separator, entry = row["factory"].partition(":")
        if not separator or not module.startswith("gpuwm."):
            raise ValueError(f"hydro stage {name} needs a gpuwm module:factory")
        hook = getattr(importlib.import_module(module), entry + "_memory", None)
        if hook is None:
            gaps.append(f"{name}: {row['factory']}_memory has no allocation inventory")
            components[name] = None
            continue
        value = hook(settings=_copy(settings), options=_copy(row.get("options", {})),
                     context=context, components=MappingProxyType(components))
        if value is None:
            gaps.append(f"{name}: prepared metadata dependencies have no allocation inventory")
        elif not isinstance(value, StageMemory):
            raise TypeError(f"{name} memory hook must return StageMemory or a named dependency gap")
        components[name] = value
    if not components:
        gaps.append("realization has no prepared stage rows")
    for name, value in components.items():
        if value is not None:
            gaps.extend(f"{name}: {gap}" for gap in value.gaps)
            if value.non_pool_bytes is None:
                gaps.append(f"{name}: incremental kernel/local-memory device bound is unpriced")
    if settings.get("driver_factory", "gpuwm.hydro.assembly:prepare") != "gpuwm.hydro.assembly:prepare":
        gaps.append("custom driver_factory has no assembly allocation contract")
    if settings.get("output_factory", "gpuwm.hydro.output:prepare") != "gpuwm.hydro.output:prepare":
        gaps.append("custom output_factory has no default writer workspace contract")
    # Reserve a non-user stage name for buffers owned by this assembly.
    if "_assembly" in components:
        raise ValueError("hydro memory stage name _assembly is reserved")
    components["_assembly"] = _workspace_memory(components)
    if components["_assembly"].non_pool_bytes is None:
        gaps.append("_assembly: generated status/copy kernel device resource bound is unpriced")
    result = HydroMemoryPlan(context, tuple((k, v) for k, v in components.items() if v is not None),
                             tuple(gaps), _digest(settings))
    result.inventory()  # Validate cross-stage aliases before device construction.
    return result


def forecast_plan(exp):
    """The current forecast scope's plan, or None for an inactive hydro door."""
    from .runtime import current_settings, resolve_settings
    settings = current_settings()
    if not settings or settings.get("enabled") is not True:
        return None
    settings = resolve_settings(settings)
    cfg = exp.root.run
    dt, minutes = float(cfg.dt), float(cfg.bldt)
    interval = max(math.floor(minutes * 60 / dt + 0.5), 1) * dt if minutes > 0 else dt
    value = plan(settings, context=MemoryContext("coupled", settings["numerics"], interval,
                 (int(cfg.nz), int(cfg.ny), int(cfg.nx)), float(exp.run_seconds)))
    # Collectors live outside stage/checkpoint arrays. No bound is inferred
    # from stage completeness or from a private product-pool capacity.
    return replace(value, gaps=value.gaps + (
        "forecast: product collector/scheduler allocation contract is unpriced",))


def construction_plan(settings, *, physics, state, mode):
    shape = tuple(getattr(getattr(state, "thp", None), "shape", ()))
    interval = physics.bldt_seconds if mode == "coupled" else settings["interval_seconds"]
    return plan(settings, context=MemoryContext(mode, settings["numerics"], float(interval), shape))


def _device_scope(device_id):
    if device_id is None:
        return nullcontext()
    import cupy as cp
    return cp.cuda.Device(device_id)


def admit_plan(value, *, device_id=None):
    """Use the existing envelope gate before any stage prebuild/allocation.

    An incomplete inventory stays runnable with a named PARTIAL receipt. It
    cannot claim a successful pre-factory admission from a lower bound.
    """
    row = value.inventory()
    row["device_id"] = device_id
    row["pre_factory_admission"] = False
    if not row["complete"]:
        row["admission"] = {"fits": None, "reason": "incomplete hydro allocation inventory"}
        return row
    from gpuwm.core.resident_admission import admit
    with _device_scope(device_id):
        row["admission"] = admit("NextGen hydro construction envelope", {
            "hydro resident arrays": row["resident_bytes"],
            "hydro prepare/step/output workspaces": row["transient_bytes"],
            "hydro incremental kernel/local memory": row["non_pool_bytes"],
        }, stage="constructing the selected hydro stages", envelope=True)
    row["pre_factory_admission"] = (row["admission"]["fits"] is True
                                    and not row["admission"].get("overridden", False))
    return row


def verify_driver(row, driver):
    """Cross-check declared checkpoint metadata; this is not admission."""
    if not row["complete"]:
        return
    declared = {a["name"]: a for a in row["arrays"] if a["checkpointed"]}
    if set(declared) != set(driver.arrays):
        raise ValueError("hydro allocation contract differs from checkpoint carrier roster")
    for name, array in driver.arrays.items():
        if (tuple(declared[name]["shape"]) != tuple(array.shape)
                or declared[name]["dtype"] != str(array.dtype)):
            raise ValueError(f"hydro allocation contract differs from checkpoint shape/dtype: {name}")
    statuses = {a.get("status_name") or name: name
                for name, a in declared.items() if a["status"]}
    if len(statuses) != sum(bool(a["status"]) for a in declared.values()) or set(statuses) != set(driver.status_arrays):
        raise ValueError("hydro allocation contract differs from status carrier roster")
    for label, name in statuses.items():
        status, carrier = driver.status_arrays[label], driver.arrays[name]
        if status is carrier:
            continue
        # The driver permits a separate contiguous view of a checkpoint
        # carrier. Verify its exact storage binding without reading values.
        pointer = getattr(getattr(status, "data", None), "ptr", None)
        if (pointer is None or pointer != getattr(getattr(carrier, "data", None), "ptr", None)
                or status.nbytes != carrier.nbytes or tuple(status.shape) != tuple(carrier.shape)
                or str(status.dtype) != str(carrier.dtype)):
            raise ValueError(f"hydro allocation contract differs from status carrier binding: {label}")
