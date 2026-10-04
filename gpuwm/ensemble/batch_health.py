"""The ordinary integration-health reduction with member-local records."""

from math import prod
from dataclasses import is_dataclass, fields as dataclass_fields, replace
from types import SimpleNamespace

import numpy as np

from gpuwm.ensemble.batch_kernel import KernelSpec, PointerSpec, prepare_batch_kernel_launch
from gpuwm.ensemble.batch_state import BatchedDomainState
from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage


def stability_memory_plan(*, members, u_shape, w_shape, theta_shape):
    """Price the same partial count as one ordinary domain for every member."""
    largest = max(prod(u_shape), prod(w_shape), prod(theta_shape))
    if min(prod(u_shape), prod(w_shape), prod(theta_shape)) <= 0:
        raise ValueError("zero-size health reductions have no maximum identity")
    blocks = min(256, max(1, (largest + 255) // 256))
    plan = BatchMemoryPlan((BatchArraySpec("health:partial", (blocks, 9), "member"),
                            BatchArraySpec("health:result", (8,), "member")), reserved_bytes=0)
    # Validate member count through the allocator's own admission inventory.
    plan.inventory(members)
    return plan, blocks


class PreparedBatchStability:
    """Two native reduction launches, one compact readback, no member mixing.

    The native max/tie/NaN/thickness statements and block reduction trees are
    unchanged. Virtual block coordinates give each member exactly the same
    partition as its ordinary single run. Forecast arrays are never written.
    """

    def __init__(self, state, cfg=None, *, boundary_width=None, available_bytes,
                 array_module=None):
        if not isinstance(state, BatchedDomainState):
            raise TypeError("member integration health needs an admitted BatchedDomainState")
        if array_module is None:
            import cupy as array_module
        from gpuwm.grid_requirements import boundary_axis
        from gpuwm.core import constants
        self.state, self.cfg, self.boundary_width = state, cfg, boundary_width
        self._xp = array_module
        self.device = int(array_module.cuda.runtime.getDevice())
        shapes = {name: state.storage.specs[name].shape for name in ("u", "w", "thp")}
        ny, nx = shapes["w"][-2:]
        width = 0 if boundary_width is None else int(boundary_width)
        if boundary_width is not None:
            if width <= 0:
                raise ValueError("boundary_width must be positive")
            if min(ny, nx) < boundary_axis(width, interior_points=1):
                raise ValueError(f"boundary_width={width} leaves an empty w interior for {ny} x {nx}")
        plan, blocks = stability_memory_plan(members=state.members, u_shape=shapes["u"],
                                              w_shape=shapes["w"], theta_shape=shapes["thp"])
        self.storage = BatchStorage(plan, state.members, array_module=array_module,
                                    available_bytes=available_bytes)
        self.plan, self.blocks = plan, blocks
        partial, result = self.storage.arrays["health:partial"], self.storage.arrays["health:result"]
        if cfg is None:
            ph, phb, ncells, phb_full = state.w, state.w, 0, 0
            phb_role = "member"
            ph_stride = phb_stride = state.storage.pointer_stride_bytes("w")
        else:
            if cfg != state.cfg:
                raise ValueError("health CFL step/grid differ from the admitted member configuration")
            ph, phb, ncells = state.php, state.phb, prod(shapes["thp"])
            phb_role = state.storage.specs["phb"].ownership
            phb_full = int(len(state.storage.specs["phb"].shape) == 3)
            ph_stride = state.storage.pointer_stride_bytes("php")
            phb_stride = state.storage.pointer_stride_bytes("phb")
        spec = KernelSpec("health", "health_partial", (
            *(PointerSpec(name, "member") for name in ("u", "w", "thp", "ph")),
            PointerSpec("phb", phb_role), PointerSpec("partial", "member")))
        strides = {name: state.storage.pointer_stride_bytes(name) for name in ("u", "w", "thp")}
        strides.update(ph=ph_stride, phb=phb_stride,
                       partial=self.storage.pointer_stride_bytes("health:partial"))
        self._partial = prepare_batch_kernel_launch(spec, state.members, (blocks,), (256,),
            (state.u, state.w, state.thp, ph, phb, partial,
             *(np.uint64(prod(shapes[name])) for name in ("u", "w", "thp")), np.uint64(ncells),
             np.int32(phb_full), np.int32(width), np.int32(ny), np.int32(nx), np.float32(constants.G)),
            pointer_strides=strides)
        spec = KernelSpec("health", "health_final", (PointerSpec("partial", "member"), PointerSpec("result", "member")))
        self._final = prepare_batch_kernel_launch(spec, state.members, (1,), (256,),
            (partial, result, np.int32(blocks)), pointer_strides={
                "partial": self.storage.pointer_stride_bytes("health:partial"),
                "result": self.storage.pointer_stride_bytes("health:result")})
        self._backings = tuple((name, id(value)) for name, value in state.storage.arrays.items())
        self.receipt = {"members": state.members, "launches_per_call": 2,
                        "record_words_per_member": 8, "partial_blocks_per_member": blocks,
                        "reduction_policy": "ordinary per-member partition and lowest-index ties",
                        "allocations": list(plan.inventory(state.members)),
                        "required_bytes": plan.required_bytes(state.members)}

    def __call__(self):
        from gpuwm.core.dycore import decode_stability_record
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("member health launch uses a different current device")
        if self._backings != tuple((name, id(value)) for name, value in self.state.storage.arrays.items()):
            raise ValueError("member state backings changed after the health reduction was bound")
        self._partial()
        self._final()
        records = self._xp.asnumpy(self.storage.arrays["health:result"])
        return tuple(decode_stability_record(record, self.cfg, boundary_width=self.boundary_width)
                     for record in records)

    def check(self, *, max_cfl, max_w_ms, phase="step"):
        """Use the unchanged scalar safety predicate on every compact record."""
        from gpuwm.core.dycore import stability_gate_failed
        reports = self()
        for member, report in enumerate(reports):
            if stability_gate_failed(report, max_cfl=max_cfl, max_w_ms=max_w_ms):
                raise FloatingPointError(f"{phase}: member {member} failed integration health: {report}")
        return reports


def strided_health_source(source):
    """Keep the native checks and map logical field indices to view strides."""
    replacements = {
        "int nfields, unsigned long long chunk)":
            "int nfields, unsigned long long chunk, const unsigned long long* shape_stride)",
        "real value = ((flags & VALIDATE_INT32)":
            "unsigned long long physical_index = member_health_offset(index, shape_stride + 12 * field);\n        real value = ((flags & VALIDATE_INT32)",
        "(real)integer_values[index] : values[index]);":
            "(real)integer_values[physical_index] : values[physical_index]);",
        "value += auxiliary[aux_index];":
            "value += auxiliary[member_health_offset(aux_index, shape_stride + 12 * field + 6)];",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("native full-state health indexing changed; its member stride audit needs renewal")
        source = source.replace(old, new)
    helper = r'''
static __device__ __forceinline__ unsigned long long member_health_offset(
    unsigned long long logical, const unsigned long long* layout) {
    unsigned long long x = logical % layout[2];
    logical /= layout[2];
    unsigned long long y = logical % layout[1];
    unsigned long long z = logical / layout[1];
    return z * layout[3] + y * layout[4] + x * layout[5];
}
'''
    return helper + source


def _layout_words(array):
    if not 1 <= array.ndim <= 3 or any(stride < 0 or stride % array.dtype.itemsize for stride in array.strides):
        raise ValueError("member health supports positive scalar-word strides in one to three dimensions")
    shape = (1,) * (3 - array.ndim) + tuple(array.shape)
    strides = (0,) * (3 - array.ndim) + tuple(stride // array.dtype.itemsize for stride in array.strides)
    return shape + strides


def _member_driver(driver, member, members, ny, nx):
    """Readout views, with no field copy or invocation of a member driver."""
    import cupy as cp
    def view(value):
        if isinstance(value, cp.ndarray):
            if value.ndim in (2, 3) and value.shape[-1] in (nx, nx + 1):
                height = value.shape[-2]
                if height in (members * ny, members * (ny + 1)) and members > 1:
                    height //= members
                    return value[:, member * height:(member + 1) * height] if value.ndim == 3 else value[member * height:(member + 1) * height]
            return value
        if isinstance(value, dict):
            return {name: view(item) for name, item in value.items()}
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{field.name: view(getattr(value, field.name))
                                    for field in dataclass_fields(value) if field.init})
        if isinstance(value, (tuple, list)):
            return type(value)(view(item) for item in value)
        return value
    names = ("pbl_tendencies", "radiation_tendencies", "cumulus_tendencies", "rthratenlw", "rthratensw",
             "cu_nca", "cu_pratec", "cu_raincv", "cu_rates", "_pending_rainbl", "fields", "microphysics")
    return SimpleNamespace(**{name: view(getattr(driver, name, None)) for name in names})


def state_health_memory_plan():
    """The exact metadata allocations used by the full-state validator."""
    from gpuwm.core.health import MAX_HEALTH_FIELDS
    declarations = tuple(BatchArraySpec("health:validate:" + name, shape, "member", dtype) for name, shape, dtype in (
        ("ptr", (MAX_HEALTH_FIELDS * 2,), "uint32"), ("aux", (MAX_HEALTH_FIELDS * 2,), "uint32"),
        ("size", (MAX_HEALTH_FIELDS * 2,), "uint32"), ("bounds", (MAX_HEALTH_FIELDS, 2), "float32"),
        ("flags", (MAX_HEALTH_FIELDS,), "uint32"), ("planes", (MAX_HEALTH_FIELDS,), "uint32"),
        ("status", (MAX_HEALTH_FIELDS * 2,), "uint32"), ("result", (2,), "uint64"),
        ("strides", (MAX_HEALTH_FIELDS, 12), "uint64")))
    return BatchMemoryPlan(declarations, reserved_bytes=0)


class PreparedBatchStateHealth:
    """The native full-state policy in one all-member validation launch.

    Packed physics is read through its actual positive strides. No model,
    physics or forcing input is copied for validation. Bounds, exclusions,
    status classes and deterministic first-bad attribution are the stock
    validator's; each member has a private two-word result.
    """

    def __init__(self, state, *, physics_driver=None, tables=None, available_bytes,
                 field_provider=None, array_module=None, member_ids=None):
        if array_module is None:
            import cupy as array_module
        from gpuwm.core.health import MAX_HEALTH_FIELDS
        if not isinstance(state, BatchedDomainState):
            raise TypeError("member full-state health needs an admitted BatchedDomainState")
        self.state, self.driver, self.tables, self.field_provider = state, physics_driver, tables, field_provider
        # The original member IDs of this pack, in pack order, so a failing
        # report names the member a user asked for and not a pack slot.
        self.member_ids = (tuple(range(state.members)) if member_ids is None
                           else tuple(int(member) for member in member_ids))
        if len(self.member_ids) != state.members:
            raise ValueError("member health needs one original member ID per packed member")
        self._xp = array_module
        self.device = int(array_module.cuda.runtime.getDevice())
        self.capacity = MAX_HEALTH_FIELDS
        self.plan = state_health_memory_plan()
        self.storage = BatchStorage(self.plan, state.members, array_module=array_module, available_bytes=available_bytes)
        self.fields = ()
        self.excluded_integer_fields = ()
        self.receipt = {"members": state.members, "launches_per_validation": 1,
            "field_input_copies": 0, "field_input_copy_bytes": 0,
            "forcing_policy": "actual member table banks with stock finiteness rules",
            "policy": "stock collect_state_fields, integer exclusions, bounds and first-bad reduction",
            "required_bytes": self.plan.required_bytes(state.members), "allocations": list(self.plan.inventory(state.members))}

    def _collect(self, member):
        from gpuwm.core.health import collect_state_fields, field_from_array
        from gpuwm.ensemble.batch_dycore import member_domain_view
        if self.field_provider is not None:
            return self.field_provider(member)
        view = member_domain_view(self.state, member)
        if self.driver is not None:
            view.physics = _member_driver(self.driver, member, self.state.members, self.state.cfg.ny, self.state.cfg.nx)
        if self.tables is not None:
            # This executor stores real forcing in separately admitted banks.
            # The unused scalar packed-table scratch is not that authority.
            view._scratch = {name: array for name, array in view._scratch.items() if not name.startswith("lbc_")}
        descriptors = list(collect_state_fields(view, backend="gpu"))
        if self.tables is not None:
            descriptors += [field_from_array("lbc." + name, self.tables.storage.member_view(name, member))
                            for name in sorted(self.tables.storage.arrays)]
        return tuple(descriptors)

    def _refresh(self):
        from gpuwm.core import health as native
        count, capacity = self.state.members, self.capacity
        host = {name: np.zeros((count,) + spec.shape, spec.dtype) for name, spec in
                ((name.removeprefix("health:validate:"), spec) for name, spec in self.storage.specs.items())}
        all_fields, all_excluded, total_elements, largest = [], [], 0, 0
        for member in range(count):
            descriptors, excluded, seen = [], [], set()
            for field in self._collect(member):
                flag = native.gpu_integer_policy(field.name, field.values.dtype)
                if flag is None:
                    excluded.append(field)
                    continue
                if not isinstance(field.values, self._xp.ndarray) or int(field.values.device.id) != self.device:
                    raise TypeError(f"member health field {field.name} needs resident owning-device storage")
                pointer = int(field.values.data.ptr)
                key = (pointer, int(field.values.size), tuple(field.values.strides), flag, field.rule,
                       id(field.auxiliary), field.aux_mode, field.plane_size)
                if key in seen:
                    continue
                seen.add(key)
                number = len(descriptors)
                if number >= capacity:
                    raise ValueError("member full-state health inventory exceeds the stock descriptor capacity")
                descriptors.append(field)
                host["ptr"][member].view(np.uint64)[number] = pointer
                host["size"][member].view(np.uint64)[number] = int(field.values.size)
                host["strides"][member, number, :6] = _layout_words(field.values)
                rule = field.rule
                host["flags"][member, number] = flag
                if rule.lower is not None:
                    host["flags"][member, number] |= native._LOWER
                    host["bounds"][member, number, 0] = np.float32(rule.lower)
                if rule.upper is not None:
                    host["flags"][member, number] |= native._UPPER
                    host["bounds"][member, number, 1] = np.float32(rule.upper)
                if rule.strict_lower:
                    host["flags"][member, number] |= native._STRICT_LOWER
                if field.auxiliary is not None:
                    auxiliary = field.auxiliary
                    if not isinstance(auxiliary, self._xp.ndarray) or auxiliary.dtype != np.dtype("float32") or int(auxiliary.device.id) != self.device:
                        raise TypeError(f"member health auxiliary for {field.name} needs owning-device float32 storage")
                    host["aux"][member].view(np.uint64)[number] = int(auxiliary.data.ptr)
                    host["strides"][member, number, 6:] = _layout_words(auxiliary)
                    host["flags"][member, number] |= native._ADD_AUX
                    if field.aux_mode == "level":
                        host["flags"][member, number] |= native._AUX_LEVEL
                        host["planes"][member, number] = field.plane_size
                    elif field.aux_mode != "direct":
                        raise ValueError("member health auxiliary mode differs from the stock direct/level policies")
                host["status"][member].view(np.uint64)[number] = rule.status_bit
            if member == 0:
                total_elements = sum(int(field.values.size) for field in descriptors)
                largest = max((int(field.values.size) for field in descriptors), default=0)
            elif [(field.name, field.values.shape, field.rule) for field in descriptors] != [(field.name, field.values.shape, field.rule) for field in all_fields[0]]:
                raise ValueError("member health inventories differ; common field ids would change first-bad attribution")
            all_fields.append(tuple(descriptors))
            all_excluded.append(tuple(excluded))
        host["result"][:, 1] = np.uint64((1 << 64) - 1)
        for name, value in host.items():
            self.storage.arrays["health:validate:" + name].set(value)
        self.fields, self.excluded_integer_fields = tuple(all_fields), tuple(all_excluded)
        chunk = max(native._MIN_CHUNK_ELEMENTS, total_elements // native._CHUNK_TARGET_BLOCKS)
        return max(1, (largest + chunk - 1) // chunk), chunk

    def validate(self, *, phase=None):
        from gpuwm.core import health as native, kernels
        from gpuwm.ensemble.batch_kernel import prepare_batch_source_launch
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("member full-state health launch uses a different current device")
        blocks, chunk = self._refresh()
        nfields = len(self.fields[0])
        if not nfields:
            return tuple(native.ValidationReport(True, 0, phase=phase) for _ in self.fields)
        names = ("field_pointer_words", "auxiliary_pointer_words", "field_size_words", "bounds", "rule_flags",
                 "plane_sizes", "status_bit_words", "result")
        slots = ("ptr", "aux", "size", "bounds", "flags", "planes", "status", "result")
        types = ("uint32", "uint32", "uint32", "float32", "uint32", "uint32", "uint32", "uint64")
        pointers = tuple(PointerSpec(name, "member", dtype) for name, dtype in zip(names, types))
        arrays = tuple(self.storage.arrays["health:validate:" + slot] for slot in slots)
        strides = {name: self.storage.pointer_stride_bytes("health:validate:" + slot) for name, slot in zip(names, slots)}
        args = arrays + (np.int32(nfields), np.uint64(chunk))
        if self.state.members == 1:
            launch = prepare_batch_kernel_launch(KernelSpec("health", "validate_full_state", pointers), 1,
                     (blocks, nfields), (256,), args, pointer_strides=strides)
        else:
            pointers += (PointerSpec("shape_stride", "member", "uint64"),)
            strides["shape_stride"] = self.storage.pointer_stride_bytes("health:validate:strides")
            launch = prepare_batch_source_launch(strided_health_source(kernels.module_source("health")),
                     KernelSpec("member_health", "validate_full_state", pointers), self.state.members,
                     (blocks, nfields), (256,), args + (self.storage.arrays["health:validate:strides"],), pointer_strides=strides)
        launch()
        words = self._xp.asnumpy(self.storage.arrays["health:validate:result"])
        reports = []
        for member, (status, packed) in enumerate(words):
            status, packed = int(status), int(packed)
            if packed == (1 << 64) - 1:
                reports.append(native.ValidationReport(True, status, phase=phase))
                continue
            number, flat = packed >> 48, packed & native._INDEX_MASK
            if number >= nfields:
                raise RuntimeError("member health kernel returned an invalid field id")
            field = self.fields[member][number]
            location = tuple(int(index) for index in np.unravel_index(flat, field.values.shape))
            value = float(field.values[location].get())
            if field.auxiliary is not None:
                auxiliary_location = location if field.aux_mode == "direct" else (flat // field.plane_size,)
                value += float(field.auxiliary[auxiliary_location].get())
            reports.append(native.ValidationReport(False, status, field.name, location, flat, value,
                                                   native._reason(value, field.rule), phase))
        self.receipt["fields_per_member"] = nfields
        self.receipt["elements_per_member"] = sum(int(field.values.size) for field in self.fields[0])
        self.receipt["field_names"] = [field.name for field in self.fields[0]]
        return tuple(reports)

    def require_healthy(self, *, phase=None):
        """Raise for the first failing member, naming it and every other one.

        All members of a pack share one step count, so nothing but this
        error can say which of them went bad. Breakage this prevents: a
        full-state failure in a pack of ten named a field and an index and
        no member, in the error and in ensemble-run.json.
        """
        from gpuwm.core.health import HealthCheckError
        reports = self.validate(phase=phase)
        failing = [(member, report) for member, report in zip(self.member_ids, reports)
                   if not report.ok]
        if failing:
            member, report = failing[0]
            error = HealthCheckError(report)
            others = [other for other, _ in failing[1:]]
            also = ("" if not others else "; also failing: member"
                    + ("s " if len(others) > 1 else " ") + ", ".join(str(other) for other in others))
            error.args = (f"member {member}: {error.args[0]}{also}",)
            error.ensemble_member_ids = (member,)
            error.ensemble_failing_member_ids = tuple(other for other, _ in failing)
            raise error
        return reports
