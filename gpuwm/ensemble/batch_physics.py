"""Column layout and preallocated native physics launch seams.

Member dycore slabs are (member, level, y, x). Column schemes consume
(level, member * y, x). Integer-only copies join or separate columns without
constructing a larger C-grid: staggered winds and lateral masks must be
prepared on each member's own grid before this seam. This module is a set of
physics primitives, not yet a full PhysicsDriver replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from operator import index
import re
from hashlib import sha256
import ast
import inspect
import textwrap

import numpy as np

from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan, BatchStorage
from gpuwm.certify.kernel_manifest import record_module


def remap_shared_column_loads(source, pointers, *, member_columns):
    """Integer-remap explicitly immutable native column loads to one slab.

    Only direct ``pointer[idx]`` or ``pointer[col]`` loads are supported.
    A different indexing expression or a store needs a separately reviewed
    adapter. No floating expression is rewritten. This source audit is not
    a GPU identity result: every modified family still needs word gates.
    """
    columns = _positive(member_columns, "member columns")
    for pointer in pointers:
        if not re.fullmatch(r"[A-Za-z_]\w*", pointer):
            raise ValueError("shared column pointers need CUDA identifiers")
        pattern = re.compile(rf"\b{re.escape(pointer)}\s*\[([^\]]+)\]")
        matches = list(pattern.finditer(source))
        if not matches:
            raise ValueError(f"shared column pointer {pointer} has no audited loads")
        for match in matches:
            if match[1].strip() not in ("idx", "col"):
                raise ValueError(f"{pointer} uses unsupported shared column indexing {match[1]!r}")
            after = source[match.end():].lstrip()
            if re.match(r"(?:=(?!=)|[+*/%&|^-]=|\+\+|--)", after):
                raise ValueError(f"{pointer} is written; sharing it would mix member state")
        source = pattern.sub(lambda m: f"{pointer}[({m[1].strip()}) % {columns}]", source)
    return source


def _native_column_kernel(module, entry, *, shared_pointers=(), member_columns):
    from gpuwm.core import kernels
    if not shared_pointers:
        return kernels.get_kernel(module, entry), {"source_policy": "unchanged native kernel"}
    import cupy as cp
    from gpuwm import wrf_exact
    from gpuwm.certify.kernel_manifest import record_module
    original = kernels.module_source(module)
    source = remap_shared_column_loads(original, shared_pointers, member_columns=member_columns)
    options = wrf_exact.effective_options(("-std=c++17",)) if wrf_exact.ENABLED else ("-std=c++17",)
    compiled = cp.RawModule(code=source, options=options)
    key = f"gpuwm.ensemble.batch_physics:{module}:{entry}:shared-column-loads[columns={member_columns}]"
    kernels._compile_observed(compiled, key)
    record_module(key, source=source, options=options, module=compiled)
    return compiled.get_function(entry), {"source_policy": "integer-only shared column indexing",
                                          "scalar_source_sha256": sha256(original.encode()).hexdigest(),
                                          "source_sha256": sha256(source.encode()).hexdigest(),
                                          "shared_pointers": list(shared_pointers)}

_LAYOUT_SOURCE = r'''
extern "C" __global__ void ensemble_column_pack_words(
    const unsigned int *source, unsigned int *destination,
    unsigned long long plane, int levels, int members, int reverse) {
    unsigned long long word = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = plane * (unsigned long long)levels * members;
    if (word >= total) return;
    unsigned long long cell = word % plane;
    unsigned long long column = word / plane;
    unsigned long long level = column % levels, member = column / levels;
    unsigned long long packed = (level * members + member) * plane + cell;
    if (reverse) destination[word] = source[packed];
    else destination[packed] = source[word];
}
extern "C" __global__ void ensemble_column_mask_ring(
    unsigned int *field, int levels, int members, int ny, int nx, int width) {
    unsigned long long point = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long plane = (unsigned long long)ny * nx;
    if (point >= plane * levels * members) return;
    unsigned long long cell = point % plane;
    int x = cell % nx, y = cell / nx;
    if (x < width || x >= nx - width || y < width || y >= ny - width)
        field[point] = 0U;
}
'''


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    value = index(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class ColumnField:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ValueError("column field needs an identifier")
        shape = tuple(_positive(n, self.name) for n in self.shape)
        if len(shape) not in (2, 3):
            raise ValueError("column field shape must be (y,x) or (level,y,x)")
        dtype = np.dtype(self.dtype)
        if dtype not in (np.dtype("float32"), np.dtype("int32"), np.dtype("uint32")):
            raise ValueError("column layout copies preserve 32-bit scalar words")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype.str)


def column_memory_plan(fields, *, reserved_bytes=0):
    fields = tuple(fields)
    if not fields or any(not isinstance(f, ColumnField) for f in fields):
        raise ValueError("column plan needs ColumnField declarations")
    # Backing allocations retain member-private ownership. Their views use
    # (level,member*y,x), and BatchStorage.member_view is deliberately not used
    # while those backings contain packed fields.
    return BatchMemoryPlan(tuple(BatchArraySpec(f"physics:{f.name}", f.shape,
                                               "member", f.dtype) for f in fields),
                           reserved_bytes=reserved_bytes)


class ColumnBuffers:
    """Explicit member-private staging, with no floating-point conversion."""

    def __init__(self, fields, *, members, available_bytes, reserved_bytes=0,
                 array_module=None):
        if array_module is None:
            import cupy as array_module
        self._xp = array_module
        self.fields = {field.name: field for field in fields}
        self.members = _positive(members, "members")
        self.device = int(array_module.cuda.runtime.getDevice())
        self.storage = BatchStorage(column_memory_plan(tuple(self.fields.values()),
                                                       reserved_bytes=reserved_bytes),
                                    self.members, array_module=array_module,
                                    available_bytes=available_bytes)
        module = array_module.RawModule(code=_LAYOUT_SOURCE, options=("--std=c++17",))
        self._copy = module.get_function("ensemble_column_pack_words")
        self._mask = module.get_function("ensemble_column_mask_ring")
        record_module("gpuwm.ensemble.batch_physics:column-layout", source=_LAYOUT_SOURCE,
                      options=("--std=c++17",), module=module)
        self.converted_bytes = 0

    def packed(self, name):
        field = self.fields[name]
        ny, nx = field.shape[-2:]
        shape = ((field.shape[0], self.members * ny, nx) if len(field.shape) == 3
                 else (self.members * ny, nx))
        return self.storage.arrays[f"physics:{name}"].reshape(shape)

    def _validate(self, name, member_array):
        field = self.fields[name]
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("physics layout launch uses a different current device")
        if (not isinstance(member_array, self._xp.ndarray)
                or member_array.shape != (self.members,) + field.shape
                or member_array.dtype != np.dtype(field.dtype)
                or not member_array.flags.c_contiguous
                or int(member_array.device.id) != self.device):
            raise ValueError(f"{name} needs contiguous member-private {field.dtype} {field.shape} slabs")
        source = int(member_array.data.ptr)
        target = self.storage.arrays[f"physics:{name}"]
        destination = int(target.data.ptr)
        if source < destination + target.nbytes and destination < source + member_array.nbytes:
            raise ValueError("physics layout input and output overlap; packing in place would destroy member words")
        return field, target

    def pack(self, name, member_array):
        field, target = self._validate(name, member_array)
        plane = prod(field.shape[-2:])
        levels = field.shape[0] if len(field.shape) == 3 else 1
        words = plane * levels * self.members
        self._copy(((words + 255) // 256,), (256,),
                   (member_array, target, np.uint64(plane), np.int32(levels),
                    np.int32(self.members), np.int32(0)))
        self.converted_bytes += 2 * member_array.nbytes
        return self.packed(name)

    def unpack(self, name, member_array):
        field, source = self._validate(name, member_array)
        plane = prod(field.shape[-2:])
        levels = field.shape[0] if len(field.shape) == 3 else 1
        words = plane * levels * self.members
        self._copy(((words + 255) // 256,), (256,),
                   (source, member_array, np.uint64(plane), np.int32(levels),
                    np.int32(self.members), np.int32(1)))
        self.converted_bytes += 2 * member_array.nbytes
        return member_array

    def mask_member_rings(self, name, *, width=1):
        """Mask each member's mass ring in a packed scalar tendency.

        This is not a C-grid momentum coupler or microphysics ring restore.
        Both have additional contracts and must use separate member-aware
        kernels. The mask uses integer stores of positive-zero words.
        """
        field = self.fields[name]
        width = _positive(width, "ring width")
        ny, nx = field.shape[-2:]
        if 2 * width >= min(ny, nx):
            raise ValueError("ring mask would consume the member's whole grid")
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("physics ring launch uses a different current device")
        levels = field.shape[0] if len(field.shape) == 3 else 1
        words = self.members * prod(field.shape)
        self._mask(((words + 255) // 256,), (256,),
                   (self.storage.arrays[f"physics:{name}"], np.int32(levels),
                    np.int32(self.members), np.int32(ny), np.int32(nx), np.int32(width)))


YSU_COLUMNS = ("u", "v", "theta", "qv", "qc", "qi", "p", "p_interface", "exner", "dz", "rthraten")
YSU_SURFACES = ("psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim", "psih", "xland", "u10", "v10")
YSU_VOLUME_OUTPUTS = ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "exch_m")
YSU_SURFACE_OUTPUTS = ("hpbl", "kpbl", "wstar", "delta", "topdown_radsum", "wstar3_2", "cloudflg")


def ysu_output_plan(*, nz, ny, nx, members, reserved_bytes=0):
    """All-member one-launch YSU outputs plus its full launch workspace."""
    from gpuwm.core.physics_inventory import ysu_workspace_floats
    nz, ny, nx, members = [_positive(v, n) for v, n in ((nz, "nz"), (ny, "ny"), (nx, "nx"), (members, "members"))]
    if not 4 <= nz <= 128:
        raise ValueError("YSU's vertical solve requires 4..128 levels")
    specs = [BatchArraySpec(f"ysu:{name}", (nz, ny, nx), "member") for name in YSU_VOLUME_OUTPUTS]
    specs.extend(BatchArraySpec(f"ysu:{name}", (ny, nx), "member",
                               "int32" if name in ("kpbl", "cloudflg") else "float32")
                 for name in YSU_SURFACE_OUTPUTS)
    # Unlike a per-member loop, this single launch prices every column's
    # workspace. Admission can choose this launch or a qualified tiled
    # family later, but cannot hide the memory behind an average multiplier.
    specs.append(BatchArraySpec("ysu:workspace", (ysu_workspace_floats(nz, members * ny * nx),), "shared"))
    return BatchMemoryPlan(tuple(specs), reserved_bytes=reserved_bytes)


def prepare_ysu_column_batch(inputs, *, members, ny, nx, dt, available_bytes,
                             topdown=1, array_module=None, reserved_bytes=0,
                             shared_fields=()):
    """Bind the unchanged native YSU kernel over already packed columns.

    Inputs must already have member rings, static ownership and A-grid winds
    established by their producer. This leaf returns packed raw rates only;
    it never constructs faces or applies rates to prognostic state.
    """
    if array_module is None:
        import cupy as array_module
    from gpuwm.core.physics_inventory import YSU_BLOCK
    members, ny, nx = _positive(members, "members"), _positive(ny, "ny"), _positive(nx, "nx")
    if set(inputs) != set(YSU_COLUMNS + YSU_SURFACES):
        raise ValueError("YSU needs the complete named packed column/surface input inventory")
    theta = inputs["theta"]
    nz = theta.shape[0]
    if not np.isfinite(dt) or dt <= 0 or topdown not in (0, 1):
        raise ValueError("YSU needs positive finite dt and topdown 0 or 1")
    device = int(array_module.cuda.runtime.getDevice())
    shared_fields = tuple(shared_fields)
    if set(shared_fields) - {"xland"}:
        raise ValueError("YSU shared fields must be immutable xland; its other inputs are member diagnostics")
    for name, array in inputs.items():
        shape = ((nz + (name == "p_interface"), members * ny, nx)
                 if name in YSU_COLUMNS else (members * ny, nx))
        if name in shared_fields:
            shape = (ny, nx)
        if (not isinstance(array, array_module.ndarray) or array.shape != shape
                or array.dtype != np.dtype("float32") or not array.flags.c_contiguous
                or int(array.device.id) != device):
            raise ValueError(f"YSU {name} needs resident contiguous float32 packed shape {shape}")
    plan = ysu_output_plan(nz=nz, ny=ny, nx=nx, members=members, reserved_bytes=reserved_bytes)
    storage = BatchStorage(plan, members, array_module=array_module, available_bytes=available_bytes)
    outputs = {}
    for name in YSU_VOLUME_OUTPUTS + YSU_SURFACE_OUTPUTS:
        array = storage.arrays[f"ysu:{name}"]
        shape = (nz, members * ny, nx) if name in YSU_VOLUME_OUTPUTS else (members * ny, nx)
        outputs[name] = array.reshape(shape)
    kernel, source_receipt = _native_column_kernel("ysu", "ysu_column",
        shared_pointers=shared_fields if members > 1 else (), member_columns=ny * nx)
    args = tuple(inputs[name] for name in YSU_COLUMNS + YSU_SURFACES)
    args += tuple(outputs[name] for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "hpbl", "kpbl", "exch_h", "exch_m", "wstar", "delta"))
    args += (np.float32(dt), outputs["topdown_radsum"], outputs["wstar3_2"], outputs["cloudflg"],
             np.int32(topdown), np.int32(nz), np.int32(members * ny), np.int32(nx),
             storage.arrays["ysu:workspace"], np.int32(nz + 1), np.int32(0))
    columns = members * ny * nx

    def launch():
        if int(array_module.cuda.runtime.getDevice()) != device:
            raise RuntimeError("YSU column launch uses a different current device")
        kernel(((columns + YSU_BLOCK - 1) // YSU_BLOCK,), (YSU_BLOCK,), args)
        return outputs

    launch.storage = storage
    launch.outputs = outputs
    launch.inputs = dict(inputs)
    launch.members = members
    launch.receipt = {"layout": "level-member-y-x", "members": members,
                      "launches_per_call": 1, "entry": "ysu_column",
                      **source_receipt,
                      "shared_fields": list(shared_fields),
                      "static_input_policy": "explicit immutable loads index one member grid",
                      "allocations": list(plan.inventory(members)),
                      "required_bytes": plan.required_bytes(members)}
    return launch


SFCLAY_INPUTS = ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "pblh", "mavail", "xland", "lakemask")


def prepare_sfclay_column_batch(inputs, outputs, *, members, ny, nx, option, dx,
                                isfflx=True, isftcflx=0, iz0tlnd=0, shared_fields=(),
                                array_module=None):
    """One MM5 launch into explicitly supplied member-private inout fields."""
    if array_module is None:
        import cupy as array_module
    from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
    from gpuwm.core.sfclay import _validate_options, _TPB
    _validate_options(option, isftcflx, iz0tlnd)
    members, ny, nx = _positive(members, "members"), _positive(ny, "ny"), _positive(nx, "nx")
    if set(inputs) != set(SFCLAY_INPUTS) or set(outputs) != set(SFCLAY_OUTPUTS):
        raise ValueError("MM5 needs its complete named input/inout/output inventory")
    shared_fields = tuple(shared_fields)
    if set(shared_fields) - {"xland", "lakemask"}:
        raise ValueError("MM5 shared fields are immutable land/lake masks only")
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("MM5 needs positive finite grid spacing")
    device = int(array_module.cuda.runtime.getDevice())
    for name, array in dict(inputs, **outputs).items():
        shape = (ny, nx) if name in shared_fields else (members * ny, nx)
        if (not isinstance(array, array_module.ndarray) or array.shape != shape
                or array.dtype != np.dtype("float32") or not array.flags.c_contiguous
                or int(array.device.id) != device):
            raise ValueError(f"MM5 {name} needs resident contiguous float32 shape {shape}")
    # Inputs are const in this family. Inout fields may be aliased with their
    # owning driver's public diagnostics but cannot alias a shared mask.
    from gpuwm.ensemble.batch_products import _refuse_overlap
    _refuse_overlap(inputs, outputs)
    kernel, source_receipt = _native_column_kernel("sfclay", "sfclay_column",
        shared_pointers=shared_fields if members > 1 else (), member_columns=ny * nx)
    points = members * ny * nx
    args = tuple(inputs[name] for name in SFCLAY_INPUTS) + tuple(outputs[name] for name in SFCLAY_OUTPUTS)
    args += (np.float32(dx), np.int32(option), np.int32(isfflx), np.int32(isftcflx), np.int32(iz0tlnd), np.int32(points))

    def launch():
        if int(array_module.cuda.runtime.getDevice()) != device:
            raise RuntimeError("MM5 column launch uses a different current device")
        kernel(((points + _TPB - 1) // _TPB,), (_TPB,), args)
        return outputs

    launch.inputs, launch.outputs = dict(inputs), dict(outputs)
    launch.receipt = {"members": members, "launches_per_call": 1,
                      "entry": "sfclay_column", "shared_fields": list(shared_fields), **source_receipt}
    return launch


NOAH_SHARED_FIELDS = frozenset({"ivgtyp", "isltyp", "shdmin", "shdmax", "tmn", "xland", "snoalb", "embck"})


def prepare_noah_column_batch(fields, params, *, members, ny, nx, dt, dzs,
                              shared_fields=(), isurban=13, isice=15,
                              xice_threshold=0.5, frpcpn=False, usemonalb=False,
                              rdlai2d=False, opt_thcnd=1, array_module=None):
    """One nonurban Noah launch, shared lookup tables and explicit static loads.

    The caller owns every mutable land-state/output backing and includes it
    in admission. The step number is supplied at launch so the initial-call
    branches follow the same fixed clock as independent members.
    """
    if array_module is None:
        import cupy as array_module
    from gpuwm.core.noah import _F2D, _F3D, _TPB, _device_tables, NUM_SOIL_LAYERS
    members, ny, nx = _positive(members, "members"), _positive(ny, "ny"), _positive(nx, "nx")
    shared_fields = tuple(shared_fields)
    if set(shared_fields) - NOAH_SHARED_FIELDS:
        raise ValueError("Noah sharing must name an immutable const grid input")
    if set(fields) != set(_F2D + _F3D) | {"ivgtyp", "isltyp", "ebal"}:
        raise ValueError("Noah needs its complete named mutable/static field inventory")
    if not np.isfinite(dt) or dt <= 0 or len(dzs) != NUM_SOIL_LAYERS:
        raise ValueError("Noah needs positive finite dt and four soil-layer thicknesses")
    device = int(array_module.cuda.runtime.getDevice())
    for name, array in fields.items():
        shape = ((4, members * ny, nx) if name in _F3D else
                 (ny, nx) if name in shared_fields else (members * ny, nx))
        dtype = np.dtype("int32" if name in ("ivgtyp", "isltyp", "ebal") else "float32")
        if (not isinstance(array, array_module.ndarray) or array.shape != shape
                or array.dtype != dtype or not array.flags.c_contiguous
                or int(array.device.id) != device):
            raise ValueError(f"Noah {name} needs resident contiguous {dtype} shape {shape}")
    tables = _device_tables(params, dzs)
    shared_pointers = tuple(name if name in ("ivgtyp", "isltyp") else f"{name}_a" for name in shared_fields)
    kernel, source_receipt = _native_column_kernel("noah", "noah_column",
        shared_pointers=shared_pointers if members > 1 else (), member_columns=ny * nx)
    args = (fields["ivgtyp"], fields["isltyp"]) + tuple(fields[name] for name in _F2D + _F3D) + (fields["ebal"],)
    args += tables + (np.float32(dt), np.int32(params.lucats), np.int32(params.slcats), np.int32(isurban),
                      np.int32(isice), np.float32(xice_threshold))
    tail = (np.int32(frpcpn), np.int32(usemonalb), np.int32(rdlai2d), np.int32(opt_thcnd),
            np.int32(members * ny), np.int32(nx))
    points = members * ny * nx

    def launch(itimestep):
        timestep = _positive(itimestep, "Noah timestep")
        if int(array_module.cuda.runtime.getDevice()) != device:
            raise RuntimeError("Noah column launch uses a different current device")
        kernel(((points + _TPB - 1) // _TPB,), (_TPB,), args + (np.int32(timestep),) + tail)
        return fields

    launch.fields, launch.tables = dict(fields), tables
    launch.receipt = {"members": members, "launches_per_call": 1, "entry": "noah_column",
                      "shared_fields": list(shared_fields), "lookup_table_bytes": sum(a.nbytes for a in tables),
                      **source_receipt}
    return launch


_RING_SOURCE = r'''
extern "C" __global__ void ensemble_physics_ring_words(
    const unsigned long long *table, int records, int members, int ny, int nx,
    int width, unsigned long long ring_cells, int restore) {
    int record = blockIdx.y;
    if (record >= records) return;
    const unsigned long long *row = table + record * 4;
    unsigned int *field = reinterpret_cast<unsigned int *>(row[0]);
    unsigned int *saved = reinterpret_cast<unsigned int *>(row[1]);
    unsigned long long levels = row[2], zero = row[3];
    unsigned long long point = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = (unsigned long long)members * levels * ring_cells;
    unsigned long long plane = (unsigned long long)ny * nx;
    for (; point < count; point += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long ring = point % ring_cells;
        unsigned long long column = point / ring_cells;
        unsigned long long level = column % levels, member = column / levels;
        unsigned long long horizontal = (unsigned long long)width * nx;
        unsigned long long cell;
        if (ring < horizontal) cell = ring;
        else if (ring < 2 * horizontal) cell = (unsigned long long)(ny - width) * nx + ring - horizontal;
        else {
            unsigned long long sides = ring - 2 * horizontal;
            unsigned long long y = sides / (2 * width) + width;
            unsigned long long x = sides % (2 * width);
            if (x >= (unsigned long long)width) x += nx - 2 * width;
            cell = y * nx + x;
        }
        unsigned long long offset = (level * members + member) * plane + cell;
        if (restore) field[offset] = zero ? 0U : saved[point];
        else if (!zero) saved[point] = field[offset];
    }
}
'''


def ring_memory_plan(field_levels, *, ny, nx, width, zero_fields=(), reserved_bytes=0):
    """Compact ring save buffers and its pointer/extent table, all priced."""
    ny, nx, width = _positive(ny, "ny"), _positive(nx, "nx"), _positive(width, "width")
    if 2 * width >= min(ny, nx):
        raise ValueError("compact physics ring requires a nonempty member interior")
    zeros = set(zero_fields)
    levels = {name: _positive(n, f"{name} levels") for name, n in field_levels.items()}
    if not levels or zeros - levels.keys():
        raise ValueError("ring fields and zero-field declarations must agree")
    cells = 2 * width * nx + 2 * width * (ny - 2 * width)
    specs = [BatchArraySpec(f"physics:ring:{name}", (count, cells), "member", "uint32")
             for name, count in levels.items() if name not in zeros]
    specs.append(BatchArraySpec("physics:ring:table", (len(levels), 4), "shared", "uint64"))
    return BatchMemoryPlan(tuple(specs), reserved_bytes=reserved_bytes)


class PreparedPhysicsRing:
    """Save/restore all members' scalar-column outer rings in one launch."""

    def __init__(self, fields, *, members, ny, nx, width, available_bytes,
                 zero_fields=(), array_module=None):
        if array_module is None:
            import cupy as array_module
        self._xp, self.fields = array_module, dict(fields)
        self.members, self.ny, self.nx, self.width = [_positive(v, n) for v, n in ((members, "members"), (ny, "ny"), (nx, "nx"), (width, "width"))]
        self.device = int(array_module.cuda.runtime.getDevice())
        levels = {}
        for name, array in self.fields.items():
            if (not isinstance(array, array_module.ndarray) or array.dtype != np.dtype("float32")
                    or not array.flags.c_contiguous or int(array.device.id) != self.device
                    or array.ndim not in (2, 3) or array.shape[-2:] != (members * ny, nx)):
                raise ValueError(f"{name} ring needs packed complete-roster float32 mass/surface storage")
            levels[name] = array.shape[0] if array.ndim == 3 else 1
        plan = ring_memory_plan(levels, ny=ny, nx=nx, width=width, zero_fields=zero_fields)
        self.storage = BatchStorage(plan, self.members, array_module=array_module, available_bytes=available_bytes)
        zeros = set(zero_fields)
        table = []
        for name, count in levels.items():
            pointer = 0 if name in zeros else int(self.storage.arrays[f"physics:ring:{name}"].data.ptr)
            table.append((int(self.fields[name].data.ptr), pointer, count, int(name in zeros)))
        self.storage.arrays["physics:ring:table"].set(np.asarray(table, np.uint64))
        self.ring_cells = 2 * width * nx + 2 * width * (ny - 2 * width)
        max_points = self.members * max(levels.values()) * self.ring_cells
        self._grid = (min(64, (max_points + 255) // 256), len(levels))
        module = array_module.RawModule(code=_RING_SOURCE, options=("--std=c++17",))
        self._kernel = module.get_function("ensemble_physics_ring_words")
        record_module("gpuwm.ensemble.batch_physics:ring-words", source=_RING_SOURCE,
                      options=("--std=c++17",), module=module)

    def _launch(self, restore):
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("physics ring launch uses a different current device")
        self._kernel(self._grid, (256,),
                     (self.storage.arrays["physics:ring:table"], np.int32(len(self.fields)),
                      np.int32(self.members), np.int32(self.ny), np.int32(self.nx),
                      np.int32(self.width), np.uint64(self.ring_cells), np.int32(restore)))

    def capture(self):
        self._launch(False)

    def restore(self):
        self._launch(True)


class PackedThompsonState:
    """An admitted stock-adapter view with no implicit scratch allocations."""

    def __init__(self, arrays, scratch, *, physics=None):
        self.__dict__.update(arrays)
        self._scratch = dict(scratch)
        self.physics = physics

    def scratch(self, shape, slot, dtype=None):
        try:
            value = self._scratch[slot]
        except KeyError:
            raise ValueError(f"Thompson slot {slot} was not admitted before integration") from None
        if value.shape != tuple(shape) or value.dtype != np.dtype(dtype or "float32"):
            raise ValueError(f"Thompson slot {slot} shape/dtype differs from its admitted column layout")
        return value

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


THOMPSON_VOLUME_SLOTS = (
    "mp_th", "mp_pii", "mp_thompson_temperature", "mp_dz8w",
    "mp_thompson_frozen_reference_density", "mp_thompson_frozen_reference_temperature",
    "mp_thompson_rain_reference_density", "mp_thompson_snow_melt_marker",
    "mp_thompson_graupel_melt_marker", "mp_thompson_snow_velocity_boost",
    "mp_thompson_graupel_number_shadow",
)
THOMPSON_SURFACE_SLOTS = (
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_sr", "mp_thompson_micro_columns",
)


def thompson_column_scratch_fields(*, nz, ny, nx, reflectivity=False):
    """The stock classic adapter's named column scratch inventory.

    Its interface buffer mp_z8w remains present although the stock adapter
    no longer reads it. Reflectivity is reserved ahead of the output-due
    call so its zero-initialized specified ring exists before capture.
    """
    nz, ny, nx = _positive(nz, "nz"), _positive(ny, "ny"), _positive(nx, "nx")
    fields = [ColumnField(name, (nz, ny, nx)) for name in THOMPSON_VOLUME_SLOTS]
    fields += [ColumnField(name, (ny, nx)) for name in THOMPSON_SURFACE_SLOTS]
    fields.append(ColumnField("mp_z8w", (nz + 1, ny, nx)))
    if reflectivity:
        fields.append(ColumnField("refl_10cm", (nz, ny, nx)))
    return tuple(fields)


def _thompson_base_source(source, *, member_columns):
    replacements = {
        "thb[thb_full ? idx : k]": "thb[thb_full ? k * __ensemble_member_columns + idx % __ensemble_member_columns : k]",
        "phb[phb_full ? idx : k]": "phb[phb_full ? k * __ensemble_member_columns + idx % __ensemble_member_columns : k]",
        "phb[phb_full ? idx + columns : k + 1]": "phb[phb_full ? (k + 1) * __ensemble_member_columns + idx % __ensemble_member_columns : k + 1]",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError("Thompson base-load indexing changed; shared terrain adapter needs a new source audit")
        source = source.replace(before, after)
    return f"#define __ensemble_member_columns {int(member_columns)}\n" + source


def _clone_thompson_column_adapter(prepare):
    """Retain the stock process order; replace only the shared-base seam.

    The original CuPy divide and power remain separate operations. Supplying
    ``out`` avoids the otherwise hidden full-volume division temporary.
    This Python source transformation is restricted to two exact call sites.
    """
    from gpuwm.core import microphysics
    from gpuwm.core import constants as c
    import cupy as cp
    source = textwrap.dedent(inspect.getsource(microphysics._apply_thompson))
    tree = ast.parse(source)
    original_assignment = ast.dump(ast.parse("pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))").body[0], include_attributes=False)
    counts = {"prepare": 0, "pii": 0}

    class Adapter(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "launch_adapter_prepare":
                node.func.id = "__ensemble_prepare_thompson"
                counts["prepare"] += 1
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if ast.dump(node, include_attributes=False) == original_assignment:
                counts["pii"] += 1
                return ast.parse("__ensemble_pii_into(state.p, pii)").body[0]
            return self.generic_visit(node)

    tree = Adapter().visit(tree)
    if counts != {"prepare": 1, "pii": 1}:
        raise ValueError("Thompson orchestration changed; column adapter needs a new call-site audit")
    ast.fix_missing_locations(tree)

    def pii_into(pressure, target):
        cp.divide(pressure, np.float32(c.P0), out=target)
        cp.power(target, np.float32(c.RCP), out=target)

    namespace = dict(microphysics.__dict__)
    namespace.update(__ensemble_prepare_thompson=prepare, __ensemble_pii_into=pii_into)
    exec(compile(tree, "<admitted-thompson-column-adapter>", "exec"), namespace)
    return namespace["_apply_thompson"], sha256(source.encode()).hexdigest()


def prepare_thompson_column_batch(state, cfg, *, members, ny, nx, dt,
                                 available_bytes, array_module=None):
    """One all-member invocation of each stock classic Thompson family.

    Mutable fields and every stock scratch slot must already be allocated in
    column layout. THB/PHB may remain one shared real-terrain/base-state slab.
    Specified rings are preserved for every member and heating is pinned to
    zero there. The caller attaches the real physics diagnostic receiver for
    reflectivity; this primitive does not invent a missing output provider.
    """
    if array_module is None:
        import cupy as array_module
    from gpuwm.core import kernels, microphysics, constants as c
    from gpuwm.core.physics_inventory import ring_guard_row
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm import wrf_exact
    members, ny, nx = _positive(members, "members"), _positive(ny, "ny"), _positive(nx, "nx")
    if int(cfg.mp_physics) != 8 or not cfg.moist or not np.isfinite(dt) or dt <= 0:
        raise ValueError("this column adapter requires moist classic Thompson 8 and positive finite dt")
    if not isinstance(state, PackedThompsonState):
        raise TypeError("Thompson column binding needs an admitted PackedThompsonState")
    device = int(array_module.cuda.runtime.getDevice())
    nz = state.p.shape[0]
    for name in ("p", "thp", "qv", "qc", "qr", "qi", "ni", "nr", "qs", "qg", "h_diabatic", "effc", "effi", "effs"):
        array = getattr(state, name)
        if (array.shape != (nz, members * ny, nx) or array.dtype != np.dtype("float32")
                or not array.flags.c_contiguous or int(array.device.id) != device):
            raise ValueError(f"Thompson {name} differs from the admitted packed shape")
    for name in ("php", "w"):
        array = getattr(state, name)
        if array.shape != (nz + 1, members * ny, nx) or array.dtype != np.dtype("float32") or not array.flags.c_contiguous:
            raise ValueError(f"Thompson {name} differs from the admitted packed interface shape")
    for name in ("thb", "phb"):
        array = getattr(state, name)
        levels = nz + (name == "phb")
        if array.shape not in ((levels,), (levels, ny, nx)) or array.dtype != np.dtype("float32") or not array.flags.c_contiguous:
            raise ValueError(f"Thompson {name} must be one shared base-state slab or vertical profile")
    # Check the complete scratch contract before a prognostic word moves.
    for field in thompson_column_scratch_fields(nz=nz, ny=ny, nx=nx):
        packed_shape = ((field.shape[0], members * ny, nx) if len(field.shape) == 3 else (members * ny, nx))
        state.scratch(packed_shape, field.name)
    if members == 1:
        prepare = None
        adapter = microphysics._apply_thompson
        scalar_hash = sha256(inspect.getsource(adapter).encode()).hexdigest()
        source_receipt = {"source_policy": "unchanged native kernel and stock adapter"}
    else:
        original = kernels.module_source("thompson")
        source = _thompson_base_source(original, member_columns=ny * nx)
        options = wrf_exact.effective_options(("-std=c++17",)) if wrf_exact.ENABLED else ("-std=c++17",)
        module = array_module.RawModule(code=source, options=options)
        key = f"gpuwm.ensemble.batch_physics:thompson:shared-base-loads[columns={ny * nx}]"
        kernels._compile_observed(module, key)
        record_module(key, source=source, options=options, module=module)
        kernel = module.get_function("thompson_adapter_prepare")

        def prepare(*args):
            # Exact stock argument order; only full shared base-state indexing
            # differs. Field strides remain native (level,all columns).
            size = nz * members * ny * nx
            kernel(((size + 255) // 256,), (256,), args + (np.float32(c.G), np.int32(size),
                   np.int32(members * ny * nx), np.int32(state.thb.ndim == 3), np.int32(state.phb.ndim == 3)))

        adapter, scalar_hash = _clone_thompson_column_adapter(prepare)
        source_receipt = {"source_policy": "integer-only shared base-state indexing",
                          "scalar_source_sha256": sha256(original.encode()).hexdigest(),
                          "source_sha256": sha256(source.encode()).hexdigest()}
    ring = None
    if (cfg.specified or cfg.nested) and cfg.spec_zone:
        row = ring_guard_row(8)
        ring_fields = {name: getattr(state, name) for name in row["state_fields"] if getattr(state, name, None) is not None}
        ring_fields.update({slot: state.existing_scratch(slot) for slot in row["surface_slots"]
                            if state.existing_scratch(slot) is not None})
        if state.existing_scratch("refl_10cm") is not None:
            ring_fields["refl_10cm"] = state.existing_scratch("refl_10cm")
        ring_fields["h_diabatic"] = state.h_diabatic
        ring = PreparedPhysicsRing(ring_fields, members=members, ny=ny, nx=nx,
                                   width=cfg.spec_zone, available_bytes=available_bytes,
                                   zero_fields=("h_diabatic",), array_module=array_module)

    def launch(*, refl_10cm_due=False):
        if int(array_module.cuda.runtime.getDevice()) != device:
            raise RuntimeError("Thompson launch uses a different current device")
        if ring is not None:
            ring.capture()
        try:
            return adapter(state, cfg, dt, refl_10cm_due=refl_10cm_due)
        finally:
            if ring is not None:
                ring.restore()

    launch.state, launch.ring = state, ring
    launch.receipt = {"members": members, "layout": "level-member-y-x", "suite": "classic-thompson-8",
                      "adapter_source_sha256": scalar_hash,
                      "ring_required_bytes": 0 if ring is None else ring.storage.plan.required_bytes(members),
                      **source_receipt}
    return launch


def physics_coupling_source(source, *, members, ny, nx):
    """Curate native mass/face coupling for column-packed member stripes.

    A face's neighboring cells remain inside its own member. Mutable rates
    keep native level/column layout; immutable map factors index one grid.
    Every float expression stays unchanged. Exact source replacements fail
    if the native family changes, so a source audit cannot silently drift.
    """
    members, ny, nx = _positive(members, "members"), _positive(ny, "ny"), _positive(nx, "nx")
    replacements = {
        "const long long j = cell / nx;": "const long long j = (cell / nx) % __ensemble_member_y;",
        "const long long i = cell - j * nx;": "const long long i = cell % nx;",
        "j == ny - 1": "j == __ensemble_member_y - 1",
        "msft[cell]": "msft[cell % __ensemble_member_plane]",
        "const long long j = r / row;": "const long long packed_j = r / row;\n        const long long j = packed_j % __ensemble_member_y;",
        "const long long f = r - j * row;": "const long long f = r - packed_j * row;",
        "const long long base = (k * ny + j) * nx;": "const long long base = (k * ny + packed_j) * nx;",
        "const long long nv = nz * (ny + 1) * nx;": "const long long nv = nz * __ensemble_members * (__ensemble_member_y + 1) * nx;",
        "const long long k = t / ((ny + 1) * nx);": "const long long k = t / (__ensemble_members * (__ensemble_member_y + 1) * nx);",
        "const long long r = t - k * (ny + 1) * nx;": "const long long r = t - k * __ensemble_members * (__ensemble_member_y + 1) * nx;",
        "const long long g = r / nx;": "const long long packed_g = r / nx;\n    const long long member = packed_g / (__ensemble_member_y + 1);\n    const long long g = packed_g % (__ensemble_member_y + 1);",
        "const long long i = r - g * nx;": "const long long i = r - packed_g * nx;",
        "const long long e = (g == ny) ? 0 : g;": "const long long e = (g == __ensemble_member_y) ? 0 : g;",
        "const long long s = (e == 0) ? ny - 1 : e - 1;": "const long long s = (e == 0) ? __ensemble_member_y - 1 : e - 1;",
        "const long long base = k * ny * nx;": "const long long base = (k * ny + member * __ensemble_member_y) * nx;",
        "g == ny": "g == __ensemble_member_y",
        "msfu[j * row + (closed ? 0 : f)]": "msfu[(j % __ensemble_member_y) * row + (closed ? 0 : f)]",
    }
    for before, after in replacements.items():
        count = source.count(before)
        expected = 2 if before == "j == ny - 1" else 3 if before == "g == ny" else 1
        if count != expected:
            raise ValueError(f"physics coupling indexing changed at {before!r}; member-stripe adapter needs a new audit")
        source = source.replace(before, after)
    header = (f"#define __ensemble_members {members}\n"
              f"#define __ensemble_member_y {ny}\n"
              f"#define __ensemble_member_plane {ny * nx}\n")
    return header + source


def physics_coupling_memory_plan(*, nz, ny, nx):
    specs = [BatchArraySpec(f"physics:coupling:{name}", (nz, ny, nx), "member")
             for name in ("rtheta", "rqv", "rqc", "rqr", "rqi", "rqs", "mass_u", "mass_v")]
    specs += [BatchArraySpec("physics:coupling:ru", (nz, ny, nx + 1), "member"),
              BatchArraySpec("physics:coupling:rv", (nz, ny + 1, nx), "member")]
    return BatchMemoryPlan(tuple(specs), reserved_bytes=0)


class PreparedPhysicsCoupling:
    """Planned member-aware scalar and C-grid momentum coupling.

    Make separate instances for held PBL and radiation components: their
    output buffers are persistent physics state and must never alias one
    another. The original source is used at one member. Larger batches
    change integer indexing only and require independent GPU word gates.
    """

    _slots = ("rtheta", "rqv", "rqc", "rqr", "rqi", "rqs")

    def __init__(self, *, members, nz, ny, nx, c1h, c2h, mut,
                 msft, msfu, msfv, has_msf, available_bytes, array_module=None):
        if array_module is None:
            import cupy as array_module
        self._xp = array_module
        self.members, self.nz, self.ny, self.nx = [_positive(v, n) for v, n in ((members, "members"), (nz, "nz"), (ny, "ny"), (nx, "nx"))]
        self.device = int(array_module.cuda.runtime.getDevice())
        self.inputs = dict(c1h=c1h, c2h=c2h, mut=mut, msft=msft, msfu=msfu, msfv=msfv)
        shapes = {"c1h": (nz,), "c2h": (nz,), "mut": (members * ny, nx),
                  "msft": (ny, nx), "msfu": (ny, nx + 1), "msfv": (ny + 1, nx)}
        for name, array in self.inputs.items():
            if (not isinstance(array, array_module.ndarray) or array.shape != shapes[name]
                    or array.dtype != np.dtype("float32") or not array.flags.c_contiguous
                    or int(array.device.id) != self.device):
                raise ValueError(f"physics coupling {name} needs resident float32 shape {shapes[name]}")
        self.plan = physics_coupling_memory_plan(nz=nz, ny=ny, nx=nx)
        self.storage = BatchStorage(self.plan, members, array_module=array_module, available_bytes=available_bytes)
        from gpuwm.ensemble.batch_products import _refuse_overlap
        _refuse_overlap(self.inputs, self.storage.arrays)
        self.outputs = {}
        for name in self._slots + ("mass_u", "mass_v", "ru", "rv"):
            shape = ((nz, members * ny, nx + 1) if name == "ru" else
                     (nz, members * (ny + 1), nx) if name == "rv" else (nz, members * ny, nx))
            self.outputs[name] = self.storage.arrays[f"physics:coupling:{name}"].reshape(shape)
        from gpuwm.core import tendency_coupling
        if members == 1:
            self._mass, self._faces = tendency_coupling._kernel("couple_mass_rates"), tendency_coupling._kernel("couple_faces")
            source = tendency_coupling._SOURCE
        else:
            from gpuwm.core.kernels import _compile_observed
            from gpuwm.certify.kernel_manifest import record_module
            source = physics_coupling_source(tendency_coupling._SOURCE, members=members, ny=ny, nx=nx)
            module = array_module.RawModule(code=source, options=())
            key = f"gpuwm.ensemble.batch_physics:member-column-tendency-coupling[members={members},ny={ny},nx={nx}]"
            _compile_observed(module, key)
            record_module(key, source=source, options=(), module=module)
            self._mass, self._faces = module.get_function("couple_mass_rates"), module.get_function("couple_faces")
        self.has_msf = bool(has_msf)
        self.receipt = {"members": members, "source_sha256": sha256(source.encode()).hexdigest(),
                        "layout": "level-member-y-x with separate member C-grid faces",
                        "allocations": list(self.plan.inventory(members))}

    def couple(self, *, cfg, rates, du=None, dv=None):
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("physics coupling launch uses a different current device")
        if set(rates) - set(self._slots):
            raise ValueError("unclassified physics scalar rate would bypass its coupled carrier")
        if (du is None) != (dv is None):
            raise ValueError("physics momentum coupling needs both du and dv")
        volume = (self.nz, self.members * self.ny, self.nx)
        for name, array in dict(rates, du=du, dv=dv).items():
            if array is not None and (not isinstance(array, self._xp.ndarray) or array.shape != volume
                or array.dtype != np.dtype("float32") or not array.flags.c_contiguous
                or int(array.device.id) != self.device):
                raise ValueError(f"physics rate {name} differs from its admitted packed volume")
        from gpuwm.ensemble.batch_products import _refuse_overlap
        _refuse_overlap({name: array for name, array in dict(rates, du=du, dv=dv).items()
                         if array is not None}, self.storage.arrays)
        dummy = self.outputs["mass_u"]
        present = sum(1 << i for i, name in enumerate(self._slots) if rates.get(name) is not None)
        # The common driver always retains theta/qv/qc, using exact zeros for
        # absent providers. Optional precipitation categories remain None.
        # Zero every absent optional backing in the same launch. The public
        # tendency still reports None until the configured driver asks to
        # materialize a category, at which point it borrows this buffer.
        wanted = 63
        pointers = tuple(rates.get(name) if rates.get(name) is not None else dummy for name in self._slots)
        out = tuple(self.outputs[name] for name in self._slots)
        inp = self.inputs
        mask = bool(cfg.specified or cfg.nested)
        args = (inp["c1h"], inp["c2h"], inp["mut"], inp["msft"], *pointers, *out,
                np.uint32(present), np.uint32(wanted), np.int32(6), np.int32(0),
                du if du is not None else dummy, dv if dv is not None else dummy,
                self.outputs["mass_u"], self.outputs["mass_v"], np.int32(du is not None),
                np.int32(mask), np.int32(self.has_msf), np.int64(self.nz),
                np.int64(self.members * self.ny), np.int64(self.nx))
        points = prod(volume)
        self._mass(((points + 255) // 256,), (256,), args)
        if du is not None:
            points = self.nz * self.members * (self.ny * (self.nx + 1) + (self.ny + 1) * self.nx)
            args = (self.outputs["mass_u"], self.outputs["mass_v"], inp["msfu"], inp["msfv"],
                    self.outputs["ru"], self.outputs["rv"], np.int32(mask),
                    np.int32(cfg.open_x), np.int32(cfg.open_y), np.int32(self.has_msf),
                    np.int64(self.nz), np.int64(self.members * self.ny), np.int64(self.nx))
            self._faces(((points + 255) // 256,), (256,), args)
        else:
            self.outputs["ru"].fill(0)
            self.outputs["rv"].fill(0)
        from gpuwm.core.physics import PhysicsTendencies
        result = PhysicsTendencies(self.outputs["ru"], self.outputs["rv"],
            self.outputs["rtheta"], self.outputs["rqv"], self.outputs["rqc"],
            self.outputs["rqr"] if rates.get("rqr") is not None else None,
            self.outputs["rqi"] if rates.get("rqi") is not None else None,
            self.outputs["rqs"] if rates.get("rqs") is not None else None)
        from types import MethodType

        def materialize(target, components):
            for name in components:
                if name not in ("rqr", "rqi", "rqs"):
                    raise ValueError(f"physics component {name} has no admitted default-suite coupling buffer")
                if getattr(target, name) is None:
                    setattr(target, name, self.outputs[name])

        result.materialize = MethodType(materialize, result)
        return result

    def couple_ysu_tendencies(self, state, cfg, rates):
        return self.couple(cfg=cfg, rates={"rtheta": rates["dtheta"], "rqv": rates["dqv"],
                                         "rqc": rates["dqc"], "rqi": rates.get("dqi")},
                           du=rates["du"], dv=rates["dv"])

    def couple_column_tendencies(self, state, cfg, **rates):
        du, dv = rates.pop("ru", None), rates.pop("rv", None)
        return self.couple(cfg=cfg, rates=rates, du=du, dv=dv)


def bind_column_driver_coupling(driver, *, pbl_coupling, radiation_coupling):
    """Bind an owned PhysicsDriver's two coupling seams privately.

    The stock method bytecode, due-call order and carrier bookkeeping remain
    intact. Each method gets its own global lookup dictionary pointing at
    the admitted member-aware coupler. No module global or other driver's
    method changes. Preparation/atmosphere and native scheme launch bindings
    remain the caller's responsibilities; this is not a complete forecast
    driver constructor.
    """
    from types import FunctionType, MethodType
    from gpuwm.core.physics import PhysicsDriver
    if not isinstance(driver, PhysicsDriver):
        raise TypeError("column coupling requires an owned PhysicsDriver")
    if not isinstance(pbl_coupling, PreparedPhysicsCoupling) or not isinstance(radiation_coupling, PreparedPhysicsCoupling):
        raise TypeError("both persistent coupling providers must be admitted")
    if pbl_coupling is radiation_coupling:
        raise ValueError("PBL and radiation coupling backings must be separate persistent components")
    expected = (pbl_coupling.nz, pbl_coupling.members * pbl_coupling.ny, pbl_coupling.nx)
    if driver.state.p.shape != expected:
        raise ValueError("PhysicsDriver atmosphere shape differs from its member-column coupling plan")
    if (pbl_coupling.members, pbl_coupling.nz, pbl_coupling.ny, pbl_coupling.nx) != (
            radiation_coupling.members, radiation_coupling.nz, radiation_coupling.ny, radiation_coupling.nx):
        raise ValueError("persistent physics coupling providers have different rosters/grids")
    for name, replacement, provider in (
        ("_couple_pbl_slot", "couple_ysu_tendencies", pbl_coupling.couple_ysu_tendencies),
        ("_run_radiation", "couple_column_tendencies", radiation_coupling.couple_column_tendencies),
    ):
        original = getattr(PhysicsDriver, name)
        if replacement not in original.__code__.co_names:
            raise ValueError(f"stock physics method {name} changed its coupling seam; a new binding audit is required")
        namespace = dict(original.__globals__)
        namespace[replacement] = provider
        function = FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        function.__annotations__ = original.__annotations__
        setattr(driver, name, MethodType(function, driver))
    driver._ensemble_persistent_couplers = (pbl_coupling, radiation_coupling)
    return driver


ATMOSPHERE_VOLUMES = ("theta", "temperature", "pressure", "exner", "u", "v", "dz", "qv", "qc", "qi", "qs", "rho", "eos_pressure")
ATMOSPHERE_INTERFACES = ("p_interface", "z_interface")


def atmosphere_memory_plan(*, nz, ny, nx, reserved_bytes=0):
    """Outer-member arithmetic carriers for the exact stock phy_prep DAG."""
    nz, ny, nx = _positive(nz, "nz"), _positive(ny, "ny"), _positive(nx, "nx")
    specs = [BatchArraySpec(f"atmosphere:{name}", (nz, ny, nx), "member")
             for name in ATMOSPHERE_VOLUMES + ("qtot", "mass_coefficient", "layer")]
    specs += [BatchArraySpec(f"atmosphere:{name}", (nz + 1, ny, nx), "member") for name in ATMOSPHERE_INTERFACES]
    specs.append(BatchArraySpec("atmosphere:mut", (ny, nx), "member"))
    return BatchMemoryPlan(tuple(specs), reserved_bytes=reserved_bytes)


class PreparedPhysicsAtmosphere:
    """Member-aware C-grid prep followed by explicit column packing.

    The elementwise operations and downward hydrostatic fold follow
    core.physics._prepare_atmosphere, in the same order and precision.
    Every intermediate has a named allocation. The per-level loop remains
    sequential across height, while each launch contains every member.
    Actual GPU word qualification against the stock helper is required.
    """

    def __init__(self, batch, *, available_bytes, array_module=None):
        if array_module is None:
            import cupy as array_module
        from gpuwm.ensemble.batch_state import BatchedDomainState
        if not isinstance(batch, BatchedDomainState):
            raise TypeError("physics atmosphere needs the admitted member dycore state")
        self.batch, self._xp = batch, array_module
        self.members = batch.members
        cfg = batch.cfg
        self.nz, self.ny, self.nx = cfg.nz, cfg.ny, cfg.nx
        outer_plan = atmosphere_memory_plan(nz=cfg.nz, ny=cfg.ny, nx=cfg.nx)
        packed_fields = tuple(ColumnField(name, (cfg.nz, cfg.ny, cfg.nx)) for name in ATMOSPHERE_VOLUMES)
        packed_fields += tuple(ColumnField(name, (cfg.nz + 1, cfg.ny, cfg.nx)) for name in ATMOSPHERE_INTERFACES)
        packed_fields += (ColumnField("mut", (cfg.ny, cfg.nx)),)
        packed_plan = column_memory_plan(packed_fields)
        combined = outer_plan.required_bytes(self.members) + packed_plan.required_bytes(self.members)
        if combined > available_bytes:
            raise MemoryError(f"member-aware physics atmosphere needs {combined} bytes before native scheme workspaces")
        self.storage = BatchStorage(outer_plan, self.members, array_module=array_module, available_bytes=available_bytes)
        self.columns = ColumnBuffers(packed_fields, members=self.members,
                                     available_bytes=available_bytes - outer_plan.required_bytes(self.members),
                                     array_module=array_module)
        self.device = int(array_module.cuda.runtime.getDevice())
        self.required_bytes = combined

    def _array(self, name):
        return self.storage.arrays[f"atmosphere:{name}"]

    def _volume_input(self, name):
        array = self.batch.storage.arrays[name]
        spec = self.batch.storage.specs[name]
        if spec.ownership == "member":
            return array.reshape((self.members,) + ((spec.shape[0], 1, 1) if len(spec.shape) == 1 else spec.shape))
        if array.ndim == 1:
            return array[None, :, None, None]
        return array[None]

    def __call__(self):
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("physics atmosphere uses a different current device")
        from gpuwm.core import constants as c
        xp, state = self._xp, self.batch
        get = self._array
        xp.add(self._volume_input("thb"), state.thp, out=get("theta"))
        xp.divide(state.p, np.float32(c.P0), out=get("exner"))
        xp.power(get("exner"), np.float32(c.RCP), out=get("exner"))
        xp.multiply(get("theta"), get("exner"), out=get("temperature"))
        # Staggered neighbors are selected inside each explicit member.
        xp.add(state.u[:, :, :, :-1], state.u[:, :, :, 1:], out=get("u"))
        xp.multiply(get("u"), np.float32(0.5), out=get("u"))
        xp.add(state.v[:, :, :-1, :], state.v[:, :, 1:, :], out=get("v"))
        xp.multiply(get("v"), np.float32(0.5), out=get("v"))
        xp.add(self._volume_input("phb"), state.php, out=get("z_interface"))
        xp.divide(get("z_interface"), np.float32(c.G), out=get("z_interface"))
        xp.subtract(get("z_interface")[:, 1:], get("z_interface")[:, :-1], out=get("dz"))
        get("qtot").fill(0)
        for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
            array = state.storage.arrays.get(name)
            if array is not None:
                xp.add(get("qtot"), array, out=get("qtot"))
        mub = state.storage.arrays["mub2d"]
        xp.add(state.mup, mub if state.storage.specs["mub2d"].ownership == "member" else mub[None], out=get("mut"))
        xp.multiply(self._volume_input("c1h"), get("mut")[:, None], out=get("mass_coefficient"))
        xp.add(get("mass_coefficient"), self._volume_input("c2h"), out=get("mass_coefficient"))
        xp.add(np.float32(1), get("qtot"), out=get("layer"))
        xp.multiply(get("layer"), get("mass_coefficient"), out=get("layer"))
        xp.multiply(get("layer"), self._volume_input("dnw"), out=get("layer"))
        get("p_interface")[:, self.nz].fill(np.float32(state.p_top))
        for level in range(self.nz - 1, -1, -1):
            xp.subtract(get("p_interface")[:, level + 1], get("layer")[:, level], out=get("p_interface")[:, level])
        xp.add(get("p_interface")[:, :-1], get("p_interface")[:, 1:], out=get("pressure"))
        xp.multiply(get("pressure"), np.float32(0.5), out=get("pressure"))
        for name in ("qv", "qc", "qi", "qs"):
            array = state.storage.arrays.get(name)
            if array is None:
                get(name).fill(0)
            else:
                xp.copyto(get(name), array)
        xp.add(np.float32(1), get("qv"), out=get("rho"))
        xp.divide(get("rho"), state.alt, out=get("rho"))
        xp.copyto(get("eos_pressure"), state.p)
        result = {name: self.columns.pack(name, get(name)) for name in ATMOSPHERE_VOLUMES + ATMOSPHERE_INTERFACES}
        self.columns.pack("mut", get("mut"))
        return {name: array for name, array in result.items() if name != "eos_pressure"}

    @property
    def packed_eos_pressure(self):
        return self.columns.packed("eos_pressure")

    @property
    def packed_mut(self):
        return self.columns.packed("mut")


def packed_physics_atmosphere_fields(*, nz, ny, nx):
    fields = [ColumnField(name, (nz, ny, nx)) for name in (
        "theta", "temperature", "pressure", "exner", "u", "v", "dz", "rho", "qtot", "mass_coefficient", "layer", "dry_qv", "dry_qc", "dry_qi", "dry_qs")]
    fields += [ColumnField(name, (nz + 1, ny, nx)) for name in ("p_interface", "z_interface")]
    fields.append(ColumnField("mut", (ny, nx)))
    return tuple(fields)


def default_column_extra_plan(*, nz, ny, nx):
    extra_specs = (
        BatchArraySpec("physics:driver:ysu_radiation_sum", (nz, ny, nx), "member"),
        BatchArraySpec("physics:driver:zero_ru", (nz, ny, nx + 1), "member"),
        BatchArraySpec("physics:driver:zero_rv", (nz, ny + 1, nx), "member"),
        *(BatchArraySpec(f"physics:driver:zero_{name}", (nz, ny, nx), "member")
          for name in ("rtheta", "rqv", "rqc")),
    )
    return BatchMemoryPlan(tuple(extra_specs), reserved_bytes=0)


class PreparedPackedPhysicsAtmosphere:
    """Stock phy_prep DAG over resident (level,member,y,x) views.

    The caller packs its prognostic bank once before the physics call and
    unpacks changed fields afterwards. Shared THB/PHB and coefficients use
    broadcast views, without N static copies. Each member's staggered wind
    neighbors are selected before flattening its columns. No field-sized
    device allocation or layout copy occurs inside this preparation call.
    """

    def __init__(self, state, *, members, ny, nx, c1h, c2h, dnw, p_top,
                 mup, mub2d, available_bytes, array_module=None):
        if array_module is None:
            import cupy as array_module
        self.state, self._xp = state, array_module
        self.members, self.ny, self.nx = _positive(members, "members"), _positive(ny, "ny"), _positive(nx, "nx")
        self.nz = state.p.shape[0]
        self.device = int(array_module.cuda.runtime.getDevice())
        shape = (self.nz, members * ny, nx)
        required = {"p": shape, "thp": shape, "alt": shape,
                    "u": (self.nz, members * ny, nx + 1),
                    "v": (self.nz, members * (ny + 1), nx),
                    "php": (self.nz + 1, members * ny, nx)}
        for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
            if getattr(state, name, None) is not None:
                required[name] = shape
        for name, wanted in required.items():
            array = getattr(state, name)
            if (not isinstance(array, array_module.ndarray) or array.shape != wanted
                    or array.dtype != np.dtype("float32") or not array.flags.c_contiguous
                    or int(array.device.id) != self.device):
                raise ValueError(f"packed physics {name} differs from its declared resident member layout")
        for name, array, wanted in (("c1h", c1h, (self.nz,)), ("c2h", c2h, (self.nz,)), ("dnw", dnw, (self.nz,)),
                                   ("mup", mup, (members * ny, nx)), ("mub2d", mub2d, (ny, nx))):
            if (not isinstance(array, array_module.ndarray) or array.shape != wanted
                    or array.dtype != np.dtype("float32") or not array.flags.c_contiguous
                    or int(array.device.id) != self.device):
                raise ValueError(f"packed physics {name} differs from its admitted coefficient/mass layout")
        self.c1h, self.c2h, self.dnw, self.p_top = c1h, c2h, dnw, np.float32(p_top)
        self.mup, self.mub2d = mup, mub2d
        fields = packed_physics_atmosphere_fields(nz=self.nz, ny=ny, nx=nx)
        self.columns = ColumnBuffers(fields, members=members, available_bytes=available_bytes, array_module=array_module)
        for name in ("thb", "phb"):
            array = getattr(state, name)
            levels = self.nz + (name == "phb")
            if array.shape == (levels,):
                view = array[:, None, None, None]
            elif array.shape == (levels, ny, nx):
                view = array[:, None, :, :]
            else:
                raise ValueError(f"packed physics {name} needs one shared vertical or real-terrain base slab")
            setattr(self, f"_{name}", view)

    def _out(self, name):
        array = self.columns.packed(name)
        if array.ndim == 2:
            return array.reshape(self.members, self.ny, self.nx)
        return array.reshape(array.shape[0], self.members, self.ny, self.nx)

    def _volume(self, array):
        return array.reshape(array.shape[0], self.members, self.ny, self.nx)

    @property
    def packed_mut(self):
        return self.columns.packed("mut")

    def __call__(self):
        if int(self._xp.cuda.runtime.getDevice()) != self.device:
            raise RuntimeError("packed physics atmosphere uses a different current device")
        from gpuwm.core import constants as c
        xp, state, out = self._xp, self.state, self._out
        xp.add(self._thb, self._volume(state.thp), out=out("theta"))
        xp.divide(self._volume(state.p), np.float32(c.P0), out=out("exner"))
        xp.power(out("exner"), np.float32(c.RCP), out=out("exner"))
        xp.multiply(out("theta"), out("exner"), out=out("temperature"))
        u = state.u.reshape(self.nz, self.members, self.ny, self.nx + 1)
        v = state.v.reshape(self.nz, self.members, self.ny + 1, self.nx)
        xp.add(u[..., :-1], u[..., 1:], out=out("u"))
        xp.multiply(out("u"), np.float32(0.5), out=out("u"))
        xp.add(v[:, :, :-1], v[:, :, 1:], out=out("v"))
        xp.multiply(out("v"), np.float32(0.5), out=out("v"))
        xp.add(self._phb, self._volume(state.php), out=out("z_interface"))
        xp.divide(out("z_interface"), np.float32(c.G), out=out("z_interface"))
        xp.subtract(out("z_interface")[1:], out("z_interface")[:-1], out=out("dz"))
        out("qtot").fill(0)
        for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
            value = getattr(state, name, None)
            if value is not None:
                xp.add(out("qtot"), self._volume(value), out=out("qtot"))
        xp.add(self.mup.reshape(self.members, self.ny, self.nx), self.mub2d[None], out=out("mut"))
        xp.multiply(self.c1h[:, None, None, None], out("mut")[None], out=out("mass_coefficient"))
        xp.add(out("mass_coefficient"), self.c2h[:, None, None, None], out=out("mass_coefficient"))
        xp.add(np.float32(1), out("qtot"), out=out("layer"))
        xp.multiply(out("layer"), out("mass_coefficient"), out=out("layer"))
        xp.multiply(out("layer"), self.dnw[:, None, None, None], out=out("layer"))
        out("p_interface")[self.nz].fill(self.p_top)
        for level in range(self.nz - 1, -1, -1):
            xp.subtract(out("p_interface")[level + 1], out("layer")[level], out=out("p_interface")[level])
        xp.add(out("p_interface")[:-1], out("p_interface")[1:], out=out("pressure"))
        xp.multiply(out("pressure"), np.float32(0.5), out=out("pressure"))
        moist = {}
        for name in ("qv", "qc", "qi", "qs"):
            value = getattr(state, name, None)
            moist[name] = value if value is not None else self.columns.packed(f"dry_{name}")
        xp.add(np.float32(1), self._volume(moist["qv"]), out=out("rho"))
        xp.divide(out("rho"), self._volume(state.alt), out=out("rho"))
        result = {name: self.columns.packed(name) for name in ("theta", "temperature", "pressure", "p_interface",
            "exner", "u", "v", "z_interface", "dz", "rho")}
        result.update(moist)
        return result


def bind_owned_default_column_driver(driver, cfg, atmosphere, *, members, ny, nx,
                                     pbl_coupling, radiation_coupling,
                                     available_bytes, array_module=None):
    """Route an initialized default-suite driver through all-member leaves.

    Initialization remains the prepared-input factory's responsibility. Its
    mutable fields, scratch and radiation callable must already describe the
    complete member-column grid. This function binds the stock scheduler,
    carrier bookkeeping and Noah forcing order to admitted native launchers.
    It does not construct a larger staggered grid or run member physics in a
    loop. One member retains the ordinary driver and its native factories.
    """
    members = _positive(members, "members")
    if members == 1:
        return driver
    if array_module is None:
        import cupy as array_module
    from types import FunctionType, MethodType
    from gpuwm.core.physics import PhysicsDriver, PhysicsTendencies
    from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
    from gpuwm.core.noah import _F2D, _F3D
    from gpuwm.ingest.soil import NOAH_LAYER_THICKNESS_M
    if not isinstance(driver, PhysicsDriver) or not isinstance(atmosphere, PreparedPackedPhysicsAtmosphere):
        raise TypeError("default column binding requires an owned PhysicsDriver and admitted packed atmosphere")
    if (cfg.mp_physics, cfg.sf_surface_physics, cfg.bl_pbl_physics) != (8, 2, 1) or cfg.sf_sfclay_physics not in (1, 91):
        raise ValueError("this native binding implements Thompson/Noah/YSU/MM5; the configured suite needs its own qualified binding")
    if cfg.cu_physics or cfg.sf_urban_physics or getattr(cfg, "sf_surface_mosaic", 0) or driver.terrain_drag is not None:
        raise ValueError("this default column binding has no cumulus, urban, mosaic or terrain-drag producer binding")
    if driver.state is not atmosphere.state:
        raise ValueError("the initialized physics driver and atmosphere preparer own different states")
    if driver.tendencies is not driver.pbl_tendencies:
        raise ValueError("positive PBL cadence needs a separate admitted composition target before native column binding")
    if getattr(cfg, "use_adaptive_time_step", False):
        raise ValueError("member adaptive clocks can diverge; this identity-qualified column binding requires the same fixed step")
    members, ny, nx = _positive(members, "members"), _positive(ny, "ny"), _positive(nx, "nx")
    if (members, ny, nx) != (atmosphere.members, atmosphere.ny, atmosphere.nx):
        raise ValueError("default physics binding and atmosphere have different rosters/grids")
    nz = atmosphere.nz
    # Fields below are the stock driver's mutable carriers. Static input banks
    # remain the preparation factory's source authority, not inferred aliases.
    f = driver.fields
    shared_surfaces = frozenset(getattr(driver, "_ensemble_shared_surface_fields", ()))
    if shared_surfaces - (NOAH_SHARED_FIELDS | {"lakemask", "landmask"}):
        raise ValueError("default shared surfaces must be the audited immutable Noah/MM5 inputs")
    active = atmosphere()
    extras_plan = default_column_extra_plan(nz=nz, ny=ny, nx=nx)
    ysu_plan = ysu_output_plan(nz=nz, ny=ny, nx=nx, members=members)
    required = extras_plan.required_bytes(members) + ysu_plan.required_bytes(members)
    if required > available_bytes:
        raise MemoryError(f"default native physics binding needs {required} bytes for YSU and held carriers")
    extras = BatchStorage(extras_plan, members, array_module=array_module, available_bytes=available_bytes)
    heating = extras.arrays["physics:driver:ysu_radiation_sum"].reshape(nz, members * ny, nx)
    inputs = {"u": active["u"], "v": active["v"], "theta": active["theta"], "qv": active["qv"],
              "qc": active["qc"], "qi": active["qi"], "p": active["pressure"], "p_interface": active["p_interface"],
              "exner": active["exner"], "dz": active["dz"], "rthraten": heating,
              "psfc": active["p_interface"][0], "znt": f["znt"], "ust": f["ust"], "hfx": f["hfx"],
              "qfx": f["qfx"], "wspd": f["wspd"], "br": f["br"], "psim": f["fm"], "psih": f["fh"],
              "xland": f["xland"], "u10": f["u10"], "v10": f["v10"]}
    ysu = prepare_ysu_column_batch(inputs, members=members, ny=ny, nx=nx, dt=driver.bldt_seconds,
        topdown=cfg.ysu_topdown_pblmix, available_bytes=available_bytes - extras_plan.required_bytes(members),
        array_module=array_module, shared_fields=tuple(sorted(shared_surfaces & {"xland"})))
    sf_inputs = {"u": active["u"][0], "v": active["v"][0], "t": active["temperature"][0], "qv": active["qv"][0],
                 "p": active["pressure"][0], "dz8w": active["dz"][0], "psfc": active["p_interface"][0],
                 "tsk": f["tsk"], "pblh": f["pblh"], "mavail": f["mavail"], "xland": f["xland"], "lakemask": f["lakemask"]}
    sf_outputs = {name: getattr(driver.sfclay_result, name) for name in SFCLAY_OUTPUTS}
    sfclay = prepare_sfclay_column_batch(sf_inputs, sf_outputs, members=members, ny=ny, nx=nx,
        option=cfg.sf_sfclay_physics, dx=cfg.dx, isfflx=bool(cfg.isfflx), isftcflx=cfg.isftcflx,
        iz0tlnd=cfg.iz0tlnd, array_module=array_module,
        shared_fields=tuple(sorted(shared_surfaces & {"xland", "lakemask"})))
    noah_fields = {name: f[name] for name in _F2D + _F3D + ("ivgtyp", "isltyp", "ebal")}
    noah = prepare_noah_column_batch(noah_fields, driver.noah_params, members=members, ny=ny, nx=nx,
        dt=driver.bldt_seconds, dzs=NOAH_LAYER_THICKNESS_M, frpcpn=True, usemonalb=driver.noah_usemonalb,
        rdlai2d=driver.noah_rdlai2d, opt_thcnd=driver.noah_opt_thcnd, array_module=array_module,
        shared_fields=tuple(sorted(shared_surfaces & NOAH_SHARED_FIELDS)))

    if "xland" in shared_surfaces:
        # A single 2-D immutable mask cannot be flattened into member stripes
        # without copying it. Apply the unchanged final Q2 operations on
        # resident (member,y,x) views, where ordinary broadcasting shares it.
        from types import SimpleNamespace
        from gpuwm.core import surface_humidity, radiation_carriers

        def cap_land_q2(q2, qv1, xland, *, xp):
            return surface_humidity.cap_land_q2(
                q2.reshape(members, ny, nx), qv1.reshape(members, ny, nx), xland[None], xp=xp)

        original = PhysicsDriver.compute
        namespace = dict(original.__globals__)
        namespace["surface_humidity"] = SimpleNamespace(cap_land_q2=cap_land_q2)
        function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        driver.compute = MethodType(function, driver)

        def consumer_cells_exist(fields):
            if fields is None or fields.get("xland") is None:
                return radiation_carriers._consumer_cells_exist(fields)
            views = {"xland": fields["xland"][None]}
            if fields.get("xice") is not None:
                views["xice"] = fields["xice"].reshape(members, ny, nx)
            return radiation_carriers._consumer_cells_exist(views)

        original = type(driver.carriers).check_before_consumption
        namespace = dict(original.__globals__)
        namespace["_consumer_cells_exist"] = consumer_cells_exist
        function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        driver.carriers.check_before_consumption = MethodType(function, driver.carriers)

    def verify_pointers(actual, expected):
        if len(actual) != len(expected) or any(int(a.data.ptr) != int(b.data.ptr) for a, b in zip(actual, expected)):
            raise ValueError("physics producer rebound an admitted native input; rebuilding its launch is required before stepping")

    def native_ysu(*args, **kwargs):
        actual = list(args[:10]) + [kwargs["rthraten"]] + [kwargs[name] for name in YSU_SURFACES]
        verify_pointers(actual, [inputs[name] for name in YSU_COLUMNS + YSU_SURFACES])
        if kwargs.get("bep") is not None or kwargs.get("topo") is not None or np.float32(kwargs["dt"]) != np.float32(driver.bldt_seconds):
            raise ValueError("YSU options changed after the native column launch was admitted")
        return ysu()

    def native_sfclay(*args, **kwargs):
        verify_pointers(args[:12], [sf_inputs[name] for name in SFCLAY_INPUTS])
        if args[12] is not driver.sfclay_result:
            raise ValueError("MM5's persistent result was replaced after admission")
        sfclay()

    def native_noah(fields, params, dt, dzs, **kwargs):
        verify_pointers([fields[name] for name in noah_fields], list(noah_fields.values()))
        if params is not driver.noah_params or kwargs.get("urban") is not None or np.float32(dt) != np.float32(driver.bldt_seconds):
            raise ValueError("Noah's table owner, timestep or urban producer changed after admission")
        noah(kwargs["itimestep"])

    def sum_radiation(owned):
        array_module.add(owned.rthratenlw, owned.rthratensw, out=heating)
        return heating

    bind_column_driver_coupling(driver, pbl_coupling=pbl_coupling, radiation_coupling=radiation_coupling)
    for method, global_name, entry in (("_run_sfclay", "launch_sfclay", native_sfclay),
                                      ("_run_noah", "launch_noah", native_noah)):
        original = getattr(PhysicsDriver, method)
        namespace = dict(original.__globals__)
        namespace[global_name] = entry
        function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        setattr(driver, method, MethodType(function, driver))
    original = PhysicsDriver._run_ysu
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    target = ast.dump(ast.parse("cp.ascontiguousarray(self.rthratenlw + self.rthratensw)", mode="eval").body, include_attributes=False)
    replacements = [0]

    class HeatingSum(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.dump(node, include_attributes=False) == target:
                replacements[0] += 1
                return ast.parse("__ensemble_sum_radiation(self)", mode="eval").body
            return self.generic_visit(node)

    tree = HeatingSum().visit(tree)
    if replacements != [1]:
        raise ValueError("YSU's held radiation sum changed; its preallocated producer needs a new source audit")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace.update(launch_ysu=native_ysu, __ensemble_sum_radiation=sum_radiation)
    exec(compile(tree, "<admitted-default-ysu-column-driver>", "exec"), namespace)
    setattr(driver, "_run_ysu", MethodType(namespace["_run_ysu"], driver))

    def held_zero(original, coupling):
        return PhysicsTendencies(coupling.outputs["ru"], coupling.outputs["rv"],
            coupling.outputs["rtheta"], coupling.outputs["rqv"], coupling.outputs["rqc"],
            coupling.outputs["rqr"] if original.rqr is not None else None,
            coupling.outputs["rqi"] if original.rqi is not None else None,
            coupling.outputs["rqs"] if original.rqs is not None else None)

    old_pbl = driver.pbl_tendencies
    shared_target = driver.tendencies is old_pbl
    driver.pbl_tendencies = held_zero(old_pbl, pbl_coupling)
    driver.radiation_tendencies = held_zero(driver.radiation_tendencies, radiation_coupling)
    arrays = extras.arrays
    driver.cumulus_tendencies = PhysicsTendencies(
        arrays["physics:driver:zero_ru"].reshape(nz, members * ny, nx + 1),
        arrays["physics:driver:zero_rv"].reshape(nz, members * (ny + 1), nx),
        *(arrays[f"physics:driver:zero_{name}"].reshape(nz, members * ny, nx) for name in ("rtheta", "rqv", "rqc")))
    assert shared_target
    driver.tendencies = driver.pbl_tendencies
    driver.state.prepared_physics_atmosphere = atmosphere
    driver._ensemble_native_column_owners = (ysu, sfclay, noah, extras, atmosphere)
    driver._ensemble_native_column_receipt = {"members": members, "suite": "thompson-noah-ysu-mm5",
        "shared_surface_fields": sorted(shared_surfaces),
        "required_binding_bytes": required, "ysu": ysu.receipt, "sfclay": sfclay.receipt, "noah": noah.receipt,
        "extra_allocations": list(extras_plan.inventory(members))}
    return driver
