"""Transcribe the retained sfctmp orchestration into explicit column code.

This runs on the CPU (it imports CuPy for the module objects and never
touches a device).  It executes the oracle's Python control flow,
``gpuwm.core.ruc.ruc_surface_temperature_step`` over the resident device
leaves of ``gpuwm.core.ruc_gpu``, with symbolic arrays, and records each
elementwise rounding boundary, each leaf argument list and each admission
check in the oracle's order.  It writes the two generated files the fused
path ships:

* ``gpuwm/core/kernels/ruc_fused_sfctmp.cuh``: three stage kernels, one
  thread per column, calling ``ruc.cu``'s leaves as device functions;
* ``gpuwm/core/ruc_sfctmp_layout.py``: the pointer table, output slots,
  ordered check messages and scratch slot sizes the launcher binds.

Run from the repository root after any change to the oracle:
``python -m tools.ruc_fused.gen_sfctmp``.  Nothing imports this at runtime.
"""
import json
import re
import inspect
from pathlib import Path
from types import FunctionType, SimpleNamespace
import numpy as np
from gpuwm.core import ruc as host

ROOT = Path(__file__).resolve().parents[2]
from gpuwm.core import ruc_gpu as gpu

N = 17
arrays = []
lines = []
checks = []
guard = 'true'
bindings = {}
cpu_checks = {}
selected_count = 0
split_offsets = []


def dtype(value):
    return np.dtype(value.dtype if isinstance(value, Cast) else value)


def literal(value):
    if isinstance(value, A):
        return value.expr
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = np.float32(value)
    return f'__int_as_float(0x{int(value.view(np.uint32)):08x})'


def emit(statement, condition=None):
    condition = guard if condition is None else condition
    lines.append(f'    if ({condition}) {{ {statement} }}')


class A:
    __array_priority__ = 10000

    def __init__(self, shape, dt='float32', expr=None, slot=None, condition='true'):
        self.shape = tuple(shape)
        self.dtype = dtype(dt)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.slot = slot
        self.expr = expr
        self.condition = condition
        self.flags = SimpleNamespace(c_contiguous=True)

    def at(self, profile=False):
        if self.expr is not None:
            return self.expr
        return f'a{self.slot}[{"k * n + c" if self.ndim == 2 else "c"}]'

    def pointer(self):
        if self.expr is not None:
            raise RuntimeError('leaf received an expression view')
        return f'a{self.slot}'

    def copy(self):
        out = allocate(self.shape, self.dtype)
        out.condition = self.condition
        assign(out, self, self.condition)
        return out

    def astype(self, dt, copy=True):
        dt = dtype(dt)
        if dt == self.dtype:
            return self
        out = allocate(self.shape, dt)
        out.condition = self.condition
        assign(out, self, self.condition)
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        shape = tuple(self.size if x == -1 else x for x in shape)
        return A(shape, self.dtype, self.expr, self.slot, self.condition)

    def __getitem__(self, key):
        if isinstance(key, Index):
            return A(self.shape, self.dtype, self.expr, self.slot, key.condition)
        if isinstance(key, tuple) and isinstance(key[-1], Index):
            return A(self.shape, self.dtype, self.expr, self.slot, key[-1].condition)
        if self.ndim == 2 and isinstance(key, int):
            return A((N,), self.dtype, expr=f'a{self.slot}[{key} * n + c]', condition=self.condition)
        raise RuntimeError(('index', self.shape, key))

    def __setitem__(self, key, value):
        selected = key[-1] if isinstance(key, tuple) else key
        condition = selected.condition if isinstance(selected, Index) else guard
        assign(self, value, condition)

    def binary(self, other, op, reverse=False, dt=None):
        shape = self.shape
        if isinstance(other, A) and other.ndim > self.ndim:
            shape = other.shape
        out = allocate(shape, dt or self.dtype)
        conditions = [self.condition]
        if isinstance(other, A):
            conditions.append(other.condition)
        out.condition = ' && '.join(dict.fromkeys(x for x in conditions if x != 'true')) or 'true'
        x = self.at()
        y = other.at() if isinstance(other, A) else literal(other)
        if op in ('<', '>', '==', '>=', '<=') and not isinstance(other, A):
            if other == 30:
                y = 'ncategory'
            elif other == 13:
                y = 'urban'
        elif (not isinstance(other, A) and isinstance(other, np.floating)
              and other == np.float32(12)):
            # The trace runs at delt = 12 (see main); no other 12.0 occurs
            # in the dispatch's arithmetic, so 12.0 there is the step.
            y = 'delt'
        if reverse:
            x, y = y, x
        expression = f'{op}({x}, {y})' if op.startswith('__f') else f'({x} {op} {y})'
        assign(out, expression, out.condition)
        return out

    def __add__(self, b): return self.binary(b, '__fadd_rn')
    def __radd__(self, b): return self.binary(b, '__fadd_rn', True)
    def __sub__(self, b):
        return self.binary(b, '-' if self.dtype.kind in 'iu' else '__fsub_rn')
    def __rsub__(self, b): return self.binary(b, '__fsub_rn', True)
    def __mul__(self, b): return self.binary(b, '__fmul_rn')
    def __rmul__(self, b): return self.binary(b, '__fmul_rn', True)
    def __truediv__(self, b): return self.binary(b, '__fdiv_rn')
    def __rtruediv__(self, b): return self.binary(b, '__fdiv_rn', True)
    def __lt__(self, b): return self.binary(b, '<', dt='bool')
    def __le__(self, b): return self.binary(b, '<=', dt='bool')
    def __gt__(self, b): return self.binary(b, '>', dt='bool')
    def __ge__(self, b): return self.binary(b, '>=', dt='bool')
    def __eq__(self, b): return self.binary(b, '==', dt='bool')
    def __ne__(self, b): return self.binary(b, '!=', dt='bool')
    def __and__(self, b): return self.binary(b, '&&', dt='bool')
    def __or__(self, b): return self.binary(b, '||', dt='bool')
    def __invert__(self):
        out = allocate(self.shape, 'bool')
        out.condition = self.condition
        assign(out, f'!({self.at()})', self.condition)
        return out
    def __neg__(self):
        out = allocate(self.shape, self.dtype)
        out.condition = self.condition
        assign(out, f'-({self.at()})', self.condition)
        return out


def allocate(shape, dt='float32', binding=None, **kw):
    dt = kw.get('dtype', dt)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    slot = len(arrays)
    out = A(shape, dt, slot=slot, condition=guard)
    arrays.append({'shape': list(shape), 'dtype': str(out.dtype), 'binding': binding})
    if binding:
        bindings[binding] = slot
    return out


def assign(out, value, condition=None):
    expression = value.at() if isinstance(value, A) else value if isinstance(value, str) else literal(value)
    target = out.at()
    statement = f'{target} = {expression};'
    if out.ndim == 2:
        statement = f'for (int k = 0; k < RUC_NZS; ++k) {{ {statement} }}'
    emit(statement, condition)


class Cast:
    dtype = np.dtype('float32')
    def __call__(self, value):
        return value.astype('float32') if isinstance(value, A) else np.float32(value)


class Index:
    size = N
    def __init__(self, condition): self.condition = condition


class Reduction:
    def __init__(self, value): self.value = value
    def __bool__(self): return True
    def __int__(self): return 1


class Stack:
    def __init__(self, values): self.values = values
    def tolist(self): return [True] * len(self.values)


def asarray(value, dt=None, dtype=None):
    dt = dtype or dt
    if isinstance(value, A):
        return value if dt is None or value.dtype == globals()['dtype'](dt) else value.astype(dt)
    raise RuntimeError(('upload', type(value), np.shape(value)))


def full(shape, value, dtype='float32'):
    out = allocate(shape, dtype)
    out.condition = 'true'
    assign(out, value, 'true')
    return out


def where(condition, left, right):
    template = next(x for x in (left, right, condition) if isinstance(x, A))
    out = allocate(template.shape, template.dtype if template is not condition else 'float32')
    out.condition = template.condition
    assign(out, f'({condition.at()} ? {left.at() if isinstance(left, A) else literal(left)} : {right.at() if isinstance(right, A) else literal(right)})', out.condition)
    return out


def extremum(left, right, name):
    template = left if isinstance(left, A) else right
    out = allocate(template.shape, template.dtype)
    out.condition = template.condition
    assign(out, f'{name}({left.at() if isinstance(left, A) else literal(left)}, {right.at() if isinstance(right, A) else literal(right)})', out.condition)
    return out


def finite(value):
    out = allocate(value.shape, 'bool')
    assign(out, f'isfinite({value.at()})')
    return out


fake = SimpleNamespace(
    float32=Cast(), int32=np.int32, intp=np.intp, integer=np.integer,
    ndarray=A, issubdtype=np.issubdtype, prod=np.prod,
    asarray=asarray, ascontiguousarray=lambda v, dtype=None: asarray(v, dtype),
    array=lambda v, dtype=None, copy=True: asarray(v, dtype).copy(),
    empty=allocate, empty_like=lambda v: allocate(v.shape, v.dtype),
    zeros=lambda shape, dtype='float32': full(shape, 0, dtype),
    ones=lambda shape, dtype='float32': full(shape, 1, dtype), full=full,
    broadcast_to=lambda v, shape: A(shape, v.dtype, v.expr, v.slot),
    isfinite=lambda v: finite(v) if isinstance(v, A) else np.isfinite(v),
    minimum=lambda a, b: extremum(a, b, 'ruc_fused_min'),
    maximum=lambda a, b: extremum(a, b, 'ruc_fused_max'),
    where=where, all=lambda v: Reduction(v), any=lambda v: Reduction(v),
    stack=lambda v: Stack(v),
    cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0)),
)


def record(condition, message, profile=False, payload=None):
    index = len(checks)
    checks.append(message)
    body = f'if ({condition}) {{ failed = true; atomicOr(flags + {index // 64}, 1ull << {index % 64});'
    if payload:
        body += f' atomicMin(flags + RUC_SFCTMP_FLAG_WORDS + {index}, {payload});'
    body += ' }'
    if profile:
        body = f'for (int k = 0; k < RUC_NZS; ++k) {{ {body} }}'
    emit(body)


def checkpoint(label):
    index = len(checks)
    checks.append({'kind': 'host', 'label': label})
    cpu_checks[label] = index
    emit(f'if (limit <= {index}) {{ ruc_sfctmp_zero_outputs(ptrs, n, c); return; }}')


class Batch:
    def __init__(self, *args): pass
    def finite(self, v, name):
        self.finite_message(v, f'{name} must be finite')
        return v
    def finite_message(self, v, message):
        record(f'!isfinite({v.at()})', message, v.ndim == 2)
        return v
    def refuse_if_any(self, v, message):
        if callable(message):
            code = message.__code__
            closure = dict(zip(code.co_freevars, [cell.cell_contents for cell in message.__closure__]))
            if 'land_type' in closure:
                land = closure['land_type']
                record(v.at(), {'kind': 'land', 'name': 'iland'},
                       payload=f'({land.at()} < 1 ? (unsigned long long)(unsigned int)({land.at()} + 2147483648u) : (0x100000000ull | (unsigned int)(0xffffffffu - (unsigned int)({land.at()} + 2147483648u))))')
                return
            raise RuntimeError(code.co_freevars)
        record(v.at(), message, v.ndim == 2)
    def flush(self): emit('if (failed) { ruc_sfctmp_zero_outputs(ptrs, n, c); return; }')


def field(v, shape, name, *, batch=None, **kw):
    v = asarray(v)
    (batch or Batch()).finite(v, name)
    return v


def root(v, shape, *, nzs=9, batch=None, **kw):
    record(f'({v.at()} < 1 || {v.at()} >= RUC_NZS)', {'kind': 'root'},
           payload=f'((unsigned long long)c << 32) | (unsigned int){v.at()}')
    if batch is None:
        emit('if (failed) { ruc_sfctmp_zero_outputs(ptrs, n, c); return; }')
    return v


def selected(v, *args):
    global guard, selected_count
    selected_count += 1
    if selected_count in (4, 6):
        split_offsets.append(len(lines))
    guard = v.at()
    return Index(guard)


def flux(v, ncolumn=None, label=None, *, arrays=None, batch=None):
    if arrays is not None:
        checkpoint('conflx')
    message = f'RUC {"CUDA " if arrays is None else ""}{label or ncolumn} conflx must be finite and nonnegative'
    (batch or Batch()).finite_message(v, message)
    (batch or Batch()).refuse_if_any(v < np.float32(0), message)
    return v


def kernel(module, name=None):
    name = name or module
    def call(grid, block, args):
        cooked = []
        for arg in args:
            if isinstance(arg, A):
                cooked.append(arg.pointer())
            elif isinstance(arg, np.integer) and int(arg) == N:
                cooked.append('n')
            elif isinstance(arg, np.floating) and arg == np.float32(12):
                cooked.append('delt')
            elif isinstance(arg, np.floating) and arg == np.float32(5000):
                cooked.append('rsmax')
            else:
                cooked.append(literal(arg))
        if name == 'ruc_snow_preparation':
            # Named scalar positions are invariant in the retained signature.
            start = 1 + len(host.RUC_SNOW_PREP_COLUMN_INPUTS) + 4
            cooked[start:start+6] = ['delt', 'c1sn', 'c2sn', 'isice', 'urban', '2']
        emit(name + '(' + ', '.join(cooked) + ');')
    return call


def main():
    global guard
    env = dict(vars(gpu))
    for name, value in list(env.items()):
        if isinstance(value, FunctionType) and value.__module__ == gpu.__name__:
            env[name] = FunctionType(value.__code__, env, value.__name__, value.__defaults__, value.__closure__)
            env[name].__kwdefaults__ = value.__kwdefaults__
    tables = SimpleNamespace(
        rstbl=allocate((30,), binding='rstbl'),
        rgltbl=allocate((30,), binding='rgltbl'), rsmax_data=5000.0)
    rough = allocate((30,), binding='z0tbl')
    emiss = allocate((30,), binding='lemitbl')
    tbq = allocate((5001,), binding='tbq')
    half = allocate((9,), binding='zshalf')
    env.update(cp=fake, _validation_batch=Batch, _float_field=field,
               _float_profile=field, _integer_field=lambda v, *a: v,
               _root_count_field=root, _resolved_soil_levels=lambda *a: 9,
               _device_constant_flux_depth=flux, get_kernel=kernel,
               _ruc_kernel=lambda name, nzs, *a, **k: kernel(name),
               _device_tbq=lambda *a: tbq,
               _snow_preparation_tables=lambda *a: (rough, emiss, 13, 30),
               _default_device_tables=lambda *a: (tables, 30, None, None),
               _upload_tables=lambda *a: (tables, 30, None, None),
               _bundle_device_tables=lambda *a: (tables, 30, None, None),
               _device_soil_half_levels=lambda *a: half,
               ruc_zshalf=lambda *a: half)
    prep_source = inspect.getsource(gpu.ruc_snow_preparation_cuda)
    prep_source = prep_source.replace('    if not 1 <= isice <= ncategory:',
        '    _ice_check()\n    if not 1 <= isice <= ncategory:')
    prep_source = prep_source.replace('    land_category = _integer_field',
        '    _checkpoint("prep_iland")\n    land_category = _integer_field')
    env['_ice_check'] = lambda: record('(isice < 1 || isice > ncategory)', 'RUC isice is outside 1..30')
    env['_checkpoint'] = checkpoint
    exec(compile(prep_source, '<snow preparation scalar checkpoint>', 'exec'), env)
    henv = dict(vars(host))
    for name in ('ruc_surface_temperature_step', '_ruc_surface_net_radiation', '_ruc_tanh_array'):
        value = henv[name]
        henv[name] = FunctionType(value.__code__, henv, value.__name__, value.__defaults__)
        henv[name].__kwdefaults__ = value.__kwdefaults__
    henv.update(RucValidationBatch=Batch, _resolved_soil_levels=lambda *a: 9,
                _horizontal_float_field=field, _horizontal_integer_field=lambda v,*a,**k:v,
                _root_count_field=root, _ruc_constant_flux_depth=flux,
                _selected=selected,
                # The snow lineage is the translation unit's define.
                _ruc_snow_lineage_mask=lambda snow, ncolumn, np: A(
                    (N,), 'bool', expr='GPUWM_RUC_SNOW_V461'))
    def horizontal(value, shape, name, **kw):
        checkpoint('column:' + name)
        return field(value, shape, name, **kw)
    henv['_horizontal_float_field'] = horizontal
    henv['_checkpoint'] = checkpoint
    surface_source = inspect.getsource(host.ruc_surface_temperature_step)
    for anchor, label in (
        ('    if not np.isfinite(timestep)', 'delt'),
        ('    if isncovr_opt not', 'isncovr_opt'),
        ('    if type(isice)', 'isice_type'),
        ('    missing = [', 'missing'), ('    bundle =', 'bundle'),
        ('    vegetation_category =', 'ivgtyp'), ('    roots =', 'nroot'),
        ('    raw_layers =', 'ilnb')):
        surface_source = re.sub('^' + re.escape(anchor),
            lambda m: f'    _checkpoint({label!r})\n' + m.group(0), surface_source, flags=re.M)
    surface_source = surface_source.replace('        array = np.asarray(values[name]',
        '        _checkpoint("profile:" + name)\n        array = np.asarray(values[name]')
    # The final batch is emitted below directly as ordered flag checks.
    # It needs no temporary boolean arrays or simulated host verdicts.
    surface_source = surface_source.split('    # One reduction per field,')[0] + '    return result\n'
    exec(compile(surface_source, '<sfctmp scalar checkpoints>', 'exec'), henv)
    def prep_stage(values, bundle=None, **kw):
        checkpoint('prep_bundle')
        checkpoint('snow_density_scalars')
        return env['ruc_snow_preparation_cuda'](values, **kw)
    fake.ruc_tanhf_glibc = env['ruc_tanhf_glibc']
    values = {name: allocate((9, N), binding='value:' + name)
              for name in host.RUC_SFCTMP_PROFILE_INPUTS}
    values.update({name: allocate((N,), binding='value:' + name)
                   for name in host.RUC_SFCTMP_COLUMN_INPUTS})
    kw = {name: allocate((N,), 'float32' if name == 'conflx' else 'int32', binding=name)
          for name in ('conflx', 'ivgtyp', 'iland', 'nroot', 'ilnb')}
    result = henv['ruc_surface_temperature_step'](
        values, **kw, delt=12, arrays=fake,
        leaves={key: env[name] for key, name in (
            ('soil','ruc_soil_step_cuda'), ('sea_ice','ruc_sea_ice_step_cuda'),
            ('snow_soil','ruc_snow_soil_step_cuda'), ('snow_sea_ice','ruc_snow_sea_ice_step_cuda'))},
        stages={'snow_prep': prep_stage})
    guard = 'true'
    outputs = {}
    for name in host.RucSurfaceTemperatureStep.__dataclass_fields__:
        v = getattr(result, name)
        outputs[name] = v.slot
        if v.dtype.kind == 'f':
            Batch().finite_message(v, f'RUC sfctmp produced non-finite {name}')
    header = ['// Generated by tools/ruc_fused/gen_sfctmp.py from the retained oracle.',
              '// Each elementwise operation is an explicit IEEE rounding boundary.',
              '#ifndef NAN\n#define NAN __int_as_float(0x7fc00000)\n#endif',
              '__device__ __forceinline__ real ruc_fused_min(real a, real b) { return (isnan(a) || isnan(b)) ? NAN : fminf(a, b); }',
              '__device__ __forceinline__ real ruc_fused_max(real a, real b) { return (isnan(a) || isnan(b)) ? NAN : fmaxf(a, b); }']
    types = {'float32':'real', 'int32':'int', 'bool':'bool'}
    header += [f'#define a{i} (({types[a["dtype"]]}*)ptrs[{i}])' for i,a in enumerate(arrays)]
    header += ['__device__ __forceinline__ void ruc_sfctmp_zero_outputs(const unsigned long long* ptrs, int n, int c) {']
    for index in outputs.values():
        if len(arrays[index]['shape']) == 2:
            header.append(f'    for (int k = 0; k < RUC_NZS; ++k) a{index}[k * n + c] = 0;')
        else:
            header.append(f'    a{index}[c] = 0;')
    header += ['}']
    header += ['extern "C" __global__ void ruc_sfctmp_fused(',
               'const unsigned long long* ptrs, const bool* run, unsigned long long* flags,',
               'real delt, real c1sn, real c2sn, real rsmax, int isice, int urban, int ncategory, int limit, int n) {',
               '    int c = blockIdx.x * blockDim.x + threadIdx.x;',
               '    if (c >= n) return;',
               '    if (!run[c]) { ruc_sfctmp_zero_outputs(ptrs, n, c); return; }']
    header.append('    bool failed = false;')
    header.insert(0, f'#define RUC_SFCTMP_FLAG_WORDS {(len(checks)+63)//64}')
    entry = header.index('extern "C" __global__ void ruc_sfctmp_fused(')
    # Three stages: one kernel over the whole sfctmp needs a larger local
    # frame at six levels than the largest existing RUC kernel.
    bodies = header[:entry]
    boundaries = [0] + split_offsets + [len(lines)]
    for stage in range(3):
        bodies += [f'extern "C" __global__ void ruc_sfctmp_stage{stage}(',
            'const unsigned long long* ptrs, const bool* run, unsigned long long* flags, bool* alive,',
            'real delt, real c1sn, real c2sn, real rsmax, int isice, int urban, int ncategory, int limit, int n) {',
            '    int c = blockIdx.x * blockDim.x + threadIdx.x;',
            '    if (c >= n) return;']
        if stage:
            bodies += ['    if (!alive[c]) return;']
        bodies += ['    alive[c] = false;',
                   '    if (!run[c]) { ruc_sfctmp_zero_outputs(ptrs, n, c); return; }',
                   '    bool failed = false;']
        bodies += lines[boundaries[stage]:boundaries[stage+1]]
        bodies += ['    alive[c] = true;', '}']
    text = '\n'.join(bodies + [f'#undef a{i}' for i in range(len(arrays))]) + '\n'
    (ROOT / 'gpuwm/core/kernels/ruc_fused_sfctmp.cuh').write_bytes(text.encode())
    # Reuse storage only after the final textual use, including guard reads.
    uses = {i: [] for i in range(len(arrays))}
    for lineno, line in enumerate(lines):
        for index in set(map(int, re.findall(r'\ba(\d+)\b', line))):
            uses[index].append(lineno)
    for index in outputs.values():
        uses[index].append(len(lines))
    locations, available, active, cursor = {}, {}, [], 0
    for index in sorted((i for i,a in enumerate(arrays) if not a['binding']), key=lambda i: min(uses[i], default=0)):
        begin, end = min(uses[index], default=0), max(uses[index], default=0)
        expired = [item for item in active if item[0] < begin]
        for item in expired:
            active.remove(item)
            available.setdefault(item[2], []).append(item[1])
        # Reuse requires the same column address stride. Boolean elements
        # occupy one byte; float and integer elements occupy four bytes.
        # Mixing those layouts would let one column overwrite another.
        size = (len(arrays[index]['shape']) == 2, np.dtype(arrays[index]['dtype']).itemsize)
        pool = available.setdefault(size, [])
        location = pool.pop() if pool else cursor
        if location == cursor:
            cursor += 1
        locations[index] = location
        active.append((end, location, size))
    slots = {}
    for index, location in locations.items():
        slots[location] = 9 if len(arrays[index]['shape']) == 2 else 1
    metadata = {'arrays': arrays, 'bindings': bindings, 'outputs': outputs, 'checks': checks,
                'locations': locations, 'slots': slots, 'cpu_checks': cpu_checks,
                'split_offsets': split_offsets}
    write_layout(json.loads(json.dumps(metadata)))
    print(len(arrays), 'arrays;', len(checks), 'checks;', len(lines), 'instructions')


def write_layout(layout):
    """The launcher's tables, as ``gpuwm/core/ruc_sfctmp_layout.py``."""
    descriptors = tuple((a['binding'], a['dtype'], len(a['shape']) == 2,
                         layout['locations'].get(str(i)))
                        for i, a in enumerate(layout['arrays']))
    text = (
        '"""The fused sfctmp launcher tables.  Generated; do not edit.' + chr(10) + chr(10)
        + 'Regenerate with ``python -m tools.ruc_fused.gen_sfctmp`` together with' + chr(10)
        + '``gpuwm/core/kernels/ruc_fused_sfctmp.cuh``: the pointer slots, output' + chr(10)
        + 'slots and check indices here are the ones that kernel source uses.' + chr(10)
        + '"""' + chr(10) + chr(10))
    for name, value in (
        ('_SFCTMP_ARRAYS', descriptors), ('_SFCTMP_OUTPUTS', layout['outputs']),
        ('_SFCTMP_CHECKS', tuple(layout['checks'])),
        ('_SFCTMP_CPU_CHECKS', layout['cpu_checks']),
        ('_SFCTMP_SLOTS', tuple(layout['slots'][str(i)] for i in range(len(layout['slots'])))),
    ):
        text += name + ' = ' + repr(value) + chr(10)
    (ROOT / 'gpuwm/core/ruc_sfctmp_layout.py').write_bytes(text.encode())


if __name__ == '__main__':
    main()
