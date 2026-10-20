"""Native replay of actual AC acoustic calls; no whole-step speed claim."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--binding', type=Path, required=True)
p.add_argument('--numerics', choices=('strict', 'fast'), required=True)
p.add_argument('--profile-only', action='store_true')
p.add_argument('--backend-dir', type=Path)
args = p.parse_args()
binding = json.loads(args.binding.read_text())
mask = os.environ['CUDA_VISIBLE_DEVICES']
os.environ.update(binding['runtime_env'])
os.environ.update(CUDA_VISIBLE_DEVICES=mask, GPUWM_NUMERICS=args.numerics)
for key, relative in {'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache',
                      'CUPY_CACHE_DIR': 'cache/cupy', 'CUDA_CACHE_PATH': 'cache/cuda',
                      'GPUWM_GENERATED_TABLE_ROOT': 'cache/generated'}.items():
    path = ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
sys.path[:0] = binding['runtime_env']['PYTHONPATH'].split(':')
import cupy as cp
import numpy as np
from gpuwm.core import dycore
from gpuwm.prepared_domain_tree_forecast import main as run
from gpuwm import numerics
backend_python = args.backend_dir / '__init__.py' if args.backend_dir else ROOT / 'gpuwm/native_step/__init__.py'
spec = importlib.util.spec_from_file_location('native_step', backend_python)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

class Frame(C.Structure):
    _fields_ = [('smdiv', C.c_float)]

class Complete(BaseException):
    pass

receipt = {'purpose': __doc__, 'numerics': args.numerics, 'wrf_exact': os.environ.get('GPUWM_WRF_EXACT'),
           'binding': binding, 'cases': [], 'profile_only': args.profile_only,
           'source': {}, 'completed_forecast': False, 'pid': os.getpid(), 'backend_python': str(backend_python),
           'backend_python_sha256': hashlib.sha256(backend_python.read_bytes()).hexdigest(),
           'backend_library_sha256': hashlib.sha256(Path(os.environ['GPUWM_NATIVE_STEP_LIBRARY']).read_bytes()).hexdigest()}
for name in ('gpuwm/core/acoustic.py', 'gpuwm/core/dycore.py',
             'gpuwm/native_step/recorder.c', 'gpuwm/native_step/__init__.py',
             'tools/native_step/measure_acoustic.py'):
    receipt['source'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

def closure_arrays(function):
    """Actual retained launch arguments, without traversing the whole state."""
    seen, found = set(), {}
    def visit(value, name):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, cp.ndarray):
            key = (value.data.ptr, value.shape, value.strides, value.dtype.str)
            found.setdefault(key, (name, value))
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                visit(item, f'{name}[{i}]')
        elif inspect.isfunction(value) and value.__closure__:
            for label, cell in zip(value.__code__.co_freevars, value.__closure__):
                visit(cell.cell_contents, f'{name}.{label}')
    visit(function, 'launch')
    return list(found.values())

def restore(arrays, saved):
    for (_, array), original in zip(arrays, saved):
        cp.copyto(array, original)
    cp.cuda.get_current_stream().synchronize()

def compare(arrays, expected):
    total = changed = 0
    details = []
    for (name, array), reference in zip(arrays, expected):
        host = cp.asnumpy(array)
        baseline = cp.asnumpy(reference)
        # Byte comparisons preserve signed zero and NaN payloads. Group by
        # original element size for the reported typed-word count.
        raw = host.tobytes(order='C')
        ref = baseline.tobytes(order='C')
        width = host.dtype.itemsize
        a = np.frombuffer(raw, np.uint8).reshape(-1, width)
        b = np.frombuffer(ref, np.uint8).reshape(-1, width)
        different = int(np.count_nonzero(np.any(a != b, axis=1)))
        total += int(host.size)
        changed += different
        if different:
            details.append({'array': name, 'words': different})
    return {'words': total, 'different_words': changed, 'different_arrays': details}

def measure(launch, cfg, stage, first):
    cells = inspect.getclosurevars(launch).nonlocals
    # The supported fixture uses specified/nested boundaries. Do not overlook
    # array writes expressed via state attributes in the open-boundary branch.
    assert not cells['radiative_x'] and not cells['radiative_y']
    uv_first, uv_later = cells['uv_first_args'], cells['uv_later_args']
    differing = [i for i, (a, b) in enumerate(zip(uv_first, uv_later))
                 if isinstance(a, np.generic) and a.tobytes() != b.tobytes()]
    assert differing == [30], differing
    assert len(uv_first) == len(uv_later)
    arrays = closure_arrays(launch)
    assert arrays, 'no retained launch arrays'
    initial = [a.copy() for _, a in arrays]
    cp.cuda.get_current_stream().synchronize()
    plan = native.Plan()
    entry = {'shape': [cfg.nx, cfg.ny, cfg.nz], 'stage': stage,
             'dtau': float(cells['dtau_arg']), 'array_count': len(arrays),
             'array_bytes': sum(a.nbytes for _, a in arrays), 'checks': [], 'timing': []}
    try:
        with plan.record(launch, *[a for _, a in arrays]):
            launch(first=True)
        cp.cuda.get_current_stream().synchronize()
        description = plan.describe()
        uv = [op for op in description['operations'] if op.get('kernel') == 'advance_uv']
        assert len(uv) == 1, uv
        plan.bind(uv[0]['index'], 30, Frame, 'smdiv')
        entry.update(operations=description['operations'], kernel_launches=description['kernel_launches'],
                     native_submissions=1, pool_reserved_bytes=description['pool_reserved_bytes'])
        for flag in (True, False):
            frame = Frame(0.0 if flag else float(cfg.smdiv))
            restore(arrays, initial)
            launch(first=flag)
            cp.cuda.get_current_stream().synchronize()
            expected = [a.copy() for _, a in arrays]
            restore(arrays, initial)
            plan.submit(frame)
            cp.cuda.get_current_stream().synchronize()
            check = compare(arrays, expected)
            check['first'] = flag
            entry['checks'].append(check)
            assert check['different_words'] == 0, check
            if args.profile_only:
                for arm in ('python', 'native'):
                    restore(arrays, initial)
                    cp.cuda.nvtx.RangePush(f'acoustic/{cfg.nx}x{cfg.ny}/stage{stage}/first{int(flag)}/{arm}')
                    if arm == 'python':
                        launch(first=flag)
                    else:
                        plan.submit(frame)
                    cp.cuda.get_current_stream().synchronize()
                    cp.cuda.nvtx.RangePop()
            else:
                samples = {'python': [], 'native': []}
                # Input restores and their synchronization are outside each
                # measurement. Both arms synchronize their submitted work.
                for repeat in range(9):
                    arms = ('python', 'native') if repeat % 2 == 0 else ('native', 'python')
                    for arm in arms:
                        restore(arrays, initial)
                        start = time.perf_counter_ns()
                        if arm == 'python':
                            launch(first=flag)
                        else:
                            plan.submit(frame)
                        cp.cuda.get_current_stream().synchronize()
                        samples[arm].append((time.perf_counter_ns() - start) / 1e6)
                entry['timing'].append({'first': flag, 'samples_ms': samples,
                    'mean_ms': {k: statistics.mean(v) for k, v in samples.items()},
                    'median_ms': {k: statistics.median(v) for k, v in samples.items()}})
            del expected
        # The enclosing model receives precisely one ordinary call, with its
        # actual flag. Native experiments cannot alter its continuation.
        restore(arrays, initial)
        launch(first=first)
        cp.cuda.get_current_stream().synchronize()
        entry['continuation'] = 'one original call from restored natural inputs'
        receipt['cases'].append(entry)
        print(json.dumps({'shape': entry['shape'], 'stage': stage,
                          'kernels': entry['kernel_launches'],
                          'checks': entry['checks'], 'timing': entry['timing']}), flush=True)
    finally:
        cp.cuda.get_current_stream().synchronize()
        plan.close()

original_step = dycore.step
original_factory = dycore.prepare_acoustic_substep_launch
counts = {}
active = None
stage_index = 0

def step(state, cfg, **kwargs):
    global active, stage_index
    key = (cfg.nx, cfg.ny, cfg.nz)
    counts[key] = counts.get(key, 0) + 1
    active = key if ((key == (120, 70, 44) and counts[key] == 2) or
                     (key == (267, 117, 44) and counts[key] == 4)) else None
    stage_index = 0
    result = original_step(state, cfg, **kwargs)
    if len(receipt['cases']) == 6:
        raise Complete()
    return result

def factory(state, cfg, dtau, coefficients, **kwargs):
    global stage_index
    launch = original_factory(state, cfg, dtau, coefficients, **kwargs)
    if active is None:
        return launch
    stage_index += 1
    stage = stage_index
    tested = False
    def wrapped(*, first):
        nonlocal tested
        if not tested:
            tested = True
            measure(launch, cfg, stage, first)
        else:
            launch(first=first)
    return wrapped

dycore.step = step
dycore.prepare_acoustic_substep_launch = factory
command = ['--prepared-root', binding['prepared_root'], binding.get('proof_option', '--prepared-head-sha256'),
           binding['prepared_head_sha256'], '--experiment-config', binding['config'],
           '--experiment-config-sha256', binding['config_sha256'], '--progress-format', 'jsonl',
           '--render-products', 'none', '--io-mode', 'none',
           '--outdir', str(ROOT / ('acoustic-check-' + str(time.time_ns())))]
receipt['argv'] = command
started = time.perf_counter()
try:
    run(command)
    raise RuntimeError('did not reach all six natural cases')
except Complete:
    receipt['status'] = 'PASS_COMPONENT_ONLY'
except BaseException as exc:
    receipt['status'] = 'FAILED'
    receipt['error'] = repr(exc)
    raise
finally:
    receipt['wall_s'] = time.perf_counter() - started
    receipt['pool_peak_reserved_bytes'] = cp.get_default_memory_pool().total_bytes()
    suffix = '-profile' if args.profile_only else ''
    (ROOT / f'receipts/acoustic-{args.numerics}{suffix}.json').write_text(json.dumps(receipt, indent=2)+'\n')
