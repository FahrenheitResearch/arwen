"""Setup orchestration for Rust Noah-OWP initialization and verified GPU transfer."""
from __future__ import annotations
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from gpuwm import bridges


def _artifact(env, name):
    candidates = list(bridges.artifact_candidates(env, name))
    target = Path(__file__).resolve().parents[3]/'tools/noahowp-init/target'
    candidates[1:1] = [target/'release'/name, target/'debug'/name]
    for path in candidates:
        if path.is_file():
            return bridges.accept_resolved(path.resolve(), executable=env == 'GPUWM_NOAHOWP_INIT')
        if os.environ.get(env) and path == Path(os.environ[env]):
            raise FileNotFoundError(f'{env} names a missing artifact: {path}')
    raise FileNotFoundError(f'Missing Rust Noah-OWP initializer artifact {name}')


def prepare_initial(request, destination, *, parameter_dir=None):
    """Initialize using shipped standard tables or explicit per-row overrides."""
    name = 'arwen-noahowp-init'+('.exe' if os.name == 'nt' else '')
    command = _artifact('GPUWM_NOAHOWP_INIT', name)
    parameters = Path(parameter_dir) if parameter_dir is not None else Path(__file__).with_name('parameters')
    run = subprocess.run([str(command), str(request), str(destination), '--parameters', str(parameters)],
                         check=True, capture_output=True, text=True)
    return json.loads(run.stdout)


def load_initial_cuda(manifest_path, *, device_id):
    """Verify payload hashes in Rust, then copy pinned buffers to the parent GPU."""
    import cupy as cp
    if sys.byteorder != 'little':
        raise RuntimeError('Noah-OWP initial slabs require little-endian payload transport')
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    schema_bytes = Path(__file__).with_name('schema.json').read_bytes()
    schema = json.loads(schema_bytes)
    if (manifest.get('schema') != 'arwen.noahowp.initial.v1' or
            manifest.get('source_revision') != schema['source_revision'] or
            manifest.get('schema_sha256') != hashlib.sha256(schema_bytes).hexdigest()):
        raise ValueError('Noah-OWP initial slab schema/source revision differs')
    n = len(manifest['columns'])
    if n < 1 or len({row['id'] for row in manifest['columns']}) != n:
        raise ValueError('Noah-OWP initial catchment roster is empty or repeats IDs')
    name = ('arwen_noahowp_init.dll' if os.name == 'nt' else
            'libarwen_noahowp_init.dylib' if sys.platform == 'darwin' else
            'libarwen_noahowp_init.so')
    library = ctypes.CDLL(str(_artifact('GPUWM_NOAHOWP_INIT_LIBRARY', name)))
    read = library.arwen_noahowp_read_into
    read.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    read.restype = ctypes.c_int
    arrays = {}
    with cp.cuda.Device(device_id):
        for dtype in ('f32', 'i32', 'f64'):
            spec = manifest['files'][dtype]
            count = max(f['offset']+f['count'] for f in schema['fields'] if f['dtype'] == dtype)
            shape = (count, n)
            cp_dtype = cp.dtype({'f32':'float32','i32':'int32','f64':'float64'}[dtype])
            size = math.prod(shape)*cp_dtype.itemsize
            if spec['shape'] != list(shape) or spec['bytes'] != size or spec['dtype'] != dtype:
                raise ValueError(f'Noah-OWP slab shape/dtype/size differs: {dtype}')
            payload = (path.parent/spec['path']).resolve()
            if not payload.is_relative_to(path.parent):
                raise ValueError('Noah-OWP initial payload escapes its directory')
            pinned = cp.cuda.alloc_pinned_memory(size)
            pointer = ctypes.addressof(ctypes.c_char.from_buffer(pinned))
            status = read(os.fsencode(payload), spec['sha256'].encode('ascii'), pointer, size)
            if status:
                raise RuntimeError(f'Rust Noah-OWP payload verification failed: {dtype}, status {status}')
            array = cp.empty(shape, dtype=cp_dtype)
            cp.cuda.runtime.memcpy(array.data.ptr, pointer, size, cp.cuda.runtime.memcpyHostToDevice)
            arrays[dtype] = array
    return arrays, manifest
