"""Ordered diagnostic programs. WRF chem/module_gocart_aerosols.F:112-144.

Index-range additions become individual terms to preserve accumulator rounding.
Column integrals follow DESIGN section 8, with dry layer mass formed first.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np


@dataclass(frozen=True)
class DiagnosticProgram:
    kind: str
    attributes: tuple
    terms: tuple
    divide_by_alt: bool
    column_scale: np.float32


def compile_diagnostic(diag, table):
    """Resolve table terms to state attributes, skipping incomplete terms."""
    attrs, terms = [], []
    for term in diag.terms:
        if not all(name in table for name in term['species']):
            continue
        slots = []
        for name in term['species']:
            attr = table.row(name).state_attr
            if attr not in attrs:
                attrs.append(attr)
            slots.append(attrs.index(attr))
        terms.append((tuple(slots), tuple(({'mul': 0, 'div': 1}[op], np.float32(v))
                                         for op, v in term.get('ops', ()))))
    if not terms:
        return None
    return DiagnosticProgram(diag.kind, tuple(attrs), tuple(terms),
                             diag.divide_by_alt, np.float32(diag.column_scale))


def _arrays(program, fields):
    return [fields[a] if isinstance(fields, Mapping) else getattr(fields, a)
            for a in program.attributes]


def _validate(program, arrays, alt, dz8w, out):
    shape = alt.shape
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('alt must have nonempty (nz, ny, nx) shape to prevent invalid grid reads')
    expected = shape if program.kind == 'term_sum_3d' else shape[1:]
    for a in [alt, *arrays]:
        if a.shape != shape or a.dtype != np.float32 or not a.flags.c_contiguous:
            raise ValueError('fields and alt must be contiguous float32 grid arrays to prevent invalid pointer reads')
    if out.shape != expected or out.dtype != np.float32 or not out.flags.c_contiguous:
        raise ValueError('out shape or layout would cause invalid diagnostic writes')
    if program.kind == 'column_integral':
        if dz8w is None or dz8w.shape != shape or dz8w.dtype != np.float32 or not dz8w.flags.c_contiguous:
            raise ValueError('column_integral needs contiguous float32 dz8w to prevent invalid layer mass reads')
    for a in [alt, *arrays, *([dz8w] if dz8w is not None else [])]:
        if hasattr(out, '__array_interface__'):
            overlap = np.shares_memory(out, a)
        else:
            lo, hi = out.data.ptr, out.data.ptr + out.nbytes
            overlap = lo < a.data.ptr + a.nbytes and a.data.ptr < hi
        if overlap:
            raise ValueError('out overlaps an input and would overwrite unread diagnostic values')


def _program_buffers(program):
    species, starts, opstarts, codes, values = [], [0], [0], [], []
    for slots, ops in program.terms:
        species.extend(slots)
        starts.append(len(species))
        for code, value in ops:
            codes.append(code)
            values.append(value)
        opstarts.append(len(codes))
    return species, starts, opstarts, codes, values


def _evaluate_host(program, fields, alt, dz8w, out):
    """Dispatch an ordinary host array diagnostic to the Rust backend."""
    import ctypes
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    arrays = _arrays(program, fields)
    _validate(program, arrays, alt, dz8w, out)
    try:
        kind = {"term_sum_3d": 0, "term_sum_surface": 1,
                "column_integral": 2}[program.kind]
    except KeyError as error:
        raise ValueError("unknown diagnostic kind would select an invalid output layout") from error
    species, starts, opstarts, codes, values = _program_buffers(program)
    buffers = [np.asarray([a.ctypes.data for a in arrays], dtype=np.uintp),
               np.asarray(species, dtype=np.int32), np.asarray(starts, dtype=np.int32),
               np.asarray(opstarts, dtype=np.int32), np.asarray(codes, dtype=np.int32),
               np.asarray(values, dtype=np.float32)]
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    function = CpuPreprocessBackend()._chem_symbol("gpuwm_chem_diagnostic_f32", [
        pointer, size, pointer, pointer, pointer, pointer, pointer, size,
        pointer, pointer, size, size, ctypes.c_uint32, ctypes.c_uint8,
        ctypes.c_float, pointer])
    nz, ny, nx = alt.shape
    code = function(buffers[0].ctypes.data, len(arrays),
                    *(buffer.ctypes.data for buffer in buffers[1:]),
                    len(program.terms), alt.ctypes.data,
                    None if dz8w is None else dz8w.ctypes.data,
                    nz, ny * nx, kind, int(program.divide_by_alt),
                    float(program.column_scale), out.ctypes.data)
    if code:
        raise ValueError(f"Rust chemistry diagnostic rejected invalid program or buffer: {code}")
    return out


def evaluate(program, fields, alt, dz8w, out):
    """Launch one kernel for the diagnostic kind; fields is state or attr mapping.

    The kernel is loaded through the engine's own ``get_kernel`` like every
    other translation unit; every float32 operation in it is spelled with
    an explicit round-to-nearest intrinsic, so no compile flag can contract
    or reassociate a word.
    """
    if isinstance(alt, np.ndarray):
        return _evaluate_host(program, fields, alt, dz8w, out)
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    arrays = _arrays(program, fields)
    _validate(program, arrays, alt, dz8w, out)
    species, starts, opstarts, codes, values = _program_buffers(program)
    buffers = [cp.asarray([a.data.ptr for a in arrays], dtype=cp.uint64),
               cp.asarray(species, dtype=cp.int32), cp.asarray(starts, dtype=cp.int32),
               cp.asarray(opstarts, dtype=cp.int32), cp.asarray(codes, dtype=cp.int32),
               cp.asarray(values, dtype=cp.float32)]
    nz, ny, nx = alt.shape
    kernel = get_kernel("chem_outputs", program.kind)
    kernel(((out.size + 127)//128,), (128,), (*buffers, np.int32(len(program.terms)),
           alt, dz8w if dz8w is not None else alt, np.int32(nz), np.int32(ny*nx),
           np.int32(program.divide_by_alt), program.column_scale, out))
    return out
