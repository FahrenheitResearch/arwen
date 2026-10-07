"""Independent ordered float32 diagnostic oracle for tests only."""
import numpy as np
from gpuwm.core.chem_outputs import _arrays, _validate


def evaluate_cpu(program, fields, alt, dz8w, out):
    """NumPy float32 oracle reference, with one rounding per WRF operation."""
    arrays = _arrays(program, fields)
    _validate(program, arrays, alt, dz8w, out)
    def level(k):
        total = np.zeros(alt.shape[1:], dtype=np.float32)
        for slots, ops in program.terms:
            value = arrays[slots[0]][k].copy()
            for slot in slots[1:]:
                value = np.add(value, arrays[slot][k], dtype=np.float32)
            for op, operand in ops:
                value = (np.multiply if op == 0 else np.divide)(value, operand, dtype=np.float32)
            total = np.add(total, value, dtype=np.float32)
        if program.divide_by_alt:
            total = np.divide(total, alt[k], dtype=np.float32)
        return total
    if program.kind == 'term_sum_3d':
        for k in range(alt.shape[0]):
            out[k] = level(k)
    elif program.kind == 'term_sum_surface':
        out[:] = level(0)
    elif program.kind == 'column_integral':
        out.fill(0)
        for k in range(alt.shape[0]):
            mass = np.multiply(np.divide(np.float32(1), alt[k], dtype=np.float32), dz8w[k], dtype=np.float32)
            out[:] = np.add(out, np.multiply(level(k), mass, dtype=np.float32), dtype=np.float32)
        out[:] = np.multiply(out, program.column_scale, dtype=np.float32)
    else:
        raise ValueError('unknown diagnostic kind would select an invalid output layout')
    return out
