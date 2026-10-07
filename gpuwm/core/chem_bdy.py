"""Outer-domain chem boundaries from WRF 4.7.1 module_input_chem_data.F.

flow_dep_bdy_chem (:1531-2031) visits Y-start, Y-end, X-start, X-end.
WRF (i,k,j), with u(i) the west face, maps to (k,j,i), with
ru_m[k,j,i] the west face and rv_m[k,j,i] the south face. Only signs
are read. dt is dt_rk + grid%dtbc (solve_em.F:2599-2616).

Compared with ingest.lateral_bc.apply_flow_dependent_boundaries and
kernels/lbc_flow.cu, corner ownership and copied inner rows are the same:
Y owns each ring's corners, and the source is the first row beyond the
entire spec_zone, with its transverse coordinate clamped to the interior.
Chem inflow instead uses each row's default_inflow or
max(epsilc, b + bt*dt), reading only boundary row 1 (:2331-2357).
There is no chem relaxation or spec tendency (solve_em.F:2504-2508).

A pointer table avoids copying full 3-D fields into a stacked arena.
One thread per field and level executes all four sides serially in one
launch, preserving source dependencies without a grid-wide barrier.
The independent verification reference retains separate float32 operations.
"""
from __future__ import annotations

import operator

import numpy as np

EPSILC = np.float32(1.e-16)


def _validate(fields, ru_m, rv_m, spec_zone, has_bc, default_inflow, bounds, xp):
    if not fields:
        return None
    if fields[0].ndim != 3:
        raise ValueError("chem boundary fields must be 3-D to prevent invalid indexing")
    nz, ny, nx = fields[0].shape
    width = operator.index(spec_zone)
    if width < 1 or nz < 1 or min(nx, ny) <= 2 * width:
        raise ValueError("spec_zone must leave an interior row to copy")
    def array(a, shape):
        if not isinstance(a, xp.ndarray) or a.shape != shape:
            raise ValueError(f"chem boundary array needs shape {shape} to prevent invalid indexing")
        if a.dtype != xp.float32 or not a.flags.c_contiguous:
            raise ValueError("chem boundary arrays must be contiguous float32 to prevent misread words")
    for f in fields:
        array(f, (nz, ny, nx))
    array(ru_m, (nz, ny, nx + 1))
    array(rv_m, (nz, ny + 1, nx))
    # The per-row flags arrive as host values and are judged on the host,
    # so no launch waits on a device read of them.
    has_host = np.asarray(has_bc, dtype=np.int32)
    defaults_host = np.asarray(default_inflow, dtype=np.float32)
    if (has_host.shape != (len(fields),)
            or defaults_host.shape != (len(fields),)):
        raise ValueError("chem boundary row parameters need one value per field to prevent row mismatch")
    supplied = bool(has_host.any())
    has = xp.asarray(has_host)
    defaults = xp.asarray(defaults_host)
    for n, b in enumerate(bounds):
        if b is None:
            if supplied:
                raise ValueError("boundary source rows require all eight arrays to prevent missing inflow")
        else:
            array(b, (len(fields), nz, ny if n < 4 else nx))
    return nz, ny, nx, width, xp.ascontiguousarray(has), xp.ascontiguousarray(defaults)




def apply_chem_flow_boundaries(fields, ru_m, rv_m, spec_zone, has_bc,
                               default_inflow, bxs, btxs, bxe, btxe,
                               bys, btys, bye, btye, dt):
    """Update device fields in place; boundaries contain only the outer row.

    X arrays are (nfields,nz,ny), Y arrays (nfields,nz,nx). They may be
    None when no row has a boundary source. Parameters follow table row
    order; no species identities are inspected. An empty batch is inert.
    """
    fields = tuple(fields)
    if not fields:
        return
    if any(isinstance(field, np.ndarray) for field in fields):
        raise TypeError(
            "chemistry flow boundaries require CUDA arrays: NumPy fields "
            "would supply host pointers to the boundary kernel")
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    bounds = (bxs, btxs, bxe, btxe, bys, btys, bye, btye)
    nz, ny, nx, w, has, defaults = _validate(
        fields, ru_m, rv_m, spec_zone, has_bc, default_inflow, bounds, cp)
    pointers = cp.asarray([f.data.ptr for f in fields], dtype=cp.uint64)
    # Unused pointers are valid arrays, but the kernel never dereferences them.
    bounds = tuple(ru_m if b is None else b for b in bounds)
    count = len(fields) * nz
    get_kernel("chem_bdy", "chem_flow_boundaries")(
        ((count + 127)//128,), (128,),
        (pointers, ru_m, rv_m, has, defaults, *bounds, np.float32(dt),
         np.int32(len(fields)), np.int32(nz), np.int32(ny), np.int32(nx), np.int32(w)))
