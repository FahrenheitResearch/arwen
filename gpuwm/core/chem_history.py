"""What a chem run writes to history (DESIGN section 8).

Generic over the chem table: every active row with an ``output_name`` is
written as its 3-D field under that name (WRF-Chem's Registry spelling,
``registry.chem``), with the row's ``long_name`` and ``units``; every array
a process declares with an ``output_name`` is written under it; every table
diagnostic whose terms are active (the PM sums of ``sum_pm_gocart``, the
near-surface and column sums) is evaluated at the frame by
:mod:`gpuwm.core.chem_outputs`; nothing here names a species.

The diagnostics depend only on the fields at the frame, so evaluating them
when the frame is written gives the words WRF's every-step ``sum_pm_gocart``
would have left in place at that time, without paying for them every step.
"""

from __future__ import annotations

import numpy as np

__all__ = ["history_fields", "history_schema", "dz8w"]


def _process_output_rows(table, module):
    """Declared history names and source slices, shared with disk pricing."""
    from gpuwm.core.chem_state import process_attr

    acted = tuple(module.rows(table))
    for alloc in getattr(module, "ALLOCATES", ()):
        if alloc.output_name is None:
            continue
        description = alloc.description or alloc.name
        if alloc.shape.startswith("rows_"):
            for i, row in enumerate(acted):
                if "{" in alloc.output_name:
                    name = alloc.output_name.format(n=i + 1, row=row.name)
                elif len(acted) == 1:
                    name = alloc.output_name
                else:
                    name = f"{alloc.output_name}_{row.name.upper()}"
                yield name, process_attr(alloc), i, alloc, description, len(acted)
        else:
            yield (alloc.output_name, process_attr(alloc), None, alloc,
                   description, len(acted))


def history_schema(cfg) -> dict:
    """Output shape, description and units without arrays or a device import."""
    from gpuwm.chem_table import load
    from gpuwm.core.chem_context import allocation_shape, chem_processes
    from gpuwm.core.chem_outputs import compile_diagnostic

    table = load(cfg)
    if table is None:
        return {}
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    mass, surface = (nz, ny, nx), (ny, nx)
    out = {row.output_name: (mass, row.long_name, row.units)
           for row in table.rows if row.output_name is not None}
    for _key, module in chem_processes(table) if table.processes else ():
        for name, _attr, index, alloc, description, nrows in _process_output_rows(
                table, module):
            shape = allocation_shape(alloc, nrows, nz, ny, nx)
            out[name] = (shape[1:] if index is not None else shape,
                         description, alloc.units)
    for diag in table.diagnostics:
        program = compile_diagnostic(diag, table)
        if program is not None:
            out[diag.output_name] = (
                mass if program.kind == "term_sum_3d" else surface,
                diag.description, diag.units)
    return out


def dz8w(state):
    """WRF chem_prep's layer thickness (chem/module_chem_utilities.F).

    ``z_at_w = (phb + ph) / g`` then ``dz8w = z_at_w(k+1) - z_at_w(k)``,
    both in float32: a division and a subtraction, which no compiler can
    contract, so the cupy expression is WRF's word for word.
    """
    from gpuwm.core import constants as c

    phb = state.phb
    if phb.ndim == 1:
        phb = phb[:, None, None]
    z_at_w = (phb + state.php) / np.float32(c.G)
    return z_at_w[1:] - z_at_w[:-1]


def history_fields(state) -> dict:
    """``{output_name: (array, long_name, units)}`` for one history frame."""
    chem = state.chem
    out: dict = {}
    for row in (*chem.transported, *chem.prescribed):
        if row.output_name is not None:
            out[row.output_name] = (getattr(state, row.state_attr),
                                    row.long_name, row.units)
    # Process-declared outputs (ChemAllocation.output_name, DESIGN section 4
    # "Output rows"): the arrays a process keeps, written generically -- a
    # per-row array once per row.  A name that is a template takes "{n}" as
    # the row's 1-based place in the process's rows (WRF's EDUST1..5 for the
    # dust bins) and "{row}" as its name; a plain name is suffixed with the
    # row's name when the process acts on more than one row.
    processes = getattr(chem, "processes", None)
    if processes is None:
        from gpuwm.core.chem_context import chem_processes
        processes = chem_processes(chem.table) if chem.table.processes else []
    for _key, module in processes:
        prepare = getattr(module, "prepare_history", None)
        if prepare is not None:
            prepare(state)
        for name, attr, index, alloc, description, _nrows in _process_output_rows(
                chem.table, module):
            array = getattr(state, attr, None)
            if array is None:
                continue
            out[name] = (array[index] if index is not None else array,
                         description, alloc.units)
    diagnostics = chem.table.diagnostics
    if diagnostics:
        import cupy as cp

        from gpuwm.core.chem_outputs import compile_diagnostic, evaluate

        thickness = None
        nz, ny, nx = state.alt.shape
        for diag in diagnostics:
            program = compile_diagnostic(diag, chem.table)
            if program is None:
                continue
            if program.kind == "column_integral" and thickness is None:
                thickness = cp.ascontiguousarray(dz8w(state))
            shape = (nz, ny, nx) if program.kind == "term_sum_3d" else (ny, nx)
            saved = getattr(chem, 'streamed_history', {})
            array = saved.get(diag.output_name)
            if array is None:
                array = cp.empty(shape, dtype=cp.float32)
            evaluate(program, state, state.alt,
                     thickness if program.kind == "column_integral" else None,
                     array)
            out[diag.output_name] = (array, diag.description, diag.units)
    return out


def streaming_fields(state):
    """Stable GPU diagnostic buffers, refreshed before a tile is scattered."""
    chem = getattr(state, 'chem', None)
    if chem is None:
        return {}
    fields = history_fields(state)
    chem.streamed_history = {name: entry[0] for name, entry in fields.items()}
    return chem.streamed_history
