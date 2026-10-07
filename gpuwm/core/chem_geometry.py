"""The mass-grid geometry a chem domain's processes read (DESIGN 2.5).

WRF hands chem_driver ``xlat``/``xlong`` with every call; gpuwm's physics
driver receives the mass-grid latitude and longitude only as radiation's
inputs and keeps no copy.  On a chem domain :func:`export_chem_geometry`
puts them in ``PhysicsDriver.fields`` as ``xlat``/``xlong`` (degrees,
float32 (ny, nx), WRF's names), where ``ChemContext.met`` finds them.  A
chem-off domain never calls it, so its field set is unchanged.
"""
from __future__ import annotations

import numpy as np

__all__ = ["export_chem_geometry"]


def export_chem_geometry(fields, latitude, longitude, *, state=None,
                         start_time=None) -> None:
    """Add ``xlat``/``xlong`` to a physics field dict when both are known.

    A run whose driver was built without them (no radiation latitude, as a
    radiation-free idealized case) gets no geometry, and a process that
    needs it refuses by name through ``ChemContext.met``'s ``KeyError``.

    ``start_time`` (the run's UTC start, which neither RunConfig nor the
    DomainState carries) is kept on the domain's ChemState as
    ``start_time``: WRF chem_driver's ``gmt``/``julday`` and every emission
    source's clock count from it.
    """
    if state is not None and start_time is not None:
        chem = getattr(state, "chem", None)
        if chem is not None:
            chem.start_time = start_time
    if latitude is None or longitude is None:
        return
    import cupy as cp

    for name, value in (("xlat", latitude), ("xlong", longitude)):
        host = (cp.asnumpy(value) if hasattr(value, "__cuda_array_interface__")
                else np.asarray(value))
        fields[name] = cp.asarray(host.astype(np.float32))
