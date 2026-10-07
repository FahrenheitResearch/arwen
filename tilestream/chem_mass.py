"""The resident chemistry mass reduction over mapped domain-store bytes."""
from __future__ import annotations

from types import SimpleNamespace


def global_masses(store, field_keys, mu, cfg, geography, template, *, rows=None):
    """Run the unchanged whole-NX native reduction, preserving its word order.

    ``mu`` is a store key or an explicit pinned old-time mu snapshot. Only
    vertical metadata and the small row-partial array visit device memory;
    the full tracer, dry column mass and metric fields are mapped native
    host allocations. Never sum already reduced tile totals here.
    """
    import cupy as cp
    from gpuwm.core.chem_driver import _masses
    from tilestream.sfire_spotting import mapped_host_array

    def source(name):
        key = "setup/" + name
        value = geography.get(key)
        if value is None:
            value = getattr(template, name)
        return mapped_host_array(value)

    state = SimpleNamespace(p=SimpleNamespace(shape=(int(cfg.nz), int(cfg.ny), int(cfg.nx))),
        mub2d=source("mub2d"), msft=source("msft"), has_msf=bool(template.has_msf),
        c1h=cp.ascontiguousarray(cp.asarray(template.c1h)),
        c2h=cp.ascontiguousarray(cp.asarray(template.c2h)),
        dnw=cp.ascontiguousarray(cp.asarray(template.dnw)))
    fields = tuple(mapped_host_array(store[key]) for key in field_keys)
    column = mapped_host_array(store[mu] if isinstance(mu, str) else mu)
    return _masses(state, cfg, tuple(template.chem.transported) if rows is None else rows,
                   fields, column)
