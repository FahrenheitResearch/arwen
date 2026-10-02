"""Controlled witness for WRF's reciprocal then inverse-density multiply."""
from __future__ import annotations


def wrf_eos_order_source(source: str) -> str:
    native = """real al = -(alt[tid] * (c1h[k] * mu_pp[c])
                    + rdnw[k] * (ph_pp[tid + st] - ph_pp[tid])) / chm;"""
    reference = """real al = (-1.0f / chm) * (alt[tid] * (c1h[k] * mu_pp[c])
                    + rdnw[k] * (ph_pp[tid + st] - ph_pp[tid]));"""
    if source.count(native) != 1:
        raise ValueError("native inverse-density division changed")
    return source.replace(native, reference)
