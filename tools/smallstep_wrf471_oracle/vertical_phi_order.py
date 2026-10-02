"""Controlled witness for WRF's geopotential subtraction round points.

WRF forms ph_1(k+1)-ph_1(k)+phb(k+1)-phb(k).  The native kernel forms
(ph_1(k+1)+phb(k+1))-(ph_1(k)+phb(k)).  This diagnostic replaces only
those two equivalent expressions; it does not alter production sources.
"""
from __future__ import annotations


def wrf_phi_order_source(source: str) -> str:
    lo = "* rdnw[0] * (ph_hi - ph_lo);"
    hi = "* rdnw[k] * (ph_hi - ph_lo);"
    if source.count(lo) != 2 or source.count(hi) != 2:
        raise ValueError("native geopotential differences changed")
    source = source.replace(lo,
        "* rdnw[0] * (php[st + c] - php[c] + phb[bstr + boff] - phb[boff]);")
    source = source.replace(hi,
        "* rdnw[k] * (php[(size_t)(k + 1) * st + c] - php[(size_t)k * st + c]"
        " + phb[(size_t)(k + 1) * bstr + boff] - phb[(size_t)k * bstr + boff]);")
    return source
