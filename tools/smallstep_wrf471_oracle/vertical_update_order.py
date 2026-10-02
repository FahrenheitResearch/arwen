"""Controlled witness for WRF's state addition and mapped surface ordering."""
from __future__ import annotations


def wrf_update_order_source(source: str) -> str:
    marker = "w_pp[f] += dtau * rw_t[f]"
    if source.count(marker) != 6:
        raise ValueError("native vertical state additions changed")
    source = source.replace(marker, "w_pp[f] = w_pp[f] + dtau * rw_t[f]")
    native = """w_pp[c] = msf_c * (0.5f * rdy * ((ht[cjp] - ht[c]) * vn
                                         + (ht[c] - ht[cjm]) * vs)
                           + 0.5f * rdx * ((ht[cip] - ht[c]) * ue
                                           + (ht[c] - ht[cim]) * uw));"""
    reference = """w_pp[c] = msf_c * 0.5f * rdy * ((ht[cjp] - ht[c]) * vn
                                         + (ht[c] - ht[cjm]) * vs)
                           + msf_c * 0.5f * rdx * ((ht[cip] - ht[c]) * ue
                                           + (ht[c] - ht[cim]) * uw);"""
    if source.count(native) != 1:
        raise ValueError("native mapped surface multiplication changed")
    return source.replace(native, reference)
