"""Controlled witness for WRF's open-top pressure multiplication order."""
from __future__ import annotations


def wrf_top_order_source(source: str) -> str:
    flat = """w_pp[f] += dtau * rw_t[f]
                 - dtau * G * c2a[h] * rdnw[nz - 1] * rdnw[nz - 1]
                   / (c1h[nz - 1] * mut + c2h[nz - 1]) * dph_dn
                 - dtau * G * (2.0f * rdnw[nz - 1] * c2a[h] * alt[h]
                               * t2_dn + c1f[nz] * muave);"""
    flat_wrf = """w_pp[f] = w_pp[f] + dtau * rw_t[f]
                 + (-0.5f * dtau * G / (c1h[nz - 1] * mut + c2h[nz - 1])
                    * (rdnw[nz - 1] * rdnw[nz - 1]) * 2.0f * c2a[h] * dph_dn
                    - dtau * G * (2.0f * rdnw[nz - 1] * c2a[h] * alt[h]
                                  * t2_dn + c1f[nz] * muave));"""
    mapped = """w_pp[f] += dtau * rw_t[f]
                 + msf_i
                   * (-dtau * G * c2a[h] * rdnw[nz - 1] * rdnw[nz - 1]
                      / (c1h[nz - 1] * mut + c2h[nz - 1]) * dph_dn
                      - dtau * G
                        * (2.0f * rdnw[nz - 1] * c2a[h] * alt[h] * t2_dn
                           + c1f[nz] * muave));"""
    mapped_wrf = """w_pp[f] = w_pp[f] + dtau * rw_t[f]
                 + msf_i
                   * (-0.5f * dtau * G / (c1h[nz - 1] * mut + c2h[nz - 1])
                      * (rdnw[nz - 1] * rdnw[nz - 1]) * 2.0f * c2a[h] * dph_dn
                      - dtau * G
                        * (2.0f * rdnw[nz - 1] * c2a[h] * alt[h] * t2_dn
                           + c1f[nz] * muave));"""
    for native, replacement in ((flat, flat_wrf), (mapped, mapped_wrf)):
        if source.count(native) != 1:
            native = native.replace("w_pp[f] +=", "w_pp[f] = w_pp[f] +")
            if source.count(native) != 1:
                raise ValueError("native open-top pressure grouping changed")
        source = source.replace(native, replacement)
    return source
