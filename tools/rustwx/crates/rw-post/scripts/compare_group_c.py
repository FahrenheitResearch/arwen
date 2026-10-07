# SPDX-License-Identifier: Apache-2.0
"""Black-box comparison of the Group C planes against an exporter's GRIB2.

Analysis tool only (not a data path).  Decodes the values of the GRIB2
messages (simple packing, template 5.0, with or without a bitmap; grid
templates 3.30 Lambert, 3.0 lat/lon and 3.20 polar) -- section 2 text is
never read -- matches each Group C field by its GRIB2 identity from the
clean-room SPEC section 6, and prints per-field distances.

  python compare_group_c.py OLD_SFC.grib2 NEW_DIR TAG [--json OUT.json]
"""
import json
import struct
import sys

import numpy as np

# field -> (discipline, category, number, first surface type, first value or
# None, second surface type or None, second value or None)
IDENTITY = {
    "sbcape": (0, 7, 6, 1, None, None, None),
    "sbcin": (0, 7, 7, 1, None, None, None),
    "mlcape": (0, 7, 6, 108, 9000, 108, 0),
    "mlcin": (0, 7, 7, 108, 9000, 108, 0),
    "mucape": (0, 7, 6, 108, 30000, 108, 0),
    "mucin": (0, 7, 7, 108, 30000, 108, 0),
    "cape_best180": (0, 7, 6, 108, 18000, 108, 0),
    "cin_best180": (0, 7, 7, 108, 18000, 108, 0),
    "lcl_height": (0, 3, 5, 5, None, None, None),
    "storm_motion_u": (0, 2, 27, None, None, None, None),
    "storm_motion_v": (0, 2, 28, None, None, None, None),
    "srh_0_1km": (0, 7, 8, 103, 1000, None, None),
    "srh_0_3km": (0, 7, 8, 103, 3000, None, None),
    "shear_u_0_1km": (0, 192, 3, 103, 1000, None, None),
    "shear_v_0_1km": (0, 192, 4, 103, 1000, None, None),
    "shear_u_0_6km": (0, 192, 3, 103, 6000, None, None),
    "shear_v_0_6km": (0, 192, 4, 103, 6000, None, None),
    "bulk_shear_0_1km": (0, 192, 1, 103, 1000, None, None),
    "bulk_shear_0_6km": (0, 192, 2, 103, 6000, None, None),
    "stp": (0, 7, 211, None, None, None, None),
    "ehi_0_1km": (0, 7, 9, None, None, None, None),
}


def sm16(b):
    v = struct.unpack(">H", b)[0]
    return -(v & 0x7FFF) if v & 0x8000 else v


def sm32(b):
    v = struct.unpack(">I", b)[0]
    return -(v & 0x7FFFFFFF) if v & 0x80000000 else v


def surface(sec4, off):
    t = sec4[off]
    scale = sec4[off + 1]
    raw = sm32(sec4[off + 2:off + 6])
    if t == 255:
        return None, None
    if scale in (0, 255) or raw in (-0x7FFFFFFF,):
        return t, raw if scale != 255 else None
    s = scale - 256 if scale > 127 else scale
    return t, raw / 10.0 ** s


def messages(path):
    data = open(path, "rb").read()
    pos = 0
    while True:
        pos = data.find(b"GRIB", pos)
        if pos < 0:
            return
        disc = data[pos + 6]
        total = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
        end = pos + total
        p = pos + 16
        nx = ny = scan = None
        sec4 = sec5 = bitmap = sec7 = None
        while p < end - 4:
            length = struct.unpack(">I", data[p:p + 4])[0]
            num = data[p + 4]
            sec = data[p:p + length]
            if num == 3:
                tmpl = struct.unpack(">H", sec[12:14])[0]
                if tmpl == 30 or tmpl == 20:
                    nx, ny = struct.unpack(">II", sec[30:38])
                    scan = sec[64] if tmpl == 30 else sec[64]
                elif tmpl == 0:
                    nx, ny = struct.unpack(">II", sec[30:38])
                    scan = sec[71]
            elif num == 4:
                sec4 = sec
            elif num == 5:
                sec5 = sec
            elif num == 6:
                bitmap = sec
            elif num == 7:
                sec7 = sec
            p += length
        cat, par = sec4[9], sec4[10]
        s1 = surface(sec4, 22)
        s2 = surface(sec4, 28)
        yield {"disc": disc, "cat": cat, "num": par, "s1": s1, "s2": s2,
               "nx": nx, "ny": ny, "scan": scan, "sec5": sec5, "bitmap": bitmap, "sec7": sec7}
        pos = end


def decode(m):
    s5 = m["sec5"]
    npts = m["nx"] * m["ny"]
    tmpl = struct.unpack(">H", s5[9:11])[0]
    assert tmpl == 0, f"packing template 5.{tmpl} not handled"
    r = struct.unpack(">f", s5[11:15])[0]
    e = sm16(s5[15:17])
    d = sm16(s5[17:19])
    nbits = s5[19]
    bm = m["bitmap"]
    if bm is not None and bm[5] == 0:
        mask = np.unpackbits(np.frombuffer(bm[6:], np.uint8))[:npts].astype(bool)
    else:
        mask = np.ones(npts, bool)
    nval = int(mask.sum())
    if nbits == 0:
        vals = np.full(nval, r, np.float64)
    else:
        bits = np.unpackbits(np.frombuffer(m["sec7"][5:], np.uint8))[: nval * nbits].reshape(nval, nbits)
        weights = (1 << np.arange(nbits - 1, -1, -1, dtype=np.uint64))
        x = (bits.astype(np.uint64) * weights).sum(axis=1).astype(np.float64)
        vals = (r + x * 2.0 ** e) / 10.0 ** d
    out = np.full(npts, np.nan)
    out[mask] = vals
    out = out.reshape(m["ny"], m["nx"])
    if m["scan"] is not None and not (m["scan"] & 0x40):
        out = out[::-1]
    return out


def matches(m, ident):
    disc, cat, num, t1, v1, t2, v2 = ident
    if (m["disc"], m["cat"], m["num"]) != (disc, cat, num):
        return False
    if t1 is not None and (m["s1"][0] != t1 or (v1 is not None and m["s1"][1] != v1)):
        return False
    if t2 is not None and (m["s2"][0] != t2 or (v2 is not None and m["s2"][1] != v2)):
        return False
    return True


def main():
    old_path, new_dir, tag = sys.argv[1:4]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    msgs = list(messages(old_path))
    inventory = sorted({(m["disc"], m["cat"], m["num"], m["s1"], m["s2"]) for m in msgs if m["cat"] in (2, 3, 7, 192) and m["disc"] == 0})
    print("inventory (disc, cat, num, surface1, surface2):")
    for row in inventory:
        print("  ", row)
    rows = {}
    for field, ident in IDENTITY.items():
        cand = [m for m in msgs if matches(m, ident)]
        if len(cand) != 1:
            rows[field] = {"status": f"{len(cand)} matching messages"}
            continue
        old = decode(cand[0]).ravel()
        new = np.fromfile(f"{new_dir}/{field}.{tag}.f32", "<f4").astype(np.float64)
        if new.size != old.size:
            rows[field] = {"status": f"size {new.size} vs {old.size}"}
            continue
        both = np.isfinite(old) & np.isfinite(new)
        d = new[both] - old[both]
        o = old[both]
        n = new[both]
        rows[field] = {
            "cells": int(both.sum()),
            "missing_old_only": int((~np.isfinite(old) & np.isfinite(new)).sum()),
            "missing_new_only": int((np.isfinite(old) & ~np.isfinite(new)).sum()),
            "old_mean": float(o.mean()), "new_mean": float(n.mean()),
            "old_p99_abs": float(np.percentile(np.abs(o), 99)),
            "bias": float(d.mean()),
            "rms": float(np.sqrt((d ** 2).mean())),
            "max_abs": float(np.abs(d).max()),
            "p99_abs": float(np.percentile(np.abs(d), 99)),
            "corr": float(np.corrcoef(o, n)[0, 1]) if o.std() > 0 and n.std() > 0 else None,
            "zero_old_nonzero_new": int(((o == 0) & (n != 0)).sum()),
            "nonzero_old_zero_new": int(((o != 0) & (n == 0)).sum()),
        }
    hdr = f"{'field':18s} {'cells':>7s} {'old_mean':>9s} {'new_mean':>9s} {'bias':>9s} {'rms':>9s} {'p99|d|':>9s} {'max|d|':>9s} {'corr':>6s}"
    print(hdr)
    for f, r in rows.items():
        if "status" in r:
            print(f"{f:18s} {r['status']}")
            continue
        c = r["corr"]
        print(f"{f:18s} {r['cells']:7d} {r['old_mean']:9.3f} {r['new_mean']:9.3f} {r['bias']:9.3f} {r['rms']:9.3f} "
              f"{r['p99_abs']:9.3f} {r['max_abs']:9.3f} {c if c is None else round(c, 4)!s:>6s}")
    if out_json:
        json.dump(rows, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
