#!/usr/bin/env python3
"""Generate src/panels/world_map_data.rs, the locator insets' embedded basemap.

Source: Natural Earth (public domain) shapefiles already vendored at
tools/rustwx/assets/basemap/ (10m coastline, land boundaries, lakes,
admin-1 lines; 110m countries only to pick which admin-1 lines to keep).

Layers written (each Douglas-Peucker simplified, then quantised to
0.01 degree i16 pairs):
  COAST    world coastline
  BORDERS  world country boundaries (land)
  LAKES    lake shorelines whose ring spans at least LAKE_MIN_SPAN degrees
  STATES   first-order admin lines of STATE_COUNTRIES (the CONUS state lines
           the inset always drew, plus Alaska, Canada and Mexico)

Usage: python gen_world_map_data.py <rustwx root> <out .rs>
Stdlib + numpy only.
"""
import struct
import sys
from pathlib import Path

import numpy as np

TOL = 0.03            # degrees; the insets show ~0.05 deg per pixel at their tightest
MIN_SPAN = 0.08       # drop rings/lines smaller than this (under ~1.5 px)
LAKE_MIN_SPAN = 1.5   # keep only lakes whose ring spans this many degrees
STATE_COUNTRIES = ("USA", "CAN", "MEX")


def read_shp(path):
    data = Path(path).read_bytes()
    records = []
    off = 100
    while off < len(data):
        _num, clen = struct.unpack(">ii", data[off:off + 8])
        rec = data[off + 8: off + 8 + clen * 2]
        off += 8 + clen * 2
        stype = struct.unpack("<i", rec[:4])[0]
        if stype not in (3, 5):
            records.append([])
            continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        starts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts]))
        base = 44 + 4 * nparts
        pts = np.frombuffer(rec[base: base + 16 * npts], dtype="<f8").reshape(-1, 2)
        starts.append(npts)
        records.append([pts[a:b] for a, b in zip(starts[:-1], starts[1:])])
    return records


def read_dbf(path, field):
    data = Path(path).read_bytes()
    nrec, hlen, rlen = struct.unpack("<IHH", data[4:12])
    fields = []
    pos = 32
    while data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b"\x00")[0].decode()
        flen = data[pos + 16]
        fields.append((name, flen))
        pos += 32
    out = []
    for i in range(nrec):
        r = data[hlen + i * rlen + 1: hlen + (i + 1) * rlen]
        p = 0
        for name, flen in fields:
            if name == field:
                out.append(r[p:p + flen].decode("latin-1").strip())
            p += flen
    return out


def dp(pts, tol):
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[a + 1:b]
        p0, p1 = pts[a], pts[b]
        d = p1 - p0
        length = np.hypot(d[0], d[1])
        if length == 0:
            dist = np.hypot(seg[:, 0] - p0[0], seg[:, 1] - p0[1])
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p0[1]) - d[1] * (seg[:, 0] - p0[0])) / length
        i = int(np.argmax(dist))
        if dist[i] > tol:
            k = a + 1 + i
            keep[k] = True
            stack.append((a, k))
            stack.append((k, b))
    return pts[keep]


def span(p):
    return max(p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min())


def simplify_layer(parts, tol, min_span):
    out = []
    for p in parts:
        if len(p) < 2 or span(p) < min_span:
            continue
        q = np.round(dp(p, tol) * 100).astype(np.int32)
        keep = np.ones(len(q), bool)
        keep[1:] = np.any(q[1:] != q[:-1], axis=1)
        q = q[keep]
        if len(q) >= 2:
            out.append(q)
    return out


def point_in_rings(x, y, rings):
    inside = np.zeros(len(x), bool)
    for r in rings:
        xr, yr = r[:, 0], r[:, 1]
        xj, yj = np.roll(xr, 1), np.roll(yr, 1)
        for xi_, yi_, xj_, yj_ in zip(xr, yr, xj, yj):
            if yj_ == yi_:
                continue
            cond = ((yi_ > y) != (yj_ > y)) & (x < (xj_ - xi_) * (y - yi_) / (yj_ - yi_) + xi_)
            inside ^= cond
    return inside


def main():
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    b10 = root / "assets/basemap/natural_earth_10m"
    b110 = root / "assets/basemap/natural_earth_110m"

    coast = [p for rec in read_shp(b10 / "ne_10m_coastline.shp") for p in rec]
    borders = [p for rec in read_shp(b10 / "ne_10m_admin_0_boundary_lines_land.shp") for p in rec]
    lakes = [p for rec in read_shp(b10 / "ne_10m_lakes.shp") for p in rec if span(p) >= LAKE_MIN_SPAN]

    names = read_dbf(b110 / "ne_110m_admin_0_countries.dbf", "ADM0_A3")
    shapes = read_shp(b110 / "ne_110m_admin_0_countries.shp")
    country_rings = [r for n, rec in zip(names, shapes) if n in STATE_COUNTRIES for r in rec]
    states = []
    for rec in read_shp(b10 / "ne_10m_admin_1_states_provinces_lines.shp"):
        for p in rec:
            # Coarse prefilter, then a majority vote of points inside the
            # 110m outline of a kept country.
            if p[:, 0].max() < -180 or p[:, 0].min() > -50 or p[:, 1].max() < 14:
                continue
            sample = p[:: max(1, len(p) // 25)]
            if point_in_rings(sample[:, 0], sample[:, 1], country_rings).mean() >= 0.5:
                states.append(p)

    layers = {
        "COAST": simplify_layer(coast, TOL, MIN_SPAN),
        "BORDERS": simplify_layer(borders, TOL, MIN_SPAN),
        "LAKES": simplify_layer(lakes, TOL, MIN_SPAN),
        "STATES": simplify_layer(states, TOL, MIN_SPAN),
    }

    lines = [
        "//! Embedded locator-inset basemap (GENERATED, do not edit by hand).",
        "//!",
        "//! Source: Natural Earth 1:10m coastline, land country boundaries, lakes",
        "//! and admin-1 lines (public domain), from tools/rustwx/assets/basemap.",
        f"//! Douglas-Peucker tolerance {TOL} deg, features under {MIN_SPAN} deg dropped,",
        f"//! lakes spanning at least {LAKE_MIN_SPAN} deg, admin-1 lines of {', '.join(STATE_COUNTRIES)}.",
        "//! Regenerate (from tools/rustwx):",
        "//!   python vendor/sharppyrs/tools/gen_world_map_data.py . \\",
        "//!     vendor/sharppyrs/src/panels/world_map_data.rs",
        "//!",
        "//! Each layer is `_STARTS`, `_BBOX`, `_XY`: `_XY` holds (lon, lat) pairs in",
        "//! hundredths of a degree; polyline `i` is points `starts[i]..starts[i+1]`;",
        "//! `bbox[4*i..4*i+4]` is its (west, south, east, north) in the same units.",
        "",
    ]
    total = 0
    for name, segs in layers.items():
        starts = [0]
        for s in segs:
            starts.append(starts[-1] + len(s))
        bbox = []
        for s in segs:
            bbox += [int(s[:, 0].min()), int(s[:, 1].min()), int(s[:, 0].max()), int(s[:, 1].max())]
        xy = np.concatenate(segs).ravel().tolist() if segs else []
        total += len(xy) // 2
        for label, ty, vals in (("STARTS", "u32", starts), ("BBOX", "i16", bbox), ("XY", "i16", xy)):
            lines.append(f"pub(super) static {name}_{label}: [{ty}; {len(vals)}] = [")
            for i in range(0, len(vals), 16):
                lines.append("    " + ", ".join(str(v) for v in vals[i:i + 16]) + ",")
            lines.append("];")
        lines.append("")
        print(f"{name}: {len(segs)} lines, {len(xy) // 2} points", file=sys.stderr)
    print(f"total points {total}", file=sys.stderr)
    out_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
