"""Build the ``gpuwm gui`` map's basemap from the renderer's vendored shapefiles.

The page draws its own map (coastlines, borders, states, lakes, land and
US counties) from one compact binary file served next to it, so the map
works with nothing fetched from outside this computer.  The geometry is
the Natural Earth 110m and 10m layers and the Census county file the
renderer already carries under ``tools/rustwx/assets/basemap`` (public
domain and US Government work; see NOTICE), simplified and quantized.

    python tools/gui_basemap.py [--out gpuwm/gui/static/map/basemap.bin]

Needs ``pyshp`` (build time only; the page and the server do not). The
line simplification is Douglas-Peucker written here in numpy with the
same arithmetic as GEOS, so the builder needs no geometry library and
reproduces the committed file byte for byte.

File layout, little-endian::

    b"AWBM" u32 version u32 header_bytes  header(JSON, space padded to 4)
    per layer, at header.layers[i].offset bytes after the header:
        u32[parts]          points per part
        i32[2 * parts]      first point of each part (lon, lat) / quantum
        i16[2 * (points - parts)]  deltas from the previous point

A delta that does not fit in 16 bits splits the part, so every value is
exact at its quantum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np
import shapefile

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "tools" / "rustwx" / "assets" / "basemap"
OUT = ROOT / "gpuwm" / "gui" / "static" / "map" / "basemap.bin"
VERSION = 1

#: (name, file, kind, level, tolerance in degrees, quantum in degrees, minimum ring area in deg^2)
#: ``level`` 0 is drawn when zoomed out, 1 when zoomed in, 2 (counties) closer still.
LAYERS = (
    ("land", "natural_earth_110m/ne_110m_admin_0_countries.shp", "poly", 0, 0.0, 0.01, 0.0),
    ("coast", "natural_earth_110m/ne_110m_coastline.shp", "line", 0, 0.0, 0.01, 0.0),
    ("borders", "natural_earth_110m/ne_110m_admin_0_boundary_lines_land.shp", "line", 0, 0.0, 0.01, 0.0),
    ("states", "natural_earth_110m/ne_110m_admin_1_states_provinces_lines.shp", "line", 0, 0.0, 0.01, 0.0),
    # Zoomed in, the coastline is the outline of the land and lake rings,
    # so the 10m coastline file itself is not carried.
    ("land", "natural_earth_10m/ne_10m_land.shp", "poly", 1, 0.008, 0.002, 0.001),
    ("lakes", "natural_earth_10m/ne_10m_lakes.shp", "poly", 1, 0.008, 0.002, 0.004),
    ("borders", "natural_earth_10m/ne_10m_admin_0_boundary_lines_land.shp", "line", 1, 0.006, 0.002, 0.0),
    ("states", "natural_earth_10m/ne_10m_admin_1_states_provinces_lines.shp", "line", 1, 0.01, 0.002, 0.0),
    ("counties", "us_counties_5m/cb_2023_us_county_5m.shp", "line", 2, 0.004, 0.002, 0.0),
)


def _parts(shape) -> list[np.ndarray]:
    points = np.asarray(shape.points, dtype=np.float64)
    if not len(points):
        return []
    bounds = list(shape.parts) + [len(points)]
    return [points[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]


def _ring_area(part: np.ndarray) -> float:
    """Unsigned area of a ring, summed in the order GEOS ``Area::ofRing`` sums it."""

    if not np.array_equal(part[0], part[-1]):
        part = np.vstack([part, part[:1]])
    x = part[1:-1, 0] - part[0, 0]
    terms = x * (part[:-2, 1] - part[2:, 1])
    return abs(float(np.cumsum(terms)[-1]) / 2.0) if len(terms) else 0.0


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to segment ``a``-``b``, as GEOS ``pointToSegment`` computes it."""

    px, py = points[:, 0], points[:, 1]
    to_a = np.sqrt((px - a[0]) * (px - a[0]) + (py - a[1]) * (py - a[1]))
    if a[0] == b[0] and a[1] == b[1]:
        return to_a
    dx, dy = b[0] - a[0], b[1] - a[1]
    len2 = dx * dx + dy * dy
    r = ((px - a[0]) * dx + (py - a[1]) * dy) / len2
    s = ((a[1] - py) * dx - (a[0] - px) * dy) / len2
    to_b = np.sqrt((px - b[0]) * (px - b[0]) + (py - b[1]) * (py - b[1]))
    return np.where(r <= 0.0, to_a, np.where(r >= 1.0, to_b, np.abs(s) * np.sqrt(len2)))


def _douglas_peucker(part: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker with both endpoints kept, the rule GEOS applies to a LineString."""

    keep = np.ones(len(part), dtype=bool)
    stack = [(0, len(part) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        distance = _segment_distance(part[i + 1:j], part[i], part[j])
        k = int(np.argmax(distance))  # the first maximum, as GEOS's strict `>` scan keeps
        if distance[k] <= tolerance:
            keep[i + 1:j] = False
        else:
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return part[keep]


def _simplify(part: np.ndarray, kind: str, tolerance: float, min_area: float) -> np.ndarray | None:
    if kind == "poly":
        if len(part) < 4:
            return None
        if min_area and _ring_area(part) < min_area:
            return None
        if tolerance:
            # Simplified as a closed line, not a polygon: a polygon that turns invalid on the way
            # (a continent does) would come back as a multipolygon and be lost.
            part = _douglas_peucker(part, tolerance)
        return part if len(part) >= 4 else None
    if tolerance:
        part = _douglas_peucker(part, tolerance)
    return part if len(part) >= 2 else None


def _encode(parts: list[np.ndarray], quantum: float) -> tuple[bytes, int, int]:
    counts: list[int] = []
    starts: list[int] = []
    deltas: list[np.ndarray] = []
    for part in parts:
        q = np.round(part / quantum).astype(np.int64)
        keep = np.ones(len(q), dtype=bool)
        keep[1:] = np.any(q[1:] != q[:-1], axis=1)
        q = q[keep]
        if len(q) < 2:
            continue
        d = np.diff(q, axis=0)
        big = np.where(np.any(np.abs(d) > 32767, axis=1))[0]
        cuts = [0] + [int(i) + 1 for i in big] + [len(q)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            piece = q[a:b]
            if len(piece) < 2:
                continue
            counts.append(len(piece))
            starts.extend((int(piece[0, 0]), int(piece[0, 1])))
            deltas.append(np.diff(piece, axis=0).astype(np.int16).reshape(-1))
    blob = np.asarray(counts, dtype="<u4").tobytes() + np.asarray(starts, dtype="<i4").tobytes()
    if deltas:
        blob += np.concatenate(deltas).astype("<i2").tobytes()
    blob += b"\0" * (-len(blob) % 4)
    return blob, len(counts), sum(counts)


def build(out: Path = OUT) -> dict:
    blobs: list[bytes] = []
    meta = []
    for name, relative, kind, level, tolerance, quantum, min_area in LAYERS:
        reader = shapefile.Reader(str(ASSETS / relative))
        parts: list[np.ndarray] = []
        for shape in reader.iterShapes():
            for part in _parts(shape):  # every ring of a polygon: outer rings and holes, filled even-odd
                simple = _simplify(part, kind, tolerance, min_area)
                if simple is not None:
                    parts.append(simple)
        blob, count, points = _encode(parts, quantum)
        meta.append({"name": name, "kind": kind, "level": level, "quantum": quantum,
                     "parts": count, "points": points, "bytes": len(blob)})
        blobs.append(blob)
    header = {"schema": "arwen.gui-basemap.v1",
              "source": "Natural Earth 110m and 10m (public domain); US Census cb_2023_us_county_5m",
              "layers": meta}
    offset = 0
    for item, blob in zip(meta, blobs):
        item["offset"] = offset
        offset += len(blob)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    body = b"".join(blobs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"AWBM" + struct.pack("<II", VERSION, len(text)) + text + body)
    return header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    header = build(args.out)
    for layer in header["layers"]:
        print(f"{layer['name']:9s} level {layer['level']} {layer['parts']:6d} parts "
              f"{layer['points']:8d} points {layer['bytes'] / 1024:8.1f} KiB")
    print(f"{args.out}: {args.out.stat().st_size / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
