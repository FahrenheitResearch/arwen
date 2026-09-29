"""Build the page's offline basemap from Natural Earth 1:50m shapefiles.

``gpuwm gui`` draws coastlines, country borders, state and province lines
and the large lakes under the Create map, with no network at run time.
This script turns the Natural Earth 1:50m layers into one compact JSON
file the page reads: each line simplified (Douglas-Peucker), rounded to
0.01 degree and delta-coded as integers.

    python tools/build_gui_basemap.py NATURAL_EARTH_DIR [OUT]

NATURAL_EARTH_DIR holds the ``ne_50m_*.shp`` files, in any layout below
it (a cartopy cache works).  OUT defaults to
``gpuwm/gui/static/map/basemap.json``.  Natural Earth is public domain
(https://www.naturalearthdata.com/); the file carries that notice.
Standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

LAYERS = {
    "coast": ("ne_50m_coastline.shp", 0.01, 0.0),
    "lake": ("ne_50m_lakes.shp", 0.01, 0.15),
    "border": ("ne_50m_admin_0_boundary_lines_land.shp", 0.01, 0.0),
    "state": ("ne_50m_admin_1_states_provinces_lines.shp", 0.01, 0.0),
}
SCALE = 100
NOTICE = ("Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com. "
          "Natural Earth 1:50m coastline, lakes, admin 0 boundary lines and admin 1 lines; public domain.")
OUT = Path(__file__).resolve().parents[1] / "gpuwm" / "gui" / "static" / "map" / "basemap.json"


def read_lines(path: Path) -> list[list[tuple[float, float]]]:
    """Every part of every polyline or polygon record in a .shp file."""

    blob = path.read_bytes()
    lines = []
    at = 100
    while at + 8 <= len(blob):
        _, words = struct.unpack(">ii", blob[at:at + 8])
        content = blob[at + 8:at + 8 + 2 * words]
        at += 8 + 2 * words
        kind = struct.unpack("<i", content[:4])[0]
        if kind not in (3, 5, 13, 15, 23, 25):
            continue
        parts_n, points_n = struct.unpack("<ii", content[36:44])
        parts = list(struct.unpack(f"<{parts_n}i", content[44:44 + 4 * parts_n]))
        base = 44 + 4 * parts_n
        coords = struct.unpack(f"<{2 * points_n}d", content[base:base + 16 * points_n])
        points = list(zip(coords[0::2], coords[1::2]))
        for i, start in enumerate(parts):
            end = parts[i + 1] if i + 1 < parts_n else points_n
            lines.append(points[start:end])
    return lines


def simplify(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        (x1, y1), (x2, y2) = points[first], points[last]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5
        worst, index = 0.0, -1
        for i in range(first + 1, last):
            px, py = points[i]
            if norm == 0.0:
                d = ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
            else:
                d = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm
            if d > worst:
                worst, index = d, i
        if worst > epsilon and index > 0:
            keep[index] = True
            stack += [(first, index), (index, last)]
    return [p for p, k in zip(points, keep) if k]


def encode(points: list[tuple[float, float]]) -> list[int]:
    out: list[int] = []
    last = None
    for lon, lat in points:
        q = (round(lon * SCALE), round(lat * SCALE))
        if q == last:
            continue
        if last is None:
            out += [q[0], q[1]]
        else:
            out += [q[0] - last[0], q[1] - last[1]]
        last = q
    return out if len(out) >= 4 else []


def find(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise SystemExit(f"{name} is not under {root}")
    return hits[0]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    root = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else OUT
    layers = {}
    for key, (name, epsilon, min_extent) in LAYERS.items():
        lines = []
        for points in read_lines(find(root, name)):
            if min_extent:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                if max(max(xs) - min(xs), max(ys) - min(ys)) < min_extent:
                    continue
            code = encode(simplify(points, epsilon))
            if code:
                lines.append(code)
        layers[key] = lines
    document = {"notice": NOTICE, "scale": SCALE, "layers": layers}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    counts = {key: sum(len(line) // 2 for line in lines) for key, lines in layers.items()}
    print(f"{out}: {out.stat().st_size} bytes, points {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
