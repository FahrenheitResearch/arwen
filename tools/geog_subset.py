"""Copy only the WPS_GEOG tiles a lat/lon box needs.

A full WPS_GEOG tree is 29 GiB, almost all of it in ``greenfrac_fpar_modis``
and ``lai_modis_30s``.  Every dataset is stored as fixed-size tiles named
``xstart-xend.ystart-yend`` in its own regular lat/lon frame, described by its
own ``index`` file (``known_x``/``known_y``/``known_lat``/``known_lon``/
``dx``/``dy``/``tile_x``/``tile_y``/``tile_bdr``).  This reads each index,
converts the requested box to that dataset's own index space, and copies the
intersecting tiles plus the index verbatim.

    python -m tools.geog_subset SRC DST --north 51 --south 17 \
        --west -112 --east -64

Verbatim: tiles are byte copies, so the built statics are identical to the
ones the full tree would produce for any domain inside the box.  A domain
outside it fails LOUDLY -- ``static.build`` raises on missing source
coverage rather than filling.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

_TILE_RE = re.compile(r"^(\d+)-(\d+)\.(\d+)-(\d+)$")


def read_index(path: Path) -> dict:
    out: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip().lower()] = value.strip().strip('"').strip("'")
    return out


def index_ranges(idx: dict, north: float, south: float,
                 west: float, east: float) -> tuple[int, int, int, int]:
    """Inclusive ``(x0, x1, y0, y1)`` 1-based index range covering the box."""
    dx = float(idx["dx"])
    dy = float(idx["dy"])
    known_x = float(idx.get("known_x", 1.0))
    known_y = float(idx.get("known_y", 1.0))
    known_lon = float(idx["known_lon"])
    known_lat = float(idx["known_lat"])

    def x_of(lon: float) -> float:
        return known_x + (lon - known_lon) / dx

    def y_of(lat: float) -> float:
        return known_y + (lat - known_lat) / dy

    xs = [x_of(west), x_of(east)]
    # A dataset whose frame starts at lon 0 (topo_gmted2010_30s) wraps: the
    # requested western hemisphere sits at the FAR end of its x axis.
    span = round(360.0 / dx)
    xs = [x + span if x < 1 else x for x in xs]
    ys = [y_of(south), y_of(north)]
    return (int(min(xs)) - 1, int(max(xs)) + 1,
            int(min(ys)) - 1, int(max(ys)) + 1)


def wanted_tiles(names, x0, x1, y0, y1) -> list[str]:
    keep = []
    for name in names:
        m = _TILE_RE.match(name)
        if not m:
            continue
        ax0, ax1, ay0, ay1 = (int(g) for g in m.groups())
        if ax1 < x0 or ax0 > x1 or ay1 < y0 or ay0 > y1:
            continue
        keep.append(name)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--north", type=float, required=True)
    ap.add_argument("--south", type=float, required=True)
    ap.add_argument("--west", type=float, required=True)
    ap.add_argument("--east", type=float, required=True)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    for ds in sorted(p for p in src.iterdir() if p.is_dir()):
        index = ds / "index"
        if not index.is_file():
            continue
        idx = read_index(index)
        names = [p.name for p in ds.iterdir() if p.is_file()]
        x0, x1, y0, y1 = index_ranges(idx, args.north, args.south,
                                      args.west, args.east)
        keep = wanted_tiles(names, x0, x1, y0, y1)
        outdir = dst / ds.name
        outdir.mkdir(exist_ok=True)
        for extra in ("index",):
            shutil.copy2(ds / extra, outdir / extra)
        nbytes = 0
        for name in keep:
            target = outdir / name
            if not target.exists():
                shutil.copy2(ds / name, target)
            nbytes += target.stat().st_size
        total += nbytes
        print(f"{ds.name:40s} x[{x0},{x1}] y[{y0},{y1}]  "
              f"{len(keep):4d}/{len(names):4d} tiles  {nbytes/2**20:8.1f} MiB",
              flush=True)
    print(f"TOTAL {total / 2**30:.2f} GiB -> {dst}")


if __name__ == "__main__":
    main()
