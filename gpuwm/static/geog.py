"""WPS_GEOG dataset readers (geogrid's source-data layer, CPU only).

Each WPS_GEOG dataset directory is self-describing: an ASCII ``index`` file
gives the storage layout (wordsize / signedness / endianness / tile
dimensions / border width / z levels), the value semantics (scale factor,
missing value, category range) and the regular lat/lon georeferencing
(``dx``/``dy``/``known_x``/``known_y``/``known_lat``/``known_lon``); the
data itself is a set of flat binary tiles named
``XSTART-XEND.YSTART-YEND`` (1-based, inclusive, x fastest within a row,
row 1 first, big-endian by default -- the write_geogrid.c convention).

Conventions implemented here (arbitrated against the bundle's geo_em files
where the format leaves room):

- ``missing_value`` is compared against the *raw* (unscaled) integers, then
  ``scale_factor`` is applied (e.g. greenfrac: raw byte 200 = missing,
  raw 0..100 scaled by 0.01).
- ``tile_bdr`` border cells are duplicated halo, never authoritative; the
  reader crops them and mosaics interior cells only.
- The index remains authoritative for ``tile_z`` when a tile contains extra
  complete, all-zero trailing planes.  Some official low-resolution WPS_GEOG
  soil tiles carry this padding; nonzero or partial surplus data is rejected.
- Negative ``dy`` (albedo-style north-to-south grids) is pure
  georeferencing: tile storage order is unchanged, only latlon<->xy flips.
- A dataset whose tiles span 360 degrees of longitude wraps in x.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TILE_RE = re.compile(r"^(\d{1,6})-(\d{1,6})\.(\d{1,6})-(\d{1,6})$")

_TRUE_STRINGS = ("yes", "true", ".true.", "1")


def _raw_index(path) -> dict[str, str]:
    """Read an index file into a {lowercased key: unquoted value} dict."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] in "'\"" and val[-1] == val[0]:
            val = val[1:-1]
        out[key.strip().lower()] = val
    return out


@dataclass(frozen=True)
class GeogIndex:
    """Typed view of a WPS_GEOG ``index`` file (defaults per WPS geogrid)."""
    type: str
    projection: str
    dx: float
    dy: float
    known_x: float
    known_y: float
    known_lat: float
    known_lon: float
    wordsize: int
    tile_x: int
    tile_y: int
    tile_z: int
    tile_z_start: int
    tile_z_end: int
    tile_bdr: int = 0
    signed: bool = False
    endian: str = "big"
    scale_factor: float = 1.0
    missing_value: float | None = None
    category_min: int | None = None
    category_max: int | None = None
    units: str = ""
    description: str = ""
    mminlu: str = ""
    iswater: int | None = None
    islake: int | None = None
    isice: int | None = None
    isurban: int | None = None
    row_order: str = "bottom_top"
    interp_option: str = ""

    @property
    def nz(self) -> int:
        return self.tile_z_end - self.tile_z_start + 1

    @property
    def dtype(self) -> np.dtype:
        base = {1: "i1", 2: "i2", 4: "i4"}[self.wordsize]
        if not self.signed:
            base = "u" + base[1:]
        order = ">" if self.endian == "big" else "<"
        return np.dtype(order + base) if self.wordsize > 1 else np.dtype(base)


def parse_index(path) -> GeogIndex:
    """Parse a WPS_GEOG index file into a :class:`GeogIndex`."""
    kv = _raw_index(path)

    def _f(key, default=None):
        return float(kv[key]) if key in kv else default

    def _i(key, default=None):
        return int(float(kv[key])) if key in kv else default

    if "tile_z" in kv:
        z0, z1 = 1, _i("tile_z")
    else:
        z0, z1 = _i("tile_z_start", 1), _i("tile_z_end", 1)
    return GeogIndex(
        type=kv.get("type", "continuous").lower(),
        projection=kv.get("projection", "regular_ll").lower(),
        dx=_f("dx"), dy=_f("dy"),
        known_x=_f("known_x", 1.0), known_y=_f("known_y", 1.0),
        known_lat=_f("known_lat"), known_lon=_f("known_lon"),
        wordsize=_i("wordsize"),
        tile_x=_i("tile_x"), tile_y=_i("tile_y"),
        tile_z=z1 - z0 + 1, tile_z_start=z0, tile_z_end=z1,
        tile_bdr=_i("tile_bdr", 0),
        signed=kv.get("signed", "no").strip().lower() in _TRUE_STRINGS,
        endian=kv.get("endian", "big").strip().lower(),
        scale_factor=_f("scale_factor", 1.0),
        missing_value=_f("missing_value"),
        category_min=_i("category_min"), category_max=_i("category_max"),
        units=kv.get("units", ""), description=kv.get("description", ""),
        mminlu=kv.get("mminlu", ""),
        iswater=_i("iswater"), islake=_i("islake"),
        isice=_i("isice"), isurban=_i("isurban"),
        row_order=kv.get("row_order", "bottom_top").strip().lower(),
        interp_option=kv.get("interp_option", "").strip(),
    )


@dataclass
class GeogWindow:
    """A mosaicked window of source data in native (raw) storage.

    ``raw`` has shape ``(nz, ny, nx)``; element ``[z, j, i]`` is source cell
    ``(x0 + i, y0 + j)`` (1-based dataset coordinates, x wrap already
    resolved).  :meth:`values` applies missing-value masking (on the raw
    integers) then the scale factor, returning float64 with NaN missing.
    """
    index: GeogIndex
    x0: int
    y0: int
    raw: np.ndarray
    coverage: np.ndarray | None = None

    @property
    def x1(self) -> int:
        return self.x0 + self.raw.shape[2] - 1

    @property
    def y1(self) -> int:
        return self.y0 + self.raw.shape[1] - 1

    def values(self, z: int = 0) -> np.ndarray:
        a = self.raw[z].astype(np.float64)
        if self.index.missing_value is not None:
            a[self.raw[z] == self.index.missing_value] = np.nan
        if self.index.scale_factor != 1.0:
            a *= self.index.scale_factor
        return a


class GeogDataset:
    """One WPS_GEOG dataset directory: index + tile inventory + windowing."""

    def __init__(self, path, *, sparse: bool | None = None):
        self.path = Path(path)
        self.index = parse_index(self.path / "index")
        raw_index = _raw_index(self.path / "index")
        declared = raw_index.get("sparse", raw_index.get("is_sparse", "no"))
        self.declared_sparse = (
            declared.strip().lower() in _TRUE_STRINGS
            if sparse is None else bool(sparse)
        )
        if self.index.projection != "regular_ll":
            raise NotImplementedError(
                f"projection {self.index.projection!r} not supported")
        self.tiles: dict[tuple[int, int], Path] = {}
        self._tile_cache: dict[tuple[int, int, bool], np.ndarray | None] = {}
        xs_min = ys_min = None
        xe_max = ye_max = 0
        for f in self.path.iterdir():
            m = _TILE_RE.match(f.name)
            if not m:
                continue
            xs, xe, ys, ye = (int(g) for g in m.groups())
            self.tiles[(xs, ys)] = f
            xs_min = xs if xs_min is None else min(xs_min, xs)
            ys_min = ys if ys_min is None else min(ys_min, ys)
            xe_max, ye_max = max(xe_max, xe), max(ye_max, ye)
        if not self.tiles:
            raise FileNotFoundError(f"no data tiles found in {self.path}")
        assert xs_min is not None and ys_min is not None
        self.tile_inventory_bounds = (xs_min, xe_max, ys_min, ye_max)

        def declared_cells(*keys: str) -> int | None:
            for key in keys:
                if key in raw_index:
                    value = int(float(raw_index[key]))
                    if value <= 0:
                        raise ValueError(
                            f"WPS GEOG index {self.path / 'index'} declares "
                            f"non-positive {key}={value}")
                    return value
            return None

        def regular_ll_cells(span: float, spacing: float) -> int | None:
            if not np.isfinite(spacing) or spacing == 0.0:
                return None
            cells = int(round(span / abs(spacing)))
            if cells <= 0:
                return None
            tolerance = max(1.0e-9, span * 2.0e-6)
            if abs(abs(spacing) * cells - span) > tolerance:
                return None
            return cells

        declared_nx = declared_cells("global_nx", "nx_global")
        declared_ny = declared_cells("global_ny", "ny_global")
        inferred_nx = regular_ll_cells(360.0, self.index.dx)
        inferred_ny = regular_ll_cells(180.0, self.index.dy)
        staged_inventory = (
            self.declared_sparse or xs_min > 1 or ys_min > 1)

        if declared_nx is not None and declared_nx < xe_max:
            raise ValueError(
                f"WPS GEOG global_nx={declared_nx} is smaller than tile "
                f"extent {xe_max} in {self.path}")
        if declared_ny is not None and declared_ny < ye_max:
            raise ValueError(
                f"WPS GEOG global_ny={declared_ny} is smaller than tile "
                f"extent {ye_max} in {self.path}")

        # A footprint-minimized WPS_GEOG staging tree keeps original global
        # tile coordinates.  Inferring dimensions solely from the largest
        # staged filename would disable longitude wrapping and can turn every
        # western-hemisphere lookup into an out-of-extent zero.  Preserve the
        # regular-LL global geometry when the tile origins prove staging (or
        # when explicit global dimensions are present).  A regional tree
        # starting at tile origin 1 retains its observed extent unless it
        # explicitly opts in or already spans the complete axis.
        use_inferred_nx = (
            inferred_nx is not None and inferred_nx >= xe_max
            and (staged_inventory or inferred_nx == xe_max))
        use_inferred_ny = (
            inferred_ny is not None and inferred_ny >= ye_max
            and (staged_inventory or inferred_ny == ye_max))
        self.nx_global = (
            declared_nx if declared_nx is not None else
            inferred_nx if use_inferred_nx else xe_max)
        self.ny_global = (
            declared_ny if declared_ny is not None else
            inferred_ny if use_inferred_ny else ye_max)
        inferred_extent_expands_inventory = (
            (use_inferred_nx and (xs_min > 1 or inferred_nx > xe_max))
            or (use_inferred_ny and (ys_min > 1 or inferred_ny > ye_max)))
        self.extent_basis = (
            "declared_global" if declared_nx is not None or declared_ny is not None
            else "regular_ll_staged_inventory"
            if staged_inventory and inferred_extent_expands_inventory
            else "regular_ll_complete"
            if use_inferred_nx or use_inferred_ny
            else "tile_inventory")
        #: does the dataset span 360 deg of longitude (x wraps)?
        self.wraps_x = (
            inferred_nx is not None and self.nx_global == inferred_nx)

    # -- georeferencing ------------------------------------------------------

    def latlon_to_xy(self, lat, lon):
        """(lat, lon) degrees -> fractional 1-based source coordinates."""
        idx = self.index
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        d = lon - idx.known_lon
        if self.wraps_x:
            d = np.mod(d, 360.0)
            x = idx.known_x + d / idx.dx
            x = np.where(x >= self.nx_global + 0.5, x - self.nx_global, x)
        else:
            d = np.mod(d + 180.0, 360.0) - 180.0
            x = idx.known_x + d / idx.dx
        y = idx.known_y + (lat - idx.known_lat) / idx.dy
        return x, y

    def xy_to_latlon(self, x, y):
        """Fractional 1-based source coordinates -> (lat, lon) degrees."""
        idx = self.index
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lat = idx.known_lat + (y - idx.known_y) * idx.dy
        lon = idx.known_lon + (x - idx.known_x) * idx.dx
        lon = np.mod(lon + 180.0, 360.0) - 180.0
        return lat, lon

    # -- tile IO --------------------------------------------------------------

    def _read_tile(self, xs: int, ys: int,
                   include_border: bool = False) -> np.ndarray | None:
        """Read tile with 1-based origin (xs, ys).

        By default only the authoritative interior is returned.  With
        ``include_border``, WPS's duplicated tile halo is retained for the
        tile-local interpolation path.  Returns None for a sparse missing
        tile.
        """
        key = (xs, ys, include_border)
        if key in self._tile_cache:
            return self._tile_cache[key]
        f = self.tiles.get((xs, ys))
        if f is None:
            self._tile_cache[key] = None
            return None
        idx = self.index
        b = idx.tile_bdr
        ny, nx = idx.tile_y + 2 * b, idx.tile_x + 2 * b
        raw = np.fromfile(f, dtype=idx.dtype)
        plane_words = ny * nx
        expect = idx.nz * plane_words
        if raw.size != expect:
            if raw.size < expect or raw.size % plane_words:
                raise ValueError(f"tile {f} has {raw.size} words, expected "
                                 f"{expect} ({idx.nz}x{ny}x{nx})")
            actual_nz = raw.size // plane_words
            raw = raw.reshape(actual_nz, ny, nx)
            padding = raw[idx.nz:]
            nonzero = np.argwhere(padding != 0)
            if nonzero.size:
                z, y, x = (int(value) for value in nonzero[0])
                raise ValueError(
                    f"tile {f} has {actual_nz} complete z planes but its "
                    f"index declares {idx.nz}; undeclared trailing planes "
                    f"contain nonzero data (first at z={idx.nz + z + 1}, "
                    f"y={y + 1}, x={x + 1})")
            raw = raw[:idx.nz]
        else:
            raw = raw.reshape(idx.nz, ny, nx)
        if b and not include_border:
            raw = raw[:, b:-b, b:-b]
        if idx.row_order == "top_bottom":
            raw = raw[:, ::-1, :]
        self._tile_cache[key] = raw
        return raw

    def read_tile_window(self, xs: int, ys: int) -> GeogWindow | None:
        """Read one native tile, retaining its interpolation border.

        WPS ``get_point`` chooses a tile from the requested source
        coordinate and confines every fallback interpolator to that tile.
        The returned coordinate origin therefore includes ``tile_bdr``.
        """
        raw = self._read_tile(xs, ys, include_border=True)
        if raw is None:
            return None
        b = self.index.tile_bdr
        return GeogWindow(index=self.index, x0=xs - b, y0=ys - b, raw=raw)

    def tile_coverage_mask(self, x0: int, x1: int,
                           y0: int, y1: int) -> np.ndarray:
        """Boolean source-cell coverage for an inclusive requested window.

        Longitude wrapping follows :meth:`read_window`.  Cells outside a
        non-wrapping dataset, rows outside its latitude extent, and cells
        belonging to absent tiles are false.  This is metadata-only: tile
        payloads are not read, so preflight can prove completeness cheaply.
        """

        if x1 < x0 or y1 < y0:
            raise ValueError("empty window")
        x = np.arange(x0, x1 + 1, dtype=np.int64)
        y = np.arange(y0, y1 + 1, dtype=np.int64)
        if self.wraps_x:
            source_x = (x - 1) % self.nx_global + 1
            x_inside = np.ones(x.size, dtype=bool)
        else:
            source_x = x
            x_inside = (x >= 1) & (x <= self.nx_global)
        y_inside = (y >= 1) & (y <= self.ny_global)
        x_origins = ((source_x - 1) // self.index.tile_x
                     * self.index.tile_x + 1)
        y_origins = ((y - 1) // self.index.tile_y
                     * self.index.tile_y + 1)
        mask = np.zeros((y.size, x.size), dtype=bool)
        for j, (ys, inside_y) in enumerate(zip(y_origins, y_inside)):
            if not inside_y:
                continue
            mask[j] = x_inside & np.fromiter(
                ((int(xs), int(ys)) in self.tiles for xs in x_origins),
                dtype=bool, count=x.size,
            )
        return mask

    def _extent_mask(self, x0: int, x1: int,
                     y0: int, y1: int) -> np.ndarray:
        """Cells in a requested window that belong to the dataset extent."""

        x = np.arange(x0, x1 + 1, dtype=np.int64)
        y = np.arange(y0, y1 + 1, dtype=np.int64)
        x_inside = (np.ones(x.size, dtype=bool) if self.wraps_x else
                    (x >= 1) & (x <= self.nx_global))
        y_inside = (y >= 1) & (y <= self.ny_global)
        return y_inside[:, None] & x_inside[None, :]

    def missing_tiles(self, x0: int, x1: int, y0: int,
                      y1: int) -> tuple[tuple[int, int], ...]:
        """Expected tile origins absent from the in-extent window cells."""

        mask = self.tile_coverage_mask(x0, x1, y0, y1)
        missing_cells = ~mask & self._extent_mask(x0, x1, y0, y1)
        missing: set[tuple[int, int]] = set()
        for j, i in np.argwhere(missing_cells):
            x = x0 + int(i)
            y = y0 + int(j)
            if self.wraps_x:
                x = (x - 1) % self.nx_global + 1
            xs = (x - 1) // self.index.tile_x * self.index.tile_x + 1
            ys = (y - 1) // self.index.tile_y * self.index.tile_y + 1
            missing.add((xs, ys))
        return tuple(sorted(missing, key=lambda item: (item[1], item[0])))

    def required_tile_origins(
            self, x0: int, x1: int, y0: int,
            y1: int) -> tuple[tuple[int, int], ...]:
        """Tile origins intersecting the in-extent part of a source window.

        Unlike :meth:`missing_tiles`, this includes present and absent
        origins.  It is metadata-only and deliberately independent of the
        dataset's ``sparse`` declaration so a staged, footprint-minimized
        tree can prove that every tile required by a model domain is present.
        """

        if x1 < x0 or y1 < y0:
            raise ValueError("empty window")
        x = np.arange(x0, x1 + 1, dtype=np.int64)
        y = np.arange(y0, y1 + 1, dtype=np.int64)
        if self.wraps_x:
            source_x = (x - 1) % self.nx_global + 1
        else:
            source_x = x[(x >= 1) & (x <= self.nx_global)]
        source_y = y[(y >= 1) & (y <= self.ny_global)]
        if source_x.size == 0 or source_y.size == 0:
            return ()
        x_origins = np.unique(
            (source_x - 1) // self.index.tile_x * self.index.tile_x + 1)
        y_origins = np.unique(
            (source_y - 1) // self.index.tile_y * self.index.tile_y + 1)
        return tuple(
            (int(xs), int(ys))
            for ys in y_origins
            for xs in x_origins
        )

    def read_window(self, x0: int, x1: int, y0: int, y1: int) -> GeogWindow:
        """Mosaic the (1-based, inclusive) index window ``[x0..x1, y0..y1]``.

        x wraps modulo the global width for global datasets; rows outside
        ``[1, ny_global]`` and absent tiles are filled with the dataset's
        missing value (or 0 when the index declares none).
        """
        idx = self.index
        if x1 < x0 or y1 < y0:
            raise ValueError("empty window")
        nxw, nyw = x1 - x0 + 1, y1 - y0 + 1
        if nxw > self.nx_global:
            raise ValueError("window wider than the global grid")
        coverage = self.tile_coverage_mask(x0, x1, y0, y1)
        unexplained = ~coverage & self._extent_mask(x0, x1, y0, y1)
        if not self.declared_sparse and bool(np.any(unexplained)):
            j, i = np.argwhere(unexplained)[0]
            source_x, source_y = x0 + int(i), y0 + int(j)
            if self.wraps_x:
                source_x = (source_x - 1) % self.nx_global + 1
            tile_x = ((source_x - 1) // idx.tile_x * idx.tile_x + 1)
            tile_y = ((source_y - 1) // idx.tile_y * idx.tile_y + 1)
            raise FileNotFoundError(
                f"WPS GEOG dataset {self.path} has unexplained fill at "
                f"source index (x={source_x}, y={source_y}); expected tile "
                f"origin ({tile_x}, {tile_y}) is absent. Declare this "
                "dataset sparse only when missing coverage is intentional."
            )
        fill = idx.missing_value if idx.missing_value is not None else 0
        out = np.full((idx.nz, nyw, nxw), fill, dtype=idx.dtype)

        # split the x range into wrap-contiguous segments of absolute coords
        if self.wraps_x:
            segs = []
            a = x0
            while a <= x1:
                wa = (a - 1) % self.nx_global + 1
                run = min(x1 - a, self.nx_global - wa) + 1
                segs.append((a, wa, run))
                a += run
        else:
            segs = [(x0, x0, nxw)]

        ty, tx = idx.tile_y, idx.tile_x
        yy0, yy1 = max(y0, 1), min(y1, self.ny_global)
        for ys in range((yy0 - 1) // ty * ty + 1, yy1 + 1, ty):
            for abs_x0, wx0, run in segs:
                wx1 = wx0 + run - 1
                for xs in range((wx0 - 1) // tx * tx + 1, wx1 + 1, tx):
                    tile = self._read_tile(xs, ys)
                    if tile is None:
                        continue
                    # overlap in wrapped source coordinates
                    ox0, ox1 = max(wx0, xs), min(wx1, xs + tx - 1)
                    oy0, oy1 = max(yy0, ys), min(yy1, ys + ty - 1)
                    if ox1 < ox0 or oy1 < oy0:
                        continue
                    di = (abs_x0 - x0) + (ox0 - wx0)
                    dj = oy0 - y0
                    out[:, dj:dj + oy1 - oy0 + 1,
                        di:di + ox1 - ox0 + 1] = \
                        tile[:, oy0 - ys:oy1 - ys + 1, ox0 - xs:ox1 - xs + 1]
        return GeogWindow(index=idx, x0=x0, y0=y0, raw=out,
                          coverage=coverage)


def tile_coverage_mask(dataset: GeogDataset | str | Path,
                       x0: int, x1: int, y0: int, y1: int, *,
                       sparse: bool | None = None) -> np.ndarray:
    """Public metadata-only WPS tile-coverage mask API."""

    ds = (dataset if isinstance(dataset, GeogDataset)
          else GeogDataset(dataset, sparse=sparse))
    return ds.tile_coverage_mask(x0, x1, y0, y1)
