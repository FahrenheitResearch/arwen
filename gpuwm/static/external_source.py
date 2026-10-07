"""Static fields from a pinned published file, with exact aligned crops.

Some forecast systems run with a static file of their own that cannot be
rebuilt from public WPS_GEOG inputs (its soil, land use, lakes, terrain
edits and drag fields come from data sets and procedures that are not
published).  For such a system the only exact route is to take the file.
This module does that, for any file a row of
``gpuwm/data/static_sources/static-sources.v1.toml`` pins:

* the row is DATA: URL, mirrors, size and SHA-256, the file's mass grid,
  the global attributes it must carry, the fields taken, and a rename map
  for fields whose engine name differs from the file's;
* a configuration selects a row by its metadata, ``[static] source =
  "<id>"`` (with an optional ``source_fields`` subset), carried to every
  geography build on the static carrier the way terrain smoothing is
  (:mod:`gpuwm.static.highres_production`, :mod:`gpuwm.static.build`);
* the exact crop is computed from grid identity, never written down: the
  case grid's first mass point is projected into the file's grid, the
  offset must be a whole number of cells, the case grid must lie wholly
  inside, and the file's own latitudes and longitudes over the window
  must equal the case grid's;
* a row may declare per-field nearest/bilinear sampling when a grid at
  the same spacing starts between source cells; file coordinates are
  verified on the original source lattice before that sampling;
* decoding, the crop and sampling run in Rust (``rw_netcdf dump``); this
  module only resolves the row, checks contracts and places the arrays.

A configuration that names no source reads no external file, so its
statics stay byte-identical. Adding another published static file is a
row, not a code path.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, MutableMapping

import numpy as np

#: The table's schema string; anything else is refused.
TABLE_SCHEMA = "gpuwm-static-sources-v1"
#: The packaged table.
TABLE_PATH = (Path(__file__).resolve().parent.parent / "data"
              / "static_sources" / "static-sources.v1.toml")
#: The cache folder, beside the WPS_GEOG datasets: ``<geog root>/
#: static_sources/<id>/<filename>``.
CACHE_SUBDIR = "static_sources"
#: Environment override for the cache folder (a shared read-only mirror).
CACHE_ENV = "GPUWM_STATIC_SOURCE_ROOT"
#: The decoder contract token for ``--window`` (rw_netcdf ``--abi``).
WINDOW_ABI_TOKEN = "dump_window_v1"
SAMPLE_ABI_TOKEN = "dump_sample_window_v1"
#: The formats this reader understands.
FORMATS = ("wps-geo_em",)

#: Engine field names by group: what ``[static] source_fields`` selects.
#: These are geo_em/WRF names, the same for every row.
FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "landuse": ("LANDUSEF", "LU_INDEX", "LANDMASK"),
    "soil": ("SOILCTOP", "SCT_DOM", "SOILCBOT", "SCB_DOM"),
    "terrain": ("HGT_M", "SLOPECAT"),
    "vegetation": ("GREENFRAC", "LAI12M"),
    "albedo": ("ALBEDO12M", "SNOALB"),
    "soil_temperature": ("SOILTEMP",),
    "lake_depth": ("LAKE_DEPTH",),
    "drag": (),  # filled from the orographic table below
}
ALL_GROUPS = "all"

#: The global attributes that name the land-use legend; a source that
#: replaces land use must carry the legend the build's own selection
#: reports, because the water-temperature statics and the history writer
#: read that legend.
LEGEND_ATTRS = ("MMINLU", "ISWATER", "ISLAKE", "ISICE", "ISURBAN",
                "NUM_LAND_CAT")

#: How far the file's latitudes and longitudes may sit from the case
#: grid's over the window.  geo_em stores them as float32 computed in
#: WPS's single-precision projection: a few 1e-5 degrees.  A one-cell
#: shift of a 3 km grid is 0.027 degrees, a thousand times this.
LATLON_TOLERANCE_DEG = 2.0e-4


def _drag_names() -> tuple[str, ...]:
    from .orographic import OROGRAPHIC_ROWS
    return tuple(OROGRAPHIC_ROWS)


def field_groups() -> dict[str, tuple[str, ...]]:
    groups = dict(FIELD_GROUPS)
    groups["drag"] = _drag_names()
    return groups


class StaticSourceError(ValueError):
    """A static source that cannot be used, with the breakage it prevents."""


@dataclass(frozen=True)
class StaticSourceRow:
    """One row of the static-source table."""

    id: str
    describes: str
    format: str
    filename: str
    url: str
    mirrors: tuple[str, ...]
    sha256: str
    bytes: int
    grid: tuple[tuple[str, object], ...]
    attrs: tuple[tuple[str, object], ...]
    fields: tuple[str, ...]
    rename: tuple[tuple[str, str], ...]
    drag_semantics: str = ""
    provenance: str = ""
    field_sampling: tuple[tuple[str, str], ...] = ()

    @property
    def grid_map(self) -> dict[str, object]:
        return dict(self.grid)

    @property
    def attrs_map(self) -> dict[str, object]:
        return dict(self.attrs)

    def served(self) -> dict[str, str]:
        """Engine name -> file variable name for every field of the row."""
        rename = dict(self.rename)
        return {rename.get(name, name): name for name in self.fields}

    @property
    def sampling_map(self) -> dict[str, str]:
        return dict(self.field_sampling)


def _row_from_table(entry: Mapping[str, object], *, where: str) -> StaticSourceRow:
    required = ("id", "describes", "format", "filename", "url", "mirrors",
                "sha256", "bytes", "grid", "attrs", "fields")
    known = set(required) | {"rename", "drag_semantics", "provenance", "field_sampling"}
    missing = [key for key in required if key not in entry]
    unknown = sorted(set(entry) - known)
    if missing or unknown:
        raise StaticSourceError(
            f"{where}: static-source row needs {missing or 'nothing more'} "
            f"and has unknown key(s) {unknown or 'none'}")
    if entry["format"] not in FORMATS:
        raise StaticSourceError(
            f"{where}: format {entry['format']!r} has no reader; known: {FORMATS}")
    sha = str(entry["sha256"]).lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise StaticSourceError(f"{where}: sha256 must be 64 hex digits")
    size = entry["bytes"]
    if type(size) is not int or size <= 0:
        raise StaticSourceError(f"{where}: bytes must be a positive integer")
    grid = dict(entry["grid"])
    for key in ("map_proj", "ref_lat", "ref_lon", "truelat1", "truelat2",
                "stand_lon", "dx", "dy", "e_we", "e_sn"):
        if key not in grid:
            raise StaticSourceError(f"{where}: [grid] lacks {key}")
    fields = tuple(str(name) for name in entry["fields"])
    rename = {str(k): str(v) for k, v in dict(entry.get("rename") or {}).items()}
    stray = sorted(set(rename) - set(fields))
    if stray:
        raise StaticSourceError(
            f"{where}: rename names fields the row does not take: {stray}")
    engine = [rename.get(name, name) for name in fields]
    if len(set(engine)) != len(engine):
        raise StaticSourceError(f"{where}: two fields map onto one engine name")
    sampling = dict(entry.get("field_sampling") or {})
    if sampling and (set(sampling) != set(fields)
                     or set(sampling.values()) - {"nearest", "bilinear"}):
        raise StaticSourceError(f"{where}: field_sampling must name every source "
                                "field once with nearest or bilinear")
    return StaticSourceRow(
        id=str(entry["id"]), describes=str(entry["describes"]),
        format=str(entry["format"]), filename=str(entry["filename"]),
        url=str(entry["url"]),
        mirrors=tuple(str(url) for url in entry["mirrors"]),
        sha256=sha, bytes=int(size),
        grid=tuple(sorted(grid.items())),
        attrs=tuple(sorted(dict(entry["attrs"]).items())),
        fields=fields, rename=tuple(sorted(rename.items())),
        drag_semantics=str(entry.get("drag_semantics", "")),
        provenance=str(entry.get("provenance", "")),
        field_sampling=tuple(sorted(sampling.items())))


@lru_cache(maxsize=None)
def _load_table(path: str) -> dict[str, StaticSourceRow]:
    import tomllib
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != TABLE_SCHEMA:
        raise StaticSourceError(
            f"{path}: schema {raw.get('schema')!r}, expected {TABLE_SCHEMA!r}")
    rows: dict[str, StaticSourceRow] = {}
    for index, entry in enumerate(raw.get("source", ())):
        row = _row_from_table(entry, where=f"{path} source[{index}]")
        if row.id in rows:
            raise StaticSourceError(f"{path}: duplicate static source {row.id!r}")
        rows[row.id] = row
    return rows


def static_source_rows(path: Path | None = None) -> dict[str, StaticSourceRow]:
    """Every row of the packaged table (or of ``path``)."""
    return dict(_load_table(str(TABLE_PATH if path is None else path)))


def static_source_row(source_id: str) -> StaticSourceRow:
    rows = static_source_rows()
    try:
        return rows[source_id]
    except KeyError:
        raise StaticSourceError(
            f"no static source {source_id!r} in {TABLE_PATH.name}; known: "
            f"{sorted(rows)}.  Naming a source that does not exist would "
            "build the statics from WPS_GEOG under that source's name") from None


@dataclass(frozen=True)
class StaticSourceSetting:
    """What a configuration asked for: a row and the groups to take."""

    id: str
    groups: tuple[str, ...] = (ALL_GROUPS,)
    #: The row's pin when the setting was made; a table whose row now pins
    #: other bytes refuses the setting rather than reading those bytes.
    sha256: str = ""

    def __post_init__(self) -> None:
        known = {ALL_GROUPS, *FIELD_GROUPS}
        if (not isinstance(self.groups, (tuple, list)) or not self.groups
                or any(not isinstance(group, str) for group in self.groups)):
            raise StaticSourceError("static source groups must be a non-empty "
                                    "list of field-group names; an empty or "
                                    "malformed selection would omit its declared fields")
        unknown = sorted(set(self.groups) - known)
        if unknown or (ALL_GROUPS in self.groups and len(self.groups) != 1):
            raise StaticSourceError(f"static source groups {list(self.groups)!r} "
                                    f"are invalid; known: {sorted(known)}, with "
                                    "'all' used alone. Otherwise the selection "
                                    "would claim a field set it does not take")
        object.__setattr__(self, "groups", tuple(self.groups))
        row = static_source_row(self.id)
        if not self.sha256:
            object.__setattr__(self, "sha256", row.sha256)
        elif self.sha256 != row.sha256:
            raise StaticSourceError(
                f"static source {self.id!r} was recorded with sha256 "
                f"{self.sha256[:12]}..., but the table now pins "
                f"{row.sha256[:12]}...; prepare again under this table, or "
                "this run would read other bytes than the ones it recorded")

    @property
    def row(self) -> StaticSourceRow:
        return static_source_row(self.id)

    def engine_names(self) -> tuple[str, ...]:
        """The engine fields this setting takes from the row, in table order."""
        served = self.row.served()
        if ALL_GROUPS in self.groups:
            return tuple(served)
        groups = field_groups()
        wanted = {name for group in self.groups for name in groups[group]}
        return tuple(name for name in served if name in wanted)

    def echo(self) -> dict[str, object]:
        return {"id": self.id, "sha256": self.sha256,
                "fields": list(self.groups)}


def parse_static_source(table: Mapping[str, object], *, source: str
                        ) -> StaticSourceSetting | None:
    """``[static] source`` / ``source_fields`` as a setting, or ``None``."""
    if "source" not in table:
        if "source_fields" in table:
            raise StaticSourceError(
                f"[static] of {source} declares source_fields without a "
                "source; name the static source the fields come from")
        return None
    source_id = table["source"]
    if not isinstance(source_id, str) or not source_id:
        raise StaticSourceError(
            f"source in [static] of {source} must be a static-source id, "
            f"got {source_id!r}")
    groups = table.get("source_fields", [ALL_GROUPS])
    if isinstance(groups, str):
        groups = [groups]
    if (not isinstance(groups, list) or not groups
            or not all(isinstance(g, str) for g in groups)):
        raise StaticSourceError(
            f"source_fields in [static] of {source} must be a non-empty "
            f"list of group names, got {groups!r}")
    known = (ALL_GROUPS, *FIELD_GROUPS)
    unknown = sorted(set(groups) - set(known))
    if unknown:
        raise StaticSourceError(
            f"source_fields in [static] of {source} names unknown group(s) "
            f"{unknown}; known: {list(known)}")
    if ALL_GROUPS in groups and len(groups) > 1:
        raise StaticSourceError(
            f"source_fields in [static] of {source}: 'all' already takes "
            "every group; list groups or 'all', not both")
    ordered = (ALL_GROUPS,) if ALL_GROUPS in groups else tuple(
        g for g in FIELD_GROUPS if g in groups)
    try:
        return StaticSourceSetting(id=source_id, groups=ordered)
    except StaticSourceError as error:
        raise StaticSourceError(f"[static] of {source}: {error}") from None


def setting_from_echo(echo, *, source: str) -> StaticSourceSetting:
    """The setting a sealed identity recorded (:meth:`StaticSourceSetting.echo`)."""
    if (not isinstance(echo, Mapping) or set(echo) != {"id", "sha256", "fields"}
            or not isinstance(echo["fields"], list)):
        raise StaticSourceError(
            f"static_source of {source} must be {{id, sha256, fields}}, got "
            f"{echo!r}; refusing to rebuild statics from a source the seal "
            "does not record")
    return StaticSourceSetting(id=str(echo["id"]),
                               groups=tuple(str(g) for g in echo["fields"]),
                               sha256=str(echo["sha256"]))


def static_source_for(carrier) -> StaticSourceSetting | None:
    """The setting a static carrier holds, or ``None``."""
    return getattr(carrier, "static_source", None)


def static_source_receipt(carrier) -> dict[str, object]:
    """``{"d01": echo}`` attesting a carrier's static source, else ``{}``."""
    setting = static_source_for(carrier)
    return {} if setting is None else {"d01": setting.echo()}


def require_root_static_source(carrier, domain_id: int, receipt, *, grid=None) -> None:
    """The static-source root seam.

    A route that built or loaded a root's statics without carrying the
    configured source has WPS_GEOG statics; running them under a
    configuration that named the source would integrate the land surface
    on soil and land use the configuration did not ask for.  Accepted
    only when the receipt (or a nested ``baseline``) attests the setting.
    """
    setting = static_source_for(carrier)
    if setting is None or int(domain_id) != 1:
        return
    if grid is not None and sampling_window(setting.row, grid) is None:
        return
    want = setting.echo()
    while isinstance(receipt, Mapping):
        attested = receipt.get("static_source")
        if isinstance(attested, Mapping) and attested.get("d01") == want:
            return
        receipt = receipt.get("baseline")
    raise StaticSourceError(
        f"static-source root seam: d01 asks for static source "
        f"{setting.id!r} ({', '.join(setting.groups)}), but this route built "
        "its statics from WPS_GEOG or loaded a prebuilt cache that does not "
        "record the source; the forecast would run on statics the "
        "configuration did not ask for.  Prepare the root through a route "
        "that builds from the declared static source, or remove "
        "[static] source")


# ---------------------------------------------------------------------------
# The local file.
# ---------------------------------------------------------------------------

def cache_roots(geog_root) -> tuple[Path, ...]:
    roots = []
    override = os.environ.get(CACHE_ENV)
    if override:
        roots.append(Path(override))
    if geog_root is not None:
        roots.append(Path(geog_root) / CACHE_SUBDIR)
    return tuple(roots)


def fetch_command(row: StaticSourceRow, geog_root) -> str:
    root = "" if geog_root is None else f" --root {geog_root}"
    return f"gpuwm fetch-geog --static-source {row.id}{root}"


def verify_local_file(path: Path, row: StaticSourceRow) -> None:
    """Verify the bytes against the row each time the file is resolved.

    Size and modification time cannot attest content: an in-place edit,
    restored timestamp or stale sidecar can leave them unchanged.
    """
    stat = path.stat()
    if stat.st_size != row.bytes:
        raise StaticSourceError(
            f"{path} holds {stat.st_size} bytes, the static source "
            f"{row.id!r} pins {row.bytes}; a truncated or other file would "
            "put other geography under the source's name")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != row.sha256:
        raise StaticSourceError(
            f"{path} has sha256 {digest.hexdigest()}, the static source "
            f"{row.id!r} pins {row.sha256}; refusing other bytes under the "
            "source's name")


def resolve_local_file(row: StaticSourceRow, geog_root) -> Path:
    """The verified local copy of a row's file, or a refusal naming the fetch."""
    for root in cache_roots(geog_root):
        candidate = root / row.id / row.filename
        if candidate.is_file():
            verify_local_file(candidate, row)
            return candidate
    looked = ", ".join(str(root / row.id / row.filename)
                       for root in cache_roots(geog_root)) or "(no GEOG root)"
    raise FileNotFoundError(
        f"the static source {row.id!r} ({row.describes}) is not staged; "
        f"looked at {looked}.  Run: {fetch_command(row, geog_root)}  "
        f"(downloads {row.bytes / 1e9:.2f} GB from {row.url}, verified "
        f"against sha256 {row.sha256}).")


def fetch_static_source(source_id: str, root: Path, *, progress=print,
                         urlopen_fn=None) -> Path:
    """Download and verify a row's file into ``<root>/static_sources/<id>/``."""
    from gpuwm import fetch_guard

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with fetch_guard.hold("fetch-geog", root, progress=progress):
        return _fetch_static_source_locked(
            source_id, root, progress=progress, urlopen_fn=urlopen_fn)


def _fetch_static_source_locked(source_id: str, root: Path, *, progress,
                                 urlopen_fn) -> Path:
    """One root writer owns the partial, verification and final rename."""
    from gpuwm import geog_assets

    row = static_source_row(source_id)
    folder = Path(root) / CACHE_SUBDIR / row.id
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / row.filename
    if final.is_file():
        verify_local_file(final, row)
        progress(f"fetch-geog: static source {row.id} already staged at {final}")
        return final
    partial = folder / (row.filename + ".part")
    errors = []
    for url in (row.url, *row.mirrors):
        kwargs = {} if urlopen_fn is None else {"urlopen_fn": urlopen_fn}
        try:
            geog_assets.download_archive(
                url, partial, expected_bytes=row.bytes, progress=progress,
                label=row.id, report_every_bytes=max(row.bytes // 10, 1),
                **kwargs)
            geog_assets.verify_archive(
                partial, expected_sha256=row.sha256, expected_bytes=row.bytes,
                source="hf", allow_drift=False, progress=progress,
                label=row.id)
        except (OSError, geog_assets.GeogFetchError) as error:
            errors.append(f"{url}: {error}")
            continue
        partial.replace(final)
        verify_local_file(final, row)
        progress(f"fetch-geog: static source {row.id} staged at {final}")
        return final
    raise geog_assets.GeogFetchError(
        f"static source {row.id!r} could not be fetched from any of its "
        f"{1 + len(row.mirrors)} URL(s): " + "; ".join(errors))


def required_static_rows(raw, *, base_dir=None) -> tuple[StaticSourceRow, ...]:
    """The static-source rows a run of this configuration will read.

    The row the configuration's ``[static] source`` names, or the one its
    ``[fetch]`` source's metadata selects on the row's own cone -- the
    same resolution the static build makes
    (:func:`gpuwm.static.highres_production.resolve_static_highres`) --
    when at least one domain is at the row's grid spacing.  A row at
    another spacing is never read (:func:`overlay_static_source` reports
    it ``NOT_APPLICABLE``), so it is not asked for.
    """
    from .highres_production import raw_domain_spacings, resolve_static_highres

    fetch = raw.get("fetch") if isinstance(raw, Mapping) else None
    source = (fetch or {}).get("source") if isinstance(fetch, Mapping) else None
    carrier = resolve_static_highres(raw, source=str(source or "configuration"),
                                     base_dir=base_dir)
    setting = static_source_for(carrier)
    if setting is None:
        return ()
    row = setting.row
    g = row.grid_map
    spacings = raw_domain_spacings(raw)
    if not any(abs(float(dx) - float(g["dx"])) <= 1e-6
               and abs(float(dx) - float(g["dy"])) <= 1e-6 for dx in spacings):
        return ()
    return (row,)


def stage_required_static_sources(raw, geog_root, *, base_dir=None,
                                  progress=print, fetch=None) -> list[Path]:
    """Fetch every static file this configuration reads and is not staged.

    The run's own statics, staged through the same fetch as ``gpuwm
    fetch-geog --static-source ID`` (:func:`fetch_static_source`: the pin,
    the resume, the verification and the cache folder are that command's),
    with a line naming the file and its size before the download and
    progress lines while it runs.  A file already staged is verified by
    :func:`resolve_local_file` and not fetched again.

    Breakage it prevents: a run of a shipped HRRR configuration recipe
    stopped at static preparation, after its forcing download, until the
    user had run ``fetch-geog --static-source hrrr-conus-v4`` by hand
    (WOOF 1.0.3 GPU smoke, 2026-10-06).
    """
    fetch = fetch_static_source if fetch is None else fetch
    staged = []
    for row in required_static_rows(raw, base_dir=base_dir):
        try:
            staged.append(resolve_local_file(row, geog_root))
            continue
        except FileNotFoundError:
            pass
        host = row.url.split("/")[2] if "://" in row.url else row.url
        progress(
            f"note: static source {row.id} is not staged; fetching "
            f"{row.bytes / 1e9:.2f} GB from {host} into "
            f"{Path(geog_root) / CACHE_SUBDIR / row.id} (the same as "
            f"`{fetch_command(row, geog_root)}`)")
        staged.append(fetch(row.id, Path(geog_root), progress=progress))
    return staged


# ---------------------------------------------------------------------------
# Grid identity and the crop.
# ---------------------------------------------------------------------------

def source_grid(row: StaticSourceRow):
    """The row's grid as a projected grid (WPS centred reference point)."""
    from .projection import projection_class
    g = row.grid_map
    return projection_class(str(g["map_proj"]))(
        float(g["ref_lat"]), float(g["ref_lon"]), float(g["truelat1"]),
        float(g["truelat2"]), float(g["stand_lon"]), float(g["dx"]),
        float(g["dy"]), int(g["e_we"]), int(g["e_sn"]))


#: The projection a case grid must share with a row before any of the
#: row's cells can be placed on it (spacing is checked on its own).
PROJECTION_KEYS = ("map_proj", "truelat1", "truelat2", "stand_lon")


def projection_mismatch(row: StaticSourceRow, projection) -> dict[str, tuple]:
    """``{key: (case value, source value)}`` for every :data:`PROJECTION_KEYS`
    entry on which ``projection`` (a grid, or a ``[projection]`` mapping)
    differs from the row's grid; empty when the cone is the row's own."""
    g = row.grid_map
    if isinstance(projection, Mapping):
        value = projection.get
    else:
        def value(key):
            return getattr(projection, key, None)
    return {key: (value(key), g[key]) for key in PROJECTION_KEYS
            if value(key) != g[key]}


def _grid_offsets(row: StaticSourceRow, grid):
    g = row.grid_map
    if (abs(float(grid.dx) - float(g["dx"])) > 1e-6
            or abs(float(grid.dy) - float(g["dy"])) > 1e-6):
        return None
    mismatch = projection_mismatch(row, grid)
    if mismatch:
        raise StaticSourceError(
            f"static source {row.id!r}: the case grid's projection differs "
            f"from the source's ({mismatch}); its source-cell positions "
            "would put the source's fields under the wrong ground")
    src = source_grid(row)
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    offsets = []
    for x, y in ((1.0, 1.0), (float(nx), float(ny)), (1.0, float(ny)),
                 (float(nx), 1.0)):
        lat, lon = grid.ij_to_latlon(x, y)
        sx, sy = src.latlon_to_ij(float(lat), float(lon))
        offsets.append((float(sx) - x, float(sy) - y))
    if not all(math.isfinite(v) for pair in offsets for v in pair):
        raise StaticSourceError("static source grid transform produced non-finite offsets")
    return offsets


def sampling_window(row: StaticSourceRow, grid) -> tuple[float, float] | None:
    """Exact integer crop, or row-declared fractional same-spacing sampling."""
    if not row.field_sampling:
        return crop_window(row, grid)
    offsets = _grid_offsets(row, grid)
    if offsets is None:
        return None
    i0, j0 = offsets[0]
    from .grid_identity import GRID_POSITION_TOLERANCE_CELLS
    if max(max(abs(a - round(i0)), abs(b - round(j0))) for a, b in offsets) <= GRID_POSITION_TOLERANCE_CELLS:
        return crop_window(row, grid)
    variation = max(max(abs(a - i0), abs(b - j0)) for a, b in offsets)
    if variation > 1.0e-6:
        raise StaticSourceError(
            f"static source {row.id!r}: the grid offset varies by {variation:.4g} "
            "cells; rectangular same-spacing sampling would assign fields "
            "to different ground locations")
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    src_nx, src_ny = int(row.grid_map["e_we"]) - 1, int(row.grid_map["e_sn"]) - 1
    if i0 < 0 or j0 < 0 or i0 + nx - 1 > src_nx - 1 or j0 + ny - 1 > src_ny - 1:
        raise StaticSourceError(
            f"static source {row.id!r}: sampled window ({nx} x {ny} at "
            f"{i0:.6g}, {j0:.6g}) leaves the source; source values outside "
            "its grid do not exist")
    return i0, j0


def crop_window(row: StaticSourceRow, grid) -> tuple[int, int] | None:
    """``(i0, j0)``: the 0-based mass-grid offset of ``grid`` in the row's
    grid, or ``None`` when ``grid`` is at another spacing (the row does not
    describe it, and its statics build from WPS_GEOG as before).

    At the row's spacing, any other projection, a fractional offset, or a
    grid that does not lie wholly inside is a refusal: statics on a grid
    that is not a sub-window of the source would put the source's land
    use under the wrong cells.
    """
    g = row.grid_map
    if (abs(float(grid.dx) - float(g["dx"])) > 1e-6
            or abs(float(grid.dy) - float(g["dy"])) > 1e-6):
        return None
    mismatch = projection_mismatch(row, grid)
    if mismatch:
        raise StaticSourceError(
            f"static source {row.id!r}: the case grid's projection differs "
            f"from the source's ({mismatch}); statics on a grid that is not "
            "a sub-window of the source would put its land use under the "
            "wrong cells")
    src = source_grid(row)
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    src_nx, src_ny = int(g["e_we"]) - 1, int(g["e_sn"]) - 1
    offsets = []
    for x, y in ((1.0, 1.0), (float(nx), float(ny)), (1.0, float(ny)),
                 (float(nx), 1.0)):
        lat, lon = grid.ij_to_latlon(x, y)
        sx, sy = src.latlon_to_ij(float(lat), float(lon))
        offsets.append((float(sx) - x, float(sy) - y))
    i_off, j_off = offsets[0]
    i0, j0 = round(i_off), round(j_off)
    from .grid_identity import GRID_POSITION_TOLERANCE_CELLS
    worst = max(max(abs(a - i0), abs(b - j0)) for a, b in offsets)
    if worst > GRID_POSITION_TOLERANCE_CELLS:
        raise StaticSourceError(
            f"static source {row.id!r}: the case grid sits {worst:.4f} cells "
            "off the source's mass points; statics on a grid that is not a "
            "sub-window of the source would put its land use under the "
            "wrong cells")
    if i0 < 0 or j0 < 0 or i0 + nx > src_nx or j0 + ny > src_ny:
        raise StaticSourceError(
            f"static source {row.id!r}: the case grid ({nx} x {ny} at offset "
            f"{i0}, {j0}) leaves the source's {src_nx} x {src_ny} grid; "
            "statics outside the source have no source cells to take")
    return int(i0), int(j0)


# ---------------------------------------------------------------------------
# The overlay.
# ---------------------------------------------------------------------------

def _require_window_reader(dataset, *, sampled=False) -> None:
    import subprocess
    try:
        abi = subprocess.run([os.fspath(dataset._executable), "--abi"],
                             capture_output=True, text=True, timeout=60).stdout
    except OSError as error:
        abi = str(error)
    required = SAMPLE_ABI_TOKEN if sampled else WINDOW_ABI_TOKEN
    if required not in abi:
        from gpuwm.netcdf_bridge import netcdf_remedy
        raise StaticSourceError(
            "the installed rw_netcdf predates the windowed read a static "
            "source is cropped with; without it the crop would move into "
            "Python.  Rebuild the reader.\n\n" + netcdf_remedy())


def served_names(setting: StaticSourceSetting | None) -> frozenset[str]:
    return frozenset(() if setting is None else setting.engine_names())


def overlay_static_source(fields: MutableMapping[str, np.ndarray], grid,
                          geog_root, setting: StaticSourceSetting, *,
                          requested: tuple[str, ...] = (),
                          landuse_attrs: Mapping[str, object] | None = None,
                          report: MutableMapping[str, object] | None = None,
                          ) -> MutableMapping[str, np.ndarray]:
    """Replace ``fields`` with the source's, cropped by grid identity.

    Only fields the build produced, plus ``requested`` fields, are taken.
    The static builder requests the full selected inventory, including
    optional fields no active physics option reads. ``TMN`` is
    recomputed from the source's ``SOILTEMP``, ``HGT_M`` and ``LANDMASK``
    whenever any of them came from the source.

    Integer origins preserve exact file values. A fractional origin is
    sampled only by the Rust reader using the row's per-field policy.
    """
    from gpuwm.netcdf_bridge import Dataset
    from .build import deep_soil_temperature_at_terrain

    row = setting.row
    window = sampling_window(row, grid)
    if window is None:
        if report is not None:
            report["static_source"] = {
                "id": row.id, "status": "NOT_APPLICABLE",
                "reason": f"grid spacing {grid.dx} m is not the source's"}
        return fields
    i0, j0 = window
    nx, ny = int(grid.e_we) - 1, int(grid.e_sn) - 1
    sampled = not isinstance(i0, int) or not isinstance(j0, int)
    served = row.served()
    wanted = [name for name in setting.engine_names()
              if name in fields or name in requested]
    missing = [name for name in requested if name not in served
               or name not in setting.engine_names()]
    if missing:
        raise StaticSourceError(
            f"static source {row.id!r} does not serve {missing}, which the "
            "build left to it")
    path = resolve_local_file(row, geog_root)
    dataset = Dataset(path)
    _require_window_reader(dataset, sampled=sampled)
    attrs = dataset.global_attributes
    g = row.grid_map
    expected_attrs = {**row.attrs_map,
                      "WEST-EAST_GRID_DIMENSION": int(g["e_we"]),
                      "SOUTH-NORTH_GRID_DIMENSION": int(g["e_sn"]),
                      "DX": float(g["dx"]), "DY": float(g["dy"]),
                      "TRUELAT1": float(g["truelat1"]),
                      "TRUELAT2": float(g["truelat2"]),
                      "STAND_LON": float(g["stand_lon"]),
                      "CEN_LAT": float(g["ref_lat"]),
                      "CEN_LON": float(g["ref_lon"])}
    drift = {}
    for key, value in expected_attrs.items():
        have = attrs.get(key)
        if isinstance(have, list) and len(have) == 1:
            have = have[0]
        if isinstance(value, float) or isinstance(have, float):
            # WPS writes projection attributes as default REAL. Stored
            # centre attributes can differ by one float32 ULP from the
            # namelist value even when the cell coordinates agree.
            tolerance = max(1e-6, 2.0 * abs(float(np.spacing(np.float32(value)))))
            same = have is not None and math.isclose(float(have), float(value),
                                                     rel_tol=0, abs_tol=tolerance)
        else:
            same = have == value
        if not same:
            drift[key] = (have, value)
    if drift:
        raise StaticSourceError(
            f"{path}: global attributes differ from static source "
            f"{row.id!r}: {drift}")
    takes_landuse = any(name in wanted for name in FIELD_GROUPS["landuse"])
    if takes_landuse and landuse_attrs is not None:
        legend = {key: (landuse_attrs.get(key), row.attrs_map.get(key))
                  for key in LEGEND_ATTRS
                  if landuse_attrs.get(key) != row.attrs_map.get(key)}
        if legend:
            raise StaticSourceError(
                f"static source {row.id!r} carries another land-use legend "
                f"than this build's selection ({legend}); the lake, water and "
                "history writers read the selection's legend, so they would "
                "misread the source's categories")
    # A sampled window checks the file's original coordinate registration
    # over every contributing source point, independently of its target.
    # An aligned crop retains the exact historical coordinate check.
    if sampled:
        ci0, cj0 = math.floor(i0), math.floor(j0)
        cni = math.ceil(i0 + nx - 1) - ci0 + 1
        cnj = math.ceil(j0 + ny - 1) - cj0 + 1
        src = source_grid(row)
        center_lat, center_lon = src.ij_to_latlon(
            ci0 + (cni + 1) / 2.0, cj0 + (cnj + 1) / 2.0)
        from .projection import projection_class
        checked_grid = projection_class(str(g["map_proj"]))(float(center_lat), float(center_lon),
            float(g["truelat1"]), float(g["truelat2"]), float(g["stand_lon"]),
            float(g["dx"]), float(g["dy"]), cni + 1, cnj + 1)
    else:
        ci0, cj0, cni, cnj = i0, j0, nx, ny
        checked_grid = grid
    lat_lon = {}
    for name in ("XLAT_M", "XLONG_M"):
        values = dataset.variables[name].read_window(i0=ci0, ni=cni, j0=cj0, nj=cnj)
        lat_lon[name] = values.reshape(values.shape[-2:])
        if not np.isfinite(lat_lon[name]).all():
            raise StaticSourceError(
                f"{path}: {name} holds missing or non-finite coordinates "
                "inside the window; its grid identity cannot be checked")
    lat, lon = checked_grid.latlon_mass()
    dlat = float(np.max(np.abs(lat_lon["XLAT_M"] - lat)))
    dlon = float(np.max(np.abs(((lat_lon["XLONG_M"] - lon) + 180.0) % 360.0
                               - 180.0)))
    if max(dlat, dlon) > LATLON_TOLERANCE_DEG:
        raise StaticSourceError(
            f"static source {row.id!r}: over the window at ({i0}, {j0}) the "
            f"file's latitudes and longitudes sit {dlat:.2e} and {dlon:.2e} "
            f"degrees from the case grid's (allowed {LATLON_TOLERANCE_DEG}); "
            "statics on a grid that is not a sub-window of the source would "
            "put its land use under the wrong cells")
    taken: dict[str, str] = {}
    for name in wanted:
        variable = dataset.variables.get(served[name])
        if variable is None:
            raise StaticSourceError(
                f"{path} lacks {served[name]}, which static source "
                f"{row.id!r} declares")
        if sampled:
            values = variable.read_sample_window(i0=i0, ni=nx, j0=j0, nj=ny,
                                                  method=row.sampling_map[served[name]])
        else:
            values = variable.read_window(i0=i0, ni=nx, j0=j0, nj=ny)
        if values.shape[0] == 1 and len(values.shape) >= 3:
            values = values[0]
        values = np.ascontiguousarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise StaticSourceError(
                f"{path}: {served[name]} holds missing or non-finite values "
                "inside the window")
        if name in fields and np.shape(fields[name]) != values.shape:
            raise StaticSourceError(
                f"static source {row.id!r}: {served[name]} has shape "
                f"{values.shape}, the build's {name} {np.shape(fields[name])}; "
                "a field of another layout would be read with the wrong "
                "categories or months")
        fields[name] = values
        taken[name] = served[name]
    if ({"SOILTEMP", "HGT_M", "LANDMASK"} & set(taken)
            and {"SOILTEMP", "HGT_M", "LANDMASK"} <= set(fields)):
        fields["TMN"] = deep_soil_temperature_at_terrain(
            fields["SOILTEMP"], fields["HGT_M"], fields["LANDMASK"])
    receipt = {
        "id": row.id, "status": "APPLIED", "sha256": row.sha256,
        "bytes": row.bytes, "path": str(path),
        "groups": list(setting.groups),
        "window": {"i0": i0, "j0": j0, "ni": nx, "nj": ny},
        "latlon_max_abs_deg": [dlat, dlon],
        "fields": taken,
        **({"sampling": {"mode": "fractional-same-spacing",
                          "methods": {name: row.sampling_map[file_name]
                                      for name, file_name in taken.items()},
                          "source_coordinate_window": {"i0": ci0, "j0": cj0,
                                                       "ni": cni, "nj": cnj}}}
           if sampled else {}),
        **({"drag_semantics": row.drag_semantics}
           if any(name in taken for name in _drag_names()) else {}),
    }
    if report is not None:
        report["static_source"] = receipt
    print(f"static source {row.id}: took {len(taken)} fields at window "
          f"i0={i0} j0={j0} ({nx} x {ny}), lat/lon within "
          f"{max(dlat, dlon):.1e} deg", flush=True)
    return fields


__all__ = [
    "ALL_GROUPS", "CACHE_ENV", "CACHE_SUBDIR", "FIELD_GROUPS",
    "StaticSourceError", "StaticSourceRow", "StaticSourceSetting",
    "PROJECTION_KEYS", "TABLE_PATH", "crop_window", "projection_mismatch",
    "sampling_window", "fetch_command", "fetch_static_source",
    "field_groups", "overlay_static_source", "parse_static_source",
    "require_root_static_source", "required_static_rows",
    "resolve_local_file", "served_names", "stage_required_static_sources",
    "setting_from_echo", "source_grid", "static_source_for",
    "static_source_receipt", "static_source_row", "static_source_rows",
    "verify_local_file",
]
