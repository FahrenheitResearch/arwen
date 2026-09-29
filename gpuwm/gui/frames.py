"""The pictures of a run: only what the Rust renderer wrote.

The renderer files every picture at
``<out>/<domain>/<product>/<valid-day>/<file>.png`` (``gpuwm.render_layout``,
the render folder ruling), a nest that retires and re-arms one segment
deeper at ``<out>/<domain>/episode-NNN/<product>/<valid-day>/<file>.png``,
and :func:`gpuwm.render_layout.iter_rendered` is
the one walker for that tree, so the page sees exactly the pictures every
other reader sees, and never the early render's dot-prefixed scratch.

The server lists these files and serves them byte for byte.  It never
draws, re-encodes or resizes a picture: a screen that needs a picture the
renderer does not make says so in words.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import threading
from typing import Any

from gpuwm.render_layout import engine_name, engine_output_time, episode_number, iter_rendered

from .files import COPY_DIR, long_path, read_json, utc_text

#: The renderer's own record of where each picture sits on the Earth,
#: written beside the render tree (``rustwx.render-georef/v1``).
GEOREF_MANIFEST = "render-georef.json"
GEOREF_SCHEMA = "rustwx.panel-georeference/v1"

#: How many pictures one listing returns at most (a long run has thousands).
LIST_LIMIT = 5000


def placement(parts: tuple[str, ...]) -> tuple[str, str, str, str, int]:
    """``(domain, episode, product, day, depth)`` of a picture from its path parts under the run folder.

    ``depth`` is how many trailing parts sit under the render root (the
    folder that holds ``render-georef.json``): 5 for an episodic nest
    (``d02-3km/episode-001/<product>/<day>/<file>``), 4 for a nest with one
    life (``episode`` is then ``""``), 1 for a flat picture.  The episode
    folder is read by the layout's own inverse of the name it writes, so an
    episode never stands in for the domain it is a life of.
    """

    if len(parts) >= 5 and episode_number(parts[-4]) is not None:
        return parts[-5], parts[-4], parts[-3], parts[-2], 5
    if len(parts) >= 4:
        return parts[-4], "", parts[-3], parts[-2], 4
    return "", "", "pictures", "", 1


def _groups(rundir: Path) -> list[dict[str, Any]]:
    """One row per ``<domain>[/<episode>]/<product>/<valid-day>`` folder, in path order."""

    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    rank_of: dict[tuple[str, str, str, str], tuple[int, float]] = {}
    for path in iter_rendered(rundir):
        relative = path.relative_to(rundir)
        parts = relative.parts
        domain, episode, product, day, _ = placement(parts)
        folder = "/".join(parts[:-1])
        key = (domain, episode, product, day)
        row = rows.get(key)
        if row is None:
            row = rows[key] = {"domain": domain, "episode": episode, "product": product, "day": day,
                               "folder": folder, "count": 0, "newest": None, "newest_mtime": 0.0}
        row["count"] += 1
        try:
            mtime = long_path(path).stat().st_mtime
        except OSError:
            mtime = 0.0
        row["newest_mtime"] = max(row["newest_mtime"], mtime)
        # A folder's newest picture is its latest forecast hour: a copied or restored run folder has its files'
        # times in copy order, and the card would show hour 0 as "the latest picture".
        hour = frame_time(path.name)["hour"]
        rank = (-1 if hour is None else hour, mtime)
        if rank >= rank_of.get(key, (-2, 0.0)):
            rank_of[key] = rank
            row["newest"] = relative.as_posix()
    return list(rows.values())


def index(rundir: Path) -> dict[str, Any]:
    """Domains, products and days, with counts and the newest picture of each."""

    groups = _groups(rundir)
    total = sum(row["count"] for row in groups)
    newest = max(groups, key=lambda row: row["newest_mtime"], default=None)
    products = order_products(sorted({row["product"] for row in groups}))
    # The picture Watching shows: the newest frame of the first favourite
    # look the run has, else the newest picture of any look.
    liked = favourites(products)
    lead = [row for row in groups if liked and row["product"] == liked[0]]
    shown = max(lead, key=lambda row: row["newest_mtime"], default=newest)
    domains = sorted({row["domain"] for row in groups})
    return {
        "count": total,
        "groups": groups,
        "domains": domains,
        "products": products,
        "newest": None if newest is None else newest["newest"],
        "watch": None if shown is None else shown["newest"],
        # The same choice made once per grid, so Watching can switch to a
        # nest while the run is still drawing it.
        "by_domain": {domain: _domain_watch(
                          [row for row in groups if row["domain"] == domain],
                          liked)
                      for domain in domains},
    }


def _domain_watch(rows: list[dict[str, Any]], liked: list[str]) -> dict[str, Any]:
    """One grid's picture count and the picture Watching shows for it.

    The newest frame of the first favourite look this grid has, else its
    newest picture of any look.
    """

    newest = max(rows, key=lambda row: row["newest_mtime"], default=None)
    shown = None
    for product in liked:
        lead = [row for row in rows if row["product"] == product]
        if lead:
            shown = max(lead, key=lambda row: row["newest_mtime"])
            break
    shown = shown or newest
    return {"count": sum(row["count"] for row in rows),
            "watch": None if shown is None else shown["newest"]}


def pictures(rundir: Path, *, domain: str | None = None, product: str | None = None,
             day: str | None = None, placed: list[dict[str, Any]] | None = None,
             episode: str | None = None) -> list[dict[str, Any]]:
    """The pictures of one domain and product (all days, or one), oldest first.

    Each carries its ``domain`` and, for a nest that retired and re-armed,
    its ``episode`` folder (``""`` for a nest with one life); ``episode``
    narrows the list to one life.

    Each carries ``valid``, the instant its file name names (ISO, UTC), or
    None.  When ``placed`` is a list, each picture also carries ``geo``:
    an index into ``placed``, which collects the distinct georeferences
    the renderer recorded for them (None for a picture it recorded none
    for, such as a composite of several maps).
    """

    out = []
    table = _GeoTable(rundir) if placed is not None else None
    for path in iter_rendered(rundir):
        relative = path.relative_to(rundir)
        parts = relative.parts
        this_domain, this_episode, this_product, this_day, depth = placement(parts)
        if domain is not None and this_domain != domain:
            continue
        if episode is not None and this_episode != episode:
            continue
        if product is not None and this_product != product:
            continue
        if day is not None and this_day != day:
            continue
        engine = engine_name(path.name, domain=this_domain or None, product=this_product or None)
        moment = engine_output_time(engine)
        item = {"path": relative.as_posix(), "name": path.name, "domain": this_domain, "episode": this_episode,
                "product": this_product, "day": this_day, "label": frame_label(path.name),
                "hour": frame_time(path.name)["hour"],
                "valid": None if moment is None else moment.strftime("%Y-%m-%dT%H:%M:%SZ")}
        if table is not None:
            item["geo"] = table.index(parts, engine, placed, depth)
        out.append(item)
        if len(out) >= LIST_LIMIT:
            break
    return out


# ------------------------------------------------------------------ georeference

_MANIFESTS: dict[str, tuple[tuple[int, float], dict[str, Any]]] = {}
_MANIFEST_LOCK = threading.Lock()


def _tail(name: str) -> str:
    """A file name without its first token: the renderer names a panel
    ``<prefix>_<model>_...`` and the delivery may carry another prefix."""

    return name.split("_", 1)[1] if "_" in name else name


_PANEL_FIELDS = ("image_width_px", "image_height_px", "plot_rect_px", "projection", "extent", "geographic_bounds")


def georef_manifest(folder: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """One render root's manifest as two lookups: ``filed`` and ``flat``.

    ``filed`` is ``{path relative to the render root: panel georeference}``,
    the key ``gpuwm.render_georef.file_pictures`` writes for every picture
    it files.  ``flat`` is ``{name tail: panel georeference}`` for entries
    keyed by a bare engine name, which is how records written before the
    record followed the filed paths named every picture; a run rendered
    then still places its pictures.
    """

    path = long_path(folder / GEOREF_MANIFEST)
    try:
        stat = path.stat()
    except OSError:
        return {"filed": {}, "flat": {}}
    key = (stat.st_size, stat.st_mtime)
    with _MANIFEST_LOCK:
        hit = _MANIFESTS.get(str(path))
        if hit is not None and hit[0] == key:
            return hit[1]
    document = read_json(path, default={}) or {}
    panels = document.get("panels") if isinstance(document, dict) else None
    filed: dict[str, dict[str, Any]] = {}
    flat: dict[str, dict[str, Any]] = {}
    for name, panel in (panels or {}).items():
        if not (isinstance(panel, dict) and panel.get("schema") == GEOREF_SCHEMA):
            continue
        place = {field: panel.get(field) for field in _PANEL_FIELDS}
        relative = str(name).replace("\\", "/")
        filed[relative] = place
        if "/" not in relative:
            flat[_tail(relative)] = place
    found = {"filed": filed, "flat": flat}
    with _MANIFEST_LOCK:
        _MANIFESTS[str(path)] = (key, found)
    return found


class _GeoTable:
    def __init__(self, rundir: Path) -> None:
        self.rundir = rundir
        self.keys: dict[str, int] = {}

    def index(self, parts: tuple[str, ...], engine: str, placed: list[dict[str, Any]],
              depth: int | None = None) -> int | None:
        # The manifest sits at the render root: the folder above
        # <domain>[/<episode>]/<product>/<day>/, or beside a flat picture.  A
        # picture is looked up by its path under that root first; an older
        # record keyed by the flat engine name is matched by that name's tail.
        depth = placement(parts)[4] if depth is None else depth
        root = self.rundir.joinpath(*parts[:-depth])
        manifest = georef_manifest(root)
        panel = manifest["filed"].get("/".join(parts[-depth:]))
        if panel is None:
            panel = manifest["flat"].get(_tail(engine))
        if panel is None:
            return None
        key = json.dumps(panel, sort_keys=True)
        if key not in self.keys:
            self.keys[key] = len(placed)
            placed.append(panel)
        return self.keys[key]


_LEAD = re.compile(r"(?:^|_)f(\d{2,4})(?:_|$)")
_STAMP = re.compile(r"(\d{8})_(\d{2})z", re.I)


def frame_label(name: str) -> str:
    """A short time label from the renderer's file name, or the name itself."""

    stem = Path(name).stem
    lead = _LEAD.search(stem)
    stamp = _STAMP.search(stem)
    parts = []
    if stamp:
        day = stamp.group(1)
        parts.append(f"{day[:4]}-{day[4:6]}-{day[6:]} {stamp.group(2)}:00 UTC run")
    if lead:
        parts.append(f"hour {int(lead.group(1))}")
    return ", ".join(parts) if parts else name


def frame_time(name: str) -> dict[str, Any]:
    """The forecast hour and valid time a renderer file name spells, where it spells them.

    ``..._20260924_12z_f003.png`` is hour 3 of the 12Z run, valid
    ``2026-09-24 15:00 UTC``.  A name without them gives ``None`` for both,
    and the page counts frames instead of clocking them.
    """

    stem = Path(name).stem
    lead = _LEAD.search(stem)
    stamp = _STAMP.search(stem)
    hour = int(lead.group(1)) if lead else None
    valid = None
    if lead and stamp:
        try:
            start = datetime.strptime(stamp.group(1) + stamp.group(2), "%Y%m%d%H")
        except ValueError:
            start = None
        if start is not None:
            valid = utc_text(start + timedelta(hours=hour))
    return {"hour": hour, "valid": valid}


_DX = re.compile(r"-(\d+(?:\.\d+)?)(k?m)$")


def domain_spacing_km(domain: str) -> float | None:
    """``d02-3km`` is 3 km, ``d03-750m`` is 0.75 km: the render folder names the spacing."""

    found = _DX.search(domain or "")
    if not found:
        return None
    value = float(found.group(1))
    return value if found.group(2) == "km" else value / 1000.0


def card(rundir: Path) -> dict[str, Any]:
    """What a run's card on My forecasts shows: its lead picture, domains and grid spacings."""

    found = index(rundir)
    domains = found["domains"]
    return {
        "thumb": found["watch"],
        "pictures": found["count"],
        "domains": domains,
        "dx_km": sorted({dx for dx in (domain_spacing_km(d) for d in domains) if dx is not None}, reverse=True),
    }


def favourite_patterns() -> list[str]:
    try:
        document = json.loads((COPY_DIR / "looks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(item) for item in document.get("favourite_patterns", [])]


def order_products(products: list[str]) -> list[str]:
    """Favourite looks first, in the order the copy file lists them, then the rest."""

    patterns = favourite_patterns()

    def rank(product: str) -> tuple[int, str]:
        for position, pattern in enumerate(patterns):
            if pattern in product.lower():
                return position, product
        return len(patterns), product

    return sorted(products, key=rank)


def favourites(products: list[str]) -> list[str]:
    patterns = favourite_patterns()
    return [product for product in order_products(products)
            if any(pattern in product.lower() for pattern in patterns)]


__all__ = ["GEOREF_MANIFEST", "card", "domain_spacing_km", "favourites", "frame_label", "frame_time", "georef_manifest",
           "index", "order_products", "pictures", "placement"]
