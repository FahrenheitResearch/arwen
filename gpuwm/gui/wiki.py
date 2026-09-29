"""The storm wiki's page store: events, phenomena and places as data, and runs as articles.

Pages are never hand-written HTML.  They are ``gpuwm.wiki.v1`` JSON
documents under ``<root>/wiki/`` (docs/dev/WIKI.md): the seed this
package ships (``wiki/seed.json``, rewritten when the package's copy
changes) and whatever the event atlas or anyone else writes beside it.  A
later document replaces an earlier record with the same id; the seed
loads first.

Every fact on a page names its sources by id, and a page reply carries
every source it cites (and the sources those cite), so the page draws a
footnote for each fact and opens the record behind it.  A run is an
article too: its facts cite the files of its own folder, served by the
run-file endpoint, and it is linked to an event when its box holds the
event and its hours overlap the event's window.  That link is computed
from the folders every time; nothing is written into a run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import re
import sys
import threading
from typing import Any, Callable, Iterable, Mapping

from . import frames, runs
from .files import read_json, utc_text

SCHEMA = "gpuwm.wiki.v1"
WIKI_DIR = "wiki"
SEED_NAME = "seed.json"
SEED_FILE = Path(__file__).resolve().parent / "seed" / "wiki-seed.json"
#: The package's recipe documents (one per seed event): the best run of each event for each card size.
SEED_RECIPES = SEED_FILE.parent / "recipes"
RECIPES_DIR = "recipes"
#: A run folder's link to the event page it was started from, written by the event page's button.
RUN_LINK = "wiki-run.json"
#: The physics composer's checked choice, written beside the plan by New forecast (see api.PHYSICS_CHOICE).
PHYSICS_CHOICE = "gui-physics.json"
#: The store reads at most this many documents, each at most JSON_LIMIT (files.py).
DOCUMENT_LIMIT = 2000
RESULT_LIMIT = 400
RECENT_LIMIT = 16
SEASON_WORDS = {"dec-feb": "December to February", "mar-may": "March to May", "jun-aug": "June to August",
                "sep-nov": "September to November"}
KIND_WORDS = {"basin": "Ocean basins", "country": "Countries", "state": "US states"}
_WORD = re.compile(r"[a-z0-9]+")
#: Words a plain question carries that name nothing in the store ("which storms hit Mexico" asks for "mexico").
STOP_WORDS = frozenset(
    "a an and any are at by did do does during event events for from had has have hit hits how in into is it its "
    "list me near of on or show storm storms struck that the their there these this those to was were what "
    "when where which who with".split())


def _mirror(source: Path, target: Path) -> Path | None:
    data = source.read_bytes()
    try:
        if target.is_file() and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(data).digest():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
    except OSError:
        return None
    return target


def ensure_seed(root: Path) -> Path | None:
    """Put the package's seed at ``<root>/wiki/seed.json`` and its recipes under ``<root>/wiki/recipes/``.

    These are the package's files: a changed copy on disk is replaced,
    and every other document in the folder is left alone.
    """

    if not SEED_FILE.is_file():
        return None
    target = _mirror(SEED_FILE, root / WIKI_DIR / SEED_NAME)
    if SEED_RECIPES.is_dir():
        for path in sorted(SEED_RECIPES.glob("*.json")):
            _mirror(path, root / WIKI_DIR / RECIPES_DIR / path.name)
    return target


def _tokens(text: str) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def query_words(text: str) -> list[str]:
    """The words of a search that can match: common question words dropped, each word once."""
    seen: list[str] = []
    for word in _tokens(text):
        if word not in STOP_WORDS and word not in seen:
            seen.append(word)
    return seen


def _stem(word: str) -> str:
    # "tornadoes" finds "tornado", "cyclones" finds "cyclone"; short words stay whole.
    for end in ("es", "s"):
        if len(word) > 4 and word.endswith(end):
            return word[: -len(end)]
    return word


def _matches(word: str, tokens: Iterable[str]) -> bool:
    stem = _stem(word)
    return any(t == word or t.startswith(word) or t.startswith(stem) for t in tokens)


def _time(text: Any) -> datetime | None:
    if not text:
        return None
    raw = str(text).strip().removesuffix(" UTC").replace(" ", "T").rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _lat(value: float) -> str:
    return f"{abs(value):.2f}°{'N' if value >= 0 else 'S'}"


def _lon(value: float) -> str:
    return f"{abs(_wrap(value)):.2f}°{'E' if _wrap(value) >= 0 else 'W'}"


def _wrap(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


class Store:
    """Every wiki document under ``<root>/wiki``, merged, reloaded when a file changes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._key: tuple[Any, ...] | None = None
        self._data: dict[str, Any] = {}

    def folder(self) -> Path:
        return self.root / WIKI_DIR

    def _files(self) -> list[Path]:
        folder = self.folder()
        if not folder.is_dir():
            return []
        found = sorted(p for p in folder.rglob("*.json") if not any(part.startswith(".") for part in p.parts[-3:]))
        # The seed loads first, so any other document replaces its records.
        found.sort(key=lambda p: (p.name != SEED_NAME or p.parent != folder, str(p)))
        return found[:DOCUMENT_LIMIT]

    def data(self) -> dict[str, Any]:
        files = self._files()
        key = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists())
        with self._lock:
            if key == self._key:
                return self._data
        merged: dict[str, Any] = {"sources": {}, "phenomena": {}, "places": {}, "events": {}, "recipes": {},
                                  "documents": []}
        for path in files:
            document = read_json(path, default=None)
            if not isinstance(document, dict) or document.get("schema") != SCHEMA:
                continue
            origin = str(document.get("origin") or "local")
            relative = path.relative_to(self.folder()).as_posix()
            merged["documents"].append({"file": relative, "origin": origin, "built": document.get("built"),
                                        "events": len(document.get("events") or [])})
            sources = document.get("sources") or {}
            if isinstance(sources, list):
                sources = {item.get("id"): item for item in sources if isinstance(item, dict)}
            for ident, record in sources.items():
                if isinstance(record, dict):
                    merged["sources"][str(ident)] = {**record, "id": str(ident)}
            for group in ("phenomena", "places", "events"):
                for record in document.get(group) or []:
                    if isinstance(record, dict) and record.get("id"):
                        merged[group][str(record["id"])] = {**record, "origin": origin, "document": relative}
            # An event's best run for each card size; a later document replaces an earlier one for that event.
            for record in document.get("recipes") or []:
                if isinstance(record, dict) and record.get("event") and isinstance(record.get("cards"), list):
                    merged["recipes"][str(record["event"])] = {**record, "document": relative}
        for event in merged["events"].values():
            event["_words"] = set(_tokens(" ".join([
                event.get("title", ""), event.get("type", ""), event.get("region", ""),
                str(event.get("decade", "")), (event.get("when") or {}).get("start", "")[:4],
                SEASON_WORDS.get(event.get("season", ""), ""),
                " ".join(merged["places"].get(p, {}).get("title", "") for p in event.get("places") or []),
                " ".join(f.get("text", "") for f in event.get("facts") or [] if isinstance(f, dict)),
                (merged["phenomena"].get(event.get("type"), {}) or {}).get("title", ""),
            ])))
        with self._lock:
            self._key = key
            self._data = merged
        return merged


# ------------------------------------------------------------------ runs as articles

def run_window(rundir: Path) -> dict[str, Any]:
    """What a run covers, from its folder: the box it asked for and its hours."""

    info = runs.status(rundir)
    intent = runs.plan_intent(read_json(rundir / runs.PLAN, default=None))
    start = _time(info.get("start_time")) or _time(intent.get("cycle"))
    seconds = info.get("run_seconds")
    hours = runs.number(intent.get("hours"))
    if not seconds and hours and 0 < hours * 3600.0 < runs.MAX_SECONDS:
        seconds = hours * 3600.0
    region = read_json(rundir / "region.geojson", default=None)
    box = None
    try:
        ring = region["coordinates"][0]
        lons = [float(p[0]) for p in ring]
        lats = [float(p[1]) for p in ring]
        if len(ring) == 5:
            # api.region_polygon's ring: west-south, east-south, east-north, west-north; west > east crosses 180.
            box = {"w": lons[0], "e": lons[1], "s": lats[0], "n": lats[2]}
        else:
            box = {"w": min(lons), "e": max(lons), "s": min(lats), "n": max(lats)}
    except (TypeError, KeyError, IndexError, ValueError):
        box = None
    link = read_json(rundir / RUN_LINK, default=None)
    return {
        "status": info, "intent": intent, "box": box, "link": link if isinstance(link, dict) else None,
        "start": start, "end": None if start is None or not seconds else start + timedelta(seconds=float(seconds)),
    }


def _inside(box: dict[str, float], lon: float, lat: float) -> bool:
    if not box["s"] <= lat <= box["n"]:
        return False
    w, e, x = box["w"], box["e"], _wrap(lon)
    return w <= x <= e if w <= e else (x >= w or x <= e)


def event_points(event: dict[str, Any]) -> list[tuple[float, float, datetime | None]]:
    geometry = event.get("geometry") or {}
    out: list[tuple[float, float, datetime | None]] = []
    for point in geometry.get("track") or []:
        try:
            out.append((float(point[0]), float(point[1]), _time(point[3]) if len(point) > 3 else None))
        except (TypeError, ValueError, IndexError):
            continue
    for point in geometry.get("path") or []:
        try:
            if float(point[0]) or float(point[1]):
                out.append((float(point[0]), float(point[1]), None))
        except (TypeError, ValueError, IndexError):
            continue
    where = event.get("where") or {}
    if not out and where.get("lat") is not None and where.get("lon") is not None:
        out.append((float(where["lon"]), float(where["lat"]), None))
    return out


def run_covers(window: dict[str, Any], event: dict[str, Any]) -> bool:
    """A run models an event when the event page started it, or its box holds a point of the event while its hours
    overlap the event's."""

    if (window.get("link") or {}).get("event") == event.get("id"):
        return True
    if window["box"] is None or window["start"] is None or window["end"] is None:
        return False
    when = event.get("when") or {}
    start, end = _time(when.get("start")), _time(when.get("end")) or _time(when.get("start"))
    if start is None or end is None:
        return False
    if window["end"] < start or window["start"] > end:
        return False
    for lon, lat, moment in event_points(event):
        if moment is not None and not (window["start"] - timedelta(hours=3) <= moment <= window["end"] + timedelta(hours=3)):
            continue
        if _inside(window["box"], lon, lat):
            return True
    return False


# ------------------------------------------------------------------ the pages

def _cited(store: dict[str, Any], ids: Iterable[str]) -> dict[str, Any]:
    """The sources named, and the sources those name (a computed record's inputs, a row's dataset)."""

    out: dict[str, Any] = {}
    todo = [str(i) for i in ids if i]
    while todo:
        ident = todo.pop()
        if ident in out:
            continue
        record = store["sources"].get(ident)
        if record is None:
            out[ident] = {"id": ident, "missing": True}
            continue
        out[ident] = record
        todo.extend(str(i) for i in record.get("inputs") or [])
        if record.get("dataset"):
            todo.append(str(record["dataset"]))
    return out


def _cites_of(value: Any) -> list[str]:
    """Every source id a record names, wherever it names one (``cite``, ``*_cite``)."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "cite" or key.endswith("_cite"):
                found.extend([item] if isinstance(item, str) else [str(i) for i in item or []])
            else:
                found.extend(_cites_of(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_cites_of(item))
    return found


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def rarity_share(rarity: Any) -> float | None:
    """The share of its place's events, in percent, at least as strong as this one: what events rank by.

    A raw count does not rank: 1 of 3 South Atlantic storms is common there, 8 of 4484 Oklahoma tornadoes is not.
    """

    if not isinstance(rarity, dict):
        return None
    try:
        count, of = float(rarity["count"]), float(rarity["of"])
    except (KeyError, TypeError, ValueError):
        return None
    return None if of <= 0 else round(100.0 * count / of, 4)


def _rarity_key(row: dict[str, Any]) -> tuple[float, str]:
    share = row.get("rarity_pct")
    return (share if share is not None else 1e9, row.get("title") or "")


def event_row(store: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    rarity = event.get("rarity") or {}
    kind = store["phenomena"].get(event.get("type")) or {}
    return {
        "kind": "event", "id": event["id"], "title": event.get("title") or event["id"],
        "type": event.get("type"), "type_title": kind.get("title") or event.get("type"),
        "region": event.get("region"), "decade": event.get("decade"), "season": event.get("season"),
        "season_words": SEASON_WORDS.get(event.get("season", ""), ""),
        "start": (event.get("when") or {}).get("start"), "rarity": rarity.get("count"), "rarity_of": rarity.get("of"),
        "rarity_pct": rarity_share(rarity),
        "rarity_text": rarity.get("text"), "places": event.get("places") or [], "seed": bool(event.get("seed")),
        "origin": event.get("origin"), "added": event.get("added"),
        "where": event.get("where"),
    }


class Wiki:
    """The wiki's answers, over the store and the runs root."""

    def __init__(self, root: Path, offered: Callable[[], Mapping[str, str] | None] | None = None) -> None:
        self.root = root
        self.store = Store(root)
        #: The sources New forecast can start from (every route the engine drives from a point and a date) as id to display name, or None.
        self.offered = offered or (lambda: None)
        self._windows: dict[str, tuple[Any, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # ---- runs

    def _run_windows(self) -> list[tuple[str, Path, dict[str, Any]]]:
        out = []
        for path in runs.iter_runs(self.root):
            run_id = path.relative_to(self.root).as_posix()
            key = (runs.folder_mtime(path),)
            with self._lock:
                hit = self._windows.get(run_id)
            if hit is None or hit[0] != key or hit[1]["status"].get("state") == "running":
                try:
                    window = run_window(path)
                except Exception as error:  # noqa: BLE001 - one unreadable folder never takes the wiki down
                    print(f"gpuwm gui: could not read the run folder {path}: {error!r}", file=sys.stderr, flush=True)
                    continue
                with self._lock:
                    self._windows[run_id] = (key, window)
            else:
                window = hit[1]
            out.append((run_id, path, window))
        return out

    def runs_of(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for run_id, path, window in self._run_windows():
            if run_covers(window, event):
                out.append(self._run_row(run_id, path, window))
        return out

    def events_of(self, window: dict[str, Any]) -> list[dict[str, Any]]:
        store = self.store.data()
        return [event_row(store, e) for e in store["events"].values() if run_covers(window, e)]

    def _run_row(self, run_id: str, path: Path, window: dict[str, Any]) -> dict[str, Any]:
        found = frames.card(path)
        info = window["status"]
        link = window.get("link") or {}
        return {"kind": "run", "id": run_id, "title": link.get("title") or run_id, "state": info.get("state"),
                "start": info.get("start_time"), "hours": None if not info.get("run_seconds")
                else round(float(info["run_seconds"]) / 3600.0, 2),
                "thumb": found["thumb"], "pictures": found["pictures"], "dx_km": found["dx_km"],
                "updated": info.get("updated_utc"), "source": info.get("source"),
                "event": link.get("event"), "card_gb": link.get("card_gb")}

    # ---- pages

    def main(self) -> dict[str, Any]:
        store = self.store.data()
        events = [event_row(store, e) for e in store["events"].values()]
        kinds = []
        for ident, kind in store["phenomena"].items():
            kinds.append({"id": ident, "title": kind.get("title"), "plural": kind.get("plural") or kind.get("title"),
                          "count": sum(1 for e in events if e["type"] == ident), "seed": bool(kind.get("seed"))})
        places = []
        for ident, place in store["places"].items():
            count = sum(1 for e in events if ident in e["places"])
            places.append({"id": ident, "title": place.get("title"), "kind": place.get("kind"), "count": count})
        places.sort(key=lambda p: (-p["count"], p["title"] or ""))
        # The featured event: the rarest for its place, turned daily among the rarest few, so it changes.
        ranked = sorted((e for e in events if e["rarity_pct"] is not None), key=_rarity_key)
        rare = ranked[:6]
        # Featured events are the rarest few whose best run this page can start, so its button runs from here.
        offered = self._offered()
        runnable = [e for e in ranked if self._runnable(store, e["id"], offered)][:6]
        pool = runnable or rare
        featured = pool[datetime.now(timezone.utc).timetuple().tm_yday % len(pool)] if pool else None
        return {
            "counts": {"events": len(events), "seed": sum(1 for e in events if e["seed"]),
                       "places": len(places), "sources": len(store["sources"])},
            "phenomena": kinds, "places": places[:40],
            "place_kinds": KIND_WORDS,
            "featured": None if featured is None else self.event_page(featured["id"], with_runs=False),
            "rare": rare, "changes": self.changes(limit=RECENT_LIMIT)["changes"],
            "documents": store["documents"],
        }

    @staticmethod
    def _runnable(store: dict[str, Any], ident: str, offered: Mapping[str, str] | set[str] | None) -> bool:
        best = store["recipes"].get(ident)
        if best is not None:
            return bool(offered) and any(row.get("fits") and row.get("source") in offered for row in best["cards"])
        return recipe_runnable((store["events"].get(ident) or {}).get("recipe"), offered)

    def _offered(self) -> Mapping[str, str] | None:
        try:
            return self.offered()
        except Exception:  # noqa: BLE001 - an engine that cannot answer offers nothing; New forecast says why
            return None

    def event_page(self, ident: str, *, with_runs: bool = True) -> dict[str, Any]:
        store = self.store.data()
        event = store["events"].get(ident)
        if event is None:
            raise FileNotFoundError(f"No event page {ident!r}.")
        record = _public(event)
        offered = self._offered()
        kind = store["phenomena"].get(event.get("type")) or {}
        places = [{"id": p, "title": (store["places"].get(p) or {}).get("title") or p,
                   "kind": (store["places"].get(p) or {}).get("kind")} for p in event.get("places") or []]
        rows = [event_row(store, e) for e in store["events"].values() if e["id"] != ident]
        mine = event_row(store, event)
        # See also: the same kind in the same place first, then the same kind, then the same place.
        same_place = set(mine["places"])

        def near(row: dict[str, Any]) -> tuple[int, int]:
            shared = bool(same_place & set(row["places"]))
            return (0 if shared and row["type"] == mine["type"] else 1 if row["type"] == mine["type"] else 2
                    if shared else 3, abs((row["decade"] or 0) - (mine["decade"] or 0)))

        see_also = sorted((r for r in rows if r["type"] == mine["type"] or same_place & set(r["places"])),
                          key=near)[:6]
        page = {
            "kind": "event", "event": record, "row": mine,
            "phenomenon": {"id": event.get("type"), "title": kind.get("title") or event.get("type"),
                           "plural": kind.get("plural")},
            "places": places, "see_also": see_also,
            "sources": _cited(store, _cites_of(record) + _cites_of(store["recipes"].get(ident) or {})),
            "categories": self._categories(mine),
            "runnable": self._runnable(store, ident, offered),
            "best": best_runs(store["recipes"].get(ident), offered),
        }
        if with_runs:
            page["runs"] = self.runs_of(event)
        return page

    @staticmethod
    def _categories(row: dict[str, Any]) -> list[dict[str, str]]:
        cats = [{"label": row["type_title"] or row["type"], "type": row["type"] or ""}]
        if row["region"]:
            cats.append({"label": row["region"], "region": row["region"]})
        if row["decade"]:
            cats.append({"label": f"{row['decade']}s", "decade": str(row["decade"])})
        if row["season"]:
            cats.append({"label": row["season_words"], "season": row["season"]})
        if row["seed"]:
            cats.append({"label": "From public records", "seed": "1"})
        return cats

    def phenomenon_page(self, ident: str) -> dict[str, Any]:
        store = self.store.data()
        kind = store["phenomena"].get(ident)
        if kind is None:
            raise FileNotFoundError(f"No phenomenon page {ident!r}.")
        events = [event_row(store, e) for e in store["events"].values() if e.get("type") == ident]
        record = _public(kind)
        return {"kind": "phenomenon", "phenomenon": record, "events": events,
                "sources": _cited(store, _cites_of(record)), "facets": self._facets(events),
                "others": [{"id": k, "title": v.get("title")} for k, v in store["phenomena"].items() if k != ident]}

    def place_page(self, ident: str) -> dict[str, Any]:
        store = self.store.data()
        place = store["places"].get(ident)
        if place is None:
            raise FileNotFoundError(f"No place page {ident!r}.")
        events = [event_row(store, e) for e in store["events"].values() if ident in (e.get("places") or [])]
        # The obscure ones first: the fewest events at least as strong in their place and season.
        events.sort(key=_rarity_key)
        parent = store["places"].get(place.get("parent") or "")
        children = [{"id": k, "title": v.get("title"), "kind": v.get("kind")}
                    for k, v in store["places"].items() if v.get("parent") == ident]
        record = _public(place)
        return {"kind": "place", "place": record, "events": events,
                "parent": None if parent is None else {"id": parent["id"], "title": parent.get("title")},
                "children": sorted(children, key=lambda c: c["title"] or ""),
                "sources": _cited(store, _cites_of(record)), "kind_words": KIND_WORDS}

    def places_index(self) -> dict[str, Any]:
        store = self.store.data()
        groups: dict[str, list[dict[str, Any]]] = {}
        for ident, place in store["places"].items():
            count = sum(1 for e in store["events"].values() if ident in (e.get("places") or []))
            groups.setdefault(place.get("kind") or "other", []).append(
                {"id": ident, "title": place.get("title"), "count": count, "seed": bool(place.get("seed"))})
        return {"groups": [{"kind": k, "title": KIND_WORDS.get(k, k.title()),
                            "places": sorted(v, key=lambda p: p["title"] or "")} for k, v in sorted(groups.items())]}

    def run_page(self, run_id: str, rundir: Path) -> dict[str, Any]:
        window = run_window(rundir)
        info = window["status"]
        intent = window["intent"]
        found = frames.index(rundir)
        card = frames.card(rundir)
        manifest = read_json(rundir / runs.MANIFEST, default=None)
        sources: dict[str, Any] = {}

        def run_file(name: str, title: str) -> str | None:
            if not (rundir / name).is_file():
                return None
            ident = f"run-file:{name}"
            sources[ident] = {"id": ident, "kind": "run-file", "title": title, "run": run_id, "path": name}
            return ident

        plan_src = run_file(runs.PLAN, "The run's plan (plan.json)")
        progress_src = run_file(runs.HEARTBEAT, "The run's heartbeat (run-progress.json)")
        events_src = run_file(runs.EVENTS, "The run's event log (events.jsonl)")
        region_src = run_file("region.geojson", "The box the run asked for (region.geojson)")
        manifest_src = run_file(runs.MANIFEST, "The run's manifest (run-manifest.json)")
        link_src = run_file(RUN_LINK, "The event page this run was started from (wiki-run.json)")
        link = window.get("link") or {}
        tree_src = None
        if found["count"]:
            tree_src = "run-tree"
            sources[tree_src] = {"id": tree_src, "kind": "run-tree", "title": "The run's pictures, as the Rust "
                                 "renderer filed them (<domain>/<product>/<day>)", "run": run_id,
                                 "folders": sorted({g["folder"] for g in found["groups"]})[:12]}
        facts = []

        def add(ident: str, label: str, text: Any, cite: list[str | None]) -> None:
            if text in (None, "", []):
                return
            facts.append({"id": ident, "label": label, "text": str(text), "cite": [c for c in cite if c],
                          "infobox": True})

        event = self.store.data()["events"].get(str(link.get("event") or ""))
        if event is not None:
            add("event", "Made from", event.get("title"), [link_src])
        if link.get("card_gb"):
            add("layout", "Best run for", f"a {link['card_gb']} GB card", [link_src])
        add("state", "State", (info.get("state") or "").capitalize(), [progress_src or events_src])
        if info.get("state") == "failed":
            # The map page shows a failed run's reason and remedy; the article said only "Failed".
            end = info.get("end") if isinstance(info.get("end"), dict) else {}
            add("reason", "Why it stopped", end.get("message"), [events_src or progress_src])
            add("remedy", "What to do", end.get("remedy"), [events_src or progress_src])
        # A plan is written by whoever wrote it: each value is read by its type, so one of the wrong kind leaves
        # its fact out instead of the run's whole article answering 500.
        source_id = intent.get("source") if isinstance(intent.get("source"), str) else info.get("source")
        names = self._offered() or {}
        # The source's display name, as New forecast shows it; the cited plan file holds the id itself.
        add("source", "Starting data", (names.get(source_id) or str(source_id).upper()) if source_id else None,
            [plan_src])
        cycle = _time(intent.get("cycle")) if isinstance(intent.get("cycle"), str) else None
        add("start", "Start", info.get("start_time") or (utc_text(cycle) if cycle else None), [events_src or plan_src])
        planned_hours = runs.number(intent.get("hours"))
        if info.get("run_seconds") or planned_hours:
            hours = float(info.get("run_seconds") or 0) / 3600.0 or planned_hours
            add("length", "Length", f"{hours:g} h", [events_src if info.get("run_seconds") else plan_src])
        root_dx = runs.number(intent.get("root_dx_km"))
        resolved = runs.event_facts(rundir / runs.EVENTS) if (rundir / runs.EVENTS).is_file() else {}
        # A downscale's grid is read from the plan gpuwm downscale wrote, not from its event log.
        child_src = (run_file(runs.DOWNSCALE_PLAN, "The downscale's plan (downscale-plan.json)")
                     if not (resolved.get("grid") or {}).get("domains") and runs.downscale_plan(rundir) else None)
        grids_km = (info.get("grids") or {}).get("grids_km") or []
        if card["dx_km"]:
            add("grid", "Grid", " / ".join(f"{dx:g} km" for dx in card["dx_km"]), [tree_src])
        elif root_dx and root_dx > 0:
            add("grid", "Grid", f"{root_dx:g} km", [plan_src])
        elif grids_km:
            add("grid", "Grid", " / ".join(f"{dx:g} km" for dx in grids_km), [child_src or events_src])
        # The level count the run resolved to (its event log or a downscale's plan), or the one its plan asked for
        # before that.
        levels = resolved.get("levels") or []
        planned_nz = runs.number(intent.get("nz"))
        if levels:
            add("levels", "Vertical levels", " / ".join(str(n) for n in levels), [child_src or events_src])
        elif planned_nz and planned_nz > 0:
            add("levels", "Vertical levels", f"{planned_nz:g}", [plan_src])
        # The physics the run was started with: the composer's checked choice when New forecast made one, else
        # the suite or the picked schemes its plan names; a plan that names neither runs its source's default suite.
        choice = read_json(rundir / PHYSICS_CHOICE, default=None)
        choice_src = run_file(PHYSICS_CHOICE, "The physics chosen in New forecast (gui-physics.json)")
        if isinstance(choice, dict) and (choice.get("suite") or choice.get("choices")):
            resolved = choice.get("resolved") or {}
            schemes = ", ".join(f"{family.replace('_', ' ')} {scheme}" for family, scheme in sorted(resolved.items()))
            name = choice.get("label") or choice.get("suite") or "Picked schemes, no named set"
            add("physics", "Physics", name + (f" ({schemes})" if schemes else ""), [choice_src, plan_src])
        elif isinstance(intent.get("physics_choices"), dict) and intent["physics_choices"]:
            picked = ", ".join(f"{family.replace('_', ' ')} {scheme}"
                               for family, scheme in sorted(dict(intent["physics_choices"]).items()))
            add("physics", "Physics", f"Picked schemes, no named set ({picked})", [plan_src])
        elif isinstance(intent.get("physics_profile"), str) and intent["physics_profile"]:
            add("physics", "Physics", intent["physics_profile"], [plan_src])
        elif intent:
            add("physics", "Physics", "The source's default set", [plan_src])
        if window["box"]:
            b = window["box"]
            add("box", "Box", f"{_lat(b['s'])} to {_lat(b['n'])}, {_lon(b['w'])} to {_lon(b['e'])}", [region_src])
        add("domains", "Domains", ", ".join(found["domains"]), [tree_src])
        add("pictures", "Pictures", f"{found['count']:,}" if found["count"] else None, [tree_src])
        if isinstance(manifest, dict):
            provenance = manifest.get("provenance")
            version = provenance.get("code_version") if isinstance(provenance, dict) else None
            add("version", "Engine version", version, [manifest_src])
        verification = []
        for pattern in ("verification*.json", "verification/*.json", "verification/*.png", "verify/*.json"):
            for path in sorted(rundir.glob(pattern))[:20]:
                verification.append(path.relative_to(rundir).as_posix())
        return {
            "kind": "run", "id": run_id, "title": link.get("title") or run_id, "status": info, "facts": facts,
            "sources": sources, "event": None if event is None else {"id": event["id"], "title": event.get("title")},
            "box": window["box"], "pictures": {"count": found["count"], "products": found["products"],
                                               "domains": found["domains"], "favourites": frames.favourites(found["products"]),
                                               "groups": found["groups"]},
            "events": self.events_of(window), "verification": verification,
        }

    # ---- search and lists

    @staticmethod
    def _facets(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        def count(key: str) -> list[dict[str, Any]]:
            seen: dict[str, int] = {}
            for row in rows:
                value = row.get(key)
                if value in (None, ""):
                    continue
                seen[str(value)] = seen.get(str(value), 0) + 1
            return [{"value": k, "count": v} for k, v in sorted(seen.items())]

        return {"region": count("region"), "decade": count("decade"), "season": count("season"),
                "type": count("type")}

    def search(self, query: dict[str, list[str]]) -> dict[str, Any]:
        def one(key: str) -> str:
            return ((query.get(key) or [""])[0] or "").strip()

        store = self.store.data()
        words = query_words(one("q"))
        results = []
        for event in store["events"].values():
            row = event_row(store, event)
            if one("type") and row["type"] != one("type"):
                continue
            if one("region") and row["region"] != one("region"):
                continue
            if one("decade") and str(row["decade"]) != one("decade"):
                continue
            if one("season") and row["season"] != one("season"):
                continue
            if one("place") and one("place") not in row["places"]:
                continue
            if one("seed") and not row["seed"]:
                continue
            if words:
                title = set(_tokens(row["title"]))
                hits = [w for w in words if _matches(w, event["_words"])]
                if not hits:
                    continue
                row["hits"] = len(hits)
                row["score"] = sum(3 if _matches(w, title) else 1 for w in hits)
            results.append(row)
        if not one("type") and not one("region") and not one("decade") and not one("season") and not one("seed"):
            for kind in ("phenomena", "places"):
                for ident, record in store[kind].items():
                    title = record.get("title") or ident
                    toks = set(_tokens(title + " " + (record.get("plural") or "") + " " + (record.get("kind") or "")))
                    hits = sum(1 for w in words if _matches(w, toks))
                    if hits:
                        results.append({"kind": "phenomenon" if kind == "phenomena" else "place", "id": ident,
                                        "title": title, "score": 10, "hits": hits, "place_kind": record.get("kind"),
                                        "seed": bool(record.get("seed"))})
            for run_id, path, window in self._run_windows():
                toks = set(_tokens(run_id + " " + str(window["status"].get("source") or "")))
                hits = sum(1 for w in words if _matches(w, toks))
                if hits:
                    results.append({**self._run_row(run_id, path, window), "score": 5, "hits": hits})
        order = one("sort") or ("score" if words else "rarity")
        if order == "rarity":
            results.sort(key=_rarity_key)
        elif order == "date":
            results.sort(key=lambda r: r.get("start") or "", reverse=True)
        elif order == "title":
            results.sort(key=lambda r: r["title"])
        else:
            results.sort(key=lambda r: (-(r.get("hits") or 0), -(r.get("score") or 0), _rarity_key(r)))
        # Pages that match every word lead; when none does, the page says these match some of the words.
        partial = bool(words) and bool(results) and not any((r.get("hits") or 0) >= len(words) for r in results)
        events = [r for r in results if r["kind"] == "event"]
        return {"query": {k: one(k) for k in ("q", "type", "region", "decade", "season", "place", "seed", "sort")},
                "words": words, "partial": partial,
                "results": results[:RESULT_LIMIT], "count": len(results), "facets": self._facets(events),
                "phenomena": [{"id": k, "title": v.get("title")} for k, v in store["phenomena"].items()]}

    def changes(self, limit: int = 60) -> dict[str, Any]:
        store = self.store.data()
        rows = []
        for event in store["events"].values():
            rows.append({"what": "event", "id": event["id"], "title": event.get("title"),
                         "when": event.get("added") or "", "origin": event.get("origin"),
                         "seed": bool(event.get("seed")), "type": event.get("type")})
        for run_id, path, window in self._run_windows():
            info = window["status"]
            when = info.get("updated_utc") or ""
            if not when:
                try:
                    when = datetime.fromtimestamp(runs.folder_mtime(path), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                except (OSError, ValueError, OverflowError):
                    when = ""
            rows.append({"what": "run", "id": run_id, "title": (window.get("link") or {}).get("title") or run_id,
                         "when": when, "state": info.get("state"),
                         "events": [e["id"] for e in self.events_of(window)]})
        rows.sort(key=lambda r: str(r["when"]), reverse=True)
        return {"changes": rows[:limit]}


#: A recipe row's fields the event page shows; the engine-facing ones (intent, args) stay on the server.
ROW_FIELDS = ("card_gb", "fits", "why", "tried", "rung", "domains", "start", "length_h", "source", "source_why",
              "physics", "fitted_grid", "memory", "disk_gib", "disk_basis", "est_minutes", "est_total_minutes", "est_kind", "est_basis",
              "what_is_given_up", "finest_km", "door", "box", "cite", "better_rungs_refused")


def best_runs(recipe: dict[str, Any] | None, offered: Mapping[str, str] | set[str] | None) -> dict[str, Any] | None:
    """An event's best run for each card size, as the event page draws it; None when the event has none."""

    if not isinstance(recipe, dict):
        return None
    names = offered if isinstance(offered, Mapping) else {}
    cards = []
    for row in sorted(recipe.get("cards") or [], key=lambda r: float(r.get("card_gb") or 0)):
        public = {key: row[key] for key in ROW_FIELDS if key in row}
        public["runnable"] = bool(row.get("fits")) and bool(offered) and row.get("source") in offered
        public["source_name"] = names.get(row.get("source")) or str(row.get("source") or "").upper()
        cards.append(public)
    return {key: recipe.get(key) for key in ("event", "title", "type", "key_hours", "ideal", "start", "length_h",
                                             "source", "cite")} | {"cards": cards}


def recipe_row(recipe: dict[str, Any] | None, card_gb: float | None) -> dict[str, Any] | None:
    """The row for a card size: that size's own row, or the biggest one that fits when no size is named."""

    if not isinstance(recipe, dict):
        return None
    rows = [row for row in recipe.get("cards") or [] if row.get("fits")]
    if card_gb is None:
        return max(rows, key=lambda r: float(r["card_gb"]), default=None)
    return next((row for row in rows if float(row["card_gb"]) == float(card_gb)), None)


def recipe_runnable(recipe: Any, offered: Mapping[str, str] | set[str] | None) -> bool:
    """Whether New forecast can start this recipe as written: its source is one New forecast offers."""
    return isinstance(recipe, dict) and bool(offered) and recipe.get("source") in offered


#: A best run's intent keys that New forecast has a control for, and the name each goes by in its draft.  The
#: button runs the intent itself, so these are read from it and never from a second copy of the same values; every
#: other intent key is carried to the plan by the server (``gpuwm.gui.api.DRAFT_INTENT_KEYS``).
PAGE_FIELDS = {"source": "source", "cycle": "cycle", "hours": "hours", "card": "card", "root_dx_km": "dx_km",
               "nz": "nz", "chain": "chain", "buffer_km": "buffer_km", "era5_provider": "era5_provider",
               "physics_profile": "profile", "forecast_start_hour": "start_hour"}

#: A best run's grid fields New forecast shows beside its own fit.
LAYOUT_DOMAIN_FIELDS = ("grid_id", "parent_id", "dx_km", "nx", "ny", "nz", "width_km", "height_km",
                        "parent_grid_ratio", "following")


def best_layout(row: dict[str, Any]) -> dict[str, Any]:
    """What an event's best run runs besides its box and start, as New forecast shows it: grids, physics, output."""

    domains = [{key: d.get(key) for key in LAYOUT_DOMAIN_FIELDS if key in d}
               for d in row.get("domains") or [] if isinstance(d, dict)]
    physics = row.get("physics") if isinstance(row.get("physics"), dict) else {}
    # How often the run writes, as a plan that keeps the layout carries it: a cyclone setup's own arguments carry
    # the row's intervals, and every other row's are the intent keys the server carries into the plan
    # (``gpuwm.gui.api.CreateMixin._event_keys``).  A row whose plan carries none says none, so Review never names
    # intervals the plan would not write.
    carried = row.get("output") if row.get("door") == "cyclone-setup" else row.get("intent")
    output = carried if isinstance(carried, dict) else {}
    return {"domains": domains, "physics": {key: physics.get(key) for key in ("profile", "why") if physics.get(key)},
            "output": {key: output[key] for key in ("history_interval_s", "nest_history_interval_s") if key in output},
            "fitted_grid": row.get("fitted_grid"), "disk_gib": row.get("disk_gib"), "door": row.get("door")}


def recipe_of(store: Store, ident: str, offered: Mapping[str, str] | set[str] | None = None,
              card_gb: float | None = None) -> dict[str, Any]:
    """An event's recipe for New forecast, from its best run for ``card_gb`` (the biggest one when None).

    The row's own words for New forecast carry over whole, nests
    included (``chain``, ``buffer_km``), so Customise starts from exactly
    what the event page's button runs: every value New forecast has a
    control for is read from the intent the button runs (:data:`PAGE_FIELDS`),
    and ``layout`` says what the run's grids, physics and output are.  An
    event with no best runs falls back to its single ``recipe``.
    ``runnable`` is false when New forecast does not offer the recipe's
    source.  Then only the box, length and grid carry over:
    ``start_cycle`` and ``start_source`` are None, because another source
    need not hold the event's date (New forecast's date picker says which
    sources do).
    """
    data = store.data()
    event = data["events"].get(ident)
    row = recipe_row(data["recipes"].get(ident), card_gb)
    if event is not None and row is not None and isinstance(row.get("recipe"), dict):
        intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
        recipe = {**row["recipe"], **{name: intent[key] for key, name in PAGE_FIELDS.items() if key in intent},
                  "card_gb": row["card_gb"], "rung": row.get("rung"), "layout": best_layout(row)}
        if row.get("door") == "cyclone-setup" and isinstance(row.get("args"), dict):
            # The cyclone setup exactly as the event page's button runs it, history intervals included: they are
            # what keeps the event's run inside the card's disk budget.  New forecast replaces only the source,
            # start, length and card.
            recipe["cyclone_setup"] = dict(row["args"])
    elif event is None or not isinstance(event.get("recipe"), dict):
        raise FileNotFoundError(f"No recipe for {ident!r}.")
    else:
        recipe = event["recipe"]
    runnable = recipe_runnable(recipe, offered)
    return {"event": ident, "title": event.get("title"), **recipe, "runnable": runnable,
            "start_cycle": recipe.get("cycle") if runnable else None,
            "start_source": recipe.get("source") if runnable else None}


__all__ = ["PAGE_FIELDS", "RUN_LINK", "SCHEMA", "SEED_FILE", "Store", "WIKI_DIR", "Wiki", "best_layout", "best_runs",
           "ensure_seed", "event_points", "query_words", "rarity_share", "recipe_of", "recipe_row", "recipe_runnable",
           "run_covers", "run_window"]
