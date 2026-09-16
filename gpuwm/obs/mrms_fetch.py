"""One MRMS composite cache, shared by everything that needs a scan.

Two callers want the same bytes: the campaign driver
``tools/obs_fetch_mrms.py``, which pulls a whole case window ahead of a
scoring pass, and a local DA run, which wants the one scan nearest a lead's
valid time while the run is still going.  Before this module they would have
been two pieces of code that both drove ``rw_mrms``, and two drivers of one
archive is two answers to "which object was taken" waiting to happen.

So the listing, the selection, the fetch, the decode and the geometry pack
live here once, behind a cache that is keyed by what was ASKED for.  A
rescore of a case that was already scored touches the network not at all:
the request-to-frame resolution is recorded beside the packs, so the second
pass reads the same pack the first pass read, by the same digest.

**Three outcomes, kept apart.**  A window that holds a frame yields it.  A
window the archive lists as empty raises :class:`MrmsFrameAbsent` -- there is
no observation at that instant and there never will be.  An archive that
could not be asked at all raises :class:`MrmsArchiveUnavailable` -- there may
well be an observation and this box could not see it.  Collapsing the last
two would turn a dropped network into a permanent hole in a receipt, which
is the difference between "we do not know" and "there was nothing there".

The cache names the archive object it took: bucket, key and the ``s3://``
URI, beside the object's own SHA-256 as the front door recorded it at fetch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

CACHE_SCHEMA = "arwen.mrms-composite-cache.v1"

#: Default product and bucket, the front door's own defaults, repeated here
#: so a receipt can quote them without running the binary.
DEFAULT_PRODUCT = "MergedReflectivityQCComposite_00.50"
DEFAULT_BUCKET = "noaa-mrms-pds"
DEFAULT_REGION = "CONUS"

#: Matching window around a requested valid time, in seconds.
DEFAULT_WINDOW_SECONDS = 240

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
_DOOR_TIME = "%Y-%m-%dT%H:%M:%SZ"

#: The stderr line the front door prints when its window holds no object.
#: Used only to classify a refusal the door already made; the authority on
#: emptiness is a successful ``list`` whose ``matched_frames`` is zero, which
#: is what :meth:`MrmsCompositeCache.ensure` falls back to.
_ABSENT_MARKER = "no MRMS frame within"


class MrmsFrameAbsent(LookupError):
    """The archive holds no composite inside the matching window."""


class MrmsArchiveUnavailable(RuntimeError):
    """The archive could not be asked: no network, or no front door."""


@dataclass(frozen=True)
class MrmsFrame:
    """One cached composite: what it is, where it came from, what it hashes to."""

    requested_valid_time: str
    valid_time: str
    offset_seconds: float
    bucket: str
    key: str
    object_uri: str
    object_sha256: str
    fetched_at: str
    pack_path: str
    pack_sha256: str
    observed_fraction: float

    def record(self) -> dict[str, object]:
        return {
            "requested_valid_time": self.requested_valid_time,
            "valid_time": self.valid_time,
            "offset_seconds": float(self.offset_seconds),
            "archive_bucket": self.bucket,
            "archive_key": self.key,
            "archive_object_uri": self.object_uri,
            "archive_object_sha256": self.object_sha256,
            "fetched_at": self.fetched_at,
            "pack_path": self.pack_path,
            "pack_sha256": self.pack_sha256,
            "observed_fraction": float(self.observed_fraction),
        }


def _parse(value: str) -> datetime:
    return datetime.strptime(str(value).replace("Z", ""), _TIME_FORMAT)


def _format(value: datetime) -> str:
    return value.strftime(_TIME_FORMAT)


def _door_time(value: str) -> str:
    return _parse(value).strftime(_DOOR_TIME)


def bbox_argument(west: float, south: float, east: float, north: float) -> str:
    """The front door's ``W,S,E,N`` spelling of a lon/lat box."""
    return f"{float(west):.4f},{float(south):.4f},{float(east):.4f},{float(north):.4f}"


def bbox_around(latitude, longitude, *, margin_deg: float = 0.25) -> str:
    """A decode box that covers a model grid with a margin.

    The full CONUS composite is 7000 x 3500 float64 cells, 196 MB a frame.
    A local DA domain is a couple of hundred kilometres across, so the box is
    not an optimisation: decoding CONUS per lead would cost more memory than
    the forecast that produced the field being scored.
    """
    import numpy as np

    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if lat.size == 0 or lon.size == 0:
        raise ValueError("a decode box needs a non-empty grid")
    return bbox_argument(float(lon.min()) - margin_deg,
                         float(lat.min()) - margin_deg,
                         float(lon.max()) + margin_deg,
                         float(lat.max()) + margin_deg)


class MrmsCompositeCache:
    """Composite frames for one case, fetched once and read many times.

    ``root`` is a directory inside the case, so a case carries the
    observations it was scored against and a rescore refetches nothing.
    """

    def __init__(self, root, *, bbox: str | None = None,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 product: str | None = None, bucket: str | None = None,
                 region: str | None = None, binary=None,
                 offline: bool = False) -> None:
        if int(window_seconds) <= 0:
            raise ValueError("the matching window is a positive number of seconds")
        self.root = Path(root)
        self.bbox = None if bbox is None else str(bbox)
        self.window_seconds = int(window_seconds)
        self.product = str(product or DEFAULT_PRODUCT)
        self.bucket = str(bucket or DEFAULT_BUCKET)
        self.region = str(region or DEFAULT_REGION)
        self.offline = bool(offline)
        self._binary = None if binary is None else Path(binary)
        self._index: dict[str, object] | None = None

    # -- layout ---------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    @property
    def packs_dir(self) -> Path:
        return self.root / "packs"

    @property
    def objects_dir(self) -> Path:
        return self.root / "objects"

    @property
    def geometry_path(self) -> Path:
        return self.packs_dir / "geometry.obspack"

    def pack_paths(self) -> list[Path]:
        """Every decoded frame this cache holds, ascending by valid time."""
        frames = self._read_index()["frames"]
        return [Path(frames[key]["pack_path"]) for key in sorted(frames)
                if Path(frames[key]["pack_path"]).is_file()]

    # -- the index ------------------------------------------------------

    def _read_index(self) -> dict[str, object]:
        if self._index is None:
            if self.index_path.is_file():
                value = json.loads(self.index_path.read_text(encoding="utf-8"))
                if value.get("schema") != CACHE_SCHEMA:
                    raise ValueError(
                        f"{self.index_path} is not an {CACHE_SCHEMA} index; "
                        f"a cache written under another contract cannot be "
                        f"read as this one")
                for field, mine in (("bucket", self.bucket),
                                    ("region", self.region),
                                    ("product", self.product),
                                    ("bbox", self.bbox)):
                    if value.get(field) != mine:
                        raise ValueError(
                            f"{self.index_path} was written for {field} "
                            f"{value.get(field)!r} and this cache asks for "
                            f"{mine!r}; the packs already in it were decoded "
                            f"under the first one, so reusing them would "
                            f"score one domain against another's cells. Use "
                            f"a separate cache directory for the new setting")
            else:
                value = {"schema": CACHE_SCHEMA, "bucket": self.bucket,
                         "region": self.region, "product": self.product,
                         "bbox": self.bbox, "frames": {}, "requests": {},
                         "geometry": None}
            self._index = value
        return self._index

    def _write_index(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self._read_index(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    def _request_key(self, valid_time: str) -> str:
        return f"{_format(_parse(valid_time))}|{self.window_seconds}"

    # -- the front door -------------------------------------------------

    def _door(self):
        from gpuwm.obs import frontdoor

        return frontdoor.MRMS

    def _binary_path(self) -> Path:
        if self._binary is None:
            try:
                self._binary = self._door().require()
            except (RuntimeError, FileNotFoundError, OSError) as error:
                raise MrmsArchiveUnavailable(
                    f"the MRMS front door is not available on this box, so no "
                    f"observation can be fetched: {error}") from error
        return self._binary

    def _common(self) -> list[str]:
        arguments = ["--product", self.product, "--bucket", self.bucket,
                     "--region", self.region,
                     "--cache", str(self.objects_dir)]
        return arguments

    def _run(self, subcommand: str, arguments: Sequence[str], *, schema: str
             ) -> dict:
        door = self._door()
        binary = self._binary_path()
        try:
            return door.run(subcommand, [*arguments, *self._common()],
                            schema=schema, binary=binary)
        except RuntimeError as error:
            if _ABSENT_MARKER in str(error):
                raise MrmsFrameAbsent(str(error)) from error
            raise MrmsArchiveUnavailable(str(error)) from error

    def _window_is_empty(self, valid_time: str) -> bool:
        """Ask the archive whether the window is empty or merely unreachable.

        A refusal from ``nearest`` is ambiguous on its own: the door prints
        one sentence for "no frame" and another for "the bucket did not
        answer", and sniffing English is not a classification.  ``list``
        separates them by construction: it succeeds with ``matched_frames``
        zero for an empty window and fails for an unreachable archive.
        """
        centre = _parse(valid_time)
        span = timedelta(seconds=self.window_seconds)
        record = self._run(
            "list",
            ["--start", (centre - span).strftime(_DOOR_TIME),
             "--end", (centre + span).strftime(_DOOR_TIME)],
            schema="gpuwm-obs.mrms-list.v1")
        return int(record.get("matched_frames", 0)) == 0

    # -- fetching -------------------------------------------------------

    def ensure(self, valid_time: str) -> MrmsFrame:
        """The composite nearest ``valid_time``, fetching it only if needed.

        Raises :class:`MrmsFrameAbsent` when the matching window holds no
        object and :class:`MrmsArchiveUnavailable` when the archive could
        not be asked.
        """
        index = self._read_index()
        request = self._request_key(valid_time)
        resolved = index["requests"].get(request)
        if isinstance(resolved, dict):
            frame = index["frames"].get(resolved["valid_time"])
            if frame is not None and Path(frame["pack_path"]).is_file():
                return self._frame(frame, valid_time,
                                   float(resolved["offset_seconds"]))
        if self.offline:
            raise MrmsArchiveUnavailable(
                f"this run is offline and {valid_time} is not in the case's "
                f"observation cache, so the scan cannot be obtained")

        try:
            nearest = self._run(
                "nearest",
                ["--valid-time", _door_time(valid_time),
                 "--window-seconds", str(self.window_seconds)],
                schema="gpuwm-obs.mrms-nearest.v1")
        except MrmsArchiveUnavailable as error:
            # The door refused for a reason its own sentence did not name as
            # an empty window.  A listing settles it: empty means there is no
            # observation, and a listing that itself fails means this box
            # could not see the archive.
            try:
                empty = self._window_is_empty(valid_time)
            except MrmsArchiveUnavailable:
                raise error from None
            if empty:
                raise MrmsFrameAbsent(
                    f"the archive lists no {self.product} object within "
                    f"{self.window_seconds} s of {valid_time}") from error
            raise
        selected = nearest["frame"]
        frame_time = _format(_parse(selected["valid_time"]))

        known = index["frames"].get(frame_time)
        if known is None or not Path(known["pack_path"]).is_file():
            known = self._fetch_and_decode(selected)
            index["frames"][frame_time] = known
        index["requests"][request] = {
            "valid_time": frame_time,
            "offset_seconds": float(nearest["offset_seconds"])}
        self._write_index()
        return self._frame(known, valid_time, float(nearest["offset_seconds"]))

    def ensure_window(self, valid_time: str) -> list[MrmsFrame]:
        """Every object inside the matching window, fetched and decoded.

        :meth:`ensure` takes the nearest object and stops, which is what a
        healthy feed needs.  The registered frame selection walks outward
        when the nearest frame is below the coverage floor, and it can only
        walk over frames that are on disk -- so a lead whose nearest scan is
        mostly mask pulls its neighbours, once, and the selection rule then
        has the candidates it was written for.
        """
        if self.offline:
            raise MrmsArchiveUnavailable(
                f"this run is offline, so the frames around {valid_time} "
                f"cannot be listed")
        centre = _parse(valid_time)
        span = timedelta(seconds=self.window_seconds)
        listing = self._run(
            "list",
            ["--start", (centre - span).strftime(_DOOR_TIME),
             "--end", (centre + span).strftime(_DOOR_TIME)],
            schema="gpuwm-obs.mrms-list.v1")
        if not listing.get("frames"):
            raise MrmsFrameAbsent(
                f"the archive lists no {self.product} object within "
                f"{self.window_seconds} s of {valid_time}")
        index = self._read_index()
        frames: list[MrmsFrame] = []
        for selected in listing["frames"]:
            stamp = _format(_parse(str(selected["valid_time"])))
            known = index["frames"].get(stamp)
            if known is None or not Path(known["pack_path"]).is_file():
                known = self._fetch_and_decode(selected)
                index["frames"][stamp] = known
            offset = (_parse(stamp) - centre).total_seconds()
            frames.append(self._frame(known, valid_time, offset))
        self._write_index()
        return frames

    @staticmethod
    def _frame(stored: Mapping[str, object], requested: str,
               offset_seconds: float) -> MrmsFrame:
        return MrmsFrame(requested_valid_time=_format(_parse(requested)),
                         offset_seconds=float(offset_seconds),
                         **{key: stored[key]
                            for key in MrmsFrame.__dataclass_fields__
                            if key not in ("requested_valid_time",
                                           "offset_seconds")})

    def _fetch_and_decode(self, selected: Mapping[str, object]
                          ) -> dict[str, object]:
        self.packs_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        stamp = _door_time(str(selected["valid_time"]))
        fetched = self._run(
            "fetch", ["--start", stamp, "--end", stamp,
                      "--out", str(self.objects_dir)],
            schema="gpuwm-obs.mrms-fetch.v1")
        pulled = fetched["files"][0]
        source = Path(pulled["path"])
        # Named for the frame's own instant, not for the archive filename:
        # the MRMS key carries dots in the product name, so splitting it at
        # the first dot gave every frame of a case the same pack name and
        # every frame after the first overwrote its predecessor.
        stamped = _parse(str(selected["valid_time"])).strftime("%Y%m%dT%H%M%S")
        pack = self.packs_dir / f"composite_{stamped}.obspack"
        decode = ["--file", str(source), "--out", str(pack)]
        if self.bbox:
            decode += ["--bbox", self.bbox]
        decoded = self._run("decode", decode,
                            schema="gpuwm-obs.mrms-decode.v1")
        self._ensure_geometry(source)
        return {
            "valid_time": _format(_parse(str(selected["valid_time"]))),
            "bucket": self.bucket,
            "key": str(selected["key"]),
            "object_uri": f"s3://{self.bucket}/{selected['key']}",
            "object_sha256": str(pulled["sha256"]).lower(),
            "fetched_at": str(pulled["fetched_at"]),
            "pack_path": str(pack),
            "pack_sha256": str(decoded["content_sha256"]).lower(),
            "observed_fraction": float(
                decoded["sentinels"]["observed_fraction"]),
        }

    def _ensure_geometry(self, source: Path) -> dict[str, object]:
        index = self._read_index()
        if index.get("geometry") and self.geometry_path.is_file():
            return index["geometry"]
        grid = ["--file", str(source), "--out", str(self.geometry_path)]
        if self.bbox:
            grid += ["--bbox", self.bbox]
        written = self._run("grid", grid, schema="gpuwm-obs.mrms-grid.v1")
        index["geometry"] = {"pack_path": str(self.geometry_path),
                             "pack_sha256": str(written["content_sha256"]).lower(),
                             "grid": written["grid"]}
        return index["geometry"]

    # -- what a receipt says about this cache ---------------------------

    def record(self) -> dict[str, object]:
        index = self._read_index()
        return {
            "schema": CACHE_SCHEMA,
            "route": "rw_mrms front door over AWS Open Data, anonymous",
            "bucket": self.bucket,
            "region": self.region,
            "product": self.product,
            "bbox": self.bbox,
            "match_window_seconds": self.window_seconds,
            "cache_root": str(self.root),
            "offline": self.offline,
            "geometry": index.get("geometry"),
            "objects": [dict(index["frames"][key])
                        for key in sorted(index["frames"])],
            "requests": {key: dict(value)
                         for key, value in sorted(index["requests"].items())},
        }


def fetch_window(cache: MrmsCompositeCache, valid_times: Sequence[str]
                 ) -> list[MrmsFrame]:
    """Every composite a set of valid times resolves to, fetched once each."""
    return [cache.ensure(value) for value in valid_times]


__all__ = [
    "CACHE_SCHEMA", "DEFAULT_BUCKET", "DEFAULT_PRODUCT", "DEFAULT_REGION",
    "DEFAULT_WINDOW_SECONDS", "MrmsArchiveUnavailable", "MrmsCompositeCache",
    "MrmsFrame", "MrmsFrameAbsent", "bbox_argument", "bbox_around",
    "fetch_window",
]
