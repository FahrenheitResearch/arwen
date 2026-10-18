"""Fetch one radar volume from an open national archive, by table.

The decoders are chosen by the file's own magic bytes (``rw_odim`` reads
ODIM_H5, JMA GRIB2 tars, IRIS RAW, Universal Format, CfRadial 1 and DORADE).
What differs between national archives is only *where the bytes are and how
they are wrapped*, and that is data, not code: every archive is one entry in
``gpuwm/obs/data/radar_archives_world.json``, and adding a network that
publishes in one of the known wrappings is a table edit (the arbitrary
acceptance test). The wrappings this module knows:

``file``
    The volume is one URL per (site, time), on a fixed cadence (JMA's NICT
    mirror: one tar per 10 minutes holding every station; KMA's API Hub).
``zip-member``
    A day of volumes is one zip per site (BoM's NCI ``rq0`` archive, about
    1 GB per site-day). Only the zip's central directory and the one member
    wanted are read, by HTTP range requests, so a single volume costs a few
    MB rather than the whole day.
``s3-listing``
    Objects under a per-site, per-day prefix of a public S3 bucket, times
    parsed from the key names (IDEAM's ``s3-radaresideam``). Every object
    stamped with the chosen time is taken, so an IRIS site's separate tasks
    for one volume arrive together.

An archive that needs a key names the environment variable and how to get
the key; nothing here creates an account or sends a key anywhere else.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

TABLE_SCHEMA = "gpuwm-obs.radar-archives.v1"
FETCH_SCHEMA = "gpuwm-obs.radar-archive-fetch.v1"
TABLE_PATH = Path(__file__).with_name("data") / "radar_archives_world.json"

_TIMEOUT_S = 60
_USER_AGENT = "gpuwm-radar-archive/1"


class RadarArchiveError(RuntimeError):
    """A volume could not be fetched; the message names why and what to do."""


@dataclass(frozen=True)
class Archive:
    id: str
    entry: dict

    @property
    def container(self) -> str:
        return self.entry["container"]


def load_table(path: Path = TABLE_PATH) -> dict[str, Archive]:
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("schema") != TABLE_SCHEMA:
        raise RadarArchiveError(
            f"{path} is schema {table.get('schema')!r}, not {TABLE_SCHEMA}")
    return {key: Archive(key, entry) for key, entry in table["archives"].items()}


def _utc(stamp: datetime) -> datetime:
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _expand(template: str, *, site: str, time: datetime, auth: str | None) -> str:
    """``{site}``, ``{auth}`` and ``{t:<strftime>}`` placeholders."""

    def time_field(match: re.Match) -> str:
        return time.strftime(match.group(1))

    text = re.sub(r"\{t:([^}]*)\}", time_field, template)
    return text.replace("{site}", site).replace("{auth}", auth or "")


def _auth(archive: Archive) -> str | None:
    spec = archive.entry.get("auth")
    if not spec:
        return None
    key = os.environ.get(spec["env"], "").strip()
    if not key:
        raise RadarArchiveError(
            f"{archive.id} needs a key in ${spec['env']}: {spec['how']}")
    return key


def _open(url: str, *, headers: dict | None = None,
          opener: Callable = urllib.request.urlopen):
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    return opener(request, timeout=_TIMEOUT_S)


def _redact(url: str, archive: Archive) -> str:
    spec = archive.entry.get("auth")
    if not spec:
        return url
    parts = urllib.parse.urlsplit(url)
    query = [(k, "***" if k == spec.get("param") else v)
             for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def _download(url: str, target: Path, *, opener: Callable) -> int:
    partial = target.with_name(target.name + ".part")
    with _open(url, opener=opener) as response:
        # Only once the archive has answered, so a 404 leaves no directory.
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    partial.replace(target)
    return target.stat().st_size


class _RangedFile(io.RawIOBase):
    """A seekable, read-only view of one URL served with byte ranges."""

    def __init__(self, url: str, opener: Callable):
        self.url = url
        self.opener = opener
        self.position = 0
        with _open(url, headers={"Range": "bytes=0-0"}, opener=opener) as response:
            span = response.headers.get("Content-Range", "")
            if response.status != 206 or "/" not in span:
                raise RadarArchiveError(
                    f"{url} does not serve byte ranges (status {response.status}); "
                    "a zip-member archive needs them to avoid downloading the "
                    "whole day")
            self.size = int(span.rsplit("/", 1)[1])
        self.requests = 1
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size:
            return 0
        end = min(self.size, self.position + len(buffer)) - 1
        with _open(self.url, headers={"Range": f"bytes={self.position}-{end}"},
                   opener=self.opener) as response:
            data = response.read()
        self.requests += 1
        self.bytes_read += len(data)
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)


def _parse_time(name: str, patterns: list[dict]) -> datetime | None:
    for pattern in patterns:
        match = re.search(pattern["regex"], name)
        if match:
            try:
                return datetime.strptime(match.group(1), pattern["format"]).replace(
                    tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _nearest(candidates: list[tuple[datetime, str]], valid: datetime,
             tolerance_s: float, what: str) -> tuple[datetime, list[str]]:
    if not candidates:
        raise RadarArchiveError(f"{what}: nothing listed")
    best = min(candidates, key=lambda item: abs((item[0] - valid).total_seconds()))
    offset = abs((best[0] - valid).total_seconds())
    if offset > tolerance_s:
        raise RadarArchiveError(
            f"{what}: the nearest volume is {best[0]:%Y-%m-%dT%H:%M:%SZ}, "
            f"{offset:.0f} s from the requested time (tolerance {tolerance_s:.0f} s)")
    return best[0], sorted(name for stamp, name in candidates if stamp == best[0])


@dataclass
class Fetched:
    archive: str
    site: str
    requested: datetime
    volume_time: datetime
    files: list[Path]
    sources: list[str]
    bytes_fetched: int
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "schema": FETCH_SCHEMA,
            "archive": self.archive,
            "site": self.site,
            "requested": self.requested.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "volume_time": self.volume_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": [str(path) for path in self.files],
            "sources": self.sources,
            "bytes_fetched": self.bytes_fetched,
            "notes": self.notes,
        }


def fetch(archive_id: str, *, site: str, valid_time: datetime, out_dir: Path,
          tolerance_s: float | None = None,
          table: dict[str, Archive] | None = None,
          opener: Callable = urllib.request.urlopen) -> Fetched:
    """Fetch the volume of ``site`` nearest ``valid_time`` from one archive."""

    table = table if table is not None else load_table()
    if archive_id not in table:
        raise RadarArchiveError(
            f"no archive {archive_id!r}; the table holds {', '.join(sorted(table))}")
    archive = table[archive_id]
    entry = archive.entry
    valid = _utc(valid_time)
    cadence = float(entry.get("cadence_s", 600))
    tolerance = float(tolerance_s if tolerance_s is not None
                      else entry.get("tolerance_s", cadence / 2))
    out_dir = Path(out_dir)
    first = entry.get("archive_start")
    if first and valid < datetime.fromisoformat(first).replace(tzinfo=timezone.utc):
        raise RadarArchiveError(f"{archive_id} starts {first}; {valid:%Y-%m-%d} is before it")
    auth = _auth(archive)
    what = f"{archive_id} {site} {valid:%Y-%m-%dT%H:%M:%SZ}"

    if archive.container == "file":
        epoch = valid.timestamp()
        slot = datetime.fromtimestamp(round(epoch / cadence) * cadence, tz=timezone.utc)
        if abs((slot - valid).total_seconds()) > tolerance:
            raise RadarArchiveError(f"{what}: no {cadence:.0f} s slot within tolerance")
        files, sources, total = [], [], 0
        for template in entry["urls"]:
            url = _expand(template, site=site, time=slot, auth=auth)
            if "file_name" in entry:
                name = _expand(entry["file_name"], site=site, time=slot, auth=None)
            else:
                name = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
            target = out_dir / name
            try:
                total += _download(url, target, opener=opener)
            except urllib.error.HTTPError as error:
                raise RadarArchiveError(
                    f"{what}: {_redact(url, archive)} returned HTTP {error.code}"
                    + (" (the archive has no volume for this time; it has gaps)"
                       if error.code == 404 else "")) from error
            files.append(target)
            sources.append(_redact(url, archive))
        return Fetched(archive_id, site, valid, slot, files, sources, total)

    if archive.container == "zip-member":
        patterns = entry["member_time"]
        sources, files, total = [], [], 0
        # The day's zip, and the neighbouring day's when the request is
        # within the tolerance of midnight.
        days = sorted({(valid + timedelta(seconds=s)).date()
                       for s in (-tolerance, 0, tolerance)})
        candidates: list[tuple[datetime, tuple[str, str]]] = []
        handles = {}
        for day in days:
            stamp = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            url = _expand(entry["urls"][0], site=site, time=stamp, auth=auth)
            try:
                remote = _RangedFile(url, opener)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    continue
                raise RadarArchiveError(f"{what}: {url} returned HTTP {error.code}") from error
            archive_zip = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16))
            handles[url] = (remote, archive_zip)
            for member in archive_zip.namelist():
                when = _parse_time(member.rsplit("/", 1)[-1], patterns)
                if when is not None:
                    candidates.append((when, (url, member)))
        if not handles:
            raise RadarArchiveError(
                f"{what}: no day zip is published yet for this site and date"
                + (f" ({entry['latency_note']})" if entry.get("latency_note") else ""))
        keyed = [(when, f"{url}#{member}") for when, (url, member) in candidates]
        chosen, names = _nearest(keyed, valid, tolerance, what)
        for name in names:
            url, member = name.split("#", 1)
            remote, archive_zip = handles[url]
            target = out_dir / member.rsplit("/", 1)[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive_zip.open(member) as source, open(target, "wb") as handle:
                shutil.copyfileobj(source, handle, length=1 << 20)
            files.append(target)
            sources.append(name)
        total = sum(remote.bytes_read for remote, _ in handles.values())
        notes = [f"read {total} bytes of the day zip by range requests"]
        return Fetched(archive_id, site, valid, chosen, files, sources, total, notes)

    if archive.container == "s3-listing":
        patterns = entry["key_time"]
        days = sorted({(valid + timedelta(seconds=s)).date()
                       for s in (-tolerance, 0, tolerance)})
        candidates = []
        for day in days:
            stamp = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            prefix = _expand(entry["prefix"], site=site, time=stamp, auth=None)
            token = None
            while True:
                query = {"list-type": "2", "prefix": prefix}
                if token:
                    query["continuation-token"] = token
                listing = entry["bucket"] + "?" + urllib.parse.urlencode(query)
                try:
                    with _open(listing, opener=opener) as response:
                        xml = response.read().decode("utf-8", "replace")
                except urllib.error.HTTPError as error:
                    # A neighbouring day the archive does not hold lists
                    # nothing; the chosen day's absence is reported below.
                    if error.code == 404:
                        break
                    raise RadarArchiveError(
                        f"{what}: listing {listing} returned HTTP {error.code}") from error
                for key in re.findall(r"<Key>([^<]+)</Key>", xml):
                    when = _parse_time(key.rsplit("/", 1)[-1], patterns)
                    if when is not None:
                        candidates.append((when, key))
                more = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
                if not more:
                    break
                token = more.group(1)
        chosen, keys = _nearest(candidates, valid, tolerance, what)
        # One object per stem: a gzipped copy beside an uncompressed one is
        # the same volume, and the smaller one is fetched.
        preferred = {}
        for key in keys:
            stem = key[:-3] if key.endswith(".gz") else key
            if stem not in preferred or key.endswith(".gz"):
                preferred[stem] = key
        files, sources, total = [], [], 0
        for key in sorted(preferred.values()):
            url = entry["bucket"] + urllib.parse.quote(key)
            target = out_dir / key.rsplit("/", 1)[-1]
            total += _download(url, target, opener=opener)
            files.append(target)
            sources.append(url)
        return Fetched(archive_id, site, valid, chosen, files, sources, total)

    raise RadarArchiveError(f"{archive_id}: unknown container {archive.container!r}")


def pack(archive_id: str, *, site: str, valid_time: datetime, out_dir: Path,
         out: Path, quantities: list[str] | None = None,
         max_range_km: float | None = None,
         table: dict[str, Archive] | None = None,
         opener: Callable = urllib.request.urlopen) -> dict:
    """Fetch, then decode into one sweep pack with ``rw_odim``."""

    from gpuwm.obs.odim import run_pack

    table = table if table is not None else load_table()
    fetched = fetch(archive_id, site=site, valid_time=valid_time, out_dir=out_dir,
                    table=table, opener=opener)
    entry = table[archive_id].entry
    record = run_pack(file=fetched.files, out=out,
                      site=site if entry.get("site_in_file") else None,
                      quantities=quantities, max_range_km=max_range_km)
    return {"fetch": fetched.to_payload(), "pack": record}


__all__ = ["FETCH_SCHEMA", "TABLE_PATH", "TABLE_SCHEMA", "Archive", "Fetched",
           "RadarArchiveError", "fetch", "load_table", "pack"]
