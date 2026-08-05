"""Which radar feed serves this observation, and what that cost in lag.

Two routes reach the same decoder.  The **archive** route lists
``{YYYY}/{MM}/{DD}/{SITE}/`` and downloads one Archive-II file per finished
volume; a volume file only exists once the volume has *ended*, so at a
random poll the newest one is on average half a volume period old and at
worst a whole one -- 3.4 to 6.8 minutes at a site running a 410 s VCP.  The
**live** route lists the real-time chunk feed and concatenates the chunks
published so far into the same Archive-II bytes; measured on 2026-08-05 the
newest chunk was 0-4 s old on every poll.

This module is the selector between them and nothing else.  It owns no
decode, no grid, no policy about what a good observation is: it returns a
file on disk plus a receipt saying which feed produced it, which objects it
is made of, whether the scan had finished, and the measured lag.  Everything
below it -- ``run_decode``, ``read_sweep_pack``, superob, the radar grid --
takes a path and never asks where it came from.

Three modes:

``archive``
    The route that exists today: nearest volume to ``valid_time``, refused
    when it is further away than ``max_offset_seconds``.

``live``
    The chunk feed, or a refusal.  Never silently falls back: a caller who
    asked for the live feed and got the archive would read a receipt saying
    "live" was requested and be told nothing about why it was not served.

``auto``
    Prefer live; fall back to the archive when the live feed cannot cover
    the request.  Every attempt, and why it was declined, lands in the
    receipt's ``attempts`` list -- so a fallback is a recorded event and not
    an absence.

**Partial volumes.**  A live assembly that stops mid-scan is a real product:
it ends on an LDM block boundary, the decoder reads the radials that
arrived, and that is what puts the lag under the scan time itself.  It is
also *never* the default.  ``allow_partial`` must be asked for, the receipt
records ``partial`` and the chunk count either way, and the file is named
``..._P{NNN}``, which the archive key parser refuses -- so a partial cannot
be mistaken for a finished volume by a name.

No case names belong here; sites and buckets are arguments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gpuwm.obs.nexrad import (ARCHIVE_FEED, LIVE_FEED, run_fetch, run_list,
                              run_live_fetch, run_live_list)

#: The selector's own schema.
SOURCE_SCHEMA = "gpuwm-obs.radar-source.v1"

#: Every mode ``source=`` accepts.  Membership is checked, so a typo is a
#: refusal rather than a silent fall-through to a default nobody chose.
SUPPORTED_SOURCES = ("live", "archive", "auto")

#: The default for nowcasting: prefer live, fall back rather than fail.
DEFAULT_SOURCE = "auto"

#: How far back ``auto`` will still try the live feed for a *named* valid
#: time before going straight to the archive.
#:
#: This is a request-saving heuristic and not a retention claim.  The chunk
#: bucket's retention was measured at 48-72 h with a lifecycle rule nobody
#: outside AWS can read, so designing to a fixed number would be designing
#: to a guess.  Being wrong here costs one declined live attempt, recorded
#: in ``attempts``, and then the archive answers.
LIVE_LOOKBACK_CEILING_SECONDS = 3600.0

#: How many live volumes to ask for when a valid time was named.  The live
#: route walks back through the retained volume ids one listing at a time,
#: so this bounds that walk; four volumes spans roughly 13-27 minutes of
#: scans depending on the VCP.
LIVE_VOLUMES_FOR_A_NAMED_TIME = 4


class RadarSourceError(RuntimeError):
    """A feed could not serve the request, and the reason is the message."""


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _iso(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Attempt:
    """One feed asked, and what it said."""

    feed: str
    outcome: str          # "selected" | "declined" | "failed"
    reason: str | None = None

    def to_payload(self) -> dict:
        return {"feed": self.feed, "outcome": self.outcome,
                "reason": self.reason}


@dataclass(frozen=True)
class SelectedVolume:
    """One Archive-II volume on disk, and the provenance of how it got there."""

    feed: str
    path: Path
    filename: str
    valid_time: str
    sha256: str
    #: Every S3 object this volume is made of: one archive key, or every
    #: chunk key in the assembled prefix.
    keys: tuple[str, ...]
    partial: bool
    #: Chunks in the assembly, or ``None`` for an archive volume, which is
    #: not made of chunks and should not report a count of one.
    chunks: int | None
    #: Seconds from the newest object landing in the bucket to the instant
    #: the bucket answered, on the bucket's own clock.  ``None`` when the
    #: ``Date`` header could not be read: unmeasured is stated as
    #: unmeasured rather than filled in from the local clock.
    lag_seconds: float | None
    observed_at: str | None
    #: Signed seconds from the requested analysis time, or ``None`` when
    #: none was requested ("give me the newest").
    offset_seconds: float | None
    receipt: dict = field(repr=False)


def _archive_lag(record: dict, newest: dict) -> tuple[float | None, str | None]:
    """(lag, observed_at) for an archive listing, on the bucket's clock."""

    observed = record.get("observed_at")
    stamped = newest.get("last_modified")
    if not observed or not stamped:
        return None, observed
    try:
        return ((_parse_iso(observed) - _parse_iso(stamped)).total_seconds(),
                observed)
    except ValueError:
        return None, observed


def _acquire_archive(binary, *, site, out_dir, valid_time, bucket, cache_dir,
                     use_cache, max_offset_seconds, now):
    """Nearest archived volume to ``valid_time``, or the newest one."""

    if valid_time is None:
        # "The newest" on a route with no newest-first listing means a
        # window ending now.  A volume is only published at scan end, so a
        # window shorter than a volume period can legitimately be empty.
        end = now
        start = now - timedelta(seconds=max(max_offset_seconds, 900.0) * 2)
    else:
        half = max(max_offset_seconds, 900.0)
        start = valid_time - timedelta(seconds=half)
        end = valid_time + timedelta(seconds=half)
    listing = run_list(binary, site=site, start=_iso(start), end=_iso(end),
                       bucket=bucket)
    candidates = [volume for volume in listing.get("volumes", ())
                  if not volume["filename"].endswith("MDM")]
    if not candidates:
        raise RadarSourceError(
            f"{site}: the archive holds no volume between {_iso(start)} and "
            f"{_iso(end)}")
    if valid_time is None:
        chosen = max(candidates, key=lambda v: _parse_iso(v["valid_time"]))
        offset = None
    else:
        chosen = min(candidates, key=lambda v: abs(
            (_parse_iso(v["valid_time"]) - valid_time).total_seconds()))
        offset = (_parse_iso(chosen["valid_time"]) - valid_time).total_seconds()
        if abs(offset) > max_offset_seconds:
            raise RadarSourceError(
                f"{site}: the nearest archived volume {chosen['filename']} is "
                f"{offset:+.0f} s from {_iso(valid_time)}, beyond the "
                f"{max_offset_seconds:.0f} s ceiling")
    stamp = _parse_iso(chosen["valid_time"])
    fetch = run_fetch(binary, site=site,
                      start=_iso(stamp - timedelta(seconds=30)),
                      end=_iso(stamp + timedelta(seconds=30)),
                      out=out_dir, bucket=bucket, cache_dir=cache_dir,
                      use_cache=use_cache)
    served = next((entry for entry in fetch.get("files", ())
                   if entry["key"] == chosen["key"]), None)
    if served is None:
        raise RadarSourceError(
            f"{site}: the archive fetch did not return {chosen['key']}")
    path = out_dir / chosen["filename"]
    if not path.is_file():
        raise RadarSourceError(f"{site}: fetch did not materialise {path}")
    lag, observed_at = _archive_lag(listing, chosen)
    return {
        "feed": ARCHIVE_FEED,
        "path": path,
        "filename": chosen["filename"],
        "valid_time": chosen["valid_time"],
        "sha256": served["sha256"],
        "keys": (chosen["key"],),
        "partial": False,
        "chunks": None,
        "listed_chunks": None,
        "complete": True,
        "truncation": None,
        "lag_seconds": lag,
        "observed_at": observed_at,
        "offset_seconds": offset,
        "bucket": listing.get("window", {}).get("bucket"),
        "bytes": chosen.get("size_bytes"),
        "clock_skew_seconds": None,
        "volume_id": None,
    }


def _acquire_live(binary, *, site, out_dir, valid_time, bucket, cache_dir,
                  use_cache, max_offset_seconds, allow_partial, min_chunks,
                  volumes):
    """The newest real-time assembly, or the one nearest ``valid_time``.

    Listed first, then assembled by name.  ``live-list`` moves no payload,
    so choosing on the listing and fetching the one volume that won costs
    two extra listings and saves downloading every candidate: a named
    analysis time asks about several volumes and uses one, and each one is
    ~110 objects and ~17 MB.
    """

    listing = run_live_list(binary, site=site, bucket=bucket, volumes=volumes,
                            allow_partial=allow_partial,
                            min_chunks=min_chunks)
    offered = [volume for volume in listing.get("volumes", ())
               if volume.get("admissible")]
    if not offered:
        refused = "; ".join(
            f"volume {volume['volume_id']} ({volume['volume_time']}): "
            f"{volume['refusal']}"
            for volume in listing.get("volumes", ())
            if volume.get("refusal"))
        raise RadarSourceError(
            f"{site}: the live feed offered no volume this run will "
            f"assemble. {refused}")
    if valid_time is None:
        picked = max(offered, key=lambda v: _parse_iso(v["volume_time"]))
        offset = None
    else:
        picked = min(offered, key=lambda v: abs(
            (_parse_iso(v["volume_time"]) - valid_time).total_seconds()))
        offset = (_parse_iso(picked["volume_time"]) - valid_time).total_seconds()
        if abs(offset) > max_offset_seconds:
            raise RadarSourceError(
                f"{site}: the nearest live volume ({picked['volume_time']}, "
                f"{picked['chunks']} chunks) is {offset:+.0f} s from "
                f"{_iso(valid_time)}, beyond the "
                f"{max_offset_seconds:.0f} s ceiling")

    record = run_live_fetch(binary, site=site, out=out_dir, bucket=bucket,
                            volume_id=picked["volume_id"],
                            allow_partial=allow_partial,
                            min_chunks=min_chunks, cache_dir=cache_dir,
                            use_cache=use_cache)
    assembled = [volume for volume in record.get("volumes", ())
                 if volume.get("sha256")]
    if not assembled:
        raise RadarSourceError(
            f"{site}: the live feed assembled no volume "
            f"({record.get('admissible_volumes', 0)} admissible of "
            f"{record.get('matched_volumes', 0)} listed)")
    chosen = assembled[0]
    # The volume may have grown between the listing and the fetch -- that
    # is the feed working, and a fresher partial is the point.  A volume
    # that changed IDENTITY is not: it would mean the counter moved under
    # the selection and the assembly answers a different question.
    if chosen["volume_time"] != picked["volume_time"]:
        raise RadarSourceError(
            f"{site}: volume {picked['volume_id']} was listed as "
            f"{picked['volume_time']} and assembled as "
            f"{chosen['volume_time']}; the real-time volume id was reused "
            "between the listing and the fetch")
    path = Path(chosen["path"])
    if not path.is_file():
        raise RadarSourceError(
            f"{site}: the live assembly did not materialise {path}")
    return {
        "feed": record.get("feed", LIVE_FEED),
        "path": path,
        "filename": chosen["filename"],
        "valid_time": chosen["volume_time"],
        "sha256": chosen["sha256"],
        "keys": tuple(chosen.get("keys", ())),
        "partial": bool(chosen["partial"]),
        "chunks": int(chosen["chunks"]),
        "listed_chunks": int(chosen["listed_chunks"]),
        "complete": bool(chosen["complete"]),
        "truncation": chosen.get("truncation"),
        "lag_seconds": chosen.get("lag_seconds"),
        "observed_at": record.get("observed_at"),
        "offset_seconds": offset,
        "bucket": record.get("bucket"),
        "bytes": chosen.get("bytes"),
        "clock_skew_seconds": record.get("clock_skew_seconds"),
        "volume_id": chosen.get("volume_id"),
    }


def live_is_plausible(valid_time: datetime | None, now: datetime, *,
                      ceiling_seconds: float = LIVE_LOOKBACK_CEILING_SECONDS
                      ) -> tuple[bool, str | None]:
    """Is the live feed worth asking for this request?

    Returns ``(plausible, why_not)``.  Pure, so ``auto``'s branch can be
    asserted without a socket.
    """

    if valid_time is None:
        return True, None
    age = (now - valid_time).total_seconds()
    if age > ceiling_seconds:
        return False, (
            f"the requested valid time is {age / 60.0:.0f} min old, past the "
            f"{ceiling_seconds / 60.0:.0f} min ceiling this route asks the "
            "real-time feed about")
    if age < -ceiling_seconds:
        return False, (
            f"the requested valid time is {-age / 60.0:.0f} min in the "
            "future; no feed has it")
    return True, None


def acquire_volume(binary, *, site: str, out_dir: Path,
                   valid_time: datetime | str | None = None,
                   source: str = DEFAULT_SOURCE,
                   bucket: str | None = None,
                   live_bucket: str | None = None,
                   cache_dir: Path | None = None,
                   use_cache: bool = True,
                   max_offset_seconds: float = 480.0,
                   allow_partial: bool = False,
                   min_chunks: int | None = None,
                   live_volumes: int | None = None,
                   now: datetime | None = None) -> SelectedVolume:
    """Put one Archive-II volume in ``out_dir`` and say where it came from.

    ``valid_time`` of ``None`` means "the newest there is", which is the
    nowcast's question.  A named time is the verification question, and
    both feeds answer it the same way: nearest volume, refused past
    ``max_offset_seconds``.
    """

    if source not in SUPPORTED_SOURCES:
        raise RadarSourceError(
            f"unknown radar source {source!r}; expected one of "
            f"{', '.join(SUPPORTED_SOURCES)}")
    if isinstance(valid_time, str):
        valid_time = _parse_iso(valid_time)
    now = now or datetime.now(timezone.utc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    volumes = live_volumes
    if volumes is None:
        volumes = 1 if valid_time is None else LIVE_VOLUMES_FOR_A_NAMED_TIME

    attempts: list[Attempt] = []
    served: dict | None = None

    if source in ("live", "auto"):
        plausible, why_not = live_is_plausible(valid_time, now)
        if not plausible and source == "auto":
            attempts.append(Attempt(LIVE_FEED, "declined", why_not))
        else:
            try:
                served = _acquire_live(
                    binary, site=site, out_dir=out_dir, valid_time=valid_time,
                    bucket=live_bucket, cache_dir=cache_dir,
                    use_cache=use_cache, max_offset_seconds=max_offset_seconds,
                    allow_partial=allow_partial, min_chunks=min_chunks,
                    volumes=volumes)
                attempts.append(Attempt(LIVE_FEED, "selected"))
            except (RadarSourceError, RuntimeError) as error:
                if source == "live":
                    # An explicit --source live never silently becomes the
                    # archive: the caller asked a question about latency and
                    # a quiet fallback answers a different one.
                    raise RadarSourceError(
                        f"the live feed could not serve {site}: {error}"
                    ) from error
                attempts.append(Attempt(LIVE_FEED, "failed", str(error)))

    if served is None:
        try:
            served = _acquire_archive(
                binary, site=site, out_dir=out_dir, valid_time=valid_time,
                bucket=bucket, cache_dir=cache_dir, use_cache=use_cache,
                max_offset_seconds=max_offset_seconds, now=now)
            attempts.append(Attempt(ARCHIVE_FEED, "selected"))
        except (RadarSourceError, RuntimeError) as error:
            attempts.append(Attempt(ARCHIVE_FEED, "failed", str(error)))
            tried = "; ".join(
                f"{attempt.feed}: {attempt.outcome}"
                + (f" ({attempt.reason})" if attempt.reason else "")
                for attempt in attempts)
            raise RadarSourceError(
                f"no feed served {site} for source={source}. {tried}"
            ) from error

    receipt = {
        "schema": SOURCE_SCHEMA,
        "requested_source": source,
        "feed": served["feed"],
        "site": site,
        "bucket": served["bucket"],
        "requested_valid_time": _iso(valid_time) if valid_time else None,
        "selected_valid_time": served["valid_time"],
        "offset_seconds": served["offset_seconds"],
        "volume": served["filename"],
        "volume_sha256": served["sha256"],
        "bytes": served["bytes"],
        "object_keys": list(served["keys"]),
        "object_count": len(served["keys"]),
        "partial": served["partial"],
        "complete": served["complete"],
        "chunks": served["chunks"],
        "listed_chunks": served["listed_chunks"],
        "truncation": served["truncation"],
        "volume_id": served["volume_id"],
        "observed_at": served["observed_at"],
        "clock_skew_seconds": served["clock_skew_seconds"],
        "lag_seconds": served["lag_seconds"],
        # Named so a reader knows the lag is not this host's clock guessing
        # at the bucket's, and knows when it could not be measured at all.
        "lag_measured_against": ("s3-date-header" if served["observed_at"]
                                 else "unmeasured"),
        "lag_definition": ("seconds from the newest object's S3 LastModified "
                           "to the instant the bucket answered the listing"),
        "allow_partial": allow_partial,
        "min_chunks": min_chunks,
        "attempts": [attempt.to_payload() for attempt in attempts],
    }
    return SelectedVolume(
        feed=served["feed"], path=served["path"], filename=served["filename"],
        valid_time=served["valid_time"], sha256=served["sha256"],
        keys=tuple(served["keys"]), partial=served["partial"],
        chunks=served["chunks"], lag_seconds=served["lag_seconds"],
        observed_at=served["observed_at"],
        offset_seconds=served["offset_seconds"], receipt=receipt)
