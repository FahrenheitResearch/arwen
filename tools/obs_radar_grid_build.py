"""Level-II -> ``gpuwm-obs.radar-grid.v1``, the production spine.

One invocation builds ONE observation file for ONE analysis time on ONE
georeference, which is the 1:1 pairing ``tools/da_cycle_prepared.py``
hard-requires (one obs file + one ``--grid-wrfout`` per leg) made
explicit at build time rather than assembled by hand afterwards:

    fetch -> decode -> verify -> superob -> grid -> receipt

Every stage is shipped code: the vendored ``rw_nexrad`` front door via
:mod:`gpuwm.obs.nexrad` (the audit-hardened S3/XML/cache subsystem; the
Message-31 VOL-block antenna resolution is active, so the volume places
its own antenna and the receipt records that source), the superobber via
:mod:`gpuwm.obs.superob`, and the writer via
:mod:`gpuwm.obs.radar_grid` with the grid identity bound to the
georeference wrfout the observations were gridded onto.

Range authority: this tool passes exactly ONE ``--max-range-km`` to BOTH
the decode and the superob parameters (default: the SuperobParams
default), closing the silent 300-vs-250 km disagreement between the two
defaults.  The receipt records the single value used.

Volume selection: the volume whose valid time is nearest the requested
``--valid-time`` wins, and it must land within ``--max-offset-seconds``
(default 480 s) or the tool refuses -- an analysis fed a stale volume
should be a decision, not an accident.

Feed selection: ``--source auto`` (the default) prefers the real-time
chunk feed and falls back to the archive when the live feed cannot cover
the request; ``--source live`` refuses rather than falling back;
``--source archive`` is the route this tool opened with.  The choice, the
objects it resolved to, and the MEASURED lag between the newest object and
the bucket's own clock all land in the receipt and in the NetCDF
provenance -- see :mod:`gpuwm.obs.radar_source`.

No case names belong here; sites, times and buckets are arguments.
EXPERIMENTAL, like everything it drives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    from gpuwm.obs.radar_source import (DEFAULT_SOURCE, SUPPORTED_SOURCES,
                                        RadarSourceError, acquire_volume)

    parser = argparse.ArgumentParser(
        prog="python -m tools.obs_radar_grid_build",
        description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True,
                        help="four-letter radar id, e.g. KXXX")
    parser.add_argument("--valid-time", required=True,
                        help="analysis time (ISO-8601, UTC)")
    parser.add_argument("--grid-wrfout", type=Path, required=True,
                        help="the wrfout whose georeference the "
                             "observations are gridded onto; its SHA-256 "
                             "joins the receipt because it is part of the "
                             "obs identity")
    parser.add_argument("--out", type=Path, required=True,
                        help="output gpuwm-obs.radar-grid.v1 path (.nc)")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="where volumes and sweep packs are staged")
    parser.add_argument("--bucket", default=None,
                        help="archive S3 bucket override (default: the "
                             "front door's own default)")
    parser.add_argument("--live-bucket", default=None,
                        help="real-time chunk bucket override (default: "
                             "the front door's own default)")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        choices=sorted(SUPPORTED_SOURCES),
                        help="which feed serves this observation: live "
                             "(refuse rather than fall back), archive, or "
                             "auto (prefer live, fall back to archive)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="accept a live volume whose scan is still in "
                             "progress. Off by default: a partial scan is "
                             "a real product and not a silent one, and it "
                             "is published under a _P{NNN} name the "
                             "archive key parser refuses")
    parser.add_argument("--min-chunks", type=int, default=None,
                        help="refuse a live assembly shorter than this "
                             "many chunks (front-door default: 2)")
    parser.add_argument("--max-offset-seconds", type=float, default=480.0)
    parser.add_argument("--max-range-km", type=float, default=None,
                        help="THE range authority, applied to decode and "
                             "superob alike (default: SuperobParams "
                             "default)")
    parser.add_argument("--max-elevation-deg", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    from gpuwm.obs.nexrad import (find_nexrad_bin, nexrad_remedy, run_decode,
                                  run_verify)
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import (SuperobParams, merge_contributions,
                                   superob_volume)
    from gpuwm.obs.sweeps import read_sweep_pack
    from gpuwm.obs.target_grid import TargetGrid

    binary = find_nexrad_bin()
    if binary is None:
        raise SystemExit(f"no rw_nexrad front door: {nexrad_remedy()}")

    target = _parse_iso(args.valid_time)
    range_km = (args.max_range_km if args.max_range_km is not None
                else SuperobParams().max_range_km)
    params = SuperobParams(max_range_km=range_km,
                           max_elevation_deg=args.max_elevation_deg)
    params.validate()

    grid = TargetGrid.from_wrfout(args.grid_wrfout)
    grid_wrfout_sha = _sha256(args.grid_wrfout)

    # -- pick the feed, then the nearest volume, refusing a stale one ------
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        selected = acquire_volume(
            binary, site=args.site, out_dir=args.work_dir, valid_time=target,
            source=args.source, bucket=args.bucket,
            live_bucket=args.live_bucket,
            max_offset_seconds=args.max_offset_seconds,
            allow_partial=args.allow_partial, min_chunks=args.min_chunks)
    except RadarSourceError as error:
        raise SystemExit(str(error)) from error
    volume_path = selected.path
    offset = selected.offset_seconds
    volume_sha = _sha256(volume_path)
    if volume_sha != selected.sha256:
        raise SystemExit(
            f"{args.site}: {volume_path} hashes {volume_sha}, the feed "
            f"recorded {selected.sha256}")

    # -- decode + verify (one range authority) ----------------------------
    pack_path = args.work_dir / (selected.filename + ".pack")
    run_decode(binary, volume=volume_path, out=pack_path,
               moments=("REF", "VEL"), max_range_km=range_km,
               max_elevation_deg=args.max_elevation_deg)
    verify = run_verify(binary, pack=pack_path)
    if verify.get("status") != "PASS":
        raise SystemExit(f"pack verify did not PASS: {verify}")

    # -- superob onto the georeference ------------------------------------
    volume = read_sweep_pack(pack_path)
    contribution = superob_volume(volume, grid, params=params)
    observations = merge_contributions([contribution], grid, params=params)

    receipt = write_radar_grid(
        args.out, observations, grid,
        valid_time=selected.valid_time, params=params,
        provenance={
            "builder": "tools/obs_radar_grid_build.py",
            "site": args.site,
            # An archived volume is one object; a live assembly is one per
            # chunk.  `volume_key` stays a STRING either way -- the object
            # that identifies the volume, which for an assembly is its
            # sequence-1 chunk -- because a field that changes type by feed
            # breaks readers written against the route that came first.
            # The full roster is beside it.
            "volume_key": selected.keys[0] if selected.keys else None,
            "volume_object_keys": list(selected.keys),
            "volume_sha256": volume_sha,
            "volume_valid_time": selected.valid_time,
            "requested_valid_time": _iso(target),
            "volume_offset_seconds": offset,
            "bucket": (selected.receipt["bucket"] or "front-door default"),
            "feed": selected.feed,
            "requested_source": args.source,
            "partial_volume": selected.partial,
            "chunks": selected.chunks,
            "feed_lag_seconds": selected.lag_seconds,
            "feed_observed_at": selected.observed_at,
            "antenna_source": volume.site.source,
            "antenna_alt_m": float(volume.site.alt_m),
            "grid_wrfout": str(args.grid_wrfout),
            "grid_wrfout_sha256": grid_wrfout_sha,
            "range_authority_km": range_km,
        },
        overwrite=args.overwrite)

    payload = {
        "schema": "gpuwm-obs.radar-grid-build.v1",
        "site": args.site,
        "volume": selected.filename,
        "volume_sha256": volume_sha,
        "volume_offset_seconds": offset,
        "source": selected.receipt,
        "antenna": {"lat_deg": float(volume.site.lat_deg),
                    "lon_deg": float(volume.site.lon_deg),
                    "alt_m": float(volume.site.alt_m),
                    "source": volume.site.source},
        "range_authority_km": range_km,
        "counts": contribution.counts.to_payload(),
        "cells": {
            "z": int(observations.z_mask.sum()),
            "vr": int(observations.vr_mask.sum()),
            "vr_rejected": int(observations.vr_rejected.sum()),
        },
        "pairing": {"obs": str(args.out),
                    "grid_wrfout": str(args.grid_wrfout),
                    "grid_wrfout_sha256": grid_wrfout_sha},
        "writer_receipt": receipt,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
