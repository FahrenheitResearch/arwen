"""Level-II -> ``gpuwm-obs.radar-grid.v1``, the production spine.

One invocation builds ONE observation file for ONE analysis time on ONE
georeference, which is the 1:1 pairing ``tools/da_cycle_prepared.py``
hard-requires (one obs file + one ``--grid-wrfout`` per leg) made
explicit at build time rather than assembled by hand afterwards:

    fetch -> decode -> verify -> superob -> grid -> receipt

**One analysis time, but not necessarily one radar.**  ``--site`` repeats,
and ``--discover-sites`` finds them from the georeference itself through
:mod:`gpuwm.obs.coverage`, which computes coverage from the front door's
own vendored site table and the grid's own mass points.  No site id is
ever a default here; ids are arguments or they are results, never
constants.

Why it matters: one radar measures the wind component along its own beam
and nothing else, so a single-radar velocity analysis is one projection of
a three-component field.  Two radars whose coverage overlaps measure two
projections of the same wind and constrain what neither can alone.  The
whole stack below this tool has carried a per-radar axis since it was
written -- ``merge_contributions`` takes a sequence, ``vr_obs`` is
``(radar, level, y, x)``, the file carries ``radar_id``/``radar_lat``/
``radar_lon``/``radar_valid_time``, and the DA adapter makes each radar
its own ``vr:<SITE>`` batch with its own localization and thinning.  This
tool was the one place that only ever passed one.

**Volumes from different radars are not synchronous, and the policy is
stated.**  Each site contributes the volume nearest the requested analysis
time within ``--max-offset-seconds``, exactly as the single-radar route
does.  The *spread* between the contributing volumes is then checked
against ``--max-radar-time-spread-seconds``: sites that agree to within a
volume period describe one atmosphere, and sites that do not describe two.
Every contributing volume's own valid time is written per radar into the
file, so the assimilating side is never told a fiction about simultaneity.

**A site that fails to fetch degrades to the ones that did.**  It is
recorded in the receipt with its refusal, counted, and the run continues
so long as ``--min-radars`` are left.  A thinner analysis that does not
say it is thinner is the failure mode this exists to prevent.

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
import sys
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


class SiteRequestError(SystemExit):
    """The caller's radar request cannot be honoured as written."""


def requested_sites(args) -> list[str]:
    """The sites this invocation NAMED, normalized, or [] for discovery.

    Pure, and called before any I/O, so a contradictory request costs
    nothing to refuse.  Discovery returns an empty list here because the
    ids do not exist yet -- they are computed from the georeference, which
    has to be read first.
    """

    if args.site and args.discover_sites:
        raise SiteRequestError(
            "--site and --discover-sites are mutually exclusive: naming "
            "some radars and discovering others leaves a receipt that "
            "cannot say which route chose which, and a rerun against a "
            "changed table would silently differ")
    if args.discover_sites:
        return []
    if not args.site:
        raise SiteRequestError(
            "name at least one --site, or ask for --discover-sites. There "
            "is no default radar and there must not be one")
    seen: set[str] = set()
    sites: list[str] = []
    for name in args.site:
        site = str(name).strip().upper()
        if site in seen:
            raise SiteRequestError(
                f"--site {site} was given twice; a radar cannot contribute "
                "two velocity batches to one analysis, and silently "
                "collapsing the repeat would hide a typo in the other one")
        seen.add(site)
        sites.append(site)
    return sites


def build_parser() -> argparse.ArgumentParser:
    """This tool's contract, as an object a caller can check against.

    Extracted from ``main`` so a caller that BUILDS this argv -- the
    nowcast front door does, for every assimilated cycle and every
    verification frame -- can prove in a test that what it emits is
    what this tool accepts, without fetching a byte.  A passthrough
    tested only against a hand-written copy of the flag list is tested
    against the copy.
    """

    from gpuwm.obs.radar_source import DEFAULT_SOURCE, SUPPORTED_SOURCES

    parser = argparse.ArgumentParser(
        prog="python -m tools.obs_radar_grid_build",
        description=__doc__.splitlines()[0])
    parser.add_argument("--site", action="append", default=[],
                        help="four-letter radar id, e.g. KXXX. Repeatable: "
                             "every site named contributes its own velocity "
                             "batch to one observation file. Mutually "
                             "exclusive with --discover-sites, because a "
                             "run should not half-choose its own radars")
    parser.add_argument("--discover-sites", action="store_true",
                        help="find the contributing radars from the "
                             "georeference and the vendored NEXRAD site "
                             "table instead of naming them. Coverage is the "
                             "fraction of the grid's own mass points inside "
                             "the range authority, so a site that clips one "
                             "corner is ranked below one that sits over the "
                             "domain and both are visible in the receipt")
    parser.add_argument("--min-coverage-fraction", type=float, default=0.05,
                        help="discovery floor: skip a site that reaches less "
                             "than this fraction of the domain (default "
                             "0.05). Raising it buys density, lowering it "
                             "buys the edges")
    parser.add_argument("--max-radars", type=int, default=None,
                        help="keep at most this many discovered sites, best "
                             "coverage first (default: all that clear the "
                             "floor)")
    parser.add_argument("--min-radars", type=int, default=1,
                        help="refuse the whole build if fewer than this many "
                             "sites produced a contribution. The default of "
                             "1 means a multi-radar request degrades to "
                             "whatever answered rather than failing the "
                             "cycle; raising it makes a specific number of "
                             "radars a hard requirement of the analysis")
    parser.add_argument("--max-radar-time-spread-seconds", type=float,
                        default=None,
                        help="refuse if the contributing volumes' own valid "
                             "times span more than this. Volumes from "
                             "different sites are never synchronous; this is "
                             "where the caller says how much asynchrony is "
                             "one atmosphere (default: unbounded, and the "
                             "measured spread is always in the receipt)")
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
    parser.add_argument("--clear-air-from-censor", action="store_true",
                        help="decode with --censor-flags and build clear-air "
                             "zeroes from the RDA's own below-threshold gate "
                             "code as well as from finite below-floor gates. "
                             "Range-folded gates stay excluded. Changes the "
                             "file's clear_air_source, and therefore what a "
                             "DA adapter will accept it as")
    parser.add_argument("--dealias", action="store_true",
                        help="unfold radial velocity per sweep before "
                             "gridding, instead of masking every gate that "
                             "might be folded. Gates the unfolder cannot "
                             "resolve are still dropped and counted; gates "
                             "it does resolve are bounded by an absolute "
                             "speed rather than a fraction of Nyquist, "
                             "which is what recovers a mesocyclone couplet "
                             "living above 0.8 * Nyquist. Records the "
                             "per-gate account in the file's provenance. "
                             "Requires scipy")
    return parser


def main(argv=None) -> int:
    from gpuwm.obs.radar_source import RadarSourceError, acquire_volume

    args = build_parser().parse_args(argv)
    # A contradictory radar request is refused before anything is read,
    # fetched or hashed: it costs nothing to catch here and the message is
    # the same either way.
    named_sites = requested_sites(args)

    from gpuwm.obs.nexrad import (find_nexrad_bin, nexrad_remedy, run_decode,
                                  run_verify)
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.dealias import SCIPY_REMEDY, DealiasParams, scipy_available
    from gpuwm.obs.superob import (SuperobParams, merge_contributions,
                                   superob_volume)
    from gpuwm.obs.sweeps import read_sweep_pack
    from gpuwm.obs.target_grid import TargetGrid

    binary = find_nexrad_bin()
    if binary is None:
        raise SystemExit(f"no rw_nexrad front door: {nexrad_remedy()}")

    # Refuse before a byte is fetched, not from inside the gridding loop.
    if args.dealias and not scipy_available():
        raise SystemExit(SCIPY_REMEDY)

    target = _parse_iso(args.valid_time)
    range_km = (args.max_range_km if args.max_range_km is not None
                else SuperobParams().max_range_km)
    params = SuperobParams(max_range_km=range_km,
                           max_elevation_deg=args.max_elevation_deg,
                           dealias=DealiasParams() if args.dealias else None)
    params.validate()

    grid = TargetGrid.from_wrfout(args.grid_wrfout)
    grid_wrfout_sha = _sha256(args.grid_wrfout)

    # -- which radars ------------------------------------------------------
    discovery = None
    sites = named_sites
    if args.discover_sites:
        from gpuwm.obs.coverage import (SiteCoverageError, discovery_receipt,
                                        read_site_table, sites_covering)
        try:
            table = read_site_table(binary)
            found = sites_covering(
                grid, table, max_range_km=range_km,
                min_coverage_fraction=args.min_coverage_fraction,
                limit=args.max_radars)
        except SiteCoverageError as error:
            raise SystemExit(str(error)) from error
        discovery = discovery_receipt(
            found, max_range_km=range_km,
            min_coverage_fraction=args.min_coverage_fraction,
            limit=args.max_radars, table_size=len(table))
        sites = [entry.site.id for entry in found]
        if not sites:  # noqa: SIM102 - the message is the whole point
            raise SystemExit(
                f"no site in the {len(table)}-entry NEXRAD table reaches "
                f"{args.min_coverage_fraction:.0%} of this domain at a "
                f"{range_km:.0f} km range authority. Lower "
                "--min-coverage-fraction, widen the range, or accept that "
                "this domain is not radar-observed")

    # -- pick the feed, then the nearest volume per site, refusing stale ---
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # The writer stages its atomic temp file BESIDE the output, so the
    # output's directory has to exist before the superob work starts --
    # not after it, and not by luck.  netCDF4 opening a file under a
    # missing directory raises PermissionError (errno 13) on Windows
    # rather than FileNotFoundError, so this failed as an access-rights
    # problem an operator could not act on, after every volume had
    # already been fetched, decoded and gridded.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    contributions = []
    per_site: list[dict] = []
    failures: list[dict] = []
    for site in sites:
        try:
            selected = acquire_volume(
                binary, site=site, out_dir=args.work_dir, valid_time=target,
                source=args.source, bucket=args.bucket,
                live_bucket=args.live_bucket,
                max_offset_seconds=args.max_offset_seconds,
                allow_partial=args.allow_partial, min_chunks=args.min_chunks)
            volume_path = selected.path
            volume_sha = _sha256(volume_path)
            if volume_sha != selected.sha256:
                raise RadarSourceError(
                    f"{site}: {volume_path} hashes {volume_sha}, the feed "
                    f"recorded {selected.sha256}")

            # -- decode + verify (one range authority) --------------------
            pack_path = args.work_dir / (selected.filename + ".pack")
            run_decode(binary, volume=volume_path, out=pack_path,
                       moments=("REF", "VEL"), max_range_km=range_km,
                       max_elevation_deg=args.max_elevation_deg,
                       censor_flags=args.clear_air_from_censor)
            verify = run_verify(binary, pack=pack_path)
            if verify.get("status") != "PASS":
                raise RadarSourceError(
                    f"{site}: pack verify did not PASS: {verify}")

            # -- superob onto the georeference ----------------------------
            volume = read_sweep_pack(pack_path)
            contribution = superob_volume(
                volume, grid, params=params,
                clear_air_from_censor=args.clear_air_from_censor)
        except (RadarSourceError, RuntimeError, OSError, ValueError) as error:
            # A site that cannot serve this analysis time degrades to the
            # ones that can.  Recorded with its reason and counted, never
            # dropped into silence -- an analysis that quietly lost half
            # its radars is indistinguishable from one that never had
            # them, and only one of those is worth trusting.
            failures.append({"site": site, "reason": str(error)})
            print(f"{site}: no contribution ({error})", file=sys.stderr,
                  flush=True)
            continue
        contributions.append(contribution)
        per_site.append({
            "site": site,
            "volume": selected.filename,
            "volume_sha256": volume_sha,
            "volume_valid_time": selected.valid_time,
            "volume_offset_seconds": selected.offset_seconds,
            "source": selected.receipt,
            "feed": selected.feed,
            "partial_volume": selected.partial,
            "chunks": selected.chunks,
            "feed_lag_seconds": selected.lag_seconds,
            "feed_observed_at": selected.observed_at,
            # An archived volume is one object; a live assembly is one per
            # chunk.  `volume_key` stays a STRING either way -- the object
            # that identifies the volume, which for an assembly is its
            # sequence-1 chunk -- because a field that changes type by feed
            # breaks readers written against the route that came first.
            # The full roster is beside it.
            "volume_key": selected.keys[0] if selected.keys else None,
            "volume_object_keys": list(selected.keys),
            "antenna": {"lat_deg": float(volume.site.lat_deg),
                        "lon_deg": float(volume.site.lon_deg),
                        "alt_m": float(volume.site.alt_m),
                        "source": volume.site.source},
            "counts": contribution.counts.to_payload(),
            "pack_schema": volume.pack_schema,
            "clear_air_source": contribution.clear_air_source,
        })

    if len(contributions) < max(1, int(args.min_radars)):
        detail = "; ".join(f"{f['site']}: {f['reason']}" for f in failures)
        raise SystemExit(
            f"{len(contributions)} radar(s) produced a contribution, "
            f"--min-radars demands {max(1, int(args.min_radars))}. "
            f"Requested {', '.join(sites)}. {detail}")

    # -- the asynchrony policy, measured and then judged -------------------
    stamps = sorted(_parse_iso(entry["volume_valid_time"])
                    for entry in per_site)
    spread = (stamps[-1] - stamps[0]).total_seconds() if stamps else 0.0
    if (args.max_radar_time_spread_seconds is not None
            and spread > float(args.max_radar_time_spread_seconds)):
        raise SystemExit(
            f"the contributing volumes span {spread:.0f} s "
            f"({_iso(stamps[0])} to {_iso(stamps[-1])}), beyond the "
            f"{float(args.max_radar_time_spread_seconds):.0f} s ceiling. "
            "Radars are never synchronous, but past some spread they are "
            "describing two atmospheres rather than one")

    # The file's own valid_time is the ANALYSIS time that was requested,
    # not any one radar's volume time -- each radar's own volume time is
    # written beside its data, so nothing downstream is told a fiction
    # about simultaneity.  Single-radar builds keep the behaviour they
    # have always had: that radar's volume time IS the file's.
    file_valid_time = (per_site[0]["volume_valid_time"]
                       if len(per_site) == 1 else _iso(target))

    observations = merge_contributions(contributions, grid, params=params)

    provenance = {
        "builder": "tools/obs_radar_grid_build.py",
        "requested_sites": list(sites),
        "contributing_sites": [entry["site"] for entry in per_site],
        "sites_that_failed": failures,
        "site_selection": ("discovered" if args.discover_sites
                           else "named"),
        "discovery": discovery,
        "requested_valid_time": _iso(target),
        "radar_time_spread_seconds": spread,
        "radar_time_spread_policy": (
            "each site contributes the volume nearest the requested "
            "analysis time within max_offset_seconds; the file's "
            "valid_time is that requested time for a multi-radar product "
            "and the volume's own time for a single-radar one; every "
            "contributing volume's valid time is written per radar"),
        "requested_source": args.source,
        "bucket": (per_site[0]["source"]["bucket"] or "front-door default"),
        "grid_wrfout": str(args.grid_wrfout),
        "grid_wrfout_sha256": grid_wrfout_sha,
        "range_authority_km": range_km,
        # One clear-air regime for the whole file: merge_contributions
        # refuses a mixture, so this is the single answer to "what were
        # these zeroes built from", and it is the value written as the
        # file's own clear_air_source attribute.
        "clear_air_source": observations.clear_air_source,
        "per_radar": [
            {key: entry[key] for key in (
                "site", "volume", "volume_key", "volume_object_keys",
                "volume_sha256", "volume_valid_time",
                "volume_offset_seconds", "feed", "partial_volume", "chunks",
                "feed_lag_seconds", "feed_observed_at", "antenna",
                "pack_schema")}
            for entry in per_site],
    }
    if len(per_site) == 1:
        # The keys the single-radar receipt has always carried, kept flat
        # so a reader written against the route that came first still
        # works.  The per_radar roster above is the general form.
        only = per_site[0]
        provenance.update({
            "site": only["site"],
            "volume_key": only["volume_key"],
            "volume_object_keys": only["volume_object_keys"],
            "volume_sha256": only["volume_sha256"],
            "volume_valid_time": only["volume_valid_time"],
            "volume_offset_seconds": only["volume_offset_seconds"],
            "feed": only["feed"],
            "partial_volume": only["partial_volume"],
            "chunks": only["chunks"],
            "feed_lag_seconds": only["feed_lag_seconds"],
            "feed_observed_at": only["feed_observed_at"],
            "antenna_source": only["antenna"]["source"],
            "antenna_alt_m": only["antenna"]["alt_m"],
            "pack_schema": only["pack_schema"],
        })

    receipt = write_radar_grid(
        args.out, observations, grid,
        valid_time=file_valid_time, params=params,
        provenance=provenance, overwrite=args.overwrite)

    # -- where the radars overlap, which is the whole point ---------------
    # Observation density is wildly uneven once coverage overlaps, and the
    # filter thins each radar's batch on its own -- so the overlap is
    # where the analysis gets two projections of one wind and also where
    # it gets twice the observation count. Both are measured here so a
    # reader can see the density they bought.
    vr_mask = observations.vr_mask.astype(bool)
    radars_seeing = vr_mask.sum(axis=0)
    overlap = {
        "cells_with_any_radar": int((radars_seeing > 0).sum()),
        "cells_by_radar_count": {
            str(n): int((radars_seeing == n).sum())
            for n in range(1, len(contributions) + 1)},
        "multi_radar_cells": int((radars_seeing > 1).sum()),
        "note": ("a cell seen by two radars carries two velocity "
                 "observations with different look directions, which is "
                 "the information a single radar cannot supply; it is "
                 "also twice the observation density, and the filter "
                 "thins each radar's batch independently, so thinning "
                 "does not equalise across the overlap"),
    }

    payload = {
        "schema": "gpuwm-obs.radar-grid-build.v1",
        "sites": [entry["site"] for entry in per_site],
        "sites_requested": list(sites),
        "sites_failed": failures,
        "site_selection": "discovered" if args.discover_sites else "named",
        "discovery": discovery,
        "radar_time_spread_seconds": spread,
        "range_authority_km": range_km,
        "per_radar": [
            {"site": entry["site"], "volume": entry["volume"],
             "volume_sha256": entry["volume_sha256"],
             "volume_valid_time": entry["volume_valid_time"],
             "volume_offset_seconds": entry["volume_offset_seconds"],
             "antenna": entry["antenna"], "source": entry["source"],
             "counts": entry["counts"],
             "pack_schema": entry["pack_schema"],
             "cells_vr": int(vr_mask[index].sum()),
             "cells_vr_rejected": int(
                 observations.vr_rejected[index].sum())}
            for index, entry in enumerate(per_site)],
        "cells": {
            "z": int(observations.z_mask.sum()),
            "z0": int(observations.z0_mask.sum()),
            "vr": int(vr_mask.sum()),
            "vr_rejected": int(observations.vr_rejected.sum()),
        },
        "overlap": overlap,
        "clear_air_source": observations.clear_air_source,
        "pairing": {"obs": str(args.out),
                    "grid_wrfout": str(args.grid_wrfout),
                    "grid_wrfout_sha256": grid_wrfout_sha},
        "writer_receipt": receipt,
    }
    if len(per_site) == 1:
        # Flat keys the single-radar route has always printed.
        only = per_site[0]
        payload.update({
            "site": only["site"], "volume": only["volume"],
            "volume_sha256": only["volume_sha256"],
            "volume_offset_seconds": only["volume_offset_seconds"],
            "source": only["source"], "antenna": only["antenna"],
            "counts": only["counts"],
            "pack_schema": only["pack_schema"],
        })
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
