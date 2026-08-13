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

Dual-pol QC: ``--cc-qc`` requests the RHO (correlation coefficient)
plane from the decode and enables the mask in :mod:`gpuwm.obs.cc_qc`,
whose rule is per moment.  In reflectivity low CC alone never drops a
gate, so hail cores, the melting layer and tornadic debris signatures
survive as echo.  In velocity there is no reflectivity shield: a
low-CC gate loses its velocity at every reflectivity, debris core
included, because a scatterer that does not move with the air is not a
wind observation.  The one exemption is the debris-signature fringe --
30 to 35 dBZ, CC above the debris floor, beside a clustered velocity
couplet in the same sweep -- which keeps its velocity because that band
is where a weak or distant tornado's rotation lives.  It is on with
``--cc-qc``; ``--cc-no-tds-fringe-exempt`` restores the strict rule,
and the receipt counts every gate the exemption kept and every
candidate each criterion turned away.

Off by default until the staged A/B proves it, and off is
byte-identical to a build from before the flag existed.  What the mask
did, sweep by sweep and moment by moment, lands in the receipt and the
NetCDF provenance -- per radar, since this tool builds one file from as
many radars as cover the grid.

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
    parser.add_argument("--cc-qc", action="store_true",
                        help="correlation-coefficient QC "
                             "(gpuwm.obs.cc_qc): request the RHO plane "
                             "from the decode and mask per moment -- "
                             "reflectivity drops only where RhoHV AND "
                             "reflectivity are both low, so hail cores, "
                             "the melting layer and tornadic debris "
                             "survive as echo; velocity drops wherever "
                             "RhoHV is low, at every reflectivity. Off "
                             "by default until the A/B proves it; off is "
                             "byte-identical to a build that predates "
                             "the flag")
    parser.add_argument("--cc-rho-min", type=float, default=None,
                        help="RhoHV threshold: compound with the shield "
                             "in reflectivity, alone in velocity "
                             "(default: the cited CcQcParams default)")
    parser.add_argument("--cc-ref-shield-dbz", type=float, default=None,
                        help="reflectivity at or above this is never "
                             "CC-dropped FROM REFLECTIVITY; it does not "
                             "protect velocity (default: the cited "
                             "CcQcParams default)")
    parser.add_argument("--cc-rho-min-velocity", type=float, default=None,
                        help="OPTIONAL separate RhoHV threshold for "
                             "velocity; unset means velocity uses "
                             "--cc-rho-min, which is the ruling. Set it "
                             "below the hail band (0.80, the TDS "
                             "criterion) to keep hail-core velocity "
                             "while still purging debris and biota")
    parser.add_argument("--cc-rho-floor", type=float, default=None,
                        help="OPTIONAL unconditional RhoHV floor, dropped "
                             "regardless of reflectivity. Deletes tornadic "
                             "debris signatures from the REFLECTIVITY "
                             "field at or above their RhoHV; off unless "
                             "explicitly given, and a decision about a "
                             "specific volume, never a habit")
    parser.add_argument("--cc-no-tds-fringe-exempt", action="store_true",
                        help="turn OFF the debris-signature fringe "
                             "exemption, restoring the strict velocity "
                             "rule: 30-35 dBZ low-CC gates beside a "
                             "velocity couplet lose their velocity like "
                             "every other low-CC gate. The exemption is "
                             "on with --cc-qc because a weak or distant "
                             "TDS lives in that band; this switch is the "
                             "way to measure what it costs")
    parser.add_argument("--cc-tds-rho-floor", type=float, default=None,
                        help="RhoHV below which a gate is never exempted "
                             "however much rotation surrounds it -- the "
                             "line between debris and receiver noise "
                             "(default: the cited CcQcParams default)")
    parser.add_argument("--cc-tds-ref-min-dbz", type=float, default=None,
                        help="reflectivity floor of the exemption band; "
                             "its ceiling is --cc-ref-shield-dbz, above "
                             "which the strict rule is untouched "
                             "(default: the cited CcQcParams default)")
    parser.add_argument("--cc-tds-couplet-delta-v-ms", type=float,
                        default=None,
                        help="velocity difference across one "
                             "adjacent-radial pair that seeds a couplet "
                             "(default: the cited CcQcParams default)")
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
                             "The default engine needs the region-global "
                             "shared library; gpuwm doctor prints how to "
                             "get it")
    add_dealias_engine_arguments(parser)
    return parser


def dealias_params_from_args(args, params_class, unavailable_reason):
    """``DealiasParams`` for these flags, or None -- refusing at the door.

    Shared by the three front doors so the refusal, the message and the
    parameter object are one implementation.  ``--dealias-refinement``
    without ``--dealias`` is refused rather than ignored: a run that asked
    for a treatment and silently did not get it is exactly the failure the
    A/B discipline exists to prevent.

    ``--dealias-refinement`` unset is None, not False, and it stays None
    all the way into ``DealiasParams``: the refinement default belongs to
    the engine (on for ``region-global``, meaningless for ``vad-region``),
    and a front door that resolved it to a bool here would have to know
    that table too.  Asked for explicitly beside the engine that has no
    such pass, it is refused by name at the door rather than as a
    traceback from the parameter object.
    """

    from gpuwm.obs.dealias import ENGINE_REGION_GLOBAL

    engine = getattr(args, "dealias_engine", None)
    refinement = getattr(args, "dealias_refinement", None)
    if not args.dealias:
        if refinement is not None:
            raise SystemExit(
                "--dealias-refinement/--no-dealias-refinement refines a "
                "dealiased field; pass --dealias as well, or drop it")
        return None
    if refinement and engine != ENGINE_REGION_GLOBAL:
        raise SystemExit(
            f"--dealias-refinement with --dealias-engine {engine}: that "
            f"engine has no refinement pass; only {ENGINE_REGION_GLOBAL} "
            "does. A switch that is accepted and then ignored is how a run "
            "gets reported as having had a treatment it never got")
    reason = unavailable_reason(engine)
    if reason is not None:
        raise SystemExit(f"--dealias-engine {engine}: {reason}")
    return params_class(engine=engine, refinement=refinement)


def add_dealias_engine_arguments(parser) -> None:
    """The engine selector, spelled the same way by every front door.

    One function so the three tools that build ``DealiasParams`` cannot
    describe the same two options differently -- which is how a reader
    ends up believing two front doors run different solvers.

    The refinement switch is a PAIR of flags with no default of its own.
    A ``store_true`` cannot express "leave it to the engine", and it
    cannot turn off a pass that is on by default -- and a default that
    can only be turned off by editing the source is not a default, it is
    a hardcoded value.
    """

    from gpuwm.obs.dealias import ENGINE_REGION_GLOBAL, ENGINES

    parser.add_argument("--dealias-engine", choices=list(ENGINES),
                        default=ENGINE_REGION_GLOBAL,
                        help="which solver unfolds a sweep. "
                             "'region-global' (default) is the vendored "
                             "region-global-dealias crate -- a Rust port "
                             "of Py-ART's dealias_region_based, verified "
                             "fold-for-fold identical to Py-ART on this "
                             "pipeline's own sweeps -- which unfolds the "
                             "region network jointly, carries no "
                             "environmental reference, and assigns a fold "
                             "to every region it resolved instead of "
                             "abstaining; its shared library must be "
                             "present (gpuwm doctor prints the build). "
                             "'vad-region' is gpuwm.obs.dealias: it "
                             "anchors regions to a fold-aware VAD fitted "
                             "from the volume and REJECTS every gate it "
                             "cannot justify, and needs scipy rather than "
                             "the library. The two make different "
                             "decisions by design; the engine is recorded "
                             "in provenance beside the velocities it made")
    refinement = parser.add_mutually_exclusive_group()
    refinement.add_argument("--dealias-refinement", dest="dealias_refinement",
                            action="store_true", default=None,
                            help="run the region-global engine's refinement "
                                 "pass, which considers small "
                                 "gate-resolution corrections where one "
                                 "connected region appears to hold two "
                                 "Nyquist branches and applies one only "
                                 "when an independent wrapped-vortex fit "
                                 "agrees. On by default with "
                                 "--dealias-engine region-global, and "
                                 "refused with any other engine")
    refinement.add_argument("--no-dealias-refinement",
                            dest="dealias_refinement", action="store_false",
                            help="turn the region-global refinement pass "
                                 "off. It abstains where it is unsure, so "
                                 "this is for isolating its effect, not "
                                 "for avoiding it")


def main(argv=None) -> int:
    from gpuwm.obs.radar_source import RadarSourceError, acquire_volume

    parser = build_parser()
    args = parser.parse_args(argv)
    # A contradictory radar request is refused before anything is read,
    # fetched or hashed: it costs nothing to catch here and the message is
    # the same either way.
    named_sites = requested_sites(args)

    # A CC threshold without --cc-qc tunes a mask that is not running.
    # Refused here rather than accepted as a silent no-op.
    if not args.cc_qc and any(value is not None for value in (
            args.cc_rho_min, args.cc_ref_shield_dbz,
            args.cc_rho_min_velocity, args.cc_rho_floor,
            args.cc_tds_rho_floor, args.cc_tds_ref_min_dbz,
            args.cc_tds_couplet_delta_v_ms)):
        parser.error("--cc-rho-min/--cc-ref-shield-dbz/"
                     "--cc-rho-min-velocity/--cc-rho-floor/"
                     "--cc-tds-rho-floor/--cc-tds-ref-min-dbz/"
                     "--cc-tds-couplet-delta-v-ms tune a mask "
                     "that --cc-qc turns on; a threshold without the "
                     "switch would be a silent no-op")
    if not args.cc_qc and args.cc_no_tds_fringe_exempt:
        parser.error("--cc-no-tds-fringe-exempt turns off part of a mask "
                     "that --cc-qc turns on; without --cc-qc there is no "
                     "exemption to withdraw and the switch would be a "
                     "silent no-op")

    from gpuwm.obs.nexrad import (find_nexrad_bin, nexrad_remedy, run_decode,
                                  run_verify)
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.dealias import DealiasParams, engine_unavailable_reason
    from gpuwm.obs.superob import (SuperobParams, merge_contributions,
                                   superob_volume)
    from gpuwm.obs.sweeps import read_sweep_pack
    from gpuwm.obs.target_grid import TargetGrid

    binary = find_nexrad_bin()
    if binary is None:
        raise SystemExit(f"no rw_nexrad front door: {nexrad_remedy()}")

    # Refuse before a byte is fetched, not from inside the gridding loop.
    dealias_params = dealias_params_from_args(args, DealiasParams,
                                     engine_unavailable_reason)

    target = _parse_iso(args.valid_time)
    range_km = (args.max_range_km if args.max_range_km is not None
                else SuperobParams().max_range_km)
    cc_params = None
    if args.cc_qc:
        from gpuwm.obs.cc_qc import CcQcParams
        overrides = {name: value for name, value in (
            ("rho_min", args.cc_rho_min),
            ("ref_shield_dbz", args.cc_ref_shield_dbz),
            ("rho_min_velocity", args.cc_rho_min_velocity),
            ("rho_floor", args.cc_rho_floor),
            ("tds_rho_floor", args.cc_tds_rho_floor),
            ("tds_ref_min_dbz", args.cc_tds_ref_min_dbz),
            ("tds_couplet_delta_v_ms", args.cc_tds_couplet_delta_v_ms),
        ) if value is not None}
        if args.cc_no_tds_fringe_exempt:
            overrides["tds_fringe_exempt"] = False
        cc_params = CcQcParams(**overrides)
    params = SuperobParams(max_range_km=range_km,
                           max_elevation_deg=args.max_elevation_deg,
                           dealias=dealias_params,
                           cc_qc=cc_params)
    params.validate()

    moments = ("REF", "VEL", "RHO") if args.cc_qc else ("REF", "VEL")

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
            # The RHO plane is requested exactly when the mask that
            # consumes it is on: a default build's pack stays
            # byte-identical, and an enabled build's pack self-describes
            # the extra plane (no schema bump -- the pack format has
            # always carried arbitrary per-moment planes).
            pack_path = args.work_dir / (selected.filename + ".pack")
            run_decode(binary, volume=volume_path, out=pack_path,
                       moments=moments, max_range_km=range_km,
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
    if cc_params is not None:
        # Present whenever --cc-qc was REQUESTED.  The guard is on the
        # parameters, not on any radar having found an RHO plane to run
        # against, so an all-zero block reads "asked for, nothing to
        # mask" and only its absence means "never asked".  That is
        # deliberately not the rule the observations file uses:
        # radar_grid._payload gates its own cc_qc provenance key on
        # `any(cc_qc)`, because a file written without the mask must keep
        # the exact key set it always had.  This is a receipt, not that
        # file, and a receipt that stayed silent about a mask that was
        # asked for and found nothing to do would be the less honest of
        # the two.  The per-radar cc_* counters already ride each entry's
        # "counts"; this is their sum across the radars that
        # contributed, so a reader sees the file-wide price of the mask
        # without adding up the roster by hand.
        cc_keys = ("cc_sweeps_masked", "cc_sweeps_paired_companion",
                   "cc_sweeps_without_rho", "cc_sweeps_without_ref",
                   "cc_gates_tested", "cc_gates_rho_missing",
                   "cc_velocity_gates_rejected",
                   "cc_reflectivity_gates_rejected",
                   "cc_velocity_gates_rejected_shielded_z",
                   # The debris-fringe exemption, as numbers a reader can
                   # audit: how many velocity gates it kept, how many
                   # clustered couplet seeds made that possible, and
                   # which conjunct turned every other low-CC velocity
                   # gate away.  The four refusals and the exemption are
                   # disjoint and together account for every gate the
                   # strict rule would have deleted.
                   "cc_velocity_gates_exempt_tds_fringe",
                   "cc_couplet_seed_gates",
                   "cc_velocity_tds_rho_below_floor",
                   "cc_velocity_tds_below_reflectivity",
                   "cc_velocity_tds_at_or_above_shield",
                   "cc_velocity_tds_no_couplet_nearby")
        payload["cc_qc"] = {
            "params": cc_params.to_payload(),
            "moments_requested": list(moments),
            "totals": {key: sum(int(entry["counts"].get(key, 0))
                                for entry in per_site)
                       for key in cc_keys},
            "per_radar": [
                {"site": entry["site"],
                 **{key: int(entry["counts"].get(key, 0))
                    for key in cc_keys}}
                for entry in per_site],
            # The per-sweep account, per radar, in contribution order.
            "sweeps": [contribution.cc_qc.get("sweeps", [])
                       for contribution in contributions],
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
