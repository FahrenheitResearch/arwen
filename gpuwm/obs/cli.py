"""``gpuwm obs`` — reach the observation front doors from an installed gpuwm.

Everything under here already existed as a library call or a ``tools/``
script. What did not exist was a way to *run* it: the European polar-volume
route was complete from the decoder down to the LETKF adapter and had no door,
which by this project's own rule means it did not exist. A capability reachable
only from a source checkout, a demo script or a private function is not
shipped.

So this module is deliberately thin. It owns argument parsing, the refusals
that argument parsing can make, and printing a JSON record. Every computation
belongs to the module it already belonged to:
:mod:`gpuwm.obs.odim` for decoding, :mod:`gpuwm.obs.radar_sites` for the site
table, and :func:`gpuwm.obs.superob.superob_volume` /
:func:`gpuwm.obs.radar_grid.write_radar_grid` for gridding.

The route, end to end, from a directory of files a national feed served::

    gpuwm obs radar doctor                     is the decoder built, and is it
                                               the one this gpuwm expects
    gpuwm obs radar volumes --dir D            what volumes are in here
    gpuwm obs radar pack --dir D --out P       assemble and decode one of them
    gpuwm obs radar nyquist --file F           can its velocity be unfolded
    gpuwm obs radar grid --pack P ...          superob it onto a model grid

``--file`` takes the one-file volumes the Netherlands and Romania publish;
``--dir`` takes the per-elevation single-sweep files Germany publishes and
assembles them. A NEXRAD Archive-II volume goes through
``gpuwm.obs.nexrad``/``rw_nexrad`` and lands in the same pack format, which is
why ``grid`` names neither continent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _print(record: dict) -> int:
    print(json.dumps(record, indent=2, default=str))
    return 0


def _quantities(value: str | None) -> list[str] | None:
    if not value:
        return None
    names = [part.strip() for part in value.split(",") if part.strip()]
    return names or None


def _bbox(value: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--bbox takes west,south,east,north; got {value!r}")
    try:
        west, south, east, north = (float(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--bbox {value!r} is not four numbers: {error}") from error
    if south >= north:
        raise argparse.ArgumentTypeError(
            f"--bbox south {south} is not below north {north}")
    return west, south, east, north


# --------------------------------------------------------------------------
# radar doctor


def _radar_doctor(args) -> int:
    from gpuwm.obs.frontdoor import ODIM

    found = ODIM.find()
    if found is None:
        # Not an exception: "it is not built" is an answer to the question
        # this subcommand asks, and the remedy is the useful half of it.
        return _print({
            "schema": "gpuwm-obs.radar-doctor.v1",
            "front_door": ODIM.name,
            "found": None,
            "abi_matches": False,
            "detail": f"{ODIM.subject} is not built or not found",
            "remedy": ODIM.remedy(),
            "searched": [str(path) for path in ODIM.candidates()],
        })
    ok, detail = ODIM.probe(found)
    return _print({
        "schema": "gpuwm-obs.radar-doctor.v1",
        "front_door": ODIM.name,
        "found": str(found),
        "abi_matches": bool(ok),
        "detail": detail,
        "remedy": None if ok else ODIM.remedy(),
        "searched": [str(path) for path in ODIM.candidates()],
    })


# --------------------------------------------------------------------------
# radar volumes / pack / nyquist


def _radar_volumes(args) -> int:
    from gpuwm.obs.odim import run_volumes

    return _print(run_volumes(directory=args.dir))


def _radar_pack(args) -> int:
    from gpuwm.obs.odim import run_pack, run_pack_dir

    if (args.file is None) == (args.dir is None):
        raise SystemExit(
            "pack needs exactly one of --file and --dir. --file is a "
            "whole-volume ODIM file (the Netherlands, Romania); --dir is a "
            "directory of the single-sweep files some feeds publish instead "
            "(Germany), which are assembled by nominal time. They are two "
            "shapes of volume, not two ways of naming one")
    if args.file is not None:
        if args.stamp is not None:
            raise SystemExit(
                "--stamp selects which volume to assemble out of a directory "
                "and means nothing for a single --file, which is already one "
                "volume")
        record = run_pack(file=args.file, out=args.out,
                          quantities=_quantities(args.quantities),
                          max_elevation_deg=args.max_elevation_deg,
                          max_range_km=args.max_range_km)
    else:
        record = run_pack_dir(directory=args.dir, out=args.out,
                              stamp=args.stamp,
                              quantities=_quantities(args.quantities),
                              max_elevation_deg=args.max_elevation_deg,
                              max_range_km=args.max_range_km)
    return _print(record)


def _radar_nyquist(args) -> int:
    from gpuwm.obs.odim import run_nyquist

    return _print(run_nyquist(file=args.file))


# --------------------------------------------------------------------------
# radar sites


def _radar_sites(args) -> int:
    from gpuwm.obs.radar_sites import (SiteNotAssimilableError, coverage_summary,
                                       load_table, require_assimilable,
                                       sites_for_bbox)

    table = load_table()
    if args.bbox is not None:
        west, south, east, north = args.bbox
        selected = sites_for_bbox(west, south, east, north, table=table)
    elif args.site is not None:
        selected = tuple(site for site in table.sites if site.id == args.site)
        if not selected:
            raise SystemExit(
                f"no site {args.site!r} in the table; "
                f"it holds {len(table.sites)} sites")
    else:
        selected = tuple(table.sites)

    refusal = None
    if args.require_assimilable:
        try:
            require_assimilable(selected, need_velocity=not args.no_velocity)
        except SiteNotAssimilableError as error:
            refusal = str(error)

    return _print({
        "schema": "gpuwm-obs.radar-sites-query.v1",
        "table": {
            "schema": table.schema,
            "source_url": table.source_url,
            "frozen_at": table.frozen_at,
            "elevation_basis": table.elevation_basis,
        },
        "coverage": coverage_summary(table),
        "selected": len(selected),
        "assimilable_checked": bool(args.require_assimilable),
        # Suppressed rather than defaulted to True: a check that did not run
        # has no verdict, and printing one would be a claim nobody made.
        "assimilable": (None if not args.require_assimilable
                        else refusal is None),
        "refusal": refusal,
        "sites": [
            {"id": site.id, "name": site.name,
             "latitude": site.latitude, "longitude": site.longitude,
             "elevation_m": site.elevation_m,
             "has_velocity": site.has_velocity,
             "has_volume_scan": site.has_volume_scan,
             "moments": list(site.moments)}
            for site in selected
        ],
    })


# --------------------------------------------------------------------------
# radar grid


def _radar_grid(args) -> int:
    from gpuwm.obs.dealias import SCIPY_REMEDY, DealiasParams, scipy_available
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import (SuperobParams, merge_contributions,
                                   superob_volume)
    from gpuwm.obs.sweeps import read_sweep_pack
    from gpuwm.obs.target_grid import TargetGrid

    if args.dealias and not scipy_available():
        raise SystemExit(SCIPY_REMEDY)

    params = SuperobParams(
        max_range_km=args.max_range_km,
        max_elevation_deg=args.max_elevation_deg,
        dealias=DealiasParams() if args.dealias else None).validate()

    grid = TargetGrid.from_wrfout(args.grid_wrfout)
    volume = read_sweep_pack(args.pack)
    contribution = superob_volume(
        volume, grid, params=params,
        clear_air_from_censor=args.clear_air_from_censor)
    observations = merge_contributions([contribution], grid, params=params)

    receipt = write_radar_grid(
        args.out, observations, grid,
        valid_time=volume.valid_time, params=params,
        provenance={
            "builder": "gpuwm obs radar grid",
            "pack": str(args.pack),
            "pack_sha256": _sha256(args.pack),
            "site": volume.site.id,
            "antenna_source": volume.site.source,
            "antenna_alt_m": float(volume.site.alt_m),
            "grid_wrfout": str(args.grid_wrfout),
            "grid_wrfout_sha256": _sha256(args.grid_wrfout),
            "range_authority_km": float(args.max_range_km),
            "pack_schema": volume.pack_schema,
            "clear_air_source": contribution.clear_air_source,
        },
        overwrite=args.overwrite)

    return _print({
        "schema": "gpuwm-obs.radar-grid-build.v2",
        "site": volume.site.id,
        "pack": str(args.pack),
        "pack_schema": volume.pack_schema,
        "valid_time": volume.valid_time,
        "range_authority_km": float(args.max_range_km),
        "clear_air_source": contribution.clear_air_source,
        "counts": contribution.counts.to_payload(),
        "cells": {
            "z": int(observations.z_mask.sum()),
            "z0": (None if observations.z0_mask is None
                   else int(observations.z0_mask.sum())),
            "vr": int(observations.vr_mask.sum()),
            "vr_rejected": int(observations.vr_rejected.sum()),
        },
        "writer_receipt": receipt,
    })


# --------------------------------------------------------------------------
# registration


def _register_radar(sub) -> None:
    radar = sub.add_parser(
        "radar",
        help="decode a radar volume into a sweep pack and grid it onto a "
             "model domain")
    radar_sub = radar.add_subparsers(dest="obs_radar_command", required=True)

    doctor = radar_sub.add_parser(
        "doctor",
        help="is the polar-volume decoder built, and is it the build this "
             "gpuwm's record contract expects")
    doctor.set_defaults(func=_radar_doctor)

    volumes = radar_sub.add_parser(
        "volumes",
        help="which volumes a directory of ODIM files holds, grouped by the "
             "nominal time inside each file")
    volumes.add_argument("--dir", type=Path, required=True, metavar="DIR",
                         help="directory of ODIM .h5 files, not searched "
                              "recursively")
    volumes.set_defaults(func=_radar_volumes)

    pack = radar_sub.add_parser(
        "pack",
        help="decode one ODIM polar volume into a gpuwm-obs.radar-sweeps.v3 "
             "pack")
    pack.add_argument("--file", type=Path, default=None, metavar="H5",
                      help="one whole-volume ODIM file (PVOL). Mutually "
                           "exclusive with --dir")
    pack.add_argument("--dir", type=Path, default=None, metavar="DIR",
                      help="directory of single-sweep ODIM files (SCAN) to "
                           "assemble into one volume, as Germany publishes "
                           "them. Mutually exclusive with --file")
    pack.add_argument("--stamp", default=None, metavar="YYYYmmddTHHMMSSZ",
                      help="which volume in --dir to assemble, in the "
                           "spelling `volumes` reports. Required only when "
                           "the directory holds more than one: taking the "
                           "newest silently would put a volume nobody asked "
                           "for behind an ordinary-looking record")
    pack.add_argument("--out", type=Path, required=True, metavar="PACK",
                      help="sweep pack to write")
    pack.add_argument("--quantities", default=None, metavar="Q,Q",
                      help="ODIM quantity names to carry (DBZH,VRADH). "
                           "Omitting this carries every quantity in the "
                           "volume, which is nine of them on a Dutch scan")
    pack.add_argument("--max-elevation-deg", type=float, default=None,
                      metavar="DEG",
                      help="drop cuts above this elevation. The 90-degree "
                           "birdbath a Dutch volume opens with is a "
                           "calibration cut, not an observation of anything a "
                           "model column exists for")
    pack.add_argument("--max-range-km", type=float, default=None,
                      metavar="KM",
                      help="trim gates beyond this range")
    pack.set_defaults(func=_radar_pack)

    nyquist = radar_sub.add_parser(
        "nyquist",
        help="per-sweep Nyquist interval and its provenance: the dealias "
             "handoff")
    nyquist.add_argument("--file", type=Path, required=True, metavar="H5",
                         help="one ODIM file; geometry is read, no payload")
    nyquist.set_defaults(func=_radar_nyquist)

    sites = radar_sub.add_parser(
        "sites",
        help="the frozen European radar site table, and whether a selection "
             "of it can be assimilated")
    sites.add_argument("--bbox", type=_bbox, default=None,
                       metavar="W,S,E,N",
                       help="select the sites inside this lon/lat box")
    sites.add_argument("--site", default=None, metavar="ID",
                       help="select one site by its table id")
    sites.add_argument("--require-assimilable", action="store_true",
                       help="also run the assimilability check over the "
                            "selection and report its refusal in full. "
                            "Without this the verdict field is null rather "
                            "than true: a check that did not run has no "
                            "verdict")
    sites.add_argument("--no-velocity", action="store_true",
                       help="do not require a radial-velocity moment in the "
                            "assimilability check, for a reflectivity-only "
                            "assimilation")
    sites.set_defaults(func=_radar_sites)

    grid = radar_sub.add_parser(
        "grid",
        help="superob a sweep pack onto a model domain and write the "
             "observation file the LETKF adapter reads")
    grid.add_argument("--pack", type=Path, required=True, metavar="PACK",
                      help="the sweep pack, from `pack` or from rw_nexrad")
    grid.add_argument("--grid-wrfout", type=Path, required=True,
                      metavar="WRFOUT",
                      help="the wrfout whose georeference the observations "
                           "are gridded onto; its SHA-256 joins the receipt")
    grid.add_argument("--out", type=Path, required=True, metavar="NC",
                      help="observation file to write")
    grid.add_argument("--max-range-km", type=float, required=True,
                      metavar="KM",
                      help="THE range authority, required rather than "
                           "defaulted: a build that quietly picked a "
                           "different range than the one it is compared "
                           "against produces a plausible, wrong answer")
    grid.add_argument("--max-elevation-deg", type=float, required=True,
                      metavar="DEG",
                      help="likewise: the elevation ceiling, stated rather "
                           "than defaulted")
    grid.add_argument("--clear-air-from-censor", action="store_true",
                      help="build clear-air zeroes from the decoder's own "
                           "gate codes as well as from finite below-floor "
                           "gates. Needs a pack carrying censor planes (v2 "
                           "or v3); a v1 pack is a hard error rather than a "
                           "silent fallback. Range-folded and ambiguous "
                           "gates stay excluded either way")
    grid.add_argument("--dealias", action="store_true",
                      help="unfold radial velocity per sweep before gridding "
                           "instead of masking every gate that might be "
                           "folded. Requires scipy")
    grid.add_argument("--overwrite", action="store_true",
                      help="replace an existing --out")
    grid.set_defaults(func=_radar_grid)


# ------------------------------------------------- the instrument doors

def _print_estate(explain: bool) -> int:
    """Every front door, where it is, and whether it is current."""

    from gpuwm.obs.frontdoor import FRONT_DOORS

    print("gpuwm obs: observation front doors "
          f"({len(FRONT_DOORS)} instrument(s))")
    missing = 0
    for instrument in sorted(FRONT_DOORS):
        door = FRONT_DOORS[instrument]
        try:
            found = door.find()
        except FileNotFoundError as error:
            # An environment override naming a missing file: explicit
            # configuration that is wrong, which is a different fault
            # from an install that never staged the binary.
            print(f"  {instrument:<8} OVERRIDE  {error}")
            missing += 1
            continue
        if found is None:
            print(f"  {instrument:<8} MISSING   {door.subject} "
                  f"({door.name}) is not staged")
            missing += 1
            continue
        ok, evidence = door.probe(found)
        state = "ok" if ok else "STALE"
        print(f"  {instrument:<8} {state:<9} {found}")
        print(f"           {evidence}")
        if not ok:
            missing += 1
    if missing:
        print(f"gpuwm obs: {missing} of {len(FRONT_DOORS)} front door(s) are "
              "not usable; run `gpuwm obs <instrument>` for the remedy "
              "for that one, or `gpuwm doctor` for the whole estate")
    else:
        print(f"gpuwm obs: all {len(FRONT_DOORS)} front doors resolved and "
              "speak this release's record contract")
    if explain:
        print("\n  why: these binaries acquire and decode the observation "
              "archives\n"
              "  gpuwm scores and assimilates against (MRMS, Stage-IV, "
              "ASOS/METAR, GOES ABI,\n"
              "  the European composite and European polar volumes).  They "
              "ship in the\n"
              "  bundle `gpuwm fetch-bridges` stages, and gpuwm.obs.sources "
              "reads the packs\n"
              "  they write.  A missing one is an incomplete install, not an\n"
              "  unexercised option.")
    # The estate printed successfully; what it found is data, not this
    # command's outcome.  A caller that wants a gate runs the
    # instrument, which refuses.
    return 0


def _obs_estate(args) -> int:
    return _print_estate(bool(getattr(args, "explain", False)))


def _instrument_main(args) -> int:
    """Resolve one instrument's binary and hand it ``ARGS`` unchanged."""

    import subprocess
    import sys

    from gpuwm.obs.frontdoor import FRONT_DOORS

    instrument = args.obs_command
    door = FRONT_DOORS[instrument]
    try:
        binary = door.require()
    except (FileNotFoundError, RuntimeError) as error:
        # The named refusal frontdoor composes, at the boundary rather
        # than as a traceback.  A door owns its own refusals.
        print(f"gpuwm obs {instrument}: {error}", file=sys.stderr)
        return 2
    forwarded = list(getattr(args, "argv", None) or [])
    if not forwarded:
        # No arguments at all: the binaries print their usage on an
        # empty argv, which is the right answer to "what can this do".
        forwarded = ["--help"]
    try:
        completed = subprocess.run([str(binary), *forwarded], check=False)
    except OSError as error:
        print(f"gpuwm obs {instrument}: {binary} failed to launch: "
              f"{error}", file=sys.stderr)
        return 2
    return int(completed.returncode)


#: The instrument doors, spelled here rather than read from
#: :data:`gpuwm.obs.frontdoor.FRONT_DOORS`, and the one-line summary each
#: gets in ``gpuwm obs --help``.  A subparser has to exist before argv is
#: seen, and the alternative -- one positional plus ``REMAINDER`` -- gives
#: ``gpuwm obs`` two grammars, because ``radar`` is a real subcommand tree
#: with its own flags and cannot be a passthrough.  The names are held
#: against the resolver's own table by
#: ``tests/test_obs_front_doors_are_shipped.py``, so a door added to one
#: and not the other fails rather than going unreachable.
_INSTRUMENTS = {
    "mrms": "MRMS composite reflectivity",
    "stage4": "Stage-IV precipitation accumulation",
    "asos": "ASOS/METAR surface reports",
    "goes": "GOES ABI cloud-product packs",
    "opera": "the European composite-reflectivity mosaic",
    "odim": "European per-site polar volumes (the raw route "
            "`gpuwm obs radar` drives)",
}


def _register_instruments(sub) -> None:
    for instrument, subject in sorted(_INSTRUMENTS.items()):
        parser = sub.add_parser(
            instrument,
            help=f"acquire and decode {subject}: resolves the front door's "
                 "binary and passes ARGS to it unchanged "
                 f"(`gpuwm obs {instrument} --help` prints its grammar)")
        parser.add_argument(
            "argv", nargs=argparse.REMAINDER, metavar="ARGS",
            help="arguments passed to the instrument's binary unchanged; "
                 "gpuwm's own flags must come before the instrument name")
        parser.set_defaults(func=_instrument_main)


def register_cli(subparsers) -> None:
    """Add ``gpuwm obs`` to the product CLI.

    One subcommand tree, not two registrations.  ``radar`` is the
    decode-and-grid route that owns its own flags; the instrument names
    are passthrough doors onto the binaries
    :mod:`gpuwm.obs.frontdoor` resolves.  ``gpuwm obs`` with neither
    prints the estate, which is the only question that is about all of
    them at once.
    """

    obs = subparsers.add_parser(
        "obs", help="fetch, decode and grid observations")
    # Not `required=True`: bare `gpuwm obs` prints where every front
    # door resolved, which is the answer to "is my observation estate
    # actually here" and is worth more than an argparse usage error.
    obs_sub = obs.add_subparsers(dest="obs_command", required=False)
    obs.set_defaults(func=_obs_estate)
    _register_radar(obs_sub)
    _register_instruments(obs_sub)


__all__ = ["register_cli"]
