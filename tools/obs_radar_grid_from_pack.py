"""Rebuild one observation file from an already-decoded sweep pack.

``tools/obs_radar_grid_build.py`` is the production spine and owns the
whole chain -- fetch, decode, verify, superob, grid, receipt.  This tool
owns only the tail of it::

    (pack already on disk) -> superob -> grid -> receipt

and it exists for one reason: **re-deriving an observation product from
the same bytes after the gridding code has changed.**  When a new
observation type is added to the writer, every file built before it
lacks the new variables, and the only way to give an old case the new
product without changing the observation itself is to run the new code
over the identical decoded volume.

Going back through the fetch and decode stages to do that would be
wrong twice over.  It needs the network, so the reconstruction stops
being reproducible; and it re-runs a decoder whose output is not
guaranteed to be the byte-for-byte pack the original run consumed, so
the reconstruction would silently be of a *different* volume.  Reading
the retained pack removes both risks: the input to the superobber is
provably the same array of gates the first build used, so any difference
in the output is attributable to the gridding code and to nothing else.

Nothing here re-implements the superobber, the merge or the writer.  It
calls :func:`gpuwm.obs.superob.superob_volume`,
:func:`gpuwm.obs.superob.merge_contributions` and
:func:`gpuwm.obs.radar_grid.write_radar_grid`, in that order, with the
same :class:`~gpuwm.obs.superob.SuperobParams` the original build
recorded -- which is why ``--max-range-km`` is REQUIRED here while the
production builder defaults it.  A reconstruction that quietly picked a
different range authority than the file it is replacing would produce a
plausible, wrong answer, and the range authority is exactly the value
the production builder was written to stop disagreeing about.

The written provenance says ``rebuilt_from_pack`` and carries the
original build record's volume identity forward, so a reader can always
tell a reconstruction from a first build and can still name the S3
object the gates came from.

EXPERIMENTAL, like the rest of this lane.  No case names belong here:
sites, times, packs and grids are arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# The engine selector is defined once, by the production spine, and
# imported here rather than restated: "same flag, same defaults and same
# provenance as tools.obs_radar_grid_build" is a claim this file makes in
# its own --dealias help, and two copies of an argparse block is how that
# claim stops being true.
from tools.obs_radar_grid_build import (add_dealias_engine_arguments,
                                        dealias_params_from_args)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.obs_radar_grid_from_pack",
        description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True,
                        help="the retained sweep pack (gpuwm-obs.radar-"
                             "sweeps.v1) the original build decoded")
    parser.add_argument("--grid-wrfout", type=Path, required=True,
                        help="the wrfout whose georeference the "
                             "observations are gridded onto; must be the "
                             "same one the original build used, and its "
                             "SHA-256 joins the receipt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-build", type=Path, default=None,
                        help="the original gpuwm-obs.radar-grid-build.v1 "
                             "record. Its volume identity (object key, "
                             "sha256, feed) is carried into the rebuilt "
                             "file's provenance so the reconstruction can "
                             "still name where its gates came from")
    parser.add_argument("--max-range-km", type=float, required=True,
                        help="THE range authority. Required, not defaulted: "
                             "this must be the value the pack was decoded "
                             "with and the file being replaced was built "
                             "with, and a default that disagreed would "
                             "produce a plausible wrong answer")
    parser.add_argument("--max-elevation-deg", type=float, required=True,
                        help="likewise -- the original build's value, not a "
                             "default")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clear-air-from-censor", action="store_true",
                        help="build clear-air zeroes from the RDA's own "
                             "below-threshold gate code as well as from "
                             "finite below-floor gates. Needs a "
                             "gpuwm-obs.radar-sweeps.v2 pack (decode with "
                             "--censor-flags); a v1 pack is a hard error "
                             "rather than a silent fallback, because the "
                             "thin product published under the wide "
                             "regime's clear_air_source would claim a "
                             "coverage it does not have. Range-folded gates "
                             "stay excluded either way")
    parser.add_argument("--dealias", action="store_true",
                        help="unfold radial velocity per sweep before "
                             "gridding instead of masking every gate that "
                             "might be folded. Same flag, same defaults and "
                             "same provenance as "
                             "tools.obs_radar_grid_build --dealias, so a "
                             "pack rebuilt through this path is comparable "
                             "with one built through that one. The default "
                             "engine needs the region-global shared "
                             "library; gpuwm doctor prints how to get it")
    add_dealias_engine_arguments(parser)
    return parser


def main(argv=None) -> int:
    from gpuwm.obs.dealias import DealiasParams, engine_unavailable_reason
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import (SuperobParams, merge_contributions,
                                   superob_volume)
    from gpuwm.obs.sweeps import read_sweep_pack
    from gpuwm.obs.target_grid import TargetGrid

    args = build_parser().parse_args(argv)

    params = SuperobParams(max_range_km=args.max_range_km,
                           max_elevation_deg=args.max_elevation_deg,
                           dealias=dealias_params_from_args(
                               args, DealiasParams,
                               engine_unavailable_reason))
    params.validate()

    grid = TargetGrid.from_wrfout(args.grid_wrfout)
    grid_wrfout_sha = _sha256(args.grid_wrfout)

    volume = read_sweep_pack(args.pack)
    contribution = superob_volume(
        volume, grid, params=params,
        clear_air_from_censor=args.clear_air_from_censor)
    observations = merge_contributions([contribution], grid, params=params)

    origin = {}
    if args.source_build is not None:
        record = json.loads(args.source_build.read_text(encoding="utf-8"))
        origin = {
            "original_build_record": str(args.source_build),
            "volume": record.get("volume"),
            "volume_sha256": record.get("volume_sha256"),
            "volume_offset_seconds": record.get("volume_offset_seconds"),
            "source": record.get("source"),
            "range_authority_km": record.get("range_authority_km"),
        }
        recorded_range = record.get("range_authority_km")
        if (recorded_range is not None
                and float(recorded_range) != float(args.max_range_km)):
            # A rebuild is only a rebuild if it observes the same volume the
            # same way.  Different range authority, different observation.
            raise SystemExit(
                f"--max-range-km {args.max_range_km} disagrees with the "
                f"original build record's range_authority_km "
                f"{recorded_range}. This would not be a reconstruction of "
                f"{args.source_build.name}; it would be a different "
                "observation wearing its name")

    receipt = write_radar_grid(
        args.out, observations, grid,
        valid_time=volume.valid_time, params=params,
        provenance={
            "builder": "tools/obs_radar_grid_from_pack.py",
            "rebuilt_from_pack": str(args.pack),
            "pack_sha256": _sha256(args.pack),
            "site": volume.site.id,
            "antenna_source": volume.site.source,
            "antenna_alt_m": float(volume.site.alt_m),
            "grid_wrfout": str(args.grid_wrfout),
            "grid_wrfout_sha256": grid_wrfout_sha,
            "range_authority_km": float(args.max_range_km),
            "pack_schema": volume.pack_schema,
            "clear_air_source": contribution.clear_air_source,
            "origin": origin,
        },
        overwrite=args.overwrite)

    payload = {
        "schema": "gpuwm-obs.radar-grid-rebuild.v1",
        "site": volume.site.id,
        "pack": str(args.pack),
        "valid_time": volume.valid_time,
        "range_authority_km": float(args.max_range_km),
        "pack_schema": volume.pack_schema,
        "clear_air_source": contribution.clear_air_source,
        "counts": contribution.counts.to_payload(),
        "cells": {
            "z": int(observations.z_mask.sum()),
            "z0": (None if observations.z0_mask is None
                   else int(observations.z0_mask.sum())),
            "vr": int(observations.vr_mask.sum()),
            "vr_rejected": int(observations.vr_rejected.sum()),
        },
        "origin": origin,
        "writer_receipt": receipt,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
