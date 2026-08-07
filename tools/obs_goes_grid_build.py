"""Build one ``gpuwm-obs.goes-grid.v1`` file from GOES packs and a grid.

The satellite twin of ``tools/obs_radar_grid_build.py``: packs in, one
gridded observation set out, bound to a model grid by digest, with a
receipt that names every choice.

    python tools/obs_goes_grid_build.py \
        --cwp-pack scan.cwp.goespack \
        --cloudtop-pack scan.cloudtop.goespack \
        --grid-wrfout wrfout_d01_2026-08-04_18:00:00 \
        --valid-time 2026-08-04T18:01:17Z \
        --out goes_cwp_1801.nc \
        --err-clear-g-m2 20 --err-rel-liquid 0.3 --err-floor-liquid-g-m2 40 \
        --err-rel-ice 0.5 --err-floor-ice-g-m2 80

The five ``--err-*`` flags are **required and have no defaults**.  There
is no measured CWP observation-error covariance for this system, so there
is nothing to default to; the numbers are the operator's stated
assumption and they are written into the product labelled UNCALIBRATED.
Anything else would let a stage inherit confidence nobody earned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpuwm.obs.goes_cwp import (JOIN_METHODS, CwpErrorModel, SuperobPolicy,
                                grid_cwp, join_cloud_top, no_join_receipt,
                                read_cloudtop_pack, read_cwp_pack)
from gpuwm.obs.goes_grid import write_goes_grid
from gpuwm.obs.target_grid import TargetGrid

RECEIPT_SCHEMA = "gpuwm-obs.goes-grid-build.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cwp-pack", required=True,
                        help="the 2 km gpuwm-obs.goes-cwp.v1 pack")
    parser.add_argument("--cloudtop-pack", default=None,
                        help="the 10 km gpuwm-obs.goes-cloudtop.v1 pack. "
                             "Without it every observation is centred at "
                             "--fallback-placement-agl-m and the receipt "
                             "says so")
    parser.add_argument("--join-method", default="nearest",
                        choices=JOIN_METHODS,
                        help="how the 10 km cloud-top grid is resampled "
                             "onto the 2 km CWP grid. This is the "
                             "interpolation the bridge deliberately "
                             "refuses to make; choosing it here is the "
                             "point")
    parser.add_argument("--coverage-slack", type=float, default=0.5,
                        help="how far outside the cloud-top grid, in its "
                             "own cells, a CWP pixel may sit and still be "
                             "served")
    parser.add_argument("--grid-wrfout", required=True,
                        help="wrfout whose georeference defines the target "
                             "grid")
    parser.add_argument("--grid-frame", type=int, default=0)
    parser.add_argument("--valid-time", required=True,
                        help="the analysis time this product is for, "
                             "ISO8601")
    parser.add_argument("--out", required=True)
    parser.add_argument("--receipt", default=None,
                        help="write the JSON receipt here as well as to "
                             "stdout")
    parser.add_argument("--overwrite", action="store_true")

    errors = parser.add_argument_group(
        "observation error (UNCALIBRATED; all required, no defaults)")
    errors.add_argument("--err-clear-g-m2", type=float, required=True,
                        help="error on a clear-sky zero")
    errors.add_argument("--err-rel-liquid", type=float, required=True,
                        help="relative error on a liquid CWP")
    errors.add_argument("--err-floor-liquid-g-m2", type=float, required=True)
    errors.add_argument("--err-rel-ice", type=float, required=True,
                        help="relative error on an ice/mixed CWP; must be "
                             ">= --err-rel-liquid, because the upstream ice "
                             "coefficient is flagged PROVISIONAL")
    errors.add_argument("--err-floor-ice-g-m2", type=float, required=True)
    errors.add_argument(
        "--err-thin-inflation", type=float, default=1.0,
        help="multiply sigma_o where the DCOMP thin-cloud bit (256) is "
             "set. Needs a gpuwm-obs.goes-cwp.v2 pack; >1.0 against a v1 "
             "pack is refused rather than silently skipped. 1.0 (the "
             "default) is the v1-equivalent behaviour")
    errors.add_argument(
        "--err-thick-inflation", type=float, default=1.0,
        help="the same for the thick-cloud bit (512). Measured on a live "
             "CONUS scan, 20.7%% of pixels carry it")

    policy = parser.add_argument_group("superob gates")
    policy.add_argument("--min-pixels", type=int, default=1)
    policy.add_argument("--min-valid-fraction", type=float, default=0.5)
    policy.add_argument("--phase-uniform-fraction", type=float, default=1.0,
                        help="1.0 is the design note's rule: a cell half "
                             "clear and half deep ice is not one "
                             "observation")
    policy.add_argument("--fallback-placement-agl-m", type=float,
                        default=3000.0)
    policy.add_argument("--max-derivation-mismatch-fraction", type=float,
                        default=0.0,
                        help="tolerated fraction of pixels whose cwp plane "
                             "does not reproduce from the pack's own cod, "
                             "cps, phase and coefficients. 0 refuses any")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    error_model = CwpErrorModel(
        clear_g_m2=args.err_clear_g_m2,
        rel_liquid=args.err_rel_liquid,
        floor_liquid_g_m2=args.err_floor_liquid_g_m2,
        rel_ice=args.err_rel_ice,
        floor_ice_g_m2=args.err_floor_ice_g_m2,
        thin_inflation=args.err_thin_inflation,
        thick_inflation=args.err_thick_inflation)
    policy = SuperobPolicy(
        min_pixels=args.min_pixels,
        min_valid_fraction=args.min_valid_fraction,
        phase_uniform_fraction=args.phase_uniform_fraction,
        fallback_placement_agl_m=args.fallback_placement_agl_m)
    error_model.validate()
    policy.validate()

    cwp_pack = read_cwp_pack(args.cwp_pack)
    cloud_top = None
    if args.cloudtop_pack is not None:
        cloudtop_pack = read_cloudtop_pack(args.cloudtop_pack)
        cloud_top, join_receipt = join_cloud_top(
            cwp_pack, cloudtop_pack, method=args.join_method,
            coverage_slack=args.coverage_slack)
    else:
        join_receipt = no_join_receipt(
            "--cloudtop-pack was not given")

    grid = TargetGrid.from_wrfout(args.grid_wrfout, frame=args.grid_frame)
    observations = grid_cwp(
        cwp_pack, grid, error_model=error_model, policy=policy,
        cloud_top_m=cloud_top, join_receipt=join_receipt,
        max_derivation_mismatch_fraction=(
            args.max_derivation_mismatch_fraction))

    written = write_goes_grid(
        args.out, observations, grid, valid_time=args.valid_time,
        provenance={
            "tool": "tools/obs_goes_grid_build.py",
            "cwp_pack": str(args.cwp_pack),
            "cloudtop_pack": (None if args.cloudtop_pack is None
                              else str(args.cloudtop_pack)),
            "grid_wrfout": str(args.grid_wrfout),
            "grid_frame": int(args.grid_frame),
        },
        overwrite=args.overwrite)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "stability": "experimental",
        "product": written,
        "counts": observations.counts,
        "join": join_receipt,
        "error_model": error_model.to_payload(),
        "superob": policy.to_payload(),
        "packs": {
            "cwp": cwp_pack.provenance(),
            "cloudtop": (None if args.cloudtop_pack is None
                         else cloudtop_pack.provenance()),
        },
    }
    encoded = json.dumps(receipt, indent=2, default=str)
    if args.receipt is not None:
        Path(args.receipt).write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
