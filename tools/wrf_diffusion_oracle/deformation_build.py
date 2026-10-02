"""Build the WRF v4.7.1 deformation and mixing-coefficient oracle."""
from __future__ import annotations

import argparse
from pathlib import Path
from build_common import build

ROUTINES = ("cal_deform_and_div", "calculate_km_kh", "calculate_N2", "cal_dampkm",
            "isotropic_km", "smag_km", "smag2d_km", "tke_km", "calc_l_scale",
            "pthl", "pu", "compute_diff_metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("constants", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bc", required=True, type=Path)
    parser.add_argument("--big-step", required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.constants, Path(__file__).with_name("deformation_wrappers.F90"),
          args.output, ROUTINES,
          extra_sources={args.bc: ["set_physical_bc3d"], args.big_step: ["phy_prep"]},
          module_declarations="   INTEGER, PARAMETER            :: bdyzone = 4\n")


if __name__ == "__main__":
    main()
