"""Compile the WRF v4.6.1 km_opt=4 / diff_opt=2 column oracle library.

Every numerical routine is a byte-unmodified slice of pristine WRF v4.6.1
source (tag v4.6.1, commit d66e442f).  The slices are wrapped in one
module together with a configuration record holding only the fields those
routines read.  The model constants module is compiled whole and unchanged.
``pipeline.F90`` is the C ABI driver; it calls the WRF routines in the order
dyn_em/module_first_rk_step_part2.F calls them.

Build flags follow the strict reference: -O0, no FMA contraction, no
vectorisation, bounds checking on.  The receipt records the source hashes,
each extracted slice's line span and hash, the compiler and every command.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.diffopt1_wrf461_oracle.build import (  # noqa: E402
    CONFIG_SOURCE as _CONFIG_SOURCE,
    FLAGS as _FLAG_SETS,
    SOURCE_PINS as SOURCE_PINS,
    WRF_COMMIT as WRF_COMMIT,
    WRF_RELEASE as WRF_RELEASE,
    _build,
    extract_routines as extract_routines,
    periodic_top_slope_control as periodic_top_slope_control,
    sha256 as sha256,
)

DIFFUSION_ROUTINES = (
    "compute_diff_metrics", "cal_deform_and_div", "calculate_km_kh", "calculate_N2",
    "cal_dampkm", "isotropic_km", "smag_km", "smag2d_km", "tke_km", "calc_l_scale",
    "phy_bc", "horizontal_diffusion_2", "horizontal_diffusion_u_2",
    "horizontal_diffusion_v_2", "horizontal_diffusion_w_2", "horizontal_diffusion_s",
    "cal_titau_11_22_33", "cal_titau_12_21", "cal_titau_13_31", "cal_titau_23_32",
    "pthl", "pu")
EXTRA_ROUTINES = {
    "dyn_em/module_big_step_utilities_em.F": ("phy_prep",),
    "share/module_bc.F": ("set_physical_bc2d", "set_physical_bc3d"),
}
FLAGS = _FLAG_SETS["strict"]
CONFIG_SOURCE = (_CONFIG_SOURCE.replace("mix_full_fields=.false.", "mix_full_fields=.true.")
                 .replace("diff_opt=1", "diff_opt=2")
                 .replace("bl_pbl_physics=1", "bl_pbl_physics=0"))


def build(wrf: Path, output: Path, control: str | None = None) -> Path:
    return _build(wrf, output, routines=DIFFUSION_ROUTINES, extra_routines=EXTRA_ROUTINES,
                  config_source=CONFIG_SOURCE, pipeline=Path(__file__).with_name("pipeline.F90"),
                  compiler_flags={None: FLAGS}, flags=None, control=control,
                  receipt_fields={"schema": "smag2d-wrf461-oracle-build-v1",
                                  "wrf_release": WRF_RELEASE, "control": control,
                                  "wrf_commit": WRF_COMMIT, "source_sha256": SOURCE_PINS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wrf_source", type=Path, help="pristine WRF v4.6.1 checkout")
    parser.add_argument("output", type=Path)
    parser.add_argument("--control", choices=("periodic-top-slope",),
                        help="attribution-only modified WRF; never the reference")
    args = parser.parse_args()
    print(build(args.wrf_source, args.output, args.control))
