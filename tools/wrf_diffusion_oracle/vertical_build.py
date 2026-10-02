"""Build the direct WRF vertical diffusion and TKE source-chain oracles."""
from __future__ import annotations

from pathlib import Path
import argparse
from build_common import build

VERTICAL_ROUTINES = ["vertical_diffusion_u_2", "vertical_diffusion_v_2",
                     "vertical_diffusion_w_2", "vertical_diffusion_s", "vertical_diffusion_2",
                     "cal_titau_13_31", "cal_titau_23_32", "cal_titau_11_22_33"]
TKE_ROUTINES = ["tke_rhs", "tke_shear", "tke_buoyancy", "tke_dissip",
                "calc_l_scale", "pu"]


def build_vertical(source_root, output_dir):
    here = Path(__file__).resolve().parent
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Preserve both complete adapters; the builder accepts a single unit.
    wrapper = out / "vertical_wrappers.F90"
    wrapper.write_bytes((here / "vertical_wrapper.F90").read_bytes() + b"\n" +
                        (here / "vertical_tke_wrapper.F90").read_bytes() + b"\n" +
                        (here / "vertical_driver_wrapper.F90").read_bytes())
    root = Path(source_root)
    return build(root / "dyn_em/module_diffusion_em.F",
                 root / "share/module_model_constants.F", wrapper, out,
                 VERTICAL_ROUTINES + TKE_ROUTINES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(build_vertical(args.source_root, args.output_dir))
