"""Build unchanged WRF 4.7.1 coordinate and exchange-coefficient routines."""
from pathlib import Path
import argparse
from build_common import build

ROUTINES = ("cal_deform_and_div", "calculate_N2", "smag2d_km", "tke_km",
            "calc_l_scale", "pthl", "pu", "compute_diff_metrics")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wrf_source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    build(a.wrf_source / "dyn_em/module_diffusion_em.F",
          a.wrf_source / "share/module_model_constants.F",
          Path(__file__).with_name("wrapper.F90"), a.output, ROUTINES,
          extra_sources={a.wrf_source / "dyn_em/module_big_step_utilities_em.F":
                         ["horizontal_diffusion", "horizontal_diffusion_3dmp"]})
