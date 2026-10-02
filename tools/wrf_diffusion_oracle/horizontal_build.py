"""Build direct WRF v4.7.1 horizontal diffusion operators."""
from pathlib import Path
import argparse
from build_common import build

ROUTINES=("horizontal_diffusion_u_2","horizontal_diffusion_v_2","horizontal_diffusion_w_2",
          "horizontal_diffusion_s","cal_titau_11_22_33","cal_titau_12_21","cal_titau_13_31","cal_titau_23_32")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path); p.add_argument("constants",type=Path); p.add_argument("output",type=Path)
    a=p.parse_args()
    build(a.source,a.constants,Path(__file__).with_name("horizontal_wrapper.F90"),a.output,ROUTINES)
