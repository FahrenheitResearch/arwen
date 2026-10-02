"""Build the unmodified WRF horizontal diffusion outer driver and leaves."""
from pathlib import Path
import argparse
from build_common import build
from horizontal_build import ROUTINES

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path);p.add_argument("constants",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args()
    build(a.source,a.constants,Path(__file__).with_name("horizontal_driver_wrapper.F90"),a.output,
          ("horizontal_diffusion_2",*ROUTINES))
