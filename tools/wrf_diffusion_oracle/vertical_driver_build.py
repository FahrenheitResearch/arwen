"""Compile the complete-output WRF vertical driver storage adapter."""
from pathlib import Path
import argparse
from build_common import build
from vertical_build import VERTICAL_ROUTINES

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path);p.add_argument("constants",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    build(a.source,a.constants,Path(__file__).with_name("vertical_driver_complete.F90"),
          a.output,VERTICAL_ROUTINES)
