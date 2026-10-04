"""Package the native captures needed by the SPP consumer gates."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stems = ("turbulence", "turbulence-high", "condensation", "dmp_mf", "driver", "surface")
    names = [stem + "-" + mode + ".csv" for stem in stems for mode in ("spp", "off")]
    names += ["gf-spp-levels.csv", "gf-spp-surface.csv", "gf-off-levels.csv",
              "gf-off-surface.csv", "gf-deep-surface.csv", "gf-shallow-surface.csv",
              "gf-inert-stock-driver.csv", "receipt.json"]
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name in sorted(names):
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(info, (args.oracle / name).read_bytes())


if __name__ == "__main__":
    main()
