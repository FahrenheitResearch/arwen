"""Native SST verification: ``python -m gpuwm.ocean.mixed_layer.verify REQUEST``.

The JSON request names forecast NetCDFs, OISST NetCDFs, NDBC standard-met
files and station positions. Rust decodes, matches and scores; Python only
resolves the installed bridges. See README.md for the request contract.
"""
import argparse
from pathlib import Path
from gpuwm.obs.frontdoor import FrontDoor
from gpuwm.netcdf_bridge import resolve_netcdf_bin

ABI = 'gpuwm-ocean-sst-verify-v1\tdaily-mean\tndbc-WTMP\tgreat-circle\tpaired-change'
DOOR = FrontDoor('rw_ocean_verify', 'GPUWM_RW_OCEAN_VERIFY',
                 'ocean SST verification', ABI)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    binary = DOOR.require()
    ok, detail = DOOR.probe(binary)
    if not ok:
        raise RuntimeError(detail)
    record = DOOR.run('score', [str(args.request.resolve()), str(resolve_netcdf_bin())],
                      schema='gpuwm-ocean-sst-verification-v1', binary=binary)
    print(record['output'])


if __name__ == '__main__':
    main()
