"""Derive the RRTMGP trace-gas climatology table from the RFMIP input file.

    python tools/derive_rrtmgp_trace_climatology.py [--source PATH] [--check]

RRTMGP's column driver needs three things from the RFMIP clear-sky input
file and nothing else: the experiment-zero global-mean mole fraction of
each well-mixed gas, the median layer pressure over the 100 sites and the
median experiment-zero ozone profile on those layers.  The file itself is
not redistributed (its embedded licence attribute names CC-BY-SA-4.0 and
links CC-BY-4.0), so the package carries only these 136 derived numbers,
in ``gpuwm-data/gpuwm_data/data/rrtmgp/rrtmgp-trace-gas-climatology.json``
under CC-BY-SA-4.0, the stricter of the two readings.

The derivation is the exact arithmetic ``RRTMGPRadiation`` performed when
it opened the NetCDF at model construction (float64 median over sites;
``float(GM[0]) * units``), and JSON keeps every float64 exactly, so the
arrays the model uploads are the bytes it uploaded before.
``tests/test_rrtmgp_trace_climatology.py`` re-derives the table from the
upstream file and requires byte equality.

Without ``--source`` the pinned upstream file is fetched into the RFMIP
cache (:mod:`gpuwm.core.rfmip_upstream`).  NetCDF is read here, in a build
tool, never on the model's data path: the package itself carries no NetCDF
reader for this file.  ``--check`` compares instead of
writing and exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gpuwm.core.rfmip_upstream import RFMIP_FILES, fetch_rfmip  # noqa: E402


def derive_trace_climatology(inputs: Path) -> dict:
    """The numbers RRTMGP's column driver reads from the RFMIP inputs.

    Exactly the arithmetic the driver ran on the NetCDF: experiment-zero
    global means times their ``units`` scale, and the float64 median over
    sites of the layer pressure and of the experiment-zero ozone.
    """

    import numpy as np
    from netCDF4 import Dataset  # a build tool, never on the model's data path

    from gpuwm.core.trace_gases import RFMIP_GAS_NAMES

    _url, size, sha256, licence = RFMIP_FILES["rfmip-clear-sky-inputs.nc"]
    with Dataset(inputs, "r") as ncfile:
        ncfile.set_auto_mask(False)
        trace = {}
        for gas, rfmip_name in RFMIP_GAS_NAMES.items():
            variable = ncfile[rfmip_name + "_GM"]
            scale = float(getattr(variable, "units", "1").replace(" ", ""))
            trace[gas] = float(variable[0]) * scale
        pressure = np.median(
            np.asarray(ncfile["pres_layer"][:], np.float64), axis=0)
        ozone = np.median(np.asarray(ncfile["ozone"][0], np.float64), axis=0)
    return {
        "schema": "gpuwm.rrtmgp-trace-gas-climatology.v1",
        "source": {
            "file": RFMIP_FILES["rfmip-clear-sky-inputs.nc"][0],
            "bytes": size,
            "sha256": sha256,
            "producer": "University of Colorado, RFMIP (CMIP6 input4MIPs "
                        "UColorado-RFMIP-1-2)",
            "licence_as_published": licence,
        },
        "licence": "CC-BY-SA-4.0",
        "derivation": "experiment 0; trace_vmr = GM[0] * units; "
                      "pressure_layer_pa and ozone_vmr are float64 medians "
                      "over the 100 sites, in the file's layer order",
        "trace_vmr": trace,
        "pressure_layer_pa": [float(value) for value in pressure],
        "ozone_vmr": [float(value) for value in ozone],
    }


def render_trace_climatology(table: dict) -> str:
    """The committed text form: stable key order, exact float64 repr."""

    return json.dumps(table, indent=1, sort_keys=True) + "\n"


TARGET = (REPO / "gpuwm-data" / "gpuwm_data" / "data" / "rrtmgp"
          / "rrtmgp-trace-gas-climatology.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=None,
                        help="a local copy of the pinned RFMIP input file")
    parser.add_argument("--check", action="store_true",
                        help="compare with the committed table; write nothing")
    args = parser.parse_args(argv)
    source = (fetch_rfmip("rfmip-clear-sky-inputs.nc", path=args.source)
              if args.source is not None
              else fetch_rfmip("rfmip-clear-sky-inputs.nc"))
    text = render_trace_climatology(derive_trace_climatology(source))
    if args.check:
        same = TARGET.is_file() and TARGET.read_bytes() == text.encode("utf-8")
        print("identical" if same else f"DIFFERS: {TARGET}")
        return 0 if same else 1
    TARGET.write_bytes(text.encode("utf-8"))
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
