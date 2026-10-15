"""Fail while the actual forecast runtime lacks the 4.8 ground contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpuwm.checkpoint_identity import URBAN_UCM_GENERATION_IDENTITIES
from gpuwm.core.urban_state import option_spec
from gpuwm.core.urban_ucm import UCM480_TABLE_COLUMNS, pack_params
from gpuwm.core.urban_tables import load_urban_params


GROUND_STATE = ("tvg_urb2d", "tt_urb2d", "xxxvg_urb2d", "tvgl_urb3d",
                "smg_urb3d", "cmcg_urb2d", "flxhumvg_urb2d", "flxhumt_urb2d")
GROUND_TABLE = ("fvg", "lai_veg", "z0vg", "z0hvg", "capvg", "epsvg", "albvg")


def contract():
    spec=option_spec(1,generation='wrf_480')
    packed=pack_params(load_urban_params(1,generation='wrf_480'))
    absent_state = [n for n in GROUND_STATE if n not in spec]
    absent_table = [n for n in GROUND_TABLE if n not in UCM480_TABLE_COLUMNS]
    absent_switch = [] if packed.switches.shape==(14,) else ['treeoption','ttscheme','water_balance']
    return {"complete": not (absent_state or absent_table or absent_switch),
            "missing_state": absent_state, "missing_table": absent_table,
            "missing_switches": absent_switch,
            "current_restart_identity": URBAN_UCM_GENERATION_IDENTITIES['wrf_480'],
            "scope": "Structural runtime contract only; numerical and forecast qualification require their independent checks.",
            "evidence": {"state": "gpuwm/core/urban_state.py:option_spec",
                         "table": "gpuwm/core/urban_ucm.py:UCM480_TABLE_COLUMNS",
                         "switches": "gpuwm/core/urban_ucm.py:pack_params",
                         "identity": "gpuwm/checkpoint_identity.py:URBAN_UCM_GENERATION_IDENTITIES"}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    result=contract()
    a.receipt.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return 0 if result["complete"] else 1


if __name__=="__main__":raise SystemExit(main())
