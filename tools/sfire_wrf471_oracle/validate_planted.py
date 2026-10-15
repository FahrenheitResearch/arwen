"""Require a measured one-word device fault, rather than a crashed control."""
import argparse
import json
from pathlib import Path

from tools.sfire_wrf471_oracle.regime_receipt import REGIMES, passed


def validate(receipt):
    for mode in ("strict", "default"):
        result = receipt["modes"][mode]
        if result.get("error") or result.get("status") != "FAIL" or result.get("exit_code") != 1:
            raise ValueError(f"{mode}: planted control did not finish with measured mismatches")
        if not result.get("planted_one_ulp") or not any(c["planted_one_ulp"] for c in result["compiles"]):
            raise ValueError(f"{mode}: the heat-flux kernel fault was not compiled")
        if set(result["regimes"]) != set(REGIMES):
            raise ValueError(f"{mode}: planted control did not grade all regimes")
        for name, regime in result["regimes"].items():
            if len(regime["steps"]) != 20 or not all(passed(g) for g in regime["initialization"].values()):
                raise ValueError(f"{mode}/{name}: initialization or carried step coverage differs")
            for step in regime["steps"]:
                fault = step["fields"]["fgrnhfx"]
                if fault["max_ulp"] != 1 or fault["different_words"] != 1 or fault["nonfinite_words"]:
                    raise ValueError(f"{mode}/{name}/{step['step']}: fault is not exactly one word at one ULP")
                for field in ("lfn", "tign", "ros", "fuel_frac", "fire_area", "fmc_g", "fgrnqfx"):
                    if not passed(step["fields"][field]):
                        raise ValueError(f"{mode}/{name}/{step['step']}: fault changed {field}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    validate(json.loads(args.receipt.read_text()))
    print("Planted device control measured exactly one word at one ULP in every step and mode.")
