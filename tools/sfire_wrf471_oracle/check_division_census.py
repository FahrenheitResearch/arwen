"""Run the production PTX census on SFIRE's actual source compositions."""
from pathlib import Path
import argparse
import json
from tools.literal_division_census import production_units, census_unit, rewrite_sites


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    units = [unit for unit in production_units(root)
             if unit.key.startswith("kernels:sfire_") or unit.key == "kernels:chem_sfire"]
    if len(units) != 10:
        raise ValueError("The SFIRE census must measure all ten observed fire, smoke and initialization CUDA compositions")
    result = {unit.key: {"census_89": census_unit(unit,"compute_89"),
                        "rewrite_120": rewrite_sites(unit,"compute_120")} for unit in units}
    Path(args.receipt).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
