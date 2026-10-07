"""Extract unchanged WRF driver/constants and nearest-point source bytes."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "driver": "7662f29bb003697d08ccee585e272a3ed7362edc73bcef5ea8dc95d100d9e578",
    "core": "7679945766e20219934daacfa4b174ffea9cabdd442ccd347ccde5a3afda7bd2",
    "constants": "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = dict(driver=ROOT.parent / "sfire_wrf471_oracle/reference/phys/module_fr_fire_driver.F",
                 core=ROOT.parent / "sfire_wrf471_oracle/reference/phys/module_fr_fire_core.F",
                 constants=Path(__file__).parent / "reference/module_model_constants.F")
    source = {}
    for key, path in paths.items():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != PINS[key]:
            raise ValueError("native geographic source pin differs: " + key)
        source[key] = raw.decode()
    declarations = "\n".join(line for line in source["constants"].splitlines()
                               if "PARAMETER :: reradius " in line or "PARAMETER ::  pi2=" in line)
    assert len(declarations.splitlines()) == 2
    start = source["driver"].index("       unit_fxlat=pi2/(360.*reradius)")
    stop = source["driver"].index("\n", source["driver"].index("       unit_fxlong=cos(", start))
    body = source["driver"][start:stop]
    start_near = source["core"].index("SUBROUTINE nearest(")
    stop_near = source["core"].index("END SUBROUTINE nearest", start_near)+len("END SUBROUTINE nearest")
    nearest = source["core"][start_near:stop_near]
    wrapper = Path(__file__).with_name("wrapper.F90").read_text()
    wrapper = wrapper.replace("!NATIVE_CONSTANTS", declarations).replace("!NATIVE_UNITS", body).replace("!NATIVE_NEAREST", nearest)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "native.F90").write_text(wrapper, newline="\n")
    receipt = dict(source_sha256=PINS, original_expression_sha256=hashlib.sha256(body.encode()).hexdigest(),
                   original_nearest_sha256=hashlib.sha256(nearest.encode()).hexdigest(),
                   original_constants_sha256=hashlib.sha256(declarations.encode()).hexdigest(),
                   scientific_transformations=[], wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest())
    (args.output / "extraction.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
