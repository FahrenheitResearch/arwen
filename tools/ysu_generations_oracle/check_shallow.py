"""Grade the declared deterministic correction after checked WRF rejects it.

    check_shallow.py ROOT OUT [--fixture replay|record|verify]

replay (the default) grades the port against the committed bounded
reference and the committed record of the checked build's rejection, so a
GPU leg needs no Fortran toolchain.  record and verify run the build.sh
oracles under ROOT; record writes the committed files, verify requires them.
"""
from pathlib import Path
import argparse
import json
import subprocess

from gpuwm.verify.ysu_wide_oracle import load_wide_oracle, run_port_wide
from tools.ysu_generations_oracle.gate import FIXTURE, grade, xz_read, xz_write

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--fixture", choices=("replay", "record", "verify"), default="replay")
parser.add_argument("--fixture-dir", type=Path, default=FIXTURE)
args = parser.parse_args()
root, out = args.root, args.out
committed = args.fixture_dir / "shallow"
work = root / "shallow-corrected"
work.mkdir(parents=True, exist_ok=True)
if args.fixture == "replay":
    (work / "ysu-wide-inputs.bin").write_bytes(xz_read(committed / "low-top-undefined.bin.xz"))
    (work / "ysu-wide-wrf.bin").write_bytes(xz_read(committed / "bounded.bin.xz"))
    rejection = json.loads((committed / "checked-rejection.json").read_text())
else:
    (work / "ysu-wide-inputs.bin").write_bytes((root / "low-top-undefined.bin").read_bytes())
    result = subprocess.run([str(root / "v4.8.0/checked/oracle"),
                             str(work / "ysu-wide-inputs.bin"), str(work / "undefined.bin")],
                            capture_output=True, text=True)
    subprocess.run([str(root / "v4.8.0/bounded/oracle"), str(work / "ysu-wide-inputs.bin"),
                    str(work / "ysu-wide-wrf.bin")], check=True, stdout=subprocess.DEVNULL)
    # The checked build's message names the array and the out-of-bounds index;
    # the traceback that follows carries build paths and is not recorded.
    message = next((line.strip() for line in result.stderr.splitlines() if "qke1d" in line), "")
    rejection = {"returncode": result.returncode, "message": message}
    if args.fixture == "record":
        xz_write(committed / "low-top-undefined.bin.xz", (work / "ysu-wide-inputs.bin").read_bytes())
        xz_write(committed / "bounded.bin.xz", (work / "ysu-wide-wrf.bin").read_bytes())
        (committed / "checked-rejection.json").write_text(json.dumps(rejection, indent=2) + "\n")
    else:
        assert (work / "ysu-wide-inputs.bin").read_bytes() == xz_read(committed / "low-top-undefined.bin.xz")
        assert (work / "ysu-wide-wrf.bin").read_bytes() == xz_read(committed / "bounded.bin.xz"), \
            "rebuilt bounded reference differs from the committed one"
        assert rejection == json.loads((committed / "checked-rejection.json").read_text())
assert rejection["returncode"] != 0 and "qke1d" in rejection["message"], \
    "checked source did not expose the undefined array read"
cols = load_wide_oracle(work)
port = run_port_wide(cols)
fields, states, cells = grade(cols, port, step=0)
receipt = {"declared_reference_correction": "Bound get_pblh to kte and set the unwritten final TKE word to zero.",
           "fixture": args.fixture,
           "unmodified_checked_exit": rejection["returncode"], "unmodified_checked_error": rejection["message"],
           "fields": fields, "states": states, "different_cells": cells}
receipt["pass"] = not cells and all(s["different"] == 0 for s in states.values())
out.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"pass": receipt["pass"], "different_cells": len(cells)}))
raise SystemExit(0 if receipt["pass"] else 1)
