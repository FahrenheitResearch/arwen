"""Grade the declared deterministic correction after checked WRF rejects it."""
from pathlib import Path
import json
import subprocess
import sys

from gpuwm.verify.ysu_wide_oracle import load_wide_oracle, run_port_wide
from tools.ysu_generations_oracle.gate import grade

root, out = map(Path, sys.argv[1:])
work = root / "shallow-corrected"
work.mkdir(parents=True, exist_ok=True)
(work / "ysu-wide-inputs.bin").write_bytes((root / "low-top-undefined.bin").read_bytes())
result = subprocess.run([str(root / "v4.8.0/checked/oracle"),
                         str(work / "ysu-wide-inputs.bin"), str(work / "undefined.bin")],
                        capture_output=True, text=True)
assert result.returncode != 0 and "qke1d" in result.stderr, "checked source did not expose the undefined array read"
subprocess.run([str(root / "v4.8.0/bounded/oracle"), str(work / "ysu-wide-inputs.bin"),
                str(work / "ysu-wide-wrf.bin")], check=True, stdout=subprocess.DEVNULL)
cols = load_wide_oracle(work)
port = run_port_wide(cols)
fields, states, cells = grade(cols, port, step=0)
receipt = {"declared_reference_correction": "Bound get_pblh to kte and set the unwritten final TKE word to zero.",
           "unmodified_checked_exit": result.returncode, "unmodified_checked_error": result.stderr,
           "fields": fields, "states": states, "different_cells": cells}
receipt["pass"] = not cells and all(s["different"] == 0 for s in states.values())
out.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"pass": receipt["pass"], "different_cells": len(cells)}))
raise SystemExit(0 if receipt["pass"] else 1)
