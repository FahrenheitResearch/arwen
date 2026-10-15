"""Extract unmodified WRF 4.6.1 aerosol routines for the column oracle."""
import hashlib
from pathlib import Path
import re
import subprocess
import sys
root, output = map(Path, sys.argv[1:])
revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if not revision.startswith("d66e442"):
    raise SystemExit(f"WRF 4.6.1 required, got {revision}")
parts = []
pins = {}
for source, names in (("module_physics_init.F", ("aerosol_in", "interp_vec")),
                      ("module_radiation_driver.F", ("aer_time_int", "aer_p_int"))):
    path = root / "phys" / source
    raw = path.read_bytes()
    pins[source] = hashlib.sha256(raw).hexdigest()
    text = raw.decode()
    for name in names:
        match = re.search(r"^\s*subroutine " + name + r"\(.*?^\s*end subroutine " + name + r"\b[^\n]*", text, re.M | re.S | re.I)
        if not match:
            raise SystemExit(f"missing WRF authority {name}")
        parts.append(match.group())
output.write_text("module aer1_authority\ncontains\n" + "\n".join(parts) + "\nend module\n")
import json
output.with_suffix(".json").write_text(json.dumps({"wrf_commit": revision, "source_sha256": pins}, indent=2)+"\n")
