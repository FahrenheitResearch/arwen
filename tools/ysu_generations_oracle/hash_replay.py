"""Hash every numeric history field through the Rust NetCDF reader."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from gpuwm.netcdf_bridge import open_dataset, resolve_netcdf_bin

root, out = map(Path, sys.argv[1:])
receipt = {"decoder": str(resolve_netcdf_bin()), "runs": {}}
for mode in ("default", "legacy", "baseline"):
    fields = {}
    paths = sorted((root / f"run-{mode}" / "wrfout").glob("wrfout_d??*"))
    assert paths, f"{mode} produced no history frames"
    for path in paths:
        if not path.is_file():
            continue
        with open_dataset(path) as dataset:
            for name, variable in dataset.variables.items():
                if variable.is_character:
                    continue
                variable.set_auto_maskandscale(False)
                values = np.ascontiguousarray(variable[:])
                fields[path.name + ":" + name] = {
                    "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "shape": list(values.shape), "dtype": str(values.dtype),
                    "words": int(values.size)}
    receipt["runs"][mode] = fields
baseline = receipt["runs"]["baseline"]
for mode in ("default", "legacy"):
    fields = receipt["runs"][mode]
    receipt[mode + "_different_fields"] = [name for name in sorted(set(baseline) | set(fields)) if baseline.get(name) != fields.get(name)]
receipt["legacy_identical"] = not receipt["legacy_different_fields"]
out.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"legacy_identical": receipt["legacy_identical"],
                  "numeric_fields": {k: len(v) for k, v in receipt["runs"].items()},
                  "default_different_fields": len(receipt["default_different_fields"])}))
assert receipt["legacy_identical"], "legacy generation changed the full-field replay"
