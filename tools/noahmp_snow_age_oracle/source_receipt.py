"""Stage pinned public sources on the lane box and record source dispositions."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

public, receipts = map(Path, sys.argv[1:])
receipts.mkdir(parents=True, exist_ok=True)
PINS = {"v4.6.1": "848f54ad3d28c4303151fe5ad83724e232694422",
        "v4.7.0": "e5c0859874407859936739e8be8741f9aed369ee",
        "v4.7.1": "e5c0859874407859936739e8be8741f9aed369ee",
        "v4.8.0": "5da0b241e48ecfd9a2a1bd667ed554765856d589"}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(public / repo), *args])


result, contents = {}, {}
for tag, pin in PINS.items():
    row = {"wrf_commit": git("WRF", "rev-parse", tag + "^{commit}").decode().strip(),
           "noahmp_commit": pin,
           "submodules": git("WRF", "ls-tree", tag, "phys").decode().strip(),
           "files": {}}
    links = git("WRF", "ls-tree", "-r", tag).decode().splitlines()
    row["submodules"] = [line for line in links if line.startswith("160000")]
    if tag == "v4.8.0":
        row["disposition"] = "Noah-MP 5.1.2 API/model port belongs to the separate generation lane."
        result[tag] = row
        continue
    full = public / ("full" + tag.replace("v", "").replace(".", ""))
    paths = [("WRF", tag, name) for name in (
        "share/module_model_constants.F", "phys/module_sf_gecros.F",
        "run/SOILPARM.TBL", "run/GENPARM.TBL", "Registry/Registry.EM_COMMON")]
    paths += [("noahmp", pin, name) for name in (
        "src/module_sf_noahmplsm.F", "src/module_sf_noahmp_glacier.F",
        "src/module_sf_noahmp_groundwater.F", "drivers/wrf/module_sf_noahmpdrv.F",
        "parameters/MPTABLE.TBL")]
    for repo, revision, name in paths:
        raw = git(repo, "show", revision + ":" + name)
        dest = full / (name if repo == "WRF" else "phys/noahmp/" + name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        row["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        contents[(tag, name)] = raw.decode()
    if tag != "v4.6.1":
        row["mmm_external"] = git("WRF", "show", tag + ":arch/Externals.cfg").decode()
        row["mmm_commit"] = git("physics_mmm", "rev-parse", "HEAD").decode().strip()
        row["mmm_tree_entries"] = len(git("physics_mmm", "ls-tree", "-r", "HEAD").decode().splitlines())
    result[tag] = row
for name in result["v4.7.0"]["files"]:
    old, new = contents[("v4.6.1", name)], contents[("v4.7.0", name)]
    if "Registry" in name:
        old = "\n".join(line for line in old.splitlines() if any(k in line.lower() for k in ("noahmp", "noah_mp", "tauss", "fage")))
        new = "\n".join(line for line in new.splitlines() if any(k in line.lower() for k in ("noahmp", "noah_mp", "tauss", "fage")))
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                      fromfile="v4.6.1/" + name, tofile="v4.7.0/" + name))
    (receipts / (Path(name).name + ".diff")).write_text(diff)
    result["v4.7.0"]["files"][name]["changed_since_461"] = bool(diff)
(receipts / "source-pins.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
