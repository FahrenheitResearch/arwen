"""Fetch public release objects and pin the exact source used by this lane."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
repo = root / "WRF"
def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args])
if not repo.exists():
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "remote", "add", "origin", "https://github.com/wrf-model/WRF.git")
files = ["phys/module_sf_urban.F", "share/module_model_constants.F",
         "frame/module_wrf_error.F", "run/URBPARM.TBL", "run/URBPARM_LCZ.TBL",
         "phys/module_sf_noahdrv.F", "phys/module_sf_noahmpdrv.F",
         "phys/module_surface_driver.F", "Registry/Registry.EM_COMMON",
         "arch/Externals.cfg", ".gitmodules"]
receipt = {}
for tag in ("v4.6.1", "v4.7.0", "v4.7.1", "v4.8.0"):
    git(repo, "fetch", "--depth=1", "--filter=blob:none", "origin", f"refs/tags/{tag}:refs/tags/{tag}")
    dest = root / tag
    dest.mkdir(exist_ok=True)
    hashes = {}
    for rel in files:
        try:
            data = git(repo, "show", f"{tag}:{rel}")
        except subprocess.CalledProcessError:
            continue
        (dest / Path(rel).name).write_bytes(data)
        hashes[rel] = hashlib.sha256(data).hexdigest()
    (dest / "SOURCES.sha256").write_text("".join(f"{v}  {Path(k).name}\n" for k,v in hashes.items()))
    receipt[tag] = {"commit": git(repo, "rev-parse", f"{tag}^{{commit}}").decode().strip(), "files": hashes}
pins = {"GFL":"873b414670323ae766ec4fe1f4b6b5ab891358ed", "MYNN-EDMF":"8f8e8b59021249949c04d1686edc5a02ca325627",
        "MYNN-SFC":"b66187f54a0c169e36592552634867474e603ac0", "TEMPO":"881614f68aa65de8f383d56139162049baf600fa",
        "fire_behavior":"eb7758055a5d436f022a634cae780f5512c42bb5", "noahmp":"5da0b241e48ecfd9a2a1bd667ed554765856d589",
        "physics_mmm":"550b5b494758ad2756394e93cb08b5709b8feb83"}
import configparser
modules = configparser.ConfigParser()
modules.read(root / "v4.8.0" / ".gitmodules")
receipt["submodules"] = {}
for section in modules.sections():
    name = modules[section]["path"].split("/")[-1]
    if name not in pins:
        continue
    path = root / ("submodule-" + name)
    path.mkdir(exist_ok=True)
    if not (path / ".git").exists():
        git(path, "init", "-q")
        git(path, "remote", "add", "origin", modules[section]["url"])
    git(path, "fetch", "--depth=1", "--filter=blob:none", "origin", pins[name])
    resolved = git(path, "rev-parse", "FETCH_HEAD").decode().strip()
    assert resolved == pins[name]
    receipt["submodules"][name] = resolved
mmm = root / "submodule-physics_mmm"
git(mmm, "fetch", "--depth=1", "origin", "refs/tags/20240626-MPASv8.2")
receipt["mmm47"] = git(mmm, "rev-parse", "FETCH_HEAD").decode().strip()
mmm_files = {}
mmm_dest = root / "mmm47-src"
mmm_dest.mkdir(exist_ok=True)
for rel in git(mmm, "ls-tree", "-r", "--name-only", "FETCH_HEAD").decode().splitlines():
    if Path(rel).suffix.lower() in (".f", ".f90"):
        data = git(mmm, "show", f"FETCH_HEAD:{rel}")
        dest = mmm_dest / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        mmm_files[rel] = hashlib.sha256(data).hexdigest()
    if rel.endswith("module_sf_urban.F"):
        data = git(mmm, "show", f"FETCH_HEAD:{rel}")
        (root / "mmm47-module_sf_urban.F").write_bytes(data)
        receipt["mmm47_urban_sha256"] = hashlib.sha256(data).hexdigest()
receipt["mmm47_materialized"] = mmm_files
# WRF's Noah-MP root entries can be links or old forwarding files. Pin and
# compare the actual urban wrapper in each release's Noah-MP gitlink.
noahmp = root / "submodule-noahmp"
for tag in ("v4.6.1", "v4.7.0", "v4.7.1", "v4.8.0"):
    link = git(repo, "ls-tree", tag, "phys/noahmp").decode().split()[2]
    git(noahmp, "fetch", "--depth=1", "--filter=blob:none", "origin", link)
    paths = git(noahmp, "ls-tree", "-r", "--name-only", link).decode().splitlines()
    candidates = [p for p in paths if Path(p).name in
                  ("module_sf_noahmpdrv.F", "NoahmpUrbanDriverMainMod.F90")]
    assert candidates, f"{tag}: no Noah-MP urban wrapper in gitlink {link}"
    chosen = next((p for p in candidates if p.endswith("NoahmpUrbanDriverMainMod.F90")), candidates[0])
    data = git(noahmp, "show", f"{link}:{chosen}")
    (root / tag / "noahmp-urban-wrapper.F90").write_bytes(data)
    receipt[tag]["effective_noahmp_wrapper"] = {
        "gitlink": link, "path": chosen, "sha256": hashlib.sha256(data).hexdigest()}
(root / "source-pins.json").write_text(json.dumps(receipt, indent=2) + "\n")
