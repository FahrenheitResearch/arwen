"""Record scheme, wrappers, tables and Registry source differences by release."""
from pathlib import Path
import difflib
import gzip
import hashlib
import json
import sys

root = Path(sys.argv[1]).resolve()
names = ("module_sf_urban.F","module_sf_noahdrv.F","module_sf_noahmpdrv.F",
         "noahmp-urban-wrapper.F90", "module_surface_driver.F",
         "Registry.EM_COMMON","URBPARM.TBL","URBPARM_LCZ.TBL")
receipts = []
for before,after in (("v4.6.1","v4.7.0"),("v4.7.0","v4.7.1"),("v4.7.1","v4.8.0")):
    for name in names:
        a = root/"sources"/before/name
        b = root/"sources"/after/name
        if not a.exists() or not b.exists():
            continue
        diff = "".join(difflib.unified_diff(a.read_text().splitlines(True),b.read_text().splitlines(True),
                                           fromfile=f"{before}/{name}",tofile=f"{after}/{name}"))
        output = root/f"source-diff-{before}-{after}-{name}.gz"
        output.write_bytes(gzip.compress(diff.encode(),mtime=0))
        receipts.append({"before":before,"after":after,"name":name,
           "before_sha256":hashlib.sha256(a.read_bytes()).hexdigest(),
           "after_sha256":hashlib.sha256(b.read_bytes()).hexdigest(),"diff":output.name,
           "diff_lines":len(diff.splitlines()),
           "disposition":"water-balance clamps implemented; rho_air/1000 retained; vegetation, trees, shadows and other 4.8 changes outside this lane"})
(root/"source-scope.json").write_text(json.dumps(receipts,indent=2)+"\n")
