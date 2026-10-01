"""Record and pack a newly generated mosaic UCM corpus in the lane checkout."""
import hashlib
import sys
from pathlib import Path
from pack_fixtures import pack_case, _known_hashes, SHA_LIST

#: ucm/ucm_lcz carry the oracle's own FRC_URB2D words in every cell; the
#: _wrfinit pair keeps the fraction urban_var_init leaves, zero wherever the
#: dominant category is not urban (WRF's rule, run_mosaic_ucm.F90 families
#: 3 and 4).  Name families after the two paths to pack only those.
FAMILIES = ("ucm", "ucm_lcz", "ucm_wrfinit", "ucm_lcz_wrfinit")
raw, out = map(Path, sys.argv[1:3])
families = tuple(sys.argv[3:]) or FAMILIES
known = _known_hashes()
added = []
for family in families:
    for directory in sorted((raw / family).iterdir()):
        if not directory.is_dir():
            continue
        label = f"{family}/{directory.name}"
        for file in sorted(directory.glob("*.bin")):
            key = f"{label}/{file.name}"
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if key in known and known[key] != digest:
                raise ValueError(f"recorded words changed: {key}")
            if key not in known:
                added.append(f"{digest}  {key}\n")
                known[key] = digest
        pack_case(directory, out / family / (directory.name + ".npz"), known, label)
with SHA_LIST.open("a", encoding="ascii", newline="\n") as stream:
    stream.writelines(added)
print(f"recorded {len(added)} new raw hashes and packed {4 * len(families)} step fixtures")
