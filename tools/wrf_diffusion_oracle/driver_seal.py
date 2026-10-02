"""Seal the complete-output compiled WRF driver fixture packet."""
import argparse
import hashlib
from pathlib import Path

def seal(root):
    root=Path(root).resolve()
    data=root/"tests/data/wrf471_diffusion"
    paths=[]
    for folder in ("horizontal-driver","vertical-driver","km-mutations"):
        paths.extend(p for p in (data/folder).iterdir() if p.is_file())
    tools=root/"tools/wrf_diffusion_oracle"
    for pattern in ("horizontal_driver*","vertical_driver*","deformation_mutations.py","deformation_compare.py",
                    "deformation_reference.py","deformation_arithmetic.py","horizontal_arithmetic.py",
                    "horizontal_compare.py","vertical_gpu.py","cases.py","build_common.py","driver_seal.py"):
        paths.extend(p for p in tools.glob(pattern) if p.is_file())
    paths.extend(tools/name for name in ("vertical_build.py","horizontal_build.py","deformation_build.py"))
    paths.append(root/"gpuwm/verify/diffusion_oracle.py")
    paths.append(root/"tests/test_diffusion_drivers_wrf471_parity.py")
    lines=[hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.relative_to(root).as_posix() for p in sorted(set(paths))]
    (data/"diffusion-driver-sha256sums.txt").write_text("\n".join(lines)+"\n",encoding="utf-8", newline="\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root")
    seal(p.parse_args().root)
