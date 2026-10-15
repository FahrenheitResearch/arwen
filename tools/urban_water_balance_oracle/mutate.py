"""Plant one bit in the changed water-conversion term and require rejection."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
kernel = Path(__file__).resolve().parents[2] / "gpuwm/core/kernels"
dest = root / "mutant"
dest.mkdir(exist_ok=True)
for name in ("urban_ucm.cu","urban_ucm_host.cpp","glibc_flt32.cuh"):
    shutil.copyfile(kernel/name,dest/name)
p = dest/"urban_ucm.cu"
text = p.read_text()
old = "FDIV(FMUL(flux, rho_air), 1000.0f)"
new = "FDIV(FMUL(flux, rho_air), __uint_as_float(0x447A0001u))"
assert text.count(old) == 1
p.write_text(text.replace(old,new))
library = dest/"mutant.so"
command = ["g++","-std=c++17","-O0","-fPIC","-shared","-fno-fast-math","-ffp-contract=off",str(dest/"urban_ucm_host.cpp"),"-o",str(library)]
subprocess.run(command,check=True)
check = [sys.executable,str(Path(__file__).with_name("terms.py")),str(root),str(library)]
result = subprocess.run(check)
assert result.returncode == 1, "the planted arithmetic error was not rejected"
(root/"mutation.json").write_text(json.dumps({"command":command,"check":check,
    "changed_term":old,"original_word":"0x447A0000","mutant_word":"0x447A0001",
    "oracle_exit":result.returncode},indent=2)+"\n")

# Flip one bit in the new generation predicate as a separate control for
# the changed clamp selection, leaving the retained unit conversion intact.
shutil.copyfile(kernel/"urban_ucm.cu",p)
text = p.read_text()
predicate = "if (generation == 0 && rain > 0.0f && water < prior)"
assert text.count(predicate) == 1
p.write_text(text.replace(predicate,predicate.replace("== 0","== 1")))
library = dest/"predicate-mutant.so"
command[-1] = str(library)
subprocess.run(command,check=True)
check[-1] = str(library)
result = subprocess.run(check)
assert result.returncode == 1,"the one-bit generation error was not rejected"
shutil.copyfile(root/"terms-mutant.json",root/"terms-predicate-mutant.json")
(root/"predicate-mutation.json").write_text(json.dumps({"command":command,"check":check,
    "changed_term":predicate,"original_word":"0x00000000","mutant_word":"0x00000001",
    "oracle_exit":result.returncode},indent=2)+"\n")
