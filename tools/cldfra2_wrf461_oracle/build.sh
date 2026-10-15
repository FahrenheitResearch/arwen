#!/usr/bin/env bash
set -euo pipefail
root=$(realpath "$1"); out=$(realpath -m "$2")
here=$(cd "$(dirname "$0")" && pwd)
# Pin the release tag and refuse edits to the public driver.
test "$(git -C "$root" describe --tags --exact-match HEAD)" = v4.6.1
git -C "$root" diff --exit-code HEAD -- phys/module_radiation_driver.F
mkdir -p "$out"
python3 - "$root" "$out" <<'PY'
import pathlib, re, sys
root, out = map(pathlib.Path, sys.argv[1:])
s = (root/'phys/module_radiation_driver.F').read_text()
m = re.search(r'(?is)^   SUBROUTINE cal_cldfra2\(.*?^   END SUBROUTINE cal_cldfra2', s, re.M)
assert m, 'public cal_cldfra2 missing'
(out/'cal_cldfra2.F90').write_text(m.group(0)+'\n')
PY
cat > "$out/state_stub.F90" <<'F90'
module module_state_description
integer, parameter :: KFCUPSCHEME=10, KFETASCHEME=1
end module
F90
cd "$out"
gfortran -O0 -ffree-form -ffree-line-length-none -fno-fast-math -ffp-contract=off state_stub.F90 cal_cldfra2.F90 "$here/run_cldfra2.F90" -o run_cldfra2
./run_cldfra2
sha256sum "$root/phys/module_radiation_driver.F" cal_cldfra2.F90 "$here/run_cldfra2.F90" columns.bin > sha256sums.txt
git -C "$root" rev-parse HEAD > wrf-commit.txt
gfortran --version | head -1 > compiler.txt
