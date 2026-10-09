#!/usr/bin/env bash
# Stock WRF with the three lost-carry assignments restricted to cold starts.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo 'usage: build_cycled.sh WRF_SOURCE_ROOT LANE_BUILD_DIR' >&2
    exit 2
fi
source_root=$(realpath "$1")
lane_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$lane_dir/referee/phys" "$lane_dir/fixtures"
cp "$source_root/phys/module_bl_mynn_common.F" "$lane_dir/referee/phys/"
cp "$source_root/phys/module_bl_mynn.F" "$lane_dir/referee/phys/"
python3 - "$lane_dir/referee/phys/module_bl_mynn.F" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = ('         cldfra_bl(its:ite,kts:kte)=0.\n'
       '         qc_bl(its:ite,kts:kte)=0.\n'
       '         qke(its:ite,kts:kte)=0.\n')
assert s.count(old) == 1, 'stock lost-carry block changed'
p.write_text(s.replace(old, '         if (.not.cycling) then\n' + old
                      + '         end if\n'))
PY
diff -u "$source_root/phys/module_bl_mynn.F" \
    "$lane_dir/referee/phys/module_bl_mynn.F" > "$lane_dir/referee.patch" || [[ $? == 1 ]]
bash "$script_dir/build.sh" "$lane_dir/referee" "$lane_dir/build"
for mixlength in 1 2; do
    for carry in empty below equal above high mixed; do
        csv="$lane_dir/fixtures/driver-carry-$carry-$mixlength.csv"
        "$lane_dir/build/run_driver_families" "$csv" "$mixlength" 24 "$carry"
        gzip -n -9 "$csv"
    done
done
# The 120-step cold referee is the unmodified source, independently built.
bash "$script_dir/build.sh" "$source_root" "$lane_dir/stock-build"
for mixlength in 1 2; do
    csv="$lane_dir/fixtures/driver-stock-120-$mixlength.csv"
    "$lane_dir/stock-build/run_driver_families" "$csv" "$mixlength" 120
    gzip -n -9 "$csv"
done
sha256sum "$source_root/phys/module_bl_mynn.F" \
    "$lane_dir/referee/phys/module_bl_mynn.F" \
    "$script_dir/run_driver_families.F90" "$lane_dir/fixtures/"*.gz \
    > "$lane_dir/provenance.txt"
gfortran --version | sed -n '1p' >> "$lane_dir/provenance.txt"
ldd --version | sed -n '1p' >> "$lane_dir/provenance.txt"
