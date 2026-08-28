#!/bin/sh
# Remaining GPU legs for the stale-guard lane, sequential on the one card.
# Each leg drops a marker when it finishes so nothing has to be polled.
set -u
cd "$(dirname "$0")/../.." || exit 1
M=runs/f4-downscale/markers
mkdir -p "$M"
GO="${WPS_GEOG:-$HOME/WPS_GEOG}"

run() {  # run NAME COMMAND...
    name=$1; shift
    "$@" > "runs/f4-downscale/${name}.log" 2>&1
    echo "$?" > "$M/${name}.done"
}

# 1. Finding 4, legacy arm: the child the RETIRED criterion admits (324x324).
#    Tests whether the retired over-size is a measured breakage or only a
#    model disagreement -- positive evidence either way.
run legacy-retired-child python -c "
from gpuwm.cli import main
import sys
sys.argv = ['gpuwm','downscale','runs/f4-downscale/parent-legacy-run/run/wrfout',
 '--parent-namelist','runs/f4-downscale/parent-legacy-namelist.input',
 '--child-config','runs/f4-downscale/child-legacy-retired.toml',
 '--ratio','3','--i-parent-start','93','--j-parent-start','64',
 '--out','runs/f4-downscale/child-legacy-retired-run','--accept-parent-cadence']
main()"

# 2-3. POOL_SLACK_FRACTION reduced re-fit: the calibration's own two legacy
#      sizes, post-#310 build, same card.
for dims in 110x88 60x48; do
    run "slack-legacy-${dims}" python -c "
from gpuwm.cli import main
import sys
sys.argv = ['gpuwm','go','runs/f4-downscale/slack-legacy-${dims}.toml',
 '--data-dir','runs/f4-downscale/data-slack-${dims}',
 '--outdir','runs/f4-downscale/slack-legacy-${dims}-run',
 '--run-stamp','off','--products','none','--geog-root','${GO}']
main()"
done

# 4-5. TROPICAL_ROOT_TIME_STEP_S: the motivating case, both clocks, 6 h,
#      under the corrected v1.1 co-located |w|/dz monitor.
for arm in 30s 60s; do
    run "tropical-${arm}" python -c "
from gpuwm.cli import main
import sys
sys.argv = ['gpuwm','go','runs/f4-downscale/tropical-${arm}.toml',
 '--data-dir','runs/f4-downscale/data-tropical',
 '--outdir','runs/f4-downscale/tropical-${arm}-run',
 '--run-stamp','off','--products','none','--geog-root','${GO}']
main()"
done

echo done > "$M/ALL.done"
