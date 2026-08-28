#!/bin/sh
# The two POOL_SLACK re-fit legs, re-run after their companion WPS
# namelists were supplied. Waits for the tropical legs to release the
# card (marker-driven, no timers).
set -u
cd "$(dirname "$0")/../.." || exit 1
M=runs/f4-downscale/markers
GO="${WPS_GEOG:-$HOME/WPS_GEOG}"

while [ ! -f "$M/ALL.done" ]; do sleep 20; done

for dims in 110x88 60x48; do
    rm -rf "runs/f4-downscale/slack-legacy-${dims}-run"
    python -c "
from gpuwm.cli import main
import sys
sys.argv = ['gpuwm','go','runs/f4-downscale/slack-legacy-${dims}.toml',
 '--data-dir','runs/f4-downscale/data-slack-${dims}',
 '--outdir','runs/f4-downscale/slack-legacy-${dims}-run',
 '--run-stamp','off','--products','none','--geog-root','${GO}']
main()" > "runs/f4-downscale/slack-legacy-${dims}.log" 2>&1
    echo "$?" > "$M/slack2-${dims}.done"
done
echo done > "$M/SLACK.done"
