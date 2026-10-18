#!/usr/bin/env bash
# WRF v4.6.1 Noah mosaic column oracle: the v4.7.1 harness, unchanged, built
# from a pristine v4.6.1 tree (tag v4.6.1, d66e442fccc04111067e29274c9f9eaccc3cef28)
# under this directory's source pins.
# Usage: build.sh WRF461_SOURCE_ROOT BUILD_DIR
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NOAH_MOSAIC_PIN_DIR="${here}" NOAH_MOSAIC_DRIVER_LINES=3003,3015 \
    bash "${here}/../noah_mosaic_wrf471_oracle/build.sh" "$@"
