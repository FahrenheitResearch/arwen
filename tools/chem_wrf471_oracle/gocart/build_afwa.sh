#!/usr/bin/env bash
# The common harness compiles all pinned modules and runs all installed drivers.
set -euo pipefail
exec bash "$(dirname "$0")/build.sh" "$@"
