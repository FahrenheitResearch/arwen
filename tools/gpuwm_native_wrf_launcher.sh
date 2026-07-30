#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime=${GPUWM_INSTALL_ROOT:-"$root/runtime"}
python_record="$runtime/.gpuwm-python"

[[ -d "$runtime/gpuwm" && -f "$python_record" ]] || {
    echo "gpuwm runtime is not installed; run $root/install.sh first" >&2
    exit 66
}
installed_python=$(<"$python_record")
python=${GPUWM_PYTHON:-$installed_python}
python_path=$(command -v "$python") || {
    echo "installed GPUWM Python is unavailable: $python" >&2
    exit 69
}
python_path=$(cd "$(dirname "$python_path")" && pwd -P)/$(basename "$python_path")
if [[ "$python_path" != "$installed_python" ]]; then
    echo "GPUWM_PYTHON differs from the interpreter used at install" >&2
    exit 65
fi
# Apply to the verification interpreter and every Python descendant launched
# by source_cli (preparation, benchmark, and worker jobs).  A command-line -B
# flag would protect only the immediate interpreter.
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "$root"
sha256sum -c --status SHA256SUMS
runtime_check=(--bridge-dir "$root/libexec/bridges")
skip_gpu=0
expect_backend=0
for argument in "$@"; do
    if (( expect_backend )); then
        case "$argument" in
            cpu|auto) skip_gpu=1 ;;
        esac
        expect_backend=0
        continue
    fi
    case "$argument" in
        -h|--help|--version|--author-only|--dry-run|--list-sources|--show-source|--show-source=*|--show-support-matrix|--namelist-support-report)
            skip_gpu=1
            ;;
        --preprocess-backend) expect_backend=1 ;;
        --preprocess-backend=cpu|--preprocess-backend=auto) skip_gpu=1 ;;
    esac
done
if (( skip_gpu )); then
    runtime_check+=(--skip-gpu)
fi
PYTHONPATH="$runtime" "$python_path" -P -m gpuwm.native_wrf_distribution \
    "${runtime_check[@]}" >/dev/null

export GPUWM_PYTHON="$python_path"
export GPUWM_HRRR_DECODER="$root/libexec/bridges/hrrr_grib2_bridge"
export GPUWM_GRIB1_BRIDGE="$root/libexec/bridges/grib1_bridge"
export GPUWM_GRIB2_INVENTORY="$root/libexec/bridges/grib2_inventory"
export GPUWM_GRIB2_DUMP="$root/libexec/bridges/grib2_dump"
export GPUWM_GFS_GRIB2_BRIDGE="$root/libexec/bridges/gfs_grib2_bridge"
export GPUWM_CPU_PREPROCESS_BRIDGE="$root/libexec/bridges/libgpuwm_preprocess_cpu.so"
export GPUWM_NATIVE_DISTRIBUTION_MANIFEST="$root/manifest.json"
export PYTHONPATH="$runtime"
exec "$python_path" -P -m gpuwm.source_cli "$@"
