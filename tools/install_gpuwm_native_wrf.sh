#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${GPUWM_PYTHON:-python3}
target=${GPUWM_INSTALL_ROOT:-"$root/runtime"}
skip_gpu=0
case "${1:-}" in
    "") ;;
    --skip-gpu) skip_gpu=1; shift ;;
    *) echo "usage: $0 [--skip-gpu]" >&2; exit 64 ;;
esac
(( $# == 0 )) || { echo "usage: $0 [--skip-gpu]" >&2; exit 64; }

[[ -f "$root/SHA256SUMS" ]] || {
    echo "distribution SHA256SUMS is missing: $root/SHA256SUMS" >&2
    exit 66
}
[[ ! -e "$target" ]] || {
    echo "refusing existing GPUWM_INSTALL_ROOT: $target" >&2
    exit 73
}
python_path=$(command -v "$python") || {
    echo "GPUWM_PYTHON is not executable: $python" >&2
    exit 69
}
# Preserve a virtual-environment launcher symlink: resolving the final Python
# binary would silently discard that environment's site-packages.
python_path=$(cd "$(dirname "$python_path")" && pwd -P)/$(basename "$python_path")
target_parent=$(dirname "$target")
[[ -d "$target_parent" ]] || {
    echo "GPUWM_INSTALL_ROOT parent does not exist: $target_parent" >&2
    exit 66
}

cd "$root"
sha256sum -c SHA256SUMS
mapfile -t wheels < <(find "$root/wheel" -maxdepth 1 -type f \
    -name 'rw_wps-*.whl' -print | sort)
(( ${#wheels[@]} == 1 )) || {
    echo "distribution must contain exactly one rw-wps wheel" >&2
    exit 65
}

partial=$(mktemp -d "$target_parent/.gpuwm-native-runtime.partial.XXXXXX")
cleanup() {
    if [[ -n "${partial:-}" && -d "$partial" ]]; then
        rm -rf -- "$partial"
    fi
}
trap cleanup EXIT
"$python_path" -m pip install \
    --disable-pip-version-check --no-index --no-deps \
    --target "$partial" "${wheels[0]}"
printf '%s\n' "$python_path" > "$partial/.gpuwm-python"

runtime_check=(
    "$python_path" -P -m gpuwm.native_wrf_distribution
    --bridge-dir "$root/libexec/bridges"
    --receipt "$partial/native-wrf-runtime-receipt.json"
)
if (( skip_gpu )); then
    runtime_check+=(--skip-gpu)
fi
PYTHONPATH="$partial" "${runtime_check[@]}"
mv "$partial" "$target"
partial=
printf 'PASS rw_wps_install=%s python=%s\n' "$target" "$python_path"
