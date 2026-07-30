#!/usr/bin/env bash
# Experimental arbitrary-Lambert HRRR -> stock-WRF preparation gate.
set -Eeuo pipefail

if (( $# < 8 || $# > 11 )); then
    cat >&2 <<'EOF'
usage: prepare_hrrr_domain_cpu_wrf.sh \
  SOURCE_ROOT SOURCE_SHA256SUMS SOURCE_MANIFEST_SHA256 GEOG_ROOT \
  DOMAIN_SPEC NAMELIST_INPUT VALID_TIME OUTPUT_ROOT \
  [RUN_SECONDS=43200] [PIPELINE_WORKERS=auto] [PREPARE_WORKERS]

Builds domain-specific native terrain/land-use/static fields, computes and
validates the exact HRRR source crop, prepares IC/LBC state, and directly
emits wrfinput_d01/wrfbdy_d01. WPS and real.exe are never invoked.

This entry point remains an acceptance lane until each target geometry passes
reopen validation and unchanged stock wrf.exe.
EOF
    exit 64
fi

source_root=$1
source_manifest=$2
source_manifest_sha=$3
geog_root=$4
domain_spec=$5
namelist_input=$6
valid_time=$7
output_root=$8
run_seconds=${9:-43200}
pipeline_workers=${10:-auto}
prepare_workers=${11:-}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${GPUWM_PYTHON:-python3}

if [[ ! "$valid_time" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_([01][0-9]|2[0-3]):00:00$ ]]; then
    echo "VALID_TIME must be an exact hourly cycle in YYYY-MM-DD_HH:00:00 form" >&2
    exit 64
fi
cycle_hour=${BASH_REMATCH[1]}

for path in "$source_root" "$source_manifest" "$geog_root" \
            "$domain_spec" "$namelist_input"; do
    [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 66; }
done
[[ ! -e "$output_root" ]] || {
    echo "refusing existing OUTPUT_ROOT: $output_root" >&2
    exit 73
}
mkdir -p "$output_root"

static_cache="$output_root/native-static.npz"
static_receipt="$output_root/native-static-receipt.json"
geometry_receipt="$output_root/native-geometry-receipt.json"
started_ns=$(date +%s%N)
PYTHONPATH="$repo" "$python" "$repo/tools/hrrr_build_native_static.py" \
    --geog-root "$geog_root" \
    --domain-spec "$domain_spec" \
    --output "$static_cache" \
    --receipt "$static_receipt"
PYTHONPATH="$repo" "$python" \
    "$repo/tools/write_hrrr_native_geometry_receipt.py" \
    --static-cache "$static_cache" \
    --hrrr-static-receipt "$static_receipt" \
    --domain-spec "$domain_spec" \
    --output "$geometry_receipt"
static_finished_ns=$(date +%s%N)

prepare_command=(
    bash
    "$repo/tools/prepare_hrrr_500_native.sh"
    "$source_root"
    "$source_manifest"
    "$source_manifest_sha"
    "$static_cache"
    "$static_receipt"
    "$namelist_input"
    "$output_root/native"
    "$run_seconds"
    "$pipeline_workers"
    "$prepare_workers"
    "$domain_spec"
)
GPUWM_HRRR_CYCLE_HOUR="$cycle_hour" \
GPUWM_HRRR_VALID_TIME="$valid_time" GPUWM_PYTHON="$python" \
    "${prepare_command[@]}"
prepared_ns=$(date +%s%N)

PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" "$python" -m gpuwm.wrf_direct \
    --prepared-cache "$output_root/native/prepared-cache" \
    --static-cache "$static_cache" \
    --geometry-receipt "$geometry_receipt" \
    --output "$output_root/wrf-native-input" \
    --valid-time "$valid_time"
finished_ns=$(date +%s%N)

static_ms=$(((static_finished_ns - started_ns) / 1000000))
prepare_ms=$(((prepared_ns - static_finished_ns) / 1000000))
export_ms=$(((finished_ns - prepared_ns) / 1000000))
total_ms=$(((finished_ns - started_ns) / 1000000))
printf 'PASS domain_spec=%s wrf_input=%s static_ms=%d prepare_ms=%d export_ms=%d total_ms=%d\n' \
    "$domain_spec" "$output_root/wrf-native-input" \
    "$static_ms" "$prepare_ms" "$export_ms" "$total_ms"
