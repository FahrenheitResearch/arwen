#!/usr/bin/env bash
# One-command downloaded-HRRR -> stock-CPU-WRF input preparation.
set -Eeuo pipefail

if (( $# < 8 || $# > 12 )); then
    cat >&2 <<'EOF'
usage: prepare_hrrr_cpu_wrf.sh \
  SOURCE_ROOT SOURCE_SHA256SUMS SOURCE_MANIFEST_SHA256 \
  STATIC_CACHE STATIC_RECEIPT NAMELIST_INPUT VALID_TIME OUTPUT_ROOT \
  [RUN_SECONDS=43200] [PIPELINE_WORKERS=8] [PREPARE_WORKERS] [DOMAIN_SPEC]

VALID_TIME uses WRF UTC form YYYY-MM-DD_HH:MM:SS.

Consumes downloaded HRRR f00..f12 atmosphere/soil GRIB2 files and emits
OUTPUT_ROOT/wrf-native-input/{wrfinput_d01,wrfbdy_d01,manifest.json}.
Neither WPS nor real.exe is invoked.
EOF
    exit 64
fi

source_root=$1
source_manifest=$2
source_manifest_sha=$3
static_cache=$4
static_receipt=$5
namelist_input=$6
valid_time=$7
output_root=$8
run_seconds=${9:-43200}
pipeline_workers=${10:-8}
prepare_workers=${11:-}
domain_spec=${12:-}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${GPUWM_PYTHON:-python3}

if [[ ! "$valid_time" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_([01][0-9]|2[0-3]):00:00$ ]]; then
    echo "VALID_TIME must be an exact hourly cycle in YYYY-MM-DD_HH:00:00 form" >&2
    exit 64
fi
cycle_hour=${BASH_REMATCH[1]}

[[ ! -e "$output_root" ]] || {
    echo "refusing existing OUTPUT_ROOT: $output_root" >&2
    exit 73
}
mkdir -p "$output_root"

started_ns=$(date +%s%N)
geometry_receipt="$output_root/native-geometry-receipt.json"
geometry_command=(
    "$python"
    "$repo/tools/write_hrrr_native_geometry_receipt.py"
    --static-cache "$static_cache"
    --hrrr-static-receipt "$static_receipt"
    --output "$geometry_receipt"
)
if [[ -n "$domain_spec" ]]; then
    geometry_command+=(--domain-spec "$domain_spec")
fi
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" "${geometry_command[@]}"
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
)
if [[ -n "$prepare_workers" ]]; then
    prepare_command+=("$prepare_workers")
fi
if [[ -n "$domain_spec" ]]; then
    if [[ -z "$prepare_workers" ]]; then
        prepare_command+=("")
    fi
    prepare_command+=("$domain_spec")
fi
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

prepare_ms=$(((prepared_ns - started_ns) / 1000000))
export_ms=$(((finished_ns - prepared_ns) / 1000000))
total_ms=$(((finished_ns - started_ns) / 1000000))
printf 'PASS wrf_input=%s prepare_ms=%d export_ms=%d total_ms=%d\n' \
    "$output_root/wrf-native-input" "$prepare_ms" "$export_ms" "$total_ms"
