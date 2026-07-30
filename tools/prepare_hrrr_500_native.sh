#!/usr/bin/env bash
# One-command downloaded-HRRR to launch-ready native IC/LBC preparation.
set -Eeuo pipefail

if (( $# < 7 || $# > 11 )); then
    cat >&2 <<'EOF'
usage: prepare_hrrr_500_native.sh \
  SOURCE_ROOT SOURCE_SHA256SUMS SOURCE_MANIFEST_SHA256 \
  STATIC_CACHE STATIC_RECEIPT NAMELIST_INPUT OUTPUT_ROOT \
  [RUN_SECONDS=43200] [PIPELINE_WORKERS=auto] [PREPARE_WORKERS] [DOMAIN_SPEC]

Consumes the canonical downloaded HRRR f00 through the final forcing hour
needed by RUN_SECONDS (at least f01 and at most f12),
then produces OUTPUT_ROOT/prepared-cache plus a hash-bound preparation report.
PREPARE_WORKERS is optional; omitting it uses the benchmarked safe default.
DOMAIN_SPEC is an optional strict gpuwm-hrrr-target-domain-v1 JSON file.
EOF
    exit 64
fi

source_root=$1
source_manifest=$2
source_manifest_sha=$3
static_cache=$4
static_receipt=$5
namelist_input=$6
output_root=$7
run_seconds=${8:-43200}
pipeline_workers=${9:-auto}
prepare_workers=${10:-}
domain_spec=${11:-}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${GPUWM_PYTHON:-python3}
decoder=${GPUWM_HRRR_DECODER:-}
cycle_hour=${GPUWM_HRRR_CYCLE_HOUR:-00}
valid_time=${GPUWM_HRRR_VALID_TIME:-}
if [[ ! "$cycle_hour" =~ ^([01][0-9]|2[0-3])$ ]]; then
    echo "GPUWM_HRRR_CYCLE_HOUR must be two digits from 00 through 23" >&2
    exit 64
fi
if [[ ! "$valid_time" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_([01][0-9]|2[0-3]):00:00$ ]]; then
    echo "GPUWM_HRRR_VALID_TIME must be an exact hourly cycle in YYYY-MM-DD_HH:00:00 form" >&2
    exit 64
fi
if [[ "${BASH_REMATCH[1]}" != "$cycle_hour" ]]; then
    echo "GPUWM_HRRR_VALID_TIME hour differs from GPUWM_HRRR_CYCLE_HOUR" >&2
    exit 64
fi
if [[ -z "$decoder" ]]; then
    decoder="$repo/tools/grib1_bridge/target/release/hrrr_grib2_bridge"
    if [[ ! -x "$decoder" ]]; then
        cargo build --release \
            --manifest-path "$repo/tools/grib1_bridge/Cargo.toml" \
            --bin hrrr_grib2_bridge
    fi
fi

for path in "$source_root" "$source_manifest" "$static_cache" \
            "$static_receipt" "$namelist_input"; do
    [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 66; }
done
if [[ -n "$domain_spec" && ! -f "$domain_spec" ]]; then
    echo "missing target domain spec: $domain_spec" >&2
    exit 66
fi
[[ -x "$decoder" ]] || { echo "HRRR decoder is not executable: $decoder" >&2; exit 66; }
[[ ! -e "$output_root" ]] || {
    echo "refusing existing OUTPUT_ROOT: $output_root" >&2
    exit 73
}
mkdir -p "$output_root"

final_forcing_hour=$(((run_seconds + 3599) / 3600))
(( final_forcing_hour < 1 )) && final_forcing_hour=1
if (( final_forcing_hour > 12 )); then
    echo "RUN_SECONDS requires HRRR forcing beyond f12" >&2
    exit 64
fi
series=$(printf '%s/hrrr-f00-f%02d-series.tsv' "$output_root" "$final_forcing_hour")
for hour in $(seq -w 0 "$final_forcing_hour"); do
    printf '%d\t%s/hrrr.t%sz.wrfnatf%s.grib2\t%s/hrrr.t%sz.soilf%s.grib2\n' \
        "$((10#$hour))" "$source_root" "$cycle_hour" "$hour" \
        "$source_root" "$cycle_hour" "$hour"
done > "$series"

command=(
    "$python" "$repo/tools/hrrr_single_domain_benchmark.py"
    --bridge "$output_root/native-bridge"
    --valid-time "$valid_time"
    --pipeline-series "$series"
    --pipeline-decoder "$decoder"
    --pipeline-signals "$output_root/pipeline-signals"
    --pipeline-workers "$pipeline_workers"
    --source-root "$source_root"
    --source-manifest "$source_manifest"
    --source-manifest-sha256 "$source_manifest_sha"
    --static-cache "$static_cache"
    --static-receipt "$static_receipt"
    --namelist-input "$namelist_input"
    --prepared-cache "$output_root/prepared-cache"
    --prepare-only
    --run-seconds "$run_seconds"
    --outdir "$output_root/preparation-report"
)
if [[ -n "$prepare_workers" ]]; then
    command+=(--prepare-workers "$prepare_workers")
fi
if [[ -n "$domain_spec" ]]; then
    command+=(--domain-spec "$domain_spec")
fi

started=$(date +%s)
cd "$repo"
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" "${command[@]}"
finished=$(date +%s)
printf 'PASS launch_ready_cache=%s wall_seconds=%d\n' \
    "$output_root/prepared-cache" "$((finished - started))"
