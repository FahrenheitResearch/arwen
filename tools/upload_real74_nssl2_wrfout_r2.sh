#!/usr/bin/env bash
# Stream only closed GPUWM NSSL validation wrfouts to their dedicated R2
# prefix, then publish the run/evidence manifests after all 64 files arrive.
set -uo pipefail

usage() {
    printf 'usage: %s RUN_DIR [STATE_DIR]\n' "$0" >&2
    exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
run_dir=$1
state_dir=${2:-"$run_dir/r2-upload-state"}
candidate_dir="$state_dir/candidates"
uploaded_dir="$state_dir/uploaded"
log_file="$state_dir/uploader.log"
receipt="$state_dir/remote-verification-receipt.tsv"
expected_sha256="$state_dir/completion-sha256.tsv"
destination='r2:wrfconus/gpuwm-validation/20260722/nssl2-500m-1974-12h/gpuwm-production/wrfout'
manifest_destination='r2:wrfconus/gpuwm-validation/20260722/nssl2-500m-1974-12h/gpuwm-production/manifests'
expected_files=64
poll_seconds=${GPUWM_R2_POLL_SECONDS:-30}
minimum_age_seconds=${GPUWM_R2_MINIMUM_AGE_SECONDS:-90}
parallel_files=${GPUWM_R2_PARALLEL_FILES:-4}

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ \
        || ! "$minimum_age_seconds" =~ ^[0-9]+$ \
        || ! "$parallel_files" =~ ^[1-9][0-9]*$ ]]; then
    printf 'uploader overrides must be integer seconds/counts (poll and parallel > 0)\n' >&2
    exit 2
fi

mkdir -p "$candidate_dir" "$uploaded_dir"
if ! command -v flock >/dev/null 2>&1; then
    printf 'flock is required to serialize uploader ownership\n' >&2
    exit 2
fi
exec 9>"$state_dir/uploader.lock"
if ! flock -n 9; then
    printf 'another uploader owns %s\n' "$state_dir/uploader.lock" >&2
    exit 2
fi

log() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" \
        | tee -a "$log_file"
}

rclone_bin=${RCLONE_BIN:-}
if [[ -z "$rclone_bin" ]]; then
    rclone_bin=$(command -v rclone 2>/dev/null || true)
fi
if [[ -z "$rclone_bin" || ! -x "$rclone_bin" ]]; then
    log 'FATAL rclone is not executable; set RCLONE_BIN explicitly'
    exit 2
fi
python_bin=${PYTHON_BIN:-}
if [[ -z "$python_bin" ]]; then
    python_bin=$(command -v python3 2>/dev/null || true)
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    log 'FATAL python3 is required to read completion.json checksums'
    exit 2
fi
if [[ ! -d "$run_dir" ]]; then
    log "FATAL run directory does not exist: $run_dir"
    exit 2
fi

rclone_copyto() {
    local source=$1
    local target=$2
    "$rclone_bin" copyto "$source" "$target" \
        --ignore-times --transfers 1 --checkers 16 \
        --s3-upload-concurrency 8 --s3-chunk-size 64M \
        --retries 20 --low-level-retries 40 \
        --contimeout 30s --timeout 2h
}

remote_size() {
    "$rclone_bin" lsl "$1" 2>/dev/null | awk 'NR == 1 {print $1}'
}

upload_verified() {
    local source=$1
    local target=$2
    local size
    local observed
    size=$(stat -c '%s' "$source") || return 1
    if ! rclone_copyto "$source" "$target"; then
        return 1
    fi
    observed=$(remote_size "$target")
    [[ "$observed" == "$size" ]]
}

log "START source=$run_dir destination=$destination expected=$expected_files parallel=$parallel_files"

while true; do
    now_epoch=$(date +%s)
    shopt -s nullglob
    files=("$run_dir"/wrfout_d0*)
    shopt -u nullglob

    upload_pids=()
    for file in "${files[@]}"; do
        [[ -f "$file" ]] || continue
        name=${file##*/}
        [[ "$name" =~ ^wrfout_d0[1-4]_1974-04-(03|04)_ ]] || continue
        size=$(stat -c '%s' "$file") || continue
        mtime=$(stat -c '%Y' "$file") || continue
        age=$((now_epoch - mtime))
        signature="$size:$mtime"
        candidate="$candidate_dir/$name"
        marker="$uploaded_dir/$name"

        if [[ -f "$marker" ]] && [[ "$(<"$marker")" == "$size" ]]; then
            continue
        fi
        if (( age < minimum_age_seconds )); then
            printf '%s\n' "$signature" > "$candidate"
            continue
        fi
        if command -v lsof >/dev/null 2>&1 && lsof "$file" >/dev/null 2>&1; then
            printf '%s\n' "$signature" > "$candidate"
            continue
        fi
        if [[ ! -f "$candidate" ]] || [[ "$(<"$candidate")" != "$signature" ]]; then
            printf '%s\n' "$signature" > "$candidate"
            continue
        fi

        (
            log "UPLOAD_BEGIN name=$name bytes=$size"
            if upload_verified "$file" "$destination/$name"; then
                printf '%s\n' "$size" > "$marker"
                log "UPLOAD_OK name=$name bytes=$size"
            else
                log "UPLOAD_RETRY name=$name bytes=$size"
            fi
        ) &
        upload_pids+=("$!")
        if (( ${#upload_pids[@]} == parallel_files )); then
            for upload_pid in "${upload_pids[@]}"; do
                wait "$upload_pid" || true
            done
            upload_pids=()
        fi
    done
    for upload_pid in "${upload_pids[@]}"; do
        wait "$upload_pid" || true
    done

    uploaded_count=$(find "$uploaded_dir" -maxdepth 1 -type f \
        -name 'wrfout_d0*' | wc -l)
    if [[ -f "$run_dir/failure-capsule.json" ]]; then
        log 'FATAL GPUWM controller published failure-capsule.json'
        exit 3
    fi
    if [[ -f "$run_dir/completion.json" ]] \
            && (( uploaded_count == expected_files )); then
        break
    fi
    sleep "$poll_seconds"
done

printf 'schema\tgpuwm-nssl2-r2-size-receipt-v1\n' > "$receipt"
printf 'destination\t%s\n' "$destination" >> "$receipt"
printf 'expected_files\t%s\n' "$expected_files" >> "$receipt"
while IFS= read -r marker; do
    name=${marker##*/}
    local_size=$(<"$marker")
    observed=$(remote_size "$destination/$name")
    if [[ "$observed" != "$local_size" ]]; then
        log "FATAL remote size mismatch name=$name local=$local_size remote=${observed:-missing}"
        exit 4
    fi
    printf 'wrfout\t%s\t%s\n' "$name" "$local_size" >> "$receipt"
done < <(find "$uploaded_dir" -maxdepth 1 -type f -name 'wrfout_d0*' | sort)

remote_count=$("$rclone_bin" lsf "$destination" --files-only 2>/dev/null \
    | grep -Ec '^wrfout_d0[1-4]_1974-04-(03|04)_')
if (( remote_count != expected_files )); then
    log "FATAL remote wrfout count is $remote_count, expected $expected_files"
    exit 4
fi

"$python_bin" - "$run_dir/completion.json" "$expected_sha256" <<'PY'
import json
from pathlib import Path
import sys

completion_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
with completion_path.open("r", encoding="utf-8") as stream:
    completion = json.load(stream)
outputs = completion.get("outputs")
if not isinstance(outputs, dict):
    raise SystemExit("completion.json has no outputs object")
rows = []
for domain in sorted(outputs):
    records = outputs[domain]
    if not isinstance(records, list):
        raise SystemExit(f"completion outputs[{domain!r}] is not a list")
    for record in records:
        if not isinstance(record, dict):
            raise SystemExit(f"completion {domain} record is not an object")
        name = Path(str(record.get("path", ""))).name
        digest = str(record.get("sha256", "")).lower()
        if not name.startswith("wrfout_d0"):
            raise SystemExit(f"invalid completion wrfout name {name!r}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SystemExit(f"invalid completion SHA-256 for {name!r}")
        rows.append((name, digest))
if len(rows) != 64 or len({name for name, _ in rows}) != 64:
    raise SystemExit(f"completion SHA-256 inventory is not 64 unique files: {len(rows)}")
with output_path.open("w", encoding="utf-8", newline="\n") as stream:
    for name, digest in sorted(rows):
        stream.write(f"{digest}\t{name}\n")
PY

verified_sha_count=0
while IFS=$'\t' read -r expected_sha name; do
    source_file="$run_dir/$name"
    if [[ ! -f "$source_file" ]]; then
        log "FATAL completion checksum names absent local file: $name"
        exit 4
    fi
    matched=0
    for attempt in 1 2 3; do
        actual_sha=$("$rclone_bin" cat "$destination/$name" \
            --retries 10 --low-level-retries 20 --contimeout 30s --timeout 2h \
            | sha256sum | awk 'NR == 1 {print $1}') || actual_sha=''
        if [[ "$actual_sha" == "$expected_sha" ]]; then
            matched=1
            break
        fi
        log "REMOTE_SHA_RETRY name=$name attempt=$attempt expected=$expected_sha actual=${actual_sha:-unavailable}"
        upload_verified "$source_file" "$destination/$name" || true
    done
    if (( matched != 1 )); then
        log "FATAL remote SHA-256 mismatch after retries: $name"
        exit 4
    fi
    printf 'sha256\t%s\t%s\n' "$name" "$expected_sha" >> "$receipt"
    verified_sha_count=$((verified_sha_count + 1))
    log "REMOTE_SHA_OK name=$name sha256=$expected_sha"
done < "$expected_sha256"
if (( verified_sha_count != expected_files )); then
    log "FATAL verified SHA-256 count is $verified_sha_count, expected $expected_files"
    exit 4
fi

manifest_files=(
    "$run_dir/completion.json"
    "$run_dir/metadata/launch-manifest.json"
    "$run_dir/metadata/preflight-receipt.json"
    "$run_dir/metadata/real74_nssl2_500m.effective.toml"
    "$receipt"
)
for file in "${manifest_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        log "FATAL required manifest is absent: $file"
        exit 5
    fi
    name=${file##*/}
    if ! upload_verified "$file" "$manifest_destination/$name"; then
        log "FATAL manifest upload/verification failed: $name"
        exit 5
    fi
done

log "COMPLETE uploaded=$uploaded_count remote=$remote_count sha256=$verified_sha_count manifests=${#manifest_files[@]}"
