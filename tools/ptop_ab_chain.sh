#!/bin/bash
# ptop A/B chain: one detached, self-driving run of the whole comparison.
#
# Two arms of one real HRRR-forced convective case, identical except
# p_top_requested (control 10000 Pa = today's shipped default, treatment
# 5000 Pa = the WRF Registry default), scored against MRMS composite
# reflectivity through tools/obs_battery_score.py.  Legs, in order:
#
#   mrms      fetch + decode MRMS composite frames for every forecast lead
#   hrrr      fetch the forcing cycle (wrfnat + wrfprs, hours 0..N)
#   prep_*    gpuwm prep, one per arm (the arms differ only in namelist p_top)
#   sim_*     gpuwm sim, one per arm, with a VRAM sampler and /usr/bin/time -v
#   render_*  gpuwm render on every history frame (the Rust render path)
#   score_*   obs battery FSS vs MRMS, one score file per arm
#   probes    tools/ptop_ab_probes.py: positive evidence both arms ran what
#             they claim (P_TOP, column depth, sponge base) plus reductions
#
# Progress markers land in $ROOT/markers/<leg>.done; the terminal marker is
# $ROOT/markers/CHAIN.done or $ROOT/markers/CHAIN.failed (leg name + rc
# inside).  No leg is retried: a failure stops the chain where it stands so
# the logs say what happened.
#
# Environment (all have defaults): PTOP_ROOT scratch root, PTOP_REPO the
# gpuwm checkout whose venv runs everything, RW_MRMS the MRMS front-door
# binary, PTOP_GEOG the WPS_GEOG root.
set -u
export PATH="$HOME/.cargo/bin:$PATH"

ROOT=${PTOP_ROOT:-$HOME/ptop-ab-scratch}
REPO=${PTOP_REPO:-$HOME/gpuwm-ptop}
GPUWM=$REPO/.venv/bin/gpuwm
PY=$REPO/.venv/bin/python
RW_MRMS=${RW_MRMS:-$HOME/.gpuwm/bridges/rw_mrms}
GEOG=${PTOP_GEOG:-$HOME/.local/share/gpuwm/WPS_GEOG}

# ---- the case ------------------------------------------------------------
CYCLE_ISO=2026-08-24T18            # HRRR cycle (UTC)
CYCLE_WRF=2026-08-24_18:00:00
HOURS=18                           # forecast window, hours 0..HOURS
RUN_SECONDS=64800
HISTORY_SECONDS=3600
CASE_ID=ptop-ab-plains-20260824
INIT_TIME=2026-08-24T18:00:00
BBOX=-106.0,41.1,-96.0,47.3        # W,S,E,N MRMS decode box around the domain
PROFILE=thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1
CFG=$REPO/configs/ptop_ab_20260824_18z
ZDAMP=5000.0
BOUNDARY_CELLS=5                   # spec_zone 1 + relax_zone 4: the rows scoring excludes

# The registration binds scores to a 40-hex evaluating commit; resolve it
# from the checkout unless the launcher pinned one.
PTOP_COMMIT=${PTOP_COMMIT:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unrecorded)}

M=$ROOT/markers
mkdir -p "$M" "$ROOT/logs" "$ROOT/obs/mrms/packs" "$ROOT/data" \
         "$ROOT/prep" "$ROOT/run" "$ROOT/png" "$ROOT/scores" "$ROOT/probes"

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$ROOT/logs/chain.log"; }
fail() { log "FAILED leg=$1 rc=$2"; printf 'leg=%s rc=%s\n' "$1" "$2" > "$M/CHAIN.failed"; exit 1; }

run_timed() {
    if [ -x /usr/bin/time ]; then /usr/bin/time -v "$@"; else "$@"; fi
}

run_leg() {
    local name=$1; shift
    if [ -f "$M/$name.done" ]; then log "skip $name (marker present)"; return 0; fi
    log "start $name"
    "$@" >> "$ROOT/logs/$name.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then fail "$name" $rc; fi
    log "done $name"
    touch "$M/$name.done"
}

# ---- leg: mrms -----------------------------------------------------------
mrms_leg() {
    local base lead t0 t1 record obj pack idx=0
    base=$(date -u -d "${INIT_TIME}Z" +%s) || return 1
    for lead in $(seq 1 "$HOURS"); do
        t0=$(date -u -d "@$((base + lead * 3600 - 120))" +%Y%m%dT%H%M%S)
        t1=$(date -u -d "@$((base + lead * 3600 + 120))" +%Y%m%dT%H%M%S)
        record=$("$RW_MRMS" fetch --start "$t0" --end "$t1" --limit 1 \
            --cache "$ROOT/obs/mrms/.cache") || return 1
        echo "$record"
        obj=$(echo "$record" | "$PY" -c \
            'import json, sys; print(json.load(sys.stdin)["files"][0]["path"])') \
            || { echo "no MRMS object for lead $lead"; return 1; }
        pack=$(printf '%s/obs/mrms/packs/mrms_h%02d.obspack' "$ROOT" "$lead")
        "$RW_MRMS" decode --file "$obj" --bbox "$BBOX" --out "$pack" || return 1
        if [ $idx -eq 0 ]; then
            "$RW_MRMS" grid --file "$obj" --bbox "$BBOX" \
                --out "$ROOT/obs/mrms/packs/geometry.obspack" || return 1
        fi
        idx=$((idx + 1))
    done
    "$RW_MRMS" verify --pack "$ROOT/obs/mrms/packs/mrms_h02.obspack" || return 1
}

# ---- leg: hrrr -----------------------------------------------------------
hrrr_leg() {
    "$GPUWM" fetch --source hrrr --cycle "$CYCLE_ISO" --hours "$HOURS" \
        --out "$ROOT/data/hrrr" || return 1
    if [ ! -f "$ROOT/data/hrrr/SHA256SUMS" ]; then
        (cd "$ROOT/data/hrrr" && sha256sum ./*.grib2 > SHA256SUMS) || return 1
    fi
}

# ---- leg: prep -----------------------------------------------------------
prep_leg() {
    local arm=$1
    local manifest_sha
    manifest_sha=$(sha256sum "$ROOT/data/hrrr/SHA256SUMS" | cut -d' ' -f1)
    run_timed "$GPUWM" prep --source hrrr \
        --source-root "$ROOT/data/hrrr" \
        --source-sha256s "$ROOT/data/hrrr/SHA256SUMS" \
        --source-sha256s-sha256 "$manifest_sha" \
        --domain-spec "$CFG.d01-target.json" \
        --namelist-input "${CFG}_${arm}.namelist.input" \
        --wps-namelist "$CFG.namelist.wps" \
        --geog-root "$GEOG" \
        --physics-profile "$PROFILE" \
        --valid-time "$CYCLE_WRF" \
        --run-seconds "$RUN_SECONDS" \
        --history-interval-seconds "$HISTORY_SECONDS" \
        --output-root "$ROOT/prep/$arm"
}

# ---- leg: sim ------------------------------------------------------------
sim_leg() {
    local arm=$1 rc smipid
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits \
        -lms 500 > "$ROOT/logs/vram_$arm.csv" 2>/dev/null &
    smipid=$!
    local wps="$ROOT/prep/$arm/namelist.wps"
    [ -f "$wps" ] || wps="$CFG.namelist.wps"
    run_timed "$GPUWM" sim "$ROOT/prep/$arm" \
        --experiment-config "$ROOT/prep/$arm/experiment.toml" \
        --wps-namelist "$wps" \
        --outdir "$ROOT/run/$arm"
    rc=$?
    kill "$smipid" 2>/dev/null
    return $rc
}

# ---- leg: render ---------------------------------------------------------
# The run folder keeps NetCDF frames under wrfout/ and same-named .json
# readiness markers under ready/; only the former are renderable.
frames_of() {
    find "$ROOT/run/$1" -path '*/wrfout/wrfout_d01_*' -not -name '*.json' \
        -type f | sort
}

render_leg() {
    local arm=$1 frame
    while IFS= read -r frame; do
        "$GPUWM" render "$frame" --out "$ROOT/png/$arm" || return 1
    done < <(frames_of "$arm")
}

# ---- leg: score ----------------------------------------------------------
run_dir_for() {
    local arm=$1 cand
    for cand in $(ls -d "$ROOT/run/$arm"/run-* 2>/dev/null) \
                $(frames_of "$arm" | head -1 | xargs -r dirname); do
        if "$PY" "$REPO/tools/obs_battery_score.py" --read-only \
                --run-directory "$cand" > /dev/null 2>&1; then
            echo "$cand"; return 0
        fi
    done
    return 1
}

score_leg() {
    local arm=$1 rundir
    rundir=$(run_dir_for "$arm") || { echo "no readable run directory for $arm"; return 1; }
    echo "scoring run directory: $rundir"
    "$PY" "$REPO/tools/obs_battery_score.py" \
        --run-directory "$rundir" \
        --case-id "$CASE_ID" \
        --arm-id "$arm" \
        --init-time "$INIT_TIME" \
        --boundary-width-cells "$BOUNDARY_CELLS" \
        --reflectivity-packs "$ROOT/obs/mrms" \
        --evaluator-commit "${PTOP_COMMIT:-unrecorded}" \
        --registration-out "$ROOT/scores/registration_$arm.json" \
        --score-out "$ROOT/scores/score_$arm.json"
}

# ---- leg: probes ---------------------------------------------------------
probes_leg() {
    local ctl trt
    ctl=$(frames_of control | head -1 | xargs -r dirname)
    trt=$(frames_of treatment | head -1 | xargs -r dirname)
    "$PY" "$REPO/tools/ptop_ab_probes.py" \
        --arm "control=$ctl" --arm "treatment=$trt" \
        --zdamp "$ZDAMP" --out "$ROOT/probes"
}

# ---- the chain -----------------------------------------------------------
rm -f "$M/CHAIN.done" "$M/CHAIN.failed"
log "chain start: case $CASE_ID cycle $CYCLE_ISO hours $HOURS root $ROOT repo $REPO commit ${PTOP_COMMIT:-unrecorded}"

run_leg mrms       mrms_leg
run_leg hrrr       hrrr_leg
run_leg prep_ctl   prep_leg control
run_leg prep_trt   prep_leg treatment
run_leg sim_ctl    sim_leg control
run_leg sim_trt    sim_leg treatment
run_leg render_ctl render_leg control
run_leg render_trt render_leg treatment
run_leg score_ctl  score_leg control
run_leg score_trt  score_leg treatment
run_leg probes     probes_leg

(cd "$ROOT" && find scores probes -type f -name '*.json' -print0 | sort -z | \
    xargs -0 -r sha256sum > SHA256SUMS.receipts)
log "chain complete"
touch "$M/CHAIN.done"
