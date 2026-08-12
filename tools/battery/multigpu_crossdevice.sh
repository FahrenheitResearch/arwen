#!/bin/bash
# The cross-device transport battery: every multi-GPU gate this lane owns,
# run with the sub-domains on SEPARATE PHYSICAL CARDS.
#
# WHY THIS FILE EXISTS
#   The evidence for cross-device reassembly was produced by a shell script
#   that lived only in a rented box's /workspace.  When the rental ends the
#   script ends with it, and the next operator reconstructs the invocations
#   from a report.  That is the failure tools/battery/stage1_files.txt and
#   tools/battery/tiles_gates.txt were both written to close, and it already
#   cost this lane once: the scratch copy's controls row carried
#   `--steps 4`, a run length at which one of the four negative controls
#   CANNOT fire, and it kept carrying it after the module learned to refuse
#   that number.  A battery nobody can re-run is a battery whose rows drift
#   from the gates they invoke.
#
#   These rows are NOT in tiles_gates.txt.  That file is the streamed-run
#   lane's list and says so; these need two or more physical cards and a
#   DEVICES assignment, which no stage-1 shard provides.
#
# THE CONTROLS ROW TAKES NO --steps, DELIBERATELY
#   tilestream.multigpu's `controls` subcommand carries its own default,
#   measured: the short-halo defect is bit-invisible against the monolithic
#   digest before step SHORT_HALO_VISIBLE_AT.  A number written here is a
#   copy of that measurement, and copies go stale.  The scratch battery
#   passed `--steps 4` and the control set reported three real results and
#   one artefact of the run length; after the module started refusing, the
#   same row failed for a second, different reason.  Passing nothing gets
#   the floor the module measured, and moves when it moves.
#
# LOG AND STATUS ARE WRITTEN TOGETHER
#   `run` writes the log, appends the status to the END OF THE LOG, and
#   writes the .rc file, in that order, from one invocation.  The scratch
#   battery wrote them as separate paths, and a hand re-run that redirected
#   to the same log path left a PASSING log beside a FAILING .rc from the
#   run before it: 3.5 minutes apart by mtime, and the report read the log.
#   A status the log does not carry is a status that can outlive its run.
#
# USAGE
#   tools/battery/multigpu_crossdevice.sh [OUTDIR]
#   Run it FROM THE REPOSITORY ROOT with a python that can import tilestream
#   and cupy.  OUTDIR defaults to ./battery-multigpu.  Needs 4 visible cards
#   for the 2x2 rows; with 2 they are skipped and SAID to be skipped.
set -u

OUT="${1:-$PWD/battery-multigpu}"
PY="${PYTHON:-python}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

NGPU=$("$PY" -c "import cupy;print(cupy.cuda.runtime.getDeviceCount())" \
       2>/dev/null || echo 0)
echo "visible cards: $NGPU" | tee "$OUT/RUNLOG"
if [ "$NGPU" -lt 2 ]; then
    echo "REFUSED: this battery is about CROSS-DEVICE transport and $NGPU " \
         "card(s) are visible.  A one-card run would pass every row without " \
         "crossing anything." | tee -a "$OUT/RUNLOG"
    exit 2
fi

PASS=0
FAIL=0
SKIP=0

run() {                       # run <name> <command...>
    local name="$1"; shift
    local log="$OUT/$name.log"
    rm -f "$log" "$OUT/$name.rc"          # no survivors from an earlier run
    echo "=== START $name  $(date -u +%FT%TZ)" | tee -a "$OUT/RUNLOG"
    ( "$@" ) > "$log" 2>&1
    local rc=$?
    # The verdict goes INTO the log first, so the two cannot be separated by
    # a later hand re-run that redirects only stdout.
    {
        echo
        echo "=== $name rc=$rc  $(date -u +%FT%TZ)"
        echo "=== command: $*"
    } >> "$log"
    echo "$rc" > "$OUT/$name.rc"
    if [ "$rc" -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
    echo "=== END   $name rc=$rc  $(date -u +%FT%TZ)" | tee -a "$OUT/RUNLOG"
}

skip() {                      # skip <name> <why...>
    local name="$1"; shift
    SKIP=$((SKIP + 1))
    echo "=== SKIP  $name: $*" | tee -a "$OUT/RUNLOG"
    echo "SKIPPED: $*" > "$OUT/$name.log"
}

# ---- 1. slice/transport acceptance: ranks spread over physical cards -------
run decomp_geography_4card \
    env PLANS=1x1,2x1,2x2 DEVICES=0,1,2,3 "$PY" -m tilestream.test_decomp_gate
run decomp_physics_4card \
    env PLANS=1x1,2x1,2x2 DEVICES=0,1,2,3 PHYSICS=1 \
    "$PY" -m tilestream.test_decomp_gate
# the one-card control, so the cross-device rows have something to equal
run decomp_geography_1card \
    env PLANS=1x1,2x1,2x2 "$PY" -m tilestream.test_decomp_gate
run decomp_physics_1card \
    env PLANS=1x1,2x1,2x2 PHYSICS=1 "$PY" -m tilestream.test_decomp_gate

# ---- 2. seam gate, both geometries, sub-domains on separate cards ----------
run seam_1x2_dev01  env GRID=1x2 DEVICES=0,1 "$PY" -m tilestream.test_seam_gate
run seam_2x1_dev01  env GRID=2x1 DEVICES=0,1 "$PY" -m tilestream.test_seam_gate
if [ "$NGPU" -ge 4 ]; then
    run seam_2x2_dev0123 env GRID=2x2 DEVICES=0,1,2,3 \
        "$PY" -m tilestream.test_seam_gate
else
    skip seam_2x2_dev0123 "needs 4 cards, $NGPU visible"
fi

# ---- 3. multigpu: geometry, the gate at all three transports, controls -----
for g in 1x2 2x1; do
    run "mg_geometry_$g" "$PY" -m tilestream.multigpu geometry --grid "$g" \
        --steps 3
done
run mg_geometry_2x2 "$PY" -m tilestream.multigpu geometry --grid 2x2 \
    --nx 256 --ny 256 --steps 3
for t in peer default host; do
    run "mg_gate_2gpu_$t" "$PY" -m tilestream.multigpu gate --ngpu 2 \
        --steps 4 --transport "$t"
done
run mg_gate_4gpu_peer "$PY" -m tilestream.multigpu gate --nx 256 --ny 256 \
    --grid 2x2 --steps 4 --transport peer
# no --steps: see the header.  The subcommand's default is the measured floor.
run mg_controls "$PY" -m tilestream.multigpu controls
for t in peer default host; do
    run "mg_exchange_$t" "$PY" -m tilestream.multigpu exchange --nx 512 \
        --ny 512 --ngpu 2 --transport "$t"
done

# ---- 4. multi-GPU streamed PHYSICS, including the real-geography rung ------
run mgphys_correctness "$PY" -m tilestream.test_mgphys --skip-negative
run mgphys_negative    "$PY" -m tilestream.test_mgphys --skip-correctness
run mgphys_geography_1v2 "$PY" -m tilestream.test_mgphys --geography \
    --skip-correctness --skip-negative

echo "=============================================================" \
    | tee -a "$OUT/RUNLOG"
echo "BATTERY TOTALS: $PASS passed, $FAIL failed, $SKIP skipped" \
    | tee -a "$OUT/RUNLOG"
echo "BATTERY_DONE $(date -u +%FT%TZ) pass=$PASS fail=$FAIL skip=$SKIP" \
    > "$OUT/BATTERY_DONE"
[ "$FAIL" -eq 0 ]
