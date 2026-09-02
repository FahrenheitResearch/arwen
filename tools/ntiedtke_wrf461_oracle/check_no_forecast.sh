#!/usr/bin/env bash
# Refuse if a forecast is running.  RUN THIS IMMEDIATELY BEFORE EVERY COMMIT.
#
# WHY, and the part that is easy to get half-right.  CLAUDE.md says "never
# git commit -- or edit tracked source -- while a forecast is running".  The
# obvious reading is that EDITING is the hazard and a commit is incidental.
# That reading is wrong and it cost a run.
#
# gpuwm/prepared_domain_tree_forecast.py:1374 `_runtime_source_identity()`
# returns
#
#     gpuwm_version, git_commit, git_tree, source_sha256
#
# where source_sha256 covers five files (core/model.py, core/nest.py,
# core/microphysics_transition.py, core/kernels/nest_microphysics.cu, and
# prepared_domain_tree_forecast.py itself).  The identity is captured at
# start and RE-CHECKED AT COMPLETION (:2307); a mismatch raises
#
#     RuntimeError("forecast implementation changed during execution")
#
# git_commit and git_tree are in that mapping.  So ANY commit -- including a
# documentation-only commit touching no source at all -- changes the identity
# and kills the run AT THE FINISH LINE, after all the compute is spent.
#
# MEASURED, 2026-08-29: seven commits were made during a 51-minute
# tc_hafs_kf3 run.  The run was recovered only by branching the work
# aside and `git reset --soft` back to the commit HEAD held at launch; it
# then completed SUCCESS.  Nothing in the tree would have caught it.
#
# Checking once per session is NOT enough -- a forecast can start at any
# time, and one did.  Check immediately before each commit.
#
#     bash tools/ntiedtke_wrf461_oracle/check_no_forecast.sh && git commit ...
set -uo pipefail

# ---------------------------------------------------------------------------
# BLIND SPOT FOUND 2026-08-29, live, at 87% of a 14-hour run.
#
# This script matched ONLY the command line.  A forecast launched through a
# wrapper -- a script that does `from gpuwm.prepared_domain_tree_forecast
# import main` and calls it -- puts none of the three names on its own
# command line, so the match failed and the gate printed "safe to commit"
# over a live run.  The process was python.exe and WAS returned by the
# query; only the pattern missed.  No commit actually landed inside that
# window, but by sequencing rather than by this script.
#
# TWO ARMS NOW, AND BOTH ARE NEEDED.  They are not two chances at the same
# catch -- each is unsound alone and each covers the other's blind spot.
# Do not delete the "redundant" one:
#
#   * SYNTACTIC (command line).  Same class of fragility that just failed,
#     and the ONLY arm that covers STARTUP: a run pays tens of seconds of
#     NVRTC compilation before its first step, and nothing writes
#     progress.jsonl in that window.  A behavioural check alone is blind
#     for a minute-plus at every launch.
#
#   * BEHAVIOURAL (a recently-written progress.jsonl).  Observes a forecast
#     DOING something instead of inferring it from how it was spelled --
#     any wrapper, any invocation shape, any future entry point.  Its
#     failure mode is a stale file inside the window, which refuses when it
#     needn't: the safe direction.
#
# The structurally correct fix is for the RUNNER to declare itself with a
# lock file, so this stops guessing from outside.  Not done: it would edit
# prepared_domain_tree_forecast.py, one of the five files whose
# source_sha256 the run identity hashes, on a runner a live campaign is
# using.  Recorded here as the shape this wants if it fails a third time.
# ---------------------------------------------------------------------------

# Entry-point names, plus flags every forecast invocation carries whatever
# wraps it.
PATTERNS='prepared_domain_tree_forecast|prepared_single_domain_forecast|run_forecast|--prepared-root|--preparation-receipt-sha256'
# The test suite narrows the pattern to a token only its own fixtures carry,
# so a real forecast on the box does not decide the outcome of a test about
# the gate's logic.  Production never sets it.
PATTERNS=${GPUWM_GATE_PATTERNS:-${PATTERNS}}

# Seconds of progress-file silence after which the behavioural arm stops
# claiming a run is live.  Generous: a slow step on a big domain can be
# seconds, and refusing when unsure is the safe direction.
PROGRESS_FRESH_SECONDS=${GPUWM_GATE_PROGRESS_FRESH_SECONDS:-90}
RUNS_ROOT=${GPUWM_RUNS_ROOT:-E:/GPUWRF/runs}

query_failed=0
if ! command -v powershell.exe >/dev/null 2>&1; then
    # Linux and macOS arm: the process table through ps, which is the
    # same question the Windows arm asks CIM.  ps failing is the query
    # failing; grep finding nothing is an answer, not a failure.
    if ps -eo pid=,args= >/dev/null 2>&1; then
        running=$(ps -eo pid=,args= 2>/dev/null | grep -E 'python' | grep -E "${PATTERNS}" | grep -v -E 'grep -E' | cut -c1-130 | sed '/^[[:space:]]*$/d') || true
    else
        query_failed=1
    fi
else
running=$(powershell.exe -NoProfile -Command   "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |
     Where-Object { \$_.CommandLine -match '${PATTERNS}' } |
     ForEach-Object { \"\$(\$_.ProcessId)  \$(\$_.CommandLine.Substring(0,[Math]::Min(120,\$_.CommandLine.Length)))\" }"   2>/dev/null | tr -d '' | sed '/^[[:space:]]*$/d') || query_failed=1
fi

# FAIL CLOSED: a gate that passes when its instrument breaks is the very
# failure this file is about.
if [[ ${query_failed} -ne 0 ]]; then
    echo "REFUSING: could not query running processes, so this script" >&2
    echo "cannot tell whether a forecast is live.  Fix the query or wait." >&2
    exit 1
fi

# Behavioural arm.  Any progress.jsonl written in the last
# PROGRESS_FRESH_SECONDS means something is stepping right now.
fresh=""
if [[ -d "${RUNS_ROOT}" ]]; then
    fresh=$(find "${RUNS_ROOT}" -name progress.jsonl -type f               -newermt "-${PROGRESS_FRESH_SECONDS} seconds" 2>/dev/null             | sed '/^[[:space:]]*$/d')
fi

if [[ -n "${running}" || -n "${fresh}" ]]; then
    echo "REFUSING: a forecast is running.  Committing would change" >&2
    echo "git_commit in its runtime identity and kill it at completion." >&2
    echo >&2
    [[ -n "${running}" ]] && { echo "matched by command line:" >&2
                               echo "${running}" >&2; }
    [[ -n "${fresh}" ]] && {
        echo "progress written in the last ${PROGRESS_FRESH_SECONDS}s:" >&2
        echo "${fresh}" >&2; }
    echo >&2
    echo "Wait for it to finish, then commit." >&2
    exit 1
fi

echo "no forecast running -- safe to commit"
