#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${GPUWM_STEP_RUNTIME:?set the local runtime root}"
ulimit -n 65536
export CUPY_GPU_MEMORY_LIMIT=21474836480
export GPUWM_NATIVE_STEP_LIBRARY=$PWD/receipts/libawstep-host-argument.so
export LD_PRELOAD=$GPUWM_NATIVE_STEP_LIBRARY
export CUDA_PATH=$GPUWM_STEP_RUNTIME/venv/lib/python3.11/site-packages/nvidia/cu13
export LD_LIBRARY_PATH=$CUDA_PATH/lib
export PYTHONPATH=$PWD:$GPUWM_STEP_RUNTIME/data
export GPUWM_STEP_BINDING=$PWD/receipts/canary-local-binding.json
export CUPY_CACHE_DIR=$PWD/cache/cupy
# The engine plan cache does not follow XDG_CACHE_HOME. Keep replayed
# compiler include paths on this runtime, away from shared-home plans.
export GPUWM_KERNEL_PLAN_DIR=$PWD/cache/kernel-plans
exec "$GPUWM_STEP_RUNTIME/venv/bin/python" -B tools/native_step/run_canary_step.py "$@"
