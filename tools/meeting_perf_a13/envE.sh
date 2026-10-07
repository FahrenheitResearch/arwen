# meeting-perf-a13 on box E: the da-e native bridges, tables and venv (read only), this lane's engine copy.
source /work/da-e/env.sh
W=/work/meeting-perf-a13
export DA_ENGINE=$W/trees/tip PYTHONPATH=$W/sitex:$W/trees/tip
export CUPY_CACHE_DIR=$W/cache/cupy XDG_CACHE_HOME=$W/cache TMPDIR=$W/tmp MPLCONFIGDIR=$W/cache/mpl
export GPUWM_STATIC_SOURCE_ROOT=$W/geog/static_sources GEOG=$W/geog
mkdir -p $W/cache/cupy $W/tmp
export GPUWM_MAPPED_ENGINE_MEMORY_BUDGET_BYTES=$((40*1024*1024*1024))
