#!/bin/bash
# setup.sh : terrain-clock-local-face lane (STEP 1, measure). Unpacks the lane tree (TREE-COMMIT) and builds this lane's
# own venv with the probe's imports. CPU only, no card. Box-local paths only.
set -u; W=/work/tclock; T=$W/tree; cd $W; t0=$(date +%s)
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
rm -rf $T && mkdir -p $T && tar -xzf $W/tree.tgz -C $T && cp $W/TREE-COMMIT $T/ && echo "untar $(( $(date +%s) - t0 ))s"
mkdir -p $W/home $W/cuda-cache $W/logs $W/results
if [ ! -x $W/venv/bin/python ]; then
  PYB=$(ls -d /work/bench/pythons/cpython-3.14.7+freethreaded-linux-x86_64-gnu/bin/python3.14t 2>/dev/null)
  uv venv -q --python "$PYB" $W/venv || exit 1
  uv pip install -q --python $W/venv/bin/python "numpy>=2.0" "scipy>=1.11" "netCDF4>=1.6" "jsonschema>=4.0" "threadpoolctl>=3.1" "cupy-cuda12x[ctk]>=14.0" || exit 1
fi
cd $T && PYTHONPATH=$T $W/venv/bin/python -c "import gpuwm,sys,numpy;print('gpuwm',gpuwm.__file__,sys.version.split()[0],numpy.__version__)"
echo "setup_total $(( $(date +%s) - t0 ))s"; echo SETUP_DONE
