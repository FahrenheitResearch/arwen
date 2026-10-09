# env-s4.sh : terrain-clock local-face lane, step 4 task B (adaptive rows), box W1. Engine = the step-3 build of the lane
# tree at fbb63fb4a (/work/tclock/s3/tree, its venv); the probe is the lane's tools/terrain_clock_probe.py with the
# blow-up criterion on the adaptive clock (uploaded to /work/tclock/s4/tools).
export S=/work/tclock/s3 T=/work/tclock/s3/tree W4=/work/tclock/s4
export HOME=$S/home PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NUMPY_MADVISE_HUGEPAGE=0
export CUDA_CACHE_PATH=$S/cuda-cache CUDA_CACHE_MAXSIZE=4294967296 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PY=$S/venv/bin/python PATH=$S/venv/bin:$PATH PYTHONPATH=$T
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
