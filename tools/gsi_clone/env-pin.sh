# Environment for building and running a GSI pin (Dockerfile.pin). It stands
# in for the WCOSS2 module loads of the pin's build.ver: the Intel compilers
# and MPI of the toolchain stage, and the pin's own libraries under
# /opt/gsi-pin/libs. Nothing from the HRRR tag's /opt/libs or /opt/nceplibs
# is on any search path here.
. /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
export PIN_ROOT=/opt/gsi-pin
export PIN_LIBS=$PIN_ROOT/libs
export TMPDIR="${TMPDIR:-$PIN_ROOT/tmp}"
mkdir -p "$TMPDIR"
export PATH=$PIN_ROOT/cmake/bin:$PIN_LIBS/bin:$PATH
export LD_LIBRARY_PATH=$PIN_LIBS/lib:${LD_LIBRARY_PATH:-}
export CMAKE_PREFIX_PATH=$PIN_LIBS
export NETCDF=$PIN_LIBS HDF5_ROOT=$PIN_LIBS
# The Cray ftn, cc and CC drivers on WCOSS2 compile everything with MPI built
# in; the Intel MPI drivers do the same here.
export CC=mpiicc CXX=mpiicpc FC=mpiifort F77=mpiifort F90=mpiifort
