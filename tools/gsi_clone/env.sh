# Environment for building and running the HRRR v4.1.21 CPU tools in the
# container. It stands in for the WCOSS2 module loads in the tag's
# modulefiles/HRRR/v4.0.0: the same variable names those modules export,
# pointing at the libraries build_libs.sh installs.
. /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1

export PREFIX=/opt/libs          # zlib, libpng, jasper, hdf5, netcdf, pnetcdf
export NCEPLIBS=/opt/nceplibs    # bacio, w3nco, bufr, g2, g2tmpl, wrf_io
export PATH=/opt/craywrap:/opt/cmake/bin:$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:${LD_LIBRARY_PATH:-}
export CMAKE_PREFIX_PATH=$PREFIX:$NCEPLIBS

# netcdf/4.7.4, hdf5/1.10.6, pnetcdf/1.12.2
export NETCDF=$PREFIX NETCDF_INCLUDE=$PREFIX/include NETCDF_LIB=$PREFIX/lib
export HDF5=$PREFIX PNETCDF=$PREFIX

# zlib/1.2.11, libpng/1.6.37, jasper/2.0.25
export Z_LIB=$PREFIX/lib/libz.a PNG_LIB=$PREFIX/lib/libpng.a JASPER_LIB=$PREFIX/lib/libjasper.a
export Z_INC=$PREFIX/include PNG_INC=$PREFIX/include JASPER_INC=$PREFIX/include

# bacio/2.4.1, w3nco/2.4.1, bufr/11.4.0, g2/3.4.5, g2tmpl/1.10.0, wrf_io/1.1.1
export BACIO_LIB4=$NCEPLIBS/lib/libbacio_4.a BACIO_LIB8=$NCEPLIBS/lib/libbacio_8.a
export W3NCO_LIB4=$NCEPLIBS/lib/libw3nco_4.a W3NCO_LIBd=$NCEPLIBS/lib/libw3nco_d.a W3NCO_LIB8=$NCEPLIBS/lib/libw3nco_8.a
export BUFR_LIB4=$NCEPLIBS/lib/libbufr_4.a BUFR_LIBd=$NCEPLIBS/lib/libbufr_d.a BUFR_LIB8=$NCEPLIBS/lib/libbufr_8.a
export BUFR_LIB4_DA=$NCEPLIBS/lib/libbufr_4_DA.a BUFR_LIBd_DA=$NCEPLIBS/lib/libbufr_d_DA.a BUFR_LIB8_DA=$NCEPLIBS/lib/libbufr_8_DA.a
export G2_LIB4=$NCEPLIBS/lib/libg2_4.a G2_LIBd=$NCEPLIBS/lib/libg2_d.a
export G2_INC4=$NCEPLIBS/include_4 G2_INCd=$NCEPLIBS/include_d
export G2TMPL_LIB=$NCEPLIBS/lib/libg2tmpl.a G2TMPL_INC=$NCEPLIBS/include
export WRF_IO_LIB=$NCEPLIBS/wrf_io/libwrfio_nf.a WRF_IO_INC=$NCEPLIBS/wrf_io

# The tag's build scripts call the Cray compiler drivers (ftn, cc), which on
# WCOSS2 wrap the Intel compilers with MPI built in. /opt/craywrap supplies
# the same two names over the Intel MPI drivers. CC/FC stay the plain names
# GSI's CMakeLists expects when CC is defined.
export CC=mpiicc CXX=mpiicpc FC=mpiifort F77=mpiifort F90=mpiifort
