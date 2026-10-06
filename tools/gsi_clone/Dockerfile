# Container for NOAA-EMC/HRRR tag v4.1.21's CPU analysis tools, built from the
# tag's own sources with no source edits: GSI (cloud analysis compiled in),
# EnKF, process_mosaic, process_cloud, process_lightning, WRF-ARW real and
# wrf, update_bc, ref2tten, WPS ungrib and metgrid.
#
# Every input is pinned: the base image by digest, Ubuntu packages through the
# dated snapshot archive, the Intel compilers by exact package version, every
# third-party and NCEPLIBS source tarball by SHA-256 (SOURCES.sha256, written
# by fetch_sources.sh), and the HRRR tree by tag and commit (stage "gsi").
# Library versions follow the tag's versions/build.ver (the WCOSS2 module set).
# BUILD.md records what each pin is and why.
#
# Build (from this folder, after ./fetch_sources.sh):
#   podman build --target toolchain -t gsi-clone-toolchain:v4.1.21 .
#   podman build --target gsi       -t gsi-clone-gsi:v4.1.21 .
#   podman build --target full      -t gsi-clone-full:v4.1.21 .

FROM docker.io/library/ubuntu:20.04@sha256:c664f8f86ed5a386b0a340d981b8f81714e21a8b9c73f658c4bea56aa179d54a AS toolchain

# Ubuntu 20.04 (glibc 2.31) is the newest base the 2021.1 Intel compilers
# list as supported. The snapshot date fixes every apt package version.
# snapshot.ubuntu.com serves https only, so ca-certificates comes first from
# the image's own (end-of-life, frozen) focal archive.
ARG APT_SNAPSHOT=20260901T000000Z
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC LC_ALL=C.UTF-8 LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN printf '%s\n' \
      "deb https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT} focal main universe" \
      "deb https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT} focal-updates main universe" \
      "deb https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT} focal-security main universe" \
      > /etc/apt/sources.list \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends \
      build-essential m4 file git curl ca-certificates gnupg tcsh csh ksh perl \
      python3 bc gawk procps time patch xz-utils \
 && rm -rf /var/lib/apt/lists/*

# Intel classic compilers. Operational HRRR v4.1.21 was built with
# intel/19.1.3.304 (Parallel Studio 2020 update 4), which Intel no longer
# distributes; 2021.1.1 is the first oneAPI release (ifort/icc 2021.1, the
# direct successor of 19.1.3) and the closest build still downloadable.
# Intel MPI stands in for cray-mpich 8.1.7 (both MPICH-derived). MKL supplies
# the LAPACK that GSI's GENERIC host links with -mkl.
#
# intel-packages.txt names every Intel package of the install by exact
# version: the four asked for (compiler-fortran, the classic C/C++ compilers,
# mpi-devel and mkl-devel, all 2021.1.1) and each package apt brings in beside
# them. Several of those carry no version in their name (the common
# variables, a second MPI runtime, a 2026 compiler runtime) and would
# otherwise float to whatever the repository holds on the build day. The two
# "latest" links are checked so the 2021.1.1 compilers and MPI are the ones
# setvars.sh selects.
COPY intel-packages.txt /opt/intel-packages.txt
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneAPI.list \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends $(cat /opt/intel-packages.txt) \
 && rm -rf /var/lib/apt/lists/* \
 && test "$(readlink /opt/intel/oneapi/compiler/latest)" = 2021.1.1 \
 && test "$(readlink /opt/intel/oneapi/mpi/latest)" = 2021.1.1

# RUN lines that need the compilers call bash by name and source env.sh:
# podman's default OCI image format ignores a SHELL instruction, and the
# Intel setvars.sh needs bash.
COPY env.sh /opt/gsi-env.sh

# Pinned sources, verified before anything is unpacked.
COPY SOURCES.sha256 /opt/src/SOURCES.sha256
COPY src/ /opt/src/
RUN cd /opt/src && sha256sum -c SOURCES.sha256

# CMake 3.18.4 (build.ver cmake_ver), the prebuilt Kitware binary.
RUN mkdir -p /opt/cmake && tar xzf /opt/src/cmake-3.18.4-Linux-x86_64.tar.gz -C /opt/cmake --strip-components=1

ARG JOBS=8
COPY build_thirdparty.sh /opt/build_thirdparty.sh
RUN bash -c ". /opt/gsi-env.sh && bash /opt/build_thirdparty.sh ${JOBS}"
COPY build_nceplibs.sh /opt/build_nceplibs.sh
RUN bash -c ". /opt/gsi-env.sh && bash /opt/build_nceplibs.sh ${JOBS}" \
 && dpkg-query -W -f='${Package} ${Version}\n' 'intel-oneapi-*' > /opt/BUILD-MANIFEST-apt-intel.txt \
 && dpkg-query -W -f='${Package} ${Version}\n' > /opt/BUILD-MANIFEST-apt-all.txt \
 && bash -c ". /opt/gsi-env.sh && { ifort --version; icc --version; mpiifort -v; }" > /opt/BUILD-MANIFEST-compilers.txt 2>&1

# ---------------------------------------------------------------------------
# GSI, EnKF and the observation preprocessors, from the tag, sources untouched.
FROM toolchain AS gsi
ARG HRRR_TAG=v4.1.21
ARG HRRR_COMMIT=a131d6c7f6ebf57ae3c43f198229ad6afdf4d886
RUN git clone --quiet --depth 1 --branch "${HRRR_TAG}" https://github.com/NOAA-EMC/HRRR.git /opt/hrrr \
 && test "$(git -C /opt/hrrr rev-parse HEAD)" = "${HRRR_COMMIT}" \
 && mkdir -p /opt/hrrr/exec
ARG JOBS=8
COPY build_gsi.sh /opt/build_gsi.sh
RUN bash -c ". /opt/gsi-env.sh && bash /opt/build_gsi.sh ${JOBS}" \
 && test -z "$(git -C /opt/hrrr status --porcelain --untracked-files=no)"
COPY build_obs_tools.sh /opt/build_obs_tools.sh
RUN bash -c ". /opt/gsi-env.sh && bash /opt/build_obs_tools.sh" \
 && test -z "$(git -C /opt/hrrr status --porcelain --untracked-files=no)"

# ---------------------------------------------------------------------------
# WRF-ARW (real and wrf) plus update_bc and ref2tten, which link pieces of
# the built WRF tree. libtirpc is the one extra system library the tag's WRF
# configure file links.
FROM gsi AS full
ARG APT_SNAPSHOT=20260901T000000Z
RUN apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends libtirpc-dev \
 && rm -rf /var/lib/apt/lists/*
ARG JOBS=8
COPY build_wrf.sh /opt/build_wrf.sh
RUN bash -c ". /opt/gsi-env.sh && bash /opt/build_wrf.sh ${JOBS}"
# WPS ungrib and metgrid, which the tag's makeguess script runs before real.
COPY build_wps.sh /opt/build_wps.sh
RUN bash -c ". /opt/gsi-env.sh && bash /opt/build_wps.sh"
# wgrib2 2.0.7, which the makeguess script runs before ungrib. WPS 3.9.1's
# ungrib misreads the complex-packed RAP file as published (MEASURED
# 2026-10-03: pressures of 1.3e11 Pa in the metgrid output), which is why the
# script repacks it to simple packing first. gfortran comes from the same
# dated snapshot.
RUN apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends gfortran \
 && rm -rf /var/lib/apt/lists/*
COPY SOURCES-run.sha256 /opt/src-run/SOURCES-run.sha256
COPY src-run/ /opt/src-run/
COPY build_wgrib2.sh /opt/build_wgrib2.sh
RUN cd /opt/src-run && sha256sum -c SOURCES-run.sha256 \
 && cp /opt/src-run/wgrib2.tgz.v2.0.7 /opt/src/ \
 && bash /opt/build_wgrib2.sh
