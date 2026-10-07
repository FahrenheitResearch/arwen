WRF 4.7.1 flow_dep_bdy_chem oracle

Source: public WRF commit f52c197ed39d12e087d02c50f412d90d418f6186.
Extracted byte for byte: chem/module_input_chem_data.F:1531-2031,
1267-1298,1299-1324,2331-2357, through extract-bdy.list.
Registry/registry.chem:3940,4022 pins tracer and GOCART package orders.
Registry/Registry.EM_COMMON:2931 gives spec_zone default 1.
module_data_radm2.F:5 gives epsilc; module_data_sorgam.F:949-950 gives conmin.

Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
glibc: Ubuntu GLIBC 2.43-2ubuntu2.4, version 2.43.
Flags: -O0 -cpp -DRWORDSIZE=4, EM defines, free form, no fast math.
The build checks source hashes and prohibits _ZGV imports in O0 objects.
The positive control imports _ZGVbN4v_expf (libmvec-report.txt beside the build).

Command:
bash tools/chem_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR run_flowdep_bdy
WRF_SOURCE_ROOT is the pinned reference tree (tools/chem_wrf471_oracle/README.md).
The build was run on a Linux host with the compiler below.
Compiler and input hashes: oracle-sha256sums.txt. The final section pins fixtures.

Cases mode1 = method 6, mode2 = GOCART_SIMPLE 300 method 7,
mode3 = CHEM_TRACER 13 method 5, mode4 = alternating supplied/fixed rows.
Widths 1 and 3, dt entries 1/2/3 = 0,0.375,37.25, three active k levels,
all four sides and ring corners, mixed negative/zero/positive face flux.
Boundary input rows 2 and 3 are offset by 100 and 200 to expose a wrong row
read. Only row 1 is published because that is the port's input interface.
Negative and zero boundary values exercise the epsilc floor.
The south corner also distinguishes separate multiplication/addition from
fused evaluation by one ULP at dt=0.375.

The prelude supplies inactive Registry indices as 1. do_pvozone is false.
Unreachable mechanism/PV routines error stop if reached. get_last_gas returns
an unused zero: none of methods 5,6,7 reads numgas. The CAM USE dependency is
a service module with an external routine that also error stops.
These fixtures qualify transport boundaries only, with no PV ozone override.

MANIFEST dimensions retain Fortran order. tests/test_chem_bdy_wrf471_parity.py
transposes chem (i,k,j,row) to (row,k,j,i), faces (i,k,j) to (k,j,i),
and boundary rows (transverse,k,row) to (row,k,transverse).
CPU identity: all 24 cases, every output row, max_ulp=0,n_nonzero=0,n=270.
GPU identity and performance remain unmeasured under this lane's CPU-only authority.
