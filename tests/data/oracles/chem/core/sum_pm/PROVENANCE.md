# sum_pm_gocart oracle

WRF v4.7.1, f52c197ed39d12e087d02c50f412d90d418f6186, public domain.
chem/module_gocart_aerosols.F:76-150 is extracted byte for byte by
extract-pm.list. chem/module_data_gocartchem.F is compiled as a whole file,
with no generated dependencies: nh4_mfac=1.375 at :18, oc_mfac=1.8 at :20.
share/module_model_constants.F:34 gives mwdry=28.966, declared REAL.
The wrapper supplies that exact REAL constant in prelude-pm.inc.

Registry/registry.chem:4022 lists the gocart_simple package (chem_opt 300):
so2 sulf dms msa p25 bc1 bc2 oc1 oc2 dust_1 dust_2 dust_3 dust_4 dust_5
seas_1 seas_2 seas_3 seas_4 p10.
tools/gen_scalar_indices.c:98 reserves slot 1; :153-169 walks package tokens
and assigns consecutive indices starting at 2, rather than declaration order.
The generated framework substitute reference-pm/registry-pm.F90 declares
only these indices and an empty module_configure, not physical calculations.
Thus p_p25..p_dust_1 is p25 bc1 bc2 oc1 oc2 dust_1;
p_p25..p_dust_3 additionally includes dust_2 dust_3;
p_seas_1..p_seas_3 is seas_1 seas_2 seas_3.

Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
glibc: Ubuntu GLIBC 2.43-2ubuntu2.4.
Flags: -O0 -cpp -ffree-form -ffree-line-length-none
-fallow-argument-mismatch, EM_CORE=1 NMM_CORE=0 RWORDSIZE=4 IWORDSIZE=4
DWORDSIZE=8 LWORDSIZE=4 WRF_CHEM=1. No fast math or FMA.
libmvec-report.txt shows no _ZGV imports at -O0 and a working positive control.
oracle-sha256sums.txt is the unmodified build receipt;
fixture-sha256sums.txt pins each published binary and manifest.

Reproduce on a CPU with gfortran using:
bash tools/chem_wrf471_oracle/build-pm.sh PINNED_WRF_ROOT OWNED_BUILD_DIR
The helper creates its source overlay only under OWNED_BUILD_DIR and checks
SOURCES-pm.sha256, including the two public files fetched from the pinned
raw GitHub commit and the generated Registry substitute.

Cases, 24 cells per output:
case_00 zero; case_01 realistic profile; case_02..20 each of the 19 package
species alone; case_21 large (1e20 scaling); case_22 tiny (1e-30 scaling)
with dense alt; case_23 high alt; case_24 unit alt.
Inputs have distinct horizontal and vertical values. Fixtures cover physical
mass cells, not WRF's duplicate terminal i/j halo projection (min(ide-1,i)).
PM2_5_DRY and PM10 each measure max_ulp=0, n_nonzero=0, n=24 in every case.
PM2_5_DRY_EC is identically zero in WRF and recorded as an oracle control,
not exposed as a new table diagnostic.

OC correction: REAL(4)(1.8) is 1.7999999523162841796875.
REAL(4)(oc_mfac-1.) is 0.7999999523162841796875, word 0x3f4ccccc.
PM rows preserve the sulfate multiply/divide/multiply order and each range
addition separately; reducing a range into a term can change rounding.
