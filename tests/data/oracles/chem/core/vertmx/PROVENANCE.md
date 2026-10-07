# WRF v4.7.1 vertmx oracle

WRF commit f52c197ed39d12e087d02c50f412d90d418f6186, public domain.
GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0, glibc 2.43-2ubuntu2.4.
Flags are defined in tools/chem_wrf471_oracle/build.sh. Exact source
SHA-256 values are in oracle-sha256sums.txt.

Build: bash tools/chem_wrf471_oracle/build.sh source build run_chem_prep run_vertmx
The owned source overlay contains the pinned chem sources and the separately
fetched share/module_model_constants.F at the same WRF commit. The script
copy had CRLF converted to LF for bash; extracted WRF routines kept their bytes.
No -O0 object imports a _ZGV symbol. The -Ofast positive control does.
Every WRF source, harness file and generated wrapper hash: oracle-sha256sums.txt.
fixtures.sha256 pins every binary and MANIFEST.txt in this directory.

18 case families at nz=49 and 59, five carried steps each. Gas and aerosol
rows share transport inputs and exercise different column units. Families
1-4 span zero/small/large/sharp-top exchange; 5-15 span exact CO, PM, fire,
Registry-presence and urban gates; 16-18 put the profile at/around epsilc.
dt=18/36/60 s and vd=0/.001/.01/.05 m s-1 are covered across the families.
All four WRF routines are byte-extracted; dry_dep_driver.F:675-805 is
included verbatim in run_vertmx.F90. Only surrounding loops and declarations
differ. The last physical mass level is solved but is not written back.
WRF ddmassn is per-step max(0,old-new); accum records five additions.
