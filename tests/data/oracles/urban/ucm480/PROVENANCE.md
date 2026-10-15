These are small Fortran column receipts, not forecast history files.

Source: WRF v4.8.0, commit 06d4240ae989cc3e50af412bb472df3d9048783c.
phys/module_sf_urban.F SHA-256:
1be801c3e11aad59be8e0e23612bbe71ba4fa29cefc930520fb4c6e2e1169f6a.
The module's pristine source is retained separately on the CPU box.

The separately defined reference comes from
tools/urban_wrf480_oracle/defined_state.py. It initializes undefined entry
memory, preserves the unassigned bottom moisture boundary, and defines the
irrigation clock when anthropogenic heat is disabled. No blanket zero-init
reference is claimed exact. The source cold-start helper itself is unchanged.

GNU Fortran 13.3.0, Ubuntu 24.04, scalar REAL kind 4 and source-declared
REAL(kind=8) Jacobian variables. Scheme flags: -O0 -g -cpp -Dwrfmodel
-DEM_CORE=1 -ffree-form -ffree-line-length-none -fcheck=all -finit-real=snan
-ffpe-trap=invalid,zero,overflow. Compiler default contraction is disabled
in the native C++ comparison: -fno-fast-math -ffp-contract=off.

The existing tools/urban_bem_regime_oracle/libm_shim.c is linked as the
reference-only float rounding control. A separate stock-platform comparison
is recorded. WOOF uses its own float32 math routines.

Rebuild: make_tree_driver.py writes the WRF-facing driver; run_rounded takes
TABLE LABEL PREFIX CH TS AH ALH IMP IRI GR FGR BOUND TREE TT FVG_ROWS.
These six receipts use CH/TS/AH/ALH/IMP/IRI/BOUND=-9, GR=0, FGR=.5,
TREE=1 and FVG=.2 for tt0/tt1, or TREE=0 and FVG=0 for tree0. TT is 0 or 1.
Every case has nine warm, cold, stable, unstable, rain and dew forcing
scenarios, with six carried Fortran steps per urban category.
