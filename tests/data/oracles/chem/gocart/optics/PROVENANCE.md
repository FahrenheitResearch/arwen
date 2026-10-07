# WRF-Chem column oracle

33 aerosol cases, three independent 50-level columns per case. The additional
rrtmg_edges case isolates conversion branches on one 50-level column.
Input chemistry is in WRF registry order with dummy scalar 1, followed by
the 19 gocart_simple species. Aerosols use ug/kg; sulf and msa use ppmv.
alt is specific dry-air volume, dz8w is m, and relhum is a fraction.

The harness calls the extracted GOCART preparation and option_mie=1 calls,
the original prep and Mie routines, extracted optical_averaging clamps,
the byte-unmodified whole module_aer_opt_out.F, and the extracted RRTMG
feedback conversion. Inactive non-GOCART methods are not compiled.
RRTMG's initial tau/ssa/asm are 0/1/0, as in its source before feedback.
AOD5502D is an ArWen-only ordered float32 integral of WRF's EXTCOF55.
WRF-Chem has no AOD5502D field.

WRF-Chem v4.7.1, commit f52c197ed39d12e087d02c50f412d90d418f6186.
Generated from the pinned sources by tools/chem_wrf471_oracle/gocart/build_optics.sh.
Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
Runtime: Ubuntu glibc 2.43-2ubuntu2.4, x86-64 SSE arithmetic.
Build: -O0 -cpp -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4
-DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -ffree-form
-ffree-line-length-none -fallow-argument-mismatch.
No _ZGV symbol in the linked oracle objects. The separate -Ofast positive
control references _ZGVbN4v_expf; the build writes both symbol lists
(undefined-O0.txt, undefined-control.txt) into its build directory.

Command, with WRF_SRC a WRF v4.7.1 tree holding the files pinned below:
```sh
bash tools/chem_wrf471_oracle/gocart/build_optics.sh WRF_SRC build fixtures tables
```
(tools/chem_wrf471_oracle/gocart/build.sh runs it too.)  A second build
from the pinned sources on another checkout, with the line-range extracts
made fresh by the script, reproduced oracle-sha256sums.txt (195c1e6f...) and
table-sha256sums.txt (d39e1c78...) byte for byte.  Extracts are byte-unmodified sed
line ranges of the pinned files. Registry stubs contain declarations only.
Message and fatal services print or stop, without replacing scientific code.
registry_declarations.inc reuses the shared stub_gocart.F90:15-50 declarations.
stub_optics.F90 extends module_configure with the aerosol output indices.
oracle_io.F90 is the chem oracles' shared writer, unchanged.

Source SHA-256 values:
```
073113232be6e8f0d800825e0af5b0c5434cd5db1a3ef428569e79a68d8c92d4  chem/module_optical_averaging.F
5afcfa5d26e3d40c42be20253d7f1d3b88cf0b279e5f87d164912bff8b792b5e  chem/module_data_rrtmgaeropt.F
67ffc5b574ad6994e572890aefa3b9069843af5e4b08fd68583437baf4300709  chem/module_data_sorgam.F
efc13a98a4899babee8cf420b23ffb9bc4de8049426bc78d85a7fb2ed3b2b3c2  chem/module_data_gocartchem.F
d7288531253596075ab0703448b164a083fb943802655adf924cc62feb078cf0  chem/module_data_gocart_seas.F
8694100875645b54a58bcff38d462ad3c9fd115261fb39941358466c84b62329  phys/module_data_gocart_dust.F
55850415d1d357f350bf975fb9a860744e19a2fb3330ded310987279c1d4169f  chem/module_data_mosaic_asect.F
46ea2932a1c1c1cfdc41c6ef23a4d0405bc768110050f9a4a3ba1dca4f7365d0  chem/module_aer_opt_out.F
447345d2658cd370e6bc97ff2ab582a5d12b84adffc58f72a938b353e017987e  phys/module_ra_rrtmg_sw.F
416a92e91bf5475b5f94f3bdf0f18fdcbb1c7bc9c88131cb5d7457e38b542f6e  Registry/registry.chem
5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062  share/module_model_constants.F
```

Extracted line ranges:
```
chem/module_optical_averaging.F 3545 4186 prep.inc
chem/module_optical_averaging.F 296 303 prep_call.inc
chem/module_optical_averaging.F 426 436 mie_call.inc
chem/module_optical_averaging.F 476 482 sw_clamps.inc
chem/module_optical_averaging.F 493 496 lw_clamps.inc
chem/module_optical_averaging.F 4453 4454 fit_limits.inc
chem/module_optical_averaging.F 4466 4467 fit_sizes.inc
chem/module_optical_averaging.F 3689 3727 section_fractions.inc
chem/module_optical_averaging.F 4221 6978 mie.inc
chem/module_data_sorgam.F 728 749 modal_parameters.inc
chem/module_data_sorgam.F 1191 1200 hygro_parameters.inc
chem/module_data_mosaic_asect.F 581 581 msa_parameter.inc
chem/module_data_gocartchem.F 18 20 mass_parameters.inc
phys/module_ra_rrtmg_sw.F 10241 10249 rrtmg_parameters.inc
phys/module_ra_rrtmg_sw.F 11355 11430 rrtmg_conversion.inc
```
