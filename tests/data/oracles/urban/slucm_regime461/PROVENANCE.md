# WRF v4.6.1 single-layer urban regime oracle

Source: WRF release tag `v4.6.1`, peeled commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`. `module_sf_urban.F` SHA-256
`433cc81d97f0096ead284cdcc984b4985f95e1bfb94c6e20d316818997e4e55e`; all five
files compiled or read are pinned in `tools/urban_wrf461_oracle/SOURCES.sha256`
and fetched by `tools/urban_wrf461_oracle/fetch_sources.sh`. Nothing in them
is edited for the fixture.

Built 2026-10-07 UTC in an Ubuntu 26.04 container (GNU Fortran 15.2.0, glibc
2.43, the platform of the `ucm` and `ucm461` fixtures) at `-O0` by
`tools/urban_slucm_regime_oracle/build_u2604.sh`. The driver
`tools/urban_slucm_regime_oracle/slucm_regime_oracle.F90` calls WRF's own
`urban_param_init`, `read_param` and `urban`; the regimes are listed in
`tools/urban_slucm_regime_oracle/REGIMES.csv`.

Rows: 18 regimes x every urban category (3 NLCD, 11 LCZ) x 6 carried steps,
under 14 switch combinations, 8,856 column steps in all. Every row holds the
forcing, the state at entry, the 19 outputs and the 44 state words after
the call, as 9-significant-digit decimals (exact for float32).

The same driver built natively on GNU Fortran 13.3.0 / glibc 2.39 with
`UCM_LIBM_SHIM=1` (atanf and log10f taken as the double routines rounded
once) is byte-identical to this fixture on all 14 variants. Without the
shim the glibc 2.39 build differs on 66 to 378 rows per variant; those are
the rows glibc 2.39's float atanf/log10f (not correctly rounded) decide.

Green-roof variants (`gr`, `griri`, `grch1`) are the `-finit-real=zero`
build: WRF reads the green-roof local `ETR` undefined on the dew arm.
`libm-report.txt` counts the rows that read it.

Controls (never fixtures of WRF's own behaviour):

* `controls/v471`: the three `IMP_SCHEME=2` variants on WRF v4.7.1's
  `module_sf_urban.F` (SHA-256 `623868c7...0ea`), whose only executable
  change from v4.6.1 is the `RHOO/1000.` unit conversion in the three
  retention updates. The engine keeps the corrected equation.
* `controls/tloc`: `slucm-lcz-grch1` on v4.6.1 with the one guard in
  `CONTROL-tloc.diff` widened, so the local hour `tloc` that the
  `IRI_SCHEME=1` arm reads is defined when `AHOPTION=0`.

`tests/test_urban_slucm_regime_wrf461.py` holds the claims.
