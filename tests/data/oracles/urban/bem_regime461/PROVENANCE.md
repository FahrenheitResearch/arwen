# BEP+BEM regime oracle, WRF v4.6.1 (`sf_urban_physics = 3`)

Eight fixtures, each the complete dump of one run of
`tools/urban_bem_regime_oracle/run_bep_bem_regime.F90` (sha256
`664175871550d9221cc05cc7624e39b535c3bbdc4a2f90e8fe2873edaec2e76e`) against
byte-unmodified WRF v4.6.1 sources: release tag `v4.6.1`, peeled commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, fetched by
`tools/urban_bem_regime_oracle/fetch_sources.sh` and pinned in
`tools/urban_bem_regime_oracle/SOURCES.sha256`:

| source | sha256 |
| --- | --- |
| `phys/module_sf_bep_bem.F` | `392ae96ad16000e7592112c12f2fe7909bec1760c67c5ef33ec876048a745c48` |
| `phys/module_sf_bem.F` | `7bc761ee592feadbaeb974b4271b956044eb02c5a4f3bf0e77af4fbabf0de8eb` |
| `phys/module_sf_urban.F` | `433cc81d97f0096ead284cdcc984b4985f95e1bfb94c6e20d316818997e4e55e` |
| `phys/module_bep_bem_helper.F` | `79403e10104e23fc2a44eb7cf9c0a33c10fabed3b963c9ccb0ea0bfd29072b6d` |
| `share/module_model_constants.F` | `5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062` |
| `frame/module_wrf_error.F` | `2ed7dc6e90e0fe442ffee84512b4998d31c8ec3400d3a7ab6078404065f784a6` |
| `run/URBPARM.TBL` | `5811226b3db503ae02d8b35cb5cb10e0e90804e451f0f4a5b76a274ee64773d0` |
| `run/URBPARM_LCZ.TBL` | `ab08e3f79d2f5d9d329aa2c953de4e7d94a71741baa0a50f1ecc145c6f81d9ab` |

Regimes: `tools/urban_bem_regime_oracle/REGIMES.csv` (sha256
`e7e4081da98977f5f7c6da23005dc314a29fc0edb12fcf5be428ba4e8337dfc6`), 14
warm-season regimes (declination +0.40 rad) and 12 cold-season regimes
(-0.40 rad): convective afternoons, tropical humid noon, convective shower
(24 mm/h) and heavy rain (90 mm/h), clear calm and cold-inversion nights,
coastal marine, high terrain day and night (surface pressure 815 hPa and
805 hPa), desert with a 25 m first level, summer and winter windstorms,
dew night with a 6 m first level, near calm sun, sunrise, the 24 h wrap of
the local hour inside a run, frozen snow-season day, freezing rain, arctic
night (236 K), southern-hemisphere summer, tropical night, morning heating
demand, and a grazing winter sun at 62 N.

Every regime is crossed with every urban class of the table (3 for
`URBPARM.TBL`, 11 for `URBPARM_LCZ.TBL`), with handed-in fractions cycling
through 0 (the table-fraction arm), 0.3, 0.5, 0.75, 0.9 and 0.99, then eight
edge columns are appended: rural land and open water (FRC 0, which BEP_BEM
must not touch), FRC 0.01, FRC 1.0, gridded morphology, gridded towers up to
the 15th 5 m height bin, a 40 m first level (the whole canopy inside level
1) and a 2 m first level.  30 mass levels.

| file | table | season | columns (urban) | calls | dt |
| --- | --- | --- | --- | --- | --- |
| `bep_bem_nlcd-summer.npz` | `URBPARM.TBL` | warm | 50 (48) | 6 | 60 s |
| `bep_bem_nlcd-winter.npz` | `URBPARM.TBL` | cold | 44 (42) | 6 | 30 s |
| `bep_bem_lcz-summer.npz` | `URBPARM_LCZ.TBL` | warm | 162 (160) | 6 | 60 s |
| `bep_bem_lcz-winter.npz` | `URBPARM_LCZ.TBL` | cold | 140 (138) | 6 | 30 s |
| `bep_bem_gr1pv-summer.npz` | gr1pv sed | warm | 50 (48) | 6 | 60 s |
| `bep_bem_gr2-winter.npz` | gr2 sed | cold | 44 (42) | 6 | 60 s |
| `bep_bem_long-summer.npz` | `URBPARM.TBL` | warm | 50 (48) | 90 | 20 s |
| `bep_bem_stop-summer.npz` | `URBPARM.TBL` | warm | 50 (48) | 40 (WRF stops in 37) | 60 s |

The raw dumps' sha256 (`data.bin`, before packing) are in
`data-sha256sums.txt`; `to_npz.py` repacks them without arithmetic.

574 urban columns in all.  The gr1pv and gr2 tables are the pinned
`URBPARM.TBL` with the same `sed` lines as
`tools/urban_wrf471_oracle/run_bep_bem.sh` (green roof type 1 with rooftop
PV, irrigation hours, an air-conditioning schedule and `SW_COND 1, 0, 1`;
green roof type 2 with irrigation every hour).

`stop-summer`: WRF's own `BEP_BEM` executes the Fortran `STOP` in
`upward_rad` (module_sf_bep_bem.F:5298, a roof layer temperature below
100 K) in call 37, on column 27 (1-based; the 28 m/s windstorm on class 3).
The roof surface heat flux alternates sign every call from call 1 on with a
growing amplitude (grdflx_urb -122, +412, -197, ... +9145 W/m2 by call 36):
the explicit roof surface update is unstable at a 60 s step under that
wind.  The same column at 20 s (`long-summer`, 90 calls) stays bounded.
The dump ends with call 37's inputs and no outputs; the engine must raise
`BepBemError` naming that check on that column in that call.

Toolchain: Ubuntu 26.04 container, GNU Fortran (Ubuntu 15.2.0-16ubuntu1)
15.2.0, glibc 2.43, `-O0` with WRF's own defines, built by
`tools/urban_bem_regime_oracle/build_u2604.sh` on W1 (2026-10-07).  libm the
`-O0` WRF objects import: `acosf asinf atanf atan2f cosf expf log10f logf
powf sinf tanf __powisf2`; no `_ZGV*`.

Controls recorded in `build-report.txt` (never fixtures):

* `-finit-real=snan -finit-integer=-2147483647`: byte-identical dumps on all
  eight variants, so no uninitialised local reaches a dumped word.  In
  particular v4.6.1's outdoor-temperature loop to `nz_um` (line 1600) reads
  `pt_u`/`pr_u` above `nzu`, which `interpol` never sets; nothing reads the
  `tmp_u` words it writes there.
* v4.7.1's `module_sf_bep_bem.F` (sha256 `42fe129d...`, whose only change
  from v4.6.1 is that loop bound, `nz_um` -> `nzu`): byte-identical dumps on
  all eight variants.  The engine transcribes the v4.7.1 bound.
* The same driver built natively on W1 (GNU Fortran 13.3.0, glibc 2.39)
  differs from these fixtures in 4,996 of 15,315,780 compared words; built
  with `BEM_LIBM_SHIM=1` (atanf, asinf, acosf, tanf and log10f taken as the
  double routine rounded once) it is byte-identical to these fixtures on all
  eight variants.  The glibc 2.39 differences are those five float routines,
  which are not correctly rounded in that C library.

Array names are the dump names with `/` written as `__`, Fortran index
order, the same names and layout as `tests/data/oracles/urban/bem`;
`col__regime`, `col__class`, `col__edge` and `col__morph` label the columns.

Reproduce:

```
bash tools/urban_bem_regime_oracle/fetch_sources.sh WORK/wrf461-src
bash tools/urban_bem_regime_oracle/build_u2604.sh WORK
python tools/urban_bem_regime_oracle/to_npz.py WORK/out-u2604 tests/data/oracles/urban/bem_regime461
```

`tests/test_urban_bem_regime_wrf461.py` holds the claims.
