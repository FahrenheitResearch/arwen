# WRF solar generations

`solar_generation = "wrf_480"` is the generic default. `wrf_461` preserves
the earlier solar consumer arithmetic for reproduction. The corrected path
uses the shared float32 device sun, WOOF math, explicit single-rounded
operations and the shared no-FTZ loader. No atmospheric radiation scheme,
table, interpolation fit or surface prognostic equation changes here.

WRF source pins:

| Release | Commit |
| --- | --- |
| v4.6.1 | d66e442fccc04111067e29274c9f9eaccc3cef28 |
| v4.7.0 | eeacb821626feb34c62c3c1e169594d886d13adb |
| v4.7.1 | f52c197ed39d12e087d02c50f412d90d418f6186 |
| v4.8.0 | 06d4240ae989cc3e50af412bb472df3d9048783c |

Official source: https://github.com/wrf-model/WRF/blob/v4.8.0/phys/module_radiation_driver.F

The census identifies two executable hunks in `calc_coszen`:

| Hunk ID | Source | Disposition |
| --- | --- | --- |
| 9dfea7131715a9e30ece1da10de0085181a8d716a2bf1aa5f3e2591c4d60a0e2 | 4.7.1 to 4.8.0, module_radiation_driver.F:3527 | Corrected generation uses julian minus 0.5; legacy retains minus 1 |
| 8bb0803cb131da8960f51f326a992fc5179fb38e9ef0e4a29f6e74d9eb1cc170 | 4.7.1 to 4.8.0, module_radiation_driver.F:3529 | Corrected generation uses 0.040849; legacy retains 0.04089 |
| 830b0f130d096a5d24b3774e3ec31956104073e863086071ad32b7ef2cb05aba | module_ra_rrtmg_swf.F:11692 | Comment only; the scheme consumes the driver's solar outputs |

The endpoint 4.6.1 to 4.8.0 comparison repeats these changes. Neither solar
routine adds a Registry state field, a table dependency, a tendency or an
output array. `DECLIN`, `SOLCON`, `COSZEN` and `HRANG` remain the outputs.
The radiation wrapper still shifts XTIME by the radiation midpoint and
keeps JULIAN at call time. Per-step interpolation has no midpoint offset.
The corrected radiation-free Noah-MP carrier advances the call calendar
at midnight; the legacy branch retains its earlier clock behavior.

Consumers are legacy RRTMG, Dudhia, RRTMGP, solar interpolation, terrain
radiation, solar albedo, urban, Noah-MP and the geometric COSZEN carriers
of analytic and longwave-only radiation. Analytic flux retains its declared
NOAA approximation. Chemistry's separate orbital implementation is outside
the WRF radiation driver's equation-of-time change.

The stock WRF namelist has no spelling for this implementation selector.
Ordinary import therefore resolves the corrected generic default. Both
generations round-trip through the existing physics-selector comment table
and `[shared]` configuration. The selector is global across a domain tree.
Restart echoes omit only `wrf_461`, which an older header without this key
already means. Corrected headers bind the explicit selector and the solar
algorithm identity. A generation change refuses the checkpoint.

The referee extracts unmodified `radconst` and `calc_coszen` from each
pinned release, with an empty `module_wrf_error` stub. Those routines use no
error-module state. GNU Fortran uses REAL(4), `-O0 -ffp-contract=off
-fcheck=all -finit-real=snan -ffree-line-length-none`. Every solar word is
graded over ten dates and 4,370 points per date. The fixture build receipts
pin source hashes, compiler, flags, kinds and dependencies. This is a solar
routine referee, not a full WRF 4.8 radiation-scheme or forecast claim.

Reproduce on a box:

```
python -m tools.solar_geometry_oracle.prepare_versions /work/<lane>/sources
python -m pytest -q tests/test_solar_generation.py
# Within a GPU queue hold, once for default and once with GPUWM_WRF_EXACT=1:
python -m tools.solar_geometry_oracle.compare_versions tests/data/oracles/solar_generations /work/<lane>/solar.json
python -m pytest -q tests/test_solar_generation_gpu.py
```

The one-bit negative control flips mantissa bit 20 of the corrected sine
coefficient in a private image. Production source stays intact. The lowest
coefficient bit was also tried and rounded away before the output clock;
it is not claimed as a successful negative control.
