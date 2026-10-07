# GSL driver subset oracle

This fixture qualifies only the override and redistribution statements in
physics/smoke_dust/module_plumerise.F90:143-166. It does not execute MAKEPLUME
and does not qualify Freitas plume rise, sounding preparation, column state,
plume top, land-use injection or the process interface.

Source: https://github.com/ufs-community/ccpp-physics/blob/3e6660c6df54e95a0871e990c2294dd397ae3860/physics/smoke_dust/module_plumerise.F90#L143-L166
License: Apache-2.0. The source statements are extracted byte for byte by
extract-gsl-plume.list, with declarations supplied by gsl_run_ebu_distribute.F90.

Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
glibc: Ubuntu GLIBC 2.43-2ubuntu2.4, 2.43.
Flags: -O0 -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none.
Reference machine stub: kind_phys=4. Every compiled public file was checked
against SOURCES.sha256. The -O0 objects import no _ZGV symbols. The positive
control imports _ZGVbN4v_expf, so the libmvec guard is effective.

Compiled source hashes:

```
27c5dc0503decb631f4e057241f35032b9a84b2f84616d8718dcee3c987ec577 rrfs_smoke_config.F90
b11265e926c10fd06e41d2cc84e6dcc0aca8337577eb2054870a6bed280f2b8c module_zero_plumegen_coms.F90
e784083e907aa1660b30db79dcac9dbe0d6a9632de8e672b491bdfdc33bf4396 module_smoke_plumerise.F90
f8c3fd2beb9157cdb5f3b3844981034a47ae5352ce9fa957afa8d0b1d7fa7994 module_plumerise.F90
```

The first three source modules are compiled for the lane's future full-plume
driver, but this subset driver calls none of their routines. There are no
debug calls, undefined icall reads, retained module-state reads or imm passes
in the extracted statements. No state-poison experiment was performed.

Cases: 12 synthetic driver inputs, each with 120 mass levels and nonuniform
w-level spacing. FRP 0, just below/equal/just above 1e7, 1e8, equal/just above
1e9 and 2e10; wind just below/equal its threshold; PBL height equal/above its
threshold; wind option off; shallow/deep kpbl; zero emissions; a 119..120
injection interval. Below-threshold cases isolate the driver's override by
supplying 4,18 as input plume indices. The full scalar's below-threshold 1,2
return is not tested here.

CPU reference versus KIND=4, all cases:

| Output | max ULP | nonzero | words |
| --- | ---: | ---: | ---: |
| k_min (integer mismatch) | 0 | 0 | 12 |
| k_max (integer mismatch) | 0 | 0 | 12 |
| flam_frac | 0 | 0 | 12 |
| ebu | 0 | 0 | 1440 |

KIND=8 output cast by oracle_io to float32 versus KIND=4: injection indices
and flam_frac identical; ebu max ULP 2, 37/1440 different words. This measures
only redistribution. Float64 plume top and index differences are unmeasured.
No GPU test was run and no CUDA compilation was performed.

Commands (the shell runs on Linux, within the lane scratch directory):

```
bash harness/build_gsl_smoke.sh ../aq-smoke/wrf-src build4 4 gsl_run_ebu_distribute
bash harness/build_gsl_smoke.sh ../aq-smoke/wrf-src build8 8 gsl_run_ebu_distribute
```

After copying build4/fixtures/ebu_distribute to this directory and the KIND=8
fixture to .plume-evidence/kind8, from the export root on the CPU host:

```
$env:GPUWM_NO_LOCAL_GPU='1'
$env:PYTHONPATH='.'
python -B tools/chem_wrf471_oracle/measure_ebu_distribute.py --kind8 .plume-evidence/kind8 --mutate
python -B -m pytest -q tests/test_chem_plumerise_frp_parity.py -m 'not gpu'
```

Mutation: .85 -> .84 in the CPU override, restored in finally. The CPU parity
test failed at flam_frac with max ULP 167773. After restoration: five tests
passed, one device test deselected. Fixture binaries and manifests are pinned
by oracle-sha256sums.txt, whose SHA-256 is also asserted by the test.
