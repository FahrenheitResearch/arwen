# Noah LSM oracle, WRF v4.6.1

Four fixture files, produced by `tools/noah_wrf461_oracle/build.sh` from the
byte-unmodified WRF v4.6.1 tree pinned at
`d66e442fccc04111067e29274c9f9eaccc3cef28`.  `build.sh` refuses to run against
any other commit, and refuses to run if `git diff` reports a change to any of
the five sources it compiles, so no number here can come from an edited tree.

| file | driver switches |
| --- | --- |
| `noah-lsm.csv` | `opt_thcnd=1`, `frpcpn=F`, `usemonalb=F`, `rdlai2d=F` |
| `noah-lsm-thcnd2.csv` | `opt_thcnd=2` (McCumber-Pielke thermal conductivity) |
| `noah-lsm-frpcpn.csv` | `frpcpn=T` (frozen fraction read from `SR`) |
| `noah-lsm-monalb.csv` | `usemonalb=T`, `rdlai2d=T` |

Each is 42 land columns x 109 fields: every input `phys/module_sf_noahdrv.F`'s
`lsm` reads, and every output it writes.  Nothing in them is a hand-computed
expectation -- each value is either a word `run_lsm.F90` placed in an input
array before the call or a word `lsm` placed in an output array during it.
The two derived echoes, `sfcprs` and `zlvl`, are labelled as such in
`run_lsm.F90` and are inputs to gpuwm where WRF forms them from `p8w3d` and
`dz8w`.

The parameter tables are read by WRF's own `SOIL_VEG_GEN_PARM`
(`module_sf_noahdrv.F:1999`) from `run/VEGPARM.TBL`, `run/SOILPARM.TBL` and
`run/GENPARM.TBL` of the same pinned tree, so the fixture also exercises
gpuwm's transcription of that list-directed READ sequence rather than assuming
it.

`libmvec-report.txt` records which libm symbols the reference object actually
carries at `-O0`, at WRF's own `-O2 -ftree-vectorize`, and at `-Ofast`, plus
the positive control proving the `_ZGV*` grep can find a vector symbol on this
toolchain at all.  `oracle-sha256sums.txt` hashes the five WRF sources, the
harness, the three tables and the four fixtures together.

Toolchain: gfortran 13.3.0, glibc 2.39 (Ubuntu 24.04 under WSL2).
