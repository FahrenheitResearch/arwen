# YSU PBL oracle, WRF v4.6.1

The first WRF number ever produced for `bl_pbl_physics=1`. Before this, the YSU
CUDA kernel had been checked only against `gpuwm.verify.npref.np_ysu_column` --
a float64 mirror of the same transcription -- so a misread line in the
transcription agreed with itself, and the registry called the scheme
`supported`.

## What it builds

```
bash tools/ysu_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
```

`WRF_SOURCE_ROOT` must be at `d66e442fccc04111067e29274c9f9eaccc3cef28` with
`phys/physics_mmm/bl_ysu.F90` and `phys/ccpp_kind_types.F` clean; build.sh
refuses otherwise, so the fixture cannot come from an edited tree.

Two artefacts land in `gpuwm/data/ysu/oracle/`:

| file | contents |
| --- | --- |
| `ysu-levels.csv` | 24 columns x 40 levels: every per-level input and output, plus `rthblten = ttnp/pi2d` (`module_bl_ysu.F:452`) and the momentum tendencies from the second, `ctopo`-present call |
| `ysu-surface.csv` | per column: the surface-coupling inputs, `hpbl`, `kpbl`, `wstar`, `delta`, both call arms |

`libmvec-report.txt` and `oracle-sha256sums.txt` are the receipts;
`tests/test_ysu_wrf461_parity.py` checks both.

## No stub_wrf.F90, on purpose

The RUC and Noah-MP harnesses need one because their modules `USE`
`module_wrf_error` / `module_model_constants` and `CALL wrf_message` and the
`wrf_dm_bcast_*` shims. `bl_ysu.F90` does neither: its only `USE` is
`ccpp_kind_types` and every `CALL` it makes (`tridin_ysu`, `tridi2n`,
`get_pblh`) is to a procedure in the same module. `nm -u` on the object is the
receipt -- `expf`, `powf`, `tanhf`, `malloc/free/memmove/memset` and
libgfortran, nothing else. `ccpp_kind_types.F` is compiled from the pinned tree
with WRF's own default `-DRWORDSIZE=4` rather than stubbed, because a stub
would be free to get `kind_phys` wrong and `kind_phys` is the precision the
whole reference is measured in.

## libmvec

WRF compiles `phys/` at `-O2 -ftree-vectorize`, where gfortran's vectoriser can
replace scalar `expf`/`powf` with glibc's 4-ULP vector forms and silently move
the reference. Measured on gfortran 13.3.0 / glibc 2.39: **it does not happen
for this module** -- `nm -u` shows no `_ZGV*` at `-O0`, at `-O2
-ftree-vectorize`, or at `-Ofast`, because every transcendental in `bl_ysu.F90`
sits in a loop carrying conditionals the vectoriser gives up on. The oracle is
still built at `-O0`, and build.sh still fails on a `_ZGV*` symbol.

A guard that has never fired proves nothing, so `libmvec_positive_control.F90`
is the same `expf` in a loop the vectoriser *can* take; build.sh fails if that
does **not** emit `_ZGVbN4v_expf`, which would mean the grep is hunting a
symbol this toolchain never produces.

`-Ofast` does one other interesting thing: `cbrtf` appears. That is
`-ffast-math` rewriting `x**h1` (`h1 = 0.33333335`, which rounds to float32
`1/3`) into `cbrt`. `kernels/ysu.cu` uses `cbrtf` unconditionally, so it is
spelling the `-Ofast` form of an expression WRF ships at `-O2`. Measured cost
on this fixture: zero for `wstar`/`delta`, and it moves `exch_h` from 7 to 9
ULP -- i.e. it is not the problem, but it is not free either.

## Fixture coverage

24 columns, chosen to reach branches rather than to look meteorological:

* dry convective (deep and shallow), stable land (weak and strong), near
  neutral;
* `br` at `+0.0`, `-0.0`, `+1.4e-45`, `-1.4e-45`, and the smallest normal
  `1.17549435e-38` -- five probes on one sign compare, because CuPy appends
  `-ftz=true` unconditionally and this project has already lost three days to
  exactly that;
* ocean columns (`xland=2`) driving the `brcr_sbro` Rossby form, one with a
  subnormal `u10` and `-0.0` `v10`;
* `ust=hfx=qfx=0` exactly, and `ust` subnormal with `hfx=+0.0`, `qfx=-0.0`;
* stratocumulus at the PBL top in liquid and in ice, with `rthraten < 0`, to
  reach the v4.6.1 top-down cloud-radiative block; plus `qc` **exactly**
  `0.01e-3` and one float32 step above it, straddling WRF's `.gt.` test (they
  give different `delta`, so the fixture discriminates the boundary);
* in-cloud `imvdif` Richardson above the PBL; strong shear (`Ri < 0`); a very
  stable free atmosphere (`prmax` clamp); `gamcrt`/`gamcrq` saturation; a
  coarse 500 m grid saturating the `rlamdz` 300 m cap; a PBL that fills the
  column (`kpbl = kte`); subnormal and signed-zero moisture;
* both values of `ysu_topdown_pblmix`.

`kpbl` spans 2 to 40 across the fixture and `wstar` 0 to 5.36 m/s.

## Not covered, deliberately

`kpbl == kte` with cloud at `kpbl-1`. WRF would read `thlix(i,k+2)` with
`k+2 == kte+1` (`bl_ysu.F90:846`), one past the end of an array declared
`kts:kte`. `ysu.cu:390` guards it with `kpbl < nz` and skips the block, which
is the defined behaviour; a fixture row there would carry whatever WRF's stack
held, so there is none. The divergence is recorded in the registry warning
instead.

## Measuring the port

```
python tools/ysu_wrf461_oracle/validate_ysu_oracle.py [FIXTURE_DIR]
```

Prints per-field ULP; asserts nothing. `tests/test_ysu_wrf461_parity.py` is the
gate, and its module docstring carries the attribution for every number in it.

## The wide oracle (lane/verify-ysu, 2026-10-07)

The 24 columns above reach branches. The wide oracle adds regimes, grids,
timesteps and more edges, and it is what the strict-build claim stands on.

```
bash tools/ysu_wrf461_oracle/build_wide.sh <WRF_SOURCE_ROOT> <BUILD_DIR> [OUT_DIR]
GPUWM_WRF_EXACT=1 python tools/ysu_wrf461_oracle/compare_wide.py   # strict
python tools/ysu_wrf461_oracle/compare_wide.py                     # default
GPUWM_WRF_EXACT=1 python -m pytest tests/test_ysu_wrf461_wide_parity.py
```

* `make_wide_columns.py` writes 131 seeded synthetic columns: convective
  land (16), stable night land (14), ocean (14), high terrain (8) and high
  terrain with `topo_wind` coefficients (10), snow and ice (12),
  stratocumulus under top-down mixing (10), tropical moist (8), and 39 edge
  probes. Grids are WRF-like stretched columns of 30 to 60 levels plus 4 and
  120; dt runs from 1 to 300 s. `--sweep N SEED` draws N more columns from
  the regime builders alone (the 20,000-column sweep is `--sweep 20000 7001`).
* `run_bl_ysu_wide.F90` calls `bl_ysu_run` once per column with the
  argument set `module_bl_ysu.F` passes in a WRF run: `uox`/`vox` and
  `ctopo`/`ctopo2` present, the BEP arrays present with `flag_bep` false,
  `tx` the float32 product `theta*pi` that `phy_prep` forms.
* `build_wide.sh` builds that Fortran twice. **WRF's own flags**
  (`-O2 -ftree-vectorize -funroll-loops`, `arch/configure.defaults`) are the
  reference, because that is the arithmetic a WRF run performs. `-O0` is
  kept beside it because the two differ: GCC folds `pow(x,2.0)` to `x*x` at
  `-O1` and above whatever the math flags, and `powf` is not correctly
  rounded at exact ties, so the five `**2.` sites (`:930`, `:949` twice,
  `:960-961`) can round differently (6 words on this fixture). Neither build
  pulls in a libmvec symbol or an FMA instruction; the script refuses if
  one does.
* `tanhf_sweep.py` holds `ysu_tanhf` (get_pblh's weight) to the C library's
  `tanhf` over every float in [-0.5, 22]. On glibc 2.39 that `tanhf` is not
  the correctly rounded tanh on 5.3% of those floats, and the kernel
  returns the library's words.

WRF writes NaN on five columns (sfcflg true and `ust**3` below the smallest
normal float, so `prfac2` is 0/0 or Inf/Inf). They are outside the claim and
the kernel returns finite numbers there.
