# Milbrandt-Yau (mp_physics = 9) WRF v4.6.1 column oracle

Runs WRF v4.6.1's own 3-D wrapper `mp_milbrandt2mom_driver` from the pinned,
byte-unmodified `phys/module_mp_milbrandt2mom.F` (WRF commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`) and WOOF's shipped launcher
`gpuwm.core.milbrandt2.launch_milbrandt2` on the same float32 columns, and
compares every output word: qv, qc, qr, qi, qs, qg, qh, the six number
moments, theta, Zet, RAINNC/RAINNCV, SNOWNC/SNOWNCV, GRPLNC/GRPLNCV,
HAILNC/HAILNCV and SR, after every one of several successive calls.

## What WRF side is run

`build.sh` preprocesses the scheme with WRF's own CPP line and compiles it
with WRF's own GNU flags (`-O2 -ftree-vectorize -funroll-loops -w
-ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4
-fallow-argument-mismatch -fallow-invalid-boz`). Given a built WRF tree it
refuses unless its preprocessed source and its object are byte-identical to
that build's `phys/module_mp_milbrandt2mom.f90` and `.o`, and it links a
second runner against the build's own object. It refuses an object that
references a libmvec (`_ZGV*`) entry point or contains FMA instructions.
The only stub is an empty `module_wrf_error` (the scheme USEs it and calls
nothing in it).

An `-O0` build is also made and is NOT the reference: at `-O0` gfortran
calls libgcc's `__powisf2` for integer powers and `powf` for `**2.`, which
WRF's `-O2` object does not, so the two differ.

## How WOOF is run

`woof_run.py` calls `launch_milbrandt2` with exactly the two wrapper
statements `gpuwm.core.milbrandt2.apply` executes around it
(`temperature = theta * pii`, then `theta = temperature / pii`) and WRF's
`p8w(i,kms,j)` as the surface pressure. `GPUWM_WRF_EXACT=1` selects the
strict build; leaving it unset is the default build.

## Rerun (verification box, Linux, gfortran 13.3, glibc 2.39)

```bash
T=<tree>; source <env.sh that puts $T on PYTHONPATH>
H=$T/tools/milbrandt2_wrf461_oracle
$H/run_fortran.sh <WRF-4.6.1 source> <workdir> [<built WRF tree>]   # CPU
<gpu mutex> $H/run_woof.sh <workdir>                                 # GPU
cat <workdir>/summary.json
```

`summary.json` lists, per input set and per run (`strict`, `default`,
`strict_layout4`), the differing words, max ULP (total order) and max
absolute difference for every field.

## Diagnosis tools (not the oracle)

- `instrument.py` + `my2dump.F90`: a copy of the scheme with dump calls at
  its own `check_values` checkpoints (end of Parts 1-4, each sedimentation
  category, the constant SAVE block). Its final output must equal the
  oracle's, which shows the calls move no arithmetic.
- `probe.py`: runs one call of the launcher kernel by kernel (the kernels
  split at WRF's part boundaries) and compares each part and the 154-entry
  constant vector with the dump.
- `sweep_log10.py`: `gfk_log10` against the host's `log10f` over every
  positive finite float (run with `GPUWM_WRF_EXACT=1`).

## Result (2026-10-07, RTX PRO 6000, NVRTC 12.9)

0 differing words in every field of every call, for all four sets
(603 distinct columns), under the strict build and the default build, and
on a (4, ncol/4) grid layout. The committed subset is
`tests/data/oracles/milbrandt2`, graded by
`tests/test_milbrandt2_wrf461_exact_gpu.py`.
