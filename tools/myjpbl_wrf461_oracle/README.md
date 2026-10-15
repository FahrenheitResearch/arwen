# MYJ PBL column oracle, WRF v4.6.1

The first WRF numbers for `bl_pbl_physics = 2`. It compiles WRF's own
`phys/module_bl_myjpbl.F` (and `share/module_model_constants.F`) byte-unmodified
at gfortran `-O0`, calls WRF's `MYJPBL` on generated columns, and writes every
input and every output so the CUDA kernel can be graded word for word.

## Build and run (CPU only)

```
bash tools/myjpbl_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [OUT.npz]
```

`WRF_SOURCE_ROOT` must be wrf-model/WRF at `d66e442fccc04111067e29274c9f9eaccc3cef28`
(tag v4.6.1) with the two files clean; the script refuses otherwise. On W1 that
tree is `/work/pverify/wrf-build/src/WRF-4.6.1`. The script needs gfortran,
gcc's `nm`/`gcov` and a Python 3 with NumPy. It takes under a minute.

What it does, in order:

1. Compiles the two WRF files and `run_myjpbl.F90` at `-O0` and links `run_myjpbl`.
2. Receipts: the `-O0` object's undefined symbols (`expf`, `powf`, `malloc`,
   `free`, libgfortran; no `_ZGV*` vector-math symbol, with a positive control
   proving the grep can see one), and the same for WRF's own
   `-O2 -ftree-vectorize -funroll-loops` object.
3. `make_columns.py` writes the five input cases.
4. `run_myjpbl` runs each case for three consecutive calls; `pack_fixture.py`
   packs everything into one `.npz`.
5. Receipt: WRF's `-O2 -ftree-vectorize -funroll-loops` build writes
   byte-identical outputs (`o2-equality.txt`).
6. Receipt: a `--coverage` build (same bytes out) and gcov list every line of
   `module_bl_myjpbl.F` the fixture never executes (`coverage-myjpbl.txt`).
   Inside MYJPBL's call tree only PRODQ2's `ELOQN <= EPS1` arm (lines
   1145-1146) is never reached; the rest of the list is `VDIFX` (MYJPBL never
   calls it) and `MYJPBLINIT`.

The committed fixture is `tests/data/oracles/myjpbl/myjpbl-wrf461.npz`, with
its receipts beside it. The build is deterministic: a rebuild on W1 reproduced
the fixture's SHA-256.

## The columns

176 columns, three calls each. Between calls the driver carries WRF's INOUT
state (TKE_MYJ, EXCH_H, QSFC, THZ0, QZ0, CT) and advances U, V, TH, QV, QCW, QCI
by the returned tendencies times DT, so calls 2 and 3 see evolved columns.

| case | columns | levels | DT | QCI |
|---|---|---|---|---|
| main_ice | 72 | 50 (to 20 km) | 30 s | present |
| main_noice | the same 72 | 50 | 30 s | absent (three species, no RQIBLTEN) |
| nz4 | 8 | 4 (the kernel's floor) | 60 s | present |
| nz128 | 8 | 128 (the kernel's ceiling) | 10 s | present |
| coarse35 | 16 | 35 | 90 s | present |

The main columns: 10 daytime convective land, 10 stable night land (half with a
low-level jet), 6 near-neutral windy, 5 warm and 5 cold ocean, 4 snow-covered,
3 sea ice, 4 stratocumulus, 3 mixed-phase, 2 cirrus, 3 deep moist tropical,
terrain from -25 m to 4500 m (PSFC down to about 57 kPa), and the edge columns:
CT = 0.4, -0.25 and 1.5 where DTH changes sign (MIXLEN's CT fix), CHKLOWQ = 0
(QFC1 = 0 keeps QSFC), ELFLX < 0 and ELFLX = 0, USTAR = 0 and 2.5 m/s, a calm
isothermal column (GM at its floor), an exactly neutral dry column (GH set to
EPSGH), TKE at its floor everywhere, TKE large everywhere (LPBL = 1, KPBL = nz),
a completely dry column, subnormal and signed-zero condensate and winds, a wind
reversing with height, and a 2 m lowest layer.

## Measuring the kernel (GPU)

```
GPUWM_WRF_EXACT=1 python tools/myjpbl_wrf461_oracle/compare_myjpbl.py
python tools/myjpbl_wrf461_oracle/compare_myjpbl.py
GPUWM_WRF_EXACT=1 python tools/myjpbl_wrf461_oracle/libm_sweep.py
```

`compare_myjpbl.py` prints, per output field, the maximum ULP, the number of
words whose bit patterns differ and the maximum absolute difference, for
replay (each call fed WRF's inputs) and free run (the kernel's own state
carried). The gate is `tests/test_myjpbl_wrf461_parity.py`. `libm_sweep.py`
compares the kernel's `gfk_exp_fma` and `gfk_pow(x, CAPA)` with the host C
library over all 2**32 float32 inputs.
