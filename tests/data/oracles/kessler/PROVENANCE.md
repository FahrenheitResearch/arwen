# Kessler (mp_physics=1) WRF v4.6.1 column oracle

Written by `tools/kessler_wrf461_oracle/build.sh` on box W1 (Ubuntu 24.04),
2026-10-07.

- WRF source: github.com/wrf-model/WRF tag v4.6.1, commit
  `d66e442fccc04111067e29274c9f9eaccc3cef28`, byte-unmodified (build.sh
  refuses another commit or a local edit).
  - `phys/module_mp_kessler.F` sha256 `56935bc95cc72084a37cdaf1b680a6015806daccc48feaf4b3cc6f167653ebfc`
  - `share/module_model_constants.F` sha256 `5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062`
- Compiler: GNU Fortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), glibc 2.39.
- Reference of record: `-O0 -ffp-contract=off`. The object imports scalar
  `expf` and `powf` only (no `_ZGV*` vector symbol) and contains no FMA
  instruction; build.sh fails otherwise.
- Driver: `run_kessler.F90` calls SUBROUTINE kessler with WRF's own
  `module_model_constants` parameters, exactly as
  `module_microphysics_driver.F:983-996` passes them.

## Files

| file | sha256 | content |
|---|---|---|
| `kessler-inputs.npz` | `2b731f5879fd0d1bfff6c3c00c6eed4456567bc5c8a39f9053a6aadf3c30c33a` | the float32 inputs, byte-equal to `kessler-inputs.bin` (sha256 `80c952c9...`) |
| `kessler-wrf461.npz` | `38b648910d23ee2ee27703737936ecee1f1ae411fcef8bcb52e70fa29a82f800` | WRF's outputs after every call, byte-equal to `kessler-wrf-O0.bin` (sha256 `daf31ebf...`) |

## Coverage

104 columns x 50 levels, 13 regimes of 8 columns: convective, stratiform,
stable_night (inversion and fog), cold_arctic (supercooled water to 200 K
aloft; Kessler has no ice), ocean, high_terrain (surface 2.5 to 4.5 km),
desert_virga, clear_dry, heavy_rain_thin (10 to 25 g/kg on 30 m layers),
saturated_onset, autoconv_threshold (qc on the c2 = 0.001 word and its two
neighbours), edge_values (subnormal and normal-boundary words, small
negatives, signed zeros, qv = 0 under rain, rain only in the top level, rain
only in the lowest level, one 50 g/kg level on quartered layers, constant
density) and random_mix.

Five cases, each restarted from the inputs and run free: dt 5 s x 4, 18 s x 8,
60 s x 6, 120 s x 4, 300 s x 3 (25 calls). The sedimentation split count
reaches 2, 7, 23, 45 and 112 in those cases, with 0, 1, 11, 36 and 230
re-split events (nfall_new differing from nfall).

## Stock-flags evidence (not the reference)

The same module compiled with WRF's stock GNU physics flags
(`-O2 -ftree-vectorize -funroll-loops`) imports `_ZGVbN4vv_powf` and
`_ZGVbN4v_expf` from libmvec, and its outputs differ from the -O0 reference
in 74,869 of 525,200 words (max 52 ULP in qv, 2 ULP in theta). That is WRF
disagreeing with itself across build flags; the scalar -O0 build is the
reference of record.
