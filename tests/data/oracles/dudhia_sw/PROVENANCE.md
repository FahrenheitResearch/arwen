# Dudhia shortwave (ra_sw_physics=1) WRF v4.6.1 column oracle

Written by `tools/dudhia_sw_wrf461_oracle/build.sh` on box W1 (Ubuntu 24.04),
2026-10-07.

- WRF source: github.com/wrf-model/WRF tag v4.6.1, commit
  `d66e442fccc04111067e29274c9f9eaccc3cef28`, byte-unmodified (build.sh
  refuses another commit or a local edit).
  - `phys/module_ra_sw.F` sha256 `cd7e1c89a962c9b62f042a8f24ee7d8892c0428866f7a44ec36df0d06705a4c0`
  - `share/module_model_constants.F` sha256 `5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062`
- Compiler: GNU Fortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), glibc 2.39.
- Reference of record: `-O0 -ffp-contract=off`. The object imports scalar
  `log10f` and `powf` (and `logf`, on the unused slope-radiation path), no
  `_ZGV*` vector symbol, and contains no FMA instruction; build.sh fails
  otherwise.
- Driver: `run_swrad.F90` calls `swinit` and SUBROUTINE SWRAD with WRF's own
  `module_model_constants` parameters (R = r_d, CP = cp, G = g, as
  `module_radiation_driver.F:2343-2369` passes them), every species flag set,
  no WRF-Chem aerosol, no eclipse, RTHRATEN zeroed on entry. Its one line of
  arithmetic is the driver's own `SWDOWN=GSW/(1.-ALBEDO)`
  (`module_radiation_driver.F:2877`).

## Files

| file | sha256 | content |
|---|---|---|
| `dudhia-inputs.npz` | `0bc3fe0154799fd0ac5bcb1d6895efadec87baeeabaa858c51aafadfbcba2ff0` | the float32 inputs, byte-equal to `dudhia-inputs.bin` (sha256 `ba3e4c1e...`) |
| `dudhia-wrf461.npz` | `030d6da7fa202e63cbdbcfb4843bf8d9dff68994436765a2d80e876bf3b5a596` | RTHRATEN, GSW, SWDOWN for each case, byte-equal to `dudhia-wrf-O0.bin` (sha256 `2a49971b...`) |

## Coverage

82 columns x 50 levels in 28 regimes: clear_midlat (8), deep_convective (10,
liquid, rain, graupel, ice and snow anvil), stable_night (8, every one dark,
COSZEN from -0.9 up to exactly 1e-9), snow_ice_polar (8, albedo 0.75 to 0.9,
low sun), ocean_stratocumulus (8, albedo 0.05 to 0.08), high_terrain (8,
surface 3.0 to 4.8 km), cirrus (6), moist_tropics_clear (6), fourteen
cosine-zenith edge words (exactly 1e-9 and the next float32 above it, 2e-9,
1e-6, 0.01, the XMUVAL words 0.2, 0.5, 1.0 and their float32 neighbours, 0
and a negative), and six more edges (one column with negative words in every
species, vapour included, on every third level; a dry column; albedo 0 and
0.99; two giant water paths).

Four cases (SOLCON, swrad_scat, icloud): (1367.0, 1.0, 1), (1412.3, 0.7, 1),
(1321.8, 1.0, 0), (1360.0, 6.0, 1).

Branches reached (tools/dudhia_sw_wrf461_oracle/coverage.py, host replay,
328 column calls, 14,200 day layers): night return 44, each XMU interval
(60 / 88 / 136 columns), each water-path interval (9,664 / 1,242 / 1,569 /
1,725 layers), ALW capped at 3.999 in 1,119 layers, XABS clamped 815,
XALB clamped 43, XABSC clamped 37, TRANS0 < 1 rescale 1,313, SDOWN floor at
1e-9 1,298.

A randomised stress set of 20,000 columns (`build.sh ... --stress 20000`,
seed 20261007, not committed: 80,000 column calls, 4,000,000 RTHRATEN
words) was graded the same way on 2026-10-07; inputs sha256 `b6b417f4...`,
outputs `9a67657d...`.

## Stock-flags evidence

The same module compiled with WRF's stock GNU physics flags
(`-O2 -ftree-vectorize -funroll-loops`) imports the same scalar functions and
no libmvec symbol, contains no FMA instruction, and its outputs are
byte-identical to the -O0 reference on both the fixture and the stress set.
