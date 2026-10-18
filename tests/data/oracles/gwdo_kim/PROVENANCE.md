# gwd_opt = 1 (KIM orographic drag) WRF v4.6.1 column oracle

Written by `tools/gwdo_kim_wrf461_oracle/build.sh` and `compare_gpu.py
--write-fixture` on box W1 (Ubuntu 24.04), 2026-10-07.

- WRF source: github.com/wrf-model/WRF tag v4.6.1, commit
  `d66e442fccc04111067e29274c9f9eaccc3cef28`, byte-unmodified (build.sh
  refuses another commit or any byte that differs from `SOURCES.sha256`).
  - `phys/physics_mmm/bl_gwdo.F90` sha256 `c294ef2eb9da1ee9ce6594b97adea7280eb18760e0c5576ab7149c35bb7c3c3b`
  - `phys/module_bl_gwdo.F` sha256 `d782ddcda578a1fde0cd74e6526bf14f27cb0fb48113c227ad971717e6ca90cc`
  - `phys/ccpp_kind_types.F` sha256 `a76e1e5b52fc7cd40be9ccb506fde28b1ef2486b812d518f7f1c882766d484db`
  - `share/module_model_constants.F` sha256 `5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062`
  - The two scheme files are byte-identical to WRF v4.7.1's, and so is the
    `gwdo` call in `module_pbl_driver.F`.
- Compiler: GNU Fortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), glibc 2.39.
- Reference of record: `-O0 -ffp-contract=off`. The scheme object imports
  scalar `atan2f` and `powf` only (no `_ZGV*` vector symbol) and contains no
  FMA instruction; build.sh fails otherwise. WRF's stock GNU physics flags
  (`-O2 -ftree-vectorize -funroll-loops`) write byte-identical outputs on
  both grids.
- Driver: `run_gwdo_kim.F90` calls SUBROUTINE gwdo with WRF's own
  `module_model_constants` CP, G, R_D, R_V, EP_1, `PI=3.141592653`, DT and
  DX, exactly as `module_pbl_driver.F:2136-2151` passes them.

## Files

| file | sha256 | content |
|---|---|---|
| `gwdo-kim-inputs-L50.npz` | `a30bba9ae5737bd0e4f76e405bac314dc3fa5726dc8a306473574544eba517f4` | float32 inputs, byte-equal to `gwdo-kim-inputs-L50.bin` (sha256 `2c427484...`) |
| `gwdo-kim-inputs-L90.npz` | `7599931644f46b9e8f5b52eed99f3486773692d7335a1fcb71a39b05a03fee49` | float32 inputs, byte-equal to `gwdo-kim-inputs-L90.bin` (sha256 `a76562b0...`) |
| `gwdo-kim-wrf461-L50.npz` | `64aafab05aa280cd2c870fae9acc977061a98c42456179fe85c1405986f32598` | WRF's outputs, byte-equal to `gwdo-kim-wrf-O0-L50.bin` (sha256 `da864431...`) |
| `gwdo-kim-wrf461-L90.npz` | `8a5033c4259b8fc929b6ee342de12779c401a2dac773d9e44d98efc105f3a66d` | WRF's outputs, byte-equal to `gwdo-kim-wrf-O0-L90.bin` (sha256 `9692ce32...`) |
| `sector-sample.npz` | `4ea4eb00dc9d19ef078915785886823a4b7bf30dd4363c33b31ae0f559b77a64` | 131,072 (ubar, vbar) pairs with the host's `atan2f` words and wind sectors (`sector_sweep.py --write-sample`) |

## Coverage

149 columns on two WRF-like eta grids (50 and 90 levels, p_top 50 hPa), 12
regimes: convective, stable_night, neutral_windy, blocking, high_terrain
(surface 2.5 to 4.6 km), ocean (statistics masked to 0, two islands),
cold_snow (232 to 262 K surface, polar map rotation), tropical_moist, jet_shear
(40 to 70 m/s jet), critical_level, sector_boundary (wind on the edge of the
scheme's eight direction sectors) and 17 edge cases (calm, light wind, VAR 0,
VAR 0.5 m, VAR 15 km, OA and OL extremes, both aspect-ratio clamps, CON 0,
Froude cap, two kinds of flow reversal, the velco clamp, 1.2 rad map
rotation, deep blocking, the dtfac limiter, subnormal and signed-zero winds).
Six (dx, dt) cases: 1 km/6 s, 3 km/18 s, 9 km/54 s, 12 km/72 s, 30 km/180 s,
50 km/300 s, each one gwdo call from the same inputs.

Columns WRF actually drags, in every (dx, dt) case: all 12 in blocking,
cold_snow, critical_level, high_terrain, jet_shear, neutral_windy,
sector_boundary and stable_night; tropical_moist 11 (L50) and 10 (L90);
convective 1 (an unstable low layer switches the scheme off); ocean 2 (the
islands); edge 11 of 17 (the other six are the drag-off cases by design).
The columns WRF leaves alone are graded too: their words must stay the PBL
tendencies bit for bit. `branch_mutations.py` shows every one
of 16 single-statement controls in `gwdo_column` (flow blocking, wave
breaking, unbroken carry, dtfac limiter, velco clamp, Froude cap, both
aspect-ratio clamps, efact cap, critical level, in-layer reversal, the kbl
floor, light wind, VAR 0, the low-layer density mean, the atan2 sector) moves
WRF words on this fixture.
