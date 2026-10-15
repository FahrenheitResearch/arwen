# WRF v4.6.1 Ferrier-Aligo column oracle (mp_physics = 5)

Reference outputs of WRF's own `phys/module_mp_fer_hires.F` (subroutine
`FER_HIRES`, package `fer_mp_hires`, moist `qv, qc, qr, qi`, scalar `qt`,
state `f_ice_phy, f_rain_phy, f_rimef_phy`), compiled byte-unmodified from
WRF tag v4.6.1 (commit `d66e442fccc04111067e29274c9f9eaccc3cef28`) with a
stub framework and a one-point column driver.

WOOF dispatches mp_physics=5 to gpuwm.core.fer_hires. Strict mode
GPUWM_WRF_EXACT=1 is graded against the stock WRF module, including its
Rcw/DUM2 reuse. The 72 original, 65 refute and 22 coverage columns are
graded by check_port.py. A missing receipt fails that check.

## Files

| file | role |
|---|---|
| `make_columns.py` | Deterministic column generator (closed-form soundings, no RNG). Writes `columns_inputs.csv` (canonical, binary32 hex) and `columns.in` (driver format). |
| `run_ferrier.F90` | Driver. Per column: `fer_hires_init` at that column's time step (GSMDT = 0, so DTPH = DT, as in a WRF run), then `FER_HIRES` for 10 steps with the state carried. Writes `levels.csv` and `surface.csv` for steps 1 and 10. |
| `stub_wrf.F90` | `wrf_error_fatal`, `wrf_debug`, `wrf_dm_on_monitor` (true), `wrf_dm_bcast_bytes` (no-op). |
| `build.sh` | Checks the pinned commit and that the module and `run/ETAMPNEW_DATA.expanded_rain` are unedited, builds a `-O0` reference and a WRF-flags (`-O2 -ftree-vectorize -funroll-loops`) twin, runs both, writes `SHA256SUMS`, compares the two. |
| `compare_ferrier.py` | Bitwise comparator: per field, rows differing, max ULP, max abs diff, worst column and regime. A port writes the same two CSVs from `columns_inputs.csv` and is graded with this. |

Committed fixtures: `tests/data/oracles/ferrier_hires/` (`columns_inputs.csv`,
`levels.csv`, `surface.csv`, `SHA256SUMS`). Test: `tests/test_ferrier_mp5_oracle.py`.

## Columns

72 columns x 50 levels, 8 per regime: convective, stable_night, snow, ocean,
high_terrain, freezing_rain, cold_anvil, edge, mixed. Domain id alternates
1 and 2 (the outer domain uses RHgrd_out = 0.975 above 850 hPa), time step
cycles 6, 20, 36, 72 s. The edge columns cover: bone dry; condensate either
side of EPSQ and EPSQ1; ice in warm air with rime factors 30 to 49.5;
temperatures exactly 0 C and -40 C; 15 g/kg rain with 5 g/kg cloud; 20 %
water supersaturation; 181 K aloft (just inside the 180 K table floor);
rime factor below 1 on input with inconsistent F_ICE / F_RAIN.

## Rerun (W1, CPU only)

```bash
bash tools/ferrier_hires_wrf461_oracle/build.sh \
    /work/pverify/wrf-build/src/WRF-4.6.1 /work/pverify/ferrier-eta-mp5/out
sha256sum out/O0/levels.csv   # equals tests/data/oracles/ferrier_hires/SHA256SUMS
```

Toolchain of record: gfortran 13.3.0, glibc 2.39 (Ubuntu 24.04). The `-O0`
object must not import any libmvec `_ZGV*` symbol (the build refuses if it
does); it imports scalar `expf`, `logf`, `log10f`, `powf` and `lroundf`.

## What the reference showed

The `-O0` and WRF-flags builds wrote bitwise-identical `levels.csv` and
`surface.csv` on W1: the scheme's own Fortran has no flag-dependent
arithmetic on this column set.

Two facts a port must reproduce rather than "fix" without evidence:

- `fer_hires_init` sets `CBFR` (Biggs freezing) from DTPH, and `FER_HIRES`
  does not recompute it, while it does recompute `CIACW`, `CIACR`, `CRACW`,
  `BRAUT` and `MY_GROWTH` from DT every call. In WRF init and step use the
  same dt, so this is consistent; a port must still take CBFR from the init
  dt.
- `FER_HIRES` overwrites the init value of `CIACW` (`0.5*DTPH*0.25*PI_E`)
  with `DT*0.25*PI_E*0.5*(1.E5)**C1`, which is the value every step uses.
