# meeting-perf-a13 box kit (open-defects ledger A13)

Scripts that measured full HRRR (1797x1057x50, HRRR physics) on 4 x RTX 5090 on box E, 2026-10-05/06.

- `mkcase.py`, `hrrr-door-c24s.toml`, `stageE.sh`, `envE.sh`: author and prepare the 2026-10-03 21Z, 3 h case on the box CPU.
- `price22.py`, `breakdown.py`: CPU admission price per card and per term for [devices] 2x2, 1x4, 2x4.
- `gpuE.sh`, `job3.sh`, `gpu3.sh`: the 4-card runs; `job3.sh` holds a lock so a second launch exits, and every arm writes a
  new directory. `gpuE.sh` is the 2026-10-06 measurement as it ran (a scratch copy of the engine with the card refusal
  disabled, `engine-meas`, which never entered the tree). `gpu3.sh`'s tip arms run the lane engine under its own default
  admission (the rank pool byte bound admits 2x2 on four 32 GB cards); only its base arms (534c6dea3, which refuses the
  layout) still use a measurement copy.
- `cmp.py`: field-by-field comparison of two history directories (wrfout netCDF only; the `ready/*.json` markers carry
  the output path and always differ).
- `hrrrplot.py`, `plot_r2.py`, `hexplot.py`, `rrfsfit.py`: plots and the hex limited-area fit.
