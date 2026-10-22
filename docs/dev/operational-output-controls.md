# Operational output controls

Authority: NOAA-EMC/HRRR commit
`40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827`, under
`sorc/hrrr_wrfarw.fd/WRFV3.9/`. This is the operational fork, not stock
WRF 4.6.1. Its diagnostic module is **phys/module_diag_gsd.F**;
`phys/module_diag_nwp.F` does not exist at this revision.

## Cadence and precipitation

`history_interval2_s` and `history_interval_change_s` are domain output
controls; a nonzero change requires the second interval. The packaged recipe writes every 900 seconds
through 64800 seconds, then every 3600 seconds. A 48-hour schedule has 103
frames including initialization. Both fixed and adaptive clocks carry the
change, as do native-namelist import/export and prepared history inventories.
Disk admission counts the two alarm phases algebraically, including the
exclusive resume boundary, instead of pricing every frame at the first interval.

`prec_acc_dt` and `prec_acc_dt1` retain WRF's units of minutes (60 and 15).
`module_diag_gsd.F:724-813` resets the corresponding bucket at step-start
`MOD(curr_secs2, interval*60)==0`, then adds the current step. History reads
never reset it. Each interval writes `PREC_ACC_C`, `PREC_ACC_NC`,
`SNOW_ACC_NC`, `GRAUP_ACC_NC`, with suffix `1` for the second interval.
Snow includes convective rain when T2 is below 273.15 K, exactly as the fork.
These are GPU accumulators and restart/streaming carriers.

## GSD inventory and reset authority

Registry.EM_COMMON:2992's `gsd_diags_rt` package contains 13 fields:
`WSPD10MAX`, `W_UP_MAX`, `W_DN_MAX`, `UP_HELI_MAX`, `W_MEAN`,
`GRPL_MAX`, `HAIL_MAXK1`, `HAIL_MAX2D`, `LWP`, `IWP`, `WSPD10`,
`WSPD80`, `MAXCLDFRA`. The module also fills the misc Registry fields
below. They are carried in history, including the fork's lowercase
vorticity variable names.

| Fields | Producer and meaning |
| --- | --- |
| WSPD10MAX, WSPD10UMAX, WSPD10VMAX | Per-step speed maximum and its U/V components |
| WSPD80MAX, WSPD80UMAX, WSPD80VMAX | Declared and reset in the authority, never assigned a nonzero value there |
| W_UP_MAX, W_DN_MAX | Per-step extrema below 100 hPa |
| W_MEAN | Time mean of column-mean W between sigma 0.5 and 0.8 |
| GRPL_MAX | Maximum column integrated graupel, kg m-2 |
| HAIL_MAXK1, HAIL_MAX2D | Thompson fork's 50-bin PSD tail diameter at the first mass level / column maximum, metres |
| REFD_MAX, REFDM10C_MAX | Derived reflectivity maximum at the first mass level / highest -10 C crossing |
| LTG1_MAX, LTG2_MAX, LTG3_MAX | Lightning threats; LTG3 scales by whole-domain LTG1/LTG2 maxima |
| UP_HELI_MAX/MIN, UH | 2-5 km AGL helicity extrema / instantaneous integral |
| UP_HELI_MAX16/MIN16, UH16 | 1-6 km AGL |
| UP_HELI_MAX02/MIN02, UH02 | 0-2 km AGL |
| UP_HELI_MAX03/MIN03, UH03 | 0-3 km AGL |
| rel_vort, rel_vort_max, rel_vort_max01 | Instantaneous 3-D vorticity and fork's 0-2 / 0-1 km extrema |
| WSPD10, WSPD80 | Model-level wind interpolation, filled on the first history-due call after the hourly sentinel reset |
| LWP, IWP, MAXCLDFRA | History-time cloud-water paths and boundary-layer cloud fraction |
| COMPOSITE_REFL_10CM, REFL_10CM_1KM, REFL_10CM_4KM | History-time reflectivity with native convective contribution, base-geopotential height selection and -10 dBZ floor |

`module_diag_gsd.F:817-874` uses `reset_interval1`, default **60 minutes**,
independent of history. It resets before folding the current step; its
position after first-RK helicity means the boundary step's helicity sample
is discarded. `module_diffusion_em.F:6068-6816` owns the four helicity bands
and vorticity, sampled before the dynamics solve. The stock completed-step
UH evaluation remains separate for moving-nest tracking; GSD output does
not change those consumers. All column math, stencil work and reductions
run on the GPU. Restart carries the interval state. Streamed history
recomputes LTG3 after full-domain carriers are assembled, avoiding tile-local
scaling. `tests/test_gsd_output.py` covers these boundaries and inertness.

The fork has an unusual vorticity overwrite: every vertical level compares
against the previous step's 0-2 km maximum, also for the 0-1 km field.
This implementation preserves that comparison rather than replacing it
with a mathematically different vertical reduction.
The previous-value selector also preserves `cal_helicity`'s integer
`MOD(itimestep-1, INT(reset_interval1*60/dt))`, including changes of dt.
Its GPU step counter survives restart and is not cleared by the separate
clock-hour reset in `module_diag_gsd`.

## What diag_int and wind_int actually gate

`diag_int=15` and `wind_int=5` do **not** gate GSD severe-weather maxima.
They gate `phys/module_diag_cl.F::clwrf_output_calc`, called under
`output_diagnostics=1` in `module_diagnostics_driver.F:1095`.
The climate-statistics module tests these periods at lines 171 and 292.
ArWen has no `clwrf_output_calc` port; the importer explicitly records that
missing output capability rather than substituting a GSD reset cadence.

## Evidence and qualification limits

`tools/gsd_output_oracle/extract.py` extracts the unmodified Thompson hail
expressions, scalarizing array indices only. Frozen gfortran outputs and
binary64 bin words are under `tests/data/oracles/gsd_output`. Seven hail
probes match float32 output words exactly. Other tests use manufactured
columns, metric helicity, reset boundaries, restart and real GPU model steps;
they do not establish full operational WRF forecast or complete-file parity.
The output kernels join preflight memory pricing and the parallel static
compile plan. Every exported entry measured zero local frame through the
production loaders on RTX 3080 / sm_86 / NVRTC 13.0.48; the recording names
that platform rather than claiming measurements on other architectures.

The separate `diag_print=1` DPSDT/DMUDT console-budget path has no history
flag in Registry.EM_COMMON:1922-1923 and is not implemented here. The
restart-only 3-D `rlv`/`rlvn` extrema are not emitted history variables and
are not native restart interchange carriers. The Registry also declares
NCI/NCA convection-initiation counters and QR_MAX_CI/QG_MAX_CI, but the pinned
GSD producer does not update them. This lane does not invent values for them.
GSD hail tables for microphysics other than 0/8/28 are explicitly refused.

The native t=0 reflectivity initialization and a full native output-file
comparison remain outside the focused evidence. The precipitation block's
additional runoff resets and ACSNOW/ACGRAUP/ACFRAIN accounting are not claimed
as part of the eight interval buckets; their physical land-surface carriers
need a separate units/ownership audit before being changed.

## Full-grid history cost

The resolved 1799 x 1059 x 50 recipe has 306 planned history fields and
**17,253,740,324 bytes (16.069 GiB)** per uncompressed full frame, including
the planner's bounded header allowance. At 48 hours, 103 frames total
1.616 TiB, compared with 49 hourly frames (before any data-dependent
compression). This is an inventory calculation; it allocates no full grid.

`gpuwm/io/wrfout.py::AsyncDomainWrfoutWriter` already has a side CUDA stream,
pinned D2H buffers, a worker thread and the Rust NetCDF writer. The worker
waits for D2H and performs file output off the stepping thread. However,
snapshot capture has a producer-stream event dependency, the bounded queue
can apply backpressure, and borrowed host-store frames must be drained
before their arrays are mutated. Therefore **fully off-step-path, full-CONUS
throughput is not established** by this lane. The RTX 3080 cannot hold this
resident domain plus snapshots. A larger-card / streamed full-grid timing
receipt is needed to qualify sustained 15-minute output; no such run was
performed on the desktop.
