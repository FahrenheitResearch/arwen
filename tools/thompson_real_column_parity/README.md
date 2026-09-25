# Thompson mp=28 and mp=8 on real columns: the port against WRF v4.6.1, process by process

This harness runs unmodified WRF v4.6.1 `module_mp_thompson.F` (tag v4.6.1,
commit d66e442f, SHA-256 `fabf19e2...`) and the port's production adapter on
the same float32 model columns, on the CPU, and compares every process rate,
the working state after the source and condensation stages, the final state,
reflectivity and surface precipitation.  Two schemes are covered by the same
module and the same instrumentation:

* mp_physics=28 (the default): WRF called with the aerosol arguments, so
  `thompson_init` sets `is_aerosol_aware` (:480), against
  `gpuwm.core.microphysics_aerosol._apply_thompson_aerosol`;
* mp_physics=8 (`--mp 8`): WRF called as the THOMPSON case of
  `module_microphysics_driver.F` calls it, with no nc, nwfa, nifa or nwfa2d,
  so `is_aerosol_aware` stays false, against
  `gpuwm.core.microphysics._apply_thompson`.  The same column files are
  read; their droplet and aerosol numbers are ignored.  The rates classic
  Thompson does not carry (the droplet and aerosol number rates and Koop
  freezing, `port_rates.NOT_CARRIED_MP8`) are reported with WRF's activity
  only.

No GPU is used anywhere.  The port side is the exact kernel source nvrtc
receives (`gpuwm.core.kernels.module_source`), compiled as C++ for the host.

## Pieces

| file | what it does |
|---|---|
| `run_columns_aero.F90`, `run_columns_classic.F90` | batch column drivers (mp=28 and mp=8): one `thompson_init`, one `mp_gt_driver` over N columns, raw float32 in and out |
| `instrument_wrf_rates.py` | writes a copy of `module_mp_thompson.F` that dumps the sixty-four process rates and the thirteen running tendencies at five points of `mp_thompson`; WRITE statements only |
| `build_wrf.sh` | builds the pristine and the instrumented module and links both drivers against each (`-O2 -fno-tree-vectorize`, no libmvec); every run refuses unless the two give byte-identical outputs |
| `cuda_host_shim.h`, `host_backend.py` | compile each kernel module for the host behind a CUDA language shim, generate one launcher per `__global__`, and install a NumPy-backed `cupy` plus a host `get_kernel`, so the adapter runs unmodified |
| `port_rates.py` | inserts one guarded `HOST_RATE` readback per rate at the one point where each kernel's rate is final, under WRF's names, plus the rain evaporation's post-adjustment `ssatw` as a diagnostic: the three aerosol units for mp=28, the classic kernels of `thompson.cu` for mp=8 |
| `extract_columns.py` | cuts inputs out of a history frame, a restart, or a restart plus a saved analysis increment |
| `real_column_parity.py` | one full comparison: WRF pristine and instrumented, port pristine and instrumented, both neutrality checks, the port's response to a one-unit nudge of every input, then every metric into `summary.json` |
| `report.py` | tables from one or more `summary.json` |
| `make_fixture.py`, `fixture_check.py` | the committed CPU gates: `tests/data/thompson_real_columns_wrf461.npz` (42 columns chosen per process regime from WRF's own checkpoints) and, with `--mp8`, its classic Thompson companion `thompson_real_columns_wrf461_mp8.npz` (the same columns, WRF's classic answers), and their checker |

## Running it

On a Linux box (or WSL) with gfortran, g++, numpy and the staged Thompson
tables (`qr_acr_qg_V4.dat`, `qr_acr_qsV2.dat`, `freezeH2O.dat`,
`CCN_ACTIVATE.BIN`):

```sh
./build_wrf.sh /path/to/WRF-v4.6.1/phys /tmp/wrfbuild ~/.gpuwm/tables/thompson
python3 extract_columns.py cols.npz 5 history wrfout_d02_...        # or restart / analysis
python3 real_column_parity.py cols.npz /tmp/wrfbuild out/            # summary.json
python3 real_column_parity.py cols.npz /tmp/wrfbuild out8/ --mp 8   # classic Thompson
python3 report.py out/summary.json
python3 make_fixture.py /tmp/wrfbuild fixture.npz cols1.npz cols2.npz ...
python3 make_fixture.py --mp8 /tmp/wrfbuild fixture_mp8.npz fixture.npz
python3 fixture_check.py fixture.npz                                 # or fixture_mp8.npz
```

`tests/test_thompson_real_column_host_parity.py` runs the fixture check where
a C++ compiler and the tables exist (on Windows through WSL) and skips,
naming the missing piece, elsewhere.

## How a difference is classified

A cell is compared where either code's value exceeds the quantity's floor
(1e-12 kg/kg for masses, 1 per kg for numbers).  A relative difference at or
below 2e-6 (the port's own end-to-end gate on the 22 committed WRF fixtures)
is rounding.  A cell beyond it still counts as rounding when:

* the port's own response to a one-unit float32 nudge of every input in the
  same cell (largest over four random-sign draws) is at least a quarter of
  the gap, so the answer there is decided by the last bit of the inputs; or
* for the final state, the gap is within four float32 units of the largest
  value the cell held anywhere in the call (a level nearly emptied in one
  step keeps a residual of that size, and the two codes' residuals differ by
  it).

Two rates carry one more named rule each, because rounding decides them:
rain evaporation where the saturation adjustment has just brought the air to
saturation (WRF's post-adjustment `ssatw` is a residual of a few float32
units, and its sign opens the evaporation gate at :3501), and rain
self-collection at the 1950 micron break-up diameter (:2159-2176), whose
relative sensitivity to the rain mean diameter is `kappa`.

## What the host build does and does not see

The host evaluates the transcription with IEEE per-operation rounding, no
fused multiply-add and glibc's libm: the same arithmetic the gfortran oracle
uses.  It does not see the device toolchain's differences (nvrtc's default
multiply-add contraction of unpinned expressions, CUDA's own transcendental
functions, sm_120's FP32 subnormal flush).  Those belong to the GPU gates.
On the 22 committed single-column fixtures the host build reproduces the
device's published G3 residuals.

## Findings

Measured on 2026-09-23 on seven saved real-data states of one convective
case (five forecast frames and two analysis states, 137,200 columns of 49
levels).  Twelve differences in the port were found and repaired in the same
change set as this harness; each kernel comment cites its WRF lines.  After
them every one of the 64 process rates agrees with WRF to float32 rounding
except the two rounding decides (rain evaporation just after the adjustment
saturates the air, and rain self-collection at the 1950 micron break-up
diameter), reflectivity is within 0.009 dB on the forecast frames and 0.024
dB on the analysis states, and surface rain within 3.6e-7 relative.

The five WRF rules the port still did not follow after those twelve were
repaired the same day, default-on in mp=28 and mp=8: G, the no-microphysics column exit and the
terminal vapour floor (:2020, :3974); T, graupel at or below R1 written as
zero (:4058-4063); A-warm, the ice mass/number balance above 0 C
(:3033-3055); J, cloud and ice at or below R1 melted or frozen before the
terminal apply removes them (:3943-3966); I, melting snow blended with the
rain pass's own fall speed (:3612-3634, :3722-3724).

Classic Thompson (``--mp 8``) was then held to the same Fortran and its
kernels and adapter given the rules the mp=28 units carry: K, the entry
rewrite; L, the cloud fallout gate; A, the 5 micron entry ice; E, the D0i
minimum crystal mass; C and F, rain and graupel emptied at the source stage
and the private graupel number balance; O, the graupel sublimation number
gate; H, the terminal ice bound in its per-kilogram form; P, presence tested
on the mixing ratio, not the concentration; R, the rain fallout's L_qr and
:3568 rewrite.  Before them mp=8 differed from WRF beyond 1e-2, unexplained
by rounding, at 89,680 process-rate cells and 159,338 final-state cells of
the seven frames, and its echo by up to 43.9 dB; after them no rate differs
beyond 1e-2 except where rounding decides, 7 final-state cells do (all
rounding residues, below), and the echo is within 0.045 dB.

Named differences that remain, all rounding's, in both schemes:

| | WRF | the port | size, seven frames |
|---|---|---|---|
| N | rounding | graupel number re-balanced on a graupel mass consumed to a one-unit residue | 2 cells (mp=28), 1 (mp=8) |
| V | rounding | a complete cloud evaporation leaves a residue one float32 unit different from WRF's (WRF forms it as qc1d + DT*qcten, the port as qc0 - rc/rho).  Where WRF's residue times rho passes R1 its L_qc stays set, so that column's low cloud falls in WRF and stays in the port; where ice melts onto the residue the sum crosses R1 in WRF only; where the residue itself passes R1, WRF keeps it and a few droplets per kg | analysis states only: 4 cloud cells beyond 1e-2 in both schemes (2 per state: levels fed by a fallout the residue left open, in mp=8 one melt level), 184 droplet-number cells (mp=28, almost all on residue cloud of 1.8e-12 to 7.3e-12 kg/kg) |
| W | rounding | the sources leave rain as a one-unit residue of a much larger entry rain, at or below R1 in one code and above it in the other, so only one evaporates it | analysis states only (mp=8): one level of each state, its rain number 2.9 percent off |

And one of arithmetic order, mp=8 only: the classic autoconversion forms
Berry-Reinhardt's Dc_b as a cancellation in its own operation order, so its
rates carry up to 9.2e-4 relative from WRF's (never beyond 1e-2); the mp=28
units follow WRF's order.

And one rule the classic port does not carry and cannot show: cloud frozen
below HGFR hands the ice WRF's running droplet number, nc1d + ncten*DT
(:3959), where the classic phase cleanup adds Nt_c/rho (``thompson.cu``,
the note in ``thompson_aerosol_sed.cu``), because classic Thompson keeps no
droplet-number tendency.  No final ice number differs beyond rounding on
the seven frames; on the raw analysis state, where WRF's freeze branch fires
at 234 levels, the port agrees there within 2.9e-7, because the terminal
bound (:4040, 999 per litre) decides the number whichever count froze.
