# 3 km grounding run: the device-memory fit

The device-memory fit that `B4-SPEED-ANCHORS.json` carries as
`device_memory_fit`, and that `tools/battery_route_preflight.py` prints beside
the allocation estimator for every route it prices. `B4-SPEED-ANCHORS.json`
pins the bytes of this file and of the samples it rests on.

The run: gpuwm 1.4.1 on an RTX 5090 (32,607 MiB) under Linux, driver 580.142,
CUDA 13.0, with no other tenant: the card read 0 MiB and 0% before the run,
and every sample lists only the run's own processes. 796 x 636 x 49 =
24,806,544 cells at dx 3 km, dt 15 s, 6 h from GFS 2026-08-02 00Z, with the
product's default suite: Thompson, RTE+RRTMGP longwave and shortwave, YSU,
Noah, Kain-Fritsch. Its run report is `grounding-3km-conus-run-report.json`
beside this file. Statements below about `gpuwm check`, the card sizing rule
and the orchestrator describe gpuwm 1.4.1, the version that ran.

The instrument: `nvidia-smi` at 4 Hz, device-wide and per process, across the
whole chain. Every sample is in `grounding-3km-conus-vram-4hz.tsv` beside this
file, one row per sample: epoch seconds, device MiB used, device MiB total,
utilisation percent, and `pid:MiB` for each process on the card. The largest
device-wide sample is 22,880 MiB (22.34 GiB). The forecast process peaked at
22,364 MiB (21.84 GiB), the preparation process at 14,414 MiB (14.08 GiB), and
the `gpuwm go` orchestrator held 498 MiB (0.486 GiB) throughout.

## The affine fit

One grid size cannot separate a slope from an intercept. What it *can* do is
test the line the product already ships. `gpuwm/core/preflight.py` carries eight
committed RTX 4080 / Linux measurements taken with the same instrument class
(machine-wide, 250 ms). Least squares over those eight:

```
slope 0.7445 KiB/cell   intercept 2.879 GiB
rung          cells     meas    fit   resid
s07         1132880     3.65   3.68   -0.03
small8      1975680     4.14   4.28   -0.14
small8-go   1975680     4.38   4.28   +0.10
s11         4531520     5.95   6.10   -0.15
edge15-go   7902720     8.75   8.49   +0.26
L12-go      8779428     9.25   9.11   +0.14
over22-go  13854456    12.59  12.72   -0.13
big24-go   15558480    13.88  13.93   -0.05
RMS residual 0.140 GiB, max |resid| 0.260 GiB
```

That is a good line -- and note its intercept, 2.879 GiB, lands within
0.03 GiB of the value `gpuwm check` itemizes independently for this suite
(CUDA context 0.42 + local-memory backing store 2.49 = 2.91 GiB). Two
unrelated routes agreeing on the intercept is the strongest thing in this file.

Extrapolated to 24,806,544 cells -- 1.6x beyond the largest point it was fitted
on -- that line predicts **20.49 GiB**. Measured: **21.84 GiB** process,
**22.34 GiB** device-wide. The line under-predicts by **+1.35 GiB (6.6%)**.

Slope from the single 5090 point, as a function of the intercept assumed:

| intercept assumed | slope |
|---|---|
| 2.91 GiB (this suite's itemized non-pool) | **0.800 KiB/cell** |
| 2.14 GiB (an independent affine sizing fit's intercept) | 0.833 KiB/cell |
| 2.00 GiB | 0.839 KiB/cell |
| 1.50 GiB | 0.860 KiB/cell |

The two-point local slope between the largest 4080 rung and this one is
0.903 KiB/cell, which is steeper than either fit -- consistent with the 5090's
intercept being *larger* than the 4080's (170 SMs against 76; the local-memory
backing store scales with multiprocessor count, so `card_local_memory_profile`
already predicts this). A 5090 intercept near 3.4 GiB would put the slope back
at 0.78 and reconcile everything. **Report the range: slope 0.75-0.84 KiB/cell,
best single value 0.80; intercept 2.9-3.4 GiB on this card.**

What this closes: the slope is **not** 0.3-0.4 KiB/cell. It is roughly double
that, and it has been double that consistently across two cards, nine points and
a 22x span of grid size.

## Largest 3 km domain on 32 GiB

**Measured and completed: 796 x 636 x 49 = 2,388 x 1,908 km**, peaking at
22.34 GiB device-wide with 8.5 GiB of the card still free. That is roughly 45%
of CONUS by area -- Arizona to Ohio, south Texas to the Canadian border.

Predicted (arithmetic, unmeasured): at 0.80 KiB/cell + 2.91 GiB intercept +
0.49 GiB for the `gpuwm go` orchestrator's own context, against the 30.86 GiB
this card actually presents, the edge is about 36.0 M cells, i.e. **~960 x 770**.
880 x 704 (30.36 M cells, predicted 26.07 GiB) should therefore have fit
comfortably. **Neither was run. Do not cite either as measured.**

**A full 3 km CONUS domain does not fit a 32 GiB card, and is not close.**
1650 x 1000 x 49 = 80.85 M cells needs **~65 GiB** of peak. Sized the way the
product sizes cards (nameplate minus 6%, minus the ~4.1 GiB reserve, minus 5%
fit headroom), that wants a **96 GiB card**; a 48 GiB card reaches about
1100 x 878, still well short. The belief that 3 km CONUS has been run on a 5090
is not supported by this measurement.
