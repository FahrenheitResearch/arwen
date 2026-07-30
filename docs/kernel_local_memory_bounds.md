# Two compile-time array bounds, 9.4 GiB of device memory

`gpuwm/core/kernels/kf.cu` declared ~47 per-thread column arrays at
`KF_KMAX = 128`, and `gpuwm/core/kernels/refl.cu` another ~14 at
`REFL_KMAX = 256`. The cases this project runs have `nz = 49`. Neither
bound is read as a value anywhere in either kernel — every loop runs to the
runtime `nz` — so both were pure allocation size, 2.6x and 5.2x larger than
the vertical grid in use.

On this card that is not a style question. When a kernel's per-thread local
frame exceeds the context's default stack limit, the driver answers the
kernel's **first launch** by allocating a backing store for the device's
whole resident-thread capacity — not for the launched grid — and holds it
for the life of the process. One allocation serves the process, sized by
the largest frame launched so far. The CuPy pool never sees it.

```
reservation_bytes = (max_local_size_bytes - default_stack_limit_bytes)
                    * max_threads_per_multiprocessor * multiprocessor_count
                  = (frame - 1024) * 1536 * 170          [RTX 5090, 610.74]
```

Both bounds are now `#ifndef`-guarded and both launchers compile them to the
field's own level count through `gpuwm.core.kernels.get_kernel_int_defines`.
This document is the measurement record.

---

## 1. The frames, and what the driver reserved for them

Every row measured on the run host: a fresh process, a bare CUDA context,
one small launch of one kernel symbol bracketed by `cudaMemGetInfo` with the
CuPy pool's own delta subtracted. The frame is
`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` read back off the driver.

| kernel | bound | frame/thread | **measured reservation** | law |
|---|---|---|---|---|
| `kf_column` | `KF_KMAX 128` | 24,064 B | **5,738.0 MiB** | 5,737.5 |
| `kf_column` | `KF_KMAX 49` | 9,216 B | **2,040.0 MiB** | 2,040.0 |
| `kf_column` | `KF_KMAX 30` | 5,640 B | **1,152.0 MiB** | 1,149.5 |
| `refl10cm_morrison_column` | `REFL_KMAX 256` | 18,432 B | **4,334.0 MiB** | 4,335.0 |
| `refl10cm_morrison_column` | `REFL_KMAX 49` | 3,528 B | **626.0 MiB** | 623.6 |
| `refl10cm_morrison_column` | `REFL_KMAX 30` | 2,160 B | **286.0 MiB** | 282.9 |
| `refl10cm_thompson_column` | `REFL_KMAX 256` | 16,128 B | **3,760.0 MiB** | 3,761.2 |
| `refl10cm_thompson_column` | `REFL_KMAX 49` | 3,088 B | **514.0 MiB** | 514.0 |
| `refl10cm_wsm6_column` | `REFL_KMAX 256` | 14,080 B | **3,250.0 MiB** | 3,251.2 |
| `refl10cm_wsm6_column` | `REFL_KMAX 49` | 2,696 B | **420.0 MiB** | 416.4 |
| `refl10cm_kessler_cell` | either | 0 B | 0 | 0 |

Ten measurements, worst disagreement with the law 3.6 MiB (0.6%). The frame
itself is linear in the bound and rounds up to the local frame's 8-byte
granularity:

| module | bytes/level | check |
|---|---|---|
| `kf` | 188 | 188 x 128 = 24,064; `align8(188 x 49)` = 9,216 |
| `refl` (Morrison) | 72 | 72 x 256 = 18,432; 72 x 49 = 3,528 |
| `refl` (Thompson) | 63 | 63 x 256 = 16,128; `align8(63 x 49)` = 3,088 |
| `refl` (WSM6) | 55 | 55 x 256 = 14,080; `align8(55 x 49)` = 2,696 |

**Saved at `nz = 49`: 3,698 MiB on `kf_column`, 3,708 MiB on
`refl10cm_morrison_column`.** They are not additive — the reservation is a
maximum over launched kernels, not a sum — but they were the two largest
frames in the tree by a factor of three, so between them they set it.

## 2. No bit moved

The bound is an allocation size. Nothing reads it as a value, and the
highest index any loop in either file forms is `nz - 1` (the reflectivity
melting scan reads `k + 1` from `k <= nz - 2`; KF's downdraft recursions
read `nd + 1` from `nd <= lfs - 1 <= nz - 2`). Two independent checks were
run rather than resting on that reading.

**Before/after digests through the production launchers.** SHA-256 over the
raw output bytes of `gpuwm.core.kf.launch_kf` (the five-column parity batch:
synthetic unstable, the two extracted 12Z soundings, shallow, guarded — all
four `KFPhaseMode` values, 17 output arrays each) and of all four
`launch_refl10cm_*` entry points (the mixed-regime stress batch: warm rain
low, snow/graupel aloft, melting layers, empty columns — Morrison at both
`morr_rimed_ice`, WSM6 at both `hail_opt`, Thompson, Kessler). Captured at
`92216d0` with the unspecialized bounds, and again after the change with the
launchers taking the specialized path.

**74 output digests, plus the 5 recorded driver frames and 2 level counts —
81 entries compared, 0 differing.**

**Side-by-side in one process**, as a permanent gate:
`tests/test_kernel_local_bounds.py` launches the specialized kernel and the
unspecialized module — no defines injected, i.e. exactly the binary that ran
before this change — on the same device buffers and compares raw bytes. 68
KF arrays over four phase modes, and every `refl.cu` kernel over every
scheme option. A stray index past `nz` would be absorbed by the slack at the
old bound and would corrupt a neighbouring column array at the new one, so
that comparison is also the out-of-bounds check.

The same file re-measures every frame in §1 against the driver, so a
compiler or source change that breaks the linear model fails there instead
of silently mispricing a rail gate.

## 3. Preflight

`gpuwm/core/preflight.py` priced the local-memory reservation from a single
per-module constant. Two of those modules no longer launch at the bound they
compile to by default, so the module table is now explicitly the **ceiling**
and `LEVEL_SPECIALIZED_KERNEL_FRAMES` carries the per-level model:

```python
LEVEL_SPECIALIZED_KERNEL_FRAMES = {
    "kf":   LevelSpecializedFrame("kf",   "KF_KMAX",   128, 188),
    "refl": LevelSpecializedFrame("refl", "REFL_KMAX", 256,  72),
}
```

An import-time assertion ties each row's unspecialized frame to the
driver-measured `KERNEL_MAX_LOCAL_SIZE_BYTES` entry, so the two tables
cannot drift apart. `kernel_local_frame_bytes(exp)` prices every launched
module, taking `kf`/`refl` at the deepest domain that launches them;
`kernel_local_memory_bytes` is the maximum over that, as before.

`KERNEL_MAX_LOCAL_SIZE_BYTES` was regenerated against the driver after the
change (`tests/test_preflight.py::test_the_recorded_local_frames_match_the_driver`,
plus an independent dump over all 51 modules): **unchanged, every row**,
including the three that still fail NVRTC. The `#ifndef` guards move
nothing when nothing overrides them.

Two historical measurements are preserved rather than re-pointed. The two
four-domain runs of 2026-07-26 were made by the as-built binary at the
unspecialized bounds, so `_as_built_overhead` in the tests prices them at
the frames those runs actually compiled to; the bracketing test still holds
them to within 4%.

`gpuwm check configs/real74_4dom_mynn_norad.toml --rail-mib 29500`:

| | before | after |
|---|---|---|
| local-memory backing store | 5,738 MiB (5.60 GiB) | **1,990 MiB (1.99 GiB)** |
| reserve | 7.12 GiB (0.60 + 6.02 + 0.50) | **3.52 GiB (0.60 + 2.41 + 0.50)** |
| estimate vs budget | 20.13 vs 19.16 GiB | **20.13 vs 22.76 GiB** |
| `alloc_estimate_le_wddm_budget` | **FAIL** | **PASS** |

The projection tracks the fix in the other direction too: the same check on
the 60 s-history variant selects **24** kernel modules instead of 23 —
`refl` is now reachable and priced — and the local-memory term does not
move. That is the time bomb, gone from the estimate as well as from the
device.

## 4. The runs

Five forecasts, all `status: complete`, all through the supervised
`python -m gpuwm.cli run` path. Device-wide VRAM sampled at 1 Hz by
`nvidia-smi` from outside the process — the whole-machine rail is the bar,
so the process's own view is not the measurement.

| | run A | run B | run C | run D | run E |
|---|---|---|---|---|---|
| config | `real74_4dom_mynn_norad.toml` | + 60 s history | + 60 s history | + 60 s history | `real74_3dom_mynn_norad.toml` |
| domains / columns | 4 / 861,001 | 4 | 4 | 4 | 3 / 501,001 |
| Kain-Fritsch | **on** (d01) | **on** | **on** | **on** | **on** |
| `refl10cm_*` launched | no | **yes, all 4** | **yes** | **yes** | no |
| in-worker instrument | — | — | yes | yes | yes |
| **device-wide peak** | **27,216 MiB** | **27,277 MiB** | **27,317 MiB** | **26,740 MiB** | **19,574 MiB** |
| card before the run | 2,679 MiB | 2,541 | 2,596 | 2,595 | 2,647 |
| **under the 29,500 rail** | **2,284 MiB** | **2,223** | **2,183** | **2,760** | **9,926** |
| wall | 123 s | 118 s | 124 s | 119 s | 69 s |

**Four domains with Kain-Fritsch fits.** Run A is
`configs/real74_4dom_mynn_norad.toml` with nothing changed. The same
configuration measured **31,130 MiB** on 2026-07-26 — 1,630 MiB *over* the
rail — and `configs/real74_4dom_mynn_norad_nocu.toml` exists only because of
that. It no longer has to.

### `refl10cm_*` was launched, and it cost nothing

The one change in runs B-D is `history_interval_s = 60.0` on every domain
(a whole number of each domain's steps), so the 60 s forecast reaches a
history frame and the microphysics drivers' `refl_10cm_due` branch fires.
Both traced probes behind the original diagnosis wrote their t=0 frames
against a 900 s interval and never launched a reflectivity kernel.

The output is the proof. `gpuwm/runtime.py:1375` emits `REFL_10CM` only for
`ticks != 0`, and only by consuming the stash the kernel wrote:

```
wrfout_d01_1974-04-03_12_00_00   REFL_10CM ABSENT   (86 variables)
wrfout_d01_1974-04-03_12_01_00   REFL_10CM present  (1,49,200,250) float32
                                 min -35.000  max 32.437 dBZ  all finite
wrfout_d04_1974-04-03_12_00_00   REFL_10CM ABSENT   (85 variables)
wrfout_d04_1974-04-03_12_01_00   REFL_10CM present  (1,49,600,600) float32
                                 min -35.000  max -3.659 dBZ  all finite
```

The in-worker instrument caught its first launch:
`refl10cm_morrison_column`, `local_size_bytes = 3528`, **non-pool step
0.0 MiB**. Mid-flight, at the first history frame of a four-domain
forecast, the reservation does not move. As built it would have gone from
64 MiB to 4,335 MiB at that instant, past the gate that let the run start.

### The in-run reservation, same instrument as the diagnosis

Run E is the *same three-domain configuration* whose as-built trace measured
`kf_column`'s first launch at **5,738.0 MiB**, non-pool, in-process:

| | as-built (2026-07-26) | specialized (run E) |
|---|---|---|
| `kf_column` frame | 24,064 B | **9,216 B** |
| first-launch non-pool step | **5,738.0 MiB** | **2,040.0 MiB** |
| non-pool at the device peak | 6,573 MiB | **4,052 MiB** |
| device-wide peak | 23,799 MiB | **19,574 MiB** |

**3,698.0 MiB returned, in a real forecast, measured by the instrument that
found the problem.**

### One thing measured and not explained

At *four* domains the same bracket reports **3,645.9 MiB** (run D) and
**3,819.5 MiB** (run C) across `kf_column`'s first launch, for the same
9,216 B frame that steps exactly 2,040.0 MiB at three domains and in every
fresh-process probe. It reproduces, so it is not another process on the
card. Two candidate explanations were tested and both are dead:

- **Grid dependence.** A fresh process launching `kf_column` at 1, 1,563,
  8,160 and 16,320 blocks reserves 2,040.0 MiB every time; at
  `KF_KMAX = 128` and 1,563 blocks it reserves 5,738.0 MiB. The reservation
  does not see the grid.
- **Sized by the widest LOADED frame rather than the widest launched one.**
  3,606 MiB is what `nssl2`'s 15,504 B frame would reserve, and the run
  compiles that module without launching it. Resolving every `nssl2` symbol
  in a fresh process and then launching `kf_column` still steps 2,040.0 MiB.
  Preflight's "maximum over launched kernels" survives.

Whatever the four-domain residual is, it is not this array bound: the
identical kernel at the identical frame reserves exactly the law's 2,040 MiB
at three domains, in seven fresh-process probes, and its own driver-reported
frame inside the four-domain worker is 9,216 B. It belongs to the in-run
non-pool accounting at the four-domain shape and is recorded here for
whoever owns that model. The rail measurement is unaffected — run A is
device-wide NVML, taken from outside the process, and it is 2,284 MiB under.


## 5. What this does not do

- **The bound is still a ceiling, not a clamp.** `nz` past 128 (KF) or 256
  (reflectivity) raises, as before. Specializing downward must never turn a
  refusal into a silent truncation.
- **One extra NVRTC compilation per distinct `nz`.** Configurations whose
  domains differ in level count compile `kf.cu`/`refl.cu` once per level
  count. CuPy's on-disk cache absorbs this across runs; the reservation is
  still one allocation, sized by the deepest.
- **Nothing else in the tree was specialized.** `ysu` (9,232 B) is now the
  widest frame in the four-domain YSU reference configuration, ahead of
  KF's specialized 9,216 B; `thompson` (11,264 B) and `nssl2` (15,504 B) are
  wider still where they are selected. Their bounds were not examined here.
- **The 3% pool-retention term and the 432 MiB context term are unchanged**
  and carry their existing provenance.
- **The four-domain in-run bracket residual of §4 is open**, and belongs to
  whoever owns the non-pool model rather than to this change.

## 6. Reproducing

```
# the bounds, the frames, and that no bit moves (device)
pytest tests/test_kernel_local_bounds.py
pytest tests/test_preflight.py::test_the_recorded_local_frames_match_the_driver

# the pricing model and the gates (CPU)
GPUWM_NO_LOCAL_GPU=1 pytest tests/test_preflight.py

# the gate that used to refuse the cumulus configuration
GPUWM_NO_LOCAL_GPU=1 python -m gpuwm.cli check \
    configs/real74_4dom_mynn_norad.toml --rail-mib 29500     # exit 0

# run A: four domains with Kain-Fritsch, nothing else changed
python -m gpuwm.cli run configs/real74_4dom_mynn_norad.toml --outdir OUT
```

Runs B-E used a copy of that config with `history_interval_s = 60.0` on
every domain (runs B-D) and `real74_3dom_mynn_norad.toml` (run E). Device-
wide VRAM was sampled at 1 Hz with
`nvidia-smi --query-gpu=memory.used`, outside the process; runs C-E
additionally put a `sitecustomize.py` on `PYTHONPATH` so the supervisor's
worker subprocess carried the launch-bracket instrument.
