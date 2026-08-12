# RRTMGP's fixed VRAM: where it goes, and what comes back

Measured on an idle RTX 4090 (23.52 GiB, CUDA 13 / cupy-cuda13x), GPU work
bound to `taskset -c 64-127` because the card sits on NUMA node 1 and an
unbound process halves D2H.

---

## 0. The headline correction

The brief's premise was that RRTMGP is ~1.63 GiB of the ~1.90 GiB fixed
intercept, and that the opportunity is its 240-step cadence. **The first half
is wrong and the second half buys nothing.** Both were measured, and both
negative results are the useful part of this report.

* The ~2 GiB intercept is **not RRTMGP**. It is non-pool device memory --
  CUDA context plus NVRTC module images -- and a one-selector-at-a-time
  ladder in fresh processes puts **YSU at 1,115.3 MiB and Kain-Fritsch at
  388.2 MiB against RRTMGP's 9.2 MiB**.
* RRTMGP *does* cost ~1 GiB that a domain-size regression books as
  intercept, but it is **pooled working set, and it is intercept only
  because it is sized by `column_chunk` (3125) rather than by the domain** --
  so it stops growing above 3,125 columns, i.e. any tile past 56x56.
* Releasing it between firings saves **0.0 MiB of peak** at every chunk
  width tested. The peak is unconditionally on the radiation step.

What *does* come back is 239.7 MiB (24.5%) of the workspace, for free, from
a layout change -- and that is the reclamation this lane ships.

---

## 1. Attribution: the allocation list

`nz = 49`, `column_chunk = 3125`, production `p_top = 10 kPa` (real74: the LW
column is 49 model layers + 25 synthetic above-model layers = 74).

`SharedRRTMGPChunkWorkspace` makes **one** allocation of the maximum over its
four solver phases. `lw_rte` is that maximum, so `lw_rte` *is* the
allocation; the other three phases are aliases of the same bytes and must not
be added to it.

### The workspace -- 978.18 MiB, every byte scales with `column_chunk`

| allocation | shape | dtype | MiB | scales with |
|---|---|---|---|---|
| `lw_rte/lev_source` | (3125, 75, 256) | f32 | 228.88 | chunk x layers x g-points |
| `lw_rte/gas_tau` | (3125, 74, 256) | f32 | 225.83 | chunk x layers x g-points |
| `lw_rte/optics_tau` | (3125, 74, 256) | f32 | 225.83 | chunk x layers x g-points |
| `lw_rte/lay_source` | (3125, 74, 256) | f32 | 225.83 | chunk x layers x g-points |
| `lw_rte/vmr` | (3125, 74, 20) | f32 | 17.64 | chunk x layers x gases |
| `lw_rte/cld_tau` | (3125, 74, 16) | f32 | 14.11 | chunk x layers x bands |
| `lw_rte/cld_ssa` | (3125, 74, 16) | f32 | 14.11 | chunk x layers x bands |
| `lw_rte/cld_asy` | (3125, 74, 16) | f32 | 14.11 | chunk x layers x bands |
| `lw_rte/sfc_source` | (3125, 256) | f32 | 3.05 | chunk x g-points |
| `lw_rte/emiss_gpt` | (3125, 256) | f32 | 3.05 | chunk x g-points |
| `lw_rte/incident` | (3125, 256) | f32 | 3.05 | chunk x g-points |
| `lw_rte/col_dry` | (3125, 74) | f32 | 0.88 | chunk x layers |
| `lw_rte/flux_up` | (3125, 75) | f32 | 0.09 | chunk x layers |
| `lw_rte/flux_dn` | (3125, 75) | f32 | 0.09 | chunk x layers |
| **total (the allocation)** | | | **978.18** | |

Phase totals, for the max: `lw_optics` 568.99, **`lw_rte` 978.18**,
`sw_optics` 738.50, `sw_rte` 712.88 MiB.

Exactly linear in the chunk -- **280,044 B/column** -- which reproduces the
brief's own figures: 3125 -> 834.6 MiB, 1024 -> 273.5, 256 -> 68.4 at the
harness `p_top`; the 834.6 vs 978.18 difference is `p_top`, not disagreement
(the WK82 20 km top gives a 63-layer LW column, real74's 10 kPa top a
74-layer one).

### Constant tables -- 22.66 MiB, `lru_cache`d once per process

`gas_lw/kmajor` (14,9,60,256) 7.38 MiB, `gas_lw/planck_fraction` 7.38,
`gas_sw/kmajor` (14,9,60,224) 6.46, everything else 1.44.

### Chunk-capped per-call transients -- 13.26 MiB

Five `metadata_*` at (3125,74) and ten `upper_peak_*` at (3125,74)/(3125,75):
the interpolation coordinates and the synthetic above-model cap. Capped at
`min(ncol, chunk)`, so these too are intercept, not slope.

### Compiled code -- 11.4 MiB

Measured as `full` minus a `full-norad` twin identical in every other
selector.

### Domain-sized -- the honest slope, 110.1 B/cell

Twenty-one (ncol,nz) column packs, six (ncol,nz+1) level packs, `emiss_bands`
(ncol,16) and four (ncol,) surface vectors.

**Fixed total: 978.18 + 22.66 + 13.26 + 11.4 = 1,025.5 MiB = 1.001 GiB.**
Independently confirmed by the ladder: at 96x80x49 RRTMGP's marginal is
+998.7 MiB of pool and +9.2 MiB of module image.

### The ladder that corrects the brief

Non-pool device bytes, one **fresh process per rung**, 96x80x49:

| rung | non-pool | marginal | pool |
|---|---:|---:|---:|
| import only (CUDA context) | 397.1 | -- | 0.0 |
| dry | 418.1 | +21.0 | 57.0 |
| mp10 Morrison | 500.9 | +82.8 | 148.2 |
| +km_opt=4 | 513.5 | +12.6 | 175.6 |
| +sfclay MM5 | 522.2 | +8.7 | 192.9 |
| **+YSU PBL** | 1637.5 | **+1115.3** | 211.6 |
| +Noah LSM | 1637.5 | +0.0 | 211.6 |
| **+RRTMGP** | 1646.8 | **+9.2** | **1210.3** |
| **+Kain-Fritsch (full)** | 2035.0 | **+388.2** | 1226.1 |

The 1.99 GiB intercept is real and it is a sixth of a 12 GB card -- but it is
YSU's and KF's compiled code, and **none of it is reclaimable by anything in
this lane**. RRTMGP's own contribution to it is 9.2 MiB.

---

## 2. What was reclaimed

### (a) Lazy release after the firing -- **0.0 MiB. Reported as a failure.**

Implemented in full (`tilestream/rrtmgp_lazy.py`): the backing comes from a
*private* `cupy.cuda.MemoryPool` -- the default pool would keep the block
against the device on `del`, so the saving would have been bookkeeping -- and
`RRTMGPRadiation.__call__` releases it after the last chunk.
`resident_bytes` between firings is **0**, measured, cycling two allocations
and two releases per window.

And it is worth nothing, because

```
persistent peak = max(peak_ordinary, peak_radiation)
lazy peak       = max(peak_ordinary - W, peak_radiation)
```

and `peak_radiation` exceeds `peak_ordinary` by `W + ~113 MiB` at **every**
chunk width -- the 113 MiB being RRTMGP's domain-sized column packs, which
chunking does not shrink. Radiation is therefore always the busiest step and
the bytes released between firings are bytes nobody else wanted.

Kept anyway, off by default, because it is the mechanism a *second* resident
consumer would need and because measuring it settled the question.

### (b) Tightening the RTE phase layouts -- **239.7 MiB, free**

`tilestream/rrtmgp_tight.py`. `lw_rte` carries five slots from `lw_optics`
that nothing in it reads: `gas_tau`, `cld_tau`, `cld_ssa`, `cld_asy`,
`col_dry` (the RTE call takes `optics_lw.tau`; Planck takes `vmr`). `sw_rte`
carries seven, including `vmr`. The shipped layout keeps them at their
offsets and *appends* the RTE outputs after them.

They cannot simply be deleted -- `phase()` assigns offsets by walking the
layout in order, so `optics_tau` moves the moment anything before it changes
size. Dropping in place and padding the holes recovers only **180.3 MiB**,
because `lev_source` (228.88 MiB) misses `gas_tau`'s hole (225.83) by three
megabytes and the padding then wastes the rest.

So the optics phases are **reordered** to put the carried slots first --
free, because every slot of an optics phase is written before it is read
inside that phase -- which turns the scattered holes into one contiguous
tail. A tail needs no padding: the RTE phase lays its outputs over it and
stops earlier. Not one `_pad` entry survives.

| phase | shipped | tight |
|---|---:|---:|
| `lw_optics` | 568.99 | 568.99 |
| `lw_rte` | 978.18 | **709.13** |
| `sw_optics` | 738.50 | 738.50 |
| `sw_rte` | 712.88 | **408.30** |
| **allocation** | **978.18** | **738.50** |

**978.18 -> 738.50 MiB, 239.68 MiB (24.5%) saved**, at the production
`p_top`; 834.60 -> 738.50 (96.09 MiB, 11.5%) at the harness's 20 km top. The
binding phase is now `sw_optics`, whose five g-point cubes are genuinely
simultaneously live inside `_finalize_cloud_optics`.

**How much is left:** aliasing the finalize kernel's outputs over its gas
inputs is safe -- `rrtmgp_finalize_cloud_sw` is strictly elementwise at
`idx`, reading `tau_gas[idx]`/`ssa_gas[idx]` before writing
`tau[idx]`/`ssa[idx]`/`asym[idx]` -- and would take the allocation from
738.50 to 709.13 MiB. **29.4 MiB for an aliasing contract on a CUDA kernel is
not worth it**, so it was measured and declined, not done. The tightening
captures 239.7 of the 269.1 MiB available: **89%**.

### (c) Chunking -- priced, and mostly a bad trade

Sizing to the tile is already done on `tilestream-vram`
(`shared_workspace.build` caps `chunk` at the tile's own column count), and
sharing one workspace across tile buffers is done there too. What this lane
adds is the *price*, and it is much worse than assumed.

---

## 3. The price

128x128x49 (16,384 columns), `full` rung, one fresh subprocess per row, three
radiation firings in every timed window (fire count printed on every row),
`p_top` = the harness's 5.7 kPa. Peaks are `cudaMemGetInfo`, sampled during
the firing by the release hook while the workspace is still standing.

| mode | chunk | W MiB | peak_ord | peak_rad | ord ms | rad ms | ratio | amort ms/step | digest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| none | -- | 0.0 | 2655.1 | 3731.1 | 18.40 | 822.4 | 44.7x | 21.747 | 75b985e4 |
| persistent | 3125 | 834.6 | 3449.1 | 3561.1 | 18.17 | 834.4 | 45.9x | 21.569 | 75b985e4 |
| lazy | 3125 | 834.6 | **2613.1** | 3561.1 | 18.48 | 820.7 | 44.4x | 21.823 | 75b985e4 |
| **tight** | 3125 | **738.5** | 3353.1 | **3465.1** | 18.23 | **817.5** | 44.9x | **21.557** | 75b985e4 |
| tight+lazy | 3125 | 738.5 | 2613.1 | 3465.1 | 18.28 | 820.3 | 44.9x | 21.624 | 75b985e4 |
| persistent | 1024 | 273.5 | 2879.1 | 2991.1 | 18.19 | 1902.1 | 104.5x | 26.044 | 75b985e4 |
| **tight** | 1024 | **242.0** | 2847.1 | **2959.1** | 18.34 | 1898.0 | 103.5x | 26.173 | 75b985e4 |
| persistent | 256 | 68.4 | 2675.1 | 2787.1 | 19.08 | 7364.2 | 386.1x | 49.680 | 75b985e4 |

**One digest across every row.** Amortised at the production cadence
(radt 12 min, dt 3 s -> one firing per 240 steps).

### Tightening: 96.0 MiB of device peak, and it is free

3561.1 -> 3465.1 MiB at chunk 3125, exactly the 96.1 MiB the workspace lost;
2991.1 -> 2959.1 at chunk 1024. **Radiation step 817.5 ms against the shipped
834.4 ms and amortised 21.557 against 21.569 ms/step** -- indistinguishable,
which is what a pure layout change should be. (At the production real74
`p_top` of 10 kPa the same change is 239.7 MiB rather than 96.1.)

### Lazy: 0.0 MiB, at every chunk

`peak_ordinary` drops by the full workspace (3449.1 -> 2613.1 MiB) and the
device high-water does not move, because `peak_radiation` was already higher.

### Chunking: expensive, and only the first step down is defensible

| step | peak saved | amortised cost | forecast |
|---|---:|---:|---|
| 3125 -> 1024 | 570.0 MiB | +4.48 ms/step | **+20.8%** |
| 1024 -> 512 | 136.0 MiB | +7.67 ms/step | +29.5% |
| 512 -> 256 | 68.0 MiB | +15.05 ms/step | +47.3% |
| 3125 -> 256 total | 774.0 MiB | +28.11 ms/step | **+130%** |

The brief expected "a saving that doubles the radiation step is probably
still worth it". It does not double it: **chunk 256 multiplies the radiation
step by 8.9x and the whole forecast by 2.3x.** The reason is that RRTMGP is
not column-bound at these sizes -- `_gas_optics` costs **33.87 ms per call at
chunk 3125 and 33.66 ms at chunk 256**, so a twelvefold smaller chunk takes
the same time per call and you simply make twelve times as many calls. (This
reproduces the adapter's own controller benchmark on 250x200x49: 256 =
21.9 s, 1024 = 5.54 s, 4096 = 1.71 s -- ~112 ms per chunk throughout.)

Stage breakdown of one 16,384-column call:

| stage | chunk 3125 | chunk 256 | ms/call 3125 | ms/call 256 |
|---|---:|---:|---:|---:|
| `_gas_optics` | 406.4 ms | 4308.7 ms | 33.87 | 33.66 |
| `_planck_sources` | 134.0 | 1240.9 | 22.34 | 19.39 |
| `_lw_rte` | 125.0 | 707.0 | 20.84 | 11.05 |
| `_sw_rte` | 93.5 | 732.9 | 15.58 | 11.45 |
| `_mcica_cloud_masks` | 41.8 | 426.7 | 3.48 | 3.33 |

Only the two RTE solvers scale with the chunk at all.

---

## 4. What it buys, in domain terms

Largest square tile that builds **two buffers**, steps twice, and survives a
forced radiation firing on the second step. One fresh subprocess per trial,
CuPy pool capped at card total minus the measured 1.988 GiB non-pool
footprint. **Every refusal cited the pool cap** (`limit set to: ...`), so no
row is a disguised device-level OOM, and every surviving trial printed
`radiation: 1`.

| card | pool cap | shipped | tight | gain |
|---|---:|---:|---:|---|
| 12 GB (11.940 GiB) | 9.952 GiB | **384^2** = 7.2 Mcell | **400^2** = 7.8 Mcell | +16 cells/side, **+8.5% cells** |
| 16 GB (15.920 GiB) | 13.932 GiB | **464^2** = 10.5 Mcell | **480^2** = 11.3 Mcell | +16 cells/side, **+7.4% cells** |

This is `full` (mp10 + km_opt=4 + MM5 sfclay + YSU + Noah + RRTMGP + KF) at
the harness `p_top`, where the tightening is worth 96.1 MiB. At the
production real74 top it is worth 239.7 MiB and the gain is correspondingly
larger.

One thing this table deliberately does **not** include: the scratch-arena and
MYNN-chunk sharing on branch `tilestream-vram`, which took the same
two-buffer ceiling from 336^2 to 416^2. That reclamation and this one touch
different allocations and multiply; they are not measured together here
because this branch's `make_physics_tile_state` does not take that branch's
`shared=`, and measuring both at once would let either claim the other's
bytes.

## 5. Gate and controls

`tilestream/test_gate.py` **PASSED before and after**: 208 PASS each, and
**every sha256 in the gate is byte-identical between the two runs**,
including the restart round trip and the gate's own negative controls. The
"before" tree carries the pristine `gpuwm/core/rrtmgp.py`; the "after" tree
carries this lane's two edits to it (the release hook, and rebinding the two
dead `work`/`mu_chunk` locals that were holding the whole backing alive past
the chunk loop).

`tilestream/test_rrtmgp_vram.py` -- **PASSED**, 48x40x49 (1,920 columns),
radiation fired in every trial:

**Equivalence.** Thirteen configurations -- `none`, and
`persistent`/`lazy`/`tight`/`tight+lazy` at chunks 3125/512/256 -- **one
digest, `def447a42a4044ad`.**

**Residency.** Released to **0 B** between firings, cycling 6 allocations /
6 releases; the shipped workspace has no release path, so the two objects are
demonstrably different.

**The strong control is a poison, and it fires both ways.** 20 runs each:

| control | fired | result |
|---|---|---|
| `poison="dead"` -- NaN over every byte the tightening reuses | **20/20** | answer unchanged **20/20**, moved 0, crashed 0 |
| `poison="live"` -- NaN over the carried prefix | **20/20** | **caught 20/20** (all by the finite check), **miss rate 0.0%** |

The second row is what licenses the first: the instrument demonstrably breaks
the run when it poisons storage the RTE phase really reads, so the dead-slot
result is evidence and not silence.

**The weak control is reported as weak.** The obvious hazard -- free the
arena mid-call and let another allocation take the bytes -- was implemented
(`hazard="release_between_phases"`) and run **40 times**:

```
hazard fired in : 40/40 runs
answer moved    : 0
MISSES          : 40
MISS RATE       : 100.0%
```

**It cannot fire, by construction**, and that is the finding rather than a
defect: CuPy will not free a block that a live view still references, so
`free_all_blocks()` releases nothing and the squatter is handed other memory.
Reporting this run as a clean checkmark -- 40 green trials, no divergence --
is exactly the shape of the seven false results this project has already
produced, which is why the miss rate is printed and why the gate rests on the
poison pair instead.

## 6. Two things a later lane should know

* **The restart manifest records `nbytes` and `phase_layouts`**
  (`gpuwm/io/restart.py:_rrtmgp_workspace_identity`), so a checkpoint written
  under the tight layout will be refused by a run configured with the shipped
  one. The answers are identical; the manifest is stricter than the physics.
  That is fail-closed and correct, but it means the layout is part of a run's
  identity and cannot be flipped mid-forecast.
* **`LazyRRTMGPChunkWorkspace.nbytes` deliberately reports the full size even
  while released**, because the preflight ledger and the restart identity
  both compare against it; `resident_bytes` is the one that answers "what is
  held right now".
