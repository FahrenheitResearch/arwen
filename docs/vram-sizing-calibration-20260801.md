# VRAM sizing calibration, 2026-08-01

What this is: the measurements behind the affine peak-envelope model, the
whole-tree ingest pricing, and the `--card` tier's assumed-free figure.
Every number here was taken on one machine with one instrument, and the
model constants in `gpuwm/core/preflight.py` cite this file.

**Instrument.** vast.ai node, RTX 4080 16 GiB (16,376 MiB physical,
15.33 GiB free to a fresh CUDA context), Linux, driver 595.58.03, 32
cores, GPU otherwise idle (1 MiB baseline). Machine-wide
`nvidia-smi --query-gpu=memory.used` sampled every 250 ms for the whole
of each process, maximum taken. Each phase is a SEPARATE process --
`rw-wps` for ingest, the prepared-cache forecast runner for the forecast
-- so the two peaks never contaminate each other.

**Provenance.** Rows marked *(go route)* are the independent tester's own
`gpuwm go` measurements on the same node, retained because they span two
grid sizes this lane did not re-run. Every other row was measured by this
lane through the staged prepared-cache route. Where both exist for the
same config they agree to within 3%: 224x180 measured 4.14 GiB staged and
4.38 GiB through `go`; 448x360 measured 8.49 staged and 8.75 through
`go`; the 2-domain tree measured 10.09 here and 10.05 for the tester.
The `go` route runs ~0.25 GiB higher, which is inside the margin the
model carries.

## 1. The forecast envelope

### The defect

v1.4.0 predicted `envelope = 1.45 x itemized alloc estimate` on Linux.
A multiplier has no intercept, and this cost has a large fixed term --
the CUDA context plus the launch-time local-memory backing store, 1.5 to
2.9 GiB depending on the device and the kernel set, which does not move
when the grid does. So the model's error changed SIGN with grid size.

### The measurements

| config | domains | itemized estimate | measured peak |
|---|---|---|---|
| 170x136 | 1 | 2.07 GiB | 3.65 GiB |
| 224x180 | 1 | 2.75 GiB | 4.14 GiB |
| 224x180 *(go route)* | 1 | 2.75 GiB | 4.38 GiB |
| 340x272 | 1 | 4.82 GiB | 5.95 GiB |
| 448x360 | 1 | 8.49 GiB (est 7.56) | 8.49 GiB |
| 448x360 *(go route)* | 1 | 7.56 GiB | 8.75 GiB |
| 474x378 *(go route)* | 1 | 8.27 GiB | 9.25 GiB |
| 594x476 *(go route)* | 1 | 12.38 GiB | 12.59 GiB |
| 630x504 *(go route)* | 1 | 13.76 GiB | 13.88 GiB |
| 156x124 + 312x248 | 2 | 4.12 GiB | 5.83 GiB |
| 242x194 + 480x384 | 2 | 8.22 GiB | 10.09 GiB |
| 66x54 .. 112x96 | 4 | 2.06 GiB | 3.86 GiB |

### The fit

Fitting `peak = a x subtotal + b` by least squares over the single-domain
rows (subtotal is the itemization before the x1.15 allocator headroom)
returns **a = 0.981, b = 2.14 GiB**. Forcing `a = 1.00` costs almost
nothing -- residuals stay within +/-0.19 GiB across a 6.6x span of grid
size -- and buys a model with one fitted constant instead of two.

That is the physical statement: **the itemization predicts the pool
essentially 1:1, and the residue is a constant.** A constant is exactly
what a CUDA context plus a per-process backing store is.

The tester's own fit through the two single-domain extremes was
`0.88 x estimate + 1.95` (in terms of the estimate, i.e. after the x1.15
headroom). Converting: `0.88 x 1.15 = 1.012`, so their slope over the
subtotal is 1.012 against this lane's 0.981, and their intercept 1.95
against 2.14. The two fits agree to 3% in slope and 0.19 GiB in
intercept, on partially disjoint data. Their coefficients were not
adopted: the shipped model decomposes the intercept instead of fitting
it whole, because 1.53 GiB of it is a term this module already computes
and which is device-dependent -- a fitted 1.95 GiB would be wrong on any
card that is not a 4080.

### The shipped model

```
peak envelope = alloc estimate
              + non_pool_device_bytes(exp, device)     # context + backing store
              + ENVELOPE_UNMODELLED_BYTES              # 0.50 GiB
              + ENVELOPE_PER_NEST_FRACTION x estimate x nests   # 5% each
```

Residuals of `measured - (estimate + non-pool)` on the 4080 (non-pool
1.535 GiB for the default suite at nz 49 on 76 SMs):

| run | domains | residual | as % of estimate |
|---|---|---|---|
| 170x136 | 1 | +0.05 GiB | |
| 224x180 | 1 | -0.14 GiB | |
| 224x180 *(go)* | 1 | +0.10 GiB | |
| 340x272 | 1 | -0.41 GiB | |
| 448x360 *(go)* | 1 | -0.34 GiB | |
| 474x378 *(go)* | 1 | -0.55 GiB | |
| 594x476 *(go)* | 1 | -1.33 GiB | |
| 630x504 *(go)* | 1 | -1.42 GiB | |
| 156x124+312x248 | 2 | +0.18 GiB | 4.3% per nest |
| 242x194+480x384 | 2 | +0.34 GiB | 4.2% per nest |
| 66x54..112x96 | 4 | +0.26 GiB | 4.2% per nest |

Worst single-domain residual +0.10 GiB, covered five times over by the
0.50 GiB constant. The multi-domain residual is the same 4.2-4.3% of the
estimate per nest at two depths and across a 4x span of estimate, so it
is priced as a fraction, rounded up to 5%.

**The non-pool term follows the device.** The backing store is
`(frame - default stack) x SMs x threads per SM`. The reference profile
in this module is a 170-SM RTX 5090; a 76-SM 4080 carries 0.447x of it.
`gpuwm check` and `gpuwm go` read the SM count off the local card
whenever the free-VRAM figure is being measured off that card. Sizing
for a card that is not in the machine uses
`CARD_CLASS_MULTIPROCESSORS`, the largest SM count sold at each
capacity, which over-prices every other card in the class.

### Cross-checks on cards this model was not fitted on

The three instrumented Linux pilots of 2026-07-30, re-read with each
card's own non-pool term:

| node | card | estimate | measured | new envelope | margin |
|---|---|---|---|---|---|
| 1 | 4090 (128 SM) | 7.20 GiB | 9.54 GiB | 10.00 GiB | +4.8% |
| 2 | 4090 (128 SM) | 7.29 GiB | 8.99 GiB | 10.09 GiB | +12.2% |
| 3 | 4070 (46 SM) | 3.51 GiB | 4.04 GiB | 4.86 GiB | +20.3% |

Conservative on all three, on two card models neither of which is the
one the constants were fitted on.

### Windows / WDDM

Unchanged evidence, and therefore unchanged behaviour: the one
instrumented WDDM run measured 1.746x its footprint projection, and that
multiplier is retained as a FLOOR under the affine form. On Windows the
envelope is `max(affine, 1.75 x footprint)`. Nobody has instrumented a
Windows run small enough to show where the two cross, so the multiplier
is not retired -- it is bounded from below by a model that cannot be
optimistic about a small configuration.

## 2. The ingest phase

### The defect

v1.4.0 priced preprocessing on the ROOT DOMAIN ONLY. A deeper ladder has
a smaller root, so the prediction FELL as domains were added while the
real cost did not move:

| config | domains | v1.4.0 predicted | measured |
|---|---|---|---|
| 448x360 | 1 | 5.46 GiB | 4.58 GiB |
| 156x124 + 312x248 | 2 | 1.03 GiB | 1.94 GiB |
| 242x194 + 480x384 | 2 | 1.89 GiB | 4.01 GiB |
| 66x54 .. 112x96 | 4 | 0.54 GiB | 0.98 GiB |
| 134x106 .. 236x176 | 4 | 0.87 GiB | 2.63 GiB |

Under by 1.9x, 2.1x, 1.8x and 3.0x on the four trees, in the unsafe
direction, on the number the before-the-fetch gate compares against the
forecast to pick a binding phase.

### The model

The hierarchy is verified and exported as ONE transaction, so every
domain's initial state is on the device together. So:

```
resident  = 2 forcing times x root per-time + root boundary tables
          + one complete initial state per NEST
transient = 0.65 x (analysis + state) of the WIDEST domain in the tree
```

Single-domain configurations are untouched by this change: a tree with
no nests contributes no nest states, and its widest domain is its root.

After: every tree is over-predicted by 10.9% to 22.1%, and every single
domain by 19.3% to 68.0% -- the whole set conservative, none under.

## 3. The `--card` tier

A card never hands a process its nameplate capacity. This node's RTX
4080 carries 16,376 MiB (15.99 GiB) and presents **15.33 GiB** free to a
fresh CUDA context -- a 0.66 GiB gap. The one 32 GiB free figure this
codebase has measured is 30.27 of 31.84, a 1.57 GiB gap, on a machine
that also had a desktop on the card.

The tier assumed the nameplate. Combined with a fit loop that grew the
grid until the envelope TOUCHED the budget, every `--card 16gb` ladder
landed 0.13-0.32 GiB over the budget a real 16 GB card leaves, and
returned `gpuwm check` rc 4 minutes after the wizard printed PASS.

Now: `assumed free = nominal - max(0.75 GiB, 6%)`, and the fit loop
stops `max(0.25 GiB, 5% of budget)` short of the budget.

| tier | assumed free | measured/derived real free |
|---|---|---|
| 12 GiB | 11.25 GiB | 11.34 GiB |
| 16 GiB | 15.04 GiB | **15.33 GiB (measured)** |
| 24 GiB | 22.56 GiB | 23.33 GiB |
| 32 GiB | 30.08 GiB | 30.27 GiB (measured, with a desktop) |

## 4. The reserve is suite-dependent

`ReservePolicy.n0_alloc` charges the local-memory backing store of the
SELECTED KERNEL SET. On the reference profile that is 1.93 GiB for
WSM6 + MYNN, 2.42 for Morrison mp10, 2.91 for the Thompson default and
**3.94 for NSSL2 double-moment**. The wizard's fit loop assumed a flat
4.0 GiB, so any suite whose reserve exceeded that was sized against one
budget and verified against a smaller one. Both NSSL2 profiles therefore
emitted a config that failed their own `gpuwm check` at every card size.

The fit loop calls `ReservePolicy.n0_alloc` on the candidate experiment
now -- the same call `gpuwm check` makes -- so the two cannot disagree
about one file.

## 5. Preprocessing memory: what the release actually delivers

Measured on this node against a real 1.3.1 install in an isolated venv
and HOME, the SAME downloaded GRIB2 files, the SAME WPS_GEOG tree and
the SAME config (242x194 @ 12 km + 480x384 @ 3 km):

| forcing times | window | 1.3.1 peak | 1.4.0 peak | reduction |
|---|---|---|---|---|
| 2 | 2 h | 4.80 GiB | 3.98 GiB | 1.20x |
| 5 | 12 h | 6.33 GiB | 4.00 GiB | 1.58x |

The architectural claim is real and better than the headline: 1.3.1's
preprocessing peak GROWS with the number of forcing times (~0.51 GiB
each) and 1.4.0's is FLAT (3.98 -> 4.00). A 24 h window now costs the
same preprocessing VRAM as a 2 h one.

"~3x" is a property of a nine-forcing-time window, not of the release.
Extrapolating 1.3.1's measured slope, the ratio reaches 3x at about 13
forcing times (~36 h). For the 2-12 h windows a first run actually uses,
expect 1.2x-1.6x.
