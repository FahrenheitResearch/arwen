# HRRR fork (WRFV3.9) advection oracle

The reference is the complete `dyn_em/module_advect_em.F` of the WRF fork
operational HRRR v4 runs: NOAA-EMC/HRRR commit
`40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827`, `sorc/hrrr_wrfarw.fd/WRFV3.9`.
`SOURCES.sha256` pins the three fork files the build reads; the pins are the
ones `tools/ieva_wrf_oracle/legacy_build.py` already carries for the same
commit. The fork's model constants are compiled unchanged; its error module
needs ESMF, so `stub_wrf.F90` stands in with inert logging routines and the
configuration type (the fork's advection module reads only the four
advection orders, `scalar_adv_opt` and the boundary flags).

```sh
# fetch the three files under SRC/dyn_em, SRC/share, SRC/frame, then
bash tools/advect_wrf_legacy_oracle/build.sh SRC BUILD_DIR
python3 tools/advect_wrf_legacy_oracle/make_cases.py tests/data/wrf471_advect tests/data/wrf_legacy_advect
nice -n 10 python3 tools/advect_wrf471_oracle/validate_advect_oracle.py \
  --directory tests/data/wrf_legacy_advect --executable BUILD_DIR/run_advect --scratch STREAM_DIR
# the positive-definite low-order flux probe (needs the WRF 4.7.1 oracle build too)
nice -n 10 python3 tools/advect_wrf_legacy_oracle/pd_low_order_probe.py \
  BUILD_DIR/run_advect WRF471_BUILD_DIR/run_advect STREAM_DIR \
  tools/advect_wrf_legacy_oracle/receipts/pd-low-order-probe.json
```

## The one source transformation

WRFV3.9 guards its hybrid vertical coordinate behind variadic C macros
(`mut(...)` becomes `(c1(k)*mut(...)+c2(k))`). gfortran preprocesses
Fortran in traditional mode and rejects variadic parameter lists, so
`expand_hybrid.py` performs exactly those five expansions on executable
text and writes `hybrid-expansion.json` with the counts and the digests
of the input and output. Nothing else in the module changes; the driver
is the WRF 4.7.1 oracle's `run_advect.F90` because the fork's
`advect_scalar`, `advect_u`, `advect_v`, `advect_w`, `advect_scalar_pd`
and `advect_scalar_mono` take the same argument lists.

## What the fork shares with WRF 4.7.1 and where it differs

The `vert_order == 5` arms of `advect_scalar`, `advect_u`, `advect_v` and
`advect_w` are the same text in both sources apart from trailing blanks
(confirmed by diff while this tool was written), so a 4.7.1 build at the
same orders is a cross-check of the fork reference for those four
routines. `advect_scalar_pd` differs: the fork's vertical low-order flux is
the GSL "SL upwind" form, `max(vel,0)*field_old(k-1)+min(vel,0)*field_old(k)`
with a semi-Lagrangian multi-cell sum where the face Courant number
exceeds 1, where 4.7.1 forms `mu*(dz/dt)*flux_upwind(field_old, cr)` with
`cr = vel*dt/dz/mu`.

The two are NOT the same flux. `dz` is negative in eta (`rdzw < 0`), so `cr`
has the sign opposite to `vel`: for `vel > 0` (Omega positive is downward)
4.7.1 takes `field_old(k)`, the cell above the face, which is the upwind
one, and the fork takes `field_old(k-1)`, the cell below it; for `vel < 0`
the two swap. (The fork's own multi-cell sums for a Courant number above 1
do take the upwind cells, so its low-order flux changes sides at Courant 1;
its high-order flux upwinds the way 4.7.1's does.) With a face Courant
number of at most 1, the only regime the implicit-explicit split leaves to
the explicit flux, the consequence is:

- where the limiter does not engage, the total tendency is the high-order
  flux divergence whatever the low-order flux was, so the two sources differ
  in float32 rounding only;
- where it engages, the fork renormalises against, and falls back to, a flux
  that takes the scalar from the downstream side. `pd_low_order_probe.py`
  measures this on the two compiled routines with the same input words
  (zero horizontal flux, uniform Omega at a vertical Courant number of 0.2,
  scalar 1 on one level and 0 elsewhere): WRF 4.7.1 moves the scalar to the
  downstream neighbour and leaves the empty upstream cell at 0; the fork
  gives that empty upstream cell a NEGATIVE tendency (-1533.3 against the
  occupied level's -1055.4 for downward Omega, -1324.0 for upward), a
  redistribution out of a cell that holds nothing
  (`receipts/pd-low-order-probe.json`). The fork's option is therefore not
  positive definite in the vertical where its limiter engages.

So a positive-definite comparison against the fork is a rounding
measurement where the limiter is idle and a material one where it works.

## Fixture layout

`tests/data/wrf_legacy_advect/cases.json` names the eight 4.7.1 input
archives by relative path with `v_sca_adv_order = v_mom_adv_order = 5`
in every case's metadata (`make_cases.py`); `gpuwm.verify.advect_oracle`
reads the orders from the metadata for both the native header and the
CUDA launchers. `<case>-wrf.npz` is the fork reference, `<case>-wrf471.npz`
the 4.7.1 reference at the same orders. The GPU receipts and word archives
follow the 4.7.1 fixture's schema.

## GPU receipts and the measured distances

`gpu-receipt.json` (RTX 4090, sm_89) and `gpu-receipt-sm120.json` (RTX 5090,
sm_120) were captured at c5b677748 with
`validate_advect_oracle.py --directory tests/data/wrf_legacy_advect --gpu --controls --mutation`
(CuPy 14.2.0, CUDA runtime 13.2); the mutation control is rejected on both.
The word folders keep the production words only, as the 4.7.1 RTX 4090 folder
does. The two cards produce the same production measurements on the four
explicit routines.

- Default build at order 5 against the fork: the same class as the 4.7.1
  receipts at order 3. Summed over the eight cases (409,600 words per
  routine), the RTX 4090 differs on 54.6 percent of `advect_scalar` words
  (4.7.1 at order 3: 54.6), 53.4 percent of `advect_u` (53.5), 52.5 percent
  of `advect_v` (52.5) and 45.7 percent of `advect_w` (45.7).
- WRF-exact build (`GPUWM_WRF_EXACT=1 GPUWM_WRF_EXACT_ADVECTION=1`,
  `tools/advect_wrf_exact/compare.py --directory tests/data/wrf_legacy_advect`):
  `advect_scalar`, `advect_u`, `advect_v` and `advect_w` equal the fork word
  for word on every specified and periodic case on both cards. On the open
  boundary case `advect_v` and `advect_w` differ in 1,389 and 2,942 words, the
  same open-boundary distance the 4.7.1 fixture shows at order 3 (1,387 and
  2,939), which the exact tests already leave out.
- `advect_scalar_pd` against the fork is the GSL low-order vertical flux
  described above (fork `module_advect_em.F:8255-8345`), the same on both
  cards. On the five real-state specified and periodic cases whose face
  Courant number stays at most 1 the fork and 4.7.1 references differ by at
  most 5.96e-7 in a tendency of order 1 (1,421 to 4,423 of 51,200 words per
  case): rounding where the limiter is idle, and a small vertical flux where
  it works. In `zero_nearzero_tracers`, whose tracers sit at the limiter's
  floor, the two references differ in 5,790 words by up to 2.4e-28 against
  a largest tendency of 2.2e-27, 11 percent of it: the low-order flux taking
  the other cell. In `map_factor_extremes`, whose vertical Courant number
  exceeds 1, the fork's semi-Lagrangian multi-cell sum moves the vertical
  tendency by up to 221.5. The open boundary case (7,350 words, at most
  0.149) carries the open-boundary distance the 4.7.1 fixture already shows
  at order 3 (4,508 words, at most 0.149).
  These distances describe the first order-5 capture, before the low-order
  vertical flux was ported. Order 5 now launches `pd_vertical_sl` after
  `pd_fluxes`, replacing only the low-order eta flux and its correction
  with the fork's exact source form. In the strict WRF verification build
  the boundary faces and every face at Courant <= 1 use its downstream
  cell choice; the interior and next boundary faces use its bounded
  semi-Lagrangian sums above Courant 1. The `deps` denominator retains its
  binary64 division. No order-3 launch loads this module. The production
  build keeps only the semi-Lagrangian sums above Courant 1 and leaves the
  upwind flux on every other face: the downstream cell drains an empty
  cell, the final-stage clamp adds the drained amount back, and a 12 h 3 km
  smoke forecast gained 52.5 t of smoke that way (lane/ec-tracer-mass), so
  production words on this fixture moved where the limiter meets an empty
  cell and each card's receipt was re-recorded with `--native-fix`.

  The exact-build test now requires all five total tendencies to equal the
  fork word for word on every specified and periodic case. The documented
  open-boundary distance remains outside that exact gate. An intentional
  native fix recaptures with `recapture_receipt.py --native-fix --install`:
  this option first runs that bitwise fork gate in a separate exact-build
  process, and cannot requalify the order-3 fixture or another card's
  receipt. The negative empty-cell tendency is inherited source behavior,
  not a positive-definiteness claim.
