# WRF v4.7.1 advection oracle

For an intentional order-5 native fork fix, the recapture tool accepts
`--native-fix` only with the fork fixture. It first requires bitwise native
total tendencies for all five routines on the specified and periodic cases
in a separate WRF-exact process. The ordinary recapture path still refuses
any changed production word, and the order-3 fixture cannot use this option.

The reference is the complete, byte-unmodified `dyn_em/module_advect_em.F`
from WRF v4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
`SOURCES.sha256` pins every WRF file compiled by the build. The actual WRF
model constants and error module are compiled, without replacing constants.

```sh
bash tools/advect_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
nice -n 10 BUILD_DIR/run_advect INPUT.bin OUTPUT.bin
nice -n 10 python3 tools/advect_wrf471_oracle/verify_compiler_controls.py BUILD_DIR INPUT_DIR CONTROL_OUTPUT_DIR
nice -n 10 python3 tools/advect_wrf471_oracle/verify_limiter_coverage.py WRF_SOURCE_ROOT INPUT_DIR COVERAGE_BUILD_DIR
```

## Where the fixtures live

The inputs, compiled WRF references, GPU receipts and word archives are in
`tests/data/wrf471_advect`, beside the diffusion oracle's
`tests/data/wrf471_diffusion`. They are test data, not package data: in 2.8.2
they sat under `gpuwm/data/advect/oracle` and, with the small-step and
big-step fixtures, made the pure wheel 184 MB, over the 100,000,000-byte
per-file distribution limit, while no runtime code reads them. The harness
(`gpuwm/verify/advect_oracle.py`) runs from a source checkout and refuses by
name in an install that has no `tests/data`. The move changed no fixture
byte: every digest in `oracle-sha256sums.txt` is the same and only its path
column moved. `receipts/h100-bw-advection-recapture.json` is a historical
record and keeps the `gpuwm/data/advect/oracle/` paths it recorded; its
`manifest_sha256_after` is the digest of `oracle-sha256sums.txt` as that file
read before its path column moved.

The reference uses gfortran, binary32 `REAL`, `-O0`, `-ffp-contract=off`, and
`-fcheck=all`. The build also creates `o2/run_advect` and `snan/run_advect`
as compiler controls. These binaries use the same unmodified WRF source.
The `o2` build uses `-O2 -ftree-vectorize -funroll-loops`; the `snan` build
initializes local real values to signalling NaN and local integers to a
sentinel. Neither control is the reference fixture.
`verify_compiler_controls.py` replays every `*.in.bin` input and writes a
word-count and SHA256 receipt for each compiler control. It refuses a
signalling-NaN control that changes any reference output word.

`verify_limiter_coverage.py` builds a separate instrumented copy containing
integer-only counters at the native positive-definite and monotonic limiter
branches. It checks the WRF source pins before copying, leaves the pinned
source unmodified, and requires all instrumented output words to equal the
existing pristine reference. Its receipt distinguishes all limiter loop
cells from physical mass-domain cells, excluding limiter halo work from the
physical counts. The coverage executable does not produce the reference.

`compiler.txt`, `source-sha256sums.txt`, `oracle-sha256sums.txt`,
`undefined.txt`, and `symbol-*.txt` are build receipts. The source and
object symbol checks prevent a fixture built from modified WRF code or
from an omitted routine being presented as a native Fortran comparison.

The service-only stub supplies a configuration type containing every field
the native advection routines read: four advection orders, periodic x/y,
specified/nested, four open and four symmetric boundary flags, and polar.
`module_bc` re-exports that type; its executable routines are unused by
these calls. `wrf_abort` stops the program, and the external debug service
does no calculation. No advection, limiter, boundary tendency, or physical
constant is replaced by a stub.

## Native routine coverage

Modes call `advect_scalar`, `advect_u`, `advect_v`, `advect_w`,
`advect_scalar_pd`, and `advect_scalar_mono` with their full WRF argument
lists. Positive-definite and monotonic limiters are inline blocks within
the last two routines, rather than separate callable subroutines.

## Binary protocol, version 1

All words use native little-endian IEEE binary32 or signed int32. Input is
one stream, without Fortran record markers. The 27-int header is:

```text
version=1, mode,
ids,ide,jds,jde,kds,kde,
ims,ime,jms,jme,kms,kme,
its,ite,jts,jte,kts,kte,
time_step,h_mom_order,v_mom_order,h_sca_order,v_sca_order,flags,tenddec
```

Modes are 1 scalar, 2 u, 3 v, 4 w, 5 positive-definite scalar, and 6
monotonic scalar. Flag bits are 0 periodic x, 1 periodic y, 2 specified,
3 nested, 4 to 7 open xs/xe/ys/ye, 8 to 11 symmetric xs/xe/ys/ye, and 12
polar. `tenddec` is 0 or 1 and controls the native optional tendency
decomposition in modes 5 and 6.

The remaining inputs, in order, are:

1. Three binary32 scalars: `rdx,rdy,dt`.
2. Seven binary32 3D arrays: `field,field_old,tendency,ru,rv,rom,romI`.
3. Nine binary32 2D arrays: `mut,mub,mu_old,msfux,msfuy,msfvx,msfvy,msftx,msfty`.
4. Six binary32 1D arrays: `c1,c2,fzm,fzp,rdzw,rdzu`.

Every 3D array has the WRF memory bounds `(ims:ime,kms:kme,jms:jme)`.
The i index is fastest, then k, then j. A NumPy array with canonical axes
`(k,j,i)` is written with `array.transpose(1,0,2).tobytes(order="C")`.
Every 2D array has `(ims:ime,jms:jme)` bounds and a canonical `(j,i)`
NumPy array is already in stream order. Vertical arrays span `kms:kme`.
The caller retains all lower bounds, staggering, and halo values supplied
by the fixture adapter; it performs no physical input conversion.

For w, `c1,c2` are the full-level coefficients and the native routine uses
`rdzu`. Other modes use half-level coefficients and `rdzw`. `kme` must
include `kde`, because native vertical fluxes address that boundary.

Output is three complete 3D arrays in the same stream order:
`tendency,h_tendency,z_tendency`. The final two are initialized to
`-999999.0` before the call and retain that value wherever the native
routine does not write them. They are active native outputs only for
modes 5 and 6 with `tenddec=1`. No header or footer is written. Comparing
the entire output preserves evidence for untouched halo and boundary words.

## CUDA comparison and exact gates

```sh
nice -n 10 python3 tools/advect_wrf471_oracle/validate_advect_oracle.py \
  --directory FIXTURE_DIR --executable BUILD_DIR/run_advect --scratch STREAM_DIR
nice -n 10 python3 tools/advect_wrf471_oracle/validate_advect_oracle.py \
  --directory FIXTURE_DIR --gpu --controls --mutation \
  --receipt GPU_RECEIPT.json --words-directory GPU_WORDS_DIR
python3 -m pytest tests/test_advect_wrf471_parity.py -q
```

The input adapter uses the real state, four lateral halo cells, WRF's
natural U/V/W staggering, and all binary32 input words. Periodic redundant
faces alias face zero. Specified and open halos use zero gradients. WRF
receives total stage mass `mut`, base mass `mub`, and the original old
perturbation mass `mu_old` as separate arguments. The CUDA launchers receive
the total old mass their public interfaces require. W uses the full-level
hybrid coefficients; the other routines use half-level coefficients.

The ordinary `gpuwm.core.advection` launchers produce the four total
tendencies. Native U/V include open-boundary radiation inside the Fortran
routine. The adapter therefore also calls the production
`gpuwm.core.dycore.apply_open_radiative_bc` for radiative-open cases, using
the actual mass fields and map factors. The production moisture launchers
produce the positive-definite total tendency.

Optional positive-definite horizontal and vertical outputs are exposed by
a test-only clone of `pd_renorm_apply`. Its limiter and flux inputs are the
production calculation; additional stores record the two components. The
total tendency is independently obtained from the unmodified production
launch. Every defined optional output word is compared. Native WRF assigns
the horizontal output in its x loop and subsequently adds its y term. At
an open or specified x edge the x assignment is absent, so the y addition
reads an undefined native horizontal value. Only these locations are
excluded from the physical-output comparison. Their raw words remain in
the dumps and the receipt records their count. No total tendency word is
excluded.

Each output receipt records its raw-word SHA256, mismatch count, signed-zero
count, nonfinite mismatch count, maximum ULP distance, maximum absolute
distance, and first and worst differing words. Vertical-level measurements
are recorded too. Tests require exact equality of these measurements and
hashes, including an improvement. There is no tolerance acceptance gate.
The test-only flux5 mutation changes coefficient 37 to 38 and must change a
compared output, demonstrating that the gate rejects a transcription error.

The committed captures include an RTX 4090 with CUDA 13, an H100 with
CUDA 12, and an RTX 5090 (sm_120) with CUDA 13. Named measured cards select
their own receipt. Other sm_120 cards replay the architecture's RTX 5090
measurement and must reproduce every output hash and metric exactly at
runtime. The receipt retains the actual measured card's name; architecture
selection does not claim that another card was measured. An unrecorded
Blackwell-or-newer architecture remains an explicit test-coverage gap. CPU
checks validate every collected capture. The RTX 4090 and H100 produce the same production
output words on these eight cases. The four isolated defect controls pass
on all three measured cards. These measurements do not establish parity
on an unmeasured GPU or compiler. The RTX 5070 Ti that runs the release GPU
stage is an sm_120 card and replays the RTX 5090 receipt: a fresh capture with
this tool on it (NVRTC 13.4.92) reproduced every production and control
measurement and every output word of that receipt, 14,336,000 words with none
different (`receipts/rtx5070ti-sm120-replay.json`).

The sm_120 capture differs from the other cards in five production arrays:
open-boundary V and W, and positive-definite totals in the open, steep-terrain
and specified-west populations. Disabling FMA contraction makes all 64
production measurement records cross-card identical. No previously exact
WRF comparison becomes nonexact. The separate receipt preserves these measured
differences without changing a tolerance or the production arithmetic.

The merged bigstep damping fix changed the `openbc` translation unit after
the RTX 4090 and H100 captures. Each receipt pins the full assembled source the
default loader compiles, and CPU checks require exact equality with the current
source. A pin moves only with a PTX identity receipt showing that the kernels
this oracle runs compile to identical code, or with a fresh capture on the
receipt's card (for the sm_120 receipt, an sm_120 card) that reproduces every
recorded word (the re-pin sections below). The
unrelated `w_damp` and CFL routines remain covered by the bigstep oracle. CUDA
replay still requires exact equality to each card's measured output words.

## Named arithmetic differences

All these comparisons use binary32 on both sides. The differences below
are expression ordering and CUDA multiply-add contraction, rather than a
binary64 intermediate. `no_fma` compiles the same CUDA source with
`--fmad=false`. `wrf_flux` replaces only the three stencil helpers with
WRF's own expression ordering. `wrf_flux_no_fma` applies both controls.
The production source and launch defaults are unchanged by the controls.

For ordinary U/V/W/scalar advection, WRF divides the centered stencil and
dissipation by 60 separately, then multiplies their difference by velocity.
CUDA `advection.cu:78-102` combines velocity times both numerators before
one `__fdiv_rn`. WRF's dissipation also adds the three terms in the reverse
order. The vertical and near-boundary third-order helpers have the same
expression-tree distinction. The native definitions appear at
`module_advect_em.F:196-209`; the equivalent definitions are repeated in
the other ordinary routines. CUDA's map weighting combines horizontal
divergences before multiplication by the map factor
(`advection.cu:407-411,602-606,785-789,1103-1107`), while native WRF computes
`mrdx=map*rdx` and `mrdy=map*rdy` and subtracts the two contributions in
separate loops (`module_advect_em.F:633-634,740-741` for U). The w lid uses
the same arithmetic distinctions after restoring its missing transport.

The positive-definite routine uses a different native stencil spelling:
the coefficients `37/60`, `2/15`, and `1/60` are independently rounded
binary32 constants (`module_advect_em.F:6165-6184`). Its additional named
differences are:

- Face mass: WRF averages two already coupled masses
  (`module_advect_em.F:6283`); CUDA couples the averaged column mass
  (`pd_advection.cu:158-160,185-187,213-215,240-242`).
- Physical spacing: WRF uses `2/(map_a+map_b)/rdx` or `/rdy`
  (`module_advect_em.F:6282,6413`); CUDA uses
  `dx*2/(map_a+map_b)` or the y equivalent (`pd_advection.cu:162-164`).
- Old coupled mass: WRF evaluates
  `(c1*mub+c2)+c1*mu_old` (`module_advect_em.F:7733`); CUDA evaluates
  `c1*(mub+mu_old)+c2` through its total-mass input
  (`pd_advection.cu:300`).
- Final divergence: WRF evaluates correction-right minus correction-left,
  plus low-right minus low-left, and subtracts z, x, and y contributions
  separately (`module_advect_em.F:7794-7796,7827-7829,7866-7868`). CUDA first
  recombines the two fluxes at each face (`pd_advection.cu:394-400,417-425`).

One measured control population is the real interior case: 24 by 24 cells,
49 mass levels, all lateral halo words included. On the original kernels,
the worst measured ULP distances changed as follows. These are measured
distances in this one population, not universal bounds.

| Routine | Production | WRF stencil ordering, no FMA |
| --- | ---: | ---: |
| `advect_scalar` | 2,066,809,130 | 262,144 |
| `advect_u` | 4,132,538 | 8,192 |
| `advect_v` | 327,927 | 4,096 |
| `advect_scalar_pd` | 10,810,058 | 88,064 |

The scalar production maximum is a sign crossing after cancellation:
CUDA wrote -0.10937722027301788 while WRF wrote 0.051025390625.
Its largest absolute difference over the whole population was
0.2186422348022461 in coupled tendency units. ULP counts near zero are
therefore not a useful general accuracy guarantee. The positive-definite
case's largest absolute difference was 1.607462763786316e-6, even though
its worst ULP count was 10,810,058. The final packaged receipt gives the
measurements for every case, output array, and arithmetic control. A
remaining structural difference is reported as DIFFERENT rather than
being folded into this arithmetic explanation.

## Supported and unsupported limiter populations

Positive-definite periodic and specified cases measure the implemented
limiter. The zero and subnormal tracer case is a separate FTZ population:
CUDA's production compile flushes subnormal inputs and outputs, while the
native binary32 reference retains them. It is excluded from a rounding-only
population. Radiative-open positive-definite kernels are an unsupported
probe: `gpuwm/core/moist.py:641-649` explicitly routes those production
domains through unlimited final-stage advection and a clamp because the
native positive-definite open-radiation blocks are unported. The reference
still measures that native routine, and the receipt retains the difference.

Monotonic transport is a distinct native routine. The production engine
documents options 0 and 1, and its namelist importer accepts option 1 only
(`docs/public/CONFIGURATION.md:429`, `gpuwm/namelist_import.py:4235-4242`).
The fixture calls native `advect_scalar_mono`; its comparison to the actual
positive-definite result demonstrates that these choices cannot be aliased.
There is no CUDA monotonic parity claim.

## Exact regression controls for the two reproduced defects

`make_w_lid_controls.py` generates isolated native horizontal and vertical
w-lid outputs. The controlled values are exactly representable. Both are
checked bit-for-bit through the normal w launcher by
`test_advect_cuda_w_lid_has_each_compiled_wrf_term_default_on`.
The realistic combined w outputs remain in the main population.

`make_mapped_radiation_controls.py` uses dry column mass 65,536, map factor
2, and normal winds 40 with a one-unit adjacent gradient. The map factor
changes the phase-speed clamp's branch. Both native normal-face arrays are
checked bit-for-bit through the ordinary radiation launcher by
`test_advect_cuda_mapped_radiation_matches_compiled_wrf_default_on`.

## The 2.8.2 merge re-pin of openbc

Merging this oracle with the compiled-WRF bigstep oracle (0673d8c51) changed
`openbc.cu` inside `w_damp` and `w_cfl_stat_impl` only, so both receipts'
`kernels.openbc` digest moved from `4916f05f` to `5bf7babe` with no capture
on the 4090 or H100. The words they record are still these cards' words:
`tools/kernel_ptx_identity/compare.py` compiled `openbc` at this oracle's tip
(5ff790c0f) and at the merged head (dd4908a0f) for compute_89, compute_90 and
compute_120, under the loader's options and under the RawModule options CuPy
compiles with (the tool's `load_module` and `rawmodule` sets), and
`open_u_radiative` and `open_v_radiative`, the only openbc kernels this
oracle runs, have identical PTX in all six readings. Only
`w_damp`, `w_cfl_stat` and `w_cfl_stat_window` differ
(`receipts/openbc-ptx-2.8.2-merge.json`). With checkouts of the two commits:

    python -m tools.kernel_ptx_identity.compare \
        --tree advect_tip_5ff790c0f=<checkout of 5ff790c0f> \
        --tree merged_dd4908a0f=<checkout of dd4908a0f> \
        --module openbc --out receipts/openbc-ptx-2.8.2-merge.json

The receipt was first written by a one-off script with the `load_module`
readings only; the general tool reproduced those three readings exactly and
added the three `rawmodule` ones, which agree.

## The 2.8.2 WRF-exact re-pin of advection and pd_advection

The opt-in WRF-exact branches (merged 1efb5a415) moved `kernels.advection` and
`kernels.pd_advection` in all three receipts from `50cc068e` and `9cdc25d0` to
`66e08b7a` and `7de1823f` with no capture, because the default compile did not
move: from the RTX 4090 and H100 capture tree (5ff790c0f) and from the RTX 5090
capture tree (7d9928995) to the merged head, both modules compile to identical
PTX, every entry, for compute_89, compute_90 and compute_120 under the loader's
options and under the RawModule options, with NVRTC 13.4 and 12.9
(`tools/kernel_ptx_identity/receipts/oracle-advect-2.8.2-nvrtc{13.4,12.9}.json`
and `oracle-advect-sm120-2.8.2-nvrtc{13.4,12.9}.json`). The harness's text
patches (the flux5 mutation, the flux controls and the h/z diagnostic clone)
patch the default compile's view of the module
(`gpuwm/verify/default_kernel_source.py`), not an opt-in branch.

## The 2.8.2 bandwidth re-pin of advection and pd_advection

The bandwidth tuning (lane/282-bw-advection 8c69d26ab and 48cfb1008, merged
7486d241c) changes the PTX of `flux_div_scalar`, `flux_div_v`, `pd_fluxes` and
`pd_renorm_apply` for compute_89, compute_90 and compute_120 with NVRTC 13.4 and
12.9 (`tools/kernel_ptx_identity/receipts/bw-advection-2.8.2-nvrtc{13.4,12.9}.json`),
so no PTX identity reading exists and each receipt moves only on a capture. A
fresh RTX 4090 capture at 7486d241c reproduced every measurement record and
level row of production and the four controls and all 2,867,200 production
words (`receipts/rtx4090-bw-advection-recapture.json`); `gpu-receipt.json` and
its line in `oracle-sha256sums.txt` moved from `66e08b7a`/`7de1823f` to
`e3fa6135`/`b4832282`. The RTX 5090 was not recaptured. A fresh capture on the
RTX 5070 Ti, also sm_120, reproduced every word of `gpu-receipt-sm120.json`,
14,336,000 words with none different
(`receipts/rtx5070ti-sm120-bw-advection-capture.json`), and that receipt moved
on this evidence; it still names the RTX 5090 that measured its words.

The H100 receipt was recaptured at 1a5783fea with CuPy 14.2.0, CUDA runtime
12.9, NVRTC 12.6.85 and driver 570.172.08 (driver API 12.8), on sm_90.
The capture reproduced all 2,867,200 original production words and every
recorded production and arithmetic-control measurement and level row.
A capture of the prior receipt source on the same H100 also matched every
raw field hash and all 14,336,000 words across production, the three
arithmetic controls and the rejected mutation. The fresh receipt and all
40 archives were copied unchanged from the capture tool, and their manifest
hashes regenerated. The H100 pin now follows `e3fa6135`/`b4832282`.

The full advection release-pin subset passed: 19 tests, no failures, errors
or skips. Capture and independent byte checks are recorded in
`receipts/h100-bw-advection-recapture.json` and
`receipts/h100-bw-advection-independent.json`; the compiler and pin readings
are in `h100-bw-advection-environment.json`, `h100-bw-advection-ptx.json` and
`h100-bw-advection-pin-stage.json` in the same folder. These are exact
reproductions of recorded component measurements, not a claim that every
production output equals native WRF.

## The 2.8.5 vertical-order re-pin of advection and pd_advection

The WRF vertical advection orders (lane/286-vadv5) give `flux_div_scalar`,
`flux_div_u`, `flux_div_v`, `flux_div_w` and `pd_fluxes` a trailing `vorder`
argument and WRF's `vert_order == 5` ladder, so the PTX of every entry this
oracle runs changes and no PTX identity reading exists: each receipt moves
only on a capture. This fixture runs at `vorder = 3`, the ladder every earlier
run took. Captures at c5b677748 with CuPy 14.2.0 and CUDA runtime 13.2:

- RTX 4090 (sm_89): all 2,867,200 production words and every production
  measurement and level row reproduced, and the mutation, `no_fma` and
  `wrf_flux_no_fma` control rows reproduced
  (`receipts/rtx4090-vadv5-recapture.json`). `gpu-receipt.json` and its line
  in `oracle-sha256sums.txt` moved: `kernels.advection` e3fa6135 to af8f5c6c,
  `kernels.pd_advection` b4832282 to 9011930b.
- RTX 5090 (sm_120): the same, with every stored control archive compared word
  for word: production, mutation, `no_fma` and `wrf_flux_no_fma` reproduced all
  2,867,200 words each (`receipts/rtx5090-vadv5-recapture.json`).
  `gpu-receipt-sm120.json` moved to the same pins.
- One control moved on both cards: `wrf_flux`, the text-patched WRF-form flux
  arithmetic compiled with FMA contraction allowed, changed in its
  positive-definite outputs only (30,938 of 2,867,200 words on the RTX 5090).
  The patch replaces `pd_flux5`, `pd_flux3` and `pd_flux3h` by name and leaves
  the new `pd_flux5v` untouched, which order 3 never calls; with contraction
  on, the extra branch in `pd_zface_half` changes which multiply-adds NVRTC
  fuses in that patched variant. The same patch without contraction
  (`wrf_flux_no_fma`) reproduced every word, as did production. Both receipts
  carry the new `wrf_flux` rows and the RTX 5090 folder its eight new
  `wrf_flux` archives.
- The H100 receipt was not recaptured: no sm_90 card was available to the
  lane. Its kernel pins still name the source before b70a94a48 (the
  vert_order 5 ladder), the commit that staled it.

## Certified architectures

Byte identity is certified on Blackwell and newer only: compute capability
10.x (sm_100) and 12.x (sm_120) and later. Ada, Ampere and Hopper run but are
not certified (ruling, 2026-10-04). `gpuwm.verify.advect_oracle` carries the
rule (`CERTIFIED_MIN_COMPUTE_MAJOR`) for both advection fixtures:

- A Blackwell receipt gates as before: its kernel and fixture pins must equal
  the tree, and a Blackwell card with no receipt is a coverage gap.
- Every other receipt (the RTX 4090 and H100 here, the RTX 4090 in the
  HRRR-fork fixture) is checked for presence and format and its committed
  words still replay against the reference. If its pins trail the tree, the
  CPU test warns `UncertifiedReceiptStale` with the pins, the current
  digests and the commit that staled them, and fails nothing. On such a card
  the device word comparisons skip with the same report; with no receipt at
  all they skip as uncertified.
- The commit comes from `uncertified_stale` in the fixture's
  `gpu-receipts.json` (`{receipt: {"staled_by": commit, "pins": {pin:
  digest}}}`). A record must be true: it may not name a Blackwell receipt, its
  digests must be the receipt's own, and they must still trail the tree.
  `recapture_receipt.py --install` drops the record of the receipt it
  installs. The H100 receipt's record names b70a94a48.

## Moving a receipt: `recapture_receipt.py`

A receipt moves only through `recapture_receipt.py`, on the card it names
(or a card of its architecture):

    python tools/advect_wrf471_oracle/recapture_receipt.py \
        --fixture tests/data/wrf471_advect --scratch <empty dir> --install
    python tools/advect_wrf471_oracle/recapture_receipt.py \
        --fixture tests/data/wrf_legacy_advect --scratch <empty dir> --install

It copies the fixture into the scratch folder, runs
`validate_advect_oracle.py --gpu --controls --mutation` there with every
`GPUWM_WRF_EXACT*` switch removed, and compares the capture with the receipt
the card selects through the fixture's `gpu-receipts.json` (the same index
the device tests read): every case's measurement and level rows (each carries
the SHA-256 of its words) and, where the folder stores them, every float32
word. A production word that moved is a finding: exit 1, nothing installed.
Otherwise `--install` moves the receipt, its stored archives and its
`oracle-sha256sums.txt` lines; a card with no receipt gets
`gpu-receipt-sm<MM>.json` and `gpu-words-sm<MM>` and an index entry; a card
of a recorded architecture moves that receipt's pins only when every control
reproduced too (otherwise `--as-card` records its own). The record lands in
`receipts/<card>-<fixture>-<pin>-recapture.json` (for the HRRR-fork fixture,
in `tools/advect_wrf_legacy_oracle/receipts/`). Its rules are held on the CPU
by `tests/test_advect_recapture_receipt.py`. Run in card mode without
`--install` at the vertical-order lane's review tip (a2994b637), it
reproduced all 2,867,200 production words and every control row of both
fixtures on the RTX 4090 and the RTX 5090 (`receipts/rtx4090-vadv5-tool-check.json`
and `receipts/rtx5090-vadv5-tool-check.json` here and in
`tools/advect_wrf_legacy_oracle/receipts/`).

Open at the vertical-order lane, informational under the ruling above: the
H100 run of both commands (the WRF 4.7.1 fixture in card mode; the HRRR-fork
fixture as a first capture).
