# Legacy RRTMG (WRF v4.6.1 option 4/4) — integration wiring dossier

Branch: `radiation/legacy-rrtmg`. This documents the driver wiring
semantics, the proof boundaries, and every known divergence of the
`ra_rrtmg_variant = "rrtmg_legacy"` path. The RTE+RRTMGP path
(`"rte-rrtmgp"`, the default) is byte-unchanged by all of it.

## 1. Selection surface and explicit non-goals

Selection: `RunConfig.ra_rrtmg_variant = "rrtmg_legacy"` on the resolved
4/4 pair; token `wrf-rrtmg-4-4-legacy-v1`; importer flag
`gpuwm import-namelist --rrtmg-variant rrtmg_legacy`. Restart identities
`wrf-v4.6.1-rrtmg-legacy-lw-v1` / `-sw-v1` and the buffer policy id
`wrf-v4.6.1-rrtmg-deltap-4mb-buffer-layers-v1` are distinct from the
RTE+RRTMGP identities: a restart written under one 4/4 implementation
refuses to resume under the other.

The only implemented combination (anything else fails closed, at import
where the key is explicit and at prep/engine entries regardless):
`icloud=1`, `cldovrlp=2` (McICA maximum-random), `idcor=0`, `o3input=2`
(CAM climatology), `ghg_input=0` (analytic year formulas; no CAMtr
reader), `aer_opt=0` (zero aerosol — the CUDA SW composition REJECTS
anything else rather than silently discarding aerosol optics; SW-audit
item 1), `swint_opt=0`, no eclipse/slope/SSiB/CAMMGMP. Intentionally
rejected WRF modes confirmed fail-closed by the SW audit: `inflag=1`,
`iceflag=1`, `icpr=0`, `icld=0`.

**Snow-discount token semantics (owner ruling, prep+ozone audit item
3):** the legacy token `wrf-rrtmg-4-4-legacy-v1` carries WRF v4.6.1's
snow treatment VERBATIM — the `min(0.99,(130/re_s)^2)` discount exists
in the transcribed wrapper and activates exactly where WRF activates it
(`iceflg==5`; latent at the production 2/3/1 flags, as in WRF). The
legacy port never shipped a pre-discount behavior, so there is NO
policy selector and no `-v2` of this token. The `-v1`/`-v2` split
belongs exclusively to the RTE+RRTMGP SUBSTITUTION token family
(`wrf-rrtmg-4-4-to-rte-rrtmgp-*`, seam branch), where adding the
discount changed behavior relative to shipped runs. Receipts
distinguish the families by the token string itself.

## 2. Radiation cadence mapping

WRF fires radiation when `MOD(itimestep, STEPRA) == 1` (with the
`itimestep==1 || radt==0 || stepra==1` short-circuits;
`module_radiation_driver.F:1113-1130`, `ra_call_offset=0`), while
cumulus/PBL use `== 0`. gpuwm's `_radiation_step_due`
(`gpuwm/core/physics.py`) is a line transcription of exactly that
predicate, and `stepra` derives from `radt` minutes exactly as WRF's
`STEPRA = nint(radt*60/dt)`. The legacy adapter plugs into the same
`radiation_due` slot RRTMGP uses; no cadence code changes.

## 3. Night-column contract (two layers, both reproduced)

* WRAPPER level (fixture-pinned): `RRTMG_SWRAD` skips the SW call for
  `coszen <= 0`, still writes `COSZR = xcoszen`, zeroes ONLY `SWCF` +
  its listed scalar diagnostics, leaves GSW / flux profiles /
  RTHRATENSW untouched (`swrad_night_outputs`, `SW_NIGHT_ZEROED`,
  `SW_NIGHT_UNTOUCHED` in `rrtmg_legacy_prep`).
* DRIVER level: WRF zeroes `GSW` and `RTHRATENSW` grid-wide at the top
  of every radiation call (`module_radiation_driver.F:1721,1738`) and
  derives `SWDOWN = GSW/(1-ALBEDO)` (`:2877`). Net: night columns end a
  radiation step with zeros; the adapter returns exactly that.

## 4. Model → wrapper input contract

| Wrapper input | gpuwm source | Note |
|---|---|---|
| `p3d`, `p8w` | `atmosphere["pressure"]`, `["p_interface"]` | already WRF's HYDROSTATIC `p_hyd`/`p_hyd_w` (phy_prep transcription in `_prepare_atmosphere`) — the fields WRF's radiation driver receives |
| `pi3d` | `atmosphere["exner"]` | EOS-pressure Exner = WRF's untouched `pi_phy` |
| `t3d` | `atmosphere["temperature"]` | |
| `t8w` | computed in-adapter | phy_prep transcription: interior `fnm/fnp` half→full weights (`state.fnm/fnp`), z-linear extrapolation at surface/top (`module_big_step_utilities_em.F:4904-4936`) |
| `dz8w` | `atmosphere["dz"]` | = WRF `dz8w(1..kte)`; WRF's `dz8w(kde)=0` slot is never read by the wrappers |
| `qv/qc/qr/qi/qs/qg` | state (missing → zeros with matching `f_q*` flags) | |
| `cldfra` | `cal_cldfra1` (already WRF-transcribed in `rrtmgp.py`) | icloud=1 |
| `re_cloud/re_ice/re_snow` (m) | `state.effc/effi/effs` (µm contract) × `F(1e-6)` — but ONLY for schemes in WRF's `use_mp_re` table (`module_physics_init.F:985-1024`): WSM6/Thompson/NSSL-2mom yes, **Morrison NO** (`has_req*=0`, relcalc/reicalc take over; Morrison's 525 µm EFFI bound would trip cldprmc's [5,140] fatal — proven on a real forecast during integration) | see §9 radii seam |
| `emiss, tsk, xland, xice, snow, albedo` | `fields` dict | `snowh = 0.001*SNOW` inside prep (WRF ignores SNOWH) |
| `o33d` | `gpuwm.ingest.wrf_ozone.o33d_profile(julday, julian, xlat, p3d_Pa)` | vmr, verbatim climatology values; see §9 nests |
| `xcoszen, declin, solcon` | in-adapter `radconst`/`calc_coszen` at `xtime + radt*0.5` | see §9 transcendentals |
| flags/calendar | cfg + run clock (`yr, julian, julday, gmt, xtime, radt`) | |

## 5. McICA

WRF derives kissvec seeds from an FP32 double-rounding of the four
bottom layer-midpoint pressures; permuteseed 150 (LW) / 1 (SW). The
certified FP32 NumPy generators (`gpuwm.core.rrtmg_mcica`) match every
stored Fortran `cldfmcl` mask; the LW audit showed the RTE+RRTMGP-side
FP64 generator (`rrtmgp_mcica.cu`) diverges in 103/178 icld=2 cases —
it was never used or adapted here. The device twin
(`rrtmg_mcica_wrf.cu`, icld=2 only, else fail-closed) is bit-gated
against BOTH the NumPy generators and the stored Fortran arrays
directly, wide-deck and chunk-invariant, dual-run. Cost: ~0.24 s per
251,001-column call vs ~94 s NumPy. Known limitation (fail-closed
documented, production-inert): the generators take `lat` as a scalar
dummy, so the batch preps group columns by identical `xlat` bits for
`idcor=1` + `icld 4/5` — degenerating to per-column calls at
all-distinct latitudes; production is `idcor=0`/`icld=2`, which never
reads `lat`. Prep-side sm_120 finding: the moisture→radii→cloud-path
block always runs host-numpy because real columns drive `ciwpth`
through FP32-subnormal values and the 5090 flushes ALL device
arithmetic, comparisons, min/max, AND the f64→f32 cast (FP64 staging
cannot even produce a subnormal f32) — the sanctioned host-side
countermeasure, gated bitwise (cupy == numpy on the full decks).

## 6. Batched engines

`gpu_rrtmg_lw_batched[_device]` (LW; kernels were already ncol-major;
chunk default 4096, ~1.51 GiB transient) and
`CudaSW.rrtmg_sw_batched[_device]` (SW; kernels factored into verbatim
`__device__` bodies + `_b` re-indexing twins; chunk default 2048,
~2.13 GiB transient; day-columns-only). Both: batched == per-column
bitwise over the full fixture decks at 4 chunk sizes, end-to-end
max_ulp 0 vs the oracle at batch width, ≥50k-column replica
determinism, honest VRAM pricing (`*_batched_vram_bytes`, estimate ==
measured pool peak), local-frame audits (worst frame:
`rlw_rtrn_march` 2048 B/thread ⇒ ~510 MiB machine-wide reservation;
every SW kernel 0 B after the spcvmc workspace restructure — which
converted a hidden ~1.65 GiB lmem reservation into pool-priced
transient VRAM). All dual-run.

## 7. Audit fold-in (LW + SW xhigh audits, 2026-07-27)

Dispositions: LW-1 (FP64 mcica quarantine) — built that way, plus
direct device-vs-Fortran assertions added. LW-2 — packaged-file →
parser → chain → `out/*` end-to-end gate added
(`test_end_to_end_from_packaged_data_file`). LW-3 — numpy>=2 pinned in
pyproject + import-time fail-closed guards in
`rrtmg_lw/rrtmg_sw/rrtmg_mcica` (NEP-50 weak promotion is part of the
bitwise contract). LW-4 — Fu `*1.0315` THEN cap `min(140, x)`:
docstring fixed; ported code already correct and fixture-gated.
LW-5 — multi-column gates shipped with the engines. SW-1 — CUDA
compositions reject `aer_opt != 0`. SW-2 — raw div/sqrt non-subnormal
invariant added to the `rrtmg_sw.cu` header contract; no in-kernel
assert (a kernel edit would force a full re-gate; the route is closed
at the API). SW-3 — `test_reftra_vrtqdr_sw` now parametrizes over
`RT_CASES` (both archives), so the seven synthetic RT taps gate (they
pass). SW-4 — night-GSW docstring corrected (see §3).

Fixture blind spots (recorded, not blockers; they become real if
layering/p_top handling ever changes): laytrop==50 in all 179 LW
fixtures (p_top=100 hPa + 4 hPa buffer construction); LW band-15
specparm edges and SW bands 22/24 `specparm>0.875` unexercised;
`taua` all zero everywhere; LW secdiff lower clamp never reached; SW
`zdenr ±1e-8` guard and conservative `ze2==1` reset unhit; band-16
laytrop+1 crossing and snow==130 exact boundary unisolated. From the
prep+ozone audit: `resnow==130` exact edge, `cldfra==0.01` equality,
and smallest-positive-coszen also unisolated; the LW and ozone fixture
decks live outside the repo (skip-gated) — a repo-self-contained
mini-fixture subset so those gates can never silently skip forever is
recorded as follow-up work. Night diagnostic bundle: WRF zeroes the
SWUPT.. optionals only IF PRESENT; gpuwm always materializes the full
bundle — all-present is the adapter contract.

## 8. Assets and restart identity

Packaged, SHA-256-pinned (PROVENANCE.md): `RRTMG_LW_DATA`,
`RRTMG_SW_DATA`, `ozone{,_lat,_plev}.formatted`. Restart asset roles
`wrf_rrtmg_lw_data`/`wrf_rrtmg_sw_data` activate with the legacy
variant. numpy>=2 is a hard dependency of the FP32 contract.

## 9. Documented divergences vs the CPU reference (ifx WRF)

1. **Transcendental libm class**: `radconst`/`calc_coszen` use
   FP32-ordered arithmetic with numpy transcendentals; ifx-WRF uses
   Intel's libm, gfortran-WRF glibc's — all differ at ~1 ulp. The SW
   fixtures pin the wrappers from `(xcoszen, declin, solcon)` inward
   (their `in/xcoszen` is the CPU run's own wrfout COSZEN — data, not a
   reproducible function), so no oracle defines "bitwise" here. Same
   class as the handoff's cross-compiler caveat. Terminator-adjacent
   day/night classification can differ per compiler in WRF itself.
2. **o33d on nests** (routing REDESIGNED per the prep+ozone audit):
   WRF computes o33d on d01 only (`o3input==2 .and. id==1`,
   `module_radiation_driver.F:1799-1823`) and hands nests
   parent-interpolated `o3rad`. The adapter reproduces that ROUTING
   STRUCTURE: the root adapter runs the (bitwise-ported) climatology
   chain on its own grid and retains the field; child adapters take a
   parent provider at construction and receive the parent's most
   recent o33d horizontally interpolated by gpuwm's certified nest
   helpers — a child never evaluates the climatology itself (gated).
   The residual divergence is the horizontal-interpolation arithmetic
   (gpuwm's vs WRF's), the same documented seam class as all nest
   interpolation.
3. **Radii carrier round-trip**: gpuwm state radii are µm-contract;
   the wrapper takes meters and multiplies by 1e6. `x_µm × 1e-6 × 1e6`
   can differ from `x_µm` by ≤1 ulp where the µm value was not itself
   born as `x_m × 1e6`.
4. **t8w**: transcribed from phy_prep, but gpuwm's geopotential/z
   carry gpuwm's own dynamics history — matched semantics, not proven
   byte-parity (same status as every other model-state seam).

## 10. Booby-trap boundary (the wiring proof)

Tripwired as "host/NumPy compute leaves": every NumPy chain routine in
`rrtmg_lw.py` (inatm, cldprmc, setcoef, every `_taugbN`, taumol,
rtrnmc, rrtmg_lw) and `rrtmg_sw.py` (inatm_sw, setcoef_sw, taumol_sw,
cldprmc_sw, reftra_sw, vrtqdr_sw, spcvmc_sw, rrtmg_sw), plus the NumPy
McICA generators on the forecast path. NOT tripwired: the driver-side
prep (`rrtmg_legacy_prep`, the Python twin of WRF's wrapper glue — it
has no CUDA twin by design) and the ozone/solar host pipelines. A
legacy-selected forecast must run with all tripwires armed and none
firing; withholding the device kernels must fire them; trapping the
batched engine entries must fail the forecast.

## 11. For the lead: seam-lane token reconciliation

`fix/cloud-radiation-seams` rewrites the same
`physics_compat`/`validate_run_config` token block this branch extends
(v1→v2 bump + `WRF_RRTMG_COMPATIBILITY_TOKENS` tuple vs. the added
`WRF_RRTMG_LEGACY`). Textual conflict is certain at merge; semantic
reconciliation: a token tuple carrying v1, v2, AND
`wrf-rrtmg-4-4-legacy-v1`, keeping the legacy pairing rules and the
seam lane's version-history comment. Not resolved unilaterally here.
Related caveat: restarts written before the Thompson radii-units fix
carry meter-scale `effc/effi/effs`; a legacy adapter radii plausibility
check will fail-closed on such states rather than radiate at clip
floors — resuming them requires the seam lane's migration story.

## 12. Cost (filled at timing)

Wall s/sim-min for the d01–d03 stack (radt 12/3/1), replicated-column
per-call methodology (Phase-A harness), same quiet-card window, same
process discipline, dual-run, `gpuwm.__file__` asserted into the
worktree:

| adapter | d01 s/call | d02 s/call | d03 s/call | s/sim-min | pool peak |
|---|---|---|---|---|---|
| RTE+RRTMGP (chunk 3125) | 2.914/2.899 | 11.438/11.597 | 14.628/14.489 | **18.68 / 18.60** | 3.44 GiB |
| legacy RRTMG (engine defaults) | 5.396/5.359 | 21.905/22.196 | 27.032/27.525 | **34.78 / 35.37** | 5.35 GiB |

Legacy ≈ 1.9× the same-window RRTMGP — inside Phase-A's 21–42
s/sim-min port estimate. The chunk lever (RRTMGP's ~5× headroom at
12500) applies to both and is untuned here; machine-wide VRAM never
exceeded ~8.4 GiB during timing (incl. ~2.3 GiB external).

## 13. Certification record (2026-07-28)

Full merged suite at `77533bb1` (single unlocked pass, superseded for
GPU content): 6 failed / 3242 passed / 16 skipped in 35:03. Of the six:
two were fixed at `95eda9b7` (the legacy container-pin update and a
PRE-EXISTING order-dependent leak in `test_rrtmgp.py` — `np.asarray`
aliasing the lru-cached `solar_source` before an in-place scale,
surfaced by the new files changing collection order); four are the
documented pre-existing preflight CQ-accounting golden pins inherited
from campaign base `4d2ce99` (delta exactly 510,408,784 B, recorded in
`f4b39b93`; trunk/campaign reconciliation, not this lane).

That unlocked pass overlapped another lane's locked GPU phase and the
machine crossed the 29,500 MiB rail (31,884 MiB observed by the lead),
so its GPU content was re-certified: all 61 CuPy-capable test files,
dual-run, under the shared mkdir-mutex GPU lock
(`tmp/gpu.lock` + owner file), with a 5-second VRAM watchdog:

* run 1: 2 failed / 1870 passed / 1 skipped (19:47)
* run 2: 2 failed / 1870 passed / 1 skipped (19:34) — identical tally
* both failures = the two pre-existing preflight pins above
* watchdog: 462 samples, machine-wide max 29,240 MiB < 29,500 — RAIL
  HELD (the near-rail peak is the preflight `--alloc` residency probe's
  own deliberate allocation, under its own headroom logic)

Every charter proof (booby-trap trio, restart round-trips, A/B digest,
cost pair) completed BEFORE the contention window: their launches
recorded 2.2–2.3 GiB machine-wide, the timing pair's own samples peaked
at 8.4 GiB, and the A/B digest ran in the same quiet window. They stand
as reported. Process rule adopted: every CuPy-capable pytest
invocation acquires the GPU lock first and queues behind other lanes;
detached GPU runs leave a completion sentinel checked on resume.

F′ certified tip: `95eda9b7`.
