# Physics options and maturity

Every physics scheme in ArWen is a transcription of WRF v4.6.1 source
(commit `d66e442f`), and every option carries a machine-readable
maturity label in the physics registry
(`gpuwm/physics_registry_v2.json`) -- the registry, not this page, is
the authority, and `tests/test_registry_reachability.py` keeps the two
from drifting. Labels never substitute for the evidence behind them;
each row below links the strongest measurement that exists for that
option.

## Maturity vocabulary

| label | meaning |
|---|---|
| **model-validated** | A matched multi-hour ArWen-vs-WRF forecast of the reference case has been run with this option and its decay tables are published ([VERIFICATION.md](VERIFICATION.md)). |
| **validation-candidate** | Executable and gated, with a ratified reference comparison, but deliberately not the default; the next candidate for full matched-run validation. |
| **supported** | Production option from the longest-certified slice: WRF-transcribed, standing unit/runtime gates, exercised by the certified reference configurations. |
| **experimental-runtime** | Executable, and carrying a documented runtime restriction or an unratified composition -- a table-bound runtime, or a nest edge between two microphysics schemes. Selecting it warns and does not block. |
| **implemented-unverified** | Runs on the GPU and is column-oracle-measured against unmodified WRF Fortran, but no dedicated ArWen/WRF forecast-trajectory comparison exists for it yet. The registry records its measured ULP distances and open divergences verbatim. |
| **planned / port-in-progress** | Not selectable. The registry publishes the target so the roadmap is machine-readable; nothing can resolve to it. |

An option that is not implemented is unreachable by construction -- a
config asking for it fails loudly at load, never silently substitutes.
The one exception is `gpuwm import-namelist`, which performs three
explicit, reported substitutions (see the end of this page).

## Microphysics (`mp_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| Kessler | 1 | supported | warm-rain certified slice; idealized + runtime gates |
| WSM6 | 6 | supported | certified slice; matched-run anchors exist for this scheme on the reference case (refl corr 0.977 at F2, 0.815 at F5, d03) |
| Thompson | 8 | **model-validated** | full matched 6 h, 4-domain run to 500 m; decay tables in [VERIFICATION.md](VERIFICATION.md); WRF's own coefficient tables packaged and SHA-256-validated at load |
| Morrison 2-moment | 10 | implemented-unverified | 28-column oracle vs unmodified WRF `MP_MORR_TWO_MOMENT`: theta within 154 ULP, but hydrometeor fields cross branch points and are not bitwise; both rimed-ice identities (graupel/hail) implemented |
| NSSL 2-moment | 18 | **validation-candidate** | full CUDA port with fused-process oracles and a ratified 500 m comparison; explicitly not the default |
| Thompson aerosol-aware | 28 | implemented-unverified | 22 WRF column fixtures end to end, 23 quantities each: 17 clear a flat 2e-6 gate, 4 do not, 1 clears only under two named allowances (numbers below); it runs multi-step and stays bounded; the one matched WRF forecast comparison is idealized only — a single-domain doubly periodic warm bubble, [validation/mp28-matched-trajectory.md](validation/mp28-matched-trajectory.md), which publishes a failed declared condition alongside a control showing that condition fails for WRF against its own recompilation — and no real-data or nested forecast has ever been validated against WRF; reachable only as a per-domain override |

### Thompson aerosol-aware (`mp_physics = 28`) — read this before selecting it

mp=28 is a second, near-complete port of Thompson, not a flag on mp=8.
Cloud droplet number `nc` becomes prognostic (WRF's `Nt_c = 100e6`
constant disappears), two aerosol number tracers `nwfa`/`nifa` and two
surface emission fields `nwfa2d`/`nifa2d` are added, and about a dozen
processes with no mp=8 counterpart run: CCN activation, aerosol-only
droplet evaporation, five cloud-number sinks under a shared balance
limiter, six wet-scavenging rates, DeMott ice nucleation replacing
Cooper, Koop homogeneous haze freezing, and number-weighted cloud
sedimentation. It lives in its own CUDA translation units and its own
Python launchers; `thompson.cu` and `thompson.py` are byte-frozen and
mp=8 is unchanged by construction.

**Exactly what has been measured.** All 22 committed WRF v4.6.1 aerosol
column fixtures, driven end to end through the shipped adapter, compared
against unmodified `phys/module_mp_thompson.F` (gfortran 13.3.0 `-O2`),
one RTX 5090, FP32. Twenty-two, not nineteen: the 19 scenarios the port
spec names (`aero-*`) plus three `wp08-*` columns from the same oracle
build that pin every reachable `nu_c` and both branches of the terminal
phase cleanup. Each fixture is compared on **23 quantities** — 15
prognostic column fields, seven surface accumulations (`RAINNC`,
`RAINNCV`, `SNOWNC`, `SNOWNCV`, `GRAUPELNC`, `GRAUPELNCV`, `SR`) and
`REFL_10CM` against WRF's own `calc_refl10cm` — at **2.0e-6 relative**
and **2.0e-4 dB**. The gate asserts that width, so it cannot be narrowed
back. The registry's residual table is published in the narrower 16-field
shape (the 15 column fields plus `RAINNC`); the gate compares all 23.

| result | fixtures |
|---|---|
| clear a **flat** gate on every one of the 23 quantities — no bounds dict, no excluded level, no per-fixture carve-out | 17 of 22 — `aero-ccn-activate`, `aero-ccn-sweep`, `aero-drop-evap`, `aero-ice-demott-dep`, `aero-ice-demott-idxin`, `aero-ice-koop`, `aero-init-profile`, `aero-nc-accrete`, `aero-nc-auto`, `aero-nc-cap`, `aero-nc-effrad`, `aero-nc-sed`, `aero-scav-frozen`, `aero-scav-rain`, `aero-sfc-emit`, `aero-warm-overlap`, `wp08-melt` (16 of the 19 spec'd `aero-*`, plus one `wp08-*`) |
| clear only under a named allowance | 1 — `aero-reduces-to-classic`, taking the gated count to 18 of 22, and it now rests on ONE allowance rather than two. See the allowance table below |
| miss the gate | 4, listed field by field below |

**The one allowance, on that one fixture, and needed for it.** Nothing
here was ever widened, and two of the three this page used to carry have
been **retired** rather than kept:

| allowance | what it does | was | is |
|---|---|---|---|
| `_NEAR_CANCELLATION_LEVELS` | 0-based level 6 held to 32 ULP of the entry value instead of a relative bound | level skipped outright | **32 ULP; measured 0.585 (`qr`) and 0.159 (`nr`), down from 14.9 and 4.05 at the 1.4.1 merge** |
| ~~`_END_TO_END_BOUNDS`~~ | `nr` held to 1.0e-5 instead of 2.0e-6 | 2.5e-3, then 1.0e-4 for `qr` and `nr` both, then 1.0e-5 for `nr` alone | **RETIRED at the 1.4.1 merge. Level 5's `nr` fell from 5.700e-06 to 4.146e-07, inside the flat 2.0e-6 gate, so the bound bought nothing and the constant is now an empty dict** |
| ~~`_REFL_DB_BOUNDS`~~ | reflectivity held to 1.0e-3 dB instead of 2.0e-4 dB | 1.0e-2 dB, then 1.0e-3 dB | **RETIRED. The residual it existed for fell to 3.242e-05 dB and then to 9.537e-06 dB, inside the flat 2.0e-4 dB gate, so the fixture is held to the flat gate on reflectivity** |

**What the 1.4.1 merge changed, and what it did not.** mp=28 sits on
`integration/release-1.4.1` and inherited the mp=8 lane's two rain
sedimentation reconciliations — `5e4af4e3` and `cb765336` — in the
byte-frozen `thompson.cu` this scheme shares for rain fallout. No mp=28
file changed. Its measurements did: `aero-reduces-to-classic` `nr` at
level 5 5.700e-06 → **4.146e-07**, that fixture's worst ULP over all 23
quantities 27.5 → **4.0**, its worst |dBZ − WRF| 3.242e-05 dB →
**9.537e-06 dB**, and the `qr` levels where mp=28 is bit-exact against WRF
and the frozen mp=8 pipeline is not went from three to **four** (1, 2, 4,
5). Nothing moved the other way, and the four missing fixtures below miss
at the same numbers they did before.

`aero-reduces-to-classic` now measures **`nr` 4.146e-07 at 0-based level
5** — inside the flat gate — and nothing on the fixture is above the flat
gate once level 6 is taken in ULPs. That level 5 is the one level of the
column where the step removes a large fraction of the rain number without
emptying it (49.75%: 3.000000e+05 → 1.507546e+05 per kg); it used to carry
5.700e-06, 27.5 ULP of the entry value, and the bound that covered it is
the one the merge retired. Every level of the column is now inside 4 ULP.
It is **not** inherited from the classic path and is
no longer claimed to be: with WRF's level-wise sedimentation density
restored, mp=28 is now **bit-exact against WRF** at `qr` levels 1, 2 and 4
and at `nr` level 1, where the frozen mp=8 pipeline on the identical entry
column is 7.61e-05, 4.00e-05, 6.27e-05 and 4.63e-05 away; over levels 0-7
mp=28 is at least as near WRF as mp=8 at every level and strictly nearer at
seven of eight. (An earlier revision of this page said the residual *was*
carried by the shared warm-rain path; that claim was true of the tree it
was measured on and is now false, so it is replaced rather than edited.)
The level-6 allowance is a *metric* change, not a looser tolerance: that
level enters with `qr` = 3.1695777e-07 kg/kg and evaporates 99.958% of it
in one 10 s step, so the survivor is the difference of two nearly equal
float32 numbers and a relative gate there measures the rate's error
amplified by 1/(1 − 0.99958) = 2370. mp=28 produces 1.3384e-10 there
against WRF's 1.3426e-10. With that level left *in* on a relative metric
the fixture reads `qr` 3.155e-03 / `nr` 3.155e-03, which is the number the
unexceptioned table reports and the reason the allowance is named.

| fixture | quantities that miss, with the measured maximum relative difference |
|---|---|
| `aero-cold-overlap` | `qc` 1.000e+00, `nc` 1.000e+00, `effc` 8.102e-01 — **one mechanism, three views, and it is a one-ULP disagreement wearing a full-scale number.** At 0-based level 4 WRF ends the step with 1.4551915228366852e-11 kg/kg of cloud water (exactly 2⁻³⁶, exactly 1.000 float32 ULP of the 2.3252160e-04 kg/kg the level entered with) and `nc` = 1.8333361 per kg, while ArWen ends at exactly zero; `effc` then reports 8.102e-01 because with no cloud water ArWen takes the 2.49 µm floor while WRF's remainder gives 1.31176e-05 m. Recorded as a MISS rather than allowanced. Separately and genuinely: `nr` 1.261e-04, `qr` 4.443e-05 at level 6, where the rain number falls 255.407 → 0.0739 per kg (99.97% consumed) and the difference is 0.611 ULP of the entry value (`qr`: 1.789 ULP) |
| `aero-cloud-freeze-nc` | `qc` 4.926e-06 — the fixture's only surviving row, at level 4, where 98.2% of the entry cloud water is frozen away and the survivor differs by exactly 1.000 ULP of the 8.247212e-05 kg/kg entry value. It is the SECOND float32 rounding of `qc` inside one step: WRF rounds once at `module_mp_thompson.F:3975` from a `qcten` carrying the source network and the condensation together, while ArWen applies the source network to `qc` and then applies the condensation to the already-rounded value |
| `wp08-nusweep` | `qr` 4.642e-06 — 2.3x the gate, at level 12, created from exactly zero and reaching 2.242e-11 kg/kg; the absolute difference is 1.04e-16 kg/kg. **This cell is ill-conditioned, measured, and no FP32 implementation can hold it to the gate.** Perturbing the level's entry cloud water by ONE float32 ULP moves the exit `qr` by 128 ULP (up) or 32 ULP (down); perturbing the entry droplet number by one ULP moves it by 256 ULP either way. The measured disagreement is 60 ULP — *smaller* than a single-ULP input change produces — so it is consistent with a sub-ULP difference in an intermediate that FP32 cannot represent. The 2.0e-6 gate at that level is ~26 ULP, i.e. below the cell's own condition number. Level 12 is also the only level of this column where the droplet number **rises** across the step (1.417475e+08 → 1.428941e+08 per kg), the signature of the number-weighted cloud sedimentation feeding it from the level above while autoconversion drains it |
| `wp08-freeze` | `nr` 2.724e-06 — 1.4x the gate, at level 0, created from exactly zero. **Attributed, un-attributed, measured, and then narrowed by the fix it prompted: 29 of its 34 ULP are the rain-presence gate, and what is left of that gate's disagreement is that the fallout kernel cannot see WRF's post-evaporation rewrite of `rr(k)`.** The shipped gate at `thompson.cu:450-452` is WRF's conjunction -- the TAU+1 test on the mixing ratio (`module_mp_thompson.F:3236`) with the R1 floor (`:3252`), then `rr(k) .gt. R1` (`:3616`) -- so it compares a mass concentration, as of `cb765336`. At 0-based level 1 of this column ArWen sees qr = 8.526513e-13 (closed, so it inherits the level-above fall speeds) and WRF sees rr = 1.174815e-12 (open, so it computes a real one 5.3x slower in number). Forcing ArWen's gate open on the shipped kernel moves level 0's `nr` from 34 ULP away from WRF to 5, while a mass change of the same size that does *not* flip the gate leaves the output bit-identical — the control that makes it a measurement (`tests/test_thompson_aerosol_adapter.py::test_the_wp08_freeze_residual_is_the_presence_gates_units_measured`). This page published the attribution as falsified for part of 2026-08-01; that falsification read `qr1d + qrten*DT` at the end of the step and took it for the value WRF's `:3236` tested, but the rain-evaporation block at `:3501` subtracts from `qrten` in between — instrumented WRF records `L_qr = .true.` there, so `:3236` took its true branch and `rr` was never floored to R1. `cb765336` DID reconcile the gate's units -- that is what it is -- and it did not move this residual because this level is in the one class that commit enumerated and left standing: `qr <= R1 < qr*rho` with `L_qr` true, 3 627 level-visits of the 2 h 12 km forecast it measured over. WRF's `L_qr` opens the evaporation block at `:3501`, whose rewrite at `:3568` leaves `rr(k)` = 1.174815e-12 above R1; ArWen's fallout kernel sees only qr = 8.526513e-13, fails the TAU+1 test, and floors rr to R1. NOT FIXED, and the reason is now a named change rather than a frozen file: closing it means carrying `L_qr` itself from the evaporation kernel to the fallout kernel, which nothing in the tree does. (An earlier revision of this row said the gate still compared a mixing ratio and that the kernel file was byte-frozen; both were true when the attribution was written and neither survived `cb765336`, so they are replaced rather than edited.) The residual <=5 ULP left over is not separately attributed |
| `aero-reduces-to-classic` | **clears the gate under the one allowance above** and is listed here only because the flat gate is the yardstick this table uses: level 5 is inside the flat gate now (`nr` 4.146e-07), and what remains is `qr`/`nr` 1.238e-04 at level 6 if that level is measured relatively rather than in ULPs |

**No surface accumulation misses any more, on any fixture.** All seven are
compared separately and all seven are now inside the flat gate on all 22
columns: `RAINNC`, `RAINNCV` and `SR` are **bitwise identical to WRF on
every fixture in the deck** (0.000e+00 relative), and `SNOWNC`/`SNOWNCV`
and `GRAUPELNC`/`GRAUPELNCV` peak at 6.285e-08 and 7.062e-08. Earlier
revisions of this page published `rainnc` / `rainncv` / `sr` residuals on
two fixtures and explained them as *one number seen three ways* —
`RAINNCV` is the fallout sum (`module_mp_thompson.F:1298`), `RAINNC`
accumulates it from zero on a first call (`:1299`), and `SR` divides the
exactly-agreeing frozen part by it (`:1308`). The explanation was correct;
the residual is gone, because the sedimentation density it came from was
found and fixed rather than tolerated.

Every surviving residual now sits where the field is either **created from
zero** inside the step or driven to **near-total consumption** — the
regimes where a fixed relative gate measures the amplified rounding of a
difference rather than the rate that produced it. That is stated, not
used: no bound above is relaxed for it. The whole 22 × 23 table is also
published denominated in float32 ULPs of `max(|entry|, |WRF after|)`, and
the two units disagree about which cell is worst, which is the point:
`aero-cold-overlap`'s two full-scale relative rows are **1.000 and 0.229
ULP**, while its `effc` row is **1.1e+07 ULP** — the signature of a branch
that flipped, not of a rounding. The worst ULP figure anywhere *inside* the
relative gate is 20. Four of the 22 fixtures are bit-exact on every one of
the 23 compared quantities.

**What changed in THIS revision, and why the count moved from 15 of 22 to
17 of 22.** Two fixtures were closed outright and one number grew. WRF
forms the working rain mass and number that sedimentation consumes
**twice**: at `module_mp_thompson.F:3237-3238` from the `:3193` τ+1
density, for every level carrying rain, and again at `:3568`/`:3570` from
the `:3490` post-condensation density — but only inside the `:3501-3502`
gate. ArWen was writing the post-condensation density unconditionally, so
every level got the `:3568` answer including the levels WRF never rewrote.
Restoring the level-wise choice closed `aero-drop-evap` (`rainnc` /
`rainncv` 5.165e-04 → 0.000e+00, `qr` 3.533e-05 → 7.35e-08, `nr`
2.258e-05 → 3.92e-07) and `aero-ice-demott-idxin` (`sr` and `rainnc` /
`rainncv` 1.279e-04 → 0.000e+00, `qr` 2.894e-05 → 6.42e-07), took
`aero-cloud-freeze-nc` from six rows to one (`qr` 2.800e-05 → 8.97e-08,
`nr` 1.797e-05 → 2.59e-07, `rainnc` / `rainncv` / `sr` 1.162e-05 →
0.000e+00), took `aero-reduces-to-classic`'s reflectivity 5.283e-04 →
3.242e-05 dB — which is what **retired** the third allowance — and took
that fixture's `qr` 7.813e-05 → 1.788e-07 and `nr` 4.832e-05 → 5.700e-06,
which is what let its bound go 1.0e-4 → 1.0e-5 with the `qr` entry
deleted. A second change pinned the contraction of WRF's terminal
`q1d(k) = q1d(k) + qten(k)*DT` apply (`:3973-4023`), which the `gfortran
-O2` oracle cannot fuse and nvrtc did; that made `aero-nc-cap`'s `qc`/`nc`
and `aero-ice-demott-idxin`'s surface accumulators bitwise exact. **It also
cost one cell, published as grown rather than absorbed:**
`aero-cold-overlap`'s `qr` at level 6 rose 3.667e-05 → 4.443e-05 (1.477 →
1.789 ULP of the entry value). Reverting that single pin restores the old
number and simultaneously un-does four of the improvements above, two of
them exact, so the pin is kept: it makes the instruction sequence WRF's at
every cell rather than at the cells that flatter the table.

**What changed in the revision before that — and the biggest change was a
correction to ArWen's own measuring stick, not to ArWen's physics.**
`aero-ice-koop` was published by the registry, this
page, `PROVENANCE.md` and the evidence document, for four waves, as *the
largest genuine physics gap in the port*, at `qi` 1.612e-03 / `ni`
1.764e-03 / `effi` 5.093e-05. It now measures 1.534e-07 / 3.396e-07 /
1.886e-07 and clears the flat gate. **No kernel changed. The oracle
harness did.**
`tools/thompson_wrf461_oracle/run_column_aero.F90` built the Exner
function with `rd_over_cp = 287.0/1004.0`, while WRF's own `rcp` is
`r_d/cp` with `r_d = 287.` and `cp = 7.*r_d/2.` = 1004.5
(`share/module_model_constants.F:19,:20,:31`) — i.e. exactly `2/7`, and
4774 float32 ULP away from what the harness used. A fixture records
pressure and temperature; the adapter has to solve for the float32
`theta` that reproduces that temperature bitwise under ArWen's own
`RCP`. Across two Exner constants 4774 ULP apart, that solve does not
survive: **47 of the deck's 528 entry levels admit an exact `theta`
under one constant and none under the other** (first published as 40,
the original environment's libm; re-pinned 2026-08-03 — the correction
note in `mp28-column-evidence.md` §3.4 records both counts and why the
number is the host libm's, not the deck's). At those levels the
adapter perturbed the entry pressure by up to 15 ULP to recover the
recorded temperature, and a perturbed entry pressure drives genuinely
different microphysics — on seven fixtures, `aero-ice-koop` among them.
The port was being measured against a yardstick that was not WRF's, and
correcting the yardstick — regenerating the deck with the kernels
untouched — is the whole of the closure. Homogeneous haze freezing was
never measured to be wrong.
`tests/test_physics_md_aerosol_claims.py::test_the_committed_deck_carries_wrfs_own_exner_constant`
re-derives exactly that from the committed CSVs, which is the half a
reader can check without rebuilding anything: all 528 levels of the
shipped deck invert exactly under WRF's `r_d/cp`, and 47 of them do not
under `287.0/1004.0`. The other half — that regenerating under the
correct constant is what moved the numbers, with nothing else changed —
is the differential rebuild recorded in
`tools/thompson_wrf461_oracle/README-AEROSOL.md`.
Against that, `aero-cold-overlap` acquired the level-4 rows above and is
worse than it was, and the two `wp08-*` residuals had never been published
at all because the registry counted 19 fixtures while the gate drove 22.

The gate is `tests/test_thompson_aerosol_adapter.py::test_g3_end_to_end_against_all_nineteen_oracle_fixtures`
(the name still says "nineteen"; it globs the fixture directory and drives
all twenty-two) and it is **honestly red**. The same numbers are published on the
registry option (`extensions.column_oracle_evidence`), and
`tests/test_physics_registry.py::test_mp28_published_residuals_still_equal_a_live_adapter_measurement`
re-measures them against a live adapter run, so nothing has to trust
this page. Beneath it, the per-kernel column gates and the
device-helper probes against a Fortran probe harness pass, including
bitwise agreement on the effective-radius branches.

**No REAL-DATA or NESTED forecast has ever been validated against WRF, and
none can be yet.** WRF's own `real.exe` is a fatal error on
`wif_input_opt = 0` with `mp_physics = 28`, and ArWen has no aerosol lateral
boundary condition — the depletion front advances at 0.993 of the wind speed,
so a 100 km nest sits at WRF's aerosol floor within 83 minutes. That, not an
absence of running, is what holds the label at `implemented-unverified`.

**A matched IDEALIZED trajectory does now exist**, and it publishes its own
failed gate:
[validation/mp28-matched-trajectory.md](validation/mp28-matched-trajectory.md).
A doubly periodic single-domain warm-bubble forecast, 120 × 120 × 40 at
dx = 2 km for 7200 s, with ArWen initialised *from WRF's own `wrfinput_d01`*
(t = 0 field difference exactly zero in ten fields of thirteen, and one
float32 rounding — 2.5e-09 to 2e-08 — in the three ArWen derives rather than
copies) and run against unmodified WRF v4.6.1
built twice from identical source. Of the four conditions declared before the
runs, three pass and **V3 fails** — but the same condition also fails when WRF
is compared against its own recompilation with one optimization flag changed,
so past t ≈ 2400 s the case is chaotic and V3 was mis-specified. The
measurements that do discriminate are favourable: mp=28's per-step
disagreement with WRF is the disagreement mp=8 already has (1.797e-02 versus
1.800e-02 RMS in `w` after five steps from a mature state), and the domain
aerosol budget matches WRF's to 1.530e-04 over two hours.

mp=28 has also been integrated multi-step against itself:
`tests/test_mp28_forecast_smoke.py` runs 150 steps × 12 s on a specified-BC
convective domain, and a second gate carries the same domain to 600 steps
(7200 s, 2.6 domain ventilation times) with 0 non-finite values, 0 bound
violations and 0 specified-zone ring violations across all 600 microphysics
calls. Finite and bounded is not correct: a scheme with a systematically
wrong activation rate passes every one of those checks for two hours. The
bounds are WRF's, but they are *clamps*, not answers. Full evidence grading, including which claims
rest on ArWen-written Fortran drivers rather than on WRF's own answer, is
in [validation/mp28-column-evidence.md](validation/mp28-column-evidence.md).

#### What the aerosol initial condition is worth

WRF's `thompson_init` fills a *synthetic* CCN/IN profile whenever the
aerosol arrays arrive unset: CCN at `module_mp_thompson.F:493-515`, ice
nuclei at `:531-551` (two independent `MAXVAL` presence tests, `:493` and
`:531`, so a domain can get one fill and not the other), and the surface
emission `nwfa2d` derived from the filled surface value at `:510`. It
decays with height towards `naCCN1` = 50.0e6 aloft from a lowest-level
value of `naCCN1 + naCCN0*exp(-(dz1/1000)*niCCN3)` (`:508`, constants at
`:96-97`) — not the bare `naCCN1 + naCCN0` = 350e6 ceiling, because the
exponent carries the first layer's thickness. On the `aero-init-profile`
fixture (lowest level 250 m, 500 m layers) WRF itself fills 1.478987e+08
at the bottom and 5.000000e+07 at the top, and 1.254902e+06 /
5.000002e+05 for the ice nuclei.

ArWen ports that fill exactly, as `gpuwm.core.microphysics.microphysics_init`,
gates it against WRF's own post-`thompson_init` snapshot
(`tests/test_thompson_aerosol_state_gpu.py::test_init_profile_matches_wrf_fixture_101`),
and **calls it**, once per domain, from
`gpuwm/core/physics.py::initialize_physics` — ArWen's
`module_physics_init.F`, the seam where WRF's `phy_init` calls `mp_init`
(`:1635`) and `mp_init`'s `CASE (THOMPSONAERO)` arm calls `thompson_init`
(`:4522-4538`). The call is presence-gated exactly as WRF's is, so a nest
that inherited its parent's aerosol and a restart about to be overwritten
by a checkpoint are both left alone. A freshly initialised mp=28 domain
therefore begins strictly **above** the terminal apply's clamps
(`MAX(11.1E6, ...)` for `nwfa`, `MAX(naIN1*0.01, ...)` = 5.0e3 for `nifa`,
`module_mp_thompson.F:3979-3982`), not pinned at them — 13.3x and 251x
above them respectively on the fixture numbers just quoted. Two tests hold
that:
`tests/test_mp28_forecast_smoke.py::test_microphysics_init_has_a_production_call_site`
scans for the call site and asserts there is **exactly one** (once per
domain silently becoming once per step would overwrite an advected,
activated and scavenged aerosol field with the synthetic one), and
`test_a_freshly_initialised_mp28_domain_carries_the_profile_not_zero`
asserts the filled state on a real `DomainState` taken through the real
`initialize_physics`.

**Until 2026-08-01 this was the port's largest measured error**: the fill
was implemented, proven and called by nothing, so every mp=28 run
integrated from `nwfa = nifa = 0` under WRF's floors. The measurement that
established the size of that gap is unchanged; what it means is not. It is
now the measured **sensitivity** of an mp=28 forecast to its aerosol
initial condition — two otherwise identical 150-step forecasts of the
convective case in
[validation/mp28-column-evidence.md](validation/mp28-column-evidence.md)
6.1, one taking the production init path and one with the profile removed:

| quantity | with the profile (what a run does today) | with it removed | change |
|---|---|---|---|
| initial mean `nwfa` | 6.653e+07 kg-1 | 0 | -- |
| final interior `nwfa` | 3.006e+07 kg-1 | 1.262e+07 kg-1 | at the floor |
| peak `nc` over the run | 1.598e+08 kg-1 | 2.844e+07 kg-1 | **5.6x fewer droplets** |
| domain-total `RAINNC` | 1.791 mm | 3.119 mm | **+74.1%** |
| peak `RAINNC` | 0.678 mm | 0.930 mm | +37.1% |

Both forecasts are re-run and this table rebuilt by
`tests/test_physics_md_aerosol_claims.py::test_the_published_aerosol_sensitivity_is_a_live_measurement`,
which compares at the precision printed here; the two runs are
bit-identical across repeats on the measurement machine, so nothing here
is a tolerance.

Read that precisely: removing the CCN loading a run starts from raises
domain-total surface rain by **74.1%** over half an hour, and cuts the peak
droplet count by a factor of 5.6. That is not a rounding difference; it is a
different forecast. It is also the magnitude the lateral-boundary deviation
below converges to, because after `L/U` the whole domain **is** the
right-hand column.

**Deviations from WRF you must know about before using it.** Every
bullet in *this* list is also a registry warning, printed whenever a
plan selects the option — `gpuwm/physics_registry_v2.json` is the
authority and this page is its summary. Two further things a reader
needs are in a second, separate list below, marked as such because
neither is an open deviation and neither is a registry warning:

- **No aerosol ingest.** There is no WIF metgrid stream, no GOCART
  climatology reader, and no black-carbon (`nbca`) species, so
  `use_aero_icbc`, `use_rap_aero_icbc`, `wif_input_opt`,
  `num_wif_levels` and `qna_update` are published unimplemented and
  refuse rather than being silently dropped. WRF's own fallback for
  exactly that case — `thompson_init`'s synthetic profile — is ported
  *and installed* (see the section immediately above), so the gap is the
  ingest lane, not the initial condition. `qnbca` is refused rather than
  zero-filled, and `taod5502d`/`taod5503d` (radiation-side aerosol
  optical depth) are not produced by `mp_gt_driver` at all.
- **WRF's own initializer refuses the configuration ArWen runs.**
  `dyn_em/module_initialize_real.F:2734-2736` calls
  `wrf_error_fatal('wif_input_opt=0 but mp_physics=28')`, so `real.exe`
  will not build a `wrfinput` for the no-WIF case at all — not even for
  the synthetic-profile variant WRF falls back to internally. The
  *physics* is WRF's line for line; the admission decision is not. A
  consequence worth stating plainly: an ArWen mp=28 run and a
  WIF-initialised WRF mp=28 run are **not** directly comparable, and a
  comparison between them must not be reported as one.
- **No aerosol lateral boundary condition, and this is the deviation
  that grows with run length.** On a specified (external BC) domain
  ArWen couples only `qv` from boundary snapshots and gives every other
  scalar flow-dependent boundaries with zero inflow, so aerosol-free air
  advects in at the upstream face and monotonically depletes
  `nwfa`/`nifa` for as long as the run continues — with no NaN, no
  negative and no health trip, because WRF's own terminal clamps
  (`nwfa >= 11.1e6`, `nifa >= 5.0e3` per m3) hold the floor. WRF's
  Registry gives `qnwfa`/`qnifa` real `bdy` arrays and forces them from
  the boundary file. **Measured** on a deliberately cloud-free run, so
  every kilogram lost is the boundary policy: with a 20.0 m/s inflow the
  depletion front advances at **19.8638 m/s** (0.99319 of the wind), and
  over 1800 s the domain-interior mean `nwfa` falls to 0.4566 of its
  initial value and `nifa` to 0.3363. At 10 m/s the front runs at
  9.808 m/s, so it is a law: *the upstream `U·t` of your domain is at
  WRF's aerosol floor after time `t`, and the whole domain after `L/U`*
  — 13.9 h for a 1000 km domain in a 20 m/s flow, **83 minutes for a
  100 km nest**. The only interior source is the fixed surface emission
  `nwfa2d` at k = 0, measured at 5540.14 kg-1 s-1, which replaces about
  5% of the lowest level's loading over 1800 s and acts on that level
  only. Separately, the `spec_zone` ring itself ends at *exactly zero*
  aerosol on three of its four faces (west/south/north 1.000, east
  0.125) because WRF's clipped microphysics tile means the terminal
  clamp never runs there — a value WRF itself cannot produce, and what a
  nest boundary or a `wrfout` reader sees. This matches ArWen's existing
  hydrometeor policy and is documented, not fixed.
- **MYNN does not mix the aerosol numbers.** ArWen passes
  `flag_qnc`/`flag_qnwfa`/`flag_qnifa` to MYNN as literal `False`, so
  `nc`/`nwfa`/`nifa` are never vertically mixed by the PBL. WRF mixes
  them at `bl_mynn_mixscalars = 1`
  (`phys/module_bl_mynn.F:4735,:4777,:4957`) or through
  `scalar_pblmix` (`phys/module_pbl_driver.F:2251`). At ArWen's pinned
  MYNN identity `bl_mynn_mixscalars = 0`, and WRF's `check_a_mundo`
  raises `scalar_pblmix` to 1 only when `use_aero_icbc` or
  `use_rap_aero_icbc` is set
  (`share/module_check_a_mundo.F:2477-2495`) — both of which ArWen
  refuses — so WRF's own value here is 0 too and today the two models
  agree. What differs is that ArWen's withholding is *structural*
  rather than a namelist value, and mp=28 is the first configuration
  in which those species carry real values and the withholding is
  physically visible. (Snow is a separate contract and is *not*
  withheld — see the second list below.)
- **Mixed mp=8 ↔ mp=28 nesting is refused by name.** An mp=28 domain
  may only sit under an mp=28 parent. No cross-scheme transition rule
  is registered, so the registry refuses the edge with
  `unsupported-component-transition` and
  `gpuwm/core/microphysics_transition.py` refuses it at runtime with a
  message that says why. WRF's own non-aerosol-aware fallbacks
  (`nc = 100e6/rho`, `nwfa = 11.1e6/rho`, `nifa = 5.0e3/rho`) would
  hand a nested child fabricated aerosol instead of its parent's, and
  no gate here would notice.
- **mp=28 and mp=8 are deliberately not bit-identical thermodynamics.**
  mp=28's `RSLF`/`RSIF` saturation Horner chains are
  contraction-pinned; mp=8's stay FMA-contracted. The two saturation
  vapour pressures therefore differ by one ULP — and
  `module_mp_thompson.F:3401` opens the whole condensation/CCN
  activation block on `ssatw > 1.E-15` (`:185`), so one ULP flips a
  branch. mp=28 matches WRF's own gfortran `-O2` arithmetic; mp=8 stays
  byte-frozen at its model-validated trajectory. Neither is a defect,
  and "make them agree" is the wrong fix in both directions.
- **`CCN_ACTIVATE.BIN` is distributed with ArWen, and a different copy
  is refused.** See the asset note below. It ships, so a clean checkout
  validates mp=28; if it is ever missing, every device gate for the
  scheme — including all 22 column fixtures — skips by name rather than
  passing, and the scheme itself fails closed rather than defaulting.
  A byte-different activation table is rejected, not used: it would be
  a silently different activation scheme.

**Two more couplings a reader needs, neither of which is a registry
warning** — one is a defect this port closed, the other is a coupling
that is correct and easy to assume is missing:

- **MYNN *does* mix mp=28's snow, and this was a real defect while it
  did not.** `FLAG_QS` is true for mp=28
  (`gpuwm/core/mynn_pbl_runtime.py::mynn_flag_qs`, whose
  `MYNN_SNOW_MICROPHYSICS` set is derived from the WRF Registry:
  `Registry.EM_COMMON:3036` declares `moist:qv,qc,qr,qi,qs,qg` for
  `thompsonaero`), so MYNN sees the snow field and its condensation
  `rh_hack` reads it. While 28 was missing from that set —
  `module_bl_mynn.F:734` and `:876` substitute `sqs = 0` and `:1324`
  skips `rqsblten` — the committed `snow_anvil` oracle column had
  `qi_bl` driven from 5.4863e-07 to exactly 0, taking `qc_bl`,
  `cldfra_bl`, `rqvblten`, `rthblten` and `exch_h` with it. Closed, and
  pinned in both directions by
  `tests/test_physics_registry.py::test_the_registry_flag_qs_contract_is_the_one_the_shipped_runtime_applies`.
- **mp=28 feeds the radiation, and it feeds it exactly the way mp=8
  does.** A prognostic `nc` only reaches a forecast's energy budget
  because `effc` reaches the cloud optics. WRF declares mp=28 with the
  same `state:re_cloud,re_ice,re_snow` inventory as mp=8
  (`Registry.EM_COMMON:3036` vs `:3024`) and lists `THOMPSON` and
  `THOMPSONAERO` in one `has_reqc`/`has_reqi`/`has_reqs` disjunction
  (`phys/module_physics_init.F:1005-1006`) — and those same three flags
  are the only gate on the block that *computes* the radii
  (`module_mp_thompson.F:1466`), so in WRF a scheme computes them if and
  only if radiation consumes them. ArWen's RTE+RRTMGP and legacy-RRTMG
  paths therefore both take Thompson's explicit-radius coupling for 28.
  A scheme that fell through to the Kessler default would not merely
  lose its radii: its ice and snow would stop producing cloud fraction,
  and an overcast ice cloud would radiate as clear sky. The one standing
  divergence here is ArWen-wide and not mp=28's — the adapter merges
  snow into a single mass-weighted ice species rather than carrying
  WRF's separate Fu snow species.

**How to select it.** mp=28 is registered with
`reachability.state = "component-override"`: it is reachable *only* as
an explicit per-domain microphysics override on the
experiment-per-domain tree route. No template selects it, it is no
template's default, and `gpuwm domain` is unchanged. That is
deliberate — an unverified scheme should be opt-in per domain, not
something a suite hands you.

Notes with teeth:

- Thompson's four table assets (`qr_acr_qg_V4.dat`, `qr_acr_qsV2.dat`,
  `freezeH2O.dat`, `thompson_aux_tables.dat`) are byte-validated
  against pinned SHA-256 values at launch.  The two smallest ship in
  the wheel; `freezeH2O.dat` (243 MiB) and `qr_acr_qg_V4.dat` (71 MiB,
  wheel-only exclusion -- a checkout carries it) are staged by
  `gpuwm fetch-tables` (the install scripts run it automatically)
  under the same pins.  `GPUWM_THOMPSON_TABLE_ROOT` can relocate the
  root but never bypasses the pins.
- **mp=28 needs a fifth table, and ArWen ships it.**
  `CCN_ACTIVATE.BIN` (35,288 bytes, sha256 `f2b8d391...`) is the CCN
  activation lookup.  It is the one Thompson artifact that is *not* a
  `thompson_init` product: `table_ccnAct`
  (`module_mp_thompson.F:5110-5166`) only reads it, and the numbers are
  offline parcel-model output (WRF's own comment at `:5102-5108`), so
  no recompilation of WRF and no ArWen code path regenerates it.  It is
  third-party data WRF redistributes; ArWen redistributes the identical
  bytes — WRF v4.6.1's own `run/CCN_ACTIVATE.BIN` under WRF's
  public-domain dedication, whose notice ships in
  `gpuwm/data/wrf_radiation/LICENSE-WRF.txt` — so a default install already
  has it in the table
  root.  `GPUWM_THOMPSON_CCN_ACTIVATE` (full path) or
  `GPUWM_THOMPSON_TABLE_ROOT` still bind a run to a copy in your own WRF
  tree instead.  Absence is fatal and never defaulted, and the size and
  SHA-256 are checked on every load, wherever it was resolved from.  It
  is deliberately outside the classic table set, so no mp=8 launch
  acquires a dependency on it.
- Thompson is the default template scheme: `gpuwm domain` emits
  `mp_physics = 8`, and the registry's declared default template
  (`thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1`) carries the same
  suite. Its warning states the one caveat verbatim: the matched-run
  decay tables were produced with the exact-port legacy RRTMG engine
  (`ra_rrtmg_variant = "rrtmg_legacy"`) to mirror the CPU reference bit
  for bit, while the default template selects the ratified RTE+RRTMGP
  substitution -- the same registry option 4/4 either way.
- Morrison's registry entry records why it is not bitwise (CUDA vs
  glibc transcendentals, FTZ at subnormal branches, FMA contraction)
  and what closing it would take. It was the default template scheme
  until 2026-07-29 and stays fully selectable at its maturity label,
  which states the verification gap honestly.
- Reflectivity (`REFL_10CM`) is computed on output-due microphysics
  steps with WRF's own 50-bin quadrature; the cold-start frame carries
  no reflectivity (no microphysics call precedes it), matching the
  registered deviation D2 in [PROVENANCE.md](../../PROVENANCE.md).

## Planetary boundary layer (`bl_pbl_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| YSU | 1 | implemented-unverified | 24-column oracle vs unmodified `bl_ysu.F90`: theta tendency 1 ULP, exchange coefficients 7 ULP, PBLH 1 ULP; momentum/moisture tendencies 4.2e-8 m/s2 / 3.1e-11 kg/kg/s (near-total cancellations); part of the model-validated reference suite alongside Thompson |
| MYNN (EDMF) | 5 | implemented-unverified | assembled driver bitwise on the warm step vs unmodified `module_bl_mynn.F`; 300-step coupled forecast gate; admitted only as the coupled 5/5 pair with the MYNN surface layer |
| Shin-Hong (scale-aware) | 11 | implemented-unverified | float32 CPU authority reproduces every output field of both `ctopo` arms at **max ULP 0** against the byte-frozen `module_bl_shinhong.F`, over 30 cases x 6 grid spacings x 40 levels; the CUDA mirror's heat tendency is bitwise (0 ULP through both tridiagonal solves), PBLH/WSTAR/DELTA 1 ULP, `EXCH_H` 8; and its resolved/subgrid partition was scored across a 3200-100 m ladder against pre-registered Honnert (2011) envelope bands -- every gated rung inside, 100 m LES anchor held ([receipts](receipts/grayzone/)) |
| SASE | none: ArWen-only, `bl_pbl_physics = 900` outside WRF's namespace | implemented-unverified, **permanently** | no WRF v4.6.1 counterpart, so no oracle comparison against WRF Fortran exists or can exist and this ladder cannot rank it; numerics self-checked; physics unvalidated -- 2 of 7 frozen acceptance bars met on a single reference case on a single day ([Selecting an experimental scheme](#selecting-an-experimental-scheme)) |

The YSU registry entry records four open items verbatim, including two
FTZ-class subnormal branch disagreements and the `topo_wind=0` driver
arm difference (up to 182 ULP on 6 of 960 fixture lanes). MYNN's entry
records four of its own, the sharpest being that its CUDA leaves are
not bitwise twins of the CPU references away from their oracle
fixtures, and that `phim`/`phih` are still evaluated on the host one
column at a time (a measured 125 µs per column).

Shin-Hong's entry records four of its own: the `q2xk(kpbl+1)`
out-of-bounds read WRF performs and ArWen deliberately does not; WRF's
own `prfac2 = 0/0` NaN reproduced rather than repaired; the sm_120
subnormal-flush branch, closed at the branch by a double-compare
countermeasure with 72 residual flush lanes pinned as counts; and the
CUDA mirror's near-total-cancellation lanes on the momentum and
moisture tendencies (the worst `du` lane is 3.4966e-06 against
3.4756e-06 m/s2). It is the only option in this table whose behaviour
*across grid spacings* has been measured against a published partition
— see [Receipts: gray zone](receipts/grayzone/) — and that measurement
is deliberately fenced: one idealized dry convective boundary layer,
six seeds, one card, one day. It is agreement with a published
similarity curve, not skill against observations, and it moved no
maturity rung. Scheme 11 remains `implemented-unverified` for the same
reason YSU and MYNN do — no matched ArWen-versus-WRF forecast
trajectory has been run with it.

Earlier revisions of this page said MYNN's registry entry recorded a
snow deviation — that snow mixing ratio was not passed to the
condensation RH adjustment (WRF's `rh_hack`). **That is no longer
true, and it was not true of the registry when it was written.**
`FLAG_QS` is derived from the WRF Registry package inventory and is
true for every snow-carrying microphysics ArWen ships (6, 8, 10, 18,
28), so MYNN receives the snow field; the mp=28 bullet above records
the defect that existed while 28 was missing from that set, and its
closure.

## Surface layer (`sf_sfclay_physics`)

| option | WRF id | maturity (registry) | reachability (registry) | notes |
|---|---|---|---|---|
| MM5 (classic) | 91 | supported | template | the certified-slice surface layer; pairs with YSU and all three LSMs |
| MYNN | 5 | implemented-unverified | template | column solver oracle-matched over land and water (max rel. err 4.3e-7); `isftcflx` 0-3 ported; only selectable as the 5/5 pair |
| MM5 (revised) | 1 | supported | unreachable | implemented in the tree, but no registered template selects it and no route allows the override (every verified run used the classic scheme), so no config can reach it |

Maturity and reachability are separate registry axes and are quoted
verbatim here: `maturity` is the option's evidence tier, while
`reachability.state` says whether any registered template or override
route can actually select it (`template` = a registered template
selects it; `unreachable` = nothing can, and the registry's blocker
text records why).

## Land surface (`sf_surface_physics`)

| option | WRF id | maturity | evidence, in one line |
|---|---|---|---|
| Noah (4-layer) | 2 | implemented-unverified | 42-column oracle vs unmodified `module_sf_noahdrv.F`: 7 of 31 outputs bit-identical incl. the whole TSLB profile; TSK within 2 ULP, HFX worst at 375 ULP; part of the model-validated reference suite |
| RUC (9-level) | 3 | implemented-unverified | column family oracle-matched vs unmodified `module_sf_ruclsm.F`; full device residency measured at production width (0.47 s per call at 360,000 columns, snow-free) |
| Noah-MP | 4 | implemented-unverified | `NOAHMP_SFLX` bitwise on all four whole-column fixtures; device slab path max ULP 0 vs the scalar authority at 360,000 columns; expert-route option pinned to the exact WRF Registry default option identity |

Divergences the registry states plainly (read the registry warnings
before relying on any of these over unusual surfaces):

- **Noah:** the glacial land-ice column (`SFLX_GLACIAL`) is not ported
  -- a domain with land-ice points must not use Noah; measured cost of
  ignoring this is listed per field in the registry. Frozen-ground
  infiltration on simultaneously-frozen-and-melting columns differs
  through CUDA vs glibc `expf`/`powf` (water redistributed, not lost).
- **Noah-MP:** glacier columns are refused during post-static
  initialization (not silently skipped); sea ice takes WRF's own skip.
  The WRF six-rate precipitation partition and radiation-cadence COSZEN
  carrier are active.
- **RUC:** uses WRF-ARW's `EM_CORE==1` species partition, lake bypass,
  fractional-sea-ice pre/post blend, and radiation-cadence GSW carrier.
  WRF's own uninitialized-`ilnb` read on thin snow (a real WRF defect:
  the value depends on grid traversal order) is *not* reproduced; ArWen
  passes the defined one-layer answer and documents the divergence.

## Radiation (`ra_lw_physics` / `ra_sw_physics`)

| option | selection | maturity | evidence, in one line |
|---|---|---|---|
| RTE+RRTMGP | 4/4 (default) | supported | the default modern k-distribution path; substitution token recorded in every config it touches |
| legacy RRTMG | 4/4 with `ra_rrtmg_variant = "rrtmg_legacy"` | verification tier* | line transcription of WRF v4.6.1 option 4/4: batched LW/SW engines bit-identical (max ULP 0) to the port oracle over the full fixture decks; McICA generators match every stored WRF Fortran mask; used for all matched-run comparisons |
| Dudhia SW (+ LW off) | 0/1 | supported | certified-slice shortwave |
| WRF RRTM + Dudhia | 1/1 | port-in-progress | not selectable; refused at load |

\* "verification tier" is this page's own term, not registry
vocabulary. Legacy RRTMG is not a separate registry component option:
it is the same registry option 4/4, selected by the
`ra_rrtmg_variant = "rrtmg_legacy"` token and kept for matched-run
verification against the WRF CPU reference. (The registry's
`rte-rrtmgp-legacy-aggregate` entry is a different thing entirely --
an RTE+RRTMGP route retained for a legacy aggregate selector, maturity
`implemented-unverified` -- not this port.)

The two 4/4 implementations are deliberately firewalled: a restart
written under one refuses to resume under the other, and the RTE+RRTMGP
path is byte-unchanged by the legacy port's existence. The legacy
variant implements exactly one WRF mode combination (`icloud=1`, McICA
maximum-random overlap, CAM ozone climatology, year-formula greenhouse
gases, zero aerosol); everything else fails closed rather than
approximating. Measured cost: legacy RRTMG is about 1.9x RTE+RRTMGP
wall time on the same three-domain stack (34.8 vs 18.7 wall-s per
simulated minute, radt 12/3/1, same card and window). Full dossier:
[rrtmg_legacy_integration.md](../rrtmg_legacy_integration.md).

## Cumulus (`cu_physics`)

| option | WRF id | maturity | notes |
|---|---|---|---|
| Kain-Fritsch | 1 | supported | outer (>=10 km) domains; packaged lookup table; cudt 5 min in the certified templates |
| Grell-Freitas (scale-aware) | 3 | implemented-unverified | whole GFDRV at the WRF v4.6.1 boundary, CPU and CUDA; per-domain override, no template selects it; runs on the model step (cudt pinned 0) |
| off | 0 | supported | the convection-permitting nests run with cumulus off |

What is certified for Grell-Freitas, and what is not. The certified
half: the entire driver -- column preparation with its mixed-precision
GFS constants, the deep cloud model, the shallow one, both `neg_check`
calls and the output algebra -- reproduces the byte-frozen WRF v4.6.1
`module_cu_gf_*.F` word for word at the GFDRV boundary over the
committed 216-column oracle (18 soundings x 6 grid spacings x 2
`ishallow` arms) on the 208 columns where GFDRV's own decomposition is
exact, with the 8 remainder bounded to the driver's own
`module_gfs_physcons` mixed precision (max 34 ULP, 3.8e-6 relative, no
branch flips). The CUDA path holds that boundary with the gamma
COMPUTED on the device: its transcribed glibc-2.39 float32
`tgammaf`/`lgammaf`/`expm1f`/`exp2f`/`powf` are bitwise against 130k
live-glibc words, which matters because one ULP of the beta-shape
normalisation moves the deep mass flux by up to 7.3 percent. The
registry entry records three deviations: the shallow `k22` trigger
ships with the section-offset indexing corrected (WRF's MAXLOC
off-by-one lives behind a parity-suite flag; measured on the fixture,
the correction moves 3 rejected cases and zero output words), the
inversion-layer search clamps WRF's out-of-bounds `t_cup(kend+8)` read
(clamp count zero on the fixture, asserted), and the engine seam feeds
the advective/boundary-layer halves of the forcing as zeros with
convective momentum tendencies not yet coupled -- the plumb-list any
label upgrade must clear. The uncertified half is the same sentence
YSU, MYNN and Shin-Hong carry: no matched ArWen-versus-WRF forecast
trajectory has been run with `cu_physics = 3`, and no real-case
verification receipt exists -- which is why the label is
`implemented-unverified` and the scheme is a per-domain override a
user must ask for by name, never a default.

## Map projections (`map_proj`)

Projections are not physics schemes and carry no physics-registry
entry; they are listed here because this is the maturity page and
their maturity differs by projection. The authority for the rows is
the binary64 projection oracle gate
(`tests/test_projection_oracle.py`, pinned per-quantity ULP ceilings
against unmodified WRF v4.6.1 `share/module_llxy.F`) and the
worldwide section of [VERIFICATION.md](VERIFICATION.md).

| option | WPS name | maturity | evidence, in one line |
|---|---|---|---|
| Lambert conformal, northern hemisphere | `lambert` | **model-validated** | the 3 April 1974 four-domain matched-run family runs on it; binary64 `module_llxy` oracle at the pinned ceilings |
| Lambert conformal, southern hemisphere | `lambert` | implemented-unverified | binary64 oracle rows (SH secant + SH tangent cones) at the pinned ceilings; Brisbane GPU smoke integration; no matched WRF run |
| Mercator | `mercator` | implemented-unverified | binary64 oracle rows (tropical, subtropical, antimeridian) at the pinned ceilings; Singapore and Fiji (antimeridian) GPU smoke integrations; no matched WRF run |
| Polar stereographic, either pole | `polar` | implemented-unverified | binary64 oracle rows (NH, SH, pole-anchored) at the pinned ceilings; Fairbanks GPU smoke integration; no matched WRF run |
| Latitude-longitude / rotated grids | -- | not implemented | refused at load, never substituted |

Domains containing or touching a pole and forcing footprints wider
than 180 degrees of longitude are refused for every projection
(genuine pipeline limits; the wizard and the loaders name them).

## The default template suite

`gpuwm domain` emits the reference-configuration physics with the
microphysics slot on the model-validated matched-run scheme: Thompson
(mp8, packaged hash-pinned tables), MM5 surface layer (91), Noah (2),
YSU (1), RTE+RRTMGP (4/4), Kain-Fritsch on the 12 km root only, the
49-level eta ladder, and the certified diffusion/damping/acoustic
settings. Morrison (mp10) remains fully selectable at its maturity
label. The acceptance forecast (6 h in 3.6 min on a 250x200x49 domain;
[FIRST-LIGHT.md](FIRST-LIGHT.md)) ran this template's Morrison variant,
the default at the time of that transcript. The matched-run validation
suite runs Thompson with the legacy RRTMG engine
(`ra_rrtmg_variant = "rrtmg_legacy"`) to mirror the CPU reference
exactly ([VERIFICATION.md](VERIFICATION.md)).

## Selecting an experimental scheme

One option on this page is experimental: SASE, the Scale-Adaptive
Stress-Energetics closure, a unified turbulence/PBL closure selected
with `bl_pbl_physics = 900`. 900 sits outside WRF's namespace on
purpose -- WRF's own `bl_pbl_physics` runs to 99 -- so it can never
collide with a scheme WRF adds later. The registry marks it
`reachability.state = "component-override"`: the prepared-domain-tree
runner admits it as a PBL component override, and the single-config
loader takes it directly. This is the whole of a configuration that
validates:

```toml
[grid]
nx = 128
ny = 128
nz = 64
dx = 1000.0
dy = 1000.0
ztop = 20000.0

[dynamics]
dt = 6.0
km_opt = 0
khdif = 0.0
kvdif = 0.0

[run]
run_seconds = 3600.0
moist = true
bl_pbl_physics = 900
sf_sfclay_physics = 91
bldt = 0.0
```

None of those companions is taste. Each is refused at config load --
refused, not warned about -- and the refusal states its own reason:

| requirement | why the closure has it |
|---|---|
| `km_opt = 0` | SASE computes its own horizontal mixing from the closure's own diffusivities, so a `km_opt` mixing operator would double-count it. 0 is admitted for this scheme and no other |
| `khdif = 0.0`, `kvdif = 0.0` | constant-K diffusion may not silently stack on the SASE mixing |
| `bldt = 0.0` | the closure produces a w tendency rebuilt every step rather than carried across a PBL call interval, so it must run every step |
| `sf_sfclay_physics != 0` | its lower boundary condition is the surface layer's friction velocity, heat and moisture fluxes and gust-corrected wind speed; with the slot off those four fields do not exist |
| `moist = true` | it mixes water vapour, cloud water and cloud ice alongside potential temperature, and forms its stability from the saturated Brunt-Vaisala frequency |
| `nz <= 128` | the implicit vertical solve carries its tridiagonal columns in per-thread local memory at that fixed depth |

On real data the path is the ordinary one, with two edits. Emit an
experiment config for your area as
[FIRST-LIGHT.md](FIRST-LIGHT.md) describes:

```bash
gpuwm domain --point=39,-98 --card 32gb --ladder 12 \
  --source gfs --cycle latest --hours 3 --out configs/myarea.toml
```

then in the `[shared]` table change `bl_pbl_physics` to `900` and
`km_opt` to `0`, and run it with `gpuwm go configs/myarea.toml`. The
emitted file already carries `khdif = 0.0`, `kvdif = 0.0`,
`bldt = 0.0`, `moist = true` and a surface-layer scheme, so those five
requirements are met as written; `nz` is the one to check against the
ceiling above. Everything else -- microphysics, radiation, cumulus,
land surface, the domain, the cycle -- is untouched, so a run against
the unedited file is the control for the one you just made.

**The closure is run-wide, never per-nest.** `bl_pbl_physics` is a
`[shared]` key, so on a domain tree the two edits above select SASE on
every domain of the tree or on none of it; there is no way to run one
nest under SASE and its parent under YSU. A `[[domain]]` table that
carries `bl_pbl_physics` is refused at load as an unknown domain key,
which is the right answer -- a per-domain PBL scheme that parsed and
was then ignored would be a silently-wrong run. The one SASE key that
*is* per-domain is the output-only `sase_flux_diag` in the table
below. (The registry lists `sase` among the domain-tree route's
`allowed_component_options`, which is a statement about what a physics
*plan document* may name; it is not a claim that the runner can vary
the PBL scheme between nests, and it cannot.)

Three optional knobs exist. Each is fail-closed on its non-default
value only -- set one without `bl_pbl_physics = 900` and the run is
refused, because a key naming a seam the run does not have would read
as a setting that took effect.

| knob | default | what it is |
|---|---|---|
| `sase_flux_diag` | `false` | output-only, per domain. True adds four history fields recording the closure's own vertical subgrid fluxes with the venting channel separated from the K_v diffusion channel; the prognostic state is bitwise identical either way |
| `sase_moist_n2` | `true` | physics selector, run-wide. True is the closure as built (saturated N^2 at the stability lengths, the subgrid-energy buoyancy source and the K_v/K_h suppression); false consumes the dry N^2 at all three points. Not per-domain: a nest whose domains ran different closures could not be compared across its own boundary |
| `sase_stable_dissipation` | `false` | physics selector, run-wide. The default is false for a measured reason, not for caution: with it true, the closure's own registered stable-boundary-layer calibration gate exits its observation band, and `tests/test_sase.py::test_jet_decoupling_stable_dissipation_exits_obs_band` pins that RED. What is falsified is the pair of stable-limb coefficients, which enter the stability ratio jointly; neither may be re-registered or tuned alone |

Maturity warns and never blocks. Every front
door says so once, and then continues. `gpuwm go` carries the clause
inline in the banner it prints before it does anything:

```
go: configs/myarea.toml -- source gfs, cycle 2026-08-01T12, 3 h, SASE PBL: EXPERIMENTAL, ArWen-original with no WRF counterpart
```

and both forecast runners print a full sentence to standard error as
the run starts -- `tools/prepared_single_domain_forecast.py`, which is
the stage `gpuwm go` drives:

```
prepared forecast: physics: SASE PBL: experimental, ArWen-original with no WRF counterpart -- the run continues.
```

and `tools/prepared_domain_tree_forecast.py`, which is how a domain
tree is run (`gpuwm go` refuses a multi-domain config and names this
runner instead):

```
prepared tree: physics: SASE PBL: experimental, ArWen-original with no WRF counterpart -- the run continues.
```

The sentence after the prefix is one string with one definition,
`gpuwm.physics_compat.experimental_selection_sentence`, and it is
built from the registry maturity rather than from any scheme name: a
second option registered at this rung is named by these same lines
without an edit to any of them.

Read that sentence literally. The status behind it:

- **It is not a WRF scheme.** There is no WRF v4.6.1 counterpart, so
  no oracle comparison against WRF Fortran exists or can exist for
  it, and it carries no WRF selector number. Every other option in
  this registry is measured as a distance from WRF; this one cannot
  be.
- **Its numerics are self-checked.** 293 tests pass on this head, 217
  CPU (`tests/test_sase.py`, plus one registered xfail) and 76 GPU
  (`tests/test_sase_gpu.py`, measured on an RTX 5090, driver
  580.126.09, CuPy 14.1.1): FP64-authority mirrors of every
  operator, analytic closed forms (box-filter transfer, single-mode
  structure function, pure-shear strain, closed-form energy decay,
  log-layer diffusivity identity, Ekman balance), energy-ledger
  theorems that close to roundoff, device-vs-authority parity at
  declared tolerances, and mutation controls and RED/GREEN
  falsification pairs on every switch. At run time the same posture
  applies: every rate the closure hands the dynamics is checked
  finite before it joins the RK stack, and a rate that is not stops
  the run naming the domain it came from and which of the producer's
  inputs -- friction velocity, surface fluxes, first-layer geometry
  -- were already degenerate when the closure received them.
- **Its physics is unvalidated.** Scored against a single reference
  case on a single day, the closure met 2 of its 7 frozen acceptance
  bars and missed 5 -- including a total absence of the low-cloud
  deck its conditional-venting limb exists to act on: median liquid
  water path 0.0 g/m2 against a reference of 132-235. Nothing
  measured speaks to any other case.
- **It has not completed a forecast on operational data.** Its first
  run on real input -- GFS 2026-08-01 18Z, one 306x244x49 domain at
  12 km, through `gpuwm go` -- reached one forecast hour and then
  failed the full-state health gate with every prognostic class
  violated. Two things were measured in the clean hour before it.
  The prognostic subgrid energy falls to its floor above the boundary
  layer as it should and then **regrows between 11.3 and 13.9 km**, to
  1.62 m2/s2 against its own boundary-layer peak of 2.60, centred on
  the level the gate then failed at; with the `km_opt = 0` this scheme
  requires, no other horizontal operator acts up there. And its
  boundary layer is deeper than the control's throughout: PBLH median
  1967 m vs 1558 m, p90 3962 vs 2793, maximum 8900 vs 4847, 9.5% of
  columns above 4 km vs 1.2%. Treat the co-location of that upper
  maximum with the failure level as a lead and not a proven cause:
  it is one case and one hour, and the same configuration also failed
  the gate under the stock YSU control -- later, at 3 h, and on the
  wind class alone. What separates the two is **not measurable today**:
  `km_opt = 0` is admitted only with this scheme, so the control that
  holds the mixing configuration fixed and varies only the closure
  cannot be expressed, and the closure publishes no diagnostic for the
  horizontal mixing it replaces (its four optional flux fields are all
  vertical).
- **It carries a known open bias.** Equilibrium subgrid TKE sits
  roughly 3-5x below the observed value (sqrt(e) about 2x low), which
  biases the stability lengths, the horizontal diffusivity, the
  buoyancy flux and every TKE product it writes. The bias is
  registered and unfixed: the TKE magnitudes this scheme reports are
  not to be believed.
- **Its horizontal operators wrap periodically.** The test filters,
  strain and horizontal flux stencils wrap in x and y
  unconditionally. On a specified or nested domain the mitigation is
  that subgrid energy is held at its floor across the outer
  `spec_bdy_width` rows every step, and the same width is excluded
  from the closure's solve reductions -- an adjudicated boundary
  treatment, not a halo exchange.

Read the maturity label as "the numerics are self-checked", never as
"the physics is validated". Use SASE to experiment. Do not use it for
a forecast you intend to believe.

## Which suites each data route can actually prepare

Selectable is not the same as preparable, and the difference is
route-dependent. State of play in v1.1.1:

| route | what it can prepare |
|---|---|
| ERA5 config door (`[case_data]` -> `gpuwm run`) | the registry-admitted combinations |
| GFS single domain | WSM6, Thompson, Morrison, NSSL2, MYNN; Noah-MP with an expert acknowledgement. RUC is deliberately withdrawn on this route |
| ERA5 single domain | the same normal profiles, plus RUC |
| HRRR single domain | the normal profiles, plus RUC and expert Noah-MP |
| prepared domain trees (GFS, ERA5, HRRR, 20CRv3) | the normal profile family, plus expert Noah-MP; microphysics may be overridden per domain, which is the only way to reach Thompson aerosol-aware (28) |

v1.0.1 restricted the GFS/HRRR door to YSU + MM5 surface layer + Noah,
because the front door unconditionally ran the stock-WRF exporter and
that exporter hard-requires `bl_pbl_physics = 1`, `sf_sfclay_physics =
91`, `sf_surface_physics = 2`. v1.1.0 removed that coupling and made
MYNN, RUC and Noah-MP reachable; v1.1.1 withdrew exactly one
combination, GFS + RUC, because the GFS route supplies none of the
soil/surface fields RUC's initialization needs and the failure landed
mid-forecast rather than at preparation.

**This table is a summary of a machine-readable authority, not the
authority itself.** The shipped registry decides, and it will answer for
your exact configuration:

```
rw-wps --show-physics-registry
```

Read `runner_routes.*.source_template_ids` and `expert_template_ids`.
Where this page and that output disagree, the output is right.

The physics suite is yours to choose. The GFS/ERA5/20CRv3 prepared
single-domain runner and the multi-domain tree runner execute any suite
the engine implements, exactly as your config writes it; there is no
profile whitelist on those routes. `--physics-profile` is optional
there: naming one asserts your config IS that shipped suite, and the
runner refuses on any switch drift, which is how you keep a run pinned
to a published product. The one route this does NOT describe is HRRR
cold start: its evidence contract is keyed by shipped profile, so it
prepares WSM6 when no profile is named and refuses a name outside its
contract -- an unnamed HRRR config's own suite does not run as written
there. Whether the suite you selected carries WRF-verification
evidence is stated -- one sentence in the wizard
output, the run receipt, and `--explain` -- and never gates: a suite
without evidence prints "supported, not yet WRF-verified" and the run
continues. An expert-template suite (Noah-MP) keeps its registry-owned
acknowledgement on every route, delivered as `--ack <id>` or
`acknowledgements = ["<id>"]` in the experiment. `gpuwm domain
--explain` lists the shipped profiles and
what each one actually runs -- several run full RTE+RRTMGP with
Kain-Fritsch, several run longwave OFF with Dudhia shortwave and no
cumulus. Read the names. What still refuses, on every route, is a
switch value the engine genuinely does not implement (the refusal names
the switch) and the registry's land-surface route blockers (for
example GFS+RUC, which dies at its first surface-temperature call).

## Namelist import substitutions

`gpuwm import-namelist` maps exactly two unimplemented WRF selections
to their nearest implemented counterparts, and reports each one in a
structured substitution report rather than silently rewriting:

| WRF namelist asks for | ArWen substitutes | class |
|---|---|---|
| `mp_physics = 55` (ISHMAEL) | Morrison 2-moment (10) | model-form change, no error bound |
| `ra_lw/sw_physics = 4` (RRTMG) | RTE+RRTMGP (default) or legacy RRTMG (`--rrtmg-variant rrtmg_legacy`) | explicit, token-recorded |

`bl_pbl_physics = 11` (Shin-Hong) was the third row of this table until
the Shin-Hong port: it now imports natively as 11 and runs the scheme
itself (registry maturity "implemented-unverified" -- its CPU authority
is bitwise against WRF v4.6.1, and no matched forecast trajectory
exists yet), so it is no longer a substitution.

Every other unimplemented scheme id is a hard error. Options WRF
accepts but ArWen has not validated -- moving nests, vertical
refinement, adaptive time step, `use_theta_m = 1`, non-SINT nest
interpolation -- are rejected loudly at load; the
complete register with WRF source citations is
[PROVENANCE.md](../../PROVENANCE.md).

`mp_physics = 28` is a **translation, not a substitution**: it imports
as 28 and runs the aerosol-aware scheme, so the table above stays at
exactly two substitutions. The aerosol knobs that surround it
(`use_aero_icbc`, `use_rap_aero_icbc`, `wif_input_opt`,
`num_wif_levels`, `qna_update`, `scalar_pblmix`, `grav_settling`,
`dust_emis`, `wif_fire_emit`, `wif_fire_inj`) are published in the
registry as unimplemented and refuse rather than being silently
dropped -- including where WRF *silently overwrites* them under mp=28.
Precisely, in `share/module_check_a_mundo.F`: `grav_settling` is forced
to 0 unconditionally for every mp=28 domain (`:2459-2474`);
`scalar_pblmix` is forced to 1 **only** when `use_aero_icbc` or
`use_rap_aero_icbc` is set (`:2477-2495`), which ArWen never reaches
because both are refused, and is forced back to 0 on any domain running
MYNN with `bl_mynn_mixscalars = 1` (`:2497-2511`). All three are debug-
or warning-level messages, not errors. ArWen's standing posture is to
refuse where WRF overwrites.

The full knob table around the scheme selectors -- every tweakable
namelist knob, its TOML spelling, default and allowed range, plus the
keys ArWen pins at a single validated value -- is
[CONFIGURATION.md](CONFIGURATION.md). The import report itself is
three-sectioned (translated / fixed-by-ArWen / not-implemented), so a
translated namelist never hides a knob decision.
