# mp_physics = 28 (Thompson aerosol-aware): what is actually measured

Status of this page: **evidence statement, deliberately conservative.**
It records what has been measured against unmodified WRF v4.6.1, what has
only been measured against something weaker, and what has not been measured
at all. Where a residual would not close, the measured number is printed
rather than absorbed into a tolerance.

The maturity label ArWen publishes for this option is
**`implemented-unverified`**. It is **not** `validation-candidate` and
**not** `model-validated`, and this page is the reason:

* `validation-candidate` would require a ratified reference comparison.
  There is none.
* `model-validated` would require a matched multi-hour ArWen-vs-WRF forecast
  with published decay tables. There is none. mp=28 *does* now integrate —
  150 steps, and 600 steps in a second gate — but not one of those steps has
  ever been compared against WRF's answer for the same step.

**Would the author defend `implemented-unverified` publicly?** Yes, and
only that. The label's published definition is "runs on the GPU and is
column-oracle-measured against unmodified WRF Fortran, but no dedicated
ArWen/WRF forecast-trajectory comparison exists for it yet; the registry
records its measured ULP distances and open divergences verbatim." Every
clause is true of this option today: 22 committed WRF column fixtures are
driven end to end and 15 of them clear a flat gate on all 23 compared
quantities; the seven that do not are published field by field, here and
on the registry option, with the ULP distance where a relative number
would mislead; and every open divergence is enumerated in §6 with WRF
line citations. What would make the label indefensible is a claim that the
column deck is clean, and it is not — §3 is honestly red, and the gate
that measures it fails.

Two things would make it *over*claimed, and neither is asserted anywhere:
that mp=28 has been compared against a WRF forecast (it has not), and that
a run which stays finite and inside WRF's clamps for two hours is therefore
right (it is not — see §5).

If you only read one section, read
[What you would get wrong today](#6-what-you-would-get-wrong-today).

---

## 1. Measurement environment

Every number on this page was produced on one machine, on one day. A
different GPU or compiler will move the last digits; it should not move any
of the conclusions.

| item | value |
| --- | --- |
| reference physics | unmodified WRF v4.6.1 `phys/module_mp_thompson.F`, commit `d66e442fccc04111067e29274c9f9eaccc3cef28` |
| Fortran | GNU Fortran 13.3.0, `-O2 -ffree-form -ffree-line-length-none`, baseline x86-64 (**no FMA instruction**) |
| GPU | NVIDIA GeForce RTX 5090 (cc 12.0), CUDA runtime 13.0, cupy 14.1.1, nvrtc defaults (`--fmad=true`) |
| host | Ubuntu 24.04.4 LTS on WSL2, Python 3.12.3, NumPy 2.5.1 |
| aerosol table | `CCN_ACTIVATE.BIN`, 35,288 bytes, sha256 `f2b8d391…c82a3dbd`, **redistributed with ArWen** since 2026-08-01 (deviation D9i in `PROVENANCE.md`; it stays outside the classic mp=8 table set, §7) |
| date of measurement | 2026-08-01 (every number on this page re-measured on this date) |

Arithmetic is float32 on the GPU and REAL(4) in the Fortran reference, which
is the comparison that matters; the ArWen kernels raise selected
sub-expressions to double exactly where WRF does and nowhere else.

---

## 2. Evidence classes, strongest first

Not all "it matches" claims are the same claim. This page grades them, and
a reader is entitled to discount the lower classes.

| class | what it is | why it is weaker than the one above | how much of mp=28 rests on it |
| --- | --- | --- | --- |
| **A — committed WRF column fixtures** | A whole column stepped by unmodified `mp_gt_driver`, dumped to CSV, committed, SHA-256'd. The comparison is against WRF's own answer for the complete scheme. | — | 22 scenarios, each a 24-level column with a `before` and an `after` row (48 data rows × 22 numeric fields) plus a 10-field surface file. This is the gate the maturity label rests on. |
| **B — committed scratch-driver Fortran output** | Fortran programs *written for this port* that call WRF's routines directly and tabulate intermediates or scalar functions. The physics is WRF's; the driver, the argument list and the choice of what to print are ArWen's. | A driver bug can make a wrong port look right, and there is no second implementation to catch it. Intermediate rates are also not something WRF itself ever exposes, so the "reference" has never been exercised by anyone else. | The five probe tables (`probe-activncloud.csv` and four siblings), the three per-kernel rate oracles and the five instrumented intermediate tables. |
| **C — host NumPy transcription** | A Python re-reading of the WRF formula, compared against the CUDA kernel. | If the transcription is wrong, both sides are wrong the same way and the test is green. This class can only find CUDA/Python disagreements, never a misreading of WRF. | Bit-exactness gates on individual device helpers and several bound/monotonicity properties. |
| **D — self-consistency, no external reference** | Runs, stays finite, stays inside WRF's own clamps, restores what it must restore. | It compares ArWen to ArWen. A scheme that is uniformly 20% wrong satisfies every one of these. | The whole of G4, the forecast smoke. Everything in §5. |

---

## 3. Class A — the committed WRF column fixtures

**Gate:** each fixture is driven end to end through the shipped adapter
(`gpuwm/core/microphysics_aerosol.py::_apply_thompson_aerosol`) and compared
against WRF's own after-state on 15 column fields (`qv`, `qc`, `qr`, `qi`,
`qs`, `qg`, `ni`, `nr`, `nc`, `nwfa`, `nifa`, `temp`, `effc`, `effi`,
`effs`), seven surface accumulations (`RAINNC`, `RAINNCV`, `SNOWNC`,
`SNOWNCV`, `GRAUPELNC`, `GRAUPELNCV`, `SR`) and `REFL_10CM` against WRF's
own `calc_refl10cm` — **23 quantities** — at a **2.0e-6 maximum relative
difference** and **2.0e-4 dB**. The tables below use the 16-field subset
(the 15 column fields plus `RAINNC`) that the registry publishes.

**The deck is twenty-two columns, not nineteen.** The gate globs
`gpuwm/data/thompson/oracle-aero/*-column.csv`: the 19 scenarios
`MP28_PORT_SPEC.md` names (ids 101-119, all `aero-*`) plus three `wp08-*`
columns (ids 120-122) that the same `build_aero.sh` invocation produced in
the same format, which pin every reachable `nu_c` (3..15) and both branches
of the terminal phase cleanup. Earlier revisions of this page, the registry
and `PROVENANCE.md` all said "nineteen" while the gate drove twenty-two,
so two residuals (`wp08-freeze`, `wp08-nusweep`) were in no published class
at all. They are below.

**Result: 17 of 22 clear the flat 2.0e-6 / 2.0e-4 dB gate on every
compared quantity, with nothing held out at all.** That is
16 of the 19 spec'd `aero-*` fixtures, plus `wp08-melt`. One more,
`aero-reduces-to-classic`, clears only under ONE named allowance (§3.2),
taking the gated count to 18 of 22. Four miss outright.

**RE-MEASURED ON THE 1.4.1 LINE (2026-08-01).** Every number on this page
was measured while the port sat on its own base, ArWen 1.3.1. Merging
`integration/release-1.4.1` inherited the mp=8 lane's two sedimentation
reconciliations — `5e4af4e3` ("the rain MVD bound belongs to TAU+1, not to
sedimentation") and `cb765336` ("the rain-presence gate is a mass
concentration, floor included") — in the byte-frozen `thompson.cu` this
port shares for rain fallout. mp=28 did not change; its inputs did. What
moved, all towards WRF:

| quantity | before the merge | after |
| --- | --- | --- |
| `aero-reduces-to-classic` `nr` at level 5 | 5.700e-06 | **4.146e-07** |
| `aero-reduces-to-classic` worst ULP, all 23 quantities | 27.5 | **4.0** |
| `aero-reduces-to-classic` worst \|dBZ − WRF\| | 3.242e-05 dB | **9.537e-06 dB** |
| `qr` levels bit-exact against WRF where mp=8 is not | 1, 2, 4 | **1, 2, 4, 5** |
| relative carve-outs live in the gate | 2 | **1** |

Nothing moved the other way, and the fixture-level verdicts below are
otherwise unchanged: the same four fixtures miss, at the same numbers.
`_END_TO_END_BOUNDS` is now an empty dict — see §3.2.

The table below gives, for every spec'd fixture, the worst field and its
measured relative difference. `PASS`/`MISS` is against the uniform 2.0e-6
gate with no allowance applied. `aero-reduces-to-classic` now reads `PASS`
here for the first time: the flat gate and the adapter's gate agree on it.

| fixture | verdict | worst field | worst relative difference | what it pins |
| --- | --- | --- | --- | --- |
| `aero-ccn-activate` | PASS | - | 0.0 | `activ_ncloud`, both clamp ends of `ta_Na`/`ta_Ww` |
| `aero-ccn-sweep` | PASS | - | 0.0 | activation over 10 updraft × 5 CCN cells |
| `aero-init-profile` | PASS | - | 0.0 | `thompson_init`'s synthetic CCN/IN fill and the `nwfa2d` derivation |
| `aero-sfc-emit` | PASS | - | 0.0 | surface emission lands only on k=kts and is unclamped |
| `aero-scav-frozen` | PASS | qg | 9.507e-08 | snow/graupel aerosol scavenging, `Eff_aero` |
| `aero-nc-effrad` | PASS | nr_per_kg | 1.863e-07 | all three `inu_c` branches of `calc_effectRad` |
| `aero-nc-sed` | PASS | nr_per_kg | 2.154e-07 | number-weighted cloud sedimentation |
| `aero-nc-auto` | PASS | nr_per_kg | 2.328e-07 | nu_c-driven Berry-Reinhardt autoconversion |
| `aero-ice-koop` | PASS | ni_per_kg | 3.396e-07 | **homogeneous haze freezing — closed a revision ago; see §3.4** |
| `aero-nc-cap` | PASS | nr_per_kg | 3.821e-07 | the `Nt_c_max` caps and the `2/rho` floor |
| `aero-scav-rain` | PASS | nr_per_kg | 3.858e-07 | rain scavenging of CCN and IN |
| `aero-drop-evap` | PASS | nr_per_kg | 3.919e-07 | the aerosol-only `tnc_wev` droplet-evaporation branch — **closed this revision; see §3.3** |
| `aero-warm-overlap` | PASS | nr_per_kg | 4.195e-07 | **cross-network `ncten`/`nwfaten` reconciliation, warm half** |
| `aero-ice-demott-dep` | PASS | ni_per_kg | 4.339e-07 | `iceDeMott` replacing Cooper nucleation |
| `aero-nc-accrete` | PASS | nr_per_kg | 5.533e-07 | nu_c-driven accretion, live `t_Efrw` second index |
| `aero-ice-demott-idxin` | PASS | qr | 6.417e-07 | the only fixture that reads a `freezeH2O` slice other than 27 — **closed this revision; see §3.3** |
| `aero-cloud-freeze-nc` | MISS | qc | 4.926e-06 | Bigg freezing with the `nc`-driven cap |
| `aero-reduces-to-classic` | PASS | nr_per_kg | 4.146e-07 | the bridge to the model-validated mp=8 port (level 6 is still held in ULPs by §3.2's one surviving allowance) |
| `aero-cold-overlap` | MISS | qc | 1.0 | **cross-network reconciliation, cold half** — and a one-ULP disagreement reported as full scale; see §3.1 |

The three columns outside the spec'd nineteen, measured the same way:
`wp08-melt` PASS (`nc_per_kg` 1.365e-07), `wp08-freeze` MISS
(`nr_per_kg` 2.724e-06, 1.4x the gate), `wp08-nusweep` MISS
(`qr` 4.642e-06, 2.3x the gate).

### 3.1 Every field that misses, with its number

| fixture | fields above 2.0e-6 |
| --- | --- |
| `aero-cold-overlap` | `qc` 1.000e+00, `nc` 1.000e+00, `effc` 8.102e-01, `nr` 1.261e-04, `qr` 4.443e-05 |
| `aero-reduces-to-classic` | nothing at level 5 any more (`nr` 4.146e-07, inside the flat gate); `qr` / `nr` 1.238e-04 at level 6 if that level is measured relatively rather than in ULPs, which is what §3.2's one surviving allowance exists for |
| `aero-cloud-freeze-nc` | `qc` 4.926e-06 |
| `wp08-nusweep` | `qr` 4.642e-06 |
| `wp08-freeze` | `nr` 2.724e-06 |

**No surface accumulation misses any more, on any fixture in the deck.**
All seven are compared separately: `RAINNC`, `RAINNCV` and `SR` are now
**bitwise identical to WRF on all 22 columns** (0.000e+00 relative), and
`SNOWNC` / `SNOWNCV` and `GRAUPELNC` / `GRAUPELNCV` peak at 6.285e-08 and
7.062e-08. Earlier revisions of this table carried `rainnc` / `rainncv` /
`sr` rows on two fixtures and explained them as **one number seen three
ways** — `RAINNCV` is the fallout sum (`module_mp_thompson.F:1298`),
`RAINNC` accumulates it from zero on a first call (`:1299`), and `SR`
divides the exactly-agreeing frozen part by it (`:1308`). That explanation
was right and the rows are gone: §3.3 records what removed them.

**`aero-cold-overlap`'s three full-scale rows are one mechanism, and the
absolute difference behind them is one ULP.** Measured at 0-based level 4:
the level enters with `qc` = 2.3252160e-04 kg/kg and `nc` = 9.1306704e+07
per kg. WRF ends the step with `qc` = 1.4551915228366852e-11 kg/kg — which
is exactly 2⁻³⁶, and exactly **1.000 float32 ULP** of that entry value —
and `nc` = 1.8333361 per kg (0.229 ULP). ArWen ends at exactly `0.0`.
`effc` follows: with no cloud water ArWen takes the 2.49 µm floor while
WRF's remainder gives 1.31176e-05 m, hence 8.102e-01. It is recorded as a
MISS rather than absorbed into an allowance, because the honest thing to
publish is that a relative metric is the wrong instrument here — not to
widen the instrument. The fixture's *other* residual is separate and real:
`nr` 1.261e-04 and `qr` 4.443e-05 at level 6, where the rain number falls
255.407 → 0.0739 per kg (99.97% consumed) and the difference is 0.611 ULP
of the entry value (`qr`: 1.789 ULP).

**Where the rest of them live.** Every surviving residual now sits in one
of two regimes: the field is **created from zero** inside the step
(`wp08-freeze` `nr` at level 0, reaching 23.808 per kg; `wp08-nusweep`
`qr` at level 12, reaching 2.242e-11 kg/kg on an absolute difference of
1.04e-16 kg/kg), or it is driven to **near-total consumption**
(`aero-cloud-freeze-nc` `qc` at level 4, 98.2% frozen away and the
survivor differing by exactly 1.000 ULP of entry; `aero-cold-overlap` at
levels 4 and 6). After the Koop closure below, there is **no surviving
residual in the "rate disagreement" class at all**. That is stated, not
used: no bound anywhere was relaxed for it.

**The one cell whose mechanism the port can name and could not reach.**
`aero-cloud-freeze-nc` `qc` at level 4 is the SECOND float32 rounding of
`qc` inside one step. WRF rounds once, at `module_mp_thompson.F:3975`,
from a `qcten` that carries the source network and the condensation
together; ArWen applies the source network to `qc` and then applies the
condensation to the already-rounded value. Measured and pinned: `qc` is
8.205620542867109e-05 entering the condensation stage and
1.4771576388739049e-06 leaving it, and nothing after that touches it —
cloud sedimentation, the terminal phase cleanup and the terminal apply all
leave it bit-identical — so the whole 4.926e-06 is that one extra rounding
and nothing else. Closing it needs a `qcten` accumulator carried across the
condensation, which needs a scratch slot in `gpuwm/core/preflight.py` and a
resequencing of the `ncten` balance limiter (`:2996-3019`), the `:3646`
cloud column mask and `thompson_aerosol_sed.cu`'s cloud sedimentation.
Recorded, attributed, measured, not absorbed.

### 3.2 The one allowance, named, with what it buys

An auditor found that `aero-reduces-to-classic` was being counted clean
through three simultaneous allowances with no single place that said so.
This is that place. The one allowance left applies to that one fixture and
is required for it; removing it puts the fixture back over the gate.
**The other two have been retired**, not kept, and each retirement
inverted its own assertion rather than deleting it.

| allowance | what it does | was | is |
| --- | --- | --- | --- |
| `_NEAR_CANCELLATION_LEVELS` | 0-based level 6 held to 32 ULP of the entry value instead of a relative bound | the level was skipped outright | **32 ULP; measured 0.585 (`qr`), 0.159 (`nr`) — the merge took these from 14.9 and 4.05** |
| ~~`_END_TO_END_BOUNDS`~~ | `nr` held to 1.0e-5 instead of 2.0e-6 | 2.5e-3, then 1.0e-4 covering `qr` and `nr` both, then 1.0e-5 for `nr` alone | **RETIRED at the 1.4.1 merge — level 5's `nr` fell to 4.146e-07, inside the flat 2.0e-6 gate, and the constant is now an empty dict** |
| ~~`_REFL_DB_BOUNDS`~~ | reflectivity held to 1.0e-3 dB instead of 2.0e-4 dB | 1.0e-2 dB, then 1.0e-3 dB | **RETIRED — the residual it covered fell to 3.242e-05 dB, and then to 9.537e-06 dB at the merge, inside the flat 2.0e-4 dB gate; the constant is now an empty dict** |

**What the "was" column is, and is not.** The current values are read out
of the gate's own constants and re-measured here. The *previous* values are
history recorded by the packages that changed them; this port lives on an
uncommitted branch, so they cannot be re-derived by diffing the tree, and
they are reproduced rather than verified. What *is* verifiable from the
tree, and is asserted by
`test_every_g3_allowance_is_named_and_buys_exactly_one_fixture`, is the
part that matters to a reader now: these two are the *only* departures
from the flat gate anywhere in the deck, they both apply to one fixture,
and removing either of them puts that fixture back over the gate.

Nothing here was widened, in this revision or in any earlier one. The
numeric bound tightened twice, each time because the mechanism it existed
for was found and closed. First: `module_mp_thompson.F:3490` overwrites
`rho(k)` inside the condensation loop while `:3384-3388` had already frozen
the rain moments from the pre-condensation density, and the adapter was not
passing that entry density through; that took the level-5 `qr` residual
from 1.915e-03 to 1.788e-07, left level 1 as the column's worst at
7.813e-05, and moved the bound 2.5e-3 → 1.0e-4 with it. Second, **in this
revision**: WRF builds the working rain mass and number that sedimentation
consumes at `:3237-3238` from the `:3193` τ+1 density for every level
carrying rain, and rebuilds them at `:3568`/`:3570` from the `:3490`
post-condensation density only inside the `:3501-3502` gate, while ArWen
wrote the post-condensation density unconditionally. Restoring the
level-wise choice made level 1 **bit-exact against WRF** in both `qr` and
`nr`, so the column's worst `qr` fell 7.813e-05 → 1.788e-07 — inside the
flat gate, so the `qr` entry was **deleted** rather than kept — and its
worst `nr` fell 4.832e-05 → 5.700e-06, with the bound following it
1.0e-4 → 1.0e-5. Both survivors are now at level 5. The same fix put
the fixture's reflectivity inside the flat dB gate and retired the third
allowance. The remaining allowance replaced a bare *skip* of level 6 with a
bound, which is strictly more than the skip asserted — and it is a metric
change rather than a looser tolerance: the level enters with `qr` =
3.1695777e-07 kg/kg and evaporates 99.958% of it in one 10 s step, so the
survivor is the difference of two nearly equal float32 numbers and a
relative gate there measures the rate's error amplified by
1/(1 − 0.99958) = 2370. ArWen produces 1.3384e-10 there against WRF's
1.3426e-10, where it used to produce exactly zero.

**The attribution of what survives CHANGED in this revision, and the old
one is now false, so it is replaced rather than edited.** This page used to
say the residual was pre-existing and not introduced by the aerosol port,
because the frozen mp=8 pipeline reproduced WRF's `qr` bitwise at levels
2-3 on the identical entry column. That was true and is not any more —
because mp=28 got **better**. With the level-wise sedimentation density
restored, mp=28 is bit-exact against WRF at `qr` levels 1, 2 and 4 and at
`nr` level 1, where mp=8 is 7.61e-05, 4.00e-05, 6.27e-05 and 4.63e-05
away; over levels 0-7 mp=28 is at least as near WRF as mp=8 at every level
and strictly nearer at seven of eight. So the surviving `nr` 5.700e-06 is
**not inherited and is no longer claimed to be**. It sits at 0-based level
5 alone — the one level of this column where the step removes a large
fraction of the rain number without emptying it (49.75%: 3.000000e+05 →
1.507546e+05 per kg) — and it is 27.5 ULP of the entry value, where every
other unexcluded level of the column is 0, 1, 2 or 3 ULP.
`test_the_reduces_to_classic_residual_is_the_classic_paths` now asserts
that port-vs-port ordering rather than the bitwise identity it used to
assert.

### 3.3 What moved since the previous revision — in both directions

**The clean count went 15 of 22 → 17 of 22 and the miss count six → four.**
Two production changes did it, both in mp=28-owned kernels, and one of them
also cost a cell.

* **WRF's sedimentation density is level-wise and ArWen's was not.** WRF
  forms the working rain mass and number that sedimentation consumes
  **twice**: at `module_mp_thompson.F:3237-3238` from the `:3193` τ+1
  density, for every level with rain, and again at `:3568`/`:3570` from the
  `:3490` post-condensation density — but only inside the `:3501-3502`
  gate. `thompson_aerosol_sat.cu` wrote the post-condensation density into
  its `reference_density` output unconditionally, so every level got the
  `:3568` answer including the levels WRF never rewrote, and
  `microphysics_aerosol.py` hands that same buffer to the rain
  sedimentation launcher. Measured closures: **`aero-drop-evap` left this
  table entirely** (`rainnc` / `rainncv` 5.165e-04 → 0.000e+00, `qr`
  3.533e-05 → 7.35e-08, `nr` 2.258e-05 → 3.92e-07) and so did
  **`aero-ice-demott-idxin`** (`sr` and `rainnc` / `rainncv` 1.279e-04 →
  0.000e+00 after the second change below, `qr` 2.894e-05 → 6.42e-07);
  `aero-cloud-freeze-nc` lost five of its six rows (`qr` 2.800e-05 →
  8.97e-08, `nr` 1.797e-05 → 2.59e-07, `rainnc` / `rainncv` / `sr`
  1.162e-05 → 0.000e+00) and keeps only `qc`; `aero-reduces-to-classic`
  went `qr` 7.813e-05 → 1.788e-07 and `nr` 4.832e-05 → 5.700e-06, and its
  reflectivity 5.283e-04 → 3.242e-05 dB, which **retired the third
  allowance**.
* **WRF's terminal apply cannot be an FMA and nvrtc made it one.**
  `:3973-4023` is `q1d(k) = q1d(k) + qten(k)*DT`, and the `gfortran -O2`
  baseline-x86-64 oracle has no FMA instruction, so `qten*DT` is rounded to
  `REAL(4)` first; nvrtc contracted the same expression in the cold and warm
  source networks and never rounded it. Both now use the same pinned
  add/sub/mul the rain-evaporation apply already used. **Verified at the
  stage, not only end to end**: the post-source-network `qc` at
  `wp08-freeze` level 0 is now 2.7275411412119865e-05 where the fused form
  gave 2.7275422326056287e-05, and at `aero-cold-overlap` level 4 it is
  1.5486410120502114e-05 where the fused form gave 1.5486406482523307e-05 —
  both of which are the rounded-product values WRF itself produces. It made
  `aero-nc-cap`'s `qc` and `nc` and `aero-ice-demott-idxin`'s surface
  accumulators **bitwise** exact.
* **What it cost, recorded rather than absorbed.** `aero-cold-overlap`'s
  `qr` at level 6 **grew**, 3.667e-05 → 4.443e-05 (1.477 → 1.789 ULP of the
  entry value, which is the scale that cell is carried at because 99.5% of
  the level's rain is consumed in the step). Bisected to a single line, the
  cold network's own `qr` apply: reverting that pin alone restores 3.667e-05
  and simultaneously loses all four `aero-ice-demott-idxin` improvements
  above, two of which are exact. The pin is kept because it makes the
  instruction sequence WRF's at every cell rather than at the cells that
  flatter this table.

**What moved in the revision before this one**, kept here because a reader
checking the direction of travel needs more than one step of it:

* **`aero-ice-koop` is closed, and it was the measuring stick that was
  wrong, not the physics.** Its `qi` 1.612e-03 / `ni` 1.764e-03 / `effi`
  5.093e-05 — which this page called "the one genuine physics gap" and
  which three auditors called the port's largest — now measure 1.534e-07 /
  3.396e-07 / 1.886e-07 and the fixture clears the flat gate. **No kernel
  changed.** See §3.4: the closure is entirely the correction of the oracle
  harness's Exner constant. A port that reports "we fixed our measuring
  stick" is more trustworthy than one that reports a physics fix it did not
  make, and homogeneous haze freezing was never measured to be wrong.
* `aero-cloud-freeze-nc` lost its `effc` row (5.018e-06 → 1.619e-06) and
  its `qc` fell 1.478e-05 → 4.926e-06; `aero-ice-demott-idxin` lost its
  `qc` row (6.031e-06 → 7.556e-08).
* **Against that: `aero-cold-overlap` is worse than it was published.** It
  gained the `qc`/`nc`/`effc` rows at level 4 described in §3.1, and its
  `qr` rose 1.174e-05 → 3.667e-05, and then rose again in this revision.
  Recorded, not absorbed.
* **And two residuals had never been published at all**, because the
  narrative counted 19 fixtures while the gate drove 22.

An evidence document that only ever moves in the author's favour is not
evidence. Of the seven bullets above, five are improvements, one is a
regression published as grown, and one is a disclosure of residuals that
were never published; the registry, `PROVENANCE.md` and
`docs/public/PHYSICS.md` now carry the same seven.

### 3.4 What actually closed `aero-ice-koop`: the oracle harness, not a kernel

This subsection exists because the alternative was to let a true sentence
("`aero-ice-koop` is clean") carry a false implication ("ArWen fixed the
homogeneous-freezing kernel"). It did not. **Nothing in
`gpuwm/core/kernels/` changed for this fixture.** What changed is the
Fortran program that produces the reference answer.

`tools/thompson_wrf461_oracle/run_column_aero.F90` builds the Exner
function the way WRF's own `phy_prep` does
(`dyn_em/module_big_step_utilities_em.F:4854`,
`pi_phy = (p_phy/p1000mb)**rcp`). Until 2026-08-01 it spelled `rcp` as
`287.0/1004.0`. WRF's `rcp` is `r_d/cp` with `r_d = 287.` and
`cp = 7.*r_d/2.` — **1004.5, not 1004** —
`share/module_model_constants.F:19`, `:20`, `:31`. In float32 WRF's value
is `0x3E924925`, bit-identical to `2./7.` and to `gpuwm/core/constants.py`'s
`RCP`; the harness's was `0x3E925BCB`, **4774 ULP away**. (The literal
`287./1004.` does occur once in the stock WRF tree, at
`phys/module_fdda_psufddagd.F:1247` — a PSU FDDA diagnostic. It is not the
model's Exner function and nothing in the dynamics or physics prep uses
it.)

The consequence was not that the fixtures were "slightly warm". It was
that the `(p, theta)` pair the harness handed `mp_gt_driver` could no
longer be **inverted** on the ArWen side. A fixture records `p_pa` and
`temp_k`; the adapter must solve for the float32 `theta` that reproduces
`temp_k` bitwise under `gpuwm`'s own `RCP`. Two Exner constants 4774 ULP
apart do not agree about whether that solve has a solution: **dozens of
the deck's 528 entry levels — 47 as measured by every current
environment, 40 as first published; see the correction below — admit an
exact `theta` under one and none under the other.** `_reconstruct_entry_state` therefore perturbed the entry
pressure by up to 15 ULP at those levels, and a perturbed entry pressure
drives genuinely different microphysics. Seven fixtures carried perturbed
levels; `aero-ice-koop` was one of them.

Be precise about which half of that is checkable here and which is
historical. **Checkable now, from the committed bytes:** the shipped deck
inverts exactly under WRF's `r_d/cp` at all 528 levels and fails at 47 of
them under `287.0/1004.0` — the table below, re-derived by a test on every
run.

**Correction, 2026-08-03 (owner-ratified), re-pinning 40 → 47.** This
section first published the superseded-constant failure count as 40, and
the pinning test asserted it. The deck's bytes never changed — the count
did, because it is not a property of the deck alone: the inversion check
runs the host's float32 `power`, and hosts disagree at the last bit.
The original 2026-08-01 environment measured 40; every environment since
measures 47, with identical per-fixture counts across toolchains as
different as glibc 2.43 / numpy 2.5.1 and MSVC / numpy 2.2.6, and the
count is re-pinned to what is measurable rather than to a dead
environment's libm. What never moved, on any environment: **528 of 528
levels invert under WRF's own constant**, and the affected fixtures are
**the same seven** either way. Those two facts carry the attribution;
the exact count under a constant this project already retired never did. **Historical, and taken from the harness's own rebuild receipt:**
that the fixtures *before* the correction were the ones with the perturbed
levels, and that regenerating them under WRF's constant is what moved the
numbers. The old fixtures are not on this tree, so that half is a
citation, not a re-measurement.

**Verifiable from this tree, without rebuilding anything.**
`tests/test_physics_md_aerosol_claims.py::test_the_committed_deck_carries_wrfs_own_exner_constant`
reads the 22 committed CSVs and re-solves the inversion under both
constants:

| Exner constant | entry levels that invert exactly | fixtures affected |
| --- | --- | --- |
| WRF's `r_d/cp` (`2/7`, `0x3E924925`) — what the deck carries | **528 of 528** | none |
| the superseded `287.0/1004.0` (`0x3E925BCB`) | 481 of 528 | 7, including `aero-ice-koop` |

The seven are `aero-cloud-freeze-nc`, `aero-cold-overlap`,
`aero-ice-demott-dep`, `aero-ice-demott-idxin` (8 levels each),
`aero-ice-koop` and `wp08-freeze` (6 each) and `aero-reduces-to-classic`
(3). Six of them are `aero-*`; the seventh, `wp08-freeze`, is outside the
nineteen the harness README tabulates, which is why that file's table names
six. Its level counts are also not the same integers as the ones above,
and they should not be: the README counts levels the adapter actually
perturbed, this table counts levels at which no exact `theta` exists at
all. The first is bounded below by the second.

And it is verifiable by rebuild: the harness's own
`tools/thompson_wrf461_oracle/README-AEROSOL.md` records the differential
regeneration — same tree, same kernels, only the fixtures regenerated —
in which `aero-ice-koop`'s worst field went `ni_per_kg` 1.764e-03 →
3.40e-07 and its perturbed-level count went 8 → 0. The gate's
`_ENTRY_STATE_PERTURBATION` table now records `(0, 0.0)` for **every one of
the twenty-two fixtures** and asserts it exactly, so no G3 residual
anywhere in this document can be attributed to the reconstruction any more.

Two things follow, and the second is the uncomfortable one:

1. The port's advertised "largest genuine physics gap" was, for two
   waves, an artifact of ArWen's own reference harness. Every publication
   that quoted it — this page, `PHYSICS.md`, `PROVENANCE.md` and the
   registry warning — quoted a number produced against a yardstick that
   was not WRF's.
2. The same correction is what made `aero-cold-overlap` **worse** (§3.1).
   The regeneration is not a favourable edit that happened to help; it
   moved fixtures in both directions, which is the only reason it can be
   trusted at all.

### 3.5 What the fixtures do NOT cover

* Every fixture supplies its own aerosol, so **no fixture exercises the
  initialisation path a real run takes** (§6.1). That path is now wired and
  separately gated, but not by anything in this section.
* Fixture states are deliberately chosen away from `activ_ncloud`'s bin
  edges. `nc` is a step function of state there — the temperature bin is
  nearest-neighbour on 10 K and `idx_d`/`idx_c`/`idx_n` are `INT`
  truncations — so near an edge an FP32 GPU and the Fortran reference can
  select different bins and differ by tens of percent in `nc` while every
  mass field agrees. Measured at the 248.15 K edge: a 1.526e-05 K change
  moves `nc` by 9.729e-02 relative (8.198490e+08 → 7.400879e+08). This is
  documented rather than absorbed into a looser gate, and **no fixture sits
  on an edge**, so the port's behaviour there is unmeasured.
* Single call, single column, no transport, no accumulation. Everything
  multi-step is class D.

---

## 4. Class B — committed scratch-driver Fortran output

Three programs in `tools/thompson_wrf461_oracle/` link the unmodified WRF
module and print things WRF never prints: scalar helper functions over a
grid, per-process rates inside the source loop, and the state at chosen
anchors inside the column sweep. Receipts, digests and reproduction
commands are in `tools/thompson_wrf461_oracle/PROBE_ORACLE_RECEIPTS.md`.

| table | rows verified | status |
| --- | --- | --- |
| `_WARM_RATE_ORACLE` (`test_thompson_aerosol_warm_gpu.py`) | 124 × 26 fields | reproduced |
| `_NCTEN_BALANCE_ORACLE` (same) | 68 × 8 fields | reproduced |
| `_WRF_COLD_WARM_LOOP` (`test_thompson_aerosol_cold_gpu.py`) | 54 × 20 fields | reproduced |
| `_WRF_COLD_REFERENCE` (same) | 360 values | reproduced |
| `SED_AERO_NC_SED` (`test_thompson_aerosol_sed_gpu.py`) | 384 values | reproduced bitwise |
| `CLEAN_CLASSIC` (same) | 408 values | reproduced bitwise |
| `SED_NU_SWEEP`, `CLEAN_MELT`, `CLEAN_FREEZE` (same) | — | **not verified.** The three scratch scenarios that produced them are not in `run_column_aero.F90` and the driver was lost. The literals are asserted against; their provenance is not reproducible from committed sources. |

The five device-helper probe tables (`probe-activncloud.csv` 1320 rows,
`probe-icekoop.csv` 480, `probe-icedemott.csv` 320, `probe-effaero.csv` 48,
`probe-effectrad.csv` 14) come from `probe_aero_functions.F90` and are
committed. The device helpers reproduce them exactly where the gates say
"bitwise" and to a documented FP32 distance elsewhere.

**Why this is class B and not class A.** `probe_aero_functions.F90` and its
siblings are ArWen code. If one of them passes an argument WRF's own driver
would never pass — a `nifa` in per-kg where WRF uses per-m3, say — the probe
and the kernel can agree perfectly and both be describing a call WRF never
makes. Only the class A fixtures, which go through `mp_gt_driver`, are
immune to that.

---

## 5. Class D — the forecast (G4), and what it does and does not show

Before WP-12b, **no multi-step mp=28 forecast had ever been run.** There is
now one: `tests/test_mp28_forecast_smoke.py`.

**Configuration.** 28 × 16 × 24 cells, dx = dy = 2 km, ztop = 16 km,
dt = 12 s, 150 RK3 steps (1800 s), specified lateral boundaries
(`spec_zone = 1`, `relax_zone = 4`), WK82's analytic sounding entered
through WRF's own moist rebalance, a 3 K / 8 km thermal, and a uniform
20 m/s zonal inflow. The domain is taken through the real
`initialize_physics`, so it carries `thompson_init`'s CCN/IN profile —
this is the configuration a user gets, not a stripped one. Attaching the
full `PhysicsDriver` is separately measured to be **bitwise irrelevant** to
this forecast (same domain-total `RAINNC` to every digit either way), which
is what makes §6.1's counterfactual a clean single-variable comparison.

**What holds, on every one of the 150 steps:**

* no NaN or Inf in any prognostic, any effective radius, `thp`, `php`,
  `mup`, `u`, `v`, `w` or `h_diabatic`;
* in the microphysics-updated interior, every bound WRF's terminal apply
  establishes: `nwfa` ∈ [1.11e7, 9.999e9], `nifa` ∈ [5.0e3, 9.999e9],
  `nc·ρ` ≤ 1.999e9, `effc` ∈ [2.49, 50] µm, `effi` ∈ [4.99, 125] µm,
  `effs` ∈ [9.99, 999] µm;
* the specified-zone ring is **bit-restored** by every microphysics call —
  checked around the call itself, not across the step, because the
  lateral-boundary path also writes that ring during RK3;
* the production health validator accepts the final state, and `nwfa`,
  `nifa` and `nc` are in its census (so that acceptance means something);
* **no persistent scratch slot is carried between steps.** All fifteen
  `mp_thompson_aero_*` slots are poisoned before every one of 40
  consecutive microphysics calls and the entire prognostic state stays
  **bitwise identical** to an unpoisoned run. This is the failure a column
  test structurally cannot find — `state.scratch` buffers survive across
  steps by design, so an unzeroed accumulator would feed the previous
  step's tendency in as a live physics input and drift plausibly forever.

**The G4 detectors were checked against injected faults** rather than
trusted: a one-row write to the ring, an out-of-band `nwfa` and `effi`, a
single NaN, scoring the 10 m/s run against the 20 m/s wind, and disabling
the accumulator zeroing (which moves 9 of 12 prognostic fields). Every one
turns the corresponding gate red.

**A 2-hour run holds too.** The same configuration integrated for 600 steps
(7200 s, 2.6 domain ventilation times) gives 0 non-finite values, 0 bound
violations and 0 ring violations across all 600 microphysics calls, peak
|w| 56.10 m/s, 1.9055 mm of total `RAINNC`, and ends with the
domain-interior mean `nwfa` at 1.3395e7 kg⁻¹ against a floor of 1.110e7 and
`nifa` at 5.9041e3 against a floor of 5.000e3 — i.e. entirely inflow air,
exactly as §5.1's law predicts. This is the longest mp=28 integration that
exists.

**What this does not show.** Nothing here compares ArWen to WRF. A scheme
with a systematically wrong activation rate would satisfy every bullet
above, for two hours, without a single violation. The bounds are WRF's, but
they are *clamps*, not answers. Class D is a statement about robustness and
nothing else.

### 5.1 The lateral-boundary aerosol depletion, measured

ArWen couples only `qv` from an external lateral-boundary snapshot; every
other advected scalar, including `nc`/`nwfa`/`nifa`, gets WRF's
`flow_dep_bdy` treatment — zero gradient out, **zero in**. WRF does not do
this: its Registry declares `qnwfa`/`qnifa` with the boundary dimension and
`bdy_interp`, so stock WRF forces aerosol at the edge from the WIF metgrid
stream ArWen has no ingest for. This is registered as deviation **D9c** in
`PROVENANCE.md`.

It cannot NaN and cannot go negative, because WRF's own floors catch it. It
therefore trips nothing. Until now there was no number for how fast it
happens. There is one now, measured on a deliberately **cloud-free** run
(`qc = qr = qi = qs = qg ≡ 0` on every step is asserted, so the aerosol has
no microphysical source or sink and every kilogram lost is the boundary
policy):

| metric | value |
| --- | --- |
| `front_speed_ms` | 19.8638 |
| `front_speed_ratio` | 0.99319 |
| `nwfa_retained` | 0.45660 |
| `nifa_retained` | 0.33627 |
| `ventilation_time_s` | 2800.0 |
| `swept_fraction` | 0.6428571428571429 |
| `surface_emission_per_kg_s` | 5540.14 |

Read that as: with a 20.0 m/s inflow the depletion front advanced at
**19.86 m/s**, i.e. at the wind speed. Over 1800 s the domain-interior mean
`nwfa` fell to **0.457** of its initial value and `nifa` to **0.336**.

The same experiment at 10 m/s gives a front speed of **9.81 m/s** and a
retained fraction of **0.736**, so the behaviour is a law and not one
number:

> **The upstream `U·t` of your domain is at WRF's aerosol floor after
> time `t`, and the whole domain after `L/U`.**

For a 1000 km operational domain in a 20 m/s flow that is **13.9 hours** to
sweep the domain, and 72 km of the upstream edge is already at the floor
after the first hour. For a 100 km nest in the same flow it is
**83 minutes**. The floor is `nwfa = 1.11e7 kg⁻¹` — about 1/4.5 of the
5.0e7 kg⁻¹ that WRF's own synthetic profile installs in the free
troposphere, and about 1/17 of the 1.909e8 kg⁻¹ it installs at the surface.

The only aerosol *source* in the scheme is the fixed surface emission
`nwfa2d` at k = 0, measured here at **5540 kg⁻¹ s⁻¹**, which replaces
9.97e6 kg⁻¹ over the 1800 s run against an initial k = 0 loading of
1.909e8 kg⁻¹ — about 5%. It cannot keep up, and it acts on the lowest model
level only.

`nifa` depletes faster than `nwfa` (0.336 vs 0.457 retained) because its
floor, 5.0e3 kg⁻¹, is roughly two orders of magnitude below its initial
value (5.00e5–1.44e6 kg⁻¹ from the synthetic profile), whereas `nwfa`'s floor
is only a factor of ~4.5 below the profile's free-tropospheric 5.0e7 kg⁻¹
and a factor of ~17 below its 1.909e8 kg⁻¹ boundary-layer value.

### 5.2 The specified-zone ring itself ends at exactly zero aerosol

Sharper, and separate from §5.1, which is about the interior. WRF's clipped
microphysics tiles never touch the `spec_zone` ring, and ArWen reproduces
that bit for bit (§5) — so the terminal clamp that guarantees
`nwfa >= 1.11e7 kg⁻¹` everywhere else **does not run there**. The ring
carries whatever `flow_dep_bdy` left, and with no aerosol in the boundary
file that is exactly `0.0`: a value WRF itself can never produce.

Measured on the same purely zonal 20 m/s run, as the fraction of each face
that ends at exactly zero:

| face | zero fraction | why |
| --- | --- | --- |
| west (inflow) | 1.000 | zero inflow |
| south (tangential, v = 0) | 1.000 | not strictly outbound, so treated as inflow |
| north (tangential, v = 0) | 1.000 | same |
| east (outflow) | 0.125 | retains the interior value except at the two corner cells |

So **three of the four faces**, not one. Microphysics is unaffected — it
never reads the ring — but three consumers are: `QNWFA`/`QNIFA` written to
`wrfout`, any nest whose boundary is fed from this ring, and any diagnostic
that reads the full array. Pinned by
`tests/test_mp28_forecast_smoke.py::test_the_specified_zone_ring_ends_at_exactly_zero_aerosol`.
It is a consequence of D9c, not a separate defect, and it closes with it.

---

## 6. What you would get wrong today

Ordered by how much it would change a forecast. Every item was measured,
not inferred. §6.7 lists what left this section since the previous
revision, so a reader can see the direction of travel without taking it on
trust.

### 6.1 Removing the aerosol initial condition moves this case's surface rain by 74%

This is a **sensitivity**, not a defect — but it is the first thing to
understand about mp=28, because it is the largest single number the port
has measured about its own behaviour.

WRF's `thompson_init` fills a *synthetic* CCN/IN profile whenever the
aerosol arrays arrive unset: CCN at `module_mp_thompson.F:493-515`, ice
nuclei at `:531-551`, and the surface emission `nwfa2d` derived from the
filled surface value at `:510`. ArWen ports that fill as
`gpuwm.core.microphysics.microphysics_init`, gates it against WRF's own
post-`thompson_init` snapshot, and **calls it once per domain** from
`gpuwm/core/physics.py::initialize_physics` — the seam where WRF's
`phy_init` calls `mp_init` (`phys/module_physics_init.F:1635`). It is
presence-gated exactly as WRF's is, so a nest that inherited its parent's
aerosol and a run about to restore a checkpoint are both left alone.

Measured, as two otherwise identical 150-step forecasts of the convective
case in §5 — one taking the production init path, one with the profile
removed:

| quantity | with the profile (what a run does today) | with it removed | change |
| --- | --- | --- | --- |
| initial mean `nwfa` | 6.653e+07 kg⁻¹ | 0 | — |
| final interior `nwfa` | 3.006e+07 kg⁻¹ | 1.262e+07 kg⁻¹ | at the floor |
| peak `nc` over the run | 1.598e+08 kg⁻¹ | 2.844e+07 kg⁻¹ | **5.6× fewer droplets** |
| domain-total `RAINNC` | 1.791 mm | 3.119 mm | **+74.1%** |
| peak `RAINNC` | 0.678 mm | 0.930 mm | +37.1% |

Both runs are re-executed and this table rebuilt by
`tests/test_physics_md_aerosol_claims.py::test_the_published_aerosol_sensitivity_is_a_live_measurement`,
compared at the precision printed here. The comparison is exact rather
than toleranced because both forecasts were repeated end to end on the
measurement machine and every value was bit-identical across repeats; if
that stops holding, the right response is to publish the spread.

Read that precisely: removing the CCN loading raises domain-total surface
precipitation by 74.1% over half an hour and cuts the peak droplet count by
a factor of 5.6. (Both `RAINNC` figures moved by one part in two thousand
at the 1.4.1 merge -- 1.792 to 1.791 mm and 3.118 to 3.119 mm -- and the
excess with them, 74.0% to 74.1%. That is the inherited mp=8 rain
sedimentation reconciliation reaching a FORECAST rather than a column: it
is the only place in this document where a merged-in change to a shared
kernel is visible in a multi-step trajectory, and it is republished rather
than rounded back.) That is not a rounding difference; it is a different
forecast. Note also that §6.2 drives the domain to exactly the right-hand
column given enough time.

**This was the port's largest measured error until 2026-08-01.** The fill
was implemented, proven against WRF and called by nothing, so every mp=28
run integrated from `nwfa = nifa = 0` and the terminal apply clamped both
to WRF's floors (`:3979-3982`) for the whole run — with no NaN, no bound
violation, no warning, and no column gate able to see it, because every
fixture supplies its own aerosol. The numbers above are unchanged; they
were the cost of that gap and they are now the value of the profile. Two
gates keep it closed: one scans for the call site and asserts there is
**exactly one** (once-per-domain silently becoming per-step would overwrite
an advected, activated and scavenged aerosol field with the synthetic one),
the other asserts the filled state on a real `DomainState` taken through
the real `initialize_physics`.

### 6.2 No aerosol at the lateral boundary — **now the largest open gap**

§5.1 and §5.2, quantified. A 6-hour specified-BC run in a 20 m/s flow on a
200 km domain has been ventilated three times over: nothing of the initial
aerosol field remains anywhere, and the whole domain is running at the CCN
floor. The scheme keeps working and nothing warns. Separately, the
`spec_zone` ring itself ends at *exactly zero* aerosol on three of its four
faces (§5.2) — below WRF's own floor, because the clipped tile means the
clamp never runs there — which is what a nest or a `wrfout` reader sees.

**The forecast impact of this deviation is NOT directly measured, and here
is why.** The obvious experiment — re-impose the driving aerosol on the
`spec_zone` ring after every step, as WRF's `bdy_interp` would — does not
work: `flow_dep_bdy` runs *inside* the RK loop and re-zeroes the ring before
transport reads it, so the emulation moves the domain-interior mean `nwfa`
by only **0.05%** over 150 steps. Measured, and reported as a failed
attempt rather than as a null result. A real number needs the LBC ingest
that does not exist.

What *is* known is the endpoint, and §6.1 now measures it directly: after
`L/U` the whole domain is at the CCN floor, which is the right-hand column
of §6.1's table — 5.6× fewer droplets and +74% domain-total surface rain.
That is an endpoint magnitude inferred from a different experiment, not a
measured trajectory difference, and it should be read as an order of
magnitude for how much this matters, not as a number to quote. It is the
largest *open* item on this page. Note what does **not** close it: the
QNWFA/QNIFA ingest lane landed (§6.5) and fixes the INITIAL condition, but it
is an initialization route only — no aerosol crosses a specified lateral
boundary, so the ventilation described above is unchanged. Closing this needs
an aerosol LBC lane, which does not exist.

### 6.3 Four column residuals remain, and one of them is worse than it was

§3.1, in full. Nothing there is a rate disagreement any more: every
surviving residual sits where a field is created from zero inside the step
or driven to near-total consumption, and no surface accumulation misses on
any fixture. The two that a reader should carry away are
`aero-cold-overlap` — three full-scale relative numbers on a one-ULP
absolute difference, and a genuine 1.261e-04 in `nr` at a second level —
and the fact that this fixture got **worse** in this revision, its level-6
`qr` growing 3.667e-05 → 4.443e-05 as the direct price of pinning the
terminal apply's contraction, while four others got better.

### 6.4 MYNN mixes the `qn` family when it is asked to — **D9d CLOSED**

This section used to record that `gpuwm/core/mynn_pbl.py` passed
`flag_qnc`/`flag_qnwfa`/`flag_qnifa` as literal `False`, so WRF's
`bl_mynn_mixscalars > 0` mixing had no ArWen counterpart and the divergence
was latent rather than active.

`bl_mynn_mixscalars` is now admitted at `{0, 1}` and implemented:
`gpuwm/core/mynn_scalar_mix.py` and its device twin carry WRF's own qn solves
(`module_bl_mynn.F:4654-4860`) plus the `scalar_opt > 0` DMP updraft-flux
terms, and `gpuwm/core/mynn_pbl_runtime.py` drives the `flag_qn*` flags true
for the five stock species when the switch is 1. The default remains
`bl_mynn_mixscalars = 0` — WRF's Registry default — so the shipped default
trajectory is unchanged; what changed is that lighting the switch now mixes
the qn family as WRF does instead of silently doing nothing.

The `1` arm is admitted only under the combination its anchored oracle
fixtures were generated at and the runtime was wired for: `bl_pbl_physics = 5`
(MYNN), `mp_physics = 28` (the one scheme whose state carries the qn family),
and `bldt = 0`. Anything else refuses by name. (Snow *is* mixed:
`flag_qs` is true for mp=28, matching `Registry.EM_COMMON:3036`, which was
a separate defect and is closed.)

### 6.5 ArWen now runs the configuration `real.exe` admits — **D9a CLOSED**

This section used to say the opposite. `dyn_em/module_initialize_real.F:2734-2736`
fatals with `wif_input_opt=0 but mp_physics=28`, and ArWen had no WIF ingest,
so it ran the case WRF's initializer refuses and an ArWen mp=28 run was
**not** directly comparable to a WIF-initialised WRF mp=28 run.

That is no longer the configuration ArWen runs. `gpuwm/ingest/wif_climatology.py`
ports WRF's global monthly QNWFA/QNIFA climatology
(`QNWFA_QNIFA_SIGMA_MONTHLY.dat`) — metgrid's `four_pt` horizontal
interpolation, `real.exe`'s `monthly_interp_to_date` temporal weighting, and
its `vert_interp` onto the dry eta pressure — matched to `real.exe` to 1e-5,
and a real-data mp=28 run takes it **by default**
(`RunConfig.mp28_aerosol_source = "auto"`). That is the state `real.exe`
reaches through `use_aero_icbc = .true.` → `aer_init_opt = 1` with
`wif_input_opt = 1`, so the two initial conditions are the same one and the
runs **are** directly comparable. The constant that carried the old warning,
`gpuwm.config.MP28_AEROSOL_SOURCE_DEVIATION`, was retired with the deviation;
`MP28_AEROSOL_SOURCE_DEFAULT` and `MP28_AEROSOL_SYNTHETIC_FALLBACK` replace it.

**What survives.** When no dataset can be located, `"auto"` falls back to
`thompson_init`'s synthetic CCN/IN profile — a real initial condition, but a
different experiment — and says so by name in the run receipt. A run that fell
back is not comparable to a WIF-initialised WRF run;
`mp28_aerosol_source = "climatology"` refuses rather than falling back, and
`"synthetic"` selects the fallback deliberately. (The `real.exe` citation was
**re-verified** against a full stock WRF v4.6.1 tree: `:2734-2735` is the
`ELSE IF (config_flags%mp_physics .EQ. THOMPSONAERO .and.
config_flags%wif_input_opt .EQ. 0 )` test and `:2736` is the
`CALL wrf_error_fatal`. Earlier revisions of this page said it had not
been checked, because the reference tree used then held only `phys/`.)

### 6.6 mp=28 and mp=8 are deliberately not bit-identical

`thompson_aerosol_common.cuh` contraction-pins the `RSLF`/`RSIF` Horner
chains; the byte-frozen `thompson.cu` leaves them contracted. nvrtc defaults
to `--fmad=true` and the gfortran baseline-x86-64 reference has no FMA
instruction, so the unpinned chain lands one ULP low — and
`module_mp_thompson.F:3401` opens the whole condensation/CCN-activation
block on `ssatw > 1.E-15`, so one ULP flips a branch. mp=28 matches WRF;
mp=8 keeps the arithmetic its model-validated trajectory was measured on.
Deviation **D9g**. This is intended, and "make them agree" is the wrong
fix in both directions. (The `:3401` citation was `:3400` in every earlier
publication of this deviation, including this page; `:3400` is
`orho = 1./rho(k)` and the branch is one line lower. Re-verified for this
revision against both reference WRF trees, which hold byte-identical copies
of the file.)

**It is also measured, and it was RE-measured for this revision** — by
substituting the unpinned Horner chains into the header the loader hands
nvrtc, with no file on disk edited, and re-driving all 22 columns. Removing
the pin takes the unexceptioned clean count from 17 of 22 to **4 of 22**
(only `aero-ice-demott-dep`, `aero-scav-frozen`, `aero-sfc-emit` and
`wp08-melt` survive), and the damage is not subtle: `aero-nc-accrete`,
`aero-nc-auto`, `aero-nc-cap`, `aero-nc-sed`, `aero-scav-rain` and
`aero-warm-overlap` all go to relative differences above 1e+20 in `qc` and
above 1e+36 in `nc` — the branch flip, not a rounding change — and
`aero-ice-koop` returns to `qi` 7.06e-03 / `ni` 8.79e-03. The pin is not a
style choice.

### 6.7 What left this section since the previous revision

Five items that were published here as open are closed, and each was
re-checked for this page rather than taken on report. **One of the five
was never open in the first place**, and it is marked as such rather than
counted as a win:

| was | now |
| --- | --- |
| "The synthetic aerosol profile is never installed — **largest**" | closed by wiring the call: `gpuwm/core/physics.py::initialize_physics`. §6.1 is the same measurement, reframed from a cost into a sensitivity |
| "An mp=28 run cannot be checkpointed, therefore cannot be resumed" | closed: `MICROPHYSICS_ALGORITHM_IDENTITIES` carries a 28 row, and a running mp=28 forecast now checkpoints |
| "`REFL_10CM` is never computed in a runtime-driven mp=28 forecast" | closed: `gpuwm/runtime.py`'s `REFL_10CM_MICROPHYSICS` is `(1, 6, 8, 10, 18, 28)` and one admission constant replaced three open-coded tuples |
| "Homogeneous haze freezing (Koop) is the weakest process" | **withdrawn, not fixed.** It was never measured against WRF's own Exner function; correcting `run_column_aero.F90`'s `287.0/1004.0` and regenerating the deck removed the residual with no kernel change at all. §3.4 |
| "Packaging: `CCN_ACTIVATE.BIN` would reach a wheel" | closed, then **superseded the same day**: the exclusion entry landed and the three packaging gates went green, after which the owner reversed the do-not-ship decision. The entry was removed, the gates were inverted to assert the table *does* reach the wheel, and all three are green in that direction |
| "`aero-drop-evap` and `aero-ice-demott-idxin` miss the gate on their surface accumulators" | closed **in this revision**, on the physics: WRF's level-wise `:3237` / `:3568` sedimentation density was restored and both fixtures now clear the flat gate on all 23 quantities. §3.3 |
| "`_REFL_DB_BOUNDS` — a third allowance on `aero-reduces-to-classic`" | **retired in this revision**: the residual it existed for fell to 3.242e-05 dB, inside the flat 2.0e-4 dB gate, and the constant is now an empty dict. §3.2 |

That is six repairs and one withdrawn claim, against two recorded
regressions in `aero-cold-overlap` — the level-4 rows, which the fixture
regeneration caused, and the level-6 `qr` growth, which the contraction pin
caused and which is published as grown rather than reverted — and one
newly-published pair of residuals (`wp08-*`). It is recorded this way so
the next reader can check the direction of travel rather than infer it —
and so that a correction to ArWen's *measuring instrument* is never
reported as a correction to ArWen's *physics*.

---

## 7. The mp=8 non-regression receipt

mp=28 lives in six new CUDA translation units plus one standalone helper
probe (seven `.cu` files, sharing one new `.cuh`) and ten new Python
modules.
`gpuwm/core/kernels/thompson.cu` and `gpuwm/core/thompson.py` are
byte-frozen, and mp=8's numerics are unchanged **by construction**: the
kernel loader compiles one module per `.cu` file from
`_preamble() + <name>.cu`, so an unedited file is an unchanged source string
and therefore unchanged PTX.

Measured rather than asserted:

| receipt | result |
| --- | --- |
| `thompson.cu` sha256 | unchanged from the pin captured at commit `789f611` |
| assembled nvrtc source string for module `thompson` | unchanged (catches a preamble or `CUDA_DEFINES` change the file hash alone would not) |
| `gpuwm/core/thompson.py` sha256 | unchanged |
| `CLASSIC_TABLE_ASSETS` / `TABLE_SET_ID` | unchanged; `CCN_ACTIVATE.BIN` absent |
| clean rebuild of the mp=8 oracle from a pristine WRF v4.6.1 tree | **re-run for this revision on 2026-08-01.** `tests/test_mp8_frozen.py` goes from 21 passed / 1 skipped to **22 passed** with `GPUWM_MP8_ORACLE_REBUILD_DIR` set: 4/4 generated `.dat` SHA-256s match the pins, and an independent byte comparison of the regenerated column oracle against the committed deck gives **88 of 92 CSVs byte-identical** |

The four CSVs that differ are `warm-column.csv`, `ice-column.csv`,
`mixed-column.csv` and `mixed-surface.csv`, and only in the *input* `p_pa`
and the `before` rows that follow from it — a one-float32-ULP
input-seed provenance drift that predates this port (max relative difference
7.26e-06, 1.30e-02, 4.06e-06 and 1.13e-07 respectively; the `mixed-column`
figure is a single near-zero field). This is the documented exception in
`tools/mp8_freeze_receipt.py::ORACLE_REBUILD_EXCEPTIONS`, and this port did
not introduce it and did not widen it.

Re-run it with:

```sh
bash tools/thompson_wrf461_oracle/build.sh /path/to/WRF-v4.6.1 /empty/dir
GPUWM_MP8_ORACLE_REBUILD_DIR=/empty/dir python -m pytest tests/test_mp8_frozen.py -q
```

---

## 8. Skip census — every test in the mp=28 suite that can decline to run

A skipped test is an unmeasured claim wearing a green tick. Seven silent
skips are how two critical defects survived wave 2 of this port, so the
complete census is published and machine-checked:
`tests/test_thompson_aerosol_gpu.py::test_the_mp28_suite_has_no_unaudited_skip_site`
fails if a skip site is **added** anywhere in the suite, as loudly as if one
were removed. Nothing in the suite uses `xfail`, and that too is asserted.

The suite is these thirteen modules: `test_kernel_loader_inert.py`,
`test_mp28_forecast_smoke.py`, `test_mp28_runnable.py`, `test_mp8_frozen.py`,
`test_thompson_aerosol_adapter.py`, `test_thompson_aerosol_cold_gpu.py`,
`test_thompson_aerosol_contract.py`,
`test_thompson_aerosol_device_helpers.py`, `test_thompson_aerosol_gpu.py`,
`test_thompson_aerosol_sat_gpu.py`, `test_thompson_aerosol_sed_gpu.py`,
`test_thompson_aerosol_state_gpu.py`, `test_thompson_aerosol_warm_gpu.py`.

**One mp=28 gate is deliberately outside that census, and its skips are
stated here instead.** `tests/test_physics_md_aerosol_claims.py` is the
publication gate — it checks this page and `PHYSICS.md` against the
adapter gate — so it is not part of the physics suite the census covers.
Twenty of its twenty-two tests are host-only and never skip. The two that
drive the device (`test_the_published_aerosol_sensitivity_is_a_live_measurement`
and `test_the_page_republishes_the_measured_depletion_numbers`) each carry
a `pytest.importorskip("cupy")`, a device-count check and the same
`CCN_ACTIVATE.BIN` guard the suite uses; on a machine without a device or
without the table they skip, and the sixteen documentary assertions still
run and still fail closed. Adding the module to `_SUITE_MODULES` and
`_SKIP_SITES` would make that machine-checked rather than stated, and is
filed as an integration request against
`tests/test_thompson_aerosol_gpu.py`, which this package does not own.

### 8.1 What actually skipped on the measurement machine: 14 of 679

| count | module | reason | is it an unmeasured claim? |
| --- | --- | --- | --- |
| 12 | `test_kernel_loader_inert.py` | the six `thompson_aerosol_*` translation units are allow-listed to receive `thompson_aerosol_common.cuh`, so "assembled source == preamble + file" is **false for them by construction** | **No.** The same file asserts the positive property for those six directly, and asserts byte-identity for every other module including `thompson` itself. The skip excludes the six from a test that is about the *other* modules. |
| 1 | `test_mp8_frozen.py::test_clean_oracle_rebuild_matches_except_the_four_documented_files` | opt-in: needs gfortran, the pristine WRF tree and ~380 MB of regenerated tables, gated on `GPUWM_MP8_ORACLE_REBUILD_DIR` | **It was, until this page.** WP-12b built the oracle and ran it; the result is §7. It skips again on a machine without gfortran. |
| 1 | `test_thompson_aerosol_device_helpers.py::test_local_reimplementations_agree_with_the_shared_definition` | "no renamed local re-implementations remain" — the test lifts every duplicated device function out of the `.cu` files and compares it against the shared header's definition, and there are none left to lift | **No, and this is the desired end state.** The uniqueness of the shared definition is asserted positively by `test_shared_helpers_are_defined_exactly_once_and_only_in_the_header` in the same file, which does not skip. The skip fires because the duplication it existed to police is gone. |

### 8.2 Skip sites that exist but did not fire here

These are the conditional paths a different machine would take. Each is
audited and pinned in `_SKIP_SITES`.

| condition | what stops being measured | how bad |
| --- | --- | --- |
| `CCN_ACTIVATE.BIN` absent (`_tables_or_skip`, 4 modules; two `skipif` markers in `test_thompson_aerosol_device_helpers.py`; one in `test_thompson_aerosol_contract.py`) | **every device gate for mp=28**, including all of class A | Severe, but **no longer expected to fire**: the asset ships with ArWen as of 2026-08-01 (deviation D9i, reversed), so a clean checkout validates mp=28. The guards are kept as defence for a tree where the file was deleted or `GPUWM_THOMPSON_CCN_ACTIVATE` points elsewhere. The skip names the one file rather than swallowing every load failure, so it can never be mistaken for a pass. |
| no CUDA device (`_require_device`, `pytest.importorskip("cupy")`, and `conftest`'s automatic `gpu` marker) | every device gate | Expected. The host-only gates — call-graph order, guards, registry, namelist, contract parsing — still run and still fail closed. |
| fixture or scratch-oracle file absent (`test_thompson_aerosol_sed_gpu.py::test_embedded_seed_matches_the_committed_fixture`, one in `test_thompson_aerosol_device_helpers.py`) | individual bitwise cross-checks | Narrow. Each is a second opinion on something a class A fixture already covers. |

**Two skip sites were RETIRED from this census, not relaxed.**
`test_thompson_aerosol_state_gpu.py`'s
`test_state_finalize_rounds_every_real4_subexpression_that_feeds_a_double`
and `test_effective_radius_is_bitwise_against_a_fortran_faithful_host_sweep`
both began `if _LIBM is None: pytest.skip("libm.so.6 unavailable")`. They
need the glibc `powf` that gfortran lowered the oracle's `REAL(4) ** REAL(4)`
to, and the suite reached it with `ctypes.CDLL("libm.so.6")` — so on any host
that is not a glibc host, two bitwise gates silently did not run. They now
call `gpuwm.core.noahmp_libm.powf`, a bit-exact transcription of the same
glibc 2.39 `sysdeps/ieee754/flt-32/e_powf.c`. What is compared is unchanged;
the condition that could stop it being compared is gone, so the skips are
gone with it and the two gates run everywhere.

### 8.3 Suite result at the time of measurement

**679 collected: 664 passed, 14 skipped, 1 failed.**

The failing test is named rather than hidden, because an honestly red gate
is the point of this document:

| test | why it is red |
| --- | --- |
| `test_thompson_aerosol_adapter.py::test_g3_end_to_end_against_all_nineteen_oracle_fixtures` | **the §3 gate.** The six fixtures in §3.1 (excluding the allowanced `aero-reduces-to-classic`) are why. This is the port's headline number and it is red on purpose. |

Two gates that were red when this page was last written are green now: the
mp=8/mp=28 sedimentation bridge ratchet and the reflectivity-residual count
ratchet in `test_mp28_runnable.py`.

The **publication** gate that reads this page and `PHYSICS.md`,
`tests/test_physics_md_aerosol_claims.py`, is **18 passed, 0 skipped, 0
failed** on the same machine — including the two device tests that re-run
§6.1's pair of forecasts and §5.1's depletion measurement.

Outside the mp=28 suite, three groups of gates are red on this machine and
all three are named rather than left for a reader to find. **None is caused
by this port**, and saying so is only worth anything with the cause
attached:

* `test_line_ending_stability.py::test_every_hashed_file_matches_its_committed_blob`
  compares tracked files against their committed blobs, which cannot hold on
  a working tree with uncommitted integration edits in it.
* `test_real74_rungs.py` (29 tests) and `test_clock.py` (18 collection
  errors) both load `configs/real74_4dom.toml`, whose `[case_data]` names an
  ERA5 GRIB file under `~/Downloads` that this machine does not have. The
  error is `forcing file … does not exist`, raised by `gpuwm/case_data.py`
  at config load, before any physics runs.
* `test_native_wrf_distribution.py` (2 tests) raises
  `importlib.metadata.PackageNotFoundError: No package metadata was found
  for gpuwm` — this virtualenv runs the tree from `sys.path` rather than
  from an installed distribution. Same class as the `setuptools` skips
  above.

Two further modules are excluded from every count on this page, on the
same grounds and with the same honesty: `tests/test_flagship_tools.py`
cannot be **collected** (no `matplotlib` in this virtualenv) and every test
in `tests/test_rrtmg_sw_cuda.py` fails with `NVRTC_ERROR_INVALID_OPTION`
because this cupy appends `-ftz=true` to a compile line that already
carries `--ftz=false`. All three RRTMG source files are byte-identical to
the branch point, so that is a toolchain fact, not a port fact.

**What was NOT run.** The whole-tree suite (~7,400 tests) was not carried
to completion for this revision. What was run is the 13-module mp=28 suite
in full, the publication gate in full, and every one of the 24 test modules
that reads a file this work edited — 888 passed, 12 skipped, and no failure
outside the three environment groups above and the §3 gate itself.

**The unsatisfiable pair the previous revision recorded here is resolved.**
`test_physics_md_aerosol_claims.py::test_the_workaround_the_page_prints_is_real`
required `docs/public/PHYSICS.md` to contain the literal
`gpuwm.core.microphysics.microphysics_init(state, cfg)`, while its sibling
`test_physics_md_does_not_claim_the_synthetic_profile_is_installed`
required — once `microphysics_init` had a production caller — that the page
NOT contain `microphysics_init(state, cfg)`. The second string contains the
first, so no page satisfied both. Neither test was deleted and neither
constraint was relaxed. The workaround test became
`test_the_production_call_site_the_page_names_is_real`, which asserts
strictly more than it did: the hook still has to exist, still has to take
`(state, cfg)` and still has to be the documented no-op away from mp=28,
*and* the caller the page names has to be real, has to be the only one, and
the page must print no manual workaround while it exists. Both halves still
fail in both directions — with the call site removed, the workaround
requirement comes straight back.

The packaging defect of the previous revision's §6.9 is closed: the
`[tool.setuptools.exclude-package-data]` entry landed, and
`tests/test_package_data_coverage.py` is 3 passed / 4 skipped here rather
than red. Stated exactly, because the skip is the interesting part: the
gate that runs everywhere -- the one that reads the declaration out of
`tomllib` -- is GREEN, and the four that measure real wheel contents SKIP in
this virtualenv because it has no `setuptools`. That is the same asymmetry
`PROVENANCE.md` D9j records, and it is why the declaration gate exists: a
suite that only had the setuptools ones would have been silently inert in
the environment this port is developed in.

---

## 9. Reproducing everything on this page

```sh
# Class A, all twenty-two fixtures, end to end (and the ratchet on §3):
python -m pytest tests/test_thompson_aerosol_adapter.py \
                 tests/test_thompson_aerosol_gpu.py -q

# Class D, the forecast and the depletion measurement (§5, §6.1):
python -m pytest tests/test_mp28_forecast_smoke.py -q -s

# The mp=8 freeze, including the empirical rebuild (§7):
bash tools/thompson_wrf461_oracle/build.sh /path/to/WRF-v4.6.1 /empty/dir
GPUWM_MP8_ORACLE_REBUILD_DIR=/empty/dir \
  python -m pytest tests/test_mp8_frozen.py -q

# This page and docs/public/PHYSICS.md against the gate that owns the
# numbers, including §3.4's Exner re-derivation and §6.1's two forecasts:
python -m pytest tests/test_physics_md_aerosol_claims.py -q -s
```

Every number in §3 and §5 is recomputed by
`tests/test_thompson_aerosol_gpu.py` and
`tests/test_mp28_forecast_smoke.py` and compared against what is printed
here. If this page and the code disagree, those tests go red — including
when a residual **improves**, because an evidence document that understates
the port is still a document nobody re-read.

**And the publication layer is now derived rather than transcribed.**
`tests/test_physics_md_aerosol_claims.py` imports the adapter gate's own
pinned partition (`_G3_UNEXCEPTIONED_CLEAN`, `_G3_GATED_CLEAN`,
`_G3_RESIDUALS`, `_END_TO_END_BOUNDS`) and rebuilds this page's and
`PHYSICS.md`'s counts, residual tables and carve-out from it, in both
directions: a fixture the gate closes may not stay in a miss table, a
fixture it misses may not leave one, and a residual may not be quoted at a
value the gate does not measure. The §6.1 sensitivity table is re-run on
the device rather than cross-referenced, and §3.4's Exner claim is
re-derived from the committed CSVs. Restating a number in prose is the
defect that recurred in every wave of this port; it is now a gate rather
than a review item.

---

## 10. What would move the label

| to reach | what is required | what is missing |
| --- | --- | --- |
| a clean `implemented-unverified` | all twenty-two class A fixtures inside 2.0e-6 with no allowance | seven fixtures, §3.1 |
| `validation-candidate` | a ratified reference comparison | there is none; §5 is self-consistency only |
| `model-validated` | a matched multi-hour ArWen-vs-WRF forecast with published decay tables | no matched run of any length exists. The wiring gaps that used to have to close first (§6.7) have closed, so what is left is the comparison itself — and §6.2 would have to be closed or bounded for a multi-hour comparison to mean anything, because after `L/U` the ArWen domain is aerosol-free and the WRF one is not |
