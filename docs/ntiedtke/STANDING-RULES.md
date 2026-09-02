# New Tiedtke port — standing rules and the road to done

Written 2026-08-29 by the reviewing session (review) and handed to the
porting session to own. **This file is the operating contract for the rest of
the port.** It exists on disk rather than in a conversation because both
sessions compact, and everything below was learned the expensive way.

the owner's goal, in his words: *a fully functional port of the cumulus scheme,
and it being ideally optimised / low on VRAM / efficient.* Correctness is the
gate; efficiency is the deliverable alongside it, not after it.

---

## 1. The rule that replaces review

Every defect the review caught fell into two classes. Neither is findable by
being more careful, and both are invisible to `max_ulp == 0`.

**Class 1 — a receipt masquerading as a gate.** The oracle digests (7 of 12
stale, and `build.sh` only ever wrote the file). The aliasing audit (a text
file nothing re-ran). The tile invariant (a paragraph). Each time, something
was recorded and nothing checked it.

**Class 2 — something WRF cannot disagree with.** Launch geometry, workspace
threading, tile decomposition, and every fixture that synthesises its own
inputs. Parity is structurally blind here: there is no Fortran analogue to
compare against, so the grade passes no matter how wrong the thing is.

> **THE RULE: if WRF cannot disagree with it, it needs its own gate.**

Applied to yourself, every slice: when you write a fact into a file, ask what
re-checks it. When you make a decision the oracle cannot see, ask what fails
when someone breaks it.

## 2. The harness lesson, which is class 2 in its most expensive form

Three defects, one structure:

| where | what the harness did |
|---|---|
| `cutypen` | passed FRESH arrays where `cumastrn` passes live ones |
| slice 4a | SKIPPED `cumastrn:500-541`, so `pmfub` was zero |
| slice 4b | never captured `paph` at the surface interface |

Each time oracle and mirror agreed — with each other, about a world WRF never
visits. **A fixture that synthesises a routine's inputs instead of producing
them by running the real chain is the only failure mode in this port that a
green grade cannot see.**

The fix is the capture architecture: `run_nt_cumastrn.F90` replicates
`cumastrn`'s body statement-by-statement, calls the real globalised
procedures, captures at every boundary, and proves itself against a real
`cu_ntiedtke_run`. Interposition was measured dead first
(`probe_interposition.sh`) — gfortran binds internal calls directly, never
through the PLT, even under `-fsemantic-interposition`.

**THE RULE, sharpened.** "Capture the inputs" was too weak. The closure
slice failed on six columns because `cumastrn:566-568` flips `ktype`
between `cuascn` and the closure — two points that look adjacent and are
not, with a promotion rule sitting between them.

> **Capture every value at the point the routine under test READS it.**
> Provenance can change between two points that look adjacent.

Fifth instance of class 2, and the first in the GRADING harness rather than
the fixture. It cost two rounds of reasoning about stale inputs; capturing
the arm's own intermediates found it in one run, by the capture file coming
back EMPTY.

**A mechanism that explains the pattern is not thereby the mechanism.** The
reviewing session predicted a stale-index bug from the fact that only the
shallow arm reads through a near-cancellation — which explains the observed
selectivity exactly. So does an arm-SELECTION bug, which is what it was.
The selectivity argument could not discriminate; only the capture could.
When a hypothesis fits the signature, that is a reason to test it, not to
believe it.

**Known limit, and say so wherever "proves itself" appears:** zero differing
words at the driver's OUTPUT proves the replication converges. It does not
prove every intermediate capture is right — a replication could differ
internally and still land in the same place. Intermediate correctness rests
on statement-order fidelity plus every callee being the real procedure. That
is an argument, not a measurement, and with interposition dead no measurement
is available. *Clears is not the same as safe.*

### CAPTURE FIRST. Reason about what survives only if capturing is impossible.

**The default is inverted, 2026-08-29, after the sixth instance** (review,
review). §2 already says capture where the routine reads. What keeps
failing is that a neighbouring capture is *cheaper* and an argument for it
is *always available* — so the reasoning happens first and the capture is
the fallback after it fails.

Instances five and six were both the same decision, not two accidents:

* `:566-568` flips `ktype` between cuascn and the closure — feeding
  cuascn's ktype ran the wrong arm.
* `pmfude_rate` looked identical at cuascn's exit and the adjustments
  block's entry; `:746-819` rescales it in between, and the mirror came
  out **1.26× low on 42 of 108 columns**.

**Both are resolution by apparent identity instead of by provenance**, and
the `zcons`/`zcons2` family is the same error in constants — same name,
different scope, factor of three, one character apart. That one is now
gated (`test_ntiedtke_constant_family.py`), built *before* porting `zcons`'s
only consumer rather than after the second misreading.

So for every remaining range: **add the capture at the real boundary as the
first action.** A rebuild costs one command and proves itself at 0 differing
words. Reasoning about what survives costs a slice and has been wrong six
times.

## 3. Phases, and what "done" means

Do not skip forward. Each phase gates the next.

**Phase 1 — transcription.** `cumastrn` replication + closure, then `cuascn`,
`cuflxn`, `cuddrafn`, `cudlfsn`, `cududvn`, `cudtdqn`. Every routine graded at
`max_ulp == 0` against a capture, never a synthesised fixture. Re-grade the
three ad-hoc reconstructions already in the tree (midlevel prologue, mfub
inputs, cutypen pre-state) against captures and report whether anything moved.

> **AND PHASE 1 CARRIES ONE CASE-TABLE ITEM**, not only code: a case that
> exercises `cumastrn:566`, the deep-to-shallow demotion. Measured, the
> flip fires only shallow-to-deep on the current fixture (6 columns,
> `:567`); `:566` never runs.
>
> It is a completion item rather than a standing gap because of what it
> costs to leave: at dx = 4500 the demotion moves a column from
> `1/scale_fac` = 8.6% to `1/scale_fac2` = 29.3% retained mass flux —
> **3.4x, larger than the entire GF-vs-New-Tiedtke gap the port was
> justified on** — and thin marginal-depth columns in an eyewall ring are
> both the population most likely to take it and the population that
> decides whether the port works.
>
> The failure mode is the argument: a reference tropical-cyclone run whose eyewall columns
> take an ungraded transition does not crash and does not look wrong. A
> disappointing f012 could not then be attributed to the physics or to
> this arm. **Unfalsifiable after the fact, cheap to prevent now.**
>
> The case table brackets it — §9's case 1 is too deep to demote, cases
> 8-11 too weak to trigger — so the target is a column deep enough for
> cutypen and cuascn but capped at modest depth. Expect the case-11 shape:
> two conditions at once, several rounds. **If it resists, say so rather
> than carrying it forward.**

> **PHASE 1 ENDS WHEN THE ASSEMBLED PIPELINE REPRODUCES `nt-levels.csv`
> BITWISE** — not when the last kernel grades. Pinned 2026-08-29 on review
> (review), because "thirteen of thirteen kernels graded" reads like
> done and is not.
>
> Every routine grading green against captures **the orchestration did not
> produce** is the same circularity the capture architecture was built to
> retire, one level up. The kernels are graded against WRF's intermediate
> state; nothing yet grades the glue that carries state between them.
>
> **And grade the assembly at EVERY capture boundary, not only end to
> end.** §8 records a real limit of the Fortran replication's self-proof:
> zero differing words at the output proves it converges, not that every
> intermediate is right — "an argument, not a measurement, and with
> interposition dead no measurement is available." The assembled CUDA
> pipeline has the identical structure, **but here the measurement IS
> available**: the captures already exist at every routine's call
> boundary, so the pipeline can be compared at each one.
>
> Two reasons beyond the principle. The boundaries **bracket the unowned
> ranges**, which is exactly where the risk is — post-flip `ktype` at the
> closure's entry brackets :566-568, cudlfsn's entry brackets :580-588,
> cududvn's entry carries `zmfuus`/`zmfdus` and so brackets the :833-915
> rescale. And **bisection**: "wrong somewhere in 306 lines plus fourteen
> kernels" is a bad place to start at 4am; "first diverges at boundary N"
> is a good one.

**Phase 2 — integration.** `CU_SCHEMES` entry in `gpuwm/config.py`, config
validation beside the existing `cu_physics` checks, the cumulus driver, the
tendency application path, `cudt`, and the prepared-cache identity fields
accepting `cu_physics = 16`. Ends when a forecast runs to completion.

**Phase 3 — the correctness gate.** Standing rule 1: two runs of unchanged
code produce byte-identical wrfout — `cmp` **every wrfout file the run
produces**, with the count recorded. A change meant to be
exact that isn't is REVERTED, not accepted with a tolerance. Also measure peak
VRAM on a real run, not a probe.

> **CORRECTED, 2026-08-29, by review (review). THE MOMENTUM-EXTENSION
> GATE AS ORIGINALLY WRITTEN PROVES THE WRONG PROPERTY**, and this matters
> because it is the one place a mistake costs the owner a campaign baseline
> rather than a test.
>
> **"33" WAS A PROPERTY OF ONE CONFIG AND IT OUTLIVED IT.** The number
> came from `_profile_1h.toml`'s cadence, and that config -- with its
> whole cycle directory `E:/GPUWRF/runs/2026-08-14_18`, its prepared
> root and both pinned sha256 values -- is no longer on disk. The gate
> is now stated over what the run produces, with the count recorded as
> evidence rather than asserted as a constant (review). Same shape
> as every other receipt-versus-gate correction here, aimed this time
> at a standing rule.
>
> The brief said: two runs, byte-identical wrfout, `cmp` all 33 — passing
> with the momentum extension in place and no scheme using it, *before* New
> Tiedtke is wired up. **Two runs of the same build prove DETERMINISM. The
> property that constraint exists to protect is INERTNESS** — that adding
> momentum slots to `CumulusResult` and the tendency-application path does
> not change what GF and KF already produce.
>
> Those are different properties and the gate cannot distinguish them. **A
> deterministic change in the answer passes it perfectly**: if the extension
> shifts an accumulation order, changes a buffer shape, or perturbs the
> optional-components set `PhysicsTendencies.zeros` is built with, then both
> runs move identically, `cmp` reports byte-identical, the gate goes green,
> and GF's forecast has moved. That is the single outcome the constraint was
> written to prevent, and it is the *likely* failure mode rather than an
> exotic one.
>
> **The gate that proves inertness is a comparison against a PRE-EXTENSION
> baseline**, not against a second post-extension run:
>
> 1. **Before touching `CumulusResult`:** one profile-config run per scheme,
>    sha256 of every wrfout file the run produces, recorded with the
>    count and with the commit hash it
>    launched at. Digests, not files — equality does not need the GB, and
>    this disk has lost 2 TB once.
> 2. **GF *and* KF.** Both go through `CumulusResult` and the
>    tendency-application path, KF through the more complex NCA branch, and
>    both `run_myj` and `run_kf` wrfout are kept deliberately as baselines
>    the structural diagnostics read 3D fields from. An extension inert for
>    GF and not for KF still costs a baseline.
> 3. **After the extension:** same config, same commit-hash discipline,
>    compare digests. 33 of 33 identical is inertness. Anything else is a
>    real finding regardless of size.
> 4. **Keep the determinism run** as the weaker second check — it catches
>    nondeterminism the comparison would mask if both sides drifted the same
>    way. One extra run is cheap insurance.
>
> **Cheap filter first, minutes rather than a run:** the existing GF and KF
> parity suites must still pass untouched. They are not *sufficient* — they
> grade the scheme kernels and the extension touches the tendency-application
> path around them, which is exactly what those suites do not cover — but
> they are the fastest way to learn you have already broken something.
>
> **Operational.** The baseline runs are forecasts, and a commit during one
> kills it at the finish line. `check_no_forecast.sh` protects against other
> sessions; it does not stop you racing yourself. Land or stash whatever you
> are holding before starting one.
>
> **These runs are the owner's GPU and the owner's call.** Ask before launching.

**Phase 4 — efficiency.** See §4. Never optimise an ungraded routine.

**Phase 5 — the point of all of it.** Run the reference tropical cyclone with `cu_physics = 16` at
13.5/4.5 km and compare f012 against: best track 958 mb, native WRF (Tiedtke)
968.1, ArWen KF 971.6, ArWen GF 977.6. The forecast is the actual bar —
396 GF parity tests pass and GF is still the wrong scheme for this storm.

## 4. Efficiency, without losing the port

**Bitwise parity is the constraint, not a target to trade against.**
`max_ulp == 0` must still hold after every optimisation, and the Phase 3
determinism gate must still pass.

*Forbidden* — these change arithmetic: reassociation, fast-math, FTZ changes,
fusing or splitting operations, changing the order of a reduction. Note ptxas
contracts by local register pressure, so a **runtime branch can leave two
clones of the same arithmetic rounding differently** — that has already failed
a bitwise gate in this project and the fix was a branch-free shape. Keep every
expression pinned to `__fmaf_rn` / `__fmul_rn` / `__fadd_rn`.

*Permitted* — these do not: memory layout and workspace packing, coalescing,
occupancy and register pressure, eliminating redundant loads, launch geometry
within the descriptor's contract.

**Measure before optimising.** `HOWTO-RUN-A-SIM.md` §13 and the 1-hour profile
loop in `CLAUDE.md`. **Drop `cumulus` from the `microphysics + cumulus + pbl`
clock proxy** — the change is IN cumulus, so leaving it in measures itself.
This box drifts ±20% run to run: ≥3 runs, take the median, and anything under
~3% is not measurable.

**Ablate first.** Delete the work before building the optimisation, to size
the ceiling. A compile-only frame/register probe costs seconds and has killed
multi-day plans in an hour. Estimating a saving from section size has been
wrong every time it was tried in this project.

**Three traps, each of which produced a wrong conclusion here before:**
a kernel-level A/B can rank things backwards against the model (the model is
the authority, the microbenchmark is a filter); SASS text comparison gives
false positives on label renumbering, only the numbers settle it; the
peak-VRAM sampler polls at 50 ms and reports one maximum, not a timeline.

**VRAM targets.** Frame stays 0 B on every kernel — that is what keeps the
local-memory reservation at zero. The reservation law is
`(frame − 1,024) × 107,520` on this card; a frame at or under the 1,024 B
default stack reserves nothing. The term that grows as physics lands is the
**workspace**: it was ~357 MiB against GF's 422 MiB at the skeleton stage, a
net −65 MiB. Re-measure it every slice and report it. Standing rule 3: peak is
~11.3 GiB of 15.92 and a change costing >50 MiB has to earn it. **If the port
ends up costing more VRAM than GF, that is a result to report, not to hide.**

Registers today: prep 40, convert 34, cuinin 72, cutypen 91, midlevel 40,
mfub 35. `cutypen` at 91 is the occupancy risk; it is a Phase 4 question, not
a Phase 1 one.

## 5. Autonomy: when to stop, and who to tell

Run without checking in. Report to **the owner** at phase boundaries only — not per
slice — with what moved, what is graded, workspace and registers, and anything
excluded from grading.

**Stop and message the reviewing session (`review`) only if:**

* a slice will not grade and one attempt has already failed
* a new class-2 blind spot appears — something the oracle cannot see
* any kernel's frame stops being 0 B, or the workspace exceeds GF's 422 MiB
* an optimisation cannot hold `max_ulp == 0` and you are tempted by a tolerance
* anything approaches the owner's uncommitted work

**the owner's working tree carries ~45 uncommitted paths from a live forecasting
campaign and they are not reconstructible from git.** Stage explicit paths.
Never `git add -A`.

### NEVER DELETE ANYTHING UNDER `E:\GPUWRF\runs`. Not to make room, ever.
uns`. Not to make room, ever.

the owner has authorised this port's compute — including a reference tropical-cyclone run at Phase 5,
bounded at 24 h — and asked that we watch disk. **Measured 2026-08-29:
`E:` has 2,251 GB free of 3,726.** Existing runs: `run_kf3` 161.3 GB,
`run_myj` 67.1, `run_kf` 55.5, `run_hafs` 28.8. Three runs at the high-water
mark is under 500 GB. **Disk is not a constraint and will not become one
from this port.**

Which makes the rule the inverse of the instruction. Because space is
plentiful there is no legitimate reason to free any — and the two
directories that most look like reclaimable bulk are the two that must never
be touched:

* **`run_myj` (67.1 GB) — the GF baseline**
* **`run_kf` (55.5 GB) — the KF baseline**

`tools/tc-intensity-diagnostics/` reads 3D fields from both. ~2 TB of older
wrfout was already deleted with only tracks surviving under
`output/_tracks_kept/`, so these two are what is left. **Every intensity
number Phase 5 is graded against traces back to them, and they are not
reconstructible without re-running the campaign.** If space is ever
genuinely needed, that goes to the owner — not to a session's judgement. Same
class as his uncommitted working tree: irreplaceable, and the loss would be
silent until something tried to read it.

Report `E:` free space with the run receipt. Do not design around a limit
that is not there.

### A COMMIT KILLS A RUNNING FORECAST. Check before every single one.

"Never commit or edit tracked source while a forecast is running" is easy to
get half-right, and getting it half-right cost a run. The obvious reading is
that EDITING is the hazard and the commit is incidental. It is not.

`prepared_domain_tree_forecast.py:1374` `_runtime_source_identity()` returns
`gpuwm_version`, **`git_commit`**, **`git_tree`**, and `source_sha256` over
five files. It is captured at launch and **re-checked at completion**
(`:2307`), raising
`RuntimeError("forecast implementation changed during execution")`.

`git_commit` is in that mapping. So **any** commit kills the run — including
a documentation-only commit touching no source at all — and it kills it **at
the finish line**, after all the compute is spent.

**MEASURED, 2026-08-29:** seven commits were made during a 51-minute
`tc_hafs_kf3` run. Recovered only by branching the work aside and
`git reset --soft` back to the commit HEAD held at launch; the run then
completed `SUCCESS`. The five hashed source files were never touched — only
the commit hash moved.

**Checking once per session is not enough. A forecast can start at any time,
and one did, fifteen minutes after a clean check.** The check is a gate:

```
bash tools/ntiedtke_wrf461_oracle/check_no_forecast.sh && git commit ...
```

It refuses, with the offending PID and command line, if any forecast entry
point is running. This is the class-1 rule applied to the rule itself: the
instruction was in CLAUDE.md the whole time and nothing re-checked it.

Commit only when asked. Do not push, open PRs, or post issues.

## 6. Deliverable shape

The trio: a `.patch`, a `.md` explaining **why** including what was tried and
rejected, and a `<topic>-commit-history.txt`. Commits on `the owner` stay topical
and one-subject-each so a range can be sliced out later. Write the `.md` as
you go — git can always reconstruct the diff, never the reasoning.

Negative results are results. A killed plan is output. The interposition probe
is the model: both preconditions tested, the one expected to fail came back
clean, and the finding was separated from "didn't work" by showing the shim
loaded, produced the pinned answer, and was never on the path.
