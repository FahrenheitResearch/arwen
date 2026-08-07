# Does a convection-allowing background improve WaH?

EXPERIMENTAL. This is the controlled comparison that decides whether
`--source hrrr` (see `docs/da-background-source.md`) is worth offering as
anything more than a capability. It is designed, validated and staged; it
has not been run.

## The claim under test

> Starting the radar-DA nowcast from a 3 km convection-allowing first
> guess produces a better 90-minute forecast than starting it from a
> 25 km global one.

That sentence hides three separate changes, and a two-arm experiment
cannot tell them apart:

| what changes when GFS becomes HRRR | direction |
|---|---|
| grid spacing of the first guess, 25 km to 3 km | the claim |
| condensate at hour zero: explicitly zero to QC/QI/QR/QS/QG | the claim's mechanism, or a separate one |
| how old the first guess is, and whether NOAA already assimilated radar into it | a confound, if the cycle is chosen for freshness |

So the experiment has three arms, and the confound is held fixed in the
one that carries the headline.

## The arms

The case is the one that has been run all night: KDMX, 2026-08-05,
init 04:00Z, six 15-minute LETKF cycles on real Level-II radial velocity
from 04:15Z to 05:30Z, then a 90-minute free forecast scored at six valid
times from 05:45Z to 07:00Z.

| arm | background | cycle | leads | age at init | role |
|---|---|---|---|---|---|
| `G-gfs` | GFS 0.25° | 2026-08-05 00Z | f004..f010 | 4 h | the published configuration, on the published case |
| `H-matched` | HRRR 3 km | 2026-08-05 00Z | f004..f008 | 4 h | **the headline** |
| `H-fresh` | HRRR 3 km | 2026-08-05 04Z | f000..f004 | 0 h | what the hourly cadence actually offers |

`H-matched` is the attributable arm. Same init time, same forecast age,
same hourly forcing interval, same grid, same eta ladder, same physics
profile, same observations, same seed, same perturbation, same
localization, same relaxation, same thinning, same leg schedule. What is
left is the model that produced the first guess.

`H-fresh` is the operationally honest arm and is deliberately
confounded: it is four hours fresher *and* it is an HRRR analysis, into
which NOAA has already assimilated radar reflectivity. Any advantage it
shows over `H-matched` is partly someone else's data assimilation, and
it is read against `H-matched`, never against `G-gfs`.

`G-gfs` is re-run rather than quoted. It has to reproduce
FSS30(27 km) `0.7274 / 0.7557 / 0.7655 / 0.7376 / 0.7316 / 0.7239`
against control `0.2360 → 0.3410`. If it does not, something moved
underneath the comparison and no HRRR number from the same queue means
anything.

## How "everything else identical" is proved rather than intended

Three mechanisms, in the order they fire.

**Before anything is fetched.**
`tools/da_background_ab/build_case_inputs.py` does not accept a
hand-written HRRR domain specification or namelist. It *derives* both
from the GFS case's own `experiment.toml` and `namelist.wps`, then
renders the experiment tables the native HRRR route would build from
them, pushes those through the ordinary `build_experiment` front door,
and compares the resulting prepared-domain identity against the one
already baked into the GFS prepared cache — with
`gpuwm.ingest.prepared_cache.compare_prepared_domain_config`, the same
comparator the forecast front door uses. It separately compares the eta
ladder and `p_top`, because those are *not* in the domain identity and
two cases can therefore pass the domain comparison while sitting on
different vertical grids. Either mismatch is a refusal.

Measured on this case: domain identity EQUAL, 50 levels, max |Δη| = 0,
|Δp_top| = 0 Pa.

**Before the queue starts.**
`tools/da_background_ab/validate_plan.py` compares the three cycling
arms flag by flag — observations, georeference frames, member count,
seed, perturbation amplitude and length scale, localization radii,
relaxation, thinning, error inflation, physics profile, history interval
and leg schedule — and refuses on any difference. It also asks the
background registry the questions a parser cannot: that both HRRR cycles
are published and reach their requested leads, that HRRR's grid covers
this domain *with its halo*, and that the ensemble
`plan_member_backgrounds` would build is eleven distinct trajectories
rather than a fabricated one.

**After the cases exist.**
`tools/da_background_ab/check_case_parity.py` compares the arrays. Static
geography — terrain, land use, map factors, Coriolis — must be identical,
because that is what "the same lower boundary" means, and a difference is
a refusal. The initial layer-interface heights are *not* expected to be
identical, and their difference is measured as a fraction of the local
layer depth (see the caveat below).

## The observations

One set of files. Gridded once, onto the GFS arm's georeference
trajectory, and passed to every arm byte-identically — so the arms are
compared observation for observation rather than against two superobbings
of the same volumes.

The price is stated rather than hidden: a gridded radar observation is
bound to the 3-D georeference it was placed on, so a gate sits in the
model layer the GFS column put it in. The horizontal grid and the terrain
are identical across arms; what differs is the hydrostatic column, and
`check_case_parity.py` reports that offset in metres and as a fraction of
a layer. Rebuilding the observations per arm would remove that offset and
replace it with a worse problem: two different observation sets, which is
exactly the confound this A/B exists to avoid.

## The metric

`tools/da_sweep_score.py`'s metric, which reproduces the published
baseline bit for bit — `gpuwm.verify.field_metrics.fss_distance`, 30 dBZ,
missing observations filled at −35 dBZ.

Three facts about it are honoured explicitly, because each has been
mis-stated before:

* the neighborhood is a **square side length**, not a radius. A rung of
  half-width `h` scores a box `2h+1` cells across; the published rung is
  `h = 4`, nine cells, **27 km across** at 3 km spacing;
* the FSS smooths **both** fields — the truth as well as the forecast;
* the published figure scores the **ensemble mean**. That is what the
  headline reports, and a per-member statistic (mean, min, max and the
  full list at the published rung) is reported beside it, never in place
  of it.

The comparison is a **curve over neighborhood size**, not a single
scale — half-widths 0, 1, 2, 3, 4, 6, 8, i.e. 3, 9, 15, 21, 27, 39 and
51 km across, averaged over the six verification times. The method is the
neighborhood ladder a sibling lane built for the 3 km / 1.5 km resolution
comparison (`tools/ens_sweep/score_resolution.py` on
`lane/ens-size-sweep`); it is reused here rather than reinvented. Its
family A/B/C split does not arise: every arm of this A/B integrates the
same grid, so there is one common grid, no reduction operator, and
nothing for a reduction-operator sensitivity to be sensitive to.

## What is scored beyond the headline

**Both controls.** Every arm carries a never-analysed control
trajectory, and it is scored on the same ladder. This is the section that
decides the interesting question. On this case the GFS control reaches
519 → 1,162 storm columns against 2,817 → 3,991 observed: it starts with
no storms and never builds many. An HRRR control starts with storms
already in it. A large part of any HRRR advantage may therefore be the
background's own forecast rather than the assimilation, and separating
those is the finding, not a nuisance.

**Spread across the cycles.** Prior and posterior spread per analysis,
plus the observation-space consistency ratio
`innovation_rms² / (spread² + obs_error²)`. The GFS arm on this case is
already 7.8× to 24.5× under-dispersive by that measure, rising through
the six analyses. The spread bar is not carried over from it unexamined —
a better-centred background produces smaller innovations, which
mechanically reduces the spread the filter needs, so the ratio is
reported per arm per cycle rather than compared as a single number.

**Analysis increment magnitude.** `mean_increment_rms` per field per
cycle. If the first guess is genuinely better the filter has less work to
do and the increments shrink. That is the mechanistic signature and it is
checked directly.

## What would falsify the claim

The verdict is computed, not narrated
(`tools/da_background_ab/score_background_ab.py`), and it can come out
NO in four distinct ways:

1. **No rung improves.** `FSS(H-matched) − FSS(G-gfs) ≤ 0` at every
   neighborhood from 3 km to 51 km, on the mean over the six
   verification times. The claim is simply false on this case.
2. **The control explains all of it.** The never-analysed control gains
   at least as much as the analysed ensemble at *every* rung. Then the
   background's own forecast improved and WaH's skill did not — a
   falsification of the claim as a statement about the nowcast system,
   however good the headline number looks.
3. **Under-dispersion bought it.** Skill improves while the HRRR arm's
   consistency ratio at the last analysis exceeds the GFS arm's. A
   collapsing ensemble has a smoother mean and FSS rewards smooth. This
   is reported as under-dispersion, not as skill.
4. **The mechanism is wrong.** Skill improves and the analysis
   increments do not shrink. Then the first guess was not closer to the
   observations, and whatever produced the gain — most plausibly
   condensate the background carried and the DA never had to create —
   is not the claimed mechanism and must be named.

Only the fifth path is support: skill improves beyond the control's own
gain, the ensemble does not become less consistent, and the increments
shrink.

## Cost

| | |
|---|---|
| GPU | two HRRR preparations (**unmeasured** — no HRRR case has been prepared at this domain size) plus three cycled runs at a measured 506 s each |
| Disk | ~5.3 GiB under the run directory; 4.09 GiB of it HRRR GRIB2, measured object by object from the published byte-range indexes |
| Network | 4.09 GiB from `noaa-hrrr-bdp-pds` |

The subset route moves **435 MB per forecast hour**, not the ~1.1 GB the
whole `wrfnat` object would cost; the field selection is roughly 58% of
the atmosphere object and 7% of the pressure object.

No new georeference forecast is run (that would have been 2.7 GB), the
cycling driver writes composites and increments rather than history
(23 MB per arm), and the HRRR cases are bound to 14400 s rather than the
GFS case's 21600 s — the legs integrate 10800 s either way, so this
removes two forecast hours per case and changes nothing about the
physics.

## Known gaps this A/B does not close

The perturbation amplitudes are **not** retuned per arm. They were tuned
against a storm-free GFS first guess, and on a background that already
contains balanced convection an unbalanced 50 km wind perturbation is a
cruder instrument — it can radiate gravity waves off real storms and
displace structure HRRR got right. Retuning per arm would make the arms
differ in two things instead of one, so the amplitudes are held and the
risk to the headline is stated instead.

Every arm is a single draw. One case, one radar, radial velocity only,
one microphysics scheme, one 90-minute window. No arm is repeated, so no
number here carries an error bar, and the standing no-ECC dual-run screen
is not applied to any of them.

## Methodology: a frozen baseline stops being a baseline when the lane moves

Recorded 2026-08-06, from the rung-0 HRRR screen, because it is a flaw in
the LADDER rather than in any one case and the next lane will meet it.

The ladder screens a variant against a frozen baseline and judges the
per-case delta. That is exactly right when the variant changes only its
treatment. It silently becomes a multi-lever comparison the moment the
lane itself has moved: rung-0 changed resolution (3 km to 1.5 km),
background (GFS to HRRR) and ensemble size (N=10 to N=8) before any
treatment was applied, so "variant minus frozen baseline" was measuring
four things at once and reporting one number.

Two consequences, both load-bearing:

* **The control test tells you which comparison you are in.** The
  unassimilated control shares the prepared case, the background and the
  model with its treatment arm and differs only in receiving no
  analysis, so a matched pair has a BIT-IDENTICAL control per case
  (verified 3/3 on the reflectivity arm). Against a frozen baseline
  built at another configuration the control cannot match, and that is
  not a defect to fix -- it is the signature that says "this comparison
  cannot attribute a treatment". Report it as "not applicable by
  construction" rather than quietly averaging it away.
* **A treatment certification needs a matched baseline at the LANE's
  configuration**, built at the same case count as the variant. The
  frozen baseline still answers a real and separate question -- does the
  whole stack beat the shipped default -- and that question is the one
  that justifies changing a default. Neither number can be read as the
  other, so a verdict that carries both must label both.

The arithmetic that keeps a marginal number from being misread: measure
the lane's own handicap first. Rung-0's configuration scored -0.0693
against the frozen baseline BEFORE any treatment, and the reflectivity
treatment scored +0.1625 against its matched arm, so a near-zero result
against the frozen baseline means the treatment roughly cancelled the
handicap -- not that the treatment failed. State that prediction before
the run finishes; stated afterwards it is a rationalization.
