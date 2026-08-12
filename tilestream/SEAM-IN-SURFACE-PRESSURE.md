# The surface-pressure seam: mechanism found, and it is NOT the gate

**Status: RESOLVED as to mechanism.** An earlier revision of this file said the
cause was unknown and that the bit-exactness gate had a blind spot covering it.
Both statements were wrong and are corrected below.

## What was observed

154.1 Mcell (2048x1536x49, dx 500 m), 4x2 across eight RTX 4090s, full physics,
under active convection.

* Composite reflectivity: **clean**, seam/background 0.63-0.93.
* Surface pressure: seam/background grows monotonically, **2.5 -> 12 -> 22 -> 38**
  over t+3 to t+12 min.

## The mechanism, in the source

`max8.py:404` builds the exchange set as

    sorted(n for n, v in inv.items() if v[3] >= 3)

where `inv[name] = (..., a.ndim)`. That is **three-dimensional carriers only.**

`state/mup` is **2-D `(ny, nx)`** (`dycore.py:1587`, `1885`). It is the
perturbation column dry mass — i.e. **the prognostic surface pressure** — and the
initialiser seeds it with amplitude-20 noise, so it carries horizontal structure
from t=0.

**It is the only one.** An earlier revision said `mu_pp` and `rmu_t` were
"excluded by the same predicate". They are not carriers at all: `state_manifest`
admits only attributes classified `serialize`, and the acoustic/tendency arrays
classify `rebuild`. **`state/mup` is the sole 2-D carrier in the `state/`
namespace.** State this narrowly — the broader phrasing invites doubt about
everything else in the decomposition, and only the narrow claim is true.

So the ranks **never exchanged surface pressure.** Stale halo column mass
explains the monotonic growth exactly, and explains why reflectivity stayed clean
— reflectivity is diagnosed from the 3-D moisture carriers, which *are*
exchanged.

## The gate is correct. A different script was not.

`gate_phys.py:282` selects

    len(inv[n]) >= 3 or n.startswith("state/")

— a strictly **larger** set that pulls `state/mup` back in. So `BITEXACT=PASS` is
a true statement about `gate_phys.py` and says **nothing** about `max8.py`. The
imagery runner and the gated runner used different exchange sets; only the
imagery runner was wrong.

The fix is one clause on `max8.py:404`. But **the 154 Mcell imagery already
rendered has a real seam in it and must not ship as physics-clean.**

## The gate's actual gap is DURATION, not carrier coverage

The digest runs **3 steps**. A seam that takes 60-240 steps to become visible
cannot be caught in a 3-step window, whatever the exchange set or the initial
condition. That duration gap — not the carrier list, and not the uniform IC — is
why this class of bug survives an otherwise strong gate with three negative
controls firing 40/40.

**Any future gate needs a long-window seam check on a smooth field** (PSFC, not
reflectivity: a seam hides in a noisy field and cannot hide in a smooth one),
integrated far enough that convection develops real surface structure.

**The actionable form, and the most useful single output of this whole episode:**
widening `gate_phys.py`'s carrier set fixes nothing by itself. What fixes it is a
**horizontally perturbed initial state** plus **enough steps for a stale halo to
propagate far enough inward to move the checksum** — more than the three the
digest currently runs. That is a testable change to the gate, not an observation
about it.

## The confound, now demoted

The same run used globally-indexed **white** noise, putting all variance at
2*dx*. That produced domain-wide speckle and drove psfc to ~1250 hPa, killing an
earlier run at t+12 min inside RRTMGP (`play range [6682.8, 124958.2] Pa outside
allowed [1.005, 109663.3] Pa`). Real, and worth fixing with coarse
periodic-smoothed noise — but it is **not** the seam mechanism. The seam is the
missing `state/mup` exchange.

## The transferable lesson: `ndim` is the wrong discriminator

Nearly every 2-D carrier IS column-local physics and correct to skip. A few are
horizontally-coupled dynamics prognostics, and column mass is the one that
matters. **Keying the exchange set on array rank silently conflates those two
populations; keying it on namespace does not.** That is why
`n.startswith("state/")` is the right rule and `v.ndim >= 3` is not.

Scope of the fix, precisely: widening to `state/` admits exactly **one** carrier,
`state/mup`. Not `mu_pp`, not `rmu_t` — those are not carriers (`rebuild`, not
`serialize`). And it does **not** admit the integer surface carriers (`isltyp`,
`ivgtyp`, `kpbl`, `isnowxy`, `pgsxy`) — those live under `fields/` and are
genuinely column-local, so they neither need exchanging nor motivate the dtype
work. A column's surface energy balance needs no horizontal neighbour; skipping
them is correct by design, not a concession.

## A second defect stacked in the same field

`state/mup` was **also rank-tiled**: the initialiser regenerated only `thp` and
`w` with global indexing, so `mup` was eight identical copies at t=0 *and* was
never exchanged thereafter. The t+0 seam test could not see it because it watched
`fields/psfc`, a diagnostic sitting at a uniform 100000 Pa at t=0 — not `mup`.

## RESOLVED CLEAN: the hardcoded float32 exchange memmap

Previously filed here as a possible pre-existing defect. It is not one.

Audit of all 228 carriers: exactly **seven** are not float32 — `fields/ebal`,
`fields/isltyp`, `fields/isnowxy`, `fields/ivgtyp`, `fields/kpbl`,
`fields/ktop_plume`, `fields/pgsxy`, all int32. **All seven are 2-D and all seven
live under `fields/`. No 3-D carrier is anything but float32.**

So the hardcoded float32 memmap on the `ndim >= 3` branch never routed an integer
field through a float32 view. There is no garbage-halo defect in the rendered
frames; the seam is explained by the `mup` exclusion alone.

Stage per-carrier dtype anyway — as defence against a latent hazard, not as a fix
for anything observed.

## What the gate actually certified

`gate_phys.py` exercises a **superset** of the shipping configuration, on a state
where the excluded fields are constant. A gate in that position **can only pass**.

**The strongest form of this, and the one to guard: a uniform-state gate cannot
detect an exchange omission IN PRINCIPLE — whatever carrier set it covers —
because the excluded fields are constant and cannot diverge by construction.**
Carrier coverage is not the variable that fixes it; horizontal structure in the
test state is.

`BITEXACT=PASS` is a true statement about `gate_phys.py` and silent about the
runner — and "the 8-GPU decomposition is bit-exact" is exactly the sentence
someone lifts out of context. Any future gate must run the configuration that
ships, not a safer neighbour of it.

Note that `gate_phys.py`'s `state/` clause looks deliberate — it serves no other
purpose — which makes this the runner diverging from an existing correct
convention, not an oversight nobody had considered.

## Related, and a cost correction

An earlier revision of this file said the exchange set goes "from 73 to 228
arrays" and that packing therefore becomes a prerequisite. **Both wrong.**
`state/mup` is the ONLY 2-D carrier in the `state/` namespace, so the fix takes
the set from 73 to **74** — one 12.6 MB plane, no measurable exchange time.

Packing is still the top throughput lever (see [[EXCHANGE-IS-COUNT-BOUND]]:
4,736 unpinned copies per step at 9-23% of the PCIe ceiling), but it is an
optimisation, not a prerequisite for this fix.

Corroborating that the reduced set is insufficient in general: `4x1` at 9.6 Mcell
raises `ValueError: MYNN mass-flux inputs must be finite`, while `4x2` and `8x1`
pass.

---

## RESOLVED: both defects fixed, and the gate now discriminates

Everything above this line was written before the fix existed.  It is
unchanged because it was right.  What follows is what happened when it was
carried out, on one RTX 5090 with the box at **0 CUDA contexts** start and
end.  Full receipt and verbatim output: [[HARDENED-SEAM-GATE]].

**The fixes.**  `arwen-8gpu-run-evidence/max8.py`, in `bowecho-dea` on branch
`lane/integrate-mup-fix`:

* the exchange predicate is now `v[3] >= 3 or n.startswith("state/")`;
* `seed_global()` regenerates `state/mup` from GLOBAL indices, mean preserved,
  closing the second stacked defect -- the rank-tiling.

**The gate.**  `tilestream/test_seam_gate.py` crosses initial condition
(uniform / smooth-perturbed-and-sliced) with exchange set (`ndim>=3` /
namespace) at 0, 3, 24 and 240 steps, and reports a matrix rather than a
verdict.  It discriminates, at both `1x2` and `2x2`:

    ic=uniform,   exchange=ndim>=3    green  green  green  green
    ic=uniform,   exchange=namespace  green  green  green  green
    ic=perturbed, exchange=ndim>=3    green    RED    RED    RED
    ic=perturbed, exchange=namespace  green  green  green  green
    ic=perturbed, namespace + rank-tiled mup
                                        RED    RED    RED    RED

**The narrow cost claim holds, re-measured on a real 228-carrier inventory:**
73 -> 74, admitting exactly `['state/mup']`; 155 2-D carriers of which one is
in `state/`; 7 non-float32 carriers, all int32, all 2-D, all under `fields/`.

### Two corrections to the prescription above

**Duration is the smaller axis.**  This file calls the >3-step window "the
single most actionable item".  Measured, the perturbed initial condition alone
is what does the work: the gate is green at 0 steps and red at 3.  The
60-240 step figure is how long the seam took to become visible in RENDERED
IMAGERY, which is a far blunter instrument than a digest.  The two claims are
about different instruments and should not be quoted as one.

**The prescribed seam check on a smooth field does not fire at all.**  Every
seam ratio in every red cell stays within 1% of 1.0 -- `mup` 0.9964 at 3
steps and 1.0038 at 240, `p[0]` 1.0153 and 1.0053 -- while the digest fails
outright.  The reason is structural: by 24 steps the error covers all 192
columns, and seam/background is a ratio of like to like, so an error that is
*everywhere* leaves it at 1 by construction.  The rendered seam reached 38
because that domain was 2048 columns wide with local convection and the error
stayed near the boundary.

**So the seam ratio is a diagnostic of WHERE an error is, never a test of
WHETHER one exists.**  A gate built on it would have been green in all four
red cells -- the same failure mode as the original: a check that cannot fire
under the conditions it is run in.

### Found on the way in

`multigpu.plan_split` passed `periodic=` to `TileSpec`, which has taken
independent `periodic_x` / `periodic_y` since the spec refactor.  **Every
multi-GPU entry point in this tree raised `TypeError` before allocating
anything** -- `gate`, `negative_controls`, `bench`, all of it.  Fixed on the
same branch; the existing 4-mode `multigpu.gate` then passes on one card,
`sequential`/`interleaved`/`threads`/`events` all matching the monolithic
digest `ad96ac84377e3b909e81fb61d21ad5bd79523c43e38c254fc5d65216554cd94c`.
