# Registration v2.1 — amendment: frame selection under a coverage floor

Amends `REGISTRATION-v2-ratified-under-delegation.json`. Written 2026-08-04,
before the B5 freeze signs, under the owner's standing delegation of
2026-08-04 relayed by the obs-battery lead.

**This file does not edit v2.** The v2 document is kept beside it, unedited,
and its `registration_sha256`
`11f834d18a61be718458b89114cae9b6ac1c03b2f44c6d3c97d54d3765b3c78f` still
hashes its own pins. An amendment that rewrote the document it amends would
destroy the only thing a registration is for.

---

## 1. The rule

Inside the registered matching tolerance
(`reflectivity.obs_match_tolerance_seconds`, **±240 s, unchanged**), the frame
scored at a lead is:

> the **nearest** frame whose recorded observed fraction meets the coverage
> floor **0.9**; ties in `|dt|` go to the **earlier** frame. If no frame
> inside the tolerance meets the floor, the lead is recorded **missing-obs**,
> named in the score record with every candidate frame's observed fraction,
> and left out of the lead mean. The tolerance itself does not widen.

The floor is the case bar's own coverage requirement — *no more than 10% of
the interior masked* — applied to the **choice** of frame and not only to the
**verdict** on it.

Pins, in `reflectivity`:

| pin | value |
|---|---|
| `frame_coverage_floor` | `0.9` |
| `frame_selection` | the sentence above, verbatim, in the parameter block |

A document minted after this amendment carries both pins. The ratified v2
document predates them, so the registered default applies to it — that is
what makes this an amendment rather than a parameter somebody might have
forgotten to set.

## 2. Why

The campaign kit build found one case day whose MRMS feed suffered an
upstream ingest outage across one scored lead. At that lead the frame nearest
the hour decodes at observed fraction **0.1586** — a mostly-masked analysis —
while a frame **169 s later, inside the same registered ±240 s tolerance**,
reads **0.969**.

Under the unamended nearest-frame rule the scorer would have scored the
0.1586 frame, and the case bar (">10% of the interior masked fails the case")
would then have failed the **whole case** over a 13-minute upstream outage,
with a qualifying frame sitting inside the tolerance the rule had already
registered. That is not a measurement of the model. It is the instrument
reading its own supply chain.

The amendment is deliberately the narrower of the two available motions: it
does **not** widen the tolerance, and it does **not** relax the case bar.
Both of those would change what "coincident" and "observed" mean. Choosing
the qualifying frame that the registration already permits changes neither.

## 3. What is NOT changed

* the ±240 s matching tolerance;
* the case bar itself, and the interior valid fraction it reads — the score
  record still publishes `interior_valid_fraction` per lead, computed on the
  regridded field over the scored interior, exactly as before;
* the primary scalar's definition, the thresholds, the neighborhoods, the
  spin-up policy, the promotion rule, and every other pin;
* the empty-window refusal: a lead with **no** frame inside the tolerance is
  still a hole in the archive and still fails the arm. Only a window that
  holds frames and none that clear the floor produces an exclusion.

**Two different quantities, named apart.** Selection screens on the frame's
own recorded `sentinels.observed_fraction` — the share of the packed
subdomain the analysis observed, which the ingest front door writes into each
pack and which is therefore knowable before decode. The case bar reads the
`interior_valid_fraction` the scorer computes after regridding onto the model
interior. They are close on this domain but they are not the same number, and
the amendment does not treat them as one: it selects on the first and reports
the second.

## 4. Determinism

* Candidates are ordered by `(|dt|, valid_time)` — nearest first, ties to the
  earlier frame — so selection never depends on pack order on disk.
* A frame that records **no** observed fraction is never screened on. That is
  the behaviour every frame had before this amendment, and inventing a
  coverage for a product that does not publish one would be worse than not
  screening. Stage-IV packs record none, so precipitation selection is
  untouched.
* An exclusion is a row in the score record
  (`reflectivity.excluded_leads[]`: lead, valid time, reason, floor, and
  every candidate's observed fraction), never a silent drop. The record also
  publishes `lead_hours_requested` and `lead_hours_scored`, so a lead mean
  taken over a changed denominator says so.
* A pass in which **every** lead is excluded raises rather than returning a
  score: an unscoreable case is not a score of zero.

## 5. The no-op proof obligation

An amendment landing between a scored receipt and the freeze must prove it
changes nothing already measured.

**Obligation.** For evidence already scored, selection under the amendment
must be identical to selection under the unamended rule.

**Discharged, on the artifact:**

1. **The B-04 score receipt** (`out/node19-shakedown/b04/score-F.json`,
   registration `11f834d1…`, primary `0.46175066058968983`) records
   `interior_valid_fraction` **1.0 at all 17 scored leads**, and
   `minimum_interior_valid_fraction` 1.0. Every frame it scored is above the
   floor by the widest possible margin, so the floor cannot have moved a
   selection: the nearest frame was already a qualifying frame.
2. **The decoded kits**, scanned per pack: **131 reflectivity frames across
   7 case-day kits**, of which **130 record observed fraction exactly 1.0**.
   The single exception is `2023-04-01T00:00:35` at **0.1586** — the outage
   above, on case day 2023-03-31, at the lead valid `2023-04-01T00:00:00`.
3. **Selection replayed on those kits**, every case day, every scored lead,
   with and without the floor: **119 lead selections, 118 identical, zero
   moved**, and one excluded — the outage lead, whose locally decoded window
   holds only the 0.1586 frame (see section 6). No selection anywhere changed
   from one frame to a different frame.
4. **A test** (`tests/test_obs_battery_ingest.py::`
   `test_a_full_coverage_frame_set_selects_exactly_as_it_did_before`) asserts
   frame-for-frame that a fully observed frame set selects identically with
   and without the floor.

## 6. Consequence for kit builds

The rule can only *choose* a qualifying frame if the kit holds the candidates.
A kit decoded at one frame per lead can only ever **exclude** the lead: on the
locally decoded 2023-03-31 kit the ±240 s window holds exactly one frame, the
0.1586 one, so the amendment records that lead missing-obs. The campaign kit
build, which decodes the frames inside the tolerance, has the 0.969 frame at
+169 s and scores it.

**So the kit build for any case day that carries an outage must decode every
frame inside the registered tolerance, not only the frame nearest the hour.**
That is an obligation on the ingest step, and it is stated here because the
selection rule is otherwise unexercisable.

## 7. Where it is implemented

| what | where |
|---|---|
| the floor, and the pins | `gpuwm/verify/obs/registration.py` (`DEFAULT_FRAME_COVERAGE_FLOOR`, `reflectivity_parameters`) |
| the selection walk | `gpuwm/obs/sources.py` (`_GriddedSource._nearest`, `_inside_window`) |
| the seam's missing-obs signal | `gpuwm/verify/obs/contracts.py` (`ObservedFractionBelowFloor`) |
| the exclusion record | `gpuwm/verify/obs/battery.py` (`score_reflectivity`) |
| the floor reaching the reader | `tools/obs_battery_score.py` (`registered_frame_coverage_floor`) |

## 8. Effect on the registration digest

None on v2: its parameter block is untouched and still hashes to
`11f834d18a61be718458b89114cae9b6ac1c03b2f44c6d3c97d54d3765b3c78f`. A v2.1
JSON minted from the amended constructors will carry `frame_coverage_floor`
and `frame_selection` in its `reflectivity` block and will therefore have its
own, different digest. Scores already published under v2 stay bound to v2 and
stay verifiable against it.
