# Certification

A run that finished is not a run that was checked. This document describes the
two commands that do the checking, the data they check against, and — as
importantly — what they refuse to do.

Both commands fail closed. Neither has a `--force`, a `--warn-only`, or a
tolerance flag. Where a check cannot be made, the answer is a refusal naming
the condition, not a pass with a caveat.

## `gpuwm certify`

```bash
gpuwm certify \
  --run-capsule out/<run>/certification-capsule.json \
  --metrics-csv out/<run>/metrics.csv \
  --band gpuwm/data/certification/bands/<config-sha256>.json \
  --wrf-reference-manifest docs/public/wrf-reference/<config-sha256>.manifest.json \
  --out-verdict out/<run>/certification-verdict.json
```

It returns 0 only when every condition below holds, and nonzero naming each
that does not. All of them are evaluated on every invocation, so a reader gets
the whole list rather than the first refusal and a rerun.

| Condition | Refuses when |
|---|---|
| `capsule_validates` | the run capsule does not validate against `gpuwm.certification-capsule/v1` |
| `band_config_identity_matches_capsule` | the band's `config_sha256` is not the configuration the capsule records |
| `wrf_reference_hashes_present` | the WRF reference manifest is missing any of its four hash groups |
| `geography_input_hashed_by_content` | a declared directory input was bound by its listing (`sha256-directory-inventory`) rather than by its bytes |
| `every_pin_resolved` | a published environment pin carries a status other than `resolved` |
| `every_metrics_column_classified` | the metrics CSV carries a column the band's `metric_coverage` does not classify |
| `every_banded_row_inside_its_interval` | a gated comparison row falls strictly outside its own interval |

The last two are the ones that catch drift rather than mistakes. A comparator
column added later is not silently ignored: it is an unclassified column, and
certification stops until somebody decides whether it is gated or merely
recorded.

## The acceptance band

The band is data, not code. One file per configuration, addressed by that
configuration's SHA-256, under `gpuwm/data/certification/bands/`. Nothing in
it names a case; two configurations that differ by one byte get two files, and
a band cannot drift onto a configuration it was not derived for.

Each band declares:

- a **provenance**, either `documented-margin` or `wrf-ensemble-envelope`;
- a **band schema version**, which the verdict records alongside the
  provenance it certified against;
- a **`metric_coverage` map** with exactly one entry per comparator metric
  column, each classified `banded` or `recorded-only`;
- an **interval** per (domain, lead, gated metric), or an explicit
  `nan_expected` marker where the comparison has no defined value.

### Today's provenance: `documented-margin`

The interim band's intervals are the deterministic output of a committed rule,
`published-anchor-margin`, applied to committed inputs. The rule
(`gpuwm/data/certification/margin_rule.json`) takes the published comparison
table as its anchor and opens a documented margin around each cell: an
absolute floor per metric family, or a relative fraction of the anchor,
whichever is larger, clamped to that family's physical range.

**The rule is the artifact; the numbers are its output.** Re-running the
derivation on an unchanged tree reproduces the committed band byte for byte,
and a test asserts exactly that — so a band edited by hand is caught by the
tool that made it. The band also records the SHA-256 of the published table it
anchored on, so editing that table makes the band visibly stale rather than
quietly wrong.

Beside each band is a **coverage receipt** recording, for every
(domain, lead, metric) triple of the published table, whether the published
value lies inside its own interval, outside it, or is a non-number against a
cell that expects one. The receipt is an observation. No test asserts a
distribution over it, and none should: a band widened until it swallowed the
whole table would satisfy such an assertion and prove nothing.

### Tomorrow's provenance: `wrf-ensemble-envelope`

When the WRF control ensemble lands, the same file and the same command carry
intervals derived from the ensemble spread instead of from a documented
margin. Such a band additionally carries an `ensemble` block — member count,
per-member configuration digests, the pair-score artifact digest, and the
interval statistic — labelled `scope: internal`. Those artifacts are
internal-scope: they are not part of any release, and this repository offers no
route to them. What is public is the interval they produced and the statistic
that produced it.

## `gpuwm dual-run`

```bash
gpuwm dual-run --capsule-a out/run-a/certification-capsule.json \
               --capsule-b out/run-b/certification-capsule.json
```

Two capsules from what should be the same run, compared leaf for leaf. Exit 0
when they are identical; nonzero naming the first field they disagree on, in a
deterministic path order that is a property of the two documents rather than of
dictionary iteration.

The comparison is total. There is no ignore list, because every field a
comparison skips is a field a silent bit flip can hide in. A field that is
absent and a field that is present carrying `null` are different claims and
compare unequal; a section emptied of its contents does not vanish from the
comparison.

This is also the detector for hardware that cannot report its own errors: on a
card without ECC, running the same configuration twice and comparing the
receipts is the screen.

## The verdict, and why flipping it does nothing

The verdict document carries `capsule_binding_sha256` — a SHA-256 over the
closed inventory of everything the decision rests on: the capsule digest, the
band and its derivation, the WRF reference binding, the metrics digest and
column list, every evaluated condition, and every bound comparison row.

The `passed` boolean is deliberately **not** in that inventory. Flipping it,
or deleting it, moves neither the binding digest nor the verdict re-derived
from the bound rows. Editing any bound row moves both. A consumer therefore
never has to trust the claimed bit: it recomputes the digest, re-derives the
decision from the rows, and compares.

Nothing under `gpuwm/certify/` imports any case module, and a test holds the
import graph to that. A certification path that ran through a case module would
be a certification path that could not certify a second case.

## WRF reference manifests

See [wrf-reference/README.md](wrf-reference/README.md). A manifest pins the
executable, the build recipe, the namelists, and the reference output bytes the
matched comparison was made against. Certification refuses while any of those
is absent — not because a missing hash is likely to be wrong, but because a
verdict that did not know what it was compared against is not a verdict.

## What certification does not claim

- It does not claim the model is correct. It claims a specific run, on a
  specific configuration, on a stated stack, produced a comparison that falls
  inside a stated interval derived by a stated rule.
- It does not claim the interval is the right one. Under
  `documented-margin` the interval is a documented margin around a published
  comparison, and it says so in its own provenance field.
- It does not claim reproducibility across hardware. The environment pins
  exist precisely because byte identity is a claim about a fixed stack; see
  [DETERMINISM.md](DETERMINISM.md).
