# Ensemble observation scores

This tool scores matched observation and forecast values. It does not create
ensembles, collocate fields, tune amplitudes, or qualify a recipe for defaults.

Build on a CPU host and score one case, recipe, quantity, unit and lead window:

```sh
rustc --edition 2021 -O tools/ensemble_calibration_score.rs -o ensemble_calibration_score
python -m tools.ensemble_calibration_score --binary ./ensemble_calibration_score \
  --matched matched.tsv --provenance provenance.json --thresholds 1,5 \
  --out scores.json
```

The TSV header starts with `sample_id`, `weight`, `observed`, followed by distinct
member identities in their declared order. Fields are tab separated. Each sample
identity identifies an observation location and valid time. Use a positive weight,
SI or explicitly declared product units, and `NaN` for missing values. Complete
member rows and observations must have the same location, time and precipitation
window. Decode, diagnostics, transformations and interpolation belong to the
native data path before this seam. Reuse one observation mask across recipes.

`provenance.json` requires `case_id`, `recipe`, `quantity`, `units`, `member_ids`,
`source_receipts`, `observation_receipts`, `match_receipt`, `valid_start`,
`valid_end` and `spinup_seconds`. Receipts identify the actual sources and native
collocation. The tool records hashes of this document, the TSV, its Rust source,
and the executable. It does not validate the source and observation receipts or
prove that their identifiers describe the TSV values.

The primary CRPS is the empirical ensemble score:

`mean(abs(member - observation)) - sum(i<j, abs(member_i-member_j)) / N^2`.

The separately reported fair CRPS replaces `N^2` with `N*(N-1)`. It is undefined
for one member. Its exchangeability assumption does not hold automatically for
members from different models or cycles. Brier events use `value >= threshold`.
Reliability bins correspond to the exact probabilities `0/N` through `N/N`.

Spread is the square root of the weighted mean unbiased member variance. Skill
is RMSE of the ensemble mean. Both the raw ratio and the ratio multiplied by
`sqrt((N+1)/N)` are recorded. The corrected ratio has expectation near one for
exchangeable, unbiased members and a perfect observation process. Observation
error, unequal member quality and dependent members require separate analysis.
A zero RMSE gives an undefined ratio, represented by JSON null.

Ranks range from zero through N. Tied observations distribute their weight
uniformly across every admissible rank. Missing observations or any missing
member exclude the entire row and increment separate counters. No missing datum
becomes zero. Repeated sample identities are rejected to prevent double counting.

Score cases separately, keep tuning and held-out cases separate, and report case
uncertainty. A pooled histogram with many nearby grid cells is not evidence of
independent verification. These scores do not establish physical balance,
conservation, initialization correctness or batch/single identity.
