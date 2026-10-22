# Ensemble tuning numerical core

This is opt-in tooling. It does not load the engine, mutate physics defaults, or
change a forecast unless a caller explicitly applies a generated setting.
Python may launch this binary and transfer JSON. Sampling, station CSV import,
score arithmetic, spread, feature reduction, and EKI arithmetic run in Rust.

Build with `cargo build --release`. Run focused checks with `cargo test` and
`cargo fmt --check`. `verify.sh` performs these checks under a CPU quota and
writes logs and a completion marker in the supplied scratch folder. Builds
and tests run on a CPU node. No GPU is required by this crate.

The CLI reads one JSON object from stdin and writes one JSON result to stdout:

```sh
tuning-harness-core sample < sample-request.json > sample-result.json
tuning-harness-core import < import-request.json > import-result.json
tuning-harness-core score < score-request.json > score-result.json
tuning-harness-core eki < eki-request.json > eki-result.json
```

An invalid request writes `{"error":"concrete reason"}` to stderr and exits 2.
Unknown request fields are rejected. Descriptive parameter-table metadata is
accepted but has no effect on numerical sampling.

## Sampling

```json
{
  "count": 24,
  "seed": 42,
  "id_prefix": "candidate",
  "parameters": [
    {"name":"surface_scale","kind":"continuous","min":0.6,"max":1.4,"default":1.0,"schemes":[]},
    {"name":"mixing_option","kind":"categorical","values":[2,3],"default":2},
    {"name":"land_model","kind":"categorical","values":["model_a","model_b","model_c"],"default":"model_a"}
  ]
}
```

Result fields are `members`, `seed`, `method`, and `applicability`. Each member
has an `id` and a `parameters` object. Continuous dimensions use independently
shuffled Latin hypercube strata and a SplitMix64 seed. Integer dimensions map
strata to an inclusive integer range. Categorical counts differ by at most one.
`schemes` is applicability metadata, returned separately. The recipe layer
must omit parameters that are inactive for a member's scheme. This keeps
categorical scheme interpretation outside the generic numerical sampler.

## Station CSV import

```json
{
  "files": [
    {"path":"output/d01/hourly.csv","day":"2024-05-01","domain":"d01","method":"bilinear"},
    {"path":"output/d02/hourly.csv","day":"2024-05-01","domain":"d02","method":"bilinear"}
  ],
  "expected_members": ["control","candidate-001","candidate-002"],
  "timezone_offset_hours": -7
}
```

Required columns are `station`, `valid_time`, `arm`, `method`, `observed_k`,
`forecast_k`, and `status`. Extra columns are ignored. Only `status=paired`
and the requested method are used. Rows for arms outside the expected list
are excluded and counted. Paired numerical values must be finite.

By default, file day D uses D 00:00Z exclusive through D+1 00:00Z inclusive,
which selects hourly forecast times f01 through f24. Per-file
`start_valid_exclusive` and `end_valid_inclusive` can supply another explicit
UTC window. Timestamps accept whole UTC seconds with `T` or a space, and
an absent suffix, `Z`, or `+00:00`. Output normalizes them to `...SSZ`.

For every station/time, the importer selects the highest numeric dNN domain
that has paired rows for every expected member. It never mixes domains
between members or counts the same station/time twice. Incomplete domain
support is counted. Observation disagreement between members or domains and
duplicate member/station/time/domain rows are errors.

Canonical output `rows` contain member, day, station, valid, period, forecast,
and observation in K. Day means local hour 06 inclusive to 18 exclusive.
`files` has support counts and exact windows, `domain_support` records the
selected domain per key, and `selected_domains` counts selected keys.

## Scoring and held-out isolation

```json
{
  "rows": [
    {"member":"control","day":"train-1","station":"station-1","valid":"time-1","period":"day","forecast":281,"observation":280}
  ],
  "config": {
    "train_days": ["train-1","train-2"],
    "heldout_days": ["held-1","held-2"],
    "baseline_member": "control",
    "expected_members": ["control","candidate-001","candidate-002"],
    "ensemble_members": ["candidate-001","candidate-002"],
    "minimum_pairs": 20,
    "frozen_winner": "candidate-001"
  }
}
```

The abbreviated rows above show the schema; a real request must include all
members on all required days and cover both day and night. Scores use the
intersection of station/time keys across all expected members for each day.
`minimum_pairs` applies to each day/night period as well as the whole day.
Counts dropped by that common mask are returned. Duplicate rows, nonfinite
values, observation/period mismatch, missing members or days, and overlapping
training/held-out days are errors. Rows outside the declared days or member
list are errors, preventing accidental use of a different scoring window.

`ranking` sorts solely by arithmetic mean of the daily training RMSE. This
gives each day equal weight even if station coverage differs. A precise tie
prefers the baseline, then member ID. Each row contains `training_score`,
`heldout_score`, `training`, and `heldout`. `metrics` contains RMSE, bias and
count for each member/day and all/day/night periods. The units match input.

Call `score` first with `heldout_days=[]` and training rows only. Save its
winner in the receipt. Call again with full rows, held-out days, and that ID
as `frozen_winner`. The core verifies that the frozen ID equals the training
winner. It never reselects using held-out scores. An eligible setting must
strictly improve on the baseline on every held-out day and in both aggregate
comparisons, with no day or night period regression on any held-out day. It
remains a review candidate; the binary does not ship it.

Training-only output has null held-out scores, comparison, and spread, and a
false `heldout_passed`. Full output includes the held-out per-day comparison.

`train_spread`, `heldout_spread`, and `daily_spread` use sample forecast variance
across `ensemble_members`. Spread is the square root of the mean variance
over the same common keys; skill is ensemble mean RMSE. Their ratio is null
when skill is exactly zero. Controls can be excluded from spread while still
participating in the common score mask.

`train_features` is a station/day/day-or-night arithmetic mean feature set
using training keys only. It returns labels with counts, member order,
member-major predictions, observations, and units. A caller may select a
fixed categorical scheme subset for EKI. Held-out rows never enter features.

## EKI

```json
{
  "parameters": [[-1],[1]],
  "predictions": [[-1],[1]],
  "observations": [1],
  "variances": [2],
  "bounds": [[-10,10]],
  "damping": 1
}
```

All matrices are member-major. Parameter columns contain continuous settings
only, from members with the same fixed categorical scheme. Variances are
positive feature uncertainty variances in input units squared. Bounds must
contain the initial settings; damping is between zero and one.

The deterministic update uses sample cross covariance and
`u_i + damping C_uy (C_yy + R)^-1 (observation - prediction_i)`.
Woodbury reduces its Cholesky solve to the number of members. The update is
clipped to supplied bounds and returns `clipped_values`, the solve dimension,
and preupdate whitened residual RMS. It requires new forecasts to measure
whether the proposed settings help. It does not predict postupdate skill.

The focused tests cover seeded strata and category balance, common masks,
missing support, duplicates and observation mismatch, equal-day weighting,
held-out leakage and frozen selection, sample spread, finest-common-domain
CSV selection, UTC windows, and analytic EKI solutions.

## Propose the next search window

```sh
python -B -m gpuwm.tuning_update --packet runs --selection scores/selection.json \
  --numerics tuning-harness-core --out ranked-proposal.json --method ranked --keep 6
python -B -m gpuwm.tuning_update --packet runs --selection scores/selection.json \
  --numerics tuning-harness-core --out eki-proposal.json --method eki \
  --variance-k2 1 --damping 0.5
```

The orchestration command verifies the sealed packet and training-selection
hash binding. It rejects held-out scores or features. Ranked proposals retain
the first K candidate settings unchanged. EKI keeps categorical groups fixed
and calls this Rust core for their active continuous parameters. Singleton
groups are retained without an update. No Python array arithmetic is used.

A fixed feature variance is an explicit assumption, with spatial and temporal
correlations unmodeled. Proposals require fresh forecasts and new unseen
held-out days. Neither mode changes a product default or deploys a setting.
