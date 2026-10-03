# WOOF against WRF: an ensemble consistency test

**This is code verification, not forecast validation.** It asks one question: is WOOF, a GPU port of WRF-ARW, statistically distinguishable from WRF itself? The yardstick is how much WRF differs from itself when its starting state is perturbed at the level of floating-point rounding. Accuracy against observations is a separate question, covered at the end.

This is the retained 2026-10-02 evidence summary. The six-hour test used
engine source `e459f79ed4c5ae74e8a23d6e1c6f66c1519dc1e1`; the corrected
24-hour envelope used `1b15157c0f9803f75c2ecd9fb3a100c706bcf453`.
Both reference WRF v4.7.1 source
`f52c197ed39d12e087d02c50f412d90d418f6186`. These results are not fresh
measurements of every later engine or assembled WOOF release.

The method follows NCAR's ensemble consistency test for CESM ([ensemble consistency test, 2015](https://doi.org/10.5194/gmd-8-2829-2015); [ultra-fast test, 2018](https://doi.org/10.5194/gmd-11-697-2018)), adapted to a regional model.

## Method

- **Reference.** For each of three weather regimes: 100 six-hour WRF v4.7.1 forecasts, built with strict floating-point flags (GNU). Each starts from initial temperatures perturbed by one float32 rounding unit.
- **Statistic.** Hour-six, area-weighted domain means of 11 variables (T, U, V, W, QVAPOR, PSFC, T2, Q2, U10, V10, RAINNC), standardized against the reference and projected onto its 11 principal components.
- **Rule, fixed before any WOOF result was scored.** The rule and cutoff were hash-sealed first. A regime fails when 3 or more components fall more than 2.00 reference standard deviations out, in at least 2 of 3 WOOF runs.
- **Calibration.** WRF runs held out of the reference and scored as if they were the candidate failed at least one regime in 33 of 5,000 trials (0.66%), against a 5% target.

## Result

| Regime | WOOF: components outside (verdict) | WRF built with Intel ifx `-O3 -fp-model fast` |
|---|---|---|
| Convective, 25 May 2024 18 UTC | 1 of 11 (**pass**) | 3 of 11 |
| Coastal, 15 Jul 2024 12 UTC | 9 of 11 (**fail**) | 10 of 11 |
| Winter, 9 Jan 2024 12 UTC | 1 of 11 (**pass**) | 9 of 11 |

The Intel column is a control: the same WRF source compiled with a common optimizing production build, one run per regime. A single run cannot get the two-of-three verdict, so it shows sensitivity, not a formal result. It shows how far an ordinary compiler change moves WRF on this same test.

**Overall result: fail.** The convective and winter regimes pass the registered rule; coastal fails. Passing this finite test does not prove equality of the models.

**Coastal.** The investigation found a real defect, but its contribution to the coastal failure has not yet been measured. The extra layers the longwave radiation scheme adds above the model top were spaced differently from WRF's. The fix reproduces WRF's captured profiles exactly in unit tests. The coastal case is being rerun against the same sealed rule, and the result will be added here either way.

## A second, descriptive test

There is a second reference: 20 strict-flag WRF v4.7.1 members across five cases, run for 24 hours. A field-hour counts as inside when its RMSE from the WRF mean is no larger than the largest WRF member's.

WOOF is inside for **1,098 of 1,440** field-hours (76%), and the Intel build for **941 of 1,440** (65%). This count has no calibrated significance level. The declared whole-case consistency result is 0/5; the count does not establish statistical indistinguishability. See `fig-envelope.png`.

## What this shows, and what it does not

- **It shows:** the registered domain-mean test did not reject the tested engine in two of three regimes and rejected it in the coastal regime. The compiler control shows that the test is sensitive to build changes. The descriptive field-hour counts answer a separate question and are not a statistical pass.
- **It does not show forecast accuracy against observations.** That is validation in the engineering sense. Meteorology calls it "forecast verification". To the extent WOOF is statistically indistinguishable from WRF for a configuration, it can inherit WRF's published validation record for that configuration. A broader station and radar validation programme is being built separately; existing individual case scores do not supply it.
- **Limits:** one event per regime, six-hour forecasts, and domain means, which can hide spatial error.

Charts: [light ECT](fig-light-ect.png) and [20-member envelope](fig-envelope.png).

The retained replay package contains fields, initial and boundary inputs with
hashes, PCA fits, scores, calibration trials, scripts, and the sealed method
and thresholds. That full package has no public download in this repository;
the summary and charts here do not by themselves permit an independent rerun.
