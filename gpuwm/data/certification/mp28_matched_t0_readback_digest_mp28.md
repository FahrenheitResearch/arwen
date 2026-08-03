# t=0 full-state parity digest

- schema: `gpuwm.t0-state-digest/v1`
- evaluator commit: `555f0498cd1f74cb4bbaee95b8b982d221e20ef3`
- registration sha256: `148b8442ec6367c8433e08df6e01a47e9469a18766b363e2332e7cd498c287fa`
- ceilings (from `gpuwm.verify.nest_gates`): max 8 ULP, p99 2 ULP, |mean signed| 0.25 ULP
- covered groups: dry_dynamics, moisture
- verdict: **PASS**

| domain | group | scored arrays | max ULP | p99 ULP | mean signed ULP | verdict |
|---|---|---:|---:|---:|---:|---|
| d01 | accumulation | 0 | - | - | - | unavailable |
| d01 | diagnostic | 0 | - | - | - | unavailable |
| d01 | dry_dynamics | 6 | 0 | 0 | +0.0000 | PASS |
| d01 | moisture | 11 | 0 | 0 | +0.0000 | PASS |
| d01 | soil | 0 | - | - | - | unavailable |
| d01 | surface | 0 | - | - | - | unavailable |

## boundary group: unavailable

Not scored: no lateral-boundary file staged on the candidate and reference side.
