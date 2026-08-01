# t=0 full-state parity digest

- schema: `gpuwm.t0-state-digest/v1`
- evaluator commit: `acb2ff3de53331f4eb70849cc55602015eacb77f`
- registration sha256: `148b8442ec6367c8433e08df6e01a47e9469a18766b363e2332e7cd498c287fa`
- ceilings (from `gpuwm.verify.nest_gates`): max 8 ULP, p99 2 ULP, |mean signed| 0.25 ULP
- covered groups: accumulation, diagnostic, dry_dynamics, moisture, soil, surface
- verdict: **FAIL**

| domain | group | scored arrays | max ULP | p99 ULP | mean signed ULP | verdict |
|---|---|---:|---:|---:|---:|---|
| d01 | accumulation | 4 | 0 | 0 | +0.0000 | PASS |
| d01 | diagnostic | 7 | 1133903872 | 1133903872 | +161986267.4286 | FAIL |
| d01 | dry_dynamics | 10 | 2174697473 | 1007772929 | -2936283.5477 | FAIL |
| d01 | moisture | 8 | 1437414 | 313 | -5.4783 | FAIL |
| d01 | soil | 6 | 1133779152 | 3337 | +8919507.1774 | FAIL |
| d01 | surface | 24 | 2106357504 | 2017730 | -99958.6513 | FAIL |
| d02 | accumulation | 3 | 0 | 0 | +0.0000 | PASS |
| d02 | diagnostic | 7 | 1133903872 | 1133903872 | +161986267.4286 | FAIL |
| d02 | dry_dynamics | 10 | 2189748225 | 1022234311 | -3451040.3231 | FAIL |
| d02 | moisture | 8 | 2178734 | 1310 | +8.9252 | FAIL |
| d02 | soil | 6 | 1133744592 | 1132987728 | +13436805.1565 | FAIL |
| d02 | surface | 24 | 2069273187 | 710424 | -42074.4007 | FAIL |
| d03 | accumulation | 3 | 0 | 0 | +0.0000 | PASS |
| d03 | diagnostic | 7 | 1133903872 | 1133903872 | +161986267.4286 | FAIL |
| d03 | dry_dynamics | 10 | 2187296769 | 1027434191 | +8615108.2799 | FAIL |
| d03 | moisture | 8 | 1373254 | 1393 | +5.5183 | FAIL |
| d03 | soil | 6 | 1133491019 | 507 | +7245613.9226 | FAIL |
| d03 | surface | 24 | 23592960 | 62955 | +5361.5342 | FAIL |
| d04 | accumulation | 3 | 0 | 0 | +0.0000 | PASS |
| d04 | diagnostic | 7 | 1133903872 | 1133903872 | +161986267.4286 | FAIL |
| d04 | dry_dynamics | 10 | 2128637952 | 1029100357 | -5739480.4637 | FAIL |
| d04 | moisture | 8 | 905965 | 479 | +4.7989 | FAIL |
| d04 | soil | 6 | 1133214160 | 40 | +1599697.4692 | FAIL |
| d04 | surface | 24 | 14680064 | 47089 | +2272.6399 | FAIL |

## boundary group: unavailable

Not scored: no lateral-boundary file staged on the candidate and reference side.
