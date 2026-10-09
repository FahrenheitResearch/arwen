# Thompson rain-graupel slab tools

What remains here after the classic CUDA unit this directory was written for
was found never to have landed (2.8.8 intake): classic Thompson
(`mp_physics=8`) runs the classic kernels of `gpuwm/core/kernels/thompson.cu`
through `gpuwm.core.microphysics._apply_thompson`, and its WRF oracle is
`tools/thompson_real_column_parity` (`--mp 8`).

| file | what it does |
|---|---|
| `corrected_mp28.py` | the one-line WRF v4.6.1 source correction (`MIN(idx_bg(k),dimNRHG)` in the eight rain-graupel table reads, nothing else) that `tools/thompson_real_column_parity/corrected_real_reference.py` compiles for both the mp=28 and the mp=8 corrected references, plus the mp=28 157-column GPU check |
| `make_columns.py` | regime columns (convective, stable night with fog, snow, cirrus, ocean, high terrain, bright band, freezing rain, threshold columns); `--stress N` perturbed columns; `--nan` NaN-state columns |

## The WRF v4.6.1 read both ports decline

Rain collecting graupel (`module_mp_thompson.F:2524-2546`). The five tables
`tcg_racg`, `tmr_racg`, `tcr_gacr`, `tnr_racg`, `tnr_gacr` are allocated with
`dimNRHG = 1` for mp=8 and mp=28 and filled in that one slab, then read at
`idx_bg(k) = idx_bg1 = 5`. Column-major addressing lands on the entry for
drop intercept bin `idx_r1 + 4`, and past the end of the allocation for the
largest rain intercepts. WOOF reads the slab the tables hold:
`thompson_racg_index` in `thompson.cu` (mp=8) and `aaf_racg_index` in
`thompson_aerosol_common.cuh` (mp=28). Both are declared divergences from
WRF wherever rain meets graupel, graded against WRF compiled with only that
index corrected.
