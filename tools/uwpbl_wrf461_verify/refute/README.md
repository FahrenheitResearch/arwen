# UW PBL refute check (2026-10-07)

An adversarial re-check of lane/verify-uw-pbl's WRF v4.6.1 column-oracle
verdict. These tools add columns the lane did not use, then grade the
product launcher on them with the lane's own `../compare.py`.

- `make_refute_cases.py OUT` writes 1,920 new columns in 13 case files (QNC pair counted twice):
  - eight random grids, 33 to 150 levels, model tops 12 to 52 km, dt from
    6 s to 180 s, plus dt = 100/3;
  - a warm-stratopause set reaching 60 km, which drives the saturation
    lookups' es > p branches;
  - a 60-step carried history;
  - a liquid-number pair: the same columns with and without QNC;
  - calm-wind probes.
- `make_refute_bulk.py OUT NFILES NCOL` writes bulk random columns from the
  same generator (seed 2026100743), 2 steps each.
- `hostcr.py DIR OUT.json` takes a DIR holding `host/` fixtures (pristine
  WRF, glibc cos/acos) and `cr/` fixtures (the same objects relinked with
  correctly rounded cos/acos). It lists, word by word, every column-step
  whose input words are equal in both but whose output words differ.
- `diag_cols.py SPEC OUT.json` (GPU) compares the WRF word with the product
  word for named column-steps.
- `calm_check.py FIXTURE OUT.json` (GPU) reads step 1 of a fixture. It
  reports mismatches on the words WRF wrote as finite values, and the
  product's values where WRF wrote NaN.
- `inspect_fx.py DIR` checks host/CR equality, counts non-finite words and
  checks that the QNC pair is identical.

To run on a box:

1. Build the oracle with `../run_oracle.sh`.
2. Run `build/pristine/run_camuwpbl` and `build/crlibm/run_camuwpbl_cr` on
   each case file, writing into `host/` and `cr/`.
3. Run `compare.py --engine gpu` on both directories, once with
   GPUWM_WRF_EXACT=1 and once without it.
