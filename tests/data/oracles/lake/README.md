# Correctly rounded lake intrinsics: pinned oracle vectors

Both files were written by `tools/lake_wrf461_oracle/fixtures.py` on the
verification box (Ubuntu 24.04, glibc 2.39, GNU Fortran 13.3.0, mpmath 1.3.0)
from the regime campaign of `tools/lake_wrf461_oracle/regimes.py`.
`tests/test_lake_f64cr.py` reads them.

`f64cr_cases.npz`: for each of exp, log, log10, sin, atan and pow, the
arguments and their correctly rounded binary64 results from a 240-bit
mpmath evaluation: 64 lake arguments where glibc 2.39 is not correctly
rounded (exp, log, atan, pow), 512 random arguments across the domain,
structured hard cases (exp of k 2^-53, log of 1 + k 2^-52, odd 27-bit
squares), and IEEE special cases whose expected words are glibc's.
SHA-256 `68eb05b99d0db0d725efafa6168feb32d7a3eccbaa20655b969a61a8e068846a`.

`lake_cr_regimes.npz`: 16 regime columns, 120 steps at 60 s: inputs,
every output word at every step and every state word at steps 30, 60, 90
and 120 of the unmodified WRF v4.6.1 `phys/module_sf_lake.F` (SHA-256
`b173ad8328ecc0aa6bb6e6306a9c6af4dc876b39dc5d304fe221122983214b98`) linked `-Bsymbolic` to correctly rounded binary64 functions
(`lake_fortran_cr.so`). With CUDA's binary64 library the CUDA lake left
these words at step 5. SHA-256 `32ab2cfe28468e5ece1e17f7b91560decf119fcdcfd19cb19973c5040942f5f2`.
