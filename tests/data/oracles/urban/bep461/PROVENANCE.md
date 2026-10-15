# BEP v4.6.1 column fixtures

`wrf461-bep-fixtures.tar.xz` (sha256
`60535187eb3a47de7fc1d5f0d895cd07d137141ee4dc843b5f26de06302c5afa`) holds
`bep461/host/` and `bep461/cr/`, each the `fixtures/` directory written by
`tools/urban_bep_wrf461_oracle/build.sh` (host C library) and
`build_crlibm_arm.sh` (correctly rounded acosf/asinf/atanf/tanf from CORE-MATH
at pinned commits: asinf bc385c2, acosf 56dd347,
atanf a8066a5, tanf 59d21d7), plus that build's
`compiler.txt`, `libmvec-report.txt`, `oracle-sha256sums.txt` and
`fixtures-sha256.txt` (`cr/` also `crlibm-sha256.txt`).

Built 2026-10-07 on box W1 (AWS us-west-2, Ubuntu 24.04): GNU Fortran 13.3.0,
glibc 2.39, from WRF v4.6.1 (tag v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`) at `-O0`, no `_ZGV*` symbol in any
object. Each case directory is oracle_io's layout (`<name>.bin` raw
little-endian float32/int32 in Fortran order, `MANIFEST.txt`);
`gpuwm.verify.urban_oracle.load_case` reads it.

Gate: `tests/test_urban_bep_wrf461_column_oracle.py`.
