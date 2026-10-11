# The pinned WRF source, in the tree

`phys/module_mp_fast_sbm.F` is WRF v4.8.0's fast spectral bin microphysics
(FSBM-2, `mp_physics = 30`), LF-normalized: 9,053 lines, SHA-256
`fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e`, the
`source` pin in `../sources.sha256`. The file is identical in WRF v4.6.1,
v4.7.1 and v4.8.0. The copy here was taken from a WRF v4.7.1 checkout and
checked against that pin after removing the carriage returns.

It is here so the tests read the artifact itself: the routine map, the
findings catalogue and the transcribed constants in
`gpuwm/core/fast_sbm/catalogue.py` and `gpuwm/core/fast_sbm/constants.py`
are checked line by line against it (`tests/test_fast_sbm_cold_contract.py`),
and every line number the port cites refers to it.

WRF is developed at NCAR, operated by UCAR, and placed in the public domain;
UCAR's notice travels in the repository `NOTICE`, in
`licenses/LICENSE-WRF-public-domain.txt` and in
`gpuwm/data/fast_sbm/LICENSE-WRF.txt`. The scheme's authors are credited in
the file's own header and in the `gpuwm.core.fast_sbm` package docstring.

The oracle builds still take both WRF sources from a WRF tree (see
`../README.md`); `module_mp_SBM_polar_radar.F` is not copied here.
