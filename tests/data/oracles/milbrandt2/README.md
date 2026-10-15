# Milbrandt-Yau (mp_physics = 9) WRF v4.6.1 column oracle fixtures

Graded by `tests/test_milbrandt2_wrf461_exact_gpu.py`. Built by
`tools/milbrandt2_wrf461_oracle` (see its README for the full recipe).

Each set is a pair of xz-compressed little-endian streams:

- `in_<set>.bin.xz`: the inputs, in the layout `gen_columns.py` writes
  (`int32 ncol, nz, nsteps`; `float32 dt`; 18 `(ncol, nz)` fields
  `qv qc qr qi qs qg qh nc nr ni ns ng nh th pii p w dz`; 5 `(ncol)` fields
  `p8w(k=1) rainnc snownc grplnc hailnc`).
- `wrf_<set>.bin.xz`: what WRF v4.6.1's own `mp_milbrandt2mom_driver`
  returned for those inputs, `nsteps` successive calls, in the layout
  `run_my2.F90` writes (per step: 15 `(ncol, nz)` fields
  `qv qc qr qi qs qg qh nc nr ni ns ng nh th zet`, then 9 `(ncol)` fields
  `rainnc rainncv snownc snowncv grplnc grplncv hailnc hailncv sr`).

The WRF side is the pinned, byte-unmodified
`phys/module_mp_milbrandt2mom.F` (WRF commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`), compiled with WRF's own GNU
flags (`-O2 -ftree-vectorize -funroll-loops -w -ffree-form
-ffree-line-length-none -fconvert=big-endian -frecord-marker=4
-fallow-argument-mismatch -fallow-invalid-boz`) by gfortran 13.3.0 against
glibc 2.39 (Ubuntu 24.04). The compiled object is byte-identical to the one
a configure-option-32 WRF build links into `wrf.exe`, and the runner linked
against that very object returned the same bytes.

| set | columns | levels | dt (s) | calls | content |
|---|---|---|---|---|---|
| dt20 | 91 | 50 | 20 | 10 | named regimes (convective core and downdraft, stratiform melting, stable night and fog, winter snow, freezing rain, maritime drizzle, high terrain, anvil, 70 columns) and 21 edge columns |
| dt60 | 91 | 50 | 60 | 6 | the same columns at a long step (more sedimentation passes) |
| stress50 | 256 | 50 | 10 | 5 | randomised columns (`gen_columns.py --set stress`) |
| stress80 | 256 | 80 | 30 | 4 | randomised columns at 80 levels (the 256-level sedimentation kernel); the first 4 of the 8 calls the harness runs |

SHA-256 of the uncompressed streams:

```
2834b5ddcc1238851d1801d2e783f29efd1cebdc1e4331d3e5a6bfcda7433c59  in_dt20.bin
daea7fbe42477e846508af616180b383eefaf536606b9a7946c890191464c6dc  wrf_dt20.bin
474ab1c81f84eefbf797b177bb9cbd83cb324df2a226188656fb5e46bd997ad8  in_dt60.bin
f0445f82103d1707de41bacbcdfb6f3ea5316f9af3bd397bd529f23920dea524  wrf_dt60.bin
7233bf199a6ead73572b32d891d22a5612e7e742d6c9792101ff88af7d3d0b44  in_stress50.bin
1a2fa644f05d25d1f4d652ed6f4204e036a7bb24cdaaa411e71c1132ad52883b  wrf_stress50.bin
2cfc5504c3dab9e5411416ee30f3caa71a35d0875d5c111730830082c99f8f41  in_stress80.bin
84a0b44374f27b21e29a84b5cbe4eda019b80be2fdc76d43ec7fd0cda074ff0c  wrf_stress80.bin
```
