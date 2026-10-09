# km_opt=3 column oracle against compiled WRF 4.6.1

The 3-D Smagorinsky closure (`diff_opt=2`, `km_opt=3`, PBL off) is compared
word for word with WRF 4.6.1's own Fortran.

## What is compiled

`build.py` extracts unmodified routine bodies from a pristine WRF v4.6.1 tree
(tag `v4.6.1`, commit `d66e442f`; every source file pinned by SHA-256):
`compute_diff_metrics`, `phy_prep`, `cal_deform_and_div`, `set_physical_bc3d`,
`calculate_km_kh` with `calculate_N2` and `smag_km`, `horizontal_diffusion_2`
and `vertical_diffusion_2` with all their leaves. The C ABI storage adapters
are the ones in `tools/wrf_diffusion_oracle` (deformation, horizontal driver,
vertical driver). The oracle build is gfortran -O0 `-ffp-contract=off
-fno-tree-vectorize` (scalar glibc libm, no contraction). `--stock` builds the
same bodies with WRF's own GNU flags (-O2, vectorised, libmvec) for comparison.

## Cases

`km3_cases.py`: the 14 retained real-state crops of
`tests/data/wrf471_diffusion` (16 x 12 columns, 49 levels, dx 3 km: open and
periodic, steep terrain, map factors 0.25 to 4, zero and 1e-20 flow, southern
hemisphere, initial and evolved winds) and 12 LES-scale states (12 x 10
columns, 40 levels, dx 50 to 200 m): convective boundary layer, shallow
cumulus, stable night with a low-level jet, cold snow/ice cloud, marine
stratocumulus over ocean, a 2.0 to 2.6 km ridge, a vortex that binds the
coefficient caps, high-latitude map factors, and edges (zero flow, 1e-20 flow,
supersaturated at every level, qc exactly at the 1e-5 switch). 4,128 columns.
Every case runs with `mix_isotropic` 0 and 1, the vertical driver with
`isfflx` 0, 1 and 2.

## Arms

`capture.py` records every WRF output: the seven deformation tensors, BN2,
xkmh/xkmv/xkhh/xkhv, the horizontal tendencies of u, v, w, theta, qv, qc, qi,
and the vertical driver's tendencies plus HFX/QFX. The vertical driver runs
twice: `stock` (WRF hands `vertical_diffusion_w_2` xkmh) and `wcontrol`
(xkmv, the engine's documented choice, bbdc11150).

`compare.py` runs the production launchers and compares bitwise:
`coef`, `h_chain` (engine K), `h_iso` (WRF K, operator only), `v{f}` (engine
K) and `vi{f}` (WRF K).

## Rerun (box with gfortran 13.3, glibc 2.39, a GPU behind the mutex)

```sh
T=<tree>; source /work/pverify/base/env.sh; cd $T
python tools/wrf461_km3_oracle/build.py <WRF-4.6.1 src> OUT/build-oracle
python tools/wrf461_km3_oracle/capture.py OUT/build-oracle/oracle.so OUT/fixtures \
    --fixtures tests/data/wrf471_diffusion
GPUWM_WRF_EXACT=1 GPUWM_WRF_EXACT_DIFFUSION=1 \
    python tools/wrf461_km3_oracle/compare.py OUT/fixtures OUT/strict.json
python tools/wrf461_km3_oracle/compare.py OUT/fixtures OUT/default.json
python tools/wrf461_km3_oracle/seal.py OUT/fixtures tests/data/wrf461_km3
GPUWM_WRF_EXACT=1 GPUWM_WRF_EXACT_DIFFUSION=1 \
    python -m pytest -q tests/test_km3_wrf461_column_oracle.py
```

`seal.py` keeps the synthetic inputs and the SHA-256 of every WRF array in
`tests/data/wrf461_km3`; the gate compares digests (bitwise equality is digest
equality), so the 49 MB capture stays on the box. The ULP instrument is
`compare.py` on the full capture.
