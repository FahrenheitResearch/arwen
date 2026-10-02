# Complete outputs from the compiled WRF vertical diffusion driver

The adapter calls WRF v4.7.1 `vertical_diffusion_2` and its unchanged leaf
routines. All 14 real-state and edge input probes run with `km_opt=2/4` and
`isfflx=0/1/2`, giving 84 driver cases. Open rows, steep terrain, map factors,
zero and near-zero flow, and both hemispheres remain in the measured arrays.

Each case retains all active tendencies, the inactive tendency slots, all
moisture input slots after the call, perturbation theta, HFX, QFX, chemistry
and all nine NBA slots. The inactive slots carry distinct sentinels and have
no engine writer in this configuration. The GPU TKE buffer is seeded before
the call for the `km_opt=4` skip-path check. Every array is measured by its
float32 words and pinned to its observed full-array hash.

```bash
python tools/wrf_diffusion_oracle/vertical_driver_build.py DIFFUSION_SOURCE CONSTANTS BUILD
python tools/wrf_diffusion_oracle/vertical_driver.py capture VERTICAL_FIXTURES PREPARATION_SO DRIVER_SO FIXTURES
python tools/wrf_diffusion_oracle/vertical_driver.py compare FIXTURES GPU_RECEIPT
python -m pytest tests/test_diffusion_drivers_wrf471_parity.py -q
```

The comparison covers the raw outer driver and the production vertical
launcher. The engine mixing package subsequently clears open dry and
moisture rows. Its final caller mask is covered separately by the boundary
regressions and model smoke; raw surface boundary fluxes are retained here.

The prescribed-heat arms previously left HFX at its previous value. They
now refresh available HFX output using WRF's `heat_flux*cpm*rho` expression.
The observed maximum distance is one ULP. A tools-only native-density replay
reproduces every HFX word, identifying density division as the remaining
rounding source. A missing HFX field leaves the dummy mass buffer unchanged.

```bash
python tools/wrf_diffusion_oracle/vertical_driver.py compare FIXTURES DENSITY_RECEIPT --reference-density-from VERTICAL_FIXTURES
```

The `km_opt=2` vertical-w tendency remains different because of the declared
coefficient choice in `launch_wrf_smag2d_vertical`: WRF passes horizontal
`xkmh`, while the engine passes vertical `xkmv`. The two coefficients
coincide for `km_opt=4`. The receipt retains this structural difference.

The `deformation_mutations.py` packet closes the coefficient routine's
mutable TKE and moisture outputs. Fourteen inputs and seven coefficient,
isotropy and seed arms give 98 native calls. All 4,609,920 returned TKE and
moisture words are bit-identical on both tested cards.
