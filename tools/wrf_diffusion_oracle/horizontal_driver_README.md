# Compiled WRF horizontal diffusion outer driver

The storage adapter calls WRF v4.7.1 `horizontal_diffusion_2` with unchanged
routine bodies and constants. The 14 real-state and edge input probes are
called with both `km_opt=2` and `km_opt=4`, giving 28 driver cases. The native
caller passes its original perturbation-theta words as the `thp` argument.

Every returned tendency is retained, including all four moisture slots.
Inactive chemistry, scalar and tracer slots retain distinct sentinels.
All nine inactive NBA stress slots are also returned and checked. These
inactive slots have no corresponding engine writer in the tested configuration.
Active fields use the engine's production launchers and open-row staging.

```bash
python tools/wrf_diffusion_oracle/horizontal_driver_build.py DIFFUSION_SOURCE CONSTANTS BUILD
python tools/wrf_diffusion_oracle/horizontal_driver.py capture DEFORMATION_FIXTURES PREPARATION_SO DRIVER_SO FIXTURES
python tools/wrf_diffusion_oracle/horizontal_driver.py compare FIXTURES GPU_RECEIPT
python -m pytest tests/test_diffusion_drivers_wrf471_parity.py -q
```

The production receipts retain every float32 word's full-array hash and exact
distance from native WRF. The RTX 4090 and RTX 5090 produce some distinct
horizontal words; each complete case must equal a recorded platform variant.
The source, compiled bodies, constants, compiler flags, producing tools and
fixture files are sealed together.

The `--diagnostic` control restores WRF's metrics, contraction setting,
interpolation, stress and scalar-flux stages, and perturbation-theta mixing.
That control reproduces every array word in all 28 native driver cases.
The recorded production differences include cancellation near zero, so a
large ULP count can accompany a small absolute tendency difference.
