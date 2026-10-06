# Compiled WRF diffusion oracles

The fixtures call the real WRF v4.7.1 Fortran routines, using their original
argument lists, C-grid staggering, tile limits and physical or periodic halos.
Numerical routine bodies are byte-unmodified source slices. Configuration
service declarations and C ABI wrappers provide only the standalone harness.
WRF's own constants, metric preparation, thermodynamic preparation and physical
boundary routines are compiled with the tested operators.

`build_common.py` pins the WRF commit and source hashes. Builds use gfortran 15,
`-O0 -ffp-contract=off -fno-tree-vectorize -fcheck=bounds`. The build receipts
record every extracted slice, wrapper, compiler version and command. The
reference uses WRF's default binary32 REAL.

The fixture corpus contains a real initialization-state crop with all 49
vertical levels. Additional flow probes combine those thermodynamics with real
one-hour C-grid winds; they are explicitly labeled operator probes. Edge cases
include steep terrain, physical boundary rows, map factors from 0.25 to 4,
zero and near-zero flow, and a hemisphere reversal. NetCDF decoding is Rust;
Python assembles cases and orchestrates the compiled reference and engine.

Run the gates under the host's GPU ownership protocol:

```sh
PYTHONPATH=.:gpuwm-data:tools/wrf_diffusion_oracle python -m pytest -q -s -o addopts= \
  tests/test_diff6_wrf471_parity.py \
  tests/test_deformation_wrf471_parity.py \
  tests/test_horizontal_diffusion_wrf471_parity.py \
  tests/test_vertical_diffusion_wrf471_parity.py \
  tests/test_constant_diffusion_wrf471_parity.py \
  tests/test_diffusion_drivers_wrf471_parity.py
```

The gates compare every output word. Bit-exact routines assert uint32 equality
to compiled WRF. Other routines assert complete product-word hashes and exact
WRF mismatch statistics against recorded platform variants. A maximum ULP
distance is a reported measurement, never an acceptance tolerance. Fixture,
producer and receipt hashes are checked separately on CPU.

`deformation_compare.py`, `horizontal_compare.py`, `vertical_compare.py` and
the outer-driver tools use the engine's normal launchers. No diagnostic
arithmetic is used by the acceptance path. Tools-only native-metric, no-FMA,
WRF operation-order and glibc-power controls identify the numerical causes.
`rounding_attribution.py --family cpu` also compiles scalar Fortran witnesses
for the square-root versus `**0.5` distinction. These controls establish a
cause; they do not upgrade production results to bit identity.

The outer-driver fixtures retain all tendency outputs, inactive moisture
slots, diagnostics and mutable inputs. Seven coefficient mutation arms cover
km2, km3, km4, isotropic settings and zero TKE. The horizontal and vertical
drivers retain the original WRF coefficient choices and surface switches.

To rebuild the most-used references, place copies of the pinned WRF tree and
input files in private CPU scratch. With `SOURCE` naming that tree:

```sh
nice -n 10 python tools/wrf_diffusion_oracle/deformation_build.py \
  SOURCE/dyn_em/module_diffusion_em.F SOURCE/share/module_model_constants.F BUILD/deformation \
  --bc SOURCE/share/module_bc.F --big-step SOURCE/dyn_em/module_big_step_utilities_em.F
nice -n 10 python tools/wrf_diffusion_oracle/horizontal_build.py \
  SOURCE/dyn_em/module_diffusion_em.F SOURCE/share/module_model_constants.F BUILD/horizontal
nice -n 10 python tools/wrf_diffusion_oracle/vertical_build.py SOURCE BUILD/vertical
```

Each capture tool's `--help` declares its input, Rust reader, prepared-library
and output arguments. Regeneration is separate from acceptance; new product
pins require a full-word comparison and explained changes. See
[diff6_README.md](diff6_README.md) and
[constant_README.md](constant_README.md) for those two reference families.

These tests qualify the listed operators and switch settings. They do not
establish whole-forecast identity or observational forecast skill. The
constant-K physical-space operator and vertical-W coefficient choice retain
their documented differences from WRF. The explicit vertical path is covered;
an implicit vertical solver or an unmatched closure has no parity claim.

The opt-in WRF-exact branches (merged 1efb5a415) moved `smag2d` in the production
receipts (deformation, horizontal, both drivers, km-mutations and the vertical
comparisons) from `381bbd6f` to `c9165bb9` with no capture, because the default
compile did not move: from the capture tree (83fde6032) to the merged head all 27
entries compile to identical PTX for compute_89, compute_90 and compute_120 under
the loader's and the RawModule options, with NVRTC 13.4 and 12.9
(`tools/kernel_ptx_identity/receipts/oracle-diffusion-2.8.2-nvrtc{13.4,12.9}.json`).
The diagnostic receipts (no-FMA, reference metrics and order, reference density,
rounding) keep the source their transforms were applied to.

The 2.8.6 Smagorinsky changes (225fc6a88, rounding pinned to native Blackwell
arithmetic, and 1f625334f, bounded 32-bit addressing) moved `smag2d` from
`b60a0290` to `400d89d0`. These did change the compiled code, so the ten
production receipts were re-measured through the unmodified launchers with the
compare functions of `deformation_compare.py`, `horizontal_compare.py`,
`horizontal_driver.py`, `vertical_driver.py` and `deformation_mutations.py`. On the
RTX 5090 every word of all 84 cases equals the 2.8.5 receipt, so no Blackwell
output moved. On the RTX 4090 the deformation, vertical-driver and
km-mutations words are unchanged; 10 of 14 horizontal cases and 24 of 28
horizontal-driver cases moved, and every one of them now equals the Blackwell
word, which is the intent of 225fc6a88. The horizontal and horizontal-driver
receipts are therefore word-identical across the two cards; the deformation
receipts still differ in 10 of 14 cases.

Byte identity is certified on Blackwell and newer only (ruling 2026-10-04), so
the driver word receipts (`horizontal-driver`, `vertical-driver`,
`km-mutations`) follow the same rule as the advection receipts. Each card that
measured a receipt is declared with its compute capability in
`RECEIPT_CAPABILITY` in `tests/test_diffusion_drivers_wrf471_parity.py`; a
receipt from an undeclared card fails. A Blackwell receipt must pin the
`smag2d` source the loader compiles, and each folder needs at least one current
Blackwell receipt. An RTX 4090 receipt that trails the source is reported
(`UncertifiedReceiptStale`) and its words are not accepted until it is
recaptured, instead of holding the test red as it did after 1f625334f. On an
uncertified card with no current receipt of its own, the device word tests skip
and say why.
