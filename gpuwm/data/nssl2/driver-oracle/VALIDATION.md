# NSSL pre-GS driver-support validation

Authority is the byte-pinned official WRF v4.6.1
`phys/module_mp_nssl_2mom.F` source. The standalone harness changes only
routine visibility and calls `calcnfromq`, `calcnfromcuten`, and `sediment1d`
from the official module. Its comparator reproduces the driver gather, the
complete cloud/rain/ice/snow/graupel/hail sediment sequence, precipitation
reducer, and final scatter.

## Evidence hashes

- Official WRF source SHA-256:
  `5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3`
- Oracle fixture SHA-256:
  `d5d8e3d13fbc3049c5c4962429d11ea7d0c519b05d13943d7becb2ccda1fe416`
- Python workspace/phase API SHA-256:
  `07bcc9598f8b945cd84b88c94507371517bb790252b301464f250176f0479154`
- CUDA implementation SHA-256:
  `4e52a801fbf2b5b1f877e5d1b0a02cdc00eec775e55271fbbdeb4cf490edfe51`
- GPU gate SHA-256:
  `271499db3ad86b31a15343029ff355950cd5615da0f284120d6ec7b5207c2ba2`
- Oracle build script SHA-256:
  `e6285aae086a6db1eb9c954d3e9922bde7a4078feddd72df782f010d1e5a5921`
- Oracle comparator SHA-256:
  `edfad1eede02a39c8da048f83a97e37ed12d1323afb7a6f97f8143601f5e3d9c`
- Oracle visibility-only patch SHA-256:
  `09c6cc8979243072bc5c5050e05f1155a63cd96bf155c6d701e61ba6780c5027`
- Oracle WRF stub SHA-256:
  `126dcfe0b800e9d79a85634201dad275bc578d7fe69f3a81df6ceb9efda4455e`

## Reproduction and gates

Fresh oracle reproduction:

```text
tools/nssl2_driver_wrf461_oracle/build.sh \
  /workspace/WRF-v4.6.1-reference /tmp/nssl2-driver-oracle-build-2
NSSL2_DRIVER_SUPPORT_ORACLE_COMPLETE driver-support.csv
d5d8e3d13fbc3049c5c4962429d11ea7d0c519b05d13943d7becb2ccda1fe416  driver-support.csv
```

Validation results:

```text
ruff check gpuwm/core/nssl2_driver_support.py tests/test_nssl2_driver_support.py
All checks passed!

python -m pytest -q -m gpu tests/test_nssl2_gpu.py tests/test_nssl2_driver_support.py
62 passed in 1.38s

python -m pytest -q tests/test_nssl2_contract.py
7 passed in 0.15s
```

The fixture contains 504 data rows spanning initialized and uninitialized
first-step states, all four KF inputs, empty/no-op columns, adaptive CFL > 1,
variable-density graupel/hail, separate and total precipitation, SR, and
water-plus-surface-export conservation. Registry arrays are verified bitwise
unchanged while the durable 16-field concentration workspace is live. The one
extreme CFL graupel-number receiving cell has a measured CUDA/Fortran relative
difference of `8.3219e-4`; its fixed gate is `1.0e-3`. Other moment gates retain
the previously admitted per-category bounds, and column conservation is gated
at `rtol=4.0e-6, atol=4.0e-7 kg m-2`.

WRF v4.6.1 loads all four KF rates, but its cloud/ice `calcnfromcuten` branches
gate on the zeroed KF number slots rather than their mass slots. The fixture
proves the exact result: rain and snow diagnose number increments; qccuten and
qicuten are consumed but make no number change in this official version.
