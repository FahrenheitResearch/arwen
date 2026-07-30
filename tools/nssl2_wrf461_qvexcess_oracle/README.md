# WRF v4.6.1 NSSL QVEXCESS oracle

This directory compiles the official WRF v4.6.1
`phys/module_mp_nssl_2mom.F` and calls its `QVEXCESS` routine directly.
The patch changes only module visibility; no executable statement is changed.

`qvexcess.F90` covers both local evaporation branches, condensation,
zero-increment boundaries, lookup-table clamps, perturbation/base splits,
ordinary and 90-percent maximum-supersaturation targets, and large finite
inputs. Its diagnostic trace is a literal single-precision transcription of
the two local iterations. Every generated row first requires the trace's final
`qvex` bits to equal the result returned by the official routine.

`workspace_qvex` is a second, independently emitted official-routine result.
It passes already-combined FP32 theta and vapor in the same representation as
the production workspace API; it is not inferred by recombining fixture
outputs from the split-input call.

The fixture also records the separate default caller update from WRF lines
11544-11567: latent potential-temperature adjustment, vapor/cloud transfer,
and `imaxsupopt=4` cloud-number/predicted-CCN coupling. That caller update is
not attributed to the pure-return `QVEXCESS` routine.

Build only in a new directory:

```bash
tools/nssl2_wrf461_qvexcess_oracle/build.sh \
  /path/to/WRF-v4.6.1 /new/oracle-build
```
