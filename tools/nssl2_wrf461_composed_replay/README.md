# WRF v4.6.1 / GPUWM composed NSSL column replay

This harness closes the gap between isolated process oracles and the live
option-18 coordinator. It extracts an actual 49-level atmospheric column from
a WRF-compatible MP18 output, then advances the identical FP32 column through:

- official NCAR WRF v4.6.1 `nssl_2mom_driver`; and
- GPUWM's production `gather/sediment -> fused GS -> NUCOND/QVEXCESS ->`
  `radar/effective radii -> scatter` coordinator.

The comparison covers all 16 Registry prognostics, potential temperature,
four precipitation categories, `sr`, `REFL_10CM`, and cloud/ice/snow effective
radii. The GPU receipt also preserves concentration-space snapshots after
sedimentation, fused GS, and NUCOND for attribution if the final comparison
fails.

The WRF source must be official commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, with
`module_mp_nssl_2mom.F` SHA-256
`5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3`.
The build changes no WRF executable statement.

Example:

```bash
python tools/nssl2_wrf461_composed_replay/extract_wrfout_column.py \
  /path/to/wrfout_d04_1974-04-03_14_00_00 /tmp/d04-1400-column \
  --dt-s 2.5

tools/nssl2_wrf461_composed_replay/build.sh \
  /path/to/WRF-v4.6.1 /tmp/nssl2-composed-build

/tmp/nssl2-composed-build/nssl2_composed_column \
  /tmp/d04-1400-column.txt /tmp/wrf.csv

python tools/nssl2_wrf461_composed_replay/run_gpu_replay.py \
  /tmp/d04-1400-column.npz /tmp/gpu.csv /tmp/gpu-stage-receipts.npz

python tools/nssl2_wrf461_composed_replay/compare_replay.py \
  /tmp/wrf.csv /tmp/gpu.csv /tmp/comparison.json
```

The current thresholds reuse the already admitted fused/process tolerances;
they are declared before inspecting a composed result. A failing field is a
diagnostic result, not permission to widen a tolerance.
