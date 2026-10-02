# Compiled WRF bookkeeping comparisons

`bookkeeping.py` calls the source-pinned WRF v4.7.1 `small_step_prep`,
`calc_p_rho`, `small_step_finish`, `sumflux`, and the divergence-filter
section of `advance_uv`. It uses the shared `build.py` C ABI shims, real
WRF argument lists, `(i,k,j)` memory, horizontal halos, and full vertical
staggering. Pressure and other forcings are zeroed only for the isolated
divergence-filter call, so the filter's own arithmetic and boundary bounds
remain the native routine's.

The six source-state cases cover an initial and evolved real state,
steep terrain, map-factor extremes, zero and near-zero fields, and the
southern hemisphere. Each case exercises periodic and open boundaries.
Periodic duplicated U/V faces and their map factors are made equal before
calling either model. Prep has RK stages 1, 2 and 3; finish has provisional
and final heating branches; sumflux has 1, 3 and 6 substeps; the filter has
coefficients 0.01 and 0.015625. There are 120 packaged cases.

Build and regenerate under CPU niceness:

```sh
nice -n 10 python tools/smallstep_wrf471_oracle/build.py /path/to/wrf471 /path/to/build
nice -n 10 python tools/smallstep_wrf471_oracle/bookkeeping.py \
  --library /path/to/build/libsmallstep_oracle.so \
  --output tests/data/wrf471_smallstep/bookkeeping
```

Run on an authorized GPU:

```sh
python tools/smallstep_wrf471_oracle/bookkeeping_receipt.py \
  --output tests/data/wrf471_smallstep/bookkeeping/measurements.json
python -m pytest -q tests/test_smallstep_bookkeeping_wrf471_parity.py
```

The fixture manifest pins every input/output archive. The measurement file
pins every output's word count, changed-word count, largest ULP distance,
nonfinite mismatch count and absolute difference. Tests assert equality to
those measurements; no number is an acceptance tolerance.

All 24 OUT/INOUT prep arrays, all 15 finish arrays, all 4 pressure-diagnosis
arrays and all 3 sumflux arrays have comparisons. The schema coverage test
fails if a WRF output is omitted. Time-level snapshots and saved fields map
to the engine's actual retained arrays. The engine's fused kernels keep
face masses and `c2a` in registers. Test-only stores expose those registers,
then the actual engine launcher is invoked with that observed source. Every
ordinary output must remain word-identical to the unobserved production
launch before an observation is used. The finish-stage total WW is
reconstructed through the actual one-substep scalar sumflux path, because
the engine does not keep WRF's separate post-finish WW buffer.

WRF uses theta minus 300 K where the engine couples full theta. The native
comparison keeps WRF's 300 K constant and explicitly maps the engine's
coupled heat perturbation. Separately labelled full-theta isolation calls
show the representation's rounding effect; those calls do not replace the
native reference. The existing declarations are `gpuwm/core/ieva.py:32-50`.
That file also declares the engine's mean of already-total column masses,
where WRF sums perturbation and base mass separately.

The CPU FP32 traces in `smallstep_bookkeeping_oracle.py` explain measured
differences using the production source's operation boundaries. They are
never the oracle expected values. Their words must agree exactly with the
real GPU launch in all 516 traced arrays. The compiled WRF output words
remain the independent reference.
