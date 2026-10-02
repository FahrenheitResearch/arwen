# Compiled WRF v4.7.1 smallstep oracle

The oracle compiles the byte-unmodified WRF smallstep module and its own
model constants. `build.py` verifies both source SHA256 pins before compiling.
The reference uses gfortran `-O0 -ffp-contract=off -fcheck=bounds`. A data-only
configuration type supplies the fields read by the source. Generated wrappers
call the eight original routines with their complete argument lists.

From the repository root on a CPU host with gfortran:

```sh
nice -n 10 python tools/smallstep_wrf471_oracle/build.py WRF_SOURCE build/smallstep
```

The input extractor uses the engine's Rust NetCDF bridge. It retains all 49
levels and both horizontal staggers of the real state. Density omitted from
the evolved history file is explicitly labelled as an input diagnostic.

```sh
PYTHONPATH=. nice -n 10 python tools/smallstep_wrf471_oracle/make_cases.py \
  wrfinput_d01 wrfout_d01_2024-05-25_19:00:00 tests/data/wrf471_smallstep
```

Regenerate reference fixtures with `GPUWM_SMALLSTEP_ORACLE_LIB` pointing at
`build/smallstep/libsmallstep_oracle.so`. GPU measurements must obey the host's
GPU ownership protocol. All reference outputs come from compiled Fortran;
NumPy packs inputs and measures word differences. Diagnostic changes to CUDA
contraction, theta coordinates and operation order are separate causal witnesses.

```sh
python tools/smallstep_wrf471_oracle/horizontal_measure.py \
  --output tests/data/wrf471_smallstep/horizontal-baseline.json \
  --fixture tests/data/wrf471_smallstep/horizontal.npz
python tools/smallstep_wrf471_oracle/bookkeeping.py \
  --library build/smallstep/libsmallstep_oracle.so \
  --output tests/data/wrf471_smallstep/bookkeeping
python tools/smallstep_wrf471_oracle/bookkeeping_receipt.py \
  --output tests/data/wrf471_smallstep/bookkeeping/measurements.json
python tools/smallstep_wrf471_oracle/vertical_receipt.py \
  --library build/smallstep/libsmallstep_oracle.so \
  --output tests/data/wrf471_smallstep/vertical-native.json \
  --arrays tests/data/wrf471_smallstep/vertical-reference.npz
```

The reference arrays are test data in `tests/data/wrf471_smallstep`, not
package data: under `gpuwm/data/smallstep/oracle` they helped push the 2.8.2
pure wheel over the 100,000,000-byte per-file limit, and no runtime code reads
them. The harness modules (`gpuwm/verify/smallstep*_oracle.py`) run from a
source checkout and refuse by name in an install that has no `tests/data`.
The move changed no fixture byte; `oracle-sha256sums.json` moved its fixture
keys to the new paths and re-pinned only the harness and tool files whose
path text changed.

The normal tests use committed reference arrays and do not need a compiler or
the shared library. They assert exact measured discrepancies and output words,
not tolerance thresholds:

```sh
PYTHONPATH=. python -m pytest -q tests/test_smallstep_oracle_contract.py \
  tests/test_smallstep_horizontal_wrf471_parity.py \
  tests/test_smallstep_vertical_wrf471_parity.py \
  tests/test_smallstep_bookkeeping_wrf471_parity.py
```

These comparisons characterize the recorded answers and differences. They do not
establish full WRF trajectory identity or forecast skill.

The opt-in WRF-exact branches (merged 1efb5a415) moved the `acoustic` pin in
`vertical-native.json` (`d790562d` to `51bf4cce`) and in `oracle-sha256sums.json`
with no capture, because the default compile did not move: from the capture tree
(46b0a09fe) to the merged head all 14 entries compile to identical PTX for
compute_89, compute_90 and compute_120 under the loader's and the RawModule
options, with NVRTC 13.4 and 12.9
(`tools/kernel_ptx_identity/receipts/oracle-smallstep-2.8.2-nvrtc{13.4,12.9}.json`).
The workspace witness and the pressure-term mutation patch the default compile's
view of the module (`gpuwm/verify/default_kernel_source.py`);
`vertical-eos-causal.json` is a diagnostic and keeps the source its transforms
were applied to.

The bandwidth implementation is re-frozen on actual recorded-word reproduction. On the RTX 5090, all 132 native arrays and 305,632 binary32 words in `vertical-reference.npz` remain identical. The production receipt now pins the assembled acoustic source `2356c7c8`. Execution, source, compiler and every output hash are retained in `receipts/rtx5090-bw-acoustic-word-reproduction.json`. The archived reference arrays and measured WRF discrepancies are unchanged.
