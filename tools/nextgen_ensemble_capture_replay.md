# Captured member output replay

`nextgen_ensemble_capture_replay.py` passes retained output vectors through the
current `ReachAccumulator`. It creates actual member sample files and the v2
reduction without running a hydrologic driver. Source histories must have
`reach_id_words[feature,2]` and native f32/f64
`discharge_m3_s[time,feature]`. Declare the selected record's UTC explicitly.

Run on the source card through its GPU pool, with the existing Rust NetCDF
reader/writer and CuPy environment. Keep HOME, temporary files and caches in
the caller's data-disk scratch. For example:

```bash
python tools/nextgen_ensemble_capture_replay.py \
  --member 4=/scratch/run/member-4.nc --member 7=/scratch/run/member-7.nc \
  --record 11 --valid-time 2023-04-02T01:00:00Z \
  --threshold 0.005 --threshold 0.15 --threshold 10 \
  --source-scope 'Describe the original member runs and input provenance' \
  --out /scratch/run/new-capture-replay
```

`handoff.json` maps each sample's original member ID, UTC, int64 ordered IDs,
native discharge dtype, units, file and variable. Exact uint32 low/high ID
words are also stored. Sample paths resolve beside `handoff.json`; reduced
frame paths resolve under its `nextgen/` directory. `nextgen/reach-ensemble.json`
uses `arwen.nextgen.reach-ensemble/v2`, float32 thresholds, uint32 counts and
fraction probabilities. Comparison is `ge`; the complete requested roster is
the denominator. Dry members remain in the reduction. Any missing finite
member masks probability while preserving counts.

Retained `arwen.nextgen.reach-ensemble/v1` files used float64 thresholds.
They remain historical artifacts: changing their dtype label or converting
only their threshold variable does not produce a v2 reduction. Recompute
counts from the original native-width member discharge into a new output
directory. In v2, CUDA rounds thresholds once to float32; comparison and
publication use those same values. Native float64 discharge remains float64,
including values immediately below the rounded threshold.

An optional source `channel_depth_m[channel]` and
`channel_reach_id_words[channel,2]` are copied separately. This source vector
is the history's final channel-output deadline: select the final discharge
record and its matching UTC when including it. Its IDs may differ from the
discharge axis; `channel_output` provides the explicit mapping and local-bed
metre reference. The sample's same-discharge-axis `channel_depth_m` stays null.
No discharge-to-depth conversion is performed. HAND depth probability remains
the products consumer's strict `>` operation.

The tool checks the written counts, finite coverage and probability bits
against CUDA vector expressions over every captured member, and records dry
reach counts. It reports cold/setup, capture, Rust publication and verification
times separately. Diagnostic event waits occur only in this offline tool.
These measurements do not time a routing step. Native physics parity and the
original members' independence/spread must be established by the source run's
receipts; replaying identical inputs does not establish forecast spread.
