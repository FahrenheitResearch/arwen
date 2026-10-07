# Recent regional campaign scoring

`tools/da_recent_score.py` reads the authored case plan and the hash-bound regional run manifest. It schedules the existing native model-manifest builder and rain scorer for every member and its seed-matched no-DA member. It performs no Python weather-field transforms. Default scope is 32 members and seeds 20261003/20261004, with 1/3/6-hour metrics on each arm.

Use an existing frozen truth manifest:

```bash
python tools/da_recent_score.py --case "$RECENT_CASE/case-plan.json" \
  --run-manifest "$SHARED/public-case.json" --truth "$SHARED/truth/truth.json" \
  --seeds 20261003 20261004 --members 32 --out "$DA_ROOT/scores/recent-current" \
  --time-budget-seconds 1200
```

Or let the same shared 20 minute scoring budget include the native truth decoder:

```bash
python tools/da_recent_score.py --case "$RECENT_CASE/case-plan.json" \
  --run-manifest "$SHARED/public-case.json" \
  --inventory "$DA_ROOT/receipts/recent-inventory.json" \
  --fetch-receipt "$DA_ROOT/receipts/recent-fetch.json" --truth-out "$SHARED/truth-current" \
  --seeds 20261003 20261004 --members 32 --out "$DA_ROOT/scores/recent-current" \
  --time-budget-seconds 1200
```

All output paths must be fresh. `--plan-only` writes the exact commands without native field reads. For a separately scheduled serial clean-start arm, give its run manifest with `--control-run-manifest`; its fork, engine, prepared content and physics identity must match the DA arm. Use `--domain 2` only when both arms save that same nested domain.

For `gpuwm-da.recent-run.v1`, DA frames are `<run.out>/seed-<seed>/da/composites`, and matched no-DA frames are `<control.out>/seed-<seed>/no-da/composites`. The older per-invocation regional layout retains its last observed-slot subdirectory. Each member selects exactly `wrfout_*_<member>.nc`, or `wrfout_*_<member>_dNN.nc` for a child. The initial post-analysis `wrfout_start` is selected at the issuance clock. Native history callbacks consume the microphysics reflectivity stash and retain the final endpoint; the leg-end writer suppresses its duplicate. Other duplicate clocks are refused by the manifest builder.

Every product emits 36 rows: footprint rain ratio, domain rain ratio, integrated 35-dBZ area multiple, and nine FSS threshold/width pairs at each of three leads. Missing or late arms produce explicit pending rows. A partial forecast can retain completed early-hour rows when the native scorer has their full support. Failed builders and scorer errors have per-product logs and command receipts. The controller interrupts and cancels its owned child group on budget expiry, then records remaining products as pending. It never launches another child after interruption.

Outputs are per-product `da.jsonl`/`no-da.jsonl` and native `.receipt.json`, plus `all-members.jsonl`, `score-plan.json`, `command-receipts.json` and `campaign-summary.json`. The compact summary includes complete member counts, source/run/truth hashes and `mean_member_*` scalar summaries. A scalar mean is withheld unless the entire requested member roster has complete measurements. Mean member FSS is explicitly a mean of FSS scores, not FSS of ensemble-mean rain. The same applies to member rain ratios and area multiples.

The public native `rw_wrfbatch --ensemble-diagnostic-reduce` already computes a fixed-roster mean, but accepts diagnosed CDF5 packs marked `gpuwm-ensemble-diagnostic-spool.v1`, not these ordinary WRF snapshots. No such mean-field spool is written by this worker. Ensemble-mean-field scoring therefore remains pending. A native spool export with a pinned precipitation and reflectivity product definition is the missing connection. An analysed deterministic issuance is also a distinct product and is not created by this wrapper.

For saved fields, `--saved-inputs FILE` accepts schema `da-rerun.saved-score-inputs.v1` with `members` rows containing `seed`, `member`, `da` and `no_da` paths to existing `regional-rain/input.v1` manifests. The native scorer still owns readers, masks, reset accounting and metrics. This door is used by the synthetic CLI test; synthetic field qualification stays pending.

One recent development event cannot satisfy the spec's unseen-event count, event-blocked simultaneous confidence bounds, independent-gauge checks or ancillary losses. Completed rows are measurements. The campaign scientific gate remains pending.
