# Regional packing benchmark

`tools/bench/ensemble_packing_bench.py` runs the original prepared forecast
runner against one sealed input bundle. Run every arm in a fresh process
inside the GPU card queue. Use the same configuration and WPS authority for
all arms. Set `restart_interval_s = 0` before preparation, since checkpointed
rosters retain sequential execution.

For each K in 1, 2, 4 and 6, grant one card and run:

```sh
python tools/bench/ensemble_packing_bench.py \
  --prepared-root "$PREPARED" --config "$CONFIG" --wps "$WPS" \
  --kind packing --members "$K" --cap "$K" --devices 1 \
  --out "$BENCH_ROOT/k-$K" --receipt "$BENCH_ROOT/k-$K.json"
```

The seeded surface-state member zero has the same realization at every K.
Actual packing capacities and execution modes are in the receipt. A K=6
request that executes in two waves is not evidence that six members fit.
Other members deliberately have different surface realizations.

For deterministic arms, grant the requested card count and use `--kind det
--members 1 --devices N`. This runs one unperturbed forecast, spatially split
when N exceeds one. A short `--seconds 60` canary can warm the disk kernel
cache before full-duration measurements.
A tree canary uses the existing typed prepared-runner API with a shorter
experiment duration after verifying the original authority. Full-duration
tree measurements use the real runner CLI. The receipt records that canary
override separately.

```sh
python tools/bench/ensemble_packing_bench.py \
  --summarize "$BENCH_ROOT"/k-*.json "$BENCH_ROOT"/det-*.json
```

The table computes g as K times the K=1 wall divided by the packed arm wall.
Deterministic speedup is the one-card wall divided by the N-card wall.
Wall includes authority preflight, initialization, forecast, history output
and the runner's final digest. It excludes queue waiting, preparation and
the additional Rust-reader history hash pass. Compare forecast-only timings
in the receipts when attributing the result.
`seconds_per_member` is amortized arm wall, not an individual member's
latency. `member_wall_seconds` and `member_realtime_factors` report the
original runner's per-member latency and model-time rate separately.
Omit `--cap` for a final check of the production default.

Every numeric history field is hashed with dtype and shape recorded. The
identity check fails for missing frames, missing fields, changed dtype,
changed shape or changed payload. Retain small receipts, configurations and
logs, then remove raw histories and prepared data from the lane's manifest.
These measurements support only the recorded geometry, physics and run
duration. They do not establish forecast skill or another grid's speed.

## G2 regional measurement, 2026-10-08

The frozen parent has 375 x 369 x 60 mass cells at 3 km and a 15 s step.
Its ratio-four child has 400 x 368 x 60 mass cells at 750 m and a 3.75 s
step, corresponding to 401 x 369 WPS dimensions. Feedback is on. The
three-hour authority uses mp8, pbl5, nine-layer RUC and rte-rrtmgp,
with restart off and history at the initial and three-hour endpoints.
Receipts select `ordinary_concurrent_members`. Prepared component admission
retains the original route with: "domain 1: selected solar/aerosol policy
needs its current packed carriers and ABI qualification". The selected
physics and this admission guard were retained for every arm.
The supplied runtime is Python 3.12.3, CuPy 14.2.0 and NVRTC 12.9.86
on RTX PRO 6000 Blackwell cards. This is a throughput and exactness check,
not operational mp28/21-member-suite qualification or forecast skill.

| Model hours | Requested K | Actual live members per wave | Wall s | Wall/K s | g |
|---|---:|---|---:|---:|---:|
| 3 | 1 | 1 | 448.288 | 448.288 | 1.0000 |
| 3 | 2 | 2 | 805.244 | 402.622 | 1.1134 |
| 3 | 3 | 3 | 1255.106 | 418.369 | 1.0715 |
| 3 | 4 | 4 | 1624.833 | 406.208 | 1.1036 |
| 2 | 1 | 1 | 299.737 | 299.737 | 1.0000 |
| 2 | 6 | 4 then 2 | 1609.597 | 268.266 | 1.1173 |

K=6 uses a two-hour prefix so its upper-bound hold estimate stays within
1800 s. Its g uses the matching two-hour K=1 baseline. It retains the
original three-hour history cadence, so its saved frames are initial
frames only; the final canonical state digest proves end-state identity.
Memory admission prices 22.263106 GiB/member against 94.427307 GiB
available, fitting at most four simultaneous ordinary members.

Whole NetCDF file SHA-256, numeric field dtype/shape/payload hashes and
all 576 canonical final arrays match member zero run alone for each
completed packed arm. The full three-hour arms cover four history files
and 474 numeric field records across both domains and both endpoints.
The two-hour prefix covers two initial files and 236 records.

| Deterministic cards | Wall s | Forecast s | Speedup |
|---:|---:|---:|---:|
| 1 | 444.105 | 424.950 | 1.0000 |
| 2 | 943.712 | 922.297 | 0.4706 |

All four deterministic history files and all 576 final canonical arrays
are byte-identical between one and two cards. The supplied GIL runtime
reports that rank threads share an interpreter lock. Keep the existing
one-card deterministic default because measured s2 is below 1.3.
Free-threaded Python and another workload's speed are unmeasured here.

Two live ordinary members per card have the highest recorded throughput
and lower original-member latency than three or four. Set the default cap
to two. These are single-arm measurements; close throughput differences
are not a statistical significance claim.

The ordinary concurrency default is resolved in
`gpuwm.ensemble.packing_policy` before live memory admission. Explicit
request caps still win, the request authority is preserved, and a smaller
memory ceiling remains stricter. Qualified native member batches and
sequential checkpoint/radar/binding routes retain their existing policy.
The bare six-member, 60-second default canary supplied no cap and selected
two live members in three waves. Its member-zero history files and final
576-array state match the standalone canary exactly.

Evidence is delivered in `ensemble-speed-packing-2card/receipts/`:
`nested-k-*.json`, `det-{1,2}.json`, `default-proof.json`,
`whole-history-file-identity.json` and the matching table JSON files.
Each arm receipt records the command, source commit, configuration hashes,
geometry, admission and per-member original runner timings. The code
changes no solver, physics selector or clock.