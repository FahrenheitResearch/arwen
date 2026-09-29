# Real one-minute ASOS fixture

Every file here is the real `rw_asos` binary's own output over the
archive's one-minute ASOS route; nothing was typed by hand.

Built on Linux from this tree's `tools/rustwx/crates/rw-obs/src/bin/asos.rs`
(`cargo build --release -p rw-obs --bin rw_asos`) and run as:

```
rw_asos stations --networks IA_ASOS --bbox -94.2,41.3,-93.2,42.2 --out stations.json
rw_asos fetch --product asos1min --stations stations.json \
    --start 2024-05-21T11:50:00Z --end 2024-05-21T12:40:00Z --out observations_1min.csv
rw_asos decode --product asos1min --stations stations.json --obs observations_1min.csv \
    --start 2024-05-21T12:00:00Z --end 2024-05-21T12:30:00Z --out surface_1min.v2.json
rw_asos verify --file surface_1min.v2.json      # PASS, observation_times_proved true
```

1. `stations.json`: the frozen table (AMW, BNW, DSM, IKV, PRO), fetched
   2026-09-28 from the archive's network metadata. sha256
   `04eca47396502200cf6098f65a825234e53ef0d1937e6531d96fb158cfef7ae5`.
2. `observations_1min.csv`: the archive's `asos1min.py` answer, 102 rows.
   Only AMW and DSM have one-minute pages in this window; the route answers
   the other three with no rows. sha256
   `6c06678be5fd33e526a17cce44444afdccfb95de7b104a6195a01ece526383c7`.
3. `surface_1min.v2.json`: `gpuwm-obs.asos-surface.v2`, provenance product
   `iem-asos-1min`, 31 valid times at a one-minute stride, 62 reports, each
   serving the minute it was taken (`match_seconds` 30). `provenance.uri`
   was set to the repository-relative CSV path; nothing else was edited.

`tests/test_da_obs_surface_asos1min.py` reads the record through the surface
seam on a four-minute analysis schedule and, when `GPUWM_RW_ASOS` names a
binary, re-runs the decode against the committed CSV and compares.
