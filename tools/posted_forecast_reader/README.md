# Posted forecast reader

This CPU reader feeds a posted deterministic forecast into the existing MRMS
and ASOS scorer. The source's records, units and conversions are a JSON table.
Adding another source on a supported grid changes that table, not this code.
Rust decodes the GRIB2, constructs coordinates, converts units and takes the
magnitude of declared wind components. The Python seam exposes read-only
native arrays and owns their lifetime. A score records the input hashes,
selected records, field-table hash and reader-library hash.

Build on a Linux CPU with a bounded worker count:

```sh
cargo build --release --manifest-path tools/posted_forecast_reader/Cargo.toml -j4
python tools/obs_battery_score.py --run-directory POSTED_FILES \
  --posted-field-table FIELD_TABLE.json \
  --posted-reader-library tools/posted_forecast_reader/target/release/libposted_forecast_reader.so \
  --read-only
```

Supported: deterministic instantaneous PDT 4.0 records on Lambert GDT 3.30,
positive i/j and i-consecutive scan 0x40. Duplicate fields, mixed cycles,
different grids, missing fields and unit-bound failures refuse the score with
the breakage named. Interval precipitation, maxima and ensemble products need
their own declared operators and are refused. Reflectivity is the published
composite, with no diagnostic recomputation.

`tests/test_posted_model_source.py` uses actual posted inputs. Set
`GPUWM_POSTED_TEST_INPUTS` to their directory and `GPUWM_POSTED_TEST_LIBRARY`
to the built library before running it. It verifies geographic round trips,
field units, native-array lifetime, ambiguous-input refusal and wrong-authority
refusal. A normal test environment without those artifacts skips this proof.

For a comparison, export each arm's admission with
`obs_battery_score.py --station-set-out CANDIDATE.json`. Intersect those
identity sets with `tools/intersect_station_sets.py ... --out SHARED.json`,
then score every arm with `--station-set-in SHARED.json`. Each arm still maps
stations through its own projection and must admit every shared station.
The file binds the observation station table and the scored valid times.
This prevents different station selections from becoming a skill difference.
Use one registration and the same observations, thresholds and lead set for
every arm. The default WRF reader and its numerical operators are unchanged.
