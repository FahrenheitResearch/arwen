# Fetch stream receipts and width rows

The packaged `gpuwm/fetch-width-rows.json` supplies the default file width and
chunk-stream budget. The generic selector reads optional deployment hints:

```sh
export GPUWM_FETCH_PROVIDER=runpod
export GPUWM_FETCH_BOX=your-box-row
export GPUWM_FETCH_BANDWIDTH_MBPS=125
```

Bandwidth is megabytes per second, not megabits per second. Add provider and
box rows to the JSON table, without adding provider adapters. Exact box rows
win; bandwidth ranges win over provider fallbacks. Unknown boxes use a
matching provider fallback or the default row. `GPUWM_FETCH_ROW=balanced`
pins a row for a width sweep. `--fetch-workers N` overrides its file width
and keeps its stream budget. Existing host caps and request pacing still
apply. The concurrency receipt records `worker_row` and `stream_budget`,
including explicit worker overrides.

Rows describe the strength of their evidence. The supplied 49-file transfer
observation is 150 seconds; its bandwidth and width were not supplied.
That observation does not establish the optimal width for a provider.

The new `rw_fetch object --url URL --output FILE` door streams declared
GRIB2 profile objects through the existing download client's retries,
range validation, host pacing and cache. It receives bounded batches of
16 MiB ranges, consumes one while the next downloads, hashes in payload
order and validates every message indicator, length, edition and terminator.
The partial publishes only after the whole stream validates and flushes.
Unique partial names tolerate interrupted attempts. Publication uses an
atomic hard link that refuses an existing destination. A filesystem without
hard links refuses publication by name. An origin without range totals uses
the existing buffered full-body fallback, which cannot overlap hashing with
network transfer. It still avoids a second disk hash pass.

The ordinary Rust model-run door uses the same consumer for whole-file and
index-subset transfers. Fresh receipts carry `sha256`, `grib2_messages`,
`bytes` and `last_byte_unix`. The last timestamp is the completion of the
last verified range response after the client's accounting, not a packet
capture. Python emits `fetch.last_byte` with that timestamp in its attributes
and a measured receipt delay. The C1 event timestamp is the receipt handoff.
Use the measured response timestamp for response-to-end comparisons, and
retain that distinction in timing reports.

Fresh Python manifest entries consume the Rust digest and envelope census
after matching the published Unix inode, device, size, mtime and ctime to
the Rust receipt. On Windows the receipt and the binding both read the
volume serial, file index and size from `GetFileInformationByHandle` and
the last-write and change times from `FILE_BASIC_INFO`; `os.stat` is not
used there because its `st_ctime` is the creation time. A rename rebinds
the same inode without reading its bytes. A kernel notice covers changes
inside one timestamp tick: inotify on Linux, a read oplock on Windows.
The oplock holds the payload open, which would refuse a rename of its
folder, so Windows drops those facts when the last fetch in the process
releases its output folder. Platforms without either retain independent
disk checks. Resumed processes still hash from disk.
An append or replacement invalidates fresh facts, so supplemental appends
are rehashed. `inputs.txt` keeps its existing ordered-path format;
`SHA256SUMS` and the fetch manifest carry the payload digests it binds.
An installed backbone without the new object door keeps the existing table
route transport. GRIB1, compressed and other format routes also keep their
existing transport. A substituted downloader or opener remains in control.

Whole-window table fetches bind planned input and supplement paths on the
first verified payload completion, even if an earlier transfer still blocks
ordered admission. This handoff does not certify missing files. Ordered
prefix receipts remain incomplete until every requested lead is verified;
the final `complete=true` manifest remains the preparation barrier. Lead
composition also runs while later inputs download. Named donor inputs are
fetched before the primary transfer pool starts.

Check a freshly built `rw_fetch` by selecting it explicitly:

```sh
export GPUWM_RW_FETCH=tools/rustwx/target/release/rw_fetch
python -m pytest -q tests/test_fetch_stream_tail.py
python tools/fetch_stream_validate.py --out fetch-stream-validation   --binary "$GPUWM_RW_FETCH"
```

The 49-object local HTTP check is transport validation, not an R1 bench.
It produces three labelled receipts, independent file digests, C1 ledgers,
and waterfalls through `gpuwm timeline`, then deletes its payload files.
It contains no forecast integration, so PROXY FF is unavailable. C3 R1
COLD/WARM savings and C4 prepared/history identity require the sealed case.
