# Object posting timestamps

Run from the repository root, with Python 3.10+ and no third-party packages:

```sh
python3 tools/race/noaa_posting_times.py --days 7 --output /path/seven/posting-times.json
python3 tools/race/noaa_posting_times.py --today --output /path/today/posting-times.json
```

The default interval is seven **complete UTC days ending yesterday**. Set
`--end-date YYYY-MM-DD` to reproduce a historical interval. `--today` collects
only cycles through the start of the current run; later cycle hours are omitted.
Use a separate output directory for each capture to retain its original evidence.

The tool requests public unsigned S3 ListObjectsV2 XML and NOMADS HTML directory
listings. It never downloads GRIB payloads or uses schedule estimates. Eight HTTP
workers run by default; `--workers` and `--timeout` control concurrency and the
per-request timeout. Source names, paths, sampled products, and boundary offsets
are declarative rows in `noaa-posting-sources.json`; a different source layout
uses `--sources PATH`.

`posting-times.json` contains:

- `rows`: one row per requested cycle and data file, including missing objects.
  `cycle`, `file`, `key`, `model`, `product`, and `lead` identify the object.
  `s3_last_modified` and `nomads_mtime` hold UTC timestamps, with numeric
  `s3_minutes_after_cycle` and `nomads_minutes_after_cycle`. Index timestamps are
  separate fields on the data-file row. `s3_status` and `nomads_status`
  distinguish absence, an unlisted NOMADS day, and request failures.
- `summary.cycles`: paired initial/terminal timestamps and the earliest terminal
  product, plus boundary f51 and **all required boundary files** (f03–f51) with
  and without indexes. The summary's required lead range is explicit.
- `summary.by_cycle_hour`: count, median, minimum, and maximum, using cycles
  with both native f00 and at least one terminal f48 product. Times are minutes.
- `requests`, `stages`, and `errors`: URLs, HTTP headers, wall timestamps, hashes,
  and failures. A request failure makes the process exit nonzero **after**
  writing the available observations. Ordinary absent objects do not fail.

The sibling `posting-times-raw/` contains original response bodies and their
HTTP receipts. S3 pagination is followed to completion. NOMADS days are discovered
from its root listing; days outside retention have null timestamps. No missing
object is assigned a synthetic publication time.

The primary race deadline is the earliest f48 among the configured products,
measured from native f00. Compute differences on each cycle before aggregating;
the difference of marginal medians is generally not the median window. Boundary
readiness is the latest posting over all needed hourly leads and, separately,
all those leads plus their indexes. The last forecast lead need not upload last.

S3 LastModified describes the current object version, so it does not recover
first posting if a file was overwritten. NOMADS directory times are interpreted
as UTC and have minute precision; these mtimes do not establish first HTTP
visibility. Positive `s3_minus_nomads_minutes` means the NOMADS mtime is earlier.
The tool is an observational study, not an availability guarantee or a live
publication watcher.

For a 48-hour run, with all costs in seconds:

```text
seconds_per_forecast_hour < (60 * window_minutes - launch_cost - publish_lag) / 48
```

Charge detection/download/preparation costs after the f00 timestamp to launch
cost, and avoid double-counting publication already included in the forecast
hour rate. Choose the endpoint whose deadline the run is intended to beat.
