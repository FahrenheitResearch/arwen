# Historical GFS acquisition

The GFS endpoint ladder reads dated archive layouts from
`gpuwm/authorities/rw-wps-archive-layouts.v1.json`. The Rust fetcher embeds the
same document. New hosts, path eras, date bounds, products and lead spacing
belong in that table.

Available rows:

| Transport | Product | Layout begins | Terms |
| --- | --- | --- | --- |
| gdex | GFS 0.25 degree, RDA ds084.1 / GDEX d084001 | 2015-01-15 | CC BY 4.0, attribution required |
| ncei-0p5 | NCEI 0.5 degree forecast objects | 2006-10-10 | US government data |
| ncei-1p0 | NCEI 1.0 degree forecast objects | 2005-02-15 | US government data |
| s3 | AWS GFS 0.25 degree, two path eras | 2021-01-01 | US government data |

These dates describe published layouts. They do not promise continuous
holdings. NCEI's online forecast archive is sparse; an analysis-only holding
does not supply future lateral boundaries. GDEX has three-hourly leads through
240 hours and twelve-hourly leads thereafter through 384.

Use the normal GFS configuration with its historical cycle and boundary
cadence. `gpuwm go CONFIG --readiness --whole-cycle` asks for every object in
that window. A dated archive `--dry-run` performs the same check before planning.
Missing objects return waiting (75); an unreachable host stays unknown rather
than being counted absent. `--no-probe` is explicitly a schedule-only answer.

The default ladder selects a dated archive for old cycles. Pin `--transport
gdex`, `ncei-0p5`, or `ncei-1p0` when a specific product is required. Pinning an
archive selects full-file acquisition automatically. Rust downloads and
validates the GRIB envelopes. The Rust ingest bridge checks actual geometry,
required pressure/surface fields, times and scan order. A manifest records the
served URL, product, terms, digests and message counts. Local series filenames
are stable identifiers; the manifest's archive product and actual grid describe
resolution.

GDEX attribution: National Centers for Environmental Prediction, NOAA,
* NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive *,
Research Data Archive / Geoscience Data Exchange, DOI
[10.5065/D65D8PWK](https://doi.org/10.5065/D65D8PWK).
The [dataset terms](https://gdex.ucar.edu/datasets/d084001/) identify
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
NCEI metadata and AWS registry links are retained beside their layout rows.

An archive admission proves acquisition capability. A real forecast and visual
inspection are required before claiming an end-to-end case result.
