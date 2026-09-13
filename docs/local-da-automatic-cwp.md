# Automatic regional cloud water path

New local DA reviews discover satellite input automatically when no explicit
satellite grids are supplied. The existing native observation bundle supplies
the decoder, NetCDF reader/writer and gridding library; no additional Python
decoder or scientific rendering path is introduced. Existing saved reviews
retain their recorded routes, and explicit grids remain bound to their files.

`gpuwm.obs.goes_window.SOURCES` is the source table. It describes the native
satellite/sector/scan-mode combinations, not operational dates or an assumed
geographic assignment. Native listings determine which products exist for the
requested time. Native geolocation and the target grid determine coverage.
Additional sources require their own compatible product and observation
operator contract; brightness temperatures are not cloud water path.

Each source contributes its newest usable scan, with older fallback after
empty or failed scans. The complete COD/CPS/ACTP trio is required. ACHA cloud
top is optional, fetched separately, and joined through the existing native
nearest method with an exact sibling pack identity. Missing ACHA uses the
existing recorded 3000 m above-ground placement assumption. No interpolation
between scan times occurs. A valid provider object can contain more area than
the target domain; listing sizes and retained source byte counts expose this
transport cost. The current native path downloads complete sector objects.

The complete measurement interval must end by analysis time, and its optical
end must be at most 1800 seconds old. This age is an explicit provisional
observation policy, not a latency limit on the domain or cadence. The source
publication cutoff is discovery start; the actual local receipt cutoff is
recorded after acquisition. Retrospective downloads are not described as
locally available at analysis time. A late unconsumed scan may enter a later
window if its measurement interval still fits. Mode/revision changes cannot
turn a consumed physical scan into a new observation.

An absent provider publication timestamp is recorded as unknown with a
warning. The actual local receipt still establishes when the data arrived;
the report does not claim it existed before discovery started. Known
publication times after the cutoff remain excluded from that window.

Overlapping source grids retain one existing observation per exact target
column, ordered by newer optical end and then stable source identity. Native
nearest-plan tie selection chooses among co-located columns. An explicit
column-index check prevents borrowing a neighbor. Values and errors are
selected without averaging. Source ownership is retained as column runs;
clear-sky zeros survive, while missing pixels remain missing. The existing
regional adapter places each column observation at one level, rather than
repeating it at every level.

## Product, quality and uncertainty

The native producer consumes COD (dimensionless), CPS (micrometers), and ACTP
phase to derive CWP in g m-2. COD/CPS source metadata describes both daytime
and nighttime retrieval algorithms. This caller adds no time-of-day screen.
It preserves the producer's actual QC: COD/CPS condemn bits 8, 16 and 64
(snow/sea ice, twilight and glint), and accepted ACTP quality/phase values.
Missing, invalid or unclassified retrievals are not substituted with zeros.
The pack coefficients and rederived CWP consistency check remain authoritative.
The ice coefficient retains its documented spherical-particle approximation.
Native ACHA height is meters above the geoid, used with the existing recorded
geoid-to-mean-sea-level assumption in the target grid.

The COD/CPS CF quality masks identify **256 as thick cloud and 512 as thin
cloud**. Earlier inflation constants reversed these labels. Numeric-input
controls now check the distinct factors independently of named constants.

Automatic acquisition states these initial, uncalibrated standard deviations:

| Retrieval | Standard deviation, g m-2 |
| --- | --- |
| Clear | 50 |
| Liquid | max(0.5 times CWP, 50) |
| Ice/mixed | max(1.0 times CWP, 100) |

Thin and thick flags multiply the stated error by 1.5 and 2.0 respectively;
both multiply together. The existing superob owner combines contributing
pixels and records its quality decisions. The 50 g m-2 floor takes a scale
from the low-water-path experiment in
[the 2019 water-path assimilation study](https://gmd.copernicus.org/articles/12/3939/2019/).
It is not an ABI-specific calibration; the phase coefficients and inflation
factors remain provisional. Their basis travels with each product. Unmeasured
uncertainty does not disable an otherwise valid observation or chosen domain.

## Failure, recovery and counts

Each attempt gets its own output generation. Source keys, intervals,
publication times, byte sizes, SHA-256 digests, packs, source decisions and
the final grid are retained. A completed window freezes that asset roster;
resume verifies its bytes and never silently refetches replacements. Empty
or failed optional feeds remain visible and contribute no observation. An
incompatible actual condensate operator skips automatic CWP; explicit inputs
report the missing species before the first forecast member starts.

Source-QC columns and downloaded bytes are acquisition measures. The execution
document's `observation_usage` instead reads the analysis owner's innovation
masks after QC and thinning from completed cycle publications. It reports
accepted observations per batch, an explicit zero for forecast-only cycles,
and unknown for old reports lacking counts. Resume preserves these completed
counts. None is evidence of a beneficial forecast change; that requires a
real analysis/forecast comparison.
