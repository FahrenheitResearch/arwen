# Automatic observed storm tracks

`cyclone-setup --storm NAME_OR_ID --cycle YYYY-MM-DDTHH --hours 24 --out storm.toml`
acquires observed guidance when `--tracks` is omitted. `--search-area=S,W,N,E`
with no storm defaults to `--storm auto` and searches the global IBTrACS archive.
The cycle selects the event, so repeated storm names do not mix years. Area
selection uses the existing pressure ranking and explicit ambiguity refusals.
A selected observed trajectory supplies setup placement and fitting guidance.
It does not steer the live moving nest or replace atmospheric forcing.

Use `--track-source auto|nhc|hurdat2|ibtracs|jtwc` and optionally
`--track-cache DIR`. Automatic named selection tries the NHC ATCF index and
best-track archive, then the global archive if no NHC event matches. JTWC uses
the NOAA NCEI IBTrACS copy of records labelled `USA_AGENCY=jtwc_*`, with
that agency's `USA_LAT`/`USA_LON` positions; it does not
depend on the JTWC web server being reachable. Source URLs, government data
terms, the IBTrACS citation, download SHA-256, byte count, cache hits and wall
time are retained in the normalized track and cyclone setup receipt. Raw
source responses are cached for one hour. The global CSV is about 333 MB.

The packaged Rust `rw_fetch cyclone-tracks --request request.json` fetches,
decodes gzip, parses ATCF BEST, HURDAT2 and IBTrACS CSV, and normalizes units.
IBTrACS uses its stable SID while an ATCF identifier appears after genesis.
Only USA wind/pressure columns supply intensity; no averaging-period mixture
is substituted when those columns are missing. Observed winds are agency
one-minute sustained values, distinct from model instantaneous 10 m maxima.
ATCF repeated radius records are deduplicated; conflicting fixes are refused.

`tc-tracks --input COMPLETED_OUTPUT --best-track auto --storm-id ID --render
--out TRACK_OUTPUT` acquires event-matched observations. The existing tracker
backends extract trajectories from completed output. The Rust
`rw_fetch cyclone-score` associates the nearest candidate within the declared
500 km initial association radius, then keeps that track identity across all
leads. Missing model leads count as misses. Observation gaps over six hours
are unscored. Shorter gaps use great-circle position interpolation, including
at the dateline. Receipts retain per-lead position, pressure and wind errors,
mean and RMS position error, misses and unobserved valid times.

Native hex history is admitted explicitly with `--mesh STATIC.nc`, optional
`--hex-init INITIAL.nc`, and `--hex-window global` (or an existing converter
window). Rust `rw_mpas_convert --remap product` verifies the declared mesh,
reads native history and matching diagnostic streams, then `rw_mlexport` and
`rw_tc_fields` supply the existing tracker field contract. Every mesh,
history, initial-file digest and conversion receipt is retained as ancestry.
Required pressure, geopotential or wind fields must exist; missing physical
fields are not synthesized by this bridge. A native file-format regression
is separate from event forecast qualification.

Source metadata is `gpuwm/authorities/cyclone-track-sources.v1.json`.
The NWS terms declare NHC information public domain, with attribution,
no endorsement and no presentation of modified data as official material.
NCEI publishes IBTrACS freely; its dataset use constraints request the v4.01
citation (DOI 10.25921/82ty-9e16) and disclaim accuracy and application
liability. Those constraints also accompany the JTWC mirror route.
