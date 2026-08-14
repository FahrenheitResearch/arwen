# Observations outside North America

What works today, what is measured, and what is waiting on a decision.

Every number on this page was taken from a live endpoint or a decoded frame,
not from documentation: the survey numbers on 2026-08-12, and the polar-volume
numbers marked CLOSED below on 2026-08-14. Where a capability could not be
reached, the reason is stated as a specific gap with a location rather than as
an absence.

## Summary

| Route | Status | Quantity | Front door |
| --- | --- | --- | --- |
| European radar composite | works | composite reflectivity, dBZ | `rw_opera` |
| European per-site polar volumes | works | reflectivity + radial velocity | `rw_odim`, via `gpuwm obs radar` |
| Worldwide surface reports | works | temperature, dewpoint, wind, MSLP | `rw_asos` |
| Japanese radar | not built, provenance question open | - | none |

## 1. European radar: the composite route works end to end

`rw_opera` fetches, decodes and packs the EUMETNET OPERA composite from the
MeteoGate OGC-EDR API. It writes `gpuwm-obs.obs-grid.v1` with quantity
`composite_reflectivity` in dBZ, which is the same pack, quantity and units
`rw_mrms` writes. `gpuwm.obs.sources.OperaCompositeSource` reads it, so a
scorer picks a grader by which network covers the domain rather than by which
code path it has to take.

Measured on a live frame (`OPERA@20260812T1930@0@DBZH.h5`):

- grid 3800 x 4400 at 1 km, 16 720 000 cells, ellipsoidal LAEA over Europe
- cadence 5 minutes, frames stamped on the minute
- 161 contributing radars listed in the frame's own `/how/nodes`
- unauthenticated, CC BY 4.0, EUMETNET

The composite's node list is **wider than the per-site inventory**. It
includes the United Kingdom (16 radars), Portugal, Slovakia, Hungary, Malta
and Slovenia, none of which publish individual site feeds through the
locations endpoint. Italy, Austria, Serbia, Bulgaria and Greece are absent
from both.

### Two sentinels, and they are opposite claims

`/dataset1/data1/what` declares `nodata = -9999000` and
`undetect = -8888000`. On the measured frame:

| Class | Cells | Share |
| --- | --- | --- |
| `nodata`, no radar coverage | 8 303 379 | 49.7 % |
| `undetect`, observed and empty | 7 726 939 | 46.2 % |
| measured echo | 689 682 | 4.1 % |

`nodata` is unobserved and belongs under the validity mask. `undetect` is the
network reporting that it looked and found nothing, which is an observation,
and on any given frame it is the most common true one. Collapsing them
deletes every correct negative a skill score is built on.

`rw_opera` separates them. Decoding the measured frame over a
Germany-and-Benelux box gives **observed fraction 0.991**. Collapsing the two
would have given roughly 0.012, and every registered coverage floor would
then have rejected a perfectly good frame as missing-obs.

This is the same trap `rw_mrms` documents for `-999` versus `-99`, in a
different archive's spelling.

### The georeference has an oracle inside it, and it is used

`/where/projdef` declares `+proj=laea ... +ellps=WGS84`. The projection is
ellipsoidal and the frame states its own four corner coordinates, so the
geometry can be checked against the file rather than trusted.

Inverting the same grid on a sphere, at the upper-left corner:

| Inversion | Latitude error | Longitude error |
| --- | --- | --- |
| spherical, authalic radius | -0.0235 deg | -0.2155 deg |
| spherical, WGS84 semi-major | -0.0249 deg | -0.1400 deg |
| **ellipsoidal, WGS84** | **0.0000 deg** | **0.0000 deg** |

0.2155 deg of longitude at 67 N is about 9.4 km, or nine cells on this grid.
`rw_opera` derives its grid ellipsoidally and refuses the frame when its
derived corners miss the declared ones by more than 1e-4 deg. On the measured
frame the worst corner offset was **5.3e-14 deg**, and the check and its
margin are recorded in the pack's `corner_check` block so a reader can see
the georeference was proved without re-running anything.

The upstream decoder in `rustwx-io` inverts this projection spherically
(`opera_laea_latlon_grid`, `inverse_spherical_laea`) and maps both sentinels
to NaN. **Question 1 for Drew** is below; nothing in BowEcho was changed.

### Part of the field is extrapolated, not observed

The measured frame's `/how/comment` states that certain DBZH volumes are
produced by Lucas-Kanade advection at Meteo France
(`DBZH.meteo-france.advection.pysteps-1.5.0`). That is a statement about what
the numbers are, so `rw_opera` carries `/how` and `/dataset1/what` into the
pack's `production` block: creator, institution, licence, comment,
contributing-node count, and the accumulation interval.

A verification campaign should read it before treating the composite as
ground truth. See the grader question below.

## 2. European per-site polar volumes: both gaps closed

The collection does serve DA-grade polar scans. Frozen inventory
(`gpuwm/obs/data/radar_sites_odim.json`, `gpuwm.obs.radar_sites`):

- **136 radars** across 20 WMO blocks, plus one composite pseudo-site
- **136** publish a `scan` method, that is a polar volume
- **134** carry a radial-velocity moment (`VRAD` or `VRADH`)
- **136** now carry a measured antenna elevation; the feed publishes none and they are read from the volumes (Gap B below)

### Gap A: the decoder read composites only — CLOSED

`rustwx-io`'s ODIM entry point requires a rank-2 `/dataset1/data1/data` and a
`/where` `projdef`, and refuses anything else. A polar volume has neither: it
carries `/where` lat/lon/height and per-sweep `elangle`, `nbins`, `nrays`,
`rscale`. Every per-site scan therefore hard-refused at the projection check,
and that is still true of `rw_opera`, which decodes the composite and only
the composite.

**It is no longer the whole story.** `rw_odim` reads ODIM `PVOL` and `SCAN`
objects and writes `gpuwm-obs.radar-sweeps.v3` — the same pack `rw_nexrad
decode` writes and the same one `gpuwm.obs.sweeps.read_sweep_pack` reads. So
the join is at the top and nothing below it needed a European variant: dealias,
superob, `radar_grid`, `da.obs_radar` and the LETKF adapter assimilate a
European radial velocity through exactly the code that assimilates an American
one.

`gpuwm.obs.opera.polar_volume_support()` now returns `(True, reason)` and the
reason names the other route rather than claiming the capability for this one.
The user-facing door is `gpuwm obs radar`; see
`docs/european-radar-front-door.md`.

**Germany's split volumes are part of the closure, not a footnote.** The
Netherlands and Romania publish one file per volume; Germany publishes one
per (elevation, quantity), so a ten-elevation volume carrying `DBZH` and
`VRADH` arrives as many files -- measured on a real Boostedt volume,
**30**, because the feed serves `TH` beside them. `gpuwm obs radar volumes`
groups a
directory by the nominal time inside each file and `gpuwm obs radar pack
--dir` assembles one group. A directory holding two nominal times is refused
rather than resolved by recency, and files merge into one sweep only on the
whole cut identity plus matching per-ray azimuths — elevation alone is not
enough, since one Dutch volume carries three sweeps at 0.30 degrees with two
different Nyquist intervals.

### Gap B: no antenna elevation in the *metadata* feed — CLOSED

**All 136 sites now carry a measured antenna height.** The paragraphs below
state the gap as it stood on 2026-08-12 and they are still the reason the
table is built the way it is; what changed is that the height is now read
from the volumes rather than looked for in the metadata.

`tools/harvest_radar_heights.py` reads `/where/height` out of one whole
volume file per site — the smallest object the site publishes, read and
discarded — and `tools/freeze_radar_sites.py --heights` merges the result.
Measured 2026-08-14: 136 of 136 resolved, from 15.0 m (`dkrom`, a flat Danish
island) to 2937.0 m (`chppm`, a Swiss alpine site), median 262 m.

The merge is checked rather than trusted, because matching the wrong file to
the wrong site would put every gate of a radar at another antenna's altitude
and nothing in the output would look wrong. Every file's declared
`/where/lat` and `/where/lon` are compared against the position the table
already holds for that site; the worst disagreement over all 136 was
**4.9e-7 degrees**, about 5 cm, and the freeze refuses to merge a document
whose worst disagreement exceeds 0.01 degrees.

Corroborated by a second, independent harvest that discovered its files
through the OGC-EDR gateway rather than the object store: 56 sites in common,
**zero height disagreements**, and **zero of the 56 read the same file** — so
the agreement is across different volumes at different times rather than a
shared read of one byte stream.

A site the harvest cannot reach keeps `null` and stays refused. That rule is
unchanged and is the whole point of the null.

---

The locations endpoint publishes a 2D point. The `detail` link on each
feature points at the WMO OSCAR/Surface station search; probed 2026-08-12 for
`0-191-0-hrdeb` it answered `totalCount: 0`. So the height is neither in the
feed nor one hop from it.

**It is, however, in the data.** Staging three Romanian volumes the same day
found the antenna height at `/where/height` in every one: `robuc` 133.0 m,
`rocra` 218.5 m, `robob` 567.0 m above mean sea level. So this gap is a
consequence of freezing the table from the locations endpoint rather than
from the volumes, and it closes with a re-freeze rather than with a new data
source. See Q5. The refusal below stays in force until that happens, because
a table that says `null` is still a table that cannot be assimilated.

Beam height above ground is a function of antenna height above mean sea
level. A site assimilated with the wrong antenna height places every gate at
the wrong altitude, smoothly, with nothing that looks like a failure.

`gpuwm.obs.radar_sites.require_assimilable` therefore refuses, by name, with
the count, and a site whose height could not be measured keeps `null` rather than a placeholder.
This is the same rule the North-American antenna table is held to. A
placeholder does not make a site usable; it makes an unusable site look
usable.

### Dual-PRF, when the decoder arrives

European radars run dual-PRF unfolding, which extends the unambiguous
velocity beyond the single-PRF Nyquist and leaves aliasing artifacts at
multiples of the *individual* PRF Nyquists rather than at the extended one.
The existing sweep contract already carries what that needs: `Sweep`
(`gpuwm/obs/sweeps.py`) holds `nyquist_velocity_ms` plus
`nyquist_radials_disagree`, which records a cut whose radials did not all
report the same Nyquist. That flag is exactly the dual-PRF signature.

No dual-PRF handling is added here, deliberately. The region-global dealias
engine on `lane/dealias-region` is the owner of this class of problem and
handles it better than the legacy VAD path; duplicating it against a decoder
that does not exist yet would be building two things wrong at once. What this
lane owes that lane is the ODIM field list, and it is recorded above: `NI`
per dataset, `highprf`/`lowprf` per sweep.

## 3. Worldwide surface reports: already reachable, now resolvable

`rw_asos` was never US-wired. Its archive root, `--networks` list and
`--bbox` are all arguments, and the network id is concatenated into a URL
template rather than branched on. The missing piece was the step before it:
given a domain, *which* networks to name. Naming the wrong ones is not an
error anyone sees, because the fetch succeeds and the station table simply
comes back short.

`gpuwm.obs.surface_networks.networks_for_bbox` answers it from a frozen
table (`gpuwm/obs/data/surface_networks.json`,
`tools/freeze_surface_networks.py`):

- **266 networks**, of which **199 are outside the United States**
- **7521 stations**, extents measured over the stations themselves

Two traps were found while building it and both are pinned by tests:

- **`AL_ASOS` is Alabama and `AL__ASOS` is Albania.** One underscore, 8000
  km. A resolver that normalized the separator would answer a Balkan domain
  with Gulf Coast stations and never say so.
- **A dateline network cannot be a `min`/`max` box.** New Zealand reports
  stations near +178 and near -176, so `min`/`max` yields `[-176, 178]`, 354
  of the 360 degrees, and the network is then offered for domains in the
  South Atlantic and the Indian Ocean. Extents are stored as the shortest
  containing longitude interval, running past +180 when they must.

The extent is a **candidate screen, not the filter**. Measured against the
live station lists of twelve networks on three continents, every archive
extent contained its stations, with the archive padding each edge by 0.1 deg.
The screen errs toward offering a network whose stations turn out to lie
elsewhere and never toward dropping one, which is the direction that matters:
an extra network costs one metadata request, a dropped one costs
observations silently. The authoritative filter remains the station-level
`--bbox` that `rw_asos stations` already applies.

## 4. Japanese radar: assessed, not built

Raw data and decoded receipts exist on this box from an earlier session
(`Documents/Codex/2026-05-31/https-pawr-nict-go-jp-jmadata/work/`): 20 JMA
sites, 655 sweeps, 512 radials, 500 m gates, elevations 0.7 to 25.0 deg,
including `Pvr` radial velocity. That is a genuine volumetric dataset and it
is better suited to DA than the European composite.

There is **no decoder for it in the Rust stack on this box**. A bounded
search over the whole Codex tree for `JMAGPV` in `*.rs` returned zero hits;
the volumes were served through a local bridge masquerading as NEXRAD.

Nothing was built on it, per the directive. The provenance question is below.

Japanese *satellite* is a different matter and is real: `rw_sat` carries
Himawari AHI against the public NOAA S3 mirror, with HSD header parsing.

## 5. What the staged case measured

A convective case over southern Romania was staged end to end on CPU on
2026-08-12: 30 composite frames, 3 polar volumes, 8 surface stations. The
bundle it produced is a development record and is not published; the five
things it measured are below, because those are what change how this route
must be used.

### The archive keeps 24 hours

A window at T-24h resolves; T-25h refuses. The bucket is named
`openradar-24h` and it means it. There is no historical European radar
through this route, so a case cannot be reconstructed after a day and any
verification campaign needs a rolling capture standing up *before* the cases
it wants to score. One frame every five minutes is about 470 MB/day for the
whole continent.

### The archive window is offset by +10 minutes

Requesting 19:30 to 20:00 returns frames from 19:40. Confirmed at four window
widths and against the raw endpoint, so it is upstream of `rw_opera` rather
than a client bug. A caller that asked for exactly its assimilation window
would silently lose the first two frames of it. Staging padded the start by
15 minutes; that pad belongs inside the fetch path, not in each caller.

### European radar is not uniformly C-band

The three Romanian radars are S-band: 10.187, 10.221 and 10.302 cm, on
LEONARDO METEOR 1700SDP10 systems. Attenuation and calibration handling
cannot be chosen per continent. `/how/wavelength` is in every file, so it can
be read per site.

### Dual-PRF schemes differ between sites in one country

`robuc` runs 652/434.67 Hz for an extended Nyquist of 33.32 m/s; `robob` and
`rocra` run 600/450 Hz for 45.99 m/s. A national constant would be wrong at
one of the three. Measured velocities reach the extended limit (`robuc`
sweep 1 spans -32.78 to +32.78 m/s against NI 33.32), so this is genuinely
folded data and not a pre-cleaned product.

### The surface route drops pressure over Europe

`mslp` was 0 of 83 rows in the case window while `alti` was 83 of 83.
European METAR reports QNH, not MSLP, and the decoder reads `mslp` only. One
of the four declared surface variables is structurally empty over Europe and
nothing in the record says so, while the pressure observation sits unread in
every row. Station elevations are already in the frozen table, so the
conversion is arithmetic.

Smaller: a station frozen into the table that returns no rows disappears from
the decoded record without entering either drop list. The count went 9 to 8
with no explanation in the receipt.

## Questions for Drew

**Q1. The upstream OPERA decoder disagrees with this front door, twice.**
`rustwx-io`'s `extract_eumetnet_opera_dbzh_from_odim_h5` maps both sentinels
to NaN and inverts the LAEA grid on a sphere. Measured, that is 46.2 % of a
frame's cells reclassified from observed-empty to unobserved, and up to 9.4
km of position error at the northern corners. `rw_opera` does neither, and
proves its geometry against the frame's own declared corners on every decode.
Nothing in BowEcho was touched. Do you want the two findings carried upstream
into rusty-weather, or should the front door keep its own reading and the
divergence be documented the way `docs/` records the other deliberate ones?

**Q2. The European verification grader.** There is no MRMS in Europe. The
OPERA composite is the natural counterpart and the route now works, but its
own metadata says part of the field is advection-extrapolated rather than
observed. The US ruling is that an own-built composite is never the grader;
OPERA is a third party's composite, which is a different thing. Options: (a)
OPERA composite as the grader, with the advection caveat recorded per frame,
which is what the pack already carries; (b) obs-space verification against
the surface network only, which is now worldwide; (c) both, with (b) as the
tiebreak where the advection note fires. Recommendation: **(a) with (b)
alongside**, because the composite is a genuine third-party product and the
caveat is per-frame and machine-readable rather than a blanket disclaimer.

**Q3. The European background.** HRRR does not cover Europe and the
HRRR-permanent ruling was CONUS-scoped. GFS is the in-house global route and
ERA5 is the hindcast route, both already built. There is a third option
already in the stack and not yet used: `rw-fetch` carries ECMWF IFS open data
at 0.25 deg (`data.ecmwf.int/forecasts/.../ifs/0p25/oper/`), which is a
better European background than GFS and is free. Recommendation: **IFS open
data for European real-time, ERA5 for European hindcasts**, with GFS as the
fallback. This needs your ruling because it adds a background source rather
than reusing one.

**Q4. Japanese feed go/no-go.** The data is DA-grade and the provenance is
what you flagged: rehosted from a university, no decoder in the Rust stack,
and the earlier session reached it through a local bridge rather than a
documented endpoint. Building an ODIM/GRIB2 polar decoder for it is
substantial work that would also unblock the European per-site route if the
formats overlap. Recommendation: **hold**, and revisit after Q1, because the
polar decoder is the shared long pole and Europe is the larger prize.

**Q5. Where do the antenna elevations come from? ANSWERED 2026-08-12, and it
is easier than feared.** The height is in each site's own ODIM volume, at
`/where/height`, in metres above mean sea level. Measured on three Romanian
radars: `robuc` 133.0 m, `rocra` 218.5 m, `robob` 567.0 m, each consistent
with the nearby airport elevation plus a tower. No national registry needs
scraping. Re-freezing `radar_sites_odim.json` from the volumes rather than
from the locations endpoint costs 136 fetches of about 1 MB, once, and picks
up wavelength, beamwidth and the dual-PRF pair in the same pass. The refusal
in `require_assimilable` should stay until that re-freeze happens. The
remaining question is only whether you want it done now or folded into the
polar-decoder work, since that decoder has to read these files anyway.
