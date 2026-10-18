# European radar, from a file to an analysis increment

`gpuwm obs radar` is the door. Everything on this page runs from a
`pip install gpuwm` — no source checkout, no demo script, no flag whose name
begins with "experimental".

This is the polar-volume route: per-site scans carrying **radial velocity**,
which is what makes a European radar assimilable. The 2-D OPERA composite is a
different product on a different binary (`rw_opera`,
`gpuwm.obs.opera`); it grades a forecast and cannot be assimilated, because a
composite has no elevation geometry and no velocity. `docs/international-obs.md`
covers that one.

## The five-minute version

```sh
gpuwm obs radar doctor                      # is the decoder here, and is it the right build
gpuwm obs radar volumes --dir ./scans       # what volumes are in this directory
gpuwm obs radar pack --dir ./scans --out v.pack --quantities DBZH,VRADH \
                     --max-elevation-deg 25 --max-range-km 200
gpuwm obs radar nyquist --file ./scans/one.h5      # can its velocity be unfolded
gpuwm obs radar grid --pack v.pack --grid-wrfout wrfout_d01_... --out obs.nc \
                     --max-range-km 200 --max-elevation-deg 25 \
                     --clear-air-from-censor --dealias
```

`grid` writes the observation file the LETKF adapter reads. From there the
path is `gpuwm.da.obs_radar` and `assimilate_radar_grid`, which are the same
functions a NEXRAD volume goes through — there is no European branch below the
pack, and that is the whole point of the design.

## Finding volumes: live and archive listings

`gpuwm obs radar-feeds` (binary `rw_radar_feeds`) turns a national feed into
dated URL rows. The catalog adapters are BowEcho's: DWD, DMI, FMI, SMHI,
GeoSphere, SHMU, CHMI, ANM, ARPA Piemonte and Lombardia, KAIA, and EUMETNET
ORD for every other OPERA country (the Netherlands' `nlhrw` and `nldhl` among
them), plus NCI (Australia) and JMA. On top of them it adds the dated archive
lookup two upstreams keep and the live adapters did not use: FMI's bucket back
to 2007 and DMI's STAC, which holds about five to six months.

```sh
gpuwm obs radar-feeds providers
gpuwm obs radar-feeds sites --provider dwd
gpuwm obs radar-feeds latest --provider ord --site nlhrw
gpuwm obs radar-feeds window --provider dwd --site ess \
      --start 2026-10-16T07:30Z --end 2026-10-16T08:15Z
gpuwm obs radar-feeds window --provider fmi --site fivan \
      --start 2010-06-01T00:00Z --end 2010-06-01T00:20Z
```

`window` answers from the archive when the feed has one and from its rolling
recent window otherwise, and says which in `route`. Each frame carries a
`stamp` read from the feed's own file names (every national spelling of a
volume time is an 8-digit date and an HHMM); a split frame (DWD's per-sweep
files, `merge: true`) is stamped by its earliest sweep, the time the volume
began. A frame whose names carry no time is refused by name rather than
placed in a window by guess. A DWD frame's parts are its per-sweep files;
`pack --dir` assembles them (DWD names them `...-hd5`, which the directory
survey reads).

## Getting the decoder

`gpuwm obs radar doctor` answers two questions at once: is `rw_odim` on this
box, and is it the build whose record contract this gpuwm was written against.
A rebuilt-but-unchanged binary passes; one whose records changed shape does
not, and the message says to rebuild rather than leaving a mismatch to surface
as a confusing parse error three commands later.

If it is missing, `gpuwm fetch-bridges` stages it — `rw_odim` travels in the
bridge bundle beside `rw_nexrad`, for the same reason `rw_nexrad` does: the
alternative is telling every user to install a Rust toolchain, which is the one
prerequisite this project otherwise never imposes.

The composite route's binary does **not**. `rw_opera`, and its three siblings
`rw_mrms`, `rw_stage4` and `rw_asos`, are in no bundle: none of them embeds the
`GPUWM_BRIDGE_SOURCE_REV` stamp a release cut proves a staged bridge with, so
they cannot be bundled until `crates/rw-obs` gains a `build.rs`. Those four
still need a source checkout and a toolchain.

## Two shapes of volume, and why the door has two flags for them

National feeds do not agree on what a file is.

| feed | one volume is | ODIM `/what/object` | use |
| --- | --- | --- | --- |
| Netherlands, Romania | one file | `PVOL` | `pack --file` |
| Germany | one file per (elevation, quantity) | `SCAN` | `pack --dir` |

A German ten-elevation volume arrives as many files that mean nothing apart —
measured on the live feed, **30**, because it serves `TH` beside `DBZH` and
`VRADH`. `pack --dir` assembles them, and
`volumes --dir` is what you run first to see what a directory actually holds.

**The assembly refuses two things rather than repairing them.** A directory
holding more than one nominal time needs `--stamp` to say which volume is
wanted; taking the newest would put a volume nobody asked for behind a record
that looks completely ordinary. And two files merge into one sweep only when
they agree on the entire cut identity — elevation, ray and bin counts, range
scale, range start, start and end times — *and* their per-ray azimuths match.
Elevation alone is not enough: a Dutch volume carries three sweeps at 0.30
degrees with two different Nyquist intervals, and separate reflectivity and
Doppler passes at one angle are ordinary radar practice. Files that do not
match become two sweeps, which is the accurate reading, and the pack records the
merge ratio so the reading is visible rather than assumed.

An assembled volume has no bytes of its own, so its `source_sha256` is a
**manifest** digest — the SHA-256 of the sorted `"<sha256>  <name>"` lines of
its members — and the record's `assembled` block carries the members so a
reader can still name the actual bytes.

## Antenna altitude

Beam height above ground is a function of antenna height above mean sea level,
so a site assimilated with the wrong antenna height places every gate at the
wrong altitude, smoothly, with nothing that looks like a failure.

`gpuwm obs radar sites --require-assimilable` runs that check and prints the
refusal in full. Without the flag the `assimilable` field is `null`, not
`true`: a check that did not run has no verdict, and printing one would be a
claim nobody made.

The frozen table is a fallback. Every ODIM volume carries `/where/height`, so a
pack built by `pack` answers the antenna height from the file and never needs
the table at all.

## Clear air is an observation, and ODIM has three states for "no echo"

`grid --clear-air-from-censor` builds clear-air zeroes from the decoder's own
per-gate reason codes rather than only from finite gates below the floor. On an
ODIM pack the admitted code is `undetect` — the processor reporting that it
looked and found nothing, which is the most common true observation on any
scan and every correct negative a skill score is built on.

Two ODIM states are **never** admitted, in any configuration:

- `nodata` — the radar reporting it did not look. Unobserved, not empty.
- `sentinel_ambiguous` — a file that declared `nodata` and `undetect` as the
  same raw value, so the gate may be either. Finnish `VRADH` does exactly
  this: 721,898 gates in one measured volume, 76 % of a single sweep. Reading
  them as clear air would assimilate "no echo" into cells that may hold one.

Both are counted, and the counts are in the written provenance, so the refusal
is visible rather than merely asserted here. The written `clear_air_source` is
`undetect_and_finite_below_floor` for an ODIM pack and
`below_threshold_and_finite_below_floor` for a NEXRAD one — the same admission
rule, different instruments making the claim, and a consumer is entitled to
know which it is holding.

## Dual-PRF velocity errors are removed before anything else

Most European C-band radars extend their velocity interval with two pulse
repetition frequencies on alternating rays (DWD and KNMI: 800 and 600 Hz).
When noise makes the combination pick the wrong fold count, a gate is wrong by
an even multiple of its own ray's single-PRF interval: isolated gates of the
opposite sign inside a smooth wind, which a dealiaser either spreads or
unfolds around wrongly. `pack` removes them first, on by default
(`rw_odim::dualprf`; `gpuwm obs odim pack ... --dual-prf off` is the knob):

- the reference is the circular mean of the gate's measured neighbours (two
  rays and two bins either side), corrected by the median of their
  deviations, and it exists only where the neighbours agree with each other
  (median absolute deviation at most half the low-PRF interval), so noise is
  never "corrected" into a wind;
- a gate more than the low-PRF interval from its reference is moved by
  `k * 2 va` (`k` = +-1, +-2), using the ray's own PRF when the file records
  per-ray PRFs (DWD's `/datasetN/how/prf`) and both otherwise, only when that
  lands it within half the low-PRF interval of the reference; the result is
  folded back into `[-NI, NI)`;
- the two intervals come from the declared `NI` and the PRF pair alone.

The pack record's `dual_prf` block counts what it did. Measured on FMI's
Ikaalinen volume of 2010-08-08 15:00Z (the archive route), 1,072 of 104,667
velocity gates were moved; in the 7 and 11 degree cuts they are the
opposite-sign speckle inside the outbound and inbound lobes (looked at,
raw beside corrected). On a live DWD frame
(per-ray PRFs read) on a dry morning, 0 to 6 gates on the third sweep
of each of the 17 sites.

## Dealiasing

`grid --dealias` unfolds radial velocity per sweep before gridding instead of
masking every gate that might be folded. It needs scipy and says so.

Nothing about it is European, and the proof is that the result sorts itself by
Nyquist exactly as the physics demands. Measured on the real Den Helder volume
of 2026-08-14 13:00Z, 15 cuts, 22,656 gates unfolded against 428,517 left
unchanged:

| Nyquist | cuts | gates unfolded |
| ---: | ---: | ---: |
| 6.0 m/s | 2 | 8,051 and 14,554 |
| 32.0 m/s | 5 | 3, 4, 1, 9, 34 |
| 48.0 m/s | 2 | 0 |
| 80.0 m/s | 6 | 0 |

Velocity folds constantly at ±6 m/s and almost nothing in this scene exceeds
±80 m/s, so that is the shape the answer has to have. A dealiaser that was
silently not running would give zeros everywhere; one running wrongly would
not sort itself this way. The account also closes on itself -- the record
carries `accounting_balances: true` over 767,542 gates offered.

`nyquist --file` is the handoff, and it reports provenance as well as value:
`Declared` (the file stated `NI`), `DerivedSinglePrf`, or `Unavailable`. A
sweep whose interval could not be established is refused by name rather than
unfolded against a guess, because a wrong ceiling does not fail loudly — it
just produces wind.

---

## Command reference

Every option below exists in the parser, and every option in the parser is
below. `tests/test_obs_cli_frontdoor.py` asserts that in both directions
against this section, so a flag added to one and not the other fails the
suite. `-h` / `--help` are argparse's own and are not listed.

### gpuwm obs radar doctor

Takes no options.

### gpuwm obs radar volumes

| option | meaning |
| --- | --- |
| `--dir` | directory of ODIM `.h5` files, not searched recursively |

### gpuwm obs radar pack

| option | meaning |
| --- | --- |
| `--file` | one whole-volume file: ODIM `PVOL`, JMA GRIB2 tar, IRIS RAW, Universal Format, CfRadial 1 or DORADE, gzipped or not (read from the file's bytes); repeat it for one station split over files (JMA's N5 and N6 tars); exclusive with `--dir` |
| `--site` | the station to take from a file holding several (a JMA tar holds the whole network; `stations` lists them) |
| `--dir` | directory of single-sweep files (`SCAN`) to assemble; exclusive with `--file` |
| `--stamp` | which volume in `--dir`, in the spelling `volumes` reports; required only when the directory holds more than one |
| `--out` | sweep pack to write |
| `--quantities` | ODIM quantity names to carry, comma separated; omitting it carries all of them |
| `--max-elevation-deg` | drop cuts above this elevation |
| `--max-range-km` | trim gates beyond this range |

### gpuwm obs radar nyquist

| option | meaning |
| --- | --- |
| `--file` | one radar file; ODIM geometry is read without payload, other formats are decoded |
| `--site` | the station, for a multi-station file |

### gpuwm obs radar stations

| option | meaning |
| --- | --- |
| `--file` | a multi-station file (a JMA tar); prints each station's id, number and position from headers alone |

### gpuwm obs radar fetch

Fetches one site's volume from an open national archive, a row of
`gpuwm/obs/data/radar_archives_world.json` (`jma-nict`, `bom-nci`,
`ideam-s3`, `kma-apihub`), and with `--out` packs it.

| option | meaning |
| --- | --- |
| `--archive` | the archive id |
| `--site` | the site in that archive's naming (the table's `site_ids` says what that is) |
| `--time` | valid time, UTC, e.g. `2019-10-12T09:00Z`; the nearest volume within tolerance is taken |
| `--out-dir` | where the fetched files go |
| `--tolerance-s` | how far from `--time` a volume may be; default about half the archive's cadence |
| `--out` | also decode the fetched files into this sweep pack |
| `--quantities` | with `--out`: quantities to carry |
| `--max-range-km` | with `--out`: trim gates beyond this range |

### gpuwm obs radar sites

| option | meaning |
| --- | --- |
| `--bbox` | select sites inside `west,south,east,north` |
| `--site` | select one site by its table id |
| `--require-assimilable` | run the assimilability check and report its refusal |
| `--no-velocity` | do not require a radial-velocity moment in that check |

### gpuwm obs radar grid

| option | meaning |
| --- | --- |
| `--pack` | the sweep pack, from `pack` or from `rw_nexrad` |
| `--grid-wrfout` | the wrfout whose georeference the observations are gridded onto |
| `--out` | observation file to write |
| `--max-range-km` | the range authority; required, never defaulted |
| `--max-elevation-deg` | the elevation ceiling; required, never defaulted |
| `--clear-air-from-censor` | admit the decoder's own clear-air gate code |
| `--dealias` | unfold velocity per sweep before gridding; needs scipy |
| `--overwrite` | replace an existing `--out` |
