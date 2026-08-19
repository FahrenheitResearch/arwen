# Sources at the domain wizard

`gpuwm domain` plans a domain FOR a source. The source decides three
things about the file it emits: the boundary cadence written into the
companion `namelist.wps`, whether the domain has to sit inside a regional
grid, and how far into a forecast the window may reach. All three are
declared in that source's row in the source registry
(`gpuwm/source_adapters.py`), so the wizard reads them rather than
carrying its own table -- and a source added to the registry reaches this
door with no change to the wizard.

List the whole registry with `gpuwm prep --list-sources`, or one row with
`gpuwm prep --show-source <id>`.

## What the wizard plans for today

`--source` takes any registered id or any of its aliases.

| source | aliases | boundary cadence | forecast horizon | native grid |
|---|---|---|---|---|
| `hrrr` | -- | 1 h | f048 | 1799x1059 Lambert at 3 km (CONUS) |
| `hrrr-prs` | `hrrr-pressure`, `hrrr-wrfprs` | 1 h | f048 | 1799x1059 Lambert at 3 km (CONUS) |
| `gem-gdps` | `gem`, `gdps`, `gem-global` | 3 h | f240 | global |
| `icon-eu` | `dwd-icon-eu`, `icon-eu-regular` | 1 h | f120 | lat 29.5..70.5, lon -23.5..62.5 |
| `gfs` | `gfs-0p25`, `gfs-0.25` | 3 h | f384 | global |
| `gdas` | `gdas-0p25`, `gdas-0.25` | 1 h | f009 | global |
| `gefs` | `gefs-ensemble` | 3 h | f384 | global |
| `aigfs` | `ai-gfs` | 6 h | f384 | global |
| `aigefs` | `ai-gefs` | 6 h | f384 | global |
| `ecmwf-open-data` | `ecmwf`, `ifs` | 3 h | f360 | global |
| `aifs` | `aifs-v2`, `aifs-single` | 6 h | f360 | global |
| `rap` | `rap-awip32` | 1 h | f051 | AWIPS 221, 349x277 Lambert at 32 km (North America) |
| `rrfs` | `rrfs-ops` | 1 h | f084 | 1799x1059 Lambert at 3 km (CONUS; HRRR's grid, measured identical) |
| `era5` | -- | 6 h | analysis only | global |
| `20crv3` | `20cr`, `twentycrv3`, `20crv3-member` | 3 h | analysis only | global |
| `20crv3-cf` | `20crv3-netcdf`, `20cr-netcdf`, `20cr-cf` | 3 h | analysis only | global |

A registered source that is NOT in this list refuses by name and says why:
either its row has no runnable initialization route yet, or its boundary
cadence is not a property of the source at all (`--source mapped` reads it
from the mapping document a caller supplies).

## Regional sources are bounded at plan time

Where a row declares a native grid, the wizard bounds the fitted ladder by
it and refuses a domain that cannot be reached -- before anything is
downloaded. The refusal names the offending point, where that point lands
in the source's OWN index space, and the window the source covers:

```
$ gpuwm domain --point=38.5,-97.5 --card 16gb --root-dx 3 --hours 6 \
      --source icon-eu --cycle 2026-08-17T00 --out kansas.toml
gpuwm domain: ladder 3 cannot be forced by icon-eu even at the minimum
layout (60x48 root): the 60x48 root's point at lat/lon (37.8377, -98.5408)
maps to source index x=-1200.652 y=133.404, and the source covers
x=0..1376 (lon -23.5..62.5) y=0..656 (lat 29.5..70.5); 2989 of 2989 root
mass points are outside it.  icon-eu's grid is centred at (50.00, 19.50)
-- move --point inside that grid, shrink the ladder, or choose a source
whose coverage includes this domain
```

(Verbatim, `gpuwm 2.5.0`, exit 2, nothing written.)

That is the same answer the preparation stage gives on real bytes, in the
same coordinates, arrived at from the registry instead of from a download.
The 2026-08-17 model battery paid a full ICON-EU acquisition and 73
seconds of preprocessing to learn it, and read it as a traceback.

Where the domain fits but the *margined* fetch box overruns the grid, the
box is clamped into coverage and the clamp is reported as an advisory --
`--area` is a coverage check for a regional source, not a crop.

HRRR keeps its own, stricter check: its certified route needs real source
cells outside the target on every side for the interpolation stencil, the
surface-fallback halo and the donor search, so it refuses domains that the
bare grid rectangle would accept.

## Which sources `gpuwm fetch` downloads

All fourteen in the table above. Ten of them are rows in the packaged
acquisition-route document
(`gpuwm/authorities/rw-wps-fetch-routes.v1.json`), read by the one engine
in `gpuwm/fetch_routes.py`; four keep the hand-written transports that
predate it. `docs/public/DATA.md` publishes the working command for each.

| how the bytes arrive | sources |
|---|---|
| table route (whole published objects, in parallel) | `hrrr-prs`, `rap`, `rrfs`, `gefs`, `aigfs`, `aigefs`, `ecmwf-open-data`, `aifs`, `icon-eu`, `gem-gdps` |
| hand-written transport (publisher-side subsetting) | `gfs`, `gdas` (NOMADS grib-filter or the S3 archive), `hrrr` (`.idx` byte ranges, live-cycle wait), `era5` (a Copernicus retrieval you run) |

`gpuwm domain` emits the `[fetch]` table and the runnable step 1 for every
one of them, because the wizard asks the fetch module the question rather
than carrying a list:

```
next:
  1. gpuwm fetch --source rap --cycle 2026-08-17T00 --hours 6 --out .../data/area_38p50n_97p50w
```

**`--area` is only for the four.** A table route takes whole published
objects -- there is no subsetting service in front of them -- so the
emitted `[fetch]` table carries no crop key for one, and passing `--area`
to it refuses by name. The crop for those sources happens at `gpuwm prep`,
where the namelist geometry is the crop. The wizard's coverage advisories
still print: they are a statement about whether the source reaches your
domain, which is a different question from how many bytes come down.

## Sources with no fetch door

Three runnable rows refuse a download by name, and the refusal states the
breakage rather than reporting a gap:

| source | why there is no route | what to do instead |
|---|---|---|
| `20crv3` | the every-member GRIB2 archive is not published on an anonymously readable public endpoint; only the ensemble-MEAN NetCDF distribution is, and a member state is not a mean | stage the files, then `gpuwm prep --source 20crv3 --source-root DIR --source-manifest DIR/SHA256SUMS --source-manifest-sha256 <digest>` |
| `20crv3-cf` | the NOAA PSL NetCDF distribution is a per-year, per-variable reanalysis archive with no cycle and no forecast lead, so `--cycle`/`--hours` describe nothing in it | same `--source-root` door; the packaged profile rebuilds the missing orography and land mask with `tools/build_pressure_level_invariant_supplement.py` |
| `mapped` | the generic declarative adapter is not a product: it names no publisher, no bucket and no file grammar, so there is nothing to resolve | it *is* the door for bytes you already have -- supply the mapping document |

For these, the emitted config carries **no `[fetch]` table** -- a table
naming a source the fetch door cannot serve is refused at every later
config load, and one that quietly loaded would advertise a download
nothing can make. The file says so in its own header, and step 1 of the
printed next-steps block is the acquisition note rather than a `gpuwm
fetch` line that would refuse. Everything else in the file is complete:
geometry, levels, physics, time step, radiation cadence and the boundary
interval. A hand-staged directory of that cycle's files runs the same
preparation chain every other mapped source runs.

Fourteen further registry rows are registered but **not runnable**
(`hgefs`, `hiresw`, `href`, `hrrr-ak`, `nam`, `nbm`, `refs`, `rrfs-a`,
`rrfs-firewx`, `rrfs-public`, `rtma`, `sref`, `urma`, `wrf`). They are not
planned by `gpuwm domain` and not downloaded by `gpuwm fetch`, and both
doors say which of the four states the row is in -- `adapter_mapping_required`,
`explicit_composition_required`, `member_selection_and_mapping_required`
or `wrf_archive_mapping_required` -- rather than "invalid choice":

```
gpuwm fetch: error: argument --source: --source nam: no fetch route.
  why: the registry row is not runnable (adapter_mapping_required);
  nothing in this ArWen could read the bytes a download produced.
  see: `gpuwm sources` for what each registered source can do today.
```

Adding a download route for one of them is a row in the route document
plus the profile work its status names. It is not a new code path.

## Adding a source

Nothing in `gpuwm/domain_wizard.py` names a model. A new row reaches this
door by declaring, in `gpuwm/source_adapters.py`:

- `runnable=True` and the profile/runner the decode route uses;
- `forcing_interval_seconds` -- the source's native spacing between valid
  times. For a row with a packaged profile this is the mapping document's
  `target.boundary_interval_seconds`, and a test fails if the two ever
  disagree, for every row that has one. It reproduces the number the
  2026-08-17 battery typed into its hand-written namelists by hand;
- `max_forecast_hour` -- 0 for an analysis or reanalysis, which also turns
  off `--forecast-start-hour` with that reason named;
- `coverage=` a `RegularLatLonWindow` or a `LambertGridWindow` if the
  product is regional; nothing at all if it is global.

`tests/test_wizard_sources.py` proves the claim rather than asserting it:
it installs a synthetic registry row and drives the real CLI, checking
that the emitted `namelist.wps` carries the cadence the row declared and
that a regional variant refuses outside the window the row declared, with
no code added anywhere.
