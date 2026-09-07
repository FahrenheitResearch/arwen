# Editable forecast starter

`regional-gfs-rrtmgp.toml` is an ordinary, editable six-hour GFS experiment:
12 km square cells, Morrison microphysics, YSU/MM5/Noah/Kain-Fritsch, and
modern RTE+RRTMGP radiation. These are starting values, not permitted ranges.
The radiation choice is not a claim of universal superior forecast accuracy.
Historical verification cases remain separate and unchanged.

Copy this file somewhere convenient and edit any TOML settings first. Choose
an available GFS cycle explicitly; the date stored in the example is fixed.
For example, with `my-forecast.toml` as the edited copy:

```sh
gpuwm domain-fit my-forecast.toml --point=40,-100 --vram-gib 8 --start-time 2026-09-05T00 --out fitted.toml --write
gpuwm go fitted.toml --dry-run
gpuwm go fitted.toml
```

The first command prints the exact changes and canonical memory estimate,
then creates a new TOML, matching WPS, and `.fit.json` receipt. Omit `--write`
to preview without creating files. Existing output paths and the starter are
never overwritten. No fit command downloads data or launches a forecast.

`--point` finds the largest centered layout within the existing fitter's
device/source constraints. For a specific area use `--polygon area.geojson`
and optional `--buffer-km` instead; the entire polygon is preserved and a
layout that exceeds the budget is refused. `--card` accepts existing named
device tiers as an alternative to an exact VRAM capacity. CPU fitting does
not require a local GPU. The memory estimate is not a measured run on your
machine, and the normal preparation and runtime checks remain authoritative.

The explicit fit changes dimensions, centered child placement, reference
latitude/longitude, matching fetch geometry, and only the start/duration you
explicitly override. It preserves dx/dy, projection family and standard
parallels, rational time steps, grid/time ratios, eta levels, physics, output
variables, output clocks, and all other settings before **every** candidate
is priced. Change those settings in the editable TOML if you want different
values; fitting never lowers resolution or switches physics to squeeze in.

Relative case-data, geography and overlay declarations retain their meaning
when output goes elsewhere. A declared WPS must be readable: its nongeometry
settings are preserved in the newly fitted WPS. Original inputs and prepared
artifacts are not rewritten or relabelled. The next `go` uses the normal
fresh preparation/authority checks for the new geometry and time.

This first fitting operation supports linear parent chains numbered 1..N
and square cells. Other valid model layouts remain runnable normally;
the fit operation does not expand its scope by silently changing topology.
The fitted TOML preserves values, not the starter's comment formatting.

Omit both --card and --vram-gib to detect this machine's GPU through the existing wizard.
