# Task modes and plot selections

Start with the weather question you want to explore, then choose the data, place,
time and model setup. Task modes organize those choices and select useful plots.
A mode does not create a storm, change the source data or choose a new physics
suite behind the scenes. Review the resulting configuration and launch plan
before running it.

## Choose a route

Open **Forecast modes** from Home and type a weather task, such as `blizzard`,
`massive supercell` or `fantasy hurricane`. Click a result or press Enter to
explore it. Up/Down and Page Up/Page Down scroll the explanation in a small
terminal. Opening a result starts no command.

For a new forecast, choose a location or GeoJSON boundary, input source, UTC
cycle and duration. The native domain wizard creates an editable configuration.
Its source and physics defaults remain authoritative; grid and time suggestions
are visible choices, not guarantees about the weather the model will produce.
New and Fit can measure the local GPU when memory capacity is left blank.
Use **Enter New** to begin this route. The suggested duration is editable;
the chosen source, cycle and starting lead must provide the entire time window.
The engine refuses a window beyond that source's available forecast horizon.

For an existing case, use **O Open** and review its configuration. **P Plot set**
reviews the task's plots for the current case; saving writes a separate
plot-preference file. It does not rewrite the case's physics, domain, timing or
saved-field choices. A newly created task configuration gets its own plot
selection. Opening an existing file does not change its saved selection.

For archived downscaling, supply actual parent history and the producing ArWen
restart or stock-WRF namelist as physics evidence. A point-centred child can be
derived from an ArWen parent; other supported archives use an explicit child
configuration. The child inherits the parent's authoritative physics. Review the
available times, refinement, surface warm start and boundary cadence before
running. A finer child cannot recover information missing from the parent.

Point downscaling defaults to a refinement ratio of 3, the full parent time
window and the parent's output cadence. Its separate capacity default is
**24 GiB** when no capacity is supplied; a blank value here does not mean local
GPU detection. Enter the capacity you intend to use and review the engine's
memory plan. Planning and running are separate actions.

After a downscale, scenario or other completed run, choose the task's **R Plot
history** action. Select its actual `wrfout` file or history folder, requested
products, saved frames, image folder and source label. Folder discovery includes
child folders, skips symbolic links and refuses an oversized selection; the
exact files appear in the command review. For an ordinary output root, the
native Rust renderer creates a new timestamped image folder. If you instead
choose an existing `run-...` folder or a folder beneath it, the renderer writes
into that run and can replace matching image names. Choose a separate output
root when you want to preserve earlier images. This action plots saved weather
without starting a forecast.

The image source label starts as `External / unspecified forecast`, or
`HYPOTHETICAL scenario / source unspecified` in the hypothetical task. Replace
it with the producing model and the scenario assumptions that should appear
on the plots. A task's plot selection is a request to the renderer; archived
downscaling does not automatically apply that selection to its outputs.

## Match a task to the available model

| Task | Useful setup and interpretation | Plot selection |
| --- | --- | --- |
| General forecast | Start a regional case or inspect an existing one. | General |
| Supercell / severe weather | Run a nested case, downscale a suitable parent archive, or configure the existing reflectivity/UH follower. The atmosphere determines whether a storm develops; severe diagnostics do not predict individual tornadoes. | Tornado / severe |
| Tropical cyclone | Use real forcing or a suitable archive and inspect pressure, winds and moisture. A pressure follower can keep a nested domain near a modelled circulation. These are model fields, not official track guidance. | Hurricane / tropical |
| Hypothetical tropical case | Open a deliberately authored scenario configuration and forcing. A scenario requires a physically specified initial and boundary state; selecting this task does not insert or guarantee a hurricane. | Hurricane / tropical |
| Winter storm / blizzard | Inspect precipitation type, temperature profiles, wind and stored snow fields. Snow depth and snow water equivalent are distinct quantities. These plots alone do not classify a blizzard or calculate blowing snow. | Snow / winter |
| Heavy rain | Inspect precipitation, moisture and storm evolution. QPF describes precipitation, not flood depth or inundation. | Rain / flooding |
| Fire weather | Inspect atmospheric dryness, wind, instability and precipitation. The fire-weather composite is an atmospheric diagnostic; no ignition, fuels or wildfire spread is simulated by this selection. | Fire weather |
| Strong wind | Inspect surface gusts, pressure and flow aloft. Local exposure and unresolved structures still depend on the chosen grid. | Wind |
| Heat / cold | Compare air temperature, humidity, heat indices, wind chill and daily extrema. Each index has its own applicable conditions; they are not interchangeable temperatures. | Heat / cold |
| Fog / aviation weather | Inspect stored visibility and cloud fields, surface moisture, winds and nearby storms. Cloud fraction is not a cloud ceiling; dewpoint depression is not visibility. This selection does not determine flight categories or replace a flight briefing. | Fog / aviation weather |
| Mountain / terrain weather | Inspect model terrain, temperature, wind and precipitation across elevation changes. Grid resolution determines which slopes and valleys are represented. | Mountain / terrain weather |
| Coastal weather | Inspect atmospheric wind, pressure, moisture, cloud and rain near the coast. These products do not calculate waves, tides, storm surge or inundation. | Coastal weather |

## Edit a scenario's initial state

Open or create an experiment configuration first. **I Scenario** opens the
initial-state controls for that configuration. Add a warm bubble,
then enter its latitude, longitude, height above ground, horizontal radius,
vertical half-depth and peak potential-temperature increase. The supported
increase is greater than zero and at most 10 K. **Preserve relative humidity**
adjusts water vapour within the bubble; false keeps water vapour unchanged.

Use **F2 Keep bubble**, **R Review**, then **Enter Apply to draft**. Applying
changes the open draft; save it explicitly, preferably as a new scenario file,
then Check and Plan before running. Cancel leaves the draft unchanged. Removing
the last bubble disables the perturbation block. Keep the source configuration
and results for comparison.

The existing engine applies these smooth temperature perturbations once to the
initial state. It checks coarse-domain containment and whether the bubble
actually reaches model cells. The experiment runtime and prepared domain-tree
runner support this block; routes that do not apply it refuse the configuration.
Start-time nests receive the initial perturbation, while delayed nests get no
fresh bubble. Review the chosen route and prepare new inputs where required.

For a new GFS scenario, keep at least one child domain so **F7 Prepare and run**
creates a prepared domain tree. A nested severe-storm recommendation provides
child domains. Preparation retains the exact scenario configuration and records
that its bubbles are deferred; the sealed initial and boundary arrays retain
the source atmosphere. The tree runner applies the bubbles after restoring
the initial states. Resuming a checkpoint does not apply them again. The
single-domain GFS route still refuses this block. Companion stock-WRF export
is also refused because those unperturbed files cannot carry the deferred
scenario; the native prepared tree remains available to run.

A warm bubble can explore convective initiation. It does not insert, relocate
or guarantee a hurricane, and selecting the hypothetical task alone leaves the
source atmosphere unchanged. Keep the scenario's actual assumptions in its
configuration and plot labels.

## Follow weather without changing what the tracker means

The domain controls support reflectivity, updraft-helicity, pressure and native
attribute tracking. UH following needs its reflectivity fallback before rotation develops.
Pressure following normally uses height minima on an 850 hPa surface; selecting
level 0 explicitly uses mean sea-level pressure. The thresholds have different
units: vortex depth in metres on a pressure surface, or an absolute pressure
ceiling in hPa for mean sea-level pressure. Review the selected signal, search
area, cadence and movement limits instead of treating every low as a hurricane.

Choose `attribute` as the follow signal to track a feature in one of these native
fields. The threshold uses the field's units; select `max` to follow values above
it or `min` to follow values below it. The same weighted centroid, search radius,
movement limits and cooldown control the resulting domain movement.

| Model attribute | Meaning | Threshold units |
| --- | --- | --- |
| `theta` | Total potential temperature, base plus perturbation | K |
| `qv` | Water-vapour mixing ratio per unit dry-air mass | kg/kg |
| `qc` | Cloud-water mixing ratio per unit dry-air mass | kg/kg |
| `qr` | Rain-water mixing ratio per unit dry-air mass | kg/kg |
| `w` | Vertical velocity averaged from adjacent faces onto mass levels | m/s |

Select `column_max`, `column_min` or `column_mean` for a vertical reduction, or
`model_level` for one explicit mass level. The mean is an unweighted average of
model levels. Model-level indices start at **0**, the lowest mass level, and
must fit the source grid. Clear the index for a column reduction. Model levels
are not fixed pressure or height surfaces. Moisture attributes require a moist
source domain; missing fields and invalid levels refuse before use. Any column
with a non-finite value in its selected levels is excluded from the search.

The attribute, direction, reduction, threshold units and selected model level
appear in the tracking evidence. Attribute track CSVs contain time and the
moving domain's own location. They do not contain hurricane intensity or
isobaric-level columns. Potential temperature is distinct from air temperature,
and low water-vapour mixing ratio is distinct from relative humidity. Selecting
these fields does not establish a fire, hail or other hazard diagnostic.

Prepared moving-domain runs need a verified statics corridor covering their
possible movements. Reprepare a bundle that lacks that corridor. Timed manual
relocation and signal following cannot both drive the same relocation mechanism.
Prepared runs currently refuse weather-triggered domain creation. Dated delayed
activation has its own supported route and exact input requirements; a fixed
delayed cache under a moving ancestor is refused. A visible domain control does
not bypass these engine checks.

## What the plot picker saves

The default stays **General: 25 products**. The original six selections keep their
order and stable IDs: `general`, `tornado`, `hurricane`, `snow`, `rain`, `wind`.
Five additional selections have the stable IDs `fire`, `temperature`, `aviation`,
`terrain` and `coastal`. The installed native renderer owns the available product
catalog; every added selector names a real canonical native product.

Selections request plots from saved history. They do not enable missing model
diagnostics or change the history interval. Visibility, cloud, gust and snow
products need their actual fields; one-hour, daily and other accumulated products
need the corresponding time windows. The renderer names unavailable products as
skips, and a request that produces no picture fails. It does not substitute an
invented field. Use Customize to remove products that are irrelevant to a short
run or unavailable input.

The terminal, CLI and remote route retain their existing configuration and
review controls. See the [terminal instructions](../../tools/arwen-tui/README.md)
and [complete CLI options](CLI-OPTIONS.md) for the exact settings available in the
installed build.
