# Complete recipes on a region

`gpuwm domain --recipe hrrr_v4_gsd41 --region PH --spacing-km 3 --source gfs --cycle 2025-11-03T00 --hours 6 --out country.toml`

Run `gpuwm check country.toml`, stage the required geography and tables, then
`gpuwm go country.toml`. The recipe contains the physics, eta ladder, model
top, clock, lateral widths and output settings. Only geometry, cycle, duration,
forcing and the declared aerosol choice change. Spacing does not rescale the
recipe clock. At very fine spacing the recipe clock may require a different
recipe row, which is data work.

Regions live in `gpuwm/data/nwpsuite/regions.v1.json`. A row's `polygon` can
be inline GeoJSON or a reference to `registries.polygons` in the adjacent
`references.v1.json`. A coverage-envelope reference declares `[west, south,
east, north]` bounds. Region rows can be keyed by id or listed with an `id`.
`sources.global_ic` declares ordered driving sources, or `sources` can be
an ordered list. `--region-table` selects an external table.
`--region` also accepts a local polygon file; `--polygon` is an alias for this
recipe door. Polygon and MultiPolygon validation, antimeridian bounds, grid
generation and containment use the existing domain operators. The grid covers
the polygon's envelope, not a masked set of cells. `--buffer-km` adds a buffer.

Recipes live in `gpuwm/data/recipes/registry.v1.json`; each row names a TOML
and an ordered `aerosol_sources` list. Adding a model requires a new recipe
and registry row. `--recipe-table` selects an external registry. A local TOML
uses the registry's optional default aerosol policy. An aerosol-aware recipe
without a declared policy refuses emission. Existing source and physics
support still applies.

The aerosol policy first tries a driving route declared to carry the analysed
QNWFA/QNIFA pair, then the monthly WRF climatology. The analysed choice still
requires both fields on every actual initial and boundary frame. A missing or
partial pair refuses; the run does not silently mix sources between frames.
The climatology choice is strict: `gpuwm fetch-tables --wif` stages its dataset;
an absent dataset refuses preparation. No synthetic or zero fallback is added.

The adjacent `.recipe.json` records the ordered policy, selected candidate,
rejected candidates, recipe and polygon hashes, and emitted config hash.
Preparation and forecast `aerosol_initialization` receipts record the source
actually consumed and dataset identity. Climatology receipts declare the
forcing difference: this is the HRRR configuration outside RAP coverage,
with monthly aerosol instead of operational analysed aerosol. The recipe's
existing numerical differences also remain; see `docs/dev/mynn-gsd41.md`.

Projection follows absolute centre latitude: below 25 degrees Mercator,
25 through 60 Lambert, above 60 polar stereographic. Pole-touching domains
refuse because source interpolation and static-tile windowing do not support
poles. Source coverage and crop longitude limits still apply. Regular and
rotated latitude-longitude grids are not built.

Worldwide surface METAR is already served by `gpuwm obs asos`: the network
extent table and generic Rust fetcher select stations by geometry. SYNOP
coverage is not claimed. Public CMORPH can provide satellite precipitation;
IMERG needs a NASA Earthdata credential.

## Dated integration proof

![Philippines 3 km recipe, simulated radar at hour 6](img/whats-new-2.9.0/01-country-recipe-ph-reflectivity-f006.png)

The Philippines run below, six hours in (valid 2025-11-03 06 UTC): simulated
radar drawn by the Rust renderer from the run's own history. A reduced copy;
the full set of eight maps is in the dated lane evidence.

At source commit `ec754823a5c6cb33f0c606905249391d171090bb`, the ordinary
recipe door emitted the suite-contract region `PH` at 3 km, 400 by 686 by
50 levels, for 2025-11-03 00 UTC. Ordinary `gpuwm go`, with GFS hourly
initial and boundary inputs and the monthly aerosol dataset, completed
21,600 model seconds on two queued cards. All seven hourly histories passed
native readback checks. All 48 aerosol boundary edge pairs had finite,
positive start and end values; the receipt identified the actual dataset
rather than a synthetic fallback.

A short US control at 2022-06-13 18 UTC, with original recipe physics and
analysed aerosol, used the same sealed preparation at the lane base
`cd336e62d6e6b6312c9636e0fdfff84801909b6a` and this source tip. All four
whole history-file SHA-256 hashes at 0, 20, 40 and 60 seconds matched. This
checks the short integration path, not full continental trajectories or
independent preparation equivalence. The packaged recipe is also byte
identical to its original config.

Focused checks: 192 passed, 3 skipped. All 22 new regression cases failed
at the base. Three imported suite-planner tests failed separately; this
proof does not qualify that planner's ensemble admission or override paths.
The dated lane report contains commands, logs, hashes, observation scores
and declared limits. This is one country integration proof, not a general
forecast-skill qualification.