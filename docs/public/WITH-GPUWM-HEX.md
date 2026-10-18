# gpuwm with gpuwm-hex

gpuwm-hex is the global model of this family: a GPU port of the
MPAS-Atmosphere v8.4.2 dynamical core on unstructured, variable-resolution
meshes. It owns no physics of its own. Every physics column runs through
gpuwm, and its initial conditions, product maps, soundings and scores are
drawn by gpuwm's Rust tools. So a gpuwm-hex release is always paired with a
gpuwm release.

The gpuwm-hex release on PyPI is 0.3.3. Its wheel declares `gpuwm>=2.8.0,<2.8.1`, so
`pip install gpuwm-hex` in this environment replaces this gpuwm with 2.8.0 and it ships no
runner for `gpuwm preset`. Keep gpuwm-hex 0.3.3 in its own environment.

`gpuwm preset list` and `gpuwm preset check` count the installed gpuwm-hex: a runner that will
not import, or a release outside the preset's declared range, is a blocker with the install
command beside it, and `gpuwm preset prepare` and `gpuwm preset run` refuse with the same words
before any card is touched.

## What gpuwm provides to gpuwm-hex

- **Physics seam.** `gpuwm.core.mpas_column_suite.run_mpas_column_physics`
  runs any registered scheme selection on MPAS columns.
- **Variable-resolution convection.** New Tiedtke uses each cell's own grid
  spacing, so it can run across a mesh whose cells range from 60 to 3 km.
- **Physics fixes hex relied on.** Kain-Fritsch accumulated convective rain no
  longer steps backwards; RRTMGP keeps positive layers above the model top
  (hex meshes produced non-finite longwave heating there).
- **Initial conditions.** `met_intermediate` reads DWD ICON global through its
  table row, and is in the bridge bundle (`gpuwm fetch-bridges` installs it).
  `rw_mpas_init` computes its momentum columns in parallel, with unchanged
  bytes.
- **Products.** `rw_mpas_convert --remap product` (interpolated state,
  area-conservative accumulated precipitation, nearest categories, with a
  precipitation budget line per frame) and `--jobs N`, and the `global-mesh` and `region:NAME` render
  windows with the `GPUWM_GRID_LABEL` source label that hex's product views
  ask for.
- **Soundings and meteograms.** `rw_hex_points` draws them at named points.
  It ships in the platform wheels and the bridge bundle (`gpuwm fetch-bridges`
  installs it), and `gpuwm doctor` checks it. To use another build, point the
  hex door at it with `$GPUWM_HEX_RW_HEX_POINTS`.

## What does not work together yet

- **WOOF point runs on hex** (a regional point forecast handing off to a hex
  fine core) are refused by name: the projected-intermediate producer they need
  is not qualified.
- A suite (`gpuwm suite`) row that runs hex plans and validates; running it
  unattended needs the same `--bindings` document as any other row.
