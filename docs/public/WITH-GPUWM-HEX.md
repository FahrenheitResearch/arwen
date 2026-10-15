# gpuwm with gpuwm-hex

gpuwm-hex is the global model of this family: a GPU port of the
MPAS-Atmosphere v8.4.2 dynamical core on unstructured, variable-resolution
meshes. It owns no physics of its own. Every physics column runs through
gpuwm, and its initial conditions, product maps, soundings and scores are
drawn by gpuwm's Rust tools. So a gpuwm-hex release is always paired with a
gpuwm release.

| gpuwm-hex | needs gpuwm | why |
|---|---|---|
| 0.4.0 | 2.9.0 | the generic column-physics entry and the renderer pieces below exist from 2.9.0 |
| 0.3.3 | 2.8.7 | its sixteen seam files are pinned to 2.8.7's bytes |

## Install

```sh
pip install "gpuwm-hex==0.4.0"     # pip installs the gpuwm it declares
gpuwm fetch-bridges                  # the Rust tools both packages drive
gpuwm-hex doctor                     # what this install can reach, and the command for each gap
```

Then follow the gpuwm-hex user manual (`docs/manual/index.md` in the
[gpuwm-hex repository](https://github.com/FahrenheitResearch/gpuwm-hex) and
its source distribution). Its chapter 10 lists what 0.4.0 adds.

## What gpuwm 2.9.0 provides to gpuwm-hex

- **Physics seam.** `gpuwm.core.mpas_column_suite.run_mpas_column_physics`
  runs any registered scheme selection on MPAS columns; hex's suites and its
  generic arbitrary-physics path call it.
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
  precipitation budget line per frame) and `--jobs N`, used by
  `gpuwm-hex products`, and the `global-mesh` and `region:NAME` render
  windows with the `GPUWM_GRID_LABEL` source label that hex's product views
  ask for.
- **Soundings and meteograms.** `rw_hex_points` draws them at named points.
  It ships in the platform wheels and the bridge bundle (`gpuwm fetch-bridges`
  installs it), and `gpuwm doctor` checks it. To use another build, point the
  hex door at it with `$GPUWM_HEX_RW_HEX_POINTS`.
- **RRFS v2 on the hex dycore.** `gpuwm preset run rrfs-v2-hex-tempo` finds
  its runner and preparation in the installed gpuwm-hex, with no source
  checkout, and `rw_mpas_init` writes the nine-level RUC soil and the aerosol
  species the preset's physics reads, from public GFS (`gpuwm preset prepare`,
  then `gpuwm preset run rrfs-v2-hex-tempo`). A one-card 12 km 6 h forecast
  prepared this way ran and was scored against observations.

## What does not work together yet

- **WOOF point runs on hex** (a regional point forecast handing off to a hex
  fine core) are refused by name: the projected-intermediate producer they need
  is not qualified.
- **gpuwm-hex 0.4.0 on gpuwm 2.8.9**: not supported. 2.8.9 lacks the column
  physics entry and the fixes above.
- A suite (`gpuwm suite`) row that runs hex plans and validates; running it
  unattended needs the same `--bindings` document as any other row.
