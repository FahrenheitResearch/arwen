Run this small coupled fire and bulk-smoke example on a GPU:

```sh
gpuwm fire-ideal examples/sfire-ideal/fire.toml \
  --sounding examples/sfire-ideal/input_sounding --outdir out/fire
```

The assembled WOOF package exposes the same arguments with `woof fire-ideal`.
The atmosphere has 600x500m horizontal extent and six mass levels; the refined
fire cells have 25m spacing. The run advances 8 seconds and writes histories and
two checkpoints. It is a short integration demonstration, not a wildfire
forecast or an observation-based validation.
The prescribed ignition expands at 100m/s until its 30m radius is reached;
subsequent fire spread uses the fuel and atmospheric Rothermel calculation.
Eight acoustic substeps resolve the sound waves on the 50m atmospheric grid.

A continuation from the 4-second checkpoint to 8 seconds uses a fresh output
folder and the same config/input bytes:

```sh
gpuwm fire-ideal examples/sfire-ideal/fire.toml \
  --sounding examples/sfire-ideal/input_sounding \
  --restart out/fire/gpuwmrst_d01_0001-01-01_00:00:04.npz \
  --outdir out/continued
```

The new output begins at the saved clock. Fire consumption and smoke source
continuation do not replay already completed steps. See docs/SFIRE-IDEAL.md
for native namelist input, source grid files and initialization controls.
