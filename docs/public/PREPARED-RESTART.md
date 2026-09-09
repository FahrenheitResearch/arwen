# Resume a prepared forecast

Set a nonzero `restart_interval_s` in the experiment before preparing it. Both single-domain and hierarchy forecasts write canonical `gpuwmrst_d01_*.npz` checkpoints at that cadence.

Use the existing prepared bundle and its exact configuration for continuation, with a new output directory:

```console
gpuwm sim prepared --experiment-config prepared/experiment.toml --wps-namelist prepared/namelist.wps --restart previous/gpuwmrst_d01_<instant>.npz --outdir continued
```

For `go`, name the existing bundle explicitly:

```console
gpuwm go prepared/experiment.toml --prepared-root prepared --wps-namelist prepared/namelist.wps --restart previous/gpuwmrst_d01_<instant>.npz --outdir continued --products none
```

The equivalent run-plan uses `route: "prepared"` and `run_options.prepared_root`, `wps_namelist`, and `restart`. These commands reuse the prepared input; they do not fetch or prepare it again. A single bundle binds the complete configuration, including the stop time. Its checkpoint continues the remaining interval and does not extend that sealed forecast. Sealed forcing extension remains a hierarchy operation.

A checkpoint resumes only on the build that wrote it: 2.7.0 refuses every 2.6.5 checkpoint by name (see the CHANGELOG's "Restart and import compatibility" section, or run `gpuwm doctor --since 2.6.5`).

History already committed at the checkpoint is retained in the previous output directory. The new directory contains only later frames. Restoring a checkpoint already at the configured stop performs no steps and writes no duplicate history. `--health-debug` on `sim` enables the shared per-step whole-domain validator, including the canonical store when streaming.

## Checkpoint format compatibility

2.7.0 writes restart format 6. A checkpoint written by 2.6.5 (format 5) is refused by name: 2.7.0 adds the adaptive-timestep and `eta_levels` configuration echo, the `fields/ustm` surface-layer member and the `held/gf_*` cumulus forcing members, none of which a format-5 file carries. Restart such a run from its initial conditions, or complete it on 2.6.5. A format-6 checkpoint written with the adaptive clock off does not need the adaptive fields in its configuration echo, and an omitted `eta_levels` matches the default (inherit the source's ladder).
