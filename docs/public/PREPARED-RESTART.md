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

History already committed at the checkpoint is retained in the previous output directory. The new directory contains only later frames. Restoring a checkpoint already at the configured stop performs no steps and writes no duplicate history. `--health-debug` on `sim` enables the shared per-step whole-domain validator, including the canonical store when streaming.
