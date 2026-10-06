# Full-ARW column oracle

`generate_real_columns.py` consumes the object-free NPZ and matching JSON
produced by the RUC fused-call observer. It preserves all sampled model
inputs in a float32/int32 stream and emits a Fortran `LSMRUC` driver.
The driver loads module parameter tables with `ruclsminit`, then restores
every captured argument before the call. The fork's snowfall accumulator
is converted between its metres and the port's millimetres.

The unchanged fork RUC and model constants are compiled with `EM_CORE=1`,
`wrf_chem=0`, and `NMM_CORE=0`. Only WRF communication, logging, and error
reporting use single-process stubs. The source URLs, hashes, interface,
and Intel build commands are in `source-pins.json`. The CPU build checks
those identities before compiling.

From an environment where `gpuwm`, NumPy, and the selected compiler are
available:

```sh
python tools/ruc_hrrr_fork_oracle/generate_real_columns.py capture.npz columns
bash tools/ruc_hrrr_fork_oracle/build_real_columns.sh HRRR_WRF_ROOT columns ifort
bash tools/ruc_hrrr_fork_oracle/build_real_columns.sh HRRR_WRF_ROOT columns gfortran
python tools/ruc_hrrr_fork_oracle/compare_real_columns.py capture.npz columns
```

`HRRR_WRF_ROOT` has the tagged `phys/`, `share/`, and `run/` files listed
in the manifest. The Intel environment may be supplied by the existing
GSI clone toolchain. The build is CPU-only and single-threaded.

`comparison.json` records differences for every RUC carrier returned by
the full driver. It compares the Intel fork result, GNU compiler control,
host port, and captured production CUDA output. Latent heat is grouped by
sampled land class. Wetness quantile means are not population-weighted
grid means. This is a measurement tool; it does not infer station skill
or assign a scientific pass from a chosen numerical tolerance.

`tests/test_ruc_real_column_oracle.py` executes an independent Fortran
reader to check profile, fraction, scalar, and integer serialization. It
also executes the generated driver against a deliberately destructive
initializer. Removing the input restore is a negative control that must
fail the argument comparison. The separate unchanged-fork run checks
actual RUC physics.
