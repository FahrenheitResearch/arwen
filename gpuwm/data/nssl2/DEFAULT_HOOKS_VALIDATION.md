# NSSL default runtime-hook validation

The default option-18 condensation hook binds the admitted WRF v4.6.1
`NUCOND` implementation directly to the durable production workspace.  The
launcher already includes the source routine's maximum-supersaturation
`QVEXCESS` call and caller update, so it is installed as the coordinator's
single combined `nucond_qvexcess` hook.  The independently admitted
`nssl2_qvexcess` module is not executed a second time.

The hook requires an explicit fused-GS callback, passes number/CCN fields in
native concentration units, owns one named write-before-read supersaturation
scratch field classified as restart-rebuilt, and contains no Registry gather
or scatter.  It does not make `mp_physics=18` selectable.

Validation commands:

```text
python -m pytest -q tests/test_nssl2_default_hooks.py
python -m pytest -q -m gpu tests/test_nssl2_nucond_diagnostics.py tests/test_nssl2_runtime.py
python -m ruff check gpuwm/core/nssl2_default_hooks.py tests/test_nssl2_default_hooks.py
```

The eventual MP18 selector must construct this bundle with the admitted fused
GS callback, call the runtime adapter with `radiation_due=True` on every
microphysics step, and pass the history-step flag only to `output_due`.
