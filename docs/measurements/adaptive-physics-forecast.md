# Adaptive physics forecast checks

Base: `49cc11670331b28d18ae119884d130e9e34cbf28`, descended from
`a806945baa520b52f3f94cc836fe07c3b77d1319`.

The first adaptive step bypassed output, restart, boundary and stop alarms.
A forecast with a 90-second starting step and 60-second history interval
completed with history at 0, 120 and 180 seconds, omitting 60 seconds.
The clock now applies its existing alarm and stop clipping to the starting
step as well as subsequent steps. No physics formulas or scheme choices change.

## Reproduction and regression

Run from the source directory on a CUDA 13 test machine, with the new test file over
the unchanged base source:

```sh
../venv/bin/python -m pytest tests/test_adaptive_physics_forecast.py -q -s -p no:cacheprovider --basetemp=../pytest-base-gpu
```

Base: 5 passed, 1 failed in 3.00 seconds. The failing case is
`test_adaptive_physics_forecast[6-16-False-True]`, whose expected history is
0, 60, 120, 180 seconds. With the fix, the same command using
`--basetemp=../pytest-tip-gpu` gives 6 passed in 2.89 seconds.

The production executor runs every case with live CFL reductions and the
adaptive driver. The domain has 24 by 24 horizontal cells and 32 levels.

| Case | Forecast duration | Checks |
| --- | --- | --- |
| cu 16, mp 6 | 180 s | Changing step, cached dt/delt/ztmst, every-step cumulus |
| cu 6 closure, mp 6 | 180 s | Same checks, using cu 16 with the closure flag |
| mp 6 | 2700 s | Cold resting column, precipitation, changing minor-loop count |
| mp 16 | 2700 s | Same checks for the double-moment scheme |
| Spectral apply | 180 s | Three-step cadence, receipt count and elapsed windows |
| cu 16 startup alarm | 180 s | First output retained when starting step exceeds cadence |

Every case checks finite prognostics, exact final clock time and all history
alarms. History uses the production consume-once reflectivity handoff without
writing weather images. These are short idealized integrations, not skill scores.

## CPU checks

Local environment: dedicated Python 3.12 venv, no CuPy,
`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `CARGO_BUILD_JOBS=4`,
`PYTHONDONTWRITEBYTECODE=1`. All commands disable the pytest cache; temporary
directories and complete logs are outside the checkout.

```sh
python -m pytest tests/test_spectral_seam.py tests/test_ntiedtke_phase2_gates.py tests/test_physics_driver.py tests/test_adaptive_clock_driver.py -q -p no:cacheprovider
python -m pytest tests/test_adaptive_timestep_controller.py tests/test_adaptive_timestep_executor.py tests/test_adaptive_timestep_checkpoint.py tests/test_adaptive_restart_sound_steps.py -q -p no:cacheprovider
python -m pytest tests/test_adaptive_clock_driver.py -k first_adaptive_step_lands -q -p no:cacheprovider
```

The first two commands on unchanged base tests give 80 passed, 57 skipped,
1 expected failure, and 46 passed, 7 skipped respectively. The third command,
with the new regression over base production code, gives 4 failed. It covers
history, restart, external boundary and run-end alarms, including nest division.

The union of the first two commands after the fix, including the four new
regressions, gives 130 passed, 64 skipped, 1 expected failure in 52.90 seconds.
Thus the pre-existing 126 passing tests remain passing and the four new
regressions turn green. No new failure or skip is introduced.

## GPU environment

The base was exported with `git archive`, then installed into a fresh uv
Python 3.12 venv with local `gpuwm-data` and `gpu-cu13`. The required
`static-fields` library was built from that source with
`cargo build --release -p static-fields --offline`. The final six-case base
and fixed runs use that native library. Only the changed clock source differs
between those runs. The box is an RTX PRO 4500 with CUDA 13.

The guard sweep found no separate refusal or workaround citing this first-step
alarm defect to retire.
