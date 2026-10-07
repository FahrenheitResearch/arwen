# Nowcast radar-heating evaluation (work package 3)

The box harness for the design in `STORMSCOPE-HEATING-2026-10-06/DESIGN.md`:
WOOF in the HRRR configuration, with and without radar heating for its first
hours, on five cases, scored against MRMS and surface stations, judged by eye
on the sheets first.

Status: next release (2.8.7+), unproven until the box test.

## One command on the box

On a machine with the engine repository:

```
bash tools/heat_eval/make_kit.sh lane/nowcast-heat-box lane/scoreboard-286 /tmp/kit
```

Copy `/tmp/kit/heat-eval-kit-<rev>.tgz` to the box, then on the box (root):

```
mkdir -p /work/nh/logs && tar -xzf heat-eval-kit-<rev>.tgz -C /work/nh
NH_START_CHAIN=1 setsid nohup bash /work/nh/kit/bootstrap.sh > /work/nh/logs/bootstrap.log 2>&1 < /dev/null &
```

The bootstrap arms the dead-man timer first, builds the engine venv and the
native bridges, fetches tables, geography and every case's HRRR and RAP
inputs, and builds the StormScope venv beside them, then (with
`NH_START_CHAIN=1`) starts `box_chain.sh`, which runs everything else from
markers. Progress: `/work/nh/coord/events.log`. Results to pull:
`/work/nh/results/` (sheets with captions in `results/gallery/`, score
tables, receipts, `decision.json`). The chain gives 45 minutes after it
ends for the pull; then the dead-man ends the box.

There is one preparation per start time and run length
(`<case>/preps/<HHMM>z-<N>h/`). The door binds the SHA-256 of the config its
preparation read and refuses any other. It cuts an arm's `[radar_heating]`
table out first (work package 2), so A, C and E share one preparation, but
it does not cut the length: it runs the config's own `run_seconds` and only
warns about a shorter `--run-seconds`. So the 6 h arms C60 and E3D get their
own 6 h preparation, and B gets its 17Z one. The first case also prepares
gate G0's one-hour door case, which work package 2's real 2-card against
4-card door comparison runs on (`GPUWM_HEAT_BOX_DOOR`); without it that test
skips, and G0 refuses to pass on a skip.

Gate G1 probes arm A, with the heating's memory added from the design, so
arm A starts on every case while G0 runs; only heated arms wait for G0.

Nothing waits forever: a failed case step leaves `steps/<STEP>.failed`, and
an arm whose inputs failed, or that `run_arm.sh` or the mutex refused, counts
as ended. A decision case that fails gate G3, or whose control cannot be
prepared, gives its slot to the reserve; the anchor's G3 is recorded and
never gates. A StormScope venv that will not build costs arms C and C60 only.

## Files

| File | What it does |
|---|---|
| `cases.toml`, `arms.toml` | the cases and arms, as rows |
| `plan.py` | loads both tables; paths, fetch lists, window classes |
| `make_arm_configs.py` | arm A per case from `door-template.toml`; every other arm as an overlay, refused if it changes anything but start, length and `[radar_heating]` |
| `fetch_case.sh` | HRRR and RAP inputs, anonymous HTTPS from the public buckets |
| `case_steps.sh` | one CPU step of a case (fetch, configs, gate G3, each arm's preparation, windows, MRMS frames), or the nowcast on one card (which halves its batch on a failed run: gate G2) |
| `run_arm.sh` | one arm on the cards the GPU mutex granted, with the gate and window refusals; a probe is stopped after f01 |
| `box_chain.sh` | the whole run, driven by marker files; sheets redrawn as each arm ends |
| `sheets.py` | every sheet through `rw_compare` only |
| `score_case.py` | radar scores (registration `nowcast-heat-v1`) and station scores (scoreboard v2) |
| `decide.py` | the decision rule; the eye's verdicts come from `results/eye.json`; a reserve stands in for a decision case that failed G3 |
| `gates.py` | gate G0 (the heating's GPU tests, no skips), gate G3 (35 dBZ area growth) and gate G1 (cards per arm) |
| `bootstrap.sh`, `lib.sh`, `deadman.sh`, `gpu_mutex_run.sh`, `make_kit.sh` | the box itself |

## The eye

Sheets are looked at before any number. Whoever judges writes
`/work/nh/results/eye.json`:

```
{"cases": {"c1-20240521": {"C_better_than_A": true, "E_better_than_A": true}, ...}}
```

Then `python -m tools.heat_eval.decide --root /work/nh --eye /work/nh/results/eye.json`.
Without it the verdict is `pending`.
