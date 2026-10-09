# `gpuwm verify-exact`: the combo sweep, replayed and scored bitwise

`gpuwm verify-exact` replays recorded stock WRF runs through WOOF on one card,
in one process, and scores every recorded step word for word. It prints one
line per run: `PASS`, or the first step that differs, the field that leads,
and how far apart it is in ULP.

Exit status:

| Status | When |
|---|---|
| 0 | at least one run or fixture passed (`PASS`, `WOOF-ONLY-OK`, fixture `PASS`) and none failed |
| 1 | any run or fixture failed |
| 2 | a usage error, including a selection that matches no run, or a `--combos`, `--cases` or `--strata` name the recording does not hold |
| 3 | nothing failed and nothing passed: no proof (only WRF refusals, unsound or unknown referees, informational builds) |

```
GPUWM_WRF_EXACT=1 gpuwm verify-exact --recordings REC --out OUT [--combos A001,A002] \
    [--cases iowa-convective] [--strata A] [--shard 2/4] [--referee scalar|strict]
```

The command refuses to start without `GPUWM_WRF_EXACT=1` (the strict build is
the 0 ULP standard of proof). `--default-arithmetic` allows a run without it,
and such a run is not a 0 ULP proof.

## What it does for each run

1. Stages a real.exe handoff from the recording: the exact `wrfinput_d01` and
   `wrfbdy_d01` bytes WRF read (linked, not copied) and the `namelist.input`
   WRF ran.
2. Resolves it through WOOF's native WRF-file door, the one door strict mode
   has qualified (it keeps WRF's `T = theta - 300` words).
3. Applies the combo's `woof_settings` to the imported TOML, exactly as the
   combo list's load check did when it proved every combo loads. Combos that
   enter through TOML (analytic radiation) or that acknowledge longwave off are
   imported with radiation 1 and given their radiation by those settings.
4. Integrates in this process, writing a history frame at every recorded
   step, for as many steps as the furthest recorded reference reached.
5. Decodes each frame through the Rust NetCDF bridge, hashes every field in
   the reference's own dtype and shape, and drops the frame.
6. Scores the replay against every recorded build of that run.

A combo whose checks include `repeat_run_0ulp` is run twice and the two WOOF
runs must match word for word.

## The referee and the outcomes

The referee decides PASS or FAIL. It is the scalar build by default (WOOF's own
float32 math is scalar), or the Thompson-fixed variant (`scalar-tfix`) wherever
one was recorded, because stock 4.6.1's Thompson answer comes from an
out-of-bounds table read. `--referee strict` picks the strict build instead.
Every other recorded build is scored too and shown on the same line.

| Outcome | Meaning | Fails the run |
|---|---|---|
| `PASS` | 0 ULP on every reference field at every step against a sound, designated referee; no reference field skipped except the listed ones below | no |
| `FAIL` | a difference against a sound, designated referee, or a reference field WOOF did not write or wrote in another shape | yes |
| `WOOF-REFUSED` | WOOF's door or loader refused the combo, or the runner raised a named `ValueError` before its first history frame | yes |
| `WOOF-CRASH` | WOOF stopped before the referee's last good step, every frame it wrote matching; or, with no referee, WOOF did not write every planned step | yes |
| `WOOF-NONFINITE` | no referee, and WOOF's state went NaN or infinite | yes |
| `REPEAT-MISS` | the combo asks for a WOOF repeat run and the two runs differ, or the repeat run is missing | yes |
| `UNSOUND-REF` | the referee is not a sound 0 ULP reference, whether WOOF matches it or not: `clean_reference` is no, there is no `clean_reference` entry (unknown), or the referee crashed or went non-finite. Never a pass | no |
| `INFO-MATCH`, `INFO-MISS` | scored against stock 4.6.1 for information: the combo's designated reference is another build | no |
| `WOOF-ONLY-OK` | no reference exists, and WOOF wrote every planned step, stayed finite and (where the combo asks) repeated | no |
| `WRF-REFUSED` | WRF refused the combo, so there was nothing to replay | no |

Soundness is decided before any PASS. A referee is sound only when its run
finished OK and `clean_reference` vouches for it (`yes`, or `yes, with the
Thompson fix build` for a `-tfix` build). A referee that crashed or went
non-finite is compared only through its last good step and is always
`UNSOUND-REF`, so the steps WOOF wrote past the crash can never hide behind a
match. Fields the recording's own repeat test showed to hold uninitialised
memory (`unreliable_fields` in `MANIFEST.json`, today `RQIBLTEN`) are never
compared.

Every other reference field must be compared. A reference field WOOF does not
write (`absent_unexcused`) or writes in another shape (`reshaped`) fails the
run, even when every field both sides write agrees. The one exception is
`ALLOWED_ABSENT` in `gpuwm/verify_exact/compare.py`: time-invariant grid,
map-factor and vertical-coordinate constants, static fields copied from
`wrfinput_d01`, and run flags and seeds, each with its reason. A listed field
is excused only when the referee's own digests show it never changed over the
compared steps (`absent_allowed` in the results). Nothing prognostic,
physical, accumulated or diagnostic is on that list, and fields WOOF does
write (its own `XLAT`, `MAPFAC_M`...) are always compared.

### Coverage over what ran

After the run lines the command prints, per stratum, how many option-value
pairs the combo list holds, how many sit in a run WOOF integrated against a
sound, designated referee (`PASS` or `FAIL`), how many sit in a run that
passed, and which option values were never integrated against a sound
referee. The same rows are written to `results/<I>of<N>.coverage.json`. The
list's own pair coverage is a property of the list; these numbers are what
the run set exercised.

### Reading a line

```
A001   iowa-convective   FAIL   vs scalar: step 1 HFX (+31 more) max 812 ulp at step 1 (1490/1600 words, first at (0,3)) | also strict: step 1 HFX
```

- `step 1`: the first model step at which any compared field differs.
- `HFX (+31 more)`: the lead field and how many other fields differ at that
  step. Fields are listed prognostic state first, then physics state, then
  diagnostics, then time-invariant fields; the results file lists them all.
- `max 812 ulp at step 1`: the lead field's largest gap, measured where both
  sides kept raw arrays (the first differing step if the recording kept it,
  otherwise the next kept step). Indices are in the array's own axis order
  (`bottom_top, south_north, west_east` for a 3-D field), 0-based.
- `[diagnostics/static only]` marks a run whose only differences are
  diagnostics (`T2`, `Q2`, `U10`, `V10`, `PSFC`, `REFL_10CM`...) or
  time-invariant fields.

## Speed

Kernels compile once per process (about 3 minutes cold under strict
arithmetic) and are cached in `CUPY_CACHE_DIR` after that. Every later run in
the same process pays only its own initialisation and integration, so put as
many runs as possible in one process. `--shard I/N` splits a selection across
processes, one card each.

## Output

Under `--out`:

| Path | What |
|---|---|
| `results/<I>of<N>.jsonl` | one JSON object per run (and per fixture): outcome, every comparison, ULP detail, WOOF status and wall time |
| `results/<I>of<N>.header.json` | arguments, `GPUWM_WRF_EXACT*` environment, engine git identity, card |
| `woof/<case>/<combo>/` | WOOF's replay in the recording layout below: digests at every step, plus the arrays of fields that differ at the reference's kept steps; `hashes.json` also carries WOOF's status and error |
| `woof-repeat/<case>/<combo>/` | the repeat run, when the combo asks for one |
| `scratch/` | each run's handoff and history while it runs; deleted after hashing unless `--keep-history` |

`--score-only` rescores the stored replays with no card.

### Measuring a miss in ULP when the reference arrays live elsewhere

A copy of the recording may hold every digest but not the raw arrays (the
replay copy on the GPU box does). Digests are enough for PASS and FAIL and for
the first differing step and field. To measure a miss in ULP:

1. `--write-needs needs.json` lists, for every miss without a ULP, the one
   referee step and the few fields that would measure it.
2. On the box with the full recording,
   `python tools/verify_exact_raw_subset.py RECORDING needs.json OVERLAY` cuts
   just those words into an overlay (numpy only; each field's SHA-256 is
   checked against the recording's index).
3. Bring the overlay back and rerun with `--score-only --raw-overlay OVERLAY`.

## The combo list

`tools/combo_sweep/gen_combos.py` writes the combo list
(`tools/combo_sweep/combos.json`). `combos-v1.json` beside it is the list the
stock WRF 4.6.1 recordings were made from. Two rules learned from WRF's own
refusals are in the generator: every `mp_physics = 28` combo carries the
aerosol climatology triple (`wif_input_opt = 1`, `num_wif_levels = 30`,
`use_aero_icbc = .true.`), and MYNN (run with `bl_mynn_edmf = 1`) never meets
a Grell-Freitas shallow plume (`ishallow = 0`).

```
python tools/combo_sweep/gen_combos.py --keep-rows-from tools/combo_sweep/combos-v1.json
```

keeps every v1 row that is still valid under its v1 id, drops the rows a rule
now forbids, and covers what they covered with new rows numbered after the old
ones (`regeneration` in the output lists all three). Without
`--keep-rows-from` the list is rebuilt from nothing. Either way the output
depends only on the inputs and `--seed`, never on Python's string hashing; the
registry, HRRR recipe and site template hashes are recorded in the output.

## The recording layout

```
REC/
  runs.csv                      build, case, combo, status, steps_requested, steps_per_frame,
                                last_good_step, raw_steps, stock461_is_designated_reference, ...
  MANIFEST.json                 unreliable_fields
  per-run-summary.csv           optional (or --run-summary): clean_reference per (case, combo)
  provenance/sweep/combos.json  the combo list (or --combos-file)
  inputs/<case>/<combo>/        wrfinput_d01, wrfbdy_d01
  <build>/<case>/<combo>/
    namelist.input              the namelist WRF ran
    hashes.json                 {"steps_per_frame": n, "fields": {NAME: {"dtype", "shape",
                                 "sha256": [one per frame]}}}; frame f is step f * n
    stepNNNN.json               {"fields": [{"name", "dtype", "shape", "offset", "nbytes", "sha256"}]}
    stepNNNN.bin                those fields' words, concatenated
```

A field's digest is SHA-256 over its words exactly as stored: little-endian,
C order, the recorded dtype and shape, with the Time dimension removed. Two
digests agree only when every word agrees.

## Per-scheme column fixtures

The per-scheme lanes' oracle inputs and outputs drop into a fixtures directory
(`--fixtures DIR`, default `REC/fixtures` when present) and are scored in the
same process, under the same arithmetic, one line each.

```
FIXTURES/<scheme>/<fixture-id>/
  fixture.json
  inputs.npz      arrays handed to the runner, by name
  expected.npz    the oracle's outputs (the words WRF wrote), by name
```

`fixture.json`:

```json
{
  "schema": "gpuwm-verify-exact-fixture-v1",
  "scheme": "wsm6",
  "runner": "package.module:function",
  "options": {"dt": 15.0},
  "compare": ["QV", "QC", "QR"],
  "reference": {"build": "WRF v4.6.1 d66e442, gfortran 13.3 -O2 -fno-fast-math -ffp-contract=off",
                "glibc": "2.39", "source": "tools/wsm6_wrf461_oracle"}
}
```

- `runner` is any importable callable `runner(inputs, **options)` returning a
  mapping that holds at least every compared name. Device arrays are copied to
  the host. A lane exposes its kernel's existing column entry point under this
  signature, usually as a thin function beside its oracle validator. Adding a
  scheme is a directory and a function, never a change to this command.
- `compare` is optional; the default is every array in `expected.npz`.
- `options` and `reference` are optional. `reference` is copied into the
  results so the line can be traced to the build that wrote the oracle.

A fixture line reads `PASS`, `FAIL` with the first differing name and its max
ULP, or `ERROR` when the runner could not be imported or raised. `FAIL` and
`ERROR` both make the command exit 1.
